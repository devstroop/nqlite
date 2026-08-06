//! Deterministic execution of `nql-ir` plans against a [`Store`].
//!
//! Every operation here is pure arithmetic over the store's BTree-ordered
//! records and append-only edge list: no randomness, no wall-clock, no
//! network, no LLM. Sorting is stable and always tie-broken by ascending
//! [`RecordId`] string form, so a given plan + store yields byte-identical
//! results every time it runs.
//!
//! kNN similarity is computed through the [`VectorIndex`] trait (see
//! [`crate::index`]): the default [`BruteForceVectorIndex`] is an exact,
//! deterministic cosine scan, which keeps this module's determinism
//! guarantee. An approximate HNSW index exists behind the opt-in `hnsw`
//! feature but is never selected by the engine.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use nql_ir::{
    Filter, MatchDirection, MatchPath, Order, Record, RecordId, RelationEdge, Select, Statement,
    Store, Value, VoteCounts,
};

use crate::bm25::{tokenize, Bm25Index};
use crate::error::{Error, Result};
use crate::index::{BruteForceVectorIndex, VectorIndex};

/// One selected record plus its computed match score.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredRecord {
    pub record: Record,
    /// Match score as defined by the select's ordering operator (see
    /// `compute_score`); `0.0` when the select has no score-producing
    /// operator and no kNN query.
    pub score: f32,
}

/// What produced a [`QueryResult`].
#[derive(Debug, Clone, PartialEq)]
pub enum QueryKind {
    /// A `SELECT` statement (with its enriched select).
    Select(Select),
    /// A `MATCH` graph traversal (with the path that was walked).
    Match(MatchPath),
    /// A `CLOSURE` transitive traversal (with the path that was walked).
    Closure(MatchPath),
}

/// The result of one read statement (`SELECT` or `MATCH`) inside a plan.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryResult {
    /// The statement that produced this result.
    pub kind: QueryKind,
    /// Matching rows, ordered per the statement's deterministic semantics.
    pub rows: Vec<ScoredRecord>,
}

/// Deterministic cosine similarity between two `f32` vectors.
///
/// Returns the dot product divided by the product of the L2 norms. A
/// zero-norm vector on either side (empty, all-zeros, or different lengths —
/// shorter is padded conceptually by `0.0`s) yields `0.0`, so the result is
/// always finite and in `[-1.0, 1.0]`.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Execute a whole `Plan` against `store`, applying statements in order and
/// collecting one [`QueryResult`] per `SELECT` (DML/DDL statements contribute
/// nothing to the output).
pub fn execute_plan(store: &mut Store, plan: &[Statement]) -> Result<Vec<QueryResult>> {
    let mut results = Vec::new();
    for stmt in plan {
        if let Some(res) = execute_statement(store, stmt)? {
            results.push(res);
        }
    }
    Ok(results)
}

/// Execute a single [`Statement`] against `store`, mutating it for DDL/DML and
/// returning a [`QueryResult`] for `SELECT`s (`None` otherwise).
pub fn execute_statement(store: &mut Store, stmt: &Statement) -> Result<Option<QueryResult>> {
    match stmt {
        Statement::CreateTable { table, vector_dim } => {
            // Declaring a table with a dim sets `vector_dims[table]`;
            // declaring without one clears any previous declaration.
            match vector_dim {
                Some(dim) => {
                    store.vector_dims.insert(table.clone(), *dim);
                }
                None => {
                    store.vector_dims.remove(table);
                }
            }
            Ok(None)
        }
        Statement::Insert(rec) => {
            validate_embedding(store, rec)?;
            store.insert(rec.clone());
            Ok(None)
        }
        Statement::Relate(edge) => {
            store.edges.push(edge.clone());
            Ok(None)
        }
        Statement::Forget { id } => {
            store.records.remove(id);
            // Drop every edge incident to the forgotten record, either end.
            store.edges.retain(|e| &e.from != id && &e.to != id);
            Ok(None)
        }
        Statement::Select(sel) => {
            let rows = run_select(store, sel);
            Ok(Some(QueryResult {
                kind: QueryKind::Select(sel.clone()),
                rows,
            }))
        }
        Statement::Match(path) => {
            let rows = run_match(store, path);
            Ok(Some(QueryResult {
                kind: QueryKind::Match(path.clone()),
                rows,
            }))
        }
        Statement::Closure(path) => {
            let rows = run_closure(store, path);
            Ok(Some(QueryResult {
                kind: QueryKind::Closure(path.clone()),
                rows,
            }))
        }
    }
}

/// Validate that a record's embedding length matches its table's declared
/// vector dimension, when one is declared.
fn validate_embedding(store: &Store, rec: &Record) -> Result<()> {
    let Some(dim) = store.vector_dims.get(&rec.id.table) else {
        return Ok(());
    };
    if let Some(emb) = &rec.embedding {
        if emb.len() != *dim {
            return Err(Error::EmbeddingDimMismatch {
                table: rec.id.table.clone(),
                expected: *dim,
                actual: emb.len(),
            });
        }
    }
    Ok(())
}

/// Scan `store.records` for the select's table, apply the filter, compute
/// per-row scores, order, and limit — all deterministically.
///
/// When the select carries a kNN clause, similarity is produced by the
/// configured [`VectorIndex`] (default: exact [`BruteForceVectorIndex`])
/// rather than an inline cosine scan. The index is rebuilt from the filtered
/// candidates on every call, so the result stays a pure, deterministic
/// function of `(store, select)`.
fn run_select(store: &Store, sel: &Select) -> Vec<ScoredRecord> {
    let candidates: Vec<Record> = store
        .records
        .values()
        .filter(|r| r.id.table == sel.table)
        .filter(|r| matches_filter(r, sel.filter.as_ref()))
        .cloned()
        .collect();

    // Rank every embedded candidate against the query through the index.
    // Non-embedded records are absent from the index and fall back to a
    // similarity of `0.0`, exactly as the inline scan did.
    let knn_sims: Option<BTreeMap<RecordId, f32>> = sel.knn.as_ref().map(|knn| {
        let index = build_default_index(&candidates);
        index
            .search(&knn.query, candidates.len())
            .into_iter()
            .collect()
    });

    // A `Filter::Bm25` turns the SELECT into lexical retrieval: build one
    // deterministic BM25 index over the filtered candidates' text field and
    // score every row with it (records missing the field / without a `Str`
    // value score 0.0). The index is rebuilt per call, so the result stays a
    // pure function of `(store, select)`.
    let bm25: Option<(Bm25Index, Vec<String>)> = match sel.filter.as_ref() {
        Some(Filter::Bm25 { field, query, .. }) => {
            let index = Bm25Index::new(field, candidates.iter());
            let query_tokens = tokenize(query);
            Some((index, query_tokens))
        }
        _ => None,
    };

    let mut rows: Vec<ScoredRecord> = candidates
        .into_iter()
        .map(|record| {
            let score = compute_score(store, sel, &record, knn_sims.as_ref(), bm25.as_ref());
            ScoredRecord { record, score }
        })
        .collect();

    order_rows(&mut rows, sel);

    if let Some(limit) = effective_limit(sel) {
        rows.truncate(limit);
    }
    rows
}

/// Execute a [`MatchPath`] against `store`.
///
/// Deterministic graph traversal: the frontier starts at `path.start` and each
/// step moves to the other endpoint of every edge with a matching name,
/// direction, and (when the step carries one) edge-property filter. Edges are
/// scanned in append order; endpoints are deduplicated by [`RecordId`] keeping
/// first appearance, so the result is a pure function of `(store, path)`. A
/// missing start record yields an empty result (no panic), and dangling edges
/// (endpoints never inserted) are skipped.
fn run_match(store: &Store, path: &MatchPath) -> Vec<ScoredRecord> {
    if !store.records.contains_key(&path.start) {
        return Vec::new();
    }

    // Frontier of reached RecordIds, in deterministic (append/dedup-first)
    // order. Scored by the edge that first reached them (weight or 0.0).
    let mut frontier: Vec<RecordId> = vec![path.start.clone()];
    let mut score_of: BTreeMap<RecordId, f32> = BTreeMap::new();
    score_of.insert(path.start.clone(), 0.0);

    for step in &path.steps {
        let mut next: Vec<RecordId> = Vec::new();
        for edge in &store.edges {
            // Is this edge part of the current frontier, in the right direction?
            let (from_side, to_side) = match step.direction {
                MatchDirection::Out => (&edge.from, &edge.to),
                MatchDirection::In => (&edge.to, &edge.from),
            };
            if edge.name != step.name || !frontier.contains(from_side) {
                continue;
            }
            if !matches_edge_props(edge, step.edge_props.as_ref()) {
                continue;
            }
            if !store.records.contains_key(to_side) {
                continue; // dangling edge: skip
            }
            if !next.contains(to_side) {
                next.push(to_side.clone());
            }
            // First edge to reach this endpoint wins its score.
            if !score_of.contains_key(to_side) {
                score_of.insert(to_side.clone(), edge.weight.unwrap_or(0.0));
            }
        }
        frontier = next;
        if frontier.is_empty() {
            break;
        }
    }

    frontier
        .into_iter()
        .filter_map(|id| {
            store.records.get(&id).map(|record| ScoredRecord {
                record: record.clone(),
                score: score_of.get(&id).copied().unwrap_or(0.0),
            })
        })
        .collect()
}

/// Execute a [`MatchPath`] as a transitive closure against `store`.
///
/// Deterministic breadth-first traversal: the frontier starts at `path.start`
/// and each step expands it to every record reachable via edges with the
/// matching name/direction/filter — including multi-hop paths — until no new
/// records are found (fixpoint). Every record ever reached (including the
/// start) is returned once, in first-visit (BFS) order, scored by the depth
/// at which it was first reached (`0.0` for the start record). A missing
/// start record yields an empty result; dangling edges are skipped.
fn run_closure(store: &Store, path: &MatchPath) -> Vec<ScoredRecord> {
    if !store.records.contains_key(&path.start) {
        return Vec::new();
    }

    let mut visited: Vec<RecordId> = vec![path.start.clone()];
    let mut depth_of: BTreeMap<RecordId, u32> = BTreeMap::new();
    depth_of.insert(path.start.clone(), 0);

    // BFS frontier: everything reached at the previous depth (start at depth 0).
    let mut frontier: Vec<RecordId> = vec![path.start.clone()];
    let mut next_depth = 1u32;

    for step in &path.steps {
        // Expand the current frontier to fixpoint along this step's edge.
        loop {
            let mut newly_reached: Vec<RecordId> = Vec::new();
            for edge in &store.edges {
                let (from_side, to_side) = match step.direction {
                    MatchDirection::Out => (&edge.from, &edge.to),
                    MatchDirection::In => (&edge.to, &edge.from),
                };
                if edge.name != step.name || !frontier.contains(from_side) {
                    continue;
                }
                if !matches_edge_props(edge, step.edge_props.as_ref()) {
                    continue;
                }
                if !store.records.contains_key(to_side) {
                    continue; // dangling edge: skip
                }
                if !visited.contains(to_side) {
                    visited.push(to_side.clone());
                    newly_reached.push(to_side.clone());
                    depth_of.insert(to_side.clone(), next_depth);
                }
            }
            if newly_reached.is_empty() {
                break; // fixpoint reached
            }
            frontier = newly_reached;
            next_depth += 1;
        }
        // After this step's closure, the next step continues from everything
        // this step reached (already in `visited`), which is the new frontier.
        frontier = visited[1..].to_vec();
    }

    visited
        .into_iter()
        .filter_map(|id| {
            store.records.get(&id).map(|record| ScoredRecord {
                record: record.clone(),
                score: depth_of.get(&id).copied().unwrap_or(0) as f32,
            })
        })
        .collect()
}

/// Apply a step's optional edge-property filter: `edge.props[field] == value`.
/// `None` accepts every edge; only `Filter::FieldEquals` is a valid filter
/// here (the parser only produces that shape).
fn matches_edge_props(edge: &RelationEdge, filter: Option<&Filter>) -> bool {
    match filter {
        None => true,
        Some(Filter::FieldEquals { field, value }) => edge.props.get(field) == Some(value),
        // The IR contract says only FieldEquals is valid; anything else is
        // treated as a non-match rather than a panic (defensive).
        Some(_) => false,
    }
}

/// Build the default vector index (exact brute-force) over the embeddings of
/// the given candidate records.
///
/// This is the engine's swap point for alternative [`VectorIndex`]
/// implementations (e.g. the feature-gated, approximate `HnswVectorIndex`):
/// swap the concrete type here and the rest of the engine is unchanged.
fn build_default_index(records: &[Record]) -> Box<dyn VectorIndex> {
    let mut index = BruteForceVectorIndex::default();
    for r in records {
        if let Some(emb) = &r.embedding {
            index.upsert(r.id.clone(), emb.clone());
        }
    }
    Box::new(index)
}

/// Apply the select's deterministic ordering. Stable sorts guarantee equal
/// keys keep their (BTree) input order; we additionally tie-break by
/// ascending [`RecordId`] so the final order is total and reproducible.
fn order_rows(rows: &mut [ScoredRecord], sel: &Select) {
    if matches!(sel.order.as_ref(), Some(Order::Recency)) {
        rows.sort_by(|a, b| {
            b.record
                .created_at
                .cmp(&a.record.created_at)
                .then_with(|| a.record.id.cmp(&b.record.id))
        });
        return;
    }

    // A kNN clause orders by similarity even without an explicit ORDER BY;
    // a `Filter::Bm25` likewise orders by its lexical score. Any other
    // explicit order sorts by its score. With no order, no kNN and no BM25,
    // rows stay in BTree key order (already sorted by RecordId).
    if sel.order.is_some()
        || sel.knn.is_some()
        || matches!(sel.filter.as_ref(), Some(Filter::Bm25 { .. }))
    {
        rows.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.record.id.cmp(&b.record.id))
        });
    }
}

/// The effective row cap: the smaller of `knn.k`, `Filter::Bm25.k` (when
/// present) and `select.limit`.
fn effective_limit(sel: &Select) -> Option<usize> {
    let knn_k = sel.knn.as_ref().map(|k| k.k);
    let bm25_k = match sel.filter.as_ref() {
        Some(Filter::Bm25 { k: Some(k), .. }) => Some(*k),
        _ => None,
    };
    let caps: Vec<usize> = [knn_k, bm25_k, sel.limit].into_iter().flatten().collect();
    caps.into_iter().min()
}

/// Apply the select's field filter. `FieldEquals` uses the derived `PartialEq`
/// on [`Value`] (exact, deterministic equality); `HasEmbedding` requires a
/// non-`None` embedding. `Bm25` is a *scoring* filter: it never prunes rows —
/// every row of the table is returned and ranked by its lexical score.
fn matches_filter(rec: &Record, filter: Option<&Filter>) -> bool {
    match filter {
        None => true,
        Some(Filter::HasEmbedding) => rec.embedding.is_some(),
        Some(Filter::FieldEquals { field, value }) => rec.body.get(field) == Some(value),
        Some(Filter::Bm25 { .. }) => true,
    }
}

/// Compute the per-row match score, which doubles as the sort key for
/// score-based orders:
///
/// - `Filter::Bm25` → the record's BM25 lexical score (see [`Bm25Index`]);
///   this overrides any explicit order, matching the operator's "rank by
///   relevance" semantics.
/// - `Order::Score` → Laplace-smoothed mean of the `:voted` edge weights
///   pointing at the record: `(sum + 1) / (n + 2)` — `0.5` with zero votes.
/// - `Order::Votes` → net up−down vote count (see [`vote_counts`]).
/// - `Order::Feedback` → time-decayed recent feedback (see [`feedback_score`]).
/// - `Order::Salience` with kNN → `0.7 * similarity + 0.3 * normalized_score`,
///   where `normalized_score` is the Laplace score clamped to `[0, 1]`
///   (weights are treated as `[0, 1]` confidence values; the clamp keeps the
///   blend in range even for out-of-range weights).
/// - `Order::Salience` without kNN → the normalized score alone.
/// - Anything else → cosine similarity vs the kNN query (`0.0` when there is
///   no kNN clause, or when the record has no embedding / zero-norm vector).
fn compute_score(
    store: &Store,
    sel: &Select,
    rec: &Record,
    knn_sims: Option<&BTreeMap<RecordId, f32>>,
    bm25: Option<&(Bm25Index, Vec<String>)>,
) -> f32 {
    // A BM25 filter is the dominant score source: every row is ranked by its
    // lexical relevance regardless of ORDER BY / kNN.
    if let (Some(Filter::Bm25 { field, .. }), Some((index, query_tokens))) =
        (sel.filter.as_ref(), bm25)
    {
        debug_assert_eq!(index.field(), field, "index built for the filter's field");
        return index.score(&rec.id, query_tokens);
    }

    let similarity = match &sel.knn {
        Some(_) => match knn_sims {
            Some(sims) => sims.get(&rec.id).copied().unwrap_or(0.0),
            None => 0.0,
        },
        None => 0.0,
    };

    match sel.order.as_ref() {
        Some(Order::Score) => score_of(store, rec),
        Some(Order::Votes) => vote_counts(store, &rec.id).net as f32,
        Some(Order::Feedback) => feedback_score(store, &rec.id),
        Some(Order::Salience) if sel.knn.is_some() => {
            0.7 * similarity + 0.3 * score_of(store, rec).clamp(0.0, 1.0)
        }
        Some(Order::Salience) => score_of(store, rec).clamp(0.0, 1.0),
        _ => similarity,
    }
}

/// Laplace-smoothed mean of `:voted` edge weights on the record.
///
/// A vote edge is any edge with `name == "voted"` pointing **to** the record;
/// `weight` defaults to `1.0` when absent (an upvote). The estimate
/// `(sum + 1) / (n + 2)` starts at `0.5` with zero votes and moves toward the
/// observed mean as votes accumulate. The edge name is stored **without** the
/// `:` prefix — the parser strips it (`RELATE (a) -> :voted -> (b)` stores
/// `"voted"`), and `::votes`/`::feedback` use the same convention.
fn score_of(store: &Store, rec: &Record) -> f32 {
    let mut sum = 0.0f32;
    let mut n = 0usize;
    for edge in &store.edges {
        if edge.name == "voted" && edge.to == rec.id {
            sum += edge.weight.unwrap_or(1.0);
            n += 1;
        }
    }
    (sum + 1.0) / (n as f32 + 2.0)
}

/// The `:voted` edges pointing **at** `id`, in store (append) order.
///
/// A vote is a directed edge `(voter)->:voted {value:+1|-1, weight:0..1}->(record)`
/// (see docs/decisions.md D9); only edges whose `name == "voted"` and whose
/// `to` is the record count. Iteration order is the store's append order,
/// which is deterministic for a given store.
fn votes_toward<'a>(store: &'a Store, id: &RecordId) -> Vec<&'a RelationEdge> {
    store
        .edges
        .iter()
        .filter(|e| e.name == "voted" && e.to == *id)
        .collect()
}

/// The signed vote value of an edge: `+1` for `value:+1`, `-1` for `value:-1`,
/// `0` for anything else (missing prop, other value). `value` lives in the
/// edge's `props` map, as produced by `RELATE ... SET value = <n>`.
fn vote_value(props: &BTreeMap<String, Value>) -> i8 {
    match props.get("value") {
        Some(Value::Int(1)) | Some(Value::Float(1.0)) => 1,
        Some(Value::Int(-1)) | Some(Value::Float(-1.0)) => -1,
        _ => 0,
    }
}

/// Aggregate vote counts over a record's `:voted` edges.
///
/// Deterministic: iterates `store.edges` in append order and only counts
/// `value == +1` (up) and `value == -1` (down) votes pointing at `id`;
/// `net = up - down`.
pub fn vote_counts(store: &Store, id: &RecordId) -> VoteCounts {
    let mut up = 0u64;
    let mut down = 0u64;
    for edge in votes_toward(store, id) {
        match vote_value(&edge.props) {
            1 => up += 1,
            -1 => down += 1,
            _ => {}
        }
    }
    VoteCounts {
        up,
        down,
        net: up as i64 - down as i64,
    }
}

/// Time-decayed recent feedback over a record's `:voted` edges.
///
/// `Σ sign(v) · decay(created_at)` where `sign` is `+1` for upvotes and `-1`
/// for downvotes (see [`vote_value`]) and
///
/// ```text
/// decay(t) = 1 / (1 + λ · (now − t)),   λ = 1.0
/// ```
///
/// with `now` = the **maximum `created_at` over all `:voted` edges in the
/// store**. Using the data's own max as "now" (instead of wall-clock) keeps
/// the score a pure, deterministic function of the store: the same input
/// always yields the same output, and `created_at` is treated as arbitrary
/// time units (engine convention: seconds; tests use synthetic units). A vote
/// at `now` contributes its full sign; each unit of age halves the
/// contribution (age 1 → 1/2, age 2 → 1/3, …). A record with no `:voted`
/// edges scores `0.0`.
pub fn feedback_score(store: &Store, id: &RecordId) -> f32 {
    let lambda = 1.0f32;
    let mut now: Option<i64> = None;
    for edge in &store.edges {
        if edge.name == "voted" {
            now = Some(now.map_or(edge.created_at, |n| n.max(edge.created_at)));
        }
    }
    let Some(now) = now else {
        return 0.0;
    };
    let mut total = 0.0f32;
    for edge in votes_toward(store, id) {
        let sign = vote_value(&edge.props) as f32;
        let age = now.saturating_sub(edge.created_at) as f32;
        total += sign * (1.0 / (1.0 + lambda * age));
    }
    total
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use nql_ir::{Plan, RecordId, RelationEdge, Value};

    use super::*;
    use crate::{Database, Knn};
    use nql_ir::MatchStep;

    fn record(id: &str, body: BTreeMap<String, Value>, embedding: Option<Vec<f32>>) -> Record {
        Record {
            id: RecordId::parse(id).unwrap(),
            body,
            embedding,
            created_at: 0,
        }
    }

    fn num(n: i64) -> Value {
        Value::Int(n)
    }

    fn str_(s: &str) -> Value {
        Value::Str(s.to_string())
    }

    fn create(table: &str, dim: Option<usize>) -> Statement {
        Statement::CreateTable {
            table: table.to_string(),
            vector_dim: dim,
        }
    }

    fn select(table: &str) -> Select {
        Select {
            table: table.to_string(),
            ..Select::default()
        }
    }

    #[test]
    fn create_insert_roundtrip() {
        let mut db = Database::default();
        db.execute(&[
            create("person", None),
            Statement::Insert(record(
                "person:1",
                BTreeMap::from([("name".into(), str_("alice"))]),
                None,
            )),
        ])
        .unwrap();
        let st = db.store();
        assert!(st
            .records
            .contains_key(&RecordId::parse("person:1").unwrap()));
        assert_eq!(st.records.len(), 1);
        assert!(!st.vector_dims.contains_key("person"));
        // Re-declare with a dim, then without: entry is removed again.
        db.execute(&[create("person", Some(3))]).unwrap();
        assert_eq!(db.store().vector_dims.get("person"), Some(&3));
        db.execute(&[create("person", None)]).unwrap();
        assert!(!db.store().vector_dims.contains_key("person"));
    }

    #[test]
    fn insert_dim_mismatch_is_error() {
        let mut db = Database::default();
        db.execute(&[create("vec", Some(3))]).unwrap();
        // Wrong length -> error.
        let err = db
            .execute(&[Statement::Insert(record(
                "vec:1",
                BTreeMap::new(),
                Some(vec![1.0, 2.0]),
            ))])
            .unwrap_err();
        assert!(matches!(
            err,
            Error::EmbeddingDimMismatch {
                table,
                expected: 3,
                actual: 2
            } if table == "vec"
        ));
        // Right length -> ok.
        db.execute(&[Statement::Insert(record(
            "vec:1",
            BTreeMap::new(),
            Some(vec![1.0, 2.0, 3.0]),
        ))])
        .unwrap();
        assert_eq!(db.store().records.len(), 1);
        // Missing embedding is allowed even with a declared dim.
        db.execute(&[Statement::Insert(record("vec:2", BTreeMap::new(), None))])
            .unwrap();
        assert_eq!(db.store().records.len(), 2);
    }

    #[test]
    fn select_filter_field_equals_and_has_embedding() {
        let mut db = Database::default();
        db.execute(&[
            create("person", Some(2)),
            Statement::Insert(record(
                "person:1",
                BTreeMap::from([("name".into(), str_("alice")), ("age".into(), num(30))]),
                Some(vec![1.0, 0.0]),
            )),
            Statement::Insert(record(
                "person:2",
                BTreeMap::from([("name".into(), str_("bob")), ("age".into(), num(40))]),
                None,
            )),
            Statement::Insert(record(
                "person:3",
                BTreeMap::from([("name".into(), str_("carol")), ("age".into(), num(30))]),
                Some(vec![0.0, 1.0]),
            )),
        ])
        .unwrap();

        let res = db
            .execute(&[Statement::Select(Select {
                filter: Some(Filter::FieldEquals {
                    field: "age".into(),
                    value: num(30),
                }),
                ..select("person")
            })])
            .unwrap();
        let ids: Vec<_> = res[0]
            .rows
            .iter()
            .map(|r| r.record.id.to_string())
            .collect();
        assert_eq!(ids, ["person:1", "person:3"]); // BTree key order

        let res = db
            .execute(&[Statement::Select(Select {
                filter: Some(Filter::HasEmbedding),
                ..select("person")
            })])
            .unwrap();
        let ids: Vec<_> = res[0]
            .rows
            .iter()
            .map(|r| r.record.id.to_string())
            .collect();
        assert_eq!(ids, ["person:1", "person:3"]);
    }

    #[test]
    fn knn_returns_nearest_by_cosine_with_k_limit() {
        let mut db = Database::default();
        db.execute(&[
            create("vec", Some(2)),
            Statement::Insert(record("vec:a", BTreeMap::new(), Some(vec![1.0, 0.0]))),
            Statement::Insert(record("vec:b", BTreeMap::new(), Some(vec![0.0, 1.0]))),
            Statement::Insert(record(
                "vec:c",
                BTreeMap::new(),
                Some(vec![0.70710677, 0.70710677]),
            )),
        ])
        .unwrap();

        let res = db
            .execute(&[Statement::Select(Select {
                knn: Some(Knn {
                    query: vec![1.0, 0.0],
                    k: 2,
                }),
                ..select("vec")
            })])
            .unwrap();
        let ids: Vec<_> = res[0]
            .rows
            .iter()
            .map(|r| r.record.id.to_string())
            .collect();
        assert_eq!(ids, ["vec:a", "vec:c"]);
        assert!(res[0].rows[0].score > res[0].rows[1].score);
        assert!((res[0].rows[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn knn_without_embedding_scores_zero_and_orders_last() {
        let mut db = Database::default();
        db.execute(&[
            create("vec", Some(2)),
            Statement::Insert(record("vec:1", BTreeMap::new(), Some(vec![1.0, 0.0]))),
            Statement::Insert(record("vec:2", BTreeMap::new(), None)),
        ])
        .unwrap();
        let res = db
            .execute(&[Statement::Select(Select {
                knn: Some(Knn {
                    query: vec![1.0, 0.0],
                    k: 10,
                }),
                ..select("vec")
            })])
            .unwrap();
        assert_eq!(res[0].rows[0].record.id.to_string(), "vec:1");
        assert_eq!(res[0].rows[1].record.id.to_string(), "vec:2");
        assert_eq!(res[0].rows[1].score, 0.0);
    }

    #[test]
    fn forget_removes_record_and_incident_edges() {
        let mut db = Database::default();
        db.execute(&[
            create("person", None),
            create("note", None),
            Statement::Insert(record("person:1", BTreeMap::new(), None)),
            Statement::Insert(record("person:2", BTreeMap::new(), None)),
            Statement::Insert(record("note:1", BTreeMap::new(), None)),
            Statement::Relate(RelationEdge {
                from: RecordId::parse("person:1").unwrap(),
                name: "wrote".into(),
                to: RecordId::parse("note:1").unwrap(),
                created_at: 1,
                weight: None,
                props: BTreeMap::new(),
            }),
            Statement::Relate(RelationEdge {
                from: RecordId::parse("note:1").unwrap(),
                name: "mentions".into(),
                to: RecordId::parse("person:2").unwrap(),
                created_at: 2,
                weight: None,
                props: BTreeMap::new(),
            }),
            // Edge not touching person:1 — must survive.
            Statement::Relate(RelationEdge {
                from: RecordId::parse("person:2").unwrap(),
                name: "knows".into(),
                to: RecordId::parse("note:1").unwrap(),
                created_at: 3,
                weight: None,
                props: BTreeMap::new(),
            }),
        ])
        .unwrap();
        assert_eq!(db.store().edges.len(), 3);

        db.execute(&[Statement::Forget {
            id: RecordId::parse("person:1").unwrap(),
        }])
        .unwrap();

        assert!(!db
            .store()
            .records
            .contains_key(&RecordId::parse("person:1").unwrap()));
        // e2 (note:1 -> person:2) and e3 (person:2 -> note:1) survive: they
        // don't touch person:1.
        assert_eq!(db.store().edges.len(), 2);
        assert!(db
            .store()
            .edges
            .iter()
            .all(|e| e.from.to_string() != "person:1" && e.to.to_string() != "person:1"));
        // Forgetting a record on the `to` side also drops its edges.
        db.execute(&[Statement::Forget {
            id: RecordId::parse("note:1").unwrap(),
        }])
        .unwrap();
        assert!(db.store().edges.is_empty());
    }

    #[test]
    fn score_operator_uses_laplace_smoothed_votes() {
        let mut db = Database::default();
        db.execute(&[
            create("post", None),
            Statement::Insert(record("post:1", BTreeMap::new(), None)),
            Statement::Insert(record("post:2", BTreeMap::new(), None)),
            Statement::Relate(RelationEdge {
                from: RecordId::parse("user:voter").unwrap(),
                name: "voted".into(),
                to: RecordId::parse("post:1").unwrap(),
                created_at: 1,
                weight: Some(1.0),
                props: BTreeMap::new(),
            }),
            Statement::Relate(RelationEdge {
                from: RecordId::parse("user:voter").unwrap(),
                name: "voted".into(),
                to: RecordId::parse("post:2").unwrap(),
                created_at: 2,
                weight: Some(0.0),
                props: BTreeMap::new(),
            }),
            // Non-`:voted` edges must not count.
            Statement::Relate(RelationEdge {
                from: RecordId::parse("user:voter").unwrap(),
                name: "mentioned".into(),
                to: RecordId::parse("post:1").unwrap(),
                created_at: 3,
                weight: Some(0.0),
                props: BTreeMap::new(),
            }),
            // A vote on a *different* record must not count either.
            Statement::Relate(RelationEdge {
                from: RecordId::parse("user:voter").unwrap(),
                name: "voted".into(),
                to: RecordId::parse("post:1").unwrap(),
                created_at: 4,
                weight: Some(1.0),
                props: BTreeMap::new(),
            }),
        ])
        .unwrap();

        // post:1 -> (1.0 + 1.0 + 1.0)/(2 + 2) = 0.75 ; post:2 -> (0.0 + 1.0)/(1 + 2) = 1/3.
        let res = db
            .execute(&[Statement::Select(Select {
                order: Some(Order::Score),
                ..select("post")
            })])
            .unwrap();
        let scores: Vec<_> = res[0].rows.iter().map(|r| r.score).collect();
        assert!((scores[0] - 0.75).abs() < 1e-6);
        assert!((scores[1] - 1.0 / 3.0).abs() < 1e-6);
        // Sorted by score desc, tie-break by id asc.
        let ids: Vec<_> = res[0]
            .rows
            .iter()
            .map(|r| r.record.id.to_string())
            .collect();
        assert_eq!(ids, ["post:1", "post:2"]);

        // Zero votes -> default 0.5.
        db.execute(&[Statement::Insert(record("post:3", BTreeMap::new(), None))])
            .unwrap();
        let res = db
            .execute(&[Statement::Select(Select {
                order: Some(Order::Score),
                ..select("post")
            })])
            .unwrap();
        let post3 = res[0]
            .rows
            .iter()
            .find(|r| r.record.id.to_string() == "post:3")
            .unwrap();
        assert!((post3.score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn salience_blends_similarity_and_normalized_score() {
        let mut db = Database::default();
        db.execute(&[
            create("doc", Some(2)),
            Statement::Insert(record("doc:1", BTreeMap::new(), Some(vec![1.0, 0.0]))),
            Statement::Relate(RelationEdge {
                from: RecordId::parse("user:v").unwrap(),
                name: "voted".into(),
                to: RecordId::parse("doc:1").unwrap(),
                created_at: 1,
                weight: Some(1.0),
                props: BTreeMap::new(),
            }),
            // doc:2: no embedding, no votes -> sim 0, score 0.5.
            Statement::Insert(record("doc:2", BTreeMap::new(), None)),
        ])
        .unwrap();
        let res = db
            .execute(&[Statement::Select(Select {
                knn: Some(Knn {
                    query: vec![1.0, 0.0],
                    k: 10,
                }),
                order: Some(Order::Salience),
                ..select("doc")
            })])
            .unwrap();
        // doc:1 -> sim 1.0, one upvote -> score (1+1)/(1+2) = 2/3,
        // salience 0.7*1.0 + 0.3*2/3 = 0.9 ; doc:2 -> sim 0, score 0.5,
        // salience 0.3*0.5 = 0.15.
        let s1 = res[0].rows[0].score;
        let s2 = res[0].rows[1].score;
        assert!((s1 - 0.9).abs() < 1e-5);
        assert!((s2 - 0.15).abs() < 1e-5);
        assert_eq!(res[0].rows[0].record.id.to_string(), "doc:1");
    }

    #[test]
    fn recency_orders_by_created_at_descending() {
        let mut db = Database::default();
        let mut old = record("msg:old", BTreeMap::new(), None);
        old.created_at = 1;
        let mut mid = record("msg:mid", BTreeMap::new(), None);
        mid.created_at = 2;
        let mut new = record("msg:new", BTreeMap::new(), None);
        new.created_at = 3;
        db.execute(&[
            create("msg", None),
            Statement::Insert(mid),
            Statement::Insert(new),
            Statement::Insert(old),
        ])
        .unwrap();
        let res = db
            .execute(&[Statement::Select(Select {
                order: Some(Order::Recency),
                ..select("msg")
            })])
            .unwrap();
        let ids: Vec<_> = res[0]
            .rows
            .iter()
            .map(|r| r.record.id.to_string())
            .collect();
        assert_eq!(ids, ["msg:new", "msg:mid", "msg:old"]);
    }

    #[test]
    fn cosine_similarity_handles_zero_norm_and_mismatched_lengths() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!((cosine_similarity(&[1.0, 0.0], &[0.0, 1.0])).abs() < 1e-6);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
        assert_eq!(cosine_similarity(&[], &[1.0, 0.0]), 0.0);
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 1.0]), 1.0);
        let r = cosine_similarity(&[1e19, 1e19], &[1e19, 1e19]);
        assert!(r.is_finite() && r > 0.0);
    }

    #[test]
    fn same_plan_on_equal_stores_yields_equal_results() {
        let plan: Plan = vec![
            create("doc", Some(2)),
            Statement::Insert(record(
                "doc:1",
                BTreeMap::from([("tag".into(), str_("a")), ("n".into(), num(1))]),
                Some(vec![1.0, 0.0]),
            )),
            Statement::Insert(record(
                "doc:2",
                BTreeMap::from([("tag".into(), str_("a")), ("n".into(), num(2))]),
                Some(vec![0.0, 1.0]),
            )),
            Statement::Insert(record(
                "doc:3",
                BTreeMap::from([("tag".into(), str_("b"))]),
                Some(vec![0.70710677, 0.70710677]),
            )),
            Statement::Relate(RelationEdge {
                from: RecordId::parse("u:v").unwrap(),
                name: "voted".into(),
                to: RecordId::parse("doc:1").unwrap(),
                created_at: 1,
                weight: Some(1.0),
                props: BTreeMap::new(),
            }),
            Statement::Select(Select {
                knn: Some(Knn {
                    query: vec![1.0, 0.0],
                    k: 2,
                }),
                filter: Some(Filter::FieldEquals {
                    field: "tag".into(),
                    value: str_("a"),
                }),
                order: Some(Order::Salience),
                ..select("doc")
            }),
        ];

        let run = || {
            let mut db = Database::default();
            db.execute(&plan).unwrap()
        };

        let a = run();
        let b = run();
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
        assert_eq!(a, b);
    }

    #[test]
    fn multi_select_plan_returns_one_result_per_select() {
        let mut db = Database::default();
        db.execute(&[
            create("t", None),
            Statement::Insert(record("t:1", BTreeMap::new(), None)),
            Statement::Insert(record("t:2", BTreeMap::new(), None)),
        ])
        .unwrap();
        let results = db
            .execute(&[
                Statement::Select(select("t")),
                Statement::Insert(record("t:3", BTreeMap::new(), None)),
                Statement::Select(select("t")),
            ])
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].rows.len(), 2);
        assert_eq!(results[1].rows.len(), 3); // sees the insert in the same plan
    }
    // -- MATCH traversal ---------------------------------------------------

    fn relate(from: &str, name: &str, to: &str, weight: Option<f32>) -> Statement {
        Statement::Relate(RelationEdge {
            from: RecordId::parse(from).unwrap(),
            name: name.into(),
            to: RecordId::parse(to).unwrap(),
            created_at: 0,
            weight,
            props: BTreeMap::new(),
        })
    }

    fn match_path(start: &str, steps: &[(MatchDirection, &str)]) -> Statement {
        Statement::Match(MatchPath {
            start: RecordId::parse(start).unwrap(),
            steps: steps
                .iter()
                .map(|(direction, name)| MatchStep {
                    direction: *direction,
                    name: (*name).into(),
                    edge_props: None,
                })
                .collect(),
        })
    }

    #[test]
    fn match_outgoing_returns_reached_records_in_edge_order() {
        let mut db = Database::default();
        db.execute(&[
            create("person", None),
            create("note", None),
            Statement::Insert(record("person:1", BTreeMap::new(), None)),
            Statement::Insert(record("note:1", BTreeMap::new(), None)),
            Statement::Insert(record("note:2", BTreeMap::new(), None)),
            Statement::Insert(record("note:3", BTreeMap::new(), None)),
            relate("person:1", "mentions", "note:2", Some(0.5)),
            relate("person:1", "mentions", "note:1", Some(0.9)),
            relate("person:1", "mentions", "note:3", None),
        ])
        .unwrap();

        let res = db
            .execute(&[match_path("person:1", &[(MatchDirection::Out, "mentions")])])
            .unwrap();
        assert!(matches!(res[0].kind, QueryKind::Match(_)));
        // Edge-append order, deduped; scores are the first edge's weight.
        let rows: Vec<(String, f32)> = res[0]
            .rows
            .iter()
            .map(|r| (r.record.id.to_string(), r.score))
            .collect();
        assert_eq!(
            rows,
            vec![
                ("note:2".to_string(), 0.5),
                ("note:1".to_string(), 0.9),
                ("note:3".to_string(), 0.0),
            ]
        );
    }

    #[test]
    fn match_incoming_returns_predecessors() {
        let mut db = Database::default();
        db.execute(&[
            create("person", None),
            create("note", None),
            Statement::Insert(record("person:1", BTreeMap::new(), None)),
            Statement::Insert(record("person:2", BTreeMap::new(), None)),
            Statement::Insert(record("note:9", BTreeMap::new(), None)),
            relate("person:1", "mentions", "note:9", Some(0.3)),
            relate("person:2", "mentions", "note:9", Some(0.7)),
        ])
        .unwrap();

        let res = db
            .execute(&[match_path("note:9", &[(MatchDirection::In, "mentions")])])
            .unwrap();
        let ids: Vec<String> = res[0]
            .rows
            .iter()
            .map(|r| r.record.id.to_string())
            .collect();
        assert_eq!(ids, vec!["person:1", "person:2"]);
    }

    #[test]
    fn match_multi_hop_walks_path() {
        let mut db = Database::default();
        db.execute(&[
            create("person", None),
            create("team", None),
            Statement::Insert(record("alice:1", BTreeMap::new(), None)),
            Statement::Insert(record("bob:2", BTreeMap::new(), None)),
            Statement::Insert(record("team:1", BTreeMap::new(), None)),
            relate("alice:1", "knows", "bob:2", Some(1.0)),
            relate("bob:2", "works_with", "team:1", Some(0.4)),
        ])
        .unwrap();

        let res = db
            .execute(&[match_path(
                "alice:1",
                &[
                    (MatchDirection::Out, "knows"),
                    (MatchDirection::Out, "works_with"),
                ],
            )])
            .unwrap();
        let rows: Vec<(String, f32)> = res[0]
            .rows
            .iter()
            .map(|r| (r.record.id.to_string(), r.score))
            .collect();
        // Only the final frontier is returned (path semantics).
        assert_eq!(rows, vec![("team:1".to_string(), 0.4)]);
    }

    #[test]
    fn match_unknown_start_or_dangling_edge_is_empty() {
        let mut db = Database::default();
        db.execute(&[
            create("person", None),
            create("note", None),
            Statement::Insert(record("person:1", BTreeMap::new(), None)),
            // Dangling edge: note:99 was never inserted.
            relate("person:1", "mentions", "note:99", None),
        ])
        .unwrap();

        // Unknown start record: empty, not an error.
        let res = db
            .execute(&[match_path(
                "person:404",
                &[(MatchDirection::Out, "mentions")],
            )])
            .unwrap();
        assert!(res[0].rows.is_empty());

        // Start exists but every edge is dangling: empty.
        let res = db
            .execute(&[match_path("person:1", &[(MatchDirection::Out, "mentions")])])
            .unwrap();
        assert!(res[0].rows.is_empty());
    }

    #[test]
    fn match_wrong_edge_name_yields_empty() {
        let mut db = Database::default();
        db.execute(&[
            create("person", None),
            create("note", None),
            Statement::Insert(record("person:1", BTreeMap::new(), None)),
            Statement::Insert(record("note:1", BTreeMap::new(), None)),
            relate("person:1", "mentions", "note:1", None),
        ])
        .unwrap();

        let res = db
            .execute(&[match_path("person:1", &[(MatchDirection::Out, "likes")])])
            .unwrap();
        assert!(res[0].rows.is_empty());
    }

    #[test]
    fn match_dedupes_repeated_targets() {
        let mut db = Database::default();
        db.execute(&[
            create("person", None),
            create("note", None),
            Statement::Insert(record("person:1", BTreeMap::new(), None)),
            Statement::Insert(record("note:1", BTreeMap::new(), None)),
            relate("person:1", "mentions", "note:1", Some(0.2)),
            relate("person:1", "mentions", "note:1", Some(0.8)),
        ])
        .unwrap();

        let res = db
            .execute(&[match_path("person:1", &[(MatchDirection::Out, "mentions")])])
            .unwrap();
        let rows: Vec<(String, f32)> = res[0]
            .rows
            .iter()
            .map(|r| (r.record.id.to_string(), r.score))
            .collect();
        // Deduped, first edge's weight wins.
        assert_eq!(rows, vec![("note:1".to_string(), 0.2)]);
    }

    // -- CLOSURE + edge-property filters ------------------------------------

    fn closure_path(start: &str, steps: &[(MatchDirection, &str)]) -> Statement {
        Statement::Closure(MatchPath {
            start: RecordId::parse(start).unwrap(),
            steps: steps
                .iter()
                .map(|(direction, name)| MatchStep {
                    direction: *direction,
                    name: (*name).into(),
                    edge_props: None,
                })
                .collect(),
        })
    }

    fn relate_with_props(
        from: &str,
        name: &str,
        to: &str,
        props: BTreeMap<String, Value>,
    ) -> Statement {
        Statement::Relate(RelationEdge {
            from: RecordId::parse(from).unwrap(),
            name: name.into(),
            to: RecordId::parse(to).unwrap(),
            created_at: 0,
            weight: None,
            props,
        })
    }

    fn match_path_with_props(
        start: &str,
        steps: &[(MatchDirection, &str, Option<Filter>)],
    ) -> Statement {
        Statement::Match(MatchPath {
            start: RecordId::parse(start).unwrap(),
            steps: steps
                .iter()
                .map(|(direction, name, edge_props)| MatchStep {
                    direction: *direction,
                    name: (*name).into(),
                    edge_props: edge_props.clone(),
                })
                .collect(),
        })
    }

    #[test]
    fn closure_reaches_transitive_neighborhood() {
        let mut db = Database::default();
        db.execute(&[
            create("person", None),
            create("note", None),
            Statement::Insert(record("person:1", BTreeMap::new(), None)),
            Statement::Insert(record("person:2", BTreeMap::new(), None)),
            Statement::Insert(record("person:3", BTreeMap::new(), None)),
            Statement::Insert(record("note:9", BTreeMap::new(), None)),
            relate("person:1", "knows", "person:2", None),
            relate("person:2", "knows", "person:3", None),
            relate("person:3", "knows", "person:1", None), // cycle
            relate("person:1", "mentions", "note:9", None),
        ])
        .unwrap();

        let res = db
            .execute(&[closure_path("person:1", &[(MatchDirection::Out, "knows")])])
            .unwrap();
        assert!(matches!(res[0].kind, QueryKind::Closure(_)));
        // BFS first-visit: start (depth 0), person:2 (1), person:3 (2). The
        // cycle back to person:1 is deduped; the :mentions edge is excluded.
        let rows: Vec<(String, f32)> = res[0]
            .rows
            .iter()
            .map(|r| (r.record.id.to_string(), r.score))
            .collect();
        assert_eq!(
            rows,
            vec![
                ("person:1".to_string(), 0.0),
                ("person:2".to_string(), 1.0),
                ("person:3".to_string(), 2.0),
            ]
        );
    }

    #[test]
    fn closure_unknown_start_is_empty() {
        let mut db = Database::default();
        db.execute(&[
            create("person", None),
            Statement::Insert(record("person:1", BTreeMap::new(), None)),
        ])
        .unwrap();
        let res = db
            .execute(&[closure_path(
                "person:404",
                &[(MatchDirection::Out, "knows")],
            )])
            .unwrap();
        assert!(res[0].rows.is_empty());
    }

    #[test]
    fn match_edge_props_filter_restricts_traversal() {
        let mut db = Database::default();
        db.execute(&[
            create("person", None),
            create("note", None),
            Statement::Insert(record("person:1", BTreeMap::new(), None)),
            Statement::Insert(record("note:1", BTreeMap::new(), None)),
            Statement::Insert(record("note:2", BTreeMap::new(), None)),
            relate_with_props(
                "person:1",
                "mentions",
                "note:1",
                BTreeMap::from([("confidence".into(), Value::Float(0.9))]),
            ),
            relate_with_props(
                "person:1",
                "mentions",
                "note:2",
                BTreeMap::from([("confidence".into(), Value::Float(0.4))]),
            ),
        ])
        .unwrap();

        let filter = Filter::FieldEquals {
            field: "confidence".into(),
            value: Value::Float(0.9),
        };
        let res = db
            .execute(&[match_path_with_props(
                "person:1",
                &[(MatchDirection::Out, "mentions", Some(filter))],
            )])
            .unwrap();
        let ids: Vec<String> = res[0]
            .rows
            .iter()
            .map(|r| r.record.id.to_string())
            .collect();
        assert_eq!(
            ids,
            vec!["note:1"],
            "only the high-confidence edge is traversed"
        );
    }

    #[test]
    fn closure_edge_props_filter_limits_fixpoint() {
        let mut db = Database::default();
        db.execute(&[
            create("person", None),
            Statement::Insert(record("person:1", BTreeMap::new(), None)),
            Statement::Insert(record("person:2", BTreeMap::new(), None)),
            Statement::Insert(record("person:3", BTreeMap::new(), None)),
            relate_with_props(
                "person:1",
                "knows",
                "person:2",
                BTreeMap::from([("trust".into(), Value::Bool(true))]),
            ),
            relate_with_props(
                "person:2",
                "knows",
                "person:3",
                BTreeMap::from([("trust".into(), Value::Bool(false))]),
            ),
        ])
        .unwrap();

        let filter = Filter::FieldEquals {
            field: "trust".into(),
            value: Value::Bool(true),
        };
        let path = Statement::Closure(MatchPath {
            start: RecordId::parse("person:1").unwrap(),
            steps: vec![MatchStep {
                direction: MatchDirection::Out,
                name: "knows".into(),
                edge_props: Some(filter),
            }],
        });
        let res = db.execute(&[path]).unwrap();
        let ids: Vec<String> = res[0]
            .rows
            .iter()
            .map(|r| r.record.id.to_string())
            .collect();
        // Fixpoint stops at the untrusted edge: person:2 reached, person:3 not.
        assert_eq!(ids, vec!["person:1", "person:2"]);
    }
}

#[cfg(test)]
mod feedback_tests {
    use std::collections::BTreeMap;

    use nql_ir::{Id, RecordId, RelationEdge, Value};

    use super::{feedback_score, vote_counts, Store};

    fn rid(s: &str) -> RecordId {
        RecordId::parse(s).unwrap()
    }

    fn vote(from: &str, to: &str, value: i64, weight: f32, created_at: i64) -> RelationEdge {
        RelationEdge {
            from: rid(from),
            name: "voted".into(),
            to: rid(to),
            created_at,
            weight: Some(weight),
            props: BTreeMap::from([("value".into(), Value::Int(value))]),
        }
    }

    fn store_with_edges(edges: Vec<RelationEdge>) -> Store {
        Store {
            edges,
            ..Store::default()
        }
    }

    #[test]
    fn vote_counts_aggregates_up_down_net() {
        let s = store_with_edges(vec![
            vote("agent:1", "doc:1", 1, 1.0, 0),
            vote("agent:2", "doc:1", 1, 1.0, 0),
            vote("agent:3", "doc:1", -1, 1.0, 0),
            vote("agent:4", "doc:2", 1, 1.0, 0), // different target: ignored
        ]);
        let c = vote_counts(&s, &rid("doc:1"));
        assert_eq!(c.up, 2);
        assert_eq!(c.down, 1);
        assert_eq!(c.net, 1);
    }

    #[test]
    fn vote_counts_zero_when_no_votes() {
        let s = store_with_edges(vec![]);
        let c = vote_counts(&s, &rid("doc:1"));
        assert_eq!((c.up, c.down, c.net), (0, 0, 0));
    }

    #[test]
    fn feedback_score_prefers_recent_positive() {
        // recent +1 edges outweigh old -1 edge
        let s = store_with_edges(vec![
            vote("agent:a", "doc:1", 1, 1.0, 100),
            vote("agent:b", "doc:1", 1, 1.0, 200),
            vote("agent:c", "doc:1", -1, 1.0, 0),
        ]);
        let recent = feedback_score(&s, &rid("doc:1"));
        let old = feedback_score(&s, &rid("doc:2")); // no votes
        assert!(
            recent > old,
            "recent positives should score higher than none"
        );
        assert!(recent > 0.0);
    }

    #[test]
    fn feedback_score_deterministic() {
        let s = store_with_edges(vec![
            vote("agent:a", "doc:1", 1, 0.5, 10),
            vote("agent:b", "doc:1", -1, 0.9, 20),
        ]);
        let s2 = s.clone();
        assert_eq!(
            feedback_score(&s, &rid("doc:1")),
            feedback_score(&s2, &rid("doc:1"))
        );
    }

    #[test]
    fn non_voted_edges_ignored() {
        let e = RelationEdge {
            name: "mentions".into(), // not a vote
            ..vote("agent:a", "doc:1", 1, 1.0, 0)
        };
        let s = store_with_edges(vec![e]);
        let c = vote_counts(&s, &rid("doc:1"));
        assert_eq!(c.net, 0);
    }

    #[test]
    fn id_imports_work() {
        // guard: Id import stays usable (numeric id construction)
        let _ = Id::Num(1);
    }
}

#[cfg(test)]
mod bm25_engine_tests {
    use std::collections::BTreeMap;

    use nql_ir::{Filter, Plan, Record, RecordId, Select, Statement, Value};

    use crate::Database;

    fn record(id: &str, body: BTreeMap<String, Value>) -> Record {
        Record {
            id: RecordId::parse(id).unwrap(),
            body,
            embedding: None,
            created_at: 0,
        }
    }

    fn str_(s: &str) -> Value {
        Value::Str(s.to_string())
    }

    fn create(table: &str) -> Statement {
        Statement::CreateTable {
            table: table.to_string(),
            vector_dim: None,
        }
    }

    fn bm25_select(table: &str, field: &str, query: &str, k: Option<usize>) -> Statement {
        Statement::Select(Select {
            table: table.to_string(),
            filter: Some(Filter::Bm25 {
                field: field.to_string(),
                query: query.to_string(),
                k,
            }),
            ..Select::default()
        })
    }

    #[test]
    fn bm25_ranks_term_dense_above_sparse_and_k_limits_rows() {
        let mut db = Database::default();
        db.execute(&[
            create("doc"),
            Statement::Insert(record(
                "doc:dense",
                BTreeMap::from([("text".into(), str_("rust rust rust rust rust"))]),
            )),
            Statement::Insert(record(
                "doc:sparse",
                BTreeMap::from([("text".into(), str_("rust is a systems language"))]),
            )),
            Statement::Insert(record(
                "doc:other",
                BTreeMap::from([("text".into(), str_("completely unrelated topic"))]),
            )),
        ])
        .unwrap();

        let res = db
            .execute(&[bm25_select("doc", "text", "rust", None)])
            .unwrap();
        let ids: Vec<_> = res[0]
            .rows
            .iter()
            .map(|r| r.record.id.to_string())
            .collect();
        assert_eq!(ids, ["doc:dense", "doc:sparse", "doc:other"]);
        assert!(res[0].rows[0].score > res[0].rows[1].score);
        assert!(res[0].rows[1].score > res[0].rows[2].score);
        assert_eq!(res[0].rows[2].score, 0.0); // no matching term

        // `k` caps the number of returned rows, densest first.
        let res = db
            .execute(&[bm25_select("doc", "text", "rust", Some(1))])
            .unwrap();
        assert_eq!(res[0].rows.len(), 1);
        assert_eq!(res[0].rows[0].record.id.to_string(), "doc:dense");
    }

    #[test]
    fn bm25_returns_only_rows_of_the_selected_table() {
        let mut db = Database::default();
        db.execute(&[
            create("note"),
            create("log"),
            Statement::Insert(record(
                "note:1",
                BTreeMap::from([("text".into(), str_("meeting notes about rust"))]),
            )),
            Statement::Insert(record(
                "log:1",
                BTreeMap::from([("text".into(), str_("rust build log entry"))]),
            )),
            Statement::Insert(record(
                "log:2",
                BTreeMap::from([("text".into(), str_("rust rust rust"))]),
            )),
        ])
        .unwrap();

        let res = db
            .execute(&[bm25_select("note", "text", "rust", None)])
            .unwrap();
        let ids: Vec<_> = res[0]
            .rows
            .iter()
            .map(|r| r.record.id.to_string())
            .collect();
        assert_eq!(ids, ["note:1"], "only `note` table rows are returned");
        assert!(res[0].rows[0].score > 0.0);

        let res = db
            .execute(&[bm25_select("log", "text", "rust", None)])
            .unwrap();
        let ids: Vec<_> = res[0]
            .rows
            .iter()
            .map(|r| r.record.id.to_string())
            .collect();
        assert_eq!(ids, ["log:2", "log:1"]);
    }

    #[test]
    fn bm25_empty_query_returns_all_rows_with_zero_score() {
        let mut db = Database::default();
        db.execute(&[
            create("doc"),
            Statement::Insert(record(
                "doc:1",
                BTreeMap::from([("text".into(), str_("hello"))]),
            )),
            Statement::Insert(record(
                "doc:2",
                BTreeMap::from([("text".into(), str_("world"))]),
            )),
        ])
        .unwrap();

        let res = db.execute(&[bm25_select("doc", "text", "", None)]).unwrap();
        let ids: Vec<_> = res[0]
            .rows
            .iter()
            .map(|r| r.record.id.to_string())
            .collect();
        // Empty query: every row is returned (BTree key order), all scored 0.
        assert_eq!(ids, ["doc:1", "doc:2"]);
        assert!(res[0].rows.iter().all(|r| r.score == 0.0));
    }

    #[test]
    fn bm25_missing_or_non_str_field_scores_zero_and_orders_last() {
        let mut db = Database::default();
        db.execute(&[
            create("doc"),
            Statement::Insert(record(
                "doc:hit",
                BTreeMap::from([("text".into(), str_("needle in the haystack"))]),
            )),
            Statement::Insert(record(
                "doc:missing",
                BTreeMap::from([("other".into(), str_("needle"))]),
            )),
            Statement::Insert(record(
                "doc:nonstr",
                BTreeMap::from([("text".into(), Value::Int(7))]),
            )),
        ])
        .unwrap();

        let res = db
            .execute(&[bm25_select("doc", "text", "needle", None)])
            .unwrap();
        let ids: Vec<_> = res[0]
            .rows
            .iter()
            .map(|r| r.record.id.to_string())
            .collect();
        assert_eq!(ids, ["doc:hit", "doc:missing", "doc:nonstr"]);
        assert!(res[0].rows[0].score > 0.0);
        // Missing / non-Str field records are returned (Bm25 never prunes)
        // but score 0.0 and therefore sort below the hit.
        assert_eq!(res[0].rows[1].score, 0.0);
        assert_eq!(res[0].rows[2].score, 0.0);
    }

    #[test]
    fn bm25_same_plan_on_equal_stores_yields_equal_results() {
        let plan: Plan = vec![
            create("doc"),
            Statement::Insert(record(
                "doc:1",
                BTreeMap::from([("text".into(), str_("the quick brown fox jumps"))]),
            )),
            Statement::Insert(record(
                "doc:2",
                BTreeMap::from([("text".into(), str_("lazy dog fox"))]),
            )),
            Statement::Insert(record(
                "doc:3",
                BTreeMap::from([("text".into(), str_("nothing to see here"))]),
            )),
            bm25_select("doc", "text", "fox dog", None),
        ];

        let run = || {
            let mut db = Database::default();
            db.execute(&plan).unwrap()
        };

        let a = run();
        let b = run();
        assert_eq!(a, b);
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
        // Sanity: ranking is non-trivial and stable across runs.
        let scores: Vec<f32> = a[0].rows.iter().map(|r| r.score).collect();
        assert_eq!(
            scores,
            b[0].rows.iter().map(|r| r.score).collect::<Vec<_>>()
        );
        assert!(scores.windows(2).all(|w| w[0] >= w[1]));
    }

    #[test]
    fn bm25_combines_with_limit_and_order_stays_score_first() {
        let mut db = Database::default();
        db.execute(&[
            create("doc"),
            Statement::Insert(record(
                "doc:1",
                BTreeMap::from([("text".into(), str_("alpha alpha alpha"))]),
            )),
            Statement::Insert(record(
                "doc:2",
                BTreeMap::from([("text".into(), str_("alpha alpha"))]),
            )),
        ])
        .unwrap();

        // Explicit LIMIT 1 + Bm25: score ordering wins, then the cap applies.
        let mut sel = match bm25_select("doc", "text", "alpha", None) {
            Statement::Select(s) => s,
            _ => unreachable!(),
        };
        sel.limit = Some(1);
        let res = db.execute(&[Statement::Select(sel)]).unwrap();
        assert_eq!(res[0].rows.len(), 1);
        assert_eq!(res[0].rows[0].record.id.to_string(), "doc:1");
    }
}
