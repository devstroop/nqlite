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

use nql_ir::{Filter, Order, Record, RecordId, Select, Statement, Store};

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

/// The result of one `SELECT` statement inside a plan.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryResult {
    /// The select that produced this result.
    pub select: Select,
    /// Matching rows, ordered per the select's order/kNN semantics.
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
                select: sel.clone(),
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

    let mut rows: Vec<ScoredRecord> = candidates
        .into_iter()
        .map(|record| {
            let score = compute_score(store, sel, &record, knn_sims.as_ref());
            ScoredRecord { record, score }
        })
        .collect();

    order_rows(&mut rows, sel);

    if let Some(limit) = effective_limit(sel) {
        rows.truncate(limit);
    }
    rows
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
    // any other explicit order sorts by its score. With no order and no kNN,
    // rows stay in BTree key order (already sorted by RecordId).
    if sel.order.is_some() || sel.knn.is_some() {
        rows.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.record.id.cmp(&b.record.id))
        });
    }
}

/// The effective row cap: the smaller of `knn.k` (when present) and
/// `select.limit`.
fn effective_limit(sel: &Select) -> Option<usize> {
    match (sel.knn.as_ref().map(|k| k.k), sel.limit) {
        (Some(k), Some(l)) => Some(k.min(l)),
        (Some(k), None) => Some(k),
        (None, limit) => limit,
    }
}

/// Apply the select's field filter. `FieldEquals` uses the derived `PartialEq`
/// on [`Value`] (exact, deterministic equality); `HasEmbedding` requires a
/// non-`None` embedding.
fn matches_filter(rec: &Record, filter: Option<&Filter>) -> bool {
    match filter {
        None => true,
        Some(Filter::HasEmbedding) => rec.embedding.is_some(),
        Some(Filter::FieldEquals { field, value }) => rec.body.get(field) == Some(value),
    }
}

/// Compute the per-row match score, which doubles as the sort key for
/// score-based orders:
///
/// - `Order::Score` → Laplace-smoothed mean of the `:voted` edge weights
///   pointing at the record: `(sum + 1) / (n + 2)` — `0.5` with zero votes.
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
) -> f32 {
    let similarity = match &sel.knn {
        Some(_) => match knn_sims {
            Some(sims) => sims.get(&rec.id).copied().unwrap_or(0.0),
            None => 0.0,
        },
        None => 0.0,
    };

    match sel.order.as_ref() {
        Some(Order::Score) => score_of(store, rec),
        Some(Order::Salience) if sel.knn.is_some() => {
            0.7 * similarity + 0.3 * score_of(store, rec).clamp(0.0, 1.0)
        }
        Some(Order::Salience) => score_of(store, rec).clamp(0.0, 1.0),
        _ => similarity,
    }
}

/// Laplace-smoothed mean of `:voted` edge weights on the record.
///
/// A vote edge is any edge with `name == ":voted"` pointing **to** the record;
/// `weight` defaults to `1.0` when absent (an upvote). The estimate
/// `(sum + 1) / (n + 2)` starts at `0.5` with zero votes and moves toward the
/// observed mean as votes accumulate.
fn score_of(store: &Store, rec: &Record) -> f32 {
    let mut sum = 0.0f32;
    let mut n = 0usize;
    for edge in &store.edges {
        if edge.name == ":voted" && edge.to == rec.id {
            sum += edge.weight.unwrap_or(1.0);
            n += 1;
        }
    }
    (sum + 1.0) / (n as f32 + 2.0)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use nql_ir::{Plan, RecordId, RelationEdge, Value};

    use super::*;
    use crate::{Database, Knn};

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
                name: ":voted".into(),
                to: RecordId::parse("post:1").unwrap(),
                created_at: 1,
                weight: Some(1.0),
                props: BTreeMap::new(),
            }),
            Statement::Relate(RelationEdge {
                from: RecordId::parse("user:voter").unwrap(),
                name: ":voted".into(),
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
                name: ":voted".into(),
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
                name: ":voted".into(),
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
                name: ":voted".into(),
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
}
