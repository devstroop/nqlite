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
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use nql_ir::{
    Aggregate, CmpOp, Filter, Id, MatchDirection, MatchPath, Order, Record, RecordId, RelationEdge,
    Select, SnapshotState, Statement, Store, Value, VoteCounts,
};
use std::borrow::Cow;

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

/// A pipeline row that *borrows* its record — `run_select`'s working form.
///
/// Filtering, scoring, ordering, offset and limit all run over these
/// references; only the ≤limit rows that survive are cloned into owned
/// [`ScoredRecord`]s (issue #144, L1): candidates must never be deep-cloned
/// wholesale per query — a dim-64 embedding plus body strings made that
/// `.collect()` the single largest cost of an exact kNN SELECT.
#[derive(Debug, Clone, Copy)]
struct ScoredRef<'a> {
    record: &'a Record,
    score: f32,
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
    /// A `HISTORY SINCE <ts>` delta read (issue #118) — one row per mutation
    /// after the cutoff (rows AND edges), labeled with the cutoff.
    History { since: i64 },
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
/// collecting one [`QueryResult`] per query statement (`SELECT`, `MATCH`,
/// `CLOSURE`; DML/DDL statements contribute nothing to the output).
///
/// When `cache` is `Some`, kNN SELECTs memoize their whole-table vector
/// index against it (issue #144, L2); `None` keeps the build-per-query
/// behavior. The cache belongs to the root `store` argument — sub-store
/// (`MEMORY`) reads never see it (see [`execute_in_context`]).
pub fn execute_plan(
    store: &mut Store,
    plan: &[Statement],
    mut cache: Option<&mut IndexCache>,
    snaps: Option<&SnapshotRing>,
) -> Result<Vec<QueryResult>> {
    let mut results = Vec::new();
    let mut current_memory: Option<String> = None;
    for stmt in plan {
        if let Some(res) = execute_in_context(
            store,
            stmt,
            &mut current_memory,
            cache.as_deref_mut(),
            snaps,
        )? {
            results.push(res);
        }
    }
    Ok(results)
}

/// Execute a single [`Statement`] within a plan's memory context.
///
/// `MEMORY <name>` switches the context (creating the named memory lazily);
/// every other statement is executed against the current memory's store (the
/// root store when no `MEMORY` statement has run yet). This is the seam used
/// by both [`execute_plan`] and WAL replay, so memory scoping survives
/// reopen. A read-only plan run entirely inside a memory sees that memory's
/// records, edges, and history only.
pub fn execute_in_context(
    store: &mut Store,
    stmt: &Statement,
    current_memory: &mut Option<String>,
    cache: Option<&mut IndexCache>,
    snaps: Option<&SnapshotRing>,
) -> Result<Option<QueryResult>> {
    if let Statement::Memory { name } = stmt {
        store.memories.entry(name.clone()).or_default();
        *current_memory = Some(name.clone());
        return Ok(None);
    }
    // WAL plan-boundary marker (issue #109): the flat write-ahead log has no
    // other way to say "the plan ended here" — reset the context so later
    // frames replay at root, matching execute_plan's fresh-plan start.
    if let Statement::ContextReset = stmt {
        *current_memory = None;
        return Ok(None);
    }
    match current_memory {
        Some(name) => {
            let memory = store.memories.get_mut(name).expect("memory created above");
            // Sub-stores never share the root's memo (issue #144, L2): they
            // live in a `BTreeMap` whose values can move, and they are
            // different stores — pass `None` and keep building per query.
            // Same rule for replay snapshots (issue #166): a root snapshot
            // is the wrong base for a memory replay — memories keep full
            // replay until per-memory rings exist.
            execute_statement(memory, stmt, None, None)
        }
        None => execute_statement(store, stmt, cache, snaps),
    }
}

/// Execute a single [`Statement`] against `store`, mutating it for DDL/DML and
/// returning a [`QueryResult`] for `SELECT`s (`None` otherwise). `cache` is
/// the optional kNN index memo (issue #144, L2) — see [`execute_plan`];
/// `snaps` is the optional replay-snapshot ring (issue #166).
pub fn execute_statement(
    store: &mut Store,
    stmt: &Statement,
    cache: Option<&mut IndexCache>,
    snaps: Option<&SnapshotRing>,
) -> Result<Option<QueryResult>> {
    match stmt {
        Statement::Memory { name } => Err(Error::MemoryWithoutContext { name: name.clone() }),
        // WAL-only marker: intercepted by `execute_in_context` before this
        // point; the arm exists for exhaustive matching (and `replay_as_of`,
        // whose per-store history never contains markers).
        Statement::ContextReset => Ok(None),
        // History compaction (issue #95): PRUNE snapshots the current state
        // (memories depth-first) and keeps only declarations + the snapshot.
        Statement::PruneHistory => {
            prune_history(store);
            Ok(None)
        }
        // History-compaction base (issue #95): only replay executes this — it
        // installs the state the pruned prefix would have reconstructed, and
        // the statements after it replay on top exactly as they did
        // originally. Never in plans or the WAL.
        Statement::Snapshot(state) => {
            *store = state.as_ref().clone().into_store();
            Ok(None)
        }
        // Exact delta read (issue #118): every mutation strictly after the
        // cutoff, as one row per entry — the sync consumer's alternative to
        // two full `AS OF` replays (and blind to nothing: edges included).
        Statement::HistorySince(since) => {
            let rows = history_since(store, *since)?;
            Ok(Some(QueryResult {
                kind: QueryKind::History { since: *since },
                rows,
            }))
        }
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
            // The tables index carries EVERY declaration (with or without a
            // dim) — the seeding source that replaces history scans (issue
            // #133 step 1).
            store.tables.insert(table.clone(), *vector_dim);
            store.log_mutation(stmt);
            Ok(None)
        }
        Statement::Insert(rec) => {
            validate_embedding(store, rec)?;
            // Clock created_at to the mutation timestamp this INSERT is about
            // to receive — but only when unset (issue #107): the parser
            // always leaves 0 ("Engine clocks created_at per-transaction"),
            // and without a stamp `ORDER BY ::recency` degenerates to id
            // order. Explicit IR-provided values are honored as-is.
            // `log_mutation` below does `clock += 1`, so `clock + 1` *is*
            // this statement's timestamp — WAL/AS OF replay re-derives the
            // same stamps from statement order (and passes explicit values
            // through unchanged), keeping the determinism contract.
            let mut rec = rec.clone();
            if rec.created_at == 0 {
                rec.created_at = store.clock + 1;
            }
            store.insert(rec);
            store.log_mutation(stmt);
            Ok(None)
        }
        Statement::Relate(edge) => {
            // Same rule as Insert (issue #107): edge.created_at drives
            // ::feedback's decay. nql `SET created_at = ...` lands in `props`
            // (only `weight` is special-cased), so the field arrives as 0
            // from every nql/MCP path and gets stamped; explicit IR values
            // pass through.
            let mut edge = edge.clone();
            if edge.created_at == 0 {
                edge.created_at = store.clock + 1;
            }
            store.edges.push(edge);
            store.log_mutation(stmt);
            Ok(None)
        }
        Statement::Forget { id } => {
            store.records.remove(id);
            // Drop every edge incident to the forgotten record, either end.
            store.edges.retain(|e| &e.from != id && &e.to != id);
            store.log_mutation(stmt);
            Ok(None)
        }
        Statement::Select(sel) => {
            let rows = run_select(store, sel, cache, snaps)?;
            Ok(Some(QueryResult {
                kind: QueryKind::Select(sel.clone()),
                rows,
            }))
        }
        Statement::Match(path) => {
            let target = temporal_target(store, path.as_of, snaps)?;
            let rows = run_match(&target, path);
            Ok(Some(QueryResult {
                kind: QueryKind::Match(path.clone()),
                rows,
            }))
        }
        Statement::MatchCount(path) => {
            // Walk-count mode (issue #94): one `{"count": n}` row; the query
            // kind stays `Match` so transports label the result unchanged.
            let table = path.start.table.clone();
            let target = temporal_target(store, path.as_of, snaps)?;
            let n = run_match_count(&target, path);
            Ok(Some(QueryResult {
                kind: QueryKind::Match(path.clone()),
                rows: vec![count_row(&table, n)],
            }))
        }
        Statement::Closure(path) => {
            let target = temporal_target(store, path.as_of, snaps)?;
            let rows = run_closure(&target, path);
            Ok(Some(QueryResult {
                kind: QueryKind::Closure(path.clone()),
                rows,
            }))
        }
    }
}

/// The store a temporal graph read runs against (spec §2.7, issue #92):
/// replayed to `as_of` when present — the exact machinery
/// `SELECT ... AS OF` uses — or the current store otherwise. `MATCH`,
/// `MATCH ... COUNT`, and `CLOSURE` all go through here, so a historical
/// traversal sees exactly the records and edges that existed at the cutoff.
fn temporal_target<'a>(
    store: &'a Store,
    as_of: Option<i64>,
    snaps: Option<&SnapshotRing>,
) -> Result<Cow<'a, Store>> {
    match as_of {
        Some(cutoff) => Ok(Cow::Owned(replay_as_of(store, cutoff, snaps)?)),
        None => Ok(Cow::Borrowed(store)),
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
/// rather than an inline cosine scan. With an [`IndexCache`] supplied
/// (issue #144, L2), the whole-table build is memoized per store-version;
/// otherwise — and always for `AS OF` reads and pruning filters — it is
/// rebuilt from the filtered candidates on every call. Either way the
/// result stays a pure, deterministic function of `(store, select)`.
/// Candidates borrow the store's records end-to-end; only the ≤limit rows
/// that survive ordering are cloned (issue #144, L1). On the kNN-only path
/// (spec §2.3 output-cap invariant) the search and row build are windowed
/// to `OFFSET + cap` candidates (issue #144, L3). A temporal read
/// whose cutoff predates the store's history snapshot returns
/// [`Error::HistoryPruned`] (issue #95).
fn run_select(
    store: &Store,
    sel: &Select,
    cache: Option<&mut IndexCache>,
    snaps: Option<&SnapshotRing>,
) -> Result<Vec<ScoredRecord>> {
    // Temporal read (`AS OF T`): replay the mutation history up to the
    // cutoff into a fresh store and query THAT — the historical view is a
    // pure function of (history, T). Everything below runs against `target`.
    let replay_store;
    let target = match sel.as_of {
        Some(cutoff) => {
            replay_store = replay_as_of(store, cutoff, snaps)?;
            &replay_store
        }
        None => store,
    };

    // L2 memo scope (issue #144): current-state, whole-table reads only.
    // `AS OF` queries a fresh replay view (never shared with anyone), and a
    // pruning filter narrows the candidate set below the whole table — both
    // bypass the memo and keep building per query. `MEMORY` sub-stores
    // never receive a cache at all (see `execute_in_context`).
    let cache =
        if sel.as_of.is_none() && matches!(sel.filter.as_ref(), None | Some(Filter::Bm25 { .. })) {
            cache
        } else {
            None
        };

    // Candidates BORROW the store's records (issue #144, L1): filtering
    // never needs owned rows, and deep-cloning every match up front made
    // this `.collect()` the largest single cost of an exact kNN SELECT.
    // Owned `Record`s materialize only for the ≤limit surviving rows below.
    let candidates: Vec<&Record> = target
        .records
        .values()
        .filter(|r| r.id.table == sel.table)
        .filter(|r| matches_filter(r, sel.filter.as_ref()))
        .collect();

    // `SELECT COUNT(*)` (spec §2.3, issue #94): one `{"count": n}` row with
    // the number of records that passed the WHERE filter — computed before
    // scoring, ordering, offset/limit, and projection (those never affect the
    // count), and no kNN/BM25 index is built for it.
    if let Some(Aggregate::CountStar) = sel.aggregate {
        return Ok(vec![count_row(&sel.table, candidates.len() as u64)]);
    }

    // The index (memoized or rebuilt) is acquired once — both the windowed
    // kNN-only path and the full path rank through it (issue #144, L2/L3).
    let fallback;
    let index: Option<&dyn VectorIndex> = match sel.knn.as_ref() {
        Some(_) => Some(match cache {
            Some(memo) => {
                memo.get_or_build(target, &sel.table, || build_default_index(&candidates))
            }
            None => {
                fallback = build_default_index(&candidates);
                &*fallback
            }
        }),
        None => None,
    };

    // kNN-only window (spec §2.3 output-cap invariant, issue #144, L3): when
    // the query ranks by the mode's own score — kNN without `::bm25`, order
    // default or `::similarity` — only the top `OFFSET + cap` candidates can
    // ever be observed, so the search and the row build are windowed to that
    // size. Every other case (score-based/structural orders, bm25/hybrid)
    // ranks ALL rows and keeps the full path below, byte-identical to before.
    let window: Option<usize> = if sel.knn.is_some()
        && !matches!(sel.filter.as_ref(), Some(Filter::Bm25 { .. }))
        && matches!(sel.order.as_ref(), None | Some(Order::Similarity))
    {
        Some(
            sel.offset
                .unwrap_or(0)
                .saturating_add(effective_limit(sel).unwrap_or(0)),
        )
    } else {
        None
    };

    // Rank every embedded candidate against the query through the index.
    // Non-embedded records are absent from the index and fall back to a
    // similarity of `0.0`, exactly as the inline scan did. On a memo hit
    // the cached whole-table index IS what a rebuild would produce (same
    // records → same BTree contents → identical search output), so rows
    // are bit-identical either way (issue #144, L2). Windowed on the
    // kNN-only path, full corpus otherwise (hybrid's RRF ranks everything).
    let knn_sims: Option<BTreeMap<RecordId, f32>> = match (&sel.knn, &index) {
        (Some(knn), Some(index)) => Some(
            index
                .search(&knn.query, window.unwrap_or(candidates.len()))
                .into_iter()
                .collect(),
        ),
        _ => None,
    };

    // A `Filter::Bm25` turns the SELECT into lexical retrieval: build one
    // deterministic BM25 index over the filtered candidates' text field and
    // score every row with it (records missing the field / without a `Str`
    // value score 0.0). The index is rebuilt per call, so the result stays a
    // pure function of `(store, select)`.
    let bm25: Option<(Bm25Index, Vec<String>)> = match sel.filter.as_ref() {
        Some(Filter::Bm25 { field, query, .. }) => {
            let index = Bm25Index::new(field, candidates.iter().copied());
            let query_tokens = tokenize(query);
            Some((index, query_tokens))
        }
        _ => None,
    };

    // Hybrid retrieval (combined optimizer, M2): when the select carries BOTH
    // a kNN clause and a `::bm25` filter, fuse the two rankings with
    // reciprocal-rank fusion (RRF): every candidate gets `1/(K + rank)` per
    // list (K = 60, rank 1-based, tie-break by RecordId asc), summed across
    // the lexical and vector lists. Deterministic and scale-free — the fused
    // score is a pure function of `(store, select)`.
    let hybrid: Option<BTreeMap<RecordId, f32>> = match (sel.knn.as_ref(), &bm25) {
        (Some(_), Some((index, query_tokens))) => {
            let mut fused: BTreeMap<RecordId, f32> = BTreeMap::new();
            let mut lexical_ranked: Vec<(RecordId, f32)> = candidates
                .iter()
                .map(|r| {
                    let s = index.score(&r.id, query_tokens);
                    (r.id.clone(), s)
                })
                .collect();
            lexical_ranked.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            for (rank, (id, _)) in lexical_ranked.iter().enumerate() {
                *fused.entry(id.clone()).or_insert(0.0) += 1.0 / (60.0 + (rank + 1) as f32);
            }
            let mut vector_ranked: Vec<(RecordId, f32)> = candidates
                .iter()
                .map(|r| {
                    let s = knn_sims
                        .as_ref()
                        .and_then(|m| m.get(&r.id))
                        .copied()
                        .unwrap_or(0.0);
                    (r.id.clone(), s)
                })
                .collect();
            vector_ranked.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            for (rank, (id, _)) in vector_ranked.iter().enumerate() {
                *fused.entry(id.clone()).or_insert(0.0) += 1.0 / (60.0 + (rank + 1) as f32);
            }
            Some(fused)
        }
        _ => None,
    };

    let mut rows: Vec<ScoredRef> = match window {
        // Windowed build (spec §2.3 output-cap invariant): the observable
        // rows are exactly the top-`need` embedded (present in the windowed
        // sims map) plus the first `need` non-embedded by id (score `0.0` —
        // their fallback can outrank negative similarities, so they compete
        // for the window too). Every true-window row is in that pool: an
        // embedded row outside the index's top-`need` has ≥`need` embedded
        // rows ahead of it, and a non-embedded row beyond the first `need`
        // by id already has `need` equal-score rows ahead of it.
        Some(need) => {
            let mut rows = Vec::with_capacity(need.saturating_mul(2).min(candidates.len()));
            let mut nonemb = 0usize;
            for &record in &candidates {
                if record.embedding.is_some() {
                    if let Some(&score) = knn_sims.as_ref().and_then(|m| m.get(&record.id)) {
                        rows.push(ScoredRef { record, score });
                    }
                } else if nonemb < need {
                    rows.push(ScoredRef { record, score: 0.0 });
                    nonemb += 1;
                }
            }
            rows
        }
        // Full path: score every candidate (hybrid/BM25/order arms below
        // keep their all-rows semantics — unchanged).
        None => candidates
            .into_iter()
            .map(|record| {
                let score = compute_score(
                    target,
                    sel,
                    record,
                    knn_sims.as_ref(),
                    bm25.as_ref(),
                    hybrid.as_ref(),
                );
                ScoredRef { record, score }
            })
            .collect(),
    };

    // `ORDER BY <field>` typo guard (issue #117): if this query returns rows
    // but NO record of the table carries the key, the sort would be silently
    // all-equal (id order) — fail loudly instead. Empty results skip the
    // check: there is nothing to mis-sort.
    if let Some(Order::Field { key, .. }) = sel.order.as_ref() {
        if !rows.is_empty()
            && !target
                .records
                .values()
                .any(|r| r.id.table == sel.table && r.body.contains_key(key))
        {
            return Err(Error::UnknownSortField {
                field: key.clone(),
                table: sel.table.clone(),
            });
        }
    }

    order_rows(&mut rows, sel);

    // `OFFSET n` (spec §2.3, issue #94): skip the first n rows after
    // ordering; the limit / kNN-k / BM25-k caps then apply to what remains.
    if let Some(offset) = sel.offset {
        let skip = offset.min(rows.len());
        rows.drain(..skip);
    }

    if let Some(limit) = effective_limit(sel) {
        rows.truncate(limit);
    }

    // Materialize the survivors (issue #144, L1): at most `limit` deep
    // clones instead of one per candidate up front — same bytes, same
    // scores, same order.
    let mut rows: Vec<ScoredRecord> = rows
        .into_iter()
        .map(|r| ScoredRecord {
            record: r.record.clone(),
            score: r.score,
        })
        .collect();

    // Field projection (spec §2.3 step 8, issue #91): keep only the listed
    // body keys — presentation only, after all filtering/scoring/ordering, so
    // a projection can never change which rows rank. Missing keys are simply
    // absent (SQL-like); `SELECT *` (`fields == None`) is untouched.
    if let Some(fields) = &sel.fields {
        for row in &mut rows {
            row.record.body.retain(|k, _| fields.iter().any(|f| f == k));
        }
    }
    Ok(rows)
}

/// Replay the store's mutation history up to (and including) logical
/// timestamp `cutoff` into a fresh [`Store`], then return it. Deterministic:
/// history is append-only in execution order, and each replayed statement is
/// executed the same way it originally was — so the view is a pure function
/// of `(store.history, cutoff)` (or an [`Error::HistoryPruned`] when a
/// compaction snapshot postdates the cutoff, issue #95).
/// The timestamp of the history-compaction snapshot, when one exists
/// (issues #95/#118): temporal reads below this horizon are unavailable —/// the pruned prefix cannot be reconstructed.
fn compaction_horizon(store: &Store) -> Option<i64> {
    store
        .history
        .iter()
        .find_map(|(ts, stmt)| matches!(stmt, Statement::Snapshot(_)).then_some(*ts))
}

fn replay_as_of(store: &Store, cutoff: i64, snaps: Option<&SnapshotRing>) -> Result<Store> {
    // Compacted history (issue #95): a cutoff before the snapshot cannot be
    // reconstructed — the pruned prefix is gone. Fail loudly rather than
    // returning a partial (declarations-only) view.
    if let Some(snap_ts) = compaction_horizon(store) {
        if cutoff < snap_ts {
            return Err(Error::HistoryPruned {
                pruned_through: snap_ts,
            });
        }
    }
    // Recent-T fast path (issue #166): a cutoff at or past the current clock
    // covers the whole log, so the view IS the current state — return it
    // without replaying a single statement. Exact by construction (the live
    // store is the full replay result); callers only ever read the view.
    if cutoff >= store.clock {
        return Ok(store.clone());
    }
    // Replay accelerator (issue #166): start from the newest snapshot base
    // at or below the cutoff instead of the empty store. The base IS the
    // state replaying `1..=base_clock` produces (captured live, never
    // edited), so applying `(base_clock..cutoff]` yields exactly the
    // from-scratch view with fewer statements executed. `None`/miss keeps
    // the previous full replay.
    let (mut view, floor) = match snaps.and_then(|r| r.select(cutoff)) {
        Some((base_clock, base)) => (base.clone(), base_clock),
        None => (Store::default(), 0),
    };
    for (ts, stmt) in &store.history {
        if *ts > cutoff {
            break;
        }
        if *ts <= floor {
            continue;
        }
        // Replay is total on a valid store (mutating statements cannot fail
        // once their preconditions were met); a failure here would mean a
        // corrupt history, so panic loudly rather than silently truncate.
        let _ = execute_statement(&mut view, stmt, None, None).expect("history replay is total");
    }
    Ok(view)
}

/// History compaction (`PRUNE HISTORY`, issue #95): replace `store`'s history
/// with the `CreateTable` declaration statements it contained (at their
/// original timestamps — the only record of empty/dim-less table
/// declarations, issue #89; re-executing them is idempotent) plus a single
/// [`Statement::Snapshot`] entry at the current clock. Memories are pruned
/// depth-first first, so the embedded stores arrive already compact.
///
/// Deterministic (a pure function of the store) and bounded: history stops
/// growing by one entry per mutation forever. `AS OF` before the snapshot now
/// fails with [`Error::HistoryPruned`]; from the snapshot onward, replay
/// rebuilds the view without walking the pruned prefix.
fn prune_history(store: &mut Store) {
    for memory in store.memories.values_mut() {
        prune_history(memory);
    }
    let decls: Vec<(i64, Statement)> = store
        .history
        .iter()
        .filter(|(_, stmt)| matches!(stmt, Statement::CreateTable { .. }))
        .cloned()
        .collect();
    let state = SnapshotState {
        records: store.records.clone(),
        edges: store.edges.clone(),
        vector_dims: store.vector_dims.clone(),
        clock: store.clock,
        memories: store.memories.clone(),
        tables: store.tables.clone(),
    };
    let mut history = decls;
    history.push((store.clock, Statement::Snapshot(Box::new(state))));
    store.history = history;
}

/// `HISTORY SINCE <ts>` (issue #118): every mutation strictly after the
/// cutoff, in append (ts-ascending) order — one row per entry carrying the
/// mutation kind and its subject ids, so a sync consumer sees changed rows
/// **and** changed edges (plus tombstones) in one read instead of diffing
/// two full `AS OF` replays — a row-diff is blind to edge-only mutations.
///
/// Pure function of `(history, since)`: append-only order, deterministic
/// kinds/subjects. The `PRUNE HISTORY` retention horizon applies (a cutoff
/// below the snapshot cannot be answered — same `HistoryPruned` contract as
/// `AS OF`), and snapshot entries themselves are compaction bookkeeping,
/// never reported as mutations. Scoped stores (MEMORY blocks) return their
/// own deltas via the usual context routing.
fn history_since(store: &Store, since: i64) -> Result<Vec<ScoredRecord>> {
    if let Some(snap_ts) = compaction_horizon(store) {
        if since < snap_ts {
            return Err(Error::HistoryPruned {
                pruned_through: snap_ts,
            });
        }
    }
    let mut rows = Vec::new();
    for (ts, stmt) in &store.history {
        if *ts <= since {
            continue;
        }
        let mut body = BTreeMap::new();
        body.insert("ts".into(), Value::Int(*ts));
        let kind = match stmt {
            Statement::CreateTable { table, vector_dim } => {
                body.insert("table".into(), Value::Str(table.clone()));
                if let Some(dim) = vector_dim {
                    body.insert("dim".into(), Value::Int(*dim as i64));
                }
                "CREATE"
            }
            Statement::Insert(rec) => {
                body.insert("id".into(), Value::Str(rec.id.to_string()));
                "INSERT"
            }
            Statement::Relate(edge) => {
                body.insert("from".into(), Value::Str(edge.from.to_string()));
                body.insert("to".into(), Value::Str(edge.to.to_string()));
                body.insert("name".into(), Value::Str(edge.name.clone()));
                "RELATE"
            }
            Statement::Forget { id } => {
                body.insert("id".into(), Value::Str(id.to_string()));
                "FORGET"
            }
            // Compaction bookkeeping (issue #95): not a mutation — the
            // horizon guard above already covered its region.
            Statement::Snapshot(_) => continue,
            // Unreachable in a well-formed history (only the four mutations
            // above are ever `log_mutation`d) — labeled deterministically
            // instead of dropped, should that ever change.
            Statement::Memory { name } => {
                body.insert("name".into(), Value::Str(name.clone()));
                "MEMORY"
            }
            Statement::PruneHistory => "PRUNE",
            Statement::HistorySince(since) => {
                body.insert("since".into(), Value::Int(*since));
                "HISTORY_SINCE"
            }
            Statement::ContextReset => "CONTEXT_RESET",
            Statement::Select(_) => "SELECT",
            Statement::Match(_) | Statement::MatchCount(_) => "MATCH",
            Statement::Closure(_) => "CLOSURE",
        };
        body.insert("kind".into(), Value::Str(kind.into()));
        rows.push(ScoredRecord {
            record: Record {
                id: RecordId::new("history", Id::Str(ts.to_string())),
                body,
                embedding: None,
                created_at: *ts,
            },
            score: *ts as f32,
        });
    }
    Ok(rows)
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
            if !edge_name_matches(&edge.name, &step.name) || !frontier.contains(from_side) {
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

/// `MATCH ... COUNT` (spec §2.5, issue #94): the number of edge-path
/// INSTANCES (walks) matching the steps — parallel edges each count, so the
/// multiplicity between two records stays observable (the endpoint dedup
/// [`run_match`] applies for rows would hide it). At every step a node's
/// walk count accumulates `walks(from)` once per matching edge, so for a
/// single step the total is exactly the number of matching edges, and for
/// multiple steps it is the number of distinct walks.
///
/// Deterministic (u64 accumulation is order-independent), saturating (never
/// panics on overflow), and spec §2.5-consistent: a missing start yields 0
/// and dangling edges are skipped.
fn run_match_count(store: &Store, path: &MatchPath) -> u64 {
    if !store.records.contains_key(&path.start) {
        return 0;
    }
    // Frontier of node → number of walks reaching it.
    let mut frontier: BTreeMap<RecordId, u64> = BTreeMap::from([(path.start.clone(), 1)]);
    for step in &path.steps {
        let mut next: BTreeMap<RecordId, u64> = BTreeMap::new();
        for edge in &store.edges {
            let (from_side, to_side) = match step.direction {
                MatchDirection::Out => (&edge.from, &edge.to),
                MatchDirection::In => (&edge.to, &edge.from),
            };
            if !edge_name_matches(&edge.name, &step.name) {
                continue;
            }
            let Some(&walks) = frontier.get(from_side) else {
                continue;
            };
            if !matches_edge_props(edge, step.edge_props.as_ref()) {
                continue;
            }
            if !store.records.contains_key(to_side) {
                continue; // dangling edge: skip
            }
            let slot = next.entry(to_side.clone()).or_insert(0);
            *slot = slot.saturating_add(walks);
        }
        if next.is_empty() {
            return 0;
        }
        frontier = next;
    }
    frontier.values().copied().fold(0u64, u64::saturating_add)
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
///
/// Edges are pre-indexed by endpoint (adjacency lists, store order within a
/// vertex) so the walk touches only the frontier's incident edges instead of
/// scanning the whole edge list per BFS level (O(N²) for long chains
/// otherwise). Per-level tie-breaks become frontier-major (frontier vertices
/// in first-visit order, their edges in store order) instead of
/// store-interleaved — both orders are deterministic; single-vertex
/// frontiers (chains, trees) are identical either way.
fn run_closure(store: &Store, path: &MatchPath) -> Vec<ScoredRecord> {
    if !store.records.contains_key(&path.start) {
        return Vec::new();
    }

    // Adjacency index: incident edges per endpoint, in store order.
    let mut by_from: HashMap<&RecordId, Vec<&RelationEdge>> = HashMap::new();
    let mut by_to: HashMap<&RecordId, Vec<&RelationEdge>> = HashMap::new();
    for edge in &store.edges {
        by_from.entry(&edge.from).or_default().push(edge);
        by_to.entry(&edge.to).or_default().push(edge);
    }

    // First-visit (BFS) order lives in `visited`; membership mirrors it in a
    // HashSet so duplicate-reach checks stay O(1) as the walk grows.
    let mut visited: Vec<RecordId> = vec![path.start.clone()];
    let mut visited_set: HashSet<RecordId> = HashSet::from([path.start.clone()]);
    let mut depth_of: BTreeMap<RecordId, u32> = BTreeMap::new();
    depth_of.insert(path.start.clone(), 0);

    // BFS frontier: everything reached at the previous depth (start at depth 0).
    let mut frontier: Vec<RecordId> = vec![path.start.clone()];
    let mut next_depth = 1u32;

    for step in &path.steps {
        // Expand the current frontier to fixpoint along this step's edge.
        loop {
            let mut newly_reached: Vec<RecordId> = Vec::new();
            for from in &frontier {
                let incident = match step.direction {
                    MatchDirection::Out => by_from.get(from),
                    MatchDirection::In => by_to.get(from),
                };
                let Some(edges) = incident else {
                    continue;
                };
                for edge in edges {
                    let to_side = match step.direction {
                        MatchDirection::Out => &edge.to,
                        MatchDirection::In => &edge.from,
                    };
                    if !edge_name_matches(&edge.name, &step.name) {
                        continue;
                    }
                    if !matches_edge_props(edge, step.edge_props.as_ref()) {
                        continue;
                    }
                    if !store.records.contains_key(to_side) {
                        continue; // dangling edge: skip
                    }
                    if visited_set.insert(to_side.clone()) {
                        visited.push(to_side.clone());
                        newly_reached.push(to_side.clone());
                        depth_of.insert(to_side.clone(), next_depth);
                    }
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

/// Apply a step's optional edge-property filter (spec §2.5): any *field*
/// predicate — equality, comparison, `IN`, `BETWEEN` (issue #93) — evaluated
/// against the edge's props via [`matches_field_pred`]. `None` accepts every
/// edge.
fn matches_edge_props(edge: &RelationEdge, filter: Option<&Filter>) -> bool {
    match filter {
        None => true,
        // Embedding presence and BM25 scoring are not edge-props predicates;
        // the parser never produces them here. Defensive, not a panic.
        Some(Filter::HasEmbedding | Filter::Bm25 { .. }) => false,
        // All-of over terms (issue #125): each term re-enters this fn, so a
        // HasEmbedding term inside an And is still a non-match on an edge.
        Some(Filter::And(terms)) => terms.iter().all(|t| matches_edge_props(edge, Some(t))),
        Some(f) => matches_field_pred(&edge.props, f),
    }
}

/// Memo of the whole-table vector index for one root [`Store`] (issue #144, L2).
///
/// Rebuilding the index (embedding re-clone + BTree inserts) on every kNN
/// query cost ~79 ms of a 213 ms exact kNN @100k — L2 keeps one built
/// index per table instead. Invalidation is **lazy and total**: every
/// record-affecting statement the engine executes goes through
/// [`Store::log_mutation`] (`clock += 1`), so an entry is served only while
/// BOTH the store's address and its `clock` match the ones it was built
/// against — there is no push-side invalidation to miss (the Zig ingest
/// lesson: missed seams fail loudly here by rebuilding, never by serving
/// stale bytes).
///
/// Correctness rules, enforced by the call chain:
///
/// - **Root store only.** Sub-stores (`MEMORY <name>`) live in a `BTreeMap`
///   whose values can move; `execute_in_context` passes `None` for them.
/// - **Current state only.** `AS OF` replays into a fresh view and
///   `run_select` drops the memo whenever `sel.as_of` is set.
/// - **Whole-table candidate sets only.** The memo equals
///   `build_default_index` over every record of the table — exactly the
///   candidate set for `filter == None` and `Filter::Bm25` (both pass every
///   record); any pruning filter still builds per query.
///
/// `PRUNE HISTORY` never changes records (same clock ⇒ still valid);
/// `Statement::Snapshot` installs happen only during replay, into stores
/// this memo never serves. Passed as `Option<&mut IndexCache>` down the
/// execute chain — `None` keeps the previous build-per-query behavior.
#[derive(Default)]
pub struct IndexCache {
    /// Address of the store these indexes were built from.
    store_addr: usize,
    /// That store's `clock` at build time.
    clock: i64,
    /// Table name → index over the table's embedded records.
    by_table: HashMap<String, Box<dyn VectorIndex>>,
}

impl std::fmt::Debug for IndexCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `Box<dyn VectorIndex>` is not `Debug`; expose what invalidation
        // depends on plus the cached table names.
        f.debug_struct("IndexCache")
            .field("store_addr", &self.store_addr)
            .field("clock", &self.clock)
            .field("tables", &self.by_table.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl IndexCache {
    /// Drop everything cached for a different store or a stale `clock`.
    fn sync(&mut self, store: &Store) {
        let addr = store as *const Store as usize;
        if self.store_addr != addr || self.clock != store.clock {
            self.by_table.clear();
            self.store_addr = addr;
            self.clock = store.clock;
        }
    }

    /// The memoized index for `table`, building it once via `build` on a miss.
    ///
    /// The returned reference is tied to `&mut self`; a caller that keeps it
    /// across mutations gets exactly what a rebuild would have produced at
    /// that clock, because any mutation invalidates first.
    pub fn get_or_build(
        &mut self,
        store: &Store,
        table: &str,
        build: impl FnOnce() -> Box<dyn VectorIndex>,
    ) -> &dyn VectorIndex {
        self.sync(store);
        if !self.by_table.contains_key(table) {
            self.by_table.insert(table.to_string(), build());
        }
        &*self.by_table[table]
    }
}

/// Build the default vector index (exact brute-force) over the embeddings of
/// the given candidate records.
///
/// This is the engine's swap point for alternative [`VectorIndex`]
/// implementations (e.g. the feature-gated, approximate `HnswVectorIndex`):
/// swap the concrete type here and the rest of the engine is unchanged.
///
/// Candidates arrive as borrowed records (issue #144, L1). The engine
/// memoizes this build per table through [`IndexCache`] (issue #144, L2),
/// so the `upsert` clones run once per store-version instead of once per
/// query.
fn build_default_index(records: &[&Record]) -> Box<dyn VectorIndex> {
    let mut index = BruteForceVectorIndex::default();
    for r in records {
        if let Some(emb) = &r.embedding {
            index.upsert(r.id.clone(), emb.clone());
        }
    }
    Box::new(index)
}

/// Mutations between persisted replay snapshots (issue #166): the ring
/// holds postcard-cheap history-stripped clones, so worst-case replay is
/// bounded by this instead of the full history length. Tuned against the
/// temporal microbench (`temporal_cold`/`temporal_warm` in benches/bench.rs):
/// a clone costs ~100 ms on a 100k-record store, amortized to ~5 µs per
/// mutation at this cadence — noise next to WAL fsync on persistent stores.
pub const SNAPSHOT_EVERY: i64 = 20000;
/// Retained snapshots per database handle (issue #166): the newest wins for
/// recent cutoffs; older ones cover older cutoffs until evicted.
pub const SNAPSHOT_RING_CAP: usize = 2;

/// In-memory ring of replay-accelerator snapshots (issue #166): full store
/// states (history stripped — replay consumes the live log, never the
/// snapshot's) keyed by clock. A pure cache in the [`IndexCache`] spirit: a
/// snapshot at clock C is by construction the state replaying mutations
/// `1..=C` produces, so replaying `(C..cutoff]` on top yields exactly the
/// from-scratch view — a miss or a loss only costs speed, never bytes.
/// Passed as `Option<&SnapshotRing>` down the execute chain — `None` keeps
/// the previous replay-from-scratch behavior everywhere (WAL replay,
/// memory sub-stores, direct engine callers).
#[derive(Debug, Default)]
pub struct SnapshotRing {
    snaps: VecDeque<(i64, Store)>,
}

impl SnapshotRing {
    /// Newest snapshot at or below `cutoff`, if any.
    pub fn select(&self, cutoff: i64) -> Option<(i64, &Store)> {
        self.snaps
            .iter()
            .rev()
            .find(|(clock, _)| *clock <= cutoff)
            .map(|(clock, store)| (*clock, store))
    }

    /// Record the current store as a snapshot at `clock` (history stripped —
    /// see above). Resets on clock regression (a different lineage must never
    /// inherit bases); evicts oldest beyond [`SNAPSHOT_RING_CAP`].
    pub fn push(&mut self, clock: i64, store: &Store) {
        if self.snaps.back().is_some_and(|(c, _)| clock <= *c) {
            self.snaps.clear();
        }
        self.snaps.push_back((clock, strip_history(store)));
        while self.snaps.len() > SNAPSHOT_RING_CAP {
            self.snaps.pop_front();
        }
    }
}

/// Clone a store without its mutation history (issue #166): snapshot bases
/// carry state only — replay always consumes the live log, so embedded
/// histories would be dead weight (roughly half the bytes on history-heavy
/// stores). Built field-by-field so the history (and sub-histories) are
/// never cloned in the first place — memories are stripped recursively for
/// the same reason.
fn strip_history(store: &Store) -> Store {
    Store {
        records: store.records.clone(),
        edges: store.edges.clone(),
        vector_dims: store.vector_dims.clone(),
        clock: store.clock,
        history: Vec::new(),
        memories: store
            .memories
            .iter()
            .map(|(name, sub)| (name.clone(), strip_history(sub)))
            .collect(),
        tables: store.tables.clone(),
    }
}

/// Apply the select's deterministic ordering. Stable sorts guarantee equal
/// keys keep their (BTree) input order; we additionally tie-break by
/// ascending [`RecordId`] so the final order is total and reproducible.
///
/// Runs over borrowed pipeline rows ([`ScoredRef`]) — the same field names
/// as [`ScoredRecord`] keep this one copy of the ordering semantics for both
/// forms (issue #144, L1).
fn order_rows(rows: &mut [ScoredRef<'_>], sel: &Select) {
    // `ORDER BY <field> [DESC]` (issue #117): explicit structural sort, same
    // precedence as `::recency` — it wins over the score-based modes below.
    // Absent/explicit-null fields rank as `null` (lowest) under
    // `Value::cmp_total`; `desc` reverses the key ONLY — ties always keep
    // ascending RecordId (the §2.1 total order holds in both directions).
    if let Some(Order::Field { key, desc }) = sel.order.as_ref() {
        let null = Value::Null;
        rows.sort_by(|a, b| {
            let ka = a.record.body.get(key).unwrap_or(&null);
            let kb = b.record.body.get(key).unwrap_or(&null);
            let ord = ka.cmp_total(kb);
            let ord = if *desc { ord.reverse() } else { ord };
            ord.then_with(|| a.record.id.cmp(&b.record.id))
        });
        return;
    }
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

/// Apply the select's field filter. Field predicates (equality, comparison,
/// `IN`, `BETWEEN`) use [`matches_field_pred`] — exact `PartialEq` for `=`
/// and the total order [`Value::cmp_total`] for ranges (issue #93);
/// `HasEmbedding` requires a non-`None` embedding. `Bm25` is a *scoring*
/// filter: it never prunes rows — every row of the table is returned and
/// ranked by its lexical score.
fn matches_filter(rec: &Record, filter: Option<&Filter>) -> bool {
    match filter {
        None => true,
        Some(Filter::HasEmbedding) => rec.embedding.is_some(),
        Some(Filter::Bm25 { .. }) => true,
        // All-of over terms (issue #125): recursion keeps each term's own
        // semantics (missing-field rules, IS NOT NULL on embeddings).
        Some(Filter::And(terms)) => terms.iter().all(|t| matches_filter(rec, Some(t))),
        // `id` pseudo-field (issue #128): bound to the record's own identity
        // — a body key named `id` never shadows it.
        Some(f) if predicate_field(f) == Some("id") => record_id_matches(rec, f),
        Some(f) => matches_field_pred(&rec.body, f),
    }
}

/// The body field a field-level predicate reads (`None` for the non-field
/// filters: embedding presence, BM25, conjunctions).
fn predicate_field(filter: &Filter) -> Option<&str> {
    match filter {
        Filter::FieldEquals { field, .. }
        | Filter::FieldCmp { field, .. }
        | Filter::FieldIn { field, .. }
        | Filter::FieldBetween { field, .. } => Some(field),
        _ => None,
    }
}

/// The `id` pseudo-field (issue #128): field predicates on `id` compare
/// against the record's own identity in display form (`table:id`) — the
/// rerank-pool predicate (`WHERE id IN [...]`). `!=` is the exact complement
/// of `=` (the #93 rule). Ordered forms are parse-rejected for `id`; an
/// IR-injected one gets a non-match rather than an invented order. Only
/// reached for RECORD filters — edge filters have no record identity, so
/// `id` there falls through to an ordinary prop lookup.
fn record_id_matches(rec: &Record, filter: &Filter) -> bool {
    let me = rec.id.to_string();
    let is_me = |v: &Value| matches!(v, Value::Str(s) if *s == me);
    match filter {
        Filter::FieldEquals { value, .. } => is_me(value),
        Filter::FieldCmp {
            op: CmpOp::Ne,
            value,
            ..
        } => !is_me(value),
        Filter::FieldIn { values, .. } => values.iter().any(is_me),
        _ => false,
    }
}

/// A field-level predicate (equality / comparison / `IN` / `BETWEEN`)
/// evaluated against a props map — record bodies and edge props share it
/// (issue #93, spec §2.3).
///
/// Semantics: a record/edge that does not carry the field **never matches**
/// (the same rule `=` has always had — so `= v` and `!= v` are complements).
/// Values that are present compare with [`Value::cmp_total`] for the range
/// operators (`<`, `<=`, `>`, `>=`, `BETWEEN`) — a total cross-type order in
/// which an explicit `null` ranks below every other type — while `=`, `!=`,
/// and `IN` use the exact derived equality on [`Value`].
fn matches_field_pred(props: &BTreeMap<String, Value>, filter: &Filter) -> bool {
    match filter {
        Filter::FieldEquals { field, value } => props.get(field) == Some(value),
        Filter::FieldCmp { field, op, value } => {
            let Some(lhs) = props.get(field) else {
                return false;
            };
            match op {
                CmpOp::Ne => lhs != value,
                CmpOp::Lt => lhs.cmp_total(value) == Ordering::Less,
                CmpOp::Le => lhs.cmp_total(value) != Ordering::Greater,
                CmpOp::Gt => lhs.cmp_total(value) == Ordering::Greater,
                CmpOp::Ge => lhs.cmp_total(value) != Ordering::Less,
            }
        }
        Filter::FieldIn { field, values } => props.get(field).is_some_and(|v| values.contains(v)),
        Filter::FieldBetween { field, lo, hi } => {
            let Some(lhs) = props.get(field) else {
                return false;
            };
            lhs.cmp_total(lo) != Ordering::Less && lhs.cmp_total(hi) != Ordering::Greater
        }
        // Handled by the callers above — not body-value predicates.
        Filter::HasEmbedding | Filter::Bm25 { .. } => true,
        // Reached only by direct calls (both wrappers intercept And first);
        // recursion preserves all-of semantics on the props-only subset.
        Filter::And(terms) => terms.iter().all(|t| matches_field_pred(props, t)),
    }
}

/// The single result row of an aggregate: `{"count": n}` as a synthetic
/// record in the queried table (id `table:count`, no embedding, score 0), so
/// it flows through every transport's normal row encoding unchanged.
fn count_row(table: &str, n: u64) -> ScoredRecord {
    ScoredRecord {
        record: Record {
            id: RecordId::new(table, Id::Str("count".into())),
            body: BTreeMap::from([("count".into(), Value::Int(n as i64))]),
            embedding: None,
            created_at: 0,
        },
        score: 0.0,
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
///   Each edge's weight is its explicit `weight`, or its signed `value` when
///   `weight` is absent (`value = -1` downvotes; see [`score_of`]).
/// - `Order::Votes` → net up−down vote count (see [`vote_counts`]).
/// - `Order::Feedback` → time-decayed recent feedback (see [`feedback_score`]).
/// - `Order::Salience` with kNN → `0.7 * similarity + 0.3 * normalized_score`,
///   where `normalized_score` is the Laplace score clamped to `[0, 1]`
///   (weights are treated as `[0, 1]` confidence values; the clamp keeps the
///   blend in range even for out-of-range weights). These are the engine
///   defaults of the spec §2.3 four-term formula (α=0.7, β=0, γ=0, δ=0.3).
/// - `Order::Salience` without kNN → the normalized score alone.
/// - `Order::SalienceWeighted([α, β, γ, δ])` → the same four-term formula with
///   agent-tuned weights (parsed from `ORDER BY ::salience(α, β, γ, δ)`):
///   `α·similarity + β·strength + γ·importance + δ·normalized_score` — see
///   [`strength_of`] and [`importance_of`] for the β/γ term definitions.
/// - Anything else → cosine similarity vs the kNN query (`0.0` when there is
///   no kNN clause, or when the record has no embedding / zero-norm vector).
fn compute_score(
    store: &Store,
    sel: &Select,
    rec: &Record,
    knn_sims: Option<&BTreeMap<RecordId, f32>>,
    bm25: Option<&(Bm25Index, Vec<String>)>,
    hybrid: Option<&BTreeMap<RecordId, f32>>,
) -> f32 {
    // Hybrid fusion is the dominant score source: with both a kNN clause and a
    // `::bm25` filter, every row is ranked by its RRF fused score regardless
    // of ORDER BY.
    if let Some(fused) = hybrid {
        return fused.get(&rec.id).copied().unwrap_or(0.0);
    }

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
        Some(Order::SalienceWeighted(w)) => {
            let [alpha, beta, gamma, delta] = *w;
            alpha * similarity
                + beta * strength_of(store, rec)
                + gamma * importance_of(rec)
                + delta * score_of(store, rec).clamp(0.0, 1.0)
        }
        _ => similarity,
    }
}

/// Does a stored edge name satisfy a requested name, tolerant of a leading
/// `:` on either side? The nql parser strips it (`-> :voted` stores
/// `"voted"`), but edges built directly through the IR may keep it — the
/// chat_memory example deliberately does (issue #98), and before this check
/// those edges matched nothing, silently collapsing `::salience`'s feedback
/// term to the no-vote baseline.
#[inline]
fn edge_name_matches(stored: &str, want: &str) -> bool {
    stored.trim_start_matches(':') == want.trim_start_matches(':')
}

/// Laplace-smoothed mean of `:voted` edge weights on the record.
///
/// A vote edge is any edge with `name == "voted"` pointing **to** the record.
/// Each edge contributes its [`vote_weight`]: the explicit `weight` when set,
/// otherwise the edge's signed `value` — so `SET value = -1` (no weight) is a
/// **downvote** here too, agreeing with `::votes`/`::feedback` instead of
/// inverting them. The estimate `(sum + 1) / (n + 2)` starts at `0.5` with
/// zero votes and moves toward the observed mean as votes accumulate.
/// The edge name is stored **without** the `:` prefix — the parser strips it
/// (`RELATE (a) -> :voted -> (b)` stores `"voted"`), and `::votes`/`::feedback`
/// use the same convention. IR-built edges may keep the colon (`:voted`);
/// readers accept both spellings via [`edge_name_matches`] (issue #98).
fn score_of(store: &Store, rec: &Record) -> f32 {
    let mut sum = 0.0f32;
    let mut n = 0usize;
    for edge in &store.edges {
        if edge_name_matches(&edge.name, "voted") && edge.to == rec.id {
            sum += vote_weight(edge);
            n += 1;
        }
    }
    (sum + 1.0) / (n as f32 + 2.0)
}

/// The signed weight `::score` counts for one `:voted` edge.
///
/// Explicit `weight` wins (an agent may down-weight a positive vote or pin an
/// exact value). When `weight` is absent the edge's `value` supplies the sign
/// (`value = -1` → `-1.0`, `value = 1` → `1.0`) — the fix for downvotes being
/// counted as upvotes (issue #85). Only an edge with *neither* field falls
/// back to the legacy `1.0` (an upvote); D9 defines votes as carrying
/// `value`, so that case is degenerate.
fn vote_weight(edge: &RelationEdge) -> f32 {
    if let Some(w) = edge.weight {
        return w;
    }
    match edge.props.get("value") {
        Some(Value::Int(v)) => *v as f32,
        Some(Value::Float(v)) => *v as f32,
        _ => 1.0,
    }
}

/// The `:voted` edges pointing **at** `id`, in store (append) order.
///
/// A vote is a directed edge `(voter)->:voted {value:+1|-1, weight:0..1}->(record)`
/// (see docs/decisions.md D9); only edges whose name matches `voted` (either
/// spelling — see [`edge_name_matches`]) and whose `to` is the record count.
/// Iteration order is the store's append order, which is deterministic for a
/// given store.
fn votes_toward<'a>(store: &'a Store, id: &RecordId) -> Vec<&'a RelationEdge> {
    store
        .edges
        .iter()
        .filter(|e| edge_name_matches(&e.name, "voted") && e.to == *id)
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

/// `strength(recency, freq)` — the β term of `::salience(α, β, γ, δ)`, in
/// `[0, 1]`. Deterministic and pure (spec §2.3):
///
/// ```text
/// strength = (recency + freq) / 2
/// recency  = 1 / (1 + age),   age = max(0, clock − created_at)
/// freq     = n / (n + 1),      n   = incident edges (either direction)
/// ```
///
/// Recency uses the same `1/(1+λ·age)` shape (λ=1) as [`feedback_score`]; freq
/// saturates toward 1 as the record draws more edges. A record stamped this
/// transaction is age 0 → recency 1.0. Engine defaults give this term zero
/// weight (β=0) — agents opt in per-query.
fn strength_of(store: &Store, rec: &Record) -> f32 {
    let age = (store.clock - rec.created_at).max(0) as f32;
    let recency = 1.0 / (1.0 + age);
    let n = store
        .edges
        .iter()
        .filter(|e| e.from == rec.id || e.to == rec.id)
        .count() as f32;
    0.5 * recency + 0.5 * (n / (n + 1.0))
}

/// The γ term of `::salience(α, β, γ, δ)`: the agent-written `importance`
/// field clamped to `[0, 1]`. Missing or non-numeric → `0.0` (spec §5: the
/// engine never invents importance).
fn importance_of(rec: &Record) -> f32 {
    let v = match rec.body.get("importance") {
        Some(Value::Float(v)) => *v as f32,
        Some(Value::Int(v)) => *v as f32,
        _ => 0.0,
    };
    v.clamp(0.0, 1.0)
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
        if edge_name_matches(&edge.name, "voted") {
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
    fn knn_index_cache_serves_hits_and_tracks_mutations() {
        // Issue #144, L2: the memoized whole-table index must serve the same
        // bytes as a fresh build, observe mutations (every record change
        // bumps `clock` — the cache's invalidation signal), bypass `AS OF`,
        // and never leak across MEMORY sub-stores.
        let mut db = Database::default();
        db.execute(&[
            create("t", Some(2)),
            Statement::Insert(record(
                "t:1",
                BTreeMap::from([("v".into(), num(1))]),
                Some(vec![0.5, 0.5]),
            )),
            Statement::Insert(record(
                "t:2",
                BTreeMap::from([("v".into(), num(2))]),
                Some(vec![0.0, 1.0]),
            )),
        ])
        .unwrap();
        let knn_select = |as_of: Option<i64>| {
            Statement::Select(Select {
                table: "t".into(),
                knn: Some(Knn {
                    query: vec![1.0, 0.0],
                    k: 4,
                }),
                as_of,
                ..Select::default()
            })
        };

        // First query builds the memo; the second hits it — identical rows.
        let built = db.execute(&[knn_select(None)]).unwrap();
        let hit = db.execute(&[knn_select(None)]).unwrap();
        assert_eq!(built, hit);
        assert_eq!(built[0].rows.len(), 2);

        // INSERT bumps `clock` → memo invalid → the new record (an exact
        // match on the query) must be observable and rank first.
        db.execute(&[Statement::Insert(record(
            "t:3",
            BTreeMap::from([("v".into(), num(3))]),
            Some(vec![1.0, 0.0]),
        ))])
        .unwrap();
        let after_insert = db.execute(&[knn_select(None)]).unwrap();
        assert_eq!(after_insert[0].rows.len(), 3);
        assert_eq!(
            after_insert[0].rows[0].record.id,
            RecordId::parse("t:3").unwrap(),
            "invalidated memo must see the fresh insert"
        );

        // `AS OF` bypasses the memo: the pre-insert view stays intact
        // (t:3 lands at clock 4; cutoff 3 = create + t:1 + t:2).
        let past = db.execute(&[knn_select(Some(3))]).unwrap();
        assert_eq!(past[0].rows.len(), 2);

        // FORGET invalidates too.
        db.execute(&[Statement::Forget {
            id: RecordId::parse("t:3").unwrap(),
        }])
        .unwrap();
        let after_forget = db.execute(&[knn_select(None)]).unwrap();
        assert_eq!(after_forget[0].rows.len(), 2);

        // MEMORY sub-stores never share the root's memo: same table name,
        // different store, its own clock.
        db.execute(&[
            Statement::Memory { name: "m".into() },
            create("t", Some(2)),
            Statement::Insert(record(
                "t:1",
                BTreeMap::from([("v".into(), num(9))]),
                Some(vec![0.0, 1.0]),
            )),
        ])
        .unwrap();
        let in_memory = db
            .execute(&[Statement::Memory { name: "m".into() }, knn_select(None)])
            .unwrap();
        assert_eq!(in_memory[0].rows.len(), 1);
        assert_eq!(in_memory[0].rows[0].record.body.get("v"), Some(&num(9)));

        // Back at root: the memory insert must not have bled through.
        let root_again = db.execute(&[knn_select(None)]).unwrap();
        assert_eq!(root_again[0].rows.len(), 2);
    }

    #[test]
    fn knn_window_matches_total_order_with_negatives_and_gaps() {
        // Spec §2.3 output-cap invariant (issue #144, L3): the windowed
        // kNN-only path returns exactly the top OFFSET+cap rows under
        // (similarity desc, id asc) — including the 0.0 rows (non-embedded
        // AND zero-norm embeddings) that outrank negative similarities, and
        // with id tie-breaks interleaving embedded and non-embedded rows.
        let mut db = Database::default();
        db.execute(&[
            create("t", Some(1)),
            Statement::Insert(record(
                "t:1",
                BTreeMap::from([("v".into(), num(1))]),
                Some(vec![-1.0]),
            )),
            Statement::Insert(record("t:2", BTreeMap::from([("v".into(), num(2))]), None)),
            Statement::Insert(record(
                "t:3",
                BTreeMap::from([("v".into(), num(3))]),
                Some(vec![0.0]),
            )),
            Statement::Insert(record(
                "t:4",
                BTreeMap::from([("v".into(), num(4))]),
                Some(vec![1.0]),
            )),
            Statement::Insert(record("t:5", BTreeMap::from([("v".into(), num(5))]), None)),
        ])
        .unwrap();
        let knn = |k: usize, offset: Option<usize>| {
            Statement::Select(Select {
                table: "t".into(),
                knn: Some(Knn {
                    query: vec![1.0],
                    k,
                }),
                offset,
                ..Select::default()
            })
        };

        // Total order over query [1.0]: t:4 (1.0), then the 0.0 tie — t:2
        // (non-emb), t:3 (zero-norm), t:5 (non-emb) by id — then t:1 (−1.0).
        // k=2: window = top-2 embedded (t:4, t:3) ∪ first-2 non-emb (t:2,
        // t:5) → sorted → [t:4, t:2] (t:2 wins the 0.0 tie by id).
        let r = db.execute(&[knn(2, None)]).unwrap();
        let ids: Vec<String> = r[0].rows.iter().map(|x| x.record.id.to_string()).collect();
        assert_eq!(ids, ["t:4", "t:2"]);
        assert_eq!(r[0].rows[0].score, 1.0);
        assert_eq!(r[0].rows[1].score, 0.0);

        // k=1 + OFFSET 1: window = top-1 embedded (t:4) ∪ first-1 non-emb
        // (t:2) → [t:4, t:2] → drain(1) → [t:2] (cap 1).
        let r = db.execute(&[knn(1, Some(1))]).unwrap();
        let ids: Vec<String> = r[0].rows.iter().map(|x| x.record.id.to_string()).collect();
        assert_eq!(ids, ["t:2"]);
        assert_eq!(r[0].rows[0].score, 0.0);
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
    fn score_defaults_weight_from_vote_value() {
        // Issue #85: `SET value = -1` without `weight` must DOWNvote under
        // ::score (previously weight defaulted to +1, inverting the vote and
        // disagreeing with ::votes/::feedback).
        let mut db = Database::default();
        let vote = |to: &str, value: i64, weight: Option<f32>| {
            let mut props = BTreeMap::new();
            props.insert("value".into(), Value::Int(value));
            Statement::Relate(RelationEdge {
                from: RecordId::parse("user:voter").unwrap(),
                name: "voted".into(),
                to: RecordId::parse(to).unwrap(),
                created_at: 1,
                weight,
                props,
            })
        };
        db.execute(&[
            create("post", None),
            Statement::Insert(record("post:a", BTreeMap::new(), None)),
            Statement::Insert(record("post:b", BTreeMap::new(), None)),
            Statement::Insert(record("post:c", BTreeMap::new(), None)),
            Statement::Insert(record("post:d", BTreeMap::new(), None)),
            // a: downvote, no weight -> -1 (was +1 before the fix)
            vote("post:a", -1, None),
            // c: upvote, no weight -> +1
            vote("post:c", 1, None),
            // d: explicit weight overrides the -1 value (agent's choice)
            vote("post:d", -1, Some(0.9)),
        ])
        .unwrap();

        let res = db
            .execute(&[Statement::Select(Select {
                order: Some(Order::Score),
                ..select("post")
            })])
            .unwrap();
        let score_of_id = |id: &str| -> f32 {
            res[0]
                .rows
                .iter()
                .find(|r| r.record.id.to_string() == id)
                .unwrap()
                .score
        };
        // a: (-1 + 1) / (1 + 2) = 0.0  (downvote pulls below the 0.5 baseline)
        assert!((score_of_id("post:a") - 0.0).abs() < 1e-6);
        // b: no votes -> 0.5 baseline
        assert!((score_of_id("post:b") - 0.5).abs() < 1e-6);
        // c: (1 + 1) / (1 + 2) = 2/3
        assert!((score_of_id("post:c") - 2.0 / 3.0).abs() < 1e-6);
        // d: (0.9 + 1) / (1 + 2) = 1.9/3 — explicit weight wins over value
        assert!((score_of_id("post:d") - 1.9 / 3.0).abs() < 1e-6);

        // Ordering: downvoted record must rank LAST (below unvoted).
        let ids: Vec<_> = res[0]
            .rows
            .iter()
            .map(|r| r.record.id.to_string())
            .collect();
        assert_eq!(ids, ["post:c", "post:d", "post:b", "post:a"]);

        // Sign agreement with ::votes: the downvoted records (a, d — both
        // value=-1) sit below the unvoted b and the upvoted c, with the same
        // relative order for a as under ::score (a is the global minimum in
        // both orderings).
        let res = db
            .execute(&[Statement::Select(Select {
                order: Some(Order::Votes),
                ..select("post")
            })])
            .unwrap();
        let net_of_id = |id: &str| -> f32 {
            res[0]
                .rows
                .iter()
                .find(|r| r.record.id.to_string() == id)
                .unwrap()
                .score
        };
        assert!((net_of_id("post:a") - (-1.0)).abs() < 1e-6);
        assert!((net_of_id("post:b") - 0.0).abs() < 1e-6);
        assert!((net_of_id("post:c") - 1.0).abs() < 1e-6);
        let pos = |id: &str| -> usize {
            res[0]
                .rows
                .iter()
                .position(|r| r.record.id.to_string() == id)
                .unwrap()
        };
        assert!(
            pos("post:a") > pos("post:b"),
            "::score and ::votes must agree on sign ordering"
        );
    }

    #[test]
    fn colon_prefixed_voted_edges_count_like_no_colon() {
        // Issue #98: IR-built ":voted" edges (the chat_memory importance
        // knob builds them with the colon kept) must count under ::score —
        // before edge_name_matches they matched nothing and the salience
        // feedback term silently collapsed to the no-vote baseline.
        let mut db = Database::default();
        db.execute(&[
            create("post", None),
            Statement::Insert(record("post:a", BTreeMap::new(), None)),
            Statement::Insert(record("post:b", BTreeMap::new(), None)),
            Statement::Relate(RelationEdge {
                from: RecordId::parse("agent:main").unwrap(),
                name: ":voted".into(),
                to: RecordId::parse("post:a").unwrap(),
                created_at: 1,
                weight: Some(1.0),
                props: BTreeMap::new(),
            }),
        ])
        .unwrap();

        let res = db
            .execute(&[Statement::Select(Select {
                order: Some(Order::Score),
                ..select("post")
            })])
            .unwrap();
        let score_of_id = |id: &str| -> f32 {
            res[0]
                .rows
                .iter()
                .find(|r| r.record.id.to_string() == id)
                .unwrap()
                .score
        };
        // a: colon-named vote with weight 1.0 -> (1 + 1)/(1 + 2) = 2/3.
        assert!((score_of_id("post:a") - 2.0 / 3.0).abs() < 1e-6);
        assert!((score_of_id("post:b") - 0.5).abs() < 1e-6, "b: no votes");
        let ids: Vec<_> = res[0]
            .rows
            .iter()
            .map(|r| r.record.id.to_string())
            .collect();
        assert_eq!(ids, ["post:a", "post:b"]);
    }

    #[test]
    fn created_at_stamped_per_mutation() {
        // Issue #107: the parser leaves created_at = 0 with the comment
        // "Engine clocks created_at per-transaction" — the clocking lives
        // HERE. Each stamp is the statement's mutation timestamp (CREATE=1),
        // which is also what AS OF replay re-derives from statement order.
        let mut db = Database::default();
        db.execute(&[
            create("post", None),
            Statement::Insert(record("post:a", BTreeMap::new(), None)),
            Statement::Insert(record("post:b", BTreeMap::new(), None)),
        ])
        .unwrap();

        let created_at_of = |id: &str| -> i64 {
            db.store()
                .records
                .values()
                .find(|r| r.id.to_string() == id)
                .unwrap()
                .created_at
        };
        assert_eq!(created_at_of("post:a"), 2, "first INSERT = ts2");
        assert_eq!(created_at_of("post:b"), 3, "second INSERT = ts3");

        // ::recency now orders newest-first (before the fix: all-zero
        // timestamps tie-broke to ascending id — the *oldest* first).
        let res = db
            .execute(&[Statement::Select(Select {
                order: Some(Order::Recency),
                ..select("post")
            })])
            .unwrap();
        let ids: Vec<_> = res[0]
            .rows
            .iter()
            .map(|r| r.record.id.to_string())
            .collect();
        assert_eq!(ids, ["post:b", "post:a"], "newest first under ::recency");
    }

    #[test]
    fn feedback_decay_uses_stamped_edge_created_at() {
        // Issue #107: edges get stamped per RELATE, so ::feedback finally
        // decays: two upvotes on one record at consecutive timestamps are
        // 1.0 (age 0) + 0.5 (age 1) = 1.5 — not the age-0 2.0.
        // NB ::feedback reads the `value` prop (like ::votes), and edges
        // arrive with created_at = 0 from nql → stamped by the engine.
        let vote = |user: &str| RelationEdge {
            from: RecordId::parse(user).unwrap(),
            name: "voted".into(),
            to: RecordId::parse("post:1").unwrap(),
            created_at: 0, // stamped by execute (issue #107)
            weight: None,
            props: BTreeMap::from([("value".into(), Value::Int(1))]),
        };
        let mut db = Database::default();
        db.execute(&[
            create("post", None),
            Statement::Insert(record("post:1", BTreeMap::new(), None)),
            Statement::Relate(vote("user:u1")),
            Statement::Relate(vote("user:u2")),
        ])
        .unwrap();

        // Stamps: CREATE=1, INSERT=2, votes at 3 and 4 → now=4, ages 1 and 0.
        let edge_ages: Vec<i64> = db.store().edges.iter().map(|e| e.created_at).collect();
        assert_eq!(edge_ages, [3, 4], "edges stamped at their mutation ts");

        let res = db
            .execute(&[Statement::Select(Select {
                order: Some(Order::Feedback),
                ..select("post")
            })])
            .unwrap();
        let score = res[0]
            .rows
            .iter()
            .find(|r| r.record.id.to_string() == "post:1")
            .unwrap()
            .score;
        assert!(
            (score - 1.5).abs() < 1e-6,
            "expected 1.5 (second vote age-1 decayed), got {score}"
        );
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
    fn salience_weighted_gamma_tunes_importance_field() {
        let mut db = Database::default();
        let body = |imp: f64| BTreeMap::from([("importance".into(), Value::Float(imp))]);
        db.execute(&[
            create("doc", Some(2)),
            Statement::Insert(record("doc:1", body(0.1), Some(vec![1.0, 0.0]))),
            Statement::Insert(record("doc:2", body(0.9), Some(vec![1.0, 0.0]))),
        ])
        .unwrap();
        let mut run = |order: Order| {
            db.execute(&[Statement::Select(Select {
                knn: Some(Knn {
                    query: vec![1.0, 0.0],
                    k: 10,
                }),
                order: Some(order),
                ..select("doc")
            })])
            .unwrap()
        };
        // γ = 1: the agent-written `importance` field drives the ranking —
        // identical embeddings, so similarity alone could not separate them.
        let res = run(Order::SalienceWeighted([0.0, 0.0, 1.0, 0.0]));
        let ids: Vec<_> = res[0]
            .rows
            .iter()
            .map(|r| r.record.id.to_string())
            .collect();
        assert_eq!(ids, ["doc:2", "doc:1"]);
        assert!((res[0].rows[0].score - 0.9).abs() < 1e-6);
        assert!((res[0].rows[1].score - 0.1).abs() < 1e-6);
        // Bare ::salience (defaults, γ = 0) ignores the field: same similarity,
        // no votes → tie → RecordId order.
        let res = run(Order::Salience);
        let ids: Vec<_> = res[0]
            .rows
            .iter()
            .map(|r| r.record.id.to_string())
            .collect();
        assert_eq!(ids, ["doc:1", "doc:2"]);
    }

    #[test]
    fn salience_weighted_beta_tunes_strength_recency_freq() {
        let mut db = Database::default();
        let mut old = record("msg:aaa-old", BTreeMap::new(), None);
        old.created_at = 1;
        let mut new = record("msg:zzz-new", BTreeMap::new(), None);
        new.created_at = 50;
        db.execute(&[
            create("msg", None),
            Statement::Insert(old),
            Statement::Insert(new),
        ])
        .unwrap();
        let mut run = |order: Order| {
            db.execute(&[Statement::Select(Select {
                order: Some(order),
                ..select("msg")
            })])
            .unwrap()
        };
        // β = 1: recency puts the newer record first even though its id sorts
        // last — the RecordId tie-break can't mask the term.
        let res = run(Order::SalienceWeighted([0.0, 1.0, 0.0, 0.0]));
        let ids: Vec<_> = res[0]
            .rows
            .iter()
            .map(|r| r.record.id.to_string())
            .collect();
        assert_eq!(ids, ["msg:zzz-new", "msg:aaa-old"]);
        // Defaults give strength zero weight: both score 0.5 → id order.
        let res = run(Order::Salience);
        let ids: Vec<_> = res[0]
            .rows
            .iter()
            .map(|r| r.record.id.to_string())
            .collect();
        assert_eq!(ids, ["msg:aaa-old", "msg:zzz-new"]);
    }

    #[test]
    fn salience_weighted_default_weights_reproduce_bare_salience() {
        // [0.7, 0, 0, 0.3] — the spec §2.3 engine defaults — must score
        // identically to bare ORDER BY ::salience (backward compatibility).
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
        let mut run = |order: Order| {
            db.execute(&[Statement::Select(Select {
                knn: Some(Knn {
                    query: vec![1.0, 0.0],
                    k: 10,
                }),
                order: Some(order),
                ..select("doc")
            })])
            .unwrap()
        };
        let bare: Vec<f32> = run(Order::Salience)[0]
            .rows
            .iter()
            .map(|r| r.score)
            .collect();
        let weighted: Vec<f32> = run(Order::SalienceWeighted([0.7, 0.0, 0.0, 0.3]))[0]
            .rows
            .iter()
            .map(|r| r.score)
            .collect();
        assert_eq!(bare, weighted);
        assert!((bare[0] - 0.9).abs() < 1e-5);
        assert!((bare[1] - 0.15).abs() < 1e-5);
    }

    #[test]
    fn comparison_filters_prune_by_total_order() {
        let mut db = Database::default();
        db.execute(&[
            create("ledger", None),
            Statement::Insert(record(
                "ledger:1",
                BTreeMap::from([("seq".into(), num(5))]),
                None,
            )),
            Statement::Insert(record(
                "ledger:2",
                BTreeMap::from([("seq".into(), num(50))]),
                None,
            )),
            Statement::Insert(record(
                "ledger:3",
                BTreeMap::from([("seq".into(), num(500))]),
                None,
            )),
            // Type-mixed value: strings rank above every number.
            Statement::Insert(record(
                "ledger:4",
                BTreeMap::from([("seq".into(), str_("abc"))]),
                None,
            )),
            // No seq at all: never matches a field predicate (the `=` rule).
            Statement::Insert(record("ledger:5", BTreeMap::new(), None)),
        ])
        .unwrap();
        let mut ids_where = |filter: Filter| {
            let res = db
                .execute(&[Statement::Select(Select {
                    filter: Some(filter),
                    ..select("ledger")
                })])
                .unwrap();
            res[0]
                .rows
                .iter()
                .map(|r| r.record.id.to_string())
                .collect::<Vec<_>>()
        };

        // `< 100`: the two small seqs — the string ranks above, the absent
        // field never matches.
        assert_eq!(
            ids_where(Filter::FieldCmp {
                field: "seq".into(),
                op: CmpOp::Lt,
                value: Value::Int(100),
            }),
            ["ledger:1", "ledger:2"]
        );
        // `> 100`: the big number AND the string (total order, cross-type).
        assert_eq!(
            ids_where(Filter::FieldCmp {
                field: "seq".into(),
                op: CmpOp::Gt,
                value: Value::Int(100),
            }),
            ["ledger:3", "ledger:4"]
        );
        // Inclusive ends.
        assert_eq!(
            ids_where(Filter::FieldCmp {
                field: "seq".into(),
                op: CmpOp::Le,
                value: Value::Int(5),
            }),
            ["ledger:1"]
        );
        // `!=` is the complement of `=` among records that carry the field.
        assert_eq!(
            ids_where(Filter::FieldCmp {
                field: "seq".into(),
                op: CmpOp::Ne,
                value: Value::Int(50),
            }),
            ["ledger:1", "ledger:3", "ledger:4"]
        );
        // IN membership (exact equality, like `=`).
        assert_eq!(
            ids_where(Filter::FieldIn {
                field: "seq".into(),
                values: vec![Value::Int(5), Value::Int(500)],
            }),
            ["ledger:1", "ledger:3"]
        );
        // BETWEEN is inclusive on both ends.
        assert_eq!(
            ids_where(Filter::FieldBetween {
                field: "seq".into(),
                lo: Value::Int(5),
                hi: Value::Int(50),
            }),
            ["ledger:1", "ledger:2"]
        );
        // Plain `=` still excludes the field-less record (regression).
        assert_eq!(
            ids_where(Filter::FieldEquals {
                field: "seq".into(),
                value: Value::Int(5),
            }),
            ["ledger:1"]
        );
    }

    #[test]
    fn count_star_returns_filtered_row_count() {
        let mut db = Database::default();
        db.execute(&[
            create("task", None),
            Statement::Insert(record(
                "task:1",
                BTreeMap::from([("done".into(), Value::Bool(true))]),
                None,
            )),
            Statement::Insert(record(
                "task:2",
                BTreeMap::from([("done".into(), Value::Bool(false))]),
                None,
            )),
            Statement::Insert(record(
                "task:3",
                BTreeMap::from([("done".into(), Value::Bool(true))]),
                None,
            )),
        ])
        .unwrap();
        let mut count_of = |filter: Option<Filter>, limit: Option<usize>| {
            let res = db
                .execute(&[Statement::Select(Select {
                    filter,
                    limit,
                    aggregate: Some(Aggregate::CountStar),
                    ..select("task")
                })])
                .unwrap();
            assert_eq!(res[0].rows.len(), 1, "COUNT returns exactly one row");
            res[0].rows[0].record.body.get("count").cloned()
        };

        // Whole table.
        assert_eq!(count_of(None, None), Some(Value::Int(3)));
        // Filtered.
        assert_eq!(
            count_of(
                Some(Filter::FieldEquals {
                    field: "done".into(),
                    value: Value::Bool(true),
                }),
                None,
            ),
            Some(Value::Int(2))
        );
        // LIMIT never affects the aggregate (count = filtered total).
        assert_eq!(count_of(None, Some(1)), Some(Value::Int(3)));
        // Empty match → a count of zero, not an empty result.
        assert_eq!(
            count_of(
                Some(Filter::FieldCmp {
                    field: "done".into(),
                    op: CmpOp::Gt,
                    value: Value::Str("m".into()),
                }),
                None,
            ),
            Some(Value::Int(0))
        );

        // The row shape: synthetic id in the queried table.
        let res = db
            .execute(&[Statement::Select(Select {
                aggregate: Some(Aggregate::CountStar),
                ..select("task")
            })])
            .unwrap();
        assert_eq!(res[0].rows[0].record.id.to_string(), "task:count");
    }

    #[test]
    fn offset_skips_after_ordering_before_limit() {
        let mut db = Database::default();
        db.execute(&[
            create("page", None),
            Statement::Insert(record("page:1", BTreeMap::new(), None)),
            Statement::Insert(record("page:2", BTreeMap::new(), None)),
            Statement::Insert(record("page:3", BTreeMap::new(), None)),
            Statement::Insert(record("page:4", BTreeMap::new(), None)),
            Statement::Insert(record("page:5", BTreeMap::new(), None)),
        ])
        .unwrap();
        let mut window = |offset: Option<usize>, limit: Option<usize>| {
            let res = db
                .execute(&[Statement::Select(Select {
                    offset,
                    limit,
                    ..select("page")
                })])
                .unwrap();
            res[0]
                .rows
                .iter()
                .map(|r| r.record.id.to_string())
                .collect::<Vec<_>>()
        };

        // OFFSET alone: everything after the skip (BTree order).
        assert_eq!(window(Some(2), None), ["page:3", "page:4", "page:5"]);
        // OFFSET + LIMIT: a pagination window.
        assert_eq!(window(Some(2), Some(2)), ["page:3", "page:4"]);
        // Past the end: empty, no panic.
        assert_eq!(window(Some(9), Some(3)), Vec::<String>::new());
        // No offset → unchanged baseline.
        assert_eq!(window(None, Some(2)), ["page:1", "page:2"]);
    }

    #[test]
    fn match_count_reports_edge_multiplicity() {
        let mut db = Database::default();
        let rel = |from: &str, to: &str, confidence: f64| {
            Statement::Relate(RelationEdge {
                from: RecordId::parse(from).unwrap(),
                name: "edge".into(),
                to: RecordId::parse(to).unwrap(),
                created_at: 0,
                weight: None,
                props: BTreeMap::from([("confidence".into(), Value::Float(confidence))]),
            })
        };
        db.execute(&[
            create("a", None),
            create("b", None),
            create("c", None),
            Statement::Insert(record("a:1", BTreeMap::new(), None)),
            Statement::Insert(record("b:1", BTreeMap::new(), None)),
            Statement::Insert(record("b:2", BTreeMap::new(), None)),
            Statement::Insert(record("c:1", BTreeMap::new(), None)),
            // Parallel edges a:1 -> b:1 that row-MATCH deduplicates away.
            rel("a:1", "b:1", 0.9),
            rel("a:1", "b:1", 0.9),
            rel("a:1", "b:1", 0.2),
            rel("a:1", "b:2", 0.9),
            // Second hop: two b:1 -> c:1 edges, one b:2 -> c:1.
            rel("b:1", "c:1", 0.5),
            rel("b:1", "c:1", 0.5),
            rel("b:2", "c:1", 0.5),
        ])
        .unwrap();
        let path = |start: &str, hops: usize, filter: Option<Filter>| MatchPath {
            start: RecordId::parse(start).unwrap(),
            steps: (0..hops)
                .map(|_| MatchStep {
                    direction: MatchDirection::Out,
                    name: "edge".into(),
                    edge_props: filter.clone(),
                })
                .collect(),
            as_of: None,
        };
        // Row-returning MATCH still dedups endpoints (the gap COUNT closes):
        // asserted BEFORE the count closure takes `db` mutably.
        let res = db
            .execute(&[Statement::Match(path("a:1", 1, None))])
            .unwrap();
        assert_eq!(res[0].rows.len(), 2);
        let mut count_walks = |start: &str, hops: usize, filter: Option<Filter>| {
            let res = db
                .execute(&[Statement::MatchCount(path(start, hops, filter))])
                .unwrap();
            assert_eq!(res[0].rows.len(), 1, "count is a single row");
            match res[0].rows[0].record.body.get("count") {
                Some(Value::Int(n)) => *n as u64,
                other => panic!("expected an integer count, got {other:?}"),
            }
        };

        // 1 hop: four parallel edges → walks = 4 (rows would dedup to 2).
        assert_eq!(count_walks("a:1", 1, None), 4);
        // Edge-prop predicate filters the walks too (confidence >= 0.5).
        assert_eq!(
            count_walks(
                "a:1",
                1,
                Some(Filter::FieldCmp {
                    field: "confidence".into(),
                    op: CmpOp::Ge,
                    value: Value::Float(0.5),
                })
            ),
            3
        );
        // 2 hops: walks multiply — 3 edges to b:1 × 2 onward + 1 × 1 = … wait,
        // step 1 reaches b:1 via 3 edges and b:2 via 1; step 2 has b:1 -> c:1
        // ×2 and b:2 -> c:1 ×1 → 3·2 + 1·1 = 7.
        assert_eq!(count_walks("a:1", 2, None), 7);
        // Missing start → count 0 (spec §2.5: empty result, never an error).
        assert_eq!(count_walks("a:404", 1, None), 0);
    }

    #[test]
    fn match_and_closure_traverse_as_of_snapshots() {
        // Timeline (one mutation per statement): create=1, g:a=2, g:b=3,
        // edge g:a→g:b=4, g:c=5, edge g:a→g:c=6. Traversals at a cutoff
        // must see exactly the records and edges that existed then (issue
        // #92) — the same replay `SELECT ... AS OF` uses.
        let mut db = Database::default();
        let edge = |to: &str| RelationEdge {
            from: RecordId::parse("g:a").unwrap(),
            name: "edge".into(),
            to: RecordId::parse(to).unwrap(),
            created_at: 0,
            weight: None,
            props: BTreeMap::new(),
        };
        db.execute(&[
            create("g", None),
            Statement::Insert(record("g:a", BTreeMap::new(), None)),
            Statement::Insert(record("g:b", BTreeMap::new(), None)),
            Statement::Relate(edge("g:b")),
            Statement::Insert(record("g:c", BTreeMap::new(), None)),
            Statement::Relate(edge("g:c")),
        ])
        .unwrap();

        let mut traverse = |as_of: Option<i64>, count: bool| {
            let path = MatchPath {
                start: RecordId::parse("g:a").unwrap(),
                steps: vec![MatchStep {
                    direction: MatchDirection::Out,
                    name: "edge".into(),
                    edge_props: None,
                }],
                as_of,
            };
            let stmt = if count {
                Statement::MatchCount(path)
            } else {
                Statement::Match(path)
            };
            let res = db.execute(&[stmt]).unwrap();
            if count {
                return match res[0].rows[0].record.body.get("count") {
                    Some(Value::Int(n)) => vec![format!("count={n}")],
                    other => panic!("expected count row, got {other:?}"),
                };
            }
            res[0]
                .rows
                .iter()
                .map(|r| r.record.id.to_string())
                .collect::<Vec<_>>()
        };

        // Before any edge: empty (start exists, no traversable edges yet).
        assert_eq!(traverse(Some(3), false), Vec::<String>::new());
        // After the first edge only: g:b — g:c doesn't exist until ts5.
        assert_eq!(traverse(Some(4), false), ["g:b"]);
        // After both edges.
        assert_eq!(traverse(Some(6), false), ["g:b", "g:c"]);
        // No AS OF = current state (regression).
        assert_eq!(traverse(None, false), ["g:b", "g:c"]);
        // COUNT over historical edge sets: 0 → 1 → 2.
        assert_eq!(traverse(Some(3), true), ["count=0"]);
        assert_eq!(traverse(Some(4), true), ["count=1"]);
        assert_eq!(traverse(None, true), ["count=2"]);

        let mut closure_ids = |as_of: Option<i64>| {
            let path = MatchPath {
                start: RecordId::parse("g:a").unwrap(),
                steps: vec![MatchStep {
                    direction: MatchDirection::Out,
                    name: "edge".into(),
                    edge_props: None,
                }],
                as_of,
            };
            let res = db.execute(&[Statement::Closure(path)]).unwrap();
            res[0]
                .rows
                .iter()
                .map(|r| r.record.id.to_string())
                .collect::<Vec<_>>()
        };
        // CLOSURE composes with AS OF: reachability at ts4 = {g:a, g:b}.
        assert_eq!(closure_ids(Some(4)), ["g:a", "g:b"]);
        assert_eq!(closure_ids(None), ["g:a", "g:b", "g:c"]);
        // Cutoff before the start record exists → empty, never an error.
        assert_eq!(closure_ids(Some(1)), Vec::<String>::new());
    }

    #[test]
    fn prune_history_compacts_and_bounds_growth() {
        // Timeline: create=1, i1=2, i2=3 → prune snapshots at clock 3.
        let mut db = Database::default();
        db.execute(&[
            create("ledger", None),
            Statement::Insert(record(
                "ledger:1",
                BTreeMap::from([("seq".into(), num(5))]),
                None,
            )),
            Statement::Insert(record(
                "ledger:2",
                BTreeMap::from([("seq".into(), num(50))]),
                None,
            )),
        ])
        .unwrap();
        assert_eq!(db.store().history.len(), 3, "one entry per mutation");

        db.execute(&[Statement::PruneHistory]).unwrap();
        {
            let store = db.store();
            // Only the retained declaration (original ts) + one snapshot.
            assert_eq!(store.history.len(), 2, "compacted: {:?}", store.history);
            assert!(matches!(store.history[0].1, Statement::CreateTable { .. }));
            assert!(matches!(store.history[1].1, Statement::Snapshot(_)));
            assert_eq!(store.history[1].0, 3, "snapshot stamped at the clock");
        }

        // Growth is now bounded: one entry per later mutation, and re-prune
        // rebuilds the snapshot in place instead of stacking them.
        db.execute(&[Statement::Insert(record("ledger:3", BTreeMap::new(), None))])
            .unwrap();
        assert_eq!(db.store().history.len(), 3);
        db.execute(&[Statement::PruneHistory]).unwrap();
        assert_eq!(
            db.store().history.len(),
            2,
            "re-prune does not stack snapshots"
        );

        // Data is untouched by compaction: current state has all three rows.
        let now = db
            .execute(&[Statement::Select(Select { ..select("ledger") })])
            .unwrap();
        assert_eq!(now[0].rows.len(), 3);
    }

    #[test]
    fn as_of_before_snapshot_errors_after_prune() {
        // create=1, i1=2, i2=3 → snapshot at 3; AS OF 1/2 must fail loudly
        // (the pruned prefix is gone) instead of returning a declarations-only
        // partial view (issue #95).
        let mut db = Database::default();
        db.execute(&[
            create("t", None),
            Statement::Insert(record("t:1", BTreeMap::new(), None)),
            Statement::Insert(record("t:2", BTreeMap::new(), None)),
        ])
        .unwrap();

        // Before pruning: the old window still replays (regression).
        let ok = db
            .execute(&[Statement::Select(Select {
                as_of: Some(2),
                ..select("t")
            })])
            .unwrap();
        assert_eq!(ok[0].rows.len(), 1, "pre-prune AS OF unchanged");

        db.execute(&[Statement::PruneHistory]).unwrap();

        let err = db.execute(&[Statement::Select(Select {
            as_of: Some(2),
            ..select("t")
        })]);
        assert!(
            matches!(err, Err(Error::HistoryPruned { pruned_through: 3 })),
            "pre-snapshot AS OF fails loudly, got {err:?}"
        );

        // From the snapshot onward everything reconstructs.
        let at = db
            .execute(&[Statement::Select(Select {
                as_of: Some(3),
                ..select("t")
            })])
            .unwrap();
        assert_eq!(at[0].rows.len(), 2, "snapshot view = state at the clock");
        db.execute(&[Statement::Insert(record("t:3", BTreeMap::new(), None))])
            .unwrap();
        let later = db
            .execute(&[Statement::Select(Select {
                as_of: Some(4),
                ..select("t")
            })])
            .unwrap();
        assert_eq!(later[0].rows.len(), 3, "post-snapshot delta replays");
    }

    #[test]
    fn prune_history_compacts_memory_blocks_too() {
        let mut db = Database::default();
        db.execute(&[
            create("root_t", None),
            Statement::Memory { name: "blk".into() },
            create("inner", None),
            Statement::Insert(record("inner:1", BTreeMap::new(), None)),
        ])
        .unwrap();
        db.execute(&[Statement::PruneHistory]).unwrap();
        {
            let store = db.store();
            let only_decls_and_snap = |h: &Vec<(i64, Statement)>| {
                h.iter().all(|(_, s)| {
                    matches!(s, Statement::CreateTable { .. } | Statement::Snapshot(_))
                })
            };
            assert!(only_decls_and_snap(&store.history), "root compacted");
            let blk = store.memories.get("blk").expect("memory block exists");
            assert!(
                only_decls_and_snap(&blk.history),
                "memory compacted with its own snapshot"
            );
            assert!(
                blk.records
                    .contains_key(&RecordId::parse("inner:1").unwrap()),
                "memory data intact"
            );
        }

        // AS OF inside the memory, before its snapshot → HistoryPruned.
        let past = db.execute(&[
            Statement::Memory { name: "blk".into() },
            Statement::Select(Select {
                as_of: Some(1),
                ..select("inner")
            }),
        ]);
        assert!(
            matches!(past, Err(Error::HistoryPruned { .. })),
            "memory-scoped temporal read fails loudly, got {past:?}"
        );

        // Current reads in the memory are untouched.
        let now = db
            .execute(&[
                Statement::Memory { name: "blk".into() },
                Statement::Select(Select { ..select("inner") }),
            ])
            .unwrap();
        assert_eq!(now[0].rows.len(), 1);
    }

    #[test]
    fn order_by_field_sorts_with_direction_and_id_tiebreak() {
        let mut db = Database::default();
        db.execute(&[
            create("t", None),
            Statement::Insert(record(
                "t:b",
                BTreeMap::from([("seq".into(), num(50))]),
                None,
            )),
            Statement::Insert(record(
                "t:a",
                BTreeMap::from([("seq".into(), num(10))]),
                None,
            )),
            Statement::Insert(record(
                "t:c",
                BTreeMap::from([("seq".into(), num(30))]),
                None,
            )),
            // Absent field = `null` = lowest rank (spec §2.3 / cmp_total).
            Statement::Insert(record("t:d", BTreeMap::new(), None)),
            // Tie with t:a — the RecordId tie-break must show in BOTH
            // directions (DESC reverses the key only).
            Statement::Insert(record(
                "t:e",
                BTreeMap::from([("seq".into(), num(10))]),
                None,
            )),
            // Cross-type value: strings rank above every number; cmp_total
            // must order it without panicking.
            Statement::Insert(record(
                "t:f",
                BTreeMap::from([("seq".into(), str_("abc"))]),
                None,
            )),
        ])
        .unwrap();
        let mut ids = |desc: bool| {
            let res = db
                .execute(&[Statement::Select(Select {
                    order: Some(Order::Field {
                        key: "seq".into(),
                        desc,
                    }),
                    ..select("t")
                })])
                .unwrap();
            res[0]
                .rows
                .iter()
                .map(|r| r.record.id.to_string())
                .collect::<Vec<_>>()
        };

        // ASC: null first, ties id-asc (t:a before t:e), then 30, 50, string.
        assert_eq!(ids(false), ["t:d", "t:a", "t:e", "t:c", "t:b", "t:f"]);
        // DESC: key reversed — but ties STAY id-asc and nulls end up last.
        assert_eq!(ids(true), ["t:f", "t:b", "t:c", "t:a", "t:e", "t:d"]);
    }

    #[test]
    fn order_by_unknown_field_errors_loudly() {
        let mut db = Database::default();
        db.execute(&[
            create("t", None),
            Statement::Insert(record("t:1", BTreeMap::from([("a".into(), num(1))]), None)),
        ])
        .unwrap();

        // Rows exist but no record of the table carries the key → almost
        // certainly a typo: fail loudly, never return id-ordered rows as a
        // plausible-looking answer.
        let err = db.execute(&[Statement::Select(Select {
            order: Some(Order::Field {
                key: "nope".into(),
                desc: false,
            }),
            ..select("t")
        })]);
        assert!(
            matches!(err, Err(Error::UnknownSortField { .. })),
            "typo field errors, got {err:?}"
        );

        // No rows at all → nothing to mis-sort → no error.
        let empty = db
            .execute(&[Statement::Select(Select {
                order: Some(Order::Field {
                    key: "nope".into(),
                    desc: false,
                }),
                filter: Some(Filter::FieldEquals {
                    field: "a".into(),
                    value: Value::Int(999),
                }),
                ..select("t")
            })])
            .unwrap();
        assert_eq!(empty[0].rows.len(), 0);
    }

    #[test]
    fn order_by_field_wins_over_knn_ranking() {
        // Explicit structural sorts apply in kNN mode — same precedence as
        // `::recency` (score-based orders defer; see #119 for the full
        // precedence table).
        let mut db = Database::default();
        db.execute(&[
            create("t", Some(2)),
            Statement::Insert(record(
                "t:1",
                BTreeMap::from([("seq".into(), num(5))]),
                Some(vec![1.0, 0.0]),
            )),
            Statement::Insert(record(
                "t:2",
                BTreeMap::from([("seq".into(), num(1))]),
                Some(vec![0.0, 1.0]),
            )),
        ])
        .unwrap();
        let res = db
            .execute(&[Statement::Select(Select {
                knn: Some(Knn {
                    query: vec![1.0, 0.0],
                    k: 10,
                }),
                order: Some(Order::Field {
                    key: "seq".into(),
                    desc: false,
                }),
                ..select("t")
            })])
            .unwrap();
        let ids: Vec<_> = res[0]
            .rows
            .iter()
            .map(|r| r.record.id.to_string())
            .collect();
        // t:1 is the nearest neighbor but t:2 has the smaller seq.
        assert_eq!(ids, ["t:2", "t:1"]);
    }

    #[test]
    fn history_since_reports_exact_deltas_including_edges() {
        // Timeline: create=1 (with dim), t:1=2, t:2=3, RELATE=4 (edge-only!),
        // FORGET t:2=5 (removes a row AND t:1's incident edge). A row-state
        // diff of two AS OF reads would see NOTHING at ts4 — this must.
        let mut db = Database::default();
        db.execute(&[
            create("t", Some(2)),
            Statement::Insert(record("t:1", BTreeMap::new(), Some(vec![1.0, 0.0]))),
            Statement::Insert(record("t:2", BTreeMap::new(), Some(vec![0.0, 1.0]))),
            Statement::Relate(RelationEdge {
                from: RecordId::parse("t:1").unwrap(),
                name: "refs".into(),
                to: RecordId::parse("t:2").unwrap(),
                created_at: 0,
                weight: None,
                props: BTreeMap::new(),
            }),
            Statement::Forget {
                id: RecordId::parse("t:2").unwrap(),
            },
        ])
        .unwrap();

        let res = db.execute(&[Statement::HistorySince(0)]).unwrap();
        assert!(matches!(res[0].kind, QueryKind::History { since: 0 }));
        let kinds: Vec<_> = res[0]
            .rows
            .iter()
            .map(|r| {
                assert_eq!(
                    r.record.id.to_string(),
                    format!("history:{}", {
                        match r.record.body.get("ts") {
                            Some(Value::Int(t)) => *t,
                            other => panic!("ts field, got {other:?}"),
                        }
                    })
                );
                match r.record.body.get("kind") {
                    Some(Value::Str(k)) => k.clone(),
                    other => panic!("kind field, got {other:?}"),
                }
            })
            .collect();
        assert_eq!(kinds, ["CREATE", "INSERT", "INSERT", "RELATE", "FORGET"]);
        // CREATE carries its declaration (table + dim) …
        assert_eq!(
            res[0].rows[0].record.body.get("table"),
            Some(&Value::Str("t".into()))
        );
        assert_eq!(res[0].rows[0].record.body.get("dim"), Some(&Value::Int(2)));
        // … and the edge-only RELATE carries both endpoints (issue #118's
        // correctness gap — a row diff would have missed this entry) …
        let rel = &res[0].rows[3].record.body;
        assert_eq!(rel.get("from"), Some(&Value::Str("t:1".into())));
        assert_eq!(rel.get("to"), Some(&Value::Str("t:2".into())));
        assert_eq!(rel.get("name"), Some(&Value::Str("refs".into())));
        // … and FORGET is a tombstone for the removed record.
        assert_eq!(
            res[0].rows[4].record.body.get("id"),
            Some(&Value::Str("t:2".into()))
        );
        // The score mirrors the mutation ts (display only; body.ts is i64).
        assert_eq!(res[0].rows[3].score, 4.0);

        // Exclusive cutoff: strictly-after semantics.
        let since3 = db.execute(&[Statement::HistorySince(3)]).unwrap();
        assert_eq!(since3[0].rows.len(), 2, "RELATE + FORGET only");
        // At/after the last mutation: an empty delta, not an error.
        let tail = db.execute(&[Statement::HistorySince(5)]).unwrap();
        assert!(tail[0].rows.is_empty());
        // The read is side-effect free.
        let cur = db
            .execute(&[Statement::Select(Select { ..select("t") })])
            .unwrap();
        assert_eq!(cur[0].rows.len(), 1, "only t:1 survives the FORGET");
    }

    #[test]
    fn history_since_respects_compaction_horizon() {
        let mut db = Database::default();
        db.execute(&[
            create("t", None),
            Statement::Insert(record("t:1", BTreeMap::new(), None)),
            Statement::Insert(record("t:2", BTreeMap::new(), None)),
        ])
        .unwrap();
        db.execute(&[Statement::PruneHistory]).unwrap(); // snapshot @3
        db.execute(&[Statement::Insert(record("t:3", BTreeMap::new(), None))])
            .unwrap(); // ts4

        // Cutoff below the snapshot: mutations (since, 3] are gone — the
        // same loud retention contract as AS OF, never a partial delta.
        let err = db.execute(&[Statement::HistorySince(2)]);
        assert!(
            matches!(err, Err(Error::HistoryPruned { pruned_through: 3 })),
            "got {err:?}"
        );

        // From the horizon on: the delta works, and the snapshot itself is
        // bookkeeping — never reported as a mutation.
        let ok = db.execute(&[Statement::HistorySince(3)]).unwrap();
        assert!(matches!(ok[0].kind, QueryKind::History { since: 3 }));
        assert_eq!(ok[0].rows.len(), 1, "only the post-prune INSERT: {ok:?}");
        assert_eq!(ok[0].rows[0].record.id.to_string(), "history:4");
    }

    #[test]
    fn history_since_scopes_to_memory_blocks() {
        let mut db = Database::default();
        db.execute(&[
            create("root_t", None), // root clock 1
            Statement::Memory { name: "blk".into() },
            create("inner", None), // blk clock 1 (own history)
            Statement::Insert(record("inner:1", BTreeMap::new(), None)), // blk 2
        ])
        .unwrap();

        // Inside the block: only the block's mutations (per-block sync).
        let blk = db
            .execute(&[
                Statement::Memory { name: "blk".into() },
                Statement::HistorySince(0),
            ])
            .unwrap();
        assert_eq!(blk[0].rows.len(), 2, "block CREATE + INSERT: {blk:?}");
        assert_eq!(
            blk[0].rows[1].record.body.get("id"),
            Some(&Value::Str("inner:1".into()))
        );

        // At root: only root's history — MEMORY statements never log.
        let root = db.execute(&[Statement::HistorySince(0)]).unwrap();
        assert_eq!(root[0].rows.len(), 1, "root CREATE only: {root:?}");
        assert_eq!(
            root[0].rows[0].record.body.get("table"),
            Some(&Value::Str("root_t".into()))
        );
    }

    /// Shared fixtures for the #119 precedence/recipe tests: three docs with
    /// text (bm25), `topic` (field sort / `IN` pool), vectors (kNN), a heavy
    /// upvote on `doc:b` (so `::score` disagrees with relevance), and
    /// insertion order a → b → c (so `::recency` is c → b → a).
    fn precedence_fixture() -> Database {
        let mut db = Database::default();
        db.execute(&[
            create("doc", Some(2)),
            Statement::Insert(record(
                "doc:a",
                BTreeMap::from([
                    ("text".into(), str_("alpha alpha")),
                    ("topic".into(), str_("z")),
                ]),
                Some(vec![1.0, 0.0]),
            )),
            Statement::Insert(record(
                "doc:b",
                BTreeMap::from([("text".into(), str_("beta")), ("topic".into(), str_("a"))]),
                Some(vec![0.0, 1.0]),
            )),
            Statement::Insert(record(
                "doc:c",
                BTreeMap::from([
                    ("text".into(), str_("alpha other")),
                    ("topic".into(), str_("m")),
                ]),
                Some(vec![0.7, 0.7]),
            )),
            Statement::Relate(RelationEdge {
                from: RecordId::parse("u:1").unwrap(),
                name: "voted".into(),
                to: RecordId::parse("doc:b").unwrap(),
                created_at: 0,
                weight: Some(5.0),
                props: BTreeMap::from([("value".into(), Value::Int(1))]),
            }),
        ])
        .unwrap();
        db
    }

    #[test]
    fn order_by_precedence_matrix_matches_spec() {
        // The verified matrix (issue #119, spec §2.3 step 5): score-based
        // orders are honored in scan/kNN modes and IGNORED in bm25/hybrid
        // (relevance/fusion wins); structural orders are honored everywhere.
        let mut db = precedence_fixture();
        let mut run = |knn: bool, bm25: bool, order: Option<Order>| {
            let res = db
                .execute(&[Statement::Select(Select {
                    knn: knn.then(|| Knn {
                        query: vec![1.0, 0.0],
                        k: 10,
                    }),
                    filter: bm25.then(|| Filter::Bm25 {
                        field: "text".into(),
                        query: "alpha".into(),
                        k: None,
                    }),
                    order,
                    ..select("doc")
                })])
                .unwrap();
            res[0]
                .rows
                .iter()
                .map(|r| r.record.id.to_string())
                .collect::<Vec<_>>()
        };
        let field_order = || Order::Field {
            key: "topic".into(),
            desc: false,
        };

        // scan mode: BTree default; the score-based op is honored (the heavy
        // upvote floats doc:b).
        assert_eq!(run(false, false, None), ["doc:a", "doc:b", "doc:c"]);
        assert_eq!(
            run(false, false, Some(Order::Score)),
            ["doc:b", "doc:a", "doc:c"]
        );
        // kNN mode: similarity default; score-based and structural honored.
        assert_eq!(run(true, false, None), ["doc:a", "doc:c", "doc:b"]);
        assert_eq!(
            run(true, false, Some(Order::Score)),
            ["doc:b", "doc:a", "doc:c"]
        );
        assert_eq!(
            run(true, false, Some(Order::Recency)),
            ["doc:c", "doc:b", "doc:a"]
        );
        // bm25 mode: relevance default; score-based IGNORED (byte-identical
        // to the unordered query — the reported silent-OK hazard); structural
        // orders honored.
        let relevance = run(false, true, None);
        assert_eq!(relevance, ["doc:a", "doc:c", "doc:b"]);
        assert_eq!(run(false, true, Some(Order::Score)), relevance);
        assert_eq!(
            run(false, true, Some(Order::Recency)),
            ["doc:c", "doc:b", "doc:a"]
        );
        assert_eq!(
            run(false, true, Some(field_order())),
            ["doc:b", "doc:c", "doc:a"]
        );
        // hybrid mode: fused default; score-based IGNORED, structural honored.
        let fused = run(true, true, None);
        assert_eq!(run(true, true, Some(Order::Score)), fused);
        assert_eq!(
            run(true, true, Some(Order::Recency)),
            ["doc:c", "doc:b", "doc:a"]
        );
        assert_eq!(
            run(true, true, Some(field_order())),
            ["doc:b", "doc:c", "doc:a"]
        );
    }

    #[test]
    fn rerank_pool_recipe_scores_only_the_pool() {
        // The #119 recipe: restrict FIRST (server-side IN, post-#93), then
        // rank by ::score — never score the whole scan.
        let mut db = precedence_fixture();
        let mut run = |filter: Option<Filter>| {
            let res = db
                .execute(&[Statement::Select(Select {
                    filter,
                    order: Some(Order::Score),
                    ..select("doc")
                })])
                .unwrap();
            res[0]
                .rows
                .iter()
                .map(|r| r.record.id.to_string())
                .collect::<Vec<_>>()
        };

        // Unrestricted: ::score ranks every row — doc:b floats on its weight.
        assert_eq!(run(None), ["doc:b", "doc:a", "doc:c"]);
        // Restricted pool: only pool members are ranked (doc:b excluded even
        // though it would have won).
        let pool = run(Some(Filter::FieldIn {
            field: "topic".into(),
            values: vec![Value::Str("z".into()), Value::Str("m".into())],
        }));
        assert_eq!(pool, ["doc:a", "doc:c"], "pool only: {pool:?}");
        assert!(!pool.contains(&"doc:b".to_string()));
    }

    #[test]
    fn feedback_weight_and_value_disagree_by_design() {
        // The #119 operator table: ::score reads `weight` (when set),
        // ::votes reads `value` only — an explicit weight splits them on
        // purpose (plus the post-#85 bare downvote).
        let mut db = Database::default();
        let vote = |to: &str, voter: &str, props: BTreeMap<String, Value>, weight: Option<f32>| {
            Statement::Relate(RelationEdge {
                from: RecordId::parse(voter).unwrap(),
                name: "voted".into(),
                to: RecordId::parse(to).unwrap(),
                created_at: 0,
                weight,
                props,
            })
        };
        db.execute(&[
            create("doc", None),
            Statement::Insert(record("doc:x", BTreeMap::new(), None)),
            Statement::Insert(record("doc:y", BTreeMap::new(), None)),
            Statement::Insert(record("doc:z", BTreeMap::new(), None)),
            // x: upvote damped by an explicit weight …
            vote(
                "doc:x",
                "u:1",
                BTreeMap::from([("value".into(), Value::Int(1))]),
                Some(0.3),
            ),
            // … y: plain upvote …
            vote(
                "doc:y",
                "u:2",
                BTreeMap::from([("value".into(), Value::Int(1))]),
                None,
            ),
            // … z: bare downvote (works since #85).
            vote(
                "doc:z",
                "u:3",
                BTreeMap::from([("value".into(), Value::Int(-1))]),
                None,
            ),
        ])
        .unwrap();
        let mut scores = |order: Order| {
            let res = db
                .execute(&[Statement::Select(Select {
                    order: Some(order),
                    ..select("doc")
                })])
                .unwrap();
            res[0]
                .rows
                .iter()
                .map(|r| (r.record.id.to_string(), r.score))
                .collect::<Vec<_>>()
        };

        // ::score follows `weight`: (0.3+1)/3 < (1+1)/3, bare downvote lowest.
        let ranked = scores(Order::Score);
        let ids: Vec<_> = ranked.iter().map(|(id, _)| id.clone()).collect();
        assert_eq!(ids, ["doc:y", "doc:x", "doc:z"]);
        assert!((ranked[0].1 - 2.0 / 3.0).abs() < 1e-4, "{ranked:?}");
        assert!((ranked[1].1 - 1.3 / 3.0).abs() < 1e-4, "{ranked:?}");
        assert!((ranked[2].1 - 0.0).abs() < 1e-4, "{ranked:?}");

        // ::votes follows `value` only: x still counts ONE upvote (the 0.3
        // weight is ignored), z counts one downvote — nets 1, 1, -1.
        let voted = scores(Order::Votes);
        let by_id = |id: &str| {
            voted
                .iter()
                .find(|(k, _)| k == id)
                .map(|(_, s)| *s)
                .unwrap_or_else(|| panic!("{id} missing from {voted:?}"))
        };
        assert!(
            (by_id("doc:x") - 1.0).abs() < 1e-4,
            "weight ignored: {voted:?}"
        );
        assert!((by_id("doc:y") - 1.0).abs() < 1e-4, "{voted:?}");
        assert!((by_id("doc:z") - -1.0).abs() < 1e-4, "{voted:?}");
    }

    #[test]
    fn where_and_requires_every_term() {
        let mut db = Database::default();
        db.execute(&[
            create("t", Some(2)),
            Statement::Insert(record(
                "t:1",
                BTreeMap::from([("a".into(), num(1)), ("b".into(), str_("x"))]),
                Some(vec![1.0, 0.0]),
            )),
            Statement::Insert(record(
                "t:2",
                BTreeMap::from([("a".into(), num(1)), ("b".into(), str_("y"))]),
                None,
            )),
            Statement::Insert(record(
                "t:3",
                BTreeMap::from([("a".into(), num(0)), ("b".into(), str_("x"))]),
                None,
            )),
            // No `a` at all: the missing-field rule applies per term.
            Statement::Insert(record(
                "t:4",
                BTreeMap::from([("b".into(), str_("x"))]),
                None,
            )),
        ])
        .unwrap();
        let mut ids_where = |filter: Filter| {
            let res = db
                .execute(&[Statement::Select(Select {
                    filter: Some(filter),
                    ..select("t")
                })])
                .unwrap();
            res[0]
                .rows
                .iter()
                .map(|r| r.record.id.to_string())
                .collect::<Vec<_>>()
        };

        // All-of: only rows satisfying BOTH terms (t:2 fails b, t:3 fails a,
        // t:4 lacks a — its term fails independently).
        assert_eq!(
            ids_where(Filter::And(vec![
                Filter::FieldEquals {
                    field: "a".into(),
                    value: Value::Int(1),
                },
                Filter::FieldEquals {
                    field: "b".into(),
                    value: Value::Str("x".into()),
                },
            ])),
            ["t:1"]
        );
        // Term order does not matter; `a >= 0` admits 0 (t:3) but excludes
        // the field-less t:4.
        assert_eq!(
            ids_where(Filter::And(vec![
                Filter::FieldEquals {
                    field: "b".into(),
                    value: Value::Str("x".into()),
                },
                Filter::FieldCmp {
                    field: "a".into(),
                    op: CmpOp::Ge,
                    value: Value::Int(0),
                },
            ])),
            ["t:1", "t:3"]
        );
        // A term no record can satisfy (missing field) empties the AND.
        assert!(ids_where(Filter::And(vec![
            Filter::FieldEquals {
                field: "missing".into(),
                value: Value::Int(1),
            },
            Filter::FieldEquals {
                field: "a".into(),
                value: Value::Int(1),
            },
        ]))
        .is_empty());
        // IS NOT NULL composes with a predicate (only t:1 is embedded).
        assert_eq!(
            ids_where(Filter::And(vec![
                Filter::HasEmbedding,
                Filter::FieldEquals {
                    field: "b".into(),
                    value: Value::Str("x".into()),
                },
            ])),
            ["t:1"]
        );
    }

    #[test]
    fn where_and_filters_edge_props_all_of() {
        // MATCH edge-prop filters accept the same conjunction (issues
        // #93/#125): all-of against the edge's props.
        let mut db = Database::default();
        let edge = |to: &str, conf: f64, kind: &str| RelationEdge {
            from: RecordId::parse("s:1").unwrap(),
            name: "e".into(),
            to: RecordId::parse(to).unwrap(),
            created_at: 0,
            weight: None,
            props: BTreeMap::from([
                ("conf".into(), Value::Float(conf)),
                ("kind".into(), Value::Str(kind.into())),
            ]),
        };
        db.execute(&[
            create("g", None),
            Statement::Insert(record("s:1", BTreeMap::new(), None)),
            Statement::Insert(record("m:1", BTreeMap::new(), None)),
            Statement::Insert(record("m:2", BTreeMap::new(), None)),
            Statement::Insert(record("m:3", BTreeMap::new(), None)),
            Statement::Relate(edge("m:1", 0.9, "x")),
            Statement::Relate(edge("m:2", 0.2, "x")),
            Statement::Relate(edge("m:3", 0.9, "y")),
        ])
        .unwrap();
        let mut reach = |terms: Vec<Filter>| {
            let props = if terms.len() == 1 {
                Some(terms.into_iter().next().unwrap())
            } else {
                Some(Filter::And(terms))
            };
            let res = db
                .execute(&[Statement::Match(MatchPath {
                    start: RecordId::parse("s:1").unwrap(),
                    steps: vec![MatchStep {
                        direction: MatchDirection::Out,
                        name: "e".into(),
                        edge_props: props,
                    }],
                    as_of: None,
                })])
                .unwrap();
            res[0]
                .rows
                .iter()
                .map(|r| r.record.id.to_string())
                .collect::<Vec<_>>()
        };
        let conf_term = |op: CmpOp, v: f64| Filter::FieldCmp {
            field: "conf".into(),
            op,
            value: Value::Float(v),
        };
        let kind_term = |k: &str| Filter::FieldEquals {
            field: "kind".into(),
            value: Value::Str(k.into()),
        };

        // All-of: high confidence AND kind x → only m:1 (m:2 fails conf,
        // m:3 fails kind).
        assert_eq!(
            reach(vec![conf_term(CmpOp::Ge, 0.5), kind_term("x")]),
            ["m:1"]
        );
        // Single term unchanged: high confidence → m:1 and m:3.
        assert_eq!(reach(vec![conf_term(CmpOp::Ge, 0.5)]), ["m:1", "m:3"]);
        // Nothing satisfies both: high confidence AND kind x at > 0.9.
        assert!(reach(vec![conf_term(CmpOp::Gt, 0.9), kind_term("x")]).is_empty());
    }

    #[test]
    fn id_predicate_binds_to_record_identity() {
        let mut db = Database::default();
        db.execute(&[
            create("t", None),
            Statement::Insert(record(
                "t:1",
                BTreeMap::from([("b".into(), str_("x"))]),
                None,
            )),
            // A body key named `id` must NOT shadow the pseudo-field
            // (issue #128's documented precedence).
            Statement::Insert(record(
                "t:2",
                BTreeMap::from([("b".into(), str_("y")), ("id".into(), str_("t:9"))]),
                None,
            )),
            Statement::Insert(record(
                "t:3",
                BTreeMap::from([("b".into(), str_("x"))]),
                None,
            )),
        ])
        .unwrap();
        let mut ids_where = |filter: Filter| {
            let res = db
                .execute(&[Statement::Select(Select {
                    filter: Some(filter),
                    ..select("t")
                })])
                .unwrap();
            res[0]
                .rows
                .iter()
                .map(|r| r.record.id.to_string())
                .collect::<Vec<_>>()
        };
        let id = |v: &str| Value::Str(v.into());

        // The rerank-pool predicate: exactly the requested identities.
        assert_eq!(
            ids_where(Filter::FieldIn {
                field: "id".into(),
                values: vec![id("t:1"), id("t:3")],
            }),
            ["t:1", "t:3"]
        );
        // Exact identity …
        assert_eq!(
            ids_where(Filter::FieldEquals {
                field: "id".into(),
                value: id("t:2"),
            }),
            ["t:2"]
        );
        // … pseudo-field wins: t:2's BODY id ("t:9") is not matchable —
        // no record's identity is t:9.
        assert!(ids_where(Filter::FieldEquals {
            field: "id".into(),
            value: id("t:9"),
        })
        .is_empty());
        // `!=` is the exact complement of `=`.
        assert_eq!(
            ids_where(Filter::FieldCmp {
                field: "id".into(),
                op: CmpOp::Ne,
                value: id("t:1"),
            }),
            ["t:2", "t:3"]
        );
        // Composes with #125 conjunctions.
        assert_eq!(
            ids_where(Filter::And(vec![
                Filter::FieldIn {
                    field: "id".into(),
                    values: vec![id("t:1"), id("t:3")],
                },
                Filter::FieldEquals {
                    field: "b".into(),
                    value: str_("x"),
                },
            ])),
            ["t:1", "t:3"]
        );
    }

    #[test]
    fn id_on_edge_filters_is_an_ordinary_prop() {
        // Edges have no record identity: `id` in an edge-prop filter falls
        // through to a normal prop lookup (issue #128, documented in §2.3).
        let mut db = Database::default();
        let edge = |to: &str, id_prop: Option<&str>| RelationEdge {
            from: RecordId::parse("s:1").unwrap(),
            name: "e".into(),
            to: RecordId::parse(to).unwrap(),
            created_at: 0,
            weight: None,
            props: id_prop
                .map(|v| BTreeMap::from([("id".into(), Value::Str(v.into()))]))
                .unwrap_or_default(),
        };
        db.execute(&[
            create("g", None),
            Statement::Insert(record("s:1", BTreeMap::new(), None)),
            Statement::Insert(record("m:1", BTreeMap::new(), None)),
            Statement::Insert(record("m:2", BTreeMap::new(), None)),
            Statement::Relate(edge("m:1", Some("custom"))),
            Statement::Relate(edge("m:2", None)),
        ])
        .unwrap();
        let mut reach = |value: &str| {
            let res = db
                .execute(&[Statement::Match(MatchPath {
                    start: RecordId::parse("s:1").unwrap(),
                    steps: vec![MatchStep {
                        direction: MatchDirection::Out,
                        name: "e".into(),
                        edge_props: Some(Filter::FieldEquals {
                            field: "id".into(),
                            value: Value::Str(value.into()),
                        }),
                    }],
                    as_of: None,
                })])
                .unwrap();
            res[0]
                .rows
                .iter()
                .map(|r| r.record.id.to_string())
                .collect::<Vec<_>>()
        };

        // Ordinary prop lookup: only the edge carrying that prop matches.
        assert_eq!(reach("custom"), ["m:1"]);
        // No identity binding on edges: the start's own id matches nothing.
        assert!(reach("s:1").is_empty());
    }

    #[test]
    fn tables_index_tracks_every_declaration() {
        // Issue #133 step 1: the `tables` index carries every CREATE TABLE
        // (with or without a dim) — the source seed_declared reads.
        let mut db = Database::default();
        db.execute(&[
            create("with_dim", Some(4)),
            create("no_dim", None),
            Statement::Insert(record(
                "with_dim:1",
                BTreeMap::new(),
                Some(vec![1.0, 0.0, 0.0, 0.0]),
            )),
        ])
        .unwrap();
        let store = db.store();
        assert_eq!(store.tables.get("with_dim"), Some(&Some(4)));
        assert_eq!(store.tables.get("no_dim"), Some(&None));
        assert!(!store.tables.contains_key("never_declared"));
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
            as_of: None,
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
            as_of: None,
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
            as_of: None,
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
            as_of: None,
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

#[cfg(test)]
mod hybrid_tests {
    //! Hybrid retrieval (combined optimizer): `::bm25(...) AND
    //! vector::similarity(...) AND k = N` fuses the lexical and vector
    //! rankings with reciprocal-rank fusion (RRF), deterministic tie-breaks.

    use std::collections::BTreeMap;

    use nql_ir::{Filter, Knn, Record, RecordId, Select, Statement, Value};

    use crate::{Database, QueryResult};

    fn create(table: &str) -> Statement {
        Statement::CreateTable {
            table: table.to_string(),
            vector_dim: None,
        }
    }

    fn str_(s: &str) -> Value {
        Value::Str(s.to_string())
    }

    fn record(id: &str, body: BTreeMap<String, Value>, embedding: Option<Vec<f32>>) -> Statement {
        Statement::Insert(Record {
            id: RecordId::parse(id).unwrap(),
            body,
            embedding,
            created_at: 0,
        })
    }

    fn hybrid_select(table: &str, query: &str, knn: Vec<f32>, k: usize) -> Statement {
        Statement::Select(Select {
            table: table.to_string(),
            knn: Some(Knn { query: knn, k }),
            filter: Some(Filter::Bm25 {
                field: "text".into(),
                query: query.into(),
                k: None,
            }),
            ..Select::default()
        })
    }

    /// Build a store where lexical and vector signals DISAGREE:
    /// - `doc:a` matches the text query but is far from the vector query
    /// - `doc:b` is near the vector query but does not match the text
    /// - `doc:c` matches both (the winner)
    /// - `doc:d` matches neither (the loser)
    fn db() -> Database {
        let mut db = Database::default();
        db.execute(&[
            create("doc"),
            record(
                "doc:a",
                BTreeMap::from([("text".into(), str_("rust rust rust rust"))]),
                Some(vec![0.0, 0.0, 1.0]), // far from q=[1,0,0]
            ),
            record(
                "doc:b",
                BTreeMap::from([("text".into(), str_("completely unrelated topic"))]),
                Some(vec![0.98, 0.0, 0.0]), // near q
            ),
            record(
                "doc:c",
                BTreeMap::from([("text".into(), str_("rust rust rust rust rust rust"))]),
                Some(vec![0.96, 0.1, 0.0]), // near q AND strongly matches
            ),
            record(
                "doc:d",
                BTreeMap::from([("text".into(), str_("alpha beta gamma"))]),
                Some(vec![0.1, 0.2, 0.0]), // weak, non-collinear vector; no lexical match
            ),
        ])
        .unwrap();
        db
    }

    fn ids(res: &QueryResult) -> Vec<String> {
        res.rows.iter().map(|r| r.record.id.to_string()).collect()
    }

    #[test]
    fn hybrid_fusion_ranks_both_signal_winners_above_either_alone() {
        let mut db = db();
        // q = [1,0,0]; lexical "rust"; k = 3
        let res = db
            .execute(&[hybrid_select("doc", "rust", vec![1.0, 0.0, 0.0], 3)])
            .unwrap();
        let got = ids(&res[0]);
        // doc:c matches both signals → fused rank 1. doc:a (lexical) and
        // doc:b (vector) both beat doc:d (neither).
        assert_eq!(got[0], "doc:c", "both-signal doc ranks first: {got:?}");
        assert!(
            got[1] == "doc:a" || got[1] == "doc:b",
            "single-signal docs next: {got:?}"
        );
        assert_eq!(got.len(), 3, "k=3 caps the hybrid result: {got:?}");
    }

    #[test]
    fn hybrid_is_deterministic_across_equal_stores() {
        let mut a = db();
        let mut b = db();
        let plan = [hybrid_select("doc", "rust", vec![1.0, 0.0, 0.0], 4)];
        let ra = a.execute(&plan).unwrap();
        let rb = b.execute(&plan).unwrap();
        assert_eq!(ids(&ra[0]), ids(&rb[0]));
        let scores_a: Vec<f32> = ra[0].rows.iter().map(|r| r.score).collect();
        let scores_b: Vec<f32> = rb[0].rows.iter().map(|r| r.score).collect();
        assert_eq!(scores_a, scores_b);
        // Non-increasing fused scores.
        assert!(scores_a.windows(2).all(|w| w[0] >= w[1]));
    }

    #[test]
    fn hybrid_without_embedding_scores_zero_and_ranks_last() {
        let mut db = Database::default();
        db.execute(&[
            create("doc"),
            record(
                "doc:lex",
                BTreeMap::from([("text".into(), str_("rust rust rust"))]),
                None, // no embedding
            ),
            record(
                "doc:both",
                BTreeMap::from([("text".into(), str_("rust rust"))]),
                Some(vec![1.0, 0.0]),
            ),
        ])
        .unwrap();
        let res = db
            .execute(&[hybrid_select("doc", "rust", vec![1.0, 0.0], 2)])
            .unwrap();
        let got = ids(&res[0]);
        assert_eq!(got[0], "doc:both", "embedded+lexical first: {got:?}");
    }
}

#[cfg(test)]
mod temporal_tests {
    //! `AS OF <int>` time-travel reads: the historical view is a pure
    //! function of (mutation history, cutoff) — replay-based, deterministic.

    use std::collections::BTreeMap;

    use nql_ir::{Filter, Record, RecordId, Select, Statement, Value};

    use crate::{Database, QueryResult};

    fn create(table: &str) -> Statement {
        Statement::CreateTable {
            table: table.to_string(),
            vector_dim: None,
        }
    }

    fn record(id: &str, body: BTreeMap<String, Value>) -> Statement {
        Statement::Insert(Record {
            id: RecordId::parse(id).unwrap(),
            body,
            embedding: None,
            created_at: 0,
        })
    }

    fn str_(s: &str) -> Value {
        Value::Str(s.to_string())
    }

    fn select(table: &str, as_of: Option<i64>) -> Statement {
        Statement::Select(Select {
            table: table.to_string(),
            knn: None,
            filter: None,
            order: None,
            limit: None,
            as_of,
            fields: None,
            offset: None,
            aggregate: None,
        })
    }

    fn ids(res: &QueryResult) -> Vec<String> {
        res.rows.iter().map(|r| r.record.id.to_string()).collect()
    }

    #[test]
    fn as_of_before_insert_is_empty() {
        let mut db = Database::default();
        db.execute(&[
            create("doc"),
            record("doc:1", BTreeMap::from([("text".into(), str_("a"))])),
        ])
        .unwrap();
        // doc:1 was the 2nd mutation (clock 2); AS OF 1 sees nothing.
        let res = db.execute(&[select("doc", Some(1))]).unwrap();
        assert!(res[0].rows.is_empty(), "ts=1 predates the insert");
    }

    #[test]
    fn as_of_after_insert_sees_the_record() {
        let mut db = Database::default();
        db.execute(&[
            create("doc"),
            record("doc:1", BTreeMap::from([("text".into(), str_("a"))])),
        ])
        .unwrap();
        let res = db.execute(&[select("doc", Some(2))]).unwrap();
        assert_eq!(ids(&res[0]), ["doc:1"]);
    }

    #[test]
    fn as_of_upsert_history_reconstructs_old_version() {
        let mut db = Database::default();
        db.execute(&[
            create("doc"),
            record("doc:1", BTreeMap::from([("text".into(), str_("v1"))])),
            record("doc:1", BTreeMap::from([("text".into(), str_("v2"))])),
        ])
        .unwrap();
        // Current store has v2; AS OF 2 (after the v1 insert, before v2).
        let now = db.execute(&[select("doc", None)]).unwrap();
        assert_eq!(now[0].rows[0].record.body.get("text"), Some(&str_("v2")));
        let past = db.execute(&[select("doc", Some(2))]).unwrap();
        assert_eq!(
            past[0].rows[0].record.body.get("text"),
            Some(&str_("v1")),
            "replay reconstructs the intermediate version"
        );
    }

    #[test]
    fn as_of_before_forget_restores_record() {
        let mut db = Database::default();
        db.execute(&[
            create("doc"),
            record("doc:1", BTreeMap::from([("text".into(), str_("a"))])),
            Statement::Forget {
                id: RecordId::parse("doc:1").unwrap(),
            },
        ])
        .unwrap();
        // Current store: forgotten.
        let now = db.execute(&[select("doc", None)]).unwrap();
        assert!(now[0].rows.is_empty());
        // AS OF 2 (after insert, before forget): record exists again.
        let past = db.execute(&[select("doc", Some(2))]).unwrap();
        assert_eq!(ids(&past[0]), ["doc:1"]);
    }

    #[test]
    fn as_of_cutoff_beyond_history_is_full_state() {
        let mut db = Database::default();
        db.execute(&[
            create("doc"),
            record("doc:1", BTreeMap::from([("text".into(), str_("a"))])),
        ])
        .unwrap();
        let res = db.execute(&[select("doc", Some(1_000_000))]).unwrap();
        assert_eq!(ids(&res[0]), ["doc:1"]);
    }

    #[test]
    fn as_of_is_deterministic_across_equal_stores() {
        let mut a = Database::default();
        let mut b = Database::default();
        let plan = [
            create("doc"),
            record("doc:1", BTreeMap::from([("text".into(), str_("x"))])),
            record("doc:2", BTreeMap::from([("text".into(), str_("y"))])),
        ];
        a.execute(&plan).unwrap();
        b.execute(&plan).unwrap();
        let q = [select("doc", Some(2)), select("doc", Some(3))];
        let ra = a.execute(&q).unwrap();
        let rb = b.execute(&q).unwrap();
        assert_eq!(ids(&ra[0]), ids(&rb[0]));
        assert_eq!(ids(&ra[1]), ids(&rb[1]));
        assert_eq!(ra[0].rows.len(), 1);
        assert_eq!(ra[1].rows.len(), 2);
    }

    #[test]
    fn as_of_respects_field_filter_on_historical_view() {
        let mut db = Database::default();
        db.execute(&[
            create("doc"),
            record("doc:1", BTreeMap::from([("text".into(), str_("alpha"))])),
            record("doc:2", BTreeMap::from([("text".into(), str_("beta"))])),
        ])
        .unwrap();
        let mut sel = match select("doc", Some(2)) {
            Statement::Select(s) => s,
            _ => unreachable!(),
        };
        sel.filter = Some(Filter::FieldEquals {
            field: "text".into(),
            value: str_("alpha"),
        });
        let res = db.execute(&[Statement::Select(sel)]).unwrap();
        assert_eq!(
            ids(&res[0]),
            ["doc:1"],
            "filter applies to the replayed view"
        );
    }
}

#[cfg(test)]
mod memory_tests {
    //! `MEMORY <name>` context blocks: named sub-stores for core/archival/
    //! shared partitions. Each memory has its own records, edges, and
    //! history — so AS OF composes with memory scoping.

    use std::collections::BTreeMap;

    use nql_ir::{Record, RecordId, Select, Statement, Value};

    use crate::{Database, QueryResult};

    fn create(table: &str) -> Statement {
        Statement::CreateTable {
            table: table.to_string(),
            vector_dim: None,
        }
    }

    fn record(id: &str, body: BTreeMap<String, Value>) -> Statement {
        Statement::Insert(Record {
            id: RecordId::parse(id).unwrap(),
            body,
            embedding: None,
            created_at: 0,
        })
    }

    fn str_(s: &str) -> Value {
        Value::Str(s.to_string())
    }

    fn select(table: &str) -> Statement {
        Statement::Select(Select {
            table: table.to_string(),
            knn: None,
            filter: None,
            order: None,
            limit: None,
            as_of: None,
            fields: None,
            offset: None,
            aggregate: None,
        })
    }

    fn ids(res: &QueryResult) -> Vec<String> {
        res.rows.iter().map(|r| r.record.id.to_string()).collect()
    }

    #[test]
    fn memory_isolates_records_from_root() {
        let mut db = Database::default();
        db.execute(&[
            create("doc"),
            record("doc:1", BTreeMap::from([("text".into(), str_("root"))])),
            Statement::Memory {
                name: "core".into(),
            },
            create("doc"),
            record("doc:1", BTreeMap::from([("text".into(), str_("core"))])),
        ])
        .unwrap();

        // Root view: only the root doc:1.
        let root = db.execute(&[select("doc")]).unwrap();
        assert_eq!(ids(&root[0]), ["doc:1"]);
        assert_eq!(root[0].rows[0].record.body.get("text"), Some(&str_("root")));

        // Memory view: the core doc:1 (separate store, same id).
        let core = db
            .execute(&[
                Statement::Memory {
                    name: "core".into(),
                },
                select("doc"),
            ])
            .unwrap();
        assert_eq!(ids(&core[0]), ["doc:1"]);
        assert_eq!(
            core[0].rows[0].record.body.get("text"),
            Some(&str_("core")),
            "same id, different memory = different record"
        );
    }

    #[test]
    fn memory_is_lazily_created() {
        let mut db = Database::default();
        db.execute(&[
            Statement::Memory {
                name: "fresh".into(),
            },
            create("t"),
            record("t:1", BTreeMap::from([("x".into(), Value::Int(1))])),
        ])
        .unwrap();
        // The memory exists with its record; root has nothing.
        assert_eq!(db.store().memories.len(), 1);
        assert!(db.store().records.is_empty());
        let res = db
            .execute(&[
                Statement::Memory {
                    name: "fresh".into(),
                },
                select("t"),
            ])
            .unwrap();
        assert_eq!(ids(&res[0]), ["t:1"]);
    }

    #[test]
    fn memory_history_is_isolated_for_as_of() {
        let mut db = Database::default();
        db.execute(&[
            create("t"),
            record("t:1", BTreeMap::from([("v".into(), str_("root1"))])),
            Statement::Memory {
                name: "core".into(),
            },
            create("t"),
            record("t:1", BTreeMap::from([("v".into(), str_("core1"))])),
            record("t:1", BTreeMap::from([("v".into(), str_("core2"))])),
        ])
        .unwrap();

        // Core memory: AS OF 2 (its own clock: create=1, insert=2) sees core1.
        let past = db
            .execute(&[
                Statement::Memory {
                    name: "core".into(),
                },
                Statement::Select(Select {
                    table: "t".into(),
                    knn: None,
                    filter: None,
                    order: None,
                    limit: None,
                    as_of: Some(2),
                    fields: None,
                    offset: None,
                    aggregate: None,
                }),
            ])
            .unwrap();
        assert_eq!(
            past[0].rows[0].record.body.get("v"),
            Some(&str_("core1")),
            "each memory keeps its own history clock"
        );

        // Root: AS OF 2 (root create=1, root insert=2) sees root1 — the root
        // clock is unaffected by memory writes.
        let root_past = db
            .execute(&[Statement::Select(Select {
                table: "t".into(),
                knn: None,
                filter: None,
                order: None,
                limit: None,
                as_of: Some(2),
                fields: None,
                offset: None,
                aggregate: None,
            })])
            .unwrap();
        assert_eq!(
            root_past[0].rows[0].record.body.get("v"),
            Some(&str_("root1"))
        );
    }

    #[test]
    fn memory_outside_plan_errors() {
        // The public `execute` always goes through execute_plan (which handles
        // MEMORY context), so the error is only reachable when a caller uses
        // execute_statement directly with a MEMORY statement.
        let mut store = nql_ir::Store::default();
        let err = crate::engine::execute_statement(
            &mut store,
            &Statement::Memory { name: "x".into() },
            None,
            None,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("MEMORY"),
            "direct Memory statement must error: {err}"
        );
    }

    #[test]
    fn memory_scoping_is_deterministic_across_equal_stores() {
        let mut a = Database::default();
        let mut b = Database::default();
        let plan = [
            create("t"),
            record("t:1", BTreeMap::from([("v".into(), str_("root"))])),
            Statement::Memory {
                name: "core".into(),
            },
            create("t"),
            record("t:1", BTreeMap::from([("v".into(), str_("core"))])),
        ];
        a.execute(&plan).unwrap();
        b.execute(&plan).unwrap();
        let q = [
            select("t"),
            Statement::Memory {
                name: "core".into(),
            },
            select("t"),
        ];
        let ra = a.execute(&q).unwrap();
        let rb = b.execute(&q).unwrap();
        assert_eq!(ids(&ra[0]), ids(&rb[0]));
        assert_eq!(ids(&ra[1]), ids(&rb[1]));
        assert_eq!(ra[0].rows.len(), 1);
        assert_eq!(ra[1].rows.len(), 1);
    }
}

#[cfg(test)]
mod snapshot_tests {
    //! Issue #166: replay-snapshot ring mechanics + replay-from-base
    //! correctness (fast, always-on). The in-session capture plumbing
    //! (`Database` counters/epochs) is exercised by the temporal bench
    //! groups and the experiments parity runs.

    use std::collections::BTreeMap;

    use nql_ir::{Record, RecordId, Statement, Store, Value};

    use super::{execute_statement, replay_as_of, strip_history, SnapshotRing};

    fn insert(id: &str, v: i64) -> Statement {
        Statement::Insert(Record {
            id: RecordId::parse(id).unwrap(),
            body: BTreeMap::from([("v".into(), Value::Int(v))]),
            embedding: None,
            created_at: 0,
        })
    }

    #[test]
    fn ring_selects_newest_base_at_or_below_cutoff() {
        let mut ring = SnapshotRing::default();
        assert!(ring.select(100).is_none());
        let s50 = Store {
            clock: 50,
            ..Store::default()
        };
        let s80 = Store {
            clock: 80,
            ..Store::default()
        };
        ring.push(50, &s50);
        ring.push(80, &s80);
        assert!(ring.select(49).is_none());
        assert_eq!(ring.select(50).map(|(c, _)| c), Some(50));
        assert_eq!(ring.select(79).map(|(c, _)| c), Some(50));
        assert_eq!(ring.select(80).map(|(c, _)| c), Some(80));
        assert_eq!(ring.select(10_000).map(|(c, _)| c), Some(80));
    }

    #[test]
    fn ring_evicts_oldest_and_resets_on_regression() {
        let mut ring = SnapshotRing::default();
        for c in [10i64, 20, 30] {
            let s = Store {
                clock: c,
                ..Store::default()
            };
            ring.push(c, &s);
        }
        // Cap is SNAPSHOT_RING_CAP (2): the oldest base is evicted.
        assert!(ring.select(15).is_none());
        assert_eq!(ring.select(20).map(|(c, _)| c), Some(20));
        // Clock regression (a different lineage) resets the ring.
        let s = Store {
            clock: 5,
            ..Store::default()
        };
        ring.push(5, &s);
        assert!(ring.select(4).is_none());
        assert_eq!(ring.select(25).map(|(c, _)| c), Some(5));
    }

    // Replaying from a snapshot base yields exactly the from-scratch view —
    // across appends, a mid-history PRUNE (the #165 bug class: bases must
    // compose with Snapshot installs), FORGETs, and a future cutoff (the
    // recent-T fast path).
    #[test]
    fn replay_from_base_matches_full_replay() {
        let stmts = vec![
            Statement::CreateTable {
                table: "t".into(),
                vector_dim: None,
            },
            insert("t:1", 1),
            insert("t:2", 2),
            Statement::PruneHistory,
            insert("t:3", 3),
            Statement::Forget {
                id: RecordId::parse("t:1").unwrap(),
            },
        ];
        let mut store = Store::default();
        for stmt in &stmts {
            execute_statement(&mut store, stmt, None, None).unwrap();
        }
        // Bases at two interior clocks (prefix states, as live capture
        // would record them).
        let mut ring = SnapshotRing::default();
        for upto in [2usize, 4] {
            let mut prefix = Store::default();
            for stmt in &stmts[..=upto] {
                execute_statement(&mut prefix, stmt, None, None).unwrap();
            }
            let clock = prefix.clock;
            ring.push(clock, &prefix);
        }
        let max_ts = store.history.iter().map(|(ts, _)| *ts).max().unwrap();
        for cutoff in [0, 1, 2, 3, 4, 5, 6, 7, max_ts, max_ts + 100] {
            let full = replay_as_of(&store, cutoff, None);
            let based = replay_as_of(&store, cutoff, Some(&ring));
            match (full, based) {
                // Views are transient (rows read state only): compare
                // history-stripped states. Histories legitimately differ —
                // a replayed `Snapshot` entry installs state (wiping the
                // view's accumulated log) while a base starts log-free —
                // but records, edges, clocks, and tables must coincide.
                (Ok(a), Ok(b)) => assert_eq!(
                    strip_history(&a),
                    strip_history(&b),
                    "cutoff {cutoff}: snapshot-base view diverged from full replay"
                ),
                (Err(_), Err(_)) => {}
                (a, b) => panic!("cutoff {cutoff}: ok/err split: {a:?} vs {b:?}"),
            }
        }
    }
}
