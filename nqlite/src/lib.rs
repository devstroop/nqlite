//! nqlite engine. Deterministic storage + execution. Zero LLM dependency.
//!
//! This crate executes an [`nql_ir::Plan`] against an [`nql_ir::Store`] fully
//! offline and deterministically: vectors arrive as BYO `f32` arrays (the
//! engine never computes embeddings), and every sort/aggregation is stable and
//! tie-broken by RecordId so identical inputs always produce identical output.
//!
//! # Determinism guarantees
//!
//! - Records live in a `BTreeMap<RecordId, ..>`, so scans are in stable key
//!   order.
//! - Edges are an append-only `Vec`; aggregation over them is a fixed,
//!   deterministic pass.
//! - All ordered queries break numeric ties by ascending [`RecordId`]
//!   (`Ord`), and unordered scans return since `BTreeMap` key order.
//! - No network, no LLM, no `rand`, no wall-clock in any execution path.
//!
//! # Vector index swap point
//!
//! kNN SELECTs run through the [`VectorIndex`] trait ([`index`] module). The
//! default [`BruteForceVectorIndex`] is an exact, deterministic cosine scan,
//! which is what keeps the engine's determinism guarantee. An approximate
//! `HnswVectorIndex` is compiled only with the opt-in `hnsw` cargo feature
//! (`cargo build --features hnsw`) and is **never** the engine default —
//! approximate search can miss true neighbours. To swap the retrieval
//! strategy, replace the concrete index in `engine::build_default_index`.

pub mod bm25;
pub mod engine;
pub mod error;
pub mod harness;
pub mod index;
pub mod storage;
pub mod v4;

pub use bm25::{tokenize, Bm25Index, B, K1};
pub use engine::{
    cosine_similarity, execute_plan, execute_statement, IndexCache, QueryKind, QueryResult,
    ScoredRecord,
};
pub use error::{Error, Result};
#[cfg(feature = "hnsw")]
pub use index::HnswVectorIndex;
pub use index::{BruteForceVectorIndex, VectorIndex};
pub use storage::{StorageError, StoreFile};
// Facade: re-export the shared IR contract (RecordId, Value, Record, Store,
// Statement, Select, Knn, Filter, Order, Plan, ...) so callers can use
// `nqlite::Value` instead of reaching into `nql_ir` directly.
pub use nql_ir::*;

/// An open database handle over a [`Store`].
///
/// In-memory by default (`Database::new`); `Database::open` additionally
/// persists every mutating plan to a single file + sidecar WAL (see
/// `spec/file-format.md`). Both modes are deterministic and zero-LLM.
#[derive(Debug)]
pub struct Database {
    store: Store,
    /// Memoized whole-table kNN vector index (issue #144, L2) — lazily
    /// invalidated by the store's identity + `clock` (see [`engine::IndexCache`]).
    index_cache: engine::IndexCache,
    /// Present when opened via [`Database::open`] — the persisted store file.
    file: Option<storage::StoreFile>,
}

impl Default for Database {
    fn default() -> Self {
        Self::new(Store::default())
    }
}

impl Database {
    /// Open a database over an existing (possibly non-empty) in-memory store.
    pub fn new(store: Store) -> Self {
        Self {
            store,
            index_cache: engine::IndexCache::default(),
            file: None,
        }
    }

    /// Open (or create) a persistent database at `path` (e.g. `data.ndb`).
    ///
    /// Loads the main file + replays the WAL (crash recovery), then logs every
    /// subsequent mutating plan to the WAL before returning. Call [`flush`]
    /// to checkpoint the WAL into the main file.
    pub fn open(path: impl AsRef<std::path::Path>) -> std::result::Result<Self, StorageError> {
        let file = storage::StoreFile::open(path)?;
        let (store, _replayed) = file.load()?;
        Ok(Self {
            store,
            index_cache: engine::IndexCache::default(),
            file: Some(file),
        })
    }

    /// Checkpoint the WAL into the main file (no-op for in-memory databases).
    pub fn flush(&mut self) -> std::result::Result<(), StorageError> {
        // Claim the lazy tail FIRST: a checkpoint rewrites
        // `store.history` into the main file, so flushing a never-ensured
        // session would silently drop the file-era history.
        self.ensure_history()?;
        if let Some(file) = &mut self.file {
            file.checkpoint(&self.store)?;
        }
        Ok(())
    }

    /// Decode the history tail on first use (issue #133): temporal
    /// statements pay the decode once per session; everything else never
    /// touches it. File entries all predate anything WAL replay appended
    /// (load runs before replay), so prepending keeps timestamps ascending.
    /// In-memory databases and legacy (v2) files have no pending tail.
    ///
    /// Idempotent (the range is taken once). **Public because anything that
    /// serializes `store.history`** (flush / threshold checkpoint,
    /// `nql-migrate`) **must claim the tail first** — otherwise the file era
    /// is silently rewritten as empty (found via the E08 100k importer
    /// proof: chunked `:flush` produced a store with `history = 0`).
    pub fn ensure_history(&mut self) -> std::result::Result<(), StorageError> {
        let Some(file) = self.file.as_ref() else {
            return Ok(());
        };
        let Some((offset, len)) = file.take_history_range() else {
            return Ok(());
        };
        let mut merged = file.read_history(offset, len)?;
        merged.append(&mut self.store.history);
        self.store.history = merged;
        Ok(())
    }

    /// Immutable access to the current store snapshot.
    pub fn store(&self) -> &Store {
        &self.store
    }

    /// Consume the database, returning the underlying store.
    pub fn into_store(self) -> Store {
        self.store
    }

    /// Execute a [`Plan`] against the store, mutating it for DDL/DML
    /// statements and returning one [`QueryResult`] per query statement
    /// (`SELECT`, `MATCH`, `CLOSURE`; DDL/DML contribute nothing).
    ///
    /// Statements before a `SELECT` are applied first, so a single plan may
    /// create tables, insert records, relate edges, and query them in one
    /// pass. Execution is deterministic for a given input plan + store.
    ///
    /// For a persistent database, all of the plan's mutating statements are
    /// appended to the WAL and fsynced **once per plan** before this returns
    /// (issue #164; `spec/file-format.md` §2: one `execute` call = one
    /// transaction), and an automatic checkpoint happens once the WAL
    /// crosses `CHECKPOINT_THRESHOLD`.
    pub fn execute(&mut self, plan: &[Statement]) -> Result<Vec<QueryResult>> {
        // Lazy history (issue #133): temporal statements claim the history
        // tail's decode once per session; current-state queries skip it.
        if plan.iter().any(needs_history) {
            self.ensure_history()?;
        }
        let results = execute_plan(&mut self.store, plan, Some(&mut self.index_cache))?;
        if let Some(file) = &mut self.file {
            // One write + fsync per plan (issue #164 — the granularity
            // `spec/file-format.md` §2 already specifies): collect the
            // mutating statements, add the plan-boundary marker, and append
            // the whole batch in a single durable write.
            let mut batch: Vec<&Statement> = plan.iter().filter(|s| is_mutating(s)).collect();
            if !batch.is_empty() {
                // Plan-boundary marker (issue #109): replay must reset the
                // memory context exactly where the runtime did — every plan
                // starts at the root (spec §2.8), and a plan that ends inside
                // a MEMORY block would otherwise leak its context into every
                // later frame during WAL replay.
                let context_reset = Statement::ContextReset;
                batch.push(&context_reset);
                file.append_batch(&batch)?;
            }
        }
        // Threshold checkpoint, decided AFTER the appends (wal_len must
        // reflect THIS plan — a pre-append check never fires for a
        // single-plan session, which is exactly how chunked ingest grows
        // the WAL). The checkpoint re-serializes `store.history` into the
        // main file: claim the lazy tail first, or the file era is
        // rewritten as absent (found by the E08 100k importer proof —
        // migrated store reported `history = 0`).
        if matches!(&self.file, Some(f) if f.needs_checkpoint()) {
            self.ensure_history()?;
            if let Some(file) = &mut self.file {
                file.checkpoint(&self.store)?;
            }
        }
        Ok(results)
    }
}

/// True for statements that change the store (or its context) and therefore
/// belong in the WAL. Read-only statements (`SELECT`, `MATCH`, `MATCH ... COUNT`,
/// `CLOSURE`) are never logged. `MEMORY` is logged: it carries the context switch that WAL
/// replay needs to reconstruct memory scoping. `ContextReset` is a WAL-only
/// sequencing marker appended by [`Database::execute`] itself — it is not a
/// plan statement and never enters `Store::history`. `Snapshot` entries live
/// only inside history (created by `PRUNE HISTORY`, issue #95) and are never
/// WAL frames; `PRUNE HISTORY` itself IS logged, so compaction survives a
/// reopen without an explicit flush.
fn is_mutating(stmt: &Statement) -> bool {
    !matches!(
        stmt,
        Statement::Select(_)
            | Statement::Match(_)
            | Statement::MatchCount(_)
            | Statement::Closure(_)
            | Statement::HistorySince(_)
            | Statement::Snapshot(_)
            | Statement::ContextReset
    )
}

/// True when a statement reads the mutation history — the only triggers for
/// decoding the lazily-loaded history frame (issue #133). A `PruneHistory`
/// needs it too: compaction must retain declarations from the *full* log.
fn needs_history(stmt: &Statement) -> bool {
    match stmt {
        Statement::Select(s) => s.as_of.is_some(),
        Statement::Match(p) | Statement::MatchCount(p) | Statement::Closure(p) => p.as_of.is_some(),
        Statement::HistorySince(_) | Statement::PruneHistory => true,
        _ => false,
    }
}
