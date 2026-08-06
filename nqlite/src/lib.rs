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

pub use bm25::{tokenize, Bm25Index, B, K1};
pub use engine::{
    cosine_similarity, execute_plan, execute_statement, QueryKind, QueryResult, ScoredRecord,
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
        Self { store, file: None }
    }

    /// Open (or create) a persistent database at `path` (e.g. `data.nql`).
    ///
    /// Loads the main file + replays the WAL (crash recovery), then logs every
    /// subsequent mutating plan to the WAL before returning. Call [`flush`]
    /// to checkpoint the WAL into the main file.
    pub fn open(path: impl AsRef<std::path::Path>) -> std::result::Result<Self, StorageError> {
        let file = storage::StoreFile::open(path)?;
        let (store, _replayed) = file.load()?;
        Ok(Self {
            store,
            file: Some(file),
        })
    }

    /// Checkpoint the WAL into the main file (no-op for in-memory databases).
    pub fn flush(&mut self) -> std::result::Result<(), StorageError> {
        if let Some(file) = &mut self.file {
            file.checkpoint(&self.store)?;
        }
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
    /// statements and returning one [`QueryResult`] per `SELECT` statement.
    ///
    /// Statements before a `SELECT` are applied first, so a single plan may
    /// create tables, insert records, relate edges, and query them in one
    /// pass. Execution is deterministic for a given input plan + store.
    ///
    /// For a persistent database, every mutating statement is appended to the
    /// WAL (fsync'd) before this returns, and an automatic checkpoint happens
    /// once the WAL crosses `CHECKPOINT_THRESHOLD`.
    pub fn execute(&mut self, plan: &[Statement]) -> Result<Vec<QueryResult>> {
        let results = execute_plan(&mut self.store, plan)?;
        if let Some(file) = &mut self.file {
            for stmt in plan {
                if is_mutating(stmt) {
                    file.append(stmt)?;
                }
            }
            if file.needs_checkpoint() {
                file.checkpoint(&self.store)?;
            }
        }
        Ok(results)
    }
}

/// True for statements that change the store and therefore belong in the WAL.
/// Read-only statements (`SELECT`, `MATCH`) are never logged.
fn is_mutating(stmt: &Statement) -> bool {
    !matches!(stmt, Statement::Select(_) | Statement::Match(_))
}
