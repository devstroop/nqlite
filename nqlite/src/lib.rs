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

pub mod engine;
pub mod error;

pub use engine::{cosine_similarity, execute_plan, execute_statement, QueryResult, ScoredRecord};
pub use error::{Error, Result};
// Facade: re-export the shared IR contract (RecordId, Value, Record, Store,
// Statement, Select, Knn, Filter, Order, Plan, ...) so callers can use
// `nqlite::Value` instead of reaching into `nql_ir` directly.
pub use nql_ir::*;

/// An open, in-memory database handle over a [`Store`].
///
/// This is the crate's primary public entry point: open a store, execute a
/// [`Plan`], and read back the resulting [`QueryResult`]s.
#[derive(Debug)]
pub struct Database {
    store: Store,
}

impl Default for Database {
    fn default() -> Self {
        Self::new(Store::default())
    }
}

impl Database {
    /// Open a database over an existing (possibly non-empty) store.
    pub fn new(store: Store) -> Self {
        Self { store }
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
    pub fn execute(&mut self, plan: &[Statement]) -> Result<Vec<QueryResult>> {
        execute_plan(&mut self.store, plan)
    }
}
