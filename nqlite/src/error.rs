//! Engine error type.

use thiserror::Error;

/// Errors raised while executing a [`nql_ir::Plan`].
#[derive(Debug, Clone, PartialEq, Error)]
pub enum Error {
    /// An `INSERT` carried an embedding whose length differs from the table's
    /// declared `VECTOR` dimension.
    #[error(
        "embedding dimension mismatch for table `{table}`: declared dim {expected}, got {actual}"
    )]
    EmbeddingDimMismatch {
        table: String,
        expected: usize,
        actual: usize,
    },

    /// A persistence error (WAL append / checkpoint) on a file-backed database.
    #[error("storage error: {0}")]
    Storage(#[from] crate::storage::StorageError),

    /// A `MEMORY <name>` statement executed outside a plan's memory context
    /// (e.g. directly via `execute_statement`): memory switching only has
    /// meaning as part of a plan, where it scopes subsequent statements.
    #[error("`MEMORY {name}` must run inside a plan to switch context")]
    MemoryWithoutContext { name: String },

    /// A temporal read (`AS OF`) asked for a timestamp earlier than the
    /// store's history snapshot: compaction (`PRUNE HISTORY`, issue #95)
    /// dropped the mutations that would reconstruct it.
    #[error(
        "history before ts {pruned_through} was compacted (PRUNE HISTORY); AS OF timestamps earlier than the snapshot are no longer available"
    )]
    HistoryPruned { pruned_through: i64 },
}

/// Convenience alias used across the crate.
pub type Result<T> = std::result::Result<T, Error>;
