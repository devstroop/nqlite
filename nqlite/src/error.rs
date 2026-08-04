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
}

/// Convenience alias used across the crate.
pub type Result<T> = std::result::Result<T, Error>;
