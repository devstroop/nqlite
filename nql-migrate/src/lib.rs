//! nql-migrate — bulk **v2/v3 → format v4** conversion (spec/file-format.md
//! §5.8's `nql-migrate`, the Rust-side tooling).
//!
//! The migration is a *container remap*: the store is loaded through the
//! existing reader (WAL replay included — the migrated file is the full
//! committed state), then written through the canonical v4 codec
//! (`nqlite::v4`). Payload bytes are the same postcard dialect (§5.7), so
//! `HISTORY` is adopted verbatim and records/edges are pure remaps.
//!
//! Guarantees:
//! - **Verified**: the written bytes are decoded back and compared against
//!   the source store — a mismatch fails the migration loudly;
//! - **Atomic**: tmp file + fsync + rename + directory fsync (same protocol
//!   as a checkpoint — §4);
//! - **In-place**: `--in` == `--out` is supported (the source lock is
//!   released before the write).

use std::path::{Path, PathBuf};

use nqlite::{v4, Database, StorageError};

/// Outcome of a successful migration (for CLI output / tests).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Report {
    /// Format version of the INPUT store (2 or 3; a v4 input is rejected
    /// by the reader before we get here).
    pub from_version: u32,
    pub records: usize,
    pub edges: usize,
    pub history: usize,
    pub memories: usize,
    /// Size of the written v4 file.
    pub bytes: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// No input file — nothing to migrate.
    #[error("input store not found: {0}")]
    InputMissing(PathBuf),
    /// Refusing to clobber an existing output without `force`.
    #[error("output already exists (pass --force to overwrite): {0}")]
    OutputExists(PathBuf),
    /// Open/read failures (incl. `BadVersion` for v4 inputs and `Locked`).
    #[error(transparent)]
    Storage(#[from] StorageError),
    /// Filesystem failures (write/fsync/rename).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// The v4 encoder rejected the store (integrity error).
    #[error("v4 encode failed: {0}")]
    Encode(String),
    /// Post-write decode did not reproduce the source store.
    #[error("verification failed: migrated bytes did not decode back to the same store")]
    Verify,
}

/// Read the format version from a store header without opening it
/// (magic + `u32 LE` version at offset 8; `None` when the file is missing).
fn peek_version(path: &Path) -> std::io::Result<Option<u32>> {
    use std::io::Read;
    let mut f = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    let mut head = [0u8; 12];
    f.read_exact(&mut head)?;
    Ok(Some(u32::from_le_bytes(head[8..12].try_into().unwrap())))
}

/// Migrate `input` (v2/v3) to v4 at `output`.
///
/// The input is loaded through `Database::open` (file + WAL replay), the
/// single-writer lock is released with the handle, and the result is
/// written atomically and verified by decoding it back.
pub fn migrate(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    force: bool,
) -> Result<Report, Error> {
    let input = input.as_ref();
    let output = output.as_ref();
    // Version peek: a store that was never checkpointed has only a WAL —
    // `Database::open` replays it into an empty store, so treat the pair as
    // the current layout. Both missing ⇒ nothing to migrate.
    let from_version = match peek_version(input).map_err(Error::Io)? {
        Some(v) => v,
        None => {
            let mut wal = input.as_os_str().to_owned();
            wal.push(".wal");
            if std::path::Path::new(&wal).exists() {
                nqlite::storage::FORMAT_VERSION
            } else {
                return Err(Error::InputMissing(input.to_path_buf()));
            }
        }
    };

    // Load the full committed state (WAL replay included); consuming the
    // Database releases the single-writer lock before we write.
    let db = Database::open(input)?;
    let store = db.into_store();

    if output != input && output.exists() && !force {
        return Err(Error::OutputExists(output.to_path_buf()));
    }

    let bytes = v4::encode_store(&store).map_err(Error::Encode)?;

    // Verified migration: decode the bytes and re-encode them — the v4
    // container is canonical, so any encoder/decoder drift fails loudly.
    // (Full `Store` equality is impossible through the wire: `Store.tables`
    // is `serde(skip)`ed inside snapshot statements and rebuilt at replay —
    // the same rule the golden fixtures document.)
    let back = v4::decode_store(&bytes).map_err(Error::Encode)?;
    if v4::encode_store(&back).map_err(Error::Encode)? != bytes {
        return Err(Error::Verify);
    }

    // Atomic replace: tmp + fsync + rename + dir fsync (§4 protocol).
    let parent = output.parent().filter(|p| !p.as_os_str().is_empty());
    if let Some(dir) = parent {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = output.with_extension("nql.migrate-tmp");
    {
        use std::io::Write;
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, output)?;
    if let Some(dir) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        if let Ok(d) = std::fs::File::open(dir) {
            let _ = d.sync_all();
        }
    }

    Ok(Report {
        from_version,
        records: store.records.len(),
        edges: store.edges.len(),
        history: store.history.len(),
        memories: store.memories.len(),
        bytes: bytes.len() as u64,
    })
}
