//! Single-file persistence with a sidecar write-ahead log (M1).
//!
//! Byte-deterministic format (see `spec/file-format.md`):
//! - main file: magic + version + length-prefixed core frame + history tail
//!   (v3; the legacy single-payload v2 layout still loads, issue #133)
//! - WAL: append-only frames of `crc32(len || payload)`, len, postcard(Statement)
//!
//! Crash-safety: the main file is replaced atomically (tmp + rename + fsync);
//! the WAL is replayed on open and truncated at the first torn frame (CRC
//! mismatch / bad length). Acknowledged transactions survive crashes; a crash
//! mid-commit can only drop the in-flight transaction, never corrupt.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use nql_ir::{Record, RecordId, RelationEdge, Statement, Store};
use thiserror::Error;

/// Checkpoint the WAL into the main file once it exceeds this size.
pub const CHECKPOINT_THRESHOLD: u64 = 1 << 20; // 1 MiB

const MAGIC: &[u8; 8] = b"NQLITE01";
/// Current layout (issue #133): length-prefixed core frame + history tail.
pub const FORMAT_VERSION: u32 = 3;
/// Previous layout (single inline `postcard(Store)` payload) — still readable.
pub const LEGACY_VERSION: u32 = 2;
/// Format v4 (spec §5) — the adopted container `nql-migrate` writes.
/// The engine reads it since #157, and checkpoints PRESERVE it (a v4
/// store rewritten as a v3 core frame would be a silent downgrade).
pub const V4_VERSION: u32 = 4;
const WAL_SUFFIX: &str = ".wal";
const LOCK_SUFFIX: &str = ".lock";

/// Storage errors (all recoverable by re-opening / re-checkpointing).
#[derive(Debug, Error)]
pub enum StorageError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Postcard(#[from] postcard::Error),
    #[error(
        "unsupported format version {0} (supported: {LEGACY_VERSION}, {FORMAT_VERSION}, {V4_VERSION})"
    )]
    BadVersion(u32),
    #[error("v4 container: {0}")]
    V4(String),
    #[error("truncated main file (core/history frame out of bounds)")]
    Truncated,
    #[error("bad magic header")]
    BadMagic,
    #[error("torn/corrupt WAL frame at offset {0} (truncated)")]
    TornFrame(u64),
    #[error(
        "store is locked by another process (lock file: {0}); \
         on platforms without advisory locking, delete the lock file only \
         if no nql process is using this store"
    )]
    Locked(PathBuf),
}

// `std::io::Error` is neither `Clone` nor `Eq`, so derive both manually by
// folding the io error into its string form.
impl Clone for StorageError {
    fn clone(&self) -> Self {
        match self {
            Self::Io(e) => Self::Io(std::io::Error::new(e.kind(), e.to_string())),
            Self::Postcard(e) => Self::Postcard(e.clone()),
            Self::BadVersion(v) => Self::BadVersion(*v),
            Self::BadMagic => Self::BadMagic,
            Self::TornFrame(o) => Self::TornFrame(*o),
            Self::Truncated => Self::Truncated,
            Self::Locked(p) => Self::Locked(p.clone()),
            Self::V4(m) => Self::V4(m.clone()),
        }
    }
}

impl PartialEq for StorageError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Io(a), Self::Io(b)) => a.kind() == b.kind() && a.to_string() == b.to_string(),
            (Self::Postcard(a), Self::Postcard(b)) => a == b,
            (Self::BadVersion(a), Self::BadVersion(b)) => a == b,
            (Self::BadMagic, Self::BadMagic) => true,
            (Self::TornFrame(a), Self::TornFrame(b)) => a == b,
            (Self::Truncated, Self::Truncated) => true,
            (Self::Locked(a), Self::Locked(b)) => a == b,
            (Self::V4(a), Self::V4(b)) => a == b,
            _ => false,
        }
    }
}

pub type Result<T> = std::result::Result<T, StorageError>;

/// An open persisted store: main file + WAL, both under `dir`.
///
/// `load` reads the main file (or an empty store if missing) and replays the
/// WAL. `append` logs one mutating statement; `checkpoint` compacts.
///
/// **Single-writer enforcement (issue #84):** opening takes an exclusive
/// cross-process lock on the sidecar `<name>.ndb.lock` and holds it until the
/// store is dropped. A second opener — another process, or another handle in
/// the same process — fails with [`StorageError::Locked`] instead of silently
/// racing the first writer to a checkpoint (which lost acknowledged writes).
#[derive(Debug)]
pub struct StoreFile {
    dir: PathBuf,
    /// Main file path `<dir>/<name>.ndb`.
    main: PathBuf,
    /// WAL path `<dir>/<name>.ndb.wal`.
    wal: PathBuf,
    /// Lock path `<dir>/<name>.ndb.lock`.
    _lock_path: PathBuf,
    /// Held lock file (RAII): the fd keeps the advisory lock alive (unix);
    /// underscore-prefixed because nothing on unix reads it back — the drop
    /// of this handle *is* the release. Non-unix Drop reads both fields.
    _lock: Option<File>,
    wal_len: u64,
    /// `(offset, len)` of the history tail in a version-3 main file — set at
    /// load, `take`n on the first temporal read (issue #133: lazy decode).
    hist_range: std::cell::Cell<Option<(u64, u64)>>,
    /// True when the main file was a format-v4 container — checkpoints
    /// must re-encode as v4, never downgrade (#157).
    is_v4: std::cell::Cell<bool>,
}

/// On-disk core frame (v3, issue #133): the store minus its history, plus the
/// explicit `tables` index (the legacy payload skips it via `serde(skip)`).
#[derive(serde::Serialize)]
struct StoreCoreRef<'a> {
    records: &'a BTreeMap<RecordId, Record>,
    edges: &'a Vec<RelationEdge>,
    vector_dims: &'a BTreeMap<String, usize>,
    clock: i64,
    memories: &'a BTreeMap<String, Store>,
    tables: &'a BTreeMap<String, Option<usize>>,
}

/// Owned core frame — what the v3 loader decodes.
#[derive(serde::Deserialize)]
struct StoreCore {
    records: BTreeMap<RecordId, Record>,
    edges: Vec<RelationEdge>,
    vector_dims: BTreeMap<String, usize>,
    clock: i64,
    memories: BTreeMap<String, Store>,
    tables: BTreeMap<String, Option<usize>>,
}

impl StoreCore {
    /// Build the in-memory store: history starts empty (its tail loads
    /// lazily); `tables` comes from the frame, nested memories get theirs
    /// rebuilt from their inline histories.
    fn into_store(self) -> Store {
        let mut store = Store {
            records: self.records,
            edges: self.edges,
            vector_dims: self.vector_dims,
            clock: self.clock,
            history: Vec::new(),
            memories: self.memories,
            tables: self.tables,
        };
        store.rebuild_tables();
        store
    }
}

impl StoreFile {
    /// Take the pending history range once (issue #133): `None` for legacy
    /// files and after the first temporal read has claimed it.
    pub(crate) fn take_history_range(&self) -> Option<(u64, u64)> {
        self.hist_range.take()
    }

    /// Decode the history tail at `(offset, len)` from the main file.
    pub(crate) fn read_history(&self, offset: u64, len: u64) -> Result<Vec<(i64, Statement)>> {
        use std::io::{Seek, SeekFrom};
        let mut f = File::open(&self.main)?;
        f.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; len as usize];
        f.read_exact(&mut buf)?;
        Ok(postcard::from_bytes(&buf)?)
    }
}

/// Take the exclusive single-writer lock for `lock_path`.
///
/// Unix: advisory `flock(LOCK_EX | LOCK_NB)` — the kernel releases it when
/// the fd closes for any reason (exit, crash, panic), so crashed writers
/// never leave stale locks behind; the lock *file* persists and is reused.
///
/// Non-unix: std file locking is below MSRV 1.82 (`File::try_lock` needs
/// 1.89), so fall back to a `create_new` lock file — exclusive but stale
/// after a crash (the error text tells the user how to clear it).
#[cfg(unix)]
fn acquire_lock(lock_path: &Path) -> Result<File> {
    use std::os::unix::io::AsRawFd;
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(lock_path)?;
    // SAFETY: `file` is a valid open fd for the duration of the call; flock(2)
    // only reads the fd and reports errno via the return value.
    let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if rc != 0 {
        let err = std::io::Error::last_os_error();
        return Err(match err.kind() {
            std::io::ErrorKind::WouldBlock => StorageError::Locked(lock_path.to_path_buf()),
            _ => StorageError::Io(err),
        });
    }
    Ok(file)
}

#[cfg(not(unix))]
fn acquire_lock(lock_path: &Path) -> Result<File> {
    match OpenOptions::new()
        .create_new(true)
        .write(true)
        .truncate(false)
        .open(lock_path)
    {
        Ok(file) => Ok(file),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            Err(StorageError::Locked(lock_path.to_path_buf()))
        }
        Err(e) => Err(StorageError::Io(e)),
    }
}

impl StoreFile {
    /// Open (or create) the store at `path` (e.g. `data.ndb`).
    /// Creates the parent directory if missing.
    ///
    /// Fails with [`StorageError::Locked`] when another handle currently owns
    /// the store (single-writer, issue #84).
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let main = path.clone();
        let wal = append_suffix(&path, WAL_SUFFIX);
        let lock_path = append_suffix(&path, LOCK_SUFFIX);
        if let Some(parent) = main.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let lock = acquire_lock(&lock_path)?;
        let wal_len = fs::metadata(&wal).map(|m| m.len()).unwrap_or(0);
        Ok(Self {
            dir: main.parent().unwrap_or(Path::new(".")).to_path_buf(),
            main,
            wal,
            _lock_path: lock_path,
            _lock: Some(lock),
            wal_len,
            hist_range: std::cell::Cell::new(None),
            is_v4: std::cell::Cell::new(false),
        })
    }

    pub fn wal_path(&self) -> &Path {
        &self.wal
    }

    /// Load the store: main file (or empty) + replay of the WAL.
    pub fn load(&self) -> Result<(Store, Vec<Statement>)> {
        let mut store = self.load_main()?;
        let replayed = self.replay_wal(&mut store)?;
        Ok((store, replayed))
    }

    fn load_main(&self) -> Result<Store> {
        let data = match fs::read(&self.main) {
            Ok(d) => d,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Store::default()),
            Err(e) => return Err(e.into()),
        };
        if data.len() < 16 {
            // Empty/corrupt main file: treat as empty (a fresh checkpoint will fix).
            return Ok(Store::default());
        }
        if &data[0..8] != MAGIC {
            return Err(StorageError::BadMagic);
        }
        let version = u32::from_le_bytes(data[8..12].try_into().unwrap());
        match version {
            // Legacy inline layout (pre-#133): one payload, history included.
            LEGACY_VERSION => {
                let mut store: Store = postcard::from_bytes(&data[16..])?;
                // `tables` is serde(skip)'ed: rebuild from the inline history.
                store.rebuild_tables();
                Ok(store)
            }
            FORMAT_VERSION => {
                // [core_len: u64 LE][core frame][history tail to EOF] (issue #133).
                if data.len() < 24 {
                    return Err(StorageError::Truncated);
                }
                let core_len = u64::from_le_bytes(data[16..24].try_into().unwrap()) as usize;
                let core_end = 24usize
                    .checked_add(core_len)
                    .filter(|&end| end <= data.len())
                    .ok_or(StorageError::Truncated)?;
                let core: StoreCore = postcard::from_bytes(&data[24..core_end])?;
                let store = core.into_store();
                // History stays undecoded (issue #133): remember where the
                // tail is; the first temporal read pays for it.
                self.hist_range
                    .set(Some((core_end as u64, (data.len() - core_end) as u64)));
                Ok(store)
            }
            V4_VERSION => {
                // Format v4 (spec §5): the container decodes the FULL store
                // including history (no lazy tail on this path — zig owns the
                // v4-native lazy decode; see #157). Checkpoints stay v4.
                let store = crate::v4::decode_store(&data).map_err(StorageError::V4)?;
                self.is_v4.set(true);
                Ok(store)
            }
            _ => Err(StorageError::BadVersion(version)),
        }
    }

    /// Replay every WAL frame into `store`, truncating at the first torn frame.
    fn replay_wal(&self, store: &mut Store) -> Result<Vec<Statement>> {
        let mut f = match File::open(&self.wal) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e.into()),
        };
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        let mut current_memory: Option<String> = None;

        let mut replayed = Vec::new();
        let mut pos = 0usize;
        let mut good_until = 0usize;
        while pos + 8 <= buf.len() {
            let crc = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap());
            let len = u32::from_le_bytes(buf[pos + 4..pos + 8].try_into().unwrap()) as usize;
            let start = pos + 8;
            if start + len > buf.len() {
                break; // torn frame: payload truncated
            }
            let payload = &buf[start..start + len];
            let mut h = crc32fast::Hasher::new();
            h.update(&(len as u32).to_le_bytes());
            h.update(payload);
            if h.finalize() != crc {
                break; // torn frame: checksum mismatch
            }
            match postcard::from_bytes::<Statement>(payload) {
                Ok(stmt) => {
                    // Replay is best-effort over logged statements: SELECT is
                    // never logged, and mutating statements are total (they
                    // cannot fail on a valid store), so any engine error here
                    // means a corrupt frame — treat it as torn. Memory
                    // statements carry the context switch so MEMORY scoping
                    // survives reopen.
                    let _ =
                        crate::engine::execute_in_context(store, &stmt, &mut current_memory, None);
                    replayed.push(stmt);
                    pos = start + len;
                    good_until = pos;
                }
                Err(_) => break, // torn frame: unserializable payload
            }
        }
        if good_until < buf.len() {
            // Truncate the torn tail so a later open doesn't retry it.
            let f = OpenOptions::new().write(true).open(&self.wal)?;
            f.set_len(good_until as u64)?;
        }
        Ok(replayed)
    }

    /// Append one statement to the WAL — the single-statement wrapper over
    /// [`StoreFile::append_batch`] (tests and one-off writers).
    pub fn append(&mut self, stmt: &Statement) -> Result<()> {
        self.append_batch(&[stmt])
    }

    /// Append statements to the WAL with **one write and one fsync** (issue
    /// #164). Every frame is byte-identical to a per-statement append (same
    /// crc32/len/payload layout) — only the syscall count changes (N+1 syncs
    /// → 1), which is exactly what `spec/file-format.md` §2 already
    /// specifies: "the file is `fsync`ed after each batch (one `execute`
    /// call = one transaction)".
    ///
    /// All payloads are serialized into a buffer **before** the file is
    /// touched, so a serialization failure leaves the WAL unchanged (the
    /// per-statement path could leave an earlier statement durable while
    /// `execute` returned an error). Durability granularity is the batch:
    /// `Database::execute` returns only after this fsync, and a crash mid-
    /// batch leaves a complete frame prefix — replay consumes complete
    /// frames only (torn-frame detection, spec §2).
    pub fn append_batch(&mut self, stmts: &[&Statement]) -> Result<()> {
        if stmts.is_empty() {
            return Ok(());
        }
        let mut buf = Vec::new();
        for stmt in stmts {
            let payload = postcard::to_allocvec(stmt)?;
            let mut h = crc32fast::Hasher::new();
            h.update(&(payload.len() as u32).to_le_bytes());
            h.update(&payload);
            let crc = h.finalize();
            buf.extend_from_slice(&crc.to_le_bytes());
            buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            buf.extend_from_slice(&payload);
        }
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.wal)?;
        f.write_all(&buf)?;
        f.sync_all()?;
        self.wal_len += buf.len() as u64;
        Ok(())
    }

    /// True when the WAL has grown past the checkpoint threshold.
    pub fn needs_checkpoint(&self) -> bool {
        self.wal_len >= CHECKPOINT_THRESHOLD
    }

    /// Atomically rewrite the main file from `store` and truncate the WAL.
    pub fn checkpoint(&mut self, store: &Store) -> Result<()> {
        let buf: Vec<u8> = if self.is_v4.get() {
            // A v4 store must STAY v4 (#157): rewriting it as a v3 core
            // frame would silently downgrade a migrated file. History is
            // eager on this path (decode_store), so it is complete here.
            crate::v4::encode_store(store).map_err(StorageError::V4)?
        } else {
            // v3 layout (issue #133): core frame (the store minus its history)
            // with an explicit length, then the history tail to EOF.
            let core = StoreCoreRef {
                records: &store.records,
                edges: &store.edges,
                vector_dims: &store.vector_dims,
                clock: store.clock,
                memories: &store.memories,
                tables: &store.tables,
            };
            let core_bytes = postcard::to_allocvec(&core)?;
            let hist_bytes = postcard::to_allocvec(&store.history)?;
            let mut buf = Vec::with_capacity(24 + core_bytes.len() + hist_bytes.len());
            buf.extend_from_slice(MAGIC);
            buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.extend_from_slice(&(core_bytes.len() as u64).to_le_bytes());
            buf.extend_from_slice(&core_bytes);
            buf.extend_from_slice(&hist_bytes);
            buf
        };

        // tmp + rename + fsync for atomic replacement.
        let tmp = self.main.with_extension("nql.tmp");
        {
            let mut f = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&tmp)?;
            f.write_all(&buf)?;
            f.sync_all()?;
        }
        fs::rename(&tmp, &self.main)?;
        // fsync the directory so the rename itself is durable.
        if let Ok(dir) = File::open(&self.dir) {
            let _ = dir.sync_all();
        }
        // Truncate the WAL now that the main file is authoritative.
        let f = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.wal)?;
        f.sync_all()?;
        self.wal_len = 0;
        Ok(())
    }
}

// Without advisory locking (non-unix) the lock *file* is the lock: close the
// fd and remove the file on graceful exit. A crash leaves the file behind and
// the next opener gets `StorageError::Locked` with clearing instructions.
// On unix no Drop impl is needed: the `File` field's drop closes the fd,
// which releases the flock (kernel-owned liveness — the file is reused as-is).
#[cfg(not(unix))]
impl Drop for StoreFile {
    fn drop(&mut self) {
        if let Some(file) = self._lock.take() {
            drop(file); // close first: Windows won't delete an open file
            let _ = fs::remove_file(&self._lock_path);
        }
    }
}

fn append_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(suffix);
    PathBuf::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nql_ir::{Id, Record, RecordId, Value};
    use std::collections::BTreeMap;

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("nqlite-storage-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn sample_store() -> Store {
        let mut s = Store::default();
        s.vector_dims.insert("t".into(), 2);
        s.insert(Record {
            id: RecordId {
                table: "t".into(),
                id: Id::Num(1),
            },
            body: BTreeMap::from([("name".into(), Value::Str("alpha".into()))]),
            embedding: Some(vec![1.0, 0.0]),
            created_at: 0,
        });
        s
    }

    #[test]
    fn roundtrip_main_file() {
        let dir = temp_dir("roundtrip");
        let path = dir.join("db.ndb");
        let mut sf = StoreFile::open(&path).unwrap();
        sf.checkpoint(&sample_store()).unwrap();
        drop(sf); // single-writer: release the store lock before reopening

        let sf2 = StoreFile::open(&path).unwrap();
        let (store, replayed) = sf2.load().unwrap();
        assert_eq!(store, sample_store(), "store survives checkpoint roundtrip");
        assert!(replayed.is_empty(), "checkpointed store has no WAL replay");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn wal_replay_applies_mutations() {
        let dir = temp_dir("wal");
        let path = dir.join("db.ndb");
        let mut sf = StoreFile::open(&path).unwrap();
        let mut store = Store::default();
        let create = Statement::CreateTable {
            table: "t".into(),
            vector_dim: Some(2),
        };
        let rec = Record {
            id: RecordId {
                table: "t".into(),
                id: Id::Num(1),
            },
            body: BTreeMap::new(),
            embedding: Some(vec![0.5, 0.5]),
            created_at: 0,
        };
        let insert = Statement::Insert(rec);
        crate::engine::execute_statement(&mut store, &create, None).unwrap();
        crate::engine::execute_statement(&mut store, &insert, None).unwrap();
        sf.append(&create).unwrap();
        sf.append(&insert).unwrap();
        drop(sf); // single-writer: release the store lock before reopening

        // Reopen from disk: WAL must reconstruct the store.
        let sf2 = StoreFile::open(&path).unwrap();
        let (loaded, replayed) = sf2.load().unwrap();
        assert_eq!(loaded, store, "WAL replay reconstructs the store");
        assert_eq!(replayed.len(), 2, "both statements replayed");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn append_batch_writes_byte_identical_wal() {
        // Issue #164: a per-plan batch must produce the exact bytes
        // per-statement appends would (frame layout unchanged) — only the
        // syscall count differs. Both WALs must replay to the same store.
        let build = |s: &mut Store| -> Vec<Statement> {
            let create = Statement::CreateTable {
                table: "t".into(),
                vector_dim: Some(2),
            };
            let insert = Statement::Insert(Record {
                id: RecordId {
                    table: "t".into(),
                    id: Id::Num(1),
                },
                body: BTreeMap::from([("name".into(), Value::Str("alpha".into()))]),
                embedding: Some(vec![0.5, 0.5]),
                created_at: 0,
            });
            crate::engine::execute_statement(s, &create, None).unwrap();
            crate::engine::execute_statement(s, &insert, None).unwrap();
            vec![create, insert]
        };

        // (a) per-statement appends
        let dir_a = temp_dir("batch-a");
        let path_a = dir_a.join("db.ndb");
        let mut store_a = Store::default();
        let stmts_a = build(&mut store_a);
        let mut sf_a = StoreFile::open(&path_a).unwrap();
        for s in &stmts_a {
            sf_a.append(s).unwrap();
        }
        let (wal_a, len_a) = (sf_a.wal.clone(), sf_a.wal_len);
        drop(sf_a);

        // (b) one batch
        let dir_b = temp_dir("batch-b");
        let path_b = dir_b.join("db.ndb");
        let mut store_b = Store::default();
        let stmts_b = build(&mut store_b);
        let mut sf_b = StoreFile::open(&path_b).unwrap();
        let refs: Vec<&Statement> = stmts_b.iter().collect();
        sf_b.append_batch(&refs).unwrap();
        let (wal_b, len_b) = (sf_b.wal.clone(), sf_b.wal_len);
        drop(sf_b);

        let bytes_a = fs::read(&wal_a).unwrap();
        let bytes_b = fs::read(&wal_b).unwrap();
        assert_eq!(
            bytes_a, bytes_b,
            "batch WAL = per-statement WAL, byte for byte"
        );
        assert_eq!(len_a, len_b, "wal_len accounting identical");

        let (load_a, rep_a) = StoreFile::open(&path_a).unwrap().load().unwrap();
        let (load_b, rep_b) = StoreFile::open(&path_b).unwrap().load().unwrap();
        assert_eq!(load_a, load_b, "both WALs replay to the same store");
        assert_eq!(rep_a.len(), 2, "per-statement replay: both statements");
        assert_eq!(rep_b.len(), 2, "batch replay: both statements");
        fs::remove_dir_all(&dir_a).ok();
        fs::remove_dir_all(&dir_b).ok();
    }

    #[test]
    fn torn_frame_is_truncated() {
        let dir = temp_dir("torn");
        let path = dir.join("db.ndb");
        let mut sf = StoreFile::open(&path).unwrap();
        let create = Statement::CreateTable {
            table: "t".into(),
            vector_dim: Some(2),
        };
        sf.append(&create).unwrap();

        // Append a garbage torn frame (bad CRC).
        {
            let mut f = OpenOptions::new().append(true).open(sf.wal_path()).unwrap();
            f.write_all(&[0xDE, 0xAD, 0xBE, 0xEF, 0x10, 0x00, 0x00, 0x00])
                .unwrap();
            f.write_all(&[0x01, 0x02]).unwrap();
        }
        drop(sf); // single-writer: release the store lock before reopening
        let sf2 = StoreFile::open(&path).unwrap();
        let (store, replayed) = sf2.load().unwrap();
        assert_eq!(replayed.len(), 1, "only the good frame is replayed");
        assert!(store.vector_dims.contains_key("t"));
        // WAL should now be truncated to the good frame only.
        assert_eq!(
            fs::metadata(sf2.wal_path()).unwrap().len(),
            8 + sf2_wal_frame_len()
        );
        fs::remove_dir_all(&dir).ok();
    }

    fn sf2_wal_frame_len() -> u64 {
        // create table "t" with dim Some(2) — must match the frame appended in
        // torn_frame_is_truncated; computed via the same serialization.
        let stmt = Statement::CreateTable {
            table: "t".into(),
            vector_dim: Some(2),
        };
        postcard::to_allocvec(&stmt).unwrap().len() as u64
    }

    #[test]
    fn deterministic_bytes() {
        let dir = temp_dir("det");
        let path = dir.join("db.ndb");
        let mut sf = StoreFile::open(&path).unwrap();
        sf.checkpoint(&sample_store()).unwrap();
        let a = fs::read(&path).unwrap();
        drop(sf); // single-writer: release the store lock before reopening
        let mut sf2 = StoreFile::open(&path).unwrap();
        sf2.checkpoint(&sample_store()).unwrap();
        let b = fs::read(&path).unwrap();
        assert_eq!(a, b, "identical stores serialize to identical bytes");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn second_open_is_rejected_while_held() {
        // Issue #84: one live handle per store path. A second open (another
        // process, or another handle in this one) must fail loudly instead of
        // racing the first writer to a checkpoint.
        let dir = temp_dir("lock");
        let path = dir.join("db.ndb");
        let sf = StoreFile::open(&path).unwrap();
        let err = StoreFile::open(&path).unwrap_err();
        assert!(
            matches!(err, StorageError::Locked(_)),
            "second open must be Locked, got: {err:?}"
        );
        // Releasing the handle makes the store reopenable (flock dropped;
        // the lock file itself is reusable).
        drop(sf);
        let sf2 = StoreFile::open(&path).unwrap();
        drop(sf2);
        fs::remove_dir_all(&dir).ok();
    }
}
