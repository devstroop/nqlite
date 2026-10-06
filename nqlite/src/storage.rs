//! Single-file persistence with a sidecar write-ahead log (M1).
//!
//! Byte-deterministic format (see `spec/file-format.md`):
//! - main file: magic + version + `postcard(Store)`
//! - WAL: append-only frames of `crc32(len || payload)`, len, postcard(Statement)
//!
//! Crash-safety: the main file is replaced atomically (tmp + rename + fsync);
//! the WAL is replayed on open and truncated at the first torn frame (CRC
//! mismatch / bad length). Acknowledged transactions survive crashes; a crash
//! mid-commit can only drop the in-flight transaction, never corrupt.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use nql_ir::{Statement, Store};
use thiserror::Error;

/// Checkpoint the WAL into the main file once it exceeds this size.
pub const CHECKPOINT_THRESHOLD: u64 = 1 << 20; // 1 MiB

const MAGIC: &[u8; 8] = b"NQLITE01";
const FORMAT_VERSION: u32 = 2;
const WAL_SUFFIX: &str = ".wal";
const LOCK_SUFFIX: &str = ".lock";

/// Storage errors (all recoverable by re-opening / re-checkpointing).
#[derive(Debug, Error)]
pub enum StorageError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Postcard(#[from] postcard::Error),
    #[error("unsupported format version {0} (expected {FORMAT_VERSION})")]
    BadVersion(u32),
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
            Self::Locked(p) => Self::Locked(p.clone()),
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
            (Self::Locked(a), Self::Locked(b)) => a == b,
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
/// cross-process lock on the sidecar `<name>.nql.lock` and holds it until the
/// store is dropped. A second opener — another process, or another handle in
/// the same process — fails with [`StorageError::Locked`] instead of silently
/// racing the first writer to a checkpoint (which lost acknowledged writes).
#[derive(Debug)]
pub struct StoreFile {
    dir: PathBuf,
    /// Main file path `<dir>/<name>.nql`.
    main: PathBuf,
    /// WAL path `<dir>/<name>.nql.wal`.
    wal: PathBuf,
    /// Lock path `<dir>/<name>.nql.lock`.
    _lock_path: PathBuf,
    /// Held lock file (RAII): the fd keeps the advisory lock alive (unix);
    /// underscore-prefixed because nothing on unix reads it back — the drop
    /// of this handle *is* the release. Non-unix Drop reads both fields.
    _lock: Option<File>,
    wal_len: u64,
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
    /// Open (or create) the store at `path` (e.g. `data.nql`).
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
        if version != FORMAT_VERSION {
            return Err(StorageError::BadVersion(version));
        }
        let store = postcard::from_bytes(&data[16..])?;
        Ok(store)
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
                    let _ = crate::engine::execute_in_context(store, &stmt, &mut current_memory);
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

    /// Append one mutating statement to the WAL and fsync.
    pub fn append(&mut self, stmt: &Statement) -> Result<()> {
        let payload = postcard::to_allocvec(stmt)?;
        let mut h = crc32fast::Hasher::new();
        h.update(&(payload.len() as u32).to_le_bytes());
        h.update(&payload);
        let crc = h.finalize();

        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.wal)?;
        f.write_all(&crc.to_le_bytes())?;
        f.write_all(&(payload.len() as u32).to_le_bytes())?;
        f.write_all(&payload)?;
        f.sync_all()?;
        self.wal_len += 8 + payload.len() as u64;
        Ok(())
    }

    /// True when the WAL has grown past the checkpoint threshold.
    pub fn needs_checkpoint(&self) -> bool {
        self.wal_len >= CHECKPOINT_THRESHOLD
    }

    /// Atomically rewrite the main file from `store` and truncate the WAL.
    pub fn checkpoint(&mut self, store: &Store) -> Result<()> {
        let mut buf = Vec::with_capacity(16 + 1024);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        let body = postcard::to_allocvec(store)?;
        buf.extend_from_slice(&body);

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
        let path = dir.join("db.nql");
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
        let path = dir.join("db.nql");
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
        crate::engine::execute_statement(&mut store, &create).unwrap();
        crate::engine::execute_statement(&mut store, &insert).unwrap();
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
    fn torn_frame_is_truncated() {
        let dir = temp_dir("torn");
        let path = dir.join("db.nql");
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
        let path = dir.join("db.nql");
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
        let path = dir.join("db.nql");
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
