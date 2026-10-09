//! WAL fsync-count probe (issue #164).
//!
//! Builds one plan of N inserts against a persistent database, executes it,
//! and checkpoints. Run under `strace` to count sync syscalls:
//!
//! ```sh
//! cargo build --release -p nqlite --example wal_fsync_probe
//! strace -c -e trace=fsync,fdatasync \
//!     target/release/examples/wal_fsync_probe /tmp/nql-wal-probe 1000
//! ```
//!
//! Expected: before #164 ≈ N + 1 (per-statement appends + `ContextReset`)
//! plus checkpoint syncs; after ≈ 1 (single per-plan batch) plus checkpoint
//! syncs — `spec/file-format.md` §2: "the file is fsynced after each batch
//! (one `execute` call = one transaction)".

use std::collections::BTreeMap;
use std::path::PathBuf;

use nqlite::{Database, Id, Record, RecordId, Statement, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().unwrap_or_else(|| "/tmp/nql-wal-probe".into()));
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1000);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("probe.ndb");

    let mut db = Database::open(&path)?;
    let mut plan = Vec::with_capacity(n + 1);
    plan.push(Statement::CreateTable {
        table: "t".into(),
        vector_dim: None,
    });
    for i in 0..n {
        plan.push(Statement::Insert(Record {
            id: RecordId::new("t", Id::Num(i as u64)),
            body: BTreeMap::from([("v".into(), Value::Int(i as i64))]),
            embedding: None,
            created_at: 0,
        }));
    }
    db.execute(&plan)?;
    db.flush()?;
    println!("probe ok: n={n} rows={}", db.store().records.len());
    Ok(())
}
