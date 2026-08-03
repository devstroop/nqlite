//! End-to-end persistence: nql text -> execute on a file-backed Database ->
//! drop -> reopen -> data + WAL replay survive.
//!
//! Deterministic and zero-LLM, like the rest of nqlite.

use nql::parse;
use nqlite::Database;

#[test]
fn file_backed_database_persists_across_reopen() {
    let dir = std::env::temp_dir().join(format!("nqlite-persist-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("db.nql");

    // Session 1: create + insert + relate, then drop (simulates process exit
    // without explicit flush — durability comes from the WAL).
    {
        let mut db = Database::open(&path).unwrap();
        db.execute(
            &parse(
                "CREATE TABLE t VECTOR<f32, 2>;
             INSERT INTO t:1 { \"name\": \"alpha\" } EMBED [1.0, 0.0];
             INSERT INTO t:2 { \"name\": \"beta\" } EMBED [0.0, 1.0];
             RELATE (t:1) -> :refs -> (t:2) SET weight = 0.5;",
            )
            .unwrap(),
        )
        .unwrap();
    }

    // Session 2: reopen — WAL replay must reconstruct everything.
    {
        let mut db = Database::open(&path).unwrap();
        assert_eq!(db.store().records.len(), 2, "both records survive reopen");
        assert_eq!(db.store().edges.len(), 1, "edge survives reopen");
        let results = db
            .execute(&parse(
                "SELECT * FROM t WHERE vector::similarity(embedding, [1.0, 0.0]) AND k = 2 ORDER BY ::similarity;",
            ).unwrap())
            .unwrap();
        assert_eq!(results[0].rows.len(), 2);
        assert_eq!(
            results[0].rows[0].record.id.to_string(),
            "t:1",
            "nearest first"
        );
        assert!(results[0].rows[0].score >= results[0].rows[1].score);
    }

    // Session 3: checkpoint (flush) then reopen — main file is authoritative.
    {
        let mut db = Database::open(&path).unwrap();
        db.flush().unwrap();
    }
    {
        let db = Database::open(&path).unwrap();
        assert_eq!(
            db.store().records.len(),
            2,
            "records survive checkpoint+reopen"
        );
        assert_eq!(db.store().edges.len(), 1);
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn deterministic_reopen_bytes() {
    let dir = std::env::temp_dir().join(format!("nqlite-persist-det-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let session = || {
        let path = dir.join(format!("db-{}.nql", std::process::id()));
        let mut db = Database::open(&path).unwrap();
        db.execute(
            &parse(
                "CREATE TABLE t VECTOR<f32, 2>;
             INSERT INTO t:1 { \"x\": 1 } EMBED [0.5, 0.5];
             SELECT * FROM t;",
            )
            .unwrap(),
        )
        .unwrap();
        db.flush().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{}.wal", path.display()));
        bytes
    };

    let a = session();
    let b = session();
    assert_eq!(a, b, "same session writes byte-identical files");
    let _ = std::fs::remove_dir_all(&dir);
}
