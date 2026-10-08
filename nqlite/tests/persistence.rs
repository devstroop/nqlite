//! End-to-end persistence: nql text -> execute on a file-backed Database ->
//! drop -> reopen -> data + WAL replay survive.
//!
//! Deterministic and zero-LLM, like the rest of nqlite.

use nql::parse;
use nqlite::{Database, Error};

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
        // The tables index is rebuilt at load (issue #133 step 1) — the
        // declared-with-dim table is known without any history scan here.
        assert_eq!(db.store().tables.get("t"), Some(&Some(2)));
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

#[test]
fn read_only_match_is_not_written_to_wal() {
    // Regression: MATCH is a read (like SELECT) — it must never be appended
    // to the WAL or replayed, only mutating statements (CREATE/INSERT/RELATE/
    // FORGET) belong there (see spec/file-format.md §2).
    let dir = std::env::temp_dir().join(format!("nqlite-persist-match-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("db.nql");

    let wal_len = |path: &std::path::Path| -> u64 {
        std::fs::metadata(format!("{}.wal", path.display()))
            .map(|m| m.len())
            .unwrap_or(0)
    };

    {
        let mut db = Database::open(&path).unwrap();
        db.execute(
            &parse(
                "CREATE TABLE t;
                 INSERT INTO t:1 { \"x\": 1 };
                 INSERT INTO t:2 { \"x\": 2 };
                 RELATE (t:1) -> :refs -> (t:2);",
            )
            .unwrap(),
        )
        .unwrap();
        let after_dml = wal_len(&path);
        assert!(after_dml > 0, "mutations are logged");

        // A read-only MATCH must not grow the WAL.
        db.execute(&parse("MATCH (t:1) -> :refs;").unwrap())
            .unwrap();
        assert_eq!(
            wal_len(&path),
            after_dml,
            "MATCH must not be appended to the WAL"
        );

        // Nor a SELECT.
        db.execute(&parse("SELECT * FROM t;").unwrap()).unwrap();
        assert_eq!(
            wal_len(&path),
            after_dml,
            "SELECT must not be appended to the WAL"
        );
    }

    // Reopen: replay works and the store is intact.
    {
        let db = Database::open(&path).unwrap();
        assert_eq!(db.store().records.len(), 2);
        assert_eq!(db.store().edges.len(), 1);
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn as_of_history_survives_checkpoint_and_reopen() {
    let dir = std::env::temp_dir().join(format!("nqlite-temporal-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("db.nql");

    // Session 1: create + insert + forget, then checkpoint + drop.
    {
        let mut db = Database::open(&path).unwrap();
        db.execute(
            &parse(
                "CREATE TABLE t;
                 INSERT INTO t:1 { \"name\": \"alpha\" };
                 FORGET t:1;",
            )
            .unwrap(),
        )
        .unwrap();
        db.flush().unwrap();
    }

    // Session 2: reopen — the mutation history (and thus AS OF) survives the
    // checkpoint: the forgotten record is visible as of before the FORGET.
    {
        let mut db = Database::open(&path).unwrap();
        let now = db.execute(&parse("SELECT * FROM t;").unwrap()).unwrap();
        assert_eq!(now[0].rows.len(), 0, "t:1 was forgotten");

        let past = db
            .execute(&parse("SELECT * FROM t AS OF 2;").unwrap())
            .unwrap();
        assert_eq!(past[0].rows.len(), 1, "AS OF sees the pre-forget state");
        assert_eq!(past[0].rows[0].record.id.to_string(), "t:1");
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn memory_context_resets_across_reopen() {
    // Regression for issue #109: a plan that ENDS inside a MEMORY block must
    // not leak its context into later plans' WAL frames. Runtime: every plan
    // starts at root (spec §2.8); the flat WAL needs an explicit boundary
    // marker (`Statement::ContextReset`) for replay to agree with the runtime.
    let dir = std::env::temp_dir().join(format!("nqlite-wal-ctx-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("db.nql");

    // Session 1: root write, then a plan that ENDS inside memory m.
    {
        let mut db = Database::open(&path).unwrap();
        db.execute(
            &parse(
                "CREATE TABLE t;
                 INSERT INTO t:a { \"text\": \"root\" };
                 MEMORY m; INSERT INTO t:mem { \"text\": \"in-m\" };",
            )
            .unwrap(),
        )
        .unwrap();
    }

    // Session 2: fresh process — this plan runs at ROOT (context resets per
    // plan). Its frames must not replay under m's context.
    {
        let mut db = Database::open(&path).unwrap();
        db.execute(&parse("INSERT INTO t:b { \"text\": \"root2\" };").unwrap())
            .unwrap();
    }

    // Session 3: replay must reconstruct both scopes exactly.
    {
        let mut db = Database::open(&path).unwrap();
        let root = db.execute(&parse("SELECT * FROM t;").unwrap()).unwrap();
        let mut ids: Vec<String> = root[0]
            .rows
            .iter()
            .map(|r| r.record.id.to_string())
            .collect();
        ids.sort();
        assert_eq!(
            ids,
            ["t:a", "t:b"],
            "root write after a memory-ending plan must replay at root (issue #109)"
        );
        let m = db
            .execute(&parse("MEMORY m; SELECT * FROM t;").unwrap())
            .unwrap();
        let mids: Vec<String> = m[0].rows.iter().map(|r| r.record.id.to_string()).collect();
        assert_eq!(
            mids,
            ["t:mem"],
            "memory store must not gain root rows, got {mids:?}"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn memory_blocks_survive_wal_replay_across_reopen() {
    let dir = std::env::temp_dir().join(format!("nqlite-memory-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("db.nql");

    // Session 1: write into the root and into a named memory, drop without
    // flushing (durability comes from the WAL, which logs MEMORY frames so
    // the context switch replays).
    {
        let mut db = Database::open(&path).unwrap();
        db.execute(
            &parse(
                "CREATE TABLE t;
                 INSERT INTO t:1 { \"name\": \"root\" };
                 MEMORY core;
                 CREATE TABLE t;
                 INSERT INTO t:1 { \"name\": \"core\" };",
            )
            .unwrap(),
        )
        .unwrap();
    }

    // Session 2: reopen — WAL replay must reconstruct memory scoping: the
    // core memory exists with its own record, the root is unaffected.
    {
        let db = Database::open(&path).unwrap();
        assert_eq!(db.store().records.len(), 1, "root record survives");
        assert_eq!(db.store().memories.len(), 1, "core memory survives");

        let mut db = db;
        let core = db
            .execute(
                &parse(
                    "MEMORY core;
                     SELECT * FROM t;",
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(core[0].rows.len(), 1, "core memory record survives");
        assert_eq!(core[0].rows[0].record.id.to_string(), "t:1");
        assert_eq!(
            core[0].rows[0].record.body.get("name"),
            Some(&nql_ir::Value::Str("core".into()))
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn prune_history_compaction_survives_reopen() {
    // Issue #95: PRUNE is WAL-logged, so compaction survives a reopen even
    // without an explicit flush — the pruned window fails loudly, the rest
    // reconstructs, and declaration-only tables keep working (issue #89).
    let dir = std::env::temp_dir().join(format!("nqlite-prune-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("db.nql");

    {
        let mut db = Database::open(&path).unwrap();
        db.execute(
            &parse(
                "CREATE TABLE empty_t;\
                 CREATE TABLE t;\
                 INSERT INTO t:1 { \"name\": \"alpha\" };\
                 INSERT INTO t:2 { \"name\": \"beta\" };",
            )
            .unwrap(),
        )
        .unwrap();
        db.execute(&parse("PRUNE HISTORY").unwrap()).unwrap();
        assert!(
            db.store()
                .history
                .iter()
                .any(|(_, s)| matches!(s, nql_ir::Statement::Snapshot(_))),
            "compacted live"
        );
    }

    {
        let mut db = Database::open(&path).unwrap();
        let snap_ts = db
            .store()
            .history
            .iter()
            .find_map(|(ts, s)| matches!(s, nql_ir::Statement::Snapshot(_)).then_some(*ts))
            .expect("snapshot survives reopen (WAL frame or checkpoint)");
        assert_eq!(
            db.store().records.len(),
            2,
            "data intact after pruned reopen"
        );

        // Declarations retained → an empty table still inserts after reopen.
        db.execute(&parse(r#"INSERT INTO empty_t:x { "v": 1 };"#).unwrap())
            .unwrap();

        // The pruned window fails loudly …
        let err = db.execute(&parse("SELECT * FROM t AS OF 1;").unwrap());
        assert!(
            matches!(err, Err(Error::HistoryPruned { .. })),
            "pre-snapshot AS OF errors, got {err:?}"
        );
        // … and from the snapshot onward temporal reads work, including for
        // mutations made after the reopen.
        let at = db
            .execute(&parse(&format!("SELECT * FROM t AS OF {snap_ts};")).unwrap())
            .unwrap();
        assert_eq!(at[0].rows.len(), 2, "snapshot view reconstructs");
        db.execute(&parse(r#"INSERT INTO t:3 { "name": "gamma" };"#).unwrap())
            .unwrap();
        let now_ts = db.store().clock;
        let later = db
            .execute(&parse(&format!("SELECT * FROM t AS OF {now_ts};")).unwrap())
            .unwrap();
        assert_eq!(later[0].rows.len(), 3, "post-snapshot delta replays");
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn legacy_v2_inline_layout_still_loads() {
    // D4 matrix (issue #133): the v3 reader must keep accepting the legacy
    // v2 layout (single inline postcard(Store) payload) — hand-crafted here
    // so the bytes are the PRE-#133 shape, independent of the current writer.
    let dir = std::env::temp_dir().join(format!("nqlite-legacy-v2-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let path = dir.join("db.nql");

    let mut db = Database::default();
    db.execute(
        &parse(
            "CREATE TABLE t VECTOR<f32, 2>;
             INSERT INTO t:1 { \"name\": \"alpha\" } EMBED [1.0, 0.0];
             INSERT INTO t:2 { \"name\": \"beta\" } EMBED [0.0, 1.0];",
        )
        .unwrap(),
    )
    .unwrap();
    let store = db.into_store();

    let mut bytes = b"NQLITE01".to_vec(); // MAGIC (storage.rs)
    bytes.extend_from_slice(&2u32.to_le_bytes()); // LEGACY_VERSION
    bytes.extend_from_slice(&0u32.to_le_bytes()); // reserved
    bytes.extend_from_slice(&postcard::to_allocvec(&store).unwrap());
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(&path, &bytes).unwrap();

    {
        let mut db = Database::open(&path).expect("legacy v2 file must load");
        assert_eq!(
            db.store().records.len(),
            2,
            "records decoded from inline payload"
        );
        // tables rebuilt from the inline history (step 1 contract).
        assert_eq!(db.store().tables.get("t"), Some(&Some(2)));
        // temporal reads work against the inline history (no lazy tail).
        let at = db
            .execute(&parse("SELECT * FROM t AS OF 2;").unwrap())
            .unwrap();
        assert_eq!(at[0].rows.len(), 1, "inline history replays");
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn unsupported_versions_and_truncated_v3_rejected() {
    // D4: loud rejections, never a partial load (issue #133).
    let dir = std::env::temp_dir().join(format!("nqlite-badver-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    for version in [1u32, 99u32] {
        let path = dir.join(format!("v{version}.nql"));
        let mut bytes = b"NQLITE01".to_vec();
        bytes.extend_from_slice(&version.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 8]);
        std::fs::write(&path, &bytes).unwrap();
        match Database::open(&path) {
            Err(nqlite::StorageError::BadVersion(v)) => assert_eq!(v, version),
            other => panic!("version {version}: expected BadVersion, got {other:?}"),
        }
    }

    // v3 header whose core length points past EOF → Truncated.
    let path = dir.join("trunc.nql");
    let mut bytes = b"NQLITE01".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&9999u64.to_le_bytes());
    bytes.extend_from_slice(b"junk");
    std::fs::write(&path, &bytes).unwrap();
    assert!(matches!(
        Database::open(&path),
        Err(nqlite::StorageError::Truncated)
    ));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn first_temporal_read_decodes_the_history_tail() {
    // The lazy seam (issue #133): a flushed v3 file (history tail) plus a
    // WAL-only mutation — the FIRST statement being temporal must claim the
    // tail and prepend it so file entries and WAL entries both answer, in
    // timestamp order. Without ensure_history, AS OF 2 would silently miss
    // the file's entries.
    let dir = std::env::temp_dir().join(format!("nqlite-lazyhist-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let path = dir.join("db.nql");

    {
        let mut db = Database::open(&path).unwrap();
        db.execute(
            &parse(
                "CREATE TABLE t;
                 INSERT INTO t:1 { \"n\": 1 };
                 INSERT INTO t:2 { \"n\": 2 };",
            )
            .unwrap(),
        )
        .unwrap();
        db.flush().unwrap(); // v3 file: core + history tail through ts3
        db.execute(&parse("INSERT INTO t:3 { \"n\": 3 };").unwrap())
            .unwrap();
    } // drop: WAL survives (no checkpoint on drop)

    {
        let mut db = Database::open(&path).unwrap();
        let at2 = db
            .execute(&parse("SELECT * FROM t AS OF 2;").unwrap())
            .unwrap();
        assert_eq!(at2[0].rows.len(), 1, "AS OF 2 needs the file history tail");
        let at4 = db
            .execute(&parse("SELECT * FROM t AS OF 4;").unwrap())
            .unwrap();
        assert_eq!(
            at4[0].rows.len(),
            3,
            "file tail + WAL entries, ascending ts"
        );
        let now = db.execute(&parse("SELECT * FROM t;").unwrap()).unwrap();
        assert_eq!(now[0].rows.len(), 3, "current state after the claim");
        // Once claimed, later temporal reads reuse the decoded history (the
        // range is taken exactly once) — another AS OF must not re-read.
        let again = db
            .execute(&parse("SELECT * FROM t AS OF 3;").unwrap())
            .unwrap();
        assert_eq!(again[0].rows.len(), 2);
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn flush_on_lazy_reopen_preserves_history() {
    // Regression (found via the E08 100k importer proof): `flush` and the
    // threshold checkpoint re-serialize `store.history` into the main file —
    // on a lazily reopened store (issue #133 tail pending) that used to
    // REWRITE the file with an EMPTY history, silently killing every
    // temporal read. Checkpoint must claim the tail first (`ensure_history`).
    let dir = std::env::temp_dir().join(format!("nqlite-flush-lazy-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("db.nql");

    // Session 1 (fresh → eager): history is in-memory, flush writes it.
    {
        let mut db = Database::open(&path).unwrap();
        db.execute(
            &parse(
                "CREATE TABLE t;
                 INSERT INTO t:1 { \"a\": 1 };
                 INSERT INTO t:2 { \"a\": 2 };",
            )
            .unwrap(),
        )
        .unwrap();
        db.flush().unwrap();
    }

    // Session 2 (reopen → the file era is a PENDING tail): a non-temporal
    // plan never claims it; the flush must do so instead of dropping it.
    {
        let mut db = Database::open(&path).unwrap();
        db.execute(&parse("INSERT INTO t:3 { \"a\": 3 };").unwrap())
            .unwrap();
        db.flush().unwrap();
    }

    // Session 3: the full log (4 entries) and temporal answers survive.
    {
        let mut db = Database::open(&path).unwrap();
        let past = db
            .execute(&parse("SELECT * FROM t AS OF 2;").unwrap())
            .unwrap();
        assert_eq!(
            past[0].rows.len(),
            1,
            "AS OF 2 sees create+first insert — file era was preserved"
        );
        assert_eq!(
            db.store().history.len(),
            4,
            "create@1 + inserts@2..4 all present after the lazy flush"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn threshold_checkpoint_on_lazy_reopen_preserves_history() {
    // The OTHER checkpoint path: `Database::execute`'s threshold fire must
    // be decided AFTER this plan's appends (a pre-append check never fires
    // for a single-plan session — exactly the E08 100k chunked-ingest
    // shape, which rewrote history with the current session's entries only).
    let dir = std::env::temp_dir().join(format!("nqlite-threshold-lazy-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("db.nql");

    // Session 1 (fresh → eager): a small history, checkpointed into main.
    {
        let mut db = Database::open(&path).unwrap();
        db.execute(&parse("CREATE TABLE t;\nINSERT INTO t:1 { \"a\": 1 };").unwrap())
            .unwrap();
        db.flush().unwrap();
    }

    // Session 2 (lazy tail): ONE plan that crosses the 1 MiB WAL threshold
    // (fat bodies: 4000 × ~330B) — the threshold fires inside this execute.
    {
        let mut db = Database::open(&path).unwrap();
        let mut prog = String::new();
        for i in 0..4000 {
            let pad = "x".repeat(300);
            prog.push_str(&format!("INSERT INTO t:{i} {{ \"pad\": \"{pad}\" }};\n"));
        }
        db.execute(&parse(&prog).unwrap()).unwrap();
    }

    // Session 3: the file-era entries were claimed before the checkpoint.
    {
        let mut db = Database::open(&path).unwrap();
        let past = db
            .execute(&parse("SELECT * FROM t AS OF 2;").unwrap())
            .unwrap();
        assert_eq!(
            past[0].rows.len(),
            1,
            "file-era view survives the threshold checkpoint"
        );
        assert_eq!(
            db.store().history.len(),
            4002,
            "2 file-era entries + 4000 from the threshold session"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn engine_reads_v4_and_checkpoint_preserves_it() {
    // #157: the engine opens format v4 (nql-migrate output), answers
    // queries — including temporal reads (history decodes eagerly on
    // this path) — and checkpoints must re-encode as v4, never
    // silently downgrading a migrated file to a v3 core frame.
    let dir = std::env::temp_dir().join(format!("nqlite-v4-open-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("store.nql");

    // A real v4 file: build the store, encode with the promoted codec.
    {
        let mut src = Database::new(nqlite::Store::default());
        src.execute(
            &parse(
                "CREATE TABLE t;
                 INSERT INTO t:1 { \"a\": 1 };
                 INSERT INTO t:2 { \"a\": 2 };
                 FORGET t:2;",
            )
            .unwrap(),
        )
        .unwrap();
        let bytes = nqlite::v4::encode_store(src.store()).unwrap();
        std::fs::write(&path, &bytes).unwrap();
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 4);
    }

    // Open: records + EAGER history + temporal answers + flush stays v4.
    {
        let mut db = Database::open(&path).unwrap();
        assert_eq!(db.store().records.len(), 1, "t:1 survives the FORGET");
        assert_eq!(
            db.store().history.len(),
            4,
            "v4 decodes history eagerly (create@1 + inserts@2,3 + forget@4)"
        );
        let past = db
            .execute(&parse("SELECT * FROM t AS OF 3;").unwrap())
            .unwrap();
        assert_eq!(past[0].rows.len(), 2, "AS OF sees both rows pre-FORGET");

        db.flush().unwrap();
        let raw = std::fs::read(&path).unwrap();
        assert_eq!(
            u32::from_le_bytes(raw[8..12].try_into().unwrap()),
            4,
            "checkpoint must not downgrade v4 to v3"
        );
        let back = nqlite::v4::decode_store(&raw).unwrap();
        assert_eq!(back.records.len(), 1, "records intact after flush");
        assert_eq!(back.history.len(), 4, "history complete after flush");

        // WAL append onto the v4 store, then reopen (replay path).
        db.execute(&parse("INSERT INTO t:3 { \"a\": 3 };").unwrap())
            .unwrap();
    }
    {
        let mut db = Database::open(&path).unwrap();
        assert_eq!(db.store().records.len(), 2, "WAL replay onto v4 main");
        let now = db.execute(&parse("SELECT * FROM t;").unwrap()).unwrap();
        assert_eq!(now[0].rows.len(), 2, "t:1 and t:3 live");
    }

    let _ = std::fs::remove_dir_all(&dir);
}
