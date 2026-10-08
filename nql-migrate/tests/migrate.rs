//! Migration gate: build a real v2/v3 store (incl. WAL + MEMORY + PRUNE),
//! migrate it, and require the decoded v4 bytes to equal the source store.

use std::collections::BTreeMap;

use nqlite::{Database, Id, Record, RecordId, RelationEdge, Statement, Value};

/// Deterministic store exercising the whole container surface: declared
/// tables (dim + no-dim), string/numeric ids, embeddings, vote edges,
/// a FORGETed record, a MEMORY block, and a pruned history (snapshots).
fn seed(path: &std::path::Path) {
    let mut db = Database::open(path).expect("open v3 store");
    let mut s = |stmts: Vec<Statement>| db.execute(&stmts).expect("execute");

    s(vec![
        Statement::CreateTable {
            table: "doc".into(),
            vector_dim: Some(2),
        },
        Statement::CreateTable {
            table: "note".into(),
            vector_dim: None,
        },
        Statement::CreateTable {
            table: "scratch".into(),
            vector_dim: None,
        }, // decl-only
    ]);
    s(vec![
        Statement::Insert(Record {
            id: RecordId::new("doc", Id::Num(1)),
            body: BTreeMap::from([
                ("text".into(), Value::Str("alpha beta".into())),
                ("n".into(), Value::Int(-7)),
            ]),
            embedding: Some(vec![0.5, -0.5]),
            created_at: 0,
        }),
        Statement::Insert(Record {
            id: RecordId::new("doc", Id::Str("two".into())),
            body: BTreeMap::from([("text".into(), Value::Str("gamma".into()))]),
            embedding: Some(vec![1.0, 0.0]),
            created_at: 0,
        }),
        Statement::Insert(Record {
            id: RecordId::new("note", Id::Num(9)),
            body: BTreeMap::from([("v".into(), Value::Float(-0.0))]),
            embedding: None,
            created_at: 0,
        }),
        Statement::Insert(Record {
            id: RecordId::new("scratch", Id::Num(1)),
            body: BTreeMap::new(),
            embedding: None,
            created_at: 0,
        }),
    ]);
    s(vec![
        Statement::Relate(RelationEdge {
            from: RecordId::new("doc", Id::Num(1)),
            name: "voted".into(),
            to: RecordId::new("note", Id::Num(9)),
            created_at: 0,
            weight: Some(0.5),
            props: BTreeMap::from([("value".into(), Value::Int(1))]),
        }),
        Statement::Relate(RelationEdge {
            from: RecordId::new("doc", Id::Str("two".into())),
            name: "refs".into(),
            to: RecordId::new("doc", Id::Num(1)),
            created_at: 0,
            weight: None,
            props: BTreeMap::new(),
        }),
    ]);
    // FORGET removes the record AND its incident edges (cascade).
    s(vec![Statement::Forget {
        id: RecordId::new("scratch", Id::Num(1)),
    }]);
    // MEMORY block: inner table + row (own clock/history).
    s(vec![
        Statement::Memory {
            name: "archive".into(),
        },
        Statement::CreateTable {
            table: "box".into(),
            vector_dim: None,
        },
        Statement::Insert(Record {
            id: RecordId::new("box", Id::Num(1)),
            body: BTreeMap::from([("kept".into(), Value::Bool(true))]),
            embedding: None,
            created_at: 0,
        }),
    ]);
    // PRUNE: declarations + snapshot ride in history (v3 stores it verbatim).
    s(vec![Statement::PruneHistory]);
    // WAL has entries at this point: reopening must replay them into the
    // store BEFORE migrate reads it (migrate opens the file itself).
    drop(db);
    let again = Database::open(path).expect("reopen (WAL replay)");
    drop(again);
}

#[test]
fn migrate_v3_to_v4_roundtrip() {
    let dir = std::env::temp_dir().join(format!("nql-migrate-rt-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let v3 = dir.join("store.ndb");
    let v4_path = dir.join("store_v4.ndb");

    seed(&v3);

    // Source of truth: what a fresh open of the v3 store yields.
    let source = Database::open(&v3).expect("reopen source").into_store();

    let report = nql_migrate::migrate(&v3, &v4_path, false).expect("migrate");
    assert_eq!(report.from_version, 3);
    assert_eq!(report.records, source.records.len());
    assert_eq!(report.edges, source.edges.len());
    assert_eq!(report.history, source.history.len());
    assert_eq!(report.memories, source.memories.len());
    assert!(report.bytes > 0);

    // The gate: migrated bytes re-encode byte-identically (canonicality),
    // and every wire-lossless part of the store survives — records (FORGET
    // cascade held), edges, the decl-only table, clock, memories + their
    // nested TABLES. (`Store.tables` inside snapshot STATEMENTS is
    // `serde(skip)`ed by design and rebuilt at replay — full equality can't
    // cross that boundary; the fixtures document the same rule.)
    let raw = std::fs::read(&v4_path).unwrap();
    let back = nqlite::v4::decode_store(&raw).expect("decode migrated v4");
    assert_eq!(
        nqlite::v4::encode_store(&back).unwrap(),
        raw,
        "re-encode stable"
    );
    assert_eq!(back.records, source.records);
    assert_eq!(back.edges, source.edges);
    assert_eq!(back.tables, source.tables);
    assert_eq!(back.clock, source.clock);
    assert_eq!(back.memories, source.memories);
    assert_eq!(back.history.len(), source.history.len());
    // Declared-only table survives (the #89 re-seed contract).
    assert_eq!(back.tables.get("scratch"), Some(&None));
    // FORGET cascade held: no scratch:1 record, no edges reference it.
    assert!(!back
        .records
        .contains_key(&RecordId::new("scratch", Id::Num(1))));
    // MEMORY block contents + nested TABLES section.
    let archive = back
        .memories
        .iter()
        .find(|(n, _)| n.as_str() == "archive")
        .expect("archive");
    assert_eq!(archive.1.records.len(), 1);
    assert_eq!(archive.1.tables.get("box"), Some(&None));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn migrate_is_idempotent_and_guards_output() {
    let dir = std::env::temp_dir().join(format!("nql-migrate-guard-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let v3 = dir.join("store.ndb");
    seed(&v3);
    let out = dir.join("out.ndb");

    nql_migrate::migrate(&v3, &out, false).expect("first migrate");
    // Existing output without --force is refused…
    let err = nql_migrate::migrate(&v3, &out, false).unwrap_err();
    assert!(matches!(err, nql_migrate::Error::OutputExists(_)), "{err}");
    // …with --force it re-runs…
    nql_migrate::migrate(&v3, &out, true).expect("force");
    // …and a second forced run is byte-identical (determinism).
    let first = std::fs::read(&out).unwrap();
    nql_migrate::migrate(&v3, &out, true).expect("force again");
    assert_eq!(first, std::fs::read(&out).unwrap());

    // Missing input is a clear error (never a silent empty migration).
    let missing = nql_migrate::migrate(dir.join("nope.ndb"), dir.join("x.ndb"), false).unwrap_err();
    assert!(
        matches!(missing, nql_migrate::Error::InputMissing(_)),
        "{missing}"
    );

    // In-place migration (in == out) works: lock released before the write.
    let inplace = dir.join("store.ndb");
    nql_migrate::migrate(&inplace, &inplace, true).expect("in-place");
    let raw = std::fs::read(&inplace).unwrap();
    assert_eq!(u32::from_le_bytes(raw[8..12].try_into().unwrap()), 4);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn migrate_reads_history_from_checkpointed_v3_input() {
    // Regression (E08 100k importer proof): a CHECKPOINTED v3 input hands
    // `Database::open` a pending history tail (issue #133). Migrate must
    // claim it (`ensure_history`) or the output silently carries EMPTY
    // history — `into_store` alone used to hand back the never-ensured store.
    let dir = std::env::temp_dir().join(format!("nql-migrate-lazy-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let v3 = dir.join("store.ndb");
    let v4_path = dir.join("store_v4.ndb");

    seed(&v3);
    // Checkpoint the WAL into a real v3 main file (history becomes a lazy
    // tail for every subsequent open).
    {
        let mut db = Database::open(&v3).expect("open for flush");
        db.flush().expect("flush → main file");
    }

    // Expected history = what a claiming open sees.
    let mut src = Database::open(&v3).expect("reopen");
    src.ensure_history().expect("claim tail");
    let expected = src.store().history.len();
    assert!(expected > 0, "seeded store has history");
    drop(src); // release the single-writer lock before migrate opens it

    let report = nql_migrate::migrate(&v3, &v4_path, false).expect("migrate");
    assert_eq!(
        report.history, expected,
        "migrated report carries the file-era history (pre-fix: 0)"
    );

    let raw = std::fs::read(&v4_path).unwrap();
    let back = nqlite::v4::decode_store(&raw).expect("decode migrated v4");
    assert_eq!(
        back.history.len(),
        expected,
        "v4 output carries the history"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
