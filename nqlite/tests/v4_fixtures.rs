//! Format-v4 golden fixtures — spec/file-format.md §5 (issue #143 sequence).
//!
//! The Rust engine is the fixture-gen **oracle** for nqlite-zig's M1
//! reader/writer: these committed bytes, plus `statements.json` (every
//! `Statement` tag in both serde-JSON and postcard hex), are what the zig
//! implementation must round-trip byte-exactly. §5.7: "Where prose is
//! ambiguous, the golden fixtures are the oracle."
//!
//! Regenerate after a *deliberate* spec change (never hand-edit):
//!
//! ```sh
//! UPDATE_FIXTURES=1 cargo test --test v4_fixtures
//! ```
//!
//! The default run is the CI gate: a fresh canonical encode must equal the
//! committed bytes, and every committed file must decode and re-encode
//! byte-identically (canonicality: §5.1's layout rules are self-checked).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use nql_ir::{
    Aggregate, CmpOp, Filter, Id, Knn, MatchDirection, MatchPath, MatchStep, Order, Record,
    RecordId, RelationEdge, Select, SnapshotState, Statement, Store, Value,
};
use nqlite::Database;

// The §5 container codec lives in `nqlite::v4` (lifted from this file — the
// fixtures below remain the byte oracle for it).
use nqlite::v4::{
    crc32, decode_store as decode_v4, encode_store as encode_v4, hex, parse_dir, section, tag_name,
    DirEnt, T_CLOCK, T_HISTORY, T_RECORDS, T_TABLES,
};

// ---------------------------------------------------------------------------
// Canonical fixture stores
// ---------------------------------------------------------------------------

fn rec(
    table: &str,
    id: Id,
    body: Vec<(&str, Value)>,
    embedding: Option<Vec<f32>>,
    created_at: i64,
) -> Record {
    Record {
        id: RecordId::new(table, id),
        body: body.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        embedding,
        created_at,
    }
}

fn edge(
    from: RecordId,
    name: &str,
    to: RecordId,
    created_at: i64,
    weight: Option<f32>,
    props: Vec<(&str, Value)>,
) -> RelationEdge {
    RelationEdge {
        from,
        name: name.to_string(),
        to,
        created_at,
        weight,
        props: props.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
    }
}

/// Hand-built: records only, no optional sections (all ids numeric,
/// history empty — a legal §5 file for an imported store).
fn plain_store() -> Store {
    let mut s = Store::default();
    s.tables.insert("note".into(), None);
    // Inserted in reverse: canonical order comes from the BTree, not input.
    s.insert(rec("note", Id::Num(2), vec![("z", Value::Int(1))], None, 9));
    s.insert(rec(
        "note",
        Id::Num(1),
        vec![("a", Value::Str("first".into()))],
        None,
        4,
    ));
    s.clock = 4;
    s
}

/// Engine-built: every `Value` kind, string/numeric ids in canonical order,
/// deliberately append-ordered (≠ sorted) edges, embeddings, a nested memory,
/// and the full mutation history.
fn rich_store() -> Store {
    let a1 = RecordId::new("alpha", Id::Num(1));
    let a007 = RecordId::new("alpha", Id::Str("007".into()));
    let n9 = RecordId::new("note", Id::Num(9));
    let z3 = RecordId::new("zebra", Id::Num(3));
    let s1 = RecordId::new("scratch", Id::Num(1));

    let mut db = Database::new(Store::default());
    db.execute(&[
        Statement::CreateTable {
            table: "alpha".into(),
            vector_dim: Some(4),
        },
        Statement::CreateTable {
            table: "note".into(),
            vector_dim: None,
        },
        Statement::CreateTable {
            table: "zebra".into(),
            vector_dim: Some(2),
        },
        Statement::CreateTable {
            table: "scratch".into(),
            vector_dim: None,
        },
    ])
    .expect("creates");

    db.execute(&[
        Statement::Insert(rec(
            "alpha",
            Id::Num(1),
            vec![
                ("a", Value::Int(-7)),
                ("b", Value::Bool(true)),
                ("c", Value::Str("γ ✓".into())),
                ("d", Value::Float(-0.0)),
                ("e", Value::Null),
                (
                    "f",
                    Value::Doc(
                        [
                            (
                                "g".to_string(),
                                Value::Arr(vec![
                                    Value::Int(1),
                                    Value::Str("x".into()),
                                    Value::Bool(false),
                                ]),
                            ),
                            ("h".to_string(), Value::Vector(vec![0.1, 0.25])),
                            ("i".to_string(), Value::Ref(a007.clone())),
                        ]
                        .into_iter()
                        .collect(),
                    ),
                ),
            ],
            Some(vec![0.5, 0.25, 0.5, 1.0]),
            0,
        )),
        Statement::Insert(rec(
            "alpha",
            Id::Num(5),
            vec![
                ("empty_doc", Value::Doc(BTreeMap::new())),
                ("empty_arr", Value::Arr(vec![])),
                ("s", Value::Str(String::new())),
            ],
            None,
            0,
        )),
        Statement::Insert(rec(
            "alpha",
            Id::Str("007".into()),
            vec![("k", Value::Str("v".into()))],
            None,
            0,
        )),
        Statement::Insert(rec(
            "alpha",
            Id::Str("10".into()),
            vec![("k", Value::Int(10))],
            None,
            0,
        )),
        Statement::Insert(rec(
            "alpha",
            Id::Str("2".into()),
            vec![("k", Value::Int(2))],
            None,
            0,
        )),
        Statement::Insert(rec(
            "note",
            Id::Num(9),
            vec![
                ("nan", Value::Float(f64::NAN)),
                ("inf", Value::Float(f64::INFINITY)),
                ("min", Value::Int(i64::MIN)),
                ("max", Value::Int(i64::MAX)),
            ],
            None,
            0,
        )),
        Statement::Insert(rec(
            "zebra",
            Id::Num(3),
            vec![("n", Value::Float(f64::MAX))],
            Some(vec![0.25, -0.5]),
            0,
        )),
        Statement::Insert(rec(
            "zebra",
            Id::Num(4),
            vec![("ok", Value::Bool(true))],
            None,
            0,
        )),
        Statement::Insert(rec(
            "scratch",
            Id::Num(1),
            vec![("t", Value::Str("x".into()))],
            None,
            0,
        )),
    ])
    .expect("inserts");

    // Append order here is deliberately NOT (from, name, to, created_at)
    // sorted — §5.5 pins append order; the fixture proves it.
    db.execute(&[
        Statement::Relate(edge(
            a1.clone(),
            "knows",
            n9.clone(),
            2,
            Some(0.5),
            vec![("src", Value::Str("seed".into()))],
        )),
        Statement::Relate(edge(n9.clone(), "knows", a1.clone(), 1, None, vec![])),
        Statement::Relate(edge(
            a1.clone(),
            "likes",
            z3.clone(),
            3,
            None,
            vec![("count", Value::Int(3))],
        )),
        Statement::Relate(edge(a1.clone(), "knows", n9.clone(), 4, None, vec![])),
    ])
    .expect("relates");

    db.execute(&[Statement::Forget { id: s1 }]).expect("forget");

    db.execute(&[
        Statement::Memory {
            name: "dreams".into(),
        },
        Statement::CreateTable {
            table: "tiny".into(),
            vector_dim: Some(2),
        },
        Statement::Insert(rec(
            "tiny",
            Id::Num(1),
            vec![("v", Value::Float(0.5))],
            Some(vec![0.1, 0.2]),
            0,
        )),
        Statement::Insert(rec("tiny", Id::Num(2), vec![("v", Value::Int(2))], None, 0)),
        Statement::Relate(edge(
            RecordId::new("tiny", Id::Num(1)),
            "next",
            RecordId::new("tiny", Id::Num(2)),
            1,
            Some(1.0),
            vec![],
        )),
        Statement::Memory { name: "d2".into() },
        Statement::CreateTable {
            table: "d2".into(),
            vector_dim: None,
        },
        Statement::Insert(rec(
            "d2",
            Id::Num(1),
            vec![("deep", Value::Bool(true))],
            None,
            0,
        )),
    ])
    .expect("memory block");

    let mut store = db.into_store();
    // The engine creates memories at the ROOT only (execute_in_context's
    // Memory arm runs on the root store), but §5.2 MEMORIES is recursive —
    // hand-nest `d2` inside `dreams` for coverage of blob-local offsets at
    // depth 2. Legal per `Store`'s type.
    if let Some(d2) = store.memories.remove("d2") {
        store
            .memories
            .get_mut("dreams")
            .expect("dreams exists")
            .memories
            .insert("d2".into(), d2);
    }
    // Pin the fixture's claim: append order ≠ sorted order (§5.5 errata).
    let mut sorted = store.edges.clone();
    sorted.sort_by(|x, y| {
        (&x.from, x.name.as_bytes(), &x.to, x.created_at).cmp(&(
            &y.from,
            y.name.as_bytes(),
            &y.to,
            y.created_at,
        ))
    });
    assert_ne!(store.edges, sorted, "fixture must demonstrate append order");
    store
}

/// Engine-built: `PRUNE HISTORY` compacted — declarations retained at their
/// original timestamps, `Statement::Snapshot` at the tail (the nested
/// memory's own history is pruned too — snapshots all the way down).
fn pruned_store() -> Store {
    let mut db = Database::default();
    db.execute(&[
        Statement::CreateTable {
            table: "note".into(),
            vector_dim: None,
        },
        Statement::Insert(rec(
            "note",
            Id::Num(1),
            vec![("a", Value::Str("one".into()))],
            None,
            0,
        )),
        Statement::Insert(rec(
            "note",
            Id::Num(2),
            vec![("a", Value::Str("two".into()))],
            None,
            0,
        )),
    ])
    .expect("creates/inserts");
    db.execute(&[
        Statement::Memory {
            name: "archive".into(),
        },
        Statement::CreateTable {
            table: "box".into(),
            vector_dim: None,
        },
        Statement::Insert(rec("box", Id::Num(1), vec![("w", Value::Int(1))], None, 0)),
    ])
    .expect("memory block");
    db.execute(&[Statement::PruneHistory]).expect("prune");
    db.into_store()
}

fn fixture_stores() -> Vec<(&'static str, Store)> {
    vec![
        ("empty", Store::default()),
        ("plain", plain_store()),
        ("rich", rich_store()),
        ("pruned", pruned_store()),
    ]
}

// ---------------------------------------------------------------------------
// Statement oracle (all 13 §5.7 tags) + serde JSON twin
// ---------------------------------------------------------------------------

fn stmt_tag(s: &Statement) -> u32 {
    match s {
        Statement::CreateTable { .. } => 0,
        Statement::Insert(_) => 1,
        Statement::Relate(_) => 2,
        Statement::Select(_) => 3,
        Statement::Match(_) => 4,
        Statement::Closure(_) => 5,
        Statement::Forget { .. } => 6,
        Statement::Memory { .. } => 7,
        Statement::ContextReset => 8,
        Statement::MatchCount(_) => 9,
        Statement::PruneHistory => 10,
        Statement::Snapshot(_) => 11,
        Statement::HistorySince(_) => 12,
    }
}

fn stmt_name(s: &Statement) -> &'static str {
    match s {
        Statement::CreateTable { .. } => "CreateTable",
        Statement::Insert(_) => "Insert",
        Statement::Relate(_) => "Relate",
        Statement::Select(_) => "Select",
        Statement::Match(_) => "Match",
        Statement::Closure(_) => "Closure",
        Statement::Forget { .. } => "Forget",
        Statement::Memory { .. } => "Memory",
        Statement::ContextReset => "ContextReset",
        Statement::MatchCount(_) => "MatchCount",
        Statement::PruneHistory => "PruneHistory",
        Statement::Snapshot(_) => "Snapshot",
        Statement::HistorySince(_) => "HistorySince",
    }
}

/// Every `Statement` tag, with coverage over the nested enums a non-Rust
/// reader must implement (`Filter` ×7, `Order` ×8, `CmpOp` ×5, directions,
/// `Aggregate`, boxed `SnapshotState`, unit/tuple/struct variants).
fn statement_samples() -> Vec<Statement> {
    let a1 = RecordId::new("alpha", Id::Num(1));
    let alice = RecordId::new("alpha", Id::Str("alice".into()));
    let sample_rec = Record {
        id: a1.clone(),
        body: BTreeMap::from([
            ("k".to_string(), Value::Int(-3)),
            ("r".to_string(), Value::Ref(alice.clone())),
        ]),
        embedding: None,
        created_at: 7,
    };
    let sample_edge = RelationEdge {
        from: a1.clone(),
        name: "knows".into(),
        to: alice.clone(),
        created_at: 2,
        weight: Some(0.5),
        props: BTreeMap::from([("w".to_string(), Value::Bool(true))]),
    };
    let path = |as_of: Option<i64>| MatchPath {
        start: a1.clone(),
        steps: vec![
            MatchStep {
                direction: MatchDirection::Out,
                name: "knows".into(),
                edge_props: Some(Filter::FieldCmp {
                    field: "w".into(),
                    op: CmpOp::Ge,
                    value: Value::Int(1),
                }),
            },
            MatchStep {
                direction: MatchDirection::In,
                name: "likes".into(),
                edge_props: None,
            },
        ],
        as_of,
    };
    let sel = |table: &str, filter: Option<Filter>, order: Option<Order>| {
        Statement::Select(Select {
            table: table.into(),
            filter,
            order,
            ..Default::default()
        })
    };
    vec![
        Statement::CreateTable {
            table: "alpha".into(),
            vector_dim: Some(4),
        },
        Statement::CreateTable {
            table: "note".into(),
            vector_dim: None,
        },
        Statement::Insert(sample_rec.clone()),
        Statement::Relate(sample_edge.clone()),
        // One fat SELECT: kNN, And-combined filters (HasEmbedding, Bm25,
        // FieldIn, FieldBetween, FieldCmp Lt/Le/Gt/Ge), SalienceWeighted,
        // as_of/fields/offset/aggregate.
        Statement::Select(Select {
            table: "alpha".into(),
            knn: Some(Knn {
                query: vec![0.25, 0.5, 1.0, 0.125],
                k: 5,
            }),
            filter: Some(Filter::And(vec![
                Filter::HasEmbedding,
                Filter::Bm25 {
                    field: "t".into(),
                    query: "hello world".into(),
                    k: Some(3),
                },
                Filter::FieldIn {
                    field: "tag".into(),
                    values: vec![Value::Str("x".into()), Value::Null],
                },
                Filter::FieldBetween {
                    field: "n".into(),
                    lo: Value::Int(1),
                    hi: Value::Int(9),
                },
                Filter::FieldCmp {
                    field: "n".into(),
                    op: CmpOp::Lt,
                    value: Value::Int(0),
                },
                Filter::FieldCmp {
                    field: "n".into(),
                    op: CmpOp::Le,
                    value: Value::Int(1),
                },
                Filter::FieldCmp {
                    field: "n".into(),
                    op: CmpOp::Gt,
                    value: Value::Int(2),
                },
                Filter::FieldCmp {
                    field: "n".into(),
                    op: CmpOp::Ge,
                    value: Value::Int(3),
                },
            ])),
            order: Some(Order::SalienceWeighted([0.7, 0.0, 0.0, 0.3])),
            limit: Some(10),
            as_of: Some(3),
            fields: Some(vec!["k".into()]),
            offset: Some(2),
            aggregate: Some(Aggregate::CountStar),
        }),
        sel(
            "note",
            Some(Filter::FieldEquals {
                field: "s".into(),
                value: Value::Float(-0.0),
            }),
            Some(Order::Votes),
        ),
        sel(
            "note",
            Some(Filter::FieldCmp {
                field: "n".into(),
                op: CmpOp::Ne,
                value: Value::Bool(false),
            }),
            Some(Order::Recency),
        ),
        sel(
            "note",
            None,
            Some(Order::Field {
                key: "n".into(),
                desc: true,
            }),
        ),
        sel("note", None, Some(Order::Similarity)),
        sel("note", None, Some(Order::Salience)),
        sel("note", None, Some(Order::Score)),
        sel("note", None, Some(Order::Feedback)),
        Statement::Match(path(None)),
        Statement::Closure(path(Some(1))),
        Statement::Forget { id: a1.clone() },
        Statement::Memory {
            name: "dreams".into(),
        },
        Statement::ContextReset,
        Statement::MatchCount(path(None)),
        Statement::PruneHistory,
        Statement::Snapshot(Box::new(SnapshotState {
            records: BTreeMap::from([(a1.clone(), sample_rec.clone())]),
            edges: vec![sample_edge.clone()],
            vector_dims: BTreeMap::from([("alpha".into(), 4)]),
            clock: 9,
            memories: BTreeMap::from([("m".into(), Store::default())]),
            tables: BTreeMap::from([("alpha".into(), Some(4))]),
        })),
        Statement::HistorySince(42),
    ]
}

fn statements_json() -> String {
    let entries: Vec<serde_json::Value> = statement_samples()
        .into_iter()
        .map(|s| {
            let hexed = hex(&postcard::to_allocvec(&s).expect("postcard sample"));
            // SnapshotState maps records by RecordId — serde_json only
            // accepts string keys, so the JSON twin is null for tag 11;
            // its semantics are pinned by pruned.nql / rich.nql history.
            let json = if stmt_name(&s) == "Snapshot" {
                serde_json::Value::Null
            } else {
                serde_json::to_value(&s).expect("json sample")
            };
            serde_json::json!({
                "name": stmt_name(&s),
                "tag": stmt_tag(&s),
                "hex": hexed,
                "json": json,
            })
        })
        .collect();
    format!("{}\n", serde_json::to_string_pretty(&entries).unwrap())
}

// ---------------------------------------------------------------------------
// Manifest + fixture I/O
// ---------------------------------------------------------------------------

fn update_mode() -> bool {
    std::env::var_os("UPDATE_FIXTURES").is_some()
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../spec/fixtures/v4")
}

fn manifest_json() -> serde_json::Value {
    let fixtures: Vec<serde_json::Value> = fixture_stores()
        .into_iter()
        .map(|(name, store)| {
            let bytes = encode_v4(&store).expect("encode");
            let dir = parse_dir(&bytes).expect("dir");
            let sections: Vec<serde_json::Value> = dir
                .iter()
                .map(|e| {
                    serde_json::json!({
                        "tag": e.tag,
                        "name": tag_name(e.tag),
                        "offset": e.off,
                        "len": e.len,
                        "crc32": e.crc,
                    })
                })
                .collect();
            serde_json::json!({
                "file": format!("{name}.nql"),
                "bytes": bytes.len(),
                "tables": store.tables.len(),
                "records": store.records.len(),
                "edges": store.edges.len(),
                "memories": store.memories.len(),
                "history": store.history.len(),
                "sections": sections,
            })
        })
        .collect();
    serde_json::json!({
        "note": "generated by nqlite/tests/v4_fixtures.rs (UPDATE_FIXTURES=1); spec/file-format.md §5; never hand-edit",
        "fixtures": fixtures,
        "statements": { "file": "statements.json", "count": statement_samples().len() },
    })
}

fn write_or_verify(path: &Path, bytes: &[u8], label: &str) {
    if update_mode() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    } else {
        let committed = std::fs::read(path).unwrap_or_else(|e| {
            panic!(
                "{label}: {} unreadable ({e}) — run with UPDATE_FIXTURES=1 once",
                path.display()
            )
        });
        assert_eq!(
            committed, bytes,
            "{label} drifted from the canonical encode — spec change? regenerate + bump .spec-pin"
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// §5.1 arithmetic, pinned: an empty store is exactly 144 bytes —
/// 24-byte header + 3×32 directory + three 8-byte required sections.
#[test]
fn empty_layout_is_pinned() {
    let bytes = encode_v4(&Store::default()).unwrap();
    let dir = parse_dir(&bytes).unwrap();
    let named: Vec<(u32, u64, u64)> = dir.iter().map(|e| (e.tag, e.off, e.len)).collect();
    assert_eq!(
        named,
        [(T_TABLES, 120, 8), (T_RECORDS, 128, 8), (T_CLOCK, 136, 8)]
    );
    assert_eq!(bytes.len(), 144);
}

/// Canonical RecordId order (§5.4): table bytes → `Num` < `Str` rank →
/// u64 / byte order — including the traps: digit-first string ids sort
/// as strings (`007` < `10` < `2`), never numerically, and numeric ids
/// of any value rank below every string id.
#[test]
fn canonical_record_order() {
    let mut s = Store::default();
    s.tables.insert("b".into(), None);
    s.tables.insert("a".into(), None);
    for (table, id) in [
        ("a", Id::Num(5)),
        ("a", Id::Str("007".into())),
        ("a", Id::Str("10".into())),
        ("a", Id::Str("2".into())),
        ("a", Id::Num(1)),
        ("b", Id::Num(0)),
    ] {
        s.insert(rec(table, id, vec![("k", Value::Int(1))], None, 1));
    }
    let bytes = encode_v4(&s).unwrap();
    let dec = decode_v4(&bytes).unwrap();
    let ids: Vec<String> = dec.records.keys().map(|k| k.to_string()).collect();
    assert_eq!(ids, ["a:1", "a:5", "a:007", "a:10", "a:2", "b:0"]);
    assert_eq!(encode_v4(&dec).unwrap(), bytes, "round-trip byte-exact");
}

/// The core gate: every fixture decodes and re-encodes byte-identically,
/// and equals the committed bytes (or is regenerated under UPDATE_FIXTURES).
/// Structural `PartialEq` is asserted where the types permit it: `rich`
/// carries a NaN (not `PartialEq`-reflexive), and `pruned` embeds stores
/// inside `Statement::Snapshot`, whose `Store.tables` is `#[serde(skip)]`
/// — dropped by postcard everywhere (v3 included) and rebuilt from the
/// inner history at replay (issue #133). Byte identity is the real gate.
#[test]
fn fixtures_roundtrip_and_match_files() {
    for (name, store) in fixture_stores() {
        let bytes = encode_v4(&store).expect(name);
        write_or_verify(&fixtures_dir().join(format!("{name}.nql")), &bytes, name);
        let dec = decode_v4(&bytes).unwrap_or_else(|e| panic!("{name}: decode: {e}"));
        assert_eq!(
            encode_v4(&dec).unwrap(),
            bytes,
            "{name}: decode → re-encode is not byte-exact"
        );
        if matches!(name, "empty" | "plain") {
            assert_eq!(dec, store, "{name}: structural round-trip");
        }
    }
}

/// Semantic spot-checks on the engine-built fixtures.
#[test]
fn fixture_semantics() {
    let by_name: BTreeMap<&str, Store> = fixture_stores().into_iter().collect();

    let plain = decode_v4(&encode_v4(&by_name["plain"]).unwrap()).unwrap();
    assert_eq!(plain.records.len(), 2);
    assert!(plain.edges.is_empty() && plain.history.is_empty() && plain.memories.is_empty());
    assert_eq!(plain.tables.get("note"), Some(&None));
    assert_eq!(plain.clock, 4);

    let rich = decode_v4(&encode_v4(&by_name["rich"]).unwrap()).unwrap();
    // Root records: scratch:1 was forgotten; tiny/d2 live under memories.
    let keys: Vec<String> = rich.records.keys().map(|k| k.to_string()).collect();
    assert_eq!(
        keys,
        [
            "alpha:1",
            "alpha:5",
            "alpha:007",
            "alpha:10",
            "alpha:2",
            "note:9",
            "zebra:3",
            "zebra:4"
        ]
    );
    let nan_body = &rich.records[&RecordId::new("note", Id::Num(9))].body;
    assert!(matches!(nan_body["nan"], Value::Float(f) if f.is_nan()));
    assert_eq!(
        rich.records[&RecordId::new("alpha", Id::Num(1))].embedding,
        Some(vec![0.5, 0.25, 0.5, 1.0])
    );
    assert_eq!(
        rich.records[&RecordId::new("zebra", Id::Num(4))].embedding,
        None
    );
    // Append order preserved (≠ sorted — see rich_store's own assert).
    assert_eq!(rich.edges.len(), 4);
    assert_eq!(rich.edges[2].name, "likes");
    assert_eq!(rich.history.len(), 18);
    assert_eq!(rich.tables.len(), 4);
    assert_eq!(rich.vector_dims.get("alpha"), Some(&4));
    assert!(!rich.vector_dims.contains_key("note"));
    // Memories: root-level `dreams` (engine), `d2` hand-nested inside it
    // for §5.2 recursion coverage.
    assert_eq!(rich.memories.len(), 1);
    let dreams = &rich.memories["dreams"];
    assert_eq!(dreams.records.len(), 2);
    assert_eq!(dreams.history.len(), 4);
    assert_eq!(dreams.tables.len(), 1);
    assert_eq!(dreams.edges.len(), 1);
    assert_eq!(dreams.memories.len(), 1);
    let d2 = &dreams.memories["d2"];
    assert_eq!(d2.records.len(), 1);
    assert_eq!(d2.history.len(), 2);
    assert_eq!(d2.tables.len(), 1);

    let pruned = decode_v4(&encode_v4(&by_name["pruned"]).unwrap()).unwrap();
    assert_eq!(pruned.records.len(), 2);
    assert!(
        matches!(pruned.history.last(), Some((_, Statement::Snapshot(_)))),
        "pruned history ends in a Snapshot"
    );
    // Declarations retained at their original timestamps (issues #95/#89).
    assert!(pruned
        .history
        .iter()
        .any(|(ts, s)| *ts == 1
            && matches!(s, Statement::CreateTable { table, .. } if table == "note")));
    // The snapshot embeds the nested memory, itself pruned to its own
    // declaration + snapshot pair (recursive Snapshot states).
    let snapshot = pruned
        .history
        .iter()
        .find_map(|(_, s)| match s {
            Statement::Snapshot(st) => Some(st),
            _ => None,
        })
        .expect("snapshot present");
    let archive = &snapshot.memories["archive"];
    assert_eq!(archive.records.len(), 1);
    assert!(
        matches!(archive.history.last(), Some((_, Statement::Snapshot(_)))),
        "nested memory history is pruned too"
    );
    assert_eq!(pruned.memories["archive"].records.len(), 1);
}

/// §5.6: the HISTORY payload is byte-identical to v3's history tail —
/// the same postcard call `storage.rs` writes (conversion = adopt verbatim).
#[test]
fn history_section_is_v3_tail_verbatim() {
    for (name, store) in fixture_stores() {
        let bytes = encode_v4(&store).unwrap();
        let dir = parse_dir(&bytes).unwrap();
        if store.history.is_empty() {
            assert!(
                !dir.iter().any(|e| e.tag == T_HISTORY),
                "{name}: empty history ⇒ no section"
            );
            continue;
        }
        let (_, payload) = section(&bytes, &dir, T_HISTORY).unwrap();
        let tail = postcard::to_allocvec(&store.history).unwrap();
        assert_eq!(
            payload, tail,
            "{name}: HISTORY must equal postcard(history)"
        );
    }
}

/// Recompute a section's CRC after a deliberate payload mutation (used to
/// reach the semantic checks behind the integrity check).
fn fix_crc(buf: &mut [u8], dir: &[DirEnt], tag: u32) {
    let idx = dir.iter().position(|e| e.tag == tag).unwrap();
    let e = &dir[idx];
    let c = crc32(&buf[e.off as usize..(e.off + e.len) as usize]);
    let crc_field = 24 + 32 * idx + 24;
    buf[crc_field..crc_field + 4].copy_from_slice(&c.to_le_bytes());
}

/// §5.1 loud-failure contract: every malformed container is rejected with a
/// specific error — never a partial load.
#[test]
fn reader_rejects_malformed() {
    let base = encode_v4(&plain_store()).unwrap();
    let dir = parse_dir(&base).unwrap();
    let err = |b: &[u8]| decode_v4(b).unwrap_err();
    let rec_off = dir.iter().find(|e| e.tag == T_RECORDS).unwrap().off as usize;

    let mut b = base.clone();
    b[0] ^= 0x01;
    assert!(err(&b).contains("magic"));

    let mut b = base.clone();
    b[8..12].copy_from_slice(&5u32.to_le_bytes());
    assert!(err(&b).contains("unsupported format version 5"));

    let mut b = base.clone();
    b[12] = 1;
    assert!(err(&b).contains("flags"));

    let mut b = base.clone();
    b.truncate(30);
    assert!(err(&b).contains("truncated"));

    // CRC: flip the last byte (inside the last section's payload).
    let mut b = base.clone();
    let last = b.len() - 1;
    b[last] ^= 0xff;
    assert!(err(&b).contains("CRC mismatch in section"));

    // Required section missing: rebuild the directory with 2 entries, keep
    // TABLES/RECORDS payloads, drop CLOCK (file then ends at RECORDS).
    let mut b = Vec::new();
    b.extend_from_slice(&base[0..16]);
    b.extend_from_slice(&2u32.to_le_bytes());
    b.extend_from_slice(&base[20..24]);
    b.extend_from_slice(&base[24..88]); // two directory entries
    b.extend_from_slice(&vec![0u8; dir[0].off as usize - b.len()]);
    b.extend_from_slice(&base[dir[0].off as usize..(dir[1].off + dir[1].len) as usize]);
    assert!(err(&b).contains("required section CLOCK"));

    // Unknown tag.
    let mut b = base.clone();
    b[24..28].copy_from_slice(&9u32.to_le_bytes());
    assert!(err(&b).contains("unknown section tag 9"));

    // Duplicate / non-ascending tag.
    let mut b = base.clone();
    b[56..60].copy_from_slice(&1u32.to_le_bytes());
    assert!(err(&b).contains("ascending"));

    // Out-of-bounds length (last entry's len field = 24 + 32·last + 16).
    let last_i = dir.len() - 1;
    let len_field = 24 + 32 * last_i + 16;
    let mut b = base.clone();
    b[len_field..len_field + 8].copy_from_slice(&(1u64 << 40).to_le_bytes());
    assert!(err(&b).contains("out of bounds"));

    // Semantic: record table_idx beyond TABLES — patch + fix the CRC so the
    // payload check passes and the cross-reference check fires.
    let mut b = base.clone();
    b[rec_off + 8..rec_off + 12].copy_from_slice(&9u32.to_le_bytes());
    fix_crc(&mut b, &dir, T_RECORDS);
    assert!(err(&b).contains("table_idx 9"));

    // Semantic: string id without a STRINGS section.
    let mut b = base.clone();
    b[rec_off + 12] = 1; // id_kind = 1
    fix_crc(&mut b, &dir, T_RECORDS);
    assert!(err(&b).contains("STRINGS"));
}

/// `statements.json` — all 13 tags as (serde JSON, postcard hex) twins.
/// The hex side is what other implementations re-encode and compare; the
/// JSON side pins the *meaning* (variant names, field names, tag numbers).
#[test]
fn statements_oracle_matches_files() {
    let text = statements_json();
    write_or_verify(
        &fixtures_dir().join("statements.json"),
        text.as_bytes(),
        "statements",
    );
    // Sanity: coverage of every §5.7 Statement tag, 0..=12.
    let tags: Vec<u32> = statement_samples().iter().map(stmt_tag).collect();
    for t in 0..=12u32 {
        assert!(tags.contains(&t), "no sample for Statement tag {t}");
    }
    // JSON twin round-trips through serde itself (guards drift between the
    // hex and json fields of each entry).
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&text).unwrap();
    assert_eq!(parsed.len(), statement_samples().len());
    for (entry, stmt) in parsed.iter().zip(statement_samples()) {
        assert_eq!(entry["tag"], serde_json::json!(stmt_tag(&stmt)));
        assert_eq!(entry["name"], serde_json::json!(stmt_name(&stmt)));
        assert_eq!(
            entry["hex"],
            serde_json::json!(hex(&postcard::to_allocvec(&stmt).unwrap()))
        );
        if stmt_name(&stmt) == "Snapshot" {
            assert!(
                entry["json"].is_null(),
                "Snapshot json twin is null by design"
            );
        } else {
            assert_eq!(entry["json"], serde_json::to_value(&stmt).unwrap());
        }
    }
}

/// `manifest.json` — independent section-table/count expectations for the
/// consuming implementation (offsets, CRCs, presence rules).
#[test]
fn manifest_matches_fixtures() {
    let text = format!(
        "{}\n",
        serde_json::to_string_pretty(&manifest_json()).unwrap()
    );
    write_or_verify(
        &fixtures_dir().join("manifest.json"),
        text.as_bytes(),
        "manifest",
    );
}
