//! Unit tests for the M0 nql grammar slice: one test per statement kind plus
//! error paths. All assertions are on the parsed IR (`nql_ir::Plan`).

use crate::{parse, parse_statement, NqlError};
use nql_ir::{
    Aggregate, CmpOp, Filter, Id, Knn, MatchDirection, Order, RecordId, Select, Statement, Value,
};
use std::collections::BTreeMap;

fn rid(s: &str) -> RecordId {
    RecordId::parse(s).expect("valid record id in test")
}

fn doc(entries: &[(&str, Value)]) -> BTreeMap<String, Value> {
    entries
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect()
}

#[test]
fn create_table_plain() {
    let plan = parse("CREATE TABLE person").unwrap();
    assert_eq!(
        plan,
        vec![Statement::CreateTable {
            table: "person".into(),
            vector_dim: None,
        }]
    );
}

#[test]
fn create_table_with_vector_dim() {
    let plan = parse("CREATE TABLE doc VECTOR<f32, 384>").unwrap();
    assert_eq!(
        plan,
        vec![Statement::CreateTable {
            table: "doc".into(),
            vector_dim: Some(384),
        }]
    );
}

#[test]
fn insert_with_body_and_embed() {
    let plan = parse(
        r#"INSERT INTO note:42 { "text": "hello\nworld", "count": 3, "pi": 3.5, "ok": true, "tags": ["a", "b"], "vec": [1.0, 2.0], "meta": {"x": 1} } EMBED [0.1, 0.2, 0.3]"#,
    )
    .unwrap();
    let Statement::Insert(rec) = &plan[0] else {
        panic!("expected Insert, got {:?}", plan[0]);
    };
    assert_eq!(rec.id, rid("note:42"));
    assert_eq!(rec.created_at, 0, "parser must not clock created_at");
    assert_eq!(
        rec.embedding,
        Some(vec![0.1, 0.2, 0.3]),
        "EMBED clause sets the record embedding"
    );
    assert_eq!(
        rec.body.get("text"),
        Some(&Value::Str("hello\nworld".into()))
    );
    assert_eq!(rec.body.get("count"), Some(&Value::Int(3)));
    assert_eq!(rec.body.get("pi"), Some(&Value::Float(3.5)));
    assert_eq!(rec.body.get("ok"), Some(&Value::Bool(true)));
    assert_eq!(
        rec.body.get("tags"),
        Some(&Value::Arr(vec![
            Value::Str("a".into()),
            Value::Str("b".into())
        ]))
    );
    assert_eq!(
        rec.body.get("vec"),
        Some(&Value::Vector(vec![1.0, 2.0])),
        "all-numeric arrays collapse to Value::Vector"
    );
    assert_eq!(
        rec.body.get("meta"),
        Some(&Value::Doc(doc(&[("x", Value::Int(1))])))
    );
}

#[test]
fn insert_uses_string_id_and_bare_keys() {
    let plan = parse(r#"INSERT INTO person:alice { name: "Alice", age: 30 }"#).unwrap();
    let Statement::Insert(rec) = &plan[0] else {
        panic!("expected Insert");
    };
    assert_eq!(rec.id, rid("person:alice"));
    assert_eq!(rec.id.id, Id::Str("alice".into()));
    assert_eq!(rec.body.get("name"), Some(&Value::Str("Alice".into())));
    assert_eq!(rec.body.get("age"), Some(&Value::Int(30)));
    assert!(rec.embedding.is_none());
}

#[test]
fn relate_with_weight_and_props() {
    let plan = parse(
        r#"RELATE (person:1) -> :likes -> (note:42) SET weight = 0.9, confidence = 0.5, note = "seen""#,
    )
    .unwrap();
    let Statement::Relate(e) = &plan[0] else {
        panic!("expected Relate, got {:?}", plan[0]);
    };
    assert_eq!(e.from, rid("person:1"));
    assert_eq!(e.name, "likes");
    assert_eq!(e.to, rid("note:42"));
    assert_eq!(e.weight, Some(0.9));
    assert_eq!(e.created_at, 0);
    assert_eq!(
        e.props,
        doc(&[
            ("confidence", Value::Float(0.5)),
            ("note", Value::Str("seen".into())),
        ])
    );
}

#[test]
fn relate_without_set() {
    let plan = parse("RELATE (a:1) -> :mentions -> (b:2)").unwrap();
    let Statement::Relate(e) = &plan[0] else {
        panic!("expected Relate");
    };
    assert_eq!(e.name, "mentions");
    assert_eq!(e.weight, None);
    assert!(e.props.is_empty());
}

#[test]
fn match_outgoing_one_hop() {
    let plan = parse("MATCH (turn:3) -> :mentions").unwrap();
    let Statement::Match(p) = &plan[0] else {
        panic!("expected Match, got {:?}", plan[0]);
    };
    assert_eq!(p.start, rid("turn:3"));
    assert_eq!(p.steps.len(), 1);
    assert_eq!(p.steps[0].direction, MatchDirection::Out);
    assert_eq!(p.steps[0].name, "mentions");
}

#[test]
fn match_incoming_one_hop() {
    let plan = parse("MATCH (note:42) <- :mentions").unwrap();
    let Statement::Match(p) = &plan[0] else {
        panic!("expected Match, got {:?}", plan[0]);
    };
    assert_eq!(p.start, rid("note:42"));
    assert_eq!(p.steps.len(), 1);
    assert_eq!(p.steps[0].direction, MatchDirection::In);
    assert_eq!(p.steps[0].name, "mentions");
}

#[test]
fn match_multi_hop_path() {
    let plan = parse("MATCH (a:1) -> :knows -> :works_with <- :knows").unwrap();
    let Statement::Match(p) = &plan[0] else {
        panic!("expected Match, got {:?}", plan[0]);
    };
    assert_eq!(p.start, rid("a:1"));
    let dirs: Vec<MatchDirection> = p.steps.iter().map(|s| s.direction).collect();
    let names: Vec<&str> = p.steps.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(
        dirs,
        vec![MatchDirection::Out, MatchDirection::Out, MatchDirection::In]
    );
    assert_eq!(names, vec!["knows", "works_with", "knows"]);
}

#[test]
fn match_requires_edge_step() {
    let err = parse("MATCH (a:1)").unwrap_err();
    assert!(
        err.to_string().contains("at least one edge step"),
        "got: {err}"
    );
}

#[test]
fn match_is_case_insensitive_keyword() {
    let plan = parse("match (a:1) -> :likes").unwrap();
    assert!(matches!(&plan[0], Statement::Match(_)));
}

#[test]
fn closure_parses_same_path_grammar() {
    let plan = parse("CLOSURE (turn:3) -> :mentions").unwrap();
    let Statement::Closure(p) = &plan[0] else {
        panic!("expected Closure, got {:?}", plan[0]);
    };
    assert_eq!(p.start, rid("turn:3"));
    assert_eq!(p.steps.len(), 1);
    assert_eq!(p.steps[0].direction, MatchDirection::Out);
    assert_eq!(p.steps[0].name, "mentions");
    assert!(p.steps[0].edge_props.is_none());
}

#[test]
fn closure_requires_edge_step() {
    let err = parse("CLOSURE (a:1)").unwrap_err();
    assert!(
        err.to_string().contains("at least one edge step"),
        "got: {err}"
    );
}

#[test]
fn match_step_edge_props_filter() {
    let plan = parse("MATCH (turn:3) -> :mentions WHERE confidence = 0.9").unwrap();
    let Statement::Match(p) = &plan[0] else {
        panic!("expected Match, got {:?}", plan[0]);
    };
    assert_eq!(
        p.steps[0].edge_props,
        Some(Filter::FieldEquals {
            field: "confidence".into(),
            value: Value::Float(0.9),
        })
    );
}

#[test]
fn match_multi_step_with_per_step_edge_props() {
    let plan = parse("MATCH (a:1) -> :knows WHERE weight = 1.0 -> :works_with WHERE weight = 0.5")
        .unwrap();
    let Statement::Match(p) = &plan[0] else {
        panic!("expected Match, got {:?}", plan[0]);
    };
    assert_eq!(p.steps.len(), 2);
    assert!(matches!(
        &p.steps[0].edge_props,
        Some(Filter::FieldEquals { field, .. }) if field == "weight"
    ));
    assert!(matches!(
        &p.steps[1].edge_props,
        Some(Filter::FieldEquals { field, .. }) if field == "weight"
    ));
}

#[test]
fn closure_is_case_insensitive_keyword() {
    let plan = parse("closure (a:1) -> :likes").unwrap();
    assert!(matches!(&plan[0], Statement::Closure(_)));
}

#[test]
fn select_knn_order_limit() {
    let plan = parse(
        "SELECT * FROM note WHERE vector::similarity(embedding, [0.5, -1.0, 2.5]) AND k = 5 ORDER BY ::similarity LIMIT 10",
    )
    .unwrap();
    let Statement::Select(s) = &plan[0] else {
        panic!("expected Select, got {:?}", plan[0]);
    };
    assert_eq!(
        s,
        &Select {
            table: "note".into(),
            knn: Some(Knn {
                query: vec![0.5, -1.0, 2.5],
                k: 5,
            }),
            filter: None,
            as_of: None,
            order: Some(Order::Similarity),
            limit: Some(10),
            fields: None,
            offset: None,
            aggregate: None,
        }
    );
}

#[test]
fn select_field_equals() {
    let plan = parse(r#"SELECT text FROM note WHERE status = "done""#).unwrap();
    let Statement::Select(s) = &plan[0] else {
        panic!("expected Select");
    };
    assert_eq!(s.table, "note");
    assert_eq!(
        s.filter,
        Some(Filter::FieldEquals {
            field: "status".into(),
            value: Value::Str("done".into()),
        })
    );
    assert!(s.knn.is_none());
    assert!(s.order.is_none());
    assert!(s.limit.is_none());
}

#[test]
fn select_embedding_is_not_null() {
    let plan = parse("SELECT * FROM doc WHERE embedding IS NOT NULL").unwrap();
    let Statement::Select(s) = &plan[0] else {
        panic!("expected Select");
    };
    assert_eq!(s.filter, Some(Filter::HasEmbedding));
}

#[test]
fn select_bm25_filter() {
    let plan = parse(r#"SELECT * FROM doc WHERE ::bm25(text, "acme corp")"#).unwrap();
    let Statement::Select(s) = &plan[0] else {
        panic!("expected Select");
    };
    assert_eq!(
        s.filter,
        Some(Filter::Bm25 {
            field: "text".into(),
            query: "acme corp".into(),
            k: None,
        })
    );
    assert!(s.knn.is_none());
}

#[test]
fn select_bm25_with_k_cap() {
    let plan = parse(r#"SELECT * FROM doc WHERE ::bm25(text, "acme") AND k = 5"#).unwrap();
    let Statement::Select(s) = &plan[0] else {
        panic!("expected Select");
    };
    assert_eq!(
        s.filter,
        Some(Filter::Bm25 {
            field: "text".into(),
            query: "acme".into(),
            k: Some(5),
        })
    );
}

#[test]
fn select_bm25_requires_string_query() {
    let err = parse("SELECT * FROM doc WHERE ::bm25(text, 42)").unwrap_err();
    assert!(err.to_string().contains("must be a string"), "got: {err}");
}

#[test]
fn select_unknown_double_colon_operator_errors() {
    let err = parse("SELECT * FROM doc WHERE ::bogus(text, \"q\")").unwrap_err();
    assert!(err.to_string().contains("::bogus"), "got: {err}");
}

#[test]
fn select_as_of_parses_timestamp() {
    let plan = parse("SELECT * FROM doc AS OF 42").unwrap();
    let Statement::Select(s) = &plan[0] else {
        panic!("expected Select");
    };
    assert_eq!(s.as_of, Some(42));
}

#[test]
fn select_as_of_is_case_insensitive() {
    let plan = parse("SELECT * FROM doc as of 7").unwrap();
    let Statement::Select(s) = &plan[0] else {
        panic!("expected Select");
    };
    assert_eq!(s.as_of, Some(7));
}

#[test]
fn select_without_as_of_has_none() {
    let plan = parse("SELECT * FROM doc LIMIT 3").unwrap();
    let Statement::Select(s) = &plan[0] else {
        panic!("expected Select");
    };
    assert_eq!(s.as_of, None);
}

#[test]
fn select_as_of_requires_integer() {
    let err = parse("SELECT * FROM doc AS OF foo").unwrap_err();
    assert!(err.to_string().contains("AS OF timestamp"), "got: {err}");
}

#[test]
fn memory_parses_name() {
    let plan = parse("MEMORY core").unwrap();
    let Statement::Memory { name } = &plan[0] else {
        panic!("expected Memory, got {:?}", plan[0]);
    };
    assert_eq!(name, "core");
}

#[test]
fn memory_is_case_insensitive_keyword() {
    let plan = parse("memory archival").unwrap();
    assert!(matches!(&plan[0], Statement::Memory { .. }));
}

#[test]
fn memory_requires_name() {
    let err = parse("MEMORY").unwrap_err();
    assert!(err.to_string().contains("memory name"), "got: {err}");
}

#[test]
fn memory_switches_context_within_a_plan() {
    // MEMORY scoping is plan-level: statements after it target the named
    // memory, statements before it target the root store.
    let plan = parse(
        "CREATE TABLE t; INSERT INTO t:1 { \"x\": 1 }; MEMORY core; INSERT INTO t:1 { \"x\": 2 };",
    )
    .unwrap();
    assert_eq!(plan.len(), 4);
    assert!(matches!(&plan[2], Statement::Memory { name } if name == "core"));
}

#[test]
fn select_hybrid_bm25_then_knn() {
    // Hybrid retrieval: lexical + vector in one WHERE (bm25 first).
    let plan = parse(
        r#"SELECT * FROM doc WHERE ::bm25(text, "acme") AND vector::similarity(embedding, [0.5, -1.0, 2.5]) AND k = 5"#,
    )
    .unwrap();
    let Statement::Select(s) = &plan[0] else {
        panic!("expected Select");
    };
    assert_eq!(
        s.filter,
        Some(Filter::Bm25 {
            field: "text".into(),
            query: "acme".into(),
            k: None,
        })
    );
    assert_eq!(
        s.knn,
        Some(Knn {
            query: vec![0.5, -1.0, 2.5],
            k: 5,
        })
    );
}

#[test]
fn select_hybrid_knn_then_bm25() {
    // Hybrid retrieval: vector first, then lexical.
    let plan = parse(
        r#"SELECT * FROM doc WHERE vector::similarity(embedding, [0.5, -1.0]) AND k = 3 AND ::bm25(text, "acme") AND k = 7"#,
    )
    .unwrap();
    let Statement::Select(s) = &plan[0] else {
        panic!("expected Select");
    };
    assert_eq!(
        s.knn,
        Some(Knn {
            query: vec![0.5, -1.0],
            k: 3,
        })
    );
    assert_eq!(
        s.filter,
        Some(Filter::Bm25 {
            field: "text".into(),
            query: "acme".into(),
            k: Some(7),
        })
    );
}

#[test]
fn select_bm25_alone_keeps_no_knn() {
    // The bm25 `AND k = N` cap must NOT be misparsed as a knn clause.
    let plan = parse(r#"SELECT * FROM doc WHERE ::bm25(text, "acme") AND k = 5"#).unwrap();
    let Statement::Select(s) = &plan[0] else {
        panic!("expected Select");
    };
    assert!(
        s.knn.is_none(),
        "k cap belongs to bm25, got knn: {:?}",
        s.knn
    );
}

#[test]
fn select_hybrid_requires_bm25_after_knn_and() {
    let err = parse(
        r#"SELECT * FROM doc WHERE vector::similarity(embedding, [0.5]) AND k = 3 AND ::bogus(text, "q")"#,
    )
    .unwrap_err();
    assert!(err.to_string().contains("::bogus"), "got: {err}");
}

#[test]
fn forget_record() {
    let plan = parse("FORGET person:42").unwrap();
    assert_eq!(
        plan,
        vec![Statement::Forget {
            id: rid("person:42")
        }]
    );
}

#[test]
fn keywords_are_case_insensitive() {
    let plan = parse(
        "cReAtE TaBlE doc VECTOR<F32, 128>\n\
         iNsErT iNtO doc:1 { title: \"hi\" } eMbEd [0.1]\n\
         rElAtE (doc:1) -> :refs -> (doc:2) SeT weight = 1\n\
         sElEcT * fRoM doc WhErE embedding iS nOt nUlL oRdEr By ::RECENCY LiMiT 3\n\
         fOrGeT doc:2",
    )
    .unwrap();
    assert_eq!(plan.len(), 5);
    assert!(matches!(
        plan[0],
        Statement::CreateTable {
            vector_dim: Some(128),
            ..
        }
    ));
    let Statement::Select(s) = &plan[3] else {
        panic!("expected Select at index 3");
    };
    assert_eq!(s.filter, Some(Filter::HasEmbedding));
    assert_eq!(s.order, Some(Order::Recency));
    assert_eq!(s.limit, Some(3));
}

#[test]
fn order_by_variants() {
    for (kw, expected) in [
        ("similarity", Order::Similarity),
        ("salience", Order::Salience),
        ("score", Order::Score),
        ("recency", Order::Recency),
    ] {
        let plan = parse(&format!("SELECT * FROM t ORDER BY ::{kw}")).unwrap();
        let Statement::Select(s) = &plan[0] else {
            panic!("expected Select");
        };
        assert_eq!(s.order, Some(expected), "ORDER BY {kw}");
    }
}

#[test]
fn salience_weighted_order_parses_and_validates() {
    let plan = parse("SELECT * FROM t ORDER BY ::salience(0.5, 0, 0.25, 0.25) LIMIT 3").unwrap();
    let Statement::Select(s) = &plan[0] else {
        panic!("expected Select");
    };
    assert_eq!(
        s.order,
        Some(Order::SalienceWeighted([0.5, 0.0, 0.25, 0.25]))
    );
    assert_eq!(s.limit, Some(3));

    // integer literals count as weights too
    let plan = parse("SELECT * FROM t ORDER BY ::salience(1, 0, 0, 0)").unwrap();
    let Statement::Select(s) = &plan[0] else {
        panic!("expected Select");
    };
    assert_eq!(s.order, Some(Order::SalienceWeighted([1.0, 0.0, 0.0, 0.0])));

    // bare ::salience keeps the engine-default variant
    let plan = parse("SELECT * FROM t ORDER BY ::salience").unwrap();
    let Statement::Select(s) = &plan[0] else {
        panic!("expected Select");
    };
    assert_eq!(s.order, Some(Order::Salience));

    // arity and shape errors: exactly four numbers, parens closed
    for bad in [
        "SELECT * FROM t ORDER BY ::salience()",
        "SELECT * FROM t ORDER BY ::salience(0.5, 0.2)",
        "SELECT * FROM t ORDER BY ::salience(0.5, 0.2, 0.1, 0.2, 0.1)",
        "SELECT * FROM t ORDER BY ::salience(0.5, 0.2, 0.1, 0.2",
        "SELECT * FROM t ORDER BY ::salience(a, 0, 0, 0)",
        "SELECT * FROM t ORDER BY ::salience(0.5, 0.2, 0.1,)",
    ] {
        assert!(parse(bad).is_err(), "expected parse error: {bad}");
    }
}

#[test]
fn comparison_in_between_filters_parse() {
    let cases: &[(&str, Filter)] = &[
        (
            "seq < 100",
            Filter::FieldCmp {
                field: "seq".into(),
                op: CmpOp::Lt,
                value: Value::Int(100),
            },
        ),
        (
            "conf <= 0.8",
            Filter::FieldCmp {
                field: "conf".into(),
                op: CmpOp::Le,
                value: Value::Float(0.8),
            },
        ),
        (
            "value > 1",
            Filter::FieldCmp {
                field: "value".into(),
                op: CmpOp::Gt,
                value: Value::Int(1),
            },
        ),
        (
            "weight >= 0.5",
            Filter::FieldCmp {
                field: "weight".into(),
                op: CmpOp::Ge,
                value: Value::Float(0.5),
            },
        ),
        (
            "name != \"x\"",
            Filter::FieldCmp {
                field: "name".into(),
                op: CmpOp::Ne,
                value: Value::Str("x".into()),
            },
        ),
        (
            "group IN [1, 2, 3]",
            Filter::FieldIn {
                field: "group".into(),
                values: vec![Value::Int(1), Value::Int(2), Value::Int(3)],
            },
        ),
        (
            "ts BETWEEN 10 AND 20",
            Filter::FieldBetween {
                field: "ts".into(),
                lo: Value::Int(10),
                hi: Value::Int(20),
            },
        ),
        (
            "tag = \"a\"",
            Filter::FieldEquals {
                field: "tag".into(),
                value: Value::Str("a".into()),
            },
        ),
    ];
    for (pred, expected) in cases {
        let plan = parse(&format!("SELECT * FROM t WHERE {pred}"))
            .unwrap_or_else(|e| panic!("{pred}: {e}"));
        let Statement::Select(s) = &plan[0] else {
            panic!("expected Select for {pred}");
        };
        assert_eq!(s.filter.as_ref(), Some(expected), "{pred}");
    }

    // Malformed predicates are hard errors, never silently dropped.
    for bad in [
        "SELECT * FROM t WHERE seq ~ 100",
        "SELECT * FROM t WHERE seq IN 5",
        "SELECT * FROM t WHERE ts BETWEEN 10 20",
        "SELECT * FROM t WHERE ts BETWEEN 10 AND",
        "SELECT * FROM t WHERE seq ! 100",
        "SELECT * FROM t WHERE seq ! = 100",
        "SELECT * FROM t WHERE seq",
    ] {
        assert!(parse(bad).is_err(), "expected parse error: {bad}");
    }
}

#[test]
fn match_count_and_edge_prop_predicates_parse() {
    // MATCH ... COUNT (walk-count mode, issue #94).
    let plan = parse("MATCH (a:1) -> :mentions COUNT").unwrap();
    let Statement::MatchCount(p) = &plan[0] else {
        panic!("expected MatchCount");
    };
    assert_eq!(p.start, RecordId::parse("a:1").unwrap());

    // Without COUNT it stays the row-returning match.
    let plan = parse("MATCH (a:1) -> :mentions").unwrap();
    assert!(matches!(plan[0], Statement::Match(_)));

    // Edge-property filters share the full predicate grammar (issue #93).
    let plan = parse("MATCH (a:1) -> :mentions WHERE confidence >= 0.5").unwrap();
    let Statement::Match(p) = &plan[0] else {
        panic!("expected Match");
    };
    assert_eq!(
        p.steps[0].edge_props,
        Some(Filter::FieldCmp {
            field: "confidence".into(),
            op: CmpOp::Ge,
            value: Value::Float(0.5),
        })
    );
}

#[test]
fn match_and_closure_accept_as_of() {
    // `... AS OF <ts>` precedes a trailing MATCH `COUNT` (issue #92).
    let plan = parse("MATCH (a:1) -> :x AS OF 5").unwrap();
    let Statement::Match(p) = &plan[0] else {
        panic!("expected Match");
    };
    assert_eq!(p.as_of, Some(5));

    let plan = parse("MATCH (a:1) -> :x AS OF 5 COUNT").unwrap();
    let Statement::MatchCount(p) = &plan[0] else {
        panic!("expected MatchCount");
    };
    assert_eq!(p.as_of, Some(5));

    let plan = parse("CLOSURE (a:1) -> :x AS OF 3").unwrap();
    let Statement::Closure(p) = &plan[0] else {
        panic!("expected Closure");
    };
    assert_eq!(p.as_of, Some(3));

    // Bare traversals keep `None` (current state) — regression.
    let plan = parse("MATCH (a:1) -> :x").unwrap();
    let Statement::Match(p) = &plan[0] else {
        panic!("expected Match");
    };
    assert_eq!(p.as_of, None);

    for bad in [
        "MATCH (a:1) -> :x AS OF",
        "MATCH (a:1) -> :x AS OF five",
        "CLOSURE (a:1) -> :x AS 5",
        "MATCH (a:1) -> :x COUNT AS OF 5",
    ] {
        assert!(parse(bad).is_err(), "expected parse error: {bad}");
    }
}

#[test]
fn order_by_field_and_desc_parse() {
    // A bare non-operator key is a body-field sort (issue #117) …
    let plan = parse("SELECT * FROM t ORDER BY seq").unwrap();
    let Statement::Select(s) = &plan[0] else {
        panic!("expected Select");
    };
    assert_eq!(
        s.order,
        Some(Order::Field {
            key: "seq".into(),
            desc: false,
        })
    );

    // … with an optional DESC that does not swallow later clauses …
    let plan = parse("SELECT * FROM t ORDER BY seq DESC LIMIT 3").unwrap();
    let Statement::Select(s) = &plan[0] else {
        panic!("expected Select");
    };
    assert_eq!(
        s.order,
        Some(Order::Field {
            key: "seq".into(),
            desc: true,
        })
    );
    assert_eq!(s.limit, Some(3));

    // … and bare operator keys keep working (the `::` is optional, as before).
    let plan = parse("SELECT * FROM t ORDER BY recency").unwrap();
    let Statement::Select(s) = &plan[0] else {
        panic!("expected Select");
    };
    assert_eq!(s.order, Some(Order::Recency));

    // `::` commits to the operator list — a field never hides behind it.
    for bad in [
        "SELECT * FROM t ORDER BY ::seq",
        // operators have fixed directions: DESC after one is an error …
        "SELECT * FROM t ORDER BY ::recency DESC",
        "SELECT * FROM t ORDER BY salience DESC",
        // … and a second DESC is statement junk.
        "SELECT * FROM t ORDER BY seq DESC DESC",
    ] {
        assert!(parse(bad).is_err(), "expected parse error: {bad}");
    }
}

#[test]
fn history_since_parses() {
    let plan = parse("HISTORY SINCE 5").unwrap();
    assert!(matches!(plan[0], Statement::HistorySince(5)));

    let plan = parse("HISTORY SINCE 0").unwrap();
    assert!(matches!(plan[0], Statement::HistorySince(0)));

    // The shared HISTORY keyword does not disturb PRUNE HISTORY (issue #95).
    let plan = parse("PRUNE HISTORY").unwrap();
    assert!(matches!(plan[0], Statement::PruneHistory));

    for bad in [
        "HISTORY",
        "HISTORY SINCE",
        "HISTORY BEFORE 5",
        "HISTORY SINCE five",
        "HISTORY SINCE 5 AGO",
    ] {
        assert!(parse(bad).is_err(), "expected parse error: {bad}");
    }
}

#[test]
fn prune_history_parses() {
    let plan = parse("PRUNE HISTORY").unwrap();
    assert!(matches!(plan[0], Statement::PruneHistory));

    for bad in [
        "PRUNE",
        "PRUNE HISTORY NOW",
        "PRUNE STORE",
        "PRUNE HISTORYARY",
    ] {
        assert!(parse(bad).is_err(), "expected parse error: {bad}");
    }
}

#[test]
fn count_star_and_offset_parse() {
    let plan = parse("SELECT COUNT(*) FROM ledger WHERE seq >= 10").unwrap();
    let Statement::Select(s) = &plan[0] else {
        panic!("expected Select");
    };
    assert_eq!(s.aggregate, Some(Aggregate::CountStar));
    assert_eq!(s.fields, None);
    assert_eq!(
        s.filter,
        Some(Filter::FieldCmp {
            field: "seq".into(),
            op: CmpOp::Ge,
            value: Value::Int(10),
        })
    );

    // LIMIT n OFFSET m in one clause …
    let plan = parse("SELECT * FROM t LIMIT 10 OFFSET 20").unwrap();
    let Statement::Select(s) = &plan[0] else {
        panic!("expected Select");
    };
    assert_eq!((s.limit, s.offset), (Some(10), Some(20)));

    // … or OFFSET on its own.
    let plan = parse("SELECT * FROM t OFFSET 5").unwrap();
    let Statement::Select(s) = &plan[0] else {
        panic!("expected Select");
    };
    assert_eq!((s.limit, s.offset), (None, Some(5)));

    // A field literally named `count` is still a projection.
    let plan = parse("SELECT count, name FROM t").unwrap();
    let Statement::Select(s) = &plan[0] else {
        panic!("expected Select");
    };
    assert_eq!(
        s.fields,
        Some(vec!["count".to_string(), "name".to_string()])
    );
    assert_eq!(s.aggregate, None);

    // COUNT arity: `*` is the only argument.
    for bad in ["SELECT COUNT(x) FROM t", "SELECT COUNT() FROM t"] {
        assert!(parse(bad).is_err(), "expected parse error: {bad}");
    }
}

#[test]
fn empty_input_parses_to_empty_plan() {
    assert!(parse("").unwrap().is_empty());
    assert!(parse("   \n\t ").unwrap().is_empty());
}

#[test]
fn parse_statement_rejects_trailing_input() {
    let err = parse_statement("CREATE TABLE t SELECT * FROM t").unwrap_err();
    assert!(matches!(err, NqlError::Parse { .. }));
    assert!(err.to_string().contains("trailing"));
}

#[test]
fn error_unknown_statement_keyword() {
    let err = parse("DROP TABLE t").unwrap_err();
    assert!(err.to_string().contains("statement keyword"), "{err}");
    assert!(err.line() >= 1 && err.col() >= 1);
}

#[test]
fn error_missing_table_after_create() {
    let err = parse("CREATE TABLE").unwrap_err();
    assert!(err.to_string().contains("table name"), "{err}");
}

#[test]
fn error_missing_from_in_select() {
    let err = parse("SELECT * t").unwrap_err();
    assert!(err.to_string().contains("FROM"), "{err}");
}

#[test]
fn error_bad_vector_dim() {
    // Zero dimension is rejected.
    let err = parse("CREATE TABLE t VECTOR<f32, 0>").unwrap_err();
    assert!(err.to_string().contains("dimension"), "{err}");
    // Missing `>` is a syntax error, not a panic.
    let err = parse("CREATE TABLE t VECTOR<f32, 8").unwrap_err();
    assert!(err.to_string().contains("closing VECTOR"), "{err}");
}

#[test]
fn error_knn_k_must_be_positive() {
    let err =
        parse("SELECT * FROM t WHERE vector::similarity(embedding, [1.0]) AND k = 0").unwrap_err();
    assert!(err.to_string().contains("k must be positive"), "{err}");
}

#[test]
fn error_unterminated_string() {
    let err = parse(r#"INSERT INTO t:1 { name: "oops }"#).unwrap_err();
    assert!(matches!(err, NqlError::Lex { .. }), "{err:?}");
    assert!(err.to_string().contains("unterminated"), "{err}");
}

#[test]
fn error_bad_vector_literal() {
    let err = parse("SELECT * FROM t WHERE vector::similarity(embedding, [0.1, \"x\"]) AND k = 1")
        .unwrap_err();
    assert!(err.to_string().contains("vector literal"), "{err}");
}

#[test]
fn error_missing_brace_in_body() {
    let err = parse("INSERT INTO t:1 { a: 1").unwrap_err();
    assert!(err.to_string().contains("closing record body"), "{err}");
}

#[test]
fn error_bad_order_key() {
    let err = parse("SELECT * FROM t ORDER BY ::random").unwrap_err();
    assert!(err.to_string().contains("ORDER BY"), "{err}");
}

#[test]
fn error_set_weight_must_be_number() {
    let err = parse(r#"RELATE (a:1) -> :e -> (b:2) SET weight = "heavy""#).unwrap_err();
    assert!(err.to_string().contains("weight"), "{err}");
}

#[test]
fn single_quoted_strings_and_escapes() {
    let plan =
        parse(r#"INSERT INTO t:1 { a: 'it\'s', b: "tab\there", c: "back\\slash" }"#).unwrap();
    let Statement::Insert(rec) = &plan[0] else {
        panic!("expected Insert");
    };
    assert_eq!(rec.body.get("a"), Some(&Value::Str("it's".into())));
    assert_eq!(rec.body.get("b"), Some(&Value::Str("tab\there".into())));
    assert_eq!(rec.body.get("c"), Some(&Value::Str("back\\slash".into())));
}

#[test]
fn plan_is_ordered_sequence_of_statements() {
    let plan = parse("CREATE TABLE t\nINSERT INTO t:1 {}\nSELECT * FROM t\nFORGET t:1").unwrap();
    assert_eq!(plan.len(), 4);
    assert!(matches!(plan[0], Statement::CreateTable { .. }));
    assert!(matches!(plan[1], Statement::Insert(_)));
    assert!(matches!(plan[2], Statement::Select(_)));
    assert!(matches!(plan[3], Statement::Forget { .. }));
}

#[test]
fn select_clauses_any_order() {
    let plan = parse("SELECT * FROM t LIMIT 2 WHERE status = 1 ORDER BY ::score").unwrap();
    let Statement::Select(s) = &plan[0] else {
        panic!("expected Select");
    };
    assert_eq!(s.limit, Some(2));
    assert_eq!(
        s.filter,
        Some(Filter::FieldEquals {
            field: "status".into(),
            value: Value::Int(1),
        })
    );
    assert_eq!(s.order, Some(Order::Score));
}

// --- comments (spec §1: `--` to end of line, `/* */`) — issue #86 ----------

#[test]
fn line_comments_are_skipped() {
    let plan = parse(
        r#"
        -- leading comment
        CREATE TABLE note;   -- trailing comment
        -- interstitial
        SELECT * FROM note;  -- another trailing
        "#,
    )
    .expect("`--` comments must parse per spec §1");
    assert_eq!(plan.len(), 2);
    assert!(matches!(plan[0], Statement::CreateTable { .. }));
    assert!(matches!(plan[1], Statement::Select(_)));
}

#[test]
fn line_comment_at_eof_without_newline_is_skipped() {
    let plan = parse("CREATE TABLE note; -- dangling comment").unwrap();
    assert_eq!(plan.len(), 1);
}

#[test]
fn block_comments_are_skipped() {
    let plan =
        parse("/* leading */ CREATE TABLE note; /* multi\nline\ncomment */ SELECT * FROM note;")
            .expect("`/* */` comments must parse per spec §1");
    assert_eq!(plan.len(), 2);
    assert!(matches!(plan[0], Statement::CreateTable { .. }));
    assert!(matches!(plan[1], Statement::Select(_)));
}

#[test]
fn unterminated_block_comment_is_an_error() {
    let err = parse("CREATE TABLE note; /* never closed").unwrap_err();
    assert!(
        err.to_string().contains("unterminated block comment"),
        "got: {err}"
    );
}

#[test]
fn comments_do_not_shadow_arrows_or_negative_numbers() {
    // `--` must not swallow `->` / `-1`: the comment only starts when the
    // `--` sequence reaches skip position.
    let plan = parse(
        r#"
        CREATE TABLE note VECTOR<f32, 1>; -- comment
        INSERT INTO note:1 { "text": "x", "delta": -1 } EMBED [0.5]; -- comment
        RELATE (note:1) -> :ref -> (note:2); -- comment
        "#,
    )
    .unwrap();
    assert_eq!(plan.len(), 3);
    let Statement::Insert(rec) = &plan[1] else {
        panic!("expected Insert");
    };
    assert_eq!(rec.body.get("delta"), Some(&Value::Int(-1)));
    assert!(matches!(plan[2], Statement::Relate(_)));
}

// --- field projection (spec §2.3 step 8) — issue #91 ------------------------

#[test]
fn select_projection_is_carried_into_ir() {
    let plan = parse("SELECT text, group FROM t").unwrap();
    let Statement::Select(s) = &plan[0] else {
        panic!("expected Select");
    };
    assert_eq!(
        s.fields,
        Some(vec!["text".to_string(), "group".to_string()]),
        "explicit field list must reach the IR"
    );
}

#[test]
fn select_star_carries_no_projection() {
    let plan = parse("SELECT * FROM t").unwrap();
    let Statement::Select(s) = &plan[0] else {
        panic!("expected Select");
    };
    assert_eq!(s.fields, None, "`*` means full records");
}

#[test]
fn select_projection_tolerates_star_mixed_syntax() {
    // `SELECT a, b` vs `SELECT *` both parse; only the list form projects.
    for (src, expect) in [
        ("SELECT * FROM t", None),
        ("SELECT a FROM t", Some(vec!["a".to_string()])),
    ] {
        let plan = parse(src).unwrap();
        let Statement::Select(s) = &plan[0] else {
            panic!("expected Select");
        };
        assert_eq!(s.fields, expect, "src: {src}");
    }
}
