//! Unit tests for the M0 nql grammar slice: one test per statement kind plus
//! error paths. All assertions are on the parsed IR (`nql_ir::Plan`).

use crate::{parse, parse_statement, NqlError};
use nql_ir::{Filter, Id, Knn, MatchDirection, Order, RecordId, Select, Statement, Value};
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
