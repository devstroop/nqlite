//! Unit tests for the analyzer: table-declaration rules, vector-dimension
//! contract, SELECT enrichment, record-id shape checks, and error messages.

use crate::analyzer::{AnalysisError, Analyzer};
use nql_ir::{
    Id, Knn, MatchDirection, MatchPath, MatchStep, Order, Record, RecordId, Select, Statement,
};
use std::collections::BTreeMap;

fn create(table: &str, dim: Option<usize>) -> Statement {
    Statement::CreateTable {
        table: table.into(),
        vector_dim: dim,
    }
}

fn insert(table: &str, id: u64, embedding: Option<Vec<f32>>) -> Statement {
    Statement::Insert(Record {
        id: RecordId::new(table, Id::Num(id)),
        body: BTreeMap::new(),
        embedding,
        created_at: 0,
    })
}

fn select(table: &str, knn: Option<Knn>, order: Option<Order>) -> Statement {
    Statement::Select(Select {
        table: table.into(),
        knn,
        filter: None,
        order,
        limit: None,
    })
}

fn knn(dim: usize) -> Knn {
    Knn {
        query: vec![1.0; dim],
        k: 5,
    }
}

/// Extract the `Select` from an analyzed statement.
fn as_select(stmt: &Statement) -> &Select {
    match stmt {
        Statement::Select(s) => s,
        other => panic!("expected Select, got {other:?}"),
    }
}

#[test]
fn valid_plan_passes_and_is_enriched() {
    let plan = vec![
        create("person", Some(3)),
        insert("person", 1, Some(vec![1.0, 2.0, 3.0])),
        select("person", Some(knn(3)), None),
    ];
    let out = Analyzer::analyze(&plan).expect("valid plan analyzes");
    assert_eq!(out.len(), 3, "statement count preserved");
    // The kNN select without ORDER BY gets Order::Similarity auto-filled.
    assert_eq!(as_select(&out[2]).order, Some(Order::Similarity));
}

#[test]
fn select_before_create_errors() {
    let plan = vec![select("person", None, None)];
    let err = Analyzer::analyze(&plan).expect_err("SELECT before CREATE must fail");
    assert!(matches!(err, AnalysisError::UnknownTableForSelect { .. }));
}

#[test]
fn insert_without_create_errors() {
    let plan = vec![insert("person", 1, None)];
    let err = Analyzer::analyze(&plan).expect_err("INSERT without CREATE must fail");
    assert!(matches!(err, AnalysisError::UnknownTableForInsert { .. }));
}

#[test]
fn insert_embedding_dim_mismatch_errors() {
    let plan = vec![
        create("person", Some(3)),
        insert("person", 1, Some(vec![1.0, 2.0])),
    ];
    let err = Analyzer::analyze(&plan).expect_err("dim mismatch must fail");
    assert!(matches!(
        err,
        AnalysisError::EmbeddingDimMismatch {
            table,
            expected: 3,
            actual: 2,
        } if table == "person"
    ));
}

#[test]
fn insert_without_embedding_is_fine_even_with_dim() {
    let plan = vec![create("person", Some(3)), insert("person", 1, None)];
    assert!(Analyzer::analyze(&plan).is_ok(), "embedding is optional");
}

#[test]
fn knn_without_order_gets_similarity() {
    let plan = vec![create("t", None), select("t", Some(knn(2)), None)];
    let out = Analyzer::analyze(&plan).expect("knn select analyzes");
    assert_eq!(as_select(&out[1]).order, Some(Order::Similarity));
    assert_eq!(as_select(&out[1]).knn, Some(knn(2)), "knn untouched");
}

#[test]
fn knn_with_similarity_order_ok() {
    let plan = vec![
        create("t", None),
        select("t", Some(knn(2)), Some(Order::Similarity)),
    ];
    let out = Analyzer::analyze(&plan).expect("explicit similarity + knn is valid");
    assert_eq!(as_select(&out[1]).order, Some(Order::Similarity));
}

#[test]
fn similarity_without_knn_errors() {
    let plan = vec![
        create("t", None),
        select("t", None, Some(Order::Similarity)),
    ];
    let err = Analyzer::analyze(&plan).expect_err("similarity without knn must fail");
    assert!(matches!(err, AnalysisError::SimilarityWithoutKnn));
}

#[test]
fn builtin_tables_are_known() {
    for name in ["global", "meta"] {
        let plan = vec![select(name, None, None)];
        assert!(
            Analyzer::analyze(&plan).is_ok(),
            "built-in table `{name}` should be selectable without CREATE"
        );
    }
}

#[test]
fn statement_order_matters() {
    // CREATE then SELECT: the declaration precedes the use.
    let ok_plan = vec![create("t", None), select("t", None, None)];
    assert!(Analyzer::analyze(&ok_plan).is_ok());

    // SELECT then CREATE: the use precedes the declaration.
    let bad_plan = vec![select("t", None, None), create("t", None)];
    let err = Analyzer::analyze(&bad_plan).expect_err("use before declaration must fail");
    assert!(matches!(err, AnalysisError::UnknownTableForSelect { .. }));
}

#[test]
fn error_has_human_message() {
    let plan = vec![select("ghosts", None, None)];
    let err = Analyzer::analyze(&plan).expect_err("unknown table must fail");
    let msg = err.to_string();
    assert!(!msg.is_empty());
    assert!(
        msg.contains("ghosts"),
        "message should name the offending table: {msg}"
    );
    assert!(msg.contains("CREATE TABLE"));
}

#[test]
fn analyze_statement_rejects_standalone_select() {
    let stmt = select("person", None, None);
    let err = Analyzer::analyze_statement(&stmt).expect_err("standalone SELECT must fail");
    assert!(matches!(err, AnalysisError::UnknownTableForSelect { .. }));
}

#[test]
fn analyze_statement_accepts_standalone_builtin_select() {
    let stmt = select("meta", None, None);
    let out = Analyzer::analyze_statement(&stmt).expect("built-in select is standalone-valid");
    assert_eq!(as_select(&out).table, "meta");
}

#[test]
fn analyze_statement_enriches_knn_select() {
    let stmt = select("meta", Some(knn(2)), None);
    let out = Analyzer::analyze_statement(&stmt).expect("built-in kNN select analyzes");
    assert_eq!(as_select(&out).order, Some(Order::Similarity));
}

#[test]
fn analyze_is_pure_input_untouched() {
    let plan = vec![create("t", None), select("t", Some(knn(2)), None)];
    let before = plan.clone();
    let out = Analyzer::analyze(&plan).expect("analyzes");
    assert_eq!(plan, before, "input plan must not be mutated");
    assert_eq!(as_select(&out[1]).order, Some(Order::Similarity));
}

#[test]
fn forget_with_empty_string_id_errors() {
    let plan = vec![Statement::Forget {
        id: RecordId::new("t", Id::Str(String::new())),
    }];
    let err = Analyzer::analyze(&plan).expect_err("empty id string must fail");
    assert!(matches!(err, AnalysisError::EmptyId { .. }));
}

#[test]
fn match_analyzes_without_table_declaration() {
    // MATCH is graph traversal, not a table read: it validates the start
    // record's id shape but requires no CREATE TABLE and no built-in table.
    let stmt = Statement::Match(MatchPath {
        start: RecordId::new("note", Id::Num(1)),
        steps: vec![MatchStep {
            direction: MatchDirection::Out,
            name: "mentions".into(),
        }],
    });
    let out = Analyzer::analyze_statement(&stmt).expect("standalone MATCH analyzes");
    assert_eq!(out, stmt, "MATCH passes through unchanged");
}

#[test]
fn match_with_empty_id_string_errors() {
    let stmt = Statement::Match(MatchPath {
        start: RecordId::new("note", Id::Str(String::new())),
        steps: vec![MatchStep {
            direction: MatchDirection::Out,
            name: "mentions".into(),
        }],
    });
    let err = Analyzer::analyze_statement(&stmt).expect_err("empty id string must fail");
    assert!(matches!(err, AnalysisError::EmptyId { .. }));
}
