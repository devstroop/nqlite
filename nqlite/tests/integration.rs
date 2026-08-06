//! End-to-end integration: nql text -> Plan -> engine execution.
//! Proves the nql (front-end) and nqlite (engine) crates interoperate through
//! the nql-ir contract, end to end, deterministically, with zero LLM.

use nql::parse;
use nql_ir::{Store, Value};
use nqlite::Database;

const SESSION: &str = r#"
    CREATE TABLE turn VECTOR<f32, 3>;
    INSERT INTO turn:1 { "text": "hello world" } EMBED [1.0, 0.0, 0.0];
    INSERT INTO turn:2 { "text": "goodbye world" } EMBED [0.9, 0.1, 0.0];
    INSERT INTO turn:3 { "text": "unrelated" } EMBED [0.0, 0.0, 1.0];
    SELECT * FROM turn
        WHERE vector::similarity(embedding, [1.0, 0.0, 0.0]) AND k = 2
        ORDER BY ::similarity;
"#;

#[test]
fn nql_to_engine_end_to_end() {
    let mut db = Database::new(Store::default());
    let plan = parse(SESSION).expect("parse nql");
    let results = db.execute(&plan).expect("execute plan");

    assert_eq!(results.len(), 1, "one SELECT statement");
    let qr = &results[0];
    assert_eq!(qr.rows.len(), 2, "k=2");
    // nearest first: turn:1 (cosine 1.0) then turn:2 (cosine ~0.99)
    assert!(
        qr.rows[0].score >= qr.rows[1].score,
        "sorted desc by similarity"
    );
    let texts: Vec<String> = qr
        .rows
        .iter()
        .map(|r| match r.record.body.get("text") {
            Some(Value::Str(s)) => s.clone(),
            _ => String::from("?"),
        })
        .collect();
    assert_eq!(texts, ["hello world", "goodbye world"]);
}

#[test]
fn nql_forget_and_relate_roundtrip() {
    let mut db = Database::new(Store::default());
    let plan = parse(
        r#"
        CREATE TABLE note;
        INSERT INTO note:1 { "body": "alpha" };
        INSERT INTO note:2 { "body": "beta" };
        RELATE (note:1) -> :references -> (note:2) SET weight = 0.5;
        SELECT * FROM note;
        FORGET note:1;
        SELECT * FROM note;
        "#,
    )
    .expect("parse");
    let results = db.execute(&plan).expect("execute");
    assert_eq!(results.len(), 2);
    // after FORGET, only note:2 remains, and its incident edges are gone
    let after = &results[1];
    let ids: Vec<String> = after.rows.iter().map(|r| r.record.id.to_string()).collect();
    assert_eq!(ids, ["note:2"]);
    let store = db.store();
    assert!(
        store
            .edges
            .iter()
            .all(|e| e.from.to_string() != "note:1" && e.to.to_string() != "note:1"),
        "FORGET removes incident edges"
    );
}

#[test]
fn same_input_same_output_determinism() {
    let run = || {
        let mut db = Database::new(Store::default());
        let plan = parse(SESSION).expect("parse");
        let results = db.execute(&plan).expect("execute");
        format!("{:?}", results)
    };
    assert_eq!(run(), run(), "identical runs produce identical results");
}

#[test]
fn nql_match_traversal_end_to_end() {
    let mut db = Database::new(Store::default());
    let plan = parse(
        r#"
        CREATE TABLE turn;
        CREATE TABLE note;
        INSERT INTO turn:3 { "text": "the turn" };
        INSERT INTO note:1 { "body": "first" };
        INSERT INTO note:2 { "body": "second" };
        RELATE (turn:3) -> :mentions -> (note:1) SET weight = 0.9;
        RELATE (turn:3) -> :mentions -> (note:2) SET weight = 0.4;
        MATCH (turn:3) -> :mentions;
        MATCH (note:1) <- :mentions;
        "#,
    )
    .expect("parse");
    let results = db.execute(&plan).expect("execute");
    assert_eq!(results.len(), 2, "two MATCH statements");

    // Outgoing: both notes, in edge-append order, first edge's weight.
    let out = &results[0];
    let ids: Vec<String> = out.rows.iter().map(|r| r.record.id.to_string()).collect();
    assert_eq!(ids, ["note:1", "note:2"]);
    assert!((out.rows[0].score - 0.9).abs() < 1e-6);

    // Incoming: only note:1 points back at turn:3 via :mentions.
    let inc = &results[1];
    let ids: Vec<String> = inc.rows.iter().map(|r| r.record.id.to_string()).collect();
    assert_eq!(ids, ["turn:3"]);
}

#[test]
fn nql_votes_flow_into_score_order() {
    // Regression: `::score` must count `:voted` edges created through the nql
    // grammar. The parser strips the edge-name colon (`:voted` -> "voted"),
    // so score_of must match the same convention as ::votes/::feedback.
    let mut db = Database::new(Store::default());
    let plan = parse(
        r#"
        CREATE TABLE turn;
        INSERT INTO turn:3 { "text": "hello" };
        RELATE (agent:main) -> :voted -> (turn:3) SET value = 1, weight = 0.9;
        SELECT * FROM turn ORDER BY ::score;
        "#,
    )
    .expect("parse");
    let results = db.execute(&plan).expect("execute");

    // Laplace-smoothed mean: (0.9 + 1) / (1 + 2) = 0.6333..., NOT the 0.5
    // no-vote baseline that a missed edge name would produce.
    let score = results[0].rows[0].score;
    assert!(
        (score - 19.0 / 30.0).abs() < 1e-6,
        "::score must reflect the vote, got {score}"
    );
}
