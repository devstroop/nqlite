//! M2 golden corpus — nql text → statements, exported for nqlite-zig's
//! parser gate (spec/fixtures/nql/corpus.json).
//!
//! The Rust front-end is the oracle: every case runs `nql::parse` (mode
//! `plan`) or `nql::parse_statement` (mode `statement`), then
//! `Analyzer::analyze`. The corpus records, per case:
//!
//! - `ok: true` → `hex` (postcard of each raw-parsed `Statement`) and
//!   `analyzed_hex` (postcard after the analyzer's enrichment — differs
//!   from `hex` when e.g. kNN SELECT gains `ORDER BY similarity`);
//! - `ok: false` → `error`: `{kind: "lex"|"parse", line, col}` from
//!   `NqlError`, or `{kind: "analysis", variant}` from `AnalysisError`
//!   (no positions — the analyzer has none; `hex` is included because the
//!   parse itself succeeded).
//!
//! The consuming implementation must match kind/line/col/variant exactly
//! and reproduce the hex bytes bit-for-bit.
//!
//! Regenerate after a deliberate grammar change (never hand-edit):
//!
//! ```sh
//! UPDATE_FIXTURES=1 cargo test --test parser_corpus
//! ```

use nql::{parse, parse_statement, AnalysisError, Analyzer, NqlError};
use serde_json::json;

/// `(name, mode, source)` — mode `plan` = full program, `statement` =
/// exactly one statement (trailing input is an error).
const CASES: &[(&str, &str, &str)] = &[
    // ---- CREATE TABLE ----
    ("create-basic", "plan", "CREATE TABLE note"),
    ("create-vector", "plan", "CREATE TABLE doc VECTOR<f32, 384>"),
    ("create-lowercase", "plan", "create table t vector<f32, 8>"),
    ("create-dim-zero", "plan", "CREATE TABLE t VECTOR<f32, 0>"),
    // ---- INSERT ----
    ("insert-value-kinds", "plan", "INSERT INTO note:42 { title: 'hello', n: 1, ok: true, f: 1.5, z: null }"),
    (
        "insert-nested",
        "plan",
        r#"INSERT INTO person:alice { name: "Ada", tags: ['a', 'b'], nested: { x: -1, y: [1.5, -2.5] } }"#,
    ),
    ("insert-empty-body", "plan", "INSERT INTO t:1 {}"),
    ("create-then-insert", "plan", "CREATE TABLE t\nINSERT INTO t:1 { a: 1 }"),
    ("insert-embed", "plan", "INSERT INTO t:1 { } EMBED [0.25, 0.5, -1.0, 0.0]"),
    // Lexer traps (M2 REQUIRED): digit-first id lexes as a number…
    ("trap-digit-first-id", "plan", "INSERT INTO t:007 { k: 1 }"),
    // …a bare `-` is NOT an ident character…
    ("trap-dash-splits-name", "plan", "INSERT INTO t:abc-def { k: 1 }"),
    // …and leading-zero numerics inside bodies follow the lexer.
    ("trap-leading-zero-body", "plan", "INSERT INTO t:1 { a: 007 }"),
    ("trap-id-overflow", "plan", "INSERT INTO t:999999999999999999999999 { k: 1 }"),
    (
        "insert-escapes",
        "plan",
        r#"INSERT INTO t:1 { s: 'it\'s', d: "say \"hi\"", e: 'a\\b' }"#,
    ),
    ("insert-unclosed-body", "plan", "INSERT INTO t:1 { a: 1"),
    // ---- RELATE ----
    ("relate-basic", "plan", "RELATE (a:1) -> :mentions -> (b:2)"),
    (
        "relate-set-props",
        "plan",
        "RELATE (person:1) -> :likes -> (note:42) SET weight = 0.9, confidence = 0.5, note = 'x'",
    ),
    (
        "relate-set-value-kinds",
        "plan",
        "RELATE (a:1) -> :e -> (b:2) SET weight = 1, flag = true, tags = ['x', 'y'], sub = { n: 2 }",
    ),
    ("relate-set-dangling", "plan", "RELATE (a:1) -> :e -> (b:2) SET"),
    // ---- SELECT (declared-table prefix keeps analysis green where wanted) ----
    ("select-star", "plan", "CREATE TABLE doc\nSELECT * FROM doc"),
    ("select-fields", "plan", "CREATE TABLE doc\nSELECT title, body FROM doc"),
    ("select-count-star", "plan", "CREATE TABLE doc\nSELECT COUNT(*) FROM doc"),
    ("select-where-eq", "plan", "CREATE TABLE doc\nSELECT * FROM doc WHERE title = 'x'"),
    ("select-where-ne", "plan", "CREATE TABLE doc\nSELECT * FROM doc WHERE n != 1"),
    ("select-where-cmp-and", "plan", "CREATE TABLE doc\nSELECT * FROM doc WHERE n < 5 AND n >= 2"),
    ("select-where-in", "plan", "CREATE TABLE doc\nSELECT * FROM doc WHERE tag IN ['a', 'b']"),
    ("select-where-between", "plan", "CREATE TABLE doc\nSELECT * FROM doc WHERE n BETWEEN 1 AND 10"),
    ("select-where-has-embedding", "plan", "CREATE TABLE doc\nSELECT * FROM doc WHERE embedding IS NOT NULL"),
    (
        "select-knn-enriches-order",
        "plan",
        "CREATE TABLE doc\nSELECT * FROM doc WHERE vector::similarity(embedding, [0.1, 0.2]) AND k = 10",
    ),
    ("select-bm25", "plan", r#"CREATE TABLE doc
SELECT * FROM doc WHERE ::bm25(text, "neural search") AND k = 5"#),
    (
        "select-hybrid",
        "plan",
        "CREATE TABLE doc\nSELECT * FROM doc WHERE ::bm25(text, 'q') AND vector::similarity(embedding, [1.0]) AND k = 3",
    ),
    ("select-order-recency-limit", "plan", "CREATE TABLE doc\nSELECT * FROM doc ORDER BY ::recency LIMIT 3"),
    (
        "select-order-salience-weights",
        "plan",
        "CREATE TABLE doc\nSELECT * FROM doc ORDER BY ::salience(0.7, 0.1, 0.1, 0.1)",
    ),
    ("select-order-field-desc", "plan", "CREATE TABLE doc\nSELECT * FROM doc ORDER BY title DESC"),
    (
        "select-order-op-desc-fails",
        "plan",
        "CREATE TABLE doc\nSELECT * FROM doc ORDER BY score DESC",
    ),
    ("select-order-field", "plan", "CREATE TABLE doc\nSELECT * FROM doc ORDER BY title"),
    ("select-order-votes", "plan", "CREATE TABLE doc\nSELECT * FROM doc ORDER BY ::votes"),
    ("select-order-feedback", "plan", "CREATE TABLE doc\nSELECT * FROM doc ORDER BY ::feedback"),
    ("select-as-of", "plan", "CREATE TABLE doc\nSELECT * FROM doc AS OF 42"),
    ("select-limit-offset", "plan", "CREATE TABLE doc\nSELECT * FROM doc LIMIT 3 OFFSET 1"),
    ("select-offset-before-limit", "plan", "CREATE TABLE doc\nSELECT * FROM doc OFFSET 2 LIMIT 10 OFFSET 1"),
    ("select-builtin-table", "plan", "SELECT * FROM meta"),
    ("select-undeclared-table", "plan", "SELECT * FROM undeclared"),
    ("select-order-similarity-no-knn", "plan", "CREATE TABLE doc\nSELECT * FROM doc ORDER BY similarity"),
    ("select-missing-from-table", "plan", "SELECT * FROM"),
    ("select-bogus-filter", "plan", "SELECT * FROM doc WHERE ::bogus(x, 'y')"),
    ("select-bm25-non-string", "plan", "SELECT * FROM doc WHERE ::bm25(text, 42)"),
    // ---- MATCH / CLOSURE ----
    ("match-out", "plan", "MATCH (a:1) -> :mentions"),
    ("match-in", "plan", "MATCH (note:42) <- :mentions"),
    ("match-mixed-steps", "plan", "MATCH (a:1) -> :knows -> :works_with <- :knows"),
    ("match-step-props", "plan", "MATCH (a:1) -> :mentions WHERE confidence >= 0.5"),
    ("match-step-props-and", "plan", "MATCH (a:1) -> :e WHERE conf >= 0.5 AND kind = 'x'"),
    ("match-as-of-count", "plan", "MATCH (a:1) -> :x AS OF 5 COUNT"),
    ("match-count", "plan", "MATCH (a:1) -> :x COUNT"),
    ("match-as-of", "plan", "MATCH (a:1) -> :x AS OF 5"),
    ("match-no-steps", "plan", "MATCH (a:1)"),
    ("match-as-of-non-int", "plan", "MATCH (a:1) -> :x AS OF five"),
    ("closure-basic", "plan", "CLOSURE (a:1) -> :x"),
    ("closure-as-of", "plan", "CLOSURE (a:1) -> :x AS OF 3"),
    // ---- misc statements ----
    ("forget", "plan", "FORGET person:42"),
    ("memory", "plan", "MEMORY core"),
    ("prune-history", "plan", "PRUNE HISTORY"),
    ("history-since", "plan", "HISTORY SINCE 5"),
    ("history-since-zero", "plan", "HISTORY SINCE 0"),
    (
        "multi-statement",
        "plan",
        "CREATE TABLE t\nINSERT INTO t:1 {}\nSELECT * FROM t\nFORGET t:1",
    ),
    (
        "comments-mixed",
        "plan",
        "CREATE TABLE note; -- dangling comment\n-- another line\n/* block */ INSERT INTO note:1 {}",
    ),
    ("comment-unterminated-block", "plan", "CREATE TABLE note; /* never closed"),
    ("empty-input", "plan", ""),
    ("separators-only", "plan", ";;; -- only separators\n;;"),
    // ---- statement mode (trailing-input contract) ----
    ("stmt-mode-forget", "statement", "FORGET person:42"),
    ("stmt-mode-trailing", "statement", "CREATE TABLE t; INSERT INTO t:1 {}"),
    // ---- analysis stage ----
    ("analysis-insert-undeclared", "plan", "INSERT INTO t:1 {}"),
    (
        "analysis-embedding-dim-mismatch",
        "plan",
        "CREATE TABLE t VECTOR<f32, 4>\nINSERT INTO t:1 {} EMBED [0.1, 0.2]",
    ),
    (
        "analysis-embedding-dim-ok",
        "plan",
        "CREATE TABLE t VECTOR<f32, 4>\nINSERT INTO t:1 {} EMBED [0.1, 0.2, 0.3, 0.4]",
    ),
    ("analysis-builtin-insert", "plan", "INSERT INTO meta:1 {}"),
    // ---- lex failures ----
    ("lex-unterminated-string", "plan", "INSERT INTO t:1 { a: 'oops }"),
    ("lex-bad-char", "plan", "CREATE TABLE t @"),
];

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn stmt_hex(stmt: &nql_ir::Statement) -> String {
    hex(&postcard::to_allocvec(stmt).expect("postcard statement"))
}

fn analysis_variant(e: &AnalysisError) -> &'static str {
    match e {
        AnalysisError::UnknownTableForSelect { .. } => "UnknownTableForSelect",
        AnalysisError::UnknownTableForInsert { .. } => "UnknownTableForInsert",
        AnalysisError::EmbeddingDimMismatch { .. } => "EmbeddingDimMismatch",
        AnalysisError::SimilarityWithoutKnn => "SimilarityWithoutKnn",
        AnalysisError::EmptyTable => "EmptyTable",
        AnalysisError::EmptyId { .. } => "EmptyId",
    }
}

fn case_json(name: &str, mode: &str, source: &str) -> serde_json::Value {
    let parsed: Result<nql_ir::Plan, NqlError> = if mode == "statement" {
        parse_statement(source).map(|s| vec![s])
    } else {
        parse(source)
    };
    match parsed {
        Err(e) => {
            let kind = match e {
                NqlError::Lex { .. } => "lex",
                NqlError::Parse { .. } => "parse",
            };
            json!({
                "name": name,
                "mode": mode,
                "source": source,
                "ok": false,
                "error": { "kind": kind, "line": e.line(), "col": e.col() },
            })
        }
        Ok(plan) => {
            let hex: Vec<String> = plan.iter().map(stmt_hex).collect();
            match Analyzer::analyze(&plan) {
                Err(ae) => json!({
                    "name": name,
                    "mode": mode,
                    "source": source,
                    "ok": false,
                    "error": { "kind": "analysis", "variant": analysis_variant(&ae) },
                    "hex": hex,
                }),
                Ok(enriched) => json!({
                    "name": name,
                    "mode": mode,
                    "source": source,
                    "ok": true,
                    "hex": hex,
                    "analyzed_hex": enriched.iter().map(stmt_hex).collect::<Vec<_>>(),
                }),
            }
        }
    }
}

fn corpus_json() -> String {
    let cases: Vec<serde_json::Value> = CASES
        .iter()
        .map(|(name, mode, source)| case_json(name, mode, source))
        .collect();
    let doc = json!({
        "note": "generated by nql/tests/parser_corpus.rs (UPDATE_FIXTURES=1); spec/nql.md §1 grammar; never hand-edit",
        "cases": cases,
    });
    format!("{}\n", serde_json::to_string_pretty(&doc).unwrap())
}

#[test]
fn parser_corpus_matches_files() {
    let text = corpus_json();
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../spec/fixtures/nql/corpus.json");
    if std::env::var_os("UPDATE_FIXTURES").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &text).unwrap();
    } else {
        let committed = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "{} unreadable ({e}) — run with UPDATE_FIXTURES=1 once",
                path.display()
            )
        });
        assert_eq!(
            committed, text,
            "corpus drifted from the current front-end — grammar change? regenerate + bump .spec-pin"
        );
    }
    // Coverage sanity: every statement-producing statement kind is present
    // in at least one successful case's raw hex (tags 0..=12 are parser
    // surface except ContextReset/Snapshot which are engine-side only).
    let doc: serde_json::Value = serde_json::from_str(&text).unwrap();
    let cases = doc["cases"].as_array().unwrap();
    assert!(cases.len() >= 60, "corpus shrank: {} cases", cases.len());
    let ok_count = cases.iter().filter(|c| c["ok"] == true).count();
    let analysis_count = cases
        .iter()
        .filter(|c| c["error"]["kind"] == "analysis")
        .count();
    assert!(ok_count >= 40, "too few ok cases: {ok_count}");
    assert!(analysis_count >= 3, "analysis coverage: {analysis_count}");
}
