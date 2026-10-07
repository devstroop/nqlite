//! M3 golden results — the engine as the execution oracle
//! (spec/fixtures/engine/results.json).
//!
//! Each case runs `nql::parse` + `Analyzer::analyze` + `Database::execute`
//! for `setup` (mutations, results ignored) then `query` (one plan, results
//! captured). Per result:
//! - `kind`: `select` | `match` | `closure` | `history`;
//! - `rows`: `[{record, score}]` — the record is serde's JSON, score is the
//!   ordering operator's f32;
//! - `rows_hex`: composite bytes `uleb128(len) + [postcard(Record) + f32 LE]…`
//!   — the byte gate other implementations re-encode and compare.
//!
//! Engine failures record `{error: <variant>}` instead.
//!
//! Regenerate after a deliberate semantics change (never hand-edit):
//!     UPDATE_FIXTURES=1 cargo test --test m3_results

use nqlite::{Database, QueryKind, QueryResult, ScoredRecord};
use serde_json::json;

/// `(name, setup, query)` — sources are NQL text (the M2 parser is part of
/// the pipeline under test on the consuming side).
const CASES: &[(&str, &[&str], &str)] = &[
    // ---- baseline ordering: BTree key order, no ORDER BY ----
    (
        "select-star-btree-order",
        &[
            "CREATE TABLE t",
            "INSERT INTO t:zebra { n: 1 }",
            "INSERT INTO t:apple { n: 2 }",
            "INSERT INTO t:banana { n: 3 }",
        ],
        "SELECT * FROM t",
    ),
    ("select-empty-table", &["CREATE TABLE t"], "SELECT * FROM t"),
    (
        "select-empty-after-filter",
        &["CREATE TABLE t", "INSERT INTO t:1 { n: 1 }"],
        "SELECT * FROM t WHERE n = 99",
    ),
    // ---- field predicates ----
    (
        "where-eq-mixed-types",
        &[
            "CREATE TABLE t",
            "INSERT INTO t:1 { v: 1 }",
            "INSERT INTO t:2 { v: 1.0 }",
            "INSERT INTO t:3 { v: \"1\" }",
            "INSERT INTO t:4 { v: null }",
            "INSERT INTO t:5 { other: 1 }",
        ],
        "SELECT * FROM t WHERE v = 1",
    ),
    (
        "where-ne-missing-never-matches",
        &["CREATE TABLE t", "INSERT INTO t:1 { v: 1 }", "INSERT INTO t:2 { w: 1 }"],
        "SELECT * FROM t WHERE v != 1",
    ),
    (
        "where-cmp-total-cross-type",
        &[
            "CREATE TABLE t",
            "INSERT INTO t:1 { v: null }",
            "INSERT INTO t:2 { v: false }",
            "INSERT INTO t:3 { v: 5 }",
            "INSERT INTO t:4 { v: \"abc\" }",
            "INSERT INTO t:5 { v: [1, 2] }",
            "INSERT INTO t:6 { v: { k: 1 } }",
        ],
        "SELECT * FROM t WHERE v < \"abc\"",
    ),
    (
        "where-in",
        &["CREATE TABLE t", "INSERT INTO t:1 { g: \"a\" }", "INSERT INTO t:2 { g: \"b\" }", "INSERT INTO t:3 { g: \"c\" }"],
        "SELECT * FROM t WHERE g IN [\"a\", \"c\"]",
    ),
    (
        "where-between",
        &["CREATE TABLE t", "INSERT INTO t:1 { n: 1 }", "INSERT INTO t:2 { n: 5 }", "INSERT INTO t:3 { n: 10 }"],
        "SELECT * FROM t WHERE n BETWEEN 2 AND 9",
    ),
    (
        "where-and-conjunction",
        &[
            "CREATE TABLE t",
            "INSERT INTO t:1 { a: 1, b: 2 }",
            "INSERT INTO t:2 { a: 1, b: 3 }",
            "INSERT INTO t:3 { a: 2, b: 2 }",
        ],
        "SELECT * FROM t WHERE a = 1 AND b >= 2",
    ),
    (
        "where-id-pseudo-field",
        &[
            "CREATE TABLE t",
            "INSERT INTO t:1 { x: 1 }",
            "INSERT INTO t:alice { x: 2 }",
        ],
        "SELECT * FROM t WHERE id IN [\"t:1\", \"t:alice\"]",
    ),
    (
        "where-has-embedding",
        &[
            "CREATE TABLE t VECTOR<f32, 2>",
            "INSERT INTO t:1 {} EMBED [1.0, 0.0]",
            "INSERT INTO t:2 {}",
        ],
        "SELECT * FROM t WHERE embedding IS NOT NULL",
    ),
    // ---- kNN (exact cosine through the brute-force index) ----
    (
        "knn-topk-order-and-tiebreak",
        &[
            "CREATE TABLE t VECTOR<f32, 3>",
            "INSERT INTO t:a {} EMBED [1.0, 0.0, 0.0]",
            "INSERT INTO t:b {} EMBED [0.0, 1.0, 0.0]",
            "INSERT INTO t:c {} EMBED [1.0, 0.0, 0.0]",
            "INSERT INTO t:d {} EMBED [0.6, 0.8, 0.0]",
        ],
        "SELECT * FROM t WHERE vector::similarity(embedding, [1.0, 0.0, 0.0]) AND k = 3",
    ),
    (
        "knn-missing-embedding-scores-zero",
        &[
            "CREATE TABLE t VECTOR<f32, 2>",
            "INSERT INTO t:1 {} EMBED [1.0, 0.0]",
            "INSERT INTO t:2 {}",
        ],
        "SELECT * FROM t WHERE vector::similarity(embedding, [1.0, 0.0]) AND k = 5",
    ),
    (
        "knn-k-vs-limit-cap",
        &[
            "CREATE TABLE t VECTOR<f32, 2>",
            "INSERT INTO t:1 {} EMBED [1.0, 0.0]",
            "INSERT INTO t:2 {} EMBED [0.9, 0.1]",
            "INSERT INTO t:3 {} EMBED [0.0, 1.0]",
        ],
        "SELECT * FROM t WHERE vector::similarity(embedding, [1.0, 0.0]) AND k = 3 LIMIT 1",
    ),
    // ---- aggregates, projection, paging ----
    (
        "count-star",
        &["CREATE TABLE t", "INSERT INTO t:1 { n: 1 }", "INSERT INTO t:2 { n: 2 }"],
        "SELECT COUNT(*) FROM t",
    ),
    (
        "count-star-filtered-and-empty",
        &["CREATE TABLE t"],
        "SELECT COUNT(*) FROM t WHERE n = 1",
    ),
    (
        "projection-fields",
        &[
            "CREATE TABLE t",
            "INSERT INTO t:1 { keep: 1, drop: 2 }",
            "INSERT INTO t:2 { keep: 3 }",
        ],
        "SELECT keep FROM t",
    ),
    (
        "offset-limit",
        &[
            "CREATE TABLE t",
            "INSERT INTO t:1 { n: 1 }",
            "INSERT INTO t:2 { n: 2 }",
            "INSERT INTO t:3 { n: 3 }",
            "INSERT INTO t:4 { n: 4 }",
        ],
        "SELECT * FROM t ORDER BY n DESC LIMIT 2 OFFSET 1",
    ),
    // ---- ORDER BY: field / recency ----
    (
        "order-field-absent-ranks-null",
        &[
            "CREATE TABLE t",
            "INSERT INTO t:1 { s: 1 }",
            "INSERT INTO t:2 { s: 3 }",
            "INSERT INTO t:3 { other: 0 }",
        ],
        "SELECT * FROM t ORDER BY s",
    ),
    (
        "order-field-desc-ties-by-id",
        &[
            "CREATE TABLE t",
            "INSERT INTO t:b { s: 1 }",
            "INSERT INTO t:a { s: 1 }",
            "INSERT INTO t:c { s: 2 }",
        ],
        "SELECT * FROM t ORDER BY s DESC",
    ),
    (
        "order-recency-stamped",
        &[
            "CREATE TABLE t",
            "INSERT INTO t:1 { n: 1 }",
            "INSERT INTO t:2 { n: 2 }",
            "INSERT INTO t:3 { n: 3 }",
        ],
        "SELECT * FROM t ORDER BY ::recency",
    ),
    (
        "order-unknown-field-errors",
        &["CREATE TABLE t", "INSERT INTO t:1 { n: 1 }"],
        "SELECT * FROM t ORDER BY missing_key",
    ),
    // ---- ORDER BY: vote-derived scores ----
    (
        "order-score-laplace-votes",
        &[
            "CREATE TABLE t",
            "CREATE TABLE v",
            "INSERT INTO t:a { }",
            "INSERT INTO t:b { }",
            "INSERT INTO t:c { }",
            "INSERT INTO v:1 { }",
            "RELATE (v:1) -> :voted -> (t:a) SET weight = 1.0, value = 1",
            "RELATE (v:1) -> :voted -> (t:b) SET value = -1",
            "RELATE (v:1) -> :voted -> (t:c) SET weight = 0.0",
        ],
        "SELECT * FROM t ORDER BY ::score",
    ),
    (
        "order-votes-net",
        &[
            "CREATE TABLE t",
            "CREATE TABLE v",
            "INSERT INTO t:a { }",
            "INSERT INTO t:b { }",
            "INSERT INTO v:1 { }",
            "INSERT INTO v:2 { }",
            "RELATE (v:1) -> :voted -> (t:a) SET value = 1",
            "RELATE (v:2) -> :voted -> (t:a) SET value = 1",
            "RELATE (v:1) -> :voted -> (t:b) SET value = -1",
        ],
        "SELECT * FROM t ORDER BY ::votes",
    ),
    (
        "order-feedback-decay",
        &[
            "CREATE TABLE t",
            "CREATE TABLE v",
            "INSERT INTO t:a { }",
            "INSERT INTO t:b { }",
            "INSERT INTO v:1 { }",
            "RELATE (v:1) -> :voted -> (t:a) SET value = 1",
            "RELATE (v:1) -> :voted -> (t:b) SET value = -1",
        ],
        "SELECT * FROM t ORDER BY ::feedback",
    ),
    (
        "order-salience-blend-with-knn",
        &[
            "CREATE TABLE t VECTOR<f32, 2>",
            "CREATE TABLE v",
            "INSERT INTO t:a {} EMBED [1.0, 0.0]",
            "INSERT INTO t:b {} EMBED [0.5, 0.5]",
            "INSERT INTO v:1 { }",
            "RELATE (v:1) -> :voted -> (t:a) SET value = 1",
        ],
        "SELECT * FROM t WHERE vector::similarity(embedding, [1.0, 0.0]) AND k = 5 ORDER BY ::salience",
    ),
    (
        "order-salience-gamma-importance",
        &[
            "CREATE TABLE t",
            "INSERT INTO t:a { importance: 0.9 }",
            "INSERT INTO t:b { importance: 5 }",
            "INSERT INTO t:c { importance: -1 }",
            "INSERT INTO t:d { }",
        ],
        "SELECT * FROM t ORDER BY ::salience(0, 0, 1, 0)",
    ),
    (
        "order-salience-beta-strength",
        &[
            "CREATE TABLE t",
            "CREATE TABLE v",
            "INSERT INTO t:a { }",
            "INSERT INTO t:b { }",
            "INSERT INTO v:1 { }",
            "RELATE (v:1) -> :x -> (t:a)",
            "RELATE (v:1) -> :x -> (t:b)",
        ],
        "SELECT * FROM t ORDER BY ::salience(0, 1, 0, 0)",
    ),
    (
        "order-similarity-explicit-with-knn",
        &[
            "CREATE TABLE t VECTOR<f32, 2>",
            "INSERT INTO t:1 {} EMBED [1.0, 0.0]",
            "INSERT INTO t:2 {} EMBED [0.0, 1.0]",
        ],
        "SELECT * FROM t WHERE vector::similarity(embedding, [1.0, 0.0]) AND k = 5 ORDER BY similarity",
    ),
    // ---- BM25 + hybrid ----
    (
        "bm25-orders-by-relevance",
        &[
            "CREATE TABLE doc",
            "INSERT INTO doc:1 { text: \"neural search engine\" }",
            "INSERT INTO doc:2 { text: \"cooking recipes\" }",
            "INSERT INTO doc:3 { text: \"search for the neural path\" }",
            "INSERT INTO doc:4 { other: \"neural\" }",
        ],
        "SELECT * FROM doc WHERE ::bm25(text, \"neural search\")",
    ),
    (
        "bm25-k-caps-rows",
        &[
            "CREATE TABLE doc",
            "INSERT INTO doc:1 { text: \"alpha beta\" }",
            "INSERT INTO doc:2 { text: \"alpha gamma\" }",
            "INSERT INTO doc:3 { text: \"alpha delta\" }",
        ],
        "SELECT * FROM doc WHERE ::bm25(text, \"alpha\") AND k = 2",
    ),
    (
        "hybrid-rrf-fusion",
        &[
            "CREATE TABLE t VECTOR<f32, 2>",
            "INSERT INTO t:1 { text: \"alpha\" } EMBED [1.0, 0.0]",
            "INSERT INTO t:2 { text: \"beta\" } EMBED [0.0, 1.0]",
            "INSERT INTO t:3 { text: \"alpha beta\" } EMBED [0.7, 0.7]",
        ],
        "SELECT * FROM t WHERE ::bm25(text, \"alpha\") AND vector::similarity(embedding, [1.0, 0.0]) AND k = 3",
    ),
    // ---- DML effects ----
    (
        "forget-removes-record-and-edges",
        &[
            "CREATE TABLE t",
            "INSERT INTO t:a { }",
            "INSERT INTO t:b { }",
            "RELATE (t:a) -> :link -> (t:b)",
        ],
        "FORGET t:a; SELECT * FROM t",
    ),
    (
        "insert-stamps-created-at",
        &[
            "CREATE TABLE t",
            "INSERT INTO t:1 { }",
            "INSERT INTO t:2 { }",
            "FORGET t:1",
            "INSERT INTO t:3 { }",
        ],
        "SELECT * FROM t ORDER BY ::recency",
    ),
    // ---- errors ----
    (
        "error-embedding-dim-mismatch",
        &["CREATE TABLE t VECTOR<f32, 2>"],
        "INSERT INTO t:1 {} EMBED [1.0, 0.0, 0.0]",
    ),
    // ---- multi-query plan ----
    (
        "two-selects-one-plan",
        &["CREATE TABLE t", "INSERT INTO t:1 { n: 1 }"],
        "SELECT * FROM t; SELECT COUNT(*) FROM t",
    ),
];

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Composite row bytes: `uleb128(len)` then per row
/// `postcard(Record)` + `f32 LE score` — documented in the fixture README.
fn rows_hex(rows: &[ScoredRecord]) -> String {
    let mut out = Vec::new();
    // uleb128 length (rows counts are tiny; match §5.7's unsigned varint).
    let mut n = rows.len() as u64;
    loop {
        let b = (n & 0x7f) as u8;
        n >>= 7;
        if n == 0 {
            out.push(b);
            break;
        }
        out.push(b | 0x80);
    }
    for r in rows {
        out.extend_from_slice(&postcard::to_allocvec(&r.record).expect("record"));
        out.extend_from_slice(&r.score.to_le_bytes());
    }
    hex(&out)
}

fn result_json(res: &QueryResult) -> serde_json::Value {
    let kind = match res.kind {
        QueryKind::Select(_) => "select",
        QueryKind::Match(_) => "match",
        QueryKind::Closure(_) => "closure",
        QueryKind::History { .. } => "history",
    };
    let rows: Vec<serde_json::Value> = res
        .rows
        .iter()
        .map(|r| {
            json!({
                "record": serde_json::to_value(&r.record).expect("record json"),
                "score": r.score,
            })
        })
        .collect();
    json!({
        "kind": kind,
        "rows": rows,
        "rows_hex": rows_hex(&res.rows),
    })
}

fn error_variant(e: &nqlite::Error) -> &'static str {
    match e {
        nqlite::Error::EmbeddingDimMismatch { .. } => "EmbeddingDimMismatch",
        nqlite::Error::Storage(_) => "Storage",
        nqlite::Error::MemoryWithoutContext { .. } => "MemoryWithoutContext",
        nqlite::Error::HistoryPruned { .. } => "HistoryPruned",
        nqlite::Error::UnknownSortField { .. } => "UnknownSortField",
    }
}

fn analysis_variant(e: &nql::AnalysisError) -> &'static str {
    match e {
        nql::AnalysisError::UnknownTableForSelect { .. } => "UnknownTableForSelect",
        nql::AnalysisError::UnknownTableForInsert { .. } => "UnknownTableForInsert",
        nql::AnalysisError::EmbeddingDimMismatch { .. } => "EmbeddingDimMismatch",
        nql::AnalysisError::SimilarityWithoutKnn => "SimilarityWithoutKnn",
        nql::AnalysisError::EmptyTable => "EmptyTable",
        nql::AnalysisError::EmptyId { .. } => "EmptyId",
    }
}

fn run_case(name: &str, setup: &[&str], query: &str) -> serde_json::Value {
    let mut db = Database::default();
    // One parse + one analyzer context for setup AND query (declarations
    // carry); execution yields results only for the query's read statements.
    let mut plan = Vec::new();
    for s in setup {
        plan.extend(nql::parse(s).unwrap_or_else(|e| panic!("case {name} setup `{s}`: {e}")));
    }
    plan.extend(nql::parse(query).unwrap_or_else(|e| panic!("case {name} query `{query}`: {e}")));
    let plan = match nql::Analyzer::analyze(&plan) {
        Ok(p) => p,
        // Analysis failures are outcomes too (stage-agnostic variant names).
        Err(ae) => {
            return json!({
                "name": name,
                "setup": setup,
                "query": query,
                "error": analysis_variant(&ae),
            });
        }
    };
    match db.execute(&plan) {
        Ok(results) => json!({
            "name": name,
            "setup": setup,
            "query": query,
            "results": results.iter().map(result_json).collect::<Vec<_>>(),
        }),
        Err(e) => json!({
            "name": name,
            "setup": setup,
            "query": query,
            "error": error_variant(&e),
        }),
    }
}

fn corpus_json() -> String {
    let cases: Vec<serde_json::Value> = CASES
        .iter()
        .map(|(name, setup, query)| run_case(name, setup, query))
        .collect();
    let doc = json!({
        "note": "generated by nqlite/tests/m3_results.rs (UPDATE_FIXTURES=1); rows_hex = uleb128(len) + [postcard(Record) + f32 LE score] per row; never hand-edit",
        "cases": cases,
    });
    format!("{}\n", serde_json::to_string_pretty(&doc).unwrap())
}

#[test]
fn engine_results_match_files() {
    let text = corpus_json();
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../spec/fixtures/engine/results.json");
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
            "engine results drifted — semantics change? regenerate + bump .spec-pin"
        );
    }
    let doc: serde_json::Value = serde_json::from_str(&text).unwrap();
    let cases = doc["cases"].as_array().unwrap();
    assert!(cases.len() >= 30, "corpus shrank: {} cases", cases.len());
    let errors = cases.iter().filter(|c| c.get("error").is_some()).count();
    assert!(errors >= 2, "error coverage: {errors}");
    assert!(cases.iter().any(|c| c["results"][0]["kind"] == "select"));
}
