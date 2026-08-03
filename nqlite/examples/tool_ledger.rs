//! Agent pattern #3 — tool-call ledger.
//!
//! Every tool invocation the agent makes is recorded as a `call` record
//! (tool name + args + result) and linked into the graph: each call
//! `(call) -[:used]-> (tool)`, and the agent keeps an aggregate
//! `(agent) -[:called]-> (tool)` usage ledger. The agent then FORGETs an
//! old entry (the engine cascades: the record and its incident edges are
//! gone) and queries the remaining ledger with a field filter.
//!
//! Forgetting is agent-initiated; the engine just removes data
//! deterministically. It never decides what to keep.

use std::error::Error;

use nql::parse;
use nql_ir::Store;
use nqlite::Database;

fn main() -> Result<(), Box<dyn Error>> {
    let mut db = Database::new(Store::default());

    // --- record the invocations + ledger edges ------------------------------
    let results = db.execute(&parse(
        r#"
        CREATE TABLE call;
        CREATE TABLE tool;

        INSERT INTO tool:search { "name": "web_search" };
        INSERT INTO tool:calc   { "name": "calculator" };
        INSERT INTO tool:code   { "name": "code_interpreter" };
        INSERT INTO tool:email  { "name": "send_email" };

        INSERT INTO call:1 { "tool": "web_search", "args": "nqlite vector index", "result": "3 hits" };
        INSERT INTO call:2 { "tool": "calculator", "args": "42 * 17",           "result": "714" };
        INSERT INTO call:3 { "tool": "web_search", "args": "rust hnsw crate",   "result": "2 hits" };
        INSERT INTO call:4 { "tool": "send_email", "args": "summary to alice",  "result": "queued" };

        RELATE (call:1) -> :used   -> (tool:search);
        RELATE (call:2) -> :used   -> (tool:calc);
        RELATE (call:3) -> :used   -> (tool:search);
        RELATE (call:4) -> :used   -> (tool:email);

        RELATE (agent:assistant) -> :called -> (tool:search);
        RELATE (agent:assistant) -> :called -> (tool:calc);
        RELATE (agent:assistant) -> :called -> (tool:search);
        RELATE (agent:assistant) -> :called -> (tool:email);

        SELECT * FROM call;
        "#,
    )?)?;

    println!(
        "tool ledger — initial state ({} invocations):",
        results[0].rows.len()
    );
    for row in &results[0].rows {
        println!(
            "  {:>7}  tool={:<11} args={:<22} result={}",
            row.record.id,
            str_field(&row.record.body, "tool"),
            str_field(&row.record.body, "args"),
            str_field(&row.record.body, "result")
        );
    }

    // --- FORGET an old entry (agent-initiated cleanup) ----------------------
    let results = db.execute(&parse(
        r#"
        FORGET call:2;
        SELECT * FROM call WHERE tool = "web_search";
        "#,
    )?)?;
    let filtered = &results[0];
    println!(
        "\nafter FORGET call:2 — web_search invocations still in the ledger ({}):",
        filtered.rows.len()
    );
    for row in &filtered.rows {
        println!(
            "  {:>7}  tool={:<11} args={:<22} result={}",
            row.record.id,
            str_field(&row.record.body, "tool"),
            str_field(&row.record.body, "args"),
            str_field(&row.record.body, "result")
        );
    }

    // --- ledger edges after the forget --------------------------------------
    let store = db.store();
    let used: Vec<_> = store.edges.iter().filter(|e| e.name == "used").collect();
    let called: Vec<_> = store.edges.iter().filter(|e| e.name == "called").collect();
    println!("\nremaining ledger edges:");
    println!("  :used   ({} edges):", used.len());
    for e in &used {
        println!("    ({}) -[:used]-> ({})", e.from, e.to);
    }
    println!("  :called ({} edges):", called.len());
    for e in &called {
        println!("    ({}) -[:called]-> ({})", e.from, e.to);
    }
    println!(
        "\nnote: FORGET call:2 removed the record AND its :used edge \
         (call:2 -> tool:calc) in one deterministic cascade."
    );

    Ok(())
}

/// A body field as a string.
fn str_field(body: &std::collections::BTreeMap<String, nql_ir::Value>, key: &str) -> String {
    match body.get(key) {
        Some(nql_ir::Value::Str(s)) => s.clone(),
        _ => String::from("?"),
    }
}
