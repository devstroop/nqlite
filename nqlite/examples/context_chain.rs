//! Agent pattern #5 — context chain.
//!
//! The agent splits its state into two named memory partitions: `MEMORY core`
//! holds the live working context, `MEMORY archival` holds the previous
//! session's context. Inside each partition, context records are linked into
//! a chain with `:follows_from` edges. `CLOSURE` then walks the chain
//! transitively (BFS to fixpoint) — every step of the reasoning, in first-
//! visit order, scored by depth — so the agent reconstructs "everything that
//! led here" with one statement. An edge-property filter (`WHERE status =
//! "final"`) prunes the walk to the edges the agent marked final.
//!
//! The agent decides which partition a record lives in, which edges are
//! `status = "final"`, and whether an edge points forward (`->`) or backward
//! (`<-`). The engine only partitions writes by memory and walks the graph
//! deterministically; it never re-orders, re-weights, or decides what to keep.

use std::error::Error;

use nql::parse;
use nql_ir::Store;
use nqlite::Database;

fn main() -> Result<(), Box<dyn Error>> {
    let mut db = Database::new(Store::default());

    // --- 1. root store: shared facts (no MEMORY yet) ------------------------
    // Plans always start at the root; anything written before the first
    // `MEMORY` belongs to every memory's parent store.
    db.execute(&parse(
        r#"
        CREATE TABLE agent;
        INSERT INTO agent:assistant { "name": "nqlite agent" };
        "#,
    )?)?;
    println!("root store: seeded 1 shared agent profile record");

    // --- 2. MEMORY core: the live working context chain ---------------------
    // Every statement after `MEMORY core` in THIS plan runs against the core
    // sub-store (its own records, edges, and history). The chain is four
    // context records linked by `:follows_from`; the extra edge
    // (context:3 -> context:2) is a back-reference — the agent re-read an
    // earlier step, and the closure must NOT re-list it.
    let results = db.execute(&parse(
        r#"
        MEMORY core;
        CREATE TABLE context;

        INSERT INTO context:1 { "role": "user",      "text": "plan the Q3 launch" };
        INSERT INTO context:2 { "role": "assistant", "text": "drafted: blog post + social plan" };
        INSERT INTO context:3 { "role": "assistant", "text": "decided: announce Friday 09:00, blog first" };
        INSERT INTO context:4 { "role": "todo",      "text": "needs customer testimonials" };

        RELATE (context:1) -> :follows_from -> (context:2) SET status = "final";
        RELATE (context:2) -> :follows_from -> (context:3) SET status = "final";
        RELATE (context:3) -> :follows_from -> (context:4) SET status = "tentative";
        RELATE (context:3) -> :follows_from -> (context:2) SET status = "final";

        SELECT * FROM context;

        CLOSURE (context:1) -> :follows_from;
        CLOSURE (context:1) -> :follows_from WHERE status = "final";
        "#,
    )?)?;

    let ctx = &results[0];
    println!("core memory — {} context records:", ctx.rows.len());
    for row in &ctx.rows {
        println!(
            "  {:>9}  {:<9}  {}",
            row.record.id,
            str_field(&row.record.body, "role"),
            str_field(&row.record.body, "text")
        );
    }

    // CLOSURE without a filter: the whole chain, in first-visit (BFS) order.
    // The back-reference (context:3 -> context:2) is already visited, so it
    // contributes nothing — each record appears exactly once, scored by depth.
    let full = &results[1];
    println!("\nCLOSURE over the full chain — depth-scored walk:");
    for row in &full.rows {
        println!(
            "  {:>9}  depth={:.0}  {}",
            row.record.id,
            row.score,
            str_field(&row.record.body, "text")
        );
    }

    // The same walk, but only across edges the agent marked final: the
    // tentative context:3 -> context:4 hop is skipped, so the todo never
    // enters the chain.
    let final_only = &results[2];
    println!("\nCLOSURE WHERE status = \"final\" — tentative hop pruned:");
    for row in &final_only.rows {
        println!(
            "  {:>9}  depth={:.0}  {}",
            row.record.id,
            row.score,
            str_field(&row.record.body, "text")
        );
    }

    // --- 3. MEMORY archival: a second partition, same table:id, new record ---
    // The archival store gets the SAME `context:1` id as core — inside a
    // memory partition, `table:id` identifiers are scoped, so these are two
    // different records. The trailing `<- :follows_from` walks the chain
    // backwards: from the latest log, every archived step that led to it.
    let results = db.execute(&parse(
        r#"
        MEMORY archival;
        CREATE TABLE context;

        INSERT INTO context:1 { "role": "log", "text": "last launch: keynote + live demo" };
        INSERT INTO context:2 { "role": "log", "text": "release notes shipped in 2 days" };
        RELATE (context:1) -> :follows_from -> (context:2) SET status = "final";

        SELECT * FROM context;
        CLOSURE (context:2) <- :follows_from;
        "#,
    )?)?;
    let archived = &results[0];
    println!(
        "\narchival memory — {} archived context records:",
        archived.rows.len()
    );
    for row in &archived.rows {
        println!(
            "  {:>9}  {:<8}  {}",
            row.record.id,
            str_field(&row.record.body, "role"),
            str_field(&row.record.body, "text")
        );
    }
    let backwalk = &results[1];
    println!("\nCLOSURE (context:2) <- :follows_from — what led up to the log:");
    for row in &backwalk.rows {
        println!(
            "  {:>9}  depth={:.0}  {}",
            row.record.id,
            row.score,
            str_field(&row.record.body, "text")
        );
    }

    // --- 4. partition isolation ---------------------------------------------
    // The root store still holds only the shared agent profile; the two
    // memories hold their own stores. Writes inside a memory never leak out.
    let store = db.store();
    println!("\nstore summary:");
    println!(
        "  root      -> {} records (only the shared profile)",
        store.records.len()
    );
    let core = &store.memories["core"];
    let archival = &store.memories["archival"];
    println!(
        "  memory core     -> {} records, {} :follows_from edges",
        core.records.len(),
        core.edges.len()
    );
    println!(
        "  memory archival -> {} records, {} :follows_from edges",
        archival.records.len(),
        archival.edges.len()
    );
    println!(
        "note: context:1 is a different record in core vs archival — memory \
         partitions scope table:id, so agents can re-number freely."
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
