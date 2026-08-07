//! Agent pattern #4 — idempotent memory.
//!
//! An agent keeps long-lived facts in a small `memory` table and re-syncs
//! them on every session. Re-asserting the SAME `memory:id` key with newer
//! text + a fresh BYO embedding is an **upsert** (the store's records are
//! keyed by RecordId), so replaying the sync routine is *idempotent*: no
//! duplicate records, no unbounded growth — the table converges to the
//! latest generation, and the state is exactly the same whether the routine
//! ran once or a hundred times.
//!
//! Every mutating statement also advances the store's logical clock and is
//! appended to its history, so `SELECT ... AS OF <ts>` replays `ts <= <ts>`
//! and time-travels to an earlier generation. The agent can diff "what I
//! knew before the refresh" against "what I know now" with zero extra
//! bookkeeping — no snapshots, no wall-clock, one deterministic counter.
//!
//! Timeline (logical timestamps, assigned in statement order):
//!
//! ```text
//! ts 1   CREATE TABLE memory VECTOR<f32, 3>;
//! ts 2   INSERT memory:1 { gen-1 } EMBED [1.00, 0.00, 0.00];
//! ts 3   INSERT memory:2 { gen-1 } EMBED [0.00, 1.00, 0.00];
//! ts 4   INSERT memory:3 { gen-1 } EMBED [0.00, 0.00, 1.00];
//! ts 5   INSERT memory:1 { gen-2 } EMBED [0.95, 0.00, 0.00];   // upsert
//! ts 6   INSERT memory:2 { gen-2 } EMBED [0.00, 0.95, 0.00];   // upsert
//! ```
//!
//! `AS OF 4` sees the full generation-1 snapshot; the current store sees
//! generation-2 — and both are exactly three rows (idempotent, not +2).

use std::collections::BTreeMap;
use std::error::Error;

use nql::parse;
use nql_ir::{Store, Value};
use nqlite::Database;

fn main() -> Result<(), Box<dyn Error>> {
    let mut db = Database::new(Store::default());

    // --- write: schema + two generations of the same memory keys ------------
    // Generation 1 is the first sync; generation 2 re-runs the same routine
    // over the same ids, so it upserts. Text and embeddings are written by
    // the agent (BYO); the engine never embeds, re-embeds, or learns.
    db.execute(&parse(
        r#"
        CREATE TABLE memory VECTOR<f32, 3>;

        INSERT INTO memory:1 { "kind": "project",
                               "text": "nqlite is deterministic: same plan, same store -> same output" }
            EMBED [1.00, 0.00, 0.00];
        INSERT INTO memory:2 { "kind": "project",
                               "text": "embeddings are BYO: the agent brings every vector" }
            EMBED [0.00, 1.00, 0.00];
        INSERT INTO memory:3 { "kind": "temporal",
                               "text": "AS OF rewinds the store to a past logical timestamp" }
            EMBED [0.00, 0.00, 1.00];

        INSERT INTO memory:1 { "kind": "project",
                               "text": "re-asserting a key is an upsert, never a duplicate" }
            EMBED [0.95, 0.00, 0.00];
        INSERT INTO memory:2 { "kind": "project",
                               "text": "the upsert keeps the table the same size" }
            EMBED [0.00, 0.95, 0.00];
        "#,
    )?)?;

    // --- read: current state vs. the temporal snapshot ----------------------
    // read 1 = current (generation 2); read 2 = temporal (generation 1, as
    // of before the refresh). Each SELECT contributes one QueryResult.
    let results = db.execute(&parse(
        r#"
        SELECT * FROM memory;
        SELECT * FROM memory AS OF 4;
        "#,
    )?)?;

    // --- current state ------------------------------------------------------
    let now = &results[0];
    println!(
        "sync routine ran twice — current memory ({} records):",
        now.rows.len()
    );
    for row in &now.rows {
        println!(
            "  {:>9}  kind={:<8}  {}",
            row.record.id,
            str_field(&row.record.body, "kind"),
            str_field(&row.record.body, "text")
        );
    }
    println!("  (same 3 keys, zero duplicates — second sync was idempotent)");

    // --- temporal read: the pre-refresh generation --------------------------
    let past = &results[1];
    println!(
        "\nSELECT * FROM memory AS OF 4 ({} records):",
        past.rows.len()
    );
    for row in &past.rows {
        println!(
            "  {:>9}  kind={:<10}  {}",
            row.record.id,
            str_field(&row.record.body, "kind"),
            str_field(&row.record.body, "text")
        );
    }
    println!(
        "\nnote: AS OF 4 replays history (ts <= 4) and reconstructs generation 1 \
         exactly — the same record ids, older texts and embeddings. The diff \
         between the two generations is the agent's own refresh, recovered \
         without any extra bookkeeping."
    );

    Ok(())
}

/// A body field as a string.
fn str_field(body: &BTreeMap<String, Value>, key: &str) -> String {
    match body.get(key) {
        Some(Value::Str(s)) => s.clone(),
        _ => String::from("?"),
    }
}
