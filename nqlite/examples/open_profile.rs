//! Cold-open profiler (issue #115): decomposes "reopen a 100K store" into
//! **store load** vs **query execution**, without the process-start, parse,
//! or row-render overhead a CLI measurement includes.
//!
//! Usage:
//!
//! ```text
//! cargo build --release --example open_profile -p nqlite
//! target/release/examples/open_profile <db-path>
//! ```
//!
//! Reported per run (deterministic stores → deterministic work; wall times
//! are machine-dependent — L2 evidence needs commit + profile + machine,
//! see the nqlite-experiments evidence ladder):
//!
//! - `load`    — `Database::open`: main-file read + postcard decode of the
//!               whole `Store` (records, edges, history) + WAL check
//! - `count`   — `SELECT COUNT(*) FROM doc` (parse + execute; one row)
//! - `scan`    — `SELECT * FROM doc` (every row materialized, nothing
//!               rendered — the render cost lives in the CLI, measured
//!               separately with `time` on a line-protocol run)
//!
//! The gap between `load` and the E08 end-to-end cold-open number is the
//! CLI's process/parse/render share; the gap between `count` and `scan` is
//! the row-materialization share.

use std::time::Instant;

use nqlite::Database;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: open_profile <db-path>");
    let file_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

    // --- 1. store load (the "reopen" itself) ----------------------------
    let t = Instant::now();
    let mut db = Database::open(&path).expect("open database");
    let load = t.elapsed();
    let (records, edges, history) = {
        let store = db.store();
        (store.records.len(), store.edges.len(), store.history.len())
    };

    // --- 2. count: parse + analyze + execute, one row out ---------------
    // Run twice: `count1` carries any one-shot first-query cost after open,
    // `count2` is the steady-state number (same plan, warm).
    let count_plan = nql::parse("SELECT COUNT(*) FROM doc;").expect("parse count");
    let t = Instant::now();
    let res = db.execute(&count_plan).expect("count");
    let count1 = t.elapsed();
    let counted = res[0]
        .rows
        .first()
        .and_then(|r| r.record.body.get("count"))
        .cloned();
    let t = Instant::now();
    db.execute(&count_plan).expect("count again");
    let count2 = t.elapsed();

    // --- 3. full scan: every row materialized in-process, not rendered ---
    let scan_plan = nql::parse("SELECT * FROM doc;").expect("parse scan");
    let t = Instant::now();
    let res = db.execute(&scan_plan).expect("scan");
    let scan = t.elapsed();
    let rows = res[0].rows.len();

    let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
    println!("db={path}");
    println!(
        "file={:.1} MB  records={records} edges={edges} history={history}",
        file_bytes as f64 / 1e6
    );
    println!(
        "load  {:8.1} ms   (open: read + decode whole Store + WAL check)",
        ms(load)
    );
    println!(
        "count1 {:7.1} ms   (first query after open, COUNT(*), result={counted:?})",
        ms(count1)
    );
    println!("count2 {:7.1} ms   (steady-state COUNT(*))", ms(count2));
    println!(
        "scan  {:8.1} ms   ({rows} rows materialized, not rendered)",
        ms(scan)
    );
}
