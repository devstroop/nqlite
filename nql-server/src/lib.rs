//! nql-server core: a deterministic line-protocol server over `nqlite`.
//!
//! The protocol is one nql program per line (`;`-separated statements). Each
//! line gets exactly one response: one output line per `SELECT` result
//! (deterministic: BTree key order, ids, scores to 4dp, no timestamps)
//! followed by a final `OK`, or a single `ERR <message>` line. The server
//! holds ONE shared [`Database`] across all lines, so a session can
//! `CREATE TABLE`, `INSERT`, and `SELECT` across separate lines.
//!
//! Because `nql::Analyzer` validates per-plan (its table context is rebuilt
//! for every `analyze` call), this crate tracks the tables declared so far
//! and prepends synthetic `CREATE TABLE` statements before analysis so that
//! cross-line references pass validation and SELECT enrichment (e.g. kNN
//! implies `ORDER BY ::similarity`) still applies.
//!
//! Zero-LLM and deterministic: no network except the optional TCP transport
//! in `main.rs`, no randomness, no wall-clock in any response.

use std::collections::BTreeMap;

use nql::Analyzer;
use nql_ir::{Statement, Value};
use nqlite::Database;

/// One nql line-protocol server: a persistent [`Database`] plus the set of
/// tables declared so far (name -> optional vector dimension), mirroring the
/// state `nql::Analyzer` rebuilds per plan.
#[derive(Debug, Default)]
pub struct Server {
    db: Database,
    /// Tables declared by earlier lines. `None` = declared without a VECTOR
    /// clause (dimension cleared, matching engine semantics).
    declared: BTreeMap<String, Option<usize>>,
}

impl Server {
    /// Start a fresh, empty server session (deterministic).
    pub fn new() -> Self {
        Server::default()
    }

    /// Open a server over a persistent single-file store (`nql-server --db`).
    ///
    /// The analyzer's cross-line table context cannot be rebuilt from records
    /// alone (an empty, dimension-less table leaves no trace outside its
    /// `CreateTable` statement), so `declared` is re-seeded by replaying the
    /// persisted mutation histories — the root store's plus every memory
    /// block's — in order, last declaration winning (matching the engine's
    /// dim-set/dim-clear semantics). Tables created before a restart stay
    /// usable after it.
    ///
    /// Fails with the store's open error (issue #84's `Locked` surfaces here
    /// when another process owns the file).
    pub fn open(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let db = Database::open(path).map_err(|e| std::io::Error::other(format!("open: {e}")))?;
        let declared = seed_declared(&db);
        Ok(Server { db, declared })
    }

    /// Handle one line of the protocol: parse, analyze (with cross-line table
    /// context), execute against the shared database, and return the response
    /// text. Never panics on bad input; a malformed line yields `ERR ...` and
    /// the shared database is left untouched by that line's failed parse or
    /// analysis.
    pub fn handle_line(&mut self, line: &str) -> String {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return "OK".to_string();
        }

        let plan = match nql::parse(trimmed) {
            Ok(p) => p,
            Err(e) => return format!("ERR {e}"),
        };

        // The analyzer's context is per-plan, so prepend a synthetic CREATE
        // TABLE for every previously declared table this line does not
        // re-declare. This makes cross-line sessions analyzable while keeping
        // this line's own declarations authoritative.
        let mut augmented: Vec<Statement> = Vec::new();
        for (table, dim) in &self.declared {
            let redeclared = plan
                .iter()
                .any(|s| matches!(s, Statement::CreateTable { table: t, .. } if t == table));
            if !redeclared {
                augmented.push(Statement::CreateTable {
                    table: table.clone(),
                    vector_dim: *dim,
                });
            }
        }
        augmented.extend(plan.iter().cloned());

        let analyzed = match Analyzer::analyze(&augmented) {
            Ok(p) => p,
            Err(e) => return format!("ERR {e}"),
        };

        // Execute only the original statements (skip the synthetic prefix).
        let prefix_len = augmented.len() - plan.len();
        match self.db.execute(&analyzed[prefix_len..]) {
            Ok(results) => {
                // Record tables this line declared (only after success, so a
                // failed plan never poisons cross-line analysis).
                for stmt in &plan {
                    if let Statement::CreateTable { table, vector_dim } = stmt {
                        self.declared.insert(table.clone(), *vector_dim);
                    }
                }
                let mut out = String::new();
                for res in &results {
                    out.push_str(&format_result(res));
                    out.push('\n');
                }
                out.push_str("OK");
                out
            }
            Err(e) => format!("ERR {e}"),
        }
    }
}

/// Rebuild the analyzer's cross-line table context from a reopened store.
///
/// Analyzer re-seeding (issue #89): collect every declared table — with its
/// optional VECTOR dim, last declaration winning (mirroring the engine's
/// `vector_dims` set/dim-clear semantics) — from the store's `tables` index:
/// maintained live by the engine's CreateTable arm and rebuilt from history
/// once at load (issue #133 step 1), so no O(history) scan lives here.
/// Records alone can't seed this: an empty, dimension-less table (e.g.
/// `CREATE TABLE notes;` with no rows yet) exists only in that registry.
fn seed_declared(db: &Database) -> BTreeMap<String, Option<usize>> {
    let store = db.store();
    let mut declared = store.tables.clone();
    for memory in store.memories.values() {
        declared.extend(memory.tables.clone());
    }
    declared
}

/// One response line for a read result (SELECT or MATCH): the statement's
/// label, row count, and every row in the engine's deterministic order, ids +
/// 4dp scores included.
fn format_result(res: &nqlite::QueryResult) -> String {
    let label = match &res.kind {
        nqlite::QueryKind::Select(sel) => format!("SELECT {}", sel.table),
        nqlite::QueryKind::Match(path) => {
            let hops: Vec<String> = path
                .steps
                .iter()
                .map(|s| {
                    format!(
                        "{}:{}",
                        match s.direction {
                            nql_ir::MatchDirection::Out => "->",
                            nql_ir::MatchDirection::In => "<-",
                        },
                        s.name
                    )
                })
                .collect();
            format!("MATCH {} {}", path.start, hops.join(" "))
        }
        nqlite::QueryKind::Closure(path) => {
            let hops: Vec<String> = path
                .steps
                .iter()
                .map(|s| {
                    format!(
                        "{}:{}",
                        match s.direction {
                            nql_ir::MatchDirection::Out => "->",
                            nql_ir::MatchDirection::In => "<-",
                        },
                        s.name
                    )
                })
                .collect();
            format!("CLOSURE {} {}", path.start, hops.join(" "))
        }
        nqlite::QueryKind::History { since } => format!("HISTORY SINCE {since}"),
    };
    let rows: Vec<String> = res.rows.iter().map(format_row).collect();
    if rows.is_empty() {
        format!("{label} (0 rows)")
    } else {
        format!("{label} ({} rows): {}", res.rows.len(), rows.join("; "))
    }
}

/// One row inside a SELECT response line.
fn format_row(s: &nqlite::ScoredRecord) -> String {
    format!(
        "{} score={:.4} {}",
        s.record.id,
        s.score,
        format_fields(&s.record.body)
    )
}

/// Deterministic field rendering: BTreeMap iteration order, values shortened
/// (vectors/arrays truncated, strings quoted) exactly like the nql CLI.
fn format_fields(body: &BTreeMap<String, Value>) -> String {
    let parts: Vec<String> = body
        .iter()
        .map(|(k, v)| format!("{k}={}", short_value(v)))
        .collect();
    if parts.is_empty() {
        "{}".to_string()
    } else {
        format!("{{{}}}", parts.join(", "))
    }
}

fn short_value(v: &Value) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Int(i) => i.to_string(),
        Value::Float(f) => format!("{f}"),
        Value::Str(s) => format!("{s:?}"),
        Value::Doc(d) => format_fields(d),
        Value::Arr(a) => {
            let shown: Vec<String> = a.iter().take(3).map(short_value).collect();
            format!("[{}]", shown.join(", "))
        }
        Value::Vector(v) => {
            let shown: Vec<String> = v.iter().take(4).map(|x| format!("{x}")).collect();
            let more = if v.len() > 4 { ", ..." } else { "" };
            format!("[{}{}]", shown.join(", "), more)
        }
        Value::Ref(rid) => rid.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convenience: run one line through a fresh server, return the response.
    fn line(server: &mut Server, input: &str) -> String {
        server.handle_line(input)
    }

    #[test]
    fn insert_then_select_session_across_lines() {
        let mut s = Server::new();
        assert_eq!(
            line(&mut s, "CREATE TABLE t VECTOR<f32, 2>"),
            "OK",
            "create table"
        );
        assert_eq!(
            line(&mut s, r#"INSERT INTO t:1 { name: "a" } EMBED [1.0, 0.0]"#),
            "OK",
            "insert 1"
        );
        assert_eq!(
            line(&mut s, r#"INSERT INTO t:2 { name: "b" } EMBED [0.9, 0.1]"#),
            "OK",
            "insert 2"
        );
        let out = line(
            &mut s,
            "SELECT * FROM t WHERE vector::similarity(embedding, [1.0, 0.0]) AND k = 2",
        );
        // kNN without ORDER BY is enriched to ORDER BY ::similarity; nearest
        // record (t:1) must sort first, scores at 4dp, then OK.
        assert!(
            out.starts_with("SELECT t (2 rows): t:1 score=1.0000"),
            "nearest first with 4dp score, got: {out}"
        );
        assert!(
            out.contains("t:2 score=0.99"),
            "second row present with 4dp score, got: {out}"
        );
        assert!(out.ends_with("\nOK"), "final OK terminator, got: {out}");
    }

    #[test]
    fn open_persists_and_reseeds_declared_tables() {
        // Issue #89 (--db mode): reopening must restore BOTH the data and the
        // analyzer's cross-line table context — including an empty, no-dim
        // table (exists only as a history statement) and a memory-block table.
        let dir = std::env::temp_dir().join(format!("nqlite-server-db-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("store.nql");

        {
            let mut s = Server::open(&path).expect("first open");
            assert_eq!(line(&mut s, "CREATE TABLE t VECTOR<f32, 2>;"), "OK");
            assert_eq!(
                line(
                    &mut s,
                    r#"INSERT INTO t:1 { "text": "x" } EMBED [0.5, 0.5];"#
                ),
                "OK"
            );
            assert_eq!(line(&mut s, "CREATE TABLE scratch;"), "OK");
            assert_eq!(line(&mut s, "MEMORY m; CREATE TABLE mt;"), "OK");
            assert_eq!(
                line(&mut s, r#"MEMORY m; INSERT INTO mt:1 { "v": 1 };"#),
                "OK"
            );
        } // drop: WAL persisted, single-writer lock released

        {
            let mut s = Server::open(&path).expect("reopen");
            // Data survived the restart.
            let out = line(&mut s, "SELECT * FROM t;");
            assert!(out.contains("t:1"), "persisted row after reopen: {out}");
            // Pre-restart tables are analyzable again...
            assert_eq!(
                line(
                    &mut s,
                    r#"INSERT INTO t:2 { "text": "y" } EMBED [0.5, 0.5];"#
                ),
                "OK",
                "INSERT into a table declared before the restart"
            );
            // ...including the empty no-dim one (history-only state)...
            assert_eq!(
                line(&mut s, r#"INSERT INTO scratch:1 { "n": 1 };"#),
                "OK",
                "empty table must be re-seeded from history"
            );
            // ...and the memory-scoped one.
            assert_eq!(
                line(&mut s, r#"MEMORY m; INSERT INTO mt:2 { "v": 2 };"#),
                "OK",
                "memory-scoped table must be re-seeded"
            );
            let out = line(&mut s, "MEMORY m; SELECT * FROM mt;");
            assert!(
                out.contains("mt:1") && out.contains("mt:2"),
                "memory rows: {out}"
            );
            let out = line(&mut s, "SELECT * FROM scratch;");
            assert!(out.contains("scratch:1"), "scratch rows: {out}");
        }

        // Third open: pins the #109 fix at the server level — scope 2's
        // root writes (t:2, scratch:1) were logged AFTER a plan that ended
        // inside memory m (scope 1's last line); replay must put them at
        // root, and the memory store must not gain them.
        {
            let mut s = Server::open(&path).expect("third open");
            let out = line(&mut s, "SELECT * FROM t;");
            assert!(
                out.contains("t:1") && out.contains("t:2"),
                "root writes after a memory-ending plan must survive replay: {out}"
            );
            let out = line(&mut s, "SELECT * FROM scratch;");
            assert!(out.contains("scratch:1"), "scratch survives replay: {out}");
            let out = line(&mut s, "MEMORY m; SELECT * FROM mt;");
            assert!(
                out.contains("mt:1") && out.contains("mt:2"),
                "memory holds its own rows: {out}"
            );
            // Row-shaped leak check ("mt:2" contains "t:2" as a substring,
            // so a bare contains() would false-positive on the last id).
            let t2_leaked = out.contains(": t:2 score=") || out.contains("; t:2 score=");
            assert!(
                !t2_leaked,
                "root t:2 must not appear in the memory store: {out}"
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn prune_history_preserves_reseed_across_reopen() {
        // Issue #95 at the server level: compaction keeps the CreateTable
        // declarations (issue #89) — an empty, no-dim table must survive a
        // PRUNE + reopen, and the pruned window must fail loudly on the line
        // protocol instead of silently returning a partial view.
        let dir = std::env::temp_dir().join(format!("nqlite-server-prune-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("store.nql");

        {
            let mut s = Server::open(&path).expect("first open");
            assert_eq!(line(&mut s, "CREATE TABLE scratch;"), "OK"); // decl-only
            assert_eq!(line(&mut s, "CREATE TABLE t;"), "OK");
            assert_eq!(line(&mut s, r#"INSERT INTO t:1 { "n": 1 };"#), "OK");
            assert_eq!(line(&mut s, "PRUNE HISTORY"), "OK");
            // Live insert into the declaration-only table still works.
            assert_eq!(
                line(&mut s, r#"INSERT INTO scratch:1 { "n": 2 };"#),
                "OK",
                "declaration retained after prune"
            );
        }

        {
            let mut s = Server::open(&path).expect("reopen");
            // Re-seeded from the retained declarations in the pruned history.
            assert_eq!(
                line(&mut s, r#"INSERT INTO scratch:2 { "n": 3 };"#),
                "OK",
                "empty table re-seeded after prune + reopen"
            );
            let out = line(&mut s, "SELECT * FROM scratch;");
            assert!(
                out.contains("scratch:1") && out.contains("scratch:2"),
                "scratch rows: {out}"
            );
            // Pre-snapshot AS OF errors loudly …
            let err = line(&mut s, "SELECT * FROM t AS OF 1;");
            assert!(err.starts_with("ERR"), "pruned window errors: {err}");
            assert!(err.contains("compacted"), "error explains why: {err}");
            // … while current reads are unaffected.
            let out = line(&mut s, "SELECT * FROM t;");
            assert!(out.contains("t:1"), "current rows unaffected: {out}");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn error_line_returns_err_and_session_survives() {
        let mut s = Server::new();
        let err = line(&mut s, "THIS IS NOT NQL");
        assert!(err.starts_with("ERR "), "parse error reported, got: {err}");
        assert!(!err.contains('\n'), "single-line ERR response");

        // SELECT before any CREATE TABLE fails analysis, not execution.
        let err2 = line(&mut s, "SELECT * FROM nowhere");
        assert!(
            err2.starts_with("ERR "),
            "analysis error reported, got: {err2}"
        );

        // The shared database still works afterwards.
        line(&mut s, "CREATE TABLE t");
        assert_eq!(line(&mut s, "INSERT INTO t:1 {}"), "OK");
        let out = line(&mut s, "SELECT * FROM t");
        assert!(out.contains("t:1"), "session survived errors, got: {out}");
    }

    #[test]
    fn multi_statement_line_returns_per_select_lines() {
        let mut s = Server::new();
        line(&mut s, "CREATE TABLE a");
        line(&mut s, "CREATE TABLE b");
        line(&mut s, "INSERT INTO a:1 {}");
        line(&mut s, "INSERT INTO b:2 {}");
        let out = line(&mut s, "SELECT * FROM a; SELECT * FROM b;");
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 3, "two SELECT lines + OK, got: {out}");
        assert!(lines[0].starts_with("SELECT a (1 rows): a:1"));
        assert!(lines[1].starts_with("SELECT b (1 rows): b:2"));
        assert_eq!(lines[2], "OK");
    }

    #[test]
    fn empty_and_whitespace_lines_are_noops() {
        let mut s = Server::new();
        assert_eq!(line(&mut s, ""), "OK");
        assert_eq!(line(&mut s, "   \t  "), "OK");
        assert_eq!(line(&mut s, "\n"), "OK");
    }

    #[test]
    fn cross_line_vector_dim_contract_is_enforced() {
        let mut s = Server::new();
        line(&mut s, "CREATE TABLE v VECTOR<f32, 2>");
        let err = line(&mut s, "INSERT INTO v:1 {} EMBED [1.0, 0.0, 0.0]");
        assert!(
            err.starts_with("ERR "),
            "dimension mismatch rejected, got: {err}"
        );
    }

    #[test]
    fn forget_and_relate_across_lines() {
        let mut s = Server::new();
        line(&mut s, "CREATE TABLE n");
        line(&mut s, "INSERT INTO n:1 {}");
        line(&mut s, "INSERT INTO n:2 {}");
        line(&mut s, "RELATE (n:1) -> :refs -> (n:2) SET weight = 0.7");
        line(&mut s, "FORGET n:2");
        let out = line(&mut s, "SELECT * FROM n");
        assert!(
            out.contains("SELECT n (1 rows): n:1"),
            "only n:1 remains, got: {out}"
        );
    }
}
