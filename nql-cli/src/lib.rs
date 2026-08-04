//! nql REPL + script runner for nqlite.
//!
//! Deterministic and zero-LLM: the CLI only parses nql (via `nql`) and executes
//! it against an in-memory `nqlite::Database`. No network, no randomness, no
//! wall-clock appears in output. Output ordering is deterministic (BTree key
//! order; fields in BTreeMap order; vectors truncated to 4 values).

use std::io::Write;

use nql::parse;
use nql_ir::{RecordId, Store, Value};
use nqlite::Database;

/// Result of running one session (each input line / complete statement).
pub struct Session {
    db: Database,
    out: Box<dyn Write>,
}

impl Session {
    /// Start a fresh deterministic session.
    pub fn new(out: Box<dyn Write>) -> Self {
        Session {
            db: Database::new(Store::default()),
            out,
        }
    }

    /// Run a full input (possibly multi-statement, `;`-separated or with
    /// newlines). Returns the number of statements executed. Deterministic.
    pub fn run(&mut self, input: &str) -> std::io::Result<usize> {
        // Skip pure whitespace/comment lines and trailing empty.
        if input.trim().is_empty() {
            return Ok(0);
        }
        let plan = match parse(input) {
            Ok(p) => p,
            Err(e) => {
                writeln!(self.out, "error: {e}")?;
                return Ok(0);
            }
        };
        let results = self
            .db
            .execute(&plan)
            .map_err(|e| std::io::Error::other(format!("execute error: {e}")))?;
        for r in &results {
            self.print_result(r)?;
        }
        Ok(plan.len())
    }

    fn print_result(&mut self, r: &nqlite::QueryResult) -> std::io::Result<()> {
        let label = match &r.kind {
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
        };
        writeln!(self.out, "{label} ({} rows)", r.rows.len())?;
        for s in &r.rows {
            let id = s.record.id.to_string();
            let fields = format_fields(&s.record.body);
            let score = s.score;
            writeln!(self.out, "  {id}  score={score:.4}  {fields}")?;
        }
        Ok(())
    }

    /// Print the whole store: records, edges, vector dims (deterministic order).
    pub fn dump_store(&mut self) -> std::io::Result<()> {
        let st = self.db.store();
        writeln!(self.out, "tables:")?;
        for (t, dim) in &st.vector_dims {
            writeln!(self.out, "  {t}  VECTOR<f32,{dim}>")?;
        }
        writeln!(self.out, "records:")?;
        for rec in st.records.values() {
            let fields = format_fields(&rec.body);
            let emb = match &rec.embedding {
                Some(v) => format!(" emb=[{}]", slice4(v)),
                None => String::new(),
            };
            writeln!(self.out, "  {}  {}{}", rec.id, fields, emb)?;
        }
        writeln!(self.out, "edges:")?;
        for e in &st.edges {
            writeln!(
                self.out,
                "  {} -[:{}]-> {}  w={:?}",
                e.from, e.name, e.to, e.weight
            )?;
        }
        Ok(())
    }

    /// Print a raw line through the session's writer (e.g. banners, help).
    pub fn writeln(&mut self, s: &str) -> std::io::Result<()> {
        writeln!(self.out, "{s}")
    }

    /// Raw access to the underlying writer.
    pub fn writer(&mut self) -> &mut dyn Write {
        &mut self.out
    }

    /// Flush the session's writer.
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.out.flush()
    }

    /// Clear to a fresh empty database.
    pub fn clear(&mut self) {
        self.db = Database::new(Store::default());
    }
}

fn format_fields(body: &std::collections::BTreeMap<String, Value>) -> String {
    // Deterministic: BTreeMap iteration order.
    let parts: Vec<String> = body
        .iter()
        .map(|(k, v)| format!("{k}={}", short_value(v)))
        .collect();
    if parts.is_empty() {
        "{}".into()
    } else {
        format!("{{{}}}", parts.join(", "))
    }
}

fn short_value(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Int(i) => i.to_string(),
        Value::Float(f) => format!("{f}"),
        Value::Str(s) => format!("{s:?}"),
        Value::Doc(d) => format_fields(d),
        Value::Arr(a) => {
            // nested array: show first elements only
            let s: Vec<String> = a.iter().take(3).map(short_value).collect();
            format!("[{}]", s.join(", "))
        }
        Value::Vector(v) => format!("[{}]", slice4(v)),
        Value::Ref(rid) => rid.to_string(),
    }
}

fn slice4(v: &[f32]) -> String {
    let shown: Vec<String> = v.iter().take(4).map(|x| format!("{x}")).collect();
    let more = if v.len() > 4 { ", ..." } else { "" };
    format!("{}{}", shown.join(", "), more)
}

/// Format a record id for user display.
pub fn display_id(rid: &RecordId) -> String {
    rid.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn session_insert_then_select() {
        let out = run_session_text(
            "CREATE TABLE t VECTOR<f32, 2>;
             INSERT INTO t:1 { \"name\": \"a\" } EMBED [1.0, 0.0];
             INSERT INTO t:2 { \"name\": \"b\" } EMBED [0.9, 0.1];
             SELECT * FROM t WHERE vector::similarity(embedding, [1.0, 0.0]) AND k = 1 ORDER BY ::similarity;",
        );
        assert!(out.contains("t:1"), "nearest is t:1, got:\n{out}");
        assert!(out.contains("name=\"a\""), "field present, got:\n{out}");
    }

    #[test]
    fn parse_error_is_reported_not_panic() {
        let out = run_session_text("THIS IS NOT NQL");
        assert!(out.contains("error:"), "got:\n{out}");
    }

    #[test]
    fn script_mode_with_forget_and_relate() {
        let out = run_session_text(
            "CREATE TABLE n;
             INSERT INTO n:1 { \"a\": 1 };
             INSERT INTO n:2 { \"a\": 2 };
             RELATE (n:1) -> :refs -> (n:2) SET weight = 0.7;
             FORGET n:2;
             SELECT * FROM n;",
        );
        assert!(
            out.contains("SELECT n (1"),
            "one row after forget, got:\n{out}"
        );
    }

    /// Run a session over a string, returning everything written.
    fn run_session_text(input: &str) -> String {
        let buf = TestBuf(Rc::new(RefCell::new(Vec::new())));
        let out;
        {
            let mut s = Session::new(Box::new(buf.clone()));
            s.run(input).unwrap();
            out = String::from_utf8(buf.0.borrow().clone()).unwrap();
        }
        out
    }

    /// Owned `Write` that captures into a shared `Vec<u8>` (so the buffered
    /// client borrow doesn't outlive the buffer).
    #[derive(Clone)]
    struct TestBuf(Rc<RefCell<Vec<u8>>>);

    impl Write for TestBuf {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.borrow_mut().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
}
