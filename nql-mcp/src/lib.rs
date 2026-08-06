//! nql-mcp: expose nqlite to AI agents over the Model Context Protocol.
//!
//! The server holds ONE shared [`nqlite::Database`] (single-writer Mutex —
//! same deterministic snapshot semantics as the engine) and exposes it as MCP
//! tools:
//!
//! - `execute_nql` — run any nql program (`;`-separated), get results as JSON
//! - `create_table`, `insert_record`, `relate`, `forget` — DDL/DML
//! - `select`, `match_path` — reads with deterministic ordering
//!
//! Every tool is a pure function of the current store: no randomness, no
//! wall-clock, no LLM. The engine's zero-LLM / byte-determinism guarantees
//! hold end to end; this crate only adapts the MCP transport.
//!
//! Transport: stdio (the MCP default for local agents). Run:
//! `nql-mcp [--db <file>]`

use std::sync::Mutex;

use nql::parse;
use nql_ir::{Record, RecordId, Store, Value};
use nqlite::Database;
use rmcp::{handler::server::wrapper::Parameters, schemars, tool, tool_router};

/// The MCP service: a shared nqlite database behind a single-writer lock.
#[derive(Clone)]
pub struct NqlMcp {
    db: std::sync::Arc<Mutex<Database>>,
}

impl Default for NqlMcp {
    fn default() -> Self {
        Self::new()
    }
}

impl NqlMcp {
    /// Start a fresh in-memory database.
    pub fn new() -> Self {
        Self {
            db: std::sync::Arc::new(Mutex::new(Database::new(Store::default()))),
        }
    }

    /// Open (or create) a persistent single-file database (see
    /// `spec/file-format.md`).
    pub fn open(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let db = Database::open(path)?;
        Ok(Self {
            db: std::sync::Arc::new(Mutex::new(db)),
        })
    }

    /// Run one nql program against the store and render every result row as
    /// JSON. Deterministic: rows in engine order, scores included.
    fn run_nql(&self, program: &str) -> anyhow::Result<serde_json::Value> {
        let plan = parse(program)?;
        let mut db = self.db.lock().unwrap();
        let results = db.execute(&plan)?;
        let out: Vec<serde_json::Value> = results
            .iter()
            .map(|res| {
                let kind = match &res.kind {
                    nqlite::QueryKind::Select(sel) => format!("SELECT {}", sel.table),
                    nqlite::QueryKind::Match(p) => format!("MATCH {}", p.start),
                    nqlite::QueryKind::Closure(p) => format!("CLOSURE {}", p.start),
                };
                serde_json::json!({
                    "kind": kind,
                    "rows": res.rows.iter().map(|r| serde_json::json!({
                        "id": r.record.id.to_string(),
                        "score": r.score,
                        "body": value_to_json(&r.record.body),
                    })).collect::<Vec<_>>(),
                })
            })
            .collect();
        Ok(serde_json::Value::Array(out))
    }
}

/// Convert an nql body map into a JSON value (deterministic key order via the
/// `BTreeMap` iteration order).
fn value_to_json(body: &std::collections::BTreeMap<String, Value>) -> serde_json::Value {
    serde_json::Value::Object(
        body.iter()
            .map(|(k, v)| (k.clone(), scalar_to_json(v)))
            .collect(),
    )
}

fn scalar_to_json(v: &Value) -> serde_json::Value {
    match v {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::Int(n) => serde_json::Value::Number((*n).into()),
        Value::Float(f) => serde_json::Number::from_f64(*f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        Value::Str(s) => serde_json::Value::String(s.clone()),
        Value::Doc(d) => value_to_json(d),
        Value::Arr(a) => serde_json::Value::Array(a.iter().map(scalar_to_json).collect()),
        Value::Vector(v) => {
            serde_json::Value::Array(v.iter().map(|x| serde_json::json!(x)).collect())
        }
        Value::Ref(r) => serde_json::Value::String(r.to_string()),
    }
}

/// Parse a `table:id` string into a [`RecordId`].
fn parse_rid(s: &str) -> anyhow::Result<RecordId> {
    RecordId::parse(s).ok_or_else(|| anyhow::anyhow!("invalid record id `{s}` (expected table:id)"))
}

/// Parse a JSON array into an `f32` embedding.
fn parse_embedding(v: &serde_json::Value) -> anyhow::Result<Vec<f32>> {
    let arr = v
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("embedding must be a JSON array of numbers"))?;
    arr.iter()
        .map(|x| {
            x.as_f64()
                .map(|f| f as f32)
                .ok_or_else(|| anyhow::anyhow!("embedding entries must be numbers"))
        })
        .collect()
}

/// Convert a JSON object into an nql body map.
fn parse_body(v: &serde_json::Value) -> anyhow::Result<std::collections::BTreeMap<String, Value>> {
    let obj = v
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("body must be a JSON object"))?;
    obj.iter()
        .map(|(k, v)| Ok((k.clone(), json_to_value(v)?)))
        .collect()
}

fn json_to_value(v: &serde_json::Value) -> anyhow::Result<Value> {
    Ok(match v {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else {
                Value::Float(n.as_f64().unwrap_or(0.0))
            }
        }
        serde_json::Value::String(s) => Value::Str(s.clone()),
        serde_json::Value::Array(a) => {
            Value::Arr(a.iter().map(json_to_value).collect::<anyhow::Result<_>>()?)
        }
        serde_json::Value::Object(o) => {
            let mut m = std::collections::BTreeMap::new();
            for (k, v) in o {
                m.insert(k.clone(), json_to_value(v)?);
            }
            Value::Doc(m)
        }
    })
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

/// Tool parameters: run an arbitrary nql program.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ExecuteNqlParams {
    /// `;`-separated nql statements (CREATE/INSERT/RELATE/SELECT/MATCH/CLOSURE/FORGET).
    pub program: String,
}

/// Tool parameters: create a table.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct CreateTableParams {
    /// Table name.
    pub table: String,
    /// Optional fixed embedding dimension (`VECTOR<f32, N>`).
    pub vector_dim: Option<usize>,
}

/// Tool parameters: insert/upsert a record.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct InsertParams {
    /// Record id in `table:id` form.
    pub id: String,
    /// Record body as a JSON object.
    pub body: serde_json::Value,
    /// Optional embedding as a JSON array of numbers.
    pub embedding: Option<serde_json::Value>,
}

/// Tool parameters: create a named edge.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct RelateParams {
    /// Source record id (`table:id`).
    pub from: String,
    /// Edge name (with or without leading `:`).
    pub name: String,
    /// Target record id (`table:id`).
    pub to: String,
    /// Optional edge weight (0..=1).
    pub weight: Option<f32>,
    /// Optional edge properties as a JSON object.
    pub props: Option<serde_json::Value>,
}

/// Tool parameters: delete a record and its incident edges.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ForgetParams {
    /// Record id in `table:id` form.
    pub id: String,
}

/// Tool parameters: SELECT with optional kNN / filter / order / limit.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SelectParams {
    /// Table to scan.
    pub table: String,
    /// Optional `WHERE <field> = <value>` (JSON scalar).
    pub field: Option<String>,
    pub value: Option<serde_json::Value>,
    /// Optional kNN query vector (JSON array) — enables similarity ranking.
    pub query: Option<serde_json::Value>,
    /// k for kNN when `query` is given.
    pub k: Option<usize>,
    /// Optional `ORDER BY ::<op>`: similarity | salience | score | votes | feedback | recency.
    pub order_by: Option<String>,
    /// Optional row cap.
    pub limit: Option<usize>,
}

/// Tool parameters: MATCH graph traversal.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct MatchParams {
    /// Start record id (`table:id`).
    pub start: String,
    /// Path steps as JSON: `[{ "direction": "out"|"in", "name": "mentions" }, ...]`.
    pub steps: serde_json::Value,
}

#[tool_router(server_handler)]
impl NqlMcp {
    /// Run an arbitrary nql program and return every result as JSON.
    #[tool(
        description = "Run a full nql program (CREATE/INSERT/RELATE/SELECT/MATCH/CLOSURE/FORGET, ';'-separated) and return all result rows as JSON."
    )]
    async fn execute_nql(
        &self,
        Parameters(ExecuteNqlParams { program }): Parameters<ExecuteNqlParams>,
    ) -> String {
        match self.run_nql(&program) {
            Ok(v) => serde_json::to_string_pretty(&v).unwrap_or_else(|_| "{}".into()),
            Err(e) => format!("ERR {e}"),
        }
    }

    /// Create a table (optionally with a fixed embedding dimension).
    #[tool(
        description = "Declare a table, optionally with VECTOR<f32, N> dimension for embeddings."
    )]
    async fn create_table(
        &self,
        Parameters(CreateTableParams { table, vector_dim }): Parameters<CreateTableParams>,
    ) -> String {
        let stmt = nql_ir::Statement::CreateTable { table, vector_dim };
        let mut db = self.db.lock().unwrap();
        match db.execute(&[stmt]) {
            Ok(_) => "OK".into(),
            Err(e) => format!("ERR {e}"),
        }
    }

    /// Insert (or upsert) a record with optional BYO embedding.
    #[tool(
        description = "Insert or upsert a record. Embedding is BYO: pass a JSON array of numbers."
    )]
    async fn insert_record(
        &self,
        Parameters(InsertParams {
            id,
            body,
            embedding,
        }): Parameters<InsertParams>,
    ) -> String {
        let rid = match parse_rid(&id) {
            Ok(r) => r,
            Err(e) => return format!("ERR {e}"),
        };
        let body = match parse_body(&body) {
            Ok(b) => b,
            Err(e) => return format!("ERR {e}"),
        };
        let embedding = match embedding {
            Some(v) => match parse_embedding(&v) {
                Ok(e) => Some(e),
                Err(e) => return format!("ERR {e}"),
            },
            None => None,
        };
        let rec = Record {
            id: rid,
            body,
            embedding,
            created_at: 0,
        };
        let stmt = nql_ir::Statement::Insert(rec);
        let mut db = self.db.lock().unwrap();
        match db.execute(&[stmt]) {
            Ok(_) => "OK".into(),
            Err(e) => format!("ERR {e}"),
        }
    }

    /// Relate two records with a named edge (weight + props optional).
    #[tool(
        description = "Create a directed, named edge (from) -> :name -> (to), with optional weight and properties."
    )]
    async fn relate(
        &self,
        Parameters(RelateParams {
            from,
            name,
            to,
            weight,
            props,
        }): Parameters<RelateParams>,
    ) -> String {
        let from = match parse_rid(&from) {
            Ok(r) => r,
            Err(e) => return format!("ERR {e}"),
        };
        let to = match parse_rid(&to) {
            Ok(r) => r,
            Err(e) => return format!("ERR {e}"),
        };
        let name = name.trim_start_matches(':').to_string();
        let props = match props {
            Some(v) => match parse_body(&v) {
                Ok(b) => b,
                Err(e) => return format!("ERR {e}"),
            },
            None => std::collections::BTreeMap::new(),
        };
        let edge = nql_ir::RelationEdge {
            from,
            name,
            to,
            created_at: 0,
            weight,
            props,
        };
        let stmt = nql_ir::Statement::Relate(edge);
        let mut db = self.db.lock().unwrap();
        match db.execute(&[stmt]) {
            Ok(_) => "OK".into(),
            Err(e) => format!("ERR {e}"),
        }
    }

    /// Delete a record and its incident edges.
    #[tool(description = "Delete a record (table:id) and every edge incident to it.")]
    async fn forget(&self, Parameters(ForgetParams { id }): Parameters<ForgetParams>) -> String {
        let rid = match parse_rid(&id) {
            Ok(r) => r,
            Err(e) => return format!("ERR {e}"),
        };
        let stmt = nql_ir::Statement::Forget { id: rid };
        let mut db = self.db.lock().unwrap();
        match db.execute(&[stmt]) {
            Ok(_) => "OK".into(),
            Err(e) => format!("ERR {e}"),
        }
    }

    /// SELECT with optional kNN / equality filter / ORDER BY / LIMIT.
    #[tool(
        description = "Scan a table, optionally filter by field equality, rank by kNN similarity, order, and limit. Returns rows as JSON in deterministic order."
    )]
    async fn select(
        &self,
        Parameters(SelectParams {
            table,
            field,
            value,
            query,
            k,
            order_by,
            limit,
        }): Parameters<SelectParams>,
    ) -> String {
        use nql_ir::{Filter, Knn, Order, Select, Statement};
        let knn = match query {
            Some(q) => match parse_embedding(&q) {
                Ok(vec) => Some(Knn {
                    query: vec,
                    k: k.unwrap_or(10).max(1),
                }),
                Err(e) => return format!("ERR {e}"),
            },
            None => None,
        };
        let filter = match (field, value) {
            (Some(f), Some(v)) => match json_to_value(&v) {
                Ok(val) => Some(Filter::FieldEquals {
                    field: f,
                    value: val,
                }),
                Err(e) => return format!("ERR {e}"),
            },
            _ => None,
        };
        let order = match order_by {
            Some(o) => match o.to_ascii_lowercase().as_str() {
                "similarity" => Some(Order::Similarity),
                "salience" => Some(Order::Salience),
                "score" => Some(Order::Score),
                "votes" => Some(Order::Votes),
                "feedback" => Some(Order::Feedback),
                "recency" => Some(Order::Recency),
                other => return format!("ERR unknown ORDER BY `{other}`"),
            },
            None => None,
        };
        let stmt = Statement::Select(Select {
            table,
            knn,
            filter,
            order,
            limit,
            as_of: None,
        });
        let mut db = self.db.lock().unwrap();
        match db.execute(&[stmt]) {
            Ok(results) => {
                let rows: Vec<serde_json::Value> = results[0]
                    .rows
                    .iter()
                    .map(|r| {
                        serde_json::json!({
                            "id": r.record.id.to_string(),
                            "score": r.score,
                            "body": value_to_json(&r.record.body),
                        })
                    })
                    .collect();
                serde_json::to_string_pretty(&serde_json::json!({ "rows": rows }))
                    .unwrap_or_else(|_| "{}".into())
            }
            Err(e) => format!("ERR {e}"),
        }
    }

    /// MATCH: walk a graph path from a start record.
    #[tool(
        description = "Walk a named-edge path from a start record (1+ hops, out/in, optional per-step edge-property filter) and return the reached records."
    )]
    async fn match_path(
        &self,
        Parameters(MatchParams { start, steps }): Parameters<MatchParams>,
    ) -> String {
        let rid = match parse_rid(&start) {
            Ok(r) => r,
            Err(e) => return format!("ERR {e}"),
        };
        let steps = match parse_steps(&steps) {
            Ok(s) => s,
            Err(e) => return format!("ERR {e}"),
        };
        let stmt = nql_ir::Statement::Match(nql_ir::MatchPath { start: rid, steps });
        let mut db = self.db.lock().unwrap();
        match db.execute(&[stmt]) {
            Ok(results) => {
                let rows: Vec<serde_json::Value> = results[0]
                    .rows
                    .iter()
                    .map(|r| {
                        serde_json::json!({
                            "id": r.record.id.to_string(),
                            "score": r.score,
                            "body": value_to_json(&r.record.body),
                        })
                    })
                    .collect();
                serde_json::to_string_pretty(&serde_json::json!({ "rows": rows }))
                    .unwrap_or_else(|_| "{}".into())
            }
            Err(e) => format!("ERR {e}"),
        }
    }

    /// CLOSURE: transitive traversal from a start record.
    #[tool(
        description = "Transitive closure: every record reachable from a start record via the named edges (any number of hops, BFS to fixpoint). Scored by BFS depth (0 = start)."
    )]
    async fn closure(
        &self,
        Parameters(MatchParams { start, steps }): Parameters<MatchParams>,
    ) -> String {
        let rid = match parse_rid(&start) {
            Ok(r) => r,
            Err(e) => return format!("ERR {e}"),
        };
        let steps = match parse_steps(&steps) {
            Ok(s) => s,
            Err(e) => return format!("ERR {e}"),
        };
        let stmt = nql_ir::Statement::Closure(nql_ir::MatchPath { start: rid, steps });
        let mut db = self.db.lock().unwrap();
        match db.execute(&[stmt]) {
            Ok(results) => {
                let rows: Vec<serde_json::Value> = results[0]
                    .rows
                    .iter()
                    .map(|r| {
                        serde_json::json!({
                            "id": r.record.id.to_string(),
                            "score": r.score,
                            "body": value_to_json(&r.record.body),
                        })
                    })
                    .collect();
                serde_json::to_string_pretty(&serde_json::json!({ "rows": rows }))
                    .unwrap_or_else(|_| "{}".into())
            }
            Err(e) => format!("ERR {e}"),
        }
    }
}

/// Parse the `steps` JSON for MATCH/CLOSURE:
/// `[{ "direction": "out"|"in", "name": "mentions", "where": {"field", "value"} }, ...]`.
fn parse_steps(v: &serde_json::Value) -> anyhow::Result<Vec<nql_ir::MatchStep>> {
    let arr = v
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("steps must be a JSON array"))?;
    if arr.is_empty() {
        anyhow::bail!("steps must contain at least one hop");
    }
    arr.iter()
        .map(|s| {
            let direction = match s.get("direction").and_then(|d| d.as_str()) {
                Some("out") => nql_ir::MatchDirection::Out,
                Some("in") => nql_ir::MatchDirection::In,
                Some(other) => anyhow::bail!("direction must be \"out\" or \"in\", got `{other}`"),
                None => anyhow::bail!("step missing `direction`"),
            };
            let name = s
                .get("name")
                .and_then(|n| n.as_str())
                .ok_or_else(|| anyhow::anyhow!("step missing `name`"))?
                .trim_start_matches(':')
                .to_string();
            let edge_props = match s.get("where") {
                Some(w) => {
                    let field = w
                        .get("field")
                        .and_then(|f| f.as_str())
                        .ok_or_else(|| anyhow::anyhow!("where missing `field`"))?
                        .to_string();
                    let value = w
                        .get("value")
                        .ok_or_else(|| anyhow::anyhow!("where missing `value`"))?;
                    Some(nql_ir::Filter::FieldEquals {
                        field,
                        value: json_to_value(value)?,
                    })
                }
                None => None,
            };
            Ok(nql_ir::MatchStep {
                direction,
                name,
                edge_props,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_service() -> NqlMcp {
        let s = NqlMcp::new();
        // Set up a small store through the tools themselves.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            s.create_table(Parameters(CreateTableParams {
                table: "turn".into(),
                vector_dim: Some(2),
            }))
            .await;
            s.insert_record(Parameters(InsertParams {
                id: "turn:1".into(),
                body: serde_json::json!({ "text": "hello world" }),
                embedding: Some(serde_json::json!([1.0, 0.0])),
            }))
            .await;
            s.insert_record(Parameters(InsertParams {
                id: "turn:2".into(),
                body: serde_json::json!({ "text": "goodbye world" }),
                embedding: Some(serde_json::json!([0.9, 0.1])),
            }))
            .await;
        });
        s
    }

    #[test]
    fn execute_nql_returns_json_rows() {
        let s = test_service();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let out = rt.block_on(async {
            s.execute_nql(Parameters(ExecuteNqlParams {
                program: "SELECT * FROM turn ORDER BY ::similarity".into(),
            }))
            .await
        });
        assert!(out.contains("\"turn:1\""), "nearest first: {out}");
        assert!(out.contains("\"turn:2\""));
        assert!(!out.contains("ERR"), "no error: {out}");
    }

    #[test]
    fn select_tool_ranks_by_similarity() {
        let s = test_service();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let out = rt.block_on(async {
            s.select(Parameters(SelectParams {
                table: "turn".into(),
                field: None,
                value: None,
                query: Some(serde_json::json!([1.0, 0.0])),
                k: Some(1),
                order_by: None,
                limit: None,
            }))
            .await
        });
        assert!(out.contains("\"turn:1\""), "top-1 is turn:1: {out}");
        assert!(!out.contains("\"turn:2\""), "k=1 caps: {out}");
    }

    #[test]
    fn relate_and_match_work_through_tools() {
        let s = NqlMcp::new();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            s.create_table(Parameters(CreateTableParams {
                table: "note".into(),
                vector_dim: None,
            }))
            .await;
            s.insert_record(Parameters(InsertParams {
                id: "note:1".into(),
                body: serde_json::json!({ "body": "alpha" }),
                embedding: None,
            }))
            .await;
            s.insert_record(Parameters(InsertParams {
                id: "note:2".into(),
                body: serde_json::json!({ "body": "beta" }),
                embedding: None,
            }))
            .await;
            s.relate(Parameters(RelateParams {
                from: "note:1".into(),
                name: "references".into(),
                to: "note:2".into(),
                weight: Some(0.5),
                props: None,
            }))
            .await;
        });
        let out = rt.block_on(async {
            s.match_path(Parameters(MatchParams {
                start: "note:1".into(),
                steps: serde_json::json!([{ "direction": "out", "name": "references" }]),
            }))
            .await
        });
        assert!(out.contains("\"note:2\""), "match returns target: {out}");
        assert!(!out.contains("ERR"), "no error: {out}");
    }

    #[test]
    fn error_returns_err_not_panic() {
        let s = NqlMcp::new();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let out = rt.block_on(async {
            s.execute_nql(Parameters(ExecuteNqlParams {
                program: "THIS IS NOT NQL".into(),
            }))
            .await
        });
        assert!(out.starts_with("ERR"), "parse errors surface: {out}");
    }

    #[test]
    fn closure_tool_reaches_transitive_neighborhood() {
        let s = NqlMcp::new();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            s.create_table(Parameters(CreateTableParams {
                table: "person".into(),
                vector_dim: None,
            }))
            .await;
            for id in ["1", "2", "3"] {
                s.insert_record(Parameters(InsertParams {
                    id: format!("person:{id}"),
                    body: serde_json::json!({ "n": id }),
                    embedding: None,
                }))
                .await;
            }
            s.relate(Parameters(RelateParams {
                from: "person:1".into(),
                name: "knows".into(),
                to: "person:2".into(),
                weight: None,
                props: None,
            }))
            .await;
            s.relate(Parameters(RelateParams {
                from: "person:2".into(),
                name: "knows".into(),
                to: "person:3".into(),
                weight: None,
                props: None,
            }))
            .await;
        });
        let out = rt.block_on(async {
            s.closure(Parameters(MatchParams {
                start: "person:1".into(),
                steps: serde_json::json!([{ "direction": "out", "name": "knows" }]),
            }))
            .await
        });
        // Closure: person:1 (depth 0), person:2 (1), person:3 (2).
        assert!(out.contains("\"person:1\""), "{out}");
        assert!(out.contains("\"person:2\""), "{out}");
        assert!(out.contains("\"person:3\""), "{out}");
        assert!(!out.contains("ERR"), "no error: {out}");
    }
}
