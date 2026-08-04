//! Shared Plan/IR and value types — produced by `nql` (front-end), executed by
//! `nqlite` (engine). This crate is the CONTRACT between the two halves.
//!
//! Zero-LLM contract: nothing in this crate (or anywhere in nqlite) ever calls
//! an embedding model, an LLM, or any network. Vectors arrive as plain `f32`
//! arrays supplied by the client (BYO-vector). This is a hard guarantee.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

/// A record identifier: `table:id`, SurrealDB-style. `id` may be numeric or a
/// string. `table` groups records into a logical collection.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RecordId {
    pub table: String,
    pub id: Id,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Id {
    Num(u64),
    Str(String),
}

impl RecordId {
    pub fn new(table: impl Into<String>, id: Id) -> Self {
        Self {
            table: table.into(),
            id,
        }
    }
    /// Parse `"table:id"` (id numeric or a bare string). Used by the nql
    /// front-end and by tests; fails on malformed input.
    pub fn parse(s: &str) -> Option<Self> {
        let (table, id) = s.split_once(':')?;
        if table.is_empty() || id.is_empty() {
            return None;
        }
        let id = match id.parse::<u64>() {
            Ok(n) => Id::Num(n),
            Err(_) => Id::Str(id.to_string()),
        };
        Some(Self {
            table: table.to_string(),
            id,
        })
    }
}

impl fmt::Display for RecordId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.id {
            Id::Num(n) => write!(f, "{}:{}", self.table, n),
            Id::Str(s) => write!(f, "{}:{}", self.table, s),
        }
    }
}

/// A typed value inside a record's document body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    /// Nested document (map).
    Doc(BTreeMap<String, Value>),
    /// Ordered array.
    Arr(Vec<Value>),
    /// An embedded vector (BYO — the engine never computes these).
    Vector(Vec<f32>),
    /// A reference to another record (used inside documents and edge props).
    Ref(RecordId),
}

/// A fixed-dimension vector column declaration, e.g. `VECTOR<f32, 384>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorSpec {
    pub dim: usize,
}

/// A named, directed relation: `(from) -[:name {props}]-> (to)`.
/// Time and weight are first-class edge properties (per Zep research: agents
/// need "started_on/ended_on", confidence, provenance on edges).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelationEdge {
    pub from: RecordId,
    pub name: String,
    pub to: RecordId,
    /// Wall-clock time the edge was created (engine clocks itself; deterministic
    /// per-transaction).
    pub created_at: i64,
    /// Optional agent-supplied weight/confidence (e.g. 0.0..=1.0).
    pub weight: Option<f32>,
    /// Optional agent-supplied provenance/note.
    pub props: BTreeMap<String, Value>,
}

/// A record as stored by the engine: id, document body, and optional embedding.
/// The embedding is a separate field (not inside the doc) so the engine can
/// index it efficiently without scanning document values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    pub id: RecordId,
    pub body: BTreeMap<String, Value>,
    /// Optional embedding vector for this record (dim = table's VECTOR spec).
    pub embedding: Option<Vec<f32>>,
    /// Created-at wall-clock (engine-managed, per-transaction deterministic).
    pub created_at: i64,
}

/// The full "database" snapshot a query runs against. In M0 this is in-memory;
/// M1 makes it the on-disk single-file store + WAL.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Store {
    pub records: BTreeMap<RecordId, Record>,
    pub edges: Vec<RelationEdge>,
    /// Per-table vector dimensions once declared.
    pub vector_dims: BTreeMap<String, usize>,
}

impl Store {
    pub fn insert(&mut self, rec: Record) {
        // Keep insertion deterministic: preserve first-seen order via BTree key
        // ordering (BTreeMap is already deterministic). Edge list is append-only
        // in insertion order, which is deterministic per transaction.
        self.records.insert(rec.id.clone(), rec);
    }
}

/// Minimal value-type smoke test so `cargo test` has a canonical harness target
/// before the real engine tests land in Milestone 0.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_id_parse_roundtrip() {
        for s in ["person:1", "person:alice", "msg:42"] {
            let rid = RecordId::parse(s).expect("parse");
            assert_eq!(rid.to_string(), s, "display roundtrips parse");
        }
        assert!(RecordId::parse("").is_none());
        assert!(RecordId::parse("nocolon").is_none());
    }

    #[test]
    fn store_insert_is_deterministic_and_sorted() {
        let mut s = Store::default();
        for (t, id) in [("b", Id::Num(2)), ("a", Id::Num(1)), ("a", Id::Num(1))] {
            s.insert(Record {
                id: RecordId::new(t, id),
                body: BTreeMap::new(),
                embedding: None,
                created_at: 0,
            });
        }
        // Duplicate insert overwrites; BTreeMap keeps key order.
        assert_eq!(s.records.len(), 2);
        let keys: Vec<_> = s.records.keys().map(|k| k.to_string()).collect();
        assert_eq!(keys, ["a:1", "b:2"]);
    }

    #[test]
    fn value_types_serialize_roundtrip_json() {
        // The IR is the shared contract between nql and nqlite; it must be
        // serializable so the contract can outlive a process boundary later.
        let rec = Record {
            id: RecordId::parse("note:42").unwrap(),
            body: BTreeMap::from([
                ("text".into(), Value::Str("hello".into())),
                ("vector".into(), Value::Vector(vec![0.1, 0.2, 0.3])),
                (
                    "ref".into(),
                    Value::Ref(RecordId::parse("person:alice").unwrap()),
                ),
            ]),
            embedding: Some(vec![0.1, 0.2, 0.3]),
            created_at: 7,
        };
        let json = serde_json::to_string(&rec).expect("serialize");
        // Round-trip equality is the real contract test: the IR must survive a
        // process boundary unchanged. (Tagged-enum JSON is verbose by design;
        // e.g. Value::Str serializes as {"Str":"..."} — see nql-ir::Value.)
        let back: Record = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, rec);
    }
}

// ---------------------------------------------------------------------------
// The Plan / Statement contract (the seam between nql and nqlite)
// ---------------------------------------------------------------------------
// A `Plan` is what `nql` (front-end) produces and `nqlite` (engine) executes.
// It lives in nql-ir so both halves compile against the SAME types. Any change
// to this section is a contract change: update nql and nqlite in lockstep.

/// A complete nql statement (M0 slice).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Statement {
    /// Declare a table and optional vector dimension for its embedding column.
    CreateTable {
        table: String,
        vector_dim: Option<usize>,
    },
    /// Insert/upsert a record (BYO vector: `embedding` field, never computed here).
    Insert(Record),
    /// Create a named, directed relation edge.
    Relate(RelationEdge),
    /// Select records (optionally kNN + filter + order + limit).
    Select(Select),
    /// Delete a record (and its incident edges).
    Forget { id: RecordId },
}

/// A SELECT with optional vector kNN, field filter, deterministic ordering, limit.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Select {
    pub table: String,
    pub knn: Option<Knn>,
    pub filter: Option<Filter>,
    pub order: Option<Order>,
    pub limit: Option<usize>,
}

/// Vector kNN clause: `WHERE vector::similarity(embedding, $q) AND k = N`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Knn {
    pub query: Vec<f32>,
    pub k: usize,
}

/// Field-level filter on record body values (M0: equality only).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Filter {
    /// `WHERE <field> = <value>`
    FieldEquals { field: String, value: Value },
    /// `WHERE embedding IS NOT NULL` — only records with vectors.
    HasEmbedding,
    /// `WHERE ::bm25(<field>, "<query>") [AND k = <N>]` — deterministic BM25
    /// lexical scoring over the given text field. Every row of the table is
    /// returned (this filter scores rather than prunes); rows are ordered by
    /// descending BM25 score with ties broken by ascending `RecordId`, and
    /// `k`, when present, caps the number of returned rows.
    Bm25 {
        /// Body field whose `Value::Str` content is tokenized and scored.
        field: String,
        /// Raw query text; tokenized identically to document content.
        query: String,
        /// Optional result cap (like `knn.k`). `None` = no cap from the filter.
        k: Option<usize>,
    },
}

/// Deterministic ordering operators (pure arithmetic — see docs/decisions.md D9).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Order {
    /// Cosine similarity (descending) vs the kNN query vector.
    Similarity,
    /// α·similarity + β·strength(recency,freq) + γ·importance + δ·feedback (agent-tuned α..δ).
    Salience,
    /// Laplace-smoothed mean of `:voted` edge values on the record.
    Score,
    /// Net up−down vote count over `:voted` edges (descending; tie-break by RecordId).
    Votes,
    /// Time-decayed recent feedback over `:voted` edges (descending; tie-break by RecordId).
    Feedback,
    /// created_at (descending).
    Recency,
}

/// Aggregated vote counts over a record's `:voted` edges (`(voter)->:voted {value:+1|-1}->(record)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct VoteCounts {
    /// Edges with `value == +1`.
    pub up: u64,
    /// Edges with `value == -1`.
    pub down: u64,
    /// `up - down`.
    pub net: i64,
}

/// A plan is a sequence of statements executed atomically (one transaction).
pub type Plan = Vec<Statement>;
