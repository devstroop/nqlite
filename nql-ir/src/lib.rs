//! Shared Plan/IR and value types — produced by `nql` (front-end), executed by
//! `nqlite` (engine). This crate is the CONTRACT between the two halves.
//!
//! Zero-LLM contract: nothing in this crate (or anywhere in nqlite) ever calls
//! an embedding model, an LLM, or any network. Vectors arrive as plain `f32`
//! arrays supplied by the client (BYO-vector). This is a hard guarantee.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
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

impl Value {
    /// Total order over [`Value`] for the comparison filters (issue #93;
    /// spec §2.3): type ranks `null < bool < number < string < array < doc <
    /// vector < ref`. Within a rank:
    ///
    /// - `Bool`: `false < true`;
    /// - numbers (`Int`/`Float`): numeric and **exact** — two ints compare as
    ///   ints, mixed int/float compare without rounding (so `1` and `1.0` are
    ///   equal even beyond 2⁵³), `NaN` sorts after every other number;
    /// - `Str`: lexicographic byte order;
    /// - `Arr`: element-wise, then shorter first;
    /// - `Doc`: (key, value) pairs in BTree key order, then fewer keys first;
    /// - `Vector`: element-wise ([`f32::total_cmp`]), then shorter first;
    /// - `Ref`: `(table, id)`.
    ///
    /// Deterministic and total: any two values compare, and the function
    /// never panics (proptest-verified in `nql::proptests`).
    pub fn cmp_total(&self, other: &Value) -> Ordering {
        fn rank(v: &Value) -> u8 {
            match v {
                Value::Null => 0,
                Value::Bool(_) => 1,
                Value::Int(_) | Value::Float(_) => 2,
                Value::Str(_) => 3,
                Value::Arr(_) => 4,
                Value::Doc(_) => 5,
                Value::Vector(_) => 6,
                Value::Ref(_) => 7,
            }
        }
        /// Exact `i64`-vs-`f64` comparison: never rounds the int through
        /// `f64` (which would break transitivity above 2⁵³). `NaN` sorts
        /// after every number, matching [`f64::total_cmp`] for floats.
        fn cmp_int_float(a: i64, b: &f64) -> Ordering {
            if b.is_nan() {
                return Ordering::Less;
            }
            // i64 as f64 has upper bound 2^63 (exclusive); beyond it every
            // float is greater than any i64.
            if *b >= i64::MAX as f64 {
                return Ordering::Less;
            }
            if *b < i64::MIN as f64 {
                return Ordering::Greater;
            }
            // In [-2^63, 2^63): casts below are exact.
            if b.fract() == 0.0 {
                return a.cmp(&(*b as i64));
            }
            // Non-integer: compare against its floor (also exact here).
            match a.cmp(&(b.floor() as i64)) {
                Ordering::Equal => Ordering::Less, // a == floor(b) < b
                other => other,
            }
        }
        let (ra, rb) = (rank(self), rank(other));
        if ra != rb {
            return ra.cmp(&rb);
        }
        match (self, other) {
            (Value::Null, Value::Null) => Ordering::Equal,
            (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
            (Value::Int(a), Value::Int(b)) => a.cmp(b),
            (Value::Int(a), Value::Float(b)) => cmp_int_float(*a, b),
            (Value::Float(a), Value::Int(b)) => cmp_int_float(*b, a).reverse(),
            (Value::Float(a), Value::Float(b)) => {
                if a == b {
                    // Equal values — including `-0.0 == 0.0`, which
                    // `total_cmp` would order: keeping zeros equal preserves
                    // transitivity with the Int↔Float comparisons above.
                    Ordering::Equal
                } else {
                    a.total_cmp(b) // NaN after every number; NaN == NaN
                }
            }
            (Value::Str(a), Value::Str(b)) => a.cmp(b),
            (Value::Arr(a), Value::Arr(b)) => {
                for (x, y) in a.iter().zip(b) {
                    let c = x.cmp_total(y);
                    if c != Ordering::Equal {
                        return c;
                    }
                }
                a.len().cmp(&b.len())
            }
            (Value::Doc(a), Value::Doc(b)) => {
                for ((ka, va), (kb, vb)) in a.iter().zip(b) {
                    let c = ka.cmp(kb);
                    if c != Ordering::Equal {
                        return c;
                    }
                    let c = va.cmp_total(vb);
                    if c != Ordering::Equal {
                        return c;
                    }
                }
                a.len().cmp(&b.len())
            }
            (Value::Vector(a), Value::Vector(b)) => {
                for (x, y) in a.iter().zip(b) {
                    let c = x.total_cmp(y);
                    if c != Ordering::Equal {
                        return c;
                    }
                }
                a.len().cmp(&b.len())
            }
            (Value::Ref(a), Value::Ref(b)) => a.cmp(b),
            // Ranks partition the enum, so same-rank pairs are always the
            // same variant (handled above); this arm keeps the match total
            // without ever panicking.
            _ => Ordering::Equal,
        }
    }
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
    /// Traverse the graph from a start record along named edges (1+ hops).
    /// Deterministic: endpoints are collected in edge-append order, deduped by
    /// `RecordId` keeping first appearance.
    Match(MatchPath),
    /// Transitive closure over the same path grammar as [`Statement::Match`]:
    /// every record reachable via the named edges (any number of hops) is
    /// returned, deduped by first-visit order, scored by BFS depth (0 = start).
    Closure(MatchPath),
    /// Delete a record (and its incident edges).
    Forget { id: RecordId },
    /// Switch the current memory context for subsequent statements in the
    /// plan (`MEMORY <name>`): DDL/DML/reads after this statement operate on
    /// the named memory's own store. Named memories are core/archival/shared
    /// partitions for agents (spec §2.8).
    Memory { name: String },
    /// WAL sequencing marker — **end of a plan** (issue #109). Never produced
    /// by the parser and never stored in `Store::history` (no clock tick, no
    /// data change): `Database::execute` appends it to the write-ahead log
    /// after a plan's mutating statements so replay resets the memory context
    /// exactly where the runtime did — every plan starts at the root
    /// (spec §2.8). Appended here: postcard tags are positional, so variants
    /// may only ever be added AFTER this one, never before (#109 discipline).
    ContextReset,
    /// `MATCH ... COUNT` — the same traversal as `Statement::Match`, but the
    /// result is a single `{"count": <n>}` row: the number of edge-path
    /// instances (walks) that matched, so parallel-edge multiplicity and
    /// ledger sizes are observable (spec §2.5, issue #94). Read-only: never
    /// logged to WAL or history.
    MatchCount(MatchPath),
    /// `PRUNE HISTORY` — compact the mutation history into one snapshot entry
    /// at the current clock (issue #95): keeps the `CreateTable` declaration
    /// statements plus the snapshot, so history growth stays bounded and
    /// replay from the snapshot is cheap. `AS OF` before the snapshot then
    /// fails loudly (`HistoryPruned`, spec §2.7). Mutating (WAL-logged);
    /// prunes every MEMORY block's history too.
    PruneHistory,
    /// A history-compaction base (issue #95): the full store state captured
    /// by [`Statement::PruneHistory`], carried inside the history itself.
    /// Never present in plans or the WAL — replay executes it to install the
    /// state the pruned prefix would have reconstructed. Boxed: the type is
    /// recursive through [`Store`].
    Snapshot(Box<SnapshotState>),
    /// `HISTORY SINCE <ts>` — exact delta read (issue #118): every mutation
    /// strictly after the cutoff, one result row per entry (kinds CREATE /
    /// INSERT / RELATE / FORGET with their subject ids — rows AND edges, so
    /// sync consumers never miss a graph-only change). Read-only: never
    /// WAL'd, never logged to history. Below the compaction horizon it fails
    /// with `HistoryPruned` (spec §2.7).
    HistorySince(i64),
}

/// Store state captured by history compaction (issue #95): everything needed
/// to reconstruct the store at `clock` without replaying the pruned prefix.
/// `memories` are embedded already-pruned (each block carries its own
/// snapshot entry in its own history) and `tables` rides along so installing
/// a snapshot never loses the declaration index (issue #133). Declaration
/// statements are NOT stored here — the pruned history retains them at their
/// original timestamps, which is what re-seeding (issue #89) reads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SnapshotState {
    pub records: BTreeMap<RecordId, Record>,
    pub edges: Vec<RelationEdge>,
    pub vector_dims: BTreeMap<String, usize>,
    pub clock: i64,
    pub memories: BTreeMap<String, Store>,
    pub tables: BTreeMap<String, Option<usize>>,
}

impl SnapshotState {
    /// Materialize the snapshot as a fresh [`Store`]: history starts empty
    /// (the pruned prefix is gone by design; replay appends from here).
    pub fn into_store(self) -> Store {
        Store {
            records: self.records,
            edges: self.edges,
            vector_dims: self.vector_dims,
            clock: self.clock,
            history: Vec::new(),
            memories: self.memories,
            tables: self.tables,
        }
    }
}

/// A graph traversal: start at `start`, walk `steps` in order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchPath {
    /// The record the traversal begins at. Must exist or the match is empty.
    pub start: RecordId,
    /// Ordered edge steps; every step must be taken (path semantics).
    pub steps: Vec<MatchStep>,
    /// Temporal read (`AS OF <int>`): traverse the store as of this logical
    /// timestamp — the mutation history replayed to that point, exactly like
    /// `SELECT ... AS OF` (spec §2.7; issue #92). `None` = current state.
    pub as_of: Option<i64>,
}

/// One hop of a [`MatchPath`]: follow edges named `name` leaving/entering the
/// current frontier, in the given direction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchStep {
    /// Which side of the edge the traversal moves across.
    pub direction: MatchDirection,
    /// Edge name to follow (`:mentions`, `:knows`, ...).
    pub name: String,
    /// Optional per-step edge filter (spec §1 `edge_props`): any field
    /// predicate — `=`, comparisons, `IN`, `BETWEEN` (issue #93) — evaluated
    /// against the edge's `props`. Non-field filters (embedding/BM25) are
    /// meaningless here and never match.
    pub edge_props: Option<Filter>,
}

/// Edge traversal direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchDirection {
    /// `(a) -> :name` — outgoing edges from the frontier record.
    Out,
    /// `(a) <- :name` — incoming edges toward the frontier record.
    In,
}

/// A SELECT with optional vector kNN, field filter, deterministic ordering, limit.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Select {
    pub table: String,
    pub knn: Option<Knn>,
    pub filter: Option<Filter>,
    pub order: Option<Order>,
    pub limit: Option<usize>,
    /// Temporal read: execute against the store as of this logical
    /// timestamp (`AS OF <int>`), replaying the mutation history up to it.
    pub as_of: Option<i64>,
    /// Field projection: `None` = full records (`SELECT *`); `Some` = only
    /// the listed body keys survive in output rows (missing keys are absent,
    /// BTree order preserved). Filters/scores always see the full record —
    /// projection is presentation-only (spec §2.3 step 8).
    pub fields: Option<Vec<String>>,
    /// Row offset: `OFFSET <n>` skips the first `n` rows after ordering,
    /// before the limit / k-cap truncate (spec §2.3; issue #94).
    pub offset: Option<usize>,
    /// Aggregate: `SELECT COUNT(*)` returns one `{"count": <n>}` row instead
    /// of records (spec §2.3; issue #94).
    pub aggregate: Option<Aggregate>,
}

/// The full "database" snapshot a query runs against. In M0 this is in-memory;
/// M1 makes it the on-disk single-file store + WAL.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Store {
    pub records: BTreeMap<RecordId, Record>,
    pub edges: Vec<RelationEdge>,
    /// Per-table vector dimensions once declared.
    pub vector_dims: BTreeMap<String, usize>,
    /// Deterministic logical clock: incremented once per mutating statement
    /// executed through the engine. The time source for `AS OF` reads — never
    /// wall-clock, so replay is a pure function of the statement order.
    pub clock: i64,
    /// Append-only mutation history: `(logical_timestamp, statement)` for
    /// every mutating statement this store has executed. `AS OF T` replays
    /// the entries with `ts <= T` into a fresh store. Persisted with the
    /// store, so time-travel survives checkpoints and WAL replay.
    pub history: Vec<(i64, Statement)>,
    /// Named memory partitions (`MEMORY <name>`): each is a full sub-store
    /// (records, edges, dims, its own clock + history), so temporal reads
    /// compose with memory scoping. The default context is the store itself.
    pub memories: BTreeMap<String, Store>,
    /// Declared tables → optional `VECTOR` dim: every `CREATE TABLE` of this
    /// store, including empty/dim-less ones (issue #89's seeding source).
    /// **Not part of the serialized payload** (`serde(skip)` — the file
    /// format is unchanged, issue #133 step 1): maintained live by the
    /// engine's `CreateTable` arm and rebuilt from history once at load
    /// ([`Store::rebuild_tables`]).
    #[serde(skip)]
    pub tables: BTreeMap<String, Option<usize>>,
}

impl Store {
    pub fn insert(&mut self, rec: Record) {
        // Keep insertion deterministic: preserve first-seen order via BTree key
        // ordering (BTreeMap is already deterministic). Edge list is append-only
        // in insertion order, which is deterministic per transaction.
        self.records.insert(rec.id.clone(), rec);
    }

    /// Record a mutating statement under the next logical timestamp.
    pub fn log_mutation(&mut self, stmt: &Statement) {
        self.clock += 1;
        self.history.push((self.clock, stmt.clone()));
    }

    /// Rebuild the `tables` index (and every nested memory's) from the
    /// mutation history — the load-time complement of the engine's
    /// `CreateTable` arm (issue #133 step 1). Idempotent; O(history), once
    /// per load. Only `CreateTable` statements matter, and pruned histories
    /// retain them (issue #95), so compaction never loses declarations.
    pub fn rebuild_tables(&mut self) {
        for (_, stmt) in &self.history {
            if let Statement::CreateTable { table, vector_dim } = stmt {
                self.tables.insert(table.clone(), *vector_dim);
            }
        }
        for memory in self.memories.values_mut() {
            memory.rebuild_tables();
        }
    }
}

/// Vector kNN clause: `WHERE vector::similarity(embedding, $q) AND k = N`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Knn {
    pub query: Vec<f32>,
    pub k: usize,
}

/// Field-level filter on record body values: equality plus the comparison,
/// membership, and range operators (spec §2.3; issue #93).
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
    /// `WHERE <field> <op> <value>` — comparison operators (issue #93).
    /// `<`, `<=`, `>`, `>=` order the record's value against the literal with
    /// [`Value::cmp_total`] (total cross-type order); `!=` is the negated
    /// exact equality of `=`. A record that does not carry the field never
    /// matches (same rule as `=`); an explicit `null` value participates and
    /// ranks below every other type (spec §2.3).
    FieldCmp {
        field: String,
        op: CmpOp,
        value: Value,
    },
    /// `WHERE <field> IN [<v1>, <v2>, ...]` — membership by the same exact
    /// value equality as `=` (so `1` and `1.0` are different values);
    /// a record without the field never matches (issue #93).
    FieldIn { field: String, values: Vec<Value> },
    /// `WHERE <field> BETWEEN <lo> AND <hi>` — inclusive range under
    /// [`Value::cmp_total`] (`lo <= value <= hi`); a record without the
    /// field never matches (issue #93).
    FieldBetween { field: String, lo: Value, hi: Value },
    /// `WHERE t1 AND t2 [AND …]` — n-ary conjunction over the combinable
    /// terms (field predicates + `IS NOT NULL`; scoring clauses — bm25/kNN —
    /// keep their own forms, issue #125). All terms must hold; evaluated
    /// per row / per edge with the same pure predicates (scan-side). A single
    /// operator, so there is no precedence to define. Appended variant
    /// (postcard-tag stable).
    And(Vec<Filter>),
}

/// Comparison operators for [`Filter::FieldCmp`] (`!=`, `<`, `<=`, `>`, `>=`).
/// Equality keeps its own variant ([`Filter::FieldEquals`]) for compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CmpOp {
    /// `!=` — negated exact equality (the complement of `=`), so exactly one
    /// of `= v` / `!= v` holds for every record that carries the field.
    Ne,
    /// `<` — less under [`Value::cmp_total`].
    Lt,
    /// `<=` — less or equal under [`Value::cmp_total`].
    Le,
    /// `>` — greater under [`Value::cmp_total`].
    Gt,
    /// `>=` — greater or equal under [`Value::cmp_total`].
    Ge,
}

/// Aggregate shaping for `SELECT` (issue #94): present ⇒ the query returns
/// one summary row instead of records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Aggregate {
    /// `SELECT COUNT(*)` — the number of records that pass the WHERE filter,
    /// as a single `{"count": <n>}` row (deterministic; spec §2.3).
    CountStar,
}

/// Deterministic ordering operators (pure arithmetic — see docs/decisions.md D9).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Order {
    /// Cosine similarity (descending) vs the kNN query vector.
    Similarity,
    /// α·similarity + β·strength(recency,freq) + γ·importance + δ·score with the
    /// engine defaults α=0.7, β=0, γ=0, δ=0.3 (spec §2.3; agent-tuned per-query
    /// via [`Order::SalienceWeighted`]).
    Salience,
    /// Laplace-smoothed mean of `:voted` edge weights on the record (each
    /// edge's explicit `weight`, falling back to its signed `value` when
    /// `weight` is absent).
    Score,
    /// Net up−down vote count over `:voted` edges (descending; tie-break by RecordId).
    Votes,
    /// Time-decayed recent feedback over `:voted` edges (descending; tie-break by RecordId).
    Feedback,
    /// created_at (descending).
    Recency,
    /// `::salience(α, β, γ, δ)` — the same four terms as [`Order::Salience`]
    /// with agent-tuned weights parsed from the parenthesized list (issue #88).
    /// Appended variant: postcard tags of earlier variants stay stable (the
    /// #109 discipline for serialized enums).
    SalienceWeighted([f32; 4]),
    /// `ORDER BY <field> [DESC]` — sort by a body field under
    /// [`Value::cmp_total`] (the same total order the filters use, issue
    /// #117). Absent fields rank as `null` (lowest); `desc` reverses the key
    /// only — ties keep ascending `RecordId`. Appended variant (postcard-tag
    /// stable).
    Field { key: String, desc: bool },
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
