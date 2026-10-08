# NQL — Neural Query Language Specification

Version: 0.1 (M0 slice) · Status: living spec — implementers target this file;
changes here are contract changes (PRs that alter grammar must update this file).

NQL is the query language of **nqlite**, a deterministic, No-LLM
database. NQL expresses records, typed relations (graph), embeddings, temporal
context, and hybrid retrieval in ONE grammar. The engine never calls an LLM;
vectors are BYO (agent-supplied `f32` arrays). This file defines the grammar
and the semantics that both the parser (`nql`) and the engine (`nqlite`) must
agree on.

**File conventions.** NQL program files (scripts passed to `--script`,
stored queries) use the **`.nql`** extension — it names the *language*.
The database store is a different artifact with a different extension:
**`.ndb`** (see `file-format.md`).

## 1. Grammar (EBNF)

Terminals: `ident` = `[A-Za-z_][A-Za-z0-9_]*`, `int` = `-?[0-9]+`,
`float` = `-?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?`, `string` = `'...' | "..."`
(minimal escapes `\" \' \\ \n \t`). `recordid` = `ident ':' (int | ident)`.
Keywords are case-insensitive. Whitespace/comments (`--` to end of line, `/* */`)
are ignored. `...` in the grammar is a list separator (`a, b, ...`).

```
statement      = create_table | insert | relate | select | match | closure | memory | forget
               | prune | history ;

create_table   = 'CREATE' 'TABLE' ident [ 'VECTOR' '<' 'f32' ',' int '>' ] ;

insert         = 'INSERT' 'INTO' recordid '{' object '}' [ 'EMBED' vector ] ;

relate         = 'RELATE' '(' recordid ')' '->' ':' ident '->' '(' recordid ')'
                 [ 'SET' (ident '=' value) (',' ident '=' value)* ] ;

select         = 'SELECT' select_list 'FROM' ident
                 [ 'WHERE' where_clause ]
                 [ 'ORDER' 'BY' order_key ]
                 [ 'AS' 'OF' int ]
                 [ 'OFFSET' int ] [ 'LIMIT' int [ 'OFFSET' int ] ] ;

forget         = 'FORGET' recordid ;

memory         = 'MEMORY' ident ;

prune          = 'PRUNE' 'HISTORY' ;

history        = 'HISTORY' 'SINCE' int ;

match          = 'MATCH' '(' recordid ')' path_step+ [ 'AS OF' int ] [ 'COUNT' ] ;
closure        = 'CLOSURE' '(' recordid ')' path_step+ [ 'AS OF' int ] ;
path_step      = ('->' | '<-') ':' ident [ edge_props ] ;
edge_props     = 'WHERE' conjunction ;

select_list    = '*' | 'COUNT' '(' '*' ')' | ident (',' ident)* ;
where_clause   = conjunction | vector_knn | bm25 | hybrid ;
conjunction    = term ( 'AND' term )* ;
term           = predicate | has_embedding ;
predicate      = field_equals | field_cmp | field_in | field_between ;
field_equals   = ident '=' value ;
field_cmp      = ident ('!=' | '<' | '<=' | '>' | '>=') value ;
field_in       = ident 'IN' '[' [ value (',' value)* ] ']' ;
field_between  = ident 'BETWEEN' value 'AND' value ;
vector_knn     = 'vector::similarity' '(' 'embedding' ',' vector ')' 'AND' 'k' '=' int ;
has_embedding  = 'embedding' 'IS' 'NOT' 'NULL' ;
bm25           = '::bm25' '(' ident ',' string ')' [ 'AND' 'k' '=' int ] ;
hybrid         = bm25 'AND' vector_knn | vector_knn 'AND' bm25 ;
order_op       = 'similarity' | 'salience' [ '(' num ',' num ',' num ',' num ')' ]
               | 'score' | 'votes' | 'feedback' | 'recency' ;
order_key      = '::'? order_op | field [ 'DESC' ] ;
num           = int | float ;

object         = ( ident ':' value ) ( ',' ident ':' value )* | '' ;
value          = 'null' | 'true' | 'false' | int | float | string
               | vector | '[' value (',' value)* ']' | '{' object '}' ;
vector         = '[' float (',' float)* ']' ;   (* all-float array *)
```

### Planned extensions (NOT in M0 — for M2+)

```
create_index   = 'CREATE' 'INDEX' ident 'ON' ident '(' ident ')' ;
```

## 2. Semantics

### 2.1 Determinism contract (non-negotiable)

- Record iteration order = `BTreeMap<RecordId, Record>` key order (RecordId Ord:
  table then id; numeric ids sort numerically, string ids lexically).
- Edges = append-only list; aggregation passes over them are fixed-order.
- Every ordered query must produce a TOTAL order: numeric sorts tie-break by
  ascending RecordId string form.
- No randomness, no wall-clock inside the engine (created_at is engine-clocked
  per transaction and deterministic within it), no network, no LLM.
- Same plan + same store => byte-identical results, every run.

### 2.2 Statements

| Statement | Effect |
|---|---|
| `CREATE TABLE t [VECTOR<f32,N>]` | Declares table `t`; optional fixed embedding dim `N`. Re-declaring with a dim enforces it on future INSERTs. |
| `INSERT INTO t:id {body} [EMBED v]` | Upsert record `t:id`. If table has declared dim, `len(embedding)` MUST equal it else `EmbeddingDimMismatch`. Embedding is BYO — engine never computes it. |
| `RELATE (a)->:name->(b) [SET ...]` | Appends a directed, named edge with properties. `weight` (float, 0..=1) and any other props. Edges are first-class: votes, provenance, temporal info all live here (see §4). |
| `MATCH (a) -> :name -> :other <- :back` | Walks the graph from record `a` along the named edges in order, returning the records reached after the last hop. Each step may carry `WHERE <predicate>` (equality, comparisons, `IN`, `BETWEEN`) to only traverse edges whose props match. `MATCH ... COUNT` returns the number of matching edge-path instances instead of records (multiplicity). `AS OF <ts>` traverses a historical snapshot. Deterministic (see §2.5). |
| `CLOSURE (a) -> :name` | Transitive closure: every record reachable from `a` via the named edges (any number of hops, BFS to fixpoint), deduped by first-visit order, scored by BFS depth (0 = start). Edge-property filters apply per step like MATCH; accepts `AS OF <ts>` like MATCH (see §2.5). |
| `SELECT ... FROM t ...` | Scans table `t`; filters (incl. `::bm25` lexical scoring); optional kNN; `AS OF <ts>` time-travels to a historical view; orders deterministically; paginates (`OFFSET`/`LIMIT`); `COUNT(*)` counts instead of listing; returns records (+computed score). |
| `FORGET t:id` | Deletes the record AND all incident edges. |
| `PRUNE HISTORY` | Compacts the mutation history into a snapshot at the current clock: bounded growth, cheap `AS OF` from the snapshot onward; `AS OF` earlier than the snapshot errors (`HistoryPruned`). Applies to MEMORY blocks too (issue #95). |
| `HISTORY SINCE <ts>` | Exact mutation delta since the cutoff — one row per CREATE/INSERT/RELATE/FORGET with its subject ids (rows AND edges + tombstones): the sync read that a two-`AS OF` diff cannot give you (issue #118). |
| `MEMORY <name>` | Switches the plan's context to the named memory (created lazily): subsequent statements target that memory's own store — records, edges, history (so `AS OF` composes). Core/archival/shared partitions for agents (see §2.8). |

### 2.3 SELECT pipeline (fixed order)

1. **Scan** — records of table `t` in BTree order.
2. **Filter** — `WHERE`:
   - `field = value`: exact, deterministic equality against body value.
   - `field != value`, `field < | <= | > | >= value`, `field IN [v, …]`,
     `field BETWEEN a AND b` (inclusive) — the comparison predicates
     (issues #93/#94). All but `!=`/`=`/`IN` order values with a **total
     cross-type order**: type ranks `null < bool < number < string < array <
     doc < vector < ref`; within numbers comparison is numeric and exact
     (ints never round through floats; `NaN` sorts after every number);
     arrays/docs/vectors compare element/key-wise, then by length. `=`,
     `!=`, and `IN` keep exact value equality (`1` and `1.0` are different
     values), so `= v` and `!= v` are complements. A record that does not
     carry the field never matches (the long-standing `=` rule); an explicit
     `null` participates and ranks lowest. The order is proptest-pinned
     (reflexive, antisymmetric, transitive, never panics).
   - `embedding IS NOT NULL`: only records with vectors.
   - **Conjunction** (issue #125): combinable terms compose with `AND` —
     `field-predicates… AND embedding IS NOT NULL AND …`, n-ary, evaluated
     all-of per row (each term keeps its own missing-field rule; a term that
     fails excludes the row). Single operator, so no precedence exists.
     Scoring clauses do **not** join conjunctions: `::bm25`/`vector::similarity`
     keep their own forms (`::bm25(...) AND vector::similarity(...)` is the
     hybrid, §2.6) — mixing them into an `AND` chain is a positioned error.
     The same conjunction grammar drives MATCH/CLOSURE edge-property filters
     (§2.5), evaluated all-of against the edge's `props`.
   - `id = | != | IN [...]` (issue #128): compares against the record's own
     identity in display form (`table:id`) — the rerank-pool predicate
     (`WHERE id IN ["doc:7", …]` server-side). The literal must be that
     string; ordered forms (`<`, `<=`, `>`, `>=`, `BETWEEN`) are positioned
     errors (ids are not an ordered value). A body key named `id` never
     shadows the pseudo-field; in edge-property filters — edges have no
     record identity — `id` is an ordinary prop lookup.
   - `vector::similarity(embedding, $q) AND k = N`: kNN candidate set (see §3).
   - `::bm25(field, "query") [AND k = N]`: lexical scoring — every row is
     ranked by BM25 relevance over the field; `k` caps the returned rows.
   - `hybrid`: `::bm25(field, "query") AND vector::similarity(embedding, $q)
     AND k = N` (or the clauses in either order) — both signals are computed
     and fused (see §2.6).
3. **kNN** — cosine similarity vs query vector; keep top-K by similarity desc,
   tie-break by RecordId asc.
4. **Fusion** — when both `::bm25` and `vector::similarity` are present, each
   row's final score is the reciprocal-rank fusion (RRF) of its rank in the
   lexical list and its rank in the vector list: `1/(60 + rank_lex) +
   1/(60 + rank_vec)`, ranks 1-based, ties broken by RecordId asc (see §2.6).
5. **Order** — `ORDER BY ::op` or `ORDER BY <field> [DESC]` (default: BTree
   order):
   - `::similarity` — cosine desc (requires kNN query).
   - `::salience` — `α·similarity + β·strength(recency,freq) + γ·importance + δ·score`
     with deterministic engine defaults **α=0.7, β=0, γ=0, δ=0.3** (α..δ are
     AGENT-side knobs, not engine config); without a kNN query, the default
     salience reduces to the feedback term. Agents tune per-query with
     `ORDER BY ::salience(α, β, γ, δ)` — exactly four comma-separated numbers.
     Term definitions (pure arithmetic, No-LLM):
     - `similarity` — cosine vs the kNN query vector (`0` without a kNN query);
     - `strength` — `(recency + freq) / 2`, where `recency = 1/(1+age)` with
       `age = max(0, clock − created_at)` and `freq = n/(n+1)` over the
       record's incident edges (either direction);
     - `importance` — the agent-written `importance` field clamped to `[0,1]`
       (missing/non-numeric → `0`; the engine never invents it, §5);
     - `score` — the Laplace-smoothed `::score` clamped to `[0,1]`.
   - `::score` — Laplace-smoothed mean over `:voted` edges, desc.
   - `::recency` — `created_at` desc.
   - `<field> [DESC]` (issue #117) — sort by a body field under the same
     total order as the §2.3 filters (`Value::cmp_total`): absent fields and
     explicit `null`s rank lowest. `DESC` reverses the **key only** — ties
     always keep ascending RecordId (the §2.1 total order holds in both
     directions). If the query returns rows but no record of the table
     carries the field, the query errors (`UnknownSortField`) — a typo must
     not become a silent all-equal sort. `DESC` after a `::` operator is a
     positioned error (operators have fixed directions). Like `::recency`,
     an explicit field sort applies even in kNN/BM25 modes (score-based
     orders defer to those rankings — precedence documented with #119).
   **Mode precedence (verified, issue #119).** The score a row *displays* is
   always the mode's score (kNN similarity, bm25 relevance, hybrid RRF
   fusion); the sort key follows this table:

   | mode \ `ORDER BY` | *(none)* | score-based ops (`::similarity`, `::salience`, `::score`, `::votes`, `::feedback`) | `::recency` | `<field> [DESC]` |
   |---|---|---|---|---|
   | scan | BTree (RecordId asc) | honored — the op's own score | honored | honored |
   | kNN | similarity desc | honored — the op's own score | honored | honored |
   | bm25 | relevance desc | **ignored — relevance wins** | honored | honored |
   | hybrid | fused (RRF) desc | **ignored — fusion wins** | honored | honored |

   Relevance dominance is the point of the bm25/hybrid modes: a score-based
   order would silently undo the ranking the query asked for (silent-OK was
   the reported hazard, issue #119). Structural orders — `::recency` and
   `<field> [DESC]` — are honored in **every** mode: they sort by data the
   mode's score does not contain. Note the display/order split: rows always
   *show* the mode's score even when a structural order re-sorts them.
   (This supersedes the earlier blanket "explicit `ORDER BY` is ignored in
   hybrid": that holds for score-based ops only — every cell above is
   pinned by the `order_by_precedence_matrix_matches_spec` test.)
6. **Offset / Limit** — `OFFSET n` skips the first `n` rows after ordering;
   then keep the first N (or the kNN/BM25 `k` cap, whichever is smallest) of
   what remains (issues #93/#94).
7. **Aggregate** — `SELECT COUNT(*)` returns ONE row `{"count": <n>}`
   instead of records: `n` is the number of records that passed step 2's
   filter, so ordering, offset, limit, and projection never affect it (a
   count is computed before them and builds no kNN/BM25 index). Deterministic
   (issue #94).
8. **Project** — keep only the fields listed in `select_list` (`SELECT *`
   keeps every field). Presentation-only: filters, scores, ordering, and
   limits all ran on the full record. A listed field a record does not have
   is simply absent from that row (no error, BTree field order preserved).

### 2.4 Transactions

A plan is a `Vec<Statement>` executed atomically in one transaction
(single-writer, snapshot readers — M1 storage; M0 is in-memory). Statements
before a SELECT apply first, so a plan may create/insert/relate/query in one pass.

### 2.5 MATCH / CLOSURE traversal (deterministic)

- The frontier starts at the start record. A missing start record yields an
  empty result — never an error.
- Each step follows every edge with the given name in the given direction
  (`->` = outgoing from a frontier record, `<-` = incoming toward it). A step
  may carry `WHERE <predicate>` — any field predicate of §2.3 (equality,
  comparisons, `IN`, `BETWEEN`) evaluated against the edge's `props`:
  only edges whose props match are traversed (issue #93).
- Edges are scanned in append order; reached endpoints are deduplicated by
  `RecordId` keeping first appearance. The result rows carry the score of the
  first edge that reached them (`weight`, or `0.0` when unset).
- Dangling edges (endpoints never inserted) are skipped.
- `MATCH` (path semantics): only the final frontier is returned — intermediate
  hops are not in the output.
- `MATCH ... COUNT` (issue #94): returns one row `{"count": <n>}` instead of
  the frontier, where `n` is the number of edge-path **instances** (walks)
  matching the steps — parallel edges count separately, so the multiplicity
  between two records stays observable. Walk counts accumulate per node per
  step as saturating `u64`s (order-independent, hence deterministic). The same
  edge/prop filters and dangling-edge rules apply; a missing start yields
  `{"count": 0}`.
- `AS OF <ts>` on `MATCH` and `CLOSURE` (issue #92): the traversal runs
  against the reconstructed snapshot — mutation history replayed to `ts`
  (§2.7) — so the start record, frontier, and edges are exactly what existed
  at that cutoff. Composes with `COUNT`. A cutoff before the start record
  exists yields the empty result (same rule as a missing start).
- `CLOSURE` (transitive closure): each step is expanded to a fixpoint (BFS,
  any number of hops) before the next step begins; every record ever reached —
  including the start — is returned once in first-visit order, scored by BFS
  depth (0 = start). Cycles are handled by dedup.

### 2.6 Hybrid retrieval (lexical + vector fusion)

`WHERE ::bm25(field, "q") AND vector::similarity(embedding, $v) AND k = N`
(and the reverse clause order) computes BOTH signals over the table's rows
and fuses them with reciprocal-rank fusion (RRF):

- Each row gets a rank in the lexical list (BM25 score desc, RecordId asc
  tie-break) and a rank in the vector list (cosine similarity desc, RecordId
  asc tie-break).
- Fused score = `1/(60 + rank_lex) + 1/(60 + rank_vec)` — deterministic,
  scale-free, and bounded; a row strong in both signals outranks a row strong
  in only one.
- The fused score is the sort key (explicit `ORDER BY` is ignored); the row
  cap is the smallest of `k`, the BM25 `k`, and `LIMIT`.

### 2.7 Temporal reads (`AS OF`)

`SELECT ... AS OF <int>`, `MATCH ... AS OF <int>`, and
`CLOSURE ... AS OF <int>` execute against the store **as of logical timestamp
`<int>`**, not the current state (graph traversals joined `SELECT` here in
issue #92 — they replay the same history and walk the reconstructed
snapshot). Semantics:

- Every mutating statement executed through the engine is appended to the
  store's mutation history under the next logical timestamp (a deterministic
  counter — never wall-clock, so replay is a pure function of statement
  order). The history is persisted with the store, so time-travel survives
  checkpoints and WAL replay.
- An `AS OF T` read replays the history entries with `ts <= T` into a fresh
  store and queries that. Upserts and `FORGET`s reconstruct correctly: an
  `AS OF` between two upserts of the same id sees the first version; `AS OF`
  before a `FORGET` sees the record again.
- `AS OF` with a cutoff beyond the last mutation is the full current state.
- The timestamp is the **logical** mutation counter, not a wall-clock
  datetime; a datetime-literal form is future work.
- **History compaction (`PRUNE HISTORY`, issue #95):** replaces the history
  with a snapshot of the current state at the current clock, plus the
  `CreateTable` declaration statements retained at their original timestamps
  (they are the only record of empty/dim-less tables — issue #89 — and
  re-executing them is idempotent). Growth stops tracking superseded
  versions, and later `AS OF` reads rebuild from the snapshot instead of
  ts0. **Retention contract:** `AS OF T` with `T` earlier than the snapshot
  fails with `HistoryPruned` ("history before ts N was compacted …") — loud,
  never a partial view. Compaction applies to every `MEMORY` block's own
  history too, is deterministic (a pure function of the store), is
  WAL-logged (it survives a reopen without an explicit flush), and lands in
  the main file at the next checkpoint. **Downside caveat:** a store pruned
  by a binary with this feature cannot be opened by older binaries (the
  snapshot entry is a new statement variant); unpruned stores decode
  unchanged.
- **Delta reads (`HISTORY SINCE <ts>`, issue #118):** every mutation strictly
  after the cutoff, in append (ts-ascending) order — one result row per entry:
  `{ ts, kind, …subjects }` where `kind` ∈ `CREATE` (table + optional `dim`),
  `INSERT` / `FORGET` (record `id`), `RELATE` (`from`, `to`, edge `name`).
  Rows **and** edges in one read: a state-diff of two `AS OF` reads is blind
  to edge-only mutations (a `RELATE` between existing records changes no
  rows) — exactly what a sync consumer must not miss. Exclusive cutoff
  (`ts > since`); an empty result means nothing changed; read-only (never
  WAL'd); runs inside a `MEMORY` block against that block's own history;
  honors the `PRUNE HISTORY` retention horizon (`HistoryPruned` below the
  snapshot, and snapshot entries themselves are never reported as
  mutations).

### 2.8 Memory blocks (`MEMORY <name>`)

`MEMORY <name>` partitions context: every statement after it in the same plan
runs against the named memory's own store (created lazily on first mention).
Semantics:

- Each memory is a full sub-store: its own records, edges, vector dims, and
  its own logical clock + mutation history. `AS OF` reads inside a memory
  replay that memory's history only.
- The default context is the root store; a plan always starts at the root, so
  an agent re-asserts `MEMORY <name>` at the top of any plan that should run
  inside a memory. Same `table:id` in different memories are different
  records.
- **Context never carries across a plan boundary.** Over `nql-server` each
  protocol line is its own plan, and in the REPL each input line is — so
  `MEMORY <name>;` on one line does **not** apply to the next line; an
  unprefixed write after it silently targets the root store (answers `OK`).
  On line-oriented inputs, prefix every statement that belongs to the
  memory (issue #87):

  ```
  MEMORY ledger; CREATE TABLE note;             -- ok: one line, one plan
  MEMORY ledger; INSERT INTO note:1 { "x": 1 }; -- ok
  MEMORY ledger; SELECT * FROM note;            -- ok
  MEMORY ledger;                                -- NO-OP for later lines!
  INSERT INTO note:2 { "x": 2 };                -- writes ROOT, not ledger
  ```

  A multi-statement script passed to `nql --script` is a *single* plan, so
  there a lone `MEMORY <name>;` does carry through the rest of the file.
- `MEMORY` statements are logged to the WAL (they carry the context switch),
  so memory scoping survives reopen via replay. Executing `MEMORY` outside a
  plan (directly via the engine's statement API) is an error.

## 3. Operators

| Operator | Definition | Determinism |
|---|---|---|
| `vector::similarity(embedding, $q)` | cosine similarity `a·b / (|a||b|)`; zero-norm operands => `0.0` | exact f32 |
| `::similarity` | order by cosine desc | total order w/ tie-break |
| `::score` | `(Σ weights + 1) / (n + 2)` over `:voted` edges; `0.5` with no votes (Laplace smoothing). Each edge's weight is its explicit `weight`, else its signed `value` (`value = -1` downvotes); an edge with neither counts as `+1` | pure arithmetic |
| `::votes(record)` | `(up, down, net)` counts over `:voted` edges | pure arithmetic |
| `::feedback(record)` | time-decayed recent feedback | engine-clock only |
| `::salience` | `α·similarity + β·strength + γ·importance + δ·score` (defaults 0.7/0/0/0.3; tune: `::salience(α, β, γ, δ)`) | fixed order, no races |
| `::bm25(field, "q")` | Okapi BM25 lexical score (k1=1.2, b=0.75); every row scored, ordered by relevance | pure arithmetic |
| `k = N` | kNN / BM25 result cap | — |

## 4. Edges & the relation model

- Directed named edges: `(from) -[:name {props}]-> (to)`.
- Edge properties: `weight` (agent-supplied confidence 0..=1), arbitrary props
  (provenance, started_on/ended_on per Zep research), `created_at` (engine).
- **Votes are edges (decision D9):** `(voter)->:voted {value:+1|-1, weight, created_at}->(record)`.
  No separate vote machinery; provenance and one-transaction semantics come free.
  The engine accepts both `voted` and `:voted` spellings at every reader
     (NQL text strips the colon on write; edges built directly through the IR
  may keep it — see issue #98).
  Vote `weight` spans `-1..=1` and defaults to the edge's `value` when omitted, so
  `SET value = -1` alone is a downvote under `::score` too; an explicit `weight`
  overrides `value` for `::score` only (`::votes`/`::feedback` always read `value`).
- `FORGET` removes incident edges, keeping the graph clean.

## 5. No-LLM & BYO-vector contract

- The engine never calls an embedder/LLM/network — to embed, chunk, summarize,
  compact, or rerank. Any learning lives in the agent/client.
- Vectors arrive as `f32` arrays at INSERT time. `importance` is a number the
  AGENT writes; the engine never invents it.
- Salience weights α..δ are agent-side (passed per-query as
  `ORDER BY ::salience(α, β, γ, δ)`); engine defaults are deterministic
  (0.7, 0, 0, 0.3).

## 6. Examples

### 6.1 Agent conversation session (M0 grammar)

```sql
-- agent stores turns, entities, and chains context
CREATE TABLE turn VECTOR<f32, 384>;
CREATE TABLE entity;

INSERT INTO turn:3 { "role": "user", "text": "I work at Acme on the ML team" }
  EMBED [0.02, -0.15, 0.43, ...];
INSERT INTO entity:acme { "kind": "organization", "name": "Acme Corp" };
INSERT INTO entity:ml { "kind": "team", "name": "ML" };

RELATE (turn:3) -> :mentions -> (entity:acme) SET weight = 0.9;
RELATE (turn:3) -> :mentions -> (entity:ml)   SET weight = 0.8;
RELATE (turn:3) -> :follows_from -> (turn:2)  SET weight = 1.0;

-- recall: semantic kNN on recent context
SELECT * FROM turn
  WHERE vector::similarity(embedding, [0.01, -0.12, 0.40, ...]) AND k = 5
  ORDER BY ::salience
  LIMIT 3;

-- graph: what entities does this turn touch?
SELECT * FROM entity
  WHERE name = "Acme Corp";

-- graph: every entity this turn mentions (1-hop outgoing)
MATCH (turn:3) -> :mentions;

-- graph: turns that mention Acme (1-hop incoming)
MATCH (entity:acme) <- :mentions;

-- graph: only high-confidence mentions (edge-property filter)
MATCH (turn:3) -> :mentions WHERE confidence = 0.9;

-- graph: the whole conversation chain reachable from turn:3
CLOSURE (turn:3) -> :follows_from;

-- lexical recall: BM25 over the turn text
SELECT * FROM turn WHERE ::bm25(text, "acme") LIMIT 5;

-- hybrid recall: lexical + semantic fused (both signals matter)
SELECT * FROM turn
  WHERE ::bm25(text, "acme") AND vector::similarity(embedding, [0.5, -1.0, 2.5])
  AND k = 5;

-- agent-side feedback: mark the recalled turn as useful
RELATE (agent:main) -> :voted -> (turn:3) SET value = 1, weight = 0.9;

-- rank by community feedback
SELECT * FROM turn ORDER BY ::score LIMIT 5;

-- time travel: the conversation as it stood at logical ts 42
SELECT * FROM turn AS OF 42;

-- memory blocks: partition core facts vs working notes
MEMORY core;
SELECT * FROM entity WHERE name = "Acme Corp";
MEMORY working;
INSERT INTO note:1 { "text": "todo: verify the recall curve" };
```

### 6.2 Planned syntax (M2+)

```sql
MEMORY core;                           -- agent memory blocks
```

## 7. Implementation notes

- Parser: hand-written lexer + recursive-descent parser (as shipped in `nql`).
  Keywords are contextual (lexed as identifiers, matched case-insensitively).
  Errors carry 1-based line/column.
- Fuzzing: the parser is a cargo-fuzz target; determinism makes fuzz results
  reproducible. Property tests (proptest) cover "parse → plan → re-execute is
  stable" invariants.
- The Plan/IR types live in `nql-ir` (the seam): `Statement`, `Select`, `Knn`,
  `Filter`, `Order`, `MatchPath`/`MatchStep`, `Plan = Vec<Statement>`.
