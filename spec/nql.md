# nql — Neural Query Language Specification

Version: 0.1 (M0 slice) · Status: living spec — implementers target this file;
changes here are contract changes (PRs that alter grammar must update this file).

nql is the query language of **nqlite**, a deterministic, zero-LLM neural
database. nql expresses records, typed relations (graph), embeddings, temporal
context, and hybrid retrieval in ONE grammar. The engine never calls an LLM;
vectors are BYO (agent-supplied `f32` arrays). This file defines the grammar
and the semantics that both the parser (`nql`) and the engine (`nqlite`) must
agree on.

## 1. Grammar (EBNF)

Terminals: `ident` = `[A-Za-z_][A-Za-z0-9_]*`, `int` = `-?[0-9]+`,
`float` = `-?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?`, `string` = `'...' | "..."`
(minimal escapes `\" \' \\ \n \t`). `recordid` = `ident ':' (int | ident)`.
Keywords are case-insensitive. Whitespace/comments (`--` to end of line, `/* */`)
are ignored. `...` in the grammar is a list separator (`a, b, ...`).

```
statement      = create_table | insert | relate | select | match | closure | forget ;

create_table   = 'CREATE' 'TABLE' ident [ 'VECTOR' '<' 'f32' ',' int '>' ] ;

insert         = 'INSERT' 'INTO' recordid '{' object '}' [ 'EMBED' vector ] ;

relate         = 'RELATE' '(' recordid ')' '->' ':' ident '->' '(' recordid ')'
                 [ 'SET' (ident '=' value) (',' ident '=' value)* ] ;

select         = 'SELECT' select_list 'FROM' ident
                 [ 'WHERE' where_clause ]
                 [ 'ORDER' 'BY' '::' order_op ]
                 [ 'LIMIT' int ] ;

forget         = 'FORGET' recordid ;

match          = 'MATCH' '(' recordid ')' path_step+ ;
closure        = 'CLOSURE' '(' recordid ')' path_step+ ;
path_step      = ('->' | '<-') ':' ident [ edge_props ] ;
edge_props     = 'WHERE' field_equals ;

select_list    = '*' | ident (',' ident)* ;
where_clause   = field_equals | vector_knn | has_embedding | bm25 | hybrid ;
field_equals   = ident '=' value ;
vector_knn     = 'vector::similarity' '(' 'embedding' ',' vector ')' 'AND' 'k' '=' int ;
has_embedding  = 'embedding' 'IS' 'NOT' 'NULL' ;
bm25           = '::bm25' '(' ident ',' string ')' [ 'AND' 'k' '=' int ] ;
hybrid         = bm25 'AND' vector_knn | vector_knn 'AND' bm25 ;
order_op       = 'similarity' | 'salience' | 'score' | 'votes' | 'feedback'
               | 'recency' ;

object         = ( ident ':' value ) ( ',' ident ':' value )* | '' ;
value          = 'null' | 'true' | 'false' | int | float | string
               | vector | '[' value (',' value)* ']' | '{' object '}' ;
vector         = '[' float (',' float)* ']' ;   (* all-float array *)
```

### Planned extensions (NOT in M0 — for M2+)

```
temporal       = 'AS' 'OF' datetime ;               (* time-travel reads *)
memory         = 'MEMORY' ident ;                   (* core/archival/shared blocks *)
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
| `MATCH (a) -> :name -> :other <- :back` | Walks the graph from record `a` along the named edges in order, returning the records reached after the last hop. Each step may carry `WHERE <edge-prop> = <value>` to only traverse edges whose props match. Deterministic (see §2.5). |
| `CLOSURE (a) -> :name` | Transitive closure: every record reachable from `a` via the named edges (any number of hops, BFS to fixpoint), deduped by first-visit order, scored by BFS depth (0 = start). Edge-property filters apply per step like MATCH (see §2.5). |
| `SELECT ... FROM t ...` | Scans table `t`; filters (incl. `::bm25` lexical scoring); optional kNN; orders deterministically; limits; returns records (+computed score). |
| `FORGET t:id` | Deletes the record AND all incident edges. |

### 2.3 SELECT pipeline (fixed order)

1. **Scan** — records of table `t` in BTree order.
2. **Filter** — `WHERE`:
   - `field = value`: exact, deterministic equality against body value.
   - `embedding IS NOT NULL`: only records with vectors.
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
5. **Order** — `ORDER BY ::op` (default: BTree order):
   - `::similarity` — cosine desc (requires kNN query).
   - `::salience` — `α·similarity + β·strength(recency,freq) + γ·importance + δ·score`
     with deterministic engine defaults (α..δ are AGENT-side knobs, not engine
     config); without a kNN query, salience reduces to the feedback term.
   - `::score` — Laplace-smoothed mean over `:voted` edges, desc.
   - `::recency` — `created_at` desc.
   (Hybrid queries always order by the fused score; explicit `ORDER BY` is
   ignored in that mode.)
6. **Limit** — keep first N (or the kNN/BM25 `k` cap, whichever is smallest).

### 2.4 Transactions

A plan is a `Vec<Statement>` executed atomically in one transaction
(single-writer, snapshot readers — M1 storage; M0 is in-memory). Statements
before a SELECT apply first, so a plan may create/insert/relate/query in one pass.

### 2.5 MATCH / CLOSURE traversal (deterministic)

- The frontier starts at the start record. A missing start record yields an
  empty result — never an error.
- Each step follows every edge with the given name in the given direction
  (`->` = outgoing from a frontier record, `<-` = incoming toward it). A step
  may carry `WHERE <edge-prop> = <value>`: only edges whose `props` field
  equals the value are traversed (exact equality against the edge's props).
- Edges are scanned in append order; reached endpoints are deduplicated by
  `RecordId` keeping first appearance. The result rows carry the score of the
  first edge that reached them (`weight`, or `0.0` when unset).
- Dangling edges (endpoints never inserted) are skipped.
- `MATCH` (path semantics): only the final frontier is returned — intermediate
  hops are not in the output.
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

## 3. Operators

| Operator | Definition | Determinism |
|---|---|---|
| `vector::similarity(embedding, $q)` | cosine similarity `a·b / (|a||b|)`; zero-norm operands => `0.0` | exact f32 |
| `::similarity` | order by cosine desc | total order w/ tie-break |
| `::score` | `(Σ weights + 1) / (n + 2)` over `:voted` edges; `0.5` with no votes (Laplace smoothing) | pure arithmetic |
| `::votes(record)` | `(up, down, net)` counts over `:voted` edges | pure arithmetic |
| `::feedback(record)` | time-decayed recent feedback | engine-clock only |
| `::salience` | `α·similarity + β·strength + γ·importance + δ·score` | fixed order, no races |
| `::bm25(field, "q")` | Okapi BM25 lexical score (k1=1.2, b=0.75); every row scored, ordered by relevance | pure arithmetic |
| `k = N` | kNN / BM25 result cap | — |

## 4. Edges & the relation model

- Directed named edges: `(from) -[:name {props}]-> (to)`.
- Edge properties: `weight` (agent-supplied confidence 0..=1), arbitrary props
  (provenance, started_on/ended_on per Zep research), `created_at` (engine).
- **Votes are edges (decision D9):** `(voter)->:voted {value:+1|-1, weight, created_at}->(record)`.
  No separate vote machinery; provenance and one-transaction semantics come free.
- `FORGET` removes incident edges, keeping the graph clean.

## 5. Zero-LLM & BYO-vector contract

- The engine never calls an embedder/LLM/network — to embed, chunk, summarize,
  compact, or rerank. Any learning lives in the agent/client.
- Vectors arrive as `f32` arrays at INSERT time. `importance` is a number the
  AGENT writes; the engine never invents it.
- Salience weights α..δ are agent-side; engine uses fixed deterministic defaults.

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
```

### 6.2 Planned syntax (M2+)

```sql
SELECT * FROM turn AS OF 2026-08-03T00:00:00Z;
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
