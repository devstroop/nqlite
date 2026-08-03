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
statement      = create_table | insert | relate | select | forget ;

create_table   = 'CREATE' 'TABLE' ident [ 'VECTOR' '<' 'f32' ',' int '>' ] ;

insert         = 'INSERT' 'INTO' recordid '{' object '}' [ 'EMBED' vector ] ;

relate         = 'RELATE' '(' recordid ')' '->' ':' ident '->' '(' recordid ')'
                 [ 'SET' (ident '=' value) (',' ident '=' value)* ] ;

select         = 'SELECT' select_list 'FROM' ident
                 [ 'WHERE' where_clause ]
                 [ 'ORDER' 'BY' '::' order_op ]
                 [ 'LIMIT' int ] ;

forget         = 'FORGET' recordid ;

select_list    = '*' | ident (',' ident)* ;
where_clause   = field_equals | vector_knn | has_embedding ;
field_equals   = ident '=' value ;
vector_knn     = 'vector::similarity' '(' 'embedding' ',' vector ')' 'AND' 'k' '=' int ;
has_embedding  = 'embedding' 'IS' 'NOT' 'NULL' ;
order_op       = 'similarity' | 'salience' | 'score' | 'recency' ;

object         = ( ident ':' value ) ( ',' ident ':' value )* | '' ;
value          = 'null' | 'true' | 'false' | int | float | string
               | vector | '[' value (',' value)* ']' | '{' object '}' ;
vector         = '[' float (',' float)* ']' ;   (* all-float array *)
```

### Planned extensions (NOT in M0 — for M2+)

```
match          = 'MATCH' path ;                     (* graph traversal *)
path           = '(' recordid ')' ( ('->' | '<-') ':' ident [edge_props] )+ ;
closure        = 'CLOSURE' '(' ident ')' ;          (* transitive closure *)
fts            = 'WHERE' '::bm25' '(' ident ',' string ')' ;
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
| `SELECT ... FROM t ...` | Scans table `t`; filters; optional kNN; orders deterministically; limits; returns records (+computed score). |
| `FORGET t:id` | Deletes the record AND all incident edges. |

### 2.3 SELECT pipeline (fixed order)

1. **Scan** — records of table `t` in BTree order.
2. **Filter** — `WHERE`:
   - `field = value`: exact, deterministic equality against body value.
   - `embedding IS NOT NULL`: only records with vectors.
   - `vector::similarity(embedding, $q) AND k = N`: kNN candidate set (see §3).
3. **kNN** — cosine similarity vs query vector; keep top-K by similarity desc,
   tie-break by RecordId asc.
4. **Order** — `ORDER BY ::op` (default: BTree order):
   - `::similarity` — cosine desc (requires kNN query).
   - `::salience` — `α·similarity + β·strength(recency,freq) + γ·importance + δ·score`
     with deterministic engine defaults (α..δ are AGENT-side knobs, not engine
     config); without a kNN query, salience reduces to the feedback term.
   - `::score` — Laplace-smoothed mean over `:voted` edges, desc.
   - `::recency` — `created_at` desc.
5. **Limit** — keep first N.

### 2.4 Transactions

A plan is a `Vec<Statement>` executed atomically in one transaction
(single-writer, snapshot readers — M1 storage; M0 is in-memory). Statements
before a SELECT apply first, so a plan may create/insert/relate/query in one pass.

## 3. Operators

| Operator | Definition | Determinism |
|---|---|---|
| `vector::similarity(embedding, $q)` | cosine similarity `a·b / (|a||b|)`; zero-norm operands => `0.0` | exact f32 |
| `::similarity` | order by cosine desc | total order w/ tie-break |
| `::score` | `(Σ weights + 1) / (n + 2)` over `:voted` edges; `0.5` with no votes (Laplace smoothing) | pure arithmetic |
| `::votes(record)` | `(up, down, net)` counts over `:voted` edges | pure arithmetic |
| `::feedback(record)` | time-decayed recent feedback | engine-clock only |
| `::salience` | `α·similarity + β·strength + γ·importance + δ·score` | fixed order, no races |
| `::bm25` (planned) | BM25 lexical score | pure arithmetic |
| `k = N` | kNN limit | — |

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

-- agent-side feedback: mark the recalled turn as useful
RELATE (agent:main) -> :voted -> (turn:3) SET value = 1, weight = 0.9;

-- rank by community feedback
SELECT * FROM turn ORDER BY ::score LIMIT 5;
```

### 6.2 Planned syntax (M2+)

```sql
MATCH (turn:3) -> :mentions -> (e);
CLOSURE (entity)                       -- transitive closure
SELECT * FROM turn WHERE ::bm25(text, "acme") ORDER BY ::score;
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
  `Filter`, `Order`, `Plan = Vec<Statement>`.
