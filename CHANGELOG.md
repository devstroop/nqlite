# Changelog

All user-visible changes to nqlite are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `ORDER BY <field> [DESC]` (issue #117): sort by a body field under the same
  proptest-pinned total order the filters use (`Value::cmp_total`); absent
  fields rank as `null`, `DESC` reverses the key only (ties keep ascending
  RecordId), and a field that exists on no record of the table errors loudly
  (`UnknownSortField`) instead of silently sorting all-equal. Like `::recency`,
  the explicit structural sort applies in kNN/BM25 modes (overall ORDER BY
  precedence tracked in #119).
- History compaction (issue #95): `PRUNE HISTORY` replaces the mutation
  history with a deterministic snapshot at the current clock (retaining the
  `CreateTable` declarations so re-seeding keeps working), bounding memory
  growth and making later `AS OF` reads rebuild from the snapshot instead of
  ts0. `AS OF` earlier than the snapshot fails loudly with `HistoryPruned`
  instead of returning a partial view; compaction covers MEMORY blocks,
  survives reopen via the WAL, and is durable at the next checkpoint.
  Spec §2.7 retention contract + file-format downgrade caveat documented.
- Temporal graph traversal (issue #92): `MATCH ... AS OF <ts>` and
  `CLOSURE ... AS OF <ts>` replay the mutation history and traverse the
  reconstructed snapshot — the same machinery as `SELECT ... AS OF`, and it
  composes with `MATCH ... COUNT`. The typed MCP `match`/`closure` tools gain
  an `as_of` parameter (schema-advertised, parity with `select`).
- Comparison and range filters (issue #93): `WHERE` accepts `!=`, `<`, `<=`,
  `>`, `>=`, `IN [..]`, and `BETWEEN a AND b` — evaluated over a documented,
  proptest-pinned total order of values (`null < bool < number < string <
  array < doc < vector < ref`, exact numeric comparison). `=`/`!=`/`IN` keep
  exact equality (complementary), records without the field never match, and
  the same predicates now work in `MATCH`/`CLOSURE` edge-property filters.
- Counting and pagination (issue #94): `SELECT COUNT(*)` returns one
  `{"count": n}` row (filtered total — ordering/offset/limit never affect
  it); `LIMIT n OFFSET m` (or `OFFSET` alone) paginates after ordering;
  `MATCH ... COUNT` reports edge-path instances so parallel-edge
  multiplicity is observable. Spec §1/§2.3/§2.5 and README updated.
- Per-query salience weights (issue #88): `ORDER BY ::salience(α, β, γ, δ)`
  tunes all four spec §2.3 terms — `α·similarity + β·strength(recency,freq)
  + γ·importance + δ·score` — with exactly four comma-separated numbers.
  Bare `::salience` keeps the deterministic engine defaults (0.7/0/0/0.3),
  value-identical to the previous behavior. Spec §2.3/§3/§5, decisions D9,
  and agent-patterns now document the same formula.
- Recall@K quality harness for the ANN path (issue #96): `nql-bench --recall`
  measures HNSW recall@10/50/100 against exact brute-force top-K on a
  deterministic dim-64 vector set (`--dim`, `--hnsw-m/--hnsw-efc/--hnsw-ef`
  for parameter sweeps; without `--features hnsw` the report says so). A gate
  test (`nqlite/tests/recall.rs`, `--features hnsw`) asserts recall@10 ≥ 0.95
  — decisions §6's target — measured **0.96 at 5k rows / 0.81 at 50k**
  (default params degrade with scale; sweep flags exist for tuning).
  `scripts/bench-compare/bench.py` carries a `rec@10` column for nqlite.
  ([#96](https://github.com/devstroop/nqlite/issues/96))
- `nql-mcp` typed `select` tool gains **`as_of`** (temporal read: logical
  timestamp, same semantics as `SELECT ... AS OF`) and **`memory`** (read a
  `MEMORY <name>` block's sub-store) — the capabilities previously reachable
  only by knowing to smuggle them through `execute_nql`. The tool schema
  advertises both (discoverable by agents), and `execute_nql`'s description
  now documents the full grammar: `AS OF`, `MEMORY` prefixing rules (each
  program starts at root), edge filters, hybrid retrieval.
  ([#90](https://github.com/devstroop/nqlite/issues/90))
- WAL plan-boundary marker (`Statement::ContextReset`): replay now resets the
  memory context exactly where the runtime does — **every plan starts at the
  root** (spec §2.8). Before this, the flat write-ahead log carried
  `current_memory` across plan boundaries, so a root write issued *after* a
  plan that ended inside a `MEMORY` block was replayed into that memory —
  silent data misplacement on every persistent line-oriented path
  (`nql-server --db`, the `nql-cli` REPL). `Database::execute` appends the
  marker after a plan's mutating statements; it is WAL-only (never in
  `Store::history`, no clock tick) and added as the last `Statement` variant
  so existing postcard tags stay stable. Downside note: a WAL written by this
  version consumed by a pre-marker binary truncates at the first marker.
  Single-plan scripts and `AS OF` (per-store history) were unaffected.
  ([#109](https://github.com/devstroop/nqlite/issues/109))
- `nql-server [--db FILE]` (both TCP and stdio modes): the line-protocol
  server can now serve a persistent single-file store with the same semantics
  as `nql-cli --db` (WAL, checkpoint, single-writer lock). On reopen the
  analyzer's cross-line table context is re-seeded by replaying the persisted
  mutation histories (root + memory blocks), so tables created before a
  restart — including empty, dimension-less ones that exist only as history
  statements — stay usable. Clean `error: ...` on open failure (e.g. `Locked`).
  ([#89](https://github.com/devstroop/nqlite/issues/89))
- Engine now stamps `created_at` on every mutation when it arrives unset —
  records get the statement's logical timestamp on INSERT, edges on RELATE —
  the contract `nql`'s parser has always documented ("Engine clocks
  created_at"). Before this, every timestamp was 0: `ORDER BY ::recency`
  degenerated to ascending record id (the *oldest* first) and `::feedback`'s
  decay was inert (every vote age 0). Explicit IR-provided values pass
  through unchanged; stamps are re-derived from statement order on WAL/AS OF
  replay, preserving the determinism contract; nql `SET created_at = ...`
  still lands in edge props (the field is engine-clocked).
  ([#107](https://github.com/devstroop/nqlite/issues/107))
- Field projection: `SELECT a, b FROM t` now returns only the listed fields
  (previously parsed and silently discarded — every row came back full, and
  typos in the field list succeeded unnoticed). Presentation-only: ranking
  and limits still run on full records; `SELECT *` unchanged; missing keys
  are absent from the row (spec §2.3 step 8).
  ([#91](https://github.com/devstroop/nqlite/issues/91))
- nql comments per spec §1: `--` to end of line and `/* */` block comments
  are skipped by the lexer (previously both were lex errors, so spec §6's own
  examples failed to parse); unterminated block comments report a positioned
  error. ([#86](https://github.com/devstroop/nqlite/issues/86))
- Hybrid retrieval: `WHERE ::bm25(field, "q") AND vector::similarity(embedding,
  $v) AND k = N` (clauses in either order) — lexical + vector signals fused
  with deterministic reciprocal-rank fusion (RRF); `ORDER BY` ignored in
  hybrid mode, cap = min(knn k, bm25 k, LIMIT).
- `CLOSURE (a) -> :name` transitive traversal (BFS fixpoint, first-visit
  order, BFS-depth scores, cycle-safe) and per-step MATCH edge-property
  filters (`MATCH (a) -> :name WHERE <prop> = <value>`); `QueryKind::Closure`.
- `nql-mcp`: MCP server (stdio, official rmcp SDK) exposing nqlite as tools
  (`execute_nql`, `create_table`, `insert_record`, `relate`, `select`,
  `match_path`, `closure`, `forget`) with deterministic JSON results.
- Retrieval regression harness (`nqlite::harness`): ground-truth `:relevant`
  edges, `recall_at_k` / `precision_at_k`, deterministic synthetic-corpus
  regression suite.
- Cross-DB benchmark harness (`nql-bench` + `scripts/bench-compare/`):
  same deterministic corpus, measured across nqlite and (when installed)
  sqlite-vec / LanceDB / Chroma.
- CLI `--db <path>` persistent sessions + `:flush` checkpoint; `nql-server`
  TCP + stdio; agent pattern examples; `ORDER BY ::votes | ::feedback`;
  Criterion benchmark harness.
- `MATCH (a) -> :name <- :name ...` graph traversal (1+ hops, both directions):
  parser rule, analyzer validation, deterministic engine execution (edges
  scanned in append order, endpoints deduped keeping first appearance), and a
  `QueryKind::Match` result discriminant alongside `QueryKind::Select`.
  `nql-cli` and `nql-server` render MATCH results with the walked path.
- Single-file persistence + sidecar WAL (`StoreFile`): ACID, crash-safe,
  deterministic reopen (byte-identical store).
- Deterministic BM25 lexical filter: `WHERE ::bm25(field, "query") [AND k = N]`.

### Changed
- `QueryResult` now carries a `kind` (`QueryKind::Select | Match | Closure`)
  instead of a `select` field; rows are unchanged.
- `::score` now matches the parser's no-colon `"voted"` edge convention
  (votes created through nql count again); read-only `MATCH` is no longer
  written to the WAL.
- Framing: "SQLite for AI memory" (see `docs/positioning.md`); "neural" now
  means embeddings-as-first-class-data, not engine intelligence.

### Deprecated / Removed / Fixed / Security
- Removed dead `tracing` dependency.
- Fixed: `::bm25` now reachable from the grammar (`WHERE ::bm25(...)`), MATCH
  edge-property filters, and vote-score colon mismatch (above).
- Fixed: `::score` no longer counts `SET value = -1` downvotes as upvotes —
  a `:voted` edge without an explicit `weight` now takes its weight from the
  signed `value` (agreeing with `::votes`/`::feedback`); explicit `weight`
  still overrides. Spec §3/§4 and decisions D9 updated (`weight: -1..=1`).
  ([#85](https://github.com/devstroop/nqlite/issues/85))
- Fixed: concurrent `--db` openers no longer lose acknowledged writes —
  `StoreFile::open` now takes an exclusive sidecar lock (`<name>.nql.lock`;
  `flock(2)` on unix, `create_new` file elsewhere, MSRV 1.82 preserved) and a
  second opener fails fast with a `Locked` storage error. Spec
  `file-format.md` §4 documents the enforcement.
  ([#84](https://github.com/devstroop/nqlite/issues/84))
- Fixed: IR-built edges that keep the leading `:` (`:voted`, as the
  chat_memory example constructs them) are now honored by every reader —
  `::score`/`::votes`/`::feedback` and MATCH/CLOSURE step matching — instead
  of silently matching nothing (the example's importance knob was dead and
  its printed claim false). The example now asserts its own output.
  ([#98](https://github.com/devstroop/nqlite/issues/98))
