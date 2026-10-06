# Changelog

All user-visible changes to nqlite are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
  are absent from the row (spec §2.3 step 7).
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
