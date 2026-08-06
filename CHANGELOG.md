# Changelog

All user-visible changes to nqlite are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
