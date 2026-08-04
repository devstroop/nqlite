# Changelog

All user-visible changes to nqlite are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `MATCH (a) -> :name <- :name ...` graph traversal (1+ hops, both directions):
  parser rule, analyzer validation, deterministic engine execution (edges
  scanned in append order, endpoints deduped keeping first appearance), and a
  `QueryKind::Match` result discriminant alongside `QueryKind::Select`.
  `nql-cli` and `nql-server` render MATCH results with the walked path.
- Single-file persistence + sidecar WAL (`StoreFile`): ACID, crash-safe,
  deterministic reopen (byte-identical store).
- Deterministic BM25 lexical filter: `WHERE ::bm25(field, "query") [AND k = N]`.
- `nql-server` (TCP + stdio line protocol) with per-line table-declaration
  tracking and an `OK`/`ERR` response envelope.
- Agent pattern examples: chat memory, RAG loop, tool ledger.
- `ORDER BY ::votes | ::feedback` (vote counts and time-decayed feedback score
  over `:voted` edges), Criterion benchmark harness.

### Changed
- `QueryResult` now carries a `kind` (`QueryKind::Select | QueryKind::Match`)
  instead of a `select` field; rows are unchanged.

### Deprecated / Removed / Fixed / Security
- None.
