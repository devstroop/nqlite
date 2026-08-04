# nqlite — Issue Tracker (local, versioned)

Each issue = one feature branch. Feature branches merge into `develop` via PR;
`develop` releases into `main` via PR (see PLAN.md). Statuses:
**open** (backlog) / **in progress** / **merged** (in develop) / **released** (in main).

## Open

(none)

## In progress

(none)

## Merged (in develop)

- **ISSUE-11 — feat/feedback** · M1 · PR #19
  `Order::Votes`/`::Feedback` in nql-ir; `vote_counts` + `feedback_score`
  (time-decayed, deterministic) in engine; parser `ORDER BY ::votes|::feedback`.
- **ISSUE-10 — feat/benchmarks** · M1 · PR #18
  Criterion harness: ingest, knn_bf, select_range, relate (deterministic
  fixed-seed synthetic data; no rand dep).
- **ISSUE-9 — feat/analyzer-planner** · M2 · PR #17
  `Analyzer::analyze` — plan validation (table-decl order, dim mismatches,
  similarity-without-knn) + SELECT enrichment (auto-Order::Similarity).
- **ISSUE-8 — feat/repl** · M2 · PR #12
  `nql-cli` workspace member: interactive REPL + `--script` runner.
- **ISSUE-7 — feat/fuzzing** · M1 · PR #11
  Proptest properties (never-panic, round-trip determinism, error positions) +
  detached cargo-fuzz harness.
- **ISSUE-6 — feat/vector-index** · M0/M1 · PR #10
  `VectorIndex` trait; brute-force exact default; feature-gated HNSW
  (fast-hnsw, seeded).
- **ISSUE-5 — feat/grammar-spec** · M2 · PR #6
  spec/nql.md — grammar + semantics + determinism contract.
- **ISSUE-4 — feat/nql-parser** · M0 · PR #5
  Hand-written lexer + recursive-descent parser (M0 slice), `;` separators,
  line/col errors.
- **ISSUE-3 — feat/engine-core** · M0 · PR #4
  `Database::execute(Plan)` — records, edges, kNN, filters, deterministic
  ordering, Laplace `::score`.

## Released (in main)

- **ISSUE-12 — feat/graph-relations** · M0/M1 · PR #32
  RELATE + 1-hop MATCH; edges with props/weight/time. `MATCH (a) -> :name <- :name`
  (1+ hops, both directions) — IR, parser, analyzer, deterministic engine
  traversal, `QueryKind::Match` result discriminant. CLOSURE remains planned.
- **ISSUE-16 — feat/agent-examples** · M3 · PR #28
  Chat memory, RAG, tool-call ledger, knowledge-graph examples (agent-side
  learning; engine stays deterministic).
- **ISSUE-15 — feat/server-mcp** · M3 · PR #27
  Network server mode + MCP server so agents can talk to nqlite over MCP.
- **ISSUE-14 — feat/bm25-fts** · M2 · PR #26
  Deterministic BM25 lexical index (`::bm25` operator).
- **ISSUE-13 — feat/storage-wal** · M1 · PR #25
  Single-file format + sidecar WAL; ACID (single-writer, snapshot readers);
  crash-safe open. The big one — real persistence.
- **ISSUE-2 — feat/ir-plan** · M0 · PR #1
  Plan/Statement contract — the nql↔nqlite seam.
- **ISSUE-1 — feat/ir-value-types** · M0 · baseline
  RecordId/Value/VectorSpec/RelationEdge/Record/Store + serde (main baseline).
- **ISSUE-0 — workspace scaffold** · baseline
  Cargo workspace (nql, nql-ir, nqlite, nql-cli), README, PLAN, CHANGELOG,
  docs index, CI, branch model.

## Milestone mapping

| Milestone | Issues |
|---|---|
| M0 — Deterministic context engine | ISSUE-1..6 (ir-value-types, ir-plan, engine-core, nql-parser, grammar-spec, vector-index-brute) + ISSUE-12 (graph-relations) |
| M1 — Real storage (file, WAL, ACID) | ISSUE-7..8, 11, 13 (fuzzing, vector-index-HNSW, feedback, storage-wal) |
| M2 — nql grammar real | ISSUE-5, 9, 10, 14 (grammar-spec, analyzer-planner, repl, bm25-fts) |
| M3 — Agents & MCP | ISSUE-15, 16 (server-mcp, agent-examples) |

_This file replaces docs/KANBAN.md. It is maintained by hand per merge; PRs that
close an issue move it from Open → Merged → Released._
