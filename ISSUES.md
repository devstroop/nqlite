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

- **ISSUE-28 — chore/bench-run** · M2 · PR #58
  First cross-DB run with real competitors on the deterministic corpus
  (1000 rows, dim-8, seed 42, 50 kNN queries): nqlite 3.74ms ingest /
  443.83ms brute-force kNN / 608.23ms bm25 / 1260.78ms hybrid vs
  sqlite-vec 22.21/3.43, lancedb 4.17/158.67, chroma 200.21/38.85.
  Harness fixes (lancedb pyarrow schema, cell alignment, venv recipe),
  report-2026-08-04.md with methodology + honest reading.
- **ISSUE-27 — feat/memory-blocks** · M2/M3 · PR #57
  `MEMORY <name>` context blocks: each memory is a full sub-store (own
  records, edges, clock + history — AS OF composes); plan-level context via
  execute_in_context; WAL replay preserves scoping across reopen.
- **ISSUE-26 — feat/temporal-reads** · M2 · PR #56
  `SELECT ... AS OF <ts>` time-travel via deterministic history replay
  (logical clock, never wall-clock): Store gains clock + persisted mutation
  history; upserts and FORGETs reconstruct correctly; is_mutating fixed to
  exclude CLOSURE; storage FORMAT_VERSION 1→2.

- **ISSUE-25 — chore/positioning** · M2/M3 · PR #51
  Positioning + messaging: docs/positioning.md (pitch, what it is/isn't,
  honest comparisons, terminology); README headline → "SQLite for AI memory"
  ("neural" = embeddings-as-first-class-data); features list gains ::bm25,
  hybrid retrieval, MATCH + CLOSURE; PLAN/spec/decisions framing note;
  CHANGELOG brought current.
- **ISSUE-24 — chore/bench-compare** · M2 · PR #50
  Cross-DB benchmark harness: nql-bench (workspace member) + 
  scripts/bench-compare/bench.py — same deterministic corpus (xorshift64*
  seed 42) across nqlite and (when installed) sqlite-vec / LanceDB / Chroma;
  JSON or table report; skips cleanly on missing drivers.
- **ISSUE-23 — feat/hybrid-retrieval** · M2 · PR #49
  Combined optimizer: `::bm25(field, "q") AND vector::similarity(embedding,
  $v) AND k = N` (clauses in either order) — both signals computed and fused
  with deterministic reciprocal-rank fusion (1/(60+rank)); fused score is the
  sort key; cap = min(knn k, bm25 k, LIMIT). Parser + engine + spec §2.6.

- **ISSUE-22 — feat/feedback-harness** · M1 · PR #44
  D9 ground-truth regression harness: relevance triples stored in the DB as
  `(query) -> :relevant -> (doc)` edges; `nqlite::harness` recall@K /
  precision@K; `tests/regression.rs` asserts kNN recall@4 ≥ 1.0 and BM25
  recall/precision@4 ≥ 0.75 over a deterministic synthetic corpus. Catches
  retrieval regressions on grammar/index/fusion changes.
- **ISSUE-21 — feat/mcp-server** · M3 · PR #43 (+#45)
  `nql-mcp`: MCP server (stdio, official rmcp SDK) exposing nqlite as tools
  (execute_nql, create_table, insert_record, relate, select, match_path,
  closure, forget) with deterministic JSON results — closes the ISSUE-15 MCP
  gap. PR #45 synced it with the ISSUE-20 IR (closure tool + edge-property
  steps).
- **ISSUE-20 — feat/graph-core** · M1 · PR #42
  CLOSURE transitive traversal (BFS fixpoint, first-visit order, depth
  scores, cycle-safe) + MATCH edge-property filters (`WHERE <prop> = <value>`
  per step) — IR, parser, analyzer, engine, server/cli; `QueryKind::Closure`.
  Graph story complete (1-hop MATCH from ISSUE-12 + closure).
- **ISSUE-19 — chore/audit-hygiene** · M0/M1 · PR #38
  CLI `--db <path>` persistent sessions + `:flush`; dropped dead `tracing`
  dep; docs honesty (ISSUE-15 MCP re-scope, PLAN status log, KANBAN archived,
  spec order_op/statement sync).
- **ISSUE-18 — feat/bm25-grammar** · M2 · PR #37
  `WHERE ::bm25(field, "query") [AND k = N]` wired into the nql parser +
  analyzer (engine `Filter::Bm25` already existed); spec sync (fts moved from
  planned to M0 grammar, `order_op` gains `::votes`/`::feedback`, statement
  list gains `match`).
- **ISSUE-17 — feat/fix-votes-wal** · M1 · PR #36
  `::score` now matches the parser's no-colon `"voted"` convention (votes
  created via nql count again); read-only `MATCH` no longer written to the
  WAL (`is_mutating` excludes SELECT and MATCH).
- **ISSUE-12 — feat/graph-relations** · M0/M1 · PR #32
  RELATE + 1-hop MATCH; edges with props/weight/time. `MATCH (a) -> :name <- :name`
  (1+ hops, both directions) — IR, parser, analyzer, deterministic engine
  traversal, `QueryKind::Match` result discriminant. CLOSURE remains planned.
- **ISSUE-16 — feat/agent-examples** · M3 · PR #28
  Chat memory, RAG, tool-call ledger, knowledge-graph examples (agent-side
  learning; engine stays deterministic).
- **ISSUE-15 — feat/server-mcp** · M3 · PR #27
  Deterministic line-protocol server (TCP + stdio). NOTE: the MCP server was
  NOT part of this issue (renamed from feat/server-mcp to reflect what
  actually landed); the MCP server shipped later via ISSUE-21 (PR #43/#45).
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
| M1 — Real storage (file, WAL, ACID) | ISSUE-7..8, 11, 13, 17, 20, 22 (fuzzing, vector-index-HNSW, feedback, storage-wal, fix-votes-wal, graph-core, feedback-harness) |
| M2 — nql grammar real | ISSUE-5, 9, 10, 14, 18, 23..28 (grammar-spec, analyzer-planner, repl, bm25-fts, bm25-grammar, hybrid-retrieval, bench-compare, positioning, temporal-reads, memory-blocks, bench-run) |
| M3 — Agents & MCP | ISSUE-15, 16, 21 (server-mode, agent-examples, mcp-server) |

_This file replaces docs/KANBAN.md. It is maintained by hand per merge; PRs that
close an issue move it from Open → Merged → Released._
