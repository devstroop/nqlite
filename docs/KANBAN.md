# nqlite — Kanban (local, versioned)

Columns: **Backlog** → **In Progress** → **Review** → **Done**.

Kanban is maintained in this file (PR/commit that moves work also moves the card).
Each card = a feature branch; feature branches merge into `develop` via PR,
`develop` releases into `main` via PR (see PLAN.md).

## Backlog

- [ ] **feat/nql-parser** — nql front-end: tokenizer + AST for the M0 slice
      (create/insert/select/relate/knn/match). winnow-based. Storage-agnostic.
- [ ] **feat/engine-core** — nqlite in-memory Store + executor: insert/select
      over records; deterministic iteration order.
- [ ] **feat/graph-relations** — RELATE + 1-hop MATCH; edges with props/weight/time.
- [ ] **feat/vector-index** — `VectorIndex` trait; exact brute-force M0; HNSW
      swap (fast-hnsw) behind the trait.
- [ ] **feat/storage-wal** — single-file format + sidecar WAL; ACID
      (single-writer, snapshot readers); crash-safe open.
- [ ] **feat/feedback** — `::votes` / `::score` (Laplace-smoothed) / `::feedback`
      (decayed) over `:voted` edges; ground-truth regression harness.
- [ ] **feat/fuzzing** — cargo-fuzz (parser) + proptest (storage invariants).
- [ ] **feat/benchmarks** — harness vs sqlite-vec / LanceDB / Chroma
      (ingest TPS, P95 kNN, recall@10, cold-open latency).
- [ ] **feat/grammar-spec** — spec/nql.md full grammar + operator semantics.
- [ ] **feat/analyzer-planner** — analyzer, optimizer, IR stability.
- [ ] **feat/bm25-fts** — deterministic BM25 lexical index.
- [ ] **feat/repl** — nql REPL + docs/playground CLI.
- [ ] **feat/server-mcp** — network server mode + MCP server for agents.
- [ ] **feat/agent-examples** — chat memory, RAG, tool-call ledger, knowledge
      graph examples (agent-side learning, engine stays deterministic).

## In Progress

- [ ] **feat/analyzer-planner** — Analyzer: plan validation (table-decl order,
      dim checks) + SELECT enrichment (auto-Order::Similarity). (wave-3: parallel)
- [ ] **feat/benchmarks** — criterion harness: ingest, knn_bf, select, relate.
      (wave-3: parallel)
- [ ] **feat/feedback** — `::votes`/`::feedback` operators + vote_counts /
      feedback_score in engine (D9). (wave-3: parallel)

## Review

- [x] **feat/vector-index** — `VectorIndex` trait; brute-force exact default;
      feature-gated HNSW (fast-hnsw, seeded). Merged via PR #10 (2026-08-03).
- [x] **feat/fuzzing** — proptest properties (never-panic, round-trip, error
      positions) + detached cargo-fuzz harness. Merged via PR #11 (2026-08-03).
- [x] **feat/repl** — `nql-cli` member: REPL + `--script` runner. Merged via
      PR #12 (2026-08-03).
- [x] **feat/engine-core** — nqlite engine: Database::execute over Store
      (create/insert/relate/select/forget, kNN, filters, deterministic order,
      Laplace ::score), 12 tests. Merged into develop via PR #4 (2026-08-03).
- [x] **feat/grammar-spec** — spec/nql.md grammar + semantics. Merged into
      develop via PR #6 (2026-08-03).

## Done

- [x] Workspace scaffold — Cargo workspace (nql, nql-ir, nqlite), README,
      PLAN, CHANGELOG, docs index. (on `main` via initial commit)
- [x] **feat/ir-value-types** — nql-ir contract: RecordId/Value/VectorSpec/
      RelationEdge/Record/Store + serde. (In `main` baseline; shipped with the
      scaffold commit — see git log 8155316.)
- [x] **feat/ir-plan** — Plan/Statement contract (nql↔nqlite seam). Merged
      into develop via PR #1 (2026-08-03).

## Milestone mapping

| Milestone | Features |
|---|---|
| M0 — Deterministic context engine | feat/ir-value-types, feat/nql-parser, feat/engine-core, feat/graph-relations (1-hop), feat/vector-index (brute-force) |
| M1 — Real storage (file, WAL, ACID) | feat/storage-wal, feat/vector-index (HNSW), feat/feedback, feat/fuzzing, feat/benchmarks |
| M2 — nql grammar real | feat/grammar-spec, feat/analyzer-planner, feat/bm25-fts, feat/repl |
| M3 — Agents & MCP | feat/server-mcp, feat/agent-examples |