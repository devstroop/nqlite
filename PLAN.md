# nqlite — Development Plan (maintainer roadmap)

State: M0–M3 features shipped (see ISSUES.md for the tracked issue log);
this is the living roadmap. README.md is the state-independent overview;
this file drives work. Research findings that sharpen these milestones are
captured under `docs/`.

Goal: a deterministic, serverless, context-first database for AI agents —
"SQLite for AI memory": SQLite ergonomics, one embedded file, ACID,
zero-LLM-dependency in the engine. (Framing: see docs/positioning.md.)

---

## Guiding principles (from docs/decisions.md)

1. Engine is 100% deterministic, zero LLM dependency, forever.
2. Learning lives in the agent; the store is the durable substrate.
3. Context is chained via records + typed graph relations + embeddings + time,
   queryable in one pass.
4. Harden everything (fuzz, property tests, crash-safety) as we build.
5. Docs stay in sync; README stays state-independent (roadmap lives here).

## Locked technical choices (research-informed — see docs/decisions.md, docs/research.md)

- Storage: our OWN single-file store + sidecar WAL (NOT RocksDB/LevelDB-backed).
  SQLite-style ACID: single-writer, snapshot readers, WAL.
- Vectors: BYO-vector only (agent pushes `f32[]`). Engine never encodes.
  `Provider` trait lives in a CLIENT-side helper, not the engine.
- Vector index: exact brute-force in M0 (correctness + determinism), behind a
  `VectorIndex` trait; swap to HNSW in M1 (`fast-hnsw` leading candidate, else
  `USearch`).
- Concurrency determinism: snapshot isolation => per-snapshot reads deterministic.
- Graph: 1-hop `MATCH` in M0; recursive `CLOSURE` in M1.
- Competitive north star: SurrealDB ("context layer for agents") is the heavy
  incumbent; sqlite-vec is the bolt-on. We own the file-serverless + zero-LLM +
  one-transaction corner. Full positioning in docs/comparison.md.

---

## Milestone 0 — Deterministic context engine (spike)

Goal: prove the whole thing works end-to-end in one file, offline, deterministic.

- [ ] Workspace scaffold (done: Cargo workspace, 3 crates, README, decisions.md).
- [ ] `nql-ir`: value types, RecordId, Value, Vec, Datetime, RelationEdge.
- [ ] `nql`: minimal parser for the v0-visionable language slice
      (create/insert/select/relate/knn/match).
- [ ] `nqlite`: in-memory engine that executes the nql Plan.
      - Store: records by table:id; IndexMap/BTree.
      - Relations: adjacency table (from, edge, to, props).
      - Vector: brute-force kNN initially (exact, correct); HNSW later.
      - Lexical: token/BM25 exact.
- [ ] Hybrid query: `SELECT ... WHERE similarity(...) ORDER BY ...` + a one-hop
      graph MATCH, in a single Plan.
- [ ] Determinism harness: property test — same input => byte-identical Plan & result.
- [ ] Local offline tests only. No network, no model, no key.
- Exit: `cargo test` green; M0 demo runs fully offline; README status consistent.

## Milestone 1 — Real storage engine (file, WAL, ACID)

- [ ] File format: single-file store + sidecar WAL; crash-safe open.
- [ ] ACID: single-writer + snapshot readers; transactions; fsync boundaries.
- [ ] Evolution the three indexes into the same transaction boundary.
- [ ] Replace in-memory vector w/ an embedded ANN (HNSW) — evaluate hnswlib-rs /
      USearch / instant-distance (see docs/research).
- [ ] Feedback operators (D9): `::votes`, `::score` (Laplace-smoothed mean),
      `::feedback` (decayed) — pure deterministic aggregation over `:voted` edges.
- [ ] Feedback-as-ground-truth regression harness: (query, retrieved, relevance)
      triples → recall@K / precision@K vs real usage data; catches retrieval
      regressions on grammar/index/fusion changes.
- [ ] Fuzz the parser (`cargo-fuzz`) and storage (proptest invariants).
- [ ] Benchmark harness vs sqlite-vec, LanceDB, Chroma (ingest TPS, P95 kNN,
      recall@10, cold-open latency).

## Milestone 2 — nql grammar real (spec), analyzer, IR stability

- [ ] Write `spec/nql.md` full grammar + semantics (docs-in-sync-with-code).
- [ ] AST -> plan lowering with a real optimizer pass.
- [ ] FTS (BM25) + vector + graph combined optimizer (pipelining/pushdown).
- [ ] REPL / nql-docs for DX.

## Milestone 3 — Agents & MCP wiring

- [x] Server mode (network) + MCP server so agents talk to nqlite over MCP
      (`nql-server` TCP/stdio line protocol, `nql-mcp` — issues #15/#21).
- [x] Example client integrations: chat memory, RAG, tool-call ledger,
      knowledge graph (`nqlite/examples/` — issues #16, wave-10 adds
      idempotent memory + context chain).
- [ ] Deterministic extension points for the agent-side "learning" (edges, NEAR,
      count/decay) — clearly outside the engine.
- [x] Benchmarks on `context_of(record)` / full session recall patterns
      (nql-bench `context_ms`/`recall_ms` scenarios, issue #78).

## Later / stretch

- Provider trait for a future local embedder (still not in the engine: a client
  helper crate, `nqlite-providers`). Q-decision: keep outside core, gate behind
  feature.
- Multi-tenant / distributed (likely out of scope; SQLite-style single-node first).
- ACID merge of parallel agents (optimistic concurrency for LWW).

---

## Milestone acceptance (shared bar)

Every milestone ships with:
- deterministic tests (proptest, fuzz) green,
- docs in sync (README/CHANGELOG/PLAN),
- benchmark numbers where relevant,
- zero LLM dependency demonstrated,
- one demo.

## Status log

- 2026-08-03: workspace scaffolded (/mnt/ext1/nqlite-workspace), philosophy
  locked (docs/decisions.md), research kicked off (delegated), PLAN skeleton.
- 2026-08-03: research completed directly (docs/research.md + Appendix A);
  decisions D1–D9 locked; nql-ir value types shipped (main baseline).
- 2026-08-03: repo created github.com/devstroop/nqlite (public, default main);
  branch model live (main → develop → feat/*, PRs + CI protection);
  per-branch worktrees under /mnt/ext1/nqlite-worktrees/.
- 2026-08-03: **wave-1** merged — ir-plan, engine-core, nql-parser, grammar-spec,
  + end-to-end integration tests + `;` separators. nql→engine pipeline proven.
- 2026-08-03: **wave-2** merged — vector-index (VectorIndex trait, brute-force
  default, feature-gated HNSW), fuzzing (proptest + cargo-fuzz), repl (nql-cli).
  Released to main via PR #13. 45+ tests across 4 crates + nql-cli.
- 2026-08-03: **wave-3** merged — analyzer (table-decl order, dim checks,
  SELECT enrichment), benchmarks (criterion, deterministic fixed-seed data),
  feedback (::votes/::feedback, D9). Released via PR #19.
- 2026-08-03: **wave-4** merged — storage-wal (single-file + sidecar WAL, ACID,
  crash-safe), bm25-fts (engine Filter::Bm25), server-mode (line protocol,
  TCP + stdio), agent-examples (chat memory, RAG loop, tool ledger).
  Released via PR #29.
- 2026-08-04: **wave-5** merged — graph-relations 1-hop MATCH (IR, parser,
  analyzer, deterministic traversal, QueryKind::Match). Released via PR #33.
  ISSUE-12 closed; tracker backlog empty.
- 2026-08-04: **audit** — full-project audit (code, spec, docs, tests). Found
  + fixed: ::score ignored nql-created votes (colon mismatch); MATCH written to
  WAL; ::bm25 unreachable from grammar; CLI had no persistence (`--db`);
  dead `tracing` dep; docs honesty (MCP re-scope, spec order_op/statement,
  PLAN state). ISSUE-17..19 tracked the follow-up.
- 2026-08-07: **wave-10** merged — idempotent-memory + context-chain agent
  examples (gated fleet process, deterministic/zero-LLM at engine level),
  release notes (`docs/release-wave-10.md`). Released via PR #66; develop→main
  via PR #67.
- 2026-08-07: **hardening wave** — GitHub issue migration (#65), crate publish
  metadata + `publish.yml` (#72, issue #68), benchmark wave (#73),
  MCP inputSchema dict fix (#74, issue #70), determinism CI job (#75).
- 2026-08-08: **docs + perf wave** — QueryResult doc fix (#76, issue #63),
  ISSUES.md finalized as archive (#77, issue #62), M3 session-recall
  benchmarks in nql-bench (#79, issue #78), adjacency-indexed CLOSURE
  traversal — 12x on the chain shape (#81, issue #80). M3 remaining:
  agent-side learning extension points.

## Tracker

- **ISSUES.md** is the issue tracker (replaces docs/KANBAN.md — see its footer).
  Each issue = one feature branch; statuses open → in progress → merged (in
  develop) → released (in main). docs/KANBAN.md is retained as an archive and
  is no longer maintained.