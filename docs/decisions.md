# nqlite — Consolidated Design Intent & Decisions

> State: design-in-progress (2026-08). This file captures the agreed philosophy and
> open questions BEFORE the research-backed plan. It is the source of truth for WHY.
> The PLAN.md in this repo is the action plan; this file is the reasoning.

## 0. The one-line pitch
A **context-first, neural database** that is serverless like SQLite — a single
embedded file, ACID, deterministic, offline — for storing AI agents' contexts,
their relationships, and their neural signals (embeddings), so that agents can
read and write their own evolving context during a live conversation.

## 1. Non-negotiable principles (locked)

1. **The engine is 100% deterministic and NEVER depends on an LLM, now or in the
   future.** No summarization, no chunking, no embedding, no compaction, no
   entity extraction runs inside the engine. Full stop.
2. **"Neural" = the data model + operators.** The database stores records,
   typed relations (graph), embedding vectors, and temporal relevance. Vector
   similarity is a deterministic distance computation on vectors *somebody else*
   already produced. The search itself is pure algorithm -> deterministic,
   fuzzable, reproducible.
3. **The learning lives in the agent, not the DB.** During an active conversation
   an agent (running your model-adapter, or a plain rule engine) decides what to
   write, relate, embed, and recall. The store is the durable substrate; the
   agent is the student. This makes the DB a truth machine that outlives any model.
4. **Context is chained through graph relations.** A conversation/knowledge graph:
   `n(3) ->:refs-> entity(8)`, `n(3) ->:follows_from-> n(2)`, plus vector fields.
   Deterministic traversal + kNN in one transaction = the core differentiated thing
   nobody does in a single embedded engine today.
5. **Harden from the ground up.** Fuzz the parser, ACID + WAL crash-safety,
   deterministic tests, benchmarks vs sqlite-vec/Chroma/LanceDB/SurrealDB.

## 2. Stack & crate split (decided)

One git workspace, one Cargo workspace, three crates (logical separation, single
repo):

```
nqlite-workspace/
  nql/        # front-end ONLY: parser + AST + analyzer. MUST NOT know storage exists.
  nql-ir/     # tiny shared contract: the lowered Plan / IR both ends compile against.
  nqlite/     # the engine: storage + indexes + executor that runs an nql Plan.
  spec/       # nql grammar spec (nql.md), file-format spec, operator semantics.
  docs/       # design reasoning, hardening, comparison, research notes.
```

Rationale for one repo (not two): a shared in-process `nql-ir` beats a serialized
cross-language contract until a real second consumer appears (then the IR can be
promoted to JSON/protobuf). The front-end is kept *compiler-enforced* isolated via
Cargo dev-dependencies so the language never bends to engine internals.

## 3. Mental model

- **Namespace › database › table**, records as `table:id` (SurrealDB-style).
- Field types: scalar, time, nested docs/arrays, `VECTOR<f32,N>`, and typed relations.
- Relate: directed + named edges with their own properties, e.g.
  `(person)->:knows->(person)`, `(source)->:contains->(chunk)`.
- One transaction covers: records + edges + vectors + their indexes.
- A single query can traverse the graph AND do kNN in one pass (hybrid).

## 4. The determinism/neural spectrum (terminology)

We distinguish three "intelligence" tiers and deliberately keep them apart:

1. **Deterministic / statistical (IN the engine, always on, offline, zero-LLM):**
   BM25 lexical, HNSW/ANN similarity, distance, re-rank-by-distance, graph
   traversal, closure, PageRank, co-occurrence edges, recency/time-decay salience,
   keyword/token NER. Reproducible => property-testable.
2. **Optional LLM/ generative (OUTSIDE, agent side, never in the engine):**
   semantic embedding computation, abstractive summarization/compaction,
   high-quality entity extraction, cross-encoder semantic reranking. A client may
   do these against your model-adapter gateway. The engine doesn't care.
3. **Agent orchestration:** writing decided edges, deciding what to forget,
   choosing what to recall — the agent's job, on top.

## 5. Open decisions (to be resolved in the plan / early milestones)

- D1. Vector ingestion in M0: **BYO-vector only** (agent pushes `f32[]`; engine
  never encodes) — lean yes, with a `Provider` trait scaffolded for a future
  local embedder. (Leading: BYO-vector, provider trait, no bundled embedder in M0.)
  **RESOLVED (research-informed)**: BYO-vector; `Provider` trait lives OUTSIDE
  the engine (client helper crate) — engine never encodes, ever.
- D2. Graph in v1: yes — relations + traversal are core to the pitch. Traversal
  scope (1-hop MATCH vs recursive CLOSURE/transitive closure) TBD by milestone.
  **RESOLVED**: 1-hop MATCH in M0; recursive CLOSURE added in M1 (graph core).
- D3. Forgetting/decay: agent-driven `FORGET` + deterministic time-decay operator
  — in v1. LLM-driven compaction: NEVER in engine; document why.
  **RESOLVED**: `FORGET`/decay are engine operators (deterministic). Compaction
  is agent-side; engine never summarizes.
- D4. Concurrency model: SQLite-style single-writer + snapshot readers, WAL.
  **RESOLVED**: same as SQLite (single-writer, snapshot readers, sidecar WAL).
- D5. File format: single file, WAL/journal sidecar not in same file. TBD: VM vs
  custom pager. Research will inform.
  **RESOLVED**: own single-file store + sidecar WAL (NOT RocksDB/LevelDB-backed —
  the whole point is a dependency-light SQLite-style file we fully own).
- D6. Whether Boost results process remains deterministic under concurrency
  (linearizability) — yes, snapshot isolation.
  **RESOLVED**: snapshot isolation => deterministic per-snapshot reads.
- D7. Language ergonomics: `SELECT`-ish + SurrealDB `RELATE` + Cypher-like MATCH
  hybrid; decide the grammar flavour to lock in the spec.
  **RESOLVED (direction)**: `SELECT ... WHERE vector::similarity(...) ... ORDER BY`
  + `RELATE (a)->:edge->(b)` + `MATCH (a)->:edge->(b)` + temporal. Full grammar
  lock happens when spec/nql.md lands in M2, but M0's parser exercises the core
  SELECT/INSERT/RELATE/MATCH/knn slice.
- D8. (new) Vector index strategy: exact brute-force in M0 (correctness +
  determinism), swap to HNSW behind a `VectorIndex` trait in M1. Candidate
  crates: `fast-hnsw` (leading: actively maintained, better recall/QPS), or
  `USearch`. Keep the trait so we can A/B and swap.

## 6. Benchmark targets (aspirational, tune later)

- reopen cold and query a 100K-record store in milliseconds.
- kNN recall@10 >= 0.95 on a standard set; ANN memory bounded.
- deterministic: same input -> byte-identical output across runs.
- crash in WAL mid-commit -> auto-recover, no corruption.
- ingest 100K records < a few seconds.

## 7. Tone for docs (repo conventions)

- README.md stays state-independent (description, features, quick start, security,
  license, contributing) — no milestone tables ("done" markers, status) in README.
- PLAN.md is the maintainer roadmap, milestone-driven.
- CHANGELOG.md is user-visible changes.
- All docs: community-standard, keep in sync with code as you already do.