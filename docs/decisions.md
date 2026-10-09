# nqlite — Consolidated Design Intent & Decisions

> State: **locked design intent — the source of truth for WHY.** D1–D9 are
> resolved (status inline); §6 reconciles the benchmark targets against
> measurements (last reconciled 2026-10-07). Action plan: [PLAN.md](../PLAN.md).

## 0. The one-line pitch
A **context-first, neural database** that is serverless like SQLite — a single
embedded file, ACID, deterministic, offline — for storing AI agents' contexts,
their relationships, and their neural signals (embeddings), so that agents can
read and write their own evolving context during a live conversation.

> **Terminology note (2026-08-04)**: "neural" here means *embeddings are
> first-class data* — nothing in the engine learns (see §4 and
> [positioning.md](positioning.md)). Public framing prefers "SQLite for AI
> memory"; the design intent is unchanged.

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
  nqlite/     # the engine: storage + indexes + executor that runs an NQL Plan.
  spec/       # NQL grammar spec (nql.md), file-format spec, operator semantics.
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

1. **Deterministic / statistical (IN the engine, always on, offline, No-LLM):**
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
  **LANDED (2026-08-04)**: `MATCH (a) -> :name <- :name ...` (1+ hops, both
  directions) in nql-ir (`MatchPath`/`MatchStep`), parser, analyzer, and engine.
  Deterministic: edges scanned in append order, endpoints deduped keeping first
  appearance, missing start record => empty result, dangling edges skipped.
  `QueryResult` gained a `QueryKind` discriminant (Select | Match). CLOSURE
  remains planned.
  **LANDED (2026-08-04, part 2)**: `CLOSURE` transitive closure (BFS fixpoint,
  first-visit order, depth scores) + per-step edge-property filters
  (`MATCH ... WHERE <edge-prop> = <value>`) in the same IR/parser/engine.
  `QueryKind::Closure` added. Graph story complete for M0/M1.
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
- D9. (new) Feedback / upvote-downvote as first-class signal:
  - **A vote is just an edge** — no new storage machinery:
    `(voter)->:voted {value:+1|-1, weight:-1..=1, created_at}->(record)`.
    Provenance, time, per-voter granularity, and one-transaction all come free.
    `weight` defaults to the `value` when omitted (so `value:-1` downvotes under
    `::score` too); an explicit `weight` overrides `value` for `::score` only.
  - Deterministic engine operators (allowed — pure arithmetic, No-LLM):
    `::votes(record)` → (up, down, net); `::score(record)` → **Laplace-smoothed
    mean** (M1 default: robust with few votes; Wilson lower bound deferred to
    ranking-API milestone); `::feedback(record)` → time-decayed recent feedback.
    Determinism discipline: fixed iteration order (BTree), tie-break by
    RecordId, no races in aggregation (snapshot isolation).
  - **Regression story**: feedback becomes ground-truth (query, retrieved,
    relevance) triples in the same store → recall@K/precision@K regression tests
    against real usage data in the M1 benchmark harness; bootstrap set for a
    future agent-side reranker. The *learning* (weight tuning α..δ, reranker
    training) lives in the agent layer, never in the engine.
  - Salience gets a fourth deterministic term:
    `::salience = α·similarity + β·strength(recency,freq) + γ·importance + δ·score`
    (weights α..δ are agent-side knobs, passed per-query as
    `ORDER BY ::salience(α, β, γ, δ)`; engine defaults 0.7/0/0/0.3 —
    LANDED 2026-10-06, issue #88).
  - **LANDED (2026-08-03)**: `Order::Votes` + `Order::Feedback` in nql-ir;
    `vote_counts` (up/down/net over `:voted` edges) and `feedback_score`
    (time-decayed: Σ sign·1/(1+λ·age), `now` = max created_at in store — pure,
    deterministic, no wall-clock) in the engine; parser accepts
    `ORDER BY ::votes | ::feedback`.
  - **LANDED (2026-08-04)**: retrieval regression harness (`nqlite::harness`):
    ground-truth relevance stored as `:relevant` edges in the store (same
    votes-as-edges model), `recall_at_k` / `precision_at_k` metrics, and
    `tests/regression.rs` asserting recall/precision floors over a
    deterministic synthetic corpus (kNN + BM25 paths). Catches retrieval
    regressions on grammar/index/fusion changes.
  - Poisoning/drift: votes carry voter + trust weight; engine stores, agent
    decides trust policy; deterministic decay for old votes.

## 6. Benchmark targets vs measured (reconciled 2026-10-06, issue #115)

Measured on the reference box (Intel Xeon E5-2640 v4 @ 2.40 GHz, release
profile, warm cache; tooling: `nqlite/examples/open_profile` + the E08
release ladder — full numbers and method on issue #115):

- **reopen cold a 100K-record store**: measured — **~0.26 s** engine load
  after #133's lazy history decode (warm-cache medians of 3; ~0.46 s first
  touch; ~0.54 s CLI end-to-end). The store file splits 31.5 MB core / 31.0 MB
  history tail: current-state queries decode only the core, and the first
  temporal read pays the deferred tail once per session. (Was ~0.7 s load /
  ~0.875 s CLI before #133.) The "milliseconds" wording stays **retired**,
  and the exact-scan floor itself moved: after the #144 L1→L3 stream
  (borrow candidates → memoized index → k-capped windowed search, behind
  the spec §2.3 output-cap invariant) a k=10 exact kNN @100K measures
  **35.6 ms idle-box mean (39.0 ms p99) / ~52 ms under load** (release
  build; was 75–140 ms).
  Sub-10 ms still needs the ANN path (#96's gate). (E08's earlier "1.8 s"
  figure included ~0.9 s of harness-side output parsing, not engine time.)
- **kNN recall@10 >= 0.95**: met at 5k rows (**0.96**, CI gate in #114);
  degrades at 50k (0.81) with default params — sweep flags exist; the gate
  pins the 5k target.
- **deterministic: same input -> byte-identical output across runs**: met —
  248 workspace tests incl. transcript-digest checks.
- **crash in WAL mid-commit -> auto-recover, no corruption**: met — CRC /
  torn-frame tests (spec file-format §4).
- **ingest 100K records < a few seconds**: met in-process (**3.74 s** line
  protocol, release); the chunked CLI file tier is slower (67.7 s — dominated
  by per-session reopens, issue #115/E08).

## 7. Tone for docs (repo conventions)

- README.md stays state-independent (description, features, quick start, security,
  license, contributing) — no milestone tables ("done" markers, status) in README.
- PLAN.md is the maintainer roadmap, milestone-driven.
- CHANGELOG.md is user-visible changes.
- All docs: community-standard, keep in sync with code as you already do.