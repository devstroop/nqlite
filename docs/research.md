# nqlite — Research Findings

Collected 2026-08 (direct, primary sources). Updates this doc when a decision
changes (docs trend concurrency with code). Sources are dated/attributed inline.

> Method note: the initial parallel research delegate batch (`deleg_d3bf401b`)
> failed to return results (upstream API 500 on final synthesis). The material
> below was instead gathered directly via primary-source fetches and official docs.

---

## 1. Competitive landscape — "embedded / context / neural" databases

### 1.1 SurrealDB — the closest mental model, and now the closest competitor
SurrealDB brand/framing in 2026 has moved to literally: **"the context layer for
AI agents — documents, graphs, vectors, time-series, and memory. One transaction,
one query, one deployment."** (surrealdb.com, 2026-08). dbdb.io classifies it:
multi-model (document + explicit graph edges), Rust, embedded + shared-nothing,
backed by an embedded KV engine (RocksDB / FusionDB / FoundationDB / TiKV; a
custom, sorted "SurrealKV" engine is evolving). It does NOT ship its own storage —
it delegates to a KV store. BSL license.

Implication for nqlite (differentiation, stated honestly):
- Overlap is real on the *data model* (documents + edges + vectors + time).
- nqlite's edge is **true serverless single-file** (SQLite-style), a **hard
  zero-LLM / zero-AI guarantee in the engine**, a minimal footprint, and a clean
  grammar authored for neural (context) workloads. SurrealDB is heavy and
  infrastructure-forward; "embedded" there still pulls in a server-less library
  tied to a KV backend and does not center the zero-agent coupling constraint.

### 1.2 Express the "context layer" gap in the sqlite world
Production-agent-memory article (Agents' Codex, 2026-05) — SQLite + FTS5 gives
**sub-1ms queries for 4,300+ memories** on local NVMe; managed vector DBs run
~25-50 ms (p95 up to 120-500 ms). Typical "SQLite for agent memory" pattern =
**structured episodic/session store (SQL) run in parallel to semantic vector search
(sqlite-vec)**, stitched by application code. SQLite holds comfortably to
~100K entries with frequent updates; that's the documented practical batch
before needing TiDB-scale migration. Mem0 self-reports >90% token savings from
summary-first store-and-recall.

That stitched-two-systems story is exactly the friction nqlite removes at the
single-file, transactional, graph-chained level — one store instead of two, with
the neural linkage as the connective tissue.

### 1.3 sqlite-vec (asg017)
- Single-file C SQLite extension, **vec0 virtual tables** with shadow tables
  (like FTS5), insert via normal INSERT, query via
  `WHERE headline_embedding MATCH $q AND k = 20`. Deployless: pip/npm/cargo.
- **Default similarity is exact/brute-force** (scans every vector). **No ANN**
  index by default; writer notes most local/AI projects are "thousands of
  vectors," where exact is fine. HNSW use recommended for multi-M tens.
- Bundled GGUF local embedders (`sqlite-lembed`) run offline on-Pi.
- Proves the "graph, vector, and lexical in one writable surface" is demanded,
  but it is a **bolt-on virtual table**, not a unified engine (no typed relations,
  no graph traversal in the sense, no single transaction owning document+graph).

### 2. Vector indexes (Rust, embedded)
| Crate | Notes |
|---|---|
| `hnswlib-rs` | Pure-Rust **HNSW**; **external K->NodeId map via any VectorStore**; distances L1/L2/Cosine/Jaccard/Hamming/Levenshtein; incremental insert/delete. Great fit: we take "you supply vectors by NodeId". |
| `fast-hnsw` | Pure-Rust HNSW v1.0.x (2026-03, ~55k dl). Claims better recall@K at same QPS (or same recall at higher QPS) vs older hnswlib; actively maintained. |
| `USearch` | SIMD-optimized, multi-language, compact; popular for mid-scale embedded+kNN. |
| `instant-distance` | HNSW with external-resizeable vectors; small-image semantics. |
| `arxiv/HD` | (comparator) |

Decision signal: prefer **exact brute-force for M0 (correctness, determinism)**,
then adopt an HNSW crate behind a **storage-engine abstraction** so we can swap
`fast-hnsw`.

### 3. Agent-memory patterns (to mine for the nql grammar + schema)
- **mem0** (mem0ai): LLM-driven memory layer. **KEY**: its extraction AND
  consolidation are **LLM-aware** (LLM discovers the "important" facts). This is
  precisely the "AI lives in the agent, not the DB" posture you've chosen — the
  engine stays deterministic; mem0 shows *what the agent* does (extract- entities,
  then consolidate). Vector DB plugins (Qdrant default) again indicate the
  push — the app bolts an agent memory layer on top of a vector store.
- **Letta / MemGPT** memory blocks: **core memory** (structured `memory` +
  `human` blocks) + **archival memory** (embedding-backed, search passages) +
  **shared memory** + subagents. Encodes what an agent *expects* of a store:
  named memory regions, searchable archival entries, sharing across agents,
  blocks that can be attached/detached/replaced. (Concepts page 404; nav shows
  the full "Memory: blocks, shared, archival, context hierarchy" surface.)
- **LangChain/LangGraph memory**: layered ShortTermBuffer / semantic long-term,
  episodic + semantic + procedural taxonomy; summary-based compaction.

These map to a bare **nql** op-set: CREATE/UPDATE record, RELATE, recall
(semantic+lexical), a `MEMORY`/context region set, attach/share, and — for the
agent that chooses — a deterministic `COMPACT`/`FORGET`/decay primitive (the LLM
is *outside*).

---

## Synthesis — what this means for nqlite (final)

1. The niche is **real and being claimed from the heavy side** (SurrealDB) and
   the **bolt-on side** (sqlite-vec). The unclaimed corner = **true
   file-serverless, deterministic, zero-LLM, with document+graph+vector+time
   under ONE transaction and a grammar authored for neural context.**
2. **Storage**: we are deliberately NOT "backed by RocksDB/LevelDB + SQLite
   = two engines". nqlite ships **its own single-file store** (SQLite-style) so it stays a dependency-light single file + a sidecar
   WAL. This is the differentiator; it also means we own the file-format spec
   and crash-safety — a real, controllable engineering surface.
3. **Vector index**: exact brute-force for M0 (deterministic/correct), then swap
   behind a trait for `fast-hnsw`/`USearch`.
4. **Agent story**: model the schema so an agent genuinely "owns its context"
   — blocks/archival/share concepts become plain tables + relations + vectors,
   never LLM calls.
5. **Determinism is our moat-testable claim**: identical input == identical bytes,
   crash-safe, reproducible — something mem0/Letta/Lance/chroma can't claim
   because the LLM is in their write path.

## Appendix A — Salvaged from the failed delegation batch (traces only, 2026-08-03)

The parallel research batch (deleg_d3bf401b) hit HTTP 500 on final synthesis for
all three tasks; briefs were lost. Their raw tool-traces survived and confirm/
extend the primary research above. Notable extra sources + facts worth keeping:

1. **langmem (LangChain's memory library)** — conceptual guide is the cleanest
   modern statement of the agent-memory model:
   - Taxonomy: **semantic** (facts/knowledge), **episodic** (past experiences),
     **procedural** (system behavior/persona). Storage: collections (searched at
     runtime) vs profiles (strict-schema, looked up directly).
   - Every memory op = "accept conversation + current memory state, **prompt an
     LLM** to expand/consolidate, respond with updated state" — i.e. the LLM is
     IN the write path. Confirms our zero-LLM-in-engine moat.
   - **Recall design rule (usable verbatim in nqlite)**: "memory relevance is
     more than semantic similarity. Recall should combine similarity with
     'importance' of the memory, and the memory's 'strength' = f(how recently /
     frequently it was used)."
     => maps directly to a deterministic nqlite `::salience` operator:
     `score = α·similarity + β·strength(recency, frequency) + γ·importance`,
     where importance is a number the AGENT writes (engine never invents it).
   - Source: https://raw.githubusercontent.com/langchain-ai/langmem/main/docs/docs/concepts/conceptual_guide.md

2. **Zep — "beyond static knowledge graphs"** (blog.getzep.com): argues KG
   usefulness for agents requires temporal edges ("started_on / ended_on"),
   confidence, and provenance — all first-class edge properties. Supports nqlite
   putting **time + weight/provenance on relation edges** (not just on records).
   Source: https://blog.getzep.com/beyond-static-knowledge-graphs/

3. **Chroma storage layout** (cookbook.chromadb.dev/core/storage-layout):
   Chroma persists via SQLite (metadata/WAL) + segment dirs (HNSW index files)
   — i.e. the two-engine bolt-on pattern, confirming the "one transaction"
   differentiation. Source: https://cookbook.chromadb.dev/core/storage-layout/

4. **Qdrant indexing docs** (qdrant.tech/documentation/manage-data/indexing):
   HNSW params m / ef_construct / ef_search; payload index + full-text index for
   filtered search — concrete knobs to expose in our `VectorIndex` trait later.

5. **SurrealDB vector index + local engine docs**
   (surrealdb.com/docs/learn/data-models/vector-search/vector-indexes,
   docs.rs/surrealdb/.../engine/local/struct.RocksDb.html): vector index via
   HNSW under the hood; local engine = RocksDB-backed. Confirms our own-file
   choice as the differentiator vs their KV-backend approach.

6. **Parser engineering sources** (for nql front-end):
   - sqlparser-rs (apache/datafusion-sqlparser-rs) — the de-facto Rust SQL
     parser; reference for dialect/spec structure.
   - winnow (github.com/winnow-rs/winnow) — maintained nom successor; better
     errors, streaming; candidate for nql parser base.
   - cargo-fuzz structure-aware fuzzing (rust-fuzz.github.io/book/cargo-fuzz/
     structure-aware-fuzzing.html) — for grammar fuzzing with structured inputs.
   - proptest (docs.rs/proptest) — property-testing for deterministic invariants.
   - SQLite `WITH` (sqlite.org/lang_with.html) + Cypher variable-length paths
     (neo4j.com/docs/cypher-manual/current/patterns/variable-length-paths/) —
     reference semantics for nql recursive `CLOSURE`/`MATCH *1..n`.

7. **pgvector benchmark** (markaicode.com/benchmarks/postgresql-pgvector-benchmark)
   and Qdrant system-design (markaicode.com/architecture/...) — useful baseline
   numbers for the benchmark harness in M1.

8. Other crates surfaced: `iqdb-ivf` (IVF in Rust), `hamming`, `numkong`,
   `stringzilla` (SIMD string libs — possible FTS accelerator later),
   `mini-qdrant` (educational HNSW/Qdrant clone), duckdb-rs (columnar reference).

## Sources
- SurrealDB dbdb.io entry (2026-08) — https://dbdb.io/db/surrealdb
- SurrealDB homepage / docs (2026-08) — https://surrealdb.com
- Agents' Codex — "Production Agent Memory: SQLite Hybrid" (2026-05) — https://agentscodex.com/posts/2026-05-08-production-agent-memory-sqlite-hybrid-long-context/
- Alex Garcia — "Introducing sqlite-vec v0.1.0" (2024-08) — https://alexgarcia.xyz/blog/2024/sqlite-vec-stable-release/index.html
- mem0 docs (vector DBs) — https://docs.mem0.ai/components/vectordbs/overview
- Letta docs (memory concepts) — https://docs.letta.com/concepts/memory
- crates.io — fast-hnsw metadata — https://crates.io/api/v1/crates/fast-hnsw
- GitHub — hnswlib-rs — https://github.com/jean-pierreBoth/hnswlib-rs

_(raw primary fetches were done via curl in mid-2026; re-verify on decision changes)