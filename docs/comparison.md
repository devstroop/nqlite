# nqlite — Position vs. adjacent systems

A living comparison. Update it as nqlite evolves (keep in sync: told in the same
"why build our own" decision). Directions: where each stores vector data, whether
it is a single file, whether the LLM/AI live in the engine, and what a single
transaction can span.

| | nqlite (planned) | sqlite-vec | LanceDB | Chroma | SurrealDB |
|---|---|---|---|---|---|
| Shape | **single-file embedded engine** | SQLite extension (vec0 virtual table) | embedded + remote columnar (Lance format) | embedded (HNSWlib) + client/server | embedded lib + server (KV-backed: RocksDB/TiKV/etc.) |
| Data model | documents + typed relations (graph) + embeddings + time | relational tables + vector virtual table | columns, vectors, full-text (BM25) | collections / embeddings | multi-model: doc + graph edges + vector + time-series |
| Graph/traversal | **first-class** (typed relations, MATCH/CLOSURE) | none (would need another ext) | no built-in graph | no | native graph + edges (RELATE/MATCH) |
| Vector index | exact brute-force M0, then HNSW behind a trait | brute-force by default (ANN optional) | IVF-PQ / bruta | HNSW (hnswlib) | vector field + index |
| Single transaction spans doc+graph+vector? | **yes** | no (virtual tables bolt-on) | vectors yes; graph not native | in collection only | **yes** |
| Zero-LLM / deterministic engine | **hard constraint, enforced** | yes (index only) | yes (store only) | yes (store only) | yes at engine, "AI-assisted" layer on top |
| Serverless single file | **yes (design goal)** | file is the .db (needs the ext-loaded binary) | Lance format dir / object store | server or emb. | not truly single-file; embedded or server |
| Deploy footprint | minimal, pure-Rust | needs SQLite loader | heavier (columnar) | medium | heavy (server, KV backend) |

## Why the niche is worth owning
- Heavy players (SurrealDB) push "context layer for agents" but are infra-forward
  (KV-backed, server + BSL).
- SQLite-world (sqlite-vec) is a **bolt-on**, not a unified transaction.
- Neither centers the **hard zero-AI guarantee** + **true single-file serverless**
  + grammar authored for *neural* (context) workload patterns.

## Wedges nqlite should lead with
1. `OPEN(path) -> deterministic, ACID, context-first store` in one small crate.
2. One transaction = documents + graph edges + vectors + timestamps.
3. nql grammar: `SELECT` + `RELATE` + `MATCH` + `::similarity` + `::knn` +
   temporal, from the start — no stitched two-DB story.
4. Zero-LLM, property-testable, byte-deterministic — the claim mem0/Letta can't
   make because their LLM is in the write path.