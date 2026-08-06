# Positioning

*How we talk about nqlite — and how we don't.*

## The one-line pitch

> **SQLite for AI memory**: a deterministic, serverless, single-file database
> that stores what an agent knows — records, graph relations, embeddings, and
> time — under one ACID transaction, with a query language written for the
> operations agents actually chain.

## What nqlite is

- **An embedded database** — one file, open a path, start recall. No service,
  no network required (the TCP/stdio line-protocol server and the MCP server
  are optional extras).
- **A context store** — the durable substrate an agent writes *into* and reads
  *from* across a conversation: turns, entities, notes, tool calls, votes,
  provenance edges.
- **A deterministic engine** — identical input ⇒ byte-identical output, always,
  everywhere. This is the property everything else hangs off: reproducible
  recall, property-testable behavior, honest benchmarks, offline operation.
- **Zero-LLM by contract** — the engine never embeds, chunks, summarizes,
  compacts, or reranks. Vectors are BYO. Learning happens *above* the database,
  in the agent, where it belongs.

## What nqlite is NOT

- **Not a vector database with a graph bolted on.** Records, edges, vectors,
  and time live in ONE store and update in ONE transaction — the anti-frankenstack
  position. The vector index is a feature of the store, not the product.
- **Not a "neural" database in the marketing sense.** "Neural" here means
  *embeddings are first-class data*, not that the engine is intelligent.
  Nothing in the engine learns, adapts, or improvises. If you need semantic
  behavior, an agent (or your embedding/reranking service) supplies it and
  stores the results here.
- **Not an agent framework.** nqlite does not decide what to remember, what to
  forget, or what to recall — that is orchestration, and it lives in the agent
  layer on top (see [decisions.md §4 — the determinism/neural
  spectrum](decisions.md)).

## The framing, in one table

| Question | Answer |
|---|---|
| What is it? | The memory/context layer for AI agents |
| What's the closest familiar thing? | SQLite — but for agent context, not relational rows |
| What's the differentiator? | Determinism + zero-LLM + one-transaction records/graph/vectors/time |
| Who computes embeddings? | You (BYO) — the engine never encodes |
| Who decides what matters? | The agent, above the DB |
| What's in scope for v1? | Storage, retrieval, traversal, ranking — all deterministic |

## Why determinism beats "smart" for memory

Non-deterministic memory is untestable memory. If a recall query can return
different answers across runs — because an LLM is in the write path, or the
index is approximate, or ordering depends on wall-clock — then:

- you cannot write a regression test that means anything;
- you cannot reproduce a failure from a bug report;
- you cannot benchmark honestly (the number drifts run to run);
- you cannot guarantee the same context to the same agent twice.

nqlite's contract is the opposite: **the query language, the engine, and the
storage format are all deterministic**. Approximate indexes (HNSW) and
non-deterministic components are either feature-gated or excluded by design —
see [spec/nql.md §2.1](../spec/nql.md).

## How we compare (honestly)

| | nqlite | sqlite-vec / LanceDB / Chroma | Neo4j | SurrealDB |
|---|---|---|---|---|
| Model | single-file embedded DB | vector stores (some embedded) | graph DB (server) | multi-model DB (server) |
| Records + vectors + edges in one txn | ✅ one transaction | ❌ vectors only (or frankenstack) | ❌ vectors awkward | ✅ but heavy |
| Deterministic engine | ✅ contract | ✅ vector math | ✅ | ✅ |
| Zero-LLM | ✅ contract | ✅ | ✅ | ✅ |
| Query language | nql — agent-native (MATCH, ::salience, ::score, ::feedback, hybrid bm25+kNN) | SQL / REST | Cypher | SurrealQL |
| Serverless single file | ✅ | sqlite-vec ✅, others ❌ | ❌ | ❌ |
| MCP server | ✅ (nql-mcp) | community | community | community |

The honest comparison: vector stores are good at *one* operation (kNN) and
graph DBs are good at *one* operation (traversal). nqlite is not the fastest
at either — it is the only embedded store that does **both plus lexical and
feedback ranking under one deterministic transaction**, which is the shape of
real agent memory. The cross-DB harness (`scripts/bench-compare/`) exists so
these claims stay measurable, not vibes.

## Terminology

- **"Neural"** (used sparingly): embeddings are first-class, queryable data.
- **"Context-first"** (preferred): the schema and query language are shaped
  around conversation context — turns, entities, mentions, votes, time.
- **"Zero-LLM"**: a hard contract, not a roadmap item. See README's
  [Zero-LLM guarantee](../README.md#zero-llm-guarantee).
- **"Deterministic"**: byte-identical output for identical input, no
  wall-clock, no randomness, no hidden model — see
  [spec/nql.md §2.1](../spec/nql.md).
