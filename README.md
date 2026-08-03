# nqlite

**A context-first, neural, serverless database.**

One embedded file, SQLite-style. Stores records, typed relations (graph),
embeddings (vectors), and temporal context under a single deterministic,
zero-LLM, ACID transaction. Built to be the durable memory/context substrate
for AI agents — while the engine itself never depends on an LLM.

## Why

Vector DBs bolt embeddings onto a store; "AI databases" put an LLM in the write
path (non-deterministic, untestable). nqlite is neither: the engine is a
deterministic truth machine. Agents — during a conversation — decide what to
write, relate, embed, and recall; the store faithfully holds their chained
context, offline, forever.

## Zero-AI guarantee

The engine will never call an LLM — to embed, chunk, summarize, compact, or
rerank. Any learning lives in the agent/client. This is the design contract.

## Status

Milestone 0 (deterministic context engine) in progress. See docs/KANBAN.md.

## Layout

```
nql/      front-end: nql language — parser + AST (storage-agnostic)
nql-ir/   shared IR / Plan contract between nql and nqlite
nqlite/   the engine: storage, indexes, execution (zero-LLM)
docs/     decisions, research, comparison, roadmap, kanban
spec/     nql grammar + file-format specs
```

## Docs

- [PLAN.md](PLAN.md) — milestone roadmap
- [docs/KANBAN.md](docs/KANBAN.md) — feature kanban
- [docs/decisions.md](docs/decisions.md) — design intent & locked decisions
- [docs/research.md](docs/research.md) — external research, sources
- [docs/comparison.md](docs/comparison.md) — vs sqlite-vec / LanceDB / Chroma / SurrealDB

## License

MIT OR Apache-2.0
