# nqlite

**A context-first, neural, serverless database.** A single embedded file — SQLite
style — that stores records, typed relations (graph), embeddings (vectors), and
temporal context under one deterministic, ACID transaction boundary. Purpose-built
to be the durable memory/context substrate for AI agents, while the engine itself
is 100% deterministic and offline — it never depends on an LLM.

> State-independent README. Roadmap lives in [PLAN.md](./PLAN.md); user-facing
> changes live in [CHANGELOG.md](./CHANGELOG.md).

## Why

Most "AI vector databases" try to be clever: they chunk, embed, summarize, and
extract inside the engine. That makes them non-deterministic, hard to test, and
coupled to a model. **nqlite flips that.** The database is the deterministic, durable
truth the agent decides into. Learning happens above, in the agent; the store just
faithfully, safely, and queryably holds the context agents build during conversations.

## Features (planned)

- **Records** — `table:id`, schemaless-ish documents + typed fields,
  `VECTOR<f32, N>` embeddings as first-class citizens.
- **Graph** — typed, named edges with properties; one transaction across
  records + vectors + edges.
- **Deterministic retrieval** — lexical (BM25) + vector (kNN) + graph traversal /
  hybrid, in one query.
- **Serverless** — open a path, start remembering. Optional server/MCP mode later.
- **Hardenable** — WAL + ACID, parser/storage fuzzing, deterministic property tests.

## Zero-AI guarantee

The engine will never call an LLM — not to embed, chunk, summarize, compact, or
rerank. Any learning happens in the client/agent. This is the design contract.

## Repository layout

```
nql/      front-end: parser + AST (knows nothing of storage)
nql-ir/   shared IR / plan contract
nqlite/   the engine: storage, indexes, execution
docs/     design reasoning, decisions, research
spec/     nql grammar + file-format specs
```

## Quick start

_(Not yet implemented. Milestone 0 is tracked in PLAN.md.)_

## License

MIT OR Apache-2.0