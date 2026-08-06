# nqlite

**A context-first, neural, serverless database for AI agents.**

nqlite is an embedded database — one file, SQLite-style ergonomics — that stores
records, typed graph relations, embedding vectors, and temporal context under a
single deterministic, **zero-LLM**, ACID transaction. It is built to be the
durable memory and context substrate for AI agents: the agent decides what to
write, relate, embed, and recall during a conversation; nqlite faithfully, safely,
and reproducibly holds the agent's chained context — offline, forever.

[![GitHub](https://img.shields.io/badge/github-devstroop%2Fnqlite-181717?logo=github)](https://github.com/devstroop/nqlite)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org)

---

## Table of contents

- [Why nqlite](#why-nqlite)
- [Features](#features)
- [Zero-LLM guarantee](#zero-llm-guarantee)
- [Installation](#installation)
- [Quick start](#quick-start)
- [The nql language](#the-nql-language)
- [Architecture](#architecture)
- [Performance](#performance)
- [Design & research](#design--research)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Why nqlite

Most "AI vector databases" try to be clever: they chunk, embed, summarize, and
extract inside the engine. That makes them non-deterministic, hard to test, and
coupled to a model. **nqlite flips the architecture.** The database is a
deterministic, durable truth the agent decides *into*; the learning happens
*above* it, in the agent. The store's job is to faithfully hold whatever context
the agent has built — records, embeddings, and the graph of relations between
them — in one transactional, queryable, offline file.

This gives you:

- **One transaction**: documents + graph edges + vectors + timestamps update
  atomically — no stitching a document DB, a graph DB, and a vector store
  together (the "frankenstack" problem).
- **Determinism you can test**: identical input produces byte-identical output.
  No hidden LLM in the write path to make results non-reproducible.
- **Agent native**: `MATCH` graph traversal, `::similarity` kNN, `::salience`,
  `::score`, and `::feedback` in a single query language — the operations an
  agent needs to chain context over a conversation.
- **Serverless**: `open a file` and start. Optional network server (line
  protocol, TCP or stdio) is built in; an MCP server is planned.

## Features

- **Records** — `table:id` identifiers, schemaless document bodies, and
  `VECTOR<f32, N>` embeddings as first-class typed fields.
- **Graph relations** — directed, named, typed edges with properties, weight,
  and provenance; one transaction across records + vectors + edges.
- **Deterministic retrieval** — lexical + vector + graph in one query:
  - `vector::similarity(...)` / `ORDER BY ::similarity` — cosine kNN
  - `ORDER BY ::salience` — `α·similarity + β·strength + γ·importance + δ·feedback`
  - `ORDER BY ::score` — Laplace-smoothed mean of `:voted` feedback edges
  - `ORDER BY ::votes` / `::feedback` — community/vote-driven ranking
  - `ORDER BY ::recency` — creation-time ordering
- **Embedded & serverless** — a single file; open a path, start recall.
  Network server mode (line protocol, TCP or stdio) is built in; MCP planned.
- **Hardened** — fuzzed parser (cargo-fuzz) + property tests, single-writer
  transaction snapshot semantics, deterministic benchmarks.

## Zero-LLM guarantee

**The engine will never call an LLM** — not to embed, chunk, summarize, compact,
or rerank. Vectors are **BYO**: the agent (or any external provider) computes them
and pushes plain `f32` arrays. Any learning lives in the agent/client. This is a
hard design contract (see [docs/decisions.md](docs/decisions.md)).

## Installation

Add the workspaces as a path/v1 dependency (crates published once stabilized):

```toml
[dependencies]
nql = "0.1"
nqlite = "0.1"
nql-ir = "0.1"
nql-cli = "0.1"   # optional: the REPL/script runner
```

Or build the CLI from source:

```bash
cargo build --release --package nql-cli
# binary: target/release/nql
```

**Requirements**: Rust 1.80+ (see `rust-version` in Cargo.toml). No system
dependencies; pure Rust.

## Quick start

The simplest way to try it is the REPL:

```bash
cargo run -q -p nql-cli
```

```text
nql 0.1.0 — type :help for help, :quit to exit
>> CREATE TABLE turn VECTOR<f32, 384>;
>> INSERT INTO turn:1 { "role": "user", "text": "I work on the ML team" };
>> SELECT * FROM turn;
SELECT turn (1)
  turn:1  score=0.0000  {role="user", text="I work on the ML team"}
```

Persist the session to a single file (sidecar WAL, ACID crash-safety):

```bash
cargo run -q -p nql-cli -- --db memory.nql        # REPL backed by memory.nql
cargo run -q -p nql-cli -- --db memory.nql --script session.nql   # script mode
# :flush inside the REPL checkpoints the WAL into the main file
```

Or in Rust, programmatically:

```rust
use nql::parse;
use nql_ir::Store;
use nqlite::Database;

fn main() {
    let mut db = Database::new(Store::default());
    let plan = parse(
        r#"
        CREATE TABLE turn VECTOR<f32, 2>;
        INSERT INTO turn:1 { "text": "hello world" } EMBED [1.0, 0.0];
        INSERT INTO turn:2 { "text": "goodbye world" } EMBED [0.9, 0.1];
        SELECT * FROM turn
            WHERE vector::similarity(embedding, [1.0, 0.0]) AND k = 1
            ORDER BY ::similarity;
        "#,
    )?;
    let results = db.execute(&plan)?;
    println!("nearest: {:?}", results[0].rows[0].record);
}
```

## The nql language

nql is a SQL-like grammar with SurrealDB-style records and graph operators,
written for neural/context workloads. Multiple statements run as one plan (one
transaction), separated by `;`:

```sql
CREATE TABLE entity;
CREATE TABLE turn VECTOR<f32, 384>;      -- declare a fixed embedding dimension

INSERT INTO entity:acme { "kind": "org", "name": "Acme Corp" };
INSERT INTO turn:3 { "role": "assistant", "text": "..." } EMBED [0.02, ...];

-- typed, named graph edge with weight + provenance
RELATE (turn:3) -> :mentions -> (entity:acme) SET weight = 0.9;

-- hybrid-retrieval SELECT
SELECT * FROM turn
    WHERE vector::similarity(embedding, [0.01, ...]) AND k = 5   -- semantic
ORDER BY ::salience                      -- or ::score / ::votes / ::feedback
LIMIT 3;

-- feedback / votes (decision D9): votes are just edges
RELATE (agent:main) -> :voted -> (turn:3) SET value = 1, weight = 0.9;
SELECT * FROM turn ORDER BY ::feedback LIMIT 5;

FORGET turn:1;                            -- deletes a record and its edges
```

The full grammar and semantics live in [spec/nql.md](spec/nql.md).

## Architecture

```
nql-cli/  (REPL + script runner)
   │  nql::parse(text)
   ▼
nql/      front-end: lexer + parser + analyzer (storage-agnostic)
   │       produces nql-ir::Plan
   ▼
nql-ir/   shared contract: value types + Statement/Select/Order/Plan
   │
   ▼
nqlite/   engine: deterministic execution over Store
   ├─ records (BTreeMap)  ──  relations (edges)  ──  vectors (VectorIndex)
   └─ ACID transaction (single-writer, snapshot readers)  [M1: file + WAL]
```

**Why three crates?** `nql` (front-end) and `nqlite` (engine) are separated by a
pure contract (`nql-ir`), so the language never bends to engine internals and
each half is hardened independently. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Performance

Benchmarks run under criterion (`cargo bench -p nqlite`) against deterministic,
fixed-seed synthetic data. Sample readings (dev build):

| Bench | Data | Throughput / latency |
|---|---|---|
| `ingest/10000` | 10k records, dim-8 vectors | ~4.4M inserts/second |
| `knn_bf/10000` | k=10, exact scan | millisecond-scale |
| `relate/10000` | 10k edges | ~2.6–3.9M edges/second |

These are **the engine only** (no LLM in the path) — the honest numbers.

## Design & research

- **[docs/decisions.md](docs/decisions.md)** — the non-negotiables, mental model,
  and every design decision D1–D9 (incl. votes-as-edges, Laplace `::score`,
  vector-strategy).
- **[docs/research.md](docs/research.md)** — external research + sources:
  agent-memory systems, embedded/vector engines, query-language design.
- **[docs/comparison.md](docs/comparison.md)** — how nqlite sits vs sqlite-vec,
  LanceDB, Chroma, and SurrealDB.

## Roadmap

Milestone plan is versioned in [PLAN.md](PLAN.md). Each feature is a tracked
issue in [ISSUES.md](ISSUES.md). (This README stays state-independent.)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) — branch model (`main → develop → feat/*`),
checklist (`fmt + clippy + test`), and the zero-LLM/ determinism rules. This is a
welcoming project; bug reports and PRs are appreciated.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for full
license text and the NOTICE.