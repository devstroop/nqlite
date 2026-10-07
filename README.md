# nqlite

**A context-first, deterministic, serverless database for AI agents — SQLite for
AI memory.**

nqlite is a single-file embedded database with SQLite-style ergonomics, built
for one job: durably hold an agent's evolving context — records, typed graph
relations, embedding vectors, and time — under a single ACID transaction and a
hard **zero-LLM** contract. Responsibilities are split by design: the engine
stores and recalls deterministically (hybrid kNN + BM25, `MATCH` traversal,
`AS OF` time travel); the agent above decides what to write, relate, embed, and
forget. All of it offline, in one file — a durable substrate for agent memory
that outlives any model.

*"Neural" here means embeddings are first-class data — nothing in the engine
learns. Full framing and terminology: [docs/positioning.md](docs/positioning.md).*

[![GitHub](https://img.shields.io/badge/github-devstroop%2Fnqlite-181717?logo=github)](https://github.com/devstroop/nqlite)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.82%2B-orange.svg)](https://www.rust-lang.org)

---

## Table of contents

- [Why nqlite](#why-nqlite)
- [Features](#features)
- [Guarantees](#guarantees)
- [Installation](#installation)
- [Quick start](#quick-start)
- [The nql language](#the-nql-language)
- [Architecture](#architecture)
- [Performance](#performance)
- [Documentation](#documentation)
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
- **Serverless**: `open a file` and start.

## Features

- **Records** — `table:id` identifiers, schemaless document bodies, and
  `VECTOR<f32, N>` embeddings as first-class typed fields.
- **Graph relations** — directed, named, typed edges with properties, weight,
  and provenance.
- **Deterministic retrieval & ranking** — cosine kNN (`vector::similarity`),
  BM25 (`::bm25`), and hybrid fusion (RRF) in one `WHERE`; ranking by
  `::salience` (4-term, agent-tunable via `::salience(α, β, γ, δ)`), `::score`
  (Laplace mean over `:voted` edges), `::votes`, `::feedback`, `::recency`,
  or any body field (`ORDER BY seq [DESC]` — documented total order, loud
  typo errors). Filters `!= < <= > >= IN [..] BETWEEN`, `COUNT(*)`, and
  `LIMIT/OFFSET` complete the slice — semantics in
  [spec/nql.md §2.3](spec/nql.md).
- **Graph traversal** — `MATCH` (1+ hops, both directions, per-step edge
  property filters) and `CLOSURE` (transitive closure, BFS-depth scores);
  both accept `AS OF <ts>` (historical snapshots) and `MATCH ... COUNT`
  (edge multiplicity).
- **Time travel** — `AS OF <ts>` on `SELECT`/`MATCH`/`CLOSURE`, backed by
  deterministic history replay; `HISTORY SINCE <ts>` returns the exact
  mutation delta (rows *and* edges, tombstones included) for sync; and
  `PRUNE HISTORY` compacts that history into a snapshot (bounded growth,
  cheap recent `AS OF`; earlier timestamps fail loudly instead of guessing).
- **Embedded & serverless** — one file, zero services; optional line-protocol
  server (TCP/stdio) and MCP server (`nql-mcp`).
- **Transactions** — one ACID transaction spans records + edges + vectors +
  their indexes (semantics under [Guarantees](#guarantees)).
- **Hardened** — fuzzed parser (cargo-fuzz) + property tests, crash-recovery
  (CRC / torn-frame) tests, deterministic benchmarks.

## Guarantees

### Zero-LLM guarantee

**The engine will never call an LLM** — not to embed, chunk, summarize, compact,
or rerank. Vectors are **BYO**: the agent (or any external provider) computes them
and pushes plain `f32` arrays. Any learning lives in the agent/client. This is a
hard design contract (see [docs/decisions.md §1](docs/decisions.md)).

### Deterministic execution

Identical `(plan, store)` ⇒ byte-identical results — no wall-clock, no
randomness, no hidden model ([spec/nql.md §2.1](spec/nql.md)). The companion
harness (`nqlite-experiments`) re-asserts this with transcript digests on every
run.

### Single-writer ACID

SQLite-style concurrency: one writer, snapshot readers per `execute`, sidecar
WAL with CRC torn-frame recovery ([spec/file-format.md §4](spec/file-format.md)).
One process owns the file (flock-guarded); readers never block writers.

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

**Requirements**: Rust 1.82+ (see `rust-version` in Cargo.toml). No system
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

Expose the database to AI agents over the Model Context Protocol (stdio):

```bash
cargo run -q -p nql-mcp                 # in-memory
cargo run -q -p nql-mcp -- --db memory.nql   # persistent
```

`nql-mcp` serves tools (`execute_nql`, `create_table`, `insert_record`,
`relate`, `select`, `match_path`, `forget`) with deterministic JSON results;
`select` supports temporal reads (`as_of`) and `MEMORY`-block reads
(`memory`), and `execute_nql` carries the full grammar (including
`AS OF` and `MEMORY` scoping).

Or speak the line protocol directly (`nql-server`, TCP or stdio). Line-protocol
rule: **each line is its own plan starting at the root store** — `MEMORY` must
prefix every statement it scopes (see [spec/nql.md §2.8](spec/nql.md)):

```bash
printf 'MEMORY core; CREATE TABLE note; MEMORY core; INSERT INTO note:1 { "text": "x" };\nMEMORY core; SELECT * FROM note;\n' \
  | cargo run -q -p nql-server -- --stdio
```

Add `--db memory.nql` (either mode) to serve a **persistent** store — same
semantics as `nql-cli --db`, including the single-writer lock; without it the
server is in-memory and everything is lost on exit:

```bash
cargo run -q -p nql-server -- --db memory.nql            # TCP on :7878
cargo run -q -p nql-server -- --db memory.nql --stdio    # line protocol on stdio
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

nql-server/  line-protocol server (TCP + stdio, optional `--db` persistence)
nql-mcp/     MCP server (stdio) — exposes nqlite as tools for AI agents
```

**Why three crates?** `nql` (front-end) and `nqlite` (engine) are separated by a
pure contract (`nql-ir`), so the language never bends to engine internals and
each half is hardened independently. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Performance

All numbers are engine-only and bound to a commit, profile, and machine —
methodology, raw runs, and an honest "where nqlite is weak" section live in
**[docs/benchmarks.md](docs/benchmarks.md)** (regenerate: `./scripts/bench.sh`;
micro-benchmarks: `cargo bench -p nqlite`).

Headline (reference box, release build): **100K-record reopen ≈ 0.26 s**
(core-only lazy load; ≈ 0.54 s CLI end-to-end), ingest 100K in **3.74 s**
in-process, exact-scan queries floor at 75–140 ms @100K — sub-10 ms needs the
feature-gated ANN path (recall@10 ≥ 0.95 CI gate at 5k rows). Cross-DB
quality matrix (sqlite-vec / LanceDB / Chroma): [scripts/bench-compare/](scripts/bench-compare/).

## Documentation

Full index: **[docs/README.md](docs/README.md)**.

| | |
|---|---|
| [docs/decisions.md](docs/decisions.md) | Design intent: non-negotiables, mental model, decisions D1–D9 |
| [docs/positioning.md](docs/positioning.md) | Pitch & framing: what nqlite is/isn't, honest comparisons, terminology |
| [spec/nql.md](spec/nql.md) · [spec/file-format.md](spec/file-format.md) | Normative: grammar & semantics · on-disk format |
| [docs/agent-patterns.md](docs/agent-patterns.md) | Runnable agent recipes (`cargo run -p nqlite --example …`) |
| [docs/benchmarks.md](docs/benchmarks.md) | Methodology, measured numbers, weaknesses |
| [docs/research.md](docs/research.md) · [docs/comparison.md](docs/comparison.md) | External sources & landscape · position vs sqlite-vec/LanceDB/Chroma/SurrealDB |

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