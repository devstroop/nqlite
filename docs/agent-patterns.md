# Agent Patterns

Runnable examples (`nqlite/examples/`) showing how an **agent** builds
long-lived context — memory, retrieval, and a tool ledger — on top of the
nqlite engine. The engine stays a deterministic, zero-LLM function of
`(plan, store)`; **all learning happens in the agent's example code**, which
decides what to write, relate, and embed.

## How to run

From the workspace root (or from `nqlite/`):

```sh
cargo run -p nqlite --example chat_memory   # agent conversation memory
cargo run -p nqlite --example rag_loop      # retrieval-augmented generation loop
cargo run -p nqlite --example tool_ledger   # tool-call ledger + FORGET
```

Each example is a self-contained `main() -> Result<(), Box<dyn Error>>`,
fully deterministic (no `rand`, no wall-clock, `created_at = 0`), and prints
human-readable output.

---

## 1. Chat memory — `examples/chat_memory.rs`

Store a simulated conversation as `turn` records (BYO dim-3 embeddings) plus
`entity` records, link each turn to the entities it mentions with
`:mentions` edges, then recall with a kNN SELECT ordered by `::salience`.

```sql
CREATE TABLE turn VECTOR<f32, 3>;
INSERT INTO turn:1 { "role": "user", "text": "What is nql?", "importance": 0.8 }
    EMBED [1.0, 0.0, 0.0];
RELATE (turn:1) -> :mentions -> (entity:nql);

SELECT * FROM turn
    WHERE vector::similarity(embedding, [1.0, 0.0, 0.0]) AND k = 3
    ORDER BY ::salience;
```

The **importance knob**: the agent hand-writes an `importance` field on each
turn and converts it into a `:voted` edge weight
(`(agent) -[:voted {weight}]-> (turn)`). The engine's salience formula blends
`0.7 · similarity + 0.3 · importance`, so a slightly less similar but much
more important turn (importance 0.9) can outrank a nearer-but-less-important
one (0.8) — the recalled "context" reflects what the *agent* judged valuable.

## 2. RAG loop — `examples/rag_loop.rs`

Minimal retrieval-augmented generation: ingest a few short documents with
hardcoded embeddings, query with kNN and take the top-2, record which docs
answered via `(query) -[:retrieved]-> (doc)` edges, then simulate user votes
(`:voted {value:+1|-1}`) and re-rank the whole corpus with `ORDER BY
::feedback`.

```sql
SELECT * FROM doc
  WHERE vector::similarity(embedding, [1.0, 0.0, 0.1]) AND k = 2;   -- retrieve

RELATE (query:1) -> :retrieved -> (doc:1);                            -- remember
RELATE (user:1)  -> :voted -> (doc:1) SET value = 1;                  -- feedback

SELECT * FROM doc ORDER BY ::feedback;                                -- re-rank
```

The feedback pass is time-decayed and fully deterministic (the engine treats
the data's own max `created_at` as "now"), so identical input always yields
the same ranking. The agent owns the embeddings and the votes; the engine
only executes the fixed formula.

## 3. Tool ledger — `examples/tool_ledger.rs`

Record each tool invocation as a `call` record (tool name + args + result),
link it to a `tool` entity, and keep an aggregate `(agent) -[:called]->
(tool)` usage ledger. The agent then FORGETs an old entry and queries what
remained with a field filter.

```sql
INSERT INTO call:1 { "tool": "web_search", "args": "nqlite vector index", "result": "3 hits" };
RELATE (call:1) -> :used-> (tool:search);
RELATE (agent:assistant) -> :called -> (tool:search);

FORGET call:2;
SELECT * FROM call WHERE tool = "web_search";
```

`FORGET` cascades deterministically — the record *and* its incident edges
are removed in one step, so the audit trail stays consistent.

---

## Why the engine stays deterministic while the agent learns

The line between "agent" and "engine" is sharp by design:

- **The agent learns.** Embeddings, importance weights, votes, edges, what to
  `FORGET` — every bit of "knowledge" the agent accumulates is written into
  the store by the agent's own code. The agent is the only component that
  changes, re-embeds, or drops information.
- **The engine never learns.** It contains no embedding model, no gradient
  update, no randomness, no wall-clock, no network. Every ordering is a fixed
  arithmetic blend (kNN cosine, Laplace-smoothed `::score`, time-decayed
  `::feedback`, binational `::salience`) tie-broken by `RecordId`, applied to
  exactly the data the agent wrote.
- **Same input → same output.** Records live in a `BTreeMap`, edges are an
  append-only list, and all sorts are stable with RecordId tie-breaks, so any
  `(plan, store)` replays to byte-identical results.
- **Provable, testable, auditable.** Because the engine is a pure function,
  agent behavior — the only part that "changes" — can be snapshotted,
  diffed, and tested. This is the whole point of the agent-pattern examples:
  demonstrate the powerful agent-side context tricks while the engine remains
  a crisp, deterministic substrate (see `docs/decisions.md`, `nqlite/src/lib.rs`).