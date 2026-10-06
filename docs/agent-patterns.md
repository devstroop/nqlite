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
turn. Two deterministic routes for it to matter — (a) the original one,
converting it into a `:voted` edge weight
(`(agent) -[:voted {weight}]-> (turn)`) so it enters the Laplace `::score`,
or (b) the direct γ weight: `ORDER BY ::salience(0.7, 0, 0.3, 0)` blends
`0.7 · similarity + 0.3 · importance` with no edge at all (spec §2.3; bare
`ORDER BY ::salience` uses the engine defaults 0.7/0/0/0.3, i.e.
`0.7 · similarity + 0.3 · score`). Either way a slightly less similar but
much more important turn (importance 0.9) can outrank a
nearer-but-less-important one (0.8) — the recalled "context" reflects what
the *agent* judged valuable.

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

## 4. Re-ranking & feedback recipes (post-#93)

### Restrict the pool first, then score

`ORDER BY ::score` ranks *the scan* — there is no candidate-set concept
(pool-by-design). The rerank recipe is therefore **restrict first, score
second**:

```sql
-- the retriever's candidates are keyed by a body field:
SELECT * FROM doc
    WHERE topic IN ["rust", "wasm"]     -- server-side pool (#93's IN)
    ORDER BY ::score
    LIMIT 2;
```

`IN` makes the restriction server-side whenever the pool carries a body key
(the `WHERE topic = …` + client-intersect workaround from exp02 still works
and still scales to pools keyed by anything else). Retriever pools keyed by
**RecordId** — the common case, since kNN returns ids — have no id predicate
yet: intersect the id set client-side, or use `WHERE id IN [...]` once #128
lands (it turns this recipe into one query). Either way the rule is the
same: `::score` over an unrestricted table is `::score` over *every* row.

### Which feedback operator reads what

After #85 the three operators are sign-consistent but they do **not** read
the same fields — an explicit `weight` splits them by design:

| operator | reads | explicit `weight` | `value` |
|---|---|---|---|
| `::score` | edge `weight` (fallback: signed `value`, then `1.0`) | **overrides `value`** — magnitude *and* sign come from `weight` | used only when `weight` is absent |
| `::votes` | edge `value` (`+1` / `-1` counts) | **ignored** | the only input |
| `::feedback` | edge `value` sign × time decay | **ignored** | the only input |

Recipes:

- plain up/down: `SET value = 1` / `SET value = -1` — since #85 a bare
  `value = -1` already *lowers* `::score` (weight derives from value).
- confidence-weighted score without changing the vote count:
  `(agent) -[:voted {value: 1, weight: 0.3}]-> (doc:x)` — `::score` sees the
  0.3, `::votes` still counts exactly one upvote, `::feedback` still decays
  its `+1`.
- never set the two to opposite signs unless you mean it: `::score` follows
  `weight`, the other two follow `value` — they *will* disagree by design.

These tables are pinned by tests that run in CI: the full mode matrix is
`order_by_precedence_matrix_matches_spec`, the pool recipe is
`rerank_pool_recipe_scores_only_the_pool`, and the weight/value split is
`feedback_weight_and_value_disagree_by_design`.

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