# Integrations — where nqlite sits in the stack

*Discussion document (2026-10-06). Companion to
[oio ADR-006](https://github.com/devstroop/oio/blob/main/docs/ADR/006-optional-nqlite-adapters.md).
Settles the integration directions **before** code exists, so each repo's
first adapter lands against an agreed boundary instead of an ad-hoc one.*

## 1. The stack, in one picture

Two systems in this workspace answer different questions, and the layer
between them is currently unnamed:

```
   nqlite            persistent truth        records, edges, vectors, time
      │                    │                 one txn, byte-deterministic, offline
      │  deterministic recall                 (hybrid kNN+BM25, MATCH/CLOSURE,
      ▼                    │                  AS OF, ::score/::feedback — E01–E10)
   Context<T>             relevant truth     typed, budgeted, provenance-carrying
      │                                        assembly — proposed N-project crate
      │  state + questions
      ▼
   oio                   inference           choice/score/noul, calibrated
      │                                        confidence, pinned checkpoint
      │  answers + confidence + routing
      ▼
   nqlite                the decision        rows + :voted + provenance →
                                               future recall improves
```

| Layer | Answers | Owner |
|---|---|---|
| **database** (nqlite) | *what is persisted/true* | this repo |
| **context** (`Context<T>`, proposed) | *what is relevant to this situation* | a third crate — neither engine |
| **decision** (oio) | *what follows from it* | [devstroop/oio](https://github.com/devstroop/oio) |

Naming: nqlite stays a **context-first database** — "database" carries the
wedge (ACID, single file, SQLite analogy); "context" is the adjective from the
data model. The assembled projection is a *type* (`Context<T>`), not a
product name. See §6.

## 2. Direction 1 — nqlite embeds oio: **never inside the engine**

oio is a learned-weight inference engine (ONNX/candle checkpoints, pinned
revisions, tokenizer, CUDA opt-ins). nqlite's non-negotiables
(`docs/decisions.md` §1) rule it out of the engine outright:

- **No-LLM/no-model is the product.** A checkpoint in the write/read path
  converts every claim into "deterministic *given revision X*" — the exact
  dilution `docs/positioning.md` exists to prevent.
- **Footprint.** nqlite is `postcard + crc32fast + serde + thiserror` (+ unix
  `libc`), pure Rust, MSRV 1.82, one file. oio needs ORT/tokenizers/
  checkpoints — a different class of dependency.
- **Unnecessary.** Everything oio could contribute (judge candidates, pick
  among options) belongs *above* the store per D9: learning lives in the
  agent.

**Permitted form:** a client-side helper crate (same standing as the planned
`nqlite-providers`) that calls an oio endpoint and writes results back as
rows/edges/votes. The engine never calls out — arrows only point *into*
nqlite.

## 3. Direction 2 — oio embeds nqlite: optional adapters, never core

Full decision record:
[oio ADR-006](https://github.com/devstroop/oio/blob/main/docs/ADR/006-optional-nqlite-adapters.md).
Summary of the two viable seams, both already present in oio's code:

| Seam | oio code | Adapter shape | What nqlite adds |
|---|---|---|---|
| **A — shortlist** | `Embedder` trait (`crates/oio/src/shortlist.rs`); `rank()` is already trait-parameterized | `NqliteEmbedder`: options as rows, kNN via `vector::similarity` | persistence + **votes**: shortlist becomes a *learned* pre-filter (`::score` over `:voted` on options) |
| **B — decision ledger** | `Predictor` trait (`crates/oio-serve/src/lib.rs`, explicitly stub-injectable) | decorator: write `(decision) <-[:decided]- (state)` after each predict | `AS OF` forensics, confidence calibration over time (`::score` per question type), regression triples over past decisions |
| C — long-doc memory | `predict_long` windowing | **explicitly out of scope** (PRD §4 discipline) | — |

Constraints every adapter must honor (each proven somewhere):

1. **One `Database` opener** — the single-writer lock rejects a second opener
   even in-process (issue #84 / PR #102). One `Arc`-shared handle per process.
2. **Off the hot path** — oio's semaphore-bounded predict must never wait on
   a WAL append; ledger writes happen after the response, ordering from the
   logical clock, not wall time.
3. **Retention or die** — an unbounded per-request ledger is issue #95's
   growth curve with extra steps (`PRUNE HISTORY`, shipped as #124, now
   bounds the active file); name a prune/snapshot policy in the ADR.
4. **MSRV direction is safe** — oio requires 1.88 ≥ nqlite's 1.82; the
   reverse direction would break nqlite's floor (another reason §2 stands).

## 4. Direction 3 — third projects using one or both

### 4.1 nqlite only

The E01–E10 harness in the workspace-sibling `nqlite-experiments` repo (not
published on GitHub) is effectively a
catalogue of these shapes: agent session memory (E01/E06), tool/spend ledgers
with reversible audit (E04/E07), provenance-carrying RAG (E02/E09), context
chain forensics (E03), persistent idempotent memory (E05), and the
decision-audit ledger validation for seam B — **E10**: forensics sweep,
precedent recall, outcome→reliability, rotation equivalence. Plus: eval
corpora-as-databases — store `(state, retrieved, judgment)` triples in the
store under test and gate CI on recall@K (`nqlite::harness` pattern).

**Integration rule:** transports first (`nql-mcp`, line protocol, CLI), link
crates only inside one Rust binary. The protocol is the stable API; the IR is
pre-1.0.

### 4.2 oio only

Typed decision endpoints (policy/retention automation), batch classification
(`/batch`, `predict_batch`, per-PRD), air-gapped/on-device serving (CPU
baseline, no network at inference). No memory claims — that's §4.3's job.

### 4.3 both — the loop

**nqlite remembers → `Context<T>` assembles → oio decides → the decision is
remembered.** Each direction covers the other's blind spot:

| nqlite lacks | oio provides |
|---|---|
| ranking ≠ judging ("which doc *answers* this?") | `choice` over candidates + calibrated confidence |
| votes need producers | model outputs as votes (confidence ≥ τ → `:voted`; < τ → review row) |

| oio lacks | nqlite provides |
|---|---|
| stateless requests | precedent: recall over logged states/decisions |
| per-call confidence, no track record | per-question-type reliability over time |
| routing by script/stopwords only | `AS OF` forensics on routing decisions |

Concrete path, cheapest first:

1. **N1 — MCP-composed loop** (no shared code): `nql-mcp` + `oio --mcp`:
   recall → pack context (E09 policy) → predict → write back. The E09 harness
   extended with an `oio_predict` step is the natural home.
2. **N2 — `Context<T>` crate**: schema + provenance per field, budget packing
   (E09's greedy generalized), rendering to oio `state`/`questions`. Pure,
   No-LLM, property-tested — owned by neither engine.
3. **N3 — decision-quality-vs-context study**: does E09's 0.25→0.97 recall
   swing move *decisions*? Requires licensed labels + held-out eval per oio
   PRD §8 — no quality claims from smoke data.
4. **N4 — forensics view**: `AS OF` state + logged prompt/answer/revision for
   one turn, rendered together (needs B + N1).

Existing evidence this builds on: oio's own
[nqlite hybrid spike](https://github.com/devstroop/oio/blob/main/docs/GITA-DEMO.md)
(holdout citation recall 1.000 vs oio BM25 0.938, abstention gap documented)
and `scripts/gita_nqlite_spike.py` / `gita_nqlite_demo.py` in oio.

## 5. Integration rules (both projects)

1. **Transports over linkage** — MCP/HTTP between processes; shared crates
   only in one binary, one file opener, one lock (issue #84).
2. **The model never touches the write path** — oio outputs become rows and
   edges, never direct mutations of recalled content; confidence thresholds
   gate auto-votes; low confidence routes to review, not silence.
3. **Pin both halves of reproducibility** — checkpoint revision + digest (oio
   N4) alongside nqlite determinism digests; a decision is reproducible iff
   *context bytes* and *checkpoint revision* are both recorded.
4. **Budgets compose** — pack exact on the nqlite side (E09), clamp
   authoritatively on the oio side (`OIO_MAX_TOKEN_BUDGET` / head budgets).
5. **Retention before scale** — any ledger/vote store names its policy first
   (issue #95).

## 6. Naming

- **nqlite: database.** "Contextbase" would discard SQLite-compatibility
   inheritance for zero technical gain and collide with context-engineering
   buzz. The honest form is already the pitch: *a context-first database* —
   persistent truth whose data model is context-shaped.
- **`Context<T>`: a type, lowercase** — the N2 crate. If anything deserves
   the "context base" word, it's this assembly layer, and even there it
   should be a library name, not a product.
- **oio: decision engine** — inference over relevant truth.

One line: **nqlite remembers, `Context<T>` assembles, oio decides — and the
decision is remembered too.**

## 7. Open questions (deliberately unresolved)

- Who owns `Context<T>` — its own repo, nqlite-experiments, or oio? (It must
  stay out of both engines.)
- Auto-vote trust policy: weight for machine voters, decay, and the
  human-review threshold — decisions D9's poisoning note applied to model
  voters.
- Ledger retention shape for adapter B: prune-by-age vs snapshot windows
  (interacts with issue #95's design).
- N3 eval corpus: where licensed labels live, and which metrics (agreement?
  ECE? Brier?) gate a "better context → better decisions" claim.

## 8. Anti-patterns (so nobody rediscovers them)

- **oio as a retriever** — it scores one state against fixed options; running
  it over a corpus is a latency disaster next to the exact-scan ladder in
  `nqlite-experiments/reports/exp08_scale_ladder.md`.
- **nqlite as a precedent cache keyed only on similarity** — precedent lookup
  must be exact-hash first, similarity second with a threshold; confident
  retrieval of a *wrong* precedent is worse than no memory.
- **Ungoverned auto-votes** — machine feedback without weight/decay/audit
  amplifies model bias (D9 poisoning/drift).
- **One file, two writers** — the lock rejects it by design (issue #84);
  separate stores and merge explicitly, or don't.

## 9. Status

- nqlite side: **10 experiments / 53 variants, all deterministic** (incl. E10,
  the ledger validation of seam B above); transports shipped (`nql-server
  --db` PR #111; MCP `select.as_of` #90 via PR #113); history retention
  closed (#95 → `PRUNE HISTORY` PR #124) and recall bench closed (#96 →
  `nql-bench --recall` PR #114, CI gate recall@10 ≥ 0.95 @ 5k).
- oio side: ADR-006 proposed; Gita spike/demo already merged (PR #24).
- Neither repo has code coupling at the time of writing — this doc is the
  agreement the first adapter PR must cite.
