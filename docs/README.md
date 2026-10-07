# docs/ — knowledge base

Evergreen documentation for nqlite. The root [README](../README.md) stays
state-independent by design; the roadmap lives in [PLAN.md](../PLAN.md) and the
tracked issue log in [ISSUES.md](../ISSUES.md).

## Design & positioning

| File | What it is |
|---|---|
| [decisions.md](decisions.md) | Locked design intent: non-negotiables, mental model, decisions D1–D9, benchmark targets |
| [positioning.md](positioning.md) | How we talk about nqlite: pitch, what it is/isn't, honest comparisons, terminology |
| [research.md](research.md) | External research with sources: competitive landscape, engines, grammar, agent-memory |
| [comparison.md](comparison.md) | Position vs sqlite-vec, LanceDB, Chroma, SurrealDB |

## Guides & reference

| File | What it is |
|---|---|
| [agent-patterns.md](agent-patterns.md) | Runnable agent recipes (`cargo run -p nqlite --example …`): memory, chains, ledgers |
| [benchmarks.md](benchmarks.md) | Benchmark methodology, measured numbers (bound to commit/profile/machine), honest weaknesses |
| [`../spec/nql.md`](../spec/nql.md) · [`../spec/file-format.md`](../spec/file-format.md) | Normative specs (live in `/spec`): grammar & semantics · on-disk format |

## Project & archive

| File | What it is |
|---|---|
| [archive/KANBAN.md](archive/KANBAN.md) | Archived kanban — superseded by ISSUES.md, history only |
| [archive/release-wave-10.md](archive/release-wave-10.md) | Dated release note (wave-10, 2026-08); current changes belong in CHANGELOG.md |
| [README.md](README.md) | This index |

## Process notes

- **README.md** (repo root) stays state-independent; roadmap/status live in
  PLAN.md and ISSUES.md.
- **PLAN.md** is the milestone roadmap; **ISSUES.md** is the live issue
  tracker (maintained per merge, kept in sync with PLAN.md milestones).
- **Dated artifacts** (release notes, superseded boards) live in `archive/`.
- **Docs move with code** — any code change that alters a decision updates the
  matching doc in the SAME PR.
  the relevant doc in the same PR.