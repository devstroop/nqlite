# docs/ — knowledge base

Everything about nqlite lives here (README.md stays compact by design).

## Index

| File | What it is |
|---|---|
| [decisions.md](decisions.md) | Locked design intent: non-negotiables, mental model, decisions D1–D9 |
| [positioning.md](positioning.md) | How we talk about nqlite: pitch, what it is/isn't, honest comparisons, terminology |
| [research.md](research.md) | External research with sources; competitive landscape, engines, grammar, agent-memory |
| [comparison.md](comparison.md) | Position vs sqlite-vec, LanceDB, Chroma, SurrealDB |
| [KANBAN.md](KANBAN.md) | Archived kanban (superseded by ISSUES.md — kept for history) |
| [README.md](README.md) | This index |

## Process notes

- **README.md** (repo root) stays state-independent. Roadmap/status live here.
- **PLAN.md** (repo root) is the milestone roadmap.
- **KANBAN.md** is archived; the live issue tracker is **ISSUES.md** (repo
  root), maintained per merge and kept in sync with PLAN.md milestones.
- Docs stay in sync with code — an experiment that changes a decision updates
  the relevant doc in the same PR.