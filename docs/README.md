# docs/ — knowledge base

Everything about nqlite lives here (README.md stays compact by design).

## Index

| File | What it is |
|---|---|
| [decisions.md](decisions.md) | Locked design intent: non-negotiables, mental model, decisions D1–D9 |
| [research.md](research.md) | External research with sources; competitive landscape, engines, grammar, agent-memory |
| [comparison.md](comparison.md) | Position vs sqlite-vec, LanceDB, Chroma, SurrealDB |
| [KANBAN.md](KANBAN.md) | Local kanban: feature breakdown, status per module |
| [README.md](README.md) | This index |

## Process notes

- **README.md** (repo root) stays state-independent. Roadmap/status live here.
- **PLAN.md** (repo root) is the milestone roadmap.
- **KANBAN.md** is the local kanban maintained during development; syncs with
  PLAN.md milestones.
- Docs stay in sync with code — an experiment that changes a decision updates
  the relevant doc in the same PR.