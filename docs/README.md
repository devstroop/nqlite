# docs/

Design documentation for nqlite. Kept in sync with the code.

## Index

- `decisions.md` — locked design intent, non-negotiables, open decisions (D1..D7).
- `research.md` — (in progress) external research findings: embedded/vector
  engines, query-language design + hardening, agent-context memory patterns,
  each with concrete sources.
- `comparison.md` — (planned) position vs sqlite-vec, LanceDB, Chroma, SurrealDB.

## How this stays honest

Per project convention, docs reflect the current state of the code. If an
experiment diverges (e.g. we pick `USearch` over `hnswlib-rs`), update the
relevant doc the same PR that changes the code.