# Wave-10 Release Notes

Wave-10 ships two new agent-pattern examples plus the gated fleet process
that delivered them. Both stay deterministic and zero-LLM at the engine
level: the agent authors every record, edge, and embedding; the engine only
reads and resolves.

## What shipped — `nqlite/examples/`

1. **`idempotent_memory.rs` — Agent pattern #4, idempotent memory.**
   Re-asserting the same `memory:id` key is an upsert (records are keyed by
   RecordId), so replaying the session resync converges to the latest
   generation — no duplicates, no unbounded growth. Every mutation advances
   the logical clock, so `SELECT ... AS OF <ts>` time-travels to an earlier
   generation; the agent diffs "before" vs "now" with no snapshots.

2. **`context_chain.rs` — Agent pattern #5, context chain.**
   Context splits across `MEMORY core` (live set) and `MEMORY archival`
   (previous session), records linked by `:follows_from` edges. `CLOSURE`
   walks the chain (BFS fixpoint, first-visit order, depth-scored) and
   reconstructs the full reasoning path in one statement; a per-step
   `WHERE status = "final"` edge filter prunes it.

Run with `cargo run -p nqlite --example idempotent_memory` / `--example context_chain`.

## Delivery process (gated fleet)

- Two parallel worktrees (`feat/parallel-a`, `feat/parallel-b`) built the
  examples concurrently, one per worker, both based on `develop`.
- Every feature branch passed a QA + reviewer gate ("QA+gate PASS") before
  merge — no direct pushes to protected branches.
- PR #61 (`release/swarm-sync` → `develop`, merged 2026-08-07, 2 files,
  +305/−0) was the single merge carrying both examples into `develop`.
- CI on the PR head: `cargo fmt --check`, `cargo clippy --workspace
  --all-targets -- -D warnings`, `cargo test --workspace` — **183 tests
  passed, 0 failed** (plus GitGuardian security check, all green).

## Release sync

Release notes land on `release/wave-10`; the release-sync target is `main`
(default branch), reached from `develop` via the tagged release merge.