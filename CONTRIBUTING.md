# Contributing

## Branch model

```
main  (releases)  ←  PR: release-sync
  ↑
develop (integration)  ←  PR: feature
  ↑
feat/<module>  (one branch per issue, ISSUES.md)
```

- `main` — released/stable state only. Protected: PR + CI required.
- `develop` — integration branch. Protected: PR + CI required.
- `feat/*` — one branch per issue (see ISSUES.md). Branch from `develop`,
  open a PR **into `develop`** when the feature is complete.
- Release: open a PR `develop → main` (release-sync), squash-merge, tag.

## Local worktrees

Parallel feature work happens in per-branch worktrees under
`/mnt/ext1/nqlite-worktrees/` (mirrors the model-adapter workflow):

```bash
git worktree add /mnt/ext1/nqlite-worktrees/<feature> feat/<feature>
```

## Checks (CI enforces on every PR)

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## Docs rule

README.md stays state-independent. Any code change that alters a decision
updates the matching doc in `docs/` in the SAME PR (docs/README.md index).

## Zero-LLM rule

No engine code may call an LLM/embedder/network. The engine is deterministic.
Feature-gated, client-side AI helpers live outside `nqlite/`.
