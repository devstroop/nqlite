#!/usr/bin/env bash
#
# Reproducible nqlite benchmark runner (issue #69).
#
# Runs nql-bench at the two headline sizes (1000 rows / 50 kNN queries and
# 10000 rows / 10 kNN queries), prints the JSON reports, and — if the optional
# cross-DB Python drivers are importable — also runs the bench.py matrix
# (scripts/bench-compare/bench.py). Missing drivers are skipped honestly.
#
# Everything lands in results/bench-<date>.json (one combined JSON report)
# plus results/bench-<date>.log (the raw human-readable transcript).
#
# Usage:
#   ./scripts/bench.sh
#
set -euo pipefail

# rustup-installed cargo lives here; make sure the toolchain is findable.
export PATH="$HOME/.cargo/bin:$PATH"

# Resolve the repo root no matter where the script is invoked from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_DATE="$(date +%F)"
mkdir -p results
LOG="results/bench-${RUN_DATE}.log"
OUT="results/bench-${RUN_DATE}.json"
XDB="results/.bench-xdb-${RUN_DATE}.json"

log() { printf '%s\n' "$*" | tee -a "$LOG"; }

log "==> nql-bench $(date '+%Y-%m-%d %H:%M:%S') | host $(uname -n) | $(nproc) cores | $(uname -m)"
log "==> corpus: xorshift64* seed 42, dim-8 vectors, 8-word vocab (dev build)"

# 1) nql-bench, headline sizes. Each run prints exactly one JSON report line.
log ""
log "==> nql-bench: 1000 rows, 50 kNN queries"
cargo run -q -p nql-bench -- --rows 1000 --knn 50 2>&1 | tee -a "$LOG"

log ""
log "==> nql-bench: 10000 rows, 10 kNN queries"
cargo run -q -p nql-bench -- --rows 10000 --knn 10 2>&1 | tee -a "$LOG"

# 2) Optional cross-DB matrix — only when all three python drivers exist.
if python3 -c "import sqlite_vec, lancedb, chromadb" >/dev/null 2>&1; then
  log ""
  log "==> cross-DB matrix (bench.py): 1000 rows, 50 kNN queries"
  python3 scripts/bench-compare/bench.py --rows 1000 --knn 50 --json \
    > "${XDB}" 2>>"$LOG" || true
  if python3 -c 'import sys,json;json.load(open(sys.argv[1]))' "$XDB" 2>/dev/null; then
    cat "$XDB" >>"$LOG"
  else
    log "    (bench.py produced no JSON — see log above)"
  fi
else
  log ""
  log "==> cross-DB matrix skipped: sqlite-vec/lancedb/chromadb not importable"
  XDB=""
fi

# 3) Assemble one JSON report for the date.
python3 - "$OUT" "$LOG" "$XDB" <<'PY'
import json, os, platform, sys
from datetime import datetime, timezone

out, log, xdb = sys.argv[1], sys.argv[2], (sys.argv[3] or "")

reports = []
for line in open(log):
    line = line.strip()
    if line.startswith("{"):
        try:
            reports.append(json.loads(line))
        except ValueError:
            pass

cross_db = []
if xdb:
    try:
        cross_db = json.load(open(xdb))
    except (OSError, ValueError):
        cross_db = []

report = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "corpus": "xorshift64* seed 42, dim-8 float vectors, 8-word vocabulary",
    "build": "dev (cargo run without --release); see docs/benchmarks.md",
    "host": {
        "uname": platform.uname().system + " " + platform.uname().release,
        "machine": platform.uname().machine,
        "nproc": os.cpu_count() or None,
    },
    "nqlite_runs": [r for r in reports if isinstance(r, dict) and r.get("db") == "nqlite"],
    "cross_db_ran": xdb != "",
    "cross_db": cross_db or None,
}
with open(out, "w") as f:
    json.dump(report, f, indent=2)
if xdb:
    os.remove(xdb)
print(f"report written to {out}")
PY

log ""
log "Done. Full log: $LOG"