#!/usr/bin/env python3
"""Percentile + throughput report over criterion samples (issue #170).

Criterion's console output leads with mean/median; tail latency (p95/p99/max)
and per-benchmark ops/s live only in its JSON artifacts. This script is a
pure function of those artifacts — no re-running, no new measurement:

    cargo bench -p nqlite [--bench bench|spike]
    python3 scripts/bench-percentiles.py            # text table
    python3 scripts/bench-percentiles.py --markdown  # paste-ready for docs/

Reads, per benchmark, `target/criterion/<id>/new/sample.json`
(`times[i]` ns wall for `iters[i]` iterations — both Linear and Flat modes)
plus `target/criterion/<id>/new/benchmark.json` (`throughput.Elements`
when the bench sets it). Per-operation samples are `times[i]/iters[i]`;
percentiles use linear interpolation (numpy 'linear' default); ops/s is
`elements / mean`.

Deterministic: same JSON in, byte-identical table out (rows sorted by id).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CRITERION = ROOT / "target" / "criterion"


def percentile(sorted_xs: list[float], p: float) -> float:
    """Linear-interpolation percentile, p in [0, 1] (numpy 'linear')."""
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    rank = (len(sorted_xs) - 1) * p
    lo = int(rank)
    frac = rank - lo
    if lo + 1 >= len(sorted_xs):
        return sorted_xs[-1]
    return sorted_xs[lo] * (1.0 - frac) + sorted_xs[lo + 1] * frac


def fmt_lat(ns: float) -> str:
    """Scale nanoseconds to a readable unit (matches criterion's style)."""
    for unit, factor in (("s", 1e9), ("ms", 1e6), ("µs", 1e3), ("ns", 1.0)):
        if ns >= factor or unit == "ns":
            return f"{ns / factor:.4g} {unit}"
    return f"{ns:.4g} ns"  # unreachable


def fmt_ops(elements: float, mean_ns: float) -> str:
    ops = elements / (mean_ns / 1e9)
    if ops >= 1e6:
        return f"{ops / 1e6:.4g} M/s"
    if ops >= 1e3:
        return f"{ops / 1e3:.4g} K/s"
    return f"{ops:.4g} /s"


def collect(criterion_dir: Path) -> list[dict]:
    rows = []
    for sample in sorted(criterion_dir.glob("**/new/sample.json")):
        bench_dir = sample.parent.parent
        try:
            s = json.loads(sample.read_text())
            times, iters = s["times"], s["iters"]
        except (OSError, ValueError, KeyError):
            continue
        ops = [t / i / 1e9 for t, i in zip(times, iters) if i > 0]
        if not ops:
            continue
        ops.sort()
        mean = sum(ops) / len(ops)
        bmark = sample.parent / "benchmark.json"
        full_id: str | None = None
        elements = None
        try:
            b = json.loads(bmark.read_text())
            full_id = b.get("full_id")
            elements = (b.get("throughput") or {}).get("Elements")
        except (OSError, ValueError):
            pass
        row: dict = {
            "id": full_id or bench_dir.relative_to(criterion_dir).as_posix(),
            "n": len(ops),
            "mean": mean * 1e9,
            "p50": percentile(ops, 0.50) * 1e9,
            "p95": percentile(ops, 0.95) * 1e9,
            "p99": percentile(ops, 0.99) * 1e9,
            "max": ops[-1] * 1e9,
            "elements": elements,
        }
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=CRITERION)
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--filter", default="", help="substring filter on benchmark id")
    args = ap.parse_args()

    rows = [r for r in collect(args.dir) if args.filter in r["id"]]
    if not rows:
        print(f"no criterion samples under {args.dir}", file=sys.stderr)
        return 1

    if args.markdown:
        print("| benchmark | n | mean | p50 | p95 | p99 | max | ops/s |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            ops = fmt_ops(r["elements"], r["mean"]) if r["elements"] else "—"
            print(f"| `{r['id']}` | {r['n']} | {fmt_lat(r['mean'])} "
                  f"| {fmt_lat(r['p50'])} | {fmt_lat(r['p95'])} "
                  f"| {fmt_lat(r['p99'])} | {fmt_lat(r['max'])} | {ops} |")
    else:
        width = max(len(r["id"]) for r in rows)
        print(f"{'benchmark':<{width}}  {'n':>4}  "
              f"{'mean':>12}  {'p50':>12}  {'p95':>12}  {'p99':>12}  {'max':>12}  ops/s")
        for r in rows:
            ops = fmt_ops(r["elements"], r["mean"]) if r["elements"] else "—"
            print(f"{r['id']:<{width}}  {r['n']:>4}  "
                  f"{fmt_lat(r['mean']):>12}  {fmt_lat(r['p50']):>12}  "
                  f"{fmt_lat(r['p95']):>12}  {fmt_lat(r['p99']):>12}  "
                  f"{fmt_lat(r['max']):>12}  {ops}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
