# Cross-DB benchmark harness

Compares nqlite's retrieval throughput against comparable embedded stores on
the **same deterministic synthetic corpus**.

## What it does

1. Generates a fixed corpus — xorshift64* (seed 42), dim-8 float vectors, an
   8-word vocabulary, `rows` documents — byte-identical to what `nql-bench`
   (the Rust runner) ingests.
2. Benchmarks **nqlite** always: ingest + kNN + BM25 + hybrid queries, via
   `cargo run -q -p nql-bench`.
3. Benchmarks competitors **if their driver is installed**; otherwise skips
   them with an honest note (the harness never fails on a missing driver):
   - **sqlite-vec** — `pip install sqlite-vec`
   - **LanceDB** — `pip install lancedb`
   - **Chroma** — `pip install chromadb`

No system pip? Use a venv (any Python 3.9+):

```sh
python3 -m venv /tmp/bench-venv
/tmp/bench-venv/bin/pip install sqlite-vec lancedb chromadb
/tmp/bench-venv/bin/python3 scripts/bench-compare/bench.py
```

## Running

```sh
# default: 1000 rows, 50 kNN queries
python3 scripts/bench-compare/bench.py

# tune sizes
python3 scripts/bench-compare/bench.py --rows 10000 --knn 200

# JSON report (machine-readable), optionally to a file
python3 scripts/bench-compare/bench.py --json
python3 scripts/bench-compare/bench.py --out results/bench-1k.json
```

## Reading the table

Only same-shape cells are comparable:

| column     | nqlite        | sqlite-vec | LanceDB | Chroma |
|------------|---------------|------------|---------|--------|
| ingest     | inserts + upserts | inserts  | inserts | inserts |
| knn        | brute-force kNN (k=10) | vec0 index | IVF-ish | HNSW |
| bm25       | `::bm25` operator | FTS5 (not wired) | — | — |
| hybrid     | BM25 + vector RRF fusion | — | — | — |

`n/a` = the competitor has no equivalent operator; compare only cells both
sides produce. The corpus is deterministic, so numbers are comparable across
runs and machines — but these are **throughput/latency** numbers only:
nqlite's engine output is always identical for identical input (determinism
contract), so this harness never measures correctness.

## Determinism guarantee

- Corpus: xorshift64* with seed 42 — the Rust `nql-bench` and the Python
  generator implement the same PRNG and vocabulary, so both sides see the
  same data distribution.
- Query workload: same kNN query vector (derived from the seed), same BM25
  query string, same iteration counts.

## Files

- `nql-bench/` — Rust runner (workspace member): builds the corpus in-process
  and emits a JSON timing report.
- `scripts/bench-compare/bench.py` — the orchestrator: corpus + nqlite +
  optional competitors + report.
