# Benchmarks & performance

Reproducible benchmark page for nqlite — how to regenerate every number that
appears in this repo's performance claims. All timings are wall-clock
milliseconds, lower is better, measured **in-memory only** (no persistence
layer yet).

## Methodology

**Corpus (deterministic, byte-identical across harnesses).** Every run
generates the same synthetic corpus:

- PRNG: xorshift64*, seed fixed at `42`.
- Each record: a dim-**8** float vector (components on a 0–999 grid scaled to
  `[0, 1)`), a **2-word text body** drawn from the fixed **8-word vocabulary**
  `alpha, beta, gamma, delta, rust, query, memory, graph`, and `group = i % 10`.
- The generator is implemented twice, byte-identically: in
  `nql-bench/src/main.rs` (Rust) and `scripts/bench-compare/bench.py`
  (Python). (Older doc comments in both files reference a
  `scripts/bench-compare/corpus.py`; the code lives inline in those two files —
  there is no standalone `corpus.py`.)

**What is measured** (`nql-bench`):

| Kind | What |
|---|---|
| ingest | one plan of N inserts, total wall time |
| kNN | `--knn` iterations of a single exact k=10 brute-force `SELECT ... kNN` query, total wall time |
| BM25 | `--knn` iterations of a lexical `Filter::Bm25` on `text`, total wall time |
| hybrid | `--knn` iterations of one fused BM25 + kNN select, total wall time |

**Build:** dev profile (`cargo build`, unoptimized) — deliberately, so any
checkout reproduces these numbers without `--release`. Do not compare these
numbers with release-build numbers from other projects without the
optimization caveat.

**Hardware** (measured 2026-08-07 on the reference box):

```
Linux linux-dev 6.17.2-1-pve #1 SMP PREEMPT_DYNAMIC PMX 6.17.2-1 (2025-10-21T11:55Z) x86_64 x86_64 x86_64 GNU/Linux
nproc: 16
```

**Run-to-run variance:** these are single-process wall-clock timings on a
shared Proxmox VM; ±10–30% run-to-run is normal at this scale. The table below
is the **median of 4 runs** per configuration, to damp that noise.

## Results (this box, 2026-08-07)

| rows | kNN queries | ingest (ms) | kNN (ms) | BM25 (ms) | hybrid (ms) |
|-----:|------------:|------------:|---------:|----------:|------------:|
| 1000 |          50 |       12.34 |   736.88 |    655.75 |     1089.54 |
| 10000 |         10 |      136.56 |  1428.93 |   1176.16 |     2134.77 |

Raw single-run samples (for noise reference):

```
# 1000 rows, 50 kNN queries
{"ingest_ms":12.43,"knn_ms":725.62,"bm25_ms":619.08,"hybrid_ms":1062.58}
{"ingest_ms":12.26,"knn_ms":726.27,"bm25_ms":657.39,"hybrid_ms":1101.27}
{"ingest_ms":12.80,"knn_ms":747.50,"bm25_ms":666.74,"hybrid_ms":1088.38}
{"ingest_ms":11.74,"knn_ms":804.11,"bm25_ms":654.12,"hybrid_ms":1090.69}
# 10000 rows, 10 kNN queries
{"ingest_ms":139.13,"knn_ms":1435.07,"bm25_ms":1075.44,"hybrid_ms":2065.11}
{"ingest_ms":135.25,"knn_ms":1496.96,"bm25_ms":1199.41,"hybrid_ms":2234.78}
{"ingest_ms":134.75,"knn_ms":1415.51,"bm25_ms":1178.16,"hybrid_ms":2166.28}
{"ingest_ms":137.87,"knn_ms":1422.79,"bm25_ms":1174.17,"hybrid_ms":2103.26}
```

Sanity note: 10× the rows costs ~2× the kNN/BM25 time (linear-ish in this
range) because both are exact scans over in-memory data; hybrid ≈ kNN + BM25.

## Cross-DB matrix

`scripts/bench-compare/bench.py` runs the same corpus against sqlite-vec,
LanceDB and Chroma **when their Python drivers are importable**; a missing
driver is reported as `skipped` — the harness never fails because a competitor
isn't installed. On this box none of the three drivers are present, so the
matrix degrades honestly to nqlite-only (see "Reproduce" for how to install
them).

## Reproduce

```sh
# full run: 1000 rows / 50 kNN + 10000 rows / 10 kNN (+ cross-DB matrix if drivers present)
./scripts/bench.sh
# results: results/bench-<date>.json + results/bench-<date>.log

# single configs, straight from source (no prior installs needed)
cargo run -q -p nql-bench -- --rows 1000 --knn 50 --seed 42
cargo run -q -p nql-bench -- --rows 10000 --knn 10 --seed 42

# cross-DB matrix (skips missing drivers honestly)
python3 scripts/bench-compare/bench.py --rows 1000 --knn 50
```

To enable the competitor columns:

```sh
python3 -m venv /tmp/bench-venv
/tmp/bench-venv/bin/pip install sqlite-vec lancedb chromadb
/tmp/bench-venv/bin/python3 scripts/bench-compare/bench.py --rows 1000 --knn 50
```

## Where nqlite is weak (honest)

- **kNN is an exact brute-force scan by default — no ANN index.** At 10k rows
  a single k=10 kNN query costs ~140 ms; that is the price of exact,
  deterministic results, and it is 1–2 orders of magnitude slower than
  sqlite-vec/LanceDB/Chroma on the same data. The upside is that the exact
  index is on by default and ANN (feature-gated HNSW-style) can be opted into
  when determinism-per-query is not the binding constraint.
- **These are dev-build numbers.** An unoptimized build is what the
  reproducible commands produce; release would be substantially faster, but
  then every reader would need the same `--release` flags to compare. State
  the build whenever quoting these numbers.
- **Single-writer, in-memory engine.** Everything above is one process, one
  writer, no persistence. Concurrent-writer and disk-backed numbers do not
  exist yet in this phase — do not extrapolate them from this page.