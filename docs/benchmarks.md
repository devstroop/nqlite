# Benchmarks & performance

Reproducible benchmark page for nqlite — how to regenerate every number that
appears in this repo's performance claims. All timings are wall-clock
milliseconds, lower is better. The `nql-bench` corpus runs are measured
**in-memory only**; persistence / cold-start numbers (the on-disk store) have
their own section below.

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
isn't installed. All three drivers are now wired **including the `rec@10`
quality column** (mean over 30 queries vs exact cosine top-10 on the shared
corpus; each store's vectors are unit-normalized at insert so its L2 ranking
is cosine-equivalent — see the shared `unit`/`mean_recall10` helpers in the
driver; lancedb rows carry an `id` column for index mapping).

Latest run with all columns: [`scripts/bench-compare/report-2026-10-06.md`](../scripts/bench-compare/report-2026-10-06.md)
(**all three competitors installed** — sqlite-vec 0.1.9, lancedb 0.39.0,
chromadb 1.5.9): recall nqlite **1.0000 @1k / 0.96 @5k** vs **1.0000 for all
three competitors at both sizes** (competitors' rec@10 = 30-query mean vs
exact cosine top-10 on the shared corpus, vectors unit-normalized so their
L2 ranking ≡ cosine — bases stated in the report). The earlier
`report-2026-08-04.md` (Raspberry Pi, all three drivers) remains the
three-driver *latency* reference — numbers are never comparable across those
two machines.

## Recall (quality, issue #96)

Latency is half the story; the other half is whether the **opt-in HNSW index
still returns the right neighbors**. `nql-bench --recall` measures the
standard ANN metric — recall@K of HNSW against the exact brute-force top-K —
on its own deterministic dim-64 vector set (seed 42, queries seeded apart
from the corpus). The dim-8 *timing* corpus is deliberately not used here:
in 8 dimensions HNSW is trivially exact and would report a meaningless 1.0.

```sh
# quality report (JSON; requires the hnsw feature for ANN numbers)
cargo run -q -p nql-bench --features hnsw -- --recall --rows 5000
cargo run -q -p nql-bench --features hnsw -- --recall --rows 50000 --queries 30 --dim 64

# parameter sweep (recall degrades with N at default params — see below)
cargo run -q -p nql-bench --features hnsw -- --recall --rows 50000 --hnsw-m 32 --hnsw-ef 256

# the gate: asserts recall@10 >= 0.95 (decisions §6 target)
cargo test -p nqlite --test recall --features hnsw
```

Measured on this box (2026-10-06, `dim=64`, `HnswVectorIndex::new(seed=42, …)`):

| rows | queries | m / efc / ef | recall@10 | recall@50 | recall@100 |
|---:|---:|---|---:|---:|---:|
| 5 000 | 20 | 16 / 200 / 64 (default) | **0.96** | 0.959 | 0.937 |
| 50 000 | 30 | 16 / 200 / 64 (default) | **0.81** | 0.756 | 0.687 |

**Parameter sweep (2026-10-07, R6's remaining item — same box, release build,
`dim=64`, seed 42, default `queries=20`):** the 50k ladder at default
`m/efc`, then the candidate profile verified down the rungs:

| rows | m / efc / ef | recall@10 | recall@50 | note |
|---:|---|---:|---:|---|
| 50 000 | 16 / 200 / 64 (default) | 0.795 | 0.752 | baseline at 20 q (the 0.81 above is the 30-q run) |
| 50 000 | 16 / 200 / 128 | 0.860 | 0.801 | |
| 50 000 | 16 / 200 / 256 | 0.945 | 0.907 | just under target |
| 50 000 | 16 / 200 / 512 | **0.960** | 0.958 | clears ≥0.95, barely |
| 50 000 | **32 / 200 / 256** | **0.975** | 0.971 | **recommended** |
| 50 000 | 32 / 400 / 256 | **0.990** | 0.981 | max-quality (heavier build) |
| 1 000 | 16 / 200 / 64 → 32 / 200 / 256 | 1.000 → **1.000** | 0.997 → 1.000 | no regression |
| 5 000 | 16 / 200 / 64 → 32 / 200 / 256 | 0.960 → **1.000** | 0.959 → 0.998 | improves the gate regime too |

**Recommendation.** For stores at **≥10k rows**, build with
`--hnsw-m 32 --hnsw-ef 256` (keep `efc=200`): recall@10 then holds
**≥0.95 at every rung** (1.000 / 1.000 / 0.975 at 1k / 5k / 50k). The
defaults (`16 / 200 / 64`) are unchanged — they pass the5k gate, remain the
compact profile, and the gate must never move silently (re-run
`cargo test -p nqlite --test recall --features hnsw` after any default
change). Cost knobs separate cleanly: **`m`/`efc` are build-time** (m32+efc200
≈ +26 s wall @50k in this harness; efc400 ≈ +68 s) while **`ef` is the live
query beam** — across the ef ladder the wall stayed within noise of the ~45 s
exact-ground-truth floor, so `ef` shows up in QPS, not in this report. Each
sweep run took 45–115 s, dominated by the exact brute-force baseline.

Keep this table updated when params or the corpus change.

Notes:

- Recall numbers are **build-profile independent** — the HNSW graph and the
  queries are seeded, so debug and release produce the same graph and the
  same neighbors. (Latencies above are dev-profile; recall is not.)
- Search beam: `fast-hnsw` widens `ef` to `max(ef, k)` — measuring with
  `k = rows` would silently turn the gate into a near-exact run. Both the
  bench and the gate cap `k` at 100 (see the comment in `nqlite/tests/recall.rs`).
- `scripts/bench-compare/bench.py` carries a `rec@10` column: nqlite's own
  HNSW-vs-exact (its dim-64 set) and — since 2026-10-06 — sqlite-vec's recall
  vs exact cosine top-10 on the shared corpus (`off` when nql-bench is built
  without `--features hnsw`; LanceDB/Chroma `n/a` — no quality metric wired).

## Cold start / persistence (issues #115, #133)

Cold-start cost of the on-disk store — how long `Database::open` takes before
the first query, and what the first *temporal* query additionally pays.

**Method:** a 100 000-record store (one `INSERT` per row) written by the
current `checkpoint` path (format **v3**: core frame + history tail, see
`spec/file-format.md` §1), then:

```sh
cargo build --release -p nqlite --example open_profile
target/release/examples/open_profile /path/to/store.ndb   # load / count / scan
# E08 harness (file_tier cold-open @100k + kNN/BM25/hybrid ladder):
cd ../nqlite-experiments && EXP08_PROFILE=release EXP08_SIZES=1000,5000,20000,50000,100000 \
  NQL_SERVER_BIN=$PWD/../nqlite/target/release/nql-server \
  NQL_CLI_BIN=$PWD/../nqlite/target/release/nql \
  python3 experiments/exp08_scale_ladder.py
```

**Measured 2026-10-07, commit `915a79b`, release profile, reference box
(see Methodology):**

| metric | v3 (this run) | pre-#133 (legacy full decode) |
|---|---:|---:|
| `open_profile` load, first touch | 459.7 ms | — |
| `open_profile` load, warm (median of 3) | **260 ms** (255.6–262.4) | ~640–715 ms |
| CLI open-only (no query) | **542–549 ms** | ~875 ms |
| CLI, first statement temporal | 958–1273 ms | (folded into load) |
| E08 `file_tier` cold-open @100k, WAL | **1338 ms** | ~1810 ms |
| E08 `file_tier` cold-open @100k, ckpt | **1135 ms** | ~1810 ms |

### Replay snapshots (issue #166)

`AS OF T` used to replay the whole mutation log on every temporal read
(`replay_as_of` from ts0 — O(history) per read). Two changes, both pure
caches (results byte-identical either way):

- **Recent-T fast path:** a cutoff at or past the current clock covers the
  whole log, so the view *is* the current state — zero replayed statements.
- **In-memory snapshot ring:** every `SNAPSHOT_EVERY` (20k) mutations (at
  most 2 retained, history-stripped) in time-travelling sessions, so
  in-session replays start from the newest base at or below the cutoff.
  Only sessions that run `AS OF` pay capture (~100 ms clone per epoch on a
  100k store, ~5 µs amortized per mutation); capture starts on the second
  temporal plan, so single-read sessions pay nothing.

What was deliberately NOT built: a persisted snapshot sidecar. Measured
A/B (interleaved cold opens, same load): postcard-decoding persisted
snapshots costs about as much as the replay it skips (both rebuild the
same record maps), while cold opens are dominated by open + tail-claim
anyway — net negative for a whole new file, so it stays out (same bar as
#144's norm-cache ~0%).

**Method:** `temporal_cold` (reopen per iteration: open + tail + replay +
read) and `temporal_warm` (one handle: replay + read) criterion groups
over 10k/50k/100k-mutation stores, `SELECT COUNT(*) ... AS OF <max>`
(recent T — isolates replay; COUNT still materializes rows, so warm
includes a ~200 ms @100k row-clone floor unrelated to replay):

```sh
cargo bench -p nqlite --bench bench -- 'temporal'
python3 scripts/bench-percentiles.py --filter temporal
```

**Measured 2026-10-09, release criterion profile, reference box (shared VM
— expect ±20–30% across runs; the A/B below is the load-proof comparison):**

| rows | cold mean | warm mean |
|-----:|----------:|----------:|
| 10 000 | 46.3 ms | 18.2 ms |
| 50 000 | 334.3 ms | 118.4 ms |
| 100 000 | 805 ms | 300.3 ms |

**Same-load A/B** (one warm handle, interleaved rounds, fat dim-8 corpus):
`AS OF max` (fast path, 0 replayed) vs `AS OF max-1` (full 100k replay)
→ **~220 ms vs ~275 ms**, i.e. the fast path removes the ~55 ms replay
slice; the remainder is the COUNT row-materialization both paths share.
Residual, by construction: cold opens still pay open + tail-claim (#133's
domain); old-T cutoffs in sessions without in-epoch snapshots still pay
full replay (bounded per base once the ring engages).

File layout of the profiled store: 62 511 796 B total = **31 547 635 B core
frame** + **30 964 137 B history tail**. `open` decodes only the core (records,
edges, tables) — the history tail is claimed lazily and decoded on the first
temporal read (`HISTORY`, `AS OF`, closures; measured above as the
temporal-first delta), which is why warm load sits at ~0.26 s (target ≤ 0.3 s,
decisions §6) while a temporal-first session pays core + tail + replay.

Compatibility (tested in `nqlite/tests/persistence.rs`): legacy **v2** files
still load (inline postcard layout, tables rebuilt), **v1 / v99** are rejected
with `BadVersion (supported: 2, 3)`, truncated v3 frames with `Truncated`.

## Latency distribution & plan-size sweep (issue #170)

Criterion's console output leads with mean/median; the tails live in its JSON
artifacts. `scripts/bench-percentiles.py` is a pure function of those
artifacts — per-operation samples (`times[i]/iters[i]`, linear-interpolation
percentiles, ops/s = throughput-elements / mean):

```sh
cargo bench -p nqlite              # release-optimized criterion harness
python3 scripts/bench-percentiles.py            # text table
python3 scripts/bench-percentiles.py --markdown # paste-ready rows
```

Tables below are that script's output, reference box, release profile,
2026-10-09 (shared VM — expect ±10–30% run-to-run; the *shapes* reproduce,
the exact decimals don't).

**Plan-size sweep** (`plan_size` group in `nqlite/benches/bench.rs`): plans of
1/10/100/1000 inserts per `execute`. `mem/*` isolates the engine's per-plan
cost; `wal/*` measures the durable plan (fresh tempdir + open + execute, so
the single per-plan fsync from #164 is inside the measured path). Latency is
per plan (p50), rate per statement:

| plan size | mem latency | mem rate | wal latency | wal rate |
|----------:|------------|---------:|------------:|---------:|
| 1 | 1.356 µs | 653.8 K/s | 869.8 µs | 1.150 K/s |
| 10 | 10.95 µs | 903.3 K/s | 947.1 µs | 10.51 K/s |
| 100 | 126.8 µs | 770.8 K/s | 1.398 ms | 70.79 K/s |
| 1000 | 1.551 ms | 635.3 K/s | 5.367 ms | 195.5 K/s |

Reading: the engine's per-plan overhead is negligible (mem rate flat
~635–903 K/s across sizes). The durable plan has a ~0.87 ms floor at size 1
(open + serialize + one fsync) that amortizes away — wal rate climbs
1.15 K/s → 195 K/s, and at size 1000 the durable plan (5.37 ms) costs only
~3.5× the in-memory plan (1.55 ms). Batching the WAL write per plan (#164)
is what makes this curve flat-ish instead of fsync-per-statement.

**Spike kNN distribution re-quote** (`spike_knn_dim64`, dim-64, k=10 —
idle-box run, loadavg ~4 on 24 cores, 2026-10-09; rows/s = rows scanned per
second). This is the clean re-measure the #144 stream owed: 100k lands at
**35.6 ms mean / 39.0 ms p99**, inside the sign-off's 35–40 ms target
(stream total 357 → ~36 ms, ~10×, determinism contract untouched):

| rows | mean | p50 | p95 | p99 | max | rows/s |
|-----:|------|-----|-----|-----|-----|-------:|
| 10 000 | 1.776 ms | 1.759 ms | 1.921 ms | 2.121 ms | 2.146 ms | 5.630 M/s |
| 50 000 | 15.78 ms | 15.47 ms | 17.54 ms | 24.44 ms | 24.59 ms | 3.168 M/s |
| 100 000 | 35.64 ms | 35.18 ms | 37.79 ms | 38.96 ms | 63.61 ms | 2.806 M/s |

Attribution split at 100k (`spike_parts`, same idle run): build 83.04 ms
mean / 123.1 ms p99 (1.204 M rows/s), scan 14.13 ms mean / 15.75 ms p99
(7.076 M rows/s) — the tail sits in build + pipeline, not the scan loop.
(An earlier same-day run under heavier load read 37.26 ms mean / 53.54 ms
p99 @100k — same shape, load-shifted. Quote the shape and the method, not
the decimals.)

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

- **kNN is an exact brute-force scan by default — no ANN index.** After the
  #144 L1→L3 stream (borrow candidates → memoized index → k-capped windowed
  search, behind the spec §2.3 output-cap invariant), a single k=10 exact
  kNN query costs **35.6 ms @100k rows idle-box mean (39.0 ms p99; ~52 ms
  under load)** and ~1.8 ms @10k — exact, deterministic, byte-identical
  output. ANN-backed stores (sqlite-vec/LanceDB/Chroma) still win at larger
  scales; that is the trade for exactness. (The previous "~140 ms @10k, 1–2
  orders slower" claim predates the stream.) The upside
  is that the exact index is on by default and ANN (feature-gated HNSW) can
  be opted into — its
  recall@10 vs exact is now measured (0.96 at 5k rows, see **Recall** above)
  and gated by `cargo test -p nqlite --test recall --features hnsw`.
- **These are dev-build numbers.** An unoptimized build is what the
  reproducible commands produce; release would be substantially faster, but
  then every reader would need the same `--release` flags to compare. State
  the build whenever quoting these numbers.
- **Single-writer engine.** Everything above is one process, one writer.
  Disk-backed durable-plan numbers now exist (the §170 plan-size sweep —
  single fsync per plan); concurrent-writer numbers do not exist yet in this
  phase — do not extrapolate them from this page.

## Cross-implementation (nqlite-zig)

The experimental Zig port runs the same wire-level benchmarks through the
shared harness (`nqlite-experiments`): its M8 numbers — bound to
machine/profile/commit, ReleaseFast, ladder through 100k rows — live in
`nqlite-zig/docs/BENCHMARKING.md` (workspace sibling, not published).
Cross-implementation correctness is gated by
`nqlite-experiments/scripts/compare_impls.py`, which diffs per-variant
`transcript_sha256` of this crate's `nql-server` against the port's
server; the current run is byte-identical across all 11 experiments
(57/57 digests, `--all` exit 0).
