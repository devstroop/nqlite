# nqlite benchmarks

Criterion benchmark harness for the **engine** (not the nql parser). All inputs
are built programmatically as `nql_ir::Statement` values from a fixed-seed,
deterministic PRNG (xorshift64* — no `rand` dependency), so every run sees
byte-identical data and results are comparable across runs and machines.

The engine is deterministic (stable BTree scans, tie-broken sorts, no
wall-clock in execution), so these benches measure **throughput/latency only**:
given identical input, engine output is always identical.

## Benches

| Group         | What it measures                                                        | Sizes  |
|---------------|-------------------------------------------------------------------------|--------|
| `ingest`      | N `INSERT`s (dim-8 vectors) in one plan vs a fresh store → inserts/sec  | 1k, 10k |
| `knn_bf`      | Brute-force kNN `SELECT` (k=10) over an N-record store → QPS, latency   | 1k, 10k |
| `select_range`| `SELECT WHERE group = 3` (field-equality filter, ~N/10 matches) → QPS   | 1k, 10k |
| `relate`      | N `RELATE` edges in one plan vs a fresh store → relates/sec             | 1k, 10k |

## Running

```sh
# all benches (default criterion warmup + measurement)
cargo bench -p nqlite

# a single group
cargo bench -p nqlite -- 'knn_bf'

# a quick smoke run (short warmup/measurement, small sample)
cargo bench -p nqlite -- --warm-up-time 0.5 --measurement-time 1 --sample-size 10
```

HTML reports land in `target/criterion/report/index.html` (enabled via the
`html_reports` feature on the `criterion` dev-dependency).

## Notes

- Benches re-run a read-only `SELECT` against the same `Database` instance to
  measure steady-state query cost; DML benches (ingest/relate) execute one plan
  against a fresh store per iteration.
- `cargo clippy --workspace --all-targets` compiles these benches; keep them
  warning-free.
- Bench harness wiring lives in `nqlite/Cargo.toml` (`[[bench]] name = "bench",
  harness = false` — criterion provides its own runner).
