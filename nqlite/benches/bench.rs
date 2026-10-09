//! Criterion benchmark harness for the nqlite engine.
//!
//! Measures **engine** throughput (not the nql parser): every input is built
//! programmatically as [`nql_ir::Statement`] values from a fixed-seed,
//! deterministic PRNG (xorshift64*, no `rand` dependency), so all runs — and
//! all machines — see byte-identical input data. The engine's own determinism
//! guarantee (stable BTree scans, tie-broken sorts, no wall-clock) means the
//! numbers below compare **throughput/latency only**: outputs for a given
//! input are always identical.
//!
//! Benches:
//! - `ingest` — execute one plan of N `INSERT`s (dim-8 embeddings + body) into
//!   a fresh store; reports inserts/sec.
//! - `knn_bf` — repeated brute-force kNN `SELECT` (k = 10) over an N-record
//!   store; reports queries/sec (QPS) and per-query latency.
//! - `select_range` — repeated `SELECT` with `WHERE group = <const>` (field
//!   equality filter, ~N/10 matches) over N records; reports QPS.
//! - `selectivity` — predicate-selectivity sweep at 10k/100k (issue #169):
//!   `count/*` (COUNT after the filter — scan + filter only) and `rows/*`
//!   (return every match — scan + filter + materialize) twins for
//!   star / eq-unique / eq-10% / range-20% predicates. Rows scanned is
//!   always N (full BTree walk); the group isolates what matching costs.
//! - `relate` — execute one plan of N `RELATE` edges; reports relates/sec.
//! - `plan_size` — execute plans of 1/10/100/1000 `INSERT`s, in-memory
//!   (`mem`, engine plan cost) and persistent (`wal`: fresh tempdir + open +
//!   execute, so the single per-plan fsync from #164 is in the measured
//!   path); the amortization curve — per-plan latency vs per-statement rate.
//! - `temporal_cold` / `temporal_warm` — `SELECT ... AS OF <max>` over stores
//!   with 10k/50k/100k mutations of history (issue #166): cold reopens the
//!   store per iteration (open + lazy-tail claim + full replay — the painful
//!   path), warm reuses one handle (replay only). Slow groups: smoke with
//!   `--sample-size 10` (see README).
//!
//! Run everything: `cargo bench -p nqlite`
//! Run a subset:   `cargo bench -p nqlite -- 'knn_bf'`

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use nql_ir::{
    Aggregate, Filter, Id, Knn, Record, RecordId, RelationEdge, Select, Statement, Store, Value,
};
use nqlite::Database;

/// Fixed embedding dimension for all synthetic records.
const DIM: usize = 8;
/// Neighbour count for the kNN bench.
const KNN_K: usize = 10;
/// Fixed PRNG seed: identical synthetic data on every run.
const SEED: u64 = 0x5EED_CAFE;

/// Deterministic xorshift64* PRNG (Vigna's constants) — no `rand` dependency.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // xorshift with state 0 is stuck; force at least one bit set.
        Rng(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform value in `[-1.0, 1.0)`.
    fn next_f32(&mut self) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        unit * 2.0 - 1.0
    }

    /// Uniform value in `[0.0, 1.0)` (edge weights / confidence).
    fn next_weight(&mut self) -> f32 {
        self.next_f32() * 0.5 + 0.5
    }
}

/// One synthetic `INSERT` statement: record `i` in table `item` with a dim-8
/// embedding and a `group` field cycling `0..10` (for the range filter bench).
fn insert_statement(rng: &mut Rng, i: usize) -> Statement {
    let embedding: Vec<f32> = (0..DIM).map(|_| rng.next_f32()).collect();
    let body = BTreeMap::from([
        ("name".into(), Value::Str(format!("item-{i}"))),
        ("group".into(), Value::Int((i % 10) as i64)),
    ]);
    Statement::Insert(Record {
        id: RecordId::new("item", Id::Num(i as u64)),
        body,
        embedding: Some(embedding),
        created_at: 0,
    })
}

/// Seed a `Database` with `n` synthetic records (setup — never measured).
fn seeded_db(n: usize) -> Database {
    let mut rng = Rng::new(SEED);
    let mut plan = Vec::with_capacity(n + 1);
    plan.push(Statement::CreateTable {
        table: "item".into(),
        vector_dim: Some(DIM),
    });
    for i in 0..n {
        plan.push(insert_statement(&mut rng, i));
    }
    let mut db = Database::new(Store::default());
    db.execute(&plan).expect("seed store");
    db
}

/// Ingest: N inserts in a single plan against a fresh store.
fn bench_ingest(c: &mut Criterion) {
    let mut group = c.benchmark_group("ingest");
    for n in [1_000usize, 10_000] {
        let mut rng = Rng::new(SEED);
        let mut plan = Vec::with_capacity(n + 1);
        plan.push(Statement::CreateTable {
            table: "item".into(),
            vector_dim: Some(DIM),
        });
        for i in 0..n {
            plan.push(insert_statement(&mut rng, i));
        }
        group.throughput(Throughput::Elements(n as u64));
        group.bench_function(format!("inserts/{n}"), |b| {
            b.iter(|| {
                let mut db = Database::new(Store::default());
                db.execute(&plan).expect("execute ingest plan");
            })
        });
    }
    group.finish();
}

/// Brute-force kNN: repeated k=10 similarity SELECTs over an N-record store.
fn bench_knn_bf(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn_bf");
    for n in [1_000usize, 10_000] {
        let mut db = seeded_db(n);
        // Fixed deterministic query vector.
        let query: Vec<f32> = (0..DIM).map(|d| (d as f32) / DIM as f32 - 0.5).collect();
        let select = Statement::Select(Select {
            table: "item".into(),
            knn: Some(Knn { query, k: KNN_K }),
            ..Select::default()
        });
        group.throughput(Throughput::Elements(n as u64));
        group.bench_function(format!("k={KNN_K}/{n}"), |b| {
            // SELECT is read-only: re-running against the same db measures the
            // steady-state query path, not store re-seeding.
            b.iter(|| {
                db.execute(std::slice::from_ref(&select))
                    .expect("knn select");
            })
        });
    }
    group.finish();
}

/// Range scan: repeated `WHERE group = 3` (field-equality) SELECTs over N.
fn bench_select_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("select_range");
    for n in [1_000usize, 10_000] {
        let mut db = seeded_db(n);
        let select = Statement::Select(Select {
            table: "item".into(),
            filter: Some(Filter::FieldEquals {
                field: "group".into(),
                value: Value::Int(3),
            }),
            ..Select::default()
        });
        group.throughput(Throughput::Elements(n as u64));
        group.bench_function(format!("field_eq/scan-{n}"), |b| {
            b.iter(|| {
                db.execute(std::slice::from_ref(&select)).expect("select");
            })
        });
    }
    group.finish();
}

/// Relate: N `RELATE` edges in a single plan against a fresh store.
fn bench_relate(c: &mut Criterion) {
    let mut group = c.benchmark_group("relate");
    for n in [1_000usize, 10_000] {
        let mut rng = Rng::new(SEED ^ 0xBEEF);
        let mut plan = Vec::with_capacity(n);
        for i in 0..n {
            plan.push(Statement::Relate(RelationEdge {
                from: RecordId::new("item", Id::Num(i as u64)),
                name: "references".into(),
                to: RecordId::new("item", Id::Num((i + 1) as u64)),
                created_at: 0,
                weight: Some(rng.next_weight()),
                props: BTreeMap::new(),
            }));
        }
        group.throughput(Throughput::Elements(n as u64));
        group.bench_function(format!("edges/{n}"), |b| {
            b.iter(|| {
                let mut db = Database::new(Store::default());
                db.execute(&plan).expect("execute relate plan");
            })
        });
    }
    group.finish();
}

/// Predicate-selectivity sweep (issue #169 — measure first).
///
/// Field predicates are a full scan: `run_select` walks `store.records` in
/// BTree order and applies `matches_filter` per row, so **rows scanned = N
/// for every shape** and only the matched count varies. Each predicate is
/// measured as a `count/*` twin (COUNT(*) after the filter — scan + filter
/// only, no row materialization) and a `rows/*` twin (return every match —
/// scan + filter + materialize clones) at three selectivities (1 row, 10%,
/// 20%) over two store sizes, so docs/benchmarks.md can split scan cost
/// from materialization cost and say what an index would actually buy.
///
/// Match counts are asserted once per store at setup (deterministic
/// corpus), so the recorded selectivities cannot drift silently.
fn bench_selectivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("selectivity");
    for n in [10_000usize, 100_000] {
        let mut db = seeded_db(n);
        let cases: [(&str, Option<Filter>, usize); 4] = [
            ("star", None, n),
            (
                "eq-unique",
                Some(Filter::FieldEquals {
                    field: "name".into(),
                    value: Value::Str(format!("item-{}", n - 1)),
                }),
                1,
            ),
            (
                "eq-10pct",
                Some(Filter::FieldEquals {
                    field: "group".into(),
                    value: Value::Int(3),
                }),
                n / 10,
            ),
            (
                "range-20pct",
                Some(Filter::FieldBetween {
                    field: "group".into(),
                    lo: Value::Int(8),
                    hi: Value::Int(9),
                }),
                2 * (n / 10),
            ),
        ];

        // Pin the selectivities: the corpus is fixed, so every recorded
        // matched-count must hold or the bench must not run.
        for (label, filter, expected) in &cases {
            let probe = Statement::Select(Select {
                table: "item".into(),
                filter: filter.clone(),
                ..Select::default()
            });
            let probe = db
                .execute(std::slice::from_ref(&probe))
                .expect("selectivity probe");
            assert_eq!(probe.len(), 1, "one result per SELECT");
            assert_eq!(
                probe[0].rows.len(),
                *expected,
                "selectivity/{label}/{n} matched"
            );
        }

        for (label, filter, _expected) in &cases {
            let rows_sel = Statement::Select(Select {
                table: "item".into(),
                filter: filter.clone(),
                ..Select::default()
            });
            let count_sel = Statement::Select(Select {
                table: "item".into(),
                filter: filter.clone(),
                aggregate: Some(Aggregate::CountStar),
                ..Select::default()
            });
            // Throughput = rows scanned (always N — full walk), so ops/s
            // converts directly to scans/s in the reporter.
            group.throughput(Throughput::Elements(n as u64));
            group.bench_function(format!("count/{label}/{n}"), |b| {
                b.iter(|| db.execute(std::slice::from_ref(&count_sel)).expect("count"))
            });
            group.bench_function(format!("rows/{label}/{n}"), |b| {
                b.iter(|| db.execute(std::slice::from_ref(&rows_sel)).expect("rows"))
            });
        }
    }
    group.finish();
}

/// Monotonic counter so concurrent `plan_wal` iterations never share a dir.
static PLAN_WAL_SEQ: AtomicU64 = AtomicU64::new(0);

/// Fresh persistent path per `plan_wal` iteration (setup — the open itself
/// is part of the measured durable-plan cost, the dir bookkeeping is not
/// timed separately from it).
fn fresh_sweep_path() -> std::path::PathBuf {
    let seq = PLAN_WAL_SEQ.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!("nqlite-plan-sweep-{}-{seq}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create plan sweep dir");
    dir.join("sweep.ndb")
}

/// Plan-size sweep: 1/10/100/1000 inserts per `execute` (issue #170).
///
/// `mem/*` isolates the engine's per-plan cost (no durability); `wal/*`
/// measures the durable plan (open + execute + the single per-plan fsync
/// from #164 — no explicit flush, the WAL stays well under the checkpoint
/// threshold at these sizes). Throughput is per statement; latency is per
/// plan — together they are the amortization curve.
fn bench_plan_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("plan_size");
    for n in [1usize, 10, 100, 1000] {
        let mut rng = Rng::new(SEED);
        let mut plan = Vec::with_capacity(n + 1);
        plan.push(Statement::CreateTable {
            table: "item".into(),
            vector_dim: Some(DIM),
        });
        for i in 0..n {
            plan.push(insert_statement(&mut rng, i));
        }
        group.throughput(Throughput::Elements(n as u64));
        group.bench_function(format!("mem/{n}"), |b| {
            b.iter(|| {
                let mut db = Database::new(Store::default());
                db.execute(&plan).expect("execute mem plan");
            })
        });
        group.bench_function(format!("wal/{n}"), |b| {
            b.iter(|| {
                let path = fresh_sweep_path();
                let mut db = Database::open(&path).expect("open sweep db");
                db.execute(&plan).expect("execute wal plan");
                drop(db);
                std::fs::remove_dir_all(path.parent().expect("sweep parent")).ok();
            })
        });
    }
    group.finish();
}

/// Build a persistent store with `n` mutations of history (issue #166
/// microbench): chunked 1k-insert executes with a flush every 10 chunks —
/// realistic accumulation with checkpoints, so open/replay costs match a
/// lived-in store. Returns the dir (kept alive by the caller for the
/// group) and the final clock (the recent-T cutoff: full-history replay).
/// Mid-build one `AS OF` runs so the snapshot cache primes the way a real
/// time-travelling session would (captures engage, final flush persists
/// the sidecar) — cold opens measure the primed path.
fn build_history_store(n: usize) -> (std::path::PathBuf, i64) {
    let dir = std::env::temp_dir().join(format!("nqlite-temporal-{n}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temporal dir");
    let path = dir.join("t.ndb");
    let mut db = Database::open(&path).expect("open temporal store");
    db.execute(&[Statement::CreateTable {
        table: "item".into(),
        vector_dim: Some(DIM),
    }])
    .expect("create table");
    let mut rng = Rng::new(SEED);
    let mut done = 0usize;
    while done < n {
        let take = (n - done).min(1000);
        let mut plan = Vec::with_capacity(take);
        for i in 0..take {
            plan.push(insert_statement(&mut rng, done + i));
        }
        db.execute(&plan).expect("ingest chunk");
        done += take;
        if done % 10_000 == 0 {
            db.flush().expect("checkpoint chunk");
        }
    }
    db.flush().expect("final checkpoint");
    let clock = db.store().clock;
    drop(db);
    (dir, clock)
}

/// Temporal replay microbench (issue #166): `SELECT COUNT(*) ... AS OF
/// <max>` (recent T — the full-history worst case) over 10k/50k/100k-
/// mutation stores. COUNT isolates replay cost (no row materialization).
/// `cold/*` reopens per iteration (open + lazy-tail claim + replay — the
/// first-read path); `warm/*` reuses one handle, so the second and later
/// reads prime the in-session snapshot ring (recent-T replays ~nothing).
fn bench_temporal(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_cold");
    for n in [10_000usize, 50_000, 100_000] {
        let (dir, clock) = build_history_store(n);
        let path = dir.join("t.ndb");
        let select = Statement::Select(Select {
            table: "item".into(),
            as_of: Some(clock),
            aggregate: Some(Aggregate::CountStar),
            ..Select::default()
        });
        group.throughput(Throughput::Elements(n as u64 + 1));
        group.bench_function(format!("asof-max/{n}"), |b| {
            b.iter(|| {
                let mut db = Database::open(&path).expect("reopen");
                db.execute(std::slice::from_ref(&select)).expect("asof");
            })
        });
        std::fs::remove_dir_all(&dir).ok();
    }
    group.finish();

    let mut group = c.benchmark_group("temporal_warm");
    for n in [10_000usize, 50_000, 100_000] {
        let (dir, clock) = build_history_store(n);
        let path = dir.join("t.ndb");
        let select = Statement::Select(Select {
            table: "item".into(),
            as_of: Some(clock),
            aggregate: Some(Aggregate::CountStar),
            ..Select::default()
        });
        let mut db = Database::open(&path).expect("open");
        group.throughput(Throughput::Elements(n as u64 + 1));
        group.bench_function(format!("asof-max/{n}"), |b| {
            b.iter(|| {
                db.execute(std::slice::from_ref(&select)).expect("asof");
            })
        });
        drop(db);
        std::fs::remove_dir_all(&dir).ok();
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_ingest,
    bench_knn_bf,
    bench_select_range,
    bench_relate,
    bench_plan_size,
    bench_selectivity,
    bench_temporal
);
criterion_main!(benches);
