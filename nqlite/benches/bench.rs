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
//! - `relate` — execute one plan of N `RELATE` edges; reports relates/sec.
//!
//! Run everything: `cargo bench -p nqlite`
//! Run a subset:   `cargo bench -p nqlite -- 'knn_bf'`

use std::collections::BTreeMap;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use nql_ir::{Filter, Id, Knn, Record, RecordId, RelationEdge, Select, Statement, Store, Value};
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

criterion_group!(
    benches,
    bench_ingest,
    bench_knn_bf,
    bench_select_range,
    bench_relate
);
criterion_main!(benches);
