//! Track B spike: exact-scan at scale (dim 64, k=10) — the regime where
//! E08's "75–140 ms floor @100K" lives. Baseline first, then candidates
//! (norm-cache, top-k selection, SIMD dot) land against these numbers.
//!
//! Run: `cargo bench -p nqlite --bench spike`

use std::collections::BTreeMap;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use nql_ir::{Id, Knn, Record, RecordId, Select, Statement, Store, Value};
use nqlite::Database;

/// Match the recall corpus regime (dim-64), not the timing corpus (dim-8).
const DIM: usize = 64;
const KNN_K: usize = 10;
const SEED: u64 = 0xBEEF_5ADE;

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
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

    fn next_f32(&mut self) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        unit * 2.0 - 1.0
    }
}

fn seeded_db(n: usize) -> Database {
    let mut rng = Rng::new(SEED);
    let mut plan = Vec::with_capacity(n + 1);
    plan.push(Statement::CreateTable {
        table: "item".into(),
        vector_dim: Some(DIM),
    });
    for i in 0..n {
        let embedding: Vec<f32> = (0..DIM).map(|_| rng.next_f32()).collect();
        let body = BTreeMap::from([
            ("name".into(), Value::Str(format!("item-{i}"))),
            ("group".into(), Value::Int((i % 10) as i64)),
        ]);
        plan.push(Statement::Insert(Record {
            id: RecordId::new("item", Id::Num(i as u64)),
            body,
            embedding: Some(embedding),
            created_at: 0,
        }));
    }
    let mut db = Database::new(Store::default());
    db.execute(&plan).expect("seed store");
    db
}

fn bench_knn_dim64(c: &mut Criterion) {
    let mut group = c.benchmark_group("spike_knn_dim64");
    for n in [10_000usize, 50_000, 100_000] {
        let mut db = seeded_db(n);
        let query: Vec<f32> = (0..DIM).map(|d| (d as f32) / DIM as f32 - 0.5).collect();
        let select = Statement::Select(Select {
            table: "item".into(),
            knn: Some(Knn { query, k: KNN_K }),
            ..Select::default()
        });
        group.throughput(Throughput::Elements(n as u64));
        group.bench_function(format!("k={KNN_K}/{n}"), |b| {
            b.iter(|| {
                db.execute(std::slice::from_ref(&select))
                    .expect("knn select");
            })
        });
    }
    group.finish();
}

/// Attribution: where does the per-query kNN time go? The engine rebuilds
/// its index (clone + BTree insert per record) on every SELECT
/// (engine.rs `build_default_index`) — split that from the scan.
fn bench_parts(c: &mut Criterion) {
    use nqlite::{BruteForceVectorIndex, VectorIndex};
    let n = 100_000usize;
    let db = seeded_db(n);
    let store = db.store();

    // (a) index construction as the engine does it per SELECT.
    let mut group = c.benchmark_group("spike_parts_build");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function(format!("build/{n}"), |b| {
        b.iter(|| {
            let mut index = BruteForceVectorIndex::default();
            for r in store.records.values() {
                if let Some(emb) = &r.embedding {
                    index.upsert(r.id.clone(), emb.clone());
                }
            }
            std::hint::black_box(&index);
        })
    });
    group.finish();

    // (b) scan over an already-built index (the hot inner loop).
    let mut index = BruteForceVectorIndex::default();
    for r in store.records.values() {
        if let Some(emb) = &r.embedding {
            index.upsert(r.id.clone(), emb.clone());
        }
    }
    let query: Vec<f32> = (0..DIM).map(|d| (d as f32) / DIM as f32 - 0.5).collect();
    let mut group = c.benchmark_group("spike_parts_scan");
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function(format!("scan/{n}"), |b| {
        b.iter(|| std::hint::black_box(index.search(&query, KNN_K)))
    });
    group.finish();
}

criterion_main!(spike);
criterion_group!(spike, bench_knn_dim64, bench_parts);
