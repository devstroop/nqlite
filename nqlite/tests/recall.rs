//! Recall@K quality gate (issue #96): the ANN path must meet decisions §6's
//! "kNN recall@10 >= 0.95 on a standard set" target against exact ground
//! truth — and the engine's kNN must agree with the exact index by
//! construction (they share the brute-force path today; this pins it).
//!
//! The corpus mirrors `nql-bench --recall`: deterministic dim-64 vectors
//! (xorshift64* family, seed 42), queries from a stream seeded apart from the
//! corpus, ids `doc:i`. HNSW is seeded (`HnswVectorIndex::default`:
//! seed=42, m=16, ef_construction=200, ef=64), so every number here is
//! reproducible — the thresholds are regression floors, not approximations.
//!
//! Run both ways:
//! ```sh
//! cargo test -p nqlite --test recall                 # exact-path gates
//! cargo test -p nqlite --test recall --features hnsw # + ANN recall gate
//! ```

use std::collections::BTreeSet;

use nql_ir::{Id, RecordId, Store};
use nqlite::{BruteForceVectorIndex, Database, VectorIndex};

const ROWS: usize = 5000;
const DIM: usize = 64;
const QUERIES: usize = 20;
const SEED: u64 = 42;
const K: usize = 10;

/// decisions §6 target, applied to the ANN path.
#[cfg(feature = "hnsw")]
const MIN_HNSW_RECALL_AT_10: f64 = 0.95;

/// xorshift64* — same family as nql-bench and the Python harness.
fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

/// Deterministic (id, vector) set: dim-64 uniform vectors, seed 42.
fn vectors(rows: usize) -> Vec<(RecordId, Vec<f32>)> {
    let mut state = SEED;
    (0..rows)
        .map(|i| {
            let v: Vec<f32> = (0..DIM)
                .map(|_| (xorshift(&mut state) % 10_000) as f32 / 10_000.0)
                .collect();
            (RecordId::new("doc", Id::Num(i as u64)), v)
        })
        .collect()
}

/// Deterministic queries, seeded apart from the corpus stream.
fn queries() -> Vec<Vec<f32>> {
    let mut state = SEED ^ 0xA5A5_A5A5_A5A5_A5A5;
    (0..QUERIES)
        .map(|_| {
            (0..DIM)
                .map(|_| (xorshift(&mut state) % 10_000) as f32 / 10_000.0)
                .collect()
        })
        .collect()
}

fn recall_vs_exact(got: &[RecordId], truth: &[RecordId], k: usize) -> f64 {
    let relevant: BTreeSet<RecordId> = truth[..k].iter().cloned().collect();
    let retrieved: Vec<RecordId> = got[..k.min(got.len())].to_vec();
    nqlite::harness::recall_at_k(&retrieved, &relevant, k)
}

#[test]
fn engine_knn_agrees_with_exact_index() {
    // The engine's kNN and BruteForceVectorIndex share the exact path today;
    // pin the agreement so an index swap (ANN by default) fails loudly here
    // unless recall gates are re-tuned alongside it.
    let vs = vectors(ROWS);
    let mut exact = BruteForceVectorIndex::default();
    for (id, v) in &vs {
        exact.upsert(id.clone(), v.clone());
    }

    let mut db = Database::new(Store::default());
    let ingest: Vec<nql_ir::Statement> = vs
        .iter()
        .map(|(id, v)| {
            nql_ir::Statement::Insert(nql_ir::Record {
                id: id.clone(),
                body: Default::default(),
                embedding: Some(v.clone()),
                created_at: 0,
            })
        })
        .collect();
    db.execute(&ingest).expect("ingest");

    for (qi, q) in queries().iter().enumerate() {
        let truth: Vec<RecordId> = exact
            .search(q, ROWS)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        let sel = nql_ir::Select {
            table: "doc".into(),
            knn: Some(nql_ir::Knn {
                query: q.clone(),
                k: K,
            }),
            ..Default::default()
        };
        let res = db
            .execute(&[nql_ir::Statement::Select(sel)])
            .expect("knn select");
        let got: Vec<RecordId> = res[0].rows.iter().map(|r| r.record.id.clone()).collect();
        let recall = recall_vs_exact(&got, &truth, K);
        assert!(
            (recall - 1.0).abs() < 1e-9,
            "query {qi}: engine kNN must equal the exact top-{K}, got recall {recall}"
        );
    }
}

#[cfg(feature = "hnsw")]
#[test]
fn hnsw_recall_at_10_meets_target() {
    let vs = vectors(ROWS);
    let mut exact = BruteForceVectorIndex::default();
    for (id, v) in &vs {
        exact.upsert(id.clone(), v.clone());
    }
    let mut hnsw = nqlite::HnswVectorIndex::default();
    for (id, v) in &vs {
        hnsw.upsert(id.clone(), v.clone());
    }
    assert_eq!(hnsw.len(), ROWS, "all vectors live in the ANN index");

    let qs = queries();
    // Search capped at 100 (NOT ROWS): fast-hnsw widens `ef` to max(ef, k),
    // so k=ROWS would silently turn the ANN gate into a near-exact run.
    // k_max=100 with the default ef=64 matches the bench `--recall` config.
    const KMAX: usize = 100;
    let mut sum = 0.0;
    for q in &qs {
        let truth: Vec<RecordId> = exact
            .search(q, KMAX)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        let got: Vec<RecordId> = hnsw.search(q, KMAX).into_iter().map(|(id, _)| id).collect();
        sum += recall_vs_exact(&got, &truth, K);
    }
    let recall = sum / qs.len() as f64;
    eprintln!(
        "hnsw recall@{K} over {} queries, rows={ROWS}: {recall}",
        qs.len()
    );
    assert!(
        recall >= MIN_HNSW_RECALL_AT_10,
        "HNSW recall@{K} = {recall:.4} below the decisions §6 target {MIN_HNSW_RECALL_AT_10}"
    );
}
