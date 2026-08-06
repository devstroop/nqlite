//! nql-bench: deterministic benchmark runner for the cross-DB harness.
//!
//! Generates the SAME fixed-seed synthetic corpus as the Python harness
//! (`scripts/bench-compare/corpus.py` — xorshift64*, seed 42, dim-8 vectors,
//! 2-word bodies from a fixed vocabulary), loads it into an in-memory nqlite
//! database, times ingest + kNN + BM25 + hybrid queries, and prints a JSON
//! report to stdout.
//!
//! Usage:
//! ```sh
//! cargo run -q -p nql-bench -- --rows 1000 --knn 100 --seed 42
//! ```
//!
//! Output: `{"db":"nqlite","rows":N,"ingest_ms":...,"knn_ms":...,"bm25_ms":...,"hybrid_ms":...}`

use std::collections::BTreeMap;
use std::time::Instant;

use nql_ir::{Filter, Id, Knn, Record, RecordId, Select, Statement, Store, Value};
use nqlite::Database;

const DIM: usize = 8;
const SEED: u64 = 42;

/// xorshift64* — must match `scripts/bench-compare/corpus.py`.
fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

fn corpus(rows: usize) -> Vec<Record> {
    let mut state = SEED;
    let vocab = [
        "alpha", "beta", "gamma", "delta", "rust", "query", "memory", "graph",
    ];
    (0..rows)
        .map(|i| {
            let mut emb = vec![0.0f32; DIM];
            for e in emb.iter_mut() {
                *e = (xorshift(&mut state) % 1000) as f32 / 1000.0;
            }
            let w1 = vocab[(xorshift(&mut state) as usize) % vocab.len()];
            let w2 = vocab[(xorshift(&mut state) as usize) % vocab.len()];
            Record {
                id: RecordId::new("doc", Id::Num(i as u64)),
                body: BTreeMap::from([
                    ("text".into(), Value::Str(format!("{w1} {w2}"))),
                    ("group".into(), Value::Int((i % 10) as i64)),
                ]),
                embedding: Some(emb),
                created_at: i as i64,
            }
        })
        .collect()
}

fn parse_args() -> (usize, usize, u64) {
    let args: Vec<String> = std::env::args().collect();
    let mut rows = 1000;
    let mut knn = 100;
    let mut seed = SEED;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--rows" => {
                i += 1;
                rows = args[i].parse().expect("--rows <n>");
            }
            "--knn" => {
                i += 1;
                knn = args[i].parse().expect("--knn <n>");
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse().expect("--seed <n>");
            }
            other => panic!("unknown flag `{other}` (--rows --knn --seed)"),
        }
        i += 1;
    }
    (rows, knn, seed)
}

fn main() {
    let (rows, knn, _seed) = parse_args();

    let mut db = Database::new(Store::default());
    let records = corpus(rows);

    // Ingest: one plan of N inserts.
    let t0 = Instant::now();
    let plan: Vec<Statement> = records
        .iter()
        .map(|r| Statement::Insert(r.clone()))
        .collect();
    db.execute(&plan).expect("ingest");
    let ingest_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Query vector: fixed, derived from the corpus seed (deterministic).
    let mut qstate = SEED;
    let qv: Vec<f32> = (0..DIM)
        .map(|_| (xorshift(&mut qstate) % 1000) as f32 / 1000.0)
        .collect();

    // kNN: repeated selects, report total.
    let knn_stmt = Statement::Select(Select {
        table: "doc".into(),
        knn: Some(Knn {
            query: qv.clone(),
            k: 10,
        }),
        filter: None,
        order: None,
        limit: None,
    });
    let t1 = Instant::now();
    for _ in 0..knn {
        db.execute(std::slice::from_ref(&knn_stmt)).expect("knn");
    }
    let knn_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // BM25: repeated lexical selects.
    let bm25_stmt = Statement::Select(Select {
        table: "doc".into(),
        knn: None,
        filter: Some(Filter::Bm25 {
            field: "text".into(),
            query: "rust query".into(),
            k: None,
        }),
        order: None,
        limit: None,
    });
    let t2 = Instant::now();
    for _ in 0..knn {
        db.execute(std::slice::from_ref(&bm25_stmt)).expect("bm25");
    }
    let bm25_ms = t2.elapsed().as_secs_f64() * 1000.0;

    // Hybrid: BM25 + kNN fused.
    let hybrid_stmt = Statement::Select(Select {
        table: "doc".into(),
        knn: Some(Knn {
            query: qv.clone(),
            k: 10,
        }),
        filter: Some(Filter::Bm25 {
            field: "text".into(),
            query: "rust query".into(),
            k: None,
        }),
        order: None,
        limit: None,
    });
    let t3 = Instant::now();
    for _ in 0..knn {
        db.execute(std::slice::from_ref(&hybrid_stmt))
            .expect("hybrid");
    }
    let hybrid_ms = t3.elapsed().as_secs_f64() * 1000.0;

    let report = serde_json::json!({
        "db": "nqlite",
        "rows": rows,
        "knn_iterations": knn,
        "ingest_ms": round2(ingest_ms),
        "knn_ms": round2(knn_ms),
        "bm25_ms": round2(bm25_ms),
        "hybrid_ms": round2(hybrid_ms),
    });
    println!("{report}");
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_is_deterministic_across_calls() {
        // Same seed → same records, same order (the cross-DB harness relies on
        // this to make nql-bench and the Python generator see identical data).
        let a = corpus(64);
        let b = corpus(64);
        assert_eq!(a.len(), b.len());
        for (ra, rb) in a.iter().zip(b.iter()) {
            assert_eq!(ra.id, rb.id);
            assert_eq!(ra.body, rb.body);
            assert_eq!(ra.embedding, rb.embedding);
        }
    }

    #[test]
    fn corpus_dim_and_vocab_match_harness_contract() {
        let docs = corpus(16);
        assert_eq!(DIM, 8);
        for d in &docs {
            assert_eq!(d.embedding.as_ref().unwrap().len(), DIM);
            if let Some(Value::Str(text)) = d.body.get("text") {
                let words: Vec<&str> = text.split(' ').collect();
                assert_eq!(words.len(), 2, "two-word bodies: {text}");
            } else {
                panic!("missing text body");
            }
        }
    }
}
