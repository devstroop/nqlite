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
//! Output: `{"db":"nqlite","rows":N,"ingest_ms":...,"knn_ms":...,"bm25_ms":...,"hybrid_ms":...,"context_ms":...,"recall_ms":...}`
//!
//! The `context_ms`/`recall_ms` fields are the M3 session-recall scenarios:
//! a deterministic `:follows_from` chain of `rows` context records, timed
//! for a single CLOSURE walk (`context_of(record)`) and a full-session
//! recall loop (SELECT all + CLOSURE, `--knn` iterations).

use std::collections::BTreeMap;
use std::time::Instant;

use nql_ir::{
    Filter, Id, Knn, MatchDirection, MatchPath, MatchStep, Record, RecordId, RelationEdge, Select,
    Statement, Store, Value,
};
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

/// Deterministic session corpus: `rows` context records chained by
/// `:follows_from` edges (`context:1 -> context:2 -> ...`), mirroring the
/// agent context-chain pattern (M3: context_of(record) / session recall).
fn chain_statements(rows: usize) -> Vec<Statement> {
    let mut plan: Vec<Statement> = (1..=rows)
        .map(|i| {
            Statement::Insert(Record {
                id: RecordId::new("context", Id::Num(i as u64)),
                body: BTreeMap::from([
                    ("role".into(), Value::Str("assistant".into())),
                    ("text".into(), Value::Str(format!("step {i}"))),
                ]),
                embedding: None,
                created_at: i as i64,
            })
        })
        .collect();
    for i in 1..rows {
        plan.push(Statement::Relate(RelationEdge {
            from: RecordId::new("context", Id::Num(i as u64)),
            name: "follows_from".into(),
            to: RecordId::new("context", Id::Num((i + 1) as u64)),
            created_at: i as i64,
            weight: None,
            props: BTreeMap::new(),
        }));
    }
    plan
}

/// `CLOSURE (context:1) -> :follows_from` — the context_of(record) walk over
/// the whole chain (first-visit order, BFS depth scores).
fn closure_stmt() -> Statement {
    Statement::Closure(MatchPath {
        start: RecordId::new("context", Id::Num(1)),
        steps: vec![MatchStep {
            direction: MatchDirection::Out,
            name: "follows_from".into(),
            edge_props: None,
        }],
    })
}

fn parse_args() -> (usize, usize, u64, bool, usize, usize, usize, usize, usize) {
    let args: Vec<String> = std::env::args().collect();
    let mut rows = 1000;
    let mut knn = 100;
    let mut seed = SEED;
    let mut recall = false;
    let mut queries = 20;
    let mut dim = 64;
    let mut hnsw_m = 16;
    let mut hnsw_efc = 200;
    let mut hnsw_ef = 64;
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
            "--recall" => {
                recall = true;
            }
            "--queries" => {
                i += 1;
                queries = args[i].parse().expect("--queries <n>");
            }
            "--dim" => {
                i += 1;
                dim = args[i].parse().expect("--dim <n>");
            }
            "--hnsw-m" => {
                i += 1;
                hnsw_m = args[i].parse().expect("--hnsw-m <n>");
            }
            "--hnsw-efc" => {
                i += 1;
                hnsw_efc = args[i].parse().expect("--hnsw-efc <n>");
            }
            "--hnsw-ef" => {
                i += 1;
                hnsw_ef = args[i].parse().expect("--hnsw-ef <n>");
            }
            other => panic!(
                "unknown flag `{other}` (--rows --knn --seed --recall --queries --dim --hnsw-m --hnsw-efc --hnsw-ef)"
            ),
        }
        i += 1;
    }
    (
        rows, knn, seed, recall, queries, dim, hnsw_m, hnsw_efc, hnsw_ef,
    )
}

/// Quality mode (issue #96): recall@K of the approximate index against the
/// **exact** brute-force top-k — the standard ANN metric. Ground truth is the
/// exact index itself, so this measures approximation loss, not task quality.
///
/// Recall uses its OWN deterministic vector set (default dim 64 — the timing
/// corpus's dim-8 uniform vectors are trivially exact for HNSW and would
/// report a meaningless 1.0): ids `doc:i`, vectors from the same xorshift64*
/// family, queries from a stream seeded apart from the corpus. Seeded index +
/// seeded data => every number reproduces run-to-run.
// Without the `hnsw` feature the m/efc/ef parameters are unused (nothing
// builds an ANN index), hence the cfg_attr.
#[cfg_attr(not(feature = "hnsw"), allow(unused_variables))]
fn run_recall(rows: usize, queries: usize, dim: usize, m: usize, efc: usize, ef: usize) {
    let mut report = serde_json::json!({
        "db": "nqlite",
        "rows": rows,
        "dim": dim,
        "recall_mode": true,
        "queries": queries,
        // Sanity anchor: exact vs itself is perfect by construction.
        "exact_recall_at_10": 1.0,
    });

    #[cfg(feature = "hnsw")]
    {
        use nqlite::VectorIndex;

        // Deterministic recall corpus: (id, vector) only — no text needed.
        let mut cstate = SEED;
        let vectors: Vec<(RecordId, Vec<f32>)> = (0..rows)
            .map(|i| {
                let v: Vec<f32> = (0..dim)
                    .map(|_| (xorshift(&mut cstate) % 10_000) as f32 / 10_000.0)
                    .collect();
                (RecordId::new("doc", Id::Num(i as u64)), v)
            })
            .collect();

        // Deterministic query stream, seeded apart from the corpus stream.
        let mut qstate = SEED ^ 0xA5A5_A5A5_A5A5_A5A5;
        let qvs: Vec<Vec<f32>> = (0..queries)
            .map(|_| {
                (0..dim)
                    .map(|_| (xorshift(&mut qstate) % 10_000) as f32 / 10_000.0)
                    .collect()
            })
            .collect();

        let ks = [10usize, 50, 100];
        let kmax = rows.min(*ks.iter().max().unwrap());
        let mut exact = nqlite::BruteForceVectorIndex::default();
        for (id, v) in &vectors {
            exact.upsert(id.clone(), v.clone());
        }
        let exact_top: Vec<Vec<RecordId>> = qvs
            .iter()
            .map(|q| {
                exact
                    .search(q, kmax)
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect()
            })
            .collect();

        let mut hnsw = nqlite::HnswVectorIndex::new(SEED, m, efc, ef);
        for (id, v) in &vectors {
            hnsw.upsert(id.clone(), v.clone());
        }

        let mut sums: BTreeMap<usize, f64> = BTreeMap::new();
        for (qi, q) in qvs.iter().enumerate() {
            let got: Vec<RecordId> = hnsw.search(q, kmax).into_iter().map(|(id, _)| id).collect();
            for k in ks {
                if k > kmax {
                    continue;
                }
                let truth: std::collections::HashSet<&RecordId> =
                    exact_top[qi][..k].iter().collect();
                let hit = got[..k.min(got.len())]
                    .iter()
                    .filter(|id| truth.contains(*id))
                    .count();
                *sums.entry(k).or_insert(0.0) += hit as f64 / k as f64;
            }
        }
        report["k_max"] = serde_json::json!(kmax);
        for k in ks {
            if let Some(sum) = sums.get(&k) {
                report[format!("hnsw_recall_at_{k}")] =
                    serde_json::json!(round4(sum / queries as f64));
            }
        }
        report["hnsw_params"] =
            serde_json::json!(format!("seed={SEED} m={m} ef_construction={efc} ef={ef}"));
        report["hnsw_live"] = serde_json::json!(hnsw.len());
    }
    #[cfg(not(feature = "hnsw"))]
    {
        report["hnsw"] = serde_json::json!(
            "disabled — build with `cargo run -p nql-bench --features hnsw -- --recall`"
        );
    }

    println!(
        "{}",
        serde_json::to_string(&report).unwrap_or_else(|_| "{}".into())
    );
}

#[cfg(feature = "hnsw")]
fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

fn main() {
    let (rows, knn, _seed, recall, queries, dim, hnsw_m, hnsw_efc, hnsw_ef) = parse_args();

    // Quality mode (issue #96): print the recall report and exit — the
    // timing scenarios below never run in this mode.
    if recall {
        run_recall(rows, queries, dim, hnsw_m, hnsw_efc, hnsw_ef);
        return;
    }

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
        as_of: None,
        fields: None,
        offset: None,
        aggregate: None,
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
        as_of: None,
        fields: None,
        offset: None,
        aggregate: None,
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
        as_of: None,
        fields: None,
        offset: None,
        aggregate: None,
    });
    let t3 = Instant::now();
    for _ in 0..knn {
        db.execute(std::slice::from_ref(&hybrid_stmt))
            .expect("hybrid");
    }
    let hybrid_ms = t3.elapsed().as_secs_f64() * 1000.0;

    // M3 session patterns: a deterministic context chain. Ingest the chain
    // OUTSIDE the timed sections; time the CLOSURE walk (context_of) and a
    // full-session recall loop (SELECT all + CLOSURE) against it.
    let mut sdb = Database::new(Store::default());
    let chain = chain_statements(rows);
    sdb.execute(&chain).expect("chain ingest");
    let context_select = Statement::Select(Select {
        table: "context".into(),
        knn: None,
        filter: None,
        order: None,
        limit: None,
        as_of: None,
        fields: None,
        offset: None,
        aggregate: None,
    });
    let closure = closure_stmt();

    let t4 = Instant::now();
    let ctx_res = sdb
        .execute(std::slice::from_ref(&closure))
        .expect("closure");
    let context_ms = t4.elapsed().as_secs_f64() * 1000.0;

    let t5 = Instant::now();
    for _ in 0..knn {
        sdb.execute(std::slice::from_ref(&context_select))
            .expect("recall select");
        sdb.execute(std::slice::from_ref(&closure))
            .expect("recall closure");
    }
    let recall_ms = t5.elapsed().as_secs_f64() * 1000.0;

    let report = serde_json::json!({
        "db": "nqlite",
        "rows": rows,
        "knn_iterations": knn,
        "ingest_ms": round2(ingest_ms),
        "knn_ms": round2(knn_ms),
        "bm25_ms": round2(bm25_ms),
        "hybrid_ms": round2(hybrid_ms),
        "context_ms": round2(context_ms),
        "context_records": ctx_res[0].rows.len(),
        "recall_ms": round2(recall_ms),
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

    #[test]
    fn chain_corpus_closure_walks_every_record_once() {
        // The M3 session corpus: a deterministic chain where CLOSURE from the
        // head must visit every record exactly once, in chain order.
        let rows = 64;
        let mut db = Database::new(Store::default());
        db.execute(&chain_statements(rows)).expect("ingest");
        let res = db
            .execute(std::slice::from_ref(&closure_stmt()))
            .expect("closure");
        // One QueryResult for the CLOSURE statement, carrying every chain
        // record in first-visit order.
        assert_eq!(res.len(), 1);
        let visited = &res[0].rows;
        assert_eq!(visited.len(), rows, "closure visits every chain record");
        for (i, r) in visited.iter().enumerate() {
            assert_eq!(
                r.record.id,
                RecordId::new("context", Id::Num((i + 1) as u64))
            );
        }
        // Determinism: a fresh database yields the identical result.
        let mut db2 = Database::new(Store::default());
        db2.execute(&chain_statements(rows)).expect("ingest");
        let res2 = db2
            .execute(std::slice::from_ref(&closure_stmt()))
            .expect("closure");
        assert_eq!(res, res2);
    }
}
