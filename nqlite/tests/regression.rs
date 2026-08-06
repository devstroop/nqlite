//! Retrieval regression harness (D9): ground-truth relevance lives in the
//! store as `:relevant` edges; the harness runs real retrievals and asserts
//! recall@K / precision@K floors so any grammar / index / fusion regression
//! is caught.
//!
//! The corpus is synthetic but deterministic: fixed embeddings, fixed bodies,
//! fixed relevance graph. No rand, no wall-clock, no LLM.

use std::collections::BTreeMap;

use nql::parse;
use nql_ir::{RecordId, Store, Value};
use nqlite::{Database, QueryKind};

/// Deterministic synthetic corpus: 12 docs in 3 topic clusters (alpha, beta,
/// gamma). Ground truth: each topic's query is relevant to its 4 docs.
fn build_corpus(db: &mut Database) {
    // topic, id, embedding (one-hot-ish, deterministic), text
    let docs: &[(&str, u64, [f32; 6], &str)] = &[
        ("alpha", 1, [1.0, 0.0, 0.0, 0.1, 0.0, 0.0], "alpha one"),
        ("alpha", 2, [0.9, 0.1, 0.0, 0.0, 0.0, 0.0], "alpha two"),
        ("alpha", 3, [0.8, 0.2, 0.0, 0.0, 0.0, 0.0], "alpha three"),
        ("alpha", 4, [0.95, 0.05, 0.0, 0.0, 0.0, 0.0], "alpha four"),
        ("beta", 5, [0.0, 0.0, 1.0, 0.0, 0.1, 0.0], "beta one"),
        ("beta", 6, [0.0, 0.0, 0.9, 0.0, 0.0, 0.0], "beta two"),
        ("beta", 7, [0.0, 0.0, 0.8, 0.0, 0.0, 0.0], "beta three"),
        ("beta", 8, [0.0, 0.0, 0.95, 0.0, 0.0, 0.0], "beta four"),
        ("gamma", 9, [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], "gamma one"),
        ("gamma", 10, [0.0, 0.0, 0.0, 0.0, 0.9, 0.0], "gamma two"),
        ("gamma", 11, [0.0, 0.0, 0.0, 0.0, 0.8, 0.0], "gamma three"),
        ("gamma", 12, [0.0, 0.0, 0.0, 0.0, 0.95, 0.0], "gamma four"),
    ];

    let mut plan = vec![nql_ir::Statement::CreateTable {
        table: "doc".into(),
        vector_dim: Some(6),
    }];
    for (topic, id, emb, text) in docs {
        plan.push(nql_ir::Statement::Insert(nql_ir::Record {
            id: RecordId::parse(&format!("doc:{id}")).unwrap(),
            body: BTreeMap::from([
                ("topic".into(), Value::Str((*topic).into())),
                ("text".into(), Value::Str((*text).into())),
            ]),
            embedding: Some(emb.to_vec()),
            created_at: 0,
        }));
    }
    // Ground truth: `(query:<topic>) -> :relevant -> (doc:<n>)` for each topic.
    for (i, (topic, _, _, _)) in docs.iter().enumerate() {
        if i % 4 == 0 {
            // one query record per topic cluster (ids 1, 5, 9 carry the edges)
            for n in 1..=4 {
                let doc_id = ((i / 4) * 4 + n) as u64;
                plan.push(nql_ir::Statement::Relate(nql_ir::RelationEdge {
                    from: RecordId::parse(&format!("query:{topic}")).unwrap(),
                    name: "relevant".into(),
                    to: RecordId::parse(&format!("doc:{doc_id}")).unwrap(),
                    created_at: 0,
                    weight: Some(1.0),
                    props: Default::default(),
                }));
            }
        }
    }
    db.execute(&plan).expect("corpus plan executes");
}

fn run_knn(db: &mut Database, query: [f32; 6], k: usize) -> Vec<RecordId> {
    let sel = nql_ir::Select {
        table: "doc".into(),
        knn: Some(nql_ir::Knn {
            query: query.to_vec(),
            k,
        }),
        filter: None,
        order: Some(nql_ir::Order::Similarity),
        limit: None,
    };
    let res = db
        .execute(&[nql_ir::Statement::Select(sel)])
        .expect("kNN executes");
    res[0].rows.iter().map(|r| r.record.id.clone()).collect()
}

fn run_bm25(db: &mut Database, term: &str, k: usize) -> Vec<RecordId> {
    let plan = parse(&format!(
        "SELECT * FROM doc WHERE ::bm25(text, \"{term}\") AND k = {k}"
    ))
    .expect("bm25 parses");
    let res = db.execute(&plan).expect("bm25 executes");
    res[0].rows.iter().map(|r| r.record.id.clone()).collect()
}

#[test]
fn knn_recall_floor_holds_on_synthetic_corpus() {
    let mut db = Database::new(Store::default());
    build_corpus(&mut db);

    // For each topic, kNN with the topic's prototype vector must retrieve the
    // 4 relevant docs within the top-K (recall@4 >= 1.0, precision@4 >= 0.8).
    for (topic, vec) in [
        ("alpha", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("beta", [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ("gamma", [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    ] {
        let query = RecordId::parse(&format!("query:{topic}")).unwrap();
        // Copy ground truth out first so `db` can be borrowed mutably below.
        let relevant = nqlite::harness::relevant_set(db.store(), &query);
        assert_eq!(relevant.len(), 4, "{topic} has 4 relevant docs");

        let retrieved = run_knn(&mut db, vec, 4);
        let rec = nqlite::harness::recall_at_k(&retrieved, &relevant, 4);
        let prec = nqlite::harness::precision_at_k(&retrieved, &relevant, 4);
        assert!(
            (rec - 1.0).abs() < 1e-9,
            "{topic} recall@4 = {rec} — kNN regression?"
        );
        assert!(
            prec >= 0.8,
            "{topic} precision@4 = {prec} — kNN regression?"
        );
    }
}

#[test]
fn bm25_recall_floor_holds_on_synthetic_corpus() {
    let mut db = Database::new(Store::default());
    build_corpus(&mut db);

    // BM25 on the shared token "alpha" must surface the alpha docs first.
    let relevant: std::collections::BTreeSet<RecordId> =
        nqlite::harness::relevant_set(db.store(), &RecordId::parse("query:alpha").unwrap());
    let retrieved = run_bm25(&mut db, "alpha", 4);
    let rec = nqlite::harness::recall_at_k(&retrieved, &relevant, 4);
    let prec = nqlite::harness::precision_at_k(&retrieved, &relevant, 4);
    assert!(rec >= 0.75, "alpha recall@4 = {rec} — BM25 regression?");
    assert!(
        prec >= 0.75,
        "alpha precision@4 = {prec} — BM25 regression?"
    );
}

#[test]
fn harness_works_over_nql_edges() {
    // Ground truth written through the nql grammar (RELATE ... :relevant)
    // must be read back by the harness: the no-colon convention matches.
    let mut db = Database::new(Store::default());
    db.execute(
        &parse(
            "CREATE TABLE doc;
             INSERT INTO doc:1 { \"t\": \"a\" };
             INSERT INTO doc:2 { \"t\": \"b\" };
             RELATE (query:q1) -> :relevant -> (doc:1) SET weight = 1.0;",
        )
        .unwrap(),
    )
    .unwrap();
    let relevant = nqlite::harness::relevant_set(db.store(), &RecordId::parse("query:q1").unwrap());
    let ids: Vec<String> = relevant.into_iter().map(|r| r.to_string()).collect();
    assert_eq!(ids, ["doc:1"]);
}

#[test]
fn results_carry_query_kind_for_disambiguation() {
    // The regression harness relies on QueryKind to pick the right result
    // shape; guard that SELECT results are identifiable.
    let mut db = Database::new(Store::default());
    build_corpus(&mut db);
    let sel = nql_ir::Select {
        table: "doc".into(),
        knn: None,
        filter: None,
        order: None,
        limit: Some(1),
    };
    let res = db
        .execute(&[nql_ir::Statement::Select(sel)])
        .expect("select executes");
    assert!(matches!(res[0].kind, QueryKind::Select(_)));
}
