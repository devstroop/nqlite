//! Agent pattern #2 — retrieval-augmented generation loop.
//!
//! Mini RAG: ingest a few short documents with hardcoded BYO embeddings
//! (dim 3), run a kNN query, take the top-2 hits, record the retrieval as
//! `(query) -[:retrieved]-> (doc)` edges, then simulate user votes and
//! re-rank the corpus with `ORDER BY ::feedback`.
//!
//! The agent supplies every vector and every vote; the engine just runs the
//! deterministic kNN scan and the fixed feedback formula. Nothing is learned
//! inside the engine.

use std::error::Error;

use nql::parse;
use nql_ir::Store;
use nqlite::Database;

fn main() -> Result<(), Box<dyn Error>> {
    let mut db = Database::new(Store::default());

    // One plan: schema + ingest + retrieval + provenance + votes + re-rank.
    // Each SELECT contributes one QueryResult, in statement order.
    let results = db.execute(&parse(
        r#"
        CREATE TABLE doc VECTOR<f32, 3>;
        CREATE TABLE query;
        CREATE TABLE user;

        INSERT INTO doc:1 { "title": "nql overview",
                            "text": "nql is a neural query language with vector kNN" }
            EMBED [1.0, 0.0, 0.0];
        INSERT INTO doc:2 { "title": "sqlite storage",
                            "text": "sqlite is an embedded relational database" }
            EMBED [0.0, 1.0, 0.0];
        INSERT INTO doc:3 { "title": "vector search",
                            "text": "vector search ranks candidates by cosine similarity" }
            EMBED [0.9, 0.1, 0.0];
        INSERT INTO doc:4 { "title": "birthday party",
                            "text": "birthday parties need cake and candles" }
            EMBED [0.0, 0.0, 1.0];

        INSERT INTO query:1 { "text": "what is a neural query language?" };

        SELECT * FROM doc
            WHERE vector::similarity(embedding, [1.0, 0.0, 0.1]) AND k = 2;

        RELATE (query:1) -> :retrieved -> (doc:1);
        RELATE (query:1) -> :retrieved -> (doc:3);

        RELATE (user:1) -> :voted -> (doc:1) SET value = 1;
        RELATE (user:2) -> :voted -> (doc:1) SET value = 1;
        RELATE (user:3) -> :voted -> (doc:3) SET value = 1;
        RELATE (user:4) -> :voted -> (doc:2) SET value = -1;
        SELECT * FROM doc ORDER BY ::feedback;
        "#,
    )?)?;

    // --- print retrieval ----------------------------------------------------
    let retrieved = &results[0];
    println!("retrieval (kNN top-2, query [1.0, 0.0, 0.1]):");
    for row in &retrieved.rows {
        println!(
            "  {:>6}  similarity={:.4}  {}",
            row.record.id,
            row.score,
            str_field(&row.record.body, "title")
        );
    }
    println!("  -> kept as (query:1) -[:retrieved]-> edges");

    // --- print the provenance edges ----------------------------------------
    let retrieved_edges: Vec<_> = db
        .store()
        .edges
        .iter()
        .filter(|e| e.name == "retrieved")
        .collect();
    println!("retrieval ledger ({} edges):", retrieved_edges.len());
    for e in &retrieved_edges {
        println!("  ({}) -[:retrieved]-> ({})", e.from, e.to);
    }

    // --- print the feedback ranking ----------------------------------------
    let ranked = &results[1];
    println!("\ncorpus re-ranked by ::feedback (up +1, down -1):");
    for row in &ranked.rows {
        println!(
            "  {:>6}  feedback={:+.2}  {}",
            row.record.id,
            row.score,
            str_field(&row.record.body, "title")
        );
    }
    println!(
        "\nnote: feedback re-ranks beyond retrieval — doc:1 stays on top \
         (2 upvotes) while doc:2 sinks below doc:4 after its downvote."
    );

    Ok(())
}

/// The `title` body field as a string.
fn str_field(body: &std::collections::BTreeMap<String, nql_ir::Value>, key: &str) -> String {
    match body.get(key) {
        Some(nql_ir::Value::Str(s)) => s.clone(),
        _ => String::from("?"),
    }
}
