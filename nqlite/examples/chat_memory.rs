//! Agent pattern #1 — chat memory.
//!
//! A simulated agent conversation. Every turn is stored with a BYO embedding
//! (dim 3) and an agent-written `importance` field (the agent-side knob);
//! turns are linked to entity records with `:mentions` edges. At recall time
//! the agent runs a kNN SELECT ordered by `::salience`, so the engine
//! deterministically blends cosine similarity with the agent-supplied
//! importance weights.
//!
//! Everything the agent "learns" lives in the data it writes (embeddings,
//! importance weights, edges). The engine only reads — it never updates
//! weights, never re-embeds, never learns.

use std::collections::BTreeMap;
use std::error::Error;

use nql::parse;
use nql_ir::{RecordId, RelationEdge, Statement, Store, Value};
use nqlite::Database;

/// Simulated conversation turns: `(id, role, text, embedding, importance)`.
/// The importance knob is hand-written by the agent, not inferred by the
/// engine.
const TURNS: &[(&str, &str, &str, [f32; 3], f32)] = &[
    ("1", "user", "What is nql?", [1.0, 0.0, 0.0], 0.8),
    (
        "2",
        "assistant",
        "nql is a neural query language with vector kNN",
        [0.9, 0.1, 0.0],
        0.9,
    ),
    (
        "3",
        "user",
        "Remind me about the birthday plan",
        [0.0, 0.0, 1.0],
        0.3,
    ),
    (
        "4",
        "assistant",
        "Birthday dinner at 7pm on Saturday",
        [0.1, 0.0, 0.9],
        0.4,
    ),
    ("5", "user", "I love vector search", [0.95, 0.0, 0.1], 0.6),
];

fn main() -> Result<(), Box<dyn Error>> {
    let mut db = Database::new(Store::default());

    // --- 1. store the conversation (turns + entities + :mentions edges) ---
    let mut nql = String::from("CREATE TABLE turn VECTOR<f32, 3>; CREATE TABLE entity;\n");
    for (id, role, text, emb, importance) in TURNS {
        nql.push_str(&format!(
            "INSERT INTO turn:{id} {{ \"role\": \"{role}\", \"text\": \"{text}\", \
             \"importance\": {importance} }} EMBED [{}, {}, {}];\n",
            emb[0], emb[1], emb[2]
        ));
    }
    nql.push_str(
        "INSERT INTO entity:nql { \"name\": \"nql\" };\n\
         INSERT INTO entity:birthday { \"name\": \"birthday\" };\n\
         INSERT INTO entity:vector { \"name\": \"vector search\" };\n\
         RELATE (turn:1) -> :mentions -> (entity:nql);\n\
         RELATE (turn:2) -> :mentions -> (entity:nql);\n\
         RELATE (turn:3) -> :mentions -> (entity:birthday);\n\
         RELATE (turn:4) -> :mentions -> (entity:birthday);\n\
         RELATE (turn:5) -> :mentions -> (entity:vector);\n",
    );
    db.execute(&parse(&nql)?)?;
    println!(
        "stored {} turns, {} entities, 5 :mentions edges",
        TURNS.len(),
        3
    );

    // --- 2. agent-side importance knob -------------------------------------
    // The agent judged some turns more important than others. It writes that
    // judgement as a `:voted` edge weight (agent-owned signal) on top of the
    // `importance` body field. The engine's salience formula reads these
    // weights in a fixed blend — it never reweights or learns by itself.
    // (Built via IR directly so the stored edge name keeps the leading `:`,
    // which the salience formula matches; nql text strips the colon.)
    let importance_edges: Vec<Statement> = TURNS
        .iter()
        .map(|(id, _, _, _, importance)| {
            Statement::Relate(RelationEdge {
                from: RecordId::parse("agent:assistant").unwrap(),
                name: ":voted".into(),
                to: RecordId::parse(&format!("turn:{id}")).unwrap(),
                created_at: 0,
                weight: Some(*importance),
                props: BTreeMap::new(),
            })
        })
        .collect();
    db.execute(&importance_edges)?;
    println!("wrote importance knob: 5 agent -> :voted -> turn edges");

    // --- 3. recall: kNN + salience -----------------------------------------
    // Query "what is nql?" -> [1.0, 0.0, 0.0]. ORDER BY ::salience blends
    // 0.7 * similarity + 0.3 * importance (Laplace-smoothed), so a slightly
    // less similar but much more important turn can outrank a closer one.
    let results = db.execute(&parse(
        "SELECT * FROM turn \
         WHERE vector::similarity(embedding, [1.0, 0.0, 0.0]) AND k = 3 \
         ORDER BY ::salience;",
    )?)?;
    let recalled = &results[0];

    println!("\nrecalled context (k = 3, ORDER BY ::salience):");
    let mentions: Vec<_> = db
        .store()
        .edges
        .iter()
        .filter(|e| e.name == "mentions")
        .collect();
    for row in &recalled.rows {
        let importance = match row.record.body.get("importance") {
            Some(Value::Float(f)) => *f,
            _ => 0.0,
        };
        let entities: Vec<String> = mentions
            .iter()
            .filter(|e| e.from == row.record.id)
            .map(|e| e.to.to_string())
            .collect();
        println!(
            "  {:>7}  score={:.4}  importance={importance:.1}  {}  {}",
            row.record.id,
            row.score,
            str_field(&row.record.body, "role"),
            str_field(&row.record.body, "text"),
        );
        println!("             mentions: {entities:?}");
    }
    println!(
        "\nnote: turn:2 outranks turn:1 despite slightly lower similarity \
         (0.994 vs 1.000) — its importance knob (0.9) lifts its salience."
    );

    Ok(())
}

/// The `text` body field as a string (debug-printed quoted).
fn str_field(body: &BTreeMap<String, Value>, key: &str) -> String {
    match body.get(key) {
        Some(Value::Str(s)) => s.clone(),
        _ => String::from("?"),
    }
}
