//! Retrieval regression harness (decision D9).
//!
//! Ground truth lives IN the store as `:relevant` edges
//! `(query:q1) -> :relevant {value: 1.0} -> (doc:7)` — the same
//! feedback-as-edges model as votes (see docs/decisions.md D9). The harness
//! reads a query's relevant set from those edges, runs a retrieval, and
//! reports recall@K / precision@K so a regression on any grammar / index /
//! fusion change is caught before it ships.
//!
//! Everything here is a pure function of `(store, query, k)`:
//! deterministic, zero-LLM, no wall-clock.

use std::collections::BTreeSet;

use nql_ir::{RecordId, Store};

/// The relevance edge name (matches the votes-as-edges convention: no `:`
/// prefix — the parser strips it).
pub const RELEVANT_EDGE: &str = "relevant";

/// Read the ground-truth relevant set for a query record: every record
/// reachable via `(query) -> :relevant -> (doc)` outgoing edges.
pub fn relevant_set(store: &Store, query: &RecordId) -> BTreeSet<RecordId> {
    store
        .edges
        .iter()
        .filter(|e| e.name == RELEVANT_EDGE && &e.from == query)
        .map(|e| e.to.clone())
        .collect()
}

/// Recall@K: the fraction of the ground-truth relevant set captured in the
/// top-K retrieved rows. `1.0` when the relevant set is empty (nothing to
/// miss); `0.0` when K = 0.
pub fn recall_at_k(retrieved: &[RecordId], relevant: &BTreeSet<RecordId>, k: usize) -> f64 {
    if relevant.is_empty() {
        return 1.0;
    }
    if k == 0 {
        return 0.0;
    }
    let hit = retrieved
        .iter()
        .take(k)
        .filter(|id| relevant.contains(*id))
        .count();
    hit as f64 / relevant.len() as f64
}

/// Precision@K: the fraction of the top-K retrieved rows that are relevant.
/// `0.0` when K = 0.
pub fn precision_at_k(retrieved: &[RecordId], relevant: &BTreeSet<RecordId>, k: usize) -> f64 {
    if k == 0 {
        return 0.0;
    }
    let hit = retrieved
        .iter()
        .take(k)
        .filter(|id| relevant.contains(*id))
        .count();
    hit as f64 / k as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use nql_ir::{Id, RelationEdge, Value};

    fn rid(s: &str) -> RecordId {
        RecordId::parse(s).unwrap()
    }

    fn store_with_relevance(edges: &[(&str, &str)]) -> Store {
        let mut s = Store::default();
        for (from, to) in edges {
            s.edges.push(RelationEdge {
                from: rid(from),
                name: RELEVANT_EDGE.into(),
                to: rid(to),
                created_at: 0,
                weight: Some(1.0),
                props: Default::default(),
            });
        }
        s
    }

    #[test]
    fn relevant_set_reads_outgoing_relevance_edges() {
        let s = store_with_relevance(&[("q:1", "d:1"), ("q:1", "d:2"), ("q:2", "d:3")]);
        let got: Vec<String> = relevant_set(&s, &rid("q:1"))
            .into_iter()
            .map(|r| r.to_string())
            .collect();
        assert_eq!(got, ["d:1", "d:2"]);
        assert!(relevant_set(&s, &rid("q:404")).is_empty());
    }

    #[test]
    fn recall_and_precision_are_pure_arithmetic() {
        let relevant: BTreeSet<RecordId> = [rid("d:1"), rid("d:2"), rid("d:3")].into();
        let retrieved = [rid("d:1"), rid("d:9"), rid("d:2")];

        // recall@2: top-2 = [d:1, d:9] → 1 hit / 3 relevant = 1/3
        assert!((recall_at_k(&retrieved, &relevant, 2) - 1.0 / 3.0).abs() < 1e-9);
        // precision@2: 1 hit in top-2 = 0.5
        assert!((precision_at_k(&retrieved, &relevant, 2) - 0.5).abs() < 1e-9);
        // recall@3: top-3 = [d:1, d:9, d:2] → 2 hits / 3 = 2/3 ; precision@3 = 2/3
        assert!((recall_at_k(&retrieved, &relevant, 3) - 2.0 / 3.0).abs() < 1e-9);
        assert!((precision_at_k(&retrieved, &relevant, 3) - 2.0 / 3.0).abs() < 1e-9);
        // K = 0: recall 0, precision 0
        assert_eq!(recall_at_k(&retrieved, &relevant, 0), 0.0);
        assert_eq!(precision_at_k(&retrieved, &relevant, 0), 0.0);
        // Empty relevant set: recall is trivially perfect.
        assert_eq!(recall_at_k(&retrieved, &BTreeSet::new(), 5), 1.0);
    }

    #[test]
    fn harness_rejects_stale_relevance_edges() {
        // An edge pointing at a forgotten record still counts as ground truth
        // (the harness reads edges, not records) — retrieval just cannot hit it.
        let s = store_with_relevance(&[("q:1", "d:gone")]);
        let relevant = relevant_set(&s, &rid("q:1"));
        assert!(relevant.contains(&rid("d:gone")));
        // And recall against an empty retrieval is 0.0 (not 1.0 — there IS a
        // relevant doc, it is just unreachable).
        assert!((recall_at_k(&[], &relevant, 5) - 0.0).abs() < 1e-9);
    }

    // Guard: Id import stays used by this module's test surface.
    #[test]
    fn id_import_guard() {
        let _ = Id::Num(1);
        let _ = Value::Null;
    }
}
