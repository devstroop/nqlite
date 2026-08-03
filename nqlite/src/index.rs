//! Vector index abstraction over record embeddings.
//!
//! The engine executes kNN SELECTs through the [`VectorIndex`] trait rather
//! than an inline cosine scan, so the exact-retrieval strategy can be swapped
//! without touching `engine.rs` or the public API. The default — and the only
//! implementation compiled into the default build — is
//! [`BruteForceVectorIndex`]: an exact, fully deterministic scan over a
//! `BTreeMap` keyed by [`RecordId`].
//!
//! # Determinism
//!
//! [`BruteForceVectorIndex`] is a pure function of its contents: vectors live
//! in a `BTreeMap<RecordId, Vec<f32>>`, similarity is the same
//! [`cosine_similarity`] the engine always used, and ties are broken by
//! ascending [`RecordId`]. The same sequence of upserts/removes therefore
//! yields byte-identical search output every time.
//!
//! # Swap point
//!
//! `HnswVectorIndex` (feature `hnsw`, opt-in) is an approximate alternative
//! for large collections. It is **not** the default: approximate search can
//! miss true nearest neighbours, so exact brute-force remains the engine's
//! deterministic default. See the crate docs for the swap mechanics.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use nql_ir::RecordId;

use crate::engine::cosine_similarity;

/// A searchable index over record embeddings.
///
/// Implementations decide their own storage and retrieval strategy; the
/// engine only relies on the contract below. Exact implementations must
/// return results ordered by descending cosine similarity with ties broken
/// by ascending [`RecordId`]; approximate implementations should at least
/// return a deterministic ranking for a given index state.
pub trait VectorIndex {
    /// Insert `v` for `id`, replacing any previous vector for the same id.
    fn upsert(&mut self, id: RecordId, v: Vec<f32>);

    /// Remove `id` from the index. Removing an absent id is a no-op.
    fn remove(&mut self, id: &RecordId);

    /// Return the `k` nearest neighbours of `query` as `(id, score)` pairs,
    /// where `score` is the cosine similarity against `query` (higher is
    /// better). Fewer than `k` results are returned when the index holds
    /// fewer live vectors.
    fn search(&self, query: &[f32], k: usize) -> Vec<(RecordId, f32)>;

    /// Number of live (non-removed) vectors in the index.
    fn len(&self) -> usize;

    /// Whether the index holds no live vectors.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Exact brute-force vector index — the deterministic default.
///
/// Vectors are stored in a [`BTreeMap`] keyed by [`RecordId`], so iteration
/// order is stable. `search` scores every vector with the shared
/// [`cosine_similarity`], sorts by score descending (ties by `RecordId`
/// ascending — the same total order the engine applies to query results),
/// and truncates to `k`.
#[derive(Debug, Clone, Default)]
pub struct BruteForceVectorIndex {
    vectors: BTreeMap<RecordId, Vec<f32>>,
}

impl VectorIndex for BruteForceVectorIndex {
    fn upsert(&mut self, id: RecordId, v: Vec<f32>) {
        self.vectors.insert(id, v);
    }

    fn remove(&mut self, id: &RecordId) {
        self.vectors.remove(id);
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(RecordId, f32)> {
        let mut scored: Vec<(RecordId, f32)> = self
            .vectors
            .iter()
            .map(|(id, v)| (id.clone(), cosine_similarity(v, query)))
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(k);
        scored
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }
}

/// Approximate HNSW vector index (feature `hnsw`, opt-in).
///
/// Wraps [`fast_hnsw`](https://docs.rs/fast-hnsw) — a pure-Rust HNSW
/// implementation — over cosine distance, keyed by the record id's string
/// form. The graph is built with a fixed seed and a deterministic upsert
/// order, so for a given sequence of operations the index state — and hence
/// its search output — is reproducible.
///
/// # Approximate, not exact
///
/// HNSW is a greedy approximate search: it can miss true nearest neighbours
/// and its ranking can differ from [`BruteForceVectorIndex`]. It exists for
/// large collections where an exact scan is too slow; the engine's default
/// remains exact brute-force. Do not rely on it when exact results or
/// byte-identical determinism are required.
///
/// # Caveats
///
/// - `upsert` on an existing id cannot evict the old entry from the HNSW
///   graph (the library has no deletion); the stale entry is tombstoned and
///   filtered out of results while a fresh entry is inserted. The index may
///   therefore grow beyond `len()` live vectors.
/// - `remove` tombstones the id; the underlying graph entry stays in memory.
#[cfg(feature = "hnsw")]
pub struct HnswVectorIndex {
    index: fast_hnsw::labeled::LabeledIndex<fast_hnsw::distance::Cosine, String>,
    /// Live ids and their current insertion index in the HNSW graph.
    live: BTreeMap<RecordId, usize>,
    /// Ids removed (or superseded by an upsert) since their last insert.
    removed: std::collections::BTreeSet<RecordId>,
    /// Beam width used for searches (`ef >= k`; larger → better recall).
    ef: usize,
}

#[cfg(feature = "hnsw")]
impl HnswVectorIndex {
    /// Build an empty index with the given HNSW parameters.
    ///
    /// `seed` fixes the RNG so identical upsert sequences produce identical
    /// graphs. `m` is the max bidirectional links per layer, `ef_construction`
    /// the insert-time beam width, and `ef` the search-time beam width.
    pub fn new(seed: u64, m: usize, ef_construction: usize, ef: usize) -> Self {
        let index = fast_hnsw::Builder::new()
            .m(m)
            .ef_construction(ef_construction)
            .seed(seed)
            .build_labeled(fast_hnsw::distance::Cosine);
        Self {
            index,
            live: BTreeMap::new(),
            removed: std::collections::BTreeSet::new(),
            ef: ef.max(1),
        }
    }

    /// Live ids (excluding removed entries), in stable `RecordId` order.
    fn live_ids(&self) -> impl Iterator<Item = &RecordId> {
        self.live.keys().filter(|id| !self.removed.contains(*id))
    }
}

#[cfg(feature = "hnsw")]
impl Default for HnswVectorIndex {
    fn default() -> Self {
        Self::new(42, 16, 200, 64)
    }
}

#[cfg(feature = "hnsw")]
impl VectorIndex for HnswVectorIndex {
    fn upsert(&mut self, id: RecordId, v: Vec<f32>) {
        // The HNSW graph is append-only: a re-insert of an existing id gets a
        // fresh insertion index and the old entry is tombstoned.
        if self.live.contains_key(&id) {
            self.removed.insert(id.clone());
        }
        let inserted_at = self.index.insert(v, id.to_string());
        self.removed.remove(&id);
        self.live.insert(id, inserted_at);
    }

    fn remove(&mut self, id: &RecordId) {
        self.removed.insert(id.clone());
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(RecordId, f32)> {
        if k == 0 || self.live_ids().next().is_none() {
            return Vec::new();
        }
        let mut scored: Vec<(RecordId, f32)> = self
            .index
            .search(query, k, self.ef)
            .into_iter()
            .filter(|r| !self.removed.contains(&id_from_payload(r.payload)))
            .map(|r| {
                let id = id_from_payload(r.payload);
                // fast-hnsw cosine distance is 1 − cosine similarity.
                (id, 1.0 - r.distance)
            })
            .collect();
        // Restore the trait contract (similarity desc, RecordId asc tie-break)
        // and cap at `k` after filtering tombstones.
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(k);
        scored
    }

    fn len(&self) -> usize {
        self.live_ids().count()
    }
}

/// Parse a `RecordId` back from its string form (the payload we store).
#[cfg(feature = "hnsw")]
fn id_from_payload(payload: &str) -> RecordId {
    RecordId::parse(payload).expect("stored payload is always a valid RecordId string")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::cosine_similarity;

    fn rid(s: &str) -> RecordId {
        RecordId::parse(s).unwrap()
    }

    /// Hand-rolled exact ranking for the known vectors below.
    fn manual_top(vectors: &[(&str, &[f32])], query: &[f32], k: usize) -> Vec<String> {
        let mut scored: Vec<(String, f32)> = vectors
            .iter()
            .map(|(id, v)| (id.to_string(), cosine_similarity(v, query)))
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(k);
        scored.into_iter().map(|(id, _)| id).collect()
    }

    const KNOWN: &[(&str, &[f32])] = &[
        ("vec:a", &[1.0, 0.0]),
        ("vec:b", &[0.0, 1.0]),
        ("vec:c", &[0.70710677, 0.70710677]),
        ("vec:d", &[-1.0, 0.0]),
    ];

    #[test]
    fn brute_force_matches_manual_cosine_ranking() {
        let mut idx = BruteForceVectorIndex::default();
        for (id, v) in KNOWN {
            idx.upsert(rid(id), v.to_vec());
        }
        for query in [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [-1.0, 0.0]] {
            for k in 1..=4 {
                let got: Vec<String> = idx
                    .search(&query, k)
                    .into_iter()
                    .map(|(id, _)| id.to_string())
                    .collect();
                let want = manual_top(KNOWN, &query, k);
                assert_eq!(got, want, "query={query:?} k={k}");
            }
        }
    }

    #[test]
    fn brute_force_returns_correct_similarity_scores() {
        let mut idx = BruteForceVectorIndex::default();
        idx.upsert(rid("vec:a"), vec![1.0, 0.0]);
        idx.upsert(rid("vec:b"), vec![0.0, 1.0]);
        let results = idx.search(&[1.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert!((results[0].1 - 1.0).abs() < 1e-6);
        assert!(results[0].1 > results[1].1);
        assert!((results[1].1 - 0.0).abs() < 1e-6);
        // Ties break by RecordId ascending.
        idx.upsert(rid("vec:c"), vec![1.0, 0.0]);
        let results = idx.search(&[1.0, 0.0], 3);
        assert_eq!(results[0].0, rid("vec:a"));
        assert_eq!(results[1].0, rid("vec:c"));
    }

    #[test]
    fn brute_force_deterministic_across_rebuilds() {
        let build = || {
            let mut idx = BruteForceVectorIndex::default();
            for (id, v) in KNOWN {
                idx.upsert(rid(id), v.to_vec());
            }
            // Same ops twice must yield identical search output.
            for (id, v) in KNOWN.iter().rev() {
                idx.upsert(rid(id), v.to_vec());
            }
            idx.search(&[0.3, 0.9], 4)
        };
        let a = build();
        let b = build();
        assert_eq!(a, b);
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
    }

    #[test]
    fn remove_drops_id_from_results() {
        let mut idx = BruteForceVectorIndex::default();
        for (id, v) in KNOWN {
            idx.upsert(rid(id), v.to_vec());
        }
        assert_eq!(idx.len(), 4);
        idx.remove(&rid("vec:a"));
        assert_eq!(idx.len(), 3);
        let ids: Vec<String> = idx
            .search(&[1.0, 0.0], 4)
            .into_iter()
            .map(|(id, _)| id.to_string())
            .collect();
        assert!(!ids.contains(&"vec:a".to_string()));
        // Removing an absent id is a no-op.
        idx.remove(&rid("vec:nope"));
        assert_eq!(idx.len(), 3);
        // Re-upsert brings the id back.
        idx.upsert(rid("vec:a"), vec![1.0, 0.0]);
        assert_eq!(idx.len(), 4);
        assert_eq!(idx.search(&[1.0, 0.0], 1)[0].0, rid("vec:a"));
    }

    #[test]
    fn empty_index_returns_nothing() {
        let idx = BruteForceVectorIndex::default();
        assert!(idx.is_empty());
        assert_eq!(idx.search(&[1.0, 0.0], 5), Vec::new());
    }

    #[cfg(feature = "hnsw")]
    mod hnsw {
        use super::*;

        #[test]
        fn hnsw_top1_agrees_with_brute_force_on_tiny_set() {
            // Well-separated vectors: HNSW with a generous ef must recover the
            // exact top-1 on a tiny, low-dimensional set.
            let mut brute = BruteForceVectorIndex::default();
            let mut hnsw = HnswVectorIndex::new(7, 16, 200, 128);
            for (id, v) in KNOWN {
                brute.upsert(rid(id), v.to_vec());
                hnsw.upsert(rid(id), v.to_vec());
            }
            for query in [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [-1.0, 0.0], [0.9, 0.4]] {
                let want = brute.search(&query, 1)[0].0.clone();
                let got = hnsw.search(&query, 1)[0].0.clone();
                assert_eq!(got, want, "query={query:?}");
            }
        }

        #[test]
        fn hnsw_is_reproducible_for_same_ops() {
            let build = || {
                let mut idx = HnswVectorIndex::new(11, 16, 200, 128);
                for (id, v) in KNOWN {
                    idx.upsert(rid(id), v.to_vec());
                }
                idx.search(&[0.3, 0.9], 4)
            };
            assert_eq!(build(), build());
        }

        #[test]
        fn hnsw_remove_tombstones_id() {
            let mut idx = HnswVectorIndex::new(3, 16, 200, 128);
            for (id, v) in KNOWN {
                idx.upsert(rid(id), v.to_vec());
            }
            assert_eq!(idx.len(), 4);
            idx.remove(&rid("vec:a"));
            assert_eq!(idx.len(), 3);
            let ids: Vec<String> = idx
                .search(&[1.0, 0.0], 4)
                .into_iter()
                .map(|(id, _)| id.to_string())
                .collect();
            assert!(!ids.contains(&"vec:a".to_string()));
        }
    }
}
