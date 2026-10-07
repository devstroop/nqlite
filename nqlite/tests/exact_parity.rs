//! Track B parity gate: the exact scan must stay **bit-identical** to the
//! reference triple-loop `cosine_similarity` algorithm — same ids, same score
//! bits — whatever internal change comes next (top-k selection, layout,
//! SIMD: see the spike findings issue). A determinism move is a failure here.
//!
//! Cases covered: dim-64 random corpus, zero vector (na = 0), zero query
//! (nb = 0), and a length-mismatched entry (shared-function fallback path).

use std::cmp::Ordering;
use std::collections::BTreeMap;

use nql_ir::{Id, RecordId};
use nqlite::engine::cosine_similarity;
use nqlite::{BruteForceVectorIndex, VectorIndex};

/// Deterministic xorshift64* (same family as the benches).
struct Rng(u64);

impl Rng {
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

/// The pre-track-B algorithm, verbatim: score every pair with the shared
/// function, sort by score desc / id asc, truncate.
fn reference_search(
    store: &BTreeMap<RecordId, Vec<f32>>,
    query: &[f32],
    k: usize,
) -> Vec<(RecordId, f32)> {
    let mut scored: Vec<(RecordId, f32)> = store
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

#[test]
fn norm_cached_scan_is_bit_identical_to_reference() {
    const DIM: usize = 64;
    const N: usize = 2048;
    let mut rng = Rng(0x5EED_5AFE);

    let mut reference: BTreeMap<RecordId, Vec<f32>> = BTreeMap::new();
    let mut index = BruteForceVectorIndex::default();
    for i in 0..N {
        let v: Vec<f32> = (0..DIM).map(|_| rng.next_f32()).collect();
        let id = RecordId::new("doc", Id::Num(i as u64));
        reference.insert(id.clone(), v.clone());
        index.upsert(id, v);
    }
    // Edge cases: all-zero stored vector (na == 0) and a short one
    // (length mismatch → shared-function fallback path).
    let zero_id = RecordId::new("doc", Id::Str("zero".into()));
    let short_id = RecordId::new("doc", Id::Str("short".into()));
    let zero = vec![0.0f32; DIM];
    let short = vec![0.5f32, -0.25, 0.125];
    reference.insert(zero_id.clone(), zero.clone());
    reference.insert(short_id.clone(), short.clone());
    index.upsert(zero_id, zero);
    index.upsert(short_id, short);

    let queries: Vec<Vec<f32>> =
        std::iter::once(vec![0.0f32; DIM]) // nb == 0
            .chain((0..4).map(|_| (0..DIM).map(|_| rng.next_f32()).collect()))
            .chain(std::iter::once(vec![1.0f32, 1.0, 1.0, 1.0])) // short query
            .collect();

    for (qi, query) in queries.iter().enumerate() {
        let want = reference_search(&reference, query, 10);
        let got = index.search(query, 10);
        assert_eq!(want.len(), got.len(), "query {qi}: result count differs");
        for ((want_id, want_score), (got_id, got_score)) in want.iter().zip(&got) {
            assert_eq!(want_id, got_id, "query {qi}: id order differs");
            assert_eq!(
                want_score.to_bits(),
                got_score.to_bits(),
                "query {qi}: score bits differ for {want_id:?} \
                 ({want_score:?} vs {got_score:?})"
            );
        }
    }
}

#[test]
fn determinism_across_repeated_searches() {
    let mut rng = Rng(0xC0FF_EE11);
    let mut index = BruteForceVectorIndex::default();
    for i in 0..512u64 {
        let v: Vec<f32> = (0..16).map(|_| rng.next_f32()).collect();
        index.upsert(RecordId::new("r", Id::Num(i)), v);
    }
    let query: Vec<f32> = (0..16).map(|_| rng.next_f32()).collect();
    let a = index.search(&query, 7);
    let b = index.search(&query, 7);
    assert_eq!(
        a.iter()
            .map(|(id, s)| (id.clone(), s.to_bits()))
            .collect::<Vec<_>>(),
        b.iter()
            .map(|(id, s)| (id.clone(), s.to_bits()))
            .collect::<Vec<_>>(),
        "identical index + query must yield identical bytes"
    );
}
