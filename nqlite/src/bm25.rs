//! Deterministic BM25 lexical retrieval over record text fields.
//!
//! This module backs the `Filter::Bm25` SELECT operator: a classic Okapi BM25
//! ranking over the tokens of one `Value::Str` body field per record. It is a
//! pure, offline, fully deterministic pipeline:
//!
//! - Tokenization is a fixed lowercase + split-on-non-alphanumeric pass
//!   ([`tokenize`]) — no stemming, no stop-word lists, no randomness.
//! - All counts live in `BTreeMap`s keyed by [`RecordId`]/token, so scans and
//!   aggregations iterate in stable order.
//! - [`bm25_score`] is pure arithmetic over those counts; the summation order
//!   over query terms is the caller-supplied token order, so a fixed input
//!   yields a byte-identical score every time.
//!
//! The engine builds one [`Bm25Index`] per SELECT (from the filtered candidate
//! records), scores each row, orders by descending score with ascending
//! [`RecordId`] tie-break, and applies the row cap — the same determinism
//! contract as the rest of [`crate::engine`].

use std::collections::BTreeMap;

use nql_ir::{Record, RecordId, Value};

/// BM25 term-frequency saturation parameter (standard Okapi default, see
/// [`bm25_score`]). Higher values increase the influence of term frequency;
/// `1.2` is the value tuned on the original TREC test collections.
pub const K1: f32 = 1.2;

/// BM25 document-length normalization parameter (standard Okapi default, see
/// [`bm25_score`]). `0.75` is the classic value: full normalization is applied
/// at `b = 1.0`, none at `b = 0.0`.
pub const B: f32 = 0.75;

/// Split `text` into deterministic lowercase tokens.
///
/// Every run of alphanumeric characters becomes one token, lowercased via
/// `to_lowercase()`; everything else (whitespace, punctuation, Unicode
/// symbols) is a separator. This is a pure function of `text`, so the same
/// input always yields the same token vector. Note: this is deliberately a
/// *naive* tokenizer — no stemming and no stop-word removal — because both
/// would add state or external tables and break the engine's zero-dependency,
/// fully deterministic guarantee.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|tok| !tok.is_empty())
        .map(|tok| tok.to_lowercase())
        .collect()
}

/// Okapi BM25 relevance of one document against one query.
///
/// ```text
/// score(D, Q) = Σ_{q ∈ Q} idf(q) · tf(q, D) · (k1 + 1)
///                                     ─────────────────────────
///                               tf(q, D) + k1 · (1 − b + b · |D| / avgdl)
/// ```
///
/// with the Robertson–Sparck Jones inverse document frequency, plus-one
/// smoothed so it is always non-negative (the Lucene-style variant):
///
/// ```text
/// idf(q) = ln(1 + (N − df(q) + 0.5) / (df(q) + 0.5))
/// ```
///
/// where `N` = number of documents in the collection, `df(q)` = number of
/// documents containing `q`, `tf(q, D)` = raw term frequency of `q` in the
/// document, `|D|` = document length in tokens, `avgdl` = mean document
/// length, and `k1`/`b` are the standard saturation and length-normalization
/// constants ([`K1`], [`B`]).
///
/// Terms absent from the document contribute `0`; `idf` is bounded below by
/// `0`, so no term can subtract from the score. The sum iterates
/// `query_tokens` in caller order (skipping duplicates), which keeps the
/// result a deterministic function of its arguments. All counts are `f32`
/// casts of `u32`/`usize` values, so the arithmetic is exact up to f32
/// rounding — identical inputs produce identical output.
#[allow(clippy::too_many_arguments)]
pub fn bm25_score(
    doc_tokens: &[String],
    query_tokens: &[String],
    df: &BTreeMap<String, u32>,
    n_docs: u32,
    avgdl: f32,
    doc_len: f32,
    k1: f32,
    b: f32,
) -> f32 {
    if n_docs == 0 || doc_len == 0.0 || query_tokens.is_empty() {
        return 0.0;
    }
    let n = n_docs as f32;
    let tf = |tok: &str| doc_tokens.iter().filter(|t| *t == tok).count() as f32;
    let mut score = 0.0f32;
    let mut seen: BTreeMap<&str, ()> = BTreeMap::new();
    for tok in query_tokens {
        // Sum each distinct query term exactly once; `seen` is only used for
        // membership, so the summation order stays `query_tokens` order.
        if seen.insert(tok.as_str(), ()).is_some() {
            continue;
        }
        let f = tf(tok);
        if f == 0.0 {
            continue;
        }
        let df_tok = df.get(tok).copied().unwrap_or(0).max(1) as f32;
        let idf = (1.0 + (n - df_tok + 0.5) / (df_tok + 0.5)).ln();
        let len_norm = 1.0 - b + b * (doc_len / avgdl.max(1.0));
        score += idf * (f * (k1 + 1.0)) / (f + k1 * len_norm);
    }
    score
}

/// A deterministic BM25 index over one text field of a table's records.
///
/// Built from the records of a single table (see [`Bm25Index::new`]); only
/// body values that are `Value::Str` are indexed — records whose field is
/// missing or of any other type simply have no entry and score `0.0`. All
/// storage is `BTreeMap`-backed, so [`Bm25Index::score`] is a pure function
/// of the index contents and the query tokens.
#[derive(Debug, Clone, PartialEq)]
pub struct Bm25Index {
    /// The body field this index was built over.
    field: String,
    /// Token counts per indexed record, keyed by `RecordId` then token.
    doc_tokens: BTreeMap<RecordId, BTreeMap<String, u32>>,
    /// Total number of indexed documents (`n_docs` for [`bm25_score`]).
    n_docs: u32,
    /// Total tokens across all indexed documents (drives `avgdl`).
    total_tokens: u64,
    /// Document frequency per token: how many indexed documents contain it.
    df: BTreeMap<String, u32>,
}

impl Bm25Index {
    /// Build the index over `field` for the given records.
    ///
    /// A record contributes to the index iff its body holds `field` as a
    /// `Value::Str`. The token counts are aggregated in `BTreeMap` order, so
    /// the resulting index is a deterministic function of the input records.
    pub fn new<'a>(field: &str, records: impl Iterator<Item = &'a Record>) -> Self {
        let mut doc_tokens: BTreeMap<RecordId, BTreeMap<String, u32>> = BTreeMap::new();
        let mut df: BTreeMap<String, u32> = BTreeMap::new();
        let mut n_docs = 0u32;
        let mut total_tokens = 0u64;

        for rec in records {
            let Some(Value::Str(text)) = rec.body.get(field) else {
                continue;
            };
            let mut counts: BTreeMap<String, u32> = BTreeMap::new();
            for tok in tokenize(text) {
                *counts.entry(tok).or_insert(0) += 1;
            }
            if counts.is_empty() {
                // An empty/whitespace-only string is a document of length 0:
                // it participates in `n_docs`/`avgdl` but matches nothing.
                n_docs += 1;
                doc_tokens.insert(rec.id.clone(), counts);
                continue;
            }
            for (tok, cnt) in &counts {
                *df.entry(tok.clone()).or_insert(0) += 1;
                total_tokens += *cnt as u64;
            }
            n_docs += 1;
            doc_tokens.insert(rec.id.clone(), counts);
        }

        Self {
            field: field.to_string(),
            doc_tokens,
            n_docs,
            total_tokens,
            df,
        }
    }

    /// BM25 score of the record `id` against `query_tokens`.
    ///
    /// Records that are not in the index (missing field, non-`Str` field, or
    /// not a record of the table the index was built from) score `0.0`.
    pub fn score(&self, id: &RecordId, query_tokens: &[String]) -> f32 {
        let Some(counts) = self.doc_tokens.get(id) else {
            return 0.0;
        };
        if counts.is_empty() || query_tokens.is_empty() {
            return 0.0;
        }
        // Reconstruct the per-document token vector (BTreeMap order) so the
        // shared `bm25_score` does the arithmetic exactly once.
        let doc_tokens: Vec<String> = counts
            .iter()
            .flat_map(|(tok, cnt)| std::iter::repeat_n(tok.clone(), *cnt as usize))
            .collect();
        let doc_len = doc_tokens.len() as f32;
        let avgdl = self.total_tokens as f32 / self.n_docs.max(1) as f32;
        bm25_score(
            &doc_tokens,
            query_tokens,
            &self.df,
            self.n_docs,
            avgdl,
            doc_len,
            K1,
            B,
        )
    }

    /// The field this index was built over.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Number of indexed documents (records whose field is `Value::Str`).
    pub fn len(&self) -> usize {
        self.doc_tokens.len()
    }

    /// Whether the index holds no documents.
    pub fn is_empty(&self) -> bool {
        self.doc_tokens.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use nql_ir::{Id, Record, RecordId, Value};

    use super::*;

    fn rec(id: &str, body: BTreeMap<String, Value>) -> Record {
        Record {
            id: RecordId::parse(id).unwrap(),
            body,
            embedding: None,
            created_at: 0,
        }
    }

    fn str_(s: &str) -> Value {
        Value::Str(s.to_string())
    }

    fn index_over(records: Vec<Record>) -> Bm25Index {
        Bm25Index::new("text", records.iter())
    }

    #[test]
    fn tokenizer_is_deterministic_and_normalizes() {
        let a = tokenize("Hello, WORLD!  123  hello-world #tag");
        let b = tokenize("Hello, WORLD!  123  hello-world #tag");
        assert_eq!(a, b, "tokenize must be a pure function of its input");
        assert_eq!(a, ["hello", "world", "123", "hello", "world", "tag"]);
        // Unicode alphanumerics survive; symbols split.
        assert_eq!(tokenize("café-au-lait 42"), ["café", "au", "lait", "42"]);
        // Empty and separator-only input tokenize to nothing.
        assert!(tokenize("").is_empty());
        assert!(tokenize("  !!!  ,,, ").is_empty());
    }

    #[test]
    fn bm25_ranks_term_dense_doc_above_sparse_doc() {
        let dense = rec(
            "doc:dense",
            BTreeMap::from([("text".into(), str_("rust rust rust rust rust"))]),
        );
        let sparse = rec(
            "doc:sparse",
            BTreeMap::from([("text".into(), str_("rust is a systems language"))]),
        );
        let idx = index_over(vec![dense, sparse]);
        let q = tokenize("rust");
        let dense_score = idx.score(&RecordId::parse("doc:dense").unwrap(), &q);
        let sparse_score = idx.score(&RecordId::parse("doc:sparse").unwrap(), &q);
        assert!(
            dense_score > sparse_score,
            "term-dense doc ({dense_score}) must outrank sparse doc ({sparse_score})"
        );
        assert!(dense_score > 0.0 && sparse_score > 0.0);
    }

    #[test]
    fn bm25_absent_term_scores_zero_and_idf_never_negative() {
        let a = rec("doc:a", BTreeMap::from([("text".into(), str_("alpha"))]));
        let b = rec("doc:b", BTreeMap::from([("text".into(), str_("beta"))]));
        let idx = index_over(vec![a, b]);
        let q = tokenize("gamma");
        assert_eq!(idx.score(&RecordId::parse("doc:a").unwrap(), &q), 0.0);
        assert_eq!(idx.score(&RecordId::parse("doc:b").unwrap(), &q), 0.0);
        // Direct formula check: idf stays >= 0 even when a term is in every doc.
        let df = BTreeMap::from([("term".to_string(), 2u32)]);
        let score = bm25_score(
            &["term".to_string()],
            &["term".to_string()],
            &df,
            2,
            1.0,
            1.0,
            K1,
            B,
        );
        assert!(score >= 0.0, "idf must never make a score negative");
    }

    #[test]
    fn empty_query_scores_zero() {
        let a = rec(
            "doc:a",
            BTreeMap::from([("text".into(), str_("hello world"))]),
        );
        let b = rec("doc:b", BTreeMap::from([("text".into(), str_("hello"))]));
        let idx = index_over(vec![a, b]);
        assert_eq!(idx.score(&RecordId::parse("doc:a").unwrap(), &[]), 0.0);
        assert_eq!(idx.score(&RecordId::parse("doc:b").unwrap(), &[]), 0.0);
    }

    #[test]
    fn missing_or_non_str_field_scores_zero_and_is_unindexed() {
        let missing = rec(
            "doc:missing",
            BTreeMap::from([("other".into(), str_("hello"))]),
        );
        let non_str = rec(
            "doc:nonstr",
            BTreeMap::from([("text".into(), Value::Int(42))]),
        );
        let empty = rec("doc:empty", BTreeMap::from([("text".into(), str_(""))]));
        let present = rec(
            "doc:present",
            BTreeMap::from([("text".into(), str_("hello"))]),
        );
        let idx = index_over(vec![missing, non_str, empty, present]);
        // Only the two `Value::Str` records are indexed.
        assert_eq!(idx.len(), 2);
        for id in ["doc:missing", "doc:nonstr", "doc:empty"] {
            assert_eq!(
                idx.score(&RecordId::parse(id).unwrap(), &tokenize("hello")),
                0.0,
                "{id} must score 0"
            );
        }
        assert!(idx.score(&RecordId::parse("doc:present").unwrap(), &tokenize("hello")) > 0.0);
    }

    #[test]
    fn index_build_and_scores_are_deterministic() {
        let records = vec![
            rec(
                "doc:1",
                BTreeMap::from([("text".into(), str_("the quick brown fox"))]),
            ),
            rec(
                "doc:2",
                BTreeMap::from([("text".into(), str_("lazy dog fox"))]),
            ),
        ];
        let a = index_over(records.clone());
        let b = index_over(records);
        assert_eq!(a, b);
        let q = tokenize("fox dog");
        for id in ["doc:1", "doc:2"] {
            let rid = RecordId::parse(id).unwrap();
            assert_eq!(a.score(&rid, &q), b.score(&rid, &q));
        }
        // Freshly built index over the same records scores identically too.
        let c = index_over(vec![
            rec(
                "doc:1",
                BTreeMap::from([("text".into(), str_("the quick brown fox"))]),
            ),
            rec(
                "doc:2",
                BTreeMap::from([("text".into(), str_("lazy dog fox"))]),
            ),
        ]);
        assert_eq!(a, c);
    }

    #[test]
    fn id_numeric_variant_works_in_index() {
        // Guard: RecordId::Num keys participate in the index like any other.
        let r = Record {
            id: RecordId::new("doc", Id::Num(7)),
            body: BTreeMap::from([("text".into(), str_("unique needle"))]),
            embedding: None,
            created_at: 0,
        };
        let idx = index_over(vec![r]);
        let rid = RecordId::new("doc", Id::Num(7));
        assert!(idx.score(&rid, &tokenize("needle")) > 0.0);
    }
}
