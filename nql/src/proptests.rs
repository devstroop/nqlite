//! Property tests for the nql parser (parser hardening, PLAN M1).
//!
//! Properties (all plain `#[test]` functions so CI picks them up — no extra
//! harness needed):
//!   * `parse` / `parse_statement` NEVER panic on arbitrary input.
//!   * Round-trip determinism: re-parsing the exact same program yields
//!     byte-identical IR (`PartialEq`/`Debug` equality on `Plan`).
//!   * Multi-statement programs keep statement count and per-statement shape.
//!   * Every `Err` carries a sane, non-empty position (`line >= 1`, `col >= 1`).
//!
//! Sizes are kept small (default 256 cases, bounded inputs) so the suite stays
//! fast in CI.

use crate::parser::{parse, parse_statement, NqlError};
use nql_ir::Statement;
use proptest::collection::vec;
use proptest::prelude::*;

/// Keywords the parser treats specially. Generated idents avoid these entirely
/// so every synthetic program is unambiguous and well-formed by construction.
const RESERVED: &[&str] = &[
    "create",
    "table",
    "insert",
    "into",
    "embed",
    "relate",
    "set",
    "select",
    "from",
    "where",
    "order",
    "by",
    "limit",
    "forget",
    "not",
    "null",
    "is",
    "and",
    "vector",
    "similarity",
    "weight",
    "embedding",
    "true",
    "false",
];

fn ident() -> impl Strategy<Value = String> {
    "[a-z]{1,8}".prop_filter("ident must not be a keyword", |s| {
        !RESERVED.contains(&s.as_str())
    })
}

/// A table name: distinct from plain idents only for readability in failing
/// counterexamples; same grammar as `ident()`.
fn table() -> impl Strategy<Value = String> {
    ident()
}

/// A numeric literal token as source text: a finite float or a small integer.
/// Formatting is stable, so re-parsing is bit-for-bit deterministic.
fn number() -> impl Strategy<Value = String> {
    prop_oneof![
        (-999i64..1000).prop_map(|n| n.to_string()),
        (-999i64..1000).prop_map(|n| format!("{}", n as f32 / 8.0)),
    ]
}

/// A quoted string literal without embedded quotes/escapes (simple, safe).
fn string_lit() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9_ ]{0,6}".prop_map(|s| format!("\"{s}\""))
}

/// Record id suffix: a non-negative number or a bare word.
fn id_suffix() -> impl Strategy<Value = String> {
    prop_oneof![(0i64..10_000i64).prop_map(|n| n.to_string()), ident()]
}

/// A JSON-ish scalar value in source text (used in bodies, SET, WHERE, ...).
fn scalar() -> impl Strategy<Value = String> {
    prop_oneof![
        number(),
        string_lit(),
        Just("true".to_string()),
        Just("false".to_string()),
        Just("null".to_string()),
        ident(),
    ]
}

/// A small non-negative integer for LIMIT / k / vector dimension (always >= 0).
fn non_neg_int() -> impl Strategy<Value = i64> {
    0i64..1000
}

// ---------------------------------------------------------------------------
// Statement generators. Each yields `(StmtKind, sql_text)` so shape assertions
// can check the parse output against what was produced.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum StmtKind {
    Create,
    Insert,
    Relate,
    Match,
    Select,
    Forget,
}

fn create_stmt() -> impl Strategy<Value = (StmtKind, String)> {
    (table(), prop::bool::ANY, 1i64..4096).prop_map(|(tbl, with_vec, dim)| {
        let mut sql = format!("CREATE TABLE {tbl}");
        if with_vec {
            // VECTOR<f32, N> with N > 0 (parser rejects N == 0).
            sql.push_str(&format!(" VECTOR<f32, {dim}>"));
        }
        (StmtKind::Create, sql)
    })
}

fn object_body() -> impl Strategy<Value = String> {
    // A `{ k: v, ... }` record body with 0..4 entries.
    vec((ident(), scalar()), 0..4).prop_map(|entries| {
        let inner = entries
            .into_iter()
            .map(|(k, v)| format!("{k}: {v}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!("{{ {inner} }}")
    })
}

fn insert_stmt() -> impl Strategy<Value = (StmtKind, String)> {
    (
        table(),
        id_suffix(),
        object_body(),
        prop::bool::ANY,
        vec(number(), 0..4),
    )
        .prop_map(|(tbl, suffix, body, with_embed, floats)| {
            let mut sql = format!("INSERT INTO {tbl}:{suffix} {body}");
            if with_embed {
                sql.push_str(&format!(" EMBED [{}]", floats.join(", ")));
            }
            (StmtKind::Insert, sql)
        })
}

fn relate_stmt() -> impl Strategy<Value = (StmtKind, String)> {
    (
        table(),
        id_suffix(),
        ident(),
        table(),
        id_suffix(),
        prop::bool::ANY,
        vec((ident(), scalar()), 0..3),
    )
        .prop_map(|(t1, s1, edge, t2, s2, with_set, props)| {
            let mut sql = format!("RELATE ({t1}:{s1}) -> :{edge} -> ({t2}:{s2})");
            if with_set && !props.is_empty() {
                let set = props
                    .into_iter()
                    .map(|(k, v)| format!("{k} = {v}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                sql.push_str(&format!(" SET {set}"));
            }
            (StmtKind::Relate, sql)
        })
}

fn select_stmt() -> impl Strategy<Value = (StmtKind, String)> {
    // WHERE shape: none | field-eq | IS NOT NULL | kNN.
    let where_clause = prop_oneof![
        Just(None),
        (ident(), scalar()).prop_map(|(f, v)| Some(format!("WHERE {f} = {v}"))),
        Just(Some("WHERE embedding IS NOT NULL".to_string())),
        (vec(number(), 1..4), 1i64..1000).prop_map(|(floats, k)| {
            Some(format!(
                "WHERE vector::similarity(embedding, [{}]) AND k = {k}",
                floats.join(", "),
            ))
        }),
    ];
    let order_key = prop_oneof![
        Just("similarity"),
        Just("salience"),
        Just("score"),
        Just("recency"),
    ];
    (
        prop_oneof![
            Just("*".to_string()),
            vec(ident(), 1..3).prop_map(|fs| fs.join(", "))
        ],
        table(),
        where_clause,
        prop::bool::ANY,
        order_key,
        prop::bool::ANY,
        non_neg_int(),
    )
        .prop_map(|(fields, tbl, w, with_order, key, with_limit, limit)| {
            let mut sql = format!("SELECT {fields} FROM {tbl}");
            if let Some(w) = w {
                sql.push(' ');
                sql.push_str(&w);
            }
            if with_order {
                sql.push_str(&format!(" ORDER BY ::{key}"));
            }
            if with_limit {
                sql.push_str(&format!(" LIMIT {limit}"));
            }
            (StmtKind::Select, sql)
        })
}

fn forget_stmt() -> impl Strategy<Value = (StmtKind, String)> {
    (table(), id_suffix())
        .prop_map(|(tbl, suffix)| (StmtKind::Forget, format!("FORGET {tbl}:{suffix}")))
}

fn match_stmt() -> impl Strategy<Value = (StmtKind, String)> {
    // One or more hops: `MATCH (t:s) -> :e <- :e ...`.
    (table(), id_suffix(), vec(ident(), 1..3), prop::bool::ANY).prop_map(
        |(tbl, suffix, edges, mix_directions)| {
            let hops = edges
                .into_iter()
                .map(|e| {
                    let arrow = if mix_directions && e.len() % 2 == 0 {
                        "<-"
                    } else {
                        "->"
                    };
                    format!("{arrow} :{e}")
                })
                .collect::<Vec<_>>()
                .join(" ");
            (StmtKind::Match, format!("MATCH ({tbl}:{suffix}) {hops}"))
        },
    )
}

/// A single well-formed statement, plus its kind.
fn stmt() -> impl Strategy<Value = (StmtKind, String)> {
    prop_oneof![
        create_stmt(),
        insert_stmt(),
        relate_stmt(),
        match_stmt(),
        select_stmt(),
        forget_stmt(),
    ]
}

/// A whole program: 0..5 statements joined with `;` (statement separators).
fn program() -> impl Strategy<Value = (Vec<StmtKind>, String)> {
    vec(stmt(), 0..5).prop_map(|stmts| {
        let kinds = stmts.iter().map(|(k, _)| *k).collect();
        let sql = stmts
            .into_iter()
            .map(|(_, s)| s)
            .collect::<Vec<_>>()
            .join("; ");
        (kinds, sql)
    })
}

fn kind_of(stmt: &Statement) -> StmtKind {
    match stmt {
        Statement::CreateTable { .. } => StmtKind::Create,
        Statement::Insert(_) => StmtKind::Insert,
        Statement::Relate(_) => StmtKind::Relate,
        Statement::Match(_) => StmtKind::Match,
        Statement::Select(_) => StmtKind::Select,
        Statement::Forget { .. } => StmtKind::Forget,
    }
}

fn err_message(err: &NqlError) -> &str {
    match err {
        NqlError::Lex { message, .. } | NqlError::Parse { message, .. } => message,
    }
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

proptest! {
    /// Any printable-ish input is either `Ok` or `Err` — never a panic.
    /// Bounded length keeps pathological nesting (deep `{` recursion) out of
    /// the CI property run; the cargo-fuzz harness covers unbounded input.
    #[test]
    fn parse_never_panics_on_arbitrary_text(s in "\\PC{0,64}") {
        let _ = parse(&s);
        let _ = parse_statement(&s);
    }
}

proptest! {
    /// The same input always produces the same IR: `parse` is a pure function.
    #[test]
    fn parse_is_deterministic(program in program()) {
        let (_, sql) = program;
        let plan = parse(&sql).expect("generated program must parse");
        assert_eq!(plan, parse(&sql).expect("re-parse must succeed"));
    }
}

proptest! {
    /// Generated programs parse, keep their statement count, and preserve the
    /// kind of every statement in order (`;`-separated multi-statement input).
    #[test]
    fn multi_statement_count_and_shape((kinds, sql) in program()) {
        let plan = parse(&sql).expect("generated program must parse");
        prop_assert_eq!(plan.len(), kinds.len(), "statement count mismatch");
        for (got, want) in plan.iter().zip(&kinds) {
            prop_assert_eq!(kind_of(got), *want, "statement kind mismatch");
        }
    }
}

proptest! {
    /// Every parse error carries a non-empty message and a sane 1-based
    /// position — never (0, 0) and never an empty message.
    #[test]
    fn errors_report_sane_positions(s in "\\PC{0,64}") {
        if let Err(err) = parse(&s) {
            prop_assert!(!err_message(&err).is_empty(), "error message must not be empty");
            prop_assert!(err.line() >= 1, "line must be >= 1, got {}", err.line());
            prop_assert!(err.col() >= 1, "col must be >= 1, got {}", err.col());
        }
        if let Err(err) = parse_statement(&s) {
            prop_assert!(!err_message(&err).is_empty(), "error message must not be empty");
            prop_assert!(err.line() >= 1, "line must be >= 1, got {}", err.line());
            prop_assert!(err.col() >= 1, "col must be >= 1, got {}", err.col());
        }
    }
}
