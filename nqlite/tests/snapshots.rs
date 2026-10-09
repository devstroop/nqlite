//! End-to-end replay-snapshot coverage (issue #166): composition of the
//! recent-T fast path, the in-session snapshot ring, `PRUNE HISTORY`, and
//! the `HistoryPruned` horizon through the public [`Database`] API.
//!
//! The ring itself is a pure cache — results are identical with or without
//! it — so this test pins observable behavior across 24k mutations (past
//! one `SNAPSHOT_EVERY` epoch, so capture engages): exact row counts at
//! pre/post-prune cutoffs, the loud horizon below the compaction snapshot,
//! and determinism across two identically-built databases. Ring mechanics
//! (select/evict/regression) and base-vs-full view equality live in the
//! engine unit tests (`snapshot_tests`), where histories stay small.

use std::collections::BTreeMap;

use nqlite::{Database, Id, Record, RecordId, Statement, Store, Value};

const N_HALF: u64 = 12_000;

fn setup() -> (Database, i64, i64) {
    let mut db = Database::new(Store::default());
    db.execute(&[Statement::CreateTable {
        table: "doc".into(),
        vector_dim: None,
    }])
    .unwrap();
    for chunk in 0..12 {
        let mut plan = Vec::with_capacity(1000);
        for i in 0..1000 {
            let id = chunk * 1000 + i;
            plan.push(Statement::Insert(Record {
                id: RecordId::new("doc", Id::Num(id)),
                body: BTreeMap::from([("group".into(), Value::Int((id % 3) as i64))]),
                embedding: None,
                created_at: 0,
            }));
        }
        db.execute(&plan).unwrap();
    }
    // Two temporal reads: the second one arms snapshot capture (single-read
    // sessions pay no clone tax).
    let count = |db: &mut Database, ts: i64| -> i64 {
        let res = db
            .execute(&[Statement::Select(nqlite::Select {
                table: "doc".into(),
                as_of: Some(ts),
                aggregate: Some(nqlite::Aggregate::CountStar),
                ..nqlite::Select::default()
            })])
            .unwrap();
        count_of(&res[0])
    };
    let mid = db.store().clock / 2;
    // Inserts are stamped from ts 2 (the CREATE takes ts 1): `mid - 1` rows
    // at or below `mid`.
    assert_eq!(count(&mut db, mid), mid - 1);
    assert_eq!(count(&mut db, mid), mid - 1);
    db.execute(&[Statement::PruneHistory]).unwrap();
    let horizon = db.store().clock;
    for chunk in 0..12 {
        let mut plan = Vec::with_capacity(1000);
        for i in 0..1000 {
            let id = N_HALF + chunk * 1000 + i;
            plan.push(Statement::Insert(Record {
                id: RecordId::new("doc", Id::Num(id)),
                body: BTreeMap::from([("group".into(), Value::Int((id % 3) as i64))]),
                embedding: None,
                created_at: 0,
            }));
        }
        db.execute(&plan).unwrap();
    }
    (db, horizon, mid)
}

fn count_of(res: &nqlite::QueryResult) -> i64 {
    match res.rows.first().map(|r| r.record.body.get("count")) {
        Some(Some(Value::Int(n))) => *n,
        other => panic!("expected {{\"count\": n}} row, got {other:?}"),
    }
}

#[test]
fn as_of_views_are_exact_across_prune_and_snapshot_epochs() {
    let (mut db, horizon, mid) = setup();
    let max = db.store().clock;

    // Below the compaction horizon: loud failure, not a partial view.
    let err = db
        .execute(&[Statement::Select(nqlite::Select {
            table: "doc".into(),
            as_of: Some(mid),
            aggregate: Some(nqlite::Aggregate::CountStar),
            ..nqlite::Select::default()
        })])
        .unwrap_err();
    assert!(
        err.to_string().contains("PRUNE HISTORY"),
        "expected HistoryPruned, got {err}"
    );

    // At the horizon: the full pre-prune state survives compaction.
    let at = |db: &mut Database, ts: i64| -> i64 {
        let res = db
            .execute(&[Statement::Select(nqlite::Select {
                table: "doc".into(),
                as_of: Some(ts),
                aggregate: Some(nqlite::Aggregate::CountStar),
                ..nqlite::Select::default()
            })])
            .unwrap();
        count_of(&res[0])
    };
    assert_eq!(at(&mut db, horizon), N_HALF as i64);
    // Mid second epoch: pre-prune rows plus post-prune inserts so far.
    assert_eq!(
        at(&mut db, horizon + N_HALF as i64 / 2),
        N_HALF as i64 + N_HALF as i64 / 2
    );
    // Recent-T (at and past the clock): the fast path — current state.
    assert_eq!(at(&mut db, max), 2 * N_HALF as i64);
    assert_eq!(at(&mut db, max + 10_000), 2 * N_HALF as i64);
}

#[test]
fn as_of_is_deterministic_across_identical_builds() {
    let (mut a, horizon, _) = setup();
    let (mut b, _, _) = setup();
    for ts in [horizon, horizon + 100, a.store().clock] {
        let ra = a
            .execute(&[Statement::Select(nqlite::Select {
                table: "doc".into(),
                as_of: Some(ts),
                aggregate: Some(nqlite::Aggregate::CountStar),
                ..nqlite::Select::default()
            })])
            .unwrap();
        let rb = b
            .execute(&[Statement::Select(nqlite::Select {
                table: "doc".into(),
                as_of: Some(ts),
                aggregate: Some(nqlite::Aggregate::CountStar),
                ..nqlite::Select::default()
            })])
            .unwrap();
        assert_eq!(ra, rb, "as_of {ts} diverged between identical builds");
    }
}
