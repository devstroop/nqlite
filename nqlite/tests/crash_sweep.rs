//! Seeded crash-point sweep for WAL/replay (issue #165).
//!
//! Targeted tests cover torn frames, CRC rejection, and clean WAL replay —
//! this sweep explores **crash points** across the plan/checkpoint lifecycle
//! with seeded randomness (the TigerBeetle DST instinct, single-node scale):
//! every run is reproducible from its seed, and a failure prints the seed,
//! the plan index, and the surgery so `seed + git commit` replays it exactly.
//!
//! Crash model (all simulated with file surgery on sandbox copies — the live
//! sequence is never disturbed, and no fault-injection infra is needed):
//!
//! - `truncate_wal(T)`: the WAL cut at byte offset `T` inside (or on the edge
//!   of) the last plan's bytes — crash mid-batch, torn `ContextReset`, torn
//!   frame. Replay must consume complete frames only, so recovery must equal
//!   the re-execution of plans `1..k-1` plus a *statement prefix* of plan `k`
//!   (found by prefix scan; `T == end` must recover the full plan, `T ==
//!   start` must drop it whole).
//! - `lost_rename`: pre-checkpoint main bytes restored after a `flush()`
//!   (the rename lost to a crash — old main + full WAL). Recovery must equal
//!   the full acknowledged state, byte-identically.
//!
//! Oracles (cheap because the engine is deterministic):
//!
//! 1. lockstep: every plan's results AND store are identical between the
//!    durable and an in-memory shadow database (`QueryResult`/`Store` are
//!    `PartialEq`);
//! 2. complete-prefix: every crash probe recovers a statement-prefix state,
//!    verified by from-scratch re-execution;
//! 3. digest-on-reopen: on full recoveries, the plan's `SELECT`s re-run
//!    identically on the reopened database.
//!
//! Running:
//!
//! ```sh
//! cargo test -p nqlite --test crash_sweep          # fixed seeds (CI, <10 s)
//! NQL_CRASH_SEED=42 NQL_CRASH_PLANS=200 \
//!   cargo test -p nqlite --test crash_sweep \
//!     -- --ignored --nocapture                      # random sweep (nightly)
//! NQL_CRASH_SEED=random cargo test -p nqlite --test crash_sweep -- --ignored --nocapture
//! ```
//!
//! `NQL_CRASH_SEED` accepts a `u64` or `random` (nanos seed, printed on
//! failure); `NQL_CRASH_PLANS` sets the plan count (default 200 for the
//! sweep, 14 for the fixed seeds). A failure message carries everything
//! needed to replay it: seed, plan index, surgery, and offsets.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use nqlite::{
    Aggregate, Database, Filter, Id, Knn, Record, RecordId, RelationEdge, Select, Statement, Store,
    Value,
};

/// Fixed CI seeds (fast, reproducible).
const FIXED_SEEDS: &[u64] = &[
    0xC0FF_EE11_1111_1111,
    0xC0FF_EE22_2222_2222,
    0xC0FF_EE33_3333_3333,
    0xC0FF_EE44_4444_4444,
    0xC0FF_EE55_5555_5555,
    0xC0FF_EE66_6666_6666,
    0xC0FF_EE77_7777_7777,
    0xC0FF_EE88_8888_8888,
];
const FIXED_PLANS: usize = 14;

/// Deterministic xorshift64* (same family as the benches).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        // Zero state is stuck; the seeds above are all non-zero, and 0 can
        // only arrive from a wrapping multiply — force a bit defensively.
        if self.0 == 0 {
            self.0 = 0x9E37_79B9_7F4A_7C15;
        }
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }

    fn chance(&mut self, p: f64) -> bool {
        ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64) < p
    }

    fn next_f32(&mut self) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        unit * 2.0 - 1.0
    }
}

static DIR_SEQ: AtomicU64 = AtomicU64::new(0);

fn fresh_dir(tag: &str, seed: u64) -> PathBuf {
    let seq = DIR_SEQ.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "nqlite-crash-{tag}-{seed:x}-{}-{seq}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create crash dir");
    dir
}

fn wal_path(main: &Path) -> PathBuf {
    let mut s = main.as_os_str().to_owned();
    s.push(".wal");
    PathBuf::from(s)
}

fn wal_len(main: &Path) -> u64 {
    fs::metadata(wal_path(main)).map(|m| m.len()).unwrap_or(0)
}

const WORDS: &[&str] = &["alpha", "beta", "gamma", "delta", "rust", "query"];

/// Seeded plan generator over two small tables: `doc` (text body, no
/// vector) and `vec` (dim-4 embeddings + text). Tracks live ids so RELATE
/// and FORGET always hit real records.
struct Gen {
    rng: Rng,
    next_doc: u64,
    next_vec: u64,
    live_doc: Vec<u64>,
    live_vec: Vec<u64>,
}

impl Gen {
    fn insert_doc(&mut self) -> Statement {
        let i = self.next_doc;
        self.next_doc += 1;
        self.live_doc.push(i);
        Statement::Insert(Record {
            id: RecordId::new("doc", Id::Num(i)),
            body: BTreeMap::from([
                (
                    "text".into(),
                    Value::Str(format!(
                        "{} {}",
                        WORDS[i as usize % WORDS.len()],
                        WORDS[(i as usize / 2) % WORDS.len()]
                    )),
                ),
                ("group".into(), Value::Int((i % 3) as i64)),
            ]),
            embedding: None,
            created_at: 0,
        })
    }

    fn insert_vec(&mut self) -> Statement {
        let i = self.next_vec;
        self.next_vec += 1;
        self.live_vec.push(i);
        Statement::Insert(Record {
            id: RecordId::new("vec", Id::Num(i)),
            body: BTreeMap::from([
                (
                    "text".into(),
                    Value::Str(format!(
                        "{} {}",
                        WORDS[(i as usize + 1) % WORDS.len()],
                        WORDS[i as usize % WORDS.len()]
                    )),
                ),
                ("group".into(), Value::Int((i % 3) as i64)),
            ]),
            embedding: Some((0..4).map(|_| self.rng.next_f32()).collect()),
            created_at: 0,
        })
    }

    fn relate(&mut self) -> Option<Statement> {
        if self.live_doc.is_empty() || self.live_vec.is_empty() {
            return None;
        }
        // Alternate directions deterministically by total id parity so both
        // edge orientations appear in the sweep.
        let a = self.live_doc[self.rng.below(self.live_doc.len())];
        let b = self.live_vec[self.rng.below(self.live_vec.len())];
        let (from, to) = if (a + b) % 2 == 0 {
            (
                RecordId::new("doc", Id::Num(a)),
                RecordId::new("vec", Id::Num(b)),
            )
        } else {
            (
                RecordId::new("vec", Id::Num(b)),
                RecordId::new("doc", Id::Num(a)),
            )
        };
        Some(Statement::Relate(RelationEdge {
            from,
            name: "links".into(),
            to,
            created_at: 0,
            weight: None,
            props: BTreeMap::new(),
        }))
    }

    fn forget(&mut self) -> Option<Statement> {
        let total = self.live_doc.len() + self.live_vec.len();
        if total == 0 {
            return None;
        }
        let pick = self.rng.below(total);
        let id = if pick < self.live_doc.len() {
            RecordId::new("doc", Id::Num(self.live_doc.remove(pick)))
        } else {
            RecordId::new(
                "vec",
                Id::Num(self.live_vec.remove(pick - self.live_doc.len())),
            )
        };
        Some(Statement::Forget { id })
    }

    fn select(&mut self) -> Statement {
        let query: Vec<f32> = (0..4).map(|_| self.rng.next_f32()).collect();
        let group = Value::Int(self.rng.below(3) as i64);
        let sel = match self.rng.below(4) {
            0 => Select {
                table: "doc".into(),
                filter: Some(Filter::FieldEquals {
                    field: "group".into(),
                    value: group,
                }),
                ..Select::default()
            },
            1 => Select {
                table: "doc".into(),
                filter: Some(Filter::Bm25 {
                    field: "text".into(),
                    query: "alpha beta".into(),
                    k: Some(5),
                }),
                ..Select::default()
            },
            2 => Select {
                table: "vec".into(),
                knn: Some(Knn { query, k: 3 }),
                ..Select::default()
            },
            _ => Select {
                table: "doc".into(),
                aggregate: Some(Aggregate::CountStar),
                ..Select::default()
            },
        };
        Statement::Select(sel)
    }

    fn plan(&mut self) -> Vec<Statement> {
        let n = 1 + self.rng.below(5);
        let mut plan = Vec::with_capacity(n);
        for _ in 0..n {
            let roll = self.rng.below(100);
            let stmt = if roll < 30 {
                Some(self.insert_doc())
            } else if roll < 50 {
                Some(self.insert_vec())
            } else if roll < 65 {
                self.relate().or_else(|| Some(self.insert_doc()))
            } else if roll < 75 {
                self.forget().or_else(|| Some(self.insert_vec()))
            } else if roll < 95 {
                Some(self.select())
            } else {
                Some(Statement::PruneHistory)
            };
            plan.extend(stmt);
        }
        // A plan of pure reads still exercises the no-WAL path; keep it.
        plan
    }
}

fn is_select(stmt: &Statement) -> bool {
    matches!(stmt, Statement::Select(_))
}

/// From-scratch re-execution of `plans[..k]` plus `plans[k][..j]` — the
/// reference for the complete-prefix oracle.
fn exec_prefix(plans: &[Vec<Statement>], k: usize, j: usize) -> Store {
    let mut db = Database::new(Store::default());
    for plan in &plans[..k] {
        db.execute(plan).expect("replay prefix plan");
    }
    if j > 0 {
        db.execute(&plans[k][..j]).expect("replay plan prefix");
    }
    db.store().clone()
}

fn copy_store_files(main: &Path, sandbox: &Path) {
    if main.exists() {
        fs::copy(main, sandbox.join("s.ndb")).expect("copy main");
    }
    let wal = wal_path(main);
    if wal.exists() {
        let mut s = sandbox.join("s.ndb").as_os_str().to_owned();
        s.push(".wal");
        fs::copy(&wal, PathBuf::from(s)).expect("copy wal");
    }
}

/// Truncation probe on a pre-flush sandbox copy (`pre_main` + its sidecar
/// WAL carry exactly the bytes `l0`/`l1` describe): cut the WAL at `t`,
/// reopen, claim any lazy file-era history tail (#133 — a checkpoint-loaded
/// core carries `history == []` until the first temporal read, which would
/// otherwise poison the comparison), and verify the complete-prefix oracle.
///
/// `t == l1` must recover the full plan (plus select digests);
/// `t == l0` must drop it whole; interior points must recover a statement
/// prefix of plan `k` (found by prefix scan — replay consumes complete
/// frames only, so a partial batch applies a statement prefix, never a
/// torn statement).
fn probe_truncate(
    seed: u64,
    k: usize,
    pre_main: &Path,
    plans: &[Vec<Statement>],
    l0: u64,
    l1: u64,
    t: u64,
) {
    if t < l1 {
        let wal = wal_path(pre_main);
        let f = fs::OpenOptions::new()
            .write(true)
            .open(&wal)
            .expect("open sandbox wal");
        f.set_len(t).expect("truncate sandbox wal");
    }
    let mut reopened = Database::open(pre_main).expect("reopen crashed store");
    // Claim the lazy tail before comparing: file-era history predates the
    // WAL replay, so prepending keeps timestamps ascending (same invariant
    // `flush` relies on) and the store becomes comparable to a from-scratch
    // re-execution. No-op for WAL-only copies.
    reopened.ensure_history().expect("claim sandbox history");
    let recovered = reopened.store().clone();

    if t == l1 {
        // Clean reopen: the full acknowledged prefix must survive intact.
        let expected = exec_prefix(plans, k, plans[k].len());
        assert_eq!(
            recovered, expected,
            "seed {seed} plan {k}: clean reopen diverged (t == l1 == {l1})"
        );
        // Digest-on-reopen: the plan's SELECTs answer identically.
        for (sel, want) in exec_shadow_selects(plans, k) {
            let got = reopened
                .execute(std::slice::from_ref(&sel))
                .unwrap_or_else(|e| panic!("seed {seed} plan {k}: digest select failed: {e}"));
            assert_eq!(
                got, want,
                "seed {seed} plan {k}: select digest moved across clean reopen"
            );
        }
        return;
    }
    if t == l0 {
        // Whole-batch drop pin: recovery must equal the k-1 prefix exactly.
        let whole = if k == 0 {
            Store::default()
        } else {
            exec_prefix(plans, k - 1, plans[k - 1].len())
        };
        assert_eq!(
            recovered, whole,
            "seed {seed} plan {k}: t == l0 ({l0}) must drop the whole batch"
        );
        return;
    }
    // Interior point: scan statement prefixes of plan k (longest first) for
    // the recovered state.
    let mut matched: Option<usize> = None;
    for j in (0..=plans[k].len()).rev() {
        if exec_prefix(plans, k, j) == recovered {
            matched = Some(j);
            break;
        }
    }
    assert!(
        matched.is_some(),
        "seed {seed} plan {k}: truncate t={t} (l0={l0} l1={l1}) recovered a \
         non-prefix state — replay applied a partial frame or dropped an \
         intact one. Replay with this seed (see module docs)."
    );
}

/// Re-execute plan k's SELECTs against a fresh shadow at the full prefix
/// (used for digest-on-reopen comparisons).
fn exec_shadow_selects(
    plans: &[Vec<Statement>],
    k: usize,
) -> Vec<(Statement, Vec<nqlite::QueryResult>)> {
    let mut db = Database::new(Store::default());
    for plan in &plans[..=k] {
        db.execute(plan).expect("shadow prefix for selects");
    }
    plans[k]
        .iter()
        .filter(|s| is_select(s))
        .map(|s| {
            let got = db.execute(std::slice::from_ref(s)).expect("shadow select");
            (s.clone(), got)
        })
        .collect()
}

/// Lost-rename probe (inlined in the flush cadence of [`run_seed`]):
/// snapshot the pre-flush files into a sandbox, flush the live database,
/// open the sandbox (old main + full WAL — the rename lost to a crash),
/// and verify full recovery. Kept inline so the snapshot/flush ordering
/// is visible at the single call site.
fn run_seed(seed: u64, n_plans: usize) {
    let mut rng = Rng(seed);
    let dir = fresh_dir("live", seed);
    let main = dir.join("s.ndb");
    let mut durable = Database::open(&main).expect("open live store");
    let mut shadow = Database::new(Store::default());
    let mut gen = Gen {
        rng: Rng(seed ^ 0x5EED),
        next_doc: 0,
        next_vec: 0,
        live_doc: Vec::new(),
        live_vec: Vec::new(),
    };
    // Plan 0 always declares both tables so later statements resolve.
    let mut plans: Vec<Vec<Statement>> = vec![vec![
        Statement::CreateTable {
            table: "doc".into(),
            vector_dim: None,
        },
        Statement::CreateTable {
            table: "vec".into(),
            vector_dim: Some(4),
        },
    ]];
    for _ in 1..n_plans {
        plans.push(gen.plan());
    }

    // Probes consume `rng`; plan generation used only `gen.rng` — the two
    // streams never couple, so probe decisions can't shift the corpus.
    for (k, plan) in plans.iter().enumerate() {
        let l0 = wal_len(&main);
        let res_durable = durable.execute(plan).unwrap_or_else(|e| {
            panic!("seed {seed} plan {k}: durable execute failed: {e}");
        });
        let res_shadow = shadow.execute(plan).unwrap_or_else(|e| {
            panic!("seed {seed} plan {k}: shadow execute failed: {e}");
        });
        assert_eq!(
            res_durable, res_shadow,
            "seed {seed} plan {k}: durable vs shadow results diverged"
        );
        assert_eq!(
            durable.store(),
            shadow.store(),
            "seed {seed} plan {k}: durable vs shadow store diverged"
        );
        let l1 = wal_len(&main);

        // Snapshot the pre-flush files FIRST: every probe below surgeries
        // a sandbox copy carrying exactly the bytes l0/l1 describe. (A
        // flush checkpoints the WAL away and materializes a main file —
        // probing post-flush bytes against pre-flush lengths, or comparing
        // a lazy-tailed checkpoint load against full history, is
        // meaningless.)
        let sandbox = fresh_dir("pre", seed);
        copy_store_files(&main, &sandbox);
        let pre_main = sandbox.join("s.ndb");

        // Checkpoint cadence (seeded): exercises flush + the lost-rename
        // crash that only exists once a checkpoint has run.
        if rng.chance(0.25) {
            let had_main = pre_main.exists();
            durable.flush().unwrap_or_else(|e| {
                panic!("seed {seed} plan {k}: flush failed: {e}");
            });
            assert_eq!(
                durable.store(),
                shadow.store(),
                "seed {seed} plan {k}: store moved across flush"
            );
            if rng.chance(0.4) {
                // Lost rename: the sandbox keeps the pre-flush files (old
                // main + full WAL) while the live database moved on — open
                // it and demand the full acknowledged state back.
                let mut renamed = Database::open(&pre_main).expect("reopen after lost rename");
                renamed.ensure_history().expect("claim sandbox history");
                assert_eq!(
                    renamed.store(),
                    shadow.store(),
                    "seed {seed} plan {k}: lost rename diverged (had_main={had_main})"
                );
            }
        }

        // Truncation probe on the pre-flush copy (the live sequence
        // continues undisturbed): T uniform in [l0, l1], endpoints
        // included — l1 is the clean-reopen pin, l0 the whole-batch-drop
        // pin.
        if rng.chance(0.5) {
            let t = if l1 > l0 {
                l0 + rng.next_u64() % (l1 - l0 + 1)
            } else {
                l1
            };
            probe_truncate(seed, k, &pre_main, &plans, l0, l1, t);
        }
        let _ = fs::remove_dir_all(&sandbox);
    }
    drop(durable);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn crash_sweep_fixed() {
    for &seed in FIXED_SEEDS {
        run_seed(seed, FIXED_PLANS);
    }
}

/// Random sweep (nightly/local): `NQL_CRASH_SEED` is a u64 or `random`
/// (nanos seed — printed, so any failure stays replayable);
/// `NQL_CRASH_PLANS` sets the plan count (default 200).
#[test]
#[ignore]
fn crash_sweep_random() {
    let seed: u64 = match std::env::var("NQL_CRASH_SEED").as_deref() {
        Ok("random") | Err(_) => {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos() as u64
        }
        Ok(s) => s.parse().unwrap_or_else(|_| {
            panic!("NQL_CRASH_SEED must be a u64 or `random`, got {s:?}");
        }),
    };
    let n: usize = std::env::var("NQL_CRASH_PLANS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    println!("crash_sweep_random: seed={seed} plans={n}");
    run_seed(seed, n);
}
