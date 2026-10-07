# nqlite file format (M1)

The store is a single main file plus a sidecar write-ahead log. Both are
byte-deterministic: identical store contents always serialize to identical
bytes (postcard; `BTreeMap` iterates in sorted key order, `Vec` in insertion
order — no `HashMap` anywhere in the format).

## 1. Main file (`<name>.nql`)

```
offset 0   : magic   = 8 bytes: "NQLITE01" (0x4E 0x51 0x4C 0x49 0x54 0x45 0x30 0x31)
offset 8   : version = u32 LE = 3   (current layout; 2 = the legacy inline layout, still readable)
offset 12  : reserved = u32 LE = 0
offset 16  : core_len = u64 LE
offset 24  : core     = postcard(core frame)
offset ...  : history  = postcard(Vec<(i64, Statement)>) — tail, to EOF
```

The **core frame** = `{ records, edges, vector_dims, clock, memories, tables }`
— the store minus its history (issue #133):

- **The history tail is not decoded at open.** Current-state queries never
  pay for it; the first temporal read (`AS OF`, `HISTORY SINCE`,
  `PRUNE HISTORY`) claims it once per session (`ensure_history`), prepending
  it to any WAL-replayed entries (file entries always predate them).
- `Store.tables` (the declared-table index, issue #133) rides in the core;
  nested memories rebuild theirs from their inline histories.
- Legacy v2 files (single inline `postcard(Store)` payload, history included)
  load through the compatibility path and rebuild `tables` from that history;
  versions ≤1 are rejected (`BadVersion`) — as they have been since `AS OF`
  landed.
- **Downside:** binaries older than the v3 layout reject version-3 files at
  the version check — loud, never a partial load (truncated frames error as
  `StorageError::Truncated`).

The history tail carries `Store.history`, which after `PRUNE HISTORY`
(issue #95) contains a `Statement::Snapshot` entry: the compacted state
plus the retained `CreateTable` declarations. **Downside caveat:** binaries
older than the compaction feature cannot decode a pruned store (unknown
statement variant) and fail loudly at `postcard` decode; unpruned stores
decode unchanged.

A missing main file means an empty store.

## 2. Write-ahead log (`<name>.nql.wal`)

Append-only sequence of frames, one per mutating `Statement` applied to the
store (CreateTable / Insert / Relate / Forget):

```
frame  : crc32 = u32 LE, len = u32 LE, payload = len bytes of postcard(Statement)
```

- Written before the statement's effects are considered durable; the file is
  `fsync`ed after each batch (one `execute` call = one transaction).
- On open, the WAL is replayed in order against the loaded store. Replay stops
  at the first frame whose `len` is out of bounds or whose CRC32 mismatches —
  that is a torn/crash frame; the WAL is truncated there.
- A `Statement` that is not mutating (Select) is never logged.

## 3. Checkpoint

When the WAL exceeds `CHECKPOINT_THRESHOLD` (default 1 MiB) — or on explicit
`flush()` — the store is re-serialized to the main file (temp file + rename,
then fsync of file + parent dir) and the WAL is truncated to zero length.

## 4. Crash-safety invariants

- Main file is only ever replaced atomically (write `tmp`, fsync, `rename`).
- WAL frames are append-only; a crash mid-frame yields a torn frame that is
  detected by CRC on next open and truncated.
- Therefore, after any crash: `open()` either recovers all acknowledged
  transactions, or (worst case) drops only a transaction whose commit was never
  fsynced — never a partially-applied one, and never corruption.
- Single-writer: one process holds the DB for writing, **enforced** by an
  exclusive sidecar lock `<name>.nql.lock` taken in `StoreFile::open` and held
  until the store is dropped (issue #84). A second opener — other process or
  other handle in the same process — fails fast with `Locked` instead of
  racing the first writer to a checkpoint (which silently lost acknowledged
  writes). On unix the lock is advisory `flock(2)`, released by the kernel on
  close/crash (no stale locks; the lock file persists and is reused); where
  std advisory locking is unavailable (MSRV 1.82 predates `File::try_lock`),
  a `create_new` lock file is used and must be removed manually after a
  crash. Concurrent readers see a consistent snapshot per `execute` (the
  store is swapped atomically at checkpoint; in-memory reads are served from
  the current `Store`).
