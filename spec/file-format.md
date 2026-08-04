# nqlite file format (M1)

The store is a single main file plus a sidecar write-ahead log. Both are
byte-deterministic: identical store contents always serialize to identical
bytes (postcard; `BTreeMap` iterates in sorted key order, `Vec` in insertion
order — no `HashMap` anywhere in the format).

## 1. Main file (`<name>.nql`)

```
offset 0   : magic   = 8 bytes: "NQLITE01" (0x4E 0x51 0x4C 0x49 0x54 0x45 0x30 0x31)
offset 8   : version = u32 LE = 1
offset 12  : reserved = u32 LE = 0
offset 16  : payload = postcard(Store)
```

`Store` = `{ records: BTreeMap<RecordId, Record>, edges: Vec<RelationEdge>,
vector_dims: BTreeMap<String, usize> }` (see `nql-ir`).

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
- Single-writer: one process holds the DB for writing. Concurrent readers see
  a consistent snapshot per `execute` (the store is swapped atomically at
  checkpoint; in-memory reads are served from the current `Store`).
