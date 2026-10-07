# nqlite file format (M1)

The store is a single main file plus a sidecar write-ahead log. Both are
byte-deterministic: identical store contents always serialize to identical
bytes (postcard; `BTreeMap` iterates in sorted key order, `Vec` in insertion
order — no `HashMap` anywhere in the format).

**Versions.** `3` is the current shipped layout (§1–§4 below); `2` is the
legacy inline layout, readable through the compatibility arm; **`4` is the
adopted target layout** (§5, issue #143) — no writer ships it yet, and the
migration switches when the nqlite-zig cutover criteria are met.

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

## 5. Version 4 main file (adopted target)

Status: **adopted as the target layout** (issue #143; Track A of the
nqlite-zig migration). No shipped writer produces v4 — §1 remains the current
layout — and the migration flips at the nqlite-zig cutover criteria. The
write-ahead log (§2) and its framing are **unchanged** by v4; only the main
file's container changes.

Goals: open cost ≈ header + directory reads (zero-copy over `mmap`), a
container specified as tables rather than "whatever postcard derives from
Rust structs", and payload bytes identical to v3 (§5.6) so conversion is a
container remap, not a re-encode.

**Fixed vs varint.** Everything in §5.1–§5.5 that is *not* a payload —
counts, offsets, entry fields, `CLOCK` — is fixed-width little-endian.
§5.7's uleb128/zigzag rules govern payload bytes only: record bodies,
edge payloads, and (verbatim) the `HISTORY` tail.

### 5.1 Header and section table

```
offset 0    : magic   = 8 bytes: "NQLITE01"  (same magic as v2/v3)
offset 8    : version = u32 LE = 4
offset 12   : flags   = u32 LE = 0   (reserved; writers must write 0)
offset 16   : section_count = u32 LE
offset 20   : reserved = u32 LE = 0
offset 24   : section table = section_count × 32 bytes
```

Section-table entry (32 bytes, little-endian):

| bytes | field | meaning |
|---|---|---|
| 0..4 | `tag` u32 | section kind (§5.2) |
| 4..8 | `reserved` u32 | = 0 |
| 8..16 | `offset` u64 | absolute file offset |
| 16..24 | `len` u64 | payload length in bytes |
| 24..28 | `crc32` u32 | CRC32 of the payload |
| 28..32 | `pad` u32 | = 0 |

- Entries appear in **ascending tag order**; section ranges are strictly
  ascending, non-overlapping, and in-bounds (else `Truncated` — same error
  class as v3, never a partial load).
- Alignment: `STRINGS`, `EMBEDS`, and `HISTORY` start at 4096-byte
  boundaries (independent page-fault domains under `mmap`); all other
  sections at 8-byte boundaries. Alignment gaps — and every `reserved`,
  `pad`, and header field not otherwise specified — are `0x00`. The file
  ends at the end of the last section (no trailing bytes).
- **Required sections:** `TABLES`, `RECORDS`, and `CLOCK` are always
  present (even when empty: `count` = 0, `CLOCK` = 8 bytes). `STRINGS`,
  `EMBEDS`, `EDGES`, `MEMORIES`, and `HISTORY` are present **iff
  non-empty** — a writer never emits a zero-payload section, so identical
  store contents always produce identical bytes.
- A missing optional tag means an absent section: absent `HISTORY` ⇒ the
  persisted history is empty — temporal reads behave exactly as on a v3
  file with an empty tail (claim-once succeeds with nothing to decode);
  §2.7's `HistoryPruned` horizon is a different rule and applies as
  always.
- CRC32 is the CRC-32 of §2's WAL frame (IEEE polynomial); it is verified
  when a section is claimed (history at first temporal read, as today),
  and a mismatch is an integrity error, never a partial decode. Whole-file
  integrity stays §9's verify-if-present manifest.

### 5.2 Section tags (fixed)

| tag | section | content |
|---|---|---|
| 1 | `TABLES` | declared-table catalog (§5.3) |
| 2 | `RECORDS` | record directory (§5.4), sorted by canonical `RecordId` order |
| 3 | `STRINGS` | UTF-8 heap; entries reference it by offset+length |
| 4 | `EMBEDS` | `f32`-LE embedding heap; entries reference it by offset |
| 5 | `EDGES` | edge directory + payloads (§5.5), in append order (spec §2.5) |
| 6 | `MEMORIES` | nested stores — a complete §5 layout, recursively (its own version field) |
| 7 | `CLOCK` | `i64` LE, exactly 8 bytes |
| 8 | `HISTORY` | statement log (§5.6) |

`MEMORIES` body: `u64 count`, then `count × { name: string (§5.7 rule),
len: u64, bytes: complete §5 layout of the nested store }` — names
sorted byte-wise, blobs tightly packed in that order, no padding. A nested
blob carries its own magic, version, and section table, and its section
offsets are absolute *within the blob* (a nested `RECORDS` `body_off`
points inside its blob); alignment inside the blob is computed from the
blob's start, as if it were a standalone file.

### 5.3 `TABLES` (declared-table catalog)

```
u64 count, then count × { name: string (§5.7 string rule), vector_dim: u64 }  — vector_dim = u64::MAX means none
```

Table names are sorted byte-wise (canonical — this catalog is a
`BTreeMap<String, Option<usize>>` on the Rust side). `table_idx` values
elsewhere in the format index this sorted order.

### 5.4 `RECORDS` (record directory)

```
u64 count, then count × 48-byte entries:
  u32 table_idx            → TABLES
  u8  id_kind              (0 = numeric, 1 = string)
  u8  reserved = 0
  u16 pad = 0
  u64 id_val               (numeric id, or STRINGS offset when id_kind = 1)
  u32 id_len               (string byte length; 0 when numeric)
  u32 body_len             (bytes at body_off)
  u64 body_off             (absolute file offset → §5.7 `Record.body` encoding)
  u64 embed_off            (absolute file offset into EMBEDS; u64::MAX = none)
  i64 created_at
```

- Entries are sorted by the **canonical `RecordId` order** — the exact total
  order of the Rust type this replaces: compare `table` bytes, then `Id`
  variant rank (`Num` < `Str`), then `u64` / string bytes. This is what makes
  directory iteration byte-equivalent to today's `BTreeMap` iteration, and
  therefore keeps cross-implementation transcript digests comparable.
- `body` is the record's document only (`BTreeMap<String, Value>`, §5.7);
  the id is *not* repeated inside the body.
- **Section layout:** `[u64 count][count × 48-byte entries][bodies]` —
  bodies follow the directory in entry order, tightly packed;
  `body_off` is each body's absolute offset (`body_len` bytes).
- **Heaps:** string ids live in `STRINGS`, packed in entry order, tightly,
  byte-to-byte — no padding, no dedup (`id_val` = absolute offset,
  `id_len` = byte length). `EMBEDS` packs one `vector_dim × 4`-byte f32-LE
  embedding per record that has one, in entry order, tightly, no dedup
  (`embed_off`, absolute); an embedding under a table with no declared
  `vector_dim` is an integrity error (the engine forbids writing one).

### 5.5 `EDGES`

```
u64 count, then count × { off: u64, len: u32, pad: u32 }  → postcard(RelationEdge) at off
```

- Entries are in the store's **append order** — byte-equivalent to v3's
  core-frame `Vec<RelationEdge>` (the engine appends; spec §2.5 scans
  "in append order", deduping by first appearance). Any other order would
  reorder observable `MATCH` traversal results across a checkpoint and
  break transcript-digest determinism, so v3→v4 edge conversion is a pure
  remap.
- **Section layout:** `[u64 count][count × 16-byte entries][payloads]` —
  payloads follow the directory in entry order, tightly packed (`off`
  absolute, `len` = payload byte length, 16-byte entries = `off` u64 LE +
  `len` u32 LE + `pad` u32 = 0). The payload is the §5.7 encoding of
  `RelationEdge` (`from`, `name`, `to`, `created_at`, `weight`, `props`).

### 5.6 `HISTORY`

A plain sequence of `(i64, Statement)` entries using the §5.7 encoding —
**byte-identical to v3's history tail**, so conversion adopts the tail
verbatim. Section bounds replace v3's inline `core_len`; the claim-once /
lazy-decode rules of §1 apply unchanged (current-state queries never touch
it; first temporal read claims it; `PRUNE HISTORY` snapshot entries ride
inside it).

### 5.7 Payload encoding (normative — shared with v3)

Everything *inside* bodies, edges, and history encodes exactly as today's
reference implementation, pinned here so non-Rust readers can implement it
without guessing. Where prose is ambiguous, **the golden fixtures are the
oracle**.

- **Unsigned lengths / counts / enum tags**: unsigned LEB128 (uleb128),
  little-endian byte order, ≤ 10 bytes for u64.
- **Signed integers** (`i64`, `created_at`, …): zigzag transform, then
  uleb128.
- **Booleans**: 1 byte (`0x00` / `0x01`).
- **f32**: 4 bytes LE · **f64**: 8 bytes LE.
- **Strings / byte arrays**: uleb128 byte length, then raw UTF-8 bytes.
- **Option**: `0x00` = none, else `0x01` followed by the value.
- **Vec / Arr**: uleb128 length, then elements.
- **Doc (map)**: uleb128 entry count, then `(key, value)` pairs — keys are
  strings by the string rule; map iteration is byte-wise key order.
- **`Value` tags** (declaration order of `nql_ir::Value` — append-only):
  `0` Null · `1` Bool · `2` Int · `3` Float (f64) · `4` Str · `5` Doc ·
  `6` Arr · `7` Vector (uleb128 count + f32-LE each) · `8` Ref (`RecordId`:
  table string, then `Id` tag `0`=Num (uleb128 u64) / `1`=Str (string rule)).
- **`Statement` tags** (declaration order of `nql_ir::Statement` —
  **append-only**: existing tags never reorder or get reused; unknown tags
  must fail decode loudly, exactly as postcard does today):
  `0` CreateTable · `1` Insert · `2` Relate · `3` Select · `4` Match ·
  `5` Closure · `6` Forget · `7` Memory · `8` ContextReset · `9` MatchCount ·
  `10` PruneHistory · `11` Snapshot · `12` HistorySince.

### 5.8 Compatibility and migration

| reader \ file | v2 | v3 | v4 |
|---|---|---|---|
| pre-v4 binaries | legacy arm | ✓ | `BadVersion` — loud, never partial |
| v4-aware | legacy arm | legacy arm → upgrade on next checkpoint | ✓ |

- A v4-aware reader opens v3 through the existing legacy arm and **writes v4
  at the next checkpoint** — the same upgrade-on-checkpoint pattern v2→v3
  used (§1). Conversion tooling (`nql-migrate`, Rust side) exists for bulk
  conversion without executing the store.
- v4-aware builds report `BadVersion (supported: 2, 3, 4)` (current builds
  print `(supported: 2, 3)`).
- The WAL (§2) is version-independent and replays into either layout.
- **Downside (symmetric to §1):** every current binary rejects v4 files at
  the version check — the migration must land as a coordinated release.

### 5.9 Non-normative implementation notes

- The format is identical whether read via `mmap` or into a heap; `flags`
  is reserved rather than used to signal either.
- Size caps are `u32` (`body_len`, `id_len`); a violation is an open-time
  error, never truncation-without-error (§4 spirit).
- Directory counts are `u64` — no practical record-count cap.
