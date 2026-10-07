//! Format-v4 golden fixtures — spec/file-format.md §5 (issue #143 sequence).
//!
//! The Rust engine is the fixture-gen **oracle** for nqlite-zig's M1
//! reader/writer: these committed bytes, plus `statements.json` (every
//! `Statement` tag in both serde-JSON and postcard hex), are what the zig
//! implementation must round-trip byte-exactly. §5.7: "Where prose is
//! ambiguous, the golden fixtures are the oracle."
//!
//! Regenerate after a *deliberate* spec change (never hand-edit):
//!
//! ```sh
//! UPDATE_FIXTURES=1 cargo test --test v4_fixtures
//! ```
//!
//! The default run is the CI gate: a fresh canonical encode must equal the
//! committed bytes, and every committed file must decode and re-encode
//! byte-identically (canonicality: §5.1's layout rules are self-checked).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use nql_ir::{
    Aggregate, CmpOp, Filter, Id, Knn, MatchDirection, MatchPath, MatchStep, Order, Record,
    RecordId, RelationEdge, Select, SnapshotState, Statement, Store, Value,
};
use nqlite::Database;

// ---------------------------------------------------------------------------
// Spec constants (§5.1 header, §5.2 tags)
// ---------------------------------------------------------------------------

const MAGIC: &[u8; 8] = b"NQLITE01";
const V4: u32 = 4;
const T_TABLES: u32 = 1;
const T_RECORDS: u32 = 2;
const T_STRINGS: u32 = 3;
const T_EMBEDS: u32 = 4;
const T_EDGES: u32 = 5;
const T_MEMORIES: u32 = 6;
const T_CLOCK: u32 = 7;
const T_HISTORY: u32 = 8;

const HEADER_LEN: u64 = 24;
const DIR_ENTRY_LEN: u64 = 32;

fn alignment(tag: u32) -> u64 {
    match tag {
        T_STRINGS | T_EMBEDS | T_HISTORY => 4096,
        _ => 8,
    }
}

fn align_up(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

fn tag_name(tag: u32) -> &'static str {
    match tag {
        T_TABLES => "TABLES",
        T_RECORDS => "RECORDS",
        T_STRINGS => "STRINGS",
        T_EMBEDS => "EMBEDS",
        T_EDGES => "EDGES",
        T_MEMORIES => "MEMORIES",
        T_CLOCK => "CLOCK",
        T_HISTORY => "HISTORY",
        _ => "?",
    }
}

// ---------------------------------------------------------------------------
// Primitive helpers (§5.7 primitives, hand-rolled so both directions agree)
// ---------------------------------------------------------------------------

fn crc32(payload: &[u8]) -> u32 {
    let mut h = crc32fast::Hasher::new();
    h.update(payload);
    h.finalize()
}

fn put_uleb(buf: &mut Vec<u8>, mut x: u64) {
    loop {
        let b = (x & 0x7f) as u8;
        x >>= 7;
        if x == 0 {
            buf.push(b);
            return;
        }
        buf.push(b | 0x80);
    }
}

fn put_str(buf: &mut Vec<u8>, s: &str) {
    put_uleb(buf, s.len() as u64);
    buf.extend_from_slice(s.as_bytes());
}

fn take_n<'a>(cur: &mut &'a [u8], n: usize) -> Result<&'a [u8], String> {
    if cur.len() < n {
        return Err("v4: payload truncated".into());
    }
    let (head, tail) = cur.split_at(n);
    *cur = tail;
    Ok(head)
}

fn get_uleb(cur: &mut &[u8]) -> Result<u64, String> {
    let mut out: u64 = 0;
    let mut shift = 0u32;
    for i in 0..10 {
        let b = *cur.first().ok_or("v4: payload truncated")?;
        *cur = &cur[1..];
        if i == 9 && b > 1 {
            return Err("v4: uleb128 overflow".into());
        }
        out |= u64::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            return Ok(out);
        }
        shift += 7;
    }
    Err("v4: uleb128 too long".into())
}

fn get_str<'a>(cur: &mut &'a [u8]) -> Result<&'a str, String> {
    let len = get_uleb(cur)? as usize;
    let bytes = take_n(cur, len)?;
    std::str::from_utf8(bytes).map_err(|_| "v4: invalid UTF-8 in string".to_string())
}

fn get_u64_le(cur: &mut &[u8]) -> Result<u64, String> {
    let b = take_n(cur, 8)?;
    Ok(u64::from_le_bytes(b.try_into().unwrap()))
}

fn get_u32_le(cur: &mut &[u8]) -> Result<u32, String> {
    let b = take_n(cur, 4)?;
    Ok(u32::from_le_bytes(b.try_into().unwrap()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------
// Encoder (§5.1–§5.6)
// ---------------------------------------------------------------------------

struct RecBuilt {
    table_idx: u32,
    kind: u8,
    /// Numeric id (kind 0) or heap-relative STRINGS offset (kind 1).
    id_val: u64,
    id_len: u32,
    body: Vec<u8>,
    body_rel: u64,
    embed_rel: Option<u64>,
    created_at: i64,
}

fn encode_v4(store: &Store) -> Result<Vec<u8>, String> {
    // ---- TABLES (§5.3, always present): names sorted by BTree iteration ----
    let mut tables_p = Vec::new();
    tables_p.extend_from_slice(&(store.tables.len() as u64).to_le_bytes());
    for (name, dim) in &store.tables {
        put_str(&mut tables_p, name);
        tables_p.extend_from_slice(&dim.map_or(u64::MAX, |d| d as u64).to_le_bytes());
    }
    let table_index: BTreeMap<&str, u32> = store
        .tables
        .keys()
        .enumerate()
        .map(|(i, k)| (k.as_str(), i as u32))
        .collect();

    // ---- RECORDS input (§5.4): heaps packed in entry order ----
    let mut strings: Vec<u8> = Vec::new();
    let mut embeds: Vec<u8> = Vec::new();
    let mut recs: Vec<RecBuilt> = Vec::with_capacity(store.records.len());
    let mut bodies_len = 0usize;
    for (rid, rec) in &store.records {
        // BTreeMap iteration = canonical RecordId order (§5.4).
        let table_idx = *table_index
            .get(rid.table.as_str())
            .ok_or_else(|| format!("record {rid} in undeclared table"))?;
        let dim = store.tables[&rid.table];
        let (kind, id_val, id_len) = match &rid.id {
            Id::Num(n) => (0u8, *n, 0u32),
            Id::Str(s) => {
                let rel = strings.len() as u64;
                strings.extend_from_slice(s.as_bytes());
                (1u8, rel, s.len() as u32)
            }
        };
        let embed_rel = match (&rec.embedding, dim) {
            (None, _) => None,
            (Some(emb), Some(d)) => {
                if emb.len() != d {
                    return Err(format!(
                        "record {rid}: embedding dim {} != declared {d}",
                        emb.len()
                    ));
                }
                let rel = embeds.len() as u64;
                for v in emb {
                    embeds.extend_from_slice(&v.to_le_bytes());
                }
                Some(rel)
            }
            (Some(_), None) => {
                return Err(format!(
                    "record {rid}: embedding under a table with no vector_dim"
                ));
            }
        };
        let body = postcard::to_allocvec(&rec.body).map_err(|e| format!("record {rid}: {e}"))?;
        // Bodies start after ALL entries: [u64 count][n × 48B][bodies…].
        let body_rel = (8 + 48 * store.records.len() + bodies_len) as u64;
        bodies_len += body.len();
        recs.push(RecBuilt {
            table_idx,
            kind,
            id_val,
            id_len,
            body,
            body_rel,
            embed_rel,
            created_at: rec.created_at,
        });
    }

    // ---- EDGES (§5.5): payloads in APPEND order (never sorted) ----
    let mut edge_payloads: Vec<Vec<u8>> = Vec::with_capacity(store.edges.len());
    for e in &store.edges {
        edge_payloads.push(postcard::to_allocvec(e).map_err(|e2| format!("edge: {e2}"))?);
    }
    let edge_dir_len = 8 + 16 * edge_payloads.len();
    let edge_bytes: usize = edge_payloads.iter().map(|p| p.len()).sum();
    let mut edge_rels: Vec<u64> = Vec::with_capacity(edge_payloads.len());
    {
        let mut rel = edge_dir_len as u64;
        for p in &edge_payloads {
            edge_rels.push(rel);
            rel += p.len() as u64;
        }
    }

    // ---- MEMORIES (§5.2): recursive blobs, names sorted (BTree) ----
    let mut memories_p: Option<Vec<u8>> = None;
    if !store.memories.is_empty() {
        let mut m = Vec::new();
        m.extend_from_slice(&(store.memories.len() as u64).to_le_bytes());
        for (name, sub) in &store.memories {
            let blob = encode_v4(sub)?;
            put_str(&mut m, name);
            m.extend_from_slice(&(blob.len() as u64).to_le_bytes());
            m.extend_from_slice(&blob);
        }
        memories_p = Some(m);
    }

    // ---- CLOCK (§5.2, always present), HISTORY (§5.6) ----
    let clock_p = store.clock.to_le_bytes().to_vec();
    let history_p = if store.history.is_empty() {
        None
    } else {
        // Byte-identical to v3's history tail (same call as storage.rs).
        Some(postcard::to_allocvec(&store.history).map_err(|e| format!("history: {e}"))?)
    };

    // ---- Presence (§5.1: required always, optional iff non-empty) ----
    let records_len = 8 + 48 * recs.len() + bodies_len;
    let mut sections: Vec<(u32, usize)> =
        vec![(T_TABLES, tables_p.len()), (T_RECORDS, records_len)];
    if !strings.is_empty() {
        sections.push((T_STRINGS, strings.len()));
    }
    if !embeds.is_empty() {
        sections.push((T_EMBEDS, embeds.len()));
    }
    if !edge_payloads.is_empty() {
        sections.push((T_EDGES, edge_dir_len + edge_bytes));
    }
    if let Some(m) = &memories_p {
        sections.push((T_MEMORIES, m.len()));
    }
    sections.push((T_CLOCK, clock_p.len()));
    if let Some(h) = &history_p {
        sections.push((T_HISTORY, h.len()));
    }

    // ---- Layout (§5.1): ascending tags, per-tag alignment, zero gaps ----
    let table_end = HEADER_LEN + DIR_ENTRY_LEN * sections.len() as u64;
    let mut cursor = table_end;
    let mut offsets: Vec<u64> = Vec::with_capacity(sections.len());
    for (tag, len) in &sections {
        let off = align_up(cursor, alignment(*tag));
        offsets.push(off);
        cursor = off + *len as u64;
    }
    let base = |tag: u32| -> Result<u64, String> {
        sections
            .iter()
            .zip(&offsets)
            .find(|((t, _), _)| *t == tag)
            .map(|(_, o)| *o)
            .ok_or_else(|| format!("section {tag} absent"))
    };
    let recs_base = base(T_RECORDS)?;
    let str_base = base(T_STRINGS).ok();
    let emb_base = base(T_EMBEDS).ok();
    let edges_base = base(T_EDGES).ok();

    // ---- Assemble payloads that carry absolute offsets ----
    let mut payloads: Vec<(u32, Vec<u8>)> = Vec::with_capacity(sections.len());

    payloads.push((T_TABLES, tables_p));

    let mut rec_p = Vec::with_capacity(records_len);
    rec_p.extend_from_slice(&(recs.len() as u64).to_le_bytes());
    for r in &recs {
        rec_p.extend_from_slice(&r.table_idx.to_le_bytes());
        rec_p.push(r.kind);
        rec_p.push(0); // reserved
        rec_p.extend_from_slice(&0u16.to_le_bytes()); // pad
        let id_val = match r.kind {
            0 => r.id_val,
            _ => str_base.ok_or("string id but STRINGS absent")? + r.id_val,
        };
        rec_p.extend_from_slice(&id_val.to_le_bytes());
        rec_p.extend_from_slice(&r.id_len.to_le_bytes());
        rec_p.extend_from_slice(&(r.body.len() as u32).to_le_bytes());
        rec_p.extend_from_slice(&(recs_base + r.body_rel).to_le_bytes());
        rec_p.extend_from_slice(
            &match r.embed_rel {
                Some(rel) => emb_base.ok_or("embedding but EMBEDS absent")? + rel,
                None => u64::MAX,
            }
            .to_le_bytes(),
        );
        rec_p.extend_from_slice(&r.created_at.to_le_bytes());
    }
    for r in &recs {
        rec_p.extend_from_slice(&r.body);
    }
    payloads.push((T_RECORDS, rec_p));

    if !strings.is_empty() {
        payloads.push((T_STRINGS, strings));
    }
    if !embeds.is_empty() {
        payloads.push((T_EMBEDS, embeds));
    }
    if !edge_payloads.is_empty() {
        let eb = edges_base.ok_or("edges absent")?;
        let mut e_p = Vec::with_capacity(edge_dir_len + edge_bytes);
        e_p.extend_from_slice(&(edge_payloads.len() as u64).to_le_bytes());
        for (rel, p) in edge_rels.iter().zip(&edge_payloads) {
            e_p.extend_from_slice(&(eb + rel).to_le_bytes());
            e_p.extend_from_slice(&(p.len() as u32).to_le_bytes());
            e_p.extend_from_slice(&0u32.to_le_bytes()); // pad
        }
        for p in &edge_payloads {
            e_p.extend_from_slice(p);
        }
        payloads.push((T_EDGES, e_p));
    }
    if let Some(m) = memories_p {
        payloads.push((T_MEMORIES, m));
    }
    payloads.push((T_CLOCK, clock_p));
    if let Some(h) = history_p {
        payloads.push((T_HISTORY, h));
    }

    // ---- Write: header + section table + zero-padded payloads ----
    let mut out = Vec::with_capacity(cursor as usize);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&V4.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // flags = 0 (§5.1)
    out.extend_from_slice(&(sections.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // reserved = 0
    for ((tag, len), off) in sections.iter().zip(&offsets) {
        let payload = &payloads
            .iter()
            .find(|(t, _)| t == tag)
            .map(|(_, p)| p)
            .expect("payload built for every section");
        debug_assert_eq!(payload.len(), *len);
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // reserved
        out.extend_from_slice(&off.to_le_bytes());
        out.extend_from_slice(&(*len as u64).to_le_bytes());
        out.extend_from_slice(&crc32(payload).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // pad
    }
    for (tag, payload) in &payloads {
        let want = *sections
            .iter()
            .zip(&offsets)
            .find(|((t, _), _)| t == tag)
            .map(|(_, o)| o)
            .expect("offset for every payload");
        assert!(out.len() as u64 <= want, "section {tag} overlaps previous");
        out.resize(want as usize, 0); // §5.1: alignment gaps are 0x00
        out.extend_from_slice(payload);
    }
    debug_assert_eq!(out.len() as u64, cursor, "file ends at last section");
    Ok(out)
}

// ---------------------------------------------------------------------------
// Decoder (§5.1 validation is loud: bad magic/version/flags, bounds,
// ordering, alignment, CRC, required sections — never a partial load)
// ---------------------------------------------------------------------------

struct DirEnt {
    tag: u32,
    off: u64,
    len: u64,
    crc: u32,
}

fn parse_dir(bytes: &[u8]) -> Result<Vec<DirEnt>, String> {
    if bytes.len() < HEADER_LEN as usize {
        return Err("v4: truncated header".into());
    }
    if &bytes[0..8] != MAGIC {
        return Err("v4: bad magic".into());
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if version != V4 {
        return Err(format!(
            "v4: unsupported format version {version} (supported: 4)"
        ));
    }
    if u32::from_le_bytes(bytes[12..16].try_into().unwrap()) != 0 {
        return Err("v4: unsupported flags (must be 0)".into());
    }
    let n = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
    if u32::from_le_bytes(bytes[20..24].try_into().unwrap()) != 0 {
        return Err("v4: nonzero reserved header field".into());
    }
    let table_end = HEADER_LEN + DIR_ENTRY_LEN * n as u64;
    if table_end as usize > bytes.len() {
        return Err("v4: truncated section table".into());
    }
    let mut out = Vec::with_capacity(n);
    let mut prev_tag = 0u32;
    let mut prev_end = table_end;
    for i in 0..n {
        let e = &bytes[(HEADER_LEN as usize + 32 * i)..(HEADER_LEN as usize + 32 * (i + 1))];
        let tag = u32::from_le_bytes(e[0..4].try_into().unwrap());
        if !(T_TABLES..=T_HISTORY).contains(&tag) {
            return Err(format!("v4: unknown section tag {tag}"));
        }
        if tag <= prev_tag {
            return Err("v4: section tags not in ascending order".into());
        }
        prev_tag = tag;
        if u32::from_le_bytes(e[4..8].try_into().unwrap()) != 0
            || u32::from_le_bytes(e[28..32].try_into().unwrap()) != 0
        {
            return Err(format!("v4: nonzero reserved/pad in section {tag}"));
        }
        let off = u64::from_le_bytes(e[8..16].try_into().unwrap());
        let len = u64::from_le_bytes(e[16..24].try_into().unwrap());
        let crc = u32::from_le_bytes(e[24..28].try_into().unwrap());
        if len == 0 {
            return Err(format!("v4: zero-length section {tag}"));
        }
        if off % alignment(tag) != 0 {
            return Err(format!("v4: section {tag} misaligned"));
        }
        if off < prev_end {
            return Err(format!("v4: section {tag} overlaps previous"));
        }
        let end = off.checked_add(len).ok_or("v4: section end overflow")?;
        if end > bytes.len() as u64 {
            return Err(format!("v4: section {tag} out of bounds"));
        }
        if crc32(&bytes[off as usize..end as usize]) != crc {
            return Err(format!("v4: CRC mismatch in section {tag}"));
        }
        out.push(DirEnt { tag, off, len, crc });
        prev_end = end;
    }
    if prev_end as usize != bytes.len() {
        return Err("v4: trailing bytes after last section".into());
    }
    for tag in [T_TABLES, T_RECORDS, T_CLOCK] {
        if !out.iter().any(|e| e.tag == tag) {
            return Err(format!("v4: required section {} absent", tag_name(tag)));
        }
    }
    Ok(out)
}

fn section<'a>(bytes: &'a [u8], dir: &[DirEnt], tag: u32) -> Result<(u64, &'a [u8]), String> {
    let e = dir
        .iter()
        .find(|e| e.tag == tag)
        .ok_or_else(|| format!("v4: section {} absent", tag_name(tag)))?;
    Ok((e.off, &bytes[e.off as usize..(e.off + e.len) as usize]))
}

fn decode_v4(bytes: &[u8]) -> Result<Store, String> {
    let dir = parse_dir(bytes)?;

    // CLOCK (§5.2: exactly 8 bytes).
    let (_, clock_p) = section(bytes, &dir, T_CLOCK)?;
    if clock_p.len() != 8 {
        return Err("v4: CLOCK is not 8 bytes".into());
    }
    let clock = i64::from_le_bytes(clock_p.try_into().unwrap());

    // TABLES (§5.3): sequential decode, positions index every table_idx.
    let (_, tables_p) = section(bytes, &dir, T_TABLES)?;
    let mut cur = tables_p;
    let count = get_u64_le(&mut cur)? as usize;
    let mut tables: BTreeMap<String, Option<usize>> = BTreeMap::new();
    let mut table_list: Vec<(String, Option<usize>)> = Vec::with_capacity(count);
    let mut prev_name: Option<Vec<u8>> = None;
    for _ in 0..count {
        let name = get_str(&mut cur)?.to_string();
        if prev_name.as_deref().is_some_and(|p| p >= name.as_bytes()) {
            return Err("v4: TABLES names not strictly ascending".into());
        }
        prev_name = Some(name.as_bytes().to_vec());
        let dim = get_u64_le(&mut cur)?;
        let d = if dim == u64::MAX {
            None
        } else {
            Some(dim as usize)
        };
        tables.insert(name.clone(), d);
        table_list.push((name, d));
    }
    if !cur.is_empty() {
        return Err("v4: trailing bytes in TABLES".into());
    }

    // STRINGS / EMBEDS entries (optional).
    let strings = section(bytes, &dir, T_STRINGS).ok();
    let embeds = section(bytes, &dir, T_EMBEDS).ok();

    // RECORDS (§5.4).
    let (recs_off, recs_p) = section(bytes, &dir, T_RECORDS)?;
    let recs_ent = dir.iter().find(|e| e.tag == T_RECORDS).unwrap();
    let mut cur = recs_p;
    let n_rec = get_u64_le(&mut cur)? as usize;
    let dir_end = (8 + 48 * n_rec) as u64;
    if (recs_p.len() as u64) < dir_end {
        return Err("v4: RECORDS directory truncated".into());
    }
    let mut records: BTreeMap<RecordId, Record> = BTreeMap::new();
    let mut prev_id: Option<RecordId> = None;
    for _ in 0..n_rec {
        let e = take_n(&mut cur, 48)?;
        let table_idx = u32::from_le_bytes(e[0..4].try_into().unwrap()) as usize;
        let kind = e[4];
        if e[5] != 0 || u16::from_le_bytes(e[6..8].try_into().unwrap()) != 0 {
            return Err("v4: nonzero reserved/pad in RECORDS entry".into());
        }
        if kind > 1 {
            return Err(format!("v4: bad id_kind {kind}"));
        }
        let id_val = u64::from_le_bytes(e[8..16].try_into().unwrap());
        let id_len = u32::from_le_bytes(e[16..20].try_into().unwrap());
        let body_len = u64::from(u32::from_le_bytes(e[20..24].try_into().unwrap()));
        let body_off = u64::from_le_bytes(e[24..32].try_into().unwrap());
        let embed_off = u64::from_le_bytes(e[32..40].try_into().unwrap());
        let created_at = i64::from_le_bytes(e[40..48].try_into().unwrap());

        let (table, table_dim) = table_list
            .get(table_idx)
            .ok_or_else(|| format!("v4: table_idx {table_idx} beyond TABLES"))?;
        let id = match kind {
            0 => {
                if id_len != 0 {
                    return Err("v4: numeric id with nonzero id_len".into());
                }
                Id::Num(id_val)
            }
            _ => {
                let (s_off, s_p) = strings
                    .as_ref()
                    .map(|(o, p)| (*o, *p))
                    .ok_or("v4: string id but no STRINGS section")?;
                let rel = id_val
                    .checked_sub(s_off)
                    .ok_or("v4: string id before STRINGS")?;
                if rel + u64::from(id_len) > s_p.len() as u64 {
                    return Err("v4: string id outside STRINGS".into());
                }
                let raw = &s_p[rel as usize..(rel + u64::from(id_len)) as usize];
                Id::Str(String::from_utf8(raw.to_vec()).map_err(|_| "v4: string id not UTF-8")?)
            }
        };

        // Body lives inside RECORDS, after the directory (§5.4 errata).
        if body_off < recs_off + dir_end || body_off + body_len > recs_off + recs_ent.len {
            return Err("v4: record body outside RECORDS payloads".into());
        }
        let body_cur = &bytes[body_off as usize..(body_off + body_len) as usize];
        let (body, rest): (BTreeMap<String, Value>, &[u8]) =
            postcard::take_from_bytes(body_cur).map_err(|e| format!("v4: record body: {e}"))?;
        if !rest.is_empty() {
            return Err("v4: trailing bytes in record body".into());
        }

        let embedding = if embed_off == u64::MAX {
            None
        } else {
            let dim = table_dim
                .ok_or_else(|| format!("v4: embedding under table {table} with no vector_dim"))?;
            let (e_off, e_p) = embeds
                .as_ref()
                .map(|(o, p)| (*o, *p))
                .ok_or("v4: embed_off set but no EMBEDS section")?;
            let rel = embed_off
                .checked_sub(e_off)
                .ok_or("v4: embedding before EMBEDS")?;
            let bytes_len = (dim as u64).checked_mul(4).ok_or("v4: dim overflow")?;
            if rel + bytes_len > e_p.len() as u64 {
                return Err("v4: embedding outside EMBEDS".into());
            }
            let mut v = Vec::with_capacity(dim);
            for c in e_p[rel as usize..(rel + bytes_len) as usize].chunks_exact(4) {
                v.push(f32::from_le_bytes(c.try_into().unwrap()));
            }
            Some(v)
        };

        let rid = RecordId {
            table: table.clone(),
            id,
        };
        if prev_id.as_ref().is_some_and(|p| p >= &rid) {
            return Err("v4: RECORDS not in canonical RecordId order".into());
        }
        prev_id = Some(rid.clone());
        let record = Record {
            id: rid.clone(),
            body,
            embedding,
            created_at,
        };
        records.insert(rid, record);
    }
    if records.len() != n_rec {
        return Err("v4: duplicate record id".into());
    }

    // EDGES (§5.5): append order, no sort at load. Directory first, then
    // payloads validated by bounds (they share the section after the index).
    let mut edges: Vec<RelationEdge> = Vec::new();
    if let Ok((ed_off, ed_p)) = section(bytes, &dir, T_EDGES) {
        let ed_ent = dir.iter().find(|e| e.tag == T_EDGES).unwrap();
        let mut cur = ed_p;
        let n_e = get_u64_le(&mut cur)? as usize;
        let idx_end = (8 + 16 * n_e) as u64;
        if (ed_p.len() as u64) < idx_end {
            return Err("v4: EDGES directory truncated".into());
        }
        let mut entries = Vec::with_capacity(n_e);
        for _ in 0..n_e {
            let off = get_u64_le(&mut cur)?;
            let len = u64::from(get_u32_le(&mut cur)?);
            let pad = get_u32_le(&mut cur)?;
            if pad != 0 {
                return Err("v4: nonzero pad in EDGES entry".into());
            }
            entries.push((off, len));
        }
        for (off, len) in entries {
            if off < ed_off + idx_end || off + len > ed_off + ed_ent.len {
                return Err("v4: edge payload outside EDGES payloads".into());
            }
            let e_cur = &bytes[off as usize..(off + len) as usize];
            let (e, rest): (RelationEdge, &[u8]) =
                postcard::take_from_bytes(e_cur).map_err(|e2| format!("v4: edge payload: {e2}"))?;
            if !rest.is_empty() {
                return Err("v4: trailing bytes in edge payload".into());
            }
            edges.push(e);
        }
    }

    // MEMORIES (§5.2): recursive complete §5 layouts, names sorted.
    let mut memories: BTreeMap<String, Store> = BTreeMap::new();
    if let Ok((_, m_p)) = section(bytes, &dir, T_MEMORIES) {
        let mut cur = m_p;
        let n_m = get_u64_le(&mut cur)? as usize;
        let mut prev_name: Option<Vec<u8>> = None;
        for _ in 0..n_m {
            let name = get_str(&mut cur)?.to_string();
            if prev_name.as_deref().is_some_and(|p| p >= name.as_bytes()) {
                return Err("v4: MEMORIES names not strictly ascending".into());
            }
            prev_name = Some(name.as_bytes().to_vec());
            let len = get_u64_le(&mut cur)? as usize;
            let blob = take_n(&mut cur, len)?;
            let sub = decode_v4(blob)?;
            memories.insert(name, sub);
        }
        if !cur.is_empty() {
            return Err("v4: trailing bytes in MEMORIES".into());
        }
    }

    // HISTORY (§5.6): verbatim postcard tail; absent = empty (§5.1 errata).
    let history = match section(bytes, &dir, T_HISTORY) {
        Ok((_, h_p)) => {
            let (h, rest): (Vec<(i64, Statement)>, &[u8]) =
                postcard::take_from_bytes(h_p).map_err(|e| format!("v4: history: {e}"))?;
            if !rest.is_empty() {
                return Err("v4: trailing bytes in HISTORY".into());
            }
            h
        }
        Err(_) => Vec::new(),
    };

    let vector_dims = tables
        .iter()
        .filter_map(|(n, d)| d.map(|x| (n.clone(), x)))
        .collect();
    Ok(Store {
        records,
        edges,
        vector_dims,
        clock,
        history,
        memories,
        tables,
    })
}

// ---------------------------------------------------------------------------
// Canonical fixture stores
// ---------------------------------------------------------------------------

fn rec(
    table: &str,
    id: Id,
    body: Vec<(&str, Value)>,
    embedding: Option<Vec<f32>>,
    created_at: i64,
) -> Record {
    Record {
        id: RecordId::new(table, id),
        body: body.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        embedding,
        created_at,
    }
}

fn edge(
    from: RecordId,
    name: &str,
    to: RecordId,
    created_at: i64,
    weight: Option<f32>,
    props: Vec<(&str, Value)>,
) -> RelationEdge {
    RelationEdge {
        from,
        name: name.to_string(),
        to,
        created_at,
        weight,
        props: props.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
    }
}

/// Hand-built: records only, no optional sections (all ids numeric,
/// history empty — a legal §5 file for an imported store).
fn plain_store() -> Store {
    let mut s = Store::default();
    s.tables.insert("note".into(), None);
    // Inserted in reverse: canonical order comes from the BTree, not input.
    s.insert(rec("note", Id::Num(2), vec![("z", Value::Int(1))], None, 9));
    s.insert(rec(
        "note",
        Id::Num(1),
        vec![("a", Value::Str("first".into()))],
        None,
        4,
    ));
    s.clock = 4;
    s
}

/// Engine-built: every `Value` kind, string/numeric ids in canonical order,
/// deliberately append-ordered (≠ sorted) edges, embeddings, a nested memory,
/// and the full mutation history.
fn rich_store() -> Store {
    let a1 = RecordId::new("alpha", Id::Num(1));
    let a007 = RecordId::new("alpha", Id::Str("007".into()));
    let n9 = RecordId::new("note", Id::Num(9));
    let z3 = RecordId::new("zebra", Id::Num(3));
    let s1 = RecordId::new("scratch", Id::Num(1));

    let mut db = Database::new(Store::default());
    db.execute(&[
        Statement::CreateTable {
            table: "alpha".into(),
            vector_dim: Some(4),
        },
        Statement::CreateTable {
            table: "note".into(),
            vector_dim: None,
        },
        Statement::CreateTable {
            table: "zebra".into(),
            vector_dim: Some(2),
        },
        Statement::CreateTable {
            table: "scratch".into(),
            vector_dim: None,
        },
    ])
    .expect("creates");

    db.execute(&[
        Statement::Insert(rec(
            "alpha",
            Id::Num(1),
            vec![
                ("a", Value::Int(-7)),
                ("b", Value::Bool(true)),
                ("c", Value::Str("γ ✓".into())),
                ("d", Value::Float(-0.0)),
                ("e", Value::Null),
                (
                    "f",
                    Value::Doc(
                        [
                            (
                                "g".to_string(),
                                Value::Arr(vec![
                                    Value::Int(1),
                                    Value::Str("x".into()),
                                    Value::Bool(false),
                                ]),
                            ),
                            ("h".to_string(), Value::Vector(vec![0.1, 0.25])),
                            ("i".to_string(), Value::Ref(a007.clone())),
                        ]
                        .into_iter()
                        .collect(),
                    ),
                ),
            ],
            Some(vec![0.5, 0.25, 0.5, 1.0]),
            0,
        )),
        Statement::Insert(rec(
            "alpha",
            Id::Num(5),
            vec![
                ("empty_doc", Value::Doc(BTreeMap::new())),
                ("empty_arr", Value::Arr(vec![])),
                ("s", Value::Str(String::new())),
            ],
            None,
            0,
        )),
        Statement::Insert(rec(
            "alpha",
            Id::Str("007".into()),
            vec![("k", Value::Str("v".into()))],
            None,
            0,
        )),
        Statement::Insert(rec(
            "alpha",
            Id::Str("10".into()),
            vec![("k", Value::Int(10))],
            None,
            0,
        )),
        Statement::Insert(rec(
            "alpha",
            Id::Str("2".into()),
            vec![("k", Value::Int(2))],
            None,
            0,
        )),
        Statement::Insert(rec(
            "note",
            Id::Num(9),
            vec![
                ("nan", Value::Float(f64::NAN)),
                ("inf", Value::Float(f64::INFINITY)),
                ("min", Value::Int(i64::MIN)),
                ("max", Value::Int(i64::MAX)),
            ],
            None,
            0,
        )),
        Statement::Insert(rec(
            "zebra",
            Id::Num(3),
            vec![("n", Value::Float(f64::MAX))],
            Some(vec![0.25, -0.5]),
            0,
        )),
        Statement::Insert(rec(
            "zebra",
            Id::Num(4),
            vec![("ok", Value::Bool(true))],
            None,
            0,
        )),
        Statement::Insert(rec(
            "scratch",
            Id::Num(1),
            vec![("t", Value::Str("x".into()))],
            None,
            0,
        )),
    ])
    .expect("inserts");

    // Append order here is deliberately NOT (from, name, to, created_at)
    // sorted — §5.5 pins append order; the fixture proves it.
    db.execute(&[
        Statement::Relate(edge(
            a1.clone(),
            "knows",
            n9.clone(),
            2,
            Some(0.5),
            vec![("src", Value::Str("seed".into()))],
        )),
        Statement::Relate(edge(n9.clone(), "knows", a1.clone(), 1, None, vec![])),
        Statement::Relate(edge(
            a1.clone(),
            "likes",
            z3.clone(),
            3,
            None,
            vec![("count", Value::Int(3))],
        )),
        Statement::Relate(edge(a1.clone(), "knows", n9.clone(), 4, None, vec![])),
    ])
    .expect("relates");

    db.execute(&[Statement::Forget { id: s1 }]).expect("forget");

    db.execute(&[
        Statement::Memory {
            name: "dreams".into(),
        },
        Statement::CreateTable {
            table: "tiny".into(),
            vector_dim: Some(2),
        },
        Statement::Insert(rec(
            "tiny",
            Id::Num(1),
            vec![("v", Value::Float(0.5))],
            Some(vec![0.1, 0.2]),
            0,
        )),
        Statement::Insert(rec("tiny", Id::Num(2), vec![("v", Value::Int(2))], None, 0)),
        Statement::Relate(edge(
            RecordId::new("tiny", Id::Num(1)),
            "next",
            RecordId::new("tiny", Id::Num(2)),
            1,
            Some(1.0),
            vec![],
        )),
        Statement::Memory { name: "d2".into() },
        Statement::CreateTable {
            table: "d2".into(),
            vector_dim: None,
        },
        Statement::Insert(rec(
            "d2",
            Id::Num(1),
            vec![("deep", Value::Bool(true))],
            None,
            0,
        )),
    ])
    .expect("memory block");

    let mut store = db.into_store();
    // The engine creates memories at the ROOT only (execute_in_context's
    // Memory arm runs on the root store), but §5.2 MEMORIES is recursive —
    // hand-nest `d2` inside `dreams` for coverage of blob-local offsets at
    // depth 2. Legal per `Store`'s type.
    if let Some(d2) = store.memories.remove("d2") {
        store
            .memories
            .get_mut("dreams")
            .expect("dreams exists")
            .memories
            .insert("d2".into(), d2);
    }
    // Pin the fixture's claim: append order ≠ sorted order (§5.5 errata).
    let mut sorted = store.edges.clone();
    sorted.sort_by(|x, y| {
        (&x.from, x.name.as_bytes(), &x.to, x.created_at).cmp(&(
            &y.from,
            y.name.as_bytes(),
            &y.to,
            y.created_at,
        ))
    });
    assert_ne!(store.edges, sorted, "fixture must demonstrate append order");
    store
}

/// Engine-built: `PRUNE HISTORY` compacted — declarations retained at their
/// original timestamps, `Statement::Snapshot` at the tail (the nested
/// memory's own history is pruned too — snapshots all the way down).
fn pruned_store() -> Store {
    let mut db = Database::default();
    db.execute(&[
        Statement::CreateTable {
            table: "note".into(),
            vector_dim: None,
        },
        Statement::Insert(rec(
            "note",
            Id::Num(1),
            vec![("a", Value::Str("one".into()))],
            None,
            0,
        )),
        Statement::Insert(rec(
            "note",
            Id::Num(2),
            vec![("a", Value::Str("two".into()))],
            None,
            0,
        )),
    ])
    .expect("creates/inserts");
    db.execute(&[
        Statement::Memory {
            name: "archive".into(),
        },
        Statement::CreateTable {
            table: "box".into(),
            vector_dim: None,
        },
        Statement::Insert(rec("box", Id::Num(1), vec![("w", Value::Int(1))], None, 0)),
    ])
    .expect("memory block");
    db.execute(&[Statement::PruneHistory]).expect("prune");
    db.into_store()
}

fn fixture_stores() -> Vec<(&'static str, Store)> {
    vec![
        ("empty", Store::default()),
        ("plain", plain_store()),
        ("rich", rich_store()),
        ("pruned", pruned_store()),
    ]
}

// ---------------------------------------------------------------------------
// Statement oracle (all 13 §5.7 tags) + serde JSON twin
// ---------------------------------------------------------------------------

fn stmt_tag(s: &Statement) -> u32 {
    match s {
        Statement::CreateTable { .. } => 0,
        Statement::Insert(_) => 1,
        Statement::Relate(_) => 2,
        Statement::Select(_) => 3,
        Statement::Match(_) => 4,
        Statement::Closure(_) => 5,
        Statement::Forget { .. } => 6,
        Statement::Memory { .. } => 7,
        Statement::ContextReset => 8,
        Statement::MatchCount(_) => 9,
        Statement::PruneHistory => 10,
        Statement::Snapshot(_) => 11,
        Statement::HistorySince(_) => 12,
    }
}

fn stmt_name(s: &Statement) -> &'static str {
    match s {
        Statement::CreateTable { .. } => "CreateTable",
        Statement::Insert(_) => "Insert",
        Statement::Relate(_) => "Relate",
        Statement::Select(_) => "Select",
        Statement::Match(_) => "Match",
        Statement::Closure(_) => "Closure",
        Statement::Forget { .. } => "Forget",
        Statement::Memory { .. } => "Memory",
        Statement::ContextReset => "ContextReset",
        Statement::MatchCount(_) => "MatchCount",
        Statement::PruneHistory => "PruneHistory",
        Statement::Snapshot(_) => "Snapshot",
        Statement::HistorySince(_) => "HistorySince",
    }
}

/// Every `Statement` tag, with coverage over the nested enums a non-Rust
/// reader must implement (`Filter` ×7, `Order` ×8, `CmpOp` ×5, directions,
/// `Aggregate`, boxed `SnapshotState`, unit/tuple/struct variants).
fn statement_samples() -> Vec<Statement> {
    let a1 = RecordId::new("alpha", Id::Num(1));
    let alice = RecordId::new("alpha", Id::Str("alice".into()));
    let sample_rec = Record {
        id: a1.clone(),
        body: BTreeMap::from([
            ("k".to_string(), Value::Int(-3)),
            ("r".to_string(), Value::Ref(alice.clone())),
        ]),
        embedding: None,
        created_at: 7,
    };
    let sample_edge = RelationEdge {
        from: a1.clone(),
        name: "knows".into(),
        to: alice.clone(),
        created_at: 2,
        weight: Some(0.5),
        props: BTreeMap::from([("w".to_string(), Value::Bool(true))]),
    };
    let path = |as_of: Option<i64>| MatchPath {
        start: a1.clone(),
        steps: vec![
            MatchStep {
                direction: MatchDirection::Out,
                name: "knows".into(),
                edge_props: Some(Filter::FieldCmp {
                    field: "w".into(),
                    op: CmpOp::Ge,
                    value: Value::Int(1),
                }),
            },
            MatchStep {
                direction: MatchDirection::In,
                name: "likes".into(),
                edge_props: None,
            },
        ],
        as_of,
    };
    let sel = |table: &str, filter: Option<Filter>, order: Option<Order>| {
        Statement::Select(Select {
            table: table.into(),
            filter,
            order,
            ..Default::default()
        })
    };
    vec![
        Statement::CreateTable {
            table: "alpha".into(),
            vector_dim: Some(4),
        },
        Statement::CreateTable {
            table: "note".into(),
            vector_dim: None,
        },
        Statement::Insert(sample_rec.clone()),
        Statement::Relate(sample_edge.clone()),
        // One fat SELECT: kNN, And-combined filters (HasEmbedding, Bm25,
        // FieldIn, FieldBetween, FieldCmp Lt/Le/Gt/Ge), SalienceWeighted,
        // as_of/fields/offset/aggregate.
        Statement::Select(Select {
            table: "alpha".into(),
            knn: Some(Knn {
                query: vec![0.25, 0.5, 1.0, 0.125],
                k: 5,
            }),
            filter: Some(Filter::And(vec![
                Filter::HasEmbedding,
                Filter::Bm25 {
                    field: "t".into(),
                    query: "hello world".into(),
                    k: Some(3),
                },
                Filter::FieldIn {
                    field: "tag".into(),
                    values: vec![Value::Str("x".into()), Value::Null],
                },
                Filter::FieldBetween {
                    field: "n".into(),
                    lo: Value::Int(1),
                    hi: Value::Int(9),
                },
                Filter::FieldCmp {
                    field: "n".into(),
                    op: CmpOp::Lt,
                    value: Value::Int(0),
                },
                Filter::FieldCmp {
                    field: "n".into(),
                    op: CmpOp::Le,
                    value: Value::Int(1),
                },
                Filter::FieldCmp {
                    field: "n".into(),
                    op: CmpOp::Gt,
                    value: Value::Int(2),
                },
                Filter::FieldCmp {
                    field: "n".into(),
                    op: CmpOp::Ge,
                    value: Value::Int(3),
                },
            ])),
            order: Some(Order::SalienceWeighted([0.7, 0.0, 0.0, 0.3])),
            limit: Some(10),
            as_of: Some(3),
            fields: Some(vec!["k".into()]),
            offset: Some(2),
            aggregate: Some(Aggregate::CountStar),
        }),
        sel(
            "note",
            Some(Filter::FieldEquals {
                field: "s".into(),
                value: Value::Float(-0.0),
            }),
            Some(Order::Votes),
        ),
        sel(
            "note",
            Some(Filter::FieldCmp {
                field: "n".into(),
                op: CmpOp::Ne,
                value: Value::Bool(false),
            }),
            Some(Order::Recency),
        ),
        sel(
            "note",
            None,
            Some(Order::Field {
                key: "n".into(),
                desc: true,
            }),
        ),
        sel("note", None, Some(Order::Similarity)),
        sel("note", None, Some(Order::Salience)),
        sel("note", None, Some(Order::Score)),
        sel("note", None, Some(Order::Feedback)),
        Statement::Match(path(None)),
        Statement::Closure(path(Some(1))),
        Statement::Forget { id: a1.clone() },
        Statement::Memory {
            name: "dreams".into(),
        },
        Statement::ContextReset,
        Statement::MatchCount(path(None)),
        Statement::PruneHistory,
        Statement::Snapshot(Box::new(SnapshotState {
            records: BTreeMap::from([(a1.clone(), sample_rec.clone())]),
            edges: vec![sample_edge.clone()],
            vector_dims: BTreeMap::from([("alpha".into(), 4)]),
            clock: 9,
            memories: BTreeMap::from([("m".into(), Store::default())]),
            tables: BTreeMap::from([("alpha".into(), Some(4))]),
        })),
        Statement::HistorySince(42),
    ]
}

fn statements_json() -> String {
    let entries: Vec<serde_json::Value> = statement_samples()
        .into_iter()
        .map(|s| {
            let hexed = hex(&postcard::to_allocvec(&s).expect("postcard sample"));
            // SnapshotState maps records by RecordId — serde_json only
            // accepts string keys, so the JSON twin is null for tag 11;
            // its semantics are pinned by pruned.nql / rich.nql history.
            let json = if stmt_name(&s) == "Snapshot" {
                serde_json::Value::Null
            } else {
                serde_json::to_value(&s).expect("json sample")
            };
            serde_json::json!({
                "name": stmt_name(&s),
                "tag": stmt_tag(&s),
                "hex": hexed,
                "json": json,
            })
        })
        .collect();
    format!("{}\n", serde_json::to_string_pretty(&entries).unwrap())
}

// ---------------------------------------------------------------------------
// Manifest + fixture I/O
// ---------------------------------------------------------------------------

fn update_mode() -> bool {
    std::env::var_os("UPDATE_FIXTURES").is_some()
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../spec/fixtures/v4")
}

fn manifest_json() -> serde_json::Value {
    let fixtures: Vec<serde_json::Value> = fixture_stores()
        .into_iter()
        .map(|(name, store)| {
            let bytes = encode_v4(&store).expect("encode");
            let dir = parse_dir(&bytes).expect("dir");
            let sections: Vec<serde_json::Value> = dir
                .iter()
                .map(|e| {
                    serde_json::json!({
                        "tag": e.tag,
                        "name": tag_name(e.tag),
                        "offset": e.off,
                        "len": e.len,
                        "crc32": e.crc,
                    })
                })
                .collect();
            serde_json::json!({
                "file": format!("{name}.nql"),
                "bytes": bytes.len(),
                "tables": store.tables.len(),
                "records": store.records.len(),
                "edges": store.edges.len(),
                "memories": store.memories.len(),
                "history": store.history.len(),
                "sections": sections,
            })
        })
        .collect();
    serde_json::json!({
        "note": "generated by nqlite/tests/v4_fixtures.rs (UPDATE_FIXTURES=1); spec/file-format.md §5; never hand-edit",
        "fixtures": fixtures,
        "statements": { "file": "statements.json", "count": statement_samples().len() },
    })
}

fn write_or_verify(path: &Path, bytes: &[u8], label: &str) {
    if update_mode() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    } else {
        let committed = std::fs::read(path).unwrap_or_else(|e| {
            panic!(
                "{label}: {} unreadable ({e}) — run with UPDATE_FIXTURES=1 once",
                path.display()
            )
        });
        assert_eq!(
            committed, bytes,
            "{label} drifted from the canonical encode — spec change? regenerate + bump .spec-pin"
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// §5.1 arithmetic, pinned: an empty store is exactly 144 bytes —
/// 24-byte header + 3×32 directory + three 8-byte required sections.
#[test]
fn empty_layout_is_pinned() {
    let bytes = encode_v4(&Store::default()).unwrap();
    let dir = parse_dir(&bytes).unwrap();
    let named: Vec<(u32, u64, u64)> = dir.iter().map(|e| (e.tag, e.off, e.len)).collect();
    assert_eq!(
        named,
        [(T_TABLES, 120, 8), (T_RECORDS, 128, 8), (T_CLOCK, 136, 8)]
    );
    assert_eq!(bytes.len(), 144);
}

/// Canonical RecordId order (§5.4): table bytes → `Num` < `Str` rank →
/// u64 / byte order — including the traps: digit-first string ids sort
/// as strings (`007` < `10` < `2`), never numerically, and numeric ids
/// of any value rank below every string id.
#[test]
fn canonical_record_order() {
    let mut s = Store::default();
    s.tables.insert("b".into(), None);
    s.tables.insert("a".into(), None);
    for (table, id) in [
        ("a", Id::Num(5)),
        ("a", Id::Str("007".into())),
        ("a", Id::Str("10".into())),
        ("a", Id::Str("2".into())),
        ("a", Id::Num(1)),
        ("b", Id::Num(0)),
    ] {
        s.insert(rec(table, id, vec![("k", Value::Int(1))], None, 1));
    }
    let bytes = encode_v4(&s).unwrap();
    let dec = decode_v4(&bytes).unwrap();
    let ids: Vec<String> = dec.records.keys().map(|k| k.to_string()).collect();
    assert_eq!(ids, ["a:1", "a:5", "a:007", "a:10", "a:2", "b:0"]);
    assert_eq!(encode_v4(&dec).unwrap(), bytes, "round-trip byte-exact");
}

/// The core gate: every fixture decodes and re-encodes byte-identically,
/// and equals the committed bytes (or is regenerated under UPDATE_FIXTURES).
/// Structural `PartialEq` is asserted where the types permit it: `rich`
/// carries a NaN (not `PartialEq`-reflexive), and `pruned` embeds stores
/// inside `Statement::Snapshot`, whose `Store.tables` is `#[serde(skip)]`
/// — dropped by postcard everywhere (v3 included) and rebuilt from the
/// inner history at replay (issue #133). Byte identity is the real gate.
#[test]
fn fixtures_roundtrip_and_match_files() {
    for (name, store) in fixture_stores() {
        let bytes = encode_v4(&store).expect(name);
        write_or_verify(&fixtures_dir().join(format!("{name}.nql")), &bytes, name);
        let dec = decode_v4(&bytes).unwrap_or_else(|e| panic!("{name}: decode: {e}"));
        assert_eq!(
            encode_v4(&dec).unwrap(),
            bytes,
            "{name}: decode → re-encode is not byte-exact"
        );
        if matches!(name, "empty" | "plain") {
            assert_eq!(dec, store, "{name}: structural round-trip");
        }
    }
}

/// Semantic spot-checks on the engine-built fixtures.
#[test]
fn fixture_semantics() {
    let by_name: BTreeMap<&str, Store> = fixture_stores().into_iter().collect();

    let plain = decode_v4(&encode_v4(&by_name["plain"]).unwrap()).unwrap();
    assert_eq!(plain.records.len(), 2);
    assert!(plain.edges.is_empty() && plain.history.is_empty() && plain.memories.is_empty());
    assert_eq!(plain.tables.get("note"), Some(&None));
    assert_eq!(plain.clock, 4);

    let rich = decode_v4(&encode_v4(&by_name["rich"]).unwrap()).unwrap();
    // Root records: scratch:1 was forgotten; tiny/d2 live under memories.
    let keys: Vec<String> = rich.records.keys().map(|k| k.to_string()).collect();
    assert_eq!(
        keys,
        [
            "alpha:1",
            "alpha:5",
            "alpha:007",
            "alpha:10",
            "alpha:2",
            "note:9",
            "zebra:3",
            "zebra:4"
        ]
    );
    let nan_body = &rich.records[&RecordId::new("note", Id::Num(9))].body;
    assert!(matches!(nan_body["nan"], Value::Float(f) if f.is_nan()));
    assert_eq!(
        rich.records[&RecordId::new("alpha", Id::Num(1))].embedding,
        Some(vec![0.5, 0.25, 0.5, 1.0])
    );
    assert_eq!(
        rich.records[&RecordId::new("zebra", Id::Num(4))].embedding,
        None
    );
    // Append order preserved (≠ sorted — see rich_store's own assert).
    assert_eq!(rich.edges.len(), 4);
    assert_eq!(rich.edges[2].name, "likes");
    assert_eq!(rich.history.len(), 18);
    assert_eq!(rich.tables.len(), 4);
    assert_eq!(rich.vector_dims.get("alpha"), Some(&4));
    assert!(!rich.vector_dims.contains_key("note"));
    // Memories: root-level `dreams` (engine), `d2` hand-nested inside it
    // for §5.2 recursion coverage.
    assert_eq!(rich.memories.len(), 1);
    let dreams = &rich.memories["dreams"];
    assert_eq!(dreams.records.len(), 2);
    assert_eq!(dreams.history.len(), 4);
    assert_eq!(dreams.tables.len(), 1);
    assert_eq!(dreams.edges.len(), 1);
    assert_eq!(dreams.memories.len(), 1);
    let d2 = &dreams.memories["d2"];
    assert_eq!(d2.records.len(), 1);
    assert_eq!(d2.history.len(), 2);
    assert_eq!(d2.tables.len(), 1);

    let pruned = decode_v4(&encode_v4(&by_name["pruned"]).unwrap()).unwrap();
    assert_eq!(pruned.records.len(), 2);
    assert!(
        matches!(pruned.history.last(), Some((_, Statement::Snapshot(_)))),
        "pruned history ends in a Snapshot"
    );
    // Declarations retained at their original timestamps (issues #95/#89).
    assert!(pruned
        .history
        .iter()
        .any(|(ts, s)| *ts == 1
            && matches!(s, Statement::CreateTable { table, .. } if table == "note")));
    // The snapshot embeds the nested memory, itself pruned to its own
    // declaration + snapshot pair (recursive Snapshot states).
    let snapshot = pruned
        .history
        .iter()
        .find_map(|(_, s)| match s {
            Statement::Snapshot(st) => Some(st),
            _ => None,
        })
        .expect("snapshot present");
    let archive = &snapshot.memories["archive"];
    assert_eq!(archive.records.len(), 1);
    assert!(
        matches!(archive.history.last(), Some((_, Statement::Snapshot(_)))),
        "nested memory history is pruned too"
    );
    assert_eq!(pruned.memories["archive"].records.len(), 1);
}

/// §5.6: the HISTORY payload is byte-identical to v3's history tail —
/// the same postcard call `storage.rs` writes (conversion = adopt verbatim).
#[test]
fn history_section_is_v3_tail_verbatim() {
    for (name, store) in fixture_stores() {
        let bytes = encode_v4(&store).unwrap();
        let dir = parse_dir(&bytes).unwrap();
        if store.history.is_empty() {
            assert!(
                !dir.iter().any(|e| e.tag == T_HISTORY),
                "{name}: empty history ⇒ no section"
            );
            continue;
        }
        let (_, payload) = section(&bytes, &dir, T_HISTORY).unwrap();
        let tail = postcard::to_allocvec(&store.history).unwrap();
        assert_eq!(
            payload, tail,
            "{name}: HISTORY must equal postcard(history)"
        );
    }
}

/// Recompute a section's CRC after a deliberate payload mutation (used to
/// reach the semantic checks behind the integrity check).
fn fix_crc(buf: &mut [u8], dir: &[DirEnt], tag: u32) {
    let idx = dir.iter().position(|e| e.tag == tag).unwrap();
    let e = &dir[idx];
    let c = crc32(&buf[e.off as usize..(e.off + e.len) as usize]);
    let crc_field = 24 + 32 * idx + 24;
    buf[crc_field..crc_field + 4].copy_from_slice(&c.to_le_bytes());
}

/// §5.1 loud-failure contract: every malformed container is rejected with a
/// specific error — never a partial load.
#[test]
fn reader_rejects_malformed() {
    let base = encode_v4(&plain_store()).unwrap();
    let dir = parse_dir(&base).unwrap();
    let err = |b: &[u8]| decode_v4(b).unwrap_err();
    let rec_off = dir.iter().find(|e| e.tag == T_RECORDS).unwrap().off as usize;

    let mut b = base.clone();
    b[0] ^= 0x01;
    assert!(err(&b).contains("magic"));

    let mut b = base.clone();
    b[8..12].copy_from_slice(&5u32.to_le_bytes());
    assert!(err(&b).contains("unsupported format version 5"));

    let mut b = base.clone();
    b[12] = 1;
    assert!(err(&b).contains("flags"));

    let mut b = base.clone();
    b.truncate(30);
    assert!(err(&b).contains("truncated"));

    // CRC: flip the last byte (inside the last section's payload).
    let mut b = base.clone();
    let last = b.len() - 1;
    b[last] ^= 0xff;
    assert!(err(&b).contains("CRC mismatch in section"));

    // Required section missing: rebuild the directory with 2 entries, keep
    // TABLES/RECORDS payloads, drop CLOCK (file then ends at RECORDS).
    let mut b = Vec::new();
    b.extend_from_slice(&base[0..16]);
    b.extend_from_slice(&2u32.to_le_bytes());
    b.extend_from_slice(&base[20..24]);
    b.extend_from_slice(&base[24..88]); // two directory entries
    b.extend_from_slice(&vec![0u8; dir[0].off as usize - b.len()]);
    b.extend_from_slice(&base[dir[0].off as usize..(dir[1].off + dir[1].len) as usize]);
    assert!(err(&b).contains("required section CLOCK"));

    // Unknown tag.
    let mut b = base.clone();
    b[24..28].copy_from_slice(&9u32.to_le_bytes());
    assert!(err(&b).contains("unknown section tag 9"));

    // Duplicate / non-ascending tag.
    let mut b = base.clone();
    b[56..60].copy_from_slice(&1u32.to_le_bytes());
    assert!(err(&b).contains("ascending"));

    // Out-of-bounds length (last entry's len field = 24 + 32·last + 16).
    let last_i = dir.len() - 1;
    let len_field = 24 + 32 * last_i + 16;
    let mut b = base.clone();
    b[len_field..len_field + 8].copy_from_slice(&(1u64 << 40).to_le_bytes());
    assert!(err(&b).contains("out of bounds"));

    // Semantic: record table_idx beyond TABLES — patch + fix the CRC so the
    // payload check passes and the cross-reference check fires.
    let mut b = base.clone();
    b[rec_off + 8..rec_off + 12].copy_from_slice(&9u32.to_le_bytes());
    fix_crc(&mut b, &dir, T_RECORDS);
    assert!(err(&b).contains("table_idx 9"));

    // Semantic: string id without a STRINGS section.
    let mut b = base.clone();
    b[rec_off + 12] = 1; // id_kind = 1
    fix_crc(&mut b, &dir, T_RECORDS);
    assert!(err(&b).contains("STRINGS"));
}

/// `statements.json` — all 13 tags as (serde JSON, postcard hex) twins.
/// The hex side is what other implementations re-encode and compare; the
/// JSON side pins the *meaning* (variant names, field names, tag numbers).
#[test]
fn statements_oracle_matches_files() {
    let text = statements_json();
    write_or_verify(
        &fixtures_dir().join("statements.json"),
        text.as_bytes(),
        "statements",
    );
    // Sanity: coverage of every §5.7 Statement tag, 0..=12.
    let tags: Vec<u32> = statement_samples().iter().map(stmt_tag).collect();
    for t in 0..=12u32 {
        assert!(tags.contains(&t), "no sample for Statement tag {t}");
    }
    // JSON twin round-trips through serde itself (guards drift between the
    // hex and json fields of each entry).
    let parsed: Vec<serde_json::Value> = serde_json::from_str(&text).unwrap();
    assert_eq!(parsed.len(), statement_samples().len());
    for (entry, stmt) in parsed.iter().zip(statement_samples()) {
        assert_eq!(entry["tag"], serde_json::json!(stmt_tag(&stmt)));
        assert_eq!(entry["name"], serde_json::json!(stmt_name(&stmt)));
        assert_eq!(
            entry["hex"],
            serde_json::json!(hex(&postcard::to_allocvec(&stmt).unwrap()))
        );
        if stmt_name(&stmt) == "Snapshot" {
            assert!(
                entry["json"].is_null(),
                "Snapshot json twin is null by design"
            );
        } else {
            assert_eq!(entry["json"], serde_json::to_value(&stmt).unwrap());
        }
    }
}

/// `manifest.json` — independent section-table/count expectations for the
/// consuming implementation (offsets, CRCs, presence rules).
#[test]
fn manifest_matches_fixtures() {
    let text = format!(
        "{}\n",
        serde_json::to_string_pretty(&manifest_json()).unwrap()
    );
    write_or_verify(
        &fixtures_dir().join("manifest.json"),
        text.as_bytes(),
        "manifest",
    );
}
