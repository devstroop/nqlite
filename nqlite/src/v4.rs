//! Format-v4 container codec — spec/file-format.md §5 (adopted target).
//!
//! The sectioned container (header/section table, alignment, CRC32,
//! iff-non-empty presence) plus §5.6's history tail — payloads are today's
//! postcard dialect (§5.7 = "exactly as the reference implementation").
//! This module was lifted from the golden-fixture generator
//! (`tests/v4_fixtures.rs`, still the byte oracle) and now also powers the
//! `nql-migrate` v3→v4 conversion tool: `Store` in, canonical v4 bytes out.

use std::collections::BTreeMap;

use nql_ir::{Id, Record, RecordId, RelationEdge, Statement, Store, Value};

pub const MAGIC: &[u8; 8] = b"NQLITE01";
pub const V4: u32 = 4;
pub const T_TABLES: u32 = 1;
pub const T_RECORDS: u32 = 2;
pub const T_STRINGS: u32 = 3;
pub const T_EMBEDS: u32 = 4;
pub const T_EDGES: u32 = 5;
pub const T_MEMORIES: u32 = 6;
pub const T_CLOCK: u32 = 7;
pub const T_HISTORY: u32 = 8;

pub const HEADER_LEN: u64 = 24;
pub const DIR_ENTRY_LEN: u64 = 32;

fn alignment(tag: u32) -> u64 {
    match tag {
        T_STRINGS | T_EMBEDS | T_HISTORY => 4096,
        _ => 8,
    }
}

fn align_up(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

pub fn tag_name(tag: u32) -> &'static str {
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

pub fn crc32(payload: &[u8]) -> u32 {
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

pub fn hex(bytes: &[u8]) -> String {
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

pub fn encode_store(store: &Store) -> Result<Vec<u8>, String> {
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
            let blob = encode_store(sub)?;
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

pub struct DirEnt {
    pub tag: u32,
    pub off: u64,
    pub len: u64,
    pub crc: u32,
}

pub fn parse_dir(bytes: &[u8]) -> Result<Vec<DirEnt>, String> {
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

pub fn section<'a>(bytes: &'a [u8], dir: &[DirEnt], tag: u32) -> Result<(u64, &'a [u8]), String> {
    let e = dir
        .iter()
        .find(|e| e.tag == tag)
        .ok_or_else(|| format!("v4: section {} absent", tag_name(tag)))?;
    Ok((e.off, &bytes[e.off as usize..(e.off + e.len) as usize]))
}

pub fn decode_store(bytes: &[u8]) -> Result<Store, String> {
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
            let sub = decode_store(blob)?;
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
