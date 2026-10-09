//! Prototype for issue #167 — v5 history-tail compression (HC0), the codec
//! sketched in `spec/file-format.md` §7.
//!
//! Reads a v3/v4 store file, slices out the history tail (the lazy part —
//! §1 / §5.6), compresses it with the HC0 block framing + fixed-parameter
//! LZSS, and proves the two properties the design note requires:
//!
//! 1. **Roundtrip** — `decompress(compress(tail)) == tail` (byte-exact).
//! 2. **Byte-identical re-encode** — `compress(decompress(c)) == c`, i.e.
//!    the codec is a pure function of its input (nql-migrate verify rule).
//!
//! HC0 is a zero-dependency, fully-specified LZSS: window 32768, min match
//! 3, max match 64, single-slot hash table (15-bit, multiplicative hash,
//! overwritten at every position incl. match interiors), LSB-first control
//! bytes, greedy first-candidate matches. No library dependence — any
//! implementation that follows §7's parameters produces identical bytes.
//!
//! Build a store to measure (profile-equivalent, exp08 corpus): see the
//! "Reproduce" block in `spec/file-format.md` §7.
//!
//! Usage: `history_compress_proto <store.ndb>`

use std::env;
use std::fs;

// ---- HC0 parameters (spec/file-format.md §7 — pin these exactly) ----------
const WINDOW: usize = 32768;
const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 64;
const RAW_BLOCK: usize = 65_536;
const HASH_SLOTS: usize = 32_768; // 15-bit table

/// 3-byte multiplicative hash → table slot. Fixed shift, fixed constant:
/// every implementation computes the same slot for the same bytes.
fn h3(b0: u8, b1: u8, b2: u8) -> usize {
    let k = ((b0 as u32) << 16) | ((b1 as u32) << 8) | b2 as u32;
    (k.wrapping_mul(0x9E37_79B1) >> 17) as usize
}

/// Greedy LZSS over one raw block (§7.3). Emits control bytes LSB-first:
/// bit `t` of a control byte governs token `t` — 0 = literal (1 byte),
/// 1 = match (`u16 LE dist-1`, `u8 len`). Table entries store `pos + 1`
/// (0 = empty); every position — including interiors of matches — is
/// inserted, so the parse is a pure function of the input bytes.
fn lzss_encode(raw: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    let mut table = vec![0u32; HASH_SLOTS];
    let mut ctrl_index = 0usize;
    let mut tok = 8usize; // forces a fresh control byte before token 0
    let mut i = 0usize;

    while i < raw.len() {
        if tok == 8 {
            ctrl_index = out.len();
            out.push(0);
            tok = 0;
        }

        // Candidate: single slot for the 3-gram at `i`.
        let mut len = 0usize;
        let mut dist = 0usize;
        if i + MIN_MATCH <= raw.len() {
            let slot = table[h3(raw[i], raw[i + 1], raw[i + 2])];
            if slot != 0 {
                let d = (i + 1) - slot as usize;
                if d <= WINDOW {
                    let max = MAX_MATCH.min(raw.len() - i);
                    let mut l = 0usize;
                    while l < max && raw[i + l] == raw[i + l - d] {
                        l += 1;
                    }
                    if l >= MIN_MATCH {
                        len = l;
                        dist = d;
                    }
                }
            }
        }

        if len >= MIN_MATCH {
            out[ctrl_index] |= 1 << tok;
            out.extend_from_slice(&((dist - 1) as u16).to_le_bytes());
            out.push(len as u8);
        } else {
            out.push(raw[i]);
        }

        // Insert every 3-gram that starts inside the token's span.
        let span = if len >= MIN_MATCH { len } else { 1 };
        for j in i..i + span {
            if j + MIN_MATCH <= raw.len() {
                table[h3(raw[j], raw[j + 1], raw[j + 2])] = (j + 1) as u32;
            }
        }
        i += span;
        tok += 1;
    }
    out
}

/// Inverse of [`lzss_encode`]. `raw_len` (from the block header) bounds the
/// output; control bits past the last token are zero padding and ignored.
fn lzss_decode(mut comp: &[u8], raw_len: usize) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(raw_len);
    let mut ctrl = 0u8;
    let mut bits_left = 0usize;
    while out.len() < raw_len {
        if bits_left == 0 {
            ctrl = comp[0];
            comp = &comp[1..];
            bits_left = 8;
        }
        let is_match = ctrl & 1 == 1;
        ctrl >>= 1;
        bits_left -= 1;
        if is_match {
            let dist = usize::from(u16::from_le_bytes([comp[0], comp[1]])) + 1;
            let len = comp[2] as usize;
            comp = &comp[3..];
            debug_assert!(dist >= 1 && dist <= out.len());
            for _ in 0..len {
                let b = out[out.len() - dist];
                out.push(b);
            }
        } else {
            out.push(comp[0]);
            comp = &comp[1..];
        }
    }
    out
}

/// HC0 container: `raw` is split into ≤65536-byte blocks; each block is
/// `[raw_len u32 LE][comp_len u32 LE][crc32(raw) u32 LE][flags u8][payload]`
/// with `flags & 1 = 1` when the payload is LZSS-compressed (`comp_len ==
/// raw_len`, flag 0 = stored verbatim — incompressible blocks).
fn hc0_compress(raw: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    for block in raw.chunks(RAW_BLOCK) {
        let comp = lzss_encode(block);
        let (flags, payload): (u8, &[u8]) = if comp.len() < block.len() {
            (1, &comp)
        } else {
            (0, block)
        };
        out.extend_from_slice(&(block.len() as u32).to_le_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&crc32fast::hash(block).to_le_bytes());
        out.push(flags);
        out.extend_from_slice(payload);
    }
    out
}

fn hc0_decompress(comp: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    let mut p = 0usize;
    while p < comp.len() {
        let raw_len = u32::from_le_bytes(comp[p..p + 4].try_into().unwrap()) as usize;
        let payload_len = u32::from_le_bytes(comp[p + 4..p + 8].try_into().unwrap()) as usize;
        let want_crc = u32::from_le_bytes(comp[p + 8..p + 12].try_into().unwrap());
        let flags = comp[p + 12];
        let payload = &comp[p + 13..p + 13 + payload_len];
        p += 13 + payload_len;
        let block = if flags & 1 == 1 {
            lzss_decode(payload, raw_len)
        } else {
            payload.to_vec()
        };
        assert_eq!(block.len(), raw_len, "HC0 block raw_len mismatch");
        assert_eq!(crc32fast::hash(&block), want_crc, "HC0 block CRC mismatch");
        out.extend_from_slice(&block);
    }
    out
}

/// Slice the history tail out of a store file (v3 §1, v4 §5.6/§5.2 tag 8).
fn history_tail(file: &[u8]) -> &[u8] {
    assert_eq!(&file[..8], b"NQLITE01", "bad magic");
    let version = u32::from_le_bytes(file[8..12].try_into().unwrap());
    match version {
        3 => {
            let core_len = u64::from_le_bytes(file[16..24].try_into().unwrap()) as usize;
            &file[24 + core_len..]
        }
        4 => {
            let count = u32::from_le_bytes(file[16..20].try_into().unwrap()) as usize;
            for i in 0..count {
                let e = &file[24 + i * 32..24 + (i + 1) * 32];
                let tag = u32::from_le_bytes(e[0..4].try_into().unwrap());
                if tag == 8 {
                    let off = u64::from_le_bytes(e[8..16].try_into().unwrap()) as usize;
                    let len = u64::from_le_bytes(e[16..24].try_into().unwrap()) as usize;
                    return &file[off..off + len];
                }
            }
            &[] // absent HISTORY section ⇒ empty persisted history (§5.1)
        }
        v => panic!("prototype handles v3/v4 stores (got v{v})"),
    }
}

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: history_compress_proto <store.ndb>");
        std::process::exit(2)
    });
    let file = fs::read(&path).expect("read store");
    let tail = history_tail(&file);

    let t0 = std::time::Instant::now();
    let comp = hc0_compress(tail);
    let compress_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = std::time::Instant::now();
    let back = hc0_decompress(&comp);
    let decompress_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // Property 1: roundtrip.
    assert_eq!(back, tail, "HC0 roundtrip mismatch");
    // Property 2: byte-identical re-encode (purity / nql-migrate verify rule).
    let re = hc0_compress(&back);
    assert_eq!(re, comp, "HC0 re-encode is not byte-identical");

    let ratio = tail.len() as f64 / comp.len().max(1) as f64;
    println!("store          : {path}");
    println!("history tail   : {:>12} B", tail.len());
    println!("HC0 compressed : {:>12} B   ratio {ratio:.3}x", comp.len());
    println!(
        "compress        : {compress_ms:>8.1} ms  ({:.2} MB/s)",
        tail.len() as f64 / 1e6 / (compress_ms / 1e3)
    );
    println!(
        "decompress      : {decompress_ms:>8.1} ms  ({:.2} MB/s)",
        tail.len() as f64 / 1e6 / (decompress_ms / 1e3)
    );
    println!("roundtrip       : byte-identical OK   re-encode: byte-identical OK");
    assert!(
        ratio >= 2.0,
        "issue #167 target: >=2x on the profile store (got {ratio:.3}x)"
    );
    println!("target >=2x     : OK");
}
