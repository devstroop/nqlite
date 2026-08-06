#!/usr/bin/env python3
"""Cross-DB benchmark harness (ISSUE-24).

Generates a FIXED synthetic corpus (xorshift64*, seed 42, dim-8 vectors,
8-word vocabulary — byte-identical to nql-bench's corpus), then benchmarks:

  - nqlite      — via `cargo run -q -p nql-bench` (always runs)
  - sqlite-vec  — via the `sqlite_vec` python driver (optional)
  - LanceDB     — via the `lancedb` python driver (optional)
  - Chroma      — via the `chromadb` python driver (optional)

Missing drivers are skipped with an honest note — the harness never fails
because a competitor isn't installed. Emits a JSON report and a human
readable side-by-side table.

Usage:
  python3 scripts/bench-compare/bench.py --rows 1000 --knn 50
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

SEED = 42
DIM = 8
VOCAB = ["alpha", "beta", "gamma", "delta", "rust", "query", "memory", "graph"]


def xorshift(state):
    """xorshift64* — must match nql-bench/src/main.rs."""
    x = state
    x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 7
    x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
    x &= 0xFFFFFFFFFFFFFFFF
    return (x * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF, x


def corpus(rows):
    state = SEED
    docs = []
    for i in range(rows):
        emb = []
        for _ in range(DIM):
            state, _ = xorshift(state)
            emb.append((state % 1000) / 1000.0)
        state, w1 = xorshift(state)
        state, w2 = xorshift(state)
        word1 = VOCAB[w1 % len(VOCAB)]
        word2 = VOCAB[w2 % len(VOCAB)]
        docs.append(
            {
                "id": f"doc:{i}",
                "text": f"{word1} {word2}",
                "group": i % 10,
                "embedding": emb,
            }
        )
    return docs


# ---------------------------------------------------------------------------
# nqlite
# ---------------------------------------------------------------------------

def bench_nqlite(rows, knn, repo_root):
    bin_path = shutil.which("nql-bench")
    if bin_path:
        cmd = [bin_path, "--rows", str(rows), "--knn", str(knn), "--seed", str(SEED)]
    else:
        cmd = [
            "cargo", "run", "-q", "-p", "nql-bench", "--",
            "--rows", str(rows), "--knn", str(knn), "--seed", str(SEED),
        ]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)
    if out.returncode != 0:
        return {"db": "nqlite", "error": out.stderr.strip()[-300:]}
    return json.loads(out.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# sqlite-vec
# ---------------------------------------------------------------------------

def bench_sqlite_vec(docs, knn):
    import sqlite3

    try:
        import sqlite_vec
    except ImportError:
        return {"db": "sqlite-vec", "skipped": "driver not installed (pip install sqlite-vec)"}

    con = sqlite3.connect(":memory:")
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.execute("CREATE VIRTUAL TABLE vec_docs USING vec0(embedding float[8]);")
    con.execute("CREATE TABLE docs(id TEXT PRIMARY KEY, text TEXT);")

    t0 = time.perf_counter()
    for d in docs:
        con.execute("INSERT INTO docs(id, text) VALUES (?, ?)", (d["id"], d["text"]))
        con.execute(
            "INSERT INTO vec_docs(rowid, embedding) VALUES (?, ?)",
            (int(d["id"].split(":")[1]), json.dumps(d["embedding"])),
        )
    con.commit()
    ingest_ms = (time.perf_counter() - t0) * 1000.0

    q = docs[0]["embedding"]
    t0 = time.perf_counter()
    for _ in range(knn):
        con.execute(
            "SELECT rowid, distance FROM vec_docs WHERE embedding MATCH ? ORDER BY distance LIMIT 10",
            (json.dumps(q),),
        ).fetchall()
    knn_ms = (time.perf_counter() - t0) * 1000.0
    con.close()
    return {"db": "sqlite-vec", "rows": len(docs), "ingest_ms": r2(ingest_ms), "knn_ms": r2(knn_ms),
            "bm25_ms": "n/a (FTS5 not wired)", "hybrid_ms": "n/a"}


# ---------------------------------------------------------------------------
# LanceDB
# ---------------------------------------------------------------------------

def bench_lance(docs, knn):
    try:
        import lancedb
    except ImportError:
        return {"db": "lancedb", "skipped": "driver not installed (pip install lancedb)"}

    db = lancedb.connect("/tmp/nql-bench-lance")
    if "docs" in db.list_tables():
        db.drop_table("docs")
    import pyarrow as pa

    schema = pa.schema([
        pa.field("embedding", pa.list_(pa.float32(), 8)),
        pa.field("text", pa.string()),
    ])
    tbl = db.create_table("docs", data=[], schema=schema)
    t0 = time.perf_counter()
    tbl.add([{"embedding": d["embedding"], "text": d["text"]} for d in docs])
    ingest_ms = (time.perf_counter() - t0) * 1000.0

    q = docs[0]["embedding"]
    t0 = time.perf_counter()
    for _ in range(knn):
        tbl.search(q).limit(10).to_list()
    knn_ms = (time.perf_counter() - t0) * 1000.0
    db.drop_table("docs")
    return {"db": "lancedb", "rows": len(docs), "ingest_ms": r2(ingest_ms), "knn_ms": r2(knn_ms),
            "bm25_ms": "n/a (no lexical index)", "hybrid_ms": "n/a"}


# ---------------------------------------------------------------------------
# Chroma
# ---------------------------------------------------------------------------

def bench_chroma(docs, knn):
    try:
        import chromadb
    except ImportError:
        return {"db": "chroma", "skipped": "driver not installed (pip install chromadb)"}

    client = chromadb.Client()
    col = client.create_collection("docs")
    t0 = time.perf_counter()
    batch = 100
    for i in range(0, len(docs), batch):
        chunk = docs[i : i + batch]
        col.add(
            ids=[d["id"] for d in chunk],
            documents=[d["text"] for d in chunk],
            embeddings=[d["embedding"] for d in chunk],
        )
    ingest_ms = (time.perf_counter() - t0) * 1000.0

    q = docs[0]["embedding"]
    t0 = time.perf_counter()
    for _ in range(knn):
        col.query(query_embeddings=[q], n_results=10)
    knn_ms = (time.perf_counter() - t0) * 1000.0
    return {"db": "chroma", "rows": len(docs), "ingest_ms": r2(ingest_ms), "knn_ms": r2(knn_ms),
            "bm25_ms": "n/a (no lexical index)", "hybrid_ms": "n/a"}


def r2(x):
    return round(x, 2)


def fmt_cell(x):
    """Right-align a cell, truncating long strings so the table stays aligned."""
    s = str(x)
    if len(s) > 12:
        s = s[:9] + "..."
    return f"{s:>12}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--knn", type=int, default=50)
    ap.add_argument("--json", action="store_true", help="emit JSON report only")
    ap.add_argument("--out", type=str, default=None, help="write JSON report to file")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    docs = corpus(args.rows)

    reports = [bench_nqlite(args.rows, args.knn, repo_root)]
    reports.append(bench_sqlite_vec(docs, args.knn))
    reports.append(bench_lance(docs, args.knn))
    reports.append(bench_chroma(docs, args.knn))

    if args.out:
        Path(args.out).write_text(json.dumps(reports, indent=2))
        print(f"report written to {args.out}")

    if args.json:
        print(json.dumps(reports, indent=2))
        return

    print()
    print(f"Cross-DB benchmark — {args.rows} rows, {args.knn} kNN queries, seed {SEED}")
    print(f"corpus: dim-{DIM} float vectors, 8-word vocabulary (xorshift64* deterministic)")
    print("-" * 78)
    hdr = f"{'db':<12}{'ingest(ms)':>12}{'knn(ms)':>12}{'bm25(ms)':>12}{'hybrid(ms)':>12}"
    print(hdr)
    print("-" * 78)
    for r in reports:
        if "skipped" in r:
            print(f"{r['db']:<12}  — skipped: {r['skipped']}")
            continue
        if "error" in r:
            print(f"{r['db']:<12}  — ERROR: {r['error']}")
            continue
        print(
            f"{r['db']:<12}{r['ingest_ms']:>12}{r['knn_ms']:>12}"
            f"{fmt_cell(r['bm25_ms'])}{fmt_cell(r['hybrid_ms'])}"
        )
    print("-" * 78)
    print("note: 'n/a' = competitor lacks that operator; compare only same-shape cells.")


if __name__ == "__main__":
    main()
