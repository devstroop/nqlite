# nql-server

A minimal, deterministic line-protocol server for `nqlite`. It exposes nql
execution (via `nql` parsing/analysis + the `nqlite` engine) over either a
plain TCP listener or plain stdio, using a simple one-program-per-line text
protocol.

The server is **zero-LLM and deterministic**: no randomness, no wall-clock, no
timestamps in any response; output ordering comes from the engine's BTree
scans and stable sorts, ids are rendered as-is, and scores use 4 decimal
places. It never panics on bad input — a malformed program yields an `ERR`
line and execution continues.

## Protocol

Each input line is one nql program (`;`-separated statements). The response is:

- **one line per `SELECT` result**, formatted as
  `SELECT <table> (<N> rows): <id> score=<s.4> {fields}; <id> ...` — rows in
  deterministic order, scores to 4dp, fields in BTree order;
- followed by a final **`OK`**, or a single **`ERR <message>`** on the first
  parse/analyze/execute error (and the line is aborted; the server never
  crashes, and the shared database keeps its prior committed state for that
  line's failed parse/analysis).

A single shared `Database` persists across **all** lines and connections, so a
session may `CREATE TABLE`, `INSERT`, and `SELECT` across separate lines.
Because the nql `Analyzer` validates per-plan, the server remembers declared
tables and re-validates them across lines (dimension contracts still enforced).

## Running

### Mode A — TCP (default)

```sh
PORT=7878 nql-server
nql-server        # uses PORT env or defaults to 7878
```

Listens on `127.0.0.1:PORT` (one line in ⇄ one line out), holding one database
that persists across lines and across connections. Ctrl-C terminates cleanly
(default SIGINT, no panic).

```sh
$ printf '%s\n' \
  'CREATE TABLE person VECTOR<f32, 2>' \
  'INSERT INTO person:1 { name: "alice" } EMBED [1.0, 0.0]' \
  'INSERT INTO person:2 { name: "bob" } EMBED [0.9, 0.1]' \
  'SELECT * FROM person WHERE vector::similarity(embedding, [1.0, 0.0]) AND k = 2' \
  | nc 127.0.0.1 7878
OK
OK
OK
SELECT person (2 rows): person:1 score=1.0000 {name="alice"}; person:2 score=0.9939 {name="bob"}
OK
```

### Mode B — stdio

```bash
nql-server --stdio
```

Reads programs from stdin, writes responses to stdout, flushing after every
line. Purely deterministic and transport-free — the base for MCP/agent use
later. Ctrl-D (EOF) exits cleanly.

```
$ nql-server --stdio
CREATE TABLE t VECTOR<f32, 2>
OK
INSERT INTO t:1 { name: "a" } EMBED [1.0, 0.0]
OK
INSERT INTO t:2 { name: "b" } EMBED [0.9, 0.1]
OK
SELECT * FROM t WHERE vector::similarity(embedding, [1.0, 0.0]) AND k = 2
SELECT t (2 rows): t:1 score=1.0000 {name="a"}; t:2 score=0.9939 {name="b"}
OK
THIS IS NOT NQL
ERR parse error at 1:1: expected a statement keyword (CREATE, INSERT, RELATE, SELECT, FORGET), found identifier `THIS`
SELECT * FROM t
SELECT t (2 rows): t:1 score=0.0000 {name="a"}; t:2 score=0.0000 {name="b"}
OK
^D
```

> Notes on the error example: `SELECT * FROM t` without a kNN clause has no
> score operator, so every row's score is `0.0000` (engine semantics). The
> exact `ERR` text comes from the parser/analyzer/engine `Display` impls.

## Development

```bash
cargo build --release -p nql-server
cargo test    -p nql-server     # unit tests for the line-handling logic
cargo clippy  -p nql-server --all-targets -- -D warnings
```

The line-handling core is `Server::handle_line(&mut self, line: &str) -> String`
in `src/lib.rs`; the TCP/stdio transports live in `src/main.rs`.