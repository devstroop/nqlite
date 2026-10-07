//! nql-server binary: deterministic line-protocol server over nqlite.
//!
//! Two modes:
//! * Default — TCP: listen on `127.0.0.1:PORT` (`PORT` env or `7878`), one nql
//!   program per line, one response per line. The shared `Database` persists
//!   across lines and connections.
//! * `--stdio` — read lines from stdin, write responses to stdout. Purely
//!   deterministic and transport-free (the base for MCP/agent use later).
//!
//! Both modes accept `--db <path>` to serve a persistent single-file store
//! (same semantics as `nql-cli --db`); without it the server is in-memory and
//! everything is lost on exit. The store's single-writer lock (issue #84)
//! means one server process owns the file — a second `--db` opener fails fast.

use std::env;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::net::TcpListener;

use nql_server::Server;

const DEFAULT_PORT: &str = "7878";

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    // `--db <path>` serves a persistent store (issue #89); without it the
    // session is in-memory (deterministic, throwaway).
    let db_path = args
        .iter()
        .position(|a| a == "--db" || a == "-d")
        .and_then(|i| args.get(i + 1).cloned());
    let server = match &db_path {
        Some(p) => match Server::open(p) {
            Ok(s) => s,
            Err(e) => {
                // Same clean `error: ...` convention as nql-cli (issue #84).
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        },
        None => Server::new(),
    };

    if args.iter().any(|a| a == "--stdio") {
        run_stdio(server)
    } else {
        let port = env::var("PORT").unwrap_or_else(|_| DEFAULT_PORT.to_string());
        run_tcp(server, &port)
    }
}

/// Mode B: line-based, transport-free. Read, respond, flush per line. EOF
/// (Ctrl-D) returns cleanly; Ctrl-C is a default SIGINT termination (no panic).
fn run_stdio(mut server: Server) -> io::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    for line in stdin.lock().lines() {
        let line = line?;
        writeln!(out, "{}", server.handle_line(&line))?;
        out.flush()?;
    }
    out.flush()?;
    Ok(())
}

/// Mode A: TCP listener on 127.0.0.1. One shared `Server` (database) persists
/// across every connection. Deterministic for sequential/single clients.
fn run_tcp(mut server: Server, port: &str) -> io::Result<()> {
    let addr = format!("127.0.0.1:{port}");
    let listener = TcpListener::bind(&addr)?;
    eprintln!("nql-server listening on {addr}");

    loop {
        let (stream, _peer) = listener.accept()?;
        // Serve one connection: read per line, respond per line. A client
        // disconnect (read or write error) just ends this connection; the
        // shared database and listener keep going.
        let reader = BufReader::new(stream.try_clone()?);
        let mut out = BufWriter::new(stream);
        for line in reader.lines() {
            let Ok(line) = line else { break };
            let resp = server.handle_line(&line);
            if writeln!(out, "{resp}").is_err() || out.flush().is_err() {
                break;
            }
        }
        let _ = out.flush();
    }
}
