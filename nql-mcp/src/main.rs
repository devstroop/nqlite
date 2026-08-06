//! nql-mcp binary: stdio MCP server over an nqlite database.
//!
//! ```
//! nql-mcp                 # in-memory database
//! nql-mcp --db memory.nql # persistent single-file store
//! ```
//!
//! Speaks the Model Context Protocol over stdio (the standard transport for
//! local agent hosts). Deterministic and zero-LLM: every tool call is a pure
//! function of the store state.

use std::io::{self, Write};

use nql_mcp::NqlMcp;
use rmcp::{transport::stdio, ServiceExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let db_path = match args.iter().position(|a| a == "--db" || a == "-d") {
        Some(i) => args.get(i + 1).cloned(),
        None => None,
    };

    let service = match &db_path {
        Some(path) => NqlMcp::open(path)?,
        None => NqlMcp::new(),
    };

    eprintln!("nql-mcp: serving nqlite over MCP stdio");
    let _ = io::stdout().flush();
    let server = service.serve(stdio()).await?;
    server.waiting().await?;
    Ok(())
}
