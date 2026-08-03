//! nql CLI: interactive REPL + `--script` runner.

use std::env;
use std::fs;
use std::io::{self, BufRead, BufWriter, Write};

use nql_cli::Session;

const HELP: &str = r#"nql — deterministic neural query REPL (zero-LLM)

Commands:
  :help          this help
  :quit | :exit  leave the REPL
  :clear         start a fresh empty database
  :store         dump current records, edges, and vector dims

Anything else is parsed as nql (multi-statement with ';' separators).
"#;

fn repl() -> io::Result<()> {
    let stdin = io::stdin();
    let mut session = Session::new(Box::new(BufWriter::new(io::stdout())));

    session.writeln(&format!(
        "nql {} — type :help for help, :quit to exit",
        env!("CARGO_PKG_VERSION")
    ))?;
    session.flush()?;

    for line in stdin.lock().lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match trimmed {
            ":help" => write!(session.writer(), "{HELP}")?,
            ":quit" | ":exit" => break,
            ":clear" => {
                session.clear();
                session.writeln("cleared")?;
            }
            ":store" => {
                session.dump_store()?;
            }
            other => {
                session.run(other)?;
            }
        }
        session.flush()?;
    }
    session.writeln("")?;
    session.flush()?;
    io::stdout().flush()?;
    Ok(())
}

fn run_script(path: &str) -> io::Result<()> {
    let src = fs::read_to_string(path)?;
    let mut session = Session::new(Box::new(BufWriter::new(io::stdout())));
    session.run(&src)?;
    session.flush()?;
    io::stdout().flush()?;
    Ok(())
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    match args.as_slice() {
        [_, flag, path] if flag == "--script" || flag == "-s" => run_script(path),
        [_] => repl(),
        _ => {
            eprintln!("usage: nql [--script FILE]");
            std::process::exit(2);
        }
    }
}
