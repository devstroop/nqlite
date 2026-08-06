//! nql CLI: interactive REPL + `--script` runner.

use std::env;
use std::fs;
use std::io::{self, BufRead, BufWriter, Write};

use nql_cli::Session;

const HELP: &str = r#"nql — deterministic neural query REPL (zero-LLM)

Commands:
  :help          this help
  :quit | :exit  leave the REPL
  :clear         start a fresh empty database (in-memory sessions only)
  :flush         checkpoint the WAL into the main file (--db sessions)
  :store         dump current records, edges, and vector dims

Anything else is parsed as nql (multi-statement with ';' separators).
"#;

fn repl(mut session: Session) -> io::Result<()> {
    let stdin = io::stdin();
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
            ":flush" => {
                session.checkpoint()?;
                session.writeln("flushed")?;
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

fn run_script(mut session: Session, path: &str) -> io::Result<()> {
    let src = fs::read_to_string(path)?;
    session.run(&src)?;
    session.flush()?;
    io::stdout().flush()?;
    Ok(())
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    // `--db <path>` opens a persistent single-file store; without it the
    // session is in-memory (deterministic, throwaway).
    let db_path = match args.iter().position(|a| a == "--db" || a == "-d") {
        Some(i) => args.get(i + 1).cloned(),
        None => None,
    };
    let rest: Vec<String> = args
        .iter()
        .filter(|a| *a != "--db" && *a != "-d")
        .filter(|a| db_path.as_ref().is_none_or(|p| *a != p))
        .cloned()
        .collect();

    let make_session = |out: Box<dyn Write>| -> io::Result<Session> {
        match &db_path {
            Some(p) => Session::open(p, out),
            None => Ok(Session::new(out)),
        }
    };

    match rest.as_slice() {
        [_, flag, path] if flag == "--script" || flag == "-s" => {
            run_script(make_session(Box::new(BufWriter::new(io::stdout())))?, path)
        }
        [_] => repl(make_session(Box::new(BufWriter::new(io::stdout())))?),
        _ => {
            eprintln!("usage: nql [--db FILE] [--script FILE]");
            std::process::exit(2);
        }
    }
}
