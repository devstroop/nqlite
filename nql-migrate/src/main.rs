//! nql-migrate CLI — bulk v2/v3 → format v4 (spec/file-format.md §5.8).
//!
//! ```text
//! nql-migrate --in <store.nql> --out <store.nql> [--force]
//! ```
//!
//! `--in == --out` performs an in-place migration (the source lock is
//! released before the rewrite). Failures print `error: …` and exit 1 —
//! the same convention as `nql-server`/`nql`.

use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "\
nql-migrate — convert an nqlite v2/v3 store to format v4 (spec/file-format.md §5)

USAGE:
    nql-migrate --in <store.nql> --out <store.nql> [--force]

OPTIONS:
    --in,  -i <path>   input store (v2/v3; WAL replayed on load)
    --out, -o <path>   output store (v4; written atomically + verified)
    --force, -f        overwrite an existing output
    --help, -h         show this help
";

struct Args {
    input: PathBuf,
    output: PathBuf,
    force: bool,
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut force = false;
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--help" | "-h" => return Err(USAGE.to_string()),
            "--force" | "-f" => force = true,
            "--in" | "-i" => {
                i += 1;
                let v = raw.get(i).ok_or_else(|| "--in needs a path".to_string())?;
                input = Some(PathBuf::from(v));
            }
            "--out" | "-o" => {
                i += 1;
                let v = raw.get(i).ok_or_else(|| "--out needs a path".to_string())?;
                output = Some(PathBuf::from(v));
            }
            other => return Err(format!("unknown argument: {other}\n\n{USAGE}")),
        }
        i += 1;
    }
    Ok(Args {
        input: input.ok_or_else(|| format!("--in is required\n\n{USAGE}"))?,
        output: output.ok_or_else(|| format!("--out is required\n\n{USAGE}"))?,
        force,
    })
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(msg) => {
            // --help is a successful exit; real usage errors are not.
            if msg == USAGE {
                print!("{msg}");
                return ExitCode::SUCCESS;
            }
            eprintln!("error: {msg}");
            return ExitCode::FAILURE;
        }
    };
    match nql_migrate::migrate(&args.input, &args.output, args.force) {
        Ok(r) => {
            println!(
                "migrated {} → {}: v{} → v4 ({} records, {} edges, {} history entries, \
                 {} memories, {} bytes)",
                args.input.display(),
                args.output.display(),
                r.from_version,
                r.records,
                r.edges,
                r.history,
                r.memories,
                r.bytes
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
