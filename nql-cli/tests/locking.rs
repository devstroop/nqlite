//! Cross-process single-writer lock (issue #84): a second `nql --db` opener
//! must fail loudly while another process holds the store, and succeed after
//! that process exits. Drives the real `nql` binary end to end.

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};

const NQL: &str = env!("CARGO_BIN_EXE_nql");

#[test]
fn second_process_open_is_rejected_while_store_held() {
    let dir = std::env::temp_dir().join(format!("nqlite-lock-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let db = dir.join("store.nql");
    let script = dir.join("q.nql");
    std::fs::write(&script, "CREATE TABLE t; INSERT INTO t:1 { \"x\": 1 };").unwrap();

    // Holder: interactive REPL on the store. The banner prints only after
    // Session::open has taken the lock, so reading it is the readiness signal.
    let mut holder = Command::new(NQL)
        .args(["--db", db.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn holder process");
    let mut stdout = holder.stdout.take().unwrap();
    let mut banner = String::new();
    BufReader::new(&mut stdout)
        .read_line(&mut banner)
        .expect("read REPL banner");
    assert!(
        banner.contains("nql"),
        "expected REPL banner, got {banner:?}"
    );

    // Second process: must be rejected at open with a lock error, non-zero exit.
    let second = Command::new(NQL)
        .args([
            "--db",
            db.to_str().unwrap(),
            "--script",
            script.to_str().unwrap(),
        ])
        .output()
        .expect("run second opener");
    assert!(
        !second.status.success(),
        "second opener must fail while the store is held"
    );
    let err = String::from_utf8_lossy(&second.stderr);
    assert!(
        err.contains("locked"),
        "expected the single-writer lock error, stderr: {err}"
    );

    // Release: close the holder's stdin -> REPL exits at EOF -> lock dropped.
    drop(holder.stdin.take());
    let status = holder.wait().expect("wait for holder to exit");
    assert!(status.success(), "holder should exit cleanly on EOF");

    // Store is free again: the same opener now succeeds (exit 0).
    let third = Command::new(NQL)
        .args([
            "--db",
            db.to_str().unwrap(),
            "--script",
            script.to_str().unwrap(),
        ])
        .output()
        .expect("run opener after release");
    assert!(
        third.status.success(),
        "opener after release must succeed; stderr: {}",
        String::from_utf8_lossy(&third.stderr)
    );

    let _ = std::fs::remove_dir_all(&dir);
}
