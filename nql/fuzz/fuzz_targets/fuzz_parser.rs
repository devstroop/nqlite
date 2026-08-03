//! cargo-fuzz harness for the nql parser.
//!
//! Run (requires nightly + cargo-fuzz):
//!   cargo +nightly fuzz run fuzz_parser
//!
//! Invariant under test: `nql::parse` must NEVER panic on any byte input
//! (valid UTF-8 is forwarded to the parser; other bytes are skipped). Found
//! inputs are minimised automatically and written to `fuzz/artifacts/`.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = nql::parse(s);
    }
});
