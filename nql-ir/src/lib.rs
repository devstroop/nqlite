//! Shared Plan/IR and value types.

/// Minimal value-type smoke test so `cargo test` has a canonical harness target
/// before the real engine tests land in Milestone 0.
#[cfg(test)]
mod smoke {
    #[test]
    fn crate_loads() {
        assert!(true, "nql-ir compiles as the shared IR contract");
    }
}
