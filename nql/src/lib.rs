//! nql front-end (parser + AST). Storage-agnostic by contract.
//!
//! Parses nql text into a deterministic [`nql_ir::Plan`] for the M0 grammar
//! slice. See [`parser`] for the accepted grammar and [`parse`] / [`parse_statement`]
//! for the public entry points.

pub mod lexer;
pub mod parser;
#[cfg(test)]
mod tests;

pub use parser::{parse, parse_statement, NqlError};
