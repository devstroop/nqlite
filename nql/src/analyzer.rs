//! Analyzer: plan validation + SELECT enrichment (M2).
//!
//! [`Analyzer::analyze`] takes the [`nql_ir::Plan`] the parser produced and
//! either rejects it with a structured [`AnalysisError`] or returns an enriched
//! copy ready for the engine:
//!
//! * **Table declaration rules** — a `SELECT` or `INSERT` may only reference a
//!   table declared by an earlier `CREATE TABLE` in the same plan, or one of
//!   the built-in tables (`global`, `meta`).
//! * **Vector dimension contract** — an `INSERT` whose record carries an
//!   embedding must match the `VECTOR<f32, N>` dimension the table declared.
//! * **SELECT enrichment** — a `SELECT` with a kNN clause but no `ORDER BY`
//!   gets `Order::Similarity`; an explicit `ORDER BY similarity` without a kNN
//!   query vector is rejected.
//! * **Record id shape** — every `table:id` must have a non-empty table and a
//!   non-empty id string.
//!
//! Analysis is deterministic and pure: the input plan is never mutated; a
//! modified [`nql_ir::Plan`] is returned on success.

use nql_ir::{Id, Order, Plan, RecordId, Select, Statement};
use std::collections::{BTreeMap, BTreeSet};

/// Built-in table names that exist without an explicit `CREATE TABLE`.
pub const BUILTIN_TABLES: [&str; 2] = ["global", "meta"];

/// A plan-level semantic error: the parse succeeded, but the statements
/// cannot be executed as written.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AnalysisError {
    /// A `SELECT` referenced a table that was never declared.
    #[error(
        "SELECT on table `{table}` requires a prior CREATE TABLE statement or a built-in table"
    )]
    UnknownTableForSelect { table: String },
    /// An `INSERT` targeted a table that was never declared.
    #[error(
        "INSERT into table `{table}` requires a prior CREATE TABLE statement or a built-in table"
    )]
    UnknownTableForInsert { table: String },
    /// An `INSERT` embedding does not match the table's declared vector dim.
    #[error(
        "INSERT into `{table}` has embedding of dimension {actual}, but the table declares VECTOR<f32, {expected}>"
    )]
    EmbeddingDimMismatch {
        table: String,
        expected: usize,
        actual: usize,
    },
    /// `ORDER BY similarity` needs a kNN query vector to rank against.
    #[error("ORDER BY similarity requires a kNN query vector (WHERE vector::similarity(...))")]
    SimilarityWithoutKnn,
    /// A `table:id` pair had an empty table name.
    #[error("record id table must not be empty")]
    EmptyTable,
    /// A `table:id` pair had an empty string id.
    #[error("record id in table `{table}` has an empty id string")]
    EmptyId { table: String },
}

/// Validates and enriches a parsed [`Plan`] for execution.
///
/// The input plan is treated as immutable: on success a new, enriched
/// [`Plan`] is returned; on failure the original plan is untouched.
pub struct Analyzer;

impl Analyzer {
    /// Analyze a whole plan, statement by statement, in order. Statements
    /// earlier in the plan declare tables for later ones.
    pub fn analyze(plan: &Plan) -> Result<Plan, AnalysisError> {
        let mut ctx = Ctx::new();
        let mut out = Vec::with_capacity(plan.len());
        for stmt in plan {
            out.push(ctx.analyze_statement(stmt)?);
        }
        Ok(out)
    }

    /// Analyze a single statement with no prior context: a standalone
    /// `SELECT`/`INSERT` (outside the built-in tables) is an error.
    pub fn analyze_statement(stmt: &Statement) -> Result<Statement, AnalysisError> {
        Ctx::new().analyze_statement(stmt)
    }
}

/// Analysis state carried across statements of one plan.
struct Ctx {
    declared: BTreeSet<String>,
    vector_dims: BTreeMap<String, usize>,
}

impl Ctx {
    fn new() -> Self {
        Self {
            declared: BUILTIN_TABLES.iter().map(|s| (*s).to_string()).collect(),
            vector_dims: BTreeMap::new(),
        }
    }

    fn analyze_statement(&mut self, stmt: &Statement) -> Result<Statement, AnalysisError> {
        match stmt {
            Statement::CreateTable { table, vector_dim } => {
                self.declared.insert(table.clone());
                if let Some(dim) = vector_dim {
                    self.vector_dims.insert(table.clone(), *dim);
                }
                Ok(stmt.clone())
            }
            Statement::Insert(rec) => {
                validate_record_id(&rec.id)?;
                let table = &rec.id.table;
                if !self.declared.contains(table) {
                    return Err(AnalysisError::UnknownTableForInsert {
                        table: table.clone(),
                    });
                }
                if let (Some(expected), Some(embedding)) =
                    (self.vector_dims.get(table), &rec.embedding)
                {
                    if embedding.len() != *expected {
                        return Err(AnalysisError::EmbeddingDimMismatch {
                            table: table.clone(),
                            expected: *expected,
                            actual: embedding.len(),
                        });
                    }
                }
                Ok(stmt.clone())
            }
            Statement::Relate(edge) => {
                validate_record_id(&edge.from)?;
                validate_record_id(&edge.to)?;
                Ok(stmt.clone())
            }
            Statement::Select(sel) => {
                if !self.declared.contains(&sel.table) {
                    return Err(AnalysisError::UnknownTableForSelect {
                        table: sel.table.clone(),
                    });
                }
                Ok(Statement::Select(enrich_select(sel)?))
            }
            Statement::Forget { id } => {
                validate_record_id(id)?;
                Ok(stmt.clone())
            }
        }
    }
}

/// Enrichment pass for a single `SELECT` (declared-table check already done).
fn enrich_select(sel: &Select) -> Result<Select, AnalysisError> {
    let mut enriched = sel.clone();
    if enriched.knn.is_some() && enriched.order.is_none() {
        enriched.order = Some(Order::Similarity);
    }
    if enriched.order == Some(Order::Similarity) && enriched.knn.is_none() {
        return Err(AnalysisError::SimilarityWithoutKnn);
    }
    Ok(enriched)
}

/// Shape-check a `table:id` pair: non-empty table, non-empty string id.
fn validate_record_id(rid: &RecordId) -> Result<(), AnalysisError> {
    if rid.table.is_empty() {
        return Err(AnalysisError::EmptyTable);
    }
    if let Id::Str(s) = &rid.id {
        if s.is_empty() {
            return Err(AnalysisError::EmptyId {
                table: rid.table.clone(),
            });
        }
    }
    Ok(())
}
