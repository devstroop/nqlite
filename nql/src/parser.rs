//! Recursive-descent parser: nql text -> `nql_ir::Plan` (M0 grammar slice).
//!
//! Grammar (keywords case-insensitive):
//!
//! ```text
//! CREATE TABLE <ident> [VECTOR<f32, <N>>]
//! INSERT INTO <ident>:<id> { <json-ish body> } [EMBED [<f32>, ...]]
//! RELATE (<ident>:<id>) -> :<edgename> -> (<ident>:<id>)
//!        [SET weight = <float>, <field> = <value>, ...]
//! SELECT <field-list|*> FROM <ident>
//!        [WHERE <field> = <value>
//!         | WHERE vector::similarity(embedding, [<f32>,...]) AND k = <N>
//!         | WHERE embedding IS NOT NULL]
//!        [ORDER BY ::<similarity|salience|score|recency>] [LIMIT <N>]
//! FORGET <ident>:<id>
//! ```
//!
//! The parser is deterministic: documents use `BTreeMap`, `created_at` is set
//! to `0` (the engine clocks it later), and there is no wall-clock or
//! randomness anywhere.

use crate::lexer::{tokenize, Spanned, Token};
use nql_ir::{
    Aggregate, CmpOp, Filter, Id, Knn, MatchDirection, MatchPath, MatchStep, Order, Plan, Record,
    RecordId, RelationEdge, Select, Statement, Value,
};
use std::collections::BTreeMap;

/// Structured parse error with a human-readable message plus 1-based line and
/// column of the offending input.
#[derive(Debug, Clone, thiserror::Error)]
pub enum NqlError {
    /// A tokenization failure (bad character, unterminated string, ...).
    #[error("lex error at {line}:{col}: {message}")]
    Lex {
        message: String,
        line: usize,
        col: usize,
    },
    /// A grammar failure (unexpected token, missing clause, bad literal, ...).
    #[error("parse error at {line}:{col}: {message}")]
    Parse {
        message: String,
        line: usize,
        col: usize,
    },
}

impl NqlError {
    pub(crate) fn syntax(message: impl Into<String>, line: usize, col: usize) -> Self {
        NqlError::Lex {
            message: message.into(),
            line,
            col,
        }
    }

    pub(crate) fn parse(message: impl Into<String>, line: usize, col: usize) -> Self {
        NqlError::Parse {
            message: message.into(),
            line,
            col,
        }
    }

    /// 1-based line of the error.
    pub fn line(&self) -> usize {
        match self {
            NqlError::Lex { line, .. } | NqlError::Parse { line, .. } => *line,
        }
    }

    /// 1-based column of the error.
    pub fn col(&self) -> usize {
        match self {
            NqlError::Lex { col, .. } | NqlError::Parse { col, .. } => *col,
        }
    }
}

/// Parse a full nql program (zero or more statements) into a `Plan`.
pub fn parse(input: &str) -> Result<Plan, NqlError> {
    let tokens = tokenize(input)?;
    let mut p = Parser::new(tokens);
    let mut plan = Vec::new();
    while !p.at_eof() {
        p.skip_semis();
        if p.at_eof() {
            break;
        }
        plan.push(p.parse_statement()?);
    }
    Ok(plan)
}

/// Parse exactly one statement; trailing input (beyond whitespace) is an error.
pub fn parse_statement(input: &str) -> Result<Statement, NqlError> {
    let tokens = tokenize(input)?;
    let mut p = Parser::new(tokens);
    let stmt = p.parse_statement()?;
    if !p.at_eof() {
        return Err(p.err_here("unexpected trailing input after statement"));
    }
    Ok(stmt)
}

struct Parser {
    toks: Vec<Spanned>,
    idx: usize,
}

impl Parser {
    fn new(toks: Vec<Spanned>) -> Self {
        Self { toks, idx: 0 }
    }

    fn peek(&self) -> &Spanned {
        self.toks
            .get(self.idx)
            .expect("parser index out of bounds (internal error)")
    }

    fn peek_tok(&self) -> &Token {
        &self.peek().tok
    }

    /// Look ahead `n` tokens without consuming (0 = next).
    fn peek_n(&self, n: usize) -> &Token {
        self.toks
            .get(self.idx + n)
            .map(|s| &s.tok)
            .unwrap_or(&Token::Eof)
    }

    fn at_eof(&self) -> bool {
        matches!(self.peek_tok(), Token::Eof)
    }

    /// Skip any run of `;` statement separators (also tolerates trailing `;`).
    fn skip_semis(&mut self) {
        while matches!(self.peek_tok(), Token::Semi) {
            self.idx += 1;
        }
    }

    fn bump(&mut self) -> Spanned {
        let s = self.peek().clone();
        if !matches!(s.tok, Token::Eof) {
            self.idx += 1;
        }
        s
    }

    fn err_here(&self, msg: impl Into<String>) -> NqlError {
        let s = self.peek();
        NqlError::parse(msg, s.line, s.col)
    }

    fn err_at(&self, s: &Spanned, msg: impl Into<String>) -> NqlError {
        NqlError::parse(msg, s.line, s.col)
    }

    /// Consume and return an identifier (any bare word).
    fn expect_ident(&mut self, what: &str) -> Result<String, NqlError> {
        let s = self.bump();
        match &s.tok {
            Token::Ident(id) => Ok(id.clone()),
            other => Err(self.err_at(&s, format!("expected {what}, found {}", describe(other)))),
        }
    }

    /// Consume a specific keyword, matching case-insensitively.
    fn expect_keyword(&mut self, kw: &str, what: &str) -> Result<(), NqlError> {
        let s = self.bump();
        match &s.tok {
            Token::Ident(id) if id.eq_ignore_ascii_case(kw) => Ok(()),
            other => Err(self.err_at(
                &s,
                format!("expected {what} (`{kw}`), found {}", describe(other)),
            )),
        }
    }

    fn expect_token(&mut self, tok: Token, what: &str) -> Result<(), NqlError> {
        let s = self.bump();
        if s.tok == tok {
            Ok(())
        } else {
            Err(self.err_at(&s, format!("expected {what}, found {}", describe(&s.tok))))
        }
    }

    /// Consume an optional comma separator; returns true if one was present.
    fn eat_comma(&mut self) -> bool {
        if matches!(self.peek_tok(), Token::Comma) {
            self.bump();
            true
        } else {
            false
        }
    }

    /// If the next token is the given keyword (case-insensitive), consume it.
    fn eat_keyword(&mut self, kw: &str) -> bool {
        if let Token::Ident(id) = self.peek_tok() {
            if id.eq_ignore_ascii_case(kw) {
                self.bump();
                return true;
            }
        }
        false
    }

    // -- statements --------------------------------------------------------

    fn parse_statement(&mut self) -> Result<Statement, NqlError> {
        match self.peek_tok() {
            Token::Ident(kw) if kw.eq_ignore_ascii_case("create") => self.parse_create(),
            Token::Ident(kw) if kw.eq_ignore_ascii_case("insert") => self.parse_insert(),
            Token::Ident(kw) if kw.eq_ignore_ascii_case("relate") => self.parse_relate(),
            Token::Ident(kw) if kw.eq_ignore_ascii_case("match") => self.parse_match(),
            Token::Ident(kw) if kw.eq_ignore_ascii_case("closure") => self.parse_closure(),
            Token::Ident(kw) if kw.eq_ignore_ascii_case("memory") => self.parse_memory(),
            Token::Ident(kw) if kw.eq_ignore_ascii_case("select") => self.parse_select(),
            Token::Ident(kw) if kw.eq_ignore_ascii_case("forget") => self.parse_forget(),
            Token::Ident(kw) if kw.eq_ignore_ascii_case("prune") => self.parse_prune(),
            Token::Ident(kw) if kw.eq_ignore_ascii_case("history") => self.parse_history(),
            other => Err(self.err_here(format!(
                "expected a statement keyword (CREATE, INSERT, RELATE, MATCH, CLOSURE, MEMORY, SELECT, FORGET, PRUNE, HISTORY), found {}",
                describe(other)
            ))),
        }
    }

    /// `HISTORY SINCE <ts>` — exact delta read (issue #118): every mutation
    /// strictly after the cutoff (rows AND edges + tombstones), for sync
    /// without two full `AS OF` replays. Read-only; never WAL'd.
    fn parse_history(&mut self) -> Result<Statement, NqlError> {
        self.expect_keyword("history", "HISTORY")?;
        self.expect_keyword("since", "SINCE after HISTORY")?;
        Ok(Statement::HistorySince(self.expect_int("SINCE timestamp")?))
    }

    /// `PRUNE HISTORY` — compact the mutation history into a snapshot at the
    /// current clock (issue #95): bounded growth, cheap replay from the
    /// snapshot; `AS OF` before it then errors loudly (spec §2.7).
    fn parse_prune(&mut self) -> Result<Statement, NqlError> {
        self.expect_keyword("prune", "PRUNE")?;
        self.expect_keyword("history", "HISTORY after PRUNE")?;
        Ok(Statement::PruneHistory)
    }

    fn parse_create(&mut self) -> Result<Statement, NqlError> {
        self.expect_keyword("create", "CREATE")?;
        self.expect_keyword("table", "TABLE")?;
        let table = self.expect_ident("table name")?;
        let mut vector_dim = None;
        if self.eat_keyword("vector") {
            self.expect_token(Token::Lt, "`<` after VECTOR")?;
            // `f32` (or any type placeholder) — accept the word, don't require it.
            self.expect_ident("vector element type (e.g. `f32`)")?;
            self.expect_token(Token::Comma, "`,` in VECTOR declaration")?;
            let dim = self.expect_usize("vector dimension")?;
            if dim == 0 {
                return Err(self.err_here("vector dimension must be positive"));
            }
            self.expect_token(Token::Gt, "`>` closing VECTOR declaration")?;
            vector_dim = Some(dim);
        }
        Ok(Statement::CreateTable { table, vector_dim })
    }

    fn parse_insert(&mut self) -> Result<Statement, NqlError> {
        self.expect_keyword("insert", "INSERT")?;
        self.expect_keyword("into", "INTO")?;
        let id = self.parse_record_id()?;
        let body = self.parse_object()?;
        let mut embedding = None;
        if self.eat_keyword("embed") {
            embedding = Some(self.parse_float_vector()?);
        }
        Ok(Statement::Insert(Record {
            id,
            body,
            embedding,
            // Engine clocks created_at per-transaction; the parser stays
            // deterministic and emits 0.
            created_at: 0,
        }))
    }

    fn parse_relate(&mut self) -> Result<Statement, NqlError> {
        self.expect_keyword("relate", "RELATE")?;
        self.expect_token(Token::LParen, "`(` before FROM record")?;
        let from = self.parse_record_id()?;
        self.expect_token(Token::RParen, "`)` after FROM record")?;
        self.expect_token(Token::Arrow, "`->` after FROM record")?;
        // Edge name: `:<edgename>` (colon optional for lenience).
        if matches!(self.peek_tok(), Token::Colon) {
            self.bump();
        }
        let name = self.expect_ident("edge name")?;
        self.expect_token(Token::Arrow, "`->` before TO record")?;
        self.expect_token(Token::LParen, "`(` before TO record")?;
        let to = self.parse_record_id()?;
        self.expect_token(Token::RParen, "`)` after TO record")?;

        let mut weight = None;
        let mut props = BTreeMap::new();
        if self.eat_keyword("set") {
            loop {
                let field = self.expect_ident("SET field name")?;
                self.expect_token(Token::Eq, "`=` in SET clause")?;
                let value = self.parse_value()?;
                if field.eq_ignore_ascii_case("weight") {
                    weight = match value {
                        Value::Int(n) => Some(n as f32),
                        Value::Float(f) => Some(f as f32),
                        other => {
                            return Err(self.err_here(format!(
                                "SET weight expects a number, found {}",
                                value_name(&other)
                            )));
                        }
                    };
                } else {
                    props.insert(field, value);
                }
                if !self.eat_comma() {
                    break;
                }
            }
        }

        Ok(Statement::Relate(RelationEdge {
            from,
            name,
            to,
            created_at: 0,
            weight,
            props,
        }))
    }

    fn parse_match(&mut self) -> Result<Statement, NqlError> {
        self.expect_keyword("match", "MATCH")?;
        let path = self.parse_path("MATCH")?;
        // `MATCH ... COUNT` — walk-count mode (issue #94). Statements can only
        // start with keywords, so a `COUNT` here can't be the next statement.
        if matches!(self.peek_tok(), Token::Ident(kw) if kw.eq_ignore_ascii_case("count")) {
            self.bump();
            return Ok(Statement::MatchCount(path));
        }
        Ok(Statement::Match(path))
    }

    fn parse_closure(&mut self) -> Result<Statement, NqlError> {
        self.expect_keyword("closure", "CLOSURE")?;
        let path = self.parse_path("CLOSURE")?;
        Ok(Statement::Closure(path))
    }

    /// `( recordid ) ( ('->' | '<-') ':' ident [WHERE ident = value] )+` —
    /// the shared path grammar of MATCH and CLOSURE. Each step may carry an
    /// optional edge-property equality filter (`edge_props` in the spec).
    fn parse_path(&mut self, kw: &str) -> Result<MatchPath, NqlError> {
        self.expect_token(Token::LParen, "`(` before start record")?;
        let start = self.parse_record_id()?;
        self.expect_token(Token::RParen, "`)` after start record")?;

        let mut steps = Vec::new();
        loop {
            let direction = if matches!(self.peek_tok(), Token::Arrow) {
                self.bump();
                MatchDirection::Out
            } else if matches!(self.peek_tok(), Token::LeftArrow) {
                self.bump();
                MatchDirection::In
            } else {
                break;
            };
            if matches!(self.peek_tok(), Token::Colon) {
                self.bump();
            }
            let name = self.expect_ident("edge name after `->`/`<-`")?;
            let mut edge_props = None;
            if matches!(self.peek_tok(), Token::Ident(kw) if kw.eq_ignore_ascii_case("where")) {
                self.bump();
                // Same term grammar as a WHERE clause (issues #93/#125):
                // predicates composed with AND — evaluated against edge props.
                edge_props = Some(self.parse_where_conjunction()?);
            }
            steps.push(MatchStep {
                direction,
                name,
                edge_props,
            });
        }
        if steps.is_empty() {
            return Err(self.err_here(format!("{kw} requires at least one edge step (`-> :name`)")));
        }
        // `[ 'AS OF' int ]` (issue #92): shared by MATCH and CLOSURE — the
        // engine traverses the history-replayed snapshot (spec §2.7), the
        // same machinery `SELECT ... AS OF` uses. It precedes a trailing
        // MATCH `COUNT` (parse_match consumes COUNT after this returns).
        let mut as_of = None;
        if matches!(self.peek_tok(), Token::Ident(kw) if kw.eq_ignore_ascii_case("as")) {
            self.bump();
            self.expect_keyword("of", "OF after AS")?;
            as_of = Some(self.expect_int("AS OF timestamp")?);
        }
        Ok(MatchPath {
            start,
            steps,
            as_of,
        })
    }

    fn parse_select(&mut self) -> Result<Statement, NqlError> {
        self.expect_keyword("select", "SELECT")?;
        // Field list (or `*`), or `COUNT(*)`: carried into the IR as the
        // projection (`None` = star = full records) and/or aggregate.
        let (fields, aggregate) = self.parse_field_list()?;
        self.expect_keyword("from", "FROM")?;
        let table = self.expect_ident("table name after FROM")?;

        let mut knn = None;
        let mut filter = None;
        let mut order = None;
        let mut limit = None;
        let mut offset = None;
        let mut as_of = None;

        loop {
            match self.peek_tok() {
                Token::Ident(kw) if kw.eq_ignore_ascii_case("where") => {
                    self.bump();
                    let (k, f) = self.parse_where()?;
                    knn = k;
                    filter = f;
                }
                Token::Ident(kw) if kw.eq_ignore_ascii_case("order") => {
                    self.bump();
                    self.expect_keyword("by", "BY after ORDER")?;
                    let colon_prefixed =
                        matches!(self.peek_tok(), Token::DoubleColon | Token::Colon);
                    if colon_prefixed {
                        self.bump();
                    }
                    let o = self.expect_ident("order key after ORDER BY")?;
                    order = Some(match o.to_ascii_lowercase().as_str() {
                        "similarity" => Order::Similarity,
                        "salience" => self.parse_salience_order()?,
                        "score" => Order::Score,
                        "recency" => Order::Recency,
                        "votes" => Order::Votes,
                        "feedback" => Order::Feedback,
                        // A bare non-operator identifier is a body-field sort
                        // (issue #117). `::` commits to the operator list, so
                        // `::seq` stays a positioned error, not a field.
                        _ if colon_prefixed => {
                            return Err(self.err_here(format!(
                                "unknown ORDER BY operator `{o}` (expected similarity, \
                                 salience, score, recency, votes, or feedback — or a \
                                 bare field name without `::`)"
                            )));
                        }
                        _ => Order::Field {
                            key: o,
                            desc: false,
                        },
                    });
                    // `<field> [DESC]` (issue #117). Operator keys have fixed
                    // directions, so DESC after one is a positioned error —
                    // never a silent no-op (#91's lesson).
                    if matches!(self.peek_tok(), Token::Ident(kw) if kw.eq_ignore_ascii_case("desc"))
                    {
                        self.bump();
                        match &mut order {
                            Some(Order::Field { desc, .. }) => *desc = true,
                            _ => {
                                return Err(self.err_here(
                                    "`DESC` applies to field sorts only \
                                     (`ORDER BY <field> DESC`); `::` operators have \
                                     fixed directions",
                                ));
                            }
                        }
                    }
                }
                Token::Ident(kw) if kw.eq_ignore_ascii_case("as") => {
                    self.bump();
                    self.expect_keyword("of", "OF after AS")?;
                    // `AS OF <int>` — temporal read at a logical timestamp.
                    as_of = Some(self.expect_int("AS OF timestamp")?);
                }
                Token::Ident(kw) if kw.eq_ignore_ascii_case("limit") => {
                    self.bump();
                    limit = Some(self.expect_usize("LIMIT count")?);
                    // `LIMIT n OFFSET m` in one clause …
                    if matches!(self.peek_tok(), Token::Ident(k) if k.eq_ignore_ascii_case("offset"))
                    {
                        self.bump();
                        offset = Some(self.expect_usize("OFFSET count")?);
                    }
                }
                // … or OFFSET on its own (either clause order).
                Token::Ident(kw) if kw.eq_ignore_ascii_case("offset") => {
                    self.bump();
                    offset = Some(self.expect_usize("OFFSET count")?);
                }
                _ => break,
            }
        }

        Ok(Statement::Select(Select {
            table,
            knn,
            filter,
            order,
            limit,
            as_of,
            fields,
            offset,
            aggregate,
        }))
    }

    fn parse_forget(&mut self) -> Result<Statement, NqlError> {
        self.expect_keyword("forget", "FORGET")?;
        let id = self.parse_record_id()?;
        Ok(Statement::Forget { id })
    }

    /// `MEMORY <name>` — switch the plan's memory context.
    fn parse_memory(&mut self) -> Result<Statement, NqlError> {
        self.expect_keyword("memory", "MEMORY")?;
        let name = self.expect_ident("memory name after MEMORY")?;
        Ok(Statement::Memory { name })
    }

    // -- shared pieces ------------------------------------------------------

    /// `SELECT <field>, ... | * | COUNT(*)` — carried into
    /// [`Select::fields`] (`None` for `*` = full records) and/or
    /// [`Select::aggregate`]; the engine applies projection as the final
    /// pipeline step (spec §2.3). `COUNT` is only an aggregate when followed
    /// by `(*)`, so a field literally named `count` still works.
    fn parse_field_list(&mut self) -> Result<(Option<Vec<String>>, Option<Aggregate>), NqlError> {
        if matches!(self.peek_tok(), Token::Ident(c) if c.eq_ignore_ascii_case("count"))
            && matches!(self.peek_n(1), Token::LParen)
        {
            self.bump(); // `count`
            self.expect_token(Token::LParen, "`(` after COUNT")?;
            self.expect_token(Token::Star, "`*` inside COUNT(")?;
            self.expect_token(Token::RParen, "`)` after COUNT(")?;
            return Ok((None, Some(Aggregate::CountStar)));
        }
        let mut fields = Vec::new();
        let mut star = false;
        loop {
            match self.peek_tok() {
                Token::Star => {
                    self.bump();
                    star = true;
                }
                Token::Ident(name) => {
                    let name = name.clone();
                    self.bump();
                    fields.push(name);
                }
                _ => {
                    return Err(self.err_here("expected a field name, `*`, or `COUNT(*)` in SELECT"))
                }
            }
            if !self.eat_comma() {
                break;
            }
        }
        Ok((if star { None } else { Some(fields) }, None))
    }

    /// `<ident>:<id>` — id is a number or a bare word.
    fn parse_record_id(&mut self) -> Result<RecordId, NqlError> {
        let table = self.expect_ident("table name")?;
        self.expect_token(Token::Colon, "`:` between table and id")?;
        let s = self.bump();
        let id = match &s.tok {
            Token::Int(n) if *n >= 0 => Id::Num(*n as u64),
            Token::Int(n) => Id::Str(n.to_string()),
            Token::Ident(word) => Id::Str(word.clone()),
            other => {
                return Err(self.err_at(
                    &s,
                    format!(
                        "expected record id (number or name), found {}",
                        describe(other)
                    ),
                ));
            }
        };
        Ok(RecordId::new(table, id))
    }

    /// The filter/knn portion after `WHERE` has been consumed.
    fn parse_where(&mut self) -> Result<(Option<Knn>, Option<Filter>), NqlError> {
        // `::bm25(field, "query") [AND k = <N>]` — lexical retrieval; may be
        // fused with a kNN clause: `::bm25(f, "q") AND vector::similarity(embedding, v) AND k = N`.
        if matches!(self.peek_tok(), Token::DoubleColon) {
            let filter = self.parse_bm25_filter()?;
            let mut knn = None;
            if self.eat_keyword("and") {
                if !matches!(self.peek_tok(), Token::Ident(kw) if kw.eq_ignore_ascii_case("vector"))
                {
                    return Err(self.err_here(
                        "`::bm25(...)` may only be ANDed with `vector::similarity(...)` \
                         (the hybrid form); field predicates belong to a plain WHERE \
                         conjunction (issue #125)",
                    ));
                }
                knn = Some(self.parse_knn()?);
            }
            return Ok((knn, Some(filter)));
        }
        // `vector::similarity(embedding, [..]) AND k = <N>` — kNN; may be fused
        // with a lexical clause: `vector::similarity(...) AND k = N AND ::bm25(f, "q")`.
        if let Token::Ident(kw) = self.peek_tok() {
            if kw.eq_ignore_ascii_case("vector") {
                let knn = self.parse_knn()?;
                let mut filter = None;
                if self.eat_keyword("and") {
                    if !matches!(self.peek_tok(), Token::DoubleColon) {
                        return Err(self.err_here(
                            "after a kNN clause, `AND` may only introduce `::bm25(...)` \
                             (hybrid form) — field predicates belong to a plain WHERE \
                             conjunction (issue #125)",
                        ));
                    }
                    filter = Some(self.parse_bm25_filter()?);
                }
                return Ok((Some(knn), filter));
            }
        }
        // Field-level terms compose with `AND` (issue #125): n-ary, all-of,
        // no precedence — one combinable subset, shared with edge props.
        let filter = self.parse_where_conjunction()?;
        Ok((None, Some(filter)))
    }

    /// A conjunction of combinable terms: `term (AND term)*` — n-ary all-of
    /// (issue #125). Shared by `WHERE` clauses and MATCH/CLOSURE
    /// edge-property filters; a single term stays unwrapped (the existing
    /// `Filter` variants, unchanged for consumers).
    fn parse_where_conjunction(&mut self) -> Result<Filter, NqlError> {
        let mut terms = vec![self.parse_where_term()?];
        while self.eat_keyword("and") {
            if matches!(self.peek_tok(), Token::DoubleColon) {
                return Err(self.err_here(
                    "`::bm25` does not take part in `AND` conjunctions — the hybrid \
                     form is `::bm25(...) AND vector::similarity(...)` (issue #125)",
                ));
            }
            if matches!(self.peek_tok(), Token::Ident(kw) if kw.eq_ignore_ascii_case("vector")) {
                return Err(self.err_here(
                    "`vector::similarity` does not take part in field-predicate `AND` \
                     conjunctions — put the kNN clause first: `WHERE \
                     vector::similarity(...) AND k = N [AND ::bm25(...)]` (issue #125)",
                ));
            }
            terms.push(self.parse_where_term()?);
        }
        Ok(if terms.len() == 1 {
            terms.remove(0)
        } else {
            Filter::And(terms)
        })
    }

    /// One combinable WHERE term: a field predicate or `IS NOT NULL` — the
    /// subset `AND` composes (scoring clauses keep their own forms,
    /// issue #125).
    fn parse_where_term(&mut self) -> Result<Filter, NqlError> {
        let field = self.expect_ident("WHERE field name")?;
        if matches!(self.peek_tok(), Token::Ident(kw) if kw.eq_ignore_ascii_case("is")) {
            self.bump();
            self.expect_keyword("not", "NOT in `IS NOT NULL`")?;
            self.expect_keyword("null", "NULL in `IS NOT NULL`")?;
            return Ok(Filter::HasEmbedding);
        }
        self.parse_field_predicate(field)
    }

    /// A field predicate once its name is consumed: `= <value>`,
    /// `!= | < | <= | > | >= <value>`, `IN [<v>, …]`, or
    /// `BETWEEN <lo> AND <hi>` (issues #93/#94, spec §2.3). Shared by
    /// `WHERE` clauses and MATCH/CLOSURE edge-property filters.
    fn parse_field_predicate(&mut self, field: String) -> Result<Filter, NqlError> {
        // `id` is the pseudo-field bound to the record's `table:id` display
        // string (issue #128): `=`, `!=`, and `IN` only — ids are not an
        // ordered value, so ordered comparisons are positioned errors.
        let is_id = field == "id";
        if is_id
            && matches!(
                self.peek_tok(),
                Token::Lt | Token::Le | Token::Gt | Token::Ge
            )
        {
            return Err(self.err_here(
                "`id` supports only `=`, `!=`, and `IN [...]` — ids are not an \
                 ordered value (issue #128)",
            ));
        }
        let cmp = match self.peek_tok() {
            Token::Eq => {
                self.bump();
                let value = self.parse_value()?;
                if is_id {
                    self.require_id_string(&value)?;
                }
                return Ok(Filter::FieldEquals { field, value });
            }
            Token::Ne => CmpOp::Ne,
            Token::Lt => CmpOp::Lt,
            Token::Le => CmpOp::Le,
            Token::Gt => CmpOp::Gt,
            Token::Ge => CmpOp::Ge,
            Token::Ident(kw) if kw.eq_ignore_ascii_case("in") => {
                self.bump();
                // Element-wise, NOT parse_value: an all-numeric literal list
                // would collapse to Value::Vector (M0 array contract) and
                // `group IN [1, 2, 3]` could never match an Int field.
                self.expect_token(Token::LBracket, "`[` after IN (e.g. `IN [1, 2, 3]`)")?;
                let mut values = Vec::new();
                loop {
                    if matches!(self.peek_tok(), Token::RBracket) {
                        self.bump();
                        break;
                    }
                    values.push(self.parse_value()?);
                    if !self.eat_comma() {
                        self.expect_token(Token::RBracket, "`]` closing the IN list")?;
                        break;
                    }
                }
                if is_id {
                    for v in &values {
                        self.require_id_string(v)?;
                    }
                }
                return Ok(Filter::FieldIn { field, values });
            }
            Token::Ident(kw) if kw.eq_ignore_ascii_case("between") => {
                if is_id {
                    return Err(self.err_here(
                        "`id` supports only `=`, `!=`, and `IN [...]` — ids are not an \
                         ordered value (issue #128)",
                    ));
                }
                self.bump();
                let lo = self.parse_value()?;
                self.expect_keyword("and", "AND between BETWEEN bounds")?;
                let hi = self.parse_value()?;
                return Ok(Filter::FieldBetween { field, lo, hi });
            }
            _ => {
                return Err(self.err_here(format!(
                    "expected an operator after `{field}` \
                     (`=`, `!=`, `<`, `<=`, `>`, `>=`, `IN [...]`, or `BETWEEN ... AND ...`)"
                )))
            }
        };
        self.bump(); // the comparison token
        let value = self.parse_value()?;
        if is_id {
            self.require_id_string(&value)?;
        }
        Ok(Filter::FieldCmp {
            field,
            op: cmp,
            value,
        })
    }

    /// `WHERE id <op> …` compares against the `table:id` display string
    /// (issue #128): a non-string literal is a positioned error, not a
    /// silent never-match.
    fn require_id_string(&self, value: &Value) -> Result<(), NqlError> {
        if matches!(value, Value::Str(_)) {
            Ok(())
        } else {
            Err(self.err_here(
                "`id` predicates compare against the `table:id` string, e.g. \
                 `id = \"doc:7\"` (issue #128)",
            ))
        }
    }

    /// `::bm25(<field>, "<query>") [AND k = <N>]` — the lexical filter alone.
    fn parse_bm25_filter(&mut self) -> Result<Filter, NqlError> {
        if !matches!(self.peek_tok(), Token::DoubleColon) {
            return Err(self.err_here("expected `::bm25` after `AND`"));
        }
        self.bump();
        let op = self.expect_ident("operator after `::`")?;
        if !op.eq_ignore_ascii_case("bm25") {
            return Err(self.err_here(format!("unknown WHERE operator `::{op}` (expected ::bm25)")));
        }
        self.expect_token(Token::LParen, "`(` after ::bm25")?;
        let field = self.expect_ident("::bm25 field name")?;
        self.expect_token(Token::Comma, "`,` between field and query")?;
        let query = match self.parse_value()? {
            Value::Str(s) => s,
            other => {
                return Err(self.err_here(format!(
                    "::bm25 query must be a string, found {}",
                    value_name(&other)
                )));
            }
        };
        self.expect_token(Token::RParen, "`)` closing ::bm25")?;
        let mut k = None;
        // Only consume `AND k = <N>` when `k` actually follows — an `AND
        // vector::similarity(...)` after the bm25 clause is the hybrid fusion
        // bridge and must be left for the caller.
        if matches!(self.peek_tok(), Token::Ident(kw) if kw.eq_ignore_ascii_case("and"))
            && matches!(self.peek_n(1), Token::Ident(kw) if kw.eq_ignore_ascii_case("k"))
        {
            self.bump(); // AND
            self.bump(); // k
            self.expect_token(Token::Eq, "`=` before k")?;
            k = Some(self.expect_usize("k")?);
        }
        Ok(Filter::Bm25 { field, query, k })
    }

    /// `vector::similarity(embedding, [<f32>, ...]) AND k = <N>`
    fn parse_knn(&mut self) -> Result<Knn, NqlError> {
        self.expect_ident("`vector`")?;
        self.expect_token(Token::DoubleColon, "`::` after `vector`")?;
        self.expect_ident("`similarity`")?;
        self.expect_token(Token::LParen, "`(` after vector::similarity")?;
        self.expect_ident("`embedding`")?;
        self.expect_token(Token::Comma, "`,` between embedding and query vector")?;
        let query = self.parse_float_vector()?;
        self.expect_token(Token::RParen, "`)` closing vector::similarity")?;
        self.expect_keyword("and", "AND in kNN WHERE clause")?;
        self.expect_ident("`k`")?;
        self.expect_token(Token::Eq, "`=` before k")?;
        let k = self.expect_usize("k")?;
        if k == 0 {
            return Err(self.err_here("k must be positive"));
        }
        Ok(Knn { query, k })
    }

    /// `[<f32>, ...]` — numeric literal list (ints allowed, cast to f32).
    fn parse_float_vector(&mut self) -> Result<Vec<f32>, NqlError> {
        self.expect_token(Token::LBracket, "`[` starting vector literal")?;
        let mut out = Vec::new();
        loop {
            match self.peek_tok() {
                Token::RBracket => {
                    self.bump();
                    break;
                }
                Token::Int(n) => {
                    let v = *n as f32;
                    self.bump();
                    out.push(v);
                }
                Token::Float(f) => {
                    let v = *f as f32;
                    self.bump();
                    out.push(v);
                }
                other => {
                    return Err(self.err_here(format!(
                        "expected a number in vector literal, found {}",
                        describe(other)
                    )));
                }
            }
            if !self.eat_comma() {
                self.expect_token(Token::RBracket, "`]` closing vector literal")?;
                break;
            }
        }
        Ok(out)
    }

    // -- JSON-ish values -----------------------------------------------------

    /// `{ ... }` object body; returns `Value::Doc` (keys are `Str`/ident).
    fn parse_object(&mut self) -> Result<BTreeMap<String, Value>, NqlError> {
        self.expect_token(Token::LBrace, "`{` starting record body")?;
        self.parse_object_body()
    }

    fn parse_object_body(&mut self) -> Result<BTreeMap<String, Value>, NqlError> {
        let mut map = BTreeMap::new();
        loop {
            match self.peek_tok() {
                Token::RBrace => {
                    self.bump();
                    break;
                }
                Token::Str(key) | Token::Ident(key) => {
                    let key = key.clone();
                    self.bump();
                    self.expect_token(Token::Colon, "`:` after object key")?;
                    let value = self.parse_value()?;
                    map.insert(key, value);
                }
                other => {
                    return Err(self.err_here(format!(
                        "expected object key or `}}`, found {}",
                        describe(other)
                    )));
                }
            }
            if !self.eat_comma() {
                self.expect_token(Token::RBrace, "`}` closing record body")?;
                break;
            }
        }
        Ok(map)
    }

    fn parse_value(&mut self) -> Result<Value, NqlError> {
        let s = self.bump();
        match &s.tok {
            Token::Str(t) => Ok(Value::Str(t.clone())),
            Token::Int(n) => Ok(Value::Int(*n)),
            Token::Float(f) => Ok(Value::Float(*f)),
            Token::Ident(w) if w.eq_ignore_ascii_case("true") => Ok(Value::Bool(true)),
            Token::Ident(w) if w.eq_ignore_ascii_case("false") => Ok(Value::Bool(false)),
            Token::Ident(w) if w.eq_ignore_ascii_case("null") => Ok(Value::Null),
            Token::Ident(w) => Ok(Value::Str(w.clone())),
            Token::LBrace => {
                let map = self.parse_object_body()?;
                Ok(Value::Doc(map))
            }
            Token::LBracket => self.parse_array_body(),
            other => Err(self.err_at(&s, format!("expected a value, found {}", describe(other)))),
        }
    }

    /// Parse the rest of an array whose opening `[` was already consumed.
    /// All-numeric arrays collapse to `Value::Vector` (M0 contract).
    fn parse_array_body(&mut self) -> Result<Value, NqlError> {
        let mut elems = Vec::new();
        loop {
            match self.peek_tok() {
                Token::RBracket => {
                    self.bump();
                    break;
                }
                _ => {
                    let v = self.parse_value()?;
                    elems.push(v);
                    if !self.eat_comma() {
                        self.expect_token(Token::RBracket, "`]` closing array")?;
                        break;
                    }
                }
            }
        }
        // All-numeric arrays become a vector (BYO float vector).
        if !elems.is_empty()
            && elems
                .iter()
                .all(|v| matches!(v, Value::Int(_) | Value::Float(_)))
        {
            let floats = elems
                .into_iter()
                .map(|v| match v {
                    Value::Int(n) => n as f32,
                    Value::Float(f) => f as f32,
                    _ => unreachable!("checked numeric above"),
                })
                .collect();
            Ok(Value::Vector(floats))
        } else {
            Ok(Value::Arr(elems))
        }
    }

    fn expect_usize(&mut self, what: &str) -> Result<usize, NqlError> {
        let s = self.bump();
        match &s.tok {
            Token::Int(n) if *n >= 0 => Ok(*n as usize),
            Token::Int(n) => {
                Err(self.err_at(&s, format!("{what} must be non-negative, found {n}")))
            }
            other => Err(self.err_at(
                &s,
                format!(
                    "expected a non-negative integer for {what}, found {}",
                    describe(other)
                ),
            )),
        }
    }

    /// Consume a signed integer token, returning its value.
    fn expect_int(&mut self, what: &str) -> Result<i64, NqlError> {
        let s = self.bump();
        match &s.tok {
            Token::Int(n) => Ok(*n),
            other => Err(self.err_at(
                &s,
                format!("expected an integer for {what}, found {}", describe(other)),
            )),
        }
    }

    /// `::salience` order key: bare (engine defaults α=0.7, β=0, γ=0, δ=0.3) or
    /// agent-tuned `::salience(α, β, γ, δ)` — four comma-separated numbers
    /// (issue #88, spec §2.3). Weights must be finite; arity is exactly 4.
    fn parse_salience_order(&mut self) -> Result<Order, NqlError> {
        if !matches!(self.peek_tok(), Token::LParen) {
            return Ok(Order::Salience);
        }
        self.bump();
        let mut w = [0.0f32; 4];
        for (i, slot) in w.iter_mut().enumerate() {
            if i > 0 {
                self.expect_token(Token::Comma, "`,` between salience weights")?;
            }
            let s = self.bump();
            let raw: f64 = match &s.tok {
                Token::Int(n) => *n as f64,
                Token::Float(f) => *f,
                other => {
                    return Err(self.err_at(
                        &s,
                        format!(
                            "expected a number for salience weight {} of 4 \
                             (α, β, γ, δ), found {}",
                            i + 1,
                            describe(other)
                        ),
                    ))
                }
            };
            let v = raw as f32;
            if !v.is_finite() {
                return Err(self.err_at(&s, "salience weights must be finite".to_string()));
            }
            *slot = v;
        }
        self.expect_token(Token::RParen, "`)` after salience weights")?;
        Ok(Order::SalienceWeighted(w))
    }
}

fn describe(t: &Token) -> String {
    match t {
        Token::Ident(s) => format!("identifier `{s}`"),
        Token::Int(n) => format!("integer `{n}`"),
        Token::Float(f) => format!("float `{f}`"),
        Token::Str(s) => format!("string `{s}`"),
        Token::LParen => "`(`".into(),
        Token::RParen => "`)`".into(),
        Token::LBrace => "`{`".into(),
        Token::RBrace => "`}`".into(),
        Token::LBracket => "`[`".into(),
        Token::RBracket => "`]`".into(),
        Token::Semi => "`;`".into(),
        Token::Comma => "`,`".into(),
        Token::Arrow => "`->`".into(),
        Token::LeftArrow => "`<-`".into(),
        Token::Plus => "`+`".into(),
        Token::Colon => "`:`".into(),
        Token::DoubleColon => "`::`".into(),
        Token::Eq => "`=`".into(),
        Token::Ne => "`!=`".into(),
        Token::Lt => "`<`".into(),
        Token::Le => "`<=`".into(),
        Token::Gt => "`>`".into(),
        Token::Ge => "`>=`".into(),
        Token::Star => "`*`".into(),
        Token::Eof => "end of input".into(),
    }
}

fn value_name(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Bool(b) => format!("boolean `{b}`"),
        Value::Int(n) => format!("integer `{n}`"),
        Value::Float(f) => format!("float `{f}`"),
        Value::Str(s) => format!("string `{s}`"),
        Value::Doc(_) => "object".into(),
        Value::Arr(_) => "array".into(),
        Value::Vector(_) => "vector".into(),
        Value::Ref(r) => format!("reference `{r}`"),
    }
}
