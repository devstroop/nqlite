//! Hand-written tokenizer for the M0 nql grammar slice.
//!
//! Produces a flat `Vec<Spanned>` of tokens, each carrying its 1-based line and
//! column so the parser can emit structured `NqlError` positions. Keywords are
//! NOT distinguished here — they are all lexed as `Ident` and the parser matches
//! them case-insensitively so they can double as table/field names where legal.

use crate::parser::NqlError;

/// A single lexical token. Keywords are contextual, so they arrive as `Ident`.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    /// A bare word: keyword, table name, field name, tag, `f32`, `vector`, ...
    Ident(String),
    Int(i64),
    Float(f64),
    /// A quoted string literal (single- or double-quoted).
    Str(String),
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    /// `->`
    Arrow,
    Plus,
    /// `:`
    Colon,
    /// `::`
    DoubleColon,
    Eq,
    Lt,
    Gt,
    Star,
    Eof,
}

/// A token plus its occurrence position (1-based line and column).
#[derive(Debug, Clone)]
pub struct Spanned {
    pub tok: Token,
    pub line: usize,
    pub col: usize,
}

struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
    line: usize,
    col: usize,
}

/// Tokenize `input`, returning the ordered token stream (terminated by `Eof`).
pub fn tokenize(input: &str) -> Result<Vec<Spanned>, NqlError> {
    let mut lx = Lexer {
        src: input.as_bytes(),
        pos: 0,
        line: 1,
        col: 1,
    };
    let mut out = Vec::new();
    loop {
        let t = lx.next_token()?;
        let end = t.tok == Token::Eof;
        out.push(t);
        if end {
            break;
        }
    }
    Ok(out)
}

impl<'a> Lexer<'a> {
    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn peek2(&self) -> Option<u8> {
        self.src.get(self.pos + 1).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        if b == b'\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(b)
    }

    fn err(&self, msg: impl Into<String>) -> NqlError {
        NqlError::syntax(msg.into(), self.line, self.col)
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            match b {
                b' ' | b'\t' | b'\n' | b'\r' => {
                    self.bump();
                }
                _ => break,
            }
        }
    }

    fn next_token(&mut self) -> Result<Spanned, NqlError> {
        self.skip_ws();
        let line = self.line;
        let col = self.col;
        let tok = match self.peek() {
            None => Token::Eof,
            Some(b'(') => {
                self.bump();
                Token::LParen
            }
            Some(b')') => {
                self.bump();
                Token::RParen
            }
            Some(b'{') => {
                self.bump();
                Token::LBrace
            }
            Some(b'}') => {
                self.bump();
                Token::RBrace
            }
            Some(b'[') => {
                self.bump();
                Token::LBracket
            }
            Some(b']') => {
                self.bump();
                Token::RBracket
            }
            Some(b',') => {
                self.bump();
                Token::Comma
            }
            Some(b'+') => {
                self.bump();
                Token::Plus
            }
            Some(b'*') => {
                self.bump();
                Token::Star
            }
            Some(b'=') => {
                self.bump();
                Token::Eq
            }
            Some(b'<') => {
                self.bump();
                Token::Lt
            }
            Some(b'>') => {
                self.bump();
                Token::Gt
            }
            Some(b':') => {
                self.bump();
                if self.peek() == Some(b':') {
                    self.bump();
                    Token::DoubleColon
                } else {
                    Token::Colon
                }
            }
            Some(b'-') => {
                self.bump();
                if self.peek() == Some(b'>') {
                    self.bump();
                    Token::Arrow
                } else if self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    self.read_number(true)?
                } else {
                    return Err(self.err("unexpected character '-' (expected `->` or a number)"));
                }
            }
            Some(q @ (b'\'' | b'"')) => self.read_string(q).map(Token::Str)?,
            Some(b) if b.is_ascii_digit() => self.read_number(false)?,
            Some(b) if is_ident_start(b) || b >= 0x80 => {
                let word = self.read_ident();
                Token::Ident(word)
            }
            Some(b) => {
                return Err(self.err(format!(
                    "unexpected character `{}`",
                    (b as char).escape_default()
                )));
            }
        };
        Ok(Spanned { tok, line, col })
    }

    fn read_ident(&mut self) -> String {
        let mut out = Vec::new();
        while let Some(b) = self.peek() {
            if is_ident_continue(b) || b >= 0x80 {
                out.push(b);
                self.bump();
            } else {
                break;
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    /// Read a possibly-negative numeric literal. Assumes `self` is positioned
    /// just after an optional sign and at a leading digit.
    fn read_number(&mut self, neg: bool) -> Result<Token, NqlError> {
        let line = self.line;
        let col = self.col;
        let mut s = String::new();
        if neg {
            s.push('-');
        }
        let mut is_float = false;

        while let Some(b) = self.peek() {
            if b.is_ascii_digit() {
                s.push(b as char);
                self.bump();
            } else {
                break;
            }
        }
        // Fractional part.
        if self.peek() == Some(b'.') && self.peek2().is_some_and(|c| c.is_ascii_digit()) {
            is_float = true;
            s.push('.');
            self.bump();
            while let Some(b) = self.peek() {
                if b.is_ascii_digit() {
                    s.push(b as char);
                    self.bump();
                } else {
                    break;
                }
            }
        }
        // Exponent.
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            is_float = true;
            s.push('e');
            self.bump();
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                s.push(self.bump().unwrap() as char);
            }
            if !self.peek().is_some_and(|c| c.is_ascii_digit()) {
                return Err(NqlError::syntax("malformed numeric exponent", line, col));
            }
            while let Some(b) = self.peek() {
                if b.is_ascii_digit() {
                    s.push(b as char);
                    self.bump();
                } else {
                    break;
                }
            }
        }

        if is_float {
            s.parse::<f64>()
                .map(Token::Float)
                .map_err(|_| NqlError::syntax(format!("invalid float literal `{s}`"), line, col))
        } else {
            match s.parse::<i64>() {
                Ok(n) => Ok(Token::Int(n)),
                // Overflow any signed i64 — fall back to a float.
                Err(_) => s.parse::<f64>().map(Token::Float).map_err(|_| {
                    NqlError::syntax(format!("invalid integer literal `{s}`"), line, col)
                }),
            }
        }
    }

    fn read_string(&mut self, quote: u8) -> Result<String, NqlError> {
        let line = self.line;
        let col = self.col;
        self.bump(); // opening quote
        let mut out = String::new();
        loop {
            let Some(b) = self.peek() else {
                return Err(NqlError::syntax("unterminated string literal", line, col));
            };
            match b {
                _ if b == quote => {
                    self.bump();
                    break;
                }
                b'\\' => {
                    self.bump();
                    let Some(esc) = self.peek() else {
                        return Err(NqlError::syntax("unterminated string literal", line, col));
                    };
                    self.bump();
                    match esc {
                        b'"' => out.push('"'),
                        b'\'' => out.push('\''),
                        b'\\' => out.push('\\'),
                        b'n' => out.push('\n'),
                        b't' => out.push('\t'),
                        b'r' => out.push('\r'),
                        other => {
                            return Err(NqlError::syntax(
                                format!(
                                    "invalid escape sequence `\\{}`",
                                    (other as char).escape_default()
                                ),
                                line,
                                col,
                            ));
                        }
                    }
                }
                b'\n' => {
                    return Err(NqlError::syntax(
                        "unterminated string literal (newline before closing quote)",
                        line,
                        col,
                    ));
                }
                _ => {
                    // Push raw UTF-8 bytes; two-byte-plus sequences decode as-is.
                    if b < 0x80 {
                        out.push(b as char);
                        self.bump();
                    } else {
                        let start = self.pos;
                        let len = utf8_len(b);
                        for _ in 0..len.saturating_sub(1) {
                            self.bump();
                        }
                        self.bump();
                        if let Ok(s) = std::str::from_utf8(&self.src[start..self.pos]) {
                            out.push_str(s);
                        } else {
                            return Err(NqlError::syntax(
                                "invalid UTF-8 in string literal",
                                line,
                                col,
                            ));
                        }
                    }
                }
            }
        }
        Ok(out)
    }
}

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

fn is_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Number of bytes in a UTF-8 leading byte.
fn utf8_len(b: u8) -> usize {
    if b >= 0xF0 {
        4
    } else if b >= 0xE0 {
        3
    } else if b >= 0xC0 {
        2
    } else {
        1
    }
}
