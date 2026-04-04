use crate::ast::{Atom, Expr, List, MetaExpr, MetaMap};
use crate::lexer::{self, Token, TokenKind};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("lex error: {0}")]
    Lex(#[from] lexer::LexError),

    #[error("unexpected end of input at byte {offset}")]
    UnexpectedEof { offset: usize },

    #[error("expected {expected}, found {found} at byte {offset}")]
    Expected {
        expected: String,
        found: String,
        offset: usize,
    },

    #[error("empty list at byte {offset}")]
    EmptyList { offset: usize },
}

struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token]) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<&Token> {
        let tok = self.tokens.get(self.pos);
        if tok.is_some() {
            self.pos += 1;
        }
        tok
    }

    /// Current byte offset for error reporting (uses last token end if at EOF).
    fn current_offset(&self) -> usize {
        if let Some(tok) = self.tokens.get(self.pos) {
            tok.span.offset
        } else if let Some(last) = self.tokens.last() {
            last.span.end()
        } else {
            0
        }
    }

    fn expect(&mut self, expected_kind: &TokenKind) -> Result<&Token, ParseError> {
        let offset = self.current_offset();
        match self.peek() {
            Some(tok)
                if std::mem::discriminant(&tok.kind) == std::mem::discriminant(expected_kind) =>
            {
                Ok(&self.tokens[{
                    let i = self.pos;
                    self.pos += 1;
                    i
                }])
            }
            Some(tok) => Err(ParseError::Expected {
                expected: format!("{:?}", expected_kind),
                found: format!("{:?}", tok.kind),
                offset: tok.span.offset,
            }),
            None => Err(ParseError::UnexpectedEof { offset }),
        }
    }

    fn parse_exprs(&mut self) -> Result<Vec<Expr>, ParseError> {
        let mut exprs = Vec::new();
        while self.peek().is_some() {
            exprs.push(self.parse_expr()?);
        }
        Ok(exprs)
    }

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        let tok = self.peek().ok_or(ParseError::UnexpectedEof {
            offset: self.current_offset(),
        })?;

        match &tok.kind {
            TokenKind::LParen => self.parse_list(),
            TokenKind::LBrace => self.parse_map(),
            TokenKind::Caret => self.parse_meta_expr(),
            TokenKind::RParen => Err(ParseError::Expected {
                expected: "expression".to_string(),
                found: ")".to_string(),
                offset: tok.span.offset,
            }),
            TokenKind::RBrace => Err(ParseError::Expected {
                expected: "expression".to_string(),
                found: "}".to_string(),
                offset: tok.span.offset,
            }),
            _ => self.parse_atom(),
        }
    }

    fn parse_atom(&mut self) -> Result<Expr, ParseError> {
        let offset = self.current_offset();
        let tok = self.advance().ok_or(ParseError::UnexpectedEof { offset })?;
        let span = tok.span;
        let atom = match &tok.kind {
            TokenKind::Symbol(s) => Atom::Symbol(s.clone()),
            TokenKind::Int(n) => Atom::Int(*n),
            TokenKind::Float(f) => Atom::Float(*f),
            TokenKind::Str(s) => Atom::Str(s.clone()),
            TokenKind::Keyword(k) => Atom::Keyword(k.clone()),
            TokenKind::Bool(b) => Atom::Bool(*b),
            other => {
                return Err(ParseError::Expected {
                    expected: "atom".to_string(),
                    found: format!("{:?}", other),
                    offset: span.offset,
                });
            }
        };
        Ok(Expr::Atom(atom, span))
    }

    fn parse_list(&mut self) -> Result<Expr, ParseError> {
        let lparen = self.advance().unwrap(); // consume '('
        let start_span = lparen.span;

        // Check for empty list — now allowed (e.g., () as empty guard)
        if let Some(tok) = self.peek() {
            if tok.kind == TokenKind::RParen {
                let end_span = tok.span;
                self.advance();
                return Ok(Expr::List(
                    List {
                        elements: Vec::new(),
                    },
                    start_span.merge(end_span),
                ));
            }
        } else {
            return Err(ParseError::UnexpectedEof {
                offset: self.current_offset(),
            });
        }

        // Read all elements until ')'
        let mut elements = Vec::new();
        loop {
            let tok = self.peek().ok_or(ParseError::UnexpectedEof {
                offset: self.current_offset(),
            })?;
            if tok.kind == TokenKind::RParen {
                let end_span = tok.span;
                self.advance(); // consume ')'
                let full_span = start_span.merge(end_span);
                return Ok(Expr::List(List { elements }, full_span));
            }
            elements.push(self.parse_expr()?);
        }
    }

    fn parse_meta_expr(&mut self) -> Result<Expr, ParseError> {
        let caret = self.advance().unwrap(); // consume '^'
        let start_span = caret.span;

        // Expect '{'
        self.expect(&TokenKind::LBrace)?;

        // Read key-value pairs until '}'
        let mut entries = Vec::new();
        loop {
            let tok = self.peek().ok_or(ParseError::UnexpectedEof {
                offset: self.current_offset(),
            })?;
            if tok.kind == TokenKind::RBrace {
                self.advance(); // consume '}'
                break;
            }

            // Key must be a keyword
            let key_offset = self.current_offset();
            let key_tok = self
                .advance()
                .ok_or(ParseError::UnexpectedEof { offset: key_offset })?;
            let key = match &key_tok.kind {
                TokenKind::Keyword(k) => k.clone(),
                other => {
                    return Err(ParseError::Expected {
                        expected: "keyword".to_string(),
                        found: format!("{:?}", other),
                        offset: key_tok.span.offset,
                    });
                }
            };

            // Value is any expression
            let value = self.parse_expr()?;
            entries.push((key, value));
        }

        // Parse the annotated expression
        let expr = self.parse_expr()?;
        let end_span = expr.span();
        let full_span = start_span.merge(end_span);

        Ok(Expr::MetaExpr(
            MetaExpr {
                entries,
                expr: Box::new(expr),
            },
            full_span,
        ))
    }

    fn parse_map(&mut self) -> Result<Expr, ParseError> {
        let lbrace = self.advance().unwrap(); // consume '{'
        let start_span = lbrace.span;

        // Check for empty map
        if let Some(tok) = self.peek()
            && tok.kind == TokenKind::RBrace
        {
            let end_span = tok.span;
            self.advance();
            return Ok(Expr::Map(MetaMap::default(), start_span.merge(end_span)));
        }

        let mut entries = Vec::new();
        loop {
            let tok = self.peek().ok_or(ParseError::UnexpectedEof {
                offset: self.current_offset(),
            })?;
            if tok.kind == TokenKind::RBrace {
                let end_span = tok.span;
                self.advance();
                return Ok(Expr::Map(MetaMap { entries }, start_span.merge(end_span)));
            }

            // Key: a symbol
            let key_offset = self.current_offset();
            let key_tok = self
                .advance()
                .ok_or(ParseError::UnexpectedEof { offset: key_offset })?;
            let key = match &key_tok.kind {
                TokenKind::Symbol(s) => s.clone(),
                other => {
                    return Err(ParseError::Expected {
                        expected: "map key (symbol)".to_string(),
                        found: format!("{:?}", other),
                        offset: key_tok.span.offset,
                    });
                }
            };

            // Expect ':'
            let colon_tok = self.peek().ok_or(ParseError::UnexpectedEof {
                offset: self.current_offset(),
            })?;
            match &colon_tok.kind {
                TokenKind::Symbol(s) if s == ":" => {
                    self.advance();
                }
                other => {
                    return Err(ParseError::Expected {
                        expected: ":".to_string(),
                        found: format!("{:?}", other),
                        offset: colon_tok.span.offset,
                    });
                }
            }

            // Value: any expression
            let value = self.parse_expr()?;
            entries.push((key, value));

            // Optional comma between entries
            if let Some(tok) = self.peek()
                && tok.kind == TokenKind::Comma
            {
                self.advance();
            }
        }
    }
}

/// Parse a token stream into a list of expressions.
pub fn parse(tokens: &[Token]) -> Result<Vec<Expr>, ParseError> {
    let mut parser = Parser::new(tokens);
    parser.parse_exprs()
}

/// Convenience: lex and parse a source string in one step.
pub fn parse_str(source: &str) -> Result<Vec<Expr>, ParseError> {
    let tokens = lexer::lex(source)?;
    let exprs = parse(&tokens)?;
    Ok(exprs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Atom, Expr};

    /// Helper to parse and unwrap.
    fn p(source: &str) -> Vec<Expr> {
        parse_str(source).unwrap()
    }

    // ── Atom tests ─────────────────────────────────────────────

    #[test]
    fn parse_int() {
        let exprs = p("42");
        assert_eq!(exprs.len(), 1);
        match &exprs[0] {
            Expr::Atom(Atom::Int(42), _) => {}
            other => panic!("expected Int(42), got {:?}", other),
        }
    }

    #[test]
    fn parse_float() {
        let exprs = p("3.125");
        match &exprs[0] {
            Expr::Atom(Atom::Float(f), _) => assert!((f - 3.125).abs() < 1e-10),
            other => panic!("expected Float, got {:?}", other),
        }
    }

    #[test]
    fn parse_string() {
        let exprs = p(r#""hello""#);
        match &exprs[0] {
            Expr::Atom(Atom::Str(s), _) => assert_eq!(s, "hello"),
            other => panic!("expected Str, got {:?}", other),
        }
    }

    #[test]
    fn parse_symbol() {
        let exprs = p("foo");
        match &exprs[0] {
            Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, "foo"),
            other => panic!("expected Symbol, got {:?}", other),
        }
    }

    #[test]
    fn parse_keyword() {
        let exprs = p(":axis");
        match &exprs[0] {
            Expr::Atom(Atom::Keyword(k), _) => assert_eq!(k, "axis"),
            other => panic!("expected Keyword, got {:?}", other),
        }
    }

    #[test]
    fn parse_bool() {
        let exprs = p("true false");
        assert_eq!(exprs.len(), 2);
        match &exprs[0] {
            Expr::Atom(Atom::Bool(true), _) => {}
            other => panic!("expected Bool(true), got {:?}", other),
        }
        match &exprs[1] {
            Expr::Atom(Atom::Bool(false), _) => {}
            other => panic!("expected Bool(false), got {:?}", other),
        }
    }

    // ── List tests ─────────────────────────────────────────────

    #[test]
    fn parse_simple_list() {
        let exprs = p("(add 1 2)");
        assert_eq!(exprs.len(), 1);
        match &exprs[0] {
            Expr::List(list, _) => {
                assert_eq!(list.elements.len(), 3);
                match &list.elements[0] {
                    Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, "add"),
                    other => panic!("expected Symbol(add), got {:?}", other),
                }
                match &list.elements[1] {
                    Expr::Atom(Atom::Int(1), _) => {}
                    other => panic!("expected Int(1), got {:?}", other),
                }
                match &list.elements[2] {
                    Expr::Atom(Atom::Int(2), _) => {}
                    other => panic!("expected Int(2), got {:?}", other),
                }
            }
            other => panic!("expected List, got {:?}", other),
        }
    }

    #[test]
    fn parse_list_no_children() {
        // A list with just a tag and no children is valid.
        let exprs = p("(nop)");
        match &exprs[0] {
            Expr::List(list, _) => {
                assert_eq!(list.elements.len(), 1);
                match &list.elements[0] {
                    Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, "nop"),
                    other => panic!("expected Symbol(nop), got {:?}", other),
                }
            }
            other => panic!("expected List, got {:?}", other),
        }
    }

    #[test]
    fn parse_nested_lists() {
        let exprs = p("(def f (sig (-> f32 f32)) (fn (x) x))");
        assert_eq!(exprs.len(), 1);
        match &exprs[0] {
            Expr::List(list, _) => {
                assert_eq!(list.elements.len(), 4);
                // element 0: symbol "def"
                match &list.elements[0] {
                    Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, "def"),
                    other => panic!("expected Symbol(def), got {:?}", other),
                }
                // element 1: symbol "f"
                match &list.elements[1] {
                    Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, "f"),
                    other => panic!("expected Symbol(f), got {:?}", other),
                }
                // element 2: (sig (-> f32 f32))
                match &list.elements[2] {
                    Expr::List(inner, _) => {
                        assert_eq!(inner.elements.len(), 2);
                        match &inner.elements[0] {
                            Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, "sig"),
                            other => panic!("expected Symbol(sig), got {:?}", other),
                        }
                        match &inner.elements[1] {
                            Expr::List(arrow, _) => {
                                assert_eq!(arrow.elements.len(), 3);
                                match &arrow.elements[0] {
                                    Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, "->"),
                                    other => panic!("expected Symbol(->), got {:?}", other),
                                }
                            }
                            other => panic!("expected arrow list, got {:?}", other),
                        }
                    }
                    other => panic!("expected sig list, got {:?}", other),
                }
                // element 3: (fn (x) x)
                match &list.elements[3] {
                    Expr::List(inner, _) => {
                        assert_eq!(inner.elements.len(), 3);
                        match &inner.elements[0] {
                            Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, "fn"),
                            other => panic!("expected Symbol(fn), got {:?}", other),
                        }
                    }
                    other => panic!("expected fn list, got {:?}", other),
                }
            }
            other => panic!("expected List, got {:?}", other),
        }
    }

    // ── MetaExpr tests ─────────────────────────────────────────

    #[test]
    fn parse_metadata() {
        let exprs = p("^{:type f32} x");
        assert_eq!(exprs.len(), 1);
        match &exprs[0] {
            Expr::MetaExpr(meta, _) => {
                assert_eq!(meta.entries.len(), 1);
                assert_eq!(meta.entries[0].0, "type");
                match &meta.entries[0].1 {
                    Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, "f32"),
                    other => panic!("expected Symbol(f32), got {:?}", other),
                }
                match meta.expr.as_ref() {
                    Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, "x"),
                    other => panic!("expected Symbol(x), got {:?}", other),
                }
            }
            other => panic!("expected MetaExpr, got {:?}", other),
        }
    }

    #[test]
    fn parse_metadata_multiple_entries() {
        let exprs = p("^{:type f32 :pure true} (add x y)");
        assert_eq!(exprs.len(), 1);
        match &exprs[0] {
            Expr::MetaExpr(meta, _) => {
                assert_eq!(meta.entries.len(), 2);
                assert_eq!(meta.entries[0].0, "type");
                assert_eq!(meta.entries[1].0, "pure");
                match meta.expr.as_ref() {
                    Expr::List(list, _) => match &list.elements[0] {
                        Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, "add"),
                        other => panic!("expected Symbol(add), got {:?}", other),
                    },
                    other => panic!("expected List, got {:?}", other),
                }
            }
            other => panic!("expected MetaExpr, got {:?}", other),
        }
    }

    // ── Multiple top-level expressions ─────────────────────────

    #[test]
    fn parse_multiple_top_level() {
        let exprs = p("42 (add 1 2) foo");
        assert_eq!(exprs.len(), 3);
        match &exprs[0] {
            Expr::Atom(Atom::Int(42), _) => {}
            other => panic!("expected Int(42), got {:?}", other),
        }
        match &exprs[1] {
            Expr::List(list, _) => match &list.elements[0] {
                Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, "add"),
                other => panic!("expected Symbol(add), got {:?}", other),
            },
            other => panic!("expected List, got {:?}", other),
        }
        match &exprs[2] {
            Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, "foo"),
            other => panic!("expected Symbol(foo), got {:?}", other),
        }
    }

    // ── Error tests ────────────────────────────────────────────

    #[test]
    fn error_unmatched_lparen() {
        let result = parse_str("(add 1 2");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("unexpected end of input"),
            "expected EOF error, got: {}",
            err
        );
    }

    #[test]
    fn error_unmatched_rparen() {
        let result = parse_str(")");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("found )"),
            "expected unexpected ')' error, got: {}",
            err
        );
    }

    #[test]
    fn parse_empty_list_allowed() {
        // Empty lists () are valid in the new spec (e.g., empty guard in match arms)
        let exprs = parse_str("()").unwrap();
        assert_eq!(exprs.len(), 1);
        match &exprs[0] {
            Expr::List(list, _) => assert!(list.elements.is_empty()),
            other => panic!("expected empty list, got: {:?}", other),
        }
    }

    #[test]
    fn parse_list_non_symbol_first_element() {
        // Any expr can be the first element of a list.
        let exprs = p("(42 a b)");
        assert_eq!(exprs.len(), 1);
        match &exprs[0] {
            Expr::List(list, _) => {
                assert_eq!(list.elements.len(), 3);
                match &list.elements[0] {
                    Expr::Atom(Atom::Int(42), _) => {}
                    other => panic!("expected Int(42), got {:?}", other),
                }
            }
            other => panic!("expected List, got {:?}", other),
        }
    }

    #[test]
    fn error_unexpected_rbrace() {
        let result = parse_str("}");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("found }"),
            "expected unexpected '}}' error, got: {}",
            err
        );
    }

    // ── Span tests ─────────────────────────────────────────────

    #[test]
    fn span_covers_full_list() {
        let exprs = p("(add 1 2)");
        let span = exprs[0].span();
        // '(' starts at 0, ')' is at byte 8, so span is 0..9
        assert_eq!(span.offset, 0);
        assert_eq!(span.end(), 9);
    }

    #[test]
    fn span_covers_meta_expr() {
        let exprs = p("^{:type f32} x");
        let span = exprs[0].span();
        // '^' at 0, 'x' at 13 with len 1 → end 14
        assert_eq!(span.offset, 0);
        assert_eq!(span.end(), 14);
    }

    // ── Empty input ────────────────────────────────────────────

    #[test]
    fn parse_empty_input() {
        let exprs = p("");
        assert!(exprs.is_empty());
    }
}
