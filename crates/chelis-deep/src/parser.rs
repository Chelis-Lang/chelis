use crate::ast::{Atom, Expr, List, MetaExpr, MetaMap};
use crate::lexer::{self, Token, TokenKind};
use thiserror::Error;

/// Returns true when `b` is a forbidden ASCII byte in a `span` metadata
/// value per `spec/03-deep-syntax.md` §1.1.1: U+0000..=U+001F (except
/// U+0020 space) or U+007F (DEL).
///
/// Mirrored in `chelis_ir::span_sanitize::is_forbidden_byte`. The two
/// definitions are kept in sync via the spec — any change to the
/// forbidden set must update both `spec/03-deep-syntax.md` §1.1.1 and
/// both implementations together.
#[inline]
fn is_forbidden_span_byte(b: u8) -> bool {
    b <= 0x1F || b == 0x7F
}

/// Render a forbidden byte for diagnostic messages, using a printable
/// canonical form. `\n`, `\r`, `\t`, `\0` get their backslash form;
/// other forbidden bytes (other C0 controls and U+007F) are shown as
/// `\xNN` two-digit lowercase hex.
fn forbidden_byte_repr(b: u8) -> String {
    match b {
        0x00 => "\\0".to_string(),
        0x09 => "\\t".to_string(),
        0x0A => "\\n".to_string(),
        0x0D => "\\r".to_string(),
        other => format!("\\x{other:02x}"),
    }
}

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

    /// A `span` metadata value contains a forbidden character per
    /// `spec/03-deep-syntax.md` §1.1.1. Span IDs must not contain ASCII
    /// control characters (U+0000..=U+001F except U+0020) or U+007F (DEL).
    /// Forbidden code points can terminate `//` line comments in generated
    /// C/HIP/Metal source and inject live code into the build artifact.
    ///
    /// `value_offset` is the byte offset of the string literal in the
    /// Deep source. `byte_in_value` is the offset of the offending byte
    /// inside the decoded span string (after escape processing). `repr`
    /// is the canonical printable form of the offending byte (`\n`,
    /// `\xNN`, …) and `code_point` is the U+NNNN form for the diagnostic.
    #[error(
        "span metadata value contains forbidden character at byte {byte_in_value} ({repr}, U+{code_point:04X}) at byte {value_offset}; span IDs must not contain ASCII control characters; see spec/03-deep-syntax.md §1.1.1"
    )]
    ForbiddenSpanChar {
        value_offset: usize,
        byte_in_value: usize,
        code_point: u32,
        repr: String,
    },
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
            // Typed-suffix literals (spec/03-deep-syntax.md §6.4.1): the
            // producer-friendly shape `(lit {} 1.0f64)` is canonicalized
            // into `(lit {type: (t-prim {} f64)} 1.0)`. When a typed
            // suffix appears as a bare atom, expand it into a synthetic
            // `(lit {type: ...} N)` list so the type checker sees the
            // suffix as an explicit type ascription.
            TokenKind::TypedInt(n, suffix) => {
                return Ok(typed_literal_lit_expr(Atom::Int(*n), *suffix, span));
            }
            TokenKind::TypedFloat(f, suffix) => {
                return Ok(typed_literal_lit_expr(Atom::Float(*f), *suffix, span));
            }
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

            // Span-charset enforcement (spec §1.1.1): same rule as in
            // `parse_map`, applied to the legacy `^{:span "..."}` prefix
            // metadata form so producers cannot bypass the forbidden-char
            // check via this entry point.
            if key == "span"
                && let Expr::Atom(Atom::Str(s), value_span) = &value
                && let Some((idx, b)) = s
                    .as_bytes()
                    .iter()
                    .copied()
                    .enumerate()
                    .find(|(_, b)| is_forbidden_span_byte(*b))
            {
                return Err(ParseError::ForbiddenSpanChar {
                    value_offset: value_span.offset,
                    byte_in_value: idx,
                    code_point: u32::from(b),
                    repr: forbidden_byte_repr(b),
                });
            }

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

            // Span-charset enforcement (spec §1.1.1): when the key is
            // `span` and the value is a string atom, reject any forbidden
            // ASCII control character. The constraint exists so backend
            // emitters can interpolate spans into `//` line comments
            // without a forbidden byte (notably `\n`) terminating the
            // comment and turning the rest of the value into live code.
            if key == "span"
                && let Expr::Atom(Atom::Str(s), value_span) = &value
                && let Some((idx, b)) = s
                    .as_bytes()
                    .iter()
                    .copied()
                    .enumerate()
                    .find(|(_, b)| is_forbidden_span_byte(*b))
            {
                return Err(ParseError::ForbiddenSpanChar {
                    value_offset: value_span.offset,
                    byte_in_value: idx,
                    code_point: u32::from(b),
                    repr: forbidden_byte_repr(b),
                });
            }

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

/// Collapse the producer-friendly shape `(lit {meta} <typed-lit>)` (where
/// `<typed-lit>` is the synthetic node created by `typed_literal_lit_expr`
/// from a suffixed literal token) into the canonical
/// `(lit {meta + type: ...} <value>)` form per `spec/03-deep-syntax.md`
/// §6.4.1.
///
/// Errors when the outer `lit` already carries a `type` metadata key that
/// disagrees with the suffix; the suffix is the user-facing intent and
/// silent overwrite would violate the "no implicit precision promotion"
/// rule.
///
/// Performance note: this is called only on `lit`-tagged lists (the
/// caller pre-filters), so the per-call cost is bounded. Marked
/// `#[inline(never)]` so the parse_list hot path keeps a small stack
/// frame and the deep-recursion test (1000 nested apps) does not bloat.
#[inline(never)]
fn collapse_typed_literal_lit(list: List, span: crate::Span) -> Result<List, ParseError> {
    // Cheap pre-checks: must be exactly 3 elements, tag must be `lit`,
    // child must itself be a synthetic typed-`lit` of the exact shape
    // `typed_literal_lit_expr` produces.
    if list.elements.len() != 3 {
        return Ok(list);
    }
    let is_outer_lit = matches!(
        &list.elements[0],
        Expr::Atom(Atom::Symbol(s), _) if s == "lit"
    );
    if !is_outer_lit {
        return Ok(list);
    }
    let outer_is_map = matches!(&list.elements[1], Expr::Map(_, _));
    if !outer_is_map {
        return Ok(list);
    }
    let inner_list = match &list.elements[2] {
        Expr::List(inner, _) => inner,
        _ => return Ok(list),
    };
    if inner_list.elements.len() != 3 {
        return Ok(list);
    }
    let inner_is_lit = matches!(
        &inner_list.elements[0],
        Expr::Atom(Atom::Symbol(s), _) if s == "lit"
    );
    if !inner_is_lit {
        return Ok(list);
    }
    let inner_meta = match &inner_list.elements[1] {
        Expr::Map(m, _) => m,
        _ => return Ok(list),
    };
    // Inner must carry exactly the synthetic single `type` key our
    // `typed_literal_lit_expr` produces. Be conservative: only collapse
    // when the inner meta is exactly one `type: (t-prim {} <prim>)` entry
    // and the inner child is an `Atom::Int` or `Atom::Float`.
    if inner_meta.entries.len() != 1 || inner_meta.entries[0].0 != "type" {
        return Ok(list);
    }
    let inner_type_ok = matches!(
        &inner_meta.entries[0].1,
        Expr::List(t_prim_list, _)
            if t_prim_list.elements.len() == 3
                && matches!(
                    &t_prim_list.elements[0],
                    Expr::Atom(Atom::Symbol(s), _) if s == "t-prim"
                )
    );
    if !inner_type_ok {
        return Ok(list);
    }
    if !matches!(
        &inner_list.elements[2],
        Expr::Atom(Atom::Int(_) | Atom::Float(_), _)
    ) {
        return Ok(list);
    }

    // Committed to collapsing — destructure to take ownership without
    // cloning the deep subtree.
    let mut iter = list.elements.into_iter();
    let outer_tag = iter.next().expect("checked above");
    let outer_meta_expr = iter.next().expect("checked above");
    let inner_expr = iter.next().expect("checked above");
    let outer_meta = match outer_meta_expr {
        Expr::Map(m, _) => m,
        _ => unreachable!("checked above"),
    };
    let inner_list = match inner_expr {
        Expr::List(l, _) => l,
        _ => unreachable!("checked above"),
    };
    let mut inner_iter = inner_list.elements.into_iter();
    let _inner_tag = inner_iter.next();
    let inner_meta_expr = inner_iter.next();
    let inner_value = inner_iter.next().expect("checked above");
    let inner_meta = match inner_meta_expr {
        Some(Expr::Map(m, _)) => m,
        _ => unreachable!("checked above"),
    };
    let inner_type_owned = inner_meta
        .entries
        .into_iter()
        .next()
        .expect("checked above")
        .1;

    // If the outer already declares a `type`, it must match the inner
    // synthetic type exactly. A mismatch is a parse error per the
    // no-implicit-promotion rule.
    if let Some((_, outer_type_expr)) = outer_meta.entries.iter().find(|(k, _)| k == "type") {
        if *outer_type_expr != inner_type_owned {
            return Err(ParseError::Expected {
                expected: "consistent literal type (outer `lit` declares one type but \
                     inner suffix declares another); see spec/03-deep-syntax.md §6.4.1"
                    .to_string(),
                found: "conflicting type metadata in nested `lit`".to_string(),
                offset: span.offset,
            });
        }
        // Outer already has a matching type entry; just unwrap.
        return Ok(List {
            elements: vec![outer_tag, Expr::Map(outer_meta, span), inner_value],
        });
    }
    // Merge: extend outer entries with the synthetic `type` entry.
    let mut merged_entries = outer_meta.entries;
    merged_entries.push(("type".to_string(), inner_type_owned));
    Ok(List {
        elements: vec![
            outer_tag,
            Expr::Map(
                MetaMap {
                    entries: merged_entries,
                },
                span,
            ),
            inner_value,
        ],
    })
}

/// Build the canonical Deep `lit`-with-type-metadata expression for a
/// suffixed literal token per `spec/03-deep-syntax.md` §6.4.1. The
/// producer-friendly bare `1.0f64` token expands into
/// `(lit {type: (t-prim {} f64)} 1.0)` so the type checker sees the
/// suffix as an explicit type ascription.
fn typed_literal_lit_expr(value: Atom, suffix: lexer::LiteralSuffix, span: crate::Span) -> Expr {
    let prim_name = suffix.t_prim_name();
    // (t-prim {} <prim_name>)
    let t_prim = Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Symbol("t-prim".to_string()), span),
                Expr::Map(MetaMap::default(), span),
                Expr::Atom(Atom::Symbol(prim_name.to_string()), span),
            ],
        },
        span,
    );
    let meta = Expr::Map(
        MetaMap {
            entries: vec![("type".to_string(), t_prim)],
        },
        span,
    );
    // (lit {type: (t-prim {} <prim_name>)} <value>)
    Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Symbol("lit".to_string()), span),
                meta,
                Expr::Atom(value, span),
            ],
        },
        span,
    )
}

/// Walk the parsed tree and apply `collapse_typed_literal_lit` to every
/// `lit`-tagged list. Done as a post-pass so it does not bloat the
/// `parse_list` stack frame on the deep-recursion hot path.
///
/// Implemented as an explicit work-stack iteration rather than a
/// recursive walk. This is foundational: the iterative collapse keeps
/// the post-pass's stack growth O(1) per AST level, so the post-pass
/// does not stack on top of the (already-recursive) `parse_list` and
/// `print_canonical` walks for deeply-nested inputs (e.g. the
/// 1000-nested-apps regression test). The previous recursive
/// implementation pushed total stack use over the macOS Smoke CI
/// thread-stack budget at depth ~1000.
fn normalize_typed_literals(exprs: &mut [Expr]) -> Result<(), ParseError> {
    for expr in exprs {
        normalize_typed_literals_in_expr(expr)?;
    }
    Ok(())
}

fn normalize_typed_literals_in_expr(expr: &mut Expr) -> Result<(), ParseError> {
    // Iterative post-order traversal: each work item is a raw pointer to
    // an `Expr` along with a visited flag. On first visit we push the
    // node back as visited and then push all of its children (so they
    // are processed before we revisit the parent). On second visit we
    // apply the per-node collapse, by which point every descendant has
    // already been normalized — matching the post-order semantics of
    // the prior recursive walk.
    //
    // Safety: we hold a unique `&mut Expr` borrow at entry and the only
    // mutation we perform on a node is via `mem::replace` on the inner
    // `List` of an `Expr::List` variant (the enum tag stays `List`, and
    // the parent's Vec slot that addresses this node is not touched).
    // Pointers to descendants are only used for their initial visit and
    // their post-order revisit, both of which are scheduled before any
    // ancestor mutation. After an ancestor's collapse runs, no pointers
    // into that ancestor's subtree remain on the stack.
    let mut stack: Vec<(*mut Expr, bool)> = Vec::new();
    stack.push((expr as *mut Expr, false));

    while let Some((ptr, visited)) = stack.pop() {
        // SAFETY: the pointer was derived from a unique `&mut Expr` we
        // own for the duration of this function. No aliasing borrows
        // exist; no ancestor mutation has invalidated the slot (see the
        // post-order argument above).
        let node = unsafe { &mut *ptr };
        if !visited {
            // Re-enqueue self for post-order processing, then enqueue
            // children for pre-order descent.
            stack.push((ptr, true));
            match node {
                Expr::List(list, _) => {
                    for child in list.elements.iter_mut() {
                        stack.push((child as *mut Expr, false));
                    }
                }
                Expr::Map(map, _) => {
                    for (_, v) in map.entries.iter_mut() {
                        stack.push((v as *mut Expr, false));
                    }
                }
                Expr::MetaExpr(me, _) => {
                    stack.push((&mut *me.expr as *mut Expr, false));
                    for (_, v) in me.entries.iter_mut() {
                        stack.push((v as *mut Expr, false));
                    }
                }
                Expr::Atom(_, _) => {}
            }
        } else if let Expr::List(list, span) = node {
            // All descendants already normalized. Check shape and
            // collapse if applicable.
            let needs_collapse = list.elements.len() == 3
                && matches!(
                    &list.elements[0],
                    Expr::Atom(Atom::Symbol(s), _) if s == "lit"
                );
            if needs_collapse {
                let span_copy = *span;
                let taken = std::mem::replace(
                    list,
                    List {
                        elements: Vec::new(),
                    },
                );
                *list = collapse_typed_literal_lit(taken, span_copy)?;
            }
        }
        // Atoms, Maps, and MetaExprs need no per-node action on the
        // post-order revisit; their children have been handled above.
    }
    Ok(())
}

/// Parse a token stream into a list of expressions.
pub fn parse(tokens: &[Token]) -> Result<Vec<Expr>, ParseError> {
    let mut parser = Parser::new(tokens);
    let mut exprs = parser.parse_exprs()?;
    normalize_typed_literals(&mut exprs)?;
    Ok(exprs)
}

/// Convenience: lex and parse a source string in one step.
pub fn parse_str(source: &str) -> Result<Vec<Expr>, ParseError> {
    let tokens = lexer::lex(source)?;
    let exprs = parse(&tokens)?;
    Ok(exprs)
}

/// Strict parse: lex, parse, then validate tag vocabulary.
/// Returns error if any unknown tags are found.
pub fn parse_str_strict(source: &str) -> Result<Vec<Expr>, ParseError> {
    let tokens = lexer::lex(source)?;
    let exprs = parse(&tokens)?;
    let warnings = crate::validate::validate(&exprs);
    if let Some(w) = warnings.first() {
        return Err(ParseError::Expected {
            expected: "valid Deep tag".to_string(),
            found: w.message.clone(),
            offset: w.offset,
        });
    }
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
        // Post-sprint canonical form: (def {} f (fn {} (params {} x) (var {} x)))
        let exprs = p("(def {} f (fn {} (params {} x) (var {} x)))");
        assert_eq!(exprs.len(), 1);
        match &exprs[0] {
            Expr::List(list, _) => {
                assert_eq!(list.elements.len(), 4); // def, {}, f, (fn ...)
                match &list.elements[0] {
                    Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, "def"),
                    other => panic!("expected Symbol(def), got {:?}", other),
                }
                match &list.elements[1] {
                    Expr::Map(m, _) => assert!(m.entries.is_empty()),
                    other => panic!("expected empty Map, got {:?}", other),
                }
                match &list.elements[2] {
                    Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, "f"),
                    other => panic!("expected Symbol(f), got {:?}", other),
                }
                match &list.elements[3] {
                    Expr::List(func, _) => match &func.elements[0] {
                        Expr::Atom(Atom::Symbol(s), _) => assert_eq!(s, "fn"),
                        other => panic!("expected Symbol(fn), got {:?}", other),
                    },
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
