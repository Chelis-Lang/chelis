use crate::ast::Expr;
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

fn validate_surf_metadata_key(key: &str, offset: usize) -> Result<(), ParseError> {
    if key.starts_with("surf_")
        && !matches!(
            key,
            "surf_path"
                | "surf_dim_group_size"
                | "surf_pipe_stage"
                | "surf_literal_style"
                | "surf_binding_type"
        )
    {
        return Err(ParseError::Expected {
            expected: "a key in the closed Surf metadata namespace (`surf_path`, `surf_dim_group_size`, `surf_pipe_stage`, `surf_literal_style`, or `surf_binding_type`)".to_string(),
            found: key.to_string(),
            offset,
        });
    }
    Ok(())
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

// ═══════════════════════════════════════════════════════════════════
// Raw parser: emits RawExpr (untyped, no tag decode, no role gate)
// ═══════════════════════════════════════════════════════════════════

use crate::raw::{RawAtom, RawExpr};

struct RawParser<'a> {
    tokens: &'a [Token],
    pos: usize,
}

impl<'a> RawParser<'a> {
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

    fn parse_exprs(&mut self) -> Result<Vec<RawExpr>, ParseError> {
        let mut exprs = Vec::new();
        while self.peek().is_some() {
            exprs.push(self.parse_expr()?);
        }
        Ok(exprs)
    }

    fn parse_expr(&mut self) -> Result<RawExpr, ParseError> {
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

    fn parse_atom(&mut self) -> Result<RawExpr, ParseError> {
        let offset = self.current_offset();
        let tok = self.advance().ok_or(ParseError::UnexpectedEof { offset })?;
        let span = tok.span;
        let atom = match &tok.kind {
            TokenKind::Symbol(s) => RawAtom::Symbol(s.clone()),
            TokenKind::Int(n) => RawAtom::Int(*n),
            TokenKind::Float(f) => RawAtom::Float(*f),
            TokenKind::TypedInt(n, suffix) => {
                return Ok(raw_typed_literal_lit_expr(RawAtom::Int(*n), *suffix, span));
            }
            TokenKind::TypedFloat(f, suffix) => {
                return Ok(raw_typed_literal_lit_expr(
                    RawAtom::Float(*f),
                    *suffix,
                    span,
                ));
            }
            TokenKind::Str(s) => RawAtom::Str(s.clone()),
            TokenKind::Keyword(_) => {
                return Err(ParseError::Expected {
                    expected: "expression (bare :keyword is valid only as a metadata map key)"
                        .to_string(),
                    found: format!("{:?}", tok.kind),
                    offset: span.offset,
                });
            }
            TokenKind::Bool(b) => RawAtom::Bool(*b),
            other => {
                return Err(ParseError::Expected {
                    expected: "atom".to_string(),
                    found: format!("{:?}", other),
                    offset: span.offset,
                });
            }
        };
        Ok(RawExpr::Atom(atom, span))
    }

    fn parse_list(&mut self) -> Result<RawExpr, ParseError> {
        let lparen = self.advance().unwrap();
        let start_span = lparen.span;

        if let Some(tok) = self.peek() {
            if tok.kind == TokenKind::RParen {
                let end_span = tok.span;
                self.advance();
                return Ok(RawExpr::List(Vec::new(), start_span.merge(end_span)));
            }
        } else {
            return Err(ParseError::UnexpectedEof {
                offset: self.current_offset(),
            });
        }

        let mut elements = Vec::new();
        loop {
            let tok = self.peek().ok_or(ParseError::UnexpectedEof {
                offset: self.current_offset(),
            })?;
            if tok.kind == TokenKind::RParen {
                let end_span = tok.span;
                self.advance();
                let full_span = start_span.merge(end_span);
                return Ok(normalize_typed_literal_wrapper(elements, full_span));
            }
            elements.push(self.parse_expr()?);
        }
    }

    fn parse_meta_expr(&mut self) -> Result<RawExpr, ParseError> {
        let caret = self.advance().unwrap();
        let start_span = caret.span;

        self.expect(&TokenKind::LBrace)?;

        let mut entries = Vec::new();
        loop {
            let tok = self.peek().ok_or(ParseError::UnexpectedEof {
                offset: self.current_offset(),
            })?;
            if tok.kind == TokenKind::RBrace {
                self.advance();
                break;
            }

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
            validate_surf_metadata_key(&key, key_tok.span.offset)?;

            let value = self.parse_expr()?;

            // Span-charset enforcement (spec §1.1.1)
            if key == "span"
                && let RawExpr::Atom(RawAtom::Str(s), value_span) = &value
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

        let expr = self.parse_expr()?;
        let end_span = expr.span();
        let full_span = start_span.merge(end_span);

        Ok(RawExpr::MetaExpr {
            entries,
            expr: Box::new(expr),
            span: full_span,
        })
    }

    fn parse_map(&mut self) -> Result<RawExpr, ParseError> {
        let lbrace = self.advance().unwrap();
        let start_span = lbrace.span;

        if let Some(tok) = self.peek()
            && tok.kind == TokenKind::RBrace
        {
            let end_span = tok.span;
            self.advance();
            return Ok(RawExpr::Map(Vec::new(), start_span.merge(end_span)));
        }

        let mut entries = Vec::new();
        loop {
            let tok = self.peek().ok_or(ParseError::UnexpectedEof {
                offset: self.current_offset(),
            })?;
            if tok.kind == TokenKind::RBrace {
                let end_span = tok.span;
                self.advance();
                return Ok(RawExpr::Map(entries, start_span.merge(end_span)));
            }

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
            validate_surf_metadata_key(&key, key_tok.span.offset)?;

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

            let value = self.parse_expr()?;

            // Span-charset enforcement (spec §1.1.1)
            if key == "span"
                && let RawExpr::Atom(RawAtom::Str(s), value_span) = &value
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

            if let Some(tok) = self.peek()
                && tok.kind == TokenKind::Comma
            {
                self.advance();
            }
        }
    }
}

/// Build a raw typed-literal `(lit {type: (t-prim {} <prim>)} <value>)` for
/// suffixed literal tokens, mirroring `typed_literal_lit_expr` but in the
/// `RawExpr` domain.
fn raw_typed_literal_lit_expr(
    value: RawAtom,
    suffix: lexer::LiteralSuffix,
    span: crate::Span,
) -> RawExpr {
    let prim_name = suffix.t_prim_name();
    let integer_spelled_float =
        matches!(value, RawAtom::Int(_)) && matches!(prim_name, "f32" | "f64" | "bf16" | "f16");
    let t_prim = RawExpr::List(
        vec![
            RawExpr::Atom(RawAtom::Symbol("t-prim".to_string()), span),
            RawExpr::Map(vec![], span),
            RawExpr::Atom(RawAtom::Symbol(prim_name.to_string()), span),
        ],
        span,
    );
    let mut entries = vec![("type".to_string(), t_prim)];
    if integer_spelled_float {
        // Retain the exact i64 payload until target-width finalization. Turning
        // it into f64 here can double-round f32/f16/bf16 literals above 2^53.
        entries.push((
            "literal_source".to_string(),
            RawExpr::Atom(RawAtom::Symbol("integer".to_string()), span),
        ));
    }
    let meta = RawExpr::Map(entries, span);
    RawExpr::List(
        vec![
            RawExpr::Atom(RawAtom::Symbol("lit".to_string()), span),
            meta,
            RawExpr::Atom(value, span),
        ],
        span,
    )
}

/// A suffixed token already expands to a complete typed `lit`. When it appears
/// in the producer-friendly spelling `(lit {} 7f32)`, avoid leaving that
/// expansion nested as the outer literal's value. The empty outer metadata is
/// deliberate: a producer that supplies metadata must emit the canonical form
/// and resolve any competing `type` entry itself.
fn normalize_typed_literal_wrapper(elements: Vec<RawExpr>, span: crate::Span) -> RawExpr {
    let expanded = match elements.as_slice() {
        [
            RawExpr::Atom(RawAtom::Symbol(outer_tag), _),
            RawExpr::Map(outer_meta, _),
            RawExpr::List(inner, _),
        ] if outer_tag == "lit" && outer_meta.is_empty() => inner,
        _ => return RawExpr::List(elements, span),
    };

    let is_typed_literal_expansion = matches!(
        expanded.as_slice(),
        [
            RawExpr::Atom(RawAtom::Symbol(inner_tag), _),
            RawExpr::Map(inner_meta, _),
            RawExpr::Atom(_, _),
        ] if inner_tag == "lit" && inner_meta.iter().any(|(key, _)| key == "type")
    );

    if is_typed_literal_expansion {
        RawExpr::List(expanded.to_vec(), span)
    } else {
        RawExpr::List(elements, span)
    }
}

/// Parse a token stream into raw (untyped) expressions.
///
/// Parse a token stream into a list of typed expressions.
///
/// Routes through the raw parser + `stamp_to_typed` pipeline. All production
/// goes through `RawParser` → `stamp_to_typed`, which produces `Expr::Node`,
/// `Expr::BareList`, and `Expr::UnknownForm`.
pub fn parse(tokens: &[Token]) -> Result<Vec<Expr>, ParseError> {
    let raw_exprs = parse_raw(tokens)?;
    let typed = crate::stamp_to_typed::stamp_exprs_lenient(raw_exprs).map_err(|e| {
        ParseError::Expected {
            expected: "valid Deep structure".to_string(),
            found: format!("{e}"),
            offset: 0,
        }
    })?;
    Ok(typed)
}

/// Decode-once stamping for hand-built `Expr::List` trees in tests.
///
/// **DEPRECATED**: Only needed by test code that manually constructs
/// `Expr::List`. New test code should use `Expr::node()` or parse via
/// `parse_str()` instead.
pub fn stamp_tags(exprs: &mut [Expr]) {
    use crate::ast::Atom;
    fn stamp(expr: &mut Expr) {
        match expr {
            Expr::List(list, _) => {
                if let Some(Expr::Atom(atom, _)) = list.elements.first_mut()
                    && let Atom::Name(symbol) = &*atom
                    && let Some(tag) = crate::tag::DeepTag::parse(symbol)
                {
                    *atom = Atom::Tag(tag);
                }
                for child in list.elements.iter_mut() {
                    stamp(child);
                }
            }
            Expr::Map(map, _) => {
                for (_, value) in map.entries.iter_mut() {
                    stamp(value);
                }
            }
            Expr::MetaExpr(meta, _) => {
                stamp(&mut meta.expr);
                for (_, value) in meta.entries.iter_mut() {
                    stamp(value);
                }
            }
            Expr::Atom(_, _) => {}
            Expr::Node(..) | Expr::BareList(..) | Expr::UnknownForm(..) => {}
        }
    }
    for expr in exprs.iter_mut() {
        stamp(expr);
    }
}

/// The raw parser mirrors the typed parser but constructs `RawExpr`/`RawAtom`
/// instead of `Expr`/`Atom`. No tag stamping, no typed-literal collapse.
pub fn parse_raw(tokens: &[Token]) -> Result<Vec<RawExpr>, ParseError> {
    let mut parser = RawParser::new(tokens);
    parser.parse_exprs()
}

/// Parse a source string into raw expressions (lex + parse_raw).
pub fn parse_raw_str(source: &str) -> Result<Vec<RawExpr>, ParseError> {
    let tokens = lexer::lex(source)?;
    parse_raw(&tokens)
}

/// Error from `parse_and_stamp` — wraps both parse errors and stamp errors.
#[derive(Debug)]
pub enum StampOrParseError {
    Parse(ParseError),
    Stamp(crate::stamp_to_typed::StampError),
}

impl std::fmt::Display for StampOrParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StampOrParseError::Parse(e) => write!(f, "parse error: {e}"),
            StampOrParseError::Stamp(e) => write!(f, "stamp error: {e}"),
        }
    }
}

impl std::error::Error for StampOrParseError {}

impl From<ParseError> for StampOrParseError {
    fn from(e: ParseError) -> Self {
        StampOrParseError::Parse(e)
    }
}

impl From<crate::stamp_to_typed::StampError> for StampOrParseError {
    fn from(e: crate::stamp_to_typed::StampError) -> Self {
        StampOrParseError::Stamp(e)
    }
}

/// Lex, parse to `RawExpr`, then stamp via `stamp_to_typed` to produce
/// typed `Expr` nodes. This is the preferred entry point for new code
/// that wants the role-directed AST.
pub fn parse_and_stamp(source: &str) -> Result<Vec<Expr>, StampOrParseError> {
    let tokens = lexer::lex(source).map_err(ParseError::from)?;
    let raw_exprs = parse_raw(&tokens)?;
    let typed = crate::stamp_to_typed::stamp_to_typed(raw_exprs)?;
    Ok(typed)
}

/// Lex, parse, then stamp via `stamp_deep_file` — the entry point for `.dp`
/// file ingestion. Accepts both a single top-level `(module ...)` and bare
/// declarations.
pub fn parse_and_stamp_file(source: &str) -> Result<Vec<Expr>, StampOrParseError> {
    let tokens = lexer::lex(source).map_err(ParseError::from)?;
    let raw_exprs = parse_raw(&tokens)?;
    let typed = crate::stamp_to_typed::stamp_deep_file(raw_exprs)?;
    Ok(typed)
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
    fn parse_suffixed_literal_inside_explicit_lit_without_nesting() {
        let exprs = p("(lit {} 7f32)");
        match &exprs[0] {
            Expr::Node(node, _) => {
                assert_eq!(node.tag(), crate::tag::DeepTag::Lit);
                assert_eq!(node.child_count(), 1);
                assert!(matches!(
                    node.children_slice()[0],
                    Expr::Atom(Atom::Int(7), _)
                ));
                assert!(node.meta().entries.iter().any(|(key, _)| key == "type"));
                assert!(node.meta().entries.iter().any(|(key, value)| {
                    key == "literal_source"
                        && matches!(value, Expr::Atom(Atom::Name(name), _) if name == "integer")
                }));
            }
            other => panic!("expected lit Node, got {other:?}"),
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
            Expr::Atom(Atom::Name(s), _) => assert_eq!(s, "foo"),
            other => panic!("expected Symbol, got {:?}", other),
        }
    }

    #[test]
    fn parse_bare_keyword_is_error() {
        let err = parse_str(":axis")
            .expect_err("bare :keyword outside metadata map must be a parse error");
        assert_eq!(
            err.to_string(),
            "expected expression (bare :keyword is valid only as a metadata map key), found Keyword(\"axis\") at byte 0"
        );
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
        // `add` is not in the closed vocabulary, so this becomes a BareList.
        let exprs = p("(add 1 2)");
        assert_eq!(exprs.len(), 1);
        match &exprs[0] {
            Expr::BareList(elems, _) => {
                assert_eq!(elems.len(), 3);
                match &elems[0] {
                    Expr::Atom(Atom::Name(s), _) => assert_eq!(s, "add"),
                    other => panic!("expected Symbol(add), got {:?}", other),
                }
                match &elems[1] {
                    Expr::Atom(Atom::Int(1), _) => {}
                    other => panic!("expected Int(1), got {:?}", other),
                }
                match &elems[2] {
                    Expr::Atom(Atom::Int(2), _) => {}
                    other => panic!("expected Int(2), got {:?}", other),
                }
            }
            other => panic!("expected BareList, got {:?}", other),
        }
    }

    #[test]
    fn parse_list_no_children() {
        // `nop` is not in the closed vocabulary, so a single-element list becomes BareList.
        let exprs = p("(nop)");
        match &exprs[0] {
            Expr::BareList(elems, _) => {
                assert_eq!(elems.len(), 1);
                match &elems[0] {
                    Expr::Atom(Atom::Name(s), _) => assert_eq!(s, "nop"),
                    other => panic!("expected Symbol(nop), got {:?}", other),
                }
            }
            other => panic!("expected BareList, got {:?}", other),
        }
    }

    #[test]
    fn parse_nested_lists() {
        // Post-sprint canonical form: (def {} f (fn {} (params {} x) (var {} x)))
        let exprs = p("(def {} f (fn {} (params {} x) (var {} x)))");
        assert_eq!(exprs.len(), 1);
        match &exprs[0] {
            Expr::Node(node, _) => {
                assert_eq!(node.tag(), crate::tag::DeepTag::Def);
                assert!(node.meta().entries.is_empty());
                // Def has 2 children: the name binder and the (fn ...) node
                assert_eq!(node.child_count(), 2);
                match &node.children_slice()[0] {
                    Expr::Atom(Atom::Name(s), _) => assert_eq!(s, "f"),
                    other => panic!("expected Symbol(f), got {:?}", other),
                }
                match &node.children_slice()[1] {
                    Expr::Node(fn_node, _) => {
                        assert_eq!(fn_node.tag(), crate::tag::DeepTag::Fn);
                    }
                    other => panic!("expected fn Node, got {:?}", other),
                }
            }
            other => panic!("expected Node, got {:?}", other),
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
                    Expr::Atom(Atom::Name(s), _) => assert_eq!(s, "f32"),
                    other => panic!("expected Symbol(f32), got {:?}", other),
                }
                match meta.expr.as_ref() {
                    Expr::Atom(Atom::Name(s), _) => assert_eq!(s, "x"),
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
                    Expr::BareList(elems, _) => match &elems[0] {
                        Expr::Atom(Atom::Name(s), _) => assert_eq!(s, "add"),
                        other => panic!("expected Symbol(add), got {:?}", other),
                    },
                    other => panic!("expected BareList, got {:?}", other),
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
            Expr::BareList(elems, _) => match &elems[0] {
                Expr::Atom(Atom::Name(s), _) => assert_eq!(s, "add"),
                other => panic!("expected Symbol(add), got {:?}", other),
            },
            other => panic!("expected BareList, got {:?}", other),
        }
        match &exprs[2] {
            Expr::Atom(Atom::Name(s), _) => assert_eq!(s, "foo"),
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
            Expr::BareList(elems, _) => assert!(elems.is_empty()),
            other => panic!("expected empty BareList, got: {:?}", other),
        }
    }

    #[test]
    fn parse_list_non_symbol_first_element() {
        // Any expr can be the first element of a list.
        let exprs = p("(42 a b)");
        assert_eq!(exprs.len(), 1);
        match &exprs[0] {
            Expr::BareList(elems, _) => {
                assert_eq!(elems.len(), 3);
                match &elems[0] {
                    Expr::Atom(Atom::Int(42), _) => {}
                    other => panic!("expected Int(42), got {:?}", other),
                }
            }
            other => panic!("expected BareList, got {:?}", other),
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
