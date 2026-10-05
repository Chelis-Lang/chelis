use crate::ast::*;
use crate::lexer::{self, LexError};
use crate::token::{Token, TokenKind};
use chelis_deep::{BoundDtype, DtypeBound, DtypeFamily, Span};
use chelis_unord::UnordSet;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("lex error: {0}")]
    Lex(#[from] LexError),
    #[error("unexpected end of input")]
    UnexpectedEof,
    #[error("expected {expected}, found {found} at byte {offset}")]
    Expected {
        expected: String,
        found: String,
        offset: usize,
    },
    #[error(
        "`{keyword}` is a reserved word that introduces {role}; to bind a value, \
         rename it (for example `{suggestion}`) at byte {offset}"
    )]
    ReservedWordBinding {
        keyword: &'static str,
        role: &'static str,
        suggestion: &'static str,
        offset: usize,
    },
    #[error("non-associative operator chained at byte {offset}")]
    NonAssocChain { offset: usize },
    #[error(
        "expression statement must be bound: a block is bindings followed by one tail expression; \
         bind the value with `_ = <expr>` or move it to tail position (byte {offset})"
    )]
    BareStatementInBlock { offset: usize },
    /// A `;` used where a canonical binding block wants a newline.
    ///
    /// The generic `Expected` shape used to render this as
    /// `expected separator (`;` or newline), found Semicolon`, which named the
    /// token it had just refused as one of the two acceptable spellings
    /// (chelis#1267). `par`, `do`, and the v0.18 compatibility grammar do take
    /// `;`, so the wording names canonical Surf v0.19 rather than claiming `;`
    /// is never a block separator.
    #[error(
        "expected a newline separator, found `;`: `;` is not a block separator in canonical \
         Surf v0.19; replace it with a newline, or run `chelis migrate surf --from 0.18` to \
         rewrite v0.18 source (byte {offset})"
    )]
    SemicolonBlockSeparator { offset: usize },
    #[error("literal `{found}` is not an accepted spelling at byte {offset}; write `{expected}`")]
    NonCanonicalLiteral {
        found: String,
        expected: String,
        offset: usize,
    },
    #[error("non-finite numeric literal `{found}` is not representable in Surf at byte {offset}")]
    NonFiniteLiteral { found: String, offset: usize },
    #[error(
        "integer magnitude `{found}` is only valid after unary `-`; write `-{found}` at byte {offset}"
    )]
    SignedMinimumMagnitudeRequiresNegation { found: String, offset: usize },
    /// A retired counter-stream spelling: the `with seed(...)` handler or the
    /// `Random` effect name. Randomness has no handler and no effect; a random
    /// primitive takes an explicit key (spec/02 §P5a, spec/05 §2.7).
    #[error(
        "`{spelling}` is not Surf: randomness has no handler or effect; a random \
         primitive takes an explicit key, for example \
         `dropout(key_from_seed(42i64), x, 0.5)` (spec/02 §P5a) at byte {offset}"
    )]
    RetiredRandomness {
        spelling: &'static str,
        offset: usize,
    },
}

struct Parser {
    // Lookahead owns a cursor, but shares the immutable stream. Cloning every
    // token for each BlockBinding made generated scalar modules quadratic.
    tokens: std::sync::Arc<[Token]>,
    closing_delimiters: std::sync::Arc<[usize]>,
    pos: usize,
    module_allowed: bool,
    mode: ParseMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParseMode {
    Canonical,
    LegacyV018,
    #[cfg(feature = "pre-020-pipe-migration")]
    PipeMigration,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub fn parse(tokens: &[Token]) -> Result<Vec<Decl>, ParseError> {
    parse_with_mode(tokens, ParseMode::Canonical)
}

/// Parse the v0.18 compatibility grammar for the explicit migration path.
/// Normal compiler, formatter, and CLI entry points must use [`parse`].
pub fn parse_legacy_v018(tokens: &[Token]) -> Result<Vec<Decl>, ParseError> {
    parse_with_mode(tokens, ParseMode::LegacyV018)
}

fn closing_delimiters(tokens: &[Token]) -> std::sync::Arc<[usize]> {
    let mut matching = vec![usize::MAX; tokens.len()];
    let mut open = Vec::new();
    for (index, token) in tokens.iter().enumerate() {
        match token.kind {
            TokenKind::LParen | TokenKind::LBracket | TokenKind::LBrace => open.push(index),
            TokenKind::RParen | TokenKind::RBracket | TokenKind::RBrace => {
                if let Some(start) = open.pop() {
                    matching[start] = index;
                }
            }
            _ => {}
        }
    }
    matching.into()
}

fn parse_with_mode(tokens: &[Token], mode: ParseMode) -> Result<Vec<Decl>, ParseError> {
    let mut p = Parser {
        tokens: tokens.into(),
        closing_delimiters: closing_delimiters(tokens),
        pos: 0,
        module_allowed: true,
        mode,
    };
    let decls = p.parse_program()?;
    validate_property_names(&decls)?;
    validate_bound_ownership(&decls)?;
    Ok(decls)
}

/// A standalone signature owns the declaration's complete binder list
/// (spec/02 §P4b, spec/03 §2.2). Reject a second list on the matching `def`
/// before desugaring could create two authorities for one declaration.
pub(crate) fn validate_bound_ownership(decls: &[Decl]) -> Result<(), ParseError> {
    let signatures: UnordSet<_> = decls
        .iter()
        .filter_map(|d| match d {
            Decl::Sig { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    for decl in decls {
        if let Decl::FunDef {
            type_binders, span, ..
        }
        | Decl::Sig {
            type_binders, span, ..
        }
        | Decl::Property {
            type_binders, span, ..
        } = decl
        {
            let mut declared = UnordSet::new();
            for binder in type_binders {
                if crate::dtype_name::is_forbidden_binder_name(&binder.name) {
                    return Err(ParseError::Expected {
                        expected:
                            "a declaration binder outside the primitive and reserved dtype vocabulary"
                                .into(),
                        found: format!(
                            "dtype spelling `{}` cannot be a declaration binder",
                            binder.name
                        ),
                        offset: span.offset,
                    });
                }
                if !declared.insert(binder.name.as_str()) {
                    return Err(ParseError::Expected {
                        expected: "one declaration per binder".into(),
                        found: format!("duplicate binder `{}`", binder.name),
                        offset: span.offset,
                    });
                }
            }
        }
        match decl {
            Decl::FunDef {
                name,
                type_binders,
                span,
                ..
            } if signatures.contains(name.as_str()) && !type_binders.is_empty() => {
                return Err(ParseError::Expected {
                    expected:
                        "binders on the signature: a declaration's `defsig` owns its binder list"
                            .into(),
                    found: format!("second binder list on def `{name}`"),
                    offset: span.offset,
                });
            }
            Decl::Module { decls, .. } => validate_bound_ownership(decls)?,
            _ => {}
        }
    }
    Ok(())
}

pub fn parse_str(source: &str) -> Result<Vec<Decl>, ParseError> {
    let tokens = lexer::lex(source)?;
    parse_canonical_source_tokens(source, &tokens)
}

/// Parse a token stream while validating source-sensitive literal semantics.
/// Harmless lexical aliases are accepted here and normalized only by the
/// printer; meaning-changing suffix/adoption forms and non-finite values stay
/// rejected.
pub fn parse_canonical_source_tokens(
    source: &str,
    tokens: &[Token],
) -> Result<Vec<Decl>, ParseError> {
    validate_literal_spellings(source, tokens)?;
    parse(tokens)
}

/// Parse the previous canonical pipe grammar solely for explicit migration.
/// All normal parser entry points enforce the new grouping contract.
#[cfg(feature = "pre-020-pipe-migration")]
pub(crate) fn parse_pipe_migration(source: &str) -> Result<Vec<Decl>, ParseError> {
    let tokens = lexer::lex(source)?;
    validate_literal_spellings(source, &tokens)?;
    parse_with_mode(&tokens, ParseMode::PipeMigration)
}

pub fn parse_str_legacy_v018(source: &str) -> Result<Vec<Decl>, ParseError> {
    let tokens = lexer::lex(source)?;
    parse_legacy_v018_source_tokens(source, &tokens)
}

pub fn parse_legacy_v018_source_tokens(
    source: &str,
    tokens: &[Token],
) -> Result<Vec<Decl>, ParseError> {
    validate_finite_literals(source, tokens)?;
    parse_legacy_v018(tokens)
}

fn validate_literal_spellings(source: &str, tokens: &[Token]) -> Result<(), ParseError> {
    validate_finite_literals(source, tokens)?;
    for token in tokens {
        let found = source
            .get(token.span.offset..token.span.end())
            .unwrap_or_default();
        let expected = match &token.kind {
            TokenKind::Int(value) => Some(value.to_string()),
            TokenKind::Float(value) => Some(crate::format::canonical_float(*value)),
            // An integer body under a float suffix keeps its integer spelling:
            // spec/02-surf-syntax.md §P10a binds it as [04-LIT-1]'s exact
            // `literal_source: integer` form, which is not the decimal-bodied
            // literal in another spelling (chelis#2119).
            TokenKind::TypedInt(value, suffix) => Some(format!("{value}{}", suffix.as_str())),
            TokenKind::TypedFloat(value, suffix) => Some(format!(
                "{}{}",
                crate::format::canonical_float(*value),
                suffix.as_str()
            )),
            TokenKind::IntMinMagnitude(suffix) => Some(format!(
                "9223372036854775808{}",
                suffix.as_ref().map_or("", |suffix| (*suffix).as_str())
            )),
            TokenKind::Str(value) => Some(crate::format::canonical_string(value)),
            _ => None,
        };
        let Some(expected) = expected else {
            continue;
        };
        if !accepted_literal_alias(found, &expected, &token.kind) {
            return Err(ParseError::NonCanonicalLiteral {
                found: found.to_string(),
                expected,
                offset: token.span.offset,
            });
        }
    }
    Ok(())
}

fn validate_finite_literals(source: &str, tokens: &[Token]) -> Result<(), ParseError> {
    for token in tokens {
        let non_finite = match &token.kind {
            TokenKind::Float(value) | TokenKind::TypedFloat(value, _) => !value.is_finite(),
            // An integer body under a float suffix binds the exact integer at
            // the suffix width (§P10a), so its representability is decided by
            // that width, not by the i64 body. `65520f16` rounds to infinity
            // and has no Surf literal, exactly as `1e400` has none
            // (chelis#2119).
            TokenKind::TypedInt(value, suffix) if suffix.is_float() => {
                !crate::resugar::round_integer_at_float_width(*value, *suffix).is_finite()
            }
            _ => false,
        };
        if non_finite {
            let found = source
                .get(token.span.offset..token.span.end())
                .unwrap_or_default();
            return Err(ParseError::NonFiniteLiteral {
                found: found.to_string(),
                offset: token.span.offset,
            });
        }
    }
    Ok(())
}

fn accepted_literal_alias(found: &str, expected: &str, kind: &TokenKind) -> bool {
    match kind {
        TokenKind::Str(_) => true,
        // An integer-bodied literal must already be the canonical decimal
        // spelling, or a value-preserving radix form: a redundant leading zero
        // reads as C octal to a human and is refused rather than normalized.
        // A float suffix does not relax this, so `007f64` stays an error.
        TokenKind::Int(_) | TokenKind::TypedInt(_, _) | TokenKind::IntMinMagnitude(_) => {
            let lower = found.to_ascii_lowercase();
            let radix = lower.starts_with("0x") || lower.starts_with("0b");
            if radix && matches!(kind, TokenKind::TypedInt(_, suffix) if suffix.is_float()) {
                // spec/04-type-system.md §5.5: an integer radix form carries no
                // float suffix. Hex is already a lex error under maximal munch;
                // this refuses the binary spelling for the same reason.
                return false;
            }
            radix || found.replace('_', "") == expected
        }
        // Every finite decimal float body decodes to the value its canonical
        // spelling round-trips to, so redundant precision, padded zeroes, and
        // exponent forms are all value-preserving input aliases the printer
        // normalizes rather than parse errors (spec/02-surf-syntax.md §P10,
        // chelis#2119). Malformed separators are already a lex error and a
        // non-finite decode is rejected by `validate_finite_literals`.
        TokenKind::Float(_) | TokenKind::TypedFloat(_, _) => true,
        _ => true,
    }
}

fn validate_property_names(decls: &[Decl]) -> Result<(), ParseError> {
    let mut value_names = UnordSet::new();
    let mut property_names = UnordSet::new();
    for decl in decls {
        match decl {
            Decl::Property { name, span, .. }
                if value_names.contains(name) || !property_names.insert(name.clone()) =>
            {
                return Err(ParseError::Expected {
                    expected: "unique property name within module value namespace".into(),
                    found: name.clone(),
                    offset: span.offset,
                });
            }
            Decl::Property { .. } => {}
            Decl::FunDef { name, span, .. }
            | Decl::LetDef { name, span, .. }
            | Decl::Sig { name, span, .. } => {
                if property_names.contains(name) {
                    return Err(ParseError::Expected {
                        expected: "unique property name within module value namespace".into(),
                        found: name.clone(),
                        offset: span.offset,
                    });
                }
                value_names.insert(name.clone());
            }
            Decl::Module { decls, .. } => validate_property_names(decls)?,
            _ => {}
        }
    }
    Ok(())
}

/// The unary-builtin keyword tokens that have a spec-meaningful bare
/// form per `spec/01-nomenclature.md` §3.6. Used by
/// `synthesize_bare_unary_builtin_lambda` to η-expand `realize`/`copy`
/// in non-call position (H1/H2 of the pipe-autofix-extras workstream;
/// see `docs/archive/investigations/pipe_autofix_and_bare_keyword_extras_diagnosis.md`).
#[derive(Copy, Clone)]
enum BareUnaryBuiltinKind {
    Realize,
    Copy,
}

#[derive(Copy, Clone)]
enum DeepSurfaceMetaForm {
    Quote,
    Unquote,
    Splice,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

impl Parser {
    fn raw_peek(&self) -> &TokenKind {
        self.tokens
            .get(self.pos)
            .map(|t| &t.kind)
            .unwrap_or(&TokenKind::Eof)
    }

    fn peek(&self) -> &TokenKind {
        let mut pos = self.pos;
        while matches!(
            self.tokens.get(pos).map(|t| &t.kind),
            Some(TokenKind::Newline)
        ) {
            pos += 1;
        }
        self.tokens
            .get(pos)
            .map(|t| &t.kind)
            .unwrap_or(&TokenKind::Eof)
    }

    fn at_eof(&self) -> bool {
        matches!(self.peek(), TokenKind::Eof)
    }

    /// Return the `n`th significant token after the current significant
    /// token. Newlines are layout and do not participate in look-ahead.
    fn peek_significant_after(&self, n: usize) -> Option<&TokenKind> {
        self.tokens
            .iter()
            .skip(self.pos)
            .filter(|token| !matches!(token.kind, TokenKind::Newline))
            .nth(n)
            .map(|token| &token.kind)
    }

    /// The record-update operator and delimited device handler share `with`.
    /// Use the actual update introducer in both parsing and the pipe guard.
    fn starts_record_update(&self, start: usize) -> bool {
        let mut kinds = self
            .tokens
            .iter()
            .skip(start)
            .filter(|token| !matches!(token.kind, TokenKind::Newline))
            .map(|token| &token.kind);
        matches!(kinds.next(), Some(TokenKind::With))
            && matches!(kinds.next(), Some(TokenKind::LBrace))
    }

    fn find_top_level_token(&self, wanted: TokenKind) -> Option<usize> {
        let mut paren = 0usize;
        let mut bracket = 0usize;
        let mut brace = 0usize;
        for (index, token) in self.tokens.iter().enumerate().skip(self.pos) {
            if paren == 0 && bracket == 0 && brace == 0 && token.kind == wanted {
                return Some(index);
            }
            match token.kind {
                TokenKind::LParen => paren += 1,
                TokenKind::RParen if paren > 0 => paren -= 1,
                TokenKind::LBracket => bracket += 1,
                TokenKind::RBracket if bracket > 0 => bracket -= 1,
                TokenKind::LBrace => brace += 1,
                TokenKind::RBrace if brace > 0 => brace -= 1,
                TokenKind::Eof => break,
                _ => {}
            }
        }
        None
    }

    /// True when the next two significant tokens are `.` followed by a
    /// PascalCase `TypeIdent` — the shape of a module-qualified path segment
    /// (`.Dropout`, `.Train`). Used to extend a constructor pattern head into
    /// a qualified path (`Demo.Dropout.Train`, chelis#316) without consuming
    /// the `.` when it is not part of such a path.
    fn peek_dot_then_typeident(&self) -> bool {
        let mut pos = self.pos;
        while matches!(
            self.tokens.get(pos).map(|t| &t.kind),
            Some(TokenKind::Newline)
        ) {
            pos += 1;
        }
        if !matches!(self.tokens.get(pos).map(|t| &t.kind), Some(TokenKind::Dot)) {
            return false;
        }
        pos += 1;
        while matches!(
            self.tokens.get(pos).map(|t| &t.kind),
            Some(TokenKind::Newline)
        ) {
            pos += 1;
        }
        matches!(
            self.tokens.get(pos).map(|t| &t.kind),
            Some(TokenKind::TypeIdent(_))
        )
    }

    fn current_offset(&self) -> usize {
        let mut pos = self.pos;
        while matches!(
            self.tokens.get(pos).map(|t| &t.kind),
            Some(TokenKind::Newline)
        ) {
            pos += 1;
        }
        self.tokens
            .get(pos)
            .map(|t| t.span.offset)
            .unwrap_or(self.tokens.last().map(|t| t.span.end()).unwrap_or(0))
    }

    fn advance_raw(&mut self) -> Token {
        let tok = self.tokens[self.pos].clone();
        self.pos += 1;
        tok
    }

    fn advance(&mut self) -> Token {
        while matches!(self.raw_peek(), TokenKind::Newline) {
            self.pos += 1;
        }
        self.advance_raw()
    }

    fn consume_block_separators(&mut self) -> usize {
        let mut consumed = 0;
        while matches!(self.raw_peek(), TokenKind::Newline)
            || (self.mode == ParseMode::LegacyV018
                && matches!(self.raw_peek(), TokenKind::Semicolon))
        {
            self.advance_raw();
            consumed += 1;
        }
        consumed
    }

    /// Reject a `;` left standing at a canonical binding-block separator
    /// boundary (chelis#1267).
    ///
    /// `consume_block_separators` takes `;` only in `LegacyV018`, so in
    /// canonical Surf v0.19 one survives wherever that function is called
    /// inside a binding block. There are four such positions, three in
    /// `parse_block_inner` (before the first binding, between bindings, and
    /// after the tail) and one in `parse_block_let_binding` (between a
    /// binding's `=` and its value, the legal v0.18 spelling `a = ; 1i64`).
    /// Each used to surface a different wrong diagnostic, none of which named
    /// the rule. Call this immediately after every `consume_block_separators`
    /// so they all report the same thing; adding a fifth call site without a
    /// matching guard reopens this defect.
    ///
    /// The two `consume_block_separators` calls in `parse_property_decl` are
    /// deliberately unguarded: they sit in a top-level declaration arm whose
    /// own diagnostic already carries a real offset.
    ///
    /// `par` and `do` genuinely require `;` (spec/02-surf-syntax.md §P5) and
    /// parse through their own functions, so they never reach this.
    fn reject_canonical_semicolon_separator(&self) -> Result<(), ParseError> {
        if self.mode != ParseMode::LegacyV018 && matches!(self.raw_peek(), TokenKind::Semicolon) {
            return Err(ParseError::SemicolonBlockSeparator {
                offset: self.current_offset(),
            });
        }
        Ok(())
    }

    fn expect(&mut self, kind: &TokenKind) -> Result<Token, ParseError> {
        if self.peek() == kind {
            Ok(self.advance())
        } else {
            Err(ParseError::Expected {
                expected: format!("{kind:?}"),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            })
        }
    }

    /// Exact newline continuations for a block sequencing expression.
    ///
    /// chelis#849. A top-level newline inside a sequencing context ends the
    /// expression being parsed unless the next significant token is one of these:
    ///
    /// * `|>` -- an infix pipeline stage has no meaning at the head of a statement.
    /// * `then` / `else` -- an `if` is not a legal expression without both, so
    ///   neither can begin one.
    ///
    /// Each member is safe because it cannot head an expression, but that is a
    /// necessary condition rather than the membership rule: other infix tokens
    /// (`+`, `*`, `==`, `&&`, ...) remain outside this deliberately closed set.
    /// A token that CAN head an expression (`with`, `match`, an identifier) must
    /// not be added: after a newline it is genuinely ambiguous between a
    /// continuation and a new statement, and admitting it would re-open the
    /// juxtaposition defect chelis#706 closed.
    fn is_block_expression_continuation(kind: &TokenKind) -> bool {
        matches!(kind, TokenKind::Pipe | TokenKind::Then | TokenKind::Else)
    }

    fn block_expr_end(&self) -> usize {
        let mut pos = self.pos;
        let mut paren_depth = 0usize;
        let mut bracket_depth = 0usize;
        let mut brace_depth = 0usize;
        while let Some(token) = self.tokens.get(pos) {
            match token.kind {
                TokenKind::Newline | TokenKind::Semicolon
                    if paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 =>
                {
                    // A top-level newline does not end the expression when
                    // the next significant token is in P12's exact closed set
                    // (chelis#849): `|>` is an infix continuation, and `then` /
                    // `else` are the mandatory continuations of an `if`. None
                    // can head a statement, so each selected continuation is
                    // unambiguous; that safety property does not admit every
                    // other infix token.
                    //
                    // This helper is the only one of the three boundary
                    // rules that is CLOSED. `decl_expr_end` and
                    // `property_expr_end` are permissive -- they end only at
                    // a declaration start (and, for a property predicate, at
                    // `with`), so a leading `then` or `else` already
                    // continued there and neither ever showed this defect.
                    // `block_expr_end` broke instead, which is the 0.17
                    // regression. spec/02 P12 states all three.
                    //
                    // The set is deliberately closed to tokens that can only
                    // continue. `with {` is NOT admitted: it heads an
                    // expression, so a leading `with` after a newline is
                    // genuinely ambiguous and is left to its own decision.
                    if matches!(token.kind, TokenKind::Newline)
                        && self
                            .tokens
                            .iter()
                            .skip(pos + 1)
                            .find(|next| !matches!(next.kind, TokenKind::Newline))
                            .is_some_and(|next| Self::is_block_expression_continuation(&next.kind))
                    {
                        pos += 1;
                        continue;
                    }
                    break;
                }
                TokenKind::RBrace if paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 => {
                    break;
                }
                TokenKind::LParen => paren_depth += 1,
                TokenKind::RParen if paren_depth > 0 => paren_depth -= 1,
                TokenKind::LBracket => bracket_depth += 1,
                TokenKind::RBracket if bracket_depth > 0 => bracket_depth -= 1,
                TokenKind::LBrace => brace_depth += 1,
                TokenKind::RBrace if brace_depth > 0 => brace_depth -= 1,
                TokenKind::Eof => break,
                _ => {}
            }
            pos += 1;
        }
        pos
    }

    fn is_decl_start_at(&self, pos: usize) -> bool {
        match self.tokens.get(pos).map(|t| &t.kind) {
            Some(
                TokenKind::Def
                | TokenKind::Sig
                | TokenKind::Ident(_)
                | TokenKind::Type
                | TokenKind::Dim
                | TokenKind::Macro
                | TokenKind::Module
                | TokenKind::Import
                | TokenKind::Export
                | TokenKind::At,
            ) => true,
            // A single-letter uppercase head starts a value binding
            // (chelis#437: `S = ...`); it must end the prior declaration's
            // expression so the binding is parsed as its own decl. Keep
            // this in lockstep with the `parse_decl` dispatch arm.
            Some(TokenKind::TypeIdent(name)) => is_single_letter_upper(name),
            _ => false,
        }
    }

    fn decl_expr_end(&self) -> usize {
        let mut pos = self.pos;
        let mut paren_depth = 0usize;
        let mut bracket_depth = 0usize;
        let mut brace_depth = 0usize;
        let mut started = false;
        while let Some(token) = self.tokens.get(pos) {
            match token.kind {
                TokenKind::Newline | TokenKind::Semicolon
                    if paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 =>
                {
                    if !started {
                        pos += 1;
                        continue;
                    }
                    let next_sig = self
                        .tokens
                        .iter()
                        .enumerate()
                        .skip(pos + 1)
                        .find(|(_, next)| !matches!(next.kind, TokenKind::Newline))
                        .map(|(idx, _)| idx);
                    if matches!(
                        next_sig.and_then(|idx| self.tokens.get(idx).map(|t| &t.kind)),
                        Some(TokenKind::Pipe)
                    ) {
                        pos += 1;
                        continue;
                    }
                    if next_sig.is_none_or(|idx| {
                        self.is_decl_start_at(idx)
                            || matches!(self.tokens[idx].kind, TokenKind::Eof)
                    }) {
                        break;
                    }
                }
                TokenKind::Eof if started => break,
                TokenKind::LParen => {
                    started = true;
                    paren_depth += 1;
                }
                TokenKind::RParen if paren_depth > 0 => paren_depth -= 1,
                TokenKind::LBracket => {
                    started = true;
                    bracket_depth += 1;
                }
                TokenKind::RBracket if bracket_depth > 0 => bracket_depth -= 1,
                TokenKind::LBrace => {
                    started = true;
                    brace_depth += 1;
                }
                TokenKind::RBrace if brace_depth > 0 => brace_depth -= 1,
                TokenKind::Eof => break,
                _ => {
                    started = true;
                }
            }
            pos += 1;
        }
        pos
    }

    fn parse_expr_until_block_separator(&mut self) -> Result<Expr, ParseError> {
        let end = self.block_expr_end();
        self.parse_expr_in_range(end, "end of block expression")
    }

    fn parse_expr_until_decl_separator(&mut self) -> Result<Expr, ParseError> {
        let end = self.decl_expr_end();
        self.parse_expr_in_range(end, "end of declaration expression")
    }

    fn property_expr_end(&self) -> usize {
        let mut pos = self.pos;
        let mut paren_depth = 0usize;
        let mut bracket_depth = 0usize;
        let mut brace_depth = 0usize;
        let mut started = false;
        while let Some(token) = self.tokens.get(pos) {
            match token.kind {
                TokenKind::Newline | TokenKind::Semicolon
                    if paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 =>
                {
                    if !started {
                        pos += 1;
                        continue;
                    }
                    let next_sig = self
                        .tokens
                        .iter()
                        .enumerate()
                        .skip(pos + 1)
                        .find(|(_, next)| !matches!(next.kind, TokenKind::Newline))
                        .map(|(idx, _)| idx);
                    if matches!(
                        next_sig.and_then(|idx| self.tokens.get(idx).map(|t| &t.kind)),
                        Some(TokenKind::Pipe)
                    ) {
                        pos += 1;
                        continue;
                    }
                    if next_sig.is_none_or(|idx| {
                        matches!(self.tokens[idx].kind, TokenKind::With | TokenKind::Eof)
                            || self.is_decl_start_at(idx)
                    }) {
                        break;
                    }
                }
                TokenKind::Eof if started => break,
                TokenKind::LParen => {
                    started = true;
                    paren_depth += 1;
                }
                TokenKind::RParen if paren_depth > 0 => paren_depth -= 1,
                TokenKind::LBracket => {
                    started = true;
                    bracket_depth += 1;
                }
                TokenKind::RBracket if bracket_depth > 0 => bracket_depth -= 1,
                TokenKind::LBrace => {
                    started = true;
                    brace_depth += 1;
                }
                TokenKind::RBrace if brace_depth > 0 => brace_depth -= 1,
                TokenKind::Eof => break,
                _ => {
                    started = true;
                }
            }
            pos += 1;
        }
        pos
    }

    fn parse_expr_until_property_option(&mut self) -> Result<Expr, ParseError> {
        let end = self.property_expr_end();
        self.parse_expr_in_range(end, "property predicate or option boundary")
    }

    fn parse_expr_in_range(&mut self, end: usize, expected: &str) -> Result<Expr, ParseError> {
        let mut expr_tokens = self.tokens[self.pos..end].to_vec();
        expr_tokens.push(Token {
            kind: TokenKind::Eof,
            span: Span::new(self.current_offset(), 0),
        });
        let mut nested = Parser {
            closing_delimiters: closing_delimiters(&expr_tokens),
            tokens: expr_tokens.into(),
            pos: 0,
            module_allowed: false,
            mode: self.mode,
        };
        let expr = nested.parse_expr(0)?;
        if !nested.at_eof() {
            return Err(ParseError::Expected {
                expected: expected.into(),
                found: format!("{:?}", nested.peek()),
                offset: nested.current_offset(),
            });
        }
        self.pos = end;
        Ok(expr)
    }

    /// Diagnose a reserved declaration keyword used as a value-binding name.
    ///
    /// `sig = 0.2f64` is a natural thing to write (`sig` for volatility), but
    /// `sig` heads a signature declaration, so `parse_decl` dispatches into
    /// `parse_sig_decl` and the failure surfaces one token later, at the `=`,
    /// as `expected identifier, found Eq` (chelis#915). By then the reserved
    /// word has already been consumed, so the *found* token is never the
    /// keyword — a special case keyed on the found token would not fire on
    /// this form at all.
    ///
    /// Call this immediately after the keyword is consumed. When the next
    /// token is `=` the user's intent is unambiguously a value binding, so
    /// name the reserved word and point at the keyword rather than the `=`.
    /// Any other following token is a genuine malformed declaration and falls
    /// through to the ordinary `expected …, found …` path.
    fn reserved_word_binding_error(
        &self,
        keyword: &'static str,
        role: &'static str,
        suggestion: &'static str,
        keyword_span: Span,
    ) -> Option<ParseError> {
        (*self.peek() == TokenKind::Eq).then_some(ParseError::ReservedWordBinding {
            keyword,
            role,
            suggestion,
            offset: keyword_span.offset,
        })
    }

    fn expect_ident(&mut self) -> Result<(String, Span), ParseError> {
        match self.peek().clone() {
            TokenKind::Ident(name) => {
                let tok = self.advance();
                Ok((name, tok.span))
            }
            _ => Err(ParseError::Expected {
                expected: "identifier".into(),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            }),
        }
    }

    /// Accept a value identifier: a lowercase-leading `Ident`, or a
    /// single-letter uppercase `TypeIdent` used in a position that
    /// unambiguously binds a value (a value-binding LHS, a parameter,
    /// or a block binding pattern). The lexer's §1.1 case-split classifies a bare
    /// uppercase identifier as `TypeIdent`; the explicit value-binding
    /// context overrides that default for single-letter names, the same
    /// way a def's `[..]` quantifier clause overrides the case-split for
    /// type variables (spec/02 §P4a). This admits finance/math notation
    /// (`S`, `K`, `T`, `N`, `P`) as value names (chelis#437) without
    /// weakening the PascalCase convention for multi-letter type and
    /// constructor names.
    fn expect_value_ident(&mut self) -> Result<(String, Span), ParseError> {
        match self.peek().clone() {
            TokenKind::Ident(name) => {
                let tok = self.advance();
                Ok((name, tok.span))
            }
            TokenKind::TypeIdent(name) if is_single_letter_upper(&name) => {
                let tok = self.advance();
                Ok((name, tok.span))
            }
            _ => Err(ParseError::Expected {
                expected: "identifier".into(),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            }),
        }
    }

    fn expect_type_ident(&mut self) -> Result<(String, Span), ParseError> {
        match self.peek().clone() {
            TokenKind::TypeIdent(name) => {
                let tok = self.advance();
                Ok((name, tok.span))
            }
            _ => Err(ParseError::Expected {
                expected: "type identifier".into(),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            }),
        }
    }

    fn expect_ident_or_type_ident(&mut self) -> Result<(String, Span), ParseError> {
        match self.peek().clone() {
            TokenKind::Ident(name) | TokenKind::TypeIdent(name) => {
                let tok = self.advance();
                Ok((name, tok.span))
            }
            _ => Err(ParseError::Expected {
                expected: "identifier or type identifier".into(),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            }),
        }
    }

    fn parse_module_path(&mut self) -> Result<(String, Span), ParseError> {
        let (first_seg, mut span) = self.expect_type_ident()?;
        let mut module = first_seg;
        while *self.peek() == TokenKind::Dot {
            self.advance();
            let (seg, seg_span) = self.expect_type_ident()?;
            module.push('.');
            module.push_str(&seg);
            span = span.merge(seg_span);
        }
        Ok((module, span))
    }

    fn parse_ident_list(&mut self, terminator: TokenKind) -> Result<Vec<String>, ParseError> {
        let mut names = Vec::new();
        if *self.peek() == terminator {
            return Ok(names);
        }
        let (name, _) = self.expect_ident_or_type_ident()?;
        names.push(name);
        while *self.peek() == TokenKind::Comma {
            self.advance();
            if self.comma_terminates_list(&terminator, names.len(), false)? {
                break;
            }
            let (name, _) = self.expect_ident_or_type_ident()?;
            names.push(name);
        }
        Ok(names)
    }

    fn comma_terminates_list(
        &self,
        terminator: &TokenKind,
        _item_count: usize,
        _singleton_tuple: bool,
    ) -> Result<bool, ParseError> {
        if self.peek() != terminator {
            return Ok(false);
        }
        Ok(true)
    }

    fn consume_trailing_comma_before(&mut self, terminator: &TokenKind) -> bool {
        if *self.peek() == TokenKind::Comma && self.peek_significant_after(1) == Some(terminator) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn parse_name_bracket_list(&mut self) -> Result<Vec<String>, ParseError> {
        let start = self.expect(&TokenKind::LBracket)?.span;
        let names = self.parse_ident_list(TokenKind::RBracket)?;
        self.expect(&TokenKind::RBracket)?;
        if names.is_empty() && self.mode != ParseMode::LegacyV018 {
            return Err(ParseError::Expected {
                expected: "omit empty `[]`".into(),
                found: "empty parameter list".into(),
                offset: start.offset,
            });
        }
        Ok(names)
    }

    /// Parse a declaration's `[..]` binder list (`spec/02-surf-syntax.md`
    /// §P4b/§P4c). Shared by `def` and `sig`; a `type` declaration's
    /// parameter list keeps [`Self::parse_name_bracket_list`], which has no
    /// bound production.
    fn parse_type_binder_list(&mut self) -> Result<Vec<TypeBinder>, ParseError> {
        let start = self.expect(&TokenKind::LBracket)?.span;
        let mut binders: Vec<TypeBinder> = Vec::new();
        if *self.peek() != TokenKind::RBracket {
            loop {
                binders.push(self.parse_type_binder()?);
                if *self.peek() != TokenKind::Comma {
                    break;
                }
                self.advance();
                if self.comma_terminates_list(&TokenKind::RBracket, binders.len(), false)? {
                    break;
                }
            }
        }
        self.expect(&TokenKind::RBracket)?;
        if binders.is_empty() && self.mode != ParseMode::LegacyV018 {
            return Err(ParseError::Expected {
                expected: "omit empty `[]`".into(),
                found: "empty parameter list".into(),
                offset: start.offset,
            });
        }
        Ok(binders)
    }

    /// One binder: `name`, `name: Family`, or `name: {d1, d2}` for
    /// `spec/02-surf-syntax.md` §P4c's two dtype-bound forms.
    ///
    /// The bound position admits only the three closed family names and a
    /// braced set of active dtype spellings, so an ADT name written there is
    /// a parse error rather than a silently accepted bound. A bare dtype
    /// without braces stays a parse error: §P4c makes a one-member bound
    /// `{f64}`, not a type ascription.
    fn parse_type_binder(&mut self) -> Result<TypeBinder, ParseError> {
        let (name, _) = self.expect_ident_or_type_ident()?;
        if *self.peek() != TokenKind::Colon {
            return Ok(TypeBinder::unbounded(name));
        }
        self.advance();
        if *self.peek() == TokenKind::LBrace {
            return self.parse_dtype_set_bound(name);
        }
        let offset = self.current_offset();
        let (family, _) = self.expect_ident_or_type_ident()?;
        match DtypeFamily::from_surf_name(&family) {
            Some(family) => Ok(TypeBinder::bounded(name, DtypeBound::Family(family))),
            None => Err(ParseError::Expected {
                expected: "a dtype family `Float`, `Int`, or `Numeric`, or a dtype set `{...}`"
                    .into(),
                found: family,
                offset,
            }),
        }
    }

    /// §P4c's explicit dtype set: `{d1, d2}`, non-empty, no repeats, members
    /// drawn from `spec/04-type-system.md` §1.1's active dtypes.
    fn parse_dtype_set_bound(&mut self, name: String) -> Result<TypeBinder, ParseError> {
        let open = self.current_offset();
        self.advance();
        let mut members = std::collections::BTreeSet::new();
        loop {
            if *self.peek() == TokenKind::RBrace {
                break;
            }
            let offset = self.current_offset();
            let (spelling, _) = self.expect_ident_or_type_ident()?;
            let Some(member) = BoundDtype::from_name(&spelling) else {
                return Err(ParseError::Expected {
                    expected: "an active dtype spelling in a dtype set".into(),
                    found: spelling,
                    offset,
                });
            };
            if !members.insert(member) {
                return Err(ParseError::Expected {
                    expected: "a dtype set without a repeated member".into(),
                    found: spelling,
                    offset,
                });
            }
            if *self.peek() == TokenKind::Comma {
                self.advance();
                continue;
            }
            break;
        }
        if *self.peek() != TokenKind::RBrace {
            return Err(ParseError::Expected {
                expected: "`}` closing a dtype set".into(),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            });
        }
        self.advance();
        if members.is_empty() {
            return Err(ParseError::Expected {
                expected: "a non-empty dtype set".into(),
                found: "{}".into(),
                offset: open,
            });
        }
        Ok(TypeBinder::bounded(name, DtypeBound::Set(members)))
    }

    // ---------------------------------------------------------------------------
    // Top-level
    // ---------------------------------------------------------------------------

    fn parse_program(&mut self) -> Result<Vec<Decl>, ParseError> {
        let mut decls = Vec::new();
        while !self.at_eof() {
            let decl = self.parse_decl()?;
            if !matches!(decl, Decl::Module { .. }) {
                self.module_allowed = false;
            }
            decls.push(decl);
        }
        Ok(decls)
    }

    fn parse_decl(&mut self) -> Result<Decl, ParseError> {
        match self.peek() {
            TokenKind::Def => self.parse_fun_def(),
            TokenKind::Sig => self.parse_sig_decl(),
            TokenKind::Ident(_) => self.parse_let_def(),
            // A single-letter uppercase head in declaration position binds
            // a value (chelis#437): `S = ...`. Multi-letter PascalCase is
            // never a value-binding LHS, so it stays a parse error here.
            TokenKind::TypeIdent(name) if is_single_letter_upper(name) => self.parse_let_def(),
            TokenKind::Type => self.parse_type_decl(),
            TokenKind::Dim => self.parse_dim_decl(),
            TokenKind::Macro => self.parse_macro_def(),
            TokenKind::Module => {
                if self.module_allowed {
                    self.parse_module()
                } else {
                    Err(ParseError::Expected {
                        expected: "module declaration only as the first declaration in a file"
                            .into(),
                        found: "Module".into(),
                        offset: self.current_offset(),
                    })
                }
            }
            TokenKind::Import => self.parse_import(),
            TokenKind::Export => self.parse_export(),
            TokenKind::At => self.parse_at_decl(),
            _ => Err(ParseError::Expected {
                expected:
                    "declaration (def, sig, binding, type, dim, macro, module, import, export, @property)"
                        .into(),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            }),
        }
    }

    // ---------------------------------------------------------------------------
    // Declarations
    // ---------------------------------------------------------------------------

    fn parse_at_decl(&mut self) -> Result<Decl, ParseError> {
        let start = self.advance().span; // consume @
        let (keyword, _) = self.expect_ident()?;
        match keyword.as_str() {
            "property" => self.parse_property_decl_after_at(start),
            "opaque" => self.parse_opaque_type_decl(start),
            // `@invariant` may only appear AFTER `@opaque` (RFC
            // D-SYNTAX). Reaching it here as the leading annotation
            // means it has no `@opaque`: assumption injection would be
            // unsound for a forgeable type.
            "invariant" => Err(ParseError::Expected {
                expected: "invariant requires @opaque".into(),
                found: "@invariant".into(),
                offset: start.offset,
            }),
            _ => Err(ParseError::Expected {
                expected: "`property` or `opaque` after `@`".into(),
                found: keyword,
                offset: self.current_offset(),
            }),
        }
    }

    fn parse_property_decl_after_at(&mut self, start: Span) -> Result<Decl, ParseError> {
        let (name, _) = self.expect_ident()?;
        let type_binders = if *self.peek() == TokenKind::LBracket {
            self.parse_type_binder_list()?
        } else {
            Vec::new()
        };
        let (forall, _) = self.expect_ident()?;
        if forall != "forall" {
            return Err(ParseError::Expected {
                expected: "`forall`".into(),
                found: forall,
                offset: self.current_offset(),
            });
        }
        self.expect(&TokenKind::LParen)?;
        let params = self.parse_params()?;
        self.expect(&TokenKind::RParen)?;
        for param in &params {
            if param.ty.is_none() {
                return Err(ParseError::Expected {
                    expected: "explicit property binder type".into(),
                    found: param.name.clone(),
                    offset: param.span.offset,
                });
            }
        }

        let preconditions = if matches!(self.peek(), TokenKind::Ident(word) if word == "where") {
            self.advance();
            self.parse_exprs_until_colon()?
        } else {
            Vec::new()
        };
        self.expect(&TokenKind::Colon)?;
        let body = self.parse_expr_until_property_option()?;
        self.consume_block_separators();

        let mut options = Vec::new();
        while *self.peek() == TokenKind::With {
            options.push(self.parse_property_option()?);
            self.consume_block_separators();
        }
        let end = options
            .last()
            .map(PropertyOption::span)
            .unwrap_or_else(|| expression_span(&body));
        Ok(Decl::Property {
            name,
            type_binders,
            params,
            preconditions,
            body,
            options,
            span: start.merge(end),
        })
    }

    fn parse_opaque_type_decl(&mut self, start: Span) -> Result<Decl, ParseError> {
        // Optional `@invariant(<binder>) <expr>` block, which must appear
        // AFTER `@opaque` and BEFORE `type` (RFC D-SYNTAX). Exactly one
        // invariant; exactly one binder.
        let invariant = self.parse_optional_invariant_block(start)?;

        if *self.peek() != TokenKind::Type {
            return Err(ParseError::Expected {
                expected: "`type` declaration after `@opaque`".into(),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            });
        }
        let decl = self.parse_type_decl_with_opaque(true)?;
        match decl {
            Decl::TypeDef {
                name,
                params,
                variants,
                span,
                ..
            } => Ok(Decl::TypeDef {
                name,
                params,
                variants,
                opaque: true,
                invariant,
                span: start.merge(span),
            }),
            _ => Err(ParseError::Expected {
                expected: "ADT type declaration after `@opaque`".into(),
                found: "type alias".into(),
                offset: start.offset,
            }),
        }
    }

    /// Parse an optional `@invariant(<binder>) <expr>` block following
    /// `@opaque`. Returns `None` when no `@invariant` follows. Rejects a
    /// second `@invariant` (only one allowed), a multi-binder binder
    /// list, and an `@invariant` whose annotation keyword is not
    /// `invariant` (RFC D-SYNTAX).
    fn parse_optional_invariant_block(
        &mut self,
        opaque_start: Span,
    ) -> Result<Option<TypeInvariant>, ParseError> {
        if *self.peek() != TokenKind::At {
            return Ok(None);
        }
        let at_span = self.advance().span; // consume @
        let (keyword, kw_span) = self.expect_ident()?;
        if keyword != "invariant" {
            return Err(ParseError::Expected {
                expected: "`invariant` after `@opaque`".into(),
                found: format!("@{keyword}"),
                offset: kw_span.offset,
            });
        }
        // Binder list: exactly one binder.
        self.expect(&TokenKind::LParen)?;
        let (binder, _) = self.expect_ident()?;
        if *self.peek() == TokenKind::Comma {
            self.advance();
            if *self.peek() != TokenKind::RParen {
                return Err(ParseError::Expected {
                    expected: "exactly one invariant binder".into(),
                    found: "multiple binders".into(),
                    offset: self.current_offset(),
                });
            }
        }
        self.expect(&TokenKind::RParen)?;

        // The predicate body runs up to the `type` keyword at depth 0.
        let end = self.invariant_expr_end();
        let body = self.parse_expr_in_range(end, "`type` after @invariant predicate")?;
        let span = at_span.merge(expression_span(&body)).merge(opaque_start);

        // A second `@invariant` is an error (exactly one allowed).
        if *self.peek() == TokenKind::At {
            let probe = self.pos;
            let at = self.advance();
            if let TokenKind::Ident(word) = self.peek()
                && word == "invariant"
            {
                return Err(ParseError::Expected {
                    expected: "exactly one @invariant".into(),
                    found: "second @invariant".into(),
                    offset: at.span.offset,
                });
            }
            // Not a second invariant; rewind so the next stage sees it.
            self.pos = probe;
        }

        Ok(Some(TypeInvariant { binder, body, span }))
    }

    /// The token index where an `@invariant` predicate body ends: the
    /// `type` keyword at bracket depth 0. Errors are surfaced by the
    /// subsequent `type` expectation, so this returns the scan position.
    fn invariant_expr_end(&self) -> usize {
        let mut pos = self.pos;
        let mut paren_depth = 0usize;
        let mut bracket_depth = 0usize;
        let mut brace_depth = 0usize;
        while let Some(token) = self.tokens.get(pos) {
            match token.kind {
                TokenKind::Type if paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 => {
                    break;
                }
                TokenKind::At if paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 => {
                    // A following `@` (e.g. a stray second annotation)
                    // also terminates the predicate scan.
                    break;
                }
                TokenKind::LParen => paren_depth += 1,
                TokenKind::RParen if paren_depth > 0 => paren_depth -= 1,
                TokenKind::LBracket => bracket_depth += 1,
                TokenKind::RBracket if bracket_depth > 0 => bracket_depth -= 1,
                TokenKind::LBrace => brace_depth += 1,
                TokenKind::RBrace if brace_depth > 0 => brace_depth -= 1,
                TokenKind::Eof => break,
                _ => {}
            }
            pos += 1;
        }
        pos
    }

    fn parse_exprs_until_colon(&mut self) -> Result<Vec<Expr>, ParseError> {
        let mut exprs = Vec::new();
        loop {
            if *self.peek() == TokenKind::Colon {
                break;
            }
            let end = self.find_property_clause_end()?;
            let redundant_group_offset = self.redundant_property_group_offset(end);
            let expr = self.parse_expr_in_range(end, "`,` or `:` in property where clause")?;
            if self.mode != ParseMode::LegacyV018
                && redundant_group_offset.is_some()
                && !matches!(expr, Expr::Tuple(..))
            {
                return Err(ParseError::Expected {
                    expected: "property precondition without redundant outer grouping".into(),
                    found: "parenthesized property precondition".into(),
                    offset: redundant_group_offset.unwrap_or_default(),
                });
            }
            exprs.push(expr);
            if *self.peek() == TokenKind::Comma {
                self.advance();
                if *self.peek() == TokenKind::Colon {
                    break;
                }
                continue;
            }
            if *self.peek() == TokenKind::Colon {
                break;
            }
            return Err(ParseError::Expected {
                expected: "`,` or `:` in property where clause".into(),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            });
        }
        Ok(exprs)
    }

    /// Return the opening-parenthesis offset when one grouping pair encloses
    /// the complete property precondition. Tuple/unit delimiters are filtered
    /// after parsing; a pair that closes before the clause end is meaningful
    /// operand or callee grouping and is therefore not reported here.
    fn redundant_property_group_offset(&self, end: usize) -> Option<usize> {
        let significant = self.tokens[self.pos..end]
            .iter()
            .filter(|token| !matches!(token.kind, TokenKind::Newline))
            .collect::<Vec<_>>();
        let first = significant.first()?;
        if !matches!(first.kind, TokenKind::LParen)
            || !matches!(significant.last()?.kind, TokenKind::RParen)
        {
            return None;
        }

        let mut depth = 0usize;
        for (index, token) in significant.iter().enumerate() {
            match token.kind {
                TokenKind::LParen => depth += 1,
                TokenKind::RParen => {
                    depth = depth.checked_sub(1)?;
                    if depth == 0 && index + 1 != significant.len() {
                        return None;
                    }
                }
                _ => {}
            }
        }
        (depth == 0).then_some(first.span.offset)
    }

    fn find_property_clause_end(&self) -> Result<usize, ParseError> {
        let mut pos = self.pos;
        let mut paren_depth = 0usize;
        let mut bracket_depth = 0usize;
        let mut brace_depth = 0usize;
        while let Some(token) = self.tokens.get(pos) {
            match token.kind {
                TokenKind::Comma | TokenKind::Colon
                    if paren_depth == 0 && bracket_depth == 0 && brace_depth == 0 =>
                {
                    return Ok(pos);
                }
                TokenKind::LParen => paren_depth += 1,
                TokenKind::RParen if paren_depth > 0 => paren_depth -= 1,
                TokenKind::LBracket => bracket_depth += 1,
                TokenKind::RBracket if bracket_depth > 0 => bracket_depth -= 1,
                TokenKind::LBrace => brace_depth += 1,
                TokenKind::RBrace if brace_depth > 0 => brace_depth -= 1,
                TokenKind::Eof => break,
                _ => {}
            }
            pos += 1;
        }
        Err(ParseError::Expected {
            expected: "`:` after property where clause".into(),
            found: format!("{:?}", self.peek()),
            offset: self.current_offset(),
        })
    }

    fn parse_property_option(&mut self) -> Result<PropertyOption, ParseError> {
        let start = self.advance().span; // consume with
        let (name, _) = self.expect_ident()?;
        self.expect(&TokenKind::Eq)?;
        let value = self.parse_expr_until_block_separator()?;
        let span = start.merge(expression_span(&value));
        match name.as_str() {
            "tolerance" => Ok(PropertyOption::Tolerance(value, span)),
            "seed" => Ok(PropertyOption::Seed(value, span)),
            "samples" => Ok(PropertyOption::Samples(value, span)),
            "contract" => match value {
                Expr::Lit(Literal::Str(id), _) => Ok(PropertyOption::Contract(id, span)),
                _ => Err(ParseError::Expected {
                    expected: "string literal contract id".into(),
                    found: format!("{value:?}"),
                    offset: start.offset,
                }),
            },
            _ => Err(ParseError::Expected {
                expected: "property option `tolerance`, `seed`, `samples`, or `contract`".into(),
                found: name,
                offset: start.offset,
            }),
        }
    }

    fn parse_fun_def(&mut self) -> Result<Decl, ParseError> {
        let start = self.advance().span; // consume Def
        let (name, _) = self.expect_ident()?;

        // Optional binder list: def f[a, b](...) / def f[p: Float](...)
        let type_binders = if *self.peek() == TokenKind::LBracket {
            self.parse_type_binder_list()?
        } else {
            Vec::new()
        };

        let params = if *self.peek() == TokenKind::LParen {
            self.advance();
            let params = self.parse_params()?;
            self.expect(&TokenKind::RParen)?;
            params
        } else if self.mode != ParseMode::LegacyV018 {
            return Err(ParseError::Expected {
                expected: "function parameter list `()`".into(),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            });
        } else {
            Vec::new()
        };

        let ret_ty = if *self.peek() == TokenKind::Arrow
            || (*self.peek() == TokenKind::Colon && self.mode == ParseMode::LegacyV018)
        {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };
        let effects = self.parse_optional_effects()?;

        self.expect(&TokenKind::Eq)?;
        let body = self.parse_expr_until_decl_separator().map_err(|e| {
            if matches!(e, ParseError::UnexpectedEof) {
                // Check if the next token after the boundary is 'if' — a common
                // mistake of splitting if/then/else onto a separate line after '='.
                let remaining = &self.tokens[self.pos..];
                let next_meaningful = remaining.iter().find(|t| {
                    !matches!(t.kind, TokenKind::Newline | TokenKind::Semicolon)
                });
                if matches!(next_meaningful.map(|t| &t.kind), Some(TokenKind::If)) {
                    return ParseError::Expected {
                        expected: "expression after `=`".into(),
                        found: "unexpected end of expression; did you mean to write the if/then/else on the same line or use braces?".into(),
                        offset: self.current_offset(),
                    };
                }
            }
            e
        })?;
        let span = start.merge(expression_span(&body));

        Ok(Decl::FunDef {
            name,
            type_binders,
            params,
            ret_ty,
            effects,
            body,
            span,
        })
    }

    fn parse_sig_decl(&mut self) -> Result<Decl, ParseError> {
        let start = self.advance().span; // consume Sig
        if let Some(err) =
            self.reserved_word_binding_error("sig", "a signature declaration", "sigma", start)
        {
            return Err(err);
        }
        let (name, _) = self.expect_ident()?;
        // Optional binder list: sig f[p: Float]: ... (§P4c). It sits in the
        // same position it occupies on a `def`, immediately after the name.
        let type_binders = if *self.peek() == TokenKind::LBracket {
            self.parse_type_binder_list()?
        } else {
            Vec::new()
        };
        self.expect(&TokenKind::Colon)?;
        let ty = self.parse_type()?;
        let effects = self.parse_optional_effects()?;
        let end = effects
            .as_ref()
            .and_then(|effects| effects.last())
            .map(EffectExpr::span)
            .unwrap_or_else(|| type_span(&ty));
        let span = start.merge(end);
        Ok(Decl::Sig {
            name,
            type_binders,
            ty,
            effects,
            span,
        })
    }

    fn parse_dim_decl(&mut self) -> Result<Decl, ParseError> {
        let start = self.advance().span; // consume Dim
        if let Some(err) =
            self.reserved_word_binding_error("dim", "a dimension declaration", "d", start)
        {
            return Err(err);
        }
        let names = self.parse_ident_list(TokenKind::Eof)?;
        let end = self.tokens[self.pos - 1].span;
        Ok(Decl::Dim {
            names,
            span: start.merge(end),
        })
    }

    fn parse_params(&mut self) -> Result<Vec<Param>, ParseError> {
        let mut params = Vec::new();
        if *self.peek() == TokenKind::RParen {
            return Ok(params);
        }
        params.push(self.parse_param()?);
        while *self.peek() == TokenKind::Comma {
            self.advance();
            if *self.peek() == TokenKind::RParen {
                break;
            }
            params.push(self.parse_param()?);
        }
        Ok(params)
    }

    fn parse_param(&mut self) -> Result<Param, ParseError> {
        let (name, span) = self.expect_value_ident()?;
        let ty = if *self.peek() == TokenKind::Colon {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };
        let end = ty.as_ref().map(type_span).unwrap_or(span);
        Ok(Param {
            name,
            ty,
            span: span.merge(end),
        })
    }

    fn parse_plain_ident_params(&mut self) -> Result<Vec<String>, ParseError> {
        let mut params = Vec::new();
        if *self.peek() == TokenKind::RParen {
            return Ok(params);
        }
        let (name, _) = self.expect_ident()?;
        params.push(name);
        while *self.peek() == TokenKind::Comma {
            self.advance();
            if *self.peek() == TokenKind::RParen {
                break;
            }
            let (name, _) = self.expect_ident()?;
            params.push(name);
        }
        Ok(params)
    }

    fn parse_let_def(&mut self) -> Result<Decl, ParseError> {
        let (name, start) = self.expect_value_ident()?;
        let ty = if *self.peek() == TokenKind::Colon {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };
        self.expect(&TokenKind::Eq)?;
        let value = self.parse_expr_until_decl_separator()?;
        let span = start.merge(expression_span(&value));
        Ok(Decl::LetDef {
            name,
            ty,
            value,
            span,
        })
    }

    fn parse_macro_def(&mut self) -> Result<Decl, ParseError> {
        let start = self.advance().span; // consume Macro
        let (name, _) = self.expect_ident()?;
        self.expect(&TokenKind::LParen)?;
        let params = self.parse_plain_ident_params()?;
        self.expect(&TokenKind::RParen)?;
        self.expect(&TokenKind::Eq)?;
        let body = self.parse_expr_until_decl_separator()?;
        let span = start.merge(expression_span(&body));
        Ok(Decl::MacroDef {
            name,
            params,
            body,
            span,
        })
    }

    fn parse_type_decl(&mut self) -> Result<Decl, ParseError> {
        self.parse_type_decl_with_opaque(false)
    }

    fn parse_type_decl_with_opaque(&mut self, opaque: bool) -> Result<Decl, ParseError> {
        let start = self.advance().span; // consume Type
        if let Some(err) =
            self.reserved_word_binding_error("type", "a type declaration", "ty", start)
        {
            return Err(err);
        }
        let (name, _) = self.expect_type_ident()?;
        let params = if *self.peek() == TokenKind::LBracket {
            self.parse_name_bracket_list()?
        } else {
            Vec::new()
        };

        self.expect(&TokenKind::Eq)?;

        if *self.peek() == TokenKind::Bar {
            let mut variants = Vec::new();
            self.advance();
            variants.push(self.parse_variant()?);
            while *self.peek() == TokenKind::Bar {
                self.advance();
                variants.push(self.parse_variant()?);
            }
            let last_span = variants.last().unwrap().span;
            Ok(Decl::TypeDef {
                name,
                params,
                variants,
                opaque,
                invariant: None,
                span: start.merge(last_span),
            })
        } else {
            let ty = self.parse_type()?;
            let span = start.merge(type_span(&ty));
            Ok(Decl::TypeAlias {
                name,
                params,
                ty,
                span,
            })
        }
    }

    fn parse_variant(&mut self) -> Result<Variant, ParseError> {
        let (name, start) = self.expect_type_ident()?;

        if *self.peek() == TokenKind::LBrace {
            // Record variant
            self.advance();
            let mut fields = Vec::new();
            if *self.peek() != TokenKind::RBrace {
                fields.push(self.parse_record_field()?);
                while *self.peek() == TokenKind::Comma {
                    self.advance();
                    if self.comma_terminates_list(&TokenKind::RBrace, fields.len(), false)? {
                        break;
                    }
                    fields.push(self.parse_record_field()?);
                }
            }
            let end = self.expect(&TokenKind::RBrace)?;
            if fields.is_empty() && self.mode != ParseMode::LegacyV018 {
                return Err(ParseError::Expected {
                    expected: format!("bare zero-field variant `{name}`"),
                    found: format!("`{name} {{}}`"),
                    offset: start.offset,
                });
            }
            Ok(Variant {
                name,
                fields: VariantFields::Record(fields),
                span: start.merge(end.span),
            })
        } else if *self.peek() == TokenKind::LParen {
            self.advance();
            let mut fields = Vec::new();
            if *self.peek() != TokenKind::RParen {
                fields.push(self.parse_type()?);
                while *self.peek() == TokenKind::Comma {
                    self.advance();
                    if self.comma_terminates_list(&TokenKind::RParen, fields.len(), false)? {
                        break;
                    }
                    fields.push(self.parse_type()?);
                }
            }
            let end = self.expect(&TokenKind::RParen)?;
            if self.mode != ParseMode::LegacyV018 && fields.is_empty() {
                return Err(ParseError::Expected {
                    expected: format!("bare zero-argument constructor `{name}`"),
                    found: format!("`{name}()`"),
                    offset: start.offset,
                });
            }
            Ok(Variant {
                name,
                fields: VariantFields::Positional(fields),
                span: start.merge(end.span),
            })
        } else {
            Ok(Variant {
                name,
                fields: VariantFields::Positional(Vec::new()),
                span: start,
            })
        }
    }

    fn parse_record_field(&mut self) -> Result<(String, TypeExpr), ParseError> {
        let (name, _) = self.expect_ident()?;
        self.expect(&TokenKind::Colon)?;
        let ty = self.parse_type()?;
        Ok((name, ty))
    }

    fn parse_module(&mut self) -> Result<Decl, ParseError> {
        let start = self.advance().span; // consume Module
        let (name, _) = self.parse_module_path()?;
        self.module_allowed = false;

        let mut decls = Vec::new();
        while !self.at_eof() {
            decls.push(self.parse_decl()?);
        }
        let end = decls.last().map(decl_span).unwrap_or(start);
        Ok(Decl::Module {
            name,
            decls,
            span: start.merge(end),
        })
    }

    fn parse_import(&mut self) -> Result<Decl, ParseError> {
        let start = self.advance().span; // consume Import
        let (module, mod_span) = self.parse_module_path()?;

        let (kind, end) = if *self.peek() == TokenKind::LParen {
            self.advance();
            let kind = if *self.peek() == TokenKind::DotDot {
                // `import Foo(..)` import-all. `..` now lexes as a single
                // DotDot token (the rank-spread marker), so accept it here
                // rather than two `Dot`s.
                self.advance();
                ImportKind::All
            } else {
                let names = self.parse_ident_list(TokenKind::RParen)?;
                if names.is_empty() {
                    if self.mode != ParseMode::LegacyV018 {
                        return Err(ParseError::Expected {
                            expected: "qualified import without empty `()`".into(),
                            found: "empty import list".into(),
                            offset: self.current_offset(),
                        });
                    }
                    ImportKind::Qualified
                } else {
                    ImportKind::Names(names)
                }
            };
            let end = self.expect(&TokenKind::RParen)?;
            (kind, end.span)
        } else {
            (ImportKind::Qualified, mod_span)
        };

        Ok(Decl::Import {
            module,
            kind,
            span: start.merge(end),
        })
    }

    fn parse_export(&mut self) -> Result<Decl, ParseError> {
        let start = self.advance().span; // consume Export
        self.expect(&TokenKind::LParen)?;
        let names = self.parse_ident_list(TokenKind::RParen)?;
        if names.is_empty() {
            return Err(ParseError::Expected {
                expected: "at least one exported name".into(),
                found: "empty export list".into(),
                offset: self.current_offset(),
            });
        }
        let end = self.expect(&TokenKind::RParen)?;
        Ok(Decl::Export {
            names,
            span: start.merge(end.span),
        })
    }

    // ---------------------------------------------------------------------------
    // Expression parsing — Pratt
    // ---------------------------------------------------------------------------

    fn parse_expr(&mut self, min_bp: u8) -> Result<Expr, ParseError> {
        if self.mode != ParseMode::LegacyV018
            && min_bp > 0
            && matches!(
                self.peek(),
                TokenKind::If | TokenKind::Match | TokenKind::Fn
            )
        {
            return Err(ParseError::Expected {
                expected: "parenthesized low-precedence expression as an operator operand".into(),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            });
        }
        let expression_start = self.pos;
        let mut lhs = self.parse_prefix()?;

        loop {
            if self.at_eof() {
                break;
            }

            if *self.peek() == TokenKind::Dot {
                let l_bp = 15u8;
                if l_bp < min_bp {
                    break;
                }
                self.advance();
                match self.peek().clone() {
                    TokenKind::Ident(field) => {
                        let tok = self.advance();
                        let start = expression_span(&lhs);
                        lhs = Expr::Access(Box::new(lhs), field, start.merge(tok.span));
                    }
                    // A PascalCase segment after `.` is a module/type path
                    // component, never a record field (fields are snake_case,
                    // §3.2). Accepting it lets a module-qualified reference
                    // such as `Demo.Dropout.Eval` (a constructor) or
                    // `Demo.Dropout.use` (a value) parse into an `Access`
                    // chain that reef resolves to the module-qualified
                    // internal name (chelis#316). Without this, the only way
                    // to disambiguate two imported modules that export the
                    // same constructor name was to rename one — and the
                    // ambiguity diagnostic's own suggestion (`Module.Eval`)
                    // did not parse.
                    TokenKind::TypeIdent(segment) => {
                        let tok = self.advance();
                        let start = expression_span(&lhs);
                        lhs = Expr::Access(Box::new(lhs), segment, start.merge(tok.span));
                    }
                    TokenKind::Int(index) => {
                        let tok = self.advance();
                        let start = expression_span(&lhs);
                        lhs = Expr::TupleGet(Box::new(lhs), index, start.merge(tok.span));
                        continue;
                    }
                    _ => {
                        return Err(ParseError::Expected {
                            expected: "field name or tuple index".into(),
                            found: format!("{:?}", self.peek()),
                            offset: self.current_offset(),
                        });
                    }
                }
                // A dotted path may be applied: `Demo.Dropout.use(m)` or the
                // qualified constructor call `Demo.List.Cons(x, xs)`. The
                // prefix-position juxtaposition handler only runs on the head
                // atom, before this postfix `.` chain is built, so consume a
                // trailing parenthesized argument list (or a curried chain of
                // them) here. This also fixes plain `Module.func(arg)`, which
                // previously failed with "expected end of declaration
                // expression, found LParen".
                while *self.peek() == TokenKind::LParen {
                    let start = expression_span(&lhs);
                    self.advance();
                    let (args, accumulator, end) = self.parse_call_args()?;
                    lhs = call_expr(lhs, args, accumulator, start.merge(end));
                    if self.mode != ParseMode::LegacyV018 {
                        break;
                    }
                }
                continue;
            }

            if self.starts_record_update(self.pos) {
                let start = expression_span(&lhs);
                self.advance();
                let fields = self.parse_record_fields()?;
                let end = self.tokens[self.pos - 1].span;
                if fields.is_empty() {
                    if self.mode != ParseMode::LegacyV018 {
                        return Err(ParseError::Expected {
                            expected: "at least one record-update field".into(),
                            found: "empty record update".into(),
                            offset: start.offset,
                        });
                    }
                    continue;
                }
                lhs = Expr::RecordUpdate(Box::new(lhs), fields, start.merge(end));
                continue;
            }

            // Check for pipe
            if *self.peek() == TokenKind::Pipe {
                let (l_bp, _r_bp) = (1u8, 2u8);
                if l_bp < min_bp {
                    break;
                }
                let start = expression_span(&lhs);
                let mut stages = Vec::new();
                while *self.peek() == TokenKind::Pipe {
                    self.advance();
                    let stage = self.parse_pipe_stage()?;
                    stages.push(stage);
                }
                let end = expression_span(stages.last().unwrap());
                lhs = Expr::Pipe(Box::new(lhs), stages, start.merge(end));
                continue;
            }

            // Check for type annotation colon
            if *self.peek() == TokenKind::Colon {
                let l_bp = 14u8;
                if l_bp < min_bp {
                    break;
                }
                self.advance();
                let ty = self.parse_type()?;
                let start = expression_span(&lhs);
                let end = type_span(&ty);
                lhs = Expr::Annotate(Box::new(lhs), ty, start.merge(end));
                continue;
            }

            // Binary operators
            if let Some((op, l_bp, r_bp)) = self.infix_bp() {
                if l_bp < min_bp {
                    break;
                }
                let non_assoc = l_bp == r_bp;
                self.advance();

                // For non-assoc ops, bump right BP so the recursive call
                // does NOT consume a same-precedence operator on the right.
                let effective_r_bp = if non_assoc { r_bp + 1 } else { r_bp };
                let rhs = self.parse_expr(effective_r_bp)?;

                // Non-assoc check: after parsing RHS, if the next token is
                // the same op class at the same precedence, that's a chain.
                if non_assoc
                    && let Some((next_op, next_l, _)) = self.infix_bp()
                    && op_class(op) == op_class(next_op)
                    && next_l == l_bp
                {
                    return Err(ParseError::NonAssocChain {
                        offset: self.current_offset(),
                    });
                }

                let start = expression_span(&lhs);
                let end = expression_span(&rhs);
                lhs = Expr::Binary(op, Box::new(lhs), Box::new(rhs), start.merge(end));
                continue;
            }

            break;
        }

        if self.mode == ParseMode::Canonical {
            self.check_pipe_grouping(expression_start, self.pos)?;
        }
        Ok(lhs)
    }

    fn check_pipe_grouping(&self, start: usize, end: usize) -> Result<(), ParseError> {
        let mut index = start;
        let mut pipe = None;
        let mut mixed = false;
        while index < end {
            let token = &self.tokens[index];
            if self.closing_delimiters[index] < end {
                index = self.closing_delimiters[index] + 1;
                continue;
            }
            index += 1;
            match token.kind {
                TokenKind::Pipe => {
                    pipe = Some(token.span.offset);
                }
                TokenKind::Plus
                | TokenKind::Minus
                | TokenKind::Star
                | TokenKind::Slash
                | TokenKind::Percent
                | TokenKind::EqEq
                | TokenKind::BangEq
                | TokenKind::Lt
                | TokenKind::Gt
                | TokenKind::LtEq
                | TokenKind::GtEq
                | TokenKind::AmpAmp
                | TokenKind::PipePipe
                | TokenKind::Bang
                | TokenKind::Amp
                | TokenKind::Colon
                | TokenKind::Dot
                | TokenKind::If
                | TokenKind::Match
                | TokenKind::Fn => mixed = true,
                TokenKind::With if self.starts_record_update(index - 1) => mixed = true,
                _ => {}
            }
        }
        if mixed && let Some(offset) = pipe {
            return Err(ParseError::Expected {
                expected: "explicit grouping around pipe operands mixed with operators or open-ended forms; write `(a + b) |> f` or `a + (b |> f)`".into(),
                found: "ungrouped pipe combination".into(), offset,
            });
        }
        Ok(())
    }

    fn infix_bp(&self) -> Option<(BinOp, u8, u8)> {
        match self.peek() {
            TokenKind::PipePipe => Some((BinOp::Or, 3, 4)),
            TokenKind::AmpAmp => Some((BinOp::And, 5, 6)),
            TokenKind::EqEq => Some((BinOp::Eq, 7, 7)),
            TokenKind::BangEq => Some((BinOp::Ne, 7, 7)),
            TokenKind::Lt => Some((BinOp::Lt, 8, 8)),
            TokenKind::Gt => Some((BinOp::Gt, 8, 8)),
            TokenKind::LtEq => Some((BinOp::Le, 8, 8)),
            TokenKind::GtEq => Some((BinOp::Ge, 8, 8)),
            TokenKind::Plus => Some((BinOp::Add, 9, 10)),
            TokenKind::Minus => Some((BinOp::Sub, 9, 10)),
            TokenKind::Star => Some((BinOp::Mul, 11, 12)),
            TokenKind::Slash => Some((BinOp::Div, 11, 12)),
            TokenKind::Percent => Some((BinOp::Mod, 11, 12)),
            _ => None,
        }
    }

    /// Parse Surf-only stage syntax without synthesizing a lambda or binder.
    fn parse_pipe_stage(&mut self) -> Result<PipeStage, ParseError> {
        let start = self.tokens[self.pos].span;
        let mode = match self.peek() {
            TokenKind::Cast => Some(CastMode::Checked),
            TokenKind::NamedCast(named) => Some(CastMode::Named(*named)),
            _ => None,
        };
        if let Some(mode) = mode
            && let Some((precision, precision_span)) = self.peek_one_arg_cast_precision()
        {
            self.reject_retired_integer_dtype_name(&precision, precision_span)?;
            self.advance();
            self.advance();
            self.advance();
            let close = self.expect(&TokenKind::RParen)?;
            return Ok(PipeStage {
                expression: Expr::Apply(
                    Box::new(Expr::Var(mode.keyword().into(), start)),
                    vec![Expr::Var(precision, precision_span)],
                    start.merge(close.span),
                ),
                syntax: PipeStageSyntax::Cast(mode),
            });
        }
        if matches!(self.peek(), TokenKind::Copy | TokenKind::Realize)
            && !matches!(self.peek_significant_after(1), Some(TokenKind::LParen))
        {
            let syntax = if *self.peek() == TokenKind::Copy {
                PipeStageSyntax::Copy
            } else {
                PipeStageSyntax::Realize
            };
            let name = if syntax == PipeStageSyntax::Copy {
                "copy"
            } else {
                "realize"
            };
            self.advance();
            return Ok(PipeStage {
                expression: Expr::Var(name.into(), start),
                syntax,
            });
        }
        Ok(self.parse_prefix()?.into())
    }

    /// If the next four tokens are `Cast LParen Ident RParen`, return
    /// the precision identifier (the type name). Otherwise `None`.
    /// Used by `parse_pipe_stage` to recognize the H3 one-arg
    /// `cast(type)` pipe-stage form without consuming tokens on miss.
    fn peek_one_arg_cast_precision(&self) -> Option<(String, Span)> {
        // Caller has already verified `self.peek() == Cast`. We need to
        // look at the token after Cast (skipping newlines), then the
        // token after that, etc. Using `peek_after_current` would only
        // see one ahead, so do a manual look-ahead here.
        let mut pos = self.pos;
        // Skip leading newlines.
        while matches!(
            self.tokens.get(pos).map(|t| &t.kind),
            Some(TokenKind::Newline)
        ) {
            pos += 1;
        }
        // Step over `Cast` / `NamedCast`.
        if !matches!(
            self.tokens.get(pos).map(|t| &t.kind),
            Some(TokenKind::Cast | TokenKind::NamedCast(_))
        ) {
            return None;
        }
        pos += 1;
        while matches!(
            self.tokens.get(pos).map(|t| &t.kind),
            Some(TokenKind::Newline)
        ) {
            pos += 1;
        }
        // Expect `LParen`.
        if !matches!(
            self.tokens.get(pos).map(|t| &t.kind),
            Some(TokenKind::LParen)
        ) {
            return None;
        }
        pos += 1;
        while matches!(
            self.tokens.get(pos).map(|t| &t.kind),
            Some(TokenKind::Newline)
        ) {
            pos += 1;
        }
        // Expect Ident (the precision name).
        let (precision, precision_span) = match self.tokens.get(pos) {
            Some(token) => match &token.kind {
                TokenKind::Ident(name) => (name.clone(), token.span),
                _ => return None,
            },
            _ => return None,
        };
        pos += 1;
        while matches!(
            self.tokens.get(pos).map(|t| &t.kind),
            Some(TokenKind::Newline)
        ) {
            pos += 1;
        }
        // Expect `RParen` (no Comma → exactly one arg, the type).
        if !matches!(
            self.tokens.get(pos).map(|t| &t.kind),
            Some(TokenKind::RParen)
        ) {
            return None;
        }
        Some((precision, precision_span))
    }

    /// Pick a fresh `__chelis_pipe[N]` parameter name. Mirrors the desugar
    /// helper at `desugar.rs::fresh_pipe_param_name`. The parser only knows
    /// the current stage's source text, so a basic counter-based choice
    /// works here; the desugarer's name-collision check covers the rest.
    fn fresh_pipe_param_name(&self) -> String {
        "__chelis_pipe".to_string()
    }

    fn parse_prefix(&mut self) -> Result<Expr, ParseError> {
        if self.at_eof() {
            return Err(ParseError::UnexpectedEof);
        }

        let expr = match self.peek().clone() {
            TokenKind::Int(n) => {
                let tok = self.advance();
                Expr::Lit(Literal::Int(n), tok.span)
            }
            TokenKind::Float(f) => {
                let tok = self.advance();
                Expr::Lit(Literal::Float(f), tok.span)
            }
            TokenKind::TypedInt(n, suffix) => {
                let tok = self.advance();
                Expr::Lit(Literal::TypedInt(n, suffix), tok.span)
            }
            TokenKind::TypedFloat(f, suffix) => {
                let tok = self.advance();
                Expr::Lit(Literal::TypedFloat(f, suffix), tok.span)
            }
            TokenKind::IntMinMagnitude(suffix) => {
                let found = format!(
                    "9223372036854775808{}",
                    suffix.as_ref().map_or("", |suffix| suffix.as_str())
                );
                return Err(ParseError::SignedMinimumMagnitudeRequiresNegation {
                    found,
                    offset: self.current_offset(),
                });
            }
            TokenKind::Str(s) => {
                let tok = self.advance();
                Expr::Lit(Literal::Str(s), tok.span)
            }
            TokenKind::True => {
                let tok = self.advance();
                Expr::Lit(Literal::Bool(true), tok.span)
            }
            TokenKind::False => {
                let tok = self.advance();
                Expr::Lit(Literal::Bool(false), tok.span)
            }
            TokenKind::Ident(name) => {
                let tok = self.advance();
                Expr::Var(name, tok.span)
            }
            TokenKind::TypeIdent(name) => {
                let tok = self.advance();
                if *self.peek() == TokenKind::LBrace {
                    self.parse_record_expr(name, tok.span)?
                } else {
                    Expr::Constructor(name, tok.span)
                }
            }
            TokenKind::Minus => {
                let tok = self.advance();
                if let TokenKind::IntMinMagnitude(suffix) = self.peek().clone() {
                    let magnitude = self.advance();
                    // The positive magnitude 2^63 cannot fit in the signed
                    // payload carried by `Literal`. Keep the signed minimum
                    // only as an internal magnitude sentinel, but preserve
                    // P10's user-visible parse shape: the expression is
                    // still unary minus, never a negative literal node.
                    let literal = match suffix {
                        None => Literal::Int(i64::MIN),
                        Some(LiteralSuffix::I64) => Literal::TypedInt(i64::MIN, LiteralSuffix::I64),
                        Some(_) => unreachable!("lexer admits only an i64 suffix here"),
                    };
                    let operand = Expr::Lit(literal, magnitude.span);
                    return Ok(Expr::Unary(
                        UnaryOp::Neg,
                        Box::new(operand),
                        tok.span.merge(magnitude.span),
                    ));
                }
                let operand = self.parse_expr(13)?;
                let span = tok.span.merge(expression_span(&operand));
                return Ok(Expr::Unary(UnaryOp::Neg, Box::new(operand), span));
            }
            TokenKind::Bang => {
                let tok = self.advance();
                let operand = self.parse_expr(13)?;
                let span = tok.span.merge(expression_span(&operand));
                return Ok(Expr::Unary(UnaryOp::Not, Box::new(operand), span));
            }
            TokenKind::Amp => {
                let tok = self.advance();
                let operand = self.parse_expr(13)?;
                let span = tok.span.merge(expression_span(&operand));
                return Ok(Expr::Borrow(Box::new(operand), span));
            }
            TokenKind::LParen => {
                let start = self.advance().span;
                if *self.peek() == TokenKind::RParen {
                    let end = self.advance().span;
                    Expr::Tuple(Vec::new(), start.merge(end))
                } else {
                    let first = self.parse_expr(0)?;
                    if *self.peek() == TokenKind::Comma {
                        // Tuple
                        let mut elems = vec![first];
                        while *self.peek() == TokenKind::Comma {
                            self.advance();
                            if self.comma_terminates_list(&TokenKind::RParen, elems.len(), true)? {
                                break;
                            }
                            elems.push(self.parse_expr(0)?);
                        }
                        let end = self.expect(&TokenKind::RParen)?;
                        Expr::Tuple(elems, start.merge(end.span))
                    } else {
                        // Grouping
                        self.expect(&TokenKind::RParen)?;
                        first
                    }
                }
            }
            TokenKind::LBracket => {
                let start = self.advance().span;
                let items = self.parse_expr_list(TokenKind::RBracket)?;
                let end = self.expect(&TokenKind::RBracket)?;
                Expr::List(items, start.merge(end.span))
            }
            TokenKind::If => self.parse_if()?,
            TokenKind::Match => self.parse_match()?,
            TokenKind::Fn => self.parse_lambda()?,
            TokenKind::Cast => self.parse_cast(CastMode::Checked)?,
            TokenKind::NamedCast(named) => self.parse_cast(CastMode::Named(named))?,
            TokenKind::Grad => self.parse_grad()?,
            TokenKind::Vmap => self.parse_vmap()?,
            TokenKind::Jit => self.parse_jit()?,
            TokenKind::Realize => self.parse_realize()?,
            TokenKind::Copy => self.parse_copy()?,
            TokenKind::With => self.parse_with_handler()?,
            TokenKind::Par => self.parse_par()?,
            TokenKind::Do => self.parse_do()?,
            TokenKind::Quote => self.parse_meta_form(DeepSurfaceMetaForm::Quote)?,
            TokenKind::Unquote => self.parse_meta_form(DeepSurfaceMetaForm::Unquote)?,
            TokenKind::Splice => self.parse_meta_form(DeepSurfaceMetaForm::Splice)?,
            TokenKind::LBrace => self.parse_block()?,
            _ => {
                return Err(ParseError::Expected {
                    expected: "expression".into(),
                    found: format!("{:?}", self.peek()),
                    offset: self.current_offset(),
                });
            }
        };

        self.parse_juxtaposition_args(expr)
    }

    fn can_start_juxtaposition_arg(&self) -> bool {
        matches!(
            self.peek(),
            TokenKind::Ident(_)
                | TokenKind::TypeIdent(_)
                | TokenKind::Int(_)
                | TokenKind::Float(_)
                | TokenKind::TypedInt(_, _)
                | TokenKind::TypedFloat(_, _)
                | TokenKind::Str(_)
                | TokenKind::True
                | TokenKind::False
                | TokenKind::LParen
                | TokenKind::Amp
                | TokenKind::Fn
                | TokenKind::If
        )
    }

    fn parse_juxtaposition_args(&mut self, mut expr: Expr) -> Result<Expr, ParseError> {
        if self.mode != ParseMode::LegacyV018 {
            if *self.peek() == TokenKind::LParen {
                let start = expression_span(&expr);
                self.advance();
                let (args, accumulator, end) = self.parse_call_args()?;
                expr = call_expr(expr, args, accumulator, start.merge(end));
            }
            return Ok(expr);
        }

        loop {
            if *self.peek() == TokenKind::LParen {
                let start = expression_span(&expr);
                self.advance();
                let (args, accumulator, end) = self.parse_call_args()?;
                let span = start.merge(end);
                expr = extend_legacy_application(expr, args, span);
                if let Some(precision) = accumulator {
                    expr = Expr::Accumulate(Box::new(expr), precision, span);
                }
                continue;
            }

            if self.can_start_juxtaposition_arg()
                && self.infix_bp().is_none()
                && *self.peek() != TokenKind::Pipe
            {
                let arg = self.parse_primary_atom()?;
                let start = expression_span(&expr);
                let end = expression_span(&arg);
                expr = extend_legacy_application(expr, vec![arg], start.merge(end));
                continue;
            }

            break;
        }

        Ok(expr)
    }

    /// Parse a single primary expression without juxtaposition chaining.
    /// Used for juxtaposition arguments to avoid infinite recursion.
    fn parse_primary_atom(&mut self) -> Result<Expr, ParseError> {
        match self.peek().clone() {
            TokenKind::Int(n) => {
                let tok = self.advance();
                Ok(Expr::Lit(Literal::Int(n), tok.span))
            }
            TokenKind::Float(f) => {
                let tok = self.advance();
                Ok(Expr::Lit(Literal::Float(f), tok.span))
            }
            TokenKind::TypedInt(n, suffix) => {
                let tok = self.advance();
                Ok(Expr::Lit(Literal::TypedInt(n, suffix), tok.span))
            }
            TokenKind::TypedFloat(f, suffix) => {
                let tok = self.advance();
                Ok(Expr::Lit(Literal::TypedFloat(f, suffix), tok.span))
            }
            TokenKind::Str(s) => {
                let tok = self.advance();
                Ok(Expr::Lit(Literal::Str(s), tok.span))
            }
            TokenKind::True => {
                let tok = self.advance();
                Ok(Expr::Lit(Literal::Bool(true), tok.span))
            }
            TokenKind::False => {
                let tok = self.advance();
                Ok(Expr::Lit(Literal::Bool(false), tok.span))
            }
            TokenKind::Ident(name) => {
                let tok = self.advance();
                let mut expr = Expr::Var(name, tok.span);
                // Allow parenthesized call as postfix
                if *self.peek() == TokenKind::LParen {
                    self.advance();
                    let (args, accumulator, end) = self.parse_call_args()?;
                    expr = call_expr(expr, args, accumulator, tok.span.merge(end));
                }
                Ok(expr)
            }
            TokenKind::TypeIdent(name) => {
                let tok = self.advance();
                if *self.peek() == TokenKind::LBrace {
                    self.parse_record_expr(name, tok.span)
                } else {
                    Ok(Expr::Constructor(name, tok.span))
                }
            }
            TokenKind::LParen => {
                let start = self.advance().span;
                if *self.peek() == TokenKind::RParen {
                    let end = self.advance().span;
                    return Ok(Expr::Tuple(Vec::new(), start.merge(end)));
                }
                let first = self.parse_expr(0)?;
                if *self.peek() == TokenKind::Comma {
                    let mut elems = vec![first];
                    while *self.peek() == TokenKind::Comma {
                        self.advance();
                        if self.comma_terminates_list(&TokenKind::RParen, elems.len(), true)? {
                            break;
                        }
                        elems.push(self.parse_expr(0)?);
                    }
                    let end = self.expect(&TokenKind::RParen)?;
                    Ok(Expr::Tuple(elems, start.merge(end.span)))
                } else {
                    self.expect(&TokenKind::RParen)?;
                    Ok(first)
                }
            }
            TokenKind::LBracket => {
                let start = self.advance().span;
                let items = self.parse_expr_list(TokenKind::RBracket)?;
                let end = self.expect(&TokenKind::RBracket)?;
                Ok(Expr::List(items, start.merge(end.span)))
            }
            TokenKind::If => self.parse_if(),
            TokenKind::Amp => {
                let tok = self.advance();
                let operand = self.parse_expr(13)?;
                let span = tok.span.merge(expression_span(&operand));
                Ok(Expr::Borrow(Box::new(operand), span))
            }
            TokenKind::Fn => self.parse_lambda(),
            _ => Err(ParseError::Expected {
                expected: "expression atom".into(),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            }),
        }
    }

    fn parse_expr_list(&mut self, terminator: TokenKind) -> Result<Vec<Expr>, ParseError> {
        let mut exprs = Vec::new();
        if *self.peek() == terminator {
            return Ok(exprs);
        }
        exprs.push(self.parse_expr(0)?);
        while *self.peek() == TokenKind::Comma {
            self.advance();
            if self.comma_terminates_list(&terminator, exprs.len(), false)? {
                break;
            }
            exprs.push(self.parse_expr(0)?);
        }
        Ok(exprs)
    }

    /// A call's arguments after its `(`, through the closing `)`: the
    /// positional arguments, then an optional final `accumulator = <dtype>`
    /// (spec/02 `CallArgs`). Returns the closing parenthesis's span.
    fn parse_call_args(&mut self) -> Result<(Vec<Expr>, Option<String>, Span), ParseError> {
        let mut args = Vec::new();
        let mut accumulator = None;
        while *self.peek() != TokenKind::RParen {
            if matches!(self.peek(), TokenKind::Ident(name) if name == "accumulator")
                && self.peek_significant_after(1) == Some(&TokenKind::Eq)
            {
                self.advance();
                self.advance();
                let (precision, precision_span) = self.expect_ident()?;
                self.reject_retired_integer_dtype_name(&precision, precision_span)?;
                if crate::dtype_name::canonical_primitive_name(&precision).is_none()
                    && !crate::dtype_name::is_reserved_dtype_name(&precision)
                {
                    return Err(ParseError::Expected {
                        expected: "a dtype after `accumulator=`".into(),
                        found: precision,
                        offset: precision_span.offset,
                    });
                }
                self.consume_trailing_comma_before(&TokenKind::RParen);
                accumulator = Some(precision);
                break;
            }
            args.push(self.parse_expr(0)?);
            if *self.peek() != TokenKind::Comma {
                break;
            }
            self.advance();
        }
        let end = self.expect(&TokenKind::RParen)?;
        Ok((args, accumulator, end.span))
    }

    fn parse_if(&mut self) -> Result<Expr, ParseError> {
        let start = self.advance().span; // consume If
        let cond = self.parse_expr(0)?;
        self.expect(&TokenKind::Then)?;
        let then_br = self.parse_expr(0)?;
        self.expect(&TokenKind::Else)?;
        let else_br = self.parse_expr(0)?;
        let span = start.merge(expression_span(&else_br));
        Ok(Expr::If(
            Box::new(cond),
            Box::new(then_br),
            Box::new(else_br),
            span,
        ))
    }

    fn parse_match(&mut self) -> Result<Expr, ParseError> {
        let start = self.advance().span; // consume Match
        let with_pos =
            self.find_top_level_token(TokenKind::With)
                .ok_or_else(|| ParseError::Expected {
                    expected: "match ... with { | pattern => expr }".into(),
                    found: format!("{:?}", self.peek()),
                    offset: self.current_offset(),
                })?;
        let scrutinee = self.parse_expr_in_range(with_pos, "`with` after match scrutinee")?;
        if *self.peek() != TokenKind::With {
            return Err(ParseError::Expected {
                expected: "match ... with { | pattern => expr }".into(),
                found: format!(
                    "{:?}; expected `with` keyword followed by braced arms",
                    self.peek()
                ),
                offset: self.current_offset(),
            });
        }
        self.advance(); // consume With
        if *self.peek() != TokenKind::LBrace {
            return Err(ParseError::Expected {
                expected: "`{` after `match ... with`; arms must be enclosed in braces: match expr with { | pattern => body }".into(),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            });
        }
        self.advance(); // consume LBrace

        let mut arms = Vec::new();
        while *self.peek() == TokenKind::Bar {
            self.advance();
            let pattern = self.parse_pattern()?;
            let guard = if *self.peek() == TokenKind::If {
                self.advance();
                Some(self.parse_expr(0)?)
            } else {
                None
            };
            self.expect(&TokenKind::FatArrow)?;
            let body = self.parse_expr(0)?;
            let arm_span = pattern_span(&pattern).merge(expression_span(&body));
            arms.push(MatchArm {
                pattern,
                guard,
                body,
                span: arm_span,
            });
        }

        let end = self.expect(&TokenKind::RBrace)?;
        Ok(Expr::Match(
            Box::new(scrutinee),
            arms,
            start.merge(end.span),
        ))
    }

    fn parse_lambda(&mut self) -> Result<Expr, ParseError> {
        let start = self.advance().span; // consume Fn
        self.expect(&TokenKind::LParen)?;
        let params = self.parse_params()?;
        self.expect(&TokenKind::RParen)?;
        self.expect(&TokenKind::Arrow)?;
        let body = self.parse_expr(0)?;
        let span = start.merge(expression_span(&body));
        Ok(Expr::Lambda(params, Box::new(body), span))
    }

    fn parse_cast(&mut self, mode: CastMode) -> Result<Expr, ParseError> {
        let start = self.advance().span; // consume Cast / NamedCast
        self.expect(&TokenKind::LParen)?;
        let expr = self.parse_expr(0)?;
        self.expect(&TokenKind::Comma)?;
        let (precision, precision_span) = self.expect_ident()?;
        self.reject_retired_integer_dtype_name(&precision, precision_span)?;
        self.consume_trailing_comma_before(&TokenKind::RParen);
        let end = self.expect(&TokenKind::RParen)?;
        Ok(Expr::Cast(
            Box::new(expr),
            precision,
            mode,
            start.merge(end.span),
        ))
    }

    fn parse_grad(&mut self) -> Result<Expr, ParseError> {
        let start = self.advance().span; // consume Grad
        self.expect(&TokenKind::LParen)?;
        let expr = self.parse_expr(0)?;
        let wrt = if self.consume_trailing_comma_before(&TokenKind::RParen) {
            None
        } else if *self.peek() == TokenKind::Comma {
            self.advance();
            let (kw, kw_span) = self.expect_ident()?;
            if kw != "wrt" {
                return Err(ParseError::Expected {
                    expected: "`wrt`".into(),
                    found: kw,
                    offset: kw_span.offset,
                });
            }
            self.expect(&TokenKind::Eq)?;
            let targets = self.parse_grad_wrt_targets()?;
            self.consume_trailing_comma_before(&TokenKind::RParen);
            Some(targets)
        } else {
            None
        };
        let end = self.expect(&TokenKind::RParen)?;
        Ok(Expr::Grad(Box::new(expr), wrt, start.merge(end.span)))
    }

    fn parse_grad_wrt_targets(&mut self) -> Result<Vec<String>, ParseError> {
        if *self.peek() == TokenKind::LParen {
            self.advance();
            let mut names = Vec::new();
            loop {
                let (name, _) = self.expect_ident()?;
                names.push(name);
                if *self.peek() == TokenKind::Comma {
                    if self.consume_trailing_comma_before(&TokenKind::RParen) {
                        break;
                    }
                    self.advance();
                    continue;
                }
                break;
            }
            self.expect(&TokenKind::RParen)?;
            if self.mode != ParseMode::LegacyV018 && names.len() == 1 {
                return Err(ParseError::Expected {
                    expected: "bare singleton `wrt=name`".into(),
                    found: "parenthesized singleton".into(),
                    offset: self.current_offset(),
                });
            }
            Ok(names)
        } else {
            let (name, _) = self.expect_ident()?;
            Ok(vec![name])
        }
    }

    fn parse_vmap(&mut self) -> Result<Expr, ParseError> {
        let start = self.advance().span; // consume Vmap
        self.expect(&TokenKind::LParen)?;
        let expr = self.parse_expr(0)?;
        let axis = if self.consume_trailing_comma_before(&TokenKind::RParen) {
            None
        } else if *self.peek() == TokenKind::Comma {
            self.advance();
            match self.peek().clone() {
                TokenKind::Ident(kw) if kw == "axis" => {
                    self.advance();
                    self.expect(&TokenKind::Eq)?;
                    match self.peek().clone() {
                        TokenKind::Int(n) => {
                            if self.mode != ParseMode::LegacyV018 && n == 0 {
                                return Err(ParseError::Expected {
                                    expected: "`vmap(f)` for the default zero axis".into(),
                                    found: "redundant `axis=0`".into(),
                                    offset: self.current_offset(),
                                });
                            }
                            self.advance();
                            Some(n)
                        }
                        _ => {
                            return Err(ParseError::Expected {
                                expected: "integer".into(),
                                found: format!("{:?}", self.peek()),
                                offset: self.current_offset(),
                            });
                        }
                    }
                }
                TokenKind::Int(n) => {
                    if self.mode != ParseMode::LegacyV018 {
                        return Err(ParseError::Expected {
                            expected: "named axis argument `axis=<integer>`".into(),
                            found: n.to_string(),
                            offset: self.current_offset(),
                        });
                    }
                    self.advance();
                    Some(n)
                }
                _ => {
                    return Err(ParseError::Expected {
                        expected: "integer".into(),
                        found: format!("{:?}", self.peek()),
                        offset: self.current_offset(),
                    });
                }
            }
        } else {
            None
        };
        self.consume_trailing_comma_before(&TokenKind::RParen);
        let end = self.expect(&TokenKind::RParen)?;
        Ok(Expr::Vmap(Box::new(expr), axis, start.merge(end.span)))
    }

    fn parse_jit(&mut self) -> Result<Expr, ParseError> {
        let start = self.advance().span; // consume Jit
        self.expect(&TokenKind::LParen)?;
        let expr = self.parse_expr(0)?;
        self.consume_trailing_comma_before(&TokenKind::RParen);
        let end = self.expect(&TokenKind::RParen)?;
        Ok(Expr::Jit(Box::new(expr), start.merge(end.span)))
    }

    fn parse_realize(&mut self) -> Result<Expr, ParseError> {
        let start = self.advance().span;
        // Bare `realize` (H1/H2): no following `LParen` → η-expand to
        // `fn (v) -> realize(v)`. Spec §3.6 says a bare callable in
        // expression position is the function itself; this synthesis
        // mirrors `parse_pipe_stage`'s bare-form handling.
        if *self.peek() != TokenKind::LParen {
            return Ok(
                self.synthesize_bare_unary_builtin_lambda(BareUnaryBuiltinKind::Realize, start)
            );
        }
        self.expect(&TokenKind::LParen)?;
        let expr = self.parse_expr(0)?;
        self.consume_trailing_comma_before(&TokenKind::RParen);
        let end = self.expect(&TokenKind::RParen)?;
        Ok(Expr::Realize(Box::new(expr), start.merge(end.span)))
    }

    fn parse_copy(&mut self) -> Result<Expr, ParseError> {
        let start = self.advance().span;
        // Bare `copy` (H1/H2): same η-expansion as bare `realize`.
        if *self.peek() != TokenKind::LParen {
            return Ok(self.synthesize_bare_unary_builtin_lambda(BareUnaryBuiltinKind::Copy, start));
        }
        self.expect(&TokenKind::LParen)?;
        let expr = self.parse_expr(0)?;
        self.consume_trailing_comma_before(&TokenKind::RParen);
        let end = self.expect(&TokenKind::RParen)?;
        Ok(Expr::Copy(Box::new(expr), start.merge(end.span)))
    }

    fn synthesize_bare_unary_builtin_lambda(&self, kind: BareUnaryBuiltinKind, span: Span) -> Expr {
        let pipe_param = self.fresh_pipe_param_name();
        let body_inner = Expr::Var(pipe_param.clone(), span);
        let body = match kind {
            BareUnaryBuiltinKind::Realize => Expr::Realize(Box::new(body_inner), span),
            BareUnaryBuiltinKind::Copy => Expr::Copy(Box::new(body_inner), span),
        };
        Expr::Lambda(
            vec![Param {
                name: pipe_param,
                ty: None,
                span,
            }],
            Box::new(body),
            span,
        )
    }

    fn parse_with_handler(&mut self) -> Result<Expr, ParseError> {
        let start = self.advance().span; // consume With
        let handler_offset = self.current_offset();
        let (handler_name, _) = self.expect_ident()?;
        if handler_name == "seed" {
            return Err(ParseError::RetiredRandomness {
                spelling: "with seed",
                offset: handler_offset,
            });
        }
        self.expect(&TokenKind::LParen)?;
        let arg = self.parse_expr(0)?;
        self.consume_trailing_comma_before(&TokenKind::RParen);
        self.expect(&TokenKind::RParen)?;
        let body = self.parse_block_inner(true)?;
        let span = start.merge(expression_span(&body));
        match handler_name.as_str() {
            "device" => Ok(Expr::WithDevice(Box::new(arg), Box::new(body), span)),
            _ => Err(ParseError::Expected {
                expected: "`device` effect handler".into(),
                found: handler_name,
                offset: self.current_offset(),
            }),
        }
    }

    fn parse_par(&mut self) -> Result<Expr, ParseError> {
        // par { e1; e2; ... }
        let start = self.advance().span; // consume Par
        self.expect(&TokenKind::LBrace)?;
        let mut exprs = Vec::new();
        while matches!(self.raw_peek(), TokenKind::Newline) {
            self.advance_raw();
        }
        if *self.peek() == TokenKind::RBrace {
            return Err(ParseError::Expected {
                expected: "at least one expression in `par`".into(),
                found: "RBrace".into(),
                offset: self.current_offset(),
            });
        }
        // chelis#706 family: par items are Sep-bounded, so a bare
        // `par { f(x)\n g(y) }` no longer cross-newline-juxtaposes into a
        // single task; the newline-only case falls through to the loop's
        // `expected separator (`;`)` error below.
        exprs.push(self.parse_expr_until_block_separator()?);
        loop {
            while matches!(self.raw_peek(), TokenKind::Newline) {
                self.advance_raw();
            }
            if *self.peek() == TokenKind::RBrace {
                break;
            }
            if *self.raw_peek() != TokenKind::Semicolon {
                return Err(ParseError::Expected {
                    expected: "separator (`;`)".into(),
                    found: format!("{:?}", self.raw_peek()),
                    offset: self.current_offset(),
                });
            }
            self.advance_raw();
            while matches!(self.raw_peek(), TokenKind::Newline) {
                self.advance_raw();
            }
            if *self.peek() == TokenKind::RBrace {
                break;
            }
            exprs.push(self.parse_expr_until_block_separator()?);
        }
        let end = self.expect(&TokenKind::RBrace)?;
        Ok(Expr::Par(exprs, start.merge(end.span)))
    }

    fn parse_do(&mut self) -> Result<Expr, ParseError> {
        let start = self.advance().span;
        self.expect(&TokenKind::LBrace)?;
        while matches!(self.raw_peek(), TokenKind::Newline) {
            self.advance_raw();
        }
        if *self.peek() == TokenKind::RBrace {
            return Err(ParseError::Expected {
                expected: "at least one expression in `do`".into(),
                found: "RBrace".into(),
                offset: self.current_offset(),
            });
        }
        let mut exprs = vec![self.parse_expr_until_block_separator()?];
        loop {
            while matches!(self.raw_peek(), TokenKind::Newline) {
                self.advance_raw();
            }
            if *self.peek() == TokenKind::RBrace {
                break;
            }
            self.expect(&TokenKind::Semicolon)?;
            while matches!(self.raw_peek(), TokenKind::Newline) {
                self.advance_raw();
            }
            if *self.peek() == TokenKind::RBrace {
                break;
            }
            exprs.push(self.parse_expr_until_block_separator()?);
        }
        let end = self.expect(&TokenKind::RBrace)?;
        Ok(Expr::Do(exprs, start.merge(end.span)))
    }

    fn parse_meta_form(&mut self, form: DeepSurfaceMetaForm) -> Result<Expr, ParseError> {
        let start = self.advance().span;
        self.expect(&TokenKind::LParen)?;
        let value = self.parse_expr(0)?;
        self.consume_trailing_comma_before(&TokenKind::RParen);
        let end = self.expect(&TokenKind::RParen)?;
        let span = start.merge(end.span);
        Ok(match form {
            DeepSurfaceMetaForm::Quote => Expr::Quote(Box::new(value), span),
            DeepSurfaceMetaForm::Unquote => Expr::Unquote(Box::new(value), span),
            DeepSurfaceMetaForm::Splice => Expr::Splice(Box::new(value), span),
        })
    }

    fn parse_block(&mut self) -> Result<Expr, ParseError> {
        self.parse_block_inner(false)
    }

    /// Parse a brace-delimited binding block. Effect handlers provide a
    /// lexical scope of their own, so their body may contain one bare tail;
    /// an ordinary `{ expr }` remains a rejected alias.
    fn parse_block_inner(&mut self, allow_unbound_tail: bool) -> Result<Expr, ParseError> {
        let start = self.advance().span; // consume LBrace
        let mut bindings = Vec::new();
        // Separator position 1 of 4: before the first binding.
        self.consume_block_separators();
        self.reject_canonical_semicolon_separator()?;
        while !self.at_eof() && self.is_short_block_binding_start() {
            bindings.push(self.parse_block_let_binding()?);
            // Separator position 2 of 4: between bindings. This runs before
            // the `sep_count` check below, so a `;` here reports the rule
            // whether or not a newline preceded it.
            let sep_count = self.consume_block_separators();
            self.reject_canonical_semicolon_separator()?;
            if *self.peek() != TokenKind::RBrace && sep_count == 0 {
                return Err(ParseError::Expected {
                    // A `;` never reaches here: canonical mode rejected it
                    // just above, and LegacyV018 consumed it as a separator.
                    // Any other stray token is a plain missing separator, so
                    // keep the generic shape and list only what this mode's
                    // grammar actually accepts (spec/02 §P5: canonical
                    // binding blocks separate on newlines alone).
                    expected: if self.mode != ParseMode::LegacyV018 {
                        "separator (newline)".into()
                    } else {
                        "separator (`;` or newline)".into()
                    },
                    found: format!("{:?}", self.raw_peek()),
                    offset: self.current_offset(),
                });
            }
        }
        // The tail is Sep-bounded exactly like a binding value
        // (`BlockBody <- (BlockBinding Sep)* Expr`, spec/02 §BlockBody):
        // a top-level newline/`;` ends it unless the next line begins a
        // continuation (`|>`, `then`, `else`; spec/02 §P12) or the break is
        // inside ()/[]/{}. An empty tail keeps today's
        // "expected expression" shape — routing `{ x = 1 }` (RBrace here)
        // through the nested parser would report Eof and disturb the
        // pinned message, so guard it explicitly first.
        if *self.peek() == TokenKind::RBrace {
            return Err(ParseError::Expected {
                expected: "expression".into(),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            });
        }
        let expr = self.parse_expr_until_block_separator()?;
        // Separator position 3 of 4: after the tail. spec/02-surf-syntax.md
        // §P5 rejects a trailing `;` by name, but it used to arrive here and
        // be reported as a bare statement, which is false twice over for
        // `{ a = 1i64\n add(a, 1i64); }`: that expression IS the tail, and
        // there is no unbound statement. Following that advice (`_ = ...;`)
        // only moved the failure onto the `;` message anyway (chelis#1267).
        self.consume_block_separators();
        self.reject_canonical_semicolon_separator()?;
        if *self.peek() != TokenKind::RBrace {
            // A second top-level expression after the tail: bare non-tail
            // statements silently juxtaposed into an application before
            // chelis#706. `current_offset` skips newlines, so this lands
            // on the stray statement's first token.
            return Err(ParseError::BareStatementInBlock {
                offset: self.current_offset(),
            });
        }
        let end = self.expect(&TokenKind::RBrace)?;
        if self.mode != ParseMode::LegacyV018 && bindings.is_empty() && !allow_unbound_tail {
            return Err(ParseError::Expected {
                expected: "binding before the tail expression; use `do` for sequencing".into(),
                found: "one-expression block".into(),
                offset: start.offset,
            });
        }
        Ok(Expr::Block(bindings, Box::new(expr), start.merge(end.span)))
    }

    fn is_short_block_binding_start(&self) -> bool {
        let mut probe = Parser {
            tokens: self.tokens.clone(),
            closing_delimiters: self.closing_delimiters.clone(),
            pos: self.pos,
            module_allowed: false,
            mode: self.mode,
        };
        let Ok(pattern) = probe.parse_let_pattern() else {
            return false;
        };
        if matches!(pattern, LetPattern::Var(_, _)) && *probe.peek() == TokenKind::Colon {
            probe.advance();
            if probe.parse_type().is_err() {
                return false;
            }
        }
        *probe.peek() == TokenKind::Eq
    }

    fn parse_block_let_binding(&mut self) -> Result<LetBinding, ParseError> {
        let pattern = self.parse_let_pattern()?;
        let ty = if matches!(pattern, LetPattern::Var(_, _)) && *self.peek() == TokenKind::Colon {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };
        self.expect(&TokenKind::Eq)?;
        // Separator position 4 of 4: between a binding's `=` and its value.
        // `a = ; 1i64` is a legal v0.18 spelling that the migrator rewrites to
        // `a = 1i64`, so migrated-era source reaches this. Unguarded it left
        // the value's token range empty and bottomed out in an offsetless
        // "unexpected end of input" (chelis#1267).
        self.consume_block_separators();
        self.reject_canonical_semicolon_separator()?;
        let value = self.parse_expr_until_block_separator()?;
        Ok(LetBinding { pattern, ty, value })
    }

    fn parse_let_pattern(&mut self) -> Result<LetPattern, ParseError> {
        match self.peek().clone() {
            TokenKind::Ident(name) => {
                let tok = self.advance();
                Ok(LetPattern::Var(name, tok.span))
            }
            // Single-letter uppercase binds a value here (chelis#437):
            // a block binding `{ S = ... }`. A binding pattern is a
            // binding position, so the §1.1 case-split default yields to
            // the explicit value binding.
            TokenKind::TypeIdent(name) if is_single_letter_upper(&name) => {
                let tok = self.advance();
                Ok(LetPattern::Var(name, tok.span))
            }
            TokenKind::Underscore => {
                let tok = self.advance();
                Ok(LetPattern::Wildcard(tok.span))
            }
            TokenKind::LParen => {
                let start = self.advance().span;
                if *self.peek() == TokenKind::RParen {
                    let end = self.advance().span;
                    return Ok(LetPattern::Tuple(Vec::new(), start.merge(end)));
                }
                let first = self.parse_let_pattern()?;
                if *self.peek() != TokenKind::Comma {
                    return Err(ParseError::Expected {
                        expected: "tuple destructuring pattern".into(),
                        found: format!("{:?}", self.peek()),
                        offset: self.current_offset(),
                    });
                }
                let mut pats = vec![first];
                while *self.peek() == TokenKind::Comma {
                    self.advance();
                    if self.comma_terminates_list(&TokenKind::RParen, pats.len(), true)? {
                        break;
                    }
                    pats.push(self.parse_let_pattern()?);
                }
                let end = self.expect(&TokenKind::RParen)?;
                Ok(LetPattern::Tuple(pats, start.merge(end.span)))
            }
            _ => Err(ParseError::Expected {
                expected: "binding pattern".into(),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            }),
        }
    }

    fn parse_record_expr(&mut self, name: String, start: Span) -> Result<Expr, ParseError> {
        let fields = self.parse_record_fields()?;
        let end = self.tokens[self.pos - 1].span;
        Ok(Expr::Record(name, fields, start.merge(end)))
    }

    fn parse_record_fields(&mut self) -> Result<Vec<(String, Expr)>, ParseError> {
        self.expect(&TokenKind::LBrace)?;
        let mut fields = Vec::new();
        if *self.peek() != TokenKind::RBrace {
            loop {
                let (field, field_span) = self.expect_ident()?;
                let explicit = *self.peek() == TokenKind::Colon;
                let value = if explicit {
                    self.advance();
                    self.parse_expr(0)?
                } else {
                    Expr::Var(field.clone(), field_span)
                };
                if self.mode != ParseMode::LegacyV018
                    && explicit
                    && matches!(&value, Expr::Var(name, _) if name == &field)
                {
                    return Err(ParseError::Expected {
                        expected: format!("record pun `{field}`"),
                        found: format!("`{field}: {field}`"),
                        offset: field_span.offset,
                    });
                }
                fields.push((field, value));
                if *self.peek() != TokenKind::Comma {
                    break;
                }
                self.advance();
                if self.comma_terminates_list(&TokenKind::RBrace, fields.len(), false)? {
                    break;
                }
            }
        }
        self.expect(&TokenKind::RBrace)?;
        Ok(fields)
    }

    // ---------------------------------------------------------------------------
    // Type expression parsing
    // ---------------------------------------------------------------------------

    fn parse_type(&mut self) -> Result<TypeExpr, ParseError> {
        // Parse first type, then check for ->
        let mut types = vec![self.parse_type_atom()?];

        while *self.peek() == TokenKind::Arrow {
            self.advance();
            types.push(self.parse_type_atom()?);
        }

        if types.len() == 1 {
            Ok(types.pop().unwrap())
        } else {
            let start = type_span(&types[0]);
            let end = type_span(types.last().unwrap());
            let ret = types.pop().unwrap();
            Ok(TypeExpr::Arrow(types, Box::new(ret), start.merge(end)))
        }
    }

    fn parse_type_atom(&mut self) -> Result<TypeExpr, ParseError> {
        match self.peek().clone() {
            TokenKind::Int(n) => {
                // spec/02-surf-syntax.md: `IntLit` has exactly one type-position
                // production, the `DimExpr` inside a tensor shape. Everywhere
                // else an integer is not a type; accepting one here silently
                // manufactured `(t-var {} <digits>)` (chelis#1179).
                Err(ParseError::Expected {
                    expected: "a type; integer literals are only dimensions inside `tensor[...]`"
                        .into(),
                    found: format!("integer literal `{n}`"),
                    offset: self.current_offset(),
                })
            }
            TokenKind::Ident(name) => {
                let tok = self.advance();
                self.reject_retired_integer_dtype_name(&name, tok.span)?;
                Ok(TypeExpr::Named(name, tok.span))
            }
            TokenKind::TypeIdent(name) => {
                let tok = self.advance();
                // Extend the type name into a module-qualified path
                // `Demo.Dropout.Mode` (chelis#316), so a consumer that imports
                // two modules exporting the same type name can still annotate
                // against one. reef resolves the dotted head to the declaring
                // module's mangled type name, mirroring qualified constructor
                // expressions and patterns. Type names are PascalCase, so only
                // `.TypeIdent` segments extend the path.
                let mut name = name;
                let mut head_span = tok.span;
                while self.peek_dot_then_typeident() {
                    self.advance(); // consume `.`
                    let seg = self.advance(); // consume the PascalCase segment
                    // `peek_dot_then_typeident` just verified this is a
                    // `TypeIdent`; `let else` pins that invariant so a future
                    // drift fails loudly instead of silently dropping a segment.
                    let TokenKind::TypeIdent(segment) = seg.kind else {
                        unreachable!("peek_dot_then_typeident guaranteed a TypeIdent segment")
                    };
                    name.push('.');
                    name.push_str(&segment);
                    head_span = seg.span;
                }
                let tok_span = tok.span.merge(head_span);
                if *self.peek() == TokenKind::LBracket {
                    self.advance();
                    let mut args = Vec::new();
                    if *self.peek() != TokenKind::RBracket {
                        args.push(self.parse_type_arg()?);
                        while *self.peek() == TokenKind::Comma {
                            self.advance();
                            if self.comma_terminates_list(
                                &TokenKind::RBracket,
                                args.len(),
                                false,
                            )? {
                                break;
                            }
                            args.push(self.parse_type_arg()?);
                        }
                    }
                    let end = self.expect(&TokenKind::RBracket)?;
                    let span = tok_span.merge(end.span);
                    if args.is_empty() {
                        if self.mode != ParseMode::LegacyV018 {
                            return Err(ParseError::Expected {
                                expected: format!("bare unapplied type `{name}`"),
                                found: format!("`{name}[]`"),
                                offset: tok_span.offset,
                            });
                        }
                        Ok(TypeExpr::Named(name, span))
                    } else {
                        Ok(TypeExpr::App(name, args, span))
                    }
                } else {
                    Ok(TypeExpr::Named(name, tok_span))
                }
            }
            TokenKind::Star => {
                // * in type position = wildcard dimension
                let tok = self.advance();
                Ok(TypeExpr::Named("*".to_string(), tok.span))
            }
            TokenKind::DotDot => {
                // `..r` rank-variable spread — only valid as the sole shape
                // element of a tensor type (enforced in the `Tensor` arm).
                let tok = self.advance();
                let (name, name_span) = self.expect_ident()?;
                Ok(TypeExpr::RankSpread(name, tok.span.merge(name_span)))
            }
            TokenKind::Tensor => {
                let tok = self.advance();
                self.expect(&TokenKind::LBracket)?;
                // Parse dims (all but last are dims, last is precision ident)
                let mut items = Vec::new();
                items.push(self.parse_tensor_item()?);
                while *self.peek() == TokenKind::Comma {
                    self.advance();
                    if *self.peek() == TokenKind::RBracket {
                        break;
                    }
                    items.push(self.parse_tensor_item()?);
                }
                let end = self.expect(&TokenKind::RBracket)?;
                // Last item should be the precision (a Named ident)
                let precision = items.pop().unwrap();
                let (prec_name, prec_span) = match &precision {
                    TypeExpr::Named(n, span) => (n.clone(), *span),
                    _ => {
                        return Err(ParseError::Expected {
                            expected: "precision type name".into(),
                            found: format!("{precision:?}"),
                            offset: self.current_offset(),
                        });
                    }
                };
                // Rank polymorphism (spec/design/rank_polymorphism.md): a `..r`
                // spread is a name-preserving run of dims and may be interleaved
                // with concrete anchors (`tensor[..pre, seq, ..post, f32]` —
                // Tier-3). The only parse-time fence is that a spread name may
                // not repeat within one tensor shape (it would bind the same run
                // twice). Two *adjacent* spreads are allowed syntactically: they
                // occur in a reduction's output type `tensor[..pre, ..post]`. An
                // undetermined adjacent-spread *split* is rejected later at
                // unification, where an output position (sound) is distinguished
                // from an input split (non-unitary).
                let mut seen_spreads: Vec<&str> = Vec::new();
                for d in &items {
                    if let TypeExpr::RankSpread(n, _) = d {
                        if seen_spreads.contains(&n.as_str()) {
                            return Err(ParseError::Expected {
                                expected: "a distinct rank-spread name; the same `..r` may \
                                           not appear twice in one tensor shape"
                                    .into(),
                                found: format!("repeated rank spread `..{n}`"),
                                offset: self.current_offset(),
                            });
                        }
                        seen_spreads.push(n);
                    }
                }
                Ok(TypeExpr::Tensor(
                    items,
                    TensorPrecision::new(prec_name, prec_span),
                    tok.span.merge(end.span),
                ))
            }
            TokenKind::Amp => {
                let tok = self.advance();
                let inner = self.parse_type_atom()?;
                let span = tok.span.merge(type_span(&inner));
                Ok(TypeExpr::Ref(Box::new(inner), span))
            }
            TokenKind::LParen => {
                let start = self.advance().span;
                if *self.peek() == TokenKind::RParen {
                    let end = self.advance().span;
                    if self.mode != ParseMode::LegacyV018 {
                        return Err(ParseError::Expected {
                            expected: "unit type `unit`".into(),
                            found: "()".into(),
                            offset: start.offset,
                        });
                    }
                    return Ok(TypeExpr::Tuple(Vec::new(), start.merge(end)));
                }
                let first = self.parse_type()?;
                if *self.peek() == TokenKind::Comma {
                    let mut types = vec![first];
                    while *self.peek() == TokenKind::Comma {
                        self.advance();
                        if self.comma_terminates_list(&TokenKind::RParen, types.len(), true)? {
                            break;
                        }
                        types.push(self.parse_type()?);
                    }
                    let end = self.expect(&TokenKind::RParen)?;
                    Ok(TypeExpr::Tuple(types, start.merge(end.span)))
                } else {
                    self.expect(&TokenKind::RParen)?;
                    Ok(first)
                }
            }
            TokenKind::Underscore => {
                let tok = self.advance();
                Ok(TypeExpr::Infer(tok.span))
            }
            _ => Err(ParseError::Expected {
                expected: "type".into(),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            }),
        }
    }

    /// One bracketed `tensor[...]` item: a dimension or the trailing
    /// precision name. Dimension positions have an integer production
    /// (`DimExpr <- IntLit / Ident / '*' / '..' Ident`,
    /// spec/02-surf-syntax.md), so the literal-dimension arm lives here
    /// rather than in `parse_type_atom` (chelis#1179).
    fn parse_tensor_item(&mut self) -> Result<TypeExpr, ParseError> {
        if let TokenKind::Int(n) = self.peek().clone() {
            let tok = self.advance();
            return Ok(TypeExpr::Named(n.to_string(), tok.span));
        }
        if *self.peek() == TokenKind::Underscore {
            let tok = self.advance();
            return Err(ParseError::Expected {
                expected: "a tensor dimension (`*` for a dynamic extent), a named rank spread, or a precision type name".into(),
                found: "inference hole `_`".into(),
                offset: tok.span.offset,
            });
        }
        self.parse_type_atom()
    }

    /// One bracketed type-application argument (`Name[...]`). An integer
    /// literal here is a concrete dimension argument to a
    /// dimension-parameterized ADT (`Frame[2]`, the chelis#940 shape,
    /// `TypeArg <- TypeExpr / IntLit`); every other argument is an
    /// ordinary type expression. Bare-type positions reject integers in
    /// `parse_type_atom` (chelis#1179).
    fn parse_type_arg(&mut self) -> Result<TypeExpr, ParseError> {
        if let TokenKind::Int(n) = self.peek().clone() {
            let tok = self.advance();
            return Ok(TypeExpr::DimensionLiteral(
                crate::ast::DimensionLiteral::new(n),
                tok.span,
            ));
        }
        self.parse_type()
    }

    // ---------------------------------------------------------------------------
    // Pattern parsing
    // ---------------------------------------------------------------------------

    fn parse_pattern(&mut self) -> Result<Pattern, ParseError> {
        match self.peek().clone() {
            TokenKind::Underscore => {
                let tok = self.advance();
                Ok(Pattern::Wildcard(tok.span))
            }
            TokenKind::Ident(name) => {
                let tok = self.advance();
                // Check for as-pattern: x @ Pattern
                if *self.peek() == TokenKind::At {
                    self.advance(); // consume @
                    let inner = self.parse_pattern()?;
                    let end = pattern_span(&inner);
                    Ok(Pattern::As(name, Box::new(inner), tok.span.merge(end)))
                } else {
                    Ok(Pattern::Var(name, tok.span))
                }
            }
            TokenKind::Int(n) => {
                let tok = self.advance();
                Ok(Pattern::Lit(Literal::Int(n), tok.span))
            }
            TokenKind::Float(f) => {
                let tok = self.advance();
                Ok(Pattern::Lit(Literal::Float(f), tok.span))
            }
            TokenKind::TypedInt(n, suffix) => {
                let tok = self.advance();
                if self.mode != ParseMode::LegacyV018 {
                    return Err(ParseError::Expected {
                        expected: "unsuffixed numeric literal pattern".into(),
                        found: format!("numeric pattern with `{}` suffix", suffix.as_str()),
                        offset: tok.span.offset,
                    });
                }
                Ok(Pattern::Lit(Literal::Int(n), tok.span))
            }
            TokenKind::TypedFloat(f, suffix) => {
                let tok = self.advance();
                if self.mode != ParseMode::LegacyV018 {
                    return Err(ParseError::Expected {
                        expected: "unsuffixed numeric literal pattern".into(),
                        found: format!("numeric pattern with `{}` suffix", suffix.as_str()),
                        offset: tok.span.offset,
                    });
                }
                Ok(Pattern::Lit(Literal::Float(f), tok.span))
            }
            TokenKind::Minus => self.parse_negative_pattern_literal(),
            TokenKind::Str(s) => {
                let tok = self.advance();
                Ok(Pattern::Lit(Literal::Str(s), tok.span))
            }
            TokenKind::True => {
                let tok = self.advance();
                Ok(Pattern::Lit(Literal::Bool(true), tok.span))
            }
            TokenKind::False => {
                let tok = self.advance();
                Ok(Pattern::Lit(Literal::Bool(false), tok.span))
            }
            TokenKind::TypeIdent(name) => {
                let tok = self.advance();
                // Extend the constructor head into a module-qualified path:
                // `| Demo.Dropout.Train =>` (chelis#316). The dotted head is
                // carried on the pattern's constructor name; reef resolves it
                // to the declaring module's mangled constructor, mirroring how
                // qualified constructor *expressions* resolve. Constructors are
                // PascalCase, so only `.TypeIdent` segments extend the path.
                let mut name = name;
                let mut head_span = tok.span;
                while self.peek_dot_then_typeident() {
                    self.advance(); // consume `.`
                    let seg = self.advance(); // consume the PascalCase segment
                    // `peek_dot_then_typeident` just verified this is a
                    // `TypeIdent`; `let else` pins that invariant so a future
                    // drift fails loudly instead of silently dropping a segment.
                    let TokenKind::TypeIdent(segment) = seg.kind else {
                        unreachable!("peek_dot_then_typeident guaranteed a TypeIdent segment")
                    };
                    name.push('.');
                    name.push_str(&segment);
                    head_span = seg.span;
                }
                let tok_span = tok.span.merge(head_span);
                // Check for record pattern: Ctor { field1, field2 }
                if *self.peek() == TokenKind::LBrace {
                    self.advance(); // consume {
                    let mut fields = Vec::new();
                    while *self.peek() != TokenKind::RBrace {
                        let (field_name, field_span) = match self.peek().clone() {
                            TokenKind::Ident(n) => {
                                let t = self.advance();
                                (n, t.span)
                            }
                            _ => {
                                return Err(ParseError::Expected {
                                    expected: "field name".into(),
                                    found: format!("{:?}", self.peek()),
                                    offset: self.current_offset(),
                                });
                            }
                        };
                        let explicit = *self.peek() == TokenKind::Colon;
                        let field_pat = if explicit {
                            self.advance();
                            self.parse_pattern()?
                        } else {
                            Pattern::Var(field_name.clone(), field_span)
                        };
                        if self.mode != ParseMode::LegacyV018
                            && explicit
                            && matches!(&field_pat, Pattern::Var(name, _) if name == &field_name)
                        {
                            return Err(ParseError::Expected {
                                expected: format!("record-pattern pun `{field_name}`"),
                                found: format!("`{field_name}: {field_name}`"),
                                offset: field_span.offset,
                            });
                        }
                        fields.push((field_name, field_pat));
                        if *self.peek() == TokenKind::Comma {
                            self.advance();
                            if self.comma_terminates_list(
                                &TokenKind::RBrace,
                                fields.len(),
                                false,
                            )? {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                    let end = self.expect(&TokenKind::RBrace)?;
                    Ok(Pattern::Record(name, fields, tok_span.merge(end.span)))
                } else if *self.peek() == TokenKind::LParen {
                    self.advance();
                    let mut sub_pats = Vec::new();
                    if *self.peek() != TokenKind::RParen {
                        sub_pats.push(self.parse_pattern()?);
                        while *self.peek() == TokenKind::Comma {
                            self.advance();
                            if self.comma_terminates_list(
                                &TokenKind::RParen,
                                sub_pats.len(),
                                false,
                            )? {
                                break;
                            }
                            sub_pats.push(self.parse_pattern()?);
                        }
                    }
                    let end = self.expect(&TokenKind::RParen)?;
                    if self.mode != ParseMode::LegacyV018 && sub_pats.is_empty() {
                        return Err(ParseError::Expected {
                            expected: format!("bare zero-argument constructor `{name}`"),
                            found: format!("`{name}()`"),
                            offset: tok_span.offset,
                        });
                    }
                    Ok(Pattern::Constructor(
                        name,
                        sub_pats,
                        tok_span.merge(end.span),
                    ))
                } else {
                    if self.mode != ParseMode::LegacyV018 {
                        return Ok(Pattern::Constructor(name, vec![], tok_span));
                    }
                    let mut sub_pats = Vec::new();
                    while self.is_pattern_arg_start() {
                        sub_pats.push(self.parse_pattern_atom()?);
                    }
                    if sub_pats.is_empty() {
                        Ok(Pattern::Constructor(name, vec![], tok_span))
                    } else {
                        let end = pattern_span(sub_pats.last().unwrap());
                        Ok(Pattern::Constructor(name, sub_pats, tok_span.merge(end)))
                    }
                }
            }
            TokenKind::LParen => {
                let start = self.advance().span;
                if *self.peek() == TokenKind::RParen {
                    let end = self.advance().span;
                    return Ok(Pattern::Tuple(Vec::new(), start.merge(end)));
                }
                let first = self.parse_pattern()?;
                if *self.peek() == TokenKind::Comma {
                    let mut pats = vec![first];
                    while *self.peek() == TokenKind::Comma {
                        self.advance();
                        if self.comma_terminates_list(&TokenKind::RParen, pats.len(), true)? {
                            break;
                        }
                        pats.push(self.parse_pattern()?);
                    }
                    let end = self.expect(&TokenKind::RParen)?;
                    Ok(Pattern::Tuple(pats, start.merge(end.span)))
                } else {
                    self.expect(&TokenKind::RParen)?;
                    Ok(first)
                }
            }
            _ => Err(ParseError::Expected {
                expected: "pattern".into(),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            }),
        }
    }

    /// Parse a single atomic pattern (no constructor application).
    fn parse_pattern_atom(&mut self) -> Result<Pattern, ParseError> {
        match self.peek().clone() {
            TokenKind::Underscore => {
                let tok = self.advance();
                Ok(Pattern::Wildcard(tok.span))
            }
            TokenKind::Ident(name) => {
                let tok = self.advance();
                Ok(Pattern::Var(name, tok.span))
            }
            TokenKind::Int(n) => {
                let tok = self.advance();
                Ok(Pattern::Lit(Literal::Int(n), tok.span))
            }
            TokenKind::Float(f) => {
                let tok = self.advance();
                Ok(Pattern::Lit(Literal::Float(f), tok.span))
            }
            TokenKind::TypedInt(n, suffix) => {
                let tok = self.advance();
                if self.mode != ParseMode::LegacyV018 {
                    return Err(ParseError::Expected {
                        expected: "unsuffixed numeric literal pattern".into(),
                        found: format!("numeric pattern with `{}` suffix", suffix.as_str()),
                        offset: tok.span.offset,
                    });
                }
                Ok(Pattern::Lit(Literal::Int(n), tok.span))
            }
            TokenKind::TypedFloat(f, suffix) => {
                let tok = self.advance();
                if self.mode != ParseMode::LegacyV018 {
                    return Err(ParseError::Expected {
                        expected: "unsuffixed numeric literal pattern".into(),
                        found: format!("numeric pattern with `{}` suffix", suffix.as_str()),
                        offset: tok.span.offset,
                    });
                }
                Ok(Pattern::Lit(Literal::Float(f), tok.span))
            }
            TokenKind::Minus => self.parse_negative_pattern_literal(),
            TokenKind::Str(s) => {
                let tok = self.advance();
                Ok(Pattern::Lit(Literal::Str(s), tok.span))
            }
            TokenKind::True => {
                let tok = self.advance();
                Ok(Pattern::Lit(Literal::Bool(true), tok.span))
            }
            TokenKind::False => {
                let tok = self.advance();
                Ok(Pattern::Lit(Literal::Bool(false), tok.span))
            }
            TokenKind::LParen => {
                let start = self.advance().span;
                if *self.peek() == TokenKind::RParen {
                    let end = self.advance().span;
                    return Ok(Pattern::Tuple(Vec::new(), start.merge(end)));
                }
                let first = self.parse_pattern()?;
                if *self.peek() == TokenKind::Comma {
                    let mut pats = vec![first];
                    while *self.peek() == TokenKind::Comma {
                        self.advance();
                        if self.comma_terminates_list(&TokenKind::RParen, pats.len(), true)? {
                            break;
                        }
                        pats.push(self.parse_pattern()?);
                    }
                    let end = self.expect(&TokenKind::RParen)?;
                    Ok(Pattern::Tuple(pats, start.merge(end.span)))
                } else {
                    self.expect(&TokenKind::RParen)?;
                    Ok(first)
                }
            }
            _ => Err(ParseError::Expected {
                expected: "pattern".into(),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            }),
        }
    }

    fn parse_negative_pattern_literal(&mut self) -> Result<Pattern, ParseError> {
        let minus = self.advance().span;
        let token = self.advance();
        let literal = match token.kind {
            TokenKind::Int(0) if self.mode != ParseMode::LegacyV018 => {
                return Err(ParseError::Expected {
                    expected: "integer zero pattern `0`".into(),
                    found: "negative integer zero pattern `-0`".into(),
                    offset: minus.offset,
                });
            }
            TokenKind::Int(value) => Literal::Int(-value),
            TokenKind::IntMinMagnitude(None) => Literal::Int(i64::MIN),
            TokenKind::IntMinMagnitude(Some(suffix)) => {
                if self.mode != ParseMode::LegacyV018 {
                    return Err(ParseError::Expected {
                        expected: "unsuffixed numeric literal pattern".into(),
                        found: format!("numeric pattern with `{}` suffix", suffix.as_str()),
                        offset: token.span.offset,
                    });
                }
                Literal::Int(i64::MIN)
            }
            TokenKind::Float(value) => Literal::Float(-value),
            TokenKind::TypedInt(value, suffix) => {
                if self.mode != ParseMode::LegacyV018 {
                    return Err(ParseError::Expected {
                        expected: "unsuffixed numeric literal pattern".into(),
                        found: format!("numeric pattern with `{}` suffix", suffix.as_str()),
                        offset: token.span.offset,
                    });
                }
                Literal::Int(-value)
            }
            TokenKind::TypedFloat(value, suffix) => {
                if self.mode != ParseMode::LegacyV018 {
                    return Err(ParseError::Expected {
                        expected: "unsuffixed numeric literal pattern".into(),
                        found: format!("numeric pattern with `{}` suffix", suffix.as_str()),
                        offset: token.span.offset,
                    });
                }
                Literal::Float(-value)
            }
            found => {
                return Err(ParseError::Expected {
                    expected: "numeric literal after `-` in a pattern".into(),
                    found: format!("{found:?}"),
                    offset: token.span.offset,
                });
            }
        };
        Ok(Pattern::Lit(literal, minus.merge(token.span)))
    }

    fn is_pattern_arg_start(&self) -> bool {
        matches!(
            self.peek(),
            TokenKind::Underscore
                | TokenKind::Ident(_)
                | TokenKind::Int(_)
                | TokenKind::Float(_)
                | TokenKind::TypedInt(_, _)
                | TokenKind::TypedFloat(_, _)
                | TokenKind::Minus
                | TokenKind::Str(_)
                | TokenKind::True
                | TokenKind::False
                | TokenKind::LParen
        )
    }

    fn parse_optional_effects(&mut self) -> Result<Option<Vec<EffectExpr>>, ParseError> {
        if *self.peek() != TokenKind::Bang {
            return Ok(None);
        }
        self.advance();
        self.expect(&TokenKind::LBrace)?;
        let mut effects = Vec::new();
        if *self.peek() != TokenKind::RBrace {
            effects.push(self.parse_effect_expr()?);
            while *self.peek() == TokenKind::Comma {
                self.advance();
                if self.comma_terminates_list(&TokenKind::RBrace, effects.len(), false)? {
                    break;
                }
                effects.push(self.parse_effect_expr()?);
            }
        }
        self.expect(&TokenKind::RBrace)?;
        Ok(Some(effects))
    }

    fn reject_retired_integer_dtype_name(&self, name: &str, span: Span) -> Result<(), ParseError> {
        if self.mode != ParseMode::LegacyV018
            && let Some(canonical) = crate::desugar::migrated_integer_dtype_name(name)
        {
            return Err(ParseError::Expected {
                expected: format!(
                    "canonical integer dtype `{canonical}`; run \
                     `chelis migrate surf --from 0.18` to rewrite v0.18 source"
                ),
                found: format!("retired v0.18 integer dtype `{name}`"),
                offset: span.offset,
            });
        }
        Ok(())
    }

    fn parse_effect_expr(&mut self) -> Result<EffectExpr, ParseError> {
        match self.peek().clone() {
            TokenKind::Ident(name) => {
                let tok = self.advance();
                match name.as_str() {
                    "diff" if self.mode == ParseMode::LegacyV018 => Ok(EffectExpr::Diff(tok.span)),
                    "random" if self.mode == ParseMode::LegacyV018 => {
                        Err(ParseError::RetiredRandomness {
                            spelling: "random",
                            offset: tok.span.offset,
                        })
                    }
                    "accum" if self.mode == ParseMode::LegacyV018 => {
                        Ok(EffectExpr::Accum(tok.span))
                    }
                    "io" if self.mode == ParseMode::LegacyV018 => Ok(EffectExpr::Io(tok.span)),
                    "test" if self.mode == ParseMode::LegacyV018 => Ok(EffectExpr::Test(tok.span)),
                    _ => Err(ParseError::Expected {
                        expected: "effect name".into(),
                        found: name,
                        offset: tok.span.offset,
                    }),
                }
            }
            TokenKind::TypeIdent(name) => {
                let tok = self.advance();
                match name.as_str() {
                    "Diff" => Ok(EffectExpr::Diff(tok.span)),
                    "Random" => Err(ParseError::RetiredRandomness {
                        spelling: "Random",
                        offset: tok.span.offset,
                    }),
                    "Accum" => Ok(EffectExpr::Accum(tok.span)),
                    "IO" => Ok(EffectExpr::Io(tok.span)),
                    "Test" => Ok(EffectExpr::Test(tok.span)),
                    "Resource" => {
                        self.expect(&TokenKind::LParen)?;
                        let device = match self.peek().clone() {
                            TokenKind::Str(device) => {
                                self.advance();
                                device
                            }
                            _ => {
                                return Err(ParseError::Expected {
                                    expected: "device string".into(),
                                    found: format!("{:?}", self.peek()),
                                    offset: self.current_offset(),
                                });
                            }
                        };
                        self.consume_trailing_comma_before(&TokenKind::RParen);
                        let end = self.expect(&TokenKind::RParen)?;
                        Ok(EffectExpr::Resource(device, tok.span.merge(end.span)))
                    }
                    _ => Err(ParseError::Expected {
                        expected: "effect name".into(),
                        found: name,
                        offset: tok.span.offset,
                    }),
                }
            }
            _ => Err(ParseError::Expected {
                expected: "effect name".into(),
                found: format!("{:?}", self.peek()),
                offset: self.current_offset(),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Span helpers
// ---------------------------------------------------------------------------

/// True for a single ASCII-uppercase letter (`S`, `K`, `T`, `N`, `P`,
/// …). The value-binding case-split override (chelis#437) is limited to
/// single-letter names so multi-letter PascalCase stays unambiguously a
/// type or constructor name (§1.1, §3.1).
fn extend_legacy_application(expr: Expr, args: Vec<Expr>, span: Span) -> Expr {
    match expr {
        Expr::Apply(function, mut existing, _) => {
            existing.extend(args);
            Expr::Apply(function, existing, span)
        }
        other => Expr::Apply(Box::new(other), args, span),
    }
}

fn is_single_letter_upper(name: &str) -> bool {
    let mut chars = name.chars();
    matches!((chars.next(), chars.next()), (Some(c), None) if c.is_ascii_uppercase())
}

/// A call of `callee`, wrapped in [`Expr::Accumulate`] when it names an
/// explicit accumulator dtype.
fn call_expr(callee: Expr, args: Vec<Expr>, accumulator: Option<String>, span: Span) -> Expr {
    let call = Expr::Apply(Box::new(callee), args, span);
    match accumulator {
        Some(precision) => Expr::Accumulate(Box::new(call), precision, span),
        None => call,
    }
}

pub(crate) fn expression_span(e: &Expr) -> Span {
    match e {
        Expr::Lit(_, s) => *s,
        Expr::Var(_, s) => *s,
        Expr::Constructor(_, s) => *s,
        Expr::Apply(_, _, s) => *s,
        Expr::Accumulate(_, _, s) => *s,
        Expr::List(_, s) => *s,
        Expr::Record(_, _, s) => *s,
        Expr::RecordUpdate(_, _, s) => *s,
        Expr::Access(_, _, s) => *s,
        Expr::TupleGet(_, _, s) => *s,
        Expr::Binary(_, _, _, s) => *s,
        Expr::Unary(_, _, s) => *s,
        Expr::Pipe(_, _, s) => *s,
        Expr::If(_, _, _, s) => *s,
        Expr::Match(_, _, s) => *s,
        Expr::Lambda(_, _, s) => *s,
        Expr::Tuple(_, s) => *s,
        Expr::Cast(_, _, _, s) => *s,
        Expr::Grad(_, _, s) => *s,
        Expr::Vmap(_, _, s) => *s,
        Expr::Jit(_, s) => *s,
        Expr::Realize(_, s) => *s,
        Expr::Copy(_, s) => *s,
        Expr::Borrow(_, s) => *s,
        Expr::WithDevice(_, _, s) => *s,
        Expr::Par(_, s) => *s,
        Expr::Do(_, s) => *s,
        Expr::Quote(_, s) => *s,
        Expr::Unquote(_, s) => *s,
        Expr::Splice(_, s) => *s,
        Expr::Annotate(_, _, s) => *s,
        Expr::Block(_, _, s) => *s,
    }
}

fn type_span(t: &TypeExpr) -> Span {
    match t {
        TypeExpr::Named(_, s) => *s,
        TypeExpr::DimensionLiteral(_, s) => *s,
        TypeExpr::RankSpread(_, s) => *s,
        TypeExpr::Tensor(_, _, s) => *s,
        TypeExpr::Arrow(_, _, s) => *s,
        TypeExpr::Ref(_, s) => *s,
        TypeExpr::App(_, _, s) => *s,
        TypeExpr::Tuple(_, s) => *s,
        TypeExpr::Infer(s) => *s,
    }
}

fn pattern_span(p: &Pattern) -> Span {
    match p {
        Pattern::Wildcard(s) => *s,
        Pattern::Var(_, s) => *s,
        Pattern::Lit(_, s) => *s,
        Pattern::Constructor(_, _, s) => *s,
        Pattern::Tuple(_, s) => *s,
        Pattern::Record(_, _, s) => *s,
        Pattern::As(_, _, s) => *s,
    }
}

fn decl_span(d: &Decl) -> Span {
    match d {
        Decl::Module { span, .. } => *span,
        Decl::Import { span, .. } => *span,
        Decl::Sig { span, .. } => *span,
        Decl::Dim { span, .. } => *span,
        Decl::TypeDef { span, .. } => *span,
        Decl::TypeAlias { span, .. } => *span,
        Decl::FunDef { span, .. } => *span,
        Decl::Property { span, .. } => *span,
        Decl::LetDef { span, .. } => *span,
        Decl::MacroDef { span, .. } => *span,
        Decl::Export { span, .. } => *span,
    }
}

fn op_class(op: BinOp) -> u8 {
    match op {
        BinOp::Eq | BinOp::Ne => 1,
        BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => 2,
        _ => 0, // associative ops don't need class checks
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Vec<Decl> {
        parse_str_legacy_v018(s).unwrap()
    }

    fn p_err(s: &str) -> ParseError {
        parse_str_legacy_v018(s).unwrap_err()
    }

    // Convenience: parse a single def, extract body expression.
    fn body(s: &str) -> Expr {
        let decls = p(s);
        match decls.into_iter().next().unwrap() {
            Decl::FunDef { body, .. } => body,
            Decl::LetDef { value, .. } => value,
            other => panic!("expected FunDef or LetDef, got {other:?}"),
        }
    }

    // ===== Declaration tests =====

    #[test]
    fn simple_fun_def() {
        let decls = p("def f(x: f32): f32 = x");
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Decl::FunDef {
                name,
                params,
                ret_ty,
                ..
            } => {
                assert_eq!(name, "f");
                assert_eq!(params.len(), 1);
                assert_eq!(params[0].name, "x");
                assert!(params[0].ty.is_some());
                assert!(ret_ty.is_some());
            }
            _ => panic!("expected FunDef"),
        }
    }

    #[test]
    fn fun_def_no_annotations() {
        let decls = p("def f(x) = x");
        match &decls[0] {
            Decl::FunDef { params, ret_ty, .. } => {
                assert_eq!(params[0].name, "x");
                assert!(params[0].ty.is_none());
                assert!(ret_ty.is_none());
            }
            _ => panic!("expected FunDef"),
        }
    }

    #[test]
    fn fun_def_multi_param() {
        let decls = p("def f(x: f32, y: f32): f32 = x");
        match &decls[0] {
            Decl::FunDef { params, .. } => {
                assert_eq!(params.len(), 2);
                assert_eq!(params[0].name, "x");
                assert_eq!(params[1].name, "y");
            }
            _ => panic!("expected FunDef"),
        }
    }

    #[test]
    fn sig_effect_annotation() {
        let decls = p("sig f: f32 -> f32 ! {Diff, Resource(\"gpu:0\")}");
        match &decls[0] {
            Decl::Sig { effects, .. } => {
                let effects = effects.as_ref().expect("effects");
                assert_eq!(effects.len(), 2);
                assert!(matches!(effects[0], EffectExpr::Diff(_)));
                assert!(
                    matches!(effects[1], EffectExpr::Resource(ref device, _) if device == "gpu:0")
                );
            }
            _ => panic!("expected Sig"),
        }
    }

    #[test]
    fn random_effect_name_is_a_typed_rejection() {
        // The counter stream's `Random` effect was retired with the explicit
        // key switch (#2413): naming it is a typed parse error that points at
        // keys, never a silently accepted or unknown-effect crash.
        let err = p_err("sig f: f32 -> f32 ! {Diff, Random}");
        assert!(
            matches!(
                err,
                ParseError::RetiredRandomness {
                    spelling: "Random",
                    ..
                }
            ),
            "got {err:?}"
        );
        assert!(err.to_string().contains("key_from_seed"), "got {err}");
    }

    #[test]
    fn sig_io_effect_annotation() {
        let decls = p("sig f: string -> unit ! {IO}");
        match &decls[0] {
            Decl::Sig { effects, .. } => {
                let effects = effects.as_ref().expect("effects");
                assert_eq!(effects.len(), 1);
                assert!(matches!(effects[0], EffectExpr::Io(_)));
            }
            _ => panic!("expected Sig"),
        }
    }

    #[test]
    fn property_contract_option_requires_string_literal() {
        let decls = p(r#"@property reflected forall(x: f32):
  x == x
  with contract = "std.normal_cdf.reflection"
"#);
        match &decls[0] {
            Decl::Property { options, .. } => {
                assert_eq!(options.len(), 1);
                match &options[0] {
                    PropertyOption::Contract(id, _) => {
                        assert_eq!(id, "std.normal_cdf.reflection");
                    }
                    other => panic!("expected contract option, got {other:?}"),
                }
            }
            other => panic!("expected property, got {other:?}"),
        }

        let err = p_err(
            "@property bad forall(x: f32):\n  x == x\n  with contract = std.normal_cdf.reflection\n",
        );
        assert!(
            err.to_string().contains("string literal contract id"),
            "wrong error: {err}"
        );
    }

    #[test]
    fn fun_def_effect_annotation() {
        let decls = p("def f(x: f32): f32 ! {Diff} = x");
        match &decls[0] {
            Decl::FunDef { effects, .. } => {
                let effects = effects.as_ref().expect("effects");
                assert_eq!(effects.len(), 1);
                assert!(matches!(effects[0], EffectExpr::Diff(_)));
            }
            _ => panic!("expected FunDef"),
        }
    }

    #[test]
    fn top_level_binding_decl() {
        let decls = p("x = 42");
        match &decls[0] {
            Decl::LetDef { name, ty, .. } => {
                assert_eq!(name, "x");
                assert!(ty.is_none());
            }
            _ => panic!("expected LetDef"),
        }
    }

    #[test]
    fn typed_binding_decl() {
        let decls = p("x: f32 = 42.0");
        match &decls[0] {
            Decl::LetDef { name, ty, .. } => {
                assert_eq!(name, "x");
                assert!(ty.is_some());
            }
            _ => panic!("expected LetDef"),
        }
    }

    #[test]
    fn macro_def() {
        let decls = p("macro relu_ref(x) = max_elem(x, 0.0)");
        match &decls[0] {
            Decl::MacroDef { name, params, .. } => {
                assert_eq!(name, "relu_ref");
                assert_eq!(params, &["x"]);
            }
            _ => panic!("expected MacroDef"),
        }
    }

    #[test]
    fn adt_type_def() {
        let decls = p("type Option[a] = | Some(a) | None");
        match &decls[0] {
            Decl::TypeDef {
                name,
                params,
                variants,
                ..
            } => {
                assert_eq!(name, "Option");
                assert_eq!(params, &["a"]);
                assert_eq!(variants.len(), 2);
                assert_eq!(variants[0].name, "Some");
                assert_eq!(variants[1].name, "None");
            }
            _ => panic!("expected TypeDef"),
        }
    }

    #[test]
    fn type_def_no_params() {
        let decls = p("type Shape = | Scalar | Vector(i64)");
        match &decls[0] {
            Decl::TypeDef {
                name,
                params,
                variants,
                ..
            } => {
                assert_eq!(name, "Shape");
                assert!(params.is_empty());
                assert_eq!(variants.len(), 2);
                assert_eq!(variants[0].name, "Scalar");
                assert_eq!(variants[1].name, "Vector");
                match &variants[1].fields {
                    VariantFields::Positional(fields) => {
                        assert_eq!(fields.len(), 1);
                    }
                    _ => panic!("expected positional"),
                }
            }
            _ => panic!("expected TypeDef"),
        }
    }

    #[test]
    fn record_variant() {
        let decls = p("type Config = | Default { lr: f32, eps: f32 }");
        match &decls[0] {
            Decl::TypeDef { variants, .. } => {
                assert_eq!(variants[0].name, "Default");
                match &variants[0].fields {
                    VariantFields::Record(fields) => {
                        assert_eq!(fields.len(), 2);
                        assert_eq!(fields[0].0, "lr");
                        assert_eq!(fields[1].0, "eps");
                    }
                    _ => panic!("expected record"),
                }
            }
            _ => panic!("expected TypeDef"),
        }
    }

    #[test]
    fn import_simple() {
        let decls = p("import Foo");
        match &decls[0] {
            Decl::Import { module, kind, .. } => {
                assert_eq!(module, "Foo");
                assert_eq!(kind, &ImportKind::Qualified);
            }
            _ => panic!("expected Import"),
        }
    }

    #[test]
    fn import_with_names() {
        let decls = p("import Foo(bar, baz)");
        match &decls[0] {
            Decl::Import { module, kind, .. } => {
                assert_eq!(module, "Foo");
                assert_eq!(
                    kind,
                    &ImportKind::Names(vec!["bar".to_string(), "baz".to_string()])
                );
            }
            _ => panic!("expected Import"),
        }
    }

    #[test]
    fn import_all_uses_dotdot_token() {
        // Regression lock: `..` now lexes as a single DotDot token (the
        // rank-spread marker), and `import Foo(..)` import-all must still parse.
        for src in ["import Foo(..)", "import Foo.Bar(..)"] {
            let decls = p(src);
            match &decls[0] {
                Decl::Import { kind, .. } => assert_eq!(kind, &ImportKind::All, "for `{src}`"),
                _ => panic!("expected Import for `{src}`"),
            }
        }
    }

    #[test]
    fn module_decl() {
        let decls = p("module M def f(x) = x");
        match &decls[0] {
            Decl::Module { name, decls, .. } => {
                assert_eq!(name, "M");
                assert_eq!(decls.len(), 1);
            }
            _ => panic!("expected Module"),
        }
    }

    // ===== Expression precedence tests =====

    #[test]
    fn add_mul_precedence() {
        let e = body("x = a + b * c");
        match e {
            Expr::Binary(BinOp::Add, lhs, rhs, _) => {
                assert!(matches!(*lhs, Expr::Var(ref n, _) if n == "a"));
                assert!(matches!(*rhs, Expr::Binary(BinOp::Mul, _, _, _)));
            }
            _ => panic!("expected Add(a, Mul(b, c)), got {e:?}"),
        }
    }

    #[test]
    fn mul_add_precedence() {
        let e = body("x = a * b + c");
        match e {
            Expr::Binary(BinOp::Add, lhs, rhs, _) => {
                assert!(matches!(*lhs, Expr::Binary(BinOp::Mul, _, _, _)));
                assert!(matches!(*rhs, Expr::Var(ref n, _) if n == "c"));
            }
            _ => panic!("expected Add(Mul(a, b), c), got {e:?}"),
        }
    }

    #[test]
    fn left_assoc_add() {
        let e = body("x = a + b + c");
        match e {
            Expr::Binary(BinOp::Add, lhs, rhs, _) => {
                assert!(matches!(*lhs, Expr::Binary(BinOp::Add, _, _, _)));
                assert!(matches!(*rhs, Expr::Var(ref n, _) if n == "c"));
            }
            _ => panic!("expected Add(Add(a, b), c), got {e:?}"),
        }
    }

    #[test]
    fn unary_neg_plus() {
        let e = body("x = -a + b");
        match e {
            Expr::Binary(BinOp::Add, lhs, _, _) => {
                assert!(matches!(*lhs, Expr::Unary(UnaryOp::Neg, _, _)));
            }
            _ => panic!("expected Add(Neg(a), b), got {e:?}"),
        }
    }

    #[test]
    fn non_assoc_eq_chain_error() {
        let err = p_err("x = a == b == c");
        assert!(matches!(err, ParseError::NonAssocChain { .. }));
    }

    #[test]
    fn parens_override() {
        let e = body("x = (a + b) * c");
        match e {
            Expr::Binary(BinOp::Mul, lhs, _, _) => {
                assert!(matches!(*lhs, Expr::Binary(BinOp::Add, _, _, _)));
            }
            _ => panic!("expected Mul(Add(a, b), c), got {e:?}"),
        }
    }

    // ===== Pipe tests =====

    #[test]
    fn pipe_chain() {
        let e = body("x = x |> f |> g");
        match e {
            Expr::Pipe(_, stages, _) => {
                assert_eq!(stages.len(), 2);
            }
            _ => panic!("expected Pipe, got {e:?}"),
        }
    }

    // ===== Application tests =====

    #[test]
    fn paren_apply() {
        let e = body("x = f(x, y)");
        match e {
            Expr::Apply(func, args, _) => {
                assert!(matches!(*func, Expr::Var(ref n, _) if n == "f"));
                assert_eq!(args.len(), 2);
            }
            _ => panic!("expected Apply, got {e:?}"),
        }
    }

    // ===== Type annotation =====

    #[test]
    fn type_annotation_expr() {
        let e = body("x = y : f32");
        match e {
            Expr::Annotate(inner, ty, _) => {
                assert!(matches!(*inner, Expr::Var(ref n, _) if n == "y"));
                assert!(matches!(ty, TypeExpr::Named(ref n, _) if n == "f32"));
            }
            _ => panic!("expected Annotate, got {e:?}"),
        }
    }

    // ===== Control flow =====

    #[test]
    fn if_then_else() {
        let e = body("x = if a then b else c");
        assert!(matches!(e, Expr::If(_, _, _, _)));
    }

    #[test]
    fn match_expr() {
        let e = body("x = match x with { | Some y => y | None => 0 }");
        match e {
            Expr::Match(_, arms, _) => {
                assert_eq!(arms.len(), 2);
                assert!(
                    matches!(&arms[0].pattern, Pattern::Constructor(n, pats, _) if n == "Some" && pats.len() == 1)
                );
                assert!(
                    matches!(&arms[1].pattern, Pattern::Constructor(n, pats, _) if n == "None" && pats.is_empty())
                );
            }
            _ => panic!("expected Match, got {e:?}"),
        }
    }

    #[test]
    fn let_in_expr_is_rejected() {
        let err = p_err("def f(x) = let y = 1 in y + 1");
        assert!(matches!(err, ParseError::Expected { .. }));
    }

    #[test]
    fn let_in_tuple_destructuring_is_rejected() {
        let err = p_err("def f(x) = let (a, b) = pair in a");
        assert!(matches!(err, ParseError::Expected { .. }));
    }

    #[test]
    fn lambda_expr() {
        let e = body("x = fn (x, y) -> x + y");
        match e {
            Expr::Lambda(params, body, _) => {
                assert_eq!(params.len(), 2);
                assert!(matches!(*body, Expr::Binary(BinOp::Add, _, _, _)));
            }
            _ => panic!("expected Lambda, got {e:?}"),
        }
    }

    #[test]
    fn block_accepts_newline_separators() {
        let e = body(
            "def f() = {
                x = 1
                y = 2
                y
            }",
        );
        match e {
            Expr::Block(bindings, body, _) => {
                assert_eq!(bindings.len(), 2);
                assert!(matches!(*body, Expr::Var(ref n, _) if n == "y"));
            }
            _ => panic!("expected Block, got {e:?}"),
        }
    }

    #[test]
    fn block_accepts_short_bindings() {
        let e = body(
            "def f() = {
                x = 1
                y = 2
                y
            }",
        );
        match e {
            Expr::Block(bindings, body, _) => {
                assert_eq!(bindings.len(), 2);
                assert!(matches!(*body, Expr::Var(ref n, _) if n == "y"));
            }
            _ => panic!("expected Block, got {e:?}"),
        }
    }

    #[test]
    fn block_accepts_typed_short_binding() {
        let e = body(
            "def f() = {
                x: tensor[n, f32] = relu(y)
                x
            }",
        );
        match e {
            Expr::Block(bindings, body, _) => {
                assert_eq!(bindings.len(), 1);
                assert!(bindings[0].ty.is_some());
                assert!(matches!(*body, Expr::Var(ref n, _) if n == "x"));
            }
            _ => panic!("expected Block, got {e:?}"),
        }
    }

    #[test]
    fn block_accepts_binding_value_broken_after_equals() {
        let e = body(
            "def f(logits, labels) = {
                loss =
                  softmax(logits, 0)
                  |> log
                  |> mul(labels)
                loss
            }",
        );
        match e {
            Expr::Block(bindings, body, _) => {
                assert_eq!(bindings.len(), 1);
                assert!(matches!(bindings[0].value, Expr::Pipe(_, _, _)));
                assert!(matches!(*body, Expr::Var(ref n, _) if n == "loss"));
            }
            _ => panic!("expected Block, got {e:?}"),
        }
    }

    #[test]
    fn block_rejects_call_like_short_binding_head() {
        let err = p_err("def f() = { g(x) = 1 x }");
        assert!(matches!(err, ParseError::Expected { .. }));
    }

    #[test]
    fn block_rejects_missing_separator_between_lets() {
        let err = p_err("def f() = { x = 1 y = 2 y }");
        assert!(matches!(err, ParseError::Expected { .. }));
    }

    #[test]
    fn par_accepts_semicolons_with_newlines() {
        let e = body(
            "def f() = par {
                a;
                b
            }",
        );
        match e {
            Expr::Par(exprs, _) => {
                assert_eq!(exprs.len(), 2);
            }
            _ => panic!("expected Par, got {e:?}"),
        }
    }

    // ===== chelis#706: bounded block tail + bare-statement diagnostic =====
    //
    // The tail expression, like a binding value, is Sep-bounded: a
    // top-level newline/`;` ends it unless the next line begins a
    // continuation (`|>`, `then`, `else`; spec/02 §P12) or the break is
    // inside ()/[]/{}. A second top-level expression after
    // the tail is rejected (previously it silently cross-newline
    // juxtaposed into an application).

    #[test]
    fn block_tail_multiline_leading_pipe_pipeline_parses() {
        // Positive #1: tail-only multi-line pipeline stays one Pipe tail.
        let e = body(
            "def f(x) = {
                x
                |> g
                |> h
            }",
        );
        match e {
            Expr::Block(bindings, body, _) => {
                assert!(bindings.is_empty());
                assert!(
                    matches!(*body, Expr::Pipe(_, _, _)),
                    "tail should be a pipeline: {body:?}"
                );
            }
            _ => panic!("expected Block, got {e:?}"),
        }
    }

    #[test]
    fn block_bindings_then_multiline_pipe_tail_parses() {
        // Positive #2: bindings followed by a multi-line pipe tail (mlp.ch shape).
        let e = body(
            "def f(logits, labels) = {
                h = relu(logits)
                softmax(h, 0)
                |> log
                |> mul(labels)
            }",
        );
        match e {
            Expr::Block(bindings, body, _) => {
                assert_eq!(bindings.len(), 1);
                assert!(
                    matches!(*body, Expr::Pipe(_, _, _)),
                    "tail should be a pipeline: {body:?}"
                );
            }
            _ => panic!("expected Block, got {e:?}"),
        }
    }

    #[test]
    fn block_tail_multiline_match_arms_parses() {
        // Positive #3: tail is a match whose arms span multiple lines;
        // the arm-separating newlines are brace-guarded.
        let e = body(
            "def f(x) = {
                match x with {
                    | Some y => y
                    | None => 0
                }
            }",
        );
        match e {
            Expr::Block(bindings, body, _) => {
                assert!(bindings.is_empty());
                match *body {
                    Expr::Match(_, ref arms, _) => assert_eq!(arms.len(), 2),
                    ref other => panic!("expected Match tail, got {other:?}"),
                }
            }
            _ => panic!("expected Block, got {e:?}"),
        }
    }

    #[test]
    fn block_tail_newlines_inside_parens_and_list_parse() {
        // Positive #4: newlines inside ()/[] in the tail are interior,
        // not separators.
        let e = body(
            "def f(a, b, c) = {
                g(
                    a,
                    [
                        b,
                        c
                    ]
                )
            }",
        );
        match e {
            Expr::Block(bindings, body, _) => {
                assert!(bindings.is_empty());
                assert!(
                    matches!(*body, Expr::Apply(_, _, _)),
                    "tail should be a call: {body:?}"
                );
            }
            _ => panic!("expected Block, got {e:?}"),
        }
    }

    #[test]
    fn block_tail_nested_block_parses() {
        // Positive #5a: tail is itself a nested block.
        let e = body(
            "def f(x) = {
                y = 1
                {
                    z = 2
                    z
                }
            }",
        );
        match e {
            Expr::Block(bindings, body, _) => {
                assert_eq!(bindings.len(), 1);
                assert!(
                    matches!(*body, Expr::Block(_, _, _)),
                    "tail should be a nested block: {body:?}"
                );
            }
            _ => panic!("expected Block, got {e:?}"),
        }
    }

    #[test]
    fn with_device_block_binding_then_tail_parses() {
        // Positive #5b: parse_with_handler routes through parse_block, so
        // the bounded tail applies to `with device(..) { .. }` too.
        let e = body(
            "def f(x) = with device(\"gpu:0\") {
                y = 1
                f(y)
            }",
        );
        match e {
            Expr::WithDevice(_, body, _) => match *body {
                Expr::Block(ref bindings, ref tail, _) => {
                    assert_eq!(bindings.len(), 1);
                    assert!(matches!(**tail, Expr::Apply(_, _, _)));
                }
                ref other => panic!("expected Block body, got {other:?}"),
            },
            _ => panic!("expected WithDevice, got {e:?}"),
        }
    }

    #[test]
    fn block_tail_trailing_separator_parses() {
        // Positive #6: a trailing separator after the tail is fine.
        let e = body("def f(x) = { g(x); }");
        match e {
            Expr::Block(bindings, body, _) => {
                assert!(bindings.is_empty());
                assert!(matches!(*body, Expr::Apply(_, _, _)));
            }
            _ => panic!("expected Block, got {e:?}"),
        }
    }

    #[test]
    fn block_bare_second_statement_is_rejected_at_second_call() {
        // Negative #8: the #706 reproducer. Two newline-separated calls;
        // the second is a bare non-tail statement.
        let src = "def f(a, b, c, d) = {
                g(a, b)
                h(c, d)
            }";
        let err = p_err(src);
        let offset = match err {
            ParseError::BareStatementInBlock { offset } => offset,
            other => panic!("expected BareStatementInBlock, got {other:?}"),
        };
        // Offset lands on the second statement's first token.
        assert_eq!(offset, src.find("h(c, d)").unwrap());
        assert!(
            err.to_string().contains("_ ="),
            "message should suggest `_ =`: {err}"
        );
    }

    #[test]
    fn block_bare_second_statement_semicolon_is_rejected() {
        // Negative #9: the explicit-`;` variant upgrades to the same
        // targeted diagnostic (was a generic "expected RBrace").
        let src = "def f(x, y) = { g(x); h(y) }";
        let err = p_err(src);
        let offset = match err {
            ParseError::BareStatementInBlock { offset } => offset,
            other => panic!("expected BareStatementInBlock, got {other:?}"),
        };
        assert_eq!(offset, src.find("h(y)").unwrap());
    }

    #[test]
    fn block_three_statements_error_at_second() {
        // Negative #10: three statements -> error at the second, not the third.
        let src = "def f(a, b, c) = {
                g(a)
                h(b)
                k(c)
            }";
        let err = p_err(src);
        let offset = match err {
            ParseError::BareStatementInBlock { offset } => offset,
            other => panic!("expected BareStatementInBlock, got {other:?}"),
        };
        assert_eq!(offset, src.find("h(b)").unwrap());
    }

    #[test]
    fn block_tail_then_paren_group_is_rejected() {
        // Negative #11: parens-absorption vector. `g(x)\n(a, b)` must not
        // juxtapose the tuple onto the call.
        let src = "def f(x, a, b) = {
                g(x)
                (a, b)
            }";
        let err = p_err(src);
        assert!(
            matches!(err, ParseError::BareStatementInBlock { .. }),
            "expected BareStatementInBlock, got {err:?}"
        );
    }

    #[test]
    fn par_bare_newline_separated_items_rejected() {
        // Negative #12: par items are Sep-bounded too; a newline between
        // items (no `;`) yields the clean separator error, not a silent
        // one-task collapse.
        let src = "def f(x, y) -> Unit = par {
                g(x)
                h(y)
            }";
        let err = p_err(src);
        match err {
            ParseError::Expected { ref expected, .. } => {
                assert!(
                    expected.contains("separator"),
                    "expected separator error, got {err:?}"
                );
            }
            other => panic!("expected separator error, got {other:?}"),
        }
    }

    #[test]
    fn block_no_tail_still_reports_expected_expression() {
        // Negative #13: a block with only a binding and no tail keeps the
        // pinned "expected expression" shape (not BareStatementInBlock).
        let src = "def f() = { x = 1 }";
        let err = p_err(src);
        match err {
            ParseError::Expected { ref expected, .. } => {
                assert!(
                    expected.contains("expression"),
                    "expected expression error, got {err:?}"
                );
            }
            other => panic!("expected `expression` error, got {other:?}"),
        }
    }

    #[test]
    fn block_semicolon_separator_is_rejected_without_offering_semicolon() {
        // Negative #14 (chelis#1267): canonical Surf v0.19 rejects `;` as a
        // block separator (spec/02-surf-syntax.md §P5, §P12), so the
        // diagnostic must not list `;` among the acceptable spellings. The
        // issue's reproducer spelled the values `cast(1, i64)`; the suffix
        // form fails identically and keeps the fixture free of type sugar.
        let src = "def main() -> i64 = { a = 1i64; b = 2i64; add(a, b) }";
        let err = parse_str(src).unwrap_err();
        let offset = match err {
            ParseError::SemicolonBlockSeparator { offset } => offset,
            ref other => panic!("expected SemicolonBlockSeparator, got {other:?}"),
        };
        // Points at the first `;`, the one just after `1i64`.
        assert_eq!(offset, src.find(';').unwrap());
        let msg = err.to_string();
        assert!(
            msg.contains("expected a newline separator, found `;`"),
            "message must name the newline as the expectation: {msg}"
        );
        assert!(
            msg.contains("not a block separator in canonical Surf v0.19"),
            "message must state the v0.19 rule: {msg}"
        );
        assert!(
            msg.contains("chelis migrate surf --from 0.18"),
            "message must name the migrator: {msg}"
        );
        // The contradiction the issue reported: the old wording offered `;`
        // as one of two acceptable separators while refusing that exact token.
        assert!(
            !msg.contains("separator (`;`"),
            "message must not offer `;` as an acceptable separator: {msg}"
        );
    }

    #[test]
    fn block_missing_separator_non_semicolon_keeps_generic_message() {
        // Negative #15 (chelis#1267 parity): a stray non-`;` token at a block
        // separator boundary keeps the generic separator diagnostic rather
        // than being rewritten into a `;` lecture. In canonical mode the
        // generic wording names only the newline, because that is the only
        // separator the grammar accepts here.
        let src = "def f() -> i64 = { a = 1i64";
        let err = parse_str(src).unwrap_err();
        match err {
            ParseError::Expected {
                ref expected,
                ref found,
                offset,
            } => {
                assert_eq!(expected, "separator (newline)", "got {err:?}");
                assert_eq!(found, "Eof", "got {err:?}");
                // Deliberate: the synthetic Eof token carries the opening
                // `{`'s offset, so an unterminated block points at the
                // construct that was never closed rather than at end of
                // input. Pinned so a future offset change is a decision.
                assert_eq!(offset, src.find('{').unwrap(), "got {err:?}");
            }
            ref other => panic!("expected generic separator error, got {other:?}"),
        }
    }

    /// Every canonical `;`-in-a-block shape must reach the one rule message,
    /// pointing at the `;` the reader actually typed.
    fn assert_semicolon_rule_at(src: &str, semicolon_index: usize) {
        let err = parse_str(src).unwrap_err();
        let offset = match err {
            ParseError::SemicolonBlockSeparator { offset } => offset,
            ref other => panic!("expected SemicolonBlockSeparator, got {other:?}"),
        };
        let expected = src
            .match_indices(';')
            .nth(semicolon_index)
            .expect("fixture has that many semicolons")
            .0;
        assert_eq!(offset, expected, "wrong `;` blamed: {err}");
    }

    #[test]
    fn block_trailing_semicolon_after_tail_names_the_semicolon_rule() {
        // Negative #17 (chelis#1267): spec/02-surf-syntax.md §P5 rejects a
        // trailing `;` by name. It used to report "expression statement must
        // be bound ... move it to tail position", which is false twice for
        // this input: `add(a, 1i64)` IS the tail, and nothing is unbound.
        // Taking that advice (`_ = add(a, 1i64);`) just landed on the `;`.
        assert_semicolon_rule_at("def f() -> i64 = {\n  a = 1i64\n  add(a, 1i64);\n}\n", 0);
    }

    #[test]
    fn block_semicolon_on_its_own_line_after_tail_names_the_semicolon_rule() {
        // Negative #18: same defect with the `;` on its own line, where the
        // preceding newline is consumed first and the old code still fell
        // through to the bare-statement message.
        assert_semicolon_rule_at(
            "def f() -> i64 = {\n  a = 1i64\n  add(a, 1i64)\n  ;\n}\n",
            0,
        );
    }

    #[test]
    fn canonical_two_statements_separated_by_semicolon_report_the_semicolon() {
        // Negative #19: `{ f(x); g(y) }` has two faults at once, a bare
        // non-tail statement and a `;`. Canonical mode now reports the `;`,
        // which is the lexically first one and the only one whose remedy is
        // not itself rejected: `_ = f(x); g(y)` still fails on the `;`.
        //
        // What makes this a reclassification rather than a lost diagnostic:
        // v0.18 reads that `;` as a real separator, so the bare statement is
        // the genuine fault there and #706's message still fires. That half
        // is already pinned by `block_bare_second_statement_semicolon_is_rejected`
        // on byte-identical source, so it is not restated here.
        assert_semicolon_rule_at("def f(x, y) -> unit = { f(x); g(y) }\n", 0);
    }

    #[test]
    fn block_semicolon_between_binding_eq_and_value_names_the_semicolon_rule() {
        // Negative #23 (chelis#1267): the fourth separator position, in
        // `parse_block_let_binding` between `=` and the value. `a = ; 1i64`
        // is a legal v0.18 spelling the migrator rewrites to `a = 1i64`, so
        // migrated-era source reaches it. Unguarded it left the value's token
        // range empty and reported an offsetless "unexpected end of input".
        assert_semicolon_rule_at("def f() -> i64 = {\n  a = ; 1i64\n  a\n}\n", 0);
    }

    #[test]
    fn block_semicolon_on_its_own_line_before_a_binding_value_names_the_semicolon_rule() {
        // Negative #24: the same position reached across newlines, where the
        // separator count is already nonzero.
        assert_semicolon_rule_at("def f() -> i64 = {\n  a =\n  ;\n  1i64\n  a\n}\n", 0);
    }

    #[test]
    fn block_semicolon_after_a_typed_binder_names_the_semicolon_rule() {
        // Negative #25: the type annotation moves the `=` but not the rule.
        assert_semicolon_rule_at("def f() -> i64 = {\n  a: i64 = ;1i64\n  a\n}\n", 0);
    }

    #[test]
    fn legacy_v018_still_parses_a_semicolon_before_a_binding_value() {
        // Positive parity for #23: `a = ; 1i64` is the v0.18 spelling the
        // migrator accepts and rewrites, so the guard must stay canonical
        // only or the migration path stops working on real source.
        let decls = parse_str_legacy_v018("def f() -> i64 = {\n  a = ; 1i64\n  a\n}\n")
            .expect("v0.18 accepts a `;` before a binding value");
        assert_eq!(decls.len(), 1);
    }

    #[test]
    fn block_semicolon_after_a_newline_between_bindings_names_the_semicolon_rule() {
        // Negative #20 (chelis#1267): when a newline precedes the `;` the
        // separator count is already nonzero, so the between-bindings arm was
        // skipped, `is_short_block_binding_start` was false at `;`, and the
        // tail parse got an empty range. That surfaced as a bare "unexpected
        // end of input" with no offset at all, which the LSP then rendered
        // past the end of the file.
        assert_semicolon_rule_at(
            "def f() -> i64 = {\n  a = 1i64\n  ;\n  b = 2i64\n  add(a, b)\n}\n",
            0,
        );
    }

    #[test]
    fn block_semicolon_leading_a_binding_line_names_the_semicolon_rule() {
        // Negative #21: the same shape with the next binding on the `;` line.
        assert_semicolon_rule_at(
            "def f() -> i64 = {\n  a = 1i64\n  ; b = 2i64\n  add(a, b)\n}\n",
            0,
        );
    }

    #[test]
    fn block_containing_only_a_semicolon_names_the_semicolon_rule() {
        // Negative #22: `;` at the leading separator position, before any
        // binding exists. Also previously "unexpected end of input".
        assert_semicolon_rule_at("def f() -> i64 = { ; }\n", 0);
    }

    #[test]
    fn canonical_newline_separated_block_parses() {
        // Positive control: the whole pre-existing block bank runs through
        // the v0.18 helpers, so nothing pinned that a canonical block parses
        // at all. Without this, every negative above could pass on a parser
        // that rejected every block.
        let decls = parse_str("def f() -> i64 = {\n  a = 1i64\n  add(a, 1i64)\n}\n")
            .expect("a newline-separated canonical block parses");
        assert_eq!(decls.len(), 1);
    }

    #[test]
    fn the_migrator_hint_in_the_semicolon_message_is_true() {
        // The message tells the reader to run `chelis migrate surf --from
        // 0.18`. Nothing pinned that the migrator actually resolves the shape
        // being diagnosed, so a migrator regression would silently turn this
        // diagnostic into the chelis#1267 defect reborn inside its own fix:
        // advice that does not work. `migrate_source_v018` is the library
        // path behind that CLI command (`cmd_migrate` in chelis-cli).
        let repro = "def main() -> i64 = { a = 1i64; b = 2i64; add(a, b) }\n";
        parse_str(repro).expect_err("the reproducer must not parse canonically");
        let migrated = crate::format::migrate_source_v018(repro)
            .expect("the migrator rewrites the `;` block the diagnostic points at");
        assert!(
            !migrated.contains(';'),
            "migration should remove the block separators: {migrated}"
        );
        parse_str(&migrated).expect("migrator output must parse under canonical Surf v0.19");
    }

    #[test]
    fn legacy_v018_missing_separator_still_offers_semicolon() {
        // Parity for #15: the generic wording is mode-dependent because the
        // two grammars accept different separators. `;` is genuinely one of
        // v0.18's, so dropping it from the legacy message would be the
        // chelis#1267 defect pointed the other way.
        let err = parse_str_legacy_v018("def f() -> i64 = { a = 1i64").unwrap_err();
        match err {
            ParseError::Expected {
                ref expected,
                ref found,
                ..
            } => {
                assert_eq!(expected, "separator (`;` or newline)", "got {err:?}");
                assert_eq!(found, "Eof", "got {err:?}");
            }
            ref other => panic!("expected generic separator error, got {other:?}"),
        }
    }

    #[test]
    fn legacy_v018_block_still_accepts_semicolon_separators() {
        // Positive parity for #14: the v0.18 compatibility grammar behind
        // `chelis migrate surf --from 0.18` still reads `;` as a block
        // separator, so the new wording is scoped to canonical Surf v0.19
        // rather than claiming `;` is never a block separator.
        let decls = parse_str_legacy_v018("def main() -> i64 = { a = 1i64; add(a, 2i64) }")
            .expect("v0.18 blocks accept `;` separators");
        assert_eq!(decls.len(), 1);
    }

    #[test]
    fn par_separator_message_still_names_semicolon() {
        // Negative control for #14: `par` and `do` genuinely require `;`
        // (spec/02-surf-syntax.md §P5), so their separator diagnostic must
        // keep offering it. Naming the newline there would be the same
        // defect in the opposite direction.
        let src = "def f(x, y) -> unit = par {\n    g(x)\n    h(y)\n}";
        let err = parse_str(src).unwrap_err();
        match err {
            ParseError::Expected { ref expected, .. } => {
                assert_eq!(expected, "separator (`;`)", "got {err:?}");
            }
            ref other => panic!("expected par separator error, got {other:?}"),
        }
    }

    #[test]
    fn retired_seed_handler_is_a_typed_rejection() {
        // `with seed` was retired with the explicit key switch (#2413): the
        // parser names the retired form and points at explicit keys.
        let err = p_err("def f() = with seed(42i64) { dropout(x, 0.5) }");
        assert!(
            matches!(
                err,
                ParseError::RetiredRandomness {
                    spelling: "with seed",
                    ..
                }
            ),
            "got {err:?}"
        );
        assert!(err.to_string().contains("key_from_seed"), "got {err}");
    }

    #[test]
    fn int64_minimum_keeps_the_normative_unary_minus_ast() {
        let expression = body("def result() = -9223372036854775808i64");

        assert!(
            matches!(expression, Expr::Unary(UnaryOp::Neg, _, _)),
            "spec/02 P10 requires every negative literal to parse as unary minus: {expression:?}"
        );
    }

    #[test]
    fn with_device_handler_expr() {
        let e = body("def f() = with device(\"gpu:0\") { x }");
        match e {
            Expr::WithDevice(device, body, _) => {
                assert!(
                    matches!(*device, Expr::Lit(Literal::Str(ref value), _) if value == "gpu:0")
                );
                assert!(matches!(*body, Expr::Block(_, _, _)));
            }
            _ => panic!("expected WithDevice, got {e:?}"),
        }
    }

    // ===== Type expression tests =====

    #[test]
    fn type_named() {
        let decls = p("x: f32 = 1.0");
        match &decls[0] {
            Decl::LetDef { ty: Some(ty), .. } => {
                assert!(matches!(ty, TypeExpr::Named(n, _) if n == "f32"));
            }
            _ => panic!("expected typed let"),
        }
    }

    // The first parameter's type annotation of a single `def`.
    fn first_param_type(s: &str) -> TypeExpr {
        match p(s).into_iter().next().unwrap() {
            Decl::FunDef { params, .. } => params.into_iter().next().unwrap().ty.unwrap(),
            other => panic!("expected FunDef, got {other:?}"),
        }
    }

    // A module-qualified type name `Demo.Dropout.Mode` (chelis#316) parses as
    // a `Named` type carrying the dotted path; reef resolves it to the
    // declaring module's type. Lets a consumer annotate against one of two
    // imported modules that export the same type name.
    #[test]
    fn qualified_named_type_parses() {
        let ty = first_param_type("def f(m: Demo.Dropout.Mode) = m");
        assert!(
            matches!(&ty, TypeExpr::Named(n, _) if n == "Demo.Dropout.Mode"),
            "expected qualified Named type, got {ty:?}"
        );
    }

    // A qualified *applied* type head: `xs: Demo.Coral.Frame[n]`.
    #[test]
    fn qualified_applied_type_parses() {
        let ty = first_param_type("def f(xs: Demo.Coral.Frame[n]) = xs");
        match &ty {
            TypeExpr::App(name, args, _) => {
                assert_eq!(name, "Demo.Coral.Frame");
                assert_eq!(args.len(), 1);
            }
            other => panic!("expected qualified App type, got {other:?}"),
        }
    }

    // A bare type name is unchanged — the dotted extension only fires on a
    // following `.PascalCase` segment.
    #[test]
    fn bare_named_type_unchanged() {
        let ty = first_param_type("def f(m: Mode) = m");
        assert!(matches!(&ty, TypeExpr::Named(n, _) if n == "Mode"));
    }

    #[test]
    fn type_tensor() {
        let decls = p("x: tensor[batch, hidden, f32] = x");
        match &decls[0] {
            Decl::LetDef { ty: Some(ty), .. } => match ty {
                TypeExpr::Tensor(dims, prec, _) => {
                    assert_eq!(dims.len(), 2);
                    assert_eq!(prec, "f32");
                }
                _ => panic!("expected Tensor type, got {ty:?}"),
            },
            _ => panic!("expected typed let"),
        }
    }

    #[test]
    fn type_tensor_with_literal_dims() {
        let decls = p("x: tensor[32, 784, f32] = x");
        match &decls[0] {
            Decl::LetDef { ty: Some(ty), .. } => match ty {
                TypeExpr::Tensor(dims, prec, _) => {
                    assert_eq!(dims.len(), 2);
                    assert!(matches!(&dims[0], TypeExpr::Named(n, _) if n == "32"));
                    assert!(matches!(&dims[1], TypeExpr::Named(n, _) if n == "784"));
                    assert_eq!(prec, "f32");
                }
                _ => panic!("expected Tensor type, got {ty:?}"),
            },
            _ => panic!("expected typed let"),
        }
    }

    #[test]
    fn tensor_inference_hole_spellings_are_rejected_at_parse_time() {
        for source in [
            "x: tensor[_, 4, f32] = x",
            "x: tensor[4, _] = x",
            "x: tensor[.._, f32] = x",
        ] {
            assert!(
                parse_str(source).is_err(),
                "tensor inference-hole spelling must not reach desugaring: {source}"
            );
        }
        assert!(
            parse_str("x: tensor[*, 4, f32] = x").is_ok(),
            "`*` remains the explicit dynamic-dimension spelling"
        );
    }

    // chelis#258 / rank polymorphism Tier-2: `..r` rank-variable spread.

    #[test]
    fn rank_spread_parses_as_sole_dim() {
        let decls = p("def f[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = relu(x)");
        let Decl::FunDef {
            ret_ty: Some(ret), ..
        } = &decls[0]
        else {
            panic!("expected fun def, got {:?}", decls[0]);
        };
        match ret {
            TypeExpr::Tensor(dims, prec, _) => {
                assert_eq!(dims.len(), 1, "rank var must be the sole shape element");
                assert!(
                    matches!(&dims[0], TypeExpr::RankSpread(n, _) if n == "r"),
                    "expected RankSpread(r), got {:?}",
                    dims[0]
                );
                assert_eq!(prec, "f32");
            }
            _ => panic!("expected Tensor return type, got {ret:?}"),
        }
    }

    #[test]
    fn rank_spread_adjacent_to_concrete_dim_parses() {
        // Tier-3: `..r` interleaved with concrete anchors is now valid syntax
        // (`tensor[..pre, seq, ..post, f32]`); the boundary moved to unification.
        let decls = p(
            "def f[pre, post](x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = x",
        );
        let Decl::FunDef {
            ret_ty: Some(ret), ..
        } = &decls[0]
        else {
            panic!("expected fun def, got {:?}", decls[0]);
        };
        match ret {
            TypeExpr::Tensor(dims, _, _) => {
                assert_eq!(
                    dims.len(),
                    2,
                    "two adjacent spreads in the reduce output type"
                );
                assert!(matches!(&dims[0], TypeExpr::RankSpread(n, _) if n == "pre"));
                assert!(matches!(&dims[1], TypeExpr::RankSpread(n, _) if n == "post"));
            }
            _ => panic!("expected Tensor return type, got {ret:?}"),
        }
    }

    #[test]
    fn duplicate_rank_spread_name_is_parse_error() {
        // The one parse-time fence: a spread name may not repeat in one shape.
        assert!(
            parse_str("x: tensor[..r, seq, ..r, f32] = x").is_err(),
            "a repeated `..r` in one tensor shape must be a parse error"
        );
    }

    #[test]
    fn rank_spread_erasure_position_parses() {
        // The erasure SHAPE (`..r` input, rank-0 `tensor[f32]` output) parses.
        // The erasure *tier* is deferred (no all-reduce primitive), so a body
        // like `sum(x, 0)` is rejected at check time by Body Discipline — this
        // test only pins that the surface shape is parseable.
        let decls = p("def sum_all[r](x: &tensor[..r, f32]) -> tensor[f32] = sum(x, 0)");
        assert!(matches!(&decls[0], Decl::FunDef { .. }));
    }

    #[test]
    fn type_arrow() {
        let decls = p("def f(x: f32 -> f32 -> bool): bool = x");
        match &decls[0] {
            Decl::FunDef { params, .. } => match &params[0].ty {
                Some(TypeExpr::Arrow(args, ret, _)) => {
                    assert_eq!(args.len(), 2);
                    assert!(matches!(ret.as_ref(), TypeExpr::Named(n, _) if n == "bool"));
                }
                other => panic!("expected Arrow type, got {other:?}"),
            },
            _ => panic!("expected FunDef"),
        }
    }

    #[test]
    fn fun_def_accepts_arrow_return_type() {
        let decls = p("def f(x: f32) -> bool = x");
        match &decls[0] {
            Decl::FunDef {
                ret_ty: Some(TypeExpr::Named(name, _)),
                ..
            } => assert_eq!(name, "bool"),
            other => panic!("expected arrow return type, got {other:?}"),
        }
    }

    #[test]
    fn type_app() {
        let decls = p("x: Option[f32] = x");
        match &decls[0] {
            Decl::LetDef { ty: Some(ty), .. } => match ty {
                TypeExpr::App(name, args, _) => {
                    assert_eq!(name, "Option");
                    assert_eq!(args.len(), 1);
                    assert!(matches!(&args[0], TypeExpr::Named(n, _) if n == "f32"));
                }
                _ => panic!("expected App type, got {ty:?}"),
            },
            _ => panic!("expected typed let"),
        }
    }

    #[test]
    fn list_literal() {
        let decls = p("xs = [1, 2, 3]");
        match &decls[0] {
            Decl::LetDef { value, .. } => match value {
                Expr::List(items, _) => {
                    assert_eq!(items.len(), 3);
                    assert!(matches!(&items[0], Expr::Lit(Literal::Int(1), _)));
                    assert!(matches!(&items[1], Expr::Lit(Literal::Int(2), _)));
                    assert!(matches!(&items[2], Expr::Lit(Literal::Int(3), _)));
                }
                other => panic!("expected list literal, got {other:?}"),
            },
            other => panic!("expected let def, got {other:?}"),
        }
    }

    #[test]
    fn empty_list_literal_with_type() {
        let decls = p("xs: List[i64] = []");
        match &decls[0] {
            Decl::LetDef {
                ty: Some(TypeExpr::App(name, args, _)),
                value,
                ..
            } => {
                assert_eq!(name, "List");
                assert_eq!(args.len(), 1);
                assert!(matches!(&args[0], TypeExpr::Named(inner, _) if inner == "i64"));
                assert!(matches!(value, Expr::List(items, _) if items.is_empty()));
            }
            other => panic!("expected typed empty list let, got {other:?}"),
        }
    }

    #[test]
    fn type_tuple() {
        let decls = p("x: (f32, f32) = x");
        match &decls[0] {
            Decl::LetDef { ty: Some(ty), .. } => match ty {
                TypeExpr::Tuple(types, _) => {
                    assert_eq!(types.len(), 2);
                }
                _ => panic!("expected Tuple type, got {ty:?}"),
            },
            _ => panic!("expected typed let"),
        }
    }

    #[test]
    fn type_infer() {
        let decls = p("x: _ = 1");
        match &decls[0] {
            Decl::LetDef { ty: Some(ty), .. } => {
                assert!(matches!(ty, TypeExpr::Infer(_)));
            }
            _ => panic!("expected typed let"),
        }
    }

    // ===== Builtins =====

    #[test]
    fn cast_expr() {
        let e = body("x = cast(y, f64)");
        match e {
            Expr::Cast(_, prec, _, _) => assert_eq!(prec, "f64"),
            _ => panic!("expected Cast, got {e:?}"),
        }
    }

    #[test]
    fn grad_expr() {
        let e = body("x = grad(f)");
        assert!(matches!(e, Expr::Grad(_, None, _)));
    }

    #[test]
    fn grad_expr_with_single_wrt() {
        let e = body("x = grad(f, wrt=w)");
        match e {
            Expr::Grad(_, Some(wrt), _) => assert_eq!(wrt, vec!["w".to_string()]),
            _ => panic!("expected Grad with wrt, got {e:?}"),
        }
    }

    #[test]
    fn grad_expr_with_multiple_wrt() {
        let e = body("x = grad(f, wrt=(w, b))");
        match e {
            Expr::Grad(_, Some(wrt), _) => {
                assert_eq!(wrt, vec!["w".to_string(), "b".to_string()])
            }
            _ => panic!("expected Grad with wrt tuple, got {e:?}"),
        }
    }

    #[test]
    fn vmap_expr() {
        let e = body("x = vmap(f)");
        match e {
            Expr::Vmap(_, axis, _) => assert!(axis.is_none()),
            _ => panic!("expected Vmap, got {e:?}"),
        }
    }

    #[test]
    fn vmap_with_axis() {
        let e = body("x = vmap(f, axis=1)");
        match e {
            Expr::Vmap(_, axis, _) => assert_eq!(axis, Some(1)),
            _ => panic!("expected Vmap, got {e:?}"),
        }
    }

    #[test]
    fn vmap_result_can_be_applied() {
        let e = body("x = vmap(f)(xs)");
        match e {
            Expr::Apply(func, args, _) => {
                assert_eq!(args.len(), 1);
                assert!(matches!(&args[0], Expr::Var(name, _) if name == "xs"));
                match func.as_ref() {
                    Expr::Vmap(inner, axis, _) => {
                        assert!(axis.is_none());
                        assert!(matches!(inner.as_ref(), Expr::Var(name, _) if name == "f"));
                    }
                    other => panic!("expected Vmap function, got {other:?}"),
                }
            }
            _ => panic!("expected Apply, got {e:?}"),
        }
    }

    #[test]
    fn jit_expr() {
        let e = body("x = jit(f)");
        assert!(matches!(e, Expr::Jit(_, _)));
    }

    // ===== Multiple declarations =====

    #[test]
    fn multiple_decls() {
        let decls = p("x = 1\ny = 2\ndef f(z) = z");
        assert_eq!(decls.len(), 3);
    }

    #[test]
    fn module_must_be_first_decl() {
        let err = p_err("x = 1\nmodule M\ny = 2");
        assert!(matches!(
            err,
            ParseError::Expected { ref expected, .. }
                if expected == "module declaration only as the first declaration in a file"
        ));
    }

    #[test]
    fn nested_module_is_rejected() {
        let err = p_err("module Outer\nmodule Inner\ny = 2");
        assert!(matches!(
            err,
            ParseError::Expected { ref expected, .. }
                if expected == "module declaration only as the first declaration in a file"
        ));
    }

    // ===== Logical operators =====

    #[test]
    fn and_or_precedence() {
        let e = body("x = a || b && c");
        match e {
            Expr::Binary(BinOp::Or, _, rhs, _) => {
                assert!(matches!(*rhs, Expr::Binary(BinOp::And, _, _, _)));
            }
            _ => panic!("expected Or(a, And(b, c)), got {e:?}"),
        }
    }

    // ===== Tuple expression =====

    #[test]
    fn tuple_expr() {
        let e = body("x = (1, 2, 3)");
        match e {
            Expr::Tuple(elems, _) => assert_eq!(elems.len(), 3),
            _ => panic!("expected Tuple, got {e:?}"),
        }
    }

    // ===== Boolean literals =====

    #[test]
    fn bool_lits() {
        let e = body("x = true");
        assert!(matches!(e, Expr::Lit(Literal::Bool(true), _)));
        let e = body("x = false");
        assert!(matches!(e, Expr::Lit(Literal::Bool(false), _)));
    }

    // ===== String literal in expr =====

    #[test]
    fn string_lit_expr() {
        let e = body("x = \"hello\"");
        assert!(matches!(e, Expr::Lit(Literal::Str(_), _)));
    }

    // ===== Constructor in expr =====

    #[test]
    fn constructor_expr() {
        let e = body("x = None");
        assert!(matches!(e, Expr::Constructor(ref n, _) if n == "None"));
    }

    // ===== Removed let/in surface =====

    #[test]
    fn chained_let_in_is_rejected() {
        let err = p_err("def f(x) = let a = 1 let b = 2 in a + b");
        assert!(matches!(err, ParseError::Expected { .. }));
    }

    #[test]
    fn let_and_in_are_valid_identifiers_in_blocks() {
        let e = body(
            "def f() = {
                let = 5
                in = 6
                add(let, in)
            }",
        );
        match e {
            Expr::Block(bindings, body, _) => {
                assert_eq!(bindings.len(), 2);
                assert!(matches!(
                    &bindings[0].pattern,
                    LetPattern::Var(name, _) if name == "let"
                ));
                assert!(matches!(
                    &bindings[1].pattern,
                    LetPattern::Var(name, _) if name == "in"
                ));
                assert!(matches!(*body, Expr::Apply(_, _, _)));
            }
            _ => panic!("expected Block, got {e:?}"),
        }
    }

    // ===== Non-assoc comparison =====

    #[test]
    fn non_assoc_lt_chain_error() {
        let err = p_err("x = a < b < c");
        assert!(matches!(err, ParseError::NonAssocChain { .. }));
    }

    #[test]
    fn different_comparison_classes_ok() {
        // == and < are in different classes, so this should parse.
        // Actually wait, both are non-assoc. Let's test same-class.
        // a < b is fine on its own.
        let _decls = p("x = a < b");
    }

    // ===== Unary bang =====

    #[test]
    fn unary_not() {
        let e = body("x = !a");
        assert!(matches!(e, Expr::Unary(UnaryOp::Not, _, _)));
    }

    // ===== Juxtaposition application tests =====

    #[test]
    fn juxtaposition_single_arg() {
        let e = body("x = f x");
        match &e {
            Expr::Apply(func, args, _) => {
                assert!(matches!(func.as_ref(), Expr::Var(n, _) if n == "f"));
                assert_eq!(args.len(), 1);
                assert!(matches!(&args[0], Expr::Var(n, _) if n == "x"));
            }
            _ => panic!("expected Apply, got {e:?}"),
        }
    }

    #[test]
    fn juxtaposition_two_args_flattens_during_v018_migration() {
        // The legacy ungrouped chain was the v0.18 spelling of a flat call.
        // Canonical v0.19 prints the resulting node as f(x, y).
        let e = body("x = f x y");
        match &e {
            Expr::Apply(func, args, _) => {
                assert!(matches!(func.as_ref(), Expr::Var(n, _) if n == "f"));
                assert_eq!(args.len(), 2);
                assert!(matches!(&args[0], Expr::Var(n, _) if n == "x"));
                assert!(matches!(&args[1], Expr::Var(n, _) if n == "y"));
            }
            _ => panic!("expected Apply, got {e:?}"),
        }
    }

    #[test]
    fn juxtaposition_with_infix() {
        // f x + g y → Binary(Add, Apply(f, [x]), Apply(g, [y]))
        let e = body("x = f x + g y");
        match &e {
            Expr::Binary(BinOp::Add, lhs, rhs, _) => {
                match lhs.as_ref() {
                    Expr::Apply(func, args, _) => {
                        assert!(matches!(func.as_ref(), Expr::Var(n, _) if n == "f"));
                        assert_eq!(args.len(), 1);
                        assert!(matches!(&args[0], Expr::Var(n, _) if n == "x"));
                    }
                    _ => panic!("expected Apply on lhs"),
                }
                match rhs.as_ref() {
                    Expr::Apply(func, args, _) => {
                        assert!(matches!(func.as_ref(), Expr::Var(n, _) if n == "g"));
                        assert_eq!(args.len(), 1);
                        assert!(matches!(&args[0], Expr::Var(n, _) if n == "y"));
                    }
                    _ => panic!("expected Apply on rhs"),
                }
            }
            _ => panic!("expected Binary Add, got {e:?}"),
        }
    }

    #[test]
    fn juxtaposition_grouped_arg() {
        // f (x + y) → Apply(f, [Binary(Add, x, y)])
        let e = body("x = f (x + y)");
        match &e {
            Expr::Apply(func, args, _) => {
                assert!(matches!(func.as_ref(), Expr::Var(n, _) if n == "f"));
                assert_eq!(args.len(), 1);
                assert!(matches!(&args[0], Expr::Binary(BinOp::Add, _, _, _)));
            }
            _ => panic!("expected Apply, got {e:?}"),
        }
    }

    #[test]
    fn borrow_prefix_parses() {
        let e = body("y = f(&x)");
        match &e {
            Expr::Apply(_, args, _) => {
                assert!(
                    matches!(&args[0], Expr::Borrow(inner, _) if matches!(inner.as_ref(), Expr::Var(name, _) if name == "x"))
                );
            }
            _ => panic!("expected Apply, got {e:?}"),
        }
    }

    #[test]
    fn borrow_juxtaposition_argument_parses() {
        let e = body("y = f &x");
        match &e {
            Expr::Apply(_, args, _) => {
                assert!(
                    matches!(&args[0], Expr::Borrow(inner, _) if matches!(inner.as_ref(), Expr::Var(name, _) if name == "x"))
                );
            }
            _ => panic!("expected Apply, got {e:?}"),
        }
    }

    #[test]
    fn borrow_type_parses() {
        let decls = p("def f(x: &tensor[4, f32]) -> tensor[4, f32] = relu(x)");
        match &decls[0] {
            Decl::FunDef { params, .. } => match &params[0].ty {
                Some(TypeExpr::Ref(inner, _)) => {
                    assert!(
                        matches!(inner.as_ref(), TypeExpr::Tensor(_, precision, _) if precision == "f32")
                    );
                }
                other => panic!("expected borrowed tensor parameter, got {other:?}"),
            },
            other => panic!("expected function definition, got {other:?}"),
        }
    }

    #[test]
    fn constructor_juxtaposition() {
        // Some x → Apply(Constructor("Some"), [Var("x")])
        let e = body("x = Some x");
        match &e {
            Expr::Apply(func, args, _) => {
                assert!(matches!(func.as_ref(), Expr::Constructor(n, _) if n == "Some"));
                assert_eq!(args.len(), 1);
            }
            _ => panic!("expected Apply, got {e:?}"),
        }
    }

    // ===== Dotted import path tests =====

    #[test]
    fn import_dotted_path() {
        let decls = p("import Foo.Bar.Baz(baz)");
        match &decls[0] {
            Decl::Import { module, kind, .. } => {
                assert_eq!(module, "Foo.Bar.Baz");
                assert_eq!(kind, &ImportKind::Names(vec!["baz".to_string()]));
            }
            _ => panic!("expected Import"),
        }
    }

    #[test]
    fn import_dotted_no_names() {
        let decls = p("import Foo.Bar");
        match &decls[0] {
            Decl::Import { module, kind, .. } => {
                assert_eq!(module, "Foo.Bar");
                assert_eq!(kind, &ImportKind::Qualified);
            }
            _ => panic!("expected Import"),
        }
    }

    // ===== Module without braces =====

    #[test]
    fn module_braceless() {
        let decls = p("module Foo\ndef f(x) = x\ndef g(y) = y");
        match &decls[0] {
            Decl::Module { name, decls, .. } => {
                assert_eq!(name, "Foo");
                assert_eq!(decls.len(), 2);
            }
            _ => panic!("expected Module"),
        }
    }

    // ===== Export declaration =====

    #[test]
    fn export_decl() {
        let decls = p("export (foo, bar, baz)");
        match &decls[0] {
            Decl::Export { names, .. } => {
                assert_eq!(names, &["foo", "bar", "baz"]);
            }
            _ => panic!("expected Export"),
        }
    }

    #[test]
    fn export_single() {
        let decls = p("export (foo)");
        match &decls[0] {
            Decl::Export { names, .. } => {
                assert_eq!(names, &["foo"]);
            }
            _ => panic!("expected Export"),
        }
    }

    // ===== Record pattern =====

    #[test]
    fn record_pattern() {
        let e = body("x = match x with { | Adam { lr, eps } => lr }");
        match e {
            Expr::Match(_, arms, _) => {
                assert_eq!(arms.len(), 1);
                match &arms[0].pattern {
                    Pattern::Record(name, fields, _) => {
                        assert_eq!(name, "Adam");
                        assert_eq!(fields.len(), 2);
                        assert_eq!(fields[0].0, "lr");
                        assert_eq!(fields[1].0, "eps");
                        assert!(matches!(&fields[0].1, Pattern::Var(n, _) if n == "lr"));
                        assert!(matches!(&fields[1].1, Pattern::Var(n, _) if n == "eps"));
                    }
                    other => panic!("expected Record pattern, got {other:?}"),
                }
            }
            _ => panic!("expected Match, got {e:?}"),
        }
    }

    // ===== As pattern =====

    #[test]
    fn as_pattern() {
        let e = body("x = match x with { | y @ Some z => y }");
        match e {
            Expr::Match(_, arms, _) => {
                assert_eq!(arms.len(), 1);
                match &arms[0].pattern {
                    Pattern::As(name, inner, _) => {
                        assert_eq!(name, "y");
                        match inner.as_ref() {
                            Pattern::Constructor(ctor, pats, _) => {
                                assert_eq!(ctor, "Some");
                                assert_eq!(pats.len(), 1);
                                assert!(matches!(&pats[0], Pattern::Var(n, _) if n == "z"));
                            }
                            other => panic!("expected Constructor inside As, got {other:?}"),
                        }
                    }
                    other => panic!("expected As pattern, got {other:?}"),
                }
            }
            _ => panic!("expected Match, got {e:?}"),
        }
    }

    // ===== Opaque invariant declarations (RFC D-SYNTAX) =====

    /// Extract the single `TypeDef` from a parsed program, descending
    /// into module wrappers.
    fn type_def(src: &str) -> (bool, Option<TypeInvariant>) {
        fn find(decls: Vec<Decl>) -> Option<(bool, Option<TypeInvariant>)> {
            for decl in decls {
                match decl {
                    Decl::TypeDef {
                        opaque, invariant, ..
                    } => return Some((opaque, invariant)),
                    Decl::Module { decls, .. } => {
                        if let Some(found) = find(decls) {
                            return Some(found);
                        }
                    }
                    _ => {}
                }
            }
            None
        }
        find(p(src)).unwrap_or_else(|| panic!("no TypeDef in: {src}"))
    }

    #[test]
    fn opaque_without_invariant_parses_none() {
        let (opaque, invariant) =
            type_def("module M\n@opaque\ntype Probability = | Probability { value: f32 }");
        assert!(opaque);
        assert!(invariant.is_none());
    }

    #[test]
    fn opaque_with_invariant_parses() {
        let (opaque, invariant) = type_def(
            "module M\n@opaque\n@invariant(p) (p.value >= 0.0) && (p.value <= 1.0)\n\
             type Probability = | Probability { value: f32 }",
        );
        assert!(opaque);
        let inv = invariant.expect("invariant parsed");
        assert_eq!(inv.binder, "p");
        // Body is the boolean `and` expression.
        assert!(
            matches!(inv.body, Expr::Binary(BinOp::And, _, _, _)),
            "expected And binary, got {:?}",
            inv.body
        );
    }

    #[test]
    fn invariant_polynomial_body_parses() {
        let (_, invariant) = type_def(
            "module M\n@opaque\n@invariant(p) (p.value * p.value) <= 1.0\n\
             type Probability = | Probability { value: f32 }",
        );
        assert_eq!(invariant.unwrap().binder, "p");
    }

    #[test]
    fn invariant_transcendental_body_parses() {
        let (_, invariant) = type_def(
            "module M\n@opaque\n@invariant(p) exp(p.value) <= 3.0\n\
             type Probability = | Probability { value: f32 }",
        );
        assert_eq!(invariant.unwrap().binder, "p");
    }

    #[test]
    fn simplex_tolerance_band_invariant_parses() {
        // The W6 flagship declaration: sum over a tensor field with a
        // module-constant tolerance band.
        let (_, invariant) = type_def(
            "module M\n@opaque\n\
             @invariant(p) (sum(p.weights) >= (1.0 - eps)) && (sum(p.weights) <= (1.0 + eps))\n\
             type Simplex = | Simplex { weights: tensor[3, f32] }",
        );
        assert_eq!(invariant.unwrap().binder, "p");
    }

    #[test]
    fn invariant_without_opaque_is_error() {
        let e = p_err("module M\n@invariant(p) p.value >= 0.0\ntype T = | T { value: f32 }");
        match e {
            ParseError::Expected { expected, .. } => {
                assert_eq!(expected, "invariant requires @opaque");
            }
            other => panic!("expected invariant-requires-opaque error, got {other:?}"),
        }
    }

    #[test]
    fn invariant_after_type_is_error() {
        // `@invariant` cannot follow `type`; it must precede it. Here the
        // type decl is parsed, then a leading `@invariant` is seen as a
        // fresh declaration with no `@opaque`.
        let e =
            p_err("module M\n@opaque\ntype T = | T { value: f32 }\n@invariant(p) p.value >= 0.0");
        match e {
            ParseError::Expected { expected, .. } => {
                assert_eq!(expected, "invariant requires @opaque");
            }
            other => panic!("expected error for @invariant after type, got {other:?}"),
        }
    }

    #[test]
    fn double_invariant_is_error() {
        let e = p_err(
            "module M\n@opaque\n@invariant(p) p.value >= 0.0\n\
             @invariant(p) p.value <= 1.0\ntype T = | T { value: f32 }",
        );
        match e {
            ParseError::Expected { expected, .. } => {
                assert_eq!(expected, "exactly one @invariant");
            }
            other => panic!("expected double-invariant error, got {other:?}"),
        }
    }

    #[test]
    fn multi_binder_invariant_is_error() {
        let e = p_err(
            "module M\n@opaque\n@invariant(p, q) p.value >= 0.0\ntype T = | T { value: f32 }",
        );
        match e {
            ParseError::Expected { expected, .. } => {
                assert_eq!(expected, "exactly one invariant binder");
            }
            other => panic!("expected multi-binder error, got {other:?}"),
        }
    }

    #[test]
    fn invariant_on_type_alias_is_error() {
        // `@opaque` (and so `@invariant`) on a type alias is rejected:
        // the ADT form is required.
        let e = p_err("module M\n@opaque\n@invariant(p) p.value >= 0.0\ntype T = f32");
        assert!(matches!(e, ParseError::Expected { .. }));
    }

    // ===== Module-qualified path tests (chelis#316) =====

    // `Demo.Dropout.Eval` is a module-qualified nullary constructor. It must
    // parse into a nested `Access` chain rooted at the head segment so reef
    // can walk the segments and resolve them to the module-qualified internal
    // name. Before the fix the parser rejected the uppercase `Dropout`
    // segment with "expected field name or tuple index".
    #[test]
    fn qualified_nullary_constructor_parses() {
        let e = body("def f() = Demo.Dropout.Eval");
        match &e {
            Expr::Access(inner, last, _) => {
                assert_eq!(last, "Eval");
                match inner.as_ref() {
                    Expr::Access(head, mid, _) => {
                        assert_eq!(mid, "Dropout");
                        assert!(matches!(head.as_ref(), Expr::Constructor(n, _) if n == "Demo"));
                    }
                    other => panic!("expected Access(Demo.Dropout), got {other:?}"),
                }
            }
            other => panic!("expected Access chain, got {other:?}"),
        }
    }

    // `Demo.Dropout.use(m)` is a module-qualified value applied to an
    // argument. The whole-path-then-call shape must yield `Apply(<access
    // chain>, [arg])`. This form previously failed with "expected end of
    // declaration expression, found LParen" even for a lowercase tail.
    #[test]
    fn qualified_call_parses() {
        let e = body("def f(m) = Demo.Dropout.use(m)");
        match &e {
            Expr::Apply(func, args, _) => {
                assert_eq!(args.len(), 1);
                assert!(matches!(&args[0], Expr::Var(n, _) if n == "m"));
                match func.as_ref() {
                    Expr::Access(inner, last, _) => {
                        assert_eq!(last, "use");
                        assert!(
                            matches!(inner.as_ref(), Expr::Access(_, mid, _) if mid == "Dropout")
                        );
                    }
                    other => panic!("expected Access chain as call head, got {other:?}"),
                }
            }
            other => panic!("expected Apply, got {other:?}"),
        }
    }

    // A qualified constructor applied to arguments: `Demo.List.Cons(x, xs)`.
    #[test]
    fn qualified_constructor_application_parses() {
        let e = body("def f(x, xs) = Demo.List.Cons(x, xs)");
        match &e {
            Expr::Apply(func, args, _) => {
                assert_eq!(args.len(), 2);
                assert!(matches!(func.as_ref(), Expr::Access(_, last, _) if last == "Cons"));
            }
            other => panic!("expected Apply, got {other:?}"),
        }
    }

    // Lowercase field access must still work and still bind tighter than a
    // following binary operator — the new trailing-call loop must not disturb
    // it.
    #[test]
    fn lowercase_field_access_still_parses() {
        let e = body("def f(r) = r.field + 1");
        match &e {
            Expr::Binary(BinOp::Add, left, _, _) => {
                assert!(matches!(left.as_ref(), Expr::Access(_, f, _) if f == "field"));
            }
            other => panic!("expected Add with field access on the left, got {other:?}"),
        }
    }

    // Tuple index access is unchanged.
    #[test]
    fn tuple_index_access_still_parses() {
        let e = body("def f(t) = t.0");
        assert!(matches!(&e, Expr::TupleGet(_, 0, _)));
    }

    // Convenience: parse a single def, return the first match arm's pattern.
    fn first_arm_pattern(s: &str) -> Pattern {
        match body(s) {
            Expr::Match(_, arms, _) => arms.into_iter().next().unwrap().pattern,
            other => panic!("expected Match, got {other:?}"),
        }
    }

    // A module-qualified *nullary* constructor pattern: `| Demo.Dropout.Train =>`.
    // The dotted head must land on the `Pattern::Constructor` name verbatim so
    // reef can resolve it to the declaring module's constructor (chelis#316).
    #[test]
    fn qualified_nullary_constructor_pattern_parses() {
        let pat = first_arm_pattern("def f(m) = match m with { | Demo.Dropout.Train => 1 }");
        match pat {
            Pattern::Constructor(name, args, _) => {
                assert_eq!(name, "Demo.Dropout.Train");
                assert!(args.is_empty());
            }
            other => panic!("expected Constructor pattern, got {other:?}"),
        }
    }

    // A qualified constructor pattern with positional sub-patterns:
    // `| Demo.List.Cons(x, xs) =>`. The dotted head and the sub-patterns must
    // both survive.
    #[test]
    fn qualified_constructor_pattern_with_args_parses() {
        let pat = first_arm_pattern("def f(m) = match m with { | Demo.List.Cons(x, xs) => 1 }");
        match pat {
            Pattern::Constructor(name, args, _) => {
                assert_eq!(name, "Demo.List.Cons");
                assert_eq!(args.len(), 2);
                assert!(matches!(&args[0], Pattern::Var(n, _) if n == "x"));
            }
            other => panic!("expected Constructor pattern, got {other:?}"),
        }
    }

    // A qualified *record* constructor pattern: `| Demo.Frame.Col { values } =>`.
    #[test]
    fn qualified_record_pattern_parses() {
        let pat = first_arm_pattern("def f(m) = match m with { | Demo.Frame.Col { values } => 1 }");
        match pat {
            Pattern::Record(name, fields, _) => {
                assert_eq!(name, "Demo.Frame.Col");
                assert_eq!(fields.len(), 1);
                assert_eq!(fields[0].0, "values");
            }
            other => panic!("expected Record pattern, got {other:?}"),
        }
    }

    // A bare constructor pattern is unchanged — the dotted-path extension only
    // fires when a `.PascalCase` segment actually follows.
    #[test]
    fn bare_constructor_pattern_unchanged() {
        let pat = first_arm_pattern("def f(m) = match m with { | None => 1 }");
        match pat {
            Pattern::Constructor(name, args, _) => {
                assert_eq!(name, "None");
                assert!(args.is_empty());
            }
            other => panic!("expected Constructor pattern, got {other:?}"),
        }
    }

    // ===== chelis#437: single-letter uppercase value identifiers =====
    //
    // §1.1 value-binding override: a single ASCII-uppercase letter is a
    // value identifier in a value-binding position (LHS, parameter, let
    // binder). Multi-letter PascalCase stays a type/constructor.

    #[test]
    fn single_letter_upper_value_binding_lhs() {
        // The exact #437 repro: `S = ...` is a value binding, not a parse
        // error. The lexer classifies `S` as a TypeIdent; the binding
        // context overrides the default and binds a value named `S`.
        let decls = p("def f(spot: f32) -> f32 = spot\nS = f(cast(2.0, f32))");
        assert_eq!(decls.len(), 2);
        match &decls[1] {
            Decl::LetDef { name, .. } => assert_eq!(name, "S"),
            other => panic!("expected LetDef named S, got {other:?}"),
        }
    }

    #[test]
    fn single_letter_upper_value_binding_each_letter() {
        // S, K, T, N, P all bind as values (finance notation).
        for letter in ["S", "K", "T", "N", "P"] {
            let src = format!("{letter} = cast(1.0, f32)");
            let decls = p(&src);
            match &decls[0] {
                Decl::LetDef { name, .. } => assert_eq!(name, letter),
                other => panic!("expected LetDef named {letter}, got {other:?}"),
            }
        }
    }

    #[test]
    fn single_letter_upper_function_name_rejected() {
        // A single-letter uppercase FUNCTION name (`def N`) is NOT part of
        // this carve-out: an applied uppercase head `N(x)` resolves to a
        // constructor, so a `def N` would be silently shadowed by
        // constructor resolution. Function names stay snake_case; the
        // carve-out covers value bindings and parameters only (chelis#437).
        assert!(
            parse_str("def N(x: f32) -> f32 = x").is_err(),
            "single-letter uppercase function name must stay rejected"
        );
    }

    #[test]
    fn single_letter_upper_param_name() {
        // `def payoff(S, K) = ...` — uppercase single-letter parameters.
        let decls = p("def payoff(S: f32, K: f32) -> f32 = S");
        match &decls[0] {
            Decl::FunDef { params, .. } => {
                assert_eq!(params.len(), 2);
                assert_eq!(params[0].name, "S");
                assert_eq!(params[1].name, "K");
            }
            other => panic!("expected FunDef, got {other:?}"),
        }
    }

    #[test]
    fn single_letter_upper_param_body_reference_is_var() {
        // In the body, a single-letter uppercase name that is a bound
        // parameter must read as a value reference. The parser emits it as
        // a constructor head (the lexer cannot see the binding); the
        // resolver later binds it to the parameter. Either way the surface
        // round-trips, so assert the body parses to a head named `S`.
        let b = body("def use_spot(S: f32) -> f32 = S");
        match b {
            Expr::Var(name, _) | Expr::Constructor(name, _) => assert_eq!(name, "S"),
            other => panic!("expected Var or Constructor head S, got {other:?}"),
        }
    }

    #[test]
    fn single_letter_upper_two_bindings_in_sequence() {
        // The decl-boundary scan must recognize a single-letter uppercase
        // head as the start of a new declaration so the prior decl's
        // expression stops before it (is_decl_start_at parity).
        let decls = p("S = cast(1.0, f32)\nT = cast(2.0, f32)");
        assert_eq!(decls.len(), 2);
        match (&decls[0], &decls[1]) {
            (Decl::LetDef { name: a, .. }, Decl::LetDef { name: b, .. }) => {
                assert_eq!(a, "S");
                assert_eq!(b, "T");
            }
            other => panic!("expected two LetDefs, got {other:?}"),
        }
    }

    #[test]
    fn single_letter_upper_block_binder() {
        // A block-let binder `{ S = x ; S }` (Surf has no `let` keyword)
        // binds a value named S. The binder reaches parse_let_pattern,
        // where the single-letter uppercase override applies.
        let src = "def f(x: f32) -> f32 = { S = x\n S }";
        let decls = parse_str(src).expect("block binder S = x should parse");
        match &decls[0] {
            Decl::FunDef { body, .. } => match body {
                Expr::Block(bindings, _, _) => {
                    assert_eq!(bindings.len(), 1);
                    match &bindings[0].pattern {
                        LetPattern::Var(name, _) => assert_eq!(name, "S"),
                        other => panic!("expected Var binder S, got {other:?}"),
                    }
                }
                other => panic!("expected Block body, got {other:?}"),
            },
            other => panic!("expected FunDef, got {other:?}"),
        }
    }

    // ----- negative parity: multi-letter PascalCase stays a type/ctor -----

    #[test]
    fn multi_letter_upper_binding_rejected() {
        // `Foo = ...` is NOT a value binding — the case-split is preserved
        // for multi-letter PascalCase names. This must stay a parse error.
        let err = p_err("Foo = cast(1.0, f32)");
        match err {
            ParseError::Expected { found, .. } => {
                assert!(
                    found.contains("Foo"),
                    "expected error mentioning Foo, got {found}"
                );
            }
            other => panic!("expected Expected error, got {other:?}"),
        }
    }

    #[test]
    fn multi_letter_upper_param_rejected() {
        // A multi-letter PascalCase parameter is rejected — params are
        // values, and multi-letter uppercase is reserved for types.
        assert!(
            parse_str("def f(Spot: f32) -> f32 = Spot").is_err(),
            "multi-letter uppercase parameter must be rejected"
        );
    }

    #[test]
    fn multi_letter_upper_expr_head_still_constructor() {
        // In expression position `Some(x)` is still a constructor
        // application; the value-binding override does not touch it.
        let b = body("def f(x: f32) -> f32 = Some(x)");
        match b {
            Expr::Apply(head, _, _) => match *head {
                Expr::Constructor(name, _) => assert_eq!(name, "Some"),
                other => panic!("expected Constructor head, got {other:?}"),
            },
            other => panic!("expected Apply, got {other:?}"),
        }
    }

    #[test]
    fn block_if_with_else_on_a_later_line_parses() {
        // chelis#849: inside a `{ }` block, `block_expr_end` ended the
        // statement at the newline after the `then` branch, so `parse_if`
        // reached Eof before its mandatory `else` and reported
        // `expected Else, found Eof`. The `else` is present -- only its line
        // placement differs. `|>` already had this continuation carve-out.
        let src = "def f(a: f32, b: f32) -> bool = {\n  c = neq(a, a)\n  if c then true\n  else lte(a, b)\n}";
        assert!(
            parse_str(src).is_ok(),
            "block + newline before `else` must parse: {:?}",
            parse_str(src).err()
        );
    }

    #[test]
    fn block_if_else_on_one_line_still_parses() {
        // Positive control for the sibling form, so the fix cannot be a
        // regression that only moves which layout works.
        let src = "def f(a: f32, b: f32) -> bool = {\n  c = neq(a, a)\n  if c then true else lte(a, b)\n}";
        assert!(parse_str(src).is_ok(), "same-line form must keep parsing");
    }

    #[test]
    fn block_if_with_a_missing_else_is_still_rejected() {
        // Negative parity: the continuation must not make `else` optional.
        // Pin the REASON, not merely that something failed -- an unrelated
        // rejection would otherwise keep this green (review of PR #1369).
        let src = "def f(a: f32, b: f32) -> bool = {\n  c = neq(a, a)\n  if c then true\n}";
        match parse_str(src) {
            Err(ParseError::Expected {
                expected, found, ..
            }) => {
                assert_eq!(expected, "Else", "the missing `else` is the reason");
                assert_eq!(found, "Eof", "the block ended before the `else`");
            }
            other => panic!("expected a missing-`else` rejection, got {other:?}"),
        }
    }

    #[test]
    fn block_if_with_a_missing_then_is_still_rejected() {
        // The `then` half of the same parity: admitting `then` as a
        // continuation must not make it optional either.
        let src = "def f(a: f32, b: f32) -> bool = {\n  c = neq(a, a)\n  if c\n}";
        match parse_str(src) {
            Err(ParseError::Expected { expected, .. }) => {
                assert_eq!(expected, "Then", "the missing `then` is the reason");
            }
            other => panic!("expected a missing-`then` rejection, got {other:?}"),
        }
    }

    #[test]
    fn block_if_with_then_on_a_later_line_parses() {
        // chelis#849 review: the same defect exists one token earlier. A
        // newline between the condition and `then` failed with
        // `expected Then, found Eof` for the identical reason.
        let src = "def f(a: f32, b: f32) -> bool = {\n  c = neq(a, a)\n  if c\n  then true\n  else lte(a, b)\n}";
        assert!(
            parse_str(src).is_ok(),
            "a newline before `then` must parse: {:?}",
            parse_str(src).err()
        );
    }

    #[test]
    fn a_multiline_else_if_chain_parses() {
        // The shape that motivated admitting `then`: each arm of a chained
        // `else if` puts its own `then` and `else` on later lines.
        let src = "def f(a: f32, b: f32) -> f32 = {\n  c = neq(a, a)\n  if c\n  then a\n  else if lt(a, b)\n  then b\n  else mul(a, b)\n}";
        assert!(
            parse_str(src).is_ok(),
            "a multiline `else if` chain must parse: {:?}",
            parse_str(src).err()
        );
    }

    /// chelis#849 review: the corrected spec/02 P12 states that a
    /// declaration body and a property predicate bound PERMISSIVELY -- they
    /// end at a declaration start (and, for a predicate, at `with`), so any
    /// other token continues them across a top-level newline.
    ///
    /// The earlier revision of P12 claimed the closed `|>`/`then`/`else` set
    /// governed declaration bodies too, which is false: a newline-led `with`
    /// continues one. Asserting it here keeps the numbered spec honest, and
    /// keeps this PR from silently narrowing a boundary it does not own.
    #[test]
    fn the_permissive_boundaries_are_not_governed_by_the_closed_set() {
        let declaration_body = concat!(
            "type Point = | Point { x: f32 }\n",
            "def update(p: Point) -> Point = p\n",
            "  with { x: 1.0f32 }\n",
        );
        assert!(
            parse_str(declaration_body).is_ok(),
            "a newline-led `with` must continue a declaration body: {:?}",
            parse_str(declaration_body).err()
        );

        // The `where` precondition routes through `property_expr_end`, the
        // third boundary rule. It is not the property-OPTION path covered
        // above; confusing the two is what left that consumer untested.
        let property_predicate =
            "@property p forall(x: i32) where if lte(x, 1i32)\n  then true\n  else false: true";
        assert!(
            parse_str(property_predicate).is_ok(),
            "a split `if` must survive a property predicate: {:?}",
            parse_str(property_predicate).err()
        );
    }

    /// chelis#849 review: `block_expr_end` is shared by seven call sites
    /// across five constructs -- block bindings, block tails, `do` items,
    /// `par` items, and property option values. The continuation rule applies
    /// to all of them, so each is covered rather than assumed.
    ///
    /// The property-option case reaches `block_expr_end` through
    /// `parse_property_option`, which needs a real `with <name> = <expr>`
    /// option. A `where` precondition looks similar and is NOT this path: it
    /// routes through `property_expr_end`, a different and permissive rule.
    /// The `where` form is covered separately below, precisely because
    /// mistaking one for the other is what left this consumer untested.
    #[test]
    fn the_continuation_rule_holds_for_every_block_expr_end_consumer() {
        let cases: &[(&str, &str)] = &[
            (
                "block binding value",
                "def f(a: f32, b: f32) -> f32 = {\n  c = neq(a, a)\n  d = if c\n  then a\n  else b\n  d\n}",
            ),
            (
                "block tail",
                "def f(a: f32, b: f32) -> f32 = {\n  c = neq(a, a)\n  if c\n  then a\n  else b\n}",
            ),
            (
                "do item",
                "def f(a: f32, b: f32) -> f32 ! { IO } = {\n  c = neq(a, a)\n  g = do {\n    if c\n    then print(\"y\")\n    else print(\"n\")\n  }\n  a\n}",
            ),
            (
                "property option value",
                "@property p forall(x: f32): true\n  with tolerance = if lte(x, 1.0f32)\n  then 1e-6f32\n  else 1e-3f32",
            ),
            (
                "par item",
                "def f(a: f32, b: f32) -> f32 = {\n  c = neq(a, a)\n  g = par {\n    if c\n    then a\n    else b\n  }\n  g\n}",
            ),
        ];
        for (label, src) in cases {
            assert!(
                parse_str(src).is_ok(),
                "{label}: the continuation rule must hold here too: {:?}",
                parse_str(src).err()
            );
        }
    }

    #[test]
    fn block_expression_continuation_set_is_exact() {
        // P12 selects exactly three safe continuations. Tokens that can head
        // an expression stay out, and so do non-selected infix operators: the
        // inability to begin an expression is necessary but not sufficient.
        assert!(Parser::is_block_expression_continuation(&TokenKind::Pipe));
        assert!(Parser::is_block_expression_continuation(&TokenKind::Then));
        assert!(Parser::is_block_expression_continuation(&TokenKind::Else));
        for excluded in [
            TokenKind::With,
            TokenKind::Match,
            TokenKind::If,
            TokenKind::Plus,
            TokenKind::Star,
            TokenKind::EqEq,
            TokenKind::AmpAmp,
            TokenKind::PipePipe,
        ] {
            assert!(!Parser::is_block_expression_continuation(&excluded));
        }
    }

    #[test]
    fn block_statement_after_an_if_else_is_not_swallowed() {
        // The continuation extends the statement only across the newline that
        // precedes `else`. A following binding must remain its own statement.
        let src = "def f(a: f32, b: f32) -> bool = {\n  c = neq(a, a)\n  d = if c then true\n  else lte(a, b)\n  d\n}";
        assert!(
            parse_str(src).is_ok(),
            "a statement after the if/else must still parse: {:?}",
            parse_str(src).err()
        );
    }

    #[test]
    fn is_single_letter_upper_predicate() {
        assert!(is_single_letter_upper("S"));
        assert!(is_single_letter_upper("K"));
        assert!(is_single_letter_upper("Z"));
        assert!(!is_single_letter_upper("s")); // lowercase
        assert!(!is_single_letter_upper("Foo")); // multi-letter
        assert!(!is_single_letter_upper("S1")); // letter+digit
        assert!(!is_single_letter_upper("")); // empty
        assert!(!is_single_letter_upper("_")); // underscore
    }
}
