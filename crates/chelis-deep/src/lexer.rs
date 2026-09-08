use crate::span::Span;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

/// Closed set of numeric literal suffixes recognized by both the Surf and
/// Deep lexers per `spec/04-type-system.md` §5.5 / `spec/02-surf-syntax.md`
/// §P10a / `spec/03-deep-syntax.md` §6.4.1. Float-typed suffixes attach to
/// either an integer or float literal token; integer-typed suffixes attach
/// to integer literal tokens only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum LiteralSuffix {
    F32,
    F64,
    Bf16,
    F16,
    I8,
    I16,
    I32,
    I64,
}

impl LiteralSuffix {
    /// Returns the lowercase suffix string (`"f32"`, `"i64"`, etc.).
    pub fn as_str(self) -> &'static str {
        match self {
            LiteralSuffix::F32 => "f32",
            LiteralSuffix::F64 => "f64",
            LiteralSuffix::Bf16 => "bf16",
            LiteralSuffix::F16 => "f16",
            LiteralSuffix::I8 => "i8",
            LiteralSuffix::I16 => "i16",
            LiteralSuffix::I32 => "i32",
            LiteralSuffix::I64 => "i64",
        }
    }

    /// True if the suffix binds to a float dtype.
    pub fn is_float(self) -> bool {
        matches!(
            self,
            LiteralSuffix::F32 | LiteralSuffix::F64 | LiteralSuffix::Bf16 | LiteralSuffix::F16
        )
    }

    /// True if the suffix binds to an integer dtype.
    pub fn is_integer(self) -> bool {
        matches!(
            self,
            LiteralSuffix::I8 | LiteralSuffix::I16 | LiteralSuffix::I32 | LiteralSuffix::I64
        )
    }

    /// Returns the canonical Deep `t-prim` precision name, e.g. `"int64"`.
    pub fn t_prim_name(self) -> &'static str {
        match self {
            LiteralSuffix::F32 => "f32",
            LiteralSuffix::F64 => "f64",
            LiteralSuffix::Bf16 => "bf16",
            LiteralSuffix::F16 => "f16",
            LiteralSuffix::I8 => "int8",
            LiteralSuffix::I16 => "int16",
            LiteralSuffix::I32 => "int32",
            LiteralSuffix::I64 => "int64",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    LParen,
    RParen,
    LBrace,
    RBrace,
    Caret,
    Symbol(String),
    Int(i64),
    Float(f64),
    /// Numeric literal carrying an explicit precision suffix per spec §5.5.
    /// The lexer emits this variant only when the suffix immediately
    /// follows the digit sequence with no intervening whitespace.
    TypedInt(i64, LiteralSuffix),
    TypedFloat(f64, LiteralSuffix),
    Str(String),
    /// Keyword without the leading `:`.
    Keyword(String),
    Bool(bool),
    Comma,
}

#[derive(Debug, Error)]
pub enum LexError {
    #[error("unterminated string starting at byte {offset}")]
    UnterminatedString { offset: usize },

    #[error("invalid escape sequence '\\{ch}' at byte {offset}")]
    InvalidEscape { ch: char, offset: usize },

    #[error("invalid number '{text}' at byte {offset}")]
    InvalidNumber { text: String, offset: usize },

    #[error("unexpected character '{ch}' at byte {offset}")]
    UnexpectedChar { ch: char, offset: usize },

    /// A suffix spelling one of the reserved-but-deferred dtype names of
    /// `spec/04-type-system.md` §1.1.1 (`f8e4m3`, `f8e5m2`, `int4`/`uint4`,
    /// `complex64`/`complex128`, `decimal128`/`decimal256`). No suffix
    /// exists for a deferred name; one is authored only when the dtype
    /// activates (§5.5).
    #[error(
        "invalid literal suffix `{suffix}` on `{literal}` at byte {offset}: \
         {suffix} is deferred per spec/04-type-system.md §1.1.1"
    )]
    DeferredSuffix {
        literal: String,
        suffix: String,
        offset: usize,
    },

    /// Unsigned integer suffix attached to a numeric literal. The `uint*`
    /// family is reserved-but-deferred per `spec/04-type-system.md`
    /// §1.1.1; the short `u*` spellings are not reserved at all (§1.1.2
    /// names `uint8`/`uint16`/`uint32`/`uint64` canonical).
    #[error(
        "invalid literal suffix `{suffix}` on `{literal}` at byte {offset}: \
         unsigned integer types are deferred per spec/04-type-system.md \
         §1.1.1 (canonical spelling uint8/uint16/uint32/uint64 per §1.1.2; \
         the short u* spellings are not reserved)"
    )]
    UnsignedSuffix {
        literal: String,
        suffix: String,
        offset: usize,
    },

    /// Integer-typed suffix (`i8`/`i16`/`i32`/`i64`) on a float literal,
    /// rejected per `spec/04-type-system.md` §5.5.
    #[error(
        "invalid literal suffix `{suffix}` on float literal `{literal}` at \
         byte {offset}: integer suffixes attach to integer literals only \
         (spec/04-type-system.md §5.5)"
    )]
    IntegerSuffixOnFloat {
        literal: String,
        suffix: String,
        offset: usize,
    },

    /// Float-typed suffix on a hex integer literal, rejected per
    /// `spec/04-type-system.md` §5.5 hex-suffix rule.
    #[error(
        "invalid literal suffix `{suffix}` on hex integer literal `{literal}` \
         at byte {offset}: hex literals cannot carry float-typed suffixes \
         (spec/04-type-system.md §5.5); use `cast({literal}, {suffix})` or \
         insert whitespace (`{literal} {suffix}`)"
    )]
    HexFloatSuffix {
        literal: String,
        suffix: String,
        offset: usize,
    },

    /// Unrecognized identifier sequence directly adjacent to a numeric
    /// literal, rejected per `spec/04-type-system.md` §5.5.
    #[error(
        "unrecognized literal suffix `{suffix}` on `{literal}` at byte \
         {offset}: only the closed set f32/f64/bf16/f16/i8/i16/i32/i64 is \
         valid (spec/04-type-system.md §5.5); insert whitespace if the \
         adjacency was unintentional"
    )]
    UnknownSuffix {
        literal: String,
        suffix: String,
        offset: usize,
    },
}

pub fn lex(source: &str) -> Result<Vec<Token>, LexError> {
    let mut tokens = Vec::new();
    let bytes = source.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        // Skip whitespace
        if bytes[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }

        // Skip line comments: ; to end of line
        if bytes[i] == b';' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }

        let start = i;

        match bytes[i] {
            b'(' => {
                tokens.push(Token {
                    kind: TokenKind::LParen,
                    span: Span::new(start, 1),
                });
                i += 1;
            }
            b')' => {
                tokens.push(Token {
                    kind: TokenKind::RParen,
                    span: Span::new(start, 1),
                });
                i += 1;
            }
            b'{' => {
                tokens.push(Token {
                    kind: TokenKind::LBrace,
                    span: Span::new(start, 1),
                });
                i += 1;
            }
            b'}' => {
                tokens.push(Token {
                    kind: TokenKind::RBrace,
                    span: Span::new(start, 1),
                });
                i += 1;
            }
            b',' => {
                tokens.push(Token {
                    kind: TokenKind::Comma,
                    span: Span::new(start, 1),
                });
                i += 1;
            }
            b'^' => {
                tokens.push(Token {
                    kind: TokenKind::Caret,
                    span: Span::new(start, 1),
                });
                i += 1;
            }
            b'"' => {
                let tok = lex_string(source, &mut i)?;
                tokens.push(tok);
            }
            b':' => {
                i += 1; // skip the ':'
                let kw_start = i;
                while i < bytes.len() && is_ident_continue(bytes[i]) {
                    i += 1;
                }
                if i == kw_start {
                    // Bare `:` -- no identifier chars followed
                    tokens.push(Token {
                        kind: TokenKind::Symbol(":".to_string()),
                        span: Span::new(start, 1),
                    });
                } else {
                    let name = &source[kw_start..i];
                    tokens.push(Token {
                        kind: TokenKind::Keyword(name.to_string()),
                        span: Span::new(start, i - start),
                    });
                }
            }
            b if b.is_ascii_digit() => {
                let tok = lex_number(source, &mut i)?;
                tokens.push(tok);
            }
            b'-' if i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() => {
                let tok = lex_number(source, &mut i)?;
                tokens.push(tok);
            }
            b'-' => {
                // '-' not followed by digit. Treat as start of symbol (e.g., `->`)
                while i < bytes.len() && is_symbol_char(bytes[i]) {
                    i += 1;
                }
                let text = &source[start..i];
                tokens.push(Token {
                    kind: TokenKind::Symbol(text.to_string()),
                    span: Span::new(start, i - start),
                });
            }
            b'*' => {
                // `*` is the canonical concrete wildcard dimension name in
                // `(d-name {} *)` (spec/03 §2.6). Its role is validated by
                // the structural type consumer; the lexer preserves the
                // spelling as one symbol rather than rejecting generated
                // canonical types before role validation.
                tokens.push(Token {
                    kind: TokenKind::Symbol("*".to_string()),
                    span: Span::new(start, 1),
                });
                i += 1;
            }
            b if is_ident_start(b) => {
                while i < bytes.len() && is_ident_continue(bytes[i]) {
                    i += 1;
                }
                let text = &source[start..i];
                let kind = match text {
                    "true" => TokenKind::Bool(true),
                    "false" => TokenKind::Bool(false),
                    _ => TokenKind::Symbol(text.to_string()),
                };
                tokens.push(Token {
                    kind,
                    span: Span::new(start, i - start),
                });
            }
            _ => {
                let ch = source[i..].chars().next().unwrap();
                return Err(LexError::UnexpectedChar { ch, offset: i });
            }
        }
    }

    Ok(tokens)
}

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

fn is_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.'
}

/// Characters valid in operator-like symbols (e.g., `->`, `_`).
fn is_symbol_char(b: u8) -> bool {
    is_ident_continue(b)
        || b == b'>'
        || b == b'<'
        || b == b'='
        || b == b'+'
        || b == b'*'
        || b == b'/'
        || b == b'!'
        || b == b'?'
}

fn lex_string(source: &str, i: &mut usize) -> Result<Token, LexError> {
    let start = *i;
    let bytes = source.as_bytes();
    *i += 1; // skip opening '"'
    let mut s = String::new();

    while *i < bytes.len() {
        match bytes[*i] {
            b'"' => {
                *i += 1;
                return Ok(Token {
                    kind: TokenKind::Str(s),
                    span: Span::new(start, *i - start),
                });
            }
            b'\\' => {
                *i += 1;
                if *i >= bytes.len() {
                    return Err(LexError::UnterminatedString { offset: start });
                }
                match bytes[*i] {
                    b'\\' => s.push('\\'),
                    b'"' => s.push('"'),
                    b'n' => s.push('\n'),
                    b't' => s.push('\t'),
                    b'r' => s.push('\r'),
                    b'0' => s.push('\0'),
                    other => {
                        return Err(LexError::InvalidEscape {
                            ch: other as char,
                            offset: *i,
                        });
                    }
                }
                *i += 1;
            }
            _ => {
                // Handle multi-byte UTF-8 correctly
                let ch = source[*i..].chars().next().unwrap();
                s.push(ch);
                *i += ch.len_utf8();
            }
        }
    }

    Err(LexError::UnterminatedString { offset: start })
}

fn lex_number(source: &str, i: &mut usize) -> Result<Token, LexError> {
    let start = *i;
    let bytes = source.as_bytes();

    // Handle negative sign
    if bytes[*i] == b'-' {
        *i += 1;
    }

    // Check for hex/binary prefix
    if *i + 1 < bytes.len() && bytes[*i] == b'0' {
        match bytes[*i + 1] {
            b'x' | b'X' => {
                *i += 2;
                while *i < bytes.len() && bytes[*i].is_ascii_hexdigit() {
                    *i += 1;
                }
                let text = &source[start..(*i)];
                let val = i64::from_str_radix(
                    text.trim_start_matches('-')
                        .trim_start_matches("0x")
                        .trim_start_matches("0X"),
                    16,
                )
                .map_err(|_| LexError::InvalidNumber {
                    text: text.to_string(),
                    offset: start,
                })?;
                let val = if text.starts_with('-') { -val } else { val };
                // Hex+float-suffix interaction per spec/04-type-system.md §5.5:
                // because `f` is a hex digit, maximal-munch hex lexing eats
                // `0xFFf32` as a single hex literal (= 0xFFf32 = 1048370).
                // To honor the spec rule that hex literals reject float-typed
                // suffixes, post-check the consumed digit sequence: if it
                // ends in a recognized float suffix (`f32`/`f64`/`f16`/`bf16`)
                // and there is at least one preceding hex digit, treat it as
                // a user-intended float-suffix on a hex literal and emit the
                // diagnostic.
                let hex_digits = text
                    .trim_start_matches('-')
                    .trim_start_matches("0x")
                    .trim_start_matches("0X");
                if let Some((prefix, suffix_str)) = detect_hex_float_suffix_tail(hex_digits) {
                    let neg_prefix = if text.starts_with('-') { "-" } else { "" };
                    return Err(LexError::HexFloatSuffix {
                        literal: format!("{neg_prefix}0x{prefix}"),
                        suffix: suffix_str.to_string(),
                        offset: start,
                    });
                }
                let suffix = lex_literal_suffix(source, i, text, start, /* on_hex = */ true)?;
                let kind = match suffix {
                    None => TokenKind::Int(val),
                    Some(s) => TokenKind::TypedInt(val, s),
                };
                return Ok(Token {
                    kind,
                    span: Span::new(start, *i - start),
                });
            }
            b'b' | b'B' => {
                *i += 2;
                while *i < bytes.len() && (bytes[*i] == b'0' || bytes[*i] == b'1') {
                    *i += 1;
                }
                let text = &source[start..(*i)];
                let val = i64::from_str_radix(
                    text.trim_start_matches('-')
                        .trim_start_matches("0b")
                        .trim_start_matches("0B"),
                    2,
                )
                .map_err(|_| LexError::InvalidNumber {
                    text: text.to_string(),
                    offset: start,
                })?;
                let val = if text.starts_with('-') { -val } else { val };
                let suffix = lex_literal_suffix(source, i, text, start, /* on_hex = */ false)?;
                let kind = match suffix {
                    None => TokenKind::Int(val),
                    Some(s) => TokenKind::TypedInt(val, s),
                };
                return Ok(Token {
                    kind,
                    span: Span::new(start, *i - start),
                });
            }
            _ => {}
        }
    }

    // Decimal digits
    while *i < bytes.len() && bytes[*i].is_ascii_digit() {
        *i += 1;
    }

    // Check for float: decimal point
    let mut is_float = false;
    if *i < bytes.len()
        && bytes[*i] == b'.'
        && (*i + 1 < bytes.len() && bytes[*i + 1].is_ascii_digit())
    {
        is_float = true;
        *i += 1; // skip '.'
        while *i < bytes.len() && bytes[*i].is_ascii_digit() {
            *i += 1;
        }
    }

    // Check for exponent (e/E) -- makes it a float even without decimal point.
    // Disambiguation vs. suffix: exponent only when followed by an optional
    // sign and at least one digit. `1e10f32` is digits-e-digits-suffix; a
    // bare `1e` (no exponent digits) falls through to suffix lex.
    if *i < bytes.len() && (bytes[*i] == b'e' || bytes[*i] == b'E') {
        let probe = *i + 1;
        let after_sign = if probe < bytes.len() && (bytes[probe] == b'+' || bytes[probe] == b'-') {
            probe + 1
        } else {
            probe
        };
        if after_sign < bytes.len() && bytes[after_sign].is_ascii_digit() {
            is_float = true;
            *i += 1;
            if *i < bytes.len() && (bytes[*i] == b'+' || bytes[*i] == b'-') {
                *i += 1;
            }
            while *i < bytes.len() && bytes[*i].is_ascii_digit() {
                *i += 1;
            }
        }
    }

    let text = &source[start..(*i)];
    let suffix = lex_literal_suffix(source, i, text, start, /* on_hex = */ false)?;
    if is_float {
        let val: f64 = text.parse().map_err(|_| LexError::InvalidNumber {
            text: text.to_string(),
            offset: start,
        })?;
        if let Some(s) = suffix {
            if s.is_integer() {
                return Err(LexError::IntegerSuffixOnFloat {
                    literal: text.to_string(),
                    suffix: s.as_str().to_string(),
                    offset: start,
                });
            }
            Ok(Token {
                kind: TokenKind::TypedFloat(val, s),
                span: Span::new(start, *i - start),
            })
        } else {
            Ok(Token {
                kind: TokenKind::Float(val),
                span: Span::new(start, *i - start),
            })
        }
    } else {
        let val: i64 = text.parse().map_err(|_| LexError::InvalidNumber {
            text: text.to_string(),
            offset: start,
        })?;
        let kind = match suffix {
            None => TokenKind::Int(val),
            Some(s) => TokenKind::TypedInt(val, s),
        };
        Ok(Token {
            kind,
            span: Span::new(start, *i - start),
        })
    }
}

/// Detect whether a hex digit sequence (without the `0x` prefix) ends in a
/// user-intended float-typed suffix. Returns `Some((prefix, suffix))`
/// where `prefix` is the hex digit sequence stripped of the suffix and
/// `suffix` is the matched suffix string. Used to surface the
/// hex+float-suffix diagnostic per `spec/04-type-system.md` §5.5 even
/// after the maximal-munch hex rule has eaten the suffix as hex digits.
///
/// Only matches when at least one hex digit precedes the suffix; a bare
/// `0xf32` would otherwise be ambiguous with a real hex literal.
fn detect_hex_float_suffix_tail(hex_digits: &str) -> Option<(&str, &str)> {
    // Order matters: longest first, so `bf16` matches before `f16`.
    for suffix in ["bf16", "f32", "f64", "f16"] {
        if hex_digits.len() > suffix.len() && hex_digits.ends_with(suffix) {
            let prefix = &hex_digits[..hex_digits.len() - suffix.len()];
            return Some((prefix, suffix));
        }
    }
    None
}

/// Consume an optional literal suffix following the digit sequence.
///
/// Per `spec/04-type-system.md` §5.5:
/// - the suffix must immediately follow the digits with no whitespace
/// - the closed set is `f32`/`f64`/`bf16`/`f16`/`i8`/`i16`/`i32`/`i64`
/// - every §1.1.1 reserved-but-deferred name (`f8e4m3`, `f8e5m2`,
///   `int4`/`uint4`, `complex*`, `decimal*`) is rejected at lex time
/// - `u8`/`u16`/`u32`/`u64` and the canonical `uint*` family are rejected
///   (deferred per §1.1.1; §1.1.2 names the `uint*` spellings canonical)
/// - any other adjacent identifier sequence is a parse error
/// - hex literals reject float-typed suffixes (the maximal-munch hex rule
///   has already swallowed any `f` digit) but accept integer-typed suffixes
///
/// `literal_text` is the digit-portion text used in diagnostics (without
/// the suffix). On success the cursor `i` is advanced past any consumed
/// suffix characters.
fn lex_literal_suffix(
    source: &str,
    i: &mut usize,
    literal_text: &str,
    literal_offset: usize,
    on_hex: bool,
) -> Result<Option<LiteralSuffix>, LexError> {
    let bytes = source.as_bytes();
    if *i >= bytes.len() {
        return Ok(None);
    }
    let first = bytes[*i];
    // Suffixes always start with an ASCII letter. Any other adjacency
    // (whitespace, punctuation) means there is no suffix.
    if !first.is_ascii_alphabetic() {
        return Ok(None);
    }
    // Greedily consume the adjacent identifier-like run. A suffix is an
    // alphanumeric sequence with no internal `.`/`-`/`_`. We deliberately
    // do NOT consume hyphens or underscores so `1.0_e3` keeps its
    // historical meaning (lexer already strips no underscores in deep,
    // but identifiers in deep can contain `-` and `.`; suffixes cannot).
    let suffix_start = *i;
    let mut probe = *i;
    while probe < bytes.len() && bytes[probe].is_ascii_alphanumeric() {
        probe += 1;
    }
    let suffix_text = &source[suffix_start..probe];
    let suffix = match suffix_text {
        "f32" => LiteralSuffix::F32,
        "f64" => LiteralSuffix::F64,
        "bf16" => LiteralSuffix::Bf16,
        "f16" => LiteralSuffix::F16,
        "i8" => LiteralSuffix::I8,
        "i16" => LiteralSuffix::I16,
        "i32" => LiteralSuffix::I32,
        "i64" => LiteralSuffix::I64,
        "f8e4m3" | "f8e5m2" | "int4" | "uint4" | "complex64" | "complex128" | "decimal128"
        | "decimal256" => {
            return Err(LexError::DeferredSuffix {
                literal: literal_text.to_string(),
                suffix: suffix_text.to_string(),
                offset: literal_offset,
            });
        }
        "u8" | "u16" | "u32" | "u64" | "uint8" | "uint16" | "uint32" | "uint64" => {
            return Err(LexError::UnsignedSuffix {
                literal: literal_text.to_string(),
                suffix: suffix_text.to_string(),
                offset: literal_offset,
            });
        }
        _ => {
            return Err(LexError::UnknownSuffix {
                literal: literal_text.to_string(),
                suffix: suffix_text.to_string(),
                offset: literal_offset,
            });
        }
    };
    if on_hex && suffix.is_float() {
        return Err(LexError::HexFloatSuffix {
            literal: literal_text.to_string(),
            suffix: suffix.as_str().to_string(),
            offset: literal_offset,
        });
    }
    *i = probe;
    Ok(Some(suffix))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex_kinds(s: &str) -> Vec<TokenKind> {
        lex(s).unwrap().into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn parens_and_braces() {
        assert_eq!(
            lex_kinds("(){}^"),
            vec![
                TokenKind::LParen,
                TokenKind::RParen,
                TokenKind::LBrace,
                TokenKind::RBrace,
                TokenKind::Caret,
            ]
        );
    }

    #[test]
    fn symbols() {
        assert_eq!(
            lex_kinds("foo bar_baz MyType x-y"),
            vec![
                TokenKind::Symbol("foo".into()),
                TokenKind::Symbol("bar_baz".into()),
                TokenKind::Symbol("MyType".into()),
                TokenKind::Symbol("x-y".into()),
            ]
        );
        assert_eq!(lex_kinds("*"), vec![TokenKind::Symbol("*".into())]);
    }

    #[test]
    fn booleans() {
        assert_eq!(
            lex_kinds("true false"),
            vec![TokenKind::Bool(true), TokenKind::Bool(false),]
        );
    }

    /// chelis#683: `i64::MIN` is writable as a Deep integer literal.
    ///
    /// Surf needs a dedicated `IntMinMagnitude` sentinel because its lexer
    /// reads the bare magnitude and the parser applies negation separately, so
    /// `2^63` overflows `i64` before the sign is known. Deep does not have that
    /// problem: `lex_number` captures `start` BEFORE the sign and then slices
    /// the token as `&source[start..i]`, so the string handed to
    /// `parse::<i64>()` is already signed and the bare magnitude is never
    /// parsed on its own. (Skipping the `-` is what lets the digit scan
    /// advance; it is the `start` capture that puts the sign in the slice.
    /// Both are load-bearing - drop either and every negative literal breaks.)
    ///
    /// Note the asymmetry inside this same function: the hex and binary arms
    /// DO strip the sign, parse the magnitude, and negate afterwards - the Surf
    /// shape - which is why hex above `i64::MAX` still fails. That is the
    /// issue's "Related" question and is deliberately left alone here.
    ///
    /// These cells pin the decimal behaviour, which is the reason no sentinel
    /// is mirrored into this crate.
    #[test]
    fn i64_boundary_literals_lex_exactly() {
        assert_eq!(
            lex_kinds("-9223372036854775808 9223372036854775807"),
            vec![TokenKind::Int(i64::MIN), TokenKind::Int(i64::MAX)]
        );
        assert_eq!(
            lex_kinds("-9223372036854775808i64 9223372036854775807i64"),
            vec![
                TokenKind::TypedInt(i64::MIN, LiteralSuffix::I64),
                TokenKind::TypedInt(i64::MAX, LiteralSuffix::I64),
            ]
        );
    }

    /// The negative half of the cell above: the UNSIGNED magnitude `2^63` is
    /// not an `i64` and must fail to lex. If this ever starts succeeding, the
    /// sign is no longer what makes `i64::MIN` representable and the reasoning
    /// in `i64_boundary_literals_lex_exactly` needs rechecking.
    #[test]
    fn unsigned_i64_min_magnitude_is_a_lex_error() {
        let err = lex("9223372036854775808").expect_err("2^63 is not an i64");
        assert!(
            matches!(err, LexError::InvalidNumber { .. }),
            "expected InvalidNumber, got: {err:?}"
        );
    }

    #[test]
    fn integers() {
        assert_eq!(
            lex_kinds("42 -7 0 0xFF 0b1010"),
            vec![
                TokenKind::Int(42),
                TokenKind::Int(-7),
                TokenKind::Int(0),
                TokenKind::Int(255),
                TokenKind::Int(10),
            ]
        );
    }

    #[test]
    fn floats() {
        assert_eq!(
            lex_kinds("3.125 -1.0 0.001 1.5e10 2.0E-3"),
            vec![
                TokenKind::Float(3.125),
                TokenKind::Float(-1.0),
                TokenKind::Float(0.001),
                TokenKind::Float(1.5e10),
                TokenKind::Float(2.0e-3),
            ]
        );
    }

    #[test]
    fn strings() {
        assert_eq!(
            lex_kinds(r#""hello" "line\n" "tab\there" "esc\"quote""#),
            vec![
                TokenKind::Str("hello".into()),
                TokenKind::Str("line\n".into()),
                TokenKind::Str("tab\there".into()),
                TokenKind::Str("esc\"quote".into()),
            ]
        );
    }

    #[test]
    fn keywords() {
        assert_eq!(
            lex_kinds(":axis :type :pure"),
            vec![
                TokenKind::Keyword("axis".into()),
                TokenKind::Keyword("type".into()),
                TokenKind::Keyword("pure".into()),
            ]
        );
    }

    #[test]
    fn comments_skipped() {
        assert_eq!(
            lex_kinds("; this is a comment\n42"),
            vec![TokenKind::Int(42)]
        );
    }

    #[test]
    fn inline_comments() {
        assert_eq!(
            lex_kinds("(add ; comment\n1 2)"),
            vec![
                TokenKind::LParen,
                TokenKind::Symbol("add".into()),
                TokenKind::Int(1),
                TokenKind::Int(2),
                TokenKind::RParen,
            ]
        );
    }

    #[test]
    fn realistic_expr() {
        // (def square (sig (-> f32 f32)) (fn (x) (apply mul x x)))
        // Count: ( def square ( sig ( -> f32 f32 ) ) ( fn ( x ) ( apply mul x x ) ) )
        //        1  2    3    4  5  6  7  8   9  10 11 12 13 14 15 16 17  18  19 20 21 22 23 24
        let tokens = lex("(def square (sig (-> f32 f32)) (fn (x) (apply mul x x)))").unwrap();
        assert_eq!(tokens.len(), 24);
        assert_eq!(tokens[0].kind, TokenKind::LParen);
        assert_eq!(tokens[1].kind, TokenKind::Symbol("def".into()));
        assert_eq!(tokens[2].kind, TokenKind::Symbol("square".into()));
        assert_eq!(tokens[6].kind, TokenKind::Symbol("->".into()));
    }

    #[test]
    fn metadata_tokens() {
        let tokens = lex("^{:type f32} x").unwrap();
        assert_eq!(tokens[0].kind, TokenKind::Caret);
        assert_eq!(tokens[1].kind, TokenKind::LBrace);
        assert_eq!(tokens[2].kind, TokenKind::Keyword("type".into()));
        assert_eq!(tokens[3].kind, TokenKind::Symbol("f32".into()));
        assert_eq!(tokens[4].kind, TokenKind::RBrace);
        assert_eq!(tokens[5].kind, TokenKind::Symbol("x".into()));
    }

    #[test]
    fn unterminated_string() {
        assert!(matches!(
            lex(r#""hello"#),
            Err(LexError::UnterminatedString { .. })
        ));
    }

    #[test]
    fn invalid_escape() {
        assert!(matches!(
            lex(r#""\q""#),
            Err(LexError::InvalidEscape { .. })
        ));
    }

    #[test]
    fn unexpected_char() {
        assert!(matches!(lex("@"), Err(LexError::UnexpectedChar { .. })));
    }

    #[test]
    fn exponent_only_floats() {
        assert_eq!(lex_kinds("-1e-5"), vec![TokenKind::Float(-1e-5)]);
        assert_eq!(lex_kinds("1e10"), vec![TokenKind::Float(1e10)]);
        assert_eq!(lex_kinds("5E3"), vec![TokenKind::Float(5e3)]);
    }

    #[test]
    fn bare_colon_is_symbol() {
        assert_eq!(
            lex_kinds("(: 42 i32)"),
            vec![
                TokenKind::LParen,
                TokenKind::Symbol(":".into()),
                TokenKind::Int(42),
                TokenKind::Symbol("i32".into()),
                TokenKind::RParen,
            ]
        );
    }

    // --- WS-B1: literal suffix grammar (spec/03-deep-syntax.md §6.4.1) ---

    #[test]
    fn typed_int_suffixes_each_member_of_closed_set() {
        assert_eq!(
            lex_kinds("42i8 42i16 42i32 42i64"),
            vec![
                TokenKind::TypedInt(42, LiteralSuffix::I8),
                TokenKind::TypedInt(42, LiteralSuffix::I16),
                TokenKind::TypedInt(42, LiteralSuffix::I32),
                TokenKind::TypedInt(42, LiteralSuffix::I64),
            ]
        );
    }

    #[test]
    fn typed_float_suffixes_each_member_of_closed_set() {
        assert_eq!(
            lex_kinds("1.0f32 1.0f64 1.0bf16 1.0f16"),
            vec![
                TokenKind::TypedFloat(1.0, LiteralSuffix::F32),
                TokenKind::TypedFloat(1.0, LiteralSuffix::F64),
                TokenKind::TypedFloat(1.0, LiteralSuffix::Bf16),
                TokenKind::TypedFloat(1.0, LiteralSuffix::F16),
            ]
        );
    }

    #[test]
    fn float_typed_suffix_attaches_to_integer_literal() {
        assert_eq!(
            lex_kinds("42f32 42f64 42bf16 42f16"),
            vec![
                TokenKind::TypedInt(42, LiteralSuffix::F32),
                TokenKind::TypedInt(42, LiteralSuffix::F64),
                TokenKind::TypedInt(42, LiteralSuffix::Bf16),
                TokenKind::TypedInt(42, LiteralSuffix::F16),
            ]
        );
    }

    #[test]
    fn integer_suffix_on_float_is_lex_error() {
        let err = lex("1.0i8").unwrap_err();
        assert!(
            matches!(err, LexError::IntegerSuffixOnFloat { ref suffix, .. } if suffix == "i8"),
            "expected IntegerSuffixOnFloat, got {err:?}"
        );
    }

    #[test]
    fn deferred_suffix_is_lex_error() {
        for (src, name) in [
            ("1.0f8e4m3", "f8e4m3"),
            ("1.0f8e5m2", "f8e5m2"),
            ("42int4", "int4"),
            ("42uint4", "uint4"),
            ("1.0complex64", "complex64"),
            ("1.0complex128", "complex128"),
            ("1.0decimal128", "decimal128"),
            ("1.0decimal256", "decimal256"),
        ] {
            let err = lex(src).unwrap_err();
            assert!(
                matches!(err, LexError::DeferredSuffix { ref suffix, .. } if suffix == name),
                "expected DeferredSuffix for {src}, got {err:?}"
            );
            let msg = format!("{err}");
            assert!(
                msg.contains(&format!("{name} is deferred")) && msg.contains("§1.1.1"),
                "diagnostic must cite §1.1.1 and contain '{name} is deferred', got: {msg}"
            );
        }
    }

    #[test]
    fn unsigned_suffix_is_lex_error() {
        for src in [
            "42u8", "42u16", "42u32", "42u64", "42uint8", "42uint16", "42uint32", "42uint64",
        ] {
            let err = lex(src).unwrap_err();
            assert!(
                matches!(err, LexError::UnsignedSuffix { .. }),
                "expected UnsignedSuffix for {src}, got {err:?}"
            );
            let msg = format!("{err}");
            assert!(
                msg.contains("unsigned integer types are deferred") && msg.contains("§1.1.1"),
                "diagnostic must cite §1.1.1 and contain 'unsigned integer types are deferred', \
                 got: {msg}"
            );
        }
    }

    #[test]
    fn unknown_suffix_is_lex_error() {
        let err = lex("1.0xyz").unwrap_err();
        assert!(
            matches!(err, LexError::UnknownSuffix { ref suffix, .. } if suffix == "xyz"),
            "expected UnknownSuffix, got {err:?}"
        );
    }

    #[test]
    fn hex_integer_suffix_is_typed_int() {
        assert_eq!(
            lex_kinds("0xFFi8 0xFFi32"),
            vec![
                TokenKind::TypedInt(0xFF, LiteralSuffix::I8),
                TokenKind::TypedInt(0xFF, LiteralSuffix::I32),
            ]
        );
    }

    #[test]
    fn hex_float_suffix_is_lex_error() {
        let err = lex("0xFFf32").unwrap_err();
        assert!(
            matches!(err, LexError::HexFloatSuffix { .. }),
            "expected HexFloatSuffix, got {err:?}"
        );
    }

    #[test]
    fn whitespace_before_suffix_is_two_tokens() {
        assert_eq!(
            lex_kinds("1.0 f32"),
            vec![TokenKind::Float(1.0), TokenKind::Symbol("f32".into()),]
        );
    }

    #[test]
    fn scientific_notation_with_float_suffix() {
        assert_eq!(
            lex_kinds("3.14e-2f32 1e10f64"),
            vec![
                TokenKind::TypedFloat(3.14e-2, LiteralSuffix::F32),
                TokenKind::TypedFloat(1e10, LiteralSuffix::F64),
            ]
        );
    }
}
