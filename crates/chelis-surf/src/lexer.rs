use chelis_deep::{LiteralSuffix, Span};
use thiserror::Error;

use crate::token::{Token, TokenKind};

#[derive(Debug, Error)]
pub enum LexError {
    #[error("unterminated string starting at byte {offset}")]
    UnterminatedString { offset: usize },

    #[error("invalid escape sequence '\\{ch}' at byte {offset}")]
    InvalidEscape { ch: char, offset: usize },

    #[error("unescaped control character U+{code:04X} in string at byte {offset}")]
    UnescapedControl { code: u32, offset: usize },

    #[error("invalid number '{text}' at byte {offset}")]
    InvalidNumber { text: String, offset: usize },

    #[error("unexpected character '{ch}' at byte {offset}")]
    UnexpectedChar { ch: char, offset: usize },

    #[error("keyword '{keyword}' is reserved for future use at byte {offset}")]
    ReservedForFuture { keyword: String, offset: usize },

    #[error("unterminated block comment starting at byte {offset}")]
    UnterminatedBlockComment { offset: usize },

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

    /// Integer-typed suffix on a float literal, per
    /// `spec/04-type-system.md` §5.5: `1.0i8` is a parse error.
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

    /// Float-typed suffix on a hex integer literal, per
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
    /// literal, per `spec/04-type-system.md` §5.5.
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

/// A source comment captured for round-trip formatting.
///
/// Comments are not part of the token stream the parser consumes — they
/// carry no semantics — but `chelis fmt` must re-emit them, so the
/// lexer records them on a side channel keyed by source span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Comment {
    /// The verbatim comment text including its delimiters: a `--` line
    /// comment keeps its leading `--`; a `{- -}` block comment keeps
    /// both braces.
    pub text: String,
    /// Byte span of the comment in the original source.
    pub span: Span,
    pub kind: CommentKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommentKind {
    /// `-- ...` to end of line.
    Line,
    /// `{- ... -}`, possibly nested.
    Block,
}

pub fn lex(source: &str) -> Result<Vec<Token>, LexError> {
    lex_inner(source, None)
}

/// Lex `source`, returning the token stream plus every comment in
/// source order. The token stream is identical to [`lex`]'s — comments
/// never become tokens — so the parser is unaffected.
pub fn lex_with_comments(source: &str) -> Result<(Vec<Token>, Vec<Comment>), LexError> {
    let mut comments = Vec::new();
    let tokens = lex_inner(source, Some(&mut comments))?;
    Ok((tokens, comments))
}

fn lex_inner(
    source: &str,
    mut comments: Option<&mut Vec<Comment>>,
) -> Result<Vec<Token>, LexError> {
    let mut tokens = Vec::new();
    let bytes = source.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        // Preserve newlines for block/par separator enforcement.
        if bytes[i] == b'\n' {
            tokens.push(Token {
                kind: TokenKind::Newline,
                span: Span::new(i, 1),
            });
            i += 1;
            continue;
        }
        if bytes[i] == b'\r' {
            let len = if i + 1 < bytes.len() && bytes[i + 1] == b'\n' {
                2
            } else {
                1
            };
            tokens.push(Token {
                kind: TokenKind::Newline,
                span: Span::new(i, len),
            });
            i += len;
            continue;
        }

        // Skip other whitespace
        if bytes[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }

        // Line comments: -- to end of line
        if i + 1 < bytes.len() && bytes[i] == b'-' && bytes[i + 1] == b'-' {
            let comment_start = i;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            if let Some(comments) = comments.as_deref_mut() {
                comments.push(Comment {
                    text: source[comment_start..i].to_string(),
                    span: Span::new(comment_start, i - comment_start),
                    kind: CommentKind::Line,
                });
            }
            continue;
        }

        // Block comments: {- ... -} (nested)
        if i + 1 < bytes.len() && bytes[i] == b'{' && bytes[i + 1] == b'-' {
            let comment_start = i;
            i += 2;
            let mut depth = 1u32;
            while i < bytes.len() && depth > 0 {
                if i + 1 < bytes.len() && bytes[i] == b'{' && bytes[i + 1] == b'-' {
                    depth += 1;
                    i += 2;
                } else if i + 1 < bytes.len() && bytes[i] == b'-' && bytes[i + 1] == b'}' {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if depth > 0 {
                return Err(LexError::UnterminatedBlockComment {
                    offset: comment_start,
                });
            }
            if let Some(comments) = comments.as_deref_mut() {
                comments.push(Comment {
                    text: source[comment_start..i].to_string(),
                    span: Span::new(comment_start, i - comment_start),
                    kind: CommentKind::Block,
                });
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
            b'[' => {
                tokens.push(Token {
                    kind: TokenKind::LBracket,
                    span: Span::new(start, 1),
                });
                i += 1;
            }
            b']' => {
                tokens.push(Token {
                    kind: TokenKind::RBracket,
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
            b':' => {
                tokens.push(Token {
                    kind: TokenKind::Colon,
                    span: Span::new(start, 1),
                });
                i += 1;
            }
            b'+' => {
                tokens.push(Token {
                    kind: TokenKind::Plus,
                    span: Span::new(start, 1),
                });
                i += 1;
            }
            b'*' => {
                tokens.push(Token {
                    kind: TokenKind::Star,
                    span: Span::new(start, 1),
                });
                i += 1;
            }
            b'/' => {
                tokens.push(Token {
                    kind: TokenKind::Slash,
                    span: Span::new(start, 1),
                });
                i += 1;
            }
            b'%' => {
                tokens.push(Token {
                    kind: TokenKind::Percent,
                    span: Span::new(start, 1),
                });
                i += 1;
            }
            b'-' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'>' {
                    tokens.push(Token {
                        kind: TokenKind::Arrow,
                        span: Span::new(start, 2),
                    });
                    i += 2;
                } else {
                    tokens.push(Token {
                        kind: TokenKind::Minus,
                        span: Span::new(start, 1),
                    });
                    i += 1;
                }
            }
            b'|' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'>' {
                    tokens.push(Token {
                        kind: TokenKind::Pipe,
                        span: Span::new(start, 2),
                    });
                    i += 2;
                } else if i + 1 < bytes.len() && bytes[i + 1] == b'|' {
                    tokens.push(Token {
                        kind: TokenKind::PipePipe,
                        span: Span::new(start, 2),
                    });
                    i += 2;
                } else {
                    tokens.push(Token {
                        kind: TokenKind::Bar,
                        span: Span::new(start, 1),
                    });
                    i += 1;
                }
            }
            b'=' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                    tokens.push(Token {
                        kind: TokenKind::EqEq,
                        span: Span::new(start, 2),
                    });
                    i += 2;
                } else if i + 1 < bytes.len() && bytes[i + 1] == b'>' {
                    tokens.push(Token {
                        kind: TokenKind::FatArrow,
                        span: Span::new(start, 2),
                    });
                    i += 2;
                } else {
                    tokens.push(Token {
                        kind: TokenKind::Eq,
                        span: Span::new(start, 1),
                    });
                    i += 1;
                }
            }
            b'!' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                    tokens.push(Token {
                        kind: TokenKind::BangEq,
                        span: Span::new(start, 2),
                    });
                    i += 2;
                } else {
                    tokens.push(Token {
                        kind: TokenKind::Bang,
                        span: Span::new(start, 1),
                    });
                    i += 1;
                }
            }
            b'<' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                    tokens.push(Token {
                        kind: TokenKind::LtEq,
                        span: Span::new(start, 2),
                    });
                    i += 2;
                } else {
                    tokens.push(Token {
                        kind: TokenKind::Lt,
                        span: Span::new(start, 1),
                    });
                    i += 1;
                }
            }
            b'>' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'=' {
                    tokens.push(Token {
                        kind: TokenKind::GtEq,
                        span: Span::new(start, 2),
                    });
                    i += 2;
                } else {
                    tokens.push(Token {
                        kind: TokenKind::Gt,
                        span: Span::new(start, 1),
                    });
                    i += 1;
                }
            }
            b'&' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'&' {
                    tokens.push(Token {
                        kind: TokenKind::AmpAmp,
                        span: Span::new(start, 2),
                    });
                    i += 2;
                } else {
                    tokens.push(Token {
                        kind: TokenKind::Amp,
                        span: Span::new(start, 1),
                    });
                    i += 1;
                }
            }
            b'.' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'.' {
                    // `..` is the rank-variable spread marker (`tensor[..r, p]`).
                    // Chelis has no range syntax, so two dots are unambiguous.
                    tokens.push(Token {
                        kind: TokenKind::DotDot,
                        span: Span::new(start, 2),
                    });
                    i += 2;
                } else {
                    tokens.push(Token {
                        kind: TokenKind::Dot,
                        span: Span::new(start, 1),
                    });
                    i += 1;
                }
            }
            b'@' => {
                tokens.push(Token {
                    kind: TokenKind::At,
                    span: Span::new(start, 1),
                });
                i += 1;
            }
            b';' => {
                tokens.push(Token {
                    kind: TokenKind::Semicolon,
                    span: Span::new(start, 1),
                });
                i += 1;
            }
            b'"' => {
                let tok = lex_string(source, &mut i)?;
                tokens.push(tok);
            }
            b if b.is_ascii_digit() => {
                let tok = lex_number(source, &mut i)?;
                tokens.push(tok);
            }
            b'_' => {
                // Check if standalone underscore vs identifier starting with _
                if i + 1 < bytes.len() && is_ident_continue(bytes[i + 1]) {
                    // _foo style identifier
                    while i < bytes.len() && is_ident_continue(bytes[i]) {
                        i += 1;
                    }
                    let text = &source[start..i];
                    let kind = classify_ident(text);
                    tokens.push(Token {
                        kind,
                        span: Span::new(start, i - start),
                    });
                } else {
                    tokens.push(Token {
                        kind: TokenKind::Underscore,
                        span: Span::new(start, 1),
                    });
                    i += 1;
                }
            }
            b if b.is_ascii_alphabetic() => {
                while i < bytes.len() && is_ident_continue(bytes[i]) {
                    i += 1;
                }
                let text = &source[start..i];
                if is_future_reserved(text) {
                    return Err(LexError::ReservedForFuture {
                        keyword: text.to_string(),
                        offset: start,
                    });
                }
                let kind = classify_ident(text);
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

fn is_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn is_future_reserved(text: &str) -> bool {
    matches!(text, "effect" | "handler" | "perform" | "resume" | "borrow")
}

fn classify_ident(text: &str) -> TokenKind {
    match text {
        "def" => TokenKind::Def,
        "sig" => TokenKind::Sig,
        "type" => TokenKind::Type,
        "dim" => TokenKind::Dim,
        "macro" => TokenKind::Macro,
        "match" => TokenKind::Match,
        "with" => TokenKind::With,
        "fn" => TokenKind::Fn,
        "module" => TokenKind::Module,
        "import" => TokenKind::Import,
        "if" => TokenKind::If,
        "then" => TokenKind::Then,
        "else" => TokenKind::Else,
        "grad" => TokenKind::Grad,
        "vmap" => TokenKind::Vmap,
        "jit" => TokenKind::Jit,
        "realize" => TokenKind::Realize,
        "copy" => TokenKind::Copy,
        "tensor" => TokenKind::Tensor,
        "cast" => TokenKind::Cast,
        "cast_trunc" => TokenKind::NamedCast(chelis_deep::NamedCastMode::Trunc),
        "export" => TokenKind::Export,
        "par" => TokenKind::Par,
        "do" => TokenKind::Do,
        "quote" => TokenKind::Quote,
        "unquote" => TokenKind::Unquote,
        "splice" => TokenKind::Splice,
        "true" => TokenKind::True,
        "false" => TokenKind::False,
        _ => {
            let first = text.chars().next().unwrap();
            if first.is_ascii_uppercase() {
                TokenKind::TypeIdent(text.to_string())
            } else {
                TokenKind::Ident(text.to_string())
            }
        }
    }
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
                    b'u' => {
                        *i += 1;
                        if *i >= bytes.len() || bytes[*i] != b'{' {
                            return Err(LexError::InvalidEscape {
                                ch: 'u',
                                offset: *i - 1,
                            });
                        }
                        *i += 1;
                        let digits_start = *i;
                        while *i < bytes.len() && bytes[*i].is_ascii_hexdigit() {
                            *i += 1;
                        }
                        if digits_start == *i || *i >= bytes.len() || bytes[*i] != b'}' {
                            return Err(LexError::InvalidEscape {
                                ch: 'u',
                                offset: digits_start.saturating_sub(2),
                            });
                        }
                        let scalar = u32::from_str_radix(&source[digits_start..*i], 16)
                            .ok()
                            .and_then(char::from_u32)
                            .ok_or(LexError::InvalidEscape {
                                ch: 'u',
                                offset: digits_start.saturating_sub(2),
                            })?;
                        s.push(scalar);
                    }
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
                let ch = source[*i..].chars().next().unwrap();
                if ch.is_control() {
                    return Err(LexError::UnescapedControl {
                        code: ch as u32,
                        offset: *i,
                    });
                }
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

    // Check for hex/binary prefix
    if *i + 1 < bytes.len() && bytes[*i] == b'0' {
        match bytes[*i + 1] {
            b'x' | b'X' => {
                *i += 2;
                while *i < bytes.len() && (bytes[*i].is_ascii_hexdigit() || bytes[*i] == b'_') {
                    *i += 1;
                }
                let text = &source[start..*i];
                if !underscores_separate_digits(&text[2..], |byte| byte.is_ascii_hexdigit()) {
                    return Err(LexError::InvalidNumber {
                        text: text.to_string(),
                        offset: start,
                    });
                }
                let digits: String = text[2..].chars().filter(|c| *c != '_').collect();
                let val =
                    i64::from_str_radix(&digits, 16).map_err(|_| LexError::InvalidNumber {
                        text: text.to_string(),
                        offset: start,
                    })?;
                // Hex+float-suffix interaction per spec/04-type-system.md §5.5:
                // because `f` is a hex digit, `0xFFf32` is consumed by the
                // maximal-munch hex rule. Detect the user-intended pattern
                // (hex digits ending in a recognized float-suffix tail) and
                // emit the hex+float-suffix diagnostic.
                if let Some((prefix, suffix_str)) = detect_hex_float_suffix_tail(&digits) {
                    return Err(LexError::HexFloatSuffix {
                        literal: format!("0x{prefix}"),
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
                while *i < bytes.len()
                    && (bytes[*i] == b'0' || bytes[*i] == b'1' || bytes[*i] == b'_')
                {
                    *i += 1;
                }
                let text = &source[start..*i];
                if !underscores_separate_digits(&text[2..], |byte| matches!(byte, b'0' | b'1')) {
                    return Err(LexError::InvalidNumber {
                        text: text.to_string(),
                        offset: start,
                    });
                }
                let digits: String = text[2..].chars().filter(|c| *c != '_').collect();
                let val = i64::from_str_radix(&digits, 2).map_err(|_| LexError::InvalidNumber {
                    text: text.to_string(),
                    offset: start,
                })?;
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

    // Decimal digits (with underscores)
    while *i < bytes.len() && (bytes[*i].is_ascii_digit() || bytes[*i] == b'_') {
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
        while *i < bytes.len() && (bytes[*i].is_ascii_digit() || bytes[*i] == b'_') {
            *i += 1;
        }
    }

    // Check for exponent (e/E) -- makes it a float even without decimal point.
    // Disambiguation vs. suffix: only treat `e`/`E` as exponent when
    // followed (after optional sign) by at least one digit. `1e10f32` is
    // exponent + suffix; a bare `1e` falls through to the suffix path.
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
            while *i < bytes.len() && (bytes[*i].is_ascii_digit() || bytes[*i] == b'_') {
                *i += 1;
            }
        }
    }

    let text = &source[start..*i];
    if !underscores_separate_digits(text, |byte| byte.is_ascii_digit()) {
        return Err(LexError::InvalidNumber {
            text: text.to_string(),
            offset: start,
        });
    }
    let clean: String = text.chars().filter(|c| *c != '_').collect();
    let suffix = lex_literal_suffix(source, i, text, start, /* on_hex = */ false)?;
    if is_float {
        let val: f64 = clean.parse().map_err(|_| LexError::InvalidNumber {
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
        if clean == "9223372036854775808" && matches!(suffix, None | Some(LiteralSuffix::I64)) {
            return Ok(Token {
                kind: TokenKind::IntMinMagnitude(suffix),
                span: Span::new(start, *i - start),
            });
        }
        let val: i64 = clean.parse().map_err(|_| LexError::InvalidNumber {
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

fn underscores_separate_digits(text: &str, is_digit: impl Fn(u8) -> bool) -> bool {
    let bytes = text.as_bytes();
    bytes.iter().enumerate().all(|(index, byte)| {
        *byte != b'_'
            || (index > 0
                && index + 1 < bytes.len()
                && is_digit(bytes[index - 1])
                && is_digit(bytes[index + 1]))
    })
}

/// Detect whether a hex digit sequence (without the `0x` prefix) ends in
/// a user-intended float-typed suffix. Returns `Some((prefix, suffix))`
/// where `prefix` is the hex digits stripped of the suffix and `suffix`
/// is the matched suffix string. Surfaces the hex+float-suffix
/// diagnostic per `spec/04-type-system.md` §5.5 even after the
/// maximal-munch hex rule has eaten the suffix as hex digits.
fn detect_hex_float_suffix_tail(hex_digits: &str) -> Option<(&str, &str)> {
    for suffix in ["bf16", "f32", "f64", "f16"] {
        if hex_digits.len() > suffix.len() && hex_digits.ends_with(suffix) {
            let prefix = &hex_digits[..hex_digits.len() - suffix.len()];
            return Some((prefix, suffix));
        }
    }
    None
}

/// Consume an optional literal suffix following the digit sequence per
/// `spec/04-type-system.md` §5.5. Mirrors `chelis_deep::lexer`.
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
    if !bytes[*i].is_ascii_alphabetic() {
        return Ok(None);
    }
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

    // ===== Keywords =====

    #[test]
    fn all_keywords() {
        assert_eq!(
            lex_kinds(
                "def sig type dim macro match with fn module import if then else grad vmap jit realize copy tensor cast cast_trunc export par do quote unquote splice"
            ),
            vec![
                TokenKind::Def,
                TokenKind::Sig,
                TokenKind::Type,
                TokenKind::Dim,
                TokenKind::Macro,
                TokenKind::Match,
                TokenKind::With,
                TokenKind::Fn,
                TokenKind::Module,
                TokenKind::Import,
                TokenKind::If,
                TokenKind::Then,
                TokenKind::Else,
                TokenKind::Grad,
                TokenKind::Vmap,
                TokenKind::Jit,
                TokenKind::Realize,
                TokenKind::Copy,
                TokenKind::Tensor,
                TokenKind::Cast,
                TokenKind::NamedCast(chelis_deep::NamedCastMode::Trunc),
                TokenKind::Export,
                TokenKind::Par,
                TokenKind::Do,
                TokenKind::Quote,
                TokenKind::Unquote,
                TokenKind::Splice,
            ]
        );
    }

    #[test]
    fn true_false_keywords() {
        assert_eq!(
            lex_kinds("true false"),
            vec![TokenKind::True, TokenKind::False]
        );
    }

    // ===== Two-char operators =====

    #[test]
    fn two_char_operators() {
        assert_eq!(
            lex_kinds("-> |> == != <= >= && ||"),
            vec![
                TokenKind::Arrow,
                TokenKind::Pipe,
                TokenKind::EqEq,
                TokenKind::BangEq,
                TokenKind::LtEq,
                TokenKind::GtEq,
                TokenKind::AmpAmp,
                TokenKind::PipePipe,
            ]
        );
    }

    // ===== Single-char operators and punctuation =====

    #[test]
    fn single_char_punctuation() {
        assert_eq!(
            lex_kinds("( ) { } [ ] , : = | + - * / % < > !"),
            vec![
                TokenKind::LParen,
                TokenKind::RParen,
                TokenKind::LBrace,
                TokenKind::RBrace,
                TokenKind::LBracket,
                TokenKind::RBracket,
                TokenKind::Comma,
                TokenKind::Colon,
                TokenKind::Eq,
                TokenKind::Bar,
                TokenKind::Plus,
                TokenKind::Minus,
                TokenKind::Star,
                TokenKind::Slash,
                TokenKind::Percent,
                TokenKind::Lt,
                TokenKind::Gt,
                TokenKind::Bang,
            ]
        );
    }

    // ===== Comments =====

    #[test]
    fn line_comment() {
        assert_eq!(
            lex_kinds("42 -- this is a comment\n7"),
            vec![TokenKind::Int(42), TokenKind::Newline, TokenKind::Int(7)]
        );
    }

    #[test]
    fn line_comment_at_eof() {
        assert_eq!(
            lex_kinds("42 -- trailing comment"),
            vec![TokenKind::Int(42)]
        );
    }

    #[test]
    fn nested_block_comments() {
        assert_eq!(
            lex_kinds("{- outer {- inner -} still outer -} 42"),
            vec![TokenKind::Int(42)]
        );
    }

    #[test]
    fn block_comment_simple() {
        assert_eq!(
            lex_kinds("1 {- comment -} 2"),
            vec![TokenKind::Int(1), TokenKind::Int(2)]
        );
    }

    #[test]
    fn newline_tokens_preserved() {
        assert_eq!(
            lex_kinds("x = 1\nlet = 2\r\nin = 3"),
            vec![
                TokenKind::Ident("x".into()),
                TokenKind::Eq,
                TokenKind::Int(1),
                TokenKind::Newline,
                TokenKind::Ident("let".into()),
                TokenKind::Eq,
                TokenKind::Int(2),
                TokenKind::Newline,
                TokenKind::Ident("in".into()),
                TokenKind::Eq,
                TokenKind::Int(3),
            ]
        );
    }

    #[test]
    fn unterminated_block_comment() {
        assert!(matches!(
            lex("{- unclosed"),
            Err(LexError::UnterminatedBlockComment { .. })
        ));
    }

    #[test]
    fn lex_with_comments_captures_line_and_block() {
        let (tokens, comments) = lex_with_comments("1 -- line one\n{- block -} 2").expect("lex");
        // The token stream is unchanged — comments never become tokens.
        assert_eq!(
            tokens.iter().map(|t| t.kind.clone()).collect::<Vec<_>>(),
            vec![TokenKind::Int(1), TokenKind::Newline, TokenKind::Int(2)]
        );
        assert_eq!(comments.len(), 2);
        assert_eq!(comments[0].text, "-- line one");
        assert_eq!(comments[0].kind, CommentKind::Line);
        assert_eq!(comments[1].text, "{- block -}");
        assert_eq!(comments[1].kind, CommentKind::Block);
    }

    #[test]
    fn lex_with_comments_captures_nested_block_verbatim() {
        let (_tokens, comments) =
            lex_with_comments("{- outer {- inner -} still outer -} 42").expect("lex");
        assert_eq!(comments.len(), 1);
        assert_eq!(comments[0].text, "{- outer {- inner -} still outer -}");
        assert_eq!(comments[0].kind, CommentKind::Block);
    }

    #[test]
    fn lex_with_comments_token_stream_matches_plain_lex() {
        let src = "module Foo\n-- doc\ndef f() -> f32 = cast(1.0, f32)\n";
        let plain = lex(src).expect("lex");
        let (with_comments, _) = lex_with_comments(src).expect("lex_with_comments");
        assert_eq!(plain, with_comments);
    }

    // ===== Ident vs TypeIdent =====

    #[test]
    fn ident_vs_type_ident() {
        assert_eq!(
            lex_kinds("foo Bar baz_quux MyType"),
            vec![
                TokenKind::Ident("foo".into()),
                TokenKind::TypeIdent("Bar".into()),
                TokenKind::Ident("baz_quux".into()),
                TokenKind::TypeIdent("MyType".into()),
            ]
        );
    }

    // ===== Underscore handling =====

    #[test]
    fn standalone_underscore() {
        assert_eq!(lex_kinds("_"), vec![TokenKind::Underscore]);
    }

    #[test]
    fn underscore_ident() {
        assert_eq!(
            lex_kinds("_foo _unused"),
            vec![
                TokenKind::Ident("_foo".into()),
                TokenKind::Ident("_unused".into()),
            ]
        );
    }

    #[test]
    fn underscore_in_pattern() {
        assert_eq!(
            lex_kinds("_ x"),
            vec![TokenKind::Underscore, TokenKind::Ident("x".into())]
        );
    }

    // ===== Number literals =====

    #[test]
    fn integers() {
        assert_eq!(
            lex_kinds("42 0 1000"),
            vec![TokenKind::Int(42), TokenKind::Int(0), TokenKind::Int(1000),]
        );
    }

    #[test]
    fn hex_literals() {
        assert_eq!(
            lex_kinds("0xFF 0X1A"),
            vec![TokenKind::Int(255), TokenKind::Int(26)]
        );
    }

    #[test]
    fn binary_literals() {
        assert_eq!(
            lex_kinds("0b1010 0B1100"),
            vec![TokenKind::Int(10), TokenKind::Int(12)]
        );
    }

    #[test]
    fn floats() {
        assert_eq!(
            lex_kinds("3.125 0.001 2.0"),
            vec![
                TokenKind::Float(3.125),
                TokenKind::Float(0.001),
                TokenKind::Float(2.0),
            ]
        );
    }

    #[test]
    fn float_with_exponent() {
        assert_eq!(
            lex_kinds("1.5e10 2.0E-3"),
            vec![TokenKind::Float(1.5e10), TokenKind::Float(2.0e-3)]
        );
    }

    #[test]
    fn exponent_only_float() {
        assert_eq!(lex_kinds("1e10"), vec![TokenKind::Float(1e10)]);
        assert_eq!(lex_kinds("5E3"), vec![TokenKind::Float(5e3)]);
    }

    #[test]
    fn number_with_underscores() {
        assert_eq!(
            lex_kinds("1_000_000 0xFF_FF"),
            vec![TokenKind::Int(1_000_000), TokenKind::Int(0xFFFF)]
        );
    }

    // ===== String literals =====

    #[test]
    fn simple_string() {
        assert_eq!(
            lex_kinds(r#""hello""#),
            vec![TokenKind::Str("hello".into())]
        );
    }

    #[test]
    fn string_with_escapes() {
        assert_eq!(
            lex_kinds(r#""line\n" "tab\there" "esc\"quote""#),
            vec![
                TokenKind::Str("line\n".into()),
                TokenKind::Str("tab\there".into()),
                TokenKind::Str("esc\"quote".into()),
            ]
        );
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

    // ===== Edge cases =====

    #[test]
    fn unexpected_char() {
        assert!(matches!(lex("~"), Err(LexError::UnexpectedChar { .. })));
    }

    #[test]
    fn at_sign() {
        assert_eq!(lex_kinds("@"), vec![TokenKind::At]);
    }

    #[test]
    fn empty_input() {
        assert_eq!(lex_kinds(""), vec![]);
    }

    #[test]
    fn arrow_not_minus() {
        // -> should be Arrow, not Minus followed by Gt
        assert_eq!(lex_kinds("->"), vec![TokenKind::Arrow]);
    }

    #[test]
    fn pipe_not_bar() {
        // |> should be Pipe, not Bar followed by Gt
        assert_eq!(lex_kinds("|>"), vec![TokenKind::Pipe]);
    }

    #[test]
    fn minus_is_minus_before_non_arrow() {
        assert_eq!(
            lex_kinds("- x"),
            vec![TokenKind::Minus, TokenKind::Ident("x".into())]
        );
    }

    #[test]
    fn realistic_def() {
        let tokens = lex_kinds("def f(x: f32): f32 = x + 1.0");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Def,
                TokenKind::Ident("f".into()),
                TokenKind::LParen,
                TokenKind::Ident("x".into()),
                TokenKind::Colon,
                TokenKind::Ident("f32".into()),
                TokenKind::RParen,
                TokenKind::Colon,
                TokenKind::Ident("f32".into()),
                TokenKind::Eq,
                TokenKind::Ident("x".into()),
                TokenKind::Plus,
                TokenKind::Float(1.0),
            ]
        );
    }

    #[test]
    fn match_expression_tokens() {
        let tokens = lex_kinds("match x with | Some y -> y | None -> 0");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Match,
                TokenKind::Ident("x".into()),
                TokenKind::With,
                TokenKind::Bar,
                TokenKind::TypeIdent("Some".into()),
                TokenKind::Ident("y".into()),
                TokenKind::Arrow,
                TokenKind::Ident("y".into()),
                TokenKind::Bar,
                TokenKind::TypeIdent("None".into()),
                TokenKind::Arrow,
                TokenKind::Int(0),
            ]
        );
    }

    #[test]
    fn pipe_chain() {
        let tokens = lex_kinds("x |> f |> g");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Ident("x".into()),
                TokenKind::Pipe,
                TokenKind::Ident("f".into()),
                TokenKind::Pipe,
                TokenKind::Ident("g".into()),
            ]
        );
    }

    #[test]
    fn lambda_tokens() {
        let tokens = lex_kinds("fn (x, y) -> x + y");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Fn,
                TokenKind::LParen,
                TokenKind::Ident("x".into()),
                TokenKind::Comma,
                TokenKind::Ident("y".into()),
                TokenKind::RParen,
                TokenKind::Arrow,
                TokenKind::Ident("x".into()),
                TokenKind::Plus,
                TokenKind::Ident("y".into()),
            ]
        );
    }

    #[test]
    fn dot_token() {
        assert_eq!(
            lex_kinds("Foo.Bar"),
            vec![
                TokenKind::TypeIdent("Foo".into()),
                TokenKind::Dot,
                TokenKind::TypeIdent("Bar".into()),
            ]
        );
    }

    #[test]
    fn dot_does_not_break_floats() {
        // 3.14 should still lex as a single float, not Int Dot Int
        assert_eq!(lex_kinds("3.125"), vec![TokenKind::Float(3.125)]);
    }

    #[test]
    fn dot_after_int_without_digit() {
        // "3." followed by a non-digit should be Int Dot (not a float)
        assert_eq!(
            lex_kinds("3.foo"),
            vec![
                TokenKind::Int(3),
                TokenKind::Dot,
                TokenKind::Ident("foo".into()),
            ]
        );
    }

    // --- WS-B1: literal suffix grammar (spec/02-surf-syntax.md §P10a /
    // spec/04-type-system.md §5.5) ---

    #[test]
    fn typed_int_suffix_each_member_of_closed_set() {
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
    fn typed_float_suffix_each_member_of_closed_set() {
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
    fn float_suffix_attaches_to_integer_literal() {
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
        // Every reserved-but-deferred name of spec/04 §1.1.1: no suffix
        // exists until the dtype activates, and the diagnostic cites the
        // owning section.
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
    fn hex_integer_suffix_typed_int() {
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
            matches!(err, LexError::HexFloatSuffix { ref suffix, .. } if suffix == "f32"),
            "expected HexFloatSuffix, got {err:?}"
        );
        let msg = format!("{err}");
        assert!(
            msg.contains("hex literals cannot carry float-typed suffixes"),
            "diagnostic must explain hex+float-suffix rule, got: {msg}"
        );
    }

    #[test]
    fn whitespace_before_suffix_is_two_tokens() {
        assert_eq!(
            lex_kinds("1.0 f32"),
            vec![TokenKind::Float(1.0), TokenKind::Ident("f32".into()),]
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

    #[test]
    fn underscore_separated_typed_literal() {
        // Underscores are still stripped before suffix parsing.
        assert_eq!(
            lex_kinds("1_000i64"),
            vec![TokenKind::TypedInt(1000, LiteralSuffix::I64)]
        );
    }
}
