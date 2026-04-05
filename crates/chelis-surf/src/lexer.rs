use chelis_deep::Span;
use thiserror::Error;

use crate::token::{Token, TokenKind};

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

    #[error("unterminated block comment starting at byte {offset}")]
    UnterminatedBlockComment { offset: usize },
}

pub fn lex(source: &str) -> Result<Vec<Token>, LexError> {
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
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
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
                    let ch = source[i..].chars().next().unwrap();
                    return Err(LexError::UnexpectedChar { ch, offset: i });
                }
            }
            b'.' => {
                tokens.push(Token {
                    kind: TokenKind::Dot,
                    span: Span::new(start, 1),
                });
                i += 1;
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

fn classify_ident(text: &str) -> TokenKind {
    match text {
        "def" => TokenKind::Def,
        "sig" => TokenKind::Sig,
        "let" => TokenKind::Let,
        "in" => TokenKind::In,
        "type" => TokenKind::Type,
        "dim" => TokenKind::Dim,
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
        "export" => TokenKind::Export,
        "par" => TokenKind::Par,
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
                let digits: String = text[2..].chars().filter(|c| *c != '_').collect();
                let val =
                    i64::from_str_radix(&digits, 16).map_err(|_| LexError::InvalidNumber {
                        text: text.to_string(),
                        offset: start,
                    })?;
                return Ok(Token {
                    kind: TokenKind::Int(val),
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
                let digits: String = text[2..].chars().filter(|c| *c != '_').collect();
                let val = i64::from_str_radix(&digits, 2).map_err(|_| LexError::InvalidNumber {
                    text: text.to_string(),
                    offset: start,
                })?;
                return Ok(Token {
                    kind: TokenKind::Int(val),
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

    // Check for exponent (e/E) -- makes it a float even without decimal point
    if *i < bytes.len() && (bytes[*i] == b'e' || bytes[*i] == b'E') {
        is_float = true;
        *i += 1;
        if *i < bytes.len() && (bytes[*i] == b'+' || bytes[*i] == b'-') {
            *i += 1;
        }
        while *i < bytes.len() && (bytes[*i].is_ascii_digit() || bytes[*i] == b'_') {
            *i += 1;
        }
    }

    let text = &source[start..*i];
    let clean: String = text.chars().filter(|c| *c != '_').collect();
    if is_float {
        let val: f64 = clean.parse().map_err(|_| LexError::InvalidNumber {
            text: text.to_string(),
            offset: start,
        })?;
        Ok(Token {
            kind: TokenKind::Float(val),
            span: Span::new(start, *i - start),
        })
    } else {
        let val: i64 = clean.parse().map_err(|_| LexError::InvalidNumber {
            text: text.to_string(),
            offset: start,
        })?;
        Ok(Token {
            kind: TokenKind::Int(val),
            span: Span::new(start, *i - start),
        })
    }
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
                "def let in type match with fn module import if then else grad vmap jit tensor cast export"
            ),
            vec![
                TokenKind::Def,
                TokenKind::Let,
                TokenKind::In,
                TokenKind::Type,
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
                TokenKind::Tensor,
                TokenKind::Cast,
                TokenKind::Export,
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
            lex_kinds("let x = 1\nlet y = 2\r\nlet z = 3"),
            vec![
                TokenKind::Let,
                TokenKind::Ident("x".into()),
                TokenKind::Eq,
                TokenKind::Int(1),
                TokenKind::Newline,
                TokenKind::Let,
                TokenKind::Ident("y".into()),
                TokenKind::Eq,
                TokenKind::Int(2),
                TokenKind::Newline,
                TokenKind::Let,
                TokenKind::Ident("z".into()),
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
}
