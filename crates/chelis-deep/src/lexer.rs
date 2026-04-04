use crate::span::Span;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
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
    Str(String),
    /// Keyword without the leading `:`.
    Keyword(String),
    Bool(bool),
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
                let name = &source[kw_start..i];
                tokens.push(Token {
                    kind: TokenKind::Keyword(name.to_string()),
                    span: Span::new(start, i - start),
                });
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
                // '-' not followed by digit — treat as start of symbol (e.g., `->`)
                while i < bytes.len() && is_symbol_char(bytes[i]) {
                    i += 1;
                }
                let text = &source[start..i];
                tokens.push(Token {
                    kind: TokenKind::Symbol(text.to_string()),
                    span: Span::new(start, i - start),
                });
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
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-'
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
                return Ok(Token {
                    kind: TokenKind::Int(val),
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
                return Ok(Token {
                    kind: TokenKind::Int(val),
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
    let is_float = *i < bytes.len()
        && bytes[*i] == b'.'
        && (*i + 1 < bytes.len() && bytes[*i + 1].is_ascii_digit());

    if is_float {
        *i += 1; // skip '.'
        while *i < bytes.len() && bytes[*i].is_ascii_digit() {
            *i += 1;
        }
        // Exponent
        if *i < bytes.len() && (bytes[*i] == b'e' || bytes[*i] == b'E') {
            *i += 1;
            if *i < bytes.len() && (bytes[*i] == b'+' || bytes[*i] == b'-') {
                *i += 1;
            }
            while *i < bytes.len() && bytes[*i].is_ascii_digit() {
                *i += 1;
            }
        }
        let text = &source[start..(*i)];
        let val: f64 = text.parse().map_err(|_| LexError::InvalidNumber {
            text: text.to_string(),
            offset: start,
        })?;
        Ok(Token {
            kind: TokenKind::Float(val),
            span: Span::new(start, *i - start),
        })
    } else {
        let text = &source[start..(*i)];
        let val: i64 = text.parse().map_err(|_| LexError::InvalidNumber {
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
    }

    #[test]
    fn booleans() {
        assert_eq!(
            lex_kinds("true false"),
            vec![TokenKind::Bool(true), TokenKind::Bool(false),]
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
}
