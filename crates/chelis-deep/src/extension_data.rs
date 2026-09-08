//! Producer data is a validated, flat syntax tree, never a program expression.
//! Keeping its canonical token stream flat also makes clone/drop stack safe.

use crate::{
    Span,
    lexer::{self, Token, TokenKind},
    parser::ParseError,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Opaque producer data under [03-META-3]. Only lexical/structural parsing can
/// construct it. Its contents are unavailable to semantic expression visitors.
///
/// ```compile_fail
/// use chelis_deep::{Expr, ExtensionData};
/// fn expression(data: ExtensionData) -> Expr { data }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtensionData {
    syntax: String,
    span: Span,
}

impl ExtensionData {
    /// Explicit producer conversion of raw syntax into data, without stamping.
    pub fn from_raw(raw: &crate::RawExpr) -> Result<Self, ParseError> {
        if let crate::RawExpr::ExtensionData(data) = raw {
            return Ok(data.clone());
        }
        // Raw atoms and public lexer tokens are constructible by callers. Check
        // their lexical identity so a forged symbol or nonfinite float cannot
        // silently turn into a different data value when rendered.
        let mut pending = vec![raw];
        while let Some(value) = pending.pop() {
            use crate::{RawAtom as A, RawExpr as R};
            match value {
                R::Atom(atom, span) => {
                    let kind = match atom {
                        A::Symbol(v) => TokenKind::Symbol(v.clone()),
                        A::Int(v) => TokenKind::Int(*v),
                        A::Float(v) => TokenKind::Float(*v),
                        A::Str(v) => TokenKind::Str(v.clone()),
                        A::Bool(v) => TokenKind::Bool(*v),
                    };
                    checked_scalar(&Token { kind, span: *span })?;
                }
                R::List(items, _) => pending.extend(items),
                R::Map(entries, _) => pending.extend(entries.iter().map(|(_, value)| value)),
                R::MetaExpr { entries, expr, .. } => {
                    pending.extend(entries.iter().map(|(_, value)| value));
                    pending.push(expr);
                }
                R::ExtensionData(_) => {}
            }
        }
        Self::parse(&crate::printer::print_raw_data(raw)).map(|value| value.at_span(raw.span()))
    }
    pub fn parse(source: &str) -> Result<Self, ParseError> {
        let tokens = lexer::lex(source)?;
        let (value, consumed) = Self::from_tokens(&tokens, Some(source))?;
        if consumed != tokens.len() {
            return Err(expected("one extension data value", &tokens[consumed..]));
        }
        Ok(value)
    }

    /// Canonical data syntax for a producer or a serialization boundary.
    pub fn syntax(&self) -> &str {
        &self.syntax
    }
    pub fn span(&self) -> Span {
        self.span
    }
    pub(crate) fn at_span(mut self, span: Span) -> Self {
        self.span = span;
        self
    }
    pub fn same_payload(&self, other: &Self) -> bool {
        self.syntax == other.syntax
    }

    pub(crate) fn from_tokens(
        tokens: &[Token],
        source: Option<&str>,
    ) -> Result<(Self, usize), ParseError> {
        enum Task {
            Value,
            ListTail(bool),
            MapTail { prefix: bool, first: bool },
            PrefixBody,
        }
        let eof = source.map_or_else(
            || tokens.last().map_or(0, |token| token.span.end()),
            str::len,
        );
        let expected = |description: &str, tail: &[Token]| match tail.first() {
            Some(_) => expected(description, tail),
            None => ParseError::UnexpectedEof { offset: eof },
        };
        let mut tasks = vec![Task::Value];
        let mut at = 0;
        let mut syntax = String::new();
        while let Some(task) = tasks.pop() {
            let tail = &tokens[at..];
            let token = tail
                .first()
                .ok_or_else(|| expected("extension data", tail))?;
            match task {
                Task::Value => match &token.kind {
                    TokenKind::LParen => {
                        syntax.push('(');
                        at += 1;
                        tasks.push(Task::ListTail(true));
                    }
                    TokenKind::LBrace => {
                        syntax.push('{');
                        at += 1;
                        tasks.push(Task::MapTail {
                            prefix: false,
                            first: true,
                        });
                    }
                    TokenKind::Caret => {
                        syntax.push_str("^{");
                        at += 1;
                        if !matches!(tokens.get(at).map(|t| &t.kind), Some(TokenKind::LBrace)) {
                            return Err(expected("{ after ^", &tokens[at..]));
                        }
                        at += 1;
                        tasks.push(Task::PrefixBody);
                        tasks.push(Task::MapTail {
                            prefix: true,
                            first: true,
                        });
                    }
                    TokenKind::Symbol(name) if name != ":" => {
                        if source.is_none() {
                            checked_scalar(token)?;
                        }
                        syntax.push_str(name);
                        at += 1;
                    }
                    TokenKind::Int(_)
                    | TokenKind::Float(_)
                    | TokenKind::TypedInt(..)
                    | TokenKind::TypedFloat(..)
                    | TokenKind::Str(_)
                    | TokenKind::Bool(_) => {
                        syntax.push_str(&if source.is_some() {
                            scalar_spelling(token, source)
                        } else {
                            checked_scalar(token)?
                        });
                        at += 1;
                    }
                    _ => return Err(expected("a scalar, list, map or prefix data record", tail)),
                },
                Task::ListTail(first) => {
                    if matches!(token.kind, TokenKind::RParen) {
                        syntax.push(')');
                        at += 1;
                    } else {
                        if !first {
                            syntax.push(' ');
                        }
                        tasks.push(Task::ListTail(false));
                        tasks.push(Task::Value);
                    }
                }
                Task::MapTail { prefix, first } => {
                    if matches!(token.kind, TokenKind::RBrace) {
                        syntax.push('}');
                        at += 1;
                        continue;
                    }
                    if !first && !prefix && matches!(token.kind, TokenKind::Comma) {
                        at += 1;
                        if matches!(tokens.get(at).map(|t| &t.kind), Some(TokenKind::RBrace)) {
                            syntax.push('}');
                            at += 1;
                            continue;
                        }
                    }
                    let tail = &tokens[at..];
                    let key = match tail.first().map(|t| &t.kind) {
                        Some(TokenKind::Keyword(key)) if prefix && identifier(key) => key,
                        Some(TokenKind::Symbol(key)) if !prefix && identifier(key) => key,
                        _ => return Err(expected("a data map key", tail)),
                    };
                    if !first {
                        syntax.push_str(if prefix { " " } else { ", " });
                    }
                    if prefix {
                        syntax.push(':');
                    }
                    syntax.push_str(key);
                    at += 1;
                    if !prefix {
                        if !matches!(tokens.get(at).map(|t| &t.kind), Some(TokenKind::Symbol(s)) if s == ":")
                        {
                            return Err(expected(": after data map key", &tokens[at..]));
                        }
                        at += 1;
                        syntax.push(':');
                    }
                    syntax.push(' ');
                    tasks.push(Task::MapTail {
                        prefix,
                        first: false,
                    });
                    tasks.push(Task::Value);
                }
                Task::PrefixBody => {
                    syntax.push(' ');
                    tasks.push(Task::Value);
                }
            }
        }
        let span = tokens[0].span.merge(tokens[at - 1].span);
        Ok((Self { syntax, span }, at))
    }
}

fn identifier(name: &str) -> bool {
    let mut bytes = name.bytes();
    bytes
        .next()
        .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
        && bytes.all(|b| b.is_ascii_alphanumeric() || b == b'_')
}
fn expected(expected: &str, tokens: &[Token]) -> ParseError {
    match tokens.first() {
        Some(token) => ParseError::Expected {
            expected: expected.into(),
            found: format!("{:?}", token.kind),
            offset: token.span.offset,
        },
        None => ParseError::UnexpectedEof { offset: 0 },
    }
}
fn scalar_spelling(token: &Token, source: Option<&str>) -> String {
    if let Some(text) = source.and_then(|s| s.get(token.span.offset..token.span.end())) {
        return text.into();
    }
    match &token.kind {
        TokenKind::Symbol(v) => v.clone(),
        TokenKind::Int(v) => v.to_string(),
        TokenKind::Float(v) => format!("{v:?}"),
        TokenKind::TypedInt(v, suffix) => format!("{v}{}", suffix.as_str()),
        TokenKind::TypedFloat(v, suffix) => format!("{v:?}{}", suffix.as_str()),
        TokenKind::Bool(v) => v.to_string(),
        TokenKind::Str(v) => {
            let mut out = String::from("\"");
            for c in v.chars() {
                match c {
                    '\\' => out.push_str("\\\\"),
                    '"' => out.push_str("\\\""),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    _ => out.push(c),
                }
            }
            out.push('"');
            out
        }
        _ => unreachable!("only scalar tokens have scalar spelling"),
    }
}

fn checked_scalar(token: &Token) -> Result<String, ParseError> {
    let spelling = scalar_spelling(token, None);
    let decoded = lexer::lex(&spelling)?;
    if decoded.len() != 1 || decoded[0].kind != token.kind {
        return Err(ParseError::Expected {
            expected: "a scalar with a faithful Deep spelling".into(),
            found: format!("{:?}", token.kind),
            offset: token.span.offset,
        });
    }
    Ok(spelling)
}

impl Serialize for ExtensionData {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (&self.syntax, self.span).serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for ExtensionData {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (syntax, span) = <(String, Span)>::deserialize(deserializer)?;
        let mut value = Self::parse(&syntax).map_err(serde::de::Error::custom)?;
        value.span = span;
        Ok(value)
    }
}
