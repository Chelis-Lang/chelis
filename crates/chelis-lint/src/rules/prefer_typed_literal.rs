//! Rule `prefer-typed-literal`: a literal whose only purpose is to bind at a
//! concrete dtype states that dtype with a suffix (§12.6), so
//! `cast(1.0, f32)` is written `1.0f32` and `cast(3, i64)` is written `3i64`.

use crate::{Context, Replacement, Rule, Severity, Surface, Violation};
use chelis_surf::token::{Token, TokenKind};

pub struct PreferTypedLiteral;

impl Rule for PreferTypedLiteral {
    fn id(&self) -> &str {
        "prefer-typed-literal"
    }

    fn spec_ref(&self) -> &str {
        "§12.6"
    }

    fn applies_to(&self) -> &[Surface] {
        &[Surface::SurfSource]
    }

    fn summary(&self) -> &str {
        "`cast(<numeric literal>, <dtype>)` is spelled as the suffixed literal (`cast(1.0, f32)` is `1.0f32`) when the two bind identically (§12.6)"
    }

    fn severity(&self) -> Severity {
        Severity::Warning
    }

    fn check(&self, ctx: &Context<'_>) -> Vec<Violation> {
        let Some(source) = ctx.source else {
            return Vec::new();
        };
        sites(source)
            .into_iter()
            .map(|site| {
                let (line, col) = line_col(source, site.start);
                Violation {
                    rule_id: self.id().to_string(),
                    spec_ref: self.spec_ref().to_string(),
                    path: ctx.path.to_path_buf(),
                    line: Some(line),
                    col: Some(col),
                    message: format!(
                        "`cast({}, {})` binds the literal at `{}` exactly as the suffixed literal does; write `{}`",
                        site.body,
                        site.dtype,
                        site.dtype,
                        site.replacement()
                    ),
                }
            })
            .collect()
    }

    fn fix(&self, ctx: &Context<'_>, violation: &Violation) -> Option<Replacement> {
        let source = ctx.source?;
        let start = line_start_offset(source, violation.line?)? + violation.col?.checked_sub(1)?;
        let site = sites(source).into_iter().find(|site| site.start == start)?;
        Some(Replacement {
            path: ctx.path.to_path_buf(),
            start: site.start,
            end: site.end,
            text: site.replacement(),
        })
    }
}

/// One `cast(<literal>, <dtype>)` call whose suffixed spelling binds the same
/// value at the same dtype.
struct Site<'a> {
    /// Byte range of the whole call, `cast` through `)`.
    start: usize,
    end: usize,
    /// The literal exactly as authored.
    body: &'a str,
    dtype: &'a str,
}

impl Site<'_> {
    fn replacement(&self) -> String {
        format!("{}{}", self.body, self.dtype)
    }
}

const FLOAT_DTYPES: [&str; 4] = ["f16", "bf16", "f32", "f64"];
const INT_DTYPES: [&str; 4] = ["i8", "i16", "i32", "i64"];

/// Find every rewritable site from the lexer's token stream, so string
/// literals and comments are never read as code.
///
/// A site is exactly `cast ( LIT , DTYPE ,? )` where LIT is an unsuffixed
/// literal token and DTYPE is a suffix name, with nothing but whitespace
/// between the tokens. The suffixed literal then binds the same value at the
/// same dtype (§12.6). Every other shape is left alone:
///
/// - a negative operand, because `-1` is unary minus applied to `1`, and the
///   suffixed form negates a value already bound at the suffix width;
/// - a float body under an integer dtype, which has no suffixed spelling;
/// - a radix body under a float dtype, which has no suffixed spelling;
/// - the body `0` under `bf16`, because `0bf16` lexes as a binary prefix;
/// - a dtype outside the suffix set, including a dtype binder;
/// - a call fed by `|>`, which supplies another argument;
/// - a call with a comment between its tokens, which the rewrite would delete.
fn sites(source: &str) -> Vec<Site<'_>> {
    let Ok(tokens) = chelis_surf::lexer::lex(source) else {
        return Vec::new();
    };
    let tokens: Vec<&Token> = tokens
        .iter()
        .filter(|token| token.kind != TokenKind::Newline)
        .collect();
    let mut out = Vec::new();
    for (index, token) in tokens.iter().enumerate() {
        if token.kind != TokenKind::Cast {
            continue;
        }
        if index > 0 && tokens[index - 1].kind == TokenKind::Pipe {
            continue;
        }
        let Some(site) = site_at(source, &tokens[index..]) else {
            continue;
        };
        out.push(site);
    }
    out
}

fn site_at<'a>(source: &'a str, tokens: &[&Token]) -> Option<Site<'a>> {
    let [cast, open, literal, comma, dtype_token, rest @ ..] = tokens else {
        return None;
    };
    let close = match rest {
        [trailing, close, ..] if trailing.kind == TokenKind::Comma => close,
        [close, ..] => close,
        [] => return None,
    };
    if open.kind != TokenKind::LParen
        || comma.kind != TokenKind::Comma
        || close.kind != TokenKind::RParen
    {
        return None;
    }
    if !matches!(dtype_token.kind, TokenKind::Ident(_)) {
        return None;
    }
    let dtype = &source[dtype_token.span.offset..dtype_token.span.end()];
    let body = &source[literal.span.offset..literal.span.end()];
    let radix = body.starts_with("0x")
        || body.starts_with("0X")
        || body.starts_with("0b")
        || body.starts_with("0B");
    let admitted = match literal.kind {
        TokenKind::Int(_) if INT_DTYPES.contains(&dtype) => true,
        TokenKind::Int(_) if FLOAT_DTYPES.contains(&dtype) => {
            !radix && !(body == "0" && dtype == "bf16")
        }
        TokenKind::Float(_) => FLOAT_DTYPES.contains(&dtype),
        _ => false,
    };
    if !admitted {
        return None;
    }
    let start = cast.span.offset;
    let end = close.span.end();
    let mut cursor = start;
    for token in tokens.iter().take_while(|token| token.span.offset < end) {
        if !source[cursor..token.span.offset].trim().is_empty() {
            return None;
        }
        cursor = token.span.end();
    }
    Some(Site {
        start,
        end,
        body,
        dtype,
    })
}

fn line_start_offset(source: &str, line_no: usize) -> Option<usize> {
    if line_no == 0 {
        return None;
    }
    let mut line = 1usize;
    if line == line_no {
        return Some(0);
    }
    for (index, byte) in source.bytes().enumerate() {
        if byte == b'\n' {
            line += 1;
            if line == line_no {
                return Some(index + 1);
            }
        }
    }
    None
}

fn line_col(source: &str, offset: usize) -> (usize, usize) {
    let before = &source[..offset];
    let line = before.matches('\n').count() + 1;
    let line_start = before.rfind('\n').map_or(0, |index| index + 1);
    (line, offset - line_start + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn ctx(src: &str) -> Context<'_> {
        Context {
            root: Path::new("/"),
            path: Path::new("test.ch"),
            source: Some(src),
            surface: Surface::SurfSource,
        }
    }

    fn run(src: &str) -> Vec<Violation> {
        PreferTypedLiteral.check(&ctx(src))
    }

    /// Apply every proposed fix to `src`, back to front.
    fn fixed(src: &str) -> String {
        let context = ctx(src);
        let mut replacements: Vec<Replacement> = run(src)
            .iter()
            .map(|violation| {
                PreferTypedLiteral
                    .fix(&context, violation)
                    .expect("every reported site has a fix")
            })
            .collect();
        replacements.sort_by_key(|replacement| replacement.start);
        let mut out = src.to_string();
        for replacement in replacements.iter().rev() {
            out.replace_range(replacement.start..replacement.end, &replacement.text);
        }
        out
    }

    fn assert_rewrites(src: &str, expected: &str) {
        assert_eq!(fixed(src), expected, "source: {src}");
    }

    fn assert_silent(src: &str) {
        let violations = run(src);
        assert!(
            violations.is_empty(),
            "expected no violation for {src:?}, got {violations:?}"
        );
    }

    #[test]
    fn rewrites_float_body_at_float_dtype() {
        assert_rewrites("x = cast(1.0, f32)\n", "x = 1.0f32\n");
        assert_rewrites("x = cast(1.1, f64)\n", "x = 1.1f64\n");
        assert_rewrites("x = cast(0.1, f16)\n", "x = 0.1f16\n");
        assert_rewrites("x = cast(0.1, bf16)\n", "x = 0.1bf16\n");
    }

    #[test]
    fn rewrites_integer_body_at_integer_dtype() {
        assert_rewrites("x = cast(7, i8)\n", "x = 7i8\n");
        assert_rewrites("x = cast(1000, i16)\n", "x = 1000i16\n");
        assert_rewrites("x = cast(5, i32)\n", "x = 5i32\n");
        assert_rewrites("x = cast(3000000000, i64)\n", "x = 3000000000i64\n");
    }

    #[test]
    fn rewrites_decimal_integer_body_at_float_dtype() {
        assert_rewrites("x = cast(1, f32)\n", "x = 1f32\n");
        assert_rewrites("x = cast(16777217, f32)\n", "x = 16777217f32\n");
        assert_rewrites("x = cast(2, f64)\n", "x = 2f64\n");
        assert_rewrites("x = cast(65504, f16)\n", "x = 65504f16\n");
        assert_rewrites("x = cast(257, bf16)\n", "x = 257bf16\n");
        assert_rewrites("x = cast(0, f32)\n", "x = 0f32\n");
    }

    #[test]
    fn keeps_the_authored_body_spelling() {
        assert_rewrites("x = cast(1e-3, f64)\n", "x = 1e-3f64\n");
        assert_rewrites("x = cast(2.5E+2, f64)\n", "x = 2.5E+2f64\n");
        assert_rewrites("x = cast(1_000.25, f64)\n", "x = 1_000.25f64\n");
        assert_rewrites("x = cast(1_000, i64)\n", "x = 1_000i64\n");
        assert_rewrites("x = cast(0xFF, i64)\n", "x = 0xFFi64\n");
        assert_rewrites("x = cast(0b101, i32)\n", "x = 0b101i32\n");
    }

    #[test]
    fn rewrites_whitespace_newline_and_trailing_comma_spellings() {
        assert_rewrites("x = cast( 1.0 ,f32 )\n", "x = 1.0f32\n");
        assert_rewrites("x = cast(\n  1.0,\n  f32\n)\n", "x = 1.0f32\n");
        assert_rewrites("x = cast(1.0, f32,)\n", "x = 1.0f32\n");
        assert_rewrites("x = cast (2, i64)\n", "x = 2i64\n");
    }

    #[test]
    fn rewrites_nested_sites_and_reports_each_position() {
        let src = "t = to_tensor([cast(1.0, f64), cast(2.0, f64)]) |> mul(cast(3, f64))\n";
        let violations = run(src);
        assert_eq!(violations.len(), 3, "{violations:?}");
        assert_eq!(violations[0].rule_id, "prefer-typed-literal");
        assert_eq!(violations[0].spec_ref, "§12.6");
        assert_eq!(violations[0].line, Some(1));
        assert_eq!(violations[0].col, Some(16));
        assert_eq!(violations[1].col, Some(32));
        assert!(
            violations[0].message.contains("`1.0f64`"),
            "{}",
            violations[0].message
        );
        assert_rewrites(src, "t = to_tensor([1.0f64, 2.0f64]) |> mul(3f64)\n");
        let second_line = "a = 1\nb = cast(2, i64)\n";
        let violations = run(second_line);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].line, Some(2));
        assert_eq!(violations[0].col, Some(5));
    }

    #[test]
    fn silent_on_negative_literals() {
        // `-1` is unary minus applied to `1`. The cast form adopts the folded
        // negative literal, while the suffixed form negates a value already
        // bound at the suffix width, so `cast(-128, i8)` is accepted and
        // `-128i8` is rejected.
        assert_silent("x = cast(-1.0, f64)\n");
        assert_silent("x = cast(-1, i64)\n");
        assert_silent("x = cast(-128, i8)\n");
        assert_silent("x = cast(- 0.5, f32)\n");
    }

    #[test]
    fn silent_where_the_suffixed_spelling_is_not_a_literal() {
        // An integer suffix never attaches to a float body.
        assert_silent("x = cast(3.0, i64)\n");
        assert_silent("x = cast(1e3, i32)\n");
        // A radix body never carries a float suffix.
        assert_silent("x = cast(0x10, f32)\n");
        assert_silent("x = cast(0b1, f64)\n");
        assert_silent("x = cast(0X10, f32)\n");
        assert_silent("x = cast(0B1, f32)\n");
        // `0bf16` lexes as a binary prefix.
        assert_silent("x = cast(0, bf16)\n");
    }

    #[test]
    fn silent_on_dtypes_outside_the_suffix_set() {
        assert_silent("def f[p: Float]() -> p = cast(1.0, p)\n");
        assert_silent("x = cast(1, bool)\n");
        assert_silent("x = cast(1, uint8)\n");
        assert_silent("x = cast(1, u8)\n");
        assert_silent("x = cast(1.0, f8e4m3)\n");
    }

    #[test]
    fn silent_on_non_literal_and_already_typed_operands() {
        assert_silent("x = cast(1.0f32, f64)\n");
        assert_silent("x = cast(1f32, f64)\n");
        assert_silent("x = cast(3i32, i64)\n");
        assert_silent("def f(y: f32) -> f64 = cast(y, f64)\n");
        assert_silent("x = cast(add(1, 2), i64)\n");
        assert_silent("x = cast(1 + 2, i64)\n");
        assert_silent("x = cast([1.0, 2.0], f64)\n");
        assert_silent("x = cast(1.0)\n");
        assert_silent("x = cast(1.0, f32, f64)\n");
    }

    #[test]
    fn silent_on_pipe_forms_and_other_callees() {
        assert_silent("x = 1.0 |> cast(f64)\n");
        assert_silent("x = y |> cast(1.0, f32)\n");
        assert_silent("x = cast_trunc(1.0, i32)\n");
        assert_silent("x = my_cast(1.0, f32)\n");
    }

    #[test]
    fn silent_in_strings_and_comments() {
        assert_silent("x = \"cast(1.0, f32)\"\n");
        assert_silent("-- x = cast(1.0, f32)\nx = 1\n");
        assert_silent("{- cast(1.0, f32) -}\nx = 1\n");
        // A comment inside the call would be deleted by the rewrite.
        assert_silent("x = cast(1.0, -- width\n  f32)\n");
        assert_silent("x = cast({- w -} 1.0, f32)\n");
    }

    #[test]
    fn silent_on_source_that_does_not_lex() {
        assert_silent("x = cast(1.0, f32) \"unterminated\n");
    }

    #[test]
    fn fix_declines_a_violation_that_does_not_point_at_a_site() {
        let src = "x = cast(1.0, f32)\n";
        let mut violation = run(src).remove(0);
        violation.col = Some(1);
        assert!(PreferTypedLiteral.fix(&ctx(src), &violation).is_none());
    }

    #[test]
    fn is_a_non_blocking_warning_with_a_lexical_fix() {
        let rule = PreferTypedLiteral;
        assert_eq!(rule.severity(), Severity::Warning);
        assert!(!rule.severity().blocks_check());
        assert!(!rule.fix_requires_typed_pipeline_check());
    }
}
