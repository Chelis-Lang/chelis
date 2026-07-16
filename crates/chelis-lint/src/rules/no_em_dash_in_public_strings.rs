//! Rule `no-em-dash-in-public-strings` -- user-facing string literals should
//! use plain punctuation instead of em dashes.

use crate::{Context, Replacement, Rule, Severity, Surface, Violation};

pub struct NoEmDashInPublicStrings;

impl Rule for NoEmDashInPublicStrings {
    fn id(&self) -> &'static str {
        "no-em-dash-in-public-strings"
    }

    fn spec_ref(&self) -> &'static str {
        "§8.6"
    }

    fn applies_to(&self) -> &[Surface] {
        &[
            Surface::SurfSource,
            Surface::DeepSource,
            Surface::RustSource,
            Surface::PythonSource,
        ]
    }

    fn summary(&self) -> &'static str {
        "Public-facing string literals should not contain em dashes"
    }

    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn check(&self, ctx: &Context<'_>) -> Vec<Violation> {
        let Some(source) = ctx.source else {
            return Vec::new();
        };
        // §8.6 scope: the rule targets user-facing strings (diagnostics,
        // raised-error text, log messages). Python module/function/class
        // docstrings are narrative prose, not user-facing strings, so we
        // detect them up front and skip any em-dash that falls inside one.
        let docstring_skip_ranges: Vec<(usize, usize)> = if ctx.surface == Surface::PythonSource {
            python_docstring_ranges(source)
        } else {
            Vec::new()
        };
        let mut out = Vec::new();
        for span in quoted_spans(source, ctx.surface) {
            for (relative, _) in source[span.start..span.end].match_indices('—') {
                let absolute = span.start + relative;
                if docstring_skip_ranges
                    .iter()
                    .any(|(s, e)| *s <= absolute && absolute < *e)
                {
                    continue;
                }
                let (line, col) = line_col(source, absolute);
                out.push(Violation {
                    rule_id: self.id().to_string(),
                    spec_ref: self.spec_ref().to_string(),
                    path: ctx.path.to_path_buf(),
                    line: Some(line),
                    col: Some(col),
                    message: "replace em dash in public-facing string literal".to_string(),
                });
            }
        }
        out
    }

    fn fix(&self, ctx: &Context<'_>, violation: &Violation) -> Option<Replacement> {
        let source = ctx.source?;
        let dash = offset_from_line_col(source, violation.line?, violation.col?)?;
        let line_start = line_start_offset(source, violation.line?)?;
        let line_end = source[line_start..]
            .find('\n')
            .map_or(source.len(), |idx| line_start + idx);
        let line = &source[line_start..line_end];
        let spacing = dash_spacing(line, line_start, dash)?;
        if !spacing.both_sides {
            return None;
        }
        let span = quoted_spans(source, ctx.surface)
            .into_iter()
            .find(|span| span.start <= dash && dash < span.end)?;
        if source[span.start..span.end].matches('—').count() >= 2 {
            return spaced_dash_replacement(ctx, dash, ", ", spacing.after_end);
        }
        clause_replacement(ctx, source, dash, spacing.after_end)
    }
}

#[derive(Debug, Clone, Copy)]
struct StringSpan {
    start: usize,
    end: usize,
}

fn quoted_spans(source: &str, surface: Surface) -> Vec<StringSpan> {
    let mut spans = Vec::new();
    let mut string_quote: Option<char> = None;
    let mut start = 0usize;
    let mut escaped = false;
    let mut in_line_comment = false;
    let mut in_block_comment = false;
    let mut in_char = false;
    let mut cursor = 0usize;
    while cursor < source.len() {
        let ch = source[cursor..].chars().next().unwrap();
        if let Some(quote) = string_quote {
            if escaped {
                escaped = false;
            } else {
                match ch {
                    '\\' => escaped = true,
                    _ if ch == quote => {
                        spans.push(StringSpan { start, end: cursor });
                        string_quote = None;
                    }
                    _ => {}
                }
            }
            cursor += ch.len_utf8();
            continue;
        }
        if in_char {
            if escaped {
                escaped = false;
            } else {
                match ch {
                    '\\' => escaped = true,
                    '\'' => in_char = false,
                    _ => {}
                }
            }
            cursor += ch.len_utf8();
            continue;
        }
        if in_line_comment {
            if ch == '\n' {
                in_line_comment = false;
            }
            cursor += ch.len_utf8();
            continue;
        }
        if in_block_comment {
            if source[cursor..].starts_with("*/") {
                in_block_comment = false;
                cursor += 2;
            } else {
                cursor += ch.len_utf8();
            }
            continue;
        }
        if matches!(surface, Surface::SurfSource | Surface::RustSource)
            && source[cursor..].starts_with("//")
        {
            in_line_comment = true;
            cursor += 2;
            continue;
        }
        if surface == Surface::DeepSource && ch == ';' {
            in_line_comment = true;
            cursor += ch.len_utf8();
            continue;
        }
        if surface == Surface::RustSource && source[cursor..].starts_with("/*") {
            in_block_comment = true;
            cursor += 2;
            continue;
        }
        if surface == Surface::PythonSource && ch == '#' {
            in_line_comment = true;
            cursor += ch.len_utf8();
            continue;
        }
        if ch == '"' || (surface == Surface::PythonSource && ch == '\'') {
            string_quote = Some(ch);
            start = cursor + ch.len_utf8();
        } else if surface == Surface::RustSource
            && ch == '\''
            && starts_rust_char_literal(source, cursor)
        {
            in_char = true;
        }
        cursor += ch.len_utf8();
    }
    spans
}

/// Return the byte ranges of Python docstring spans in `source`.
///
/// A docstring is heuristically defined as a triple-quoted string
/// (`"""..."""` or `'''...'''`) whose **opening triple-quote is the
/// first non-whitespace token on its line**. This catches the three
/// canonical docstring shapes:
///
/// - Module docstring: triple-quote at column 0 of an early line.
/// - Function/method/class docstring: triple-quote at the indent column
///   immediately after a `def`/`class` header line, with no other
///   code preceding it on that line.
///
/// String literals that appear mid-line (`print("""x""")`,
/// `raise ValueError("""x""")`, a triple-quoted assignment RHS that
/// shares the line with the `=`) are NOT docstrings under this rule
/// and are not skipped.
///
/// Each returned range is `(start, end)` where `start` is the byte
/// position of the first character inside the opening triple-quote and
/// `end` is the byte position of the first character of the closing
/// triple-quote. Em-dashes whose absolute offset falls in `[start, end)`
/// are inside the docstring's content and excluded from the rule.
fn python_docstring_ranges(source: &str) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let bytes = source.as_bytes();
    let mut cursor = 0usize;
    let mut at_line_start = true;
    while cursor < bytes.len() {
        if at_line_start {
            // Skip leading whitespace.
            let mut scan = cursor;
            while scan < bytes.len() && (bytes[scan] == b' ' || bytes[scan] == b'\t') {
                scan += 1;
            }
            if scan + 3 <= bytes.len() {
                let head = &bytes[scan..scan + 3];
                if head == b"\"\"\"" || head == b"'''" {
                    let quote_byte = head[0];
                    let content_start = scan + 3;
                    if let Some(close) = find_triple_close(bytes, content_start, quote_byte) {
                        ranges.push((content_start, close));
                        cursor = close + 3;
                        // After the closing triple-quote, the rest of
                        // the line is post-string content — treat that
                        // line end as a new line start.
                        at_line_start = false;
                        continue;
                    }
                }
            }
        }
        // Advance one byte. Track line-start across newlines.
        let b = bytes[cursor];
        if b == b'\n' {
            at_line_start = true;
        } else if b != b' ' && b != b'\t' && b != b'\r' {
            at_line_start = false;
        }
        cursor += 1;
    }
    ranges
}

/// Locate the byte offset of the closing triple-quote of the given
/// `quote_byte` (b'"' or b'\'') starting from `from`. Returns the byte
/// offset of the opening of the close triple. Returns `None` if no
/// close triple is found before end-of-source (treat as unterminated).
fn find_triple_close(bytes: &[u8], from: usize, quote_byte: u8) -> Option<usize> {
    let mut cursor = from;
    while cursor + 3 <= bytes.len() {
        if bytes[cursor] == b'\\' {
            cursor += 2;
            continue;
        }
        if bytes[cursor] == quote_byte
            && bytes[cursor + 1] == quote_byte
            && bytes[cursor + 2] == quote_byte
        {
            return Some(cursor);
        }
        cursor += 1;
    }
    None
}

fn starts_rust_char_literal(source: &str, quote: usize) -> bool {
    let mut cursor = quote + 1;
    let Some(ch) = source[cursor..].chars().next() else {
        return false;
    };
    if ch == '\\' {
        cursor += ch.len_utf8();
        let Some(escaped) = source[cursor..].chars().next() else {
            return false;
        };
        cursor += escaped.len_utf8();
    } else if ch == '\n' || ch == '\'' {
        return false;
    } else {
        cursor += ch.len_utf8();
    }
    source[cursor..].starts_with('\'')
}

#[derive(Debug, Clone, Copy)]
struct DashSpacing {
    both_sides: bool,
    /// Source-absolute byte offset of the first byte AFTER the dash
    /// and its (optional) trailing ASCII whitespace. Callers slice
    /// `source[..]` with this, so it must be source-absolute even
    /// though the spacing probe runs against the dash's containing
    /// line. The earlier shape of this function returned a
    /// line-relative offset, and `clause_replacement` /
    /// `spaced_dash_replacement` happily passed it into `source[..]`
    /// — fine on line 1, panic (or silently wrong rewrite) on every
    /// other line. See chelis#209.
    after_end: usize,
}

/// Probe the ASCII whitespace around the em dash at source-absolute
/// byte offset `dash`. `line` is the dash's containing line and
/// `line_start` its source-absolute start, so the spacing scan stays
/// within the line while the returned `after_end` is source-absolute
/// and safe to feed back into `source[..]` slices.
fn dash_spacing(line: &str, line_start: usize, dash: usize) -> Option<DashSpacing> {
    let dash_in_line = dash.checked_sub(line_start)?;
    let before = dash_in_line
        .checked_sub(1)
        .and_then(|idx| line.as_bytes().get(idx))
        .is_some_and(u8::is_ascii_whitespace);
    let after_start_in_line = dash_in_line + '—'.len_utf8();
    let after = line
        .as_bytes()
        .get(after_start_in_line)
        .is_some_and(u8::is_ascii_whitespace);
    if before != after {
        return None;
    }
    Some(DashSpacing {
        both_sides: before && after,
        after_end: line_start + after_start_in_line + usize::from(after),
    })
}

fn spaced_dash_replacement(
    ctx: &Context<'_>,
    dash: usize,
    text: &str,
    after_end: usize,
) -> Option<Replacement> {
    Some(Replacement {
        path: ctx.path.to_path_buf(),
        start: dash.checked_sub(1)?,
        end: after_end,
        text: text.to_string(),
    })
}

fn clause_replacement(
    ctx: &Context<'_>,
    source: &str,
    dash: usize,
    after_end: usize,
) -> Option<Replacement> {
    let next_start = after_end;
    let next = source[next_start..].chars().next()?;
    if !next.is_ascii_alphabetic() {
        return spaced_dash_replacement(ctx, dash, ": ", after_end);
    }
    let mut capitalized = String::new();
    capitalized.push(next.to_ascii_uppercase());
    Some(Replacement {
        path: ctx.path.to_path_buf(),
        start: dash.checked_sub(1)?,
        end: next_start + next.len_utf8(),
        text: format!(". {capitalized}"),
    })
}

fn line_col(source: &str, offset: usize) -> (usize, usize) {
    let mut line = 1usize;
    let mut line_start = 0usize;
    for (index, byte) in source.bytes().enumerate() {
        if index >= offset {
            break;
        }
        if byte == b'\n' {
            line += 1;
            line_start = index + 1;
        }
    }
    (line, offset.saturating_sub(line_start) + 1)
}

fn offset_from_line_col(source: &str, line_no: usize, col_no: usize) -> Option<usize> {
    let line_start = line_start_offset(source, line_no)?;
    Some(line_start + col_no.checked_sub(1)?)
}

fn line_start_offset(source: &str, line_no: usize) -> Option<usize> {
    if line_no == 0 {
        return None;
    }
    if line_no == 1 {
        return Some(0);
    }
    let mut line = 1usize;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn ctx(src: &str) -> Context<'_> {
        Context {
            root: Path::new("/"),
            path: Path::new("test.rs"),
            source: Some(src),
            surface: Surface::RustSource,
        }
    }

    #[test]
    fn flags_em_dash_in_string_literal() {
        let dash = '\u{2014}';
        let src = format!("fn main() {{ println!(\"one {dash} two\"); }}\n");
        let violations = NoEmDashInPublicStrings.check(&ctx(&src));
        assert_eq!(violations.len(), 1);
        let fix = NoEmDashInPublicStrings
            .fix(&ctx(&src), &violations[0])
            .expect("fix");
        assert_eq!(fix.text, ". T");
    }

    #[test]
    fn ignores_comments() {
        let dash = '\u{2014}';
        let src = format!("// println!(\"one {dash} two\")\n");
        assert!(NoEmDashInPublicStrings.check(&ctx(&src)).is_empty());
    }

    #[test]
    fn ignores_rust_char_literals() {
        let dash = '\u{2014}';
        let src = format!("const EM_DASH: char = '{dash}';\n");
        assert!(NoEmDashInPublicStrings.check(&ctx(&src)).is_empty());
    }

    #[test]
    fn flags_python_single_quoted_strings() {
        let dash = '\u{2014}';
        let src = format!("print('one {dash} two')\n");
        let ctx = Context {
            root: Path::new("/"),
            path: Path::new("test.py"),
            source: Some(&src),
            surface: Surface::PythonSource,
        };
        assert_eq!(NoEmDashInPublicStrings.check(&ctx).len(), 1);
    }

    #[test]
    fn flags_rust_string_after_lifetime_parameter() {
        let dash = '\u{2014}';
        let src = format!("fn f<'a>() {{ println!(\"one {dash} two\"); }}\n");
        let violations = NoEmDashInPublicStrings.check(&ctx(&src));
        assert_eq!(violations.len(), 1);
    }

    #[test]
    fn ignores_deep_semicolon_comments() {
        let dash = '\u{2014}';
        let src = format!("; diagnostic example: (lit \"one {dash} two\")\n");
        let ctx = Context {
            root: Path::new("/"),
            path: Path::new("test.dp"),
            source: Some(&src),
            surface: Surface::DeepSource,
        };
        assert!(NoEmDashInPublicStrings.check(&ctx).is_empty());
    }

    #[test]
    fn fixes_each_string_literal_independently() {
        let dash = '\u{2014}';
        let src = format!(
            "fn main() {{ println!(\"one {dash} two\"); println!(\"three {dash} four\"); }}\n"
        );
        let violations = NoEmDashInPublicStrings.check(&ctx(&src));
        assert_eq!(violations.len(), 2);
        let fix = NoEmDashInPublicStrings
            .fix(&ctx(&src), &violations[0])
            .expect("fix");
        assert_eq!(fix.text, ". T");
    }

    #[test]
    fn refuses_unspaced_auto_fix() {
        let dash = '\u{2014}';
        let src = format!("fn main() {{ println!(\"one{dash}two\"); }}\n");
        let violations = NoEmDashInPublicStrings.check(&ctx(&src));
        assert_eq!(violations.len(), 1);
        assert!(
            NoEmDashInPublicStrings
                .fix(&ctx(&src), &violations[0])
                .is_none()
        );
    }

    // §8.6 scope: the rule targets user-facing strings (diagnostics,
    // log messages, raised-error text). Module/function/class
    // docstrings in Python are narrative prose, not user-facing
    // strings — they must not trip the rule.
    fn py_ctx(src: &str) -> Context<'_> {
        Context {
            root: Path::new("/"),
            path: Path::new("test.py"),
            source: Some(src),
            surface: Surface::PythonSource,
        }
    }

    #[test]
    fn ignores_python_module_docstring() {
        let dash = '\u{2014}';
        let src = format!(
            "\"\"\"Top-level module docstring {dash} narrative prose.\"\"\"\n\nimport sys\n"
        );
        assert!(
            NoEmDashInPublicStrings.check(&py_ctx(&src)).is_empty(),
            "module docstring should be excluded from §8.6",
        );
    }

    #[test]
    fn ignores_python_function_docstring() {
        let dash = '\u{2014}';
        let src = format!(
            "def f(x):\n    \"\"\"Compute the thing {dash} returns float.\"\"\"\n    return x\n"
        );
        assert!(
            NoEmDashInPublicStrings.check(&py_ctx(&src)).is_empty(),
            "function docstring should be excluded from §8.6",
        );
    }

    #[test]
    fn ignores_python_class_docstring() {
        let dash = '\u{2014}';
        let src =
            format!("class Foo:\n    \"\"\"Class docstring {dash} narrative.\"\"\"\n    pass\n");
        assert!(
            NoEmDashInPublicStrings.check(&py_ctx(&src)).is_empty(),
            "class docstring should be excluded from §8.6",
        );
    }

    #[test]
    fn ignores_python_triple_single_quoted_docstring() {
        let dash = '\u{2014}';
        let src =
            format!("def f(x):\n    '''Compute the thing {dash} returns float.'''\n    return x\n");
        assert!(
            NoEmDashInPublicStrings.check(&py_ctx(&src)).is_empty(),
            "triple-single-quoted docstring should be excluded too",
        );
    }

    #[test]
    fn flags_python_print_with_em_dash() {
        // Negative control: a user-facing print() call still fires.
        let dash = '\u{2014}';
        let src = format!("print(\"hello {dash} world\")\n");
        let v = NoEmDashInPublicStrings.check(&py_ctx(&src));
        assert_eq!(v.len(), 1, "print() with em-dash must still fire");
    }

    #[test]
    fn flags_python_raise_with_em_dash() {
        let dash = '\u{2014}';
        let src = format!("raise ValueError(\"bad {dash} bad\")\n");
        let v = NoEmDashInPublicStrings.check(&py_ctx(&src));
        assert_eq!(v.len(), 1, "raise with em-dash must still fire");
    }

    #[test]
    fn flags_python_inline_triple_quoted_call_argument() {
        // A triple-quoted string passed as a call argument is NOT a
        // docstring (it isn't a bare-statement expression). It must
        // still fire because it's reaching a user-facing call site.
        let dash = '\u{2014}';
        let src = format!("print(\"\"\"hello {dash} world\"\"\")\n");
        let v = NoEmDashInPublicStrings.check(&py_ctx(&src));
        assert_eq!(
            v.len(),
            1,
            "triple-quoted in print() is user-facing, not docstring"
        );
    }

    // Guard 2 (test toolchain footgun guards): the em-dash §8.6 rule
    // has been bitten twice by em dashes inside Rust test-function
    // string literals (WS-B2 acceptance panic messages; the RT-1
    // adversarial panic message). The investigation found the rule
    // already CATCHES every one of those constructs; the failure mode
    // was visibility (the blocking errors were buried under hundreds
    // of advisory warnings in `chelis lint --check .` output), fixed
    // in `cmd_lint`. These cases mirror the historical misses and
    // lock that `quoted_spans` keeps reaching each construct, plus
    // the ASCII-hyphen negative parity. See
    // docs/investigations/test_toolchain_guards_design.md.

    #[test]
    fn flags_em_dash_in_raw_string_inside_test_fn() {
        let dash = '\u{2014}';
        let src = format!(
            "#[test]\nfn t() {{\n    let s = r#\"expected NO int64 {dash} silent widening\"#;\n    assert!(!s.is_empty());\n}}\n"
        );
        let violations = NoEmDashInPublicStrings.check(&ctx(&src));
        assert_eq!(
            violations.len(),
            1,
            "em dash in a raw string literal inside a #[test] fn must fire",
        );
    }

    #[test]
    fn flags_em_dash_in_format_macro_arg_inside_test_fn() {
        let dash = '\u{2014}';
        let src = format!(
            "#[test]\nfn t() {{\n    let msg = format!(\"declared return type f64 {dash} must override\");\n    assert!(!msg.is_empty());\n}}\n"
        );
        let violations = NoEmDashInPublicStrings.check(&ctx(&src));
        assert_eq!(
            violations.len(),
            1,
            "em dash in a format! macro string arg inside a #[test] fn must fire",
        );
    }

    #[test]
    fn flags_em_dash_in_multi_line_string_inside_test_fn() {
        // The two historical misses both used `\`-continued multi-line
        // panic-message string literals; reproduce that exact shape.
        let dash = '\u{2014}';
        let src = format!(
            "#[test]\nfn t() {{\n    panic!(\n        \"bare unannotated list must not silently default \\\n         to int64 {dash} got something\"\n    );\n}}\n"
        );
        let violations = NoEmDashInPublicStrings.check(&ctx(&src));
        assert_eq!(
            violations.len(),
            1,
            "em dash in a multi-line string literal inside a #[test] fn must fire",
        );
    }

    #[test]
    fn ignores_ascii_hyphen_in_test_fn_string() {
        // Negative parity: a plain ASCII hyphen-minus must NOT fire,
        // including inside raw strings and macro args.
        let src = "#[test]\nfn t() {\n    let a = \"this-is-fine no em dash here\";\n    let b = r#\"raw-string with-hyphens only\"#;\n    let c = format!(\"format-arg with-a-hyphen\");\n    assert!(!a.is_empty() && !b.is_empty() && !c.is_empty());\n}\n";
        assert!(
            NoEmDashInPublicStrings.check(&ctx(src)).is_empty(),
            "ASCII hyphen-minus must not be flagged as an em dash",
        );
    }
}
