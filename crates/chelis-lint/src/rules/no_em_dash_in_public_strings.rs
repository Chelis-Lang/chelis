//! Rule `no-em-dash-in-public-strings` -- user-facing string literals should
//! use plain punctuation instead of em dashes.

use crate::{Context, Replacement, Rule, Severity, Surface, Violation};

pub struct NoEmDashInPublicStrings;

impl Rule for NoEmDashInPublicStrings {
    fn id(&self) -> &str {
        "no-em-dash-in-public-strings"
    }

    fn spec_ref(&self) -> &str {
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

    fn summary(&self) -> &str {
        "Public-facing string literals should not contain em dashes"
    }

    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn check(&self, ctx: &Context<'_>) -> Vec<Violation> {
        let Some(source) = ctx.source else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for span in quoted_spans(source, ctx.surface) {
            for (relative, _) in source[span.start..span.end].match_indices('—') {
                let absolute = span.start + relative;
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
            .map(|idx| line_start + idx)
            .unwrap_or(source.len());
        let line = &source[line_start..line_end];
        let dash_in_line = dash - line_start;
        let spacing = dash_spacing(line, dash_in_line)?;
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
    after_end: usize,
}

fn dash_spacing(line: &str, dash: usize) -> Option<DashSpacing> {
    let before = dash
        .checked_sub(1)
        .and_then(|idx| line.as_bytes().get(idx))
        .is_some_and(|b| b.is_ascii_whitespace());
    let after_start = dash + '—'.len_utf8();
    let after = line
        .as_bytes()
        .get(after_start)
        .is_some_and(|b| b.is_ascii_whitespace());
    if before != after {
        return None;
    }
    Some(DashSpacing {
        both_sides: before && after,
        after_end: after_start + usize::from(after),
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
}
