//! Rule `redundant-linearity-call` — source-level `copy()` and `drop()` are
//! accepted for migration compatibility, but the implicit linearity model
//! inserts equivalent IR nodes automatically.
//!
//! This rule is advisory-only. It is wired through `registry::advisory_rules`
//! instead of `registry::all_rules`, so it does not fail style-gated build,
//! check, eval, or validate paths.

use crate::{Context, Replacement, Rule, Severity, Surface, Violation};
use regex::Regex;
use std::sync::OnceLock;

static CALL_RE: OnceLock<Regex> = OnceLock::new();

fn call_re() -> &'static Regex {
    CALL_RE.get_or_init(|| Regex::new(r"\b(copy|drop)[ \t\r\n]*\(").unwrap())
}

pub struct RedundantLinearityCall;

impl Rule for RedundantLinearityCall {
    fn id(&self) -> &str {
        "redundant-linearity-call"
    }

    fn spec_ref(&self) -> &str {
        "implicit-linearity"
    }

    fn applies_to(&self) -> &[Surface] {
        &[Surface::SurfSource]
    }

    fn summary(&self) -> &str {
        "Source-level `copy()` and `drop()` are valid but redundant under implicit linearity"
    }

    fn severity(&self) -> Severity {
        Severity::Warning
    }

    fn check(&self, ctx: &Context<'_>) -> Vec<Violation> {
        let Some(source) = ctx.source else {
            return Vec::new();
        };
        let mut out = Vec::new();
        let mut offset = 0usize;
        for line in source.lines() {
            for segment in code_segments(line) {
                let code = &line[segment.clone()];
                for matched in call_re().captures_iter(code) {
                    let whole = matched.get(0).unwrap();
                    let call = matched.get(1).unwrap().as_str();
                    let absolute = offset + segment.start + whole.start();
                    let open = absolute + whole.as_str().len() - 1;
                    let Some(close) = matching_paren(source, open) else {
                        continue;
                    };
                    if !has_single_top_level_argument(source, open, close) {
                        continue;
                    }
                    let (line_no, col_no) = line_col(source, absolute);
                    out.push(Violation {
                    rule_id: self.id().to_string(),
                    spec_ref: self.spec_ref().to_string(),
                    path: ctx.path.to_path_buf(),
                    line: Some(line_no),
                    col: Some(col_no),
                    message: format!(
                        "`{call}()` is valid for migration compatibility but redundant; implicit linearity inserts the corresponding IR node"
                    ),
                    });
                }
            }
            offset += line.len() + 1;
        }
        out
    }

    fn fix_requires_typed_pipeline_check(&self) -> bool {
        // The strip `copy(x) -> x` is safe iff the resulting program still
        // type/effect/linearity-checks. After Item 1 of the 0.7.6 toolchain
        // hygiene workstream (PR #29) extended implicit linearity to handle
        // var-RHS let-bindings, the broad class of programs the lint targets
        // is safe to rewrite — but the CLI fix driver still verifies each
        // candidate against the typed pipeline before writing. See
        // `docs/investigations/redundant_linearity_autofix_architecture.md`
        // for the architectural decision (Path 1B).
        true
    }

    fn fix(&self, ctx: &Context<'_>, violation: &Violation) -> Option<Replacement> {
        let source = ctx.source?;
        let line_start = line_start_offset(source, violation.line?)?;
        let col = violation.col?.checked_sub(1)?;
        let start = line_start + col;
        let call_match = call_re().find_at(source, start)?;
        if call_match.start() != start {
            return None;
        }
        let open = call_match.end().checked_sub(1)?;
        let close = matching_paren(source, open)?;
        if !has_single_top_level_argument(source, open, close) {
            return None;
        }
        let inner = source[open + 1..close].trim();
        Some(Replacement {
            path: ctx.path.to_path_buf(),
            start,
            end: close + 1,
            text: inner.to_string(),
        })
    }
}

fn has_single_top_level_argument(source: &str, open: usize, close: usize) -> bool {
    let inner = &source[open + 1..close];
    if inner.trim().is_empty() {
        return false;
    }
    let mut depth = 0usize;
    let mut cursor = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    while cursor < inner.len() {
        let ch = inner[cursor..].chars().next().unwrap();
        if in_string {
            if escaped {
                escaped = false;
            } else {
                match ch {
                    '\\' => escaped = true,
                    '"' => in_string = false,
                    _ => {}
                }
            }
            cursor += ch.len_utf8();
            continue;
        }
        match ch {
            '"' => in_string = true,
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => {
                let Some(next) = depth.checked_sub(1) else {
                    return false;
                };
                depth = next;
            }
            ',' if depth == 0 => return false,
            _ => {}
        }
        cursor += ch.len_utf8();
    }
    !in_string && depth == 0
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

fn matching_paren(source: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (offset, ch) in source[open..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(open + offset);
                }
            }
            _ => {}
        }
    }
    None
}

fn code_segments(line: &str) -> Vec<std::ops::Range<usize>> {
    let mut segments = Vec::new();
    let mut start = 0usize;
    let mut cursor = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    while cursor < line.len() {
        let ch = line[cursor..].chars().next().unwrap();
        if in_string {
            if escaped {
                escaped = false;
            } else {
                match ch {
                    '\\' => escaped = true,
                    '"' => {
                        in_string = false;
                        start = cursor + ch.len_utf8();
                    }
                    _ => {}
                }
            }
            cursor += ch.len_utf8();
            continue;
        }
        if line[cursor..].starts_with("//") {
            break;
        }
        if ch == '"' {
            if start < cursor {
                segments.push(start..cursor);
            }
            in_string = true;
        }
        cursor += ch.len_utf8();
    }
    if !in_string && start < cursor {
        segments.push(start..cursor);
    }
    segments
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn run(src: &str) -> Vec<Violation> {
        let path = Path::new("test.ch");
        let ctx = Context {
            root: Path::new("/"),
            path,
            source: Some(src),
            surface: Surface::SurfSource,
        };
        RedundantLinearityCall.check(&ctx)
    }

    #[test]
    fn flags_copy_and_drop_calls() {
        let violations = run("def f(x: tensor[2, f32]) -> tensor[2, f32] = copy(x)\n\
             result = drop(f(to_tensor([1.0, 2.0])))\n");
        assert_eq!(violations.len(), 2);
        assert_eq!(violations[0].rule_id, "redundant-linearity-call");
        assert!(violations[0].message.contains("`copy()`"));
        assert!(violations[1].message.contains("`drop()`"));
    }

    #[test]
    fn ignores_comments_and_other_names() {
        let violations = run("// copy(x)\n\
             def f(x: tensor[2, f32]) -> tensor[2, f32] = dropout(x, 0.5)\n");
        assert!(violations.is_empty());
    }

    #[test]
    fn ignores_string_literals() {
        let violations = run("def f() -> string = \"copy(x)\"\n");
        assert!(violations.is_empty());
    }

    #[test]
    fn ignores_list_drop_with_two_arguments() {
        let violations = run("def f(ys: list[int64]) -> list[int64] = drop(ys, cast(1, int64))\n");
        assert!(violations.is_empty());
    }

    #[test]
    fn fix_proposes_strip_and_opts_into_typed_pipeline_verification() {
        // The rule proposes the source-level strip `copy(x) -> x`. The
        // typed-pipeline safety check is enforced by the CLI fix driver
        // (see `apply_lint_fixes` in chelis-cli), keyed on
        // `fix_requires_typed_pipeline_check`. The unit test here only
        // pins the rule's local behavior: it proposes the strip and
        // signals that the driver must verify before writing.
        let src = "def f(x: tensor[2, f32]) -> tensor[2, f32] = add(copy(x), x)\n";
        let violations = run(src);
        assert_eq!(violations.len(), 1);
        let path = Path::new("test.ch");
        let ctx = Context {
            root: Path::new("/"),
            path,
            source: Some(src),
            surface: Surface::SurfSource,
        };
        let rule = RedundantLinearityCall;
        assert!(rule.fix_requires_typed_pipeline_check());
        let replacement = rule
            .fix(&ctx, &violations[0])
            .expect("fix should propose the strip");
        // The replacement covers exactly the `copy(x)` span and replaces
        // it with the inner argument `x`.
        let pre = &src[..replacement.start];
        let post = &src[replacement.end..];
        assert_eq!(replacement.text, "x");
        assert_eq!(
            format!("{pre}{}{post}", replacement.text),
            "def f(x: tensor[2, f32]) -> tensor[2, f32] = add(x, x)\n",
        );
    }
}
