//! Rule `prefer-pipe-operator` -- nested first-argument call chains are easier
//! to read as Surf pipes.

use crate::{Context, Replacement, Rule, Severity, Surface, Violation};

pub struct PreferPipeOperator;

impl Rule for PreferPipeOperator {
    fn id(&self) -> &str {
        "prefer-pipe-operator"
    }

    fn spec_ref(&self) -> &str {
        "§3.6"
    }

    fn applies_to(&self) -> &[Surface] {
        &[Surface::SurfSource]
    }

    fn summary(&self) -> &str {
        "Prefer `|>` for valid first-argument call chains"
    }

    fn severity(&self) -> Severity {
        Severity::Warning
    }

    fn check(&self, ctx: &Context<'_>) -> Vec<Violation> {
        let Some(source) = ctx.source else {
            return Vec::new();
        };
        find_pipe_candidates(source)
            .into_iter()
            .map(|candidate| {
                let (line, col) = line_col(source, candidate.start);
                Violation {
                    rule_id: self.id().to_string(),
                    spec_ref: self.spec_ref().to_string(),
                    path: ctx.path.to_path_buf(),
                    line: Some(line),
                    col: Some(col),
                    message: "nested first-argument call chain can be written with `|>`"
                        .to_string(),
                }
            })
            .collect()
    }

    fn fix_requires_typed_pipeline_check(&self) -> bool {
        // The pipe rewrite is safe iff the rewritten program still passes
        // the typed/effect/linearity pipeline. Item 5 (PR #34) added the
        // CLI-driver gate that runs the same pipeline `chelis check` uses;
        // opting in here re-enables the autofix that 477bd0d disabled.
        // Architectural decision in
        // `docs/investigations/redundant_linearity_autofix_architecture.md`
        // (Path 1B). The per-rule re-enable rationale is documented in
        // `docs/investigations/pipe_autofix_and_bare_keyword_extras_diagnosis.md`.
        true
    }

    fn fix(&self, ctx: &Context<'_>, violation: &Violation) -> Option<Replacement> {
        let source = ctx.source?;
        let start = offset_from_line_col(source, violation.line?, violation.col?)?;
        let candidate = pipe_candidate_at(source, start)?;
        Some(Replacement {
            path: ctx.path.to_path_buf(),
            start: candidate.start,
            end: candidate.end,
            text: candidate.replacement,
        })
    }
}

#[derive(Debug, Clone)]
struct Candidate {
    start: usize,
    end: usize,
    replacement: String,
}

#[derive(Debug, Clone)]
struct Call {
    name: String,
    args: Vec<Arg>,
    start: usize,
    end: usize,
}

#[derive(Debug, Clone)]
struct Arg {
    start: usize,
    text: String,
}

fn find_pipe_candidates(source: &str) -> Vec<Candidate> {
    let mut out = Vec::new();
    let mut offset = 0usize;
    for line in source.lines() {
        for segment in code_segments(line) {
            let mut index = segment.start;
            while index < segment.end {
                let absolute = offset + index;
                if let Some(candidate) = pipe_candidate_at(source, absolute)
                    && candidate.start == absolute
                {
                    index += candidate.end.saturating_sub(candidate.start).max(1);
                    out.push(candidate);
                    continue;
                }
                index += 1;
            }
        }
        offset += line.len() + 1;
    }
    out
}

fn pipe_candidate_at(source: &str, start: usize) -> Option<Candidate> {
    let call = parse_call_at(source, start)?;
    let mut stages: Vec<String> = Vec::new();
    let mut current = call.clone();
    loop {
        let first = current.args.first()?;
        let Some(inner) = parse_arg_as_call(source, first) else {
            if stages.is_empty() {
                return None;
            }
            // Innermost first argument becomes the seed of the pipe
            // (e.g., the `x` in `outer(inner(x), scale)` → `x |> inner |> outer(scale)`).
            let seed = first.text.trim().to_string();
            stages.push(render_stage(&current));
            stages.reverse();
            let mut replacement = seed;
            for stage in stages {
                replacement.push_str(" |> ");
                replacement.push_str(&stage);
            }
            return Some(Candidate {
                start: call.start,
                end: call.end,
                replacement,
            });
        };
        stages.push(render_stage(&current));
        current = inner;
    }
}

fn render_stage(call: &Call) -> String {
    let rest: Vec<String> = call
        .args
        .iter()
        .skip(1)
        .map(|arg| arg.text.trim().to_string())
        .collect();
    if rest.is_empty() {
        call.name.clone()
    } else {
        format!("{}({})", call.name, rest.join(", "))
    }
}

fn parse_arg_as_call(source: &str, arg: &Arg) -> Option<Call> {
    let leading = arg.text.len() - arg.text.trim_start().len();
    let trailing = arg.text.trim_end().len();
    let start = arg.start + leading;
    let call = parse_call_at(source, start)?;
    (call.end == arg.start + trailing).then_some(call)
}

fn parse_call_at(source: &str, start: usize) -> Option<Call> {
    if start >= source.len() || !is_ident_start(source.as_bytes()[start]) {
        return None;
    }
    let mut cursor = start + 1;
    while cursor < source.len() && is_ident_continue(source.as_bytes()[cursor]) {
        cursor += 1;
    }
    let name = source[start..cursor].to_string();
    while cursor < source.len() && source.as_bytes()[cursor].is_ascii_whitespace() {
        cursor += 1;
    }
    if source.as_bytes().get(cursor) != Some(&b'(') {
        return None;
    }
    let close = matching_paren(source, cursor)?;
    let args = split_args(source, cursor + 1, close)?;
    Some(Call {
        name,
        args,
        start,
        end: close + 1,
    })
}

fn split_args(source: &str, start: usize, end: usize) -> Option<Vec<Arg>> {
    if source[start..end].trim().is_empty() {
        return Some(Vec::new());
    }
    let mut args = Vec::new();
    let mut depth = 0usize;
    let mut arg_start = start;
    let mut cursor = start;
    let mut in_string = false;
    let mut escaped = false;
    while cursor < end {
        let ch = source[cursor..].chars().next()?;
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
            ')' | ']' | '}' => depth = depth.checked_sub(1)?,
            ',' if depth == 0 => {
                args.push(Arg {
                    start: arg_start,
                    text: source[arg_start..cursor].to_string(),
                });
                arg_start = cursor + 1;
            }
            _ => {}
        }
        cursor += ch.len_utf8();
    }
    if in_string {
        return None;
    }
    args.push(Arg {
        start: arg_start,
        text: source[arg_start..end].to_string(),
    });
    Some(args)
}

fn matching_paren(source: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut cursor = open;
    let mut in_string = false;
    let mut escaped = false;
    while cursor < source.len() {
        let ch = source[cursor..].chars().next()?;
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
            '(' => depth += 1,
            ')' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(cursor);
                }
            }
            _ => {}
        }
        cursor += ch.len_utf8();
    }
    None
}

fn is_ident_start(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_'
}

fn is_ident_continue(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.')
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
    if line_no == 0 || col_no == 0 {
        return None;
    }
    let mut line = 1usize;
    let mut line_start = 0usize;
    for (index, byte) in source.bytes().enumerate() {
        if line == line_no {
            return Some(line_start + col_no - 1);
        }
        if byte == b'\n' {
            line += 1;
            line_start = index + 1;
        }
    }
    (line == line_no).then_some(line_start + col_no - 1)
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

    #[test]
    fn detects_nested_first_arg_chain() {
        // The source-text walker proposes the pipe rewrite; the CLI
        // driver's Path 1B gate (see
        // `docs/investigations/redundant_linearity_autofix_architecture.md`)
        // verifies safety before writing. The unit test pins the local
        // proposal shape only.
        let src = "def f(x: f32) -> f32 = outer(inner(x), scale)\n";
        let violations = PreferPipeOperator.check(&ctx(src));
        assert_eq!(violations.len(), 1);
        let rule = PreferPipeOperator;
        assert!(rule.fix_requires_typed_pipeline_check());
        let replacement = rule
            .fix(&ctx(src), &violations[0])
            .expect("fix should propose the pipe rewrite");
        assert_eq!(replacement.text, "x |> inner |> outer(scale)");
    }

    #[test]
    fn ignores_single_call() {
        let src = "def f(x: f32) -> f32 = outer(x)\n";
        assert!(PreferPipeOperator.check(&ctx(src)).is_empty());
    }

    #[test]
    fn ignores_string_literals() {
        let src = "def f() -> string = \"outer(inner(x), scale)\"\n";
        assert!(PreferPipeOperator.check(&ctx(src)).is_empty());
    }

    #[test]
    fn preserves_string_argument_contents() {
        // The inner call's `"a,b"` string argument keeps its exact text
        // when assembled into the rewrite's trailing `outer("a,b")` stage.
        let src = "def f(x: f32) -> f32 = outer(inner(x), \"a,b\")\n";
        let violations = PreferPipeOperator.check(&ctx(src));
        assert_eq!(violations.len(), 1);
        let replacement = PreferPipeOperator
            .fix(&ctx(src), &violations[0])
            .expect("fix should propose the pipe rewrite");
        assert_eq!(replacement.text, "x |> inner |> outer(\"a,b\")");
    }

    #[test]
    fn fix_proposes_literal_seed_for_inner_cast_chain() {
        // The walker descends through the outer `beta(...)`'s first arg
        // (`cast(2.0, f32)`), then through that call's first arg (the
        // literal `2.0`). The proposed rewrite uses `2.0` as the pipe
        // seed and accumulates the enclosing calls as pipe stages.
        // This is a syntactic proposal only; the CLI driver's Path 1B
        // gate decides whether the rewrite is semantically safe.
        let src = "def f() -> f32 = beta(cast(2.0, f32), cast(3.0, f32))\n";
        let violations = PreferPipeOperator.check(&ctx(src));
        assert_eq!(violations.len(), 1);
        let replacement = PreferPipeOperator
            .fix(&ctx(src), &violations[0])
            .expect("fix should propose the rewrite");
        assert_eq!(replacement.text, "2.0 |> cast(f32) |> beta(cast(3.0, f32))");
    }
}
