//! Rule `prefer-pipe-operator` -- nested first-argument call chains are easier
//! to read as Surf pipes.

use crate::{Context, Replacement, Rule, Severity, Surface, Violation};

/// Mirror of `chelis_surf::format::format_pipe_layout`'s flat-shape gate.
/// The formatter emits the flat single-line `seed |> stage1 |> ...` form
/// only when `total_stages = 1 + stages.len()` is at most 3 and the flat
/// rendering fits in `FMT_LINE_WIDTH` characters. Outside that window the
/// formatter emits one element per line and `format_function_body` wraps
/// the result in `{ ... }` with 2-space indent. The autofix here would
/// only be replacing the call expression's byte span, so it cannot emit
/// the brace-wrapped multi-line form without widening the replacement
/// surface; instead, we restrict the autofix to candidates whose flat
/// shape the formatter would also emit. See
/// `docs/investigations/prefer_pipe_autofix_output_diagnosis.md`.
const FMT_LINE_WIDTH: usize = 80;
const FMT_FLAT_MAX_STAGES: usize = 3;

pub struct PreferPipeOperator;

impl Rule for PreferPipeOperator {
    fn id(&self) -> &'static str {
        "prefer-pipe-operator"
    }

    fn spec_ref(&self) -> &'static str {
        "§3.6"
    }

    fn applies_to(&self) -> &[Surface] {
        &[Surface::SurfSource]
    }

    fn summary(&self) -> &'static str {
        "Prefer `|>` for valid first-argument call chains"
    }

    fn severity(&self) -> Severity {
        Severity::Warning
    }

    fn check(&self, ctx: &Context<'_>) -> Vec<Violation> {
        let Some(source) = ctx.source else {
            return Vec::new();
        };
        // Mirror the post-PR-55 `fix()` syntactic bail-out
        // (`candidate_is_fmt_clean`) at the trigger so the rule does
        // not propose violations the autofix would silently decline
        // on shape grounds. This closes one half of the V2-F3
        // trigger-emit asymmetry; the typed-pipeline half is closed
        // by the CLI driver via `check_mirrors_fix`. See
        // `docs/investigations/prefer_pipe_trigger_emit_diagnosis.md`.
        find_pipe_candidates(source)
            .into_iter()
            .filter(candidate_is_fmt_clean)
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

    fn check_mirrors_fix(&self) -> bool {
        // V2-F3 (PR #58): the warning is only actionable when the
        // rule can offer a safe pipe rewrite. When the autofix bails
        // out — either on syntactic emit-window grounds (filtered
        // inside `check()` above by `candidate_is_fmt_clean`) or on
        // typed-pipeline grounds (rewrite would consume a fan-out
        // variable, etc.) — surfacing the warning is misleading and
        // makes `chelis lint --fix` non-convergent. The CLI driver
        // mirrors the typed-pipeline gate at the warning-emit path
        // for rules that opt in here, so `--fix` reaches a fixpoint
        // for this rule. See
        // `docs/investigations/prefer_pipe_trigger_emit_diagnosis.md`.
        true
    }

    fn fix(&self, ctx: &Context<'_>, violation: &Violation) -> Option<Replacement> {
        let source = ctx.source?;
        let start = offset_from_line_col(source, violation.line?, violation.col?)?;
        let candidate = pipe_candidate_at(source, start)?;
        if !candidate_is_fmt_clean(&candidate) {
            // The candidate's flat single-line text would not survive
            // `chelis fmt --check`: the formatter would re-emit the pipe
            // as a brace-wrapped multi-line form, which the autofix
            // cannot produce inside its call-expression-only replacement
            // span. Drop the autofix; the `check()` warning still fires
            // so the user can rewrite manually. See Finding 3b at
            // `docs/investigations/prefer_pipe_autofix_output_diagnosis.md`.
            return None;
        }
        Some(Replacement {
            path: ctx.path.to_path_buf(),
            start: candidate.start,
            end: candidate.end,
            text: candidate.replacement,
        })
    }
}

/// Predict whether `chelis_surf::format::format_pipe_layout` would emit
/// the flat single-line form for the assembled pipe replacement. The
/// formatter gates the flat shape on
/// `total_stages <= FMT_FLAT_MAX_STAGES && flat.chars().count() <= FMT_LINE_WIDTH`,
/// where `total_stages = 1 + stages.len()`.
fn candidate_is_fmt_clean(candidate: &Candidate) -> bool {
    let total_stages = 1 + candidate.stage_count;
    total_stages <= FMT_FLAT_MAX_STAGES && candidate.replacement.chars().count() <= FMT_LINE_WIDTH
}

#[derive(Debug, Clone)]
struct Candidate {
    start: usize,
    end: usize,
    replacement: String,
    /// Number of pipe stages following the seed (so total parts in the
    /// pipe expression is `1 + stage_count`). Tracked separately from
    /// the replacement text so the fmt-clean predicate does not have to
    /// re-parse ` |> ` separators out of string-argument contents.
    stage_count: usize,
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
            let stage_count = stages.len();
            let mut replacement = seed;
            for stage in &stages {
                replacement.push_str(" |> ");
                replacement.push_str(stage);
            }
            return Some(Candidate {
                start: call.start,
                end: call.end,
                replacement,
                stage_count,
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

    #[test]
    fn check_suppresses_when_fmt_would_emit_multi_line() {
        // V2-F3 (PR #58): the `check()` trigger mirrors the
        // post-PR-55 `fix()` syntactic bail-out so the rule does not
        // propose violations the autofix would silently decline on
        // shape grounds. A four-part pipe (`sigmoid(relu(neg(x)))`
        // -> `x |> neg |> relu |> sigmoid`) would format multi-line,
        // so the trigger drops the candidate. The CLI driver further
        // suppresses warnings whose autofix is rejected by the
        // typed-pipeline gate via `check_mirrors_fix`. See
        // `docs/investigations/prefer_pipe_trigger_emit_diagnosis.md`.
        let src = "def f(x: f32) -> f32 = sigmoid(relu(neg(x)))\n";
        let violations = PreferPipeOperator.check(&ctx(src));
        assert!(
            violations.is_empty(),
            "expected check() to drop a four-part pipe candidate; got {violations:?}"
        );
    }

    #[test]
    fn check_mirrors_fix_opts_in_for_prefer_pipe_operator() {
        // V2-F3: `prefer-pipe-operator` opts in to the
        // `check_mirrors_fix` filter so the CLI suppresses warnings
        // whose autofix would be silently dropped by the typed-pipeline
        // gate (e.g., fan-out re-use of the seed in
        // `add(mul(x, x), x)`). This pins the opt-in.
        let rule = PreferPipeOperator;
        assert!(rule.check_mirrors_fix());
    }

    #[test]
    fn fix_emits_flat_form_at_exactly_three_pipe_parts() {
        // `relu(neg(x))` rewrites to a 3-part pipe (`x |> neg |> relu`),
        // which the formatter does emit flat. The autofix proposes the
        // flat text verbatim.
        let src = "def f(x: f32) -> f32 = relu(neg(x))\n";
        let violations = PreferPipeOperator.check(&ctx(src));
        assert_eq!(violations.len(), 1);
        let replacement = PreferPipeOperator
            .fix(&ctx(src), &violations[0])
            .expect("fix should propose the three-part pipe");
        assert_eq!(replacement.text, "x |> neg |> relu");
    }
}
