//! Rule `redundant-linearity-call` — source-level `copy()` and `drop()` are
//! accepted for migration compatibility, but the implicit linearity model
//! inserts equivalent IR nodes automatically.
//!
//! This rule is advisory-only. It is wired through `registry::advisory_rules`
//! instead of `registry::all_rules`, so it does not fail style-gated build,
//! check, eval, or validate paths.

use crate::{Context, Rule, Surface, Violation};
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

    fn check(&self, ctx: &Context<'_>) -> Vec<Violation> {
        let Some(source) = ctx.source else {
            return Vec::new();
        };
        let mut out = Vec::new();
        let mut offset = 0usize;
        for line in source.lines() {
            let code = line.split("//").next().unwrap_or(line);
            for matched in call_re().captures_iter(code) {
                let whole = matched.get(0).unwrap();
                let call = matched.get(1).unwrap().as_str();
                let absolute = offset + whole.start();
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
            offset += line.len() + 1;
        }
        out
    }
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
}
