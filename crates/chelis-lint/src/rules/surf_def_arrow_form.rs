//! Rule `surf-def-arrow-form` — `def name(params) -> T = expr` is the
//! canonical Surf signature form; the colon variant
//! `def name(params) : T = expr` is flagged.
//!
//! Spec authority: `spec/01-nomenclature.md` §3.5. The canonical parser
//! accepts only the arrow form. The lint still scans raw source so style-gate
//! callers can name the legacy colon spelling directly instead of reporting
//! only a generic parse error. `chelis migrate surf --from 0.18` owns the
//! mechanical rewrite; `chelis fmt` is not a dialect translator.
//!
//! Implementation notes: the rule scans each `def NAME` start, balances
//! the parameter parentheses (so types like `(i32, i32)` don't fool a
//! naive regex), then checks the first non-whitespace token following
//! the parameter list. If it is `:`, the rule fires. If it is `->`, `=`,
//! or anything else (signature wrap onto next line followed by a `:` is
//! reported only when the colon appears within the same logical
//! signature; we conservatively limit scanning to the next ~512 bytes
//! to avoid pathological scans).

use crate::{Context, Rule, Surface, Violation};
use regex::Regex;
use std::sync::OnceLock;

static DEF_HEAD_RE: OnceLock<Regex> = OnceLock::new();

fn def_head_re() -> &'static Regex {
    // Matches `def NAME` at line start, capturing the byte position so
    // we can hand off to a balanced scanner.
    DEF_HEAD_RE.get_or_init(|| Regex::new(r"(?m)^[ \t]*def[ \t]+([A-Za-z_][A-Za-z0-9_]*)").unwrap())
}

pub struct SurfDefArrowForm;

impl Rule for SurfDefArrowForm {
    fn id(&self) -> &str {
        "surf-def-arrow-form"
    }

    fn spec_ref(&self) -> &str {
        "§3.5"
    }

    fn applies_to(&self) -> &[Surface] {
        &[Surface::SurfSource]
    }

    fn summary(&self) -> &str {
        "Surf `def` declarations use `def name(params) -> T = expr`, not the colon form `def name(params) : T = expr`"
    }

    fn check(&self, ctx: &Context<'_>) -> Vec<Violation> {
        let Some(source) = ctx.source else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for caps in def_head_re().captures_iter(source) {
            let head_match = caps.get(0).unwrap();
            let name = caps.get(1).unwrap().as_str();
            // Start scanning right after the captured `def NAME` head.
            let scan_start = head_match.end();
            let Some(annotation_kind) = scan_for_annotation(&source[scan_start..]) else {
                continue;
            };
            if !matches!(annotation_kind, AnnotationKind::Colon) {
                continue;
            }
            let line = source[..head_match.start()]
                .bytes()
                .filter(|&b| b == b'\n')
                .count()
                + 1;
            out.push(Violation {
                rule_id: self.id().to_string(),
                spec_ref: self.spec_ref().to_string(),
                path: ctx.path.to_path_buf(),
                line: Some(line),
                col: None,
                message: format!(
                    "function `{name}` uses colon-form return type (`def {name}(...) : T = ...`); use arrow form `def {name}(...) -> T = ...` per §3.5"
                ),
            });
        }
        out
    }
}

#[derive(Debug, PartialEq, Eq)]
enum AnnotationKind {
    /// `def name(...) : T = ...` — the form the rule flags.
    Colon,
    /// `def name(...) -> T = ...` — canonical.
    Arrow,
    /// `def name(...) = ...` — no return-type annotation.
    None,
}

/// Scan the body following `def NAME`, balance the parameter parens
/// (handling optional generic-parameter brackets `[a, b]` first), and
/// classify the token that introduces the body.
///
/// Returns `None` if the scanner cannot find a parameter-list close
/// within the budget — partial input or pathological lines are ignored
/// rather than reported as false-positive violations.
fn scan_for_annotation(s: &str) -> Option<AnnotationKind> {
    const BUDGET: usize = 4096;
    let bytes = s.as_bytes();
    let limit = bytes.len().min(BUDGET);
    let mut i = 0;
    // Optional generic-param brackets: `def name[a, b](...)`.
    while i < limit && (bytes[i] == b' ' || bytes[i] == b'\t' || bytes[i] == b'\n') {
        i += 1;
    }
    if i < limit && bytes[i] == b'[' {
        let mut depth = 1usize;
        i += 1;
        while i < limit && depth > 0 {
            match bytes[i] {
                b'[' => depth += 1,
                b']' => depth -= 1,
                _ => {}
            }
            i += 1;
        }
        if depth != 0 {
            return None;
        }
    }
    while i < limit && (bytes[i] == b' ' || bytes[i] == b'\t' || bytes[i] == b'\n') {
        i += 1;
    }
    // Parameter parens are required in canonical function definitions.
    // Retain a narrow scan of legacy no-parameter-list input so a direct
    // colon annotation still receives this rule's actionable diagnostic.
    if i >= limit || bytes[i] != b'(' {
        // No parameter list — could be legacy `def name = expr` or
        // `def name : T = expr`. Treat only the latter as a colon-form
        // violation; canonical parsing reports the missing `()` separately.
        if i < limit && bytes[i] == b':' {
            return Some(AnnotationKind::Colon);
        }
        if i < limit && bytes[i] == b'-' && i + 1 < limit && bytes[i + 1] == b'>' {
            return Some(AnnotationKind::Arrow);
        }
        if i < limit && bytes[i] == b'=' {
            return Some(AnnotationKind::None);
        }
        return None;
    }
    // Balance parameter parens.
    let mut depth = 1usize;
    i += 1;
    while i < limit && depth > 0 {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => depth -= 1,
            _ => {}
        }
        i += 1;
    }
    if depth != 0 {
        return None;
    }
    // Skip whitespace; classify the next token.
    while i < limit && (bytes[i] == b' ' || bytes[i] == b'\t' || bytes[i] == b'\n') {
        i += 1;
    }
    if i >= limit {
        return None;
    }
    match bytes[i] {
        b':' => Some(AnnotationKind::Colon),
        b'-' if i + 1 < limit && bytes[i + 1] == b'>' => Some(AnnotationKind::Arrow),
        b'=' => Some(AnnotationKind::None),
        _ => None,
    }
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
        SurfDefArrowForm.check(&ctx)
    }

    #[test]
    fn accepts_arrow_form() {
        assert!(run("def add(x: i32, y: i32) -> i32 = x + y\n").is_empty());
        assert!(run("def f(x: f32) -> f32 = x\n").is_empty());
        assert!(run("def two() -> i32 = 2\n").is_empty());
    }

    #[test]
    fn accepts_no_annotation() {
        assert!(run("def add(x: i32, y: i32) = x + y\n").is_empty());
    }

    #[test]
    fn rejects_colon_form() {
        let v = run("def add(x: i32, y: i32) : i32 = x + y\n");
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("colon-form"));
        assert!(v[0].message.contains("add"));
        assert_eq!(v[0].line, Some(1));
    }

    #[test]
    fn rejects_colon_form_no_space() {
        let v = run("def add(x: i32):i32 = x\n");
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn handles_nested_paren_types() {
        // Tuple type in parameter must not break paren balancing.
        assert!(run("def first(p: (i32, i32)) -> i32 = p\n").is_empty());
        let v = run("def first(p: (i32, i32)) : i32 = p\n");
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn handles_generic_parameters() {
        assert!(run("def head[a](xs: List[a]) -> a = xs\n").is_empty());
        let v = run("def head[a](xs: List[a]) : a = xs\n");
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn ignores_match_arms_in_body() {
        // `): ` inside the BODY (after the first `=`) must not be
        // misinterpreted; the scanner stops at the parameter-list close.
        let src = "def fff(x: i32) -> i32 = match foo with { Some(y) -> 1 | None -> 0 }\n";
        assert!(run(src).is_empty());
    }

    #[test]
    fn flags_each_violation_on_separate_lines() {
        let src = "def a(x: i32) -> i32 = x\ndef b(x: i32) : i32 = x\ndef c(x: i32) -> i32 = x\ndef d(x: i32) : i32 = x\n";
        let v = run(src);
        assert_eq!(v.len(), 2);
        assert!(v.iter().any(|x| x.message.contains("`b`")));
        assert!(v.iter().any(|x| x.message.contains("`d`")));
    }

    #[test]
    fn handles_wrapped_signature() {
        // Realistic multi-line signature: `def NAME` stays on one line,
        // parameters wrap across lines, return-type appears on a later
        // line. The paren-balancing scanner handles this; only `def`
        // and NAME need to be on the same line for the regex to fire.
        let src = "def add(\n  x: i32,\n  y: i32,\n)\n  : i32 = x + y\n";
        let v = run(src);
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn scan_for_annotation_classifies_each_form() {
        assert_eq!(
            scan_for_annotation("(x: i32) -> i32 = x"),
            Some(AnnotationKind::Arrow)
        );
        assert_eq!(
            scan_for_annotation("(x: i32) : i32 = x"),
            Some(AnnotationKind::Colon)
        );
        assert_eq!(
            scan_for_annotation("(x: i32) = x"),
            Some(AnnotationKind::None)
        );
        // Unbalanced — return None, not a panic or false positive.
        assert_eq!(scan_for_annotation("(x: i32"), None);
    }
}
