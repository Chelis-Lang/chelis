//! Rule `surf-value-snake-case` — Surf top-level `def` names are `snake_case`.
//!
//! Spec authority: `spec/01-nomenclature.md` §3.2 (Functions and values).
//!
//! `snake_case` here means: starts with a lowercase ASCII letter, followed
//! by lowercase ASCII letters, ASCII digits, or `_`. No uppercase letters
//! anywhere; no leading underscore. Single-letter math-style function
//! names (`f`, `g`) satisfy this trivially.
//!
//! The §3.2 single-letter math-notation carve-out (chelis#437) admits a
//! single ASCII-uppercase letter as a value BINDING name (`S = ...`) and
//! as a parameter, but NOT as a `def`/`sig` function name: an applied
//! uppercase head `N(x)` resolves to a constructor, so an uppercase
//! function name would be silently shadowed. This rule therefore keeps
//! flagging uppercase `def` names; it never sees bare value bindings
//! (the regex matches only `def name`, which value bindings lack).
//!
//! Detects `^def name(...)` and `^def name<...>` lines where `name`
//! contains an uppercase letter or leading underscore. The regex is
//! permissive on the captured identifier so the rule (not the regex)
//! decides what to flag.

use crate::{Context, Rule, Surface, Violation};
use regex::Regex;
use std::sync::OnceLock;

static DEF_RE: OnceLock<Regex> = OnceLock::new();

fn def_re() -> &'static Regex {
    DEF_RE.get_or_init(|| Regex::new(r"(?m)^[ \t]*def[ \t]+([A-Za-z_][A-Za-z0-9_]*)").unwrap())
}

pub struct SurfValueSnakeCase;

impl Rule for SurfValueSnakeCase {
    fn id(&self) -> &'static str {
        "surf-value-snake-case"
    }

    fn spec_ref(&self) -> &'static str {
        "§3.2"
    }

    fn applies_to(&self) -> &[Surface] {
        &[Surface::SurfSource]
    }

    fn summary(&self) -> &'static str {
        "Surf `def` names are snake_case: lowercase ASCII, optional digits and underscores, no leading underscore"
    }

    fn check(&self, ctx: &Context<'_>) -> Vec<Violation> {
        let Some(source) = ctx.source else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for caps in def_re().captures_iter(source) {
            let name = caps.get(1).unwrap().as_str();
            if is_snake_case(name) {
                continue;
            }
            let match_start = caps.get(0).unwrap().start();
            let line = source[..match_start]
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
                    "function `{name}` is not snake_case (§3.2 requires lowercase ASCII, digits, and `_`; no uppercase letters; no leading underscore)"
                ),
            });
        }
        out
    }
}

fn is_snake_case(s: &str) -> bool {
    let mut chars = s.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_ascii_lowercase() {
        return false;
    }
    chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
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
        SurfValueSnakeCase.check(&ctx)
    }

    #[test]
    fn accepts_snake_case_defs() {
        assert!(run("def add_one(x: i32) -> i32 = x\n").is_empty());
        assert!(run("def predict(x: f32) -> f32 = x\n").is_empty());
        assert!(run("def golden_section_search() -> f32 = 0.5\n").is_empty());
        assert!(run("def f(x: f32) -> f32 = x\n").is_empty());
        assert!(run("def s0() -> f32 = 1.0\n").is_empty());
        assert!(run("def bs_call_scalar(s: f32) -> f32 = s\n").is_empty());
    }

    #[test]
    fn rejects_single_letter_uppercase_function_name() {
        // The §3.2 carve-out (chelis#437) does NOT extend to function
        // names: an applied uppercase head resolves to a constructor, so a
        // `def N` would be silently shadowed. The rule keeps flagging it.
        let v = run("def N(x: f32) -> f32 = x\n");
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains('N'));
    }

    #[test]
    fn rejects_camel_case() {
        let v = run("def addOne(x: i32) -> i32 = x\n");
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("addOne"));
        assert_eq!(v[0].rule_id, "surf-value-snake-case");
        assert_eq!(v[0].line, Some(1));
    }

    #[test]
    fn rejects_pascal_case_function() {
        let v = run("def AddOne(x: i32) -> i32 = x\n");
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("AddOne"));
    }

    #[test]
    fn rejects_screaming_snake() {
        let v = run("def MAX_DIM() -> i32 = 8\n");
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("MAX_DIM"));
    }

    #[test]
    fn rejects_leading_underscore() {
        let v = run("def _internal(x: i32) -> i32 = x\n");
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("_internal"));
    }

    #[test]
    fn ignores_type_decls() {
        // `type Frame = Foo` should be invisible to this rule.
        assert!(run("type Frame = Foo\n").is_empty());
    }

    #[test]
    fn reports_correct_line_numbers() {
        let src = "// header\nmodule Foo\n\ndef BadName() -> i32 = 1\n";
        let v = run(src);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].line, Some(4));
    }

    #[test]
    fn flags_each_violation() {
        let src = "def ok_one(x: i32) -> i32 = x\ndef BadOne(x: i32) -> i32 = x\ndef bad_two(x: i32) -> i32 = x\ndef Worse_Three(x: i32) -> i32 = x\n";
        let v = run(src);
        // ok_one fine. BadOne flagged. bad_two fine. Worse_Three flagged.
        assert_eq!(v.len(), 2);
        assert!(v.iter().any(|x| x.message.contains("BadOne")));
        assert!(v.iter().any(|x| x.message.contains("Worse_Three")));
    }

    #[test]
    fn snake_case_predicate_examples() {
        assert!(is_snake_case("predict"));
        assert!(is_snake_case("add_one"));
        assert!(is_snake_case("f"));
        assert!(is_snake_case("s0"));
        assert!(is_snake_case("bs_call_scalar"));
        assert!(!is_snake_case(""));
        assert!(!is_snake_case("AddOne"));
        assert!(!is_snake_case("addOne"));
        assert!(!is_snake_case("_internal"));
        assert!(!is_snake_case("MAX_DIM"));
        assert!(!is_snake_case("0bad"));
    }
}
