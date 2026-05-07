//! Rule `surf-type-pascal-case` — Surf type declarations are PascalCase.
//!
//! Spec authority: `spec/01-nomenclature.md` §3.1 (Types and ADT
//! constructors).
//!
//! Detects `^type Name[...]? = ...` lines where `Name` is not strictly
//! PascalCase. PascalCase here means: ASCII alphanumeric, starting with
//! an uppercase ASCII letter, no underscores. ADT constructor checking
//! is intentionally deferred — `|` introduces both constructors (in a
//! `type` body) and pattern-match alternatives, so distinguishing them
//! requires real parsing rather than line-level regex. This rule scopes
//! to the `type Name` declaration form, which closes §3.1's biggest
//! enforcement gap without false positives on `match` expressions.

use crate::{Context, Rule, Surface, Violation};
use regex::Regex;
use std::sync::OnceLock;

static TYPE_RE: OnceLock<Regex> = OnceLock::new();

fn type_re() -> &'static Regex {
    // `type Foo` or `type Foo[a]` at the start of a line. Captures the
    // type name only (group 1). Permissive on charset so the rule —
    // not the regex — flags violations.
    TYPE_RE.get_or_init(|| Regex::new(r"(?m)^[ \t]*type[ \t]+([A-Za-z_][A-Za-z0-9_]*)").unwrap())
}

pub struct SurfTypePascalCase;

impl Rule for SurfTypePascalCase {
    fn id(&self) -> &str {
        "surf-type-pascal-case"
    }

    fn spec_ref(&self) -> &str {
        "§3.1"
    }

    fn applies_to(&self) -> &[Surface] {
        &[Surface::SurfSource]
    }

    fn summary(&self) -> &str {
        "Surf type declarations are PascalCase: ASCII alphanumeric, leading uppercase, no underscores"
    }

    fn check(&self, ctx: &Context<'_>) -> Vec<Violation> {
        let Some(source) = ctx.source else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for caps in type_re().captures_iter(source) {
            let name = caps.get(1).unwrap().as_str();
            if is_pascal_case(name) {
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
                    "type `{name}` is not PascalCase (§3.1 requires leading uppercase, no underscores, ASCII alphanumeric)"
                ),
            });
        }
        out
    }
}

fn is_pascal_case(s: &str) -> bool {
    let mut chars = s.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_ascii_uppercase() {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric())
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
        SurfTypePascalCase.check(&ctx)
    }

    #[test]
    fn accepts_pascal_case_types() {
        assert!(run("type Frame = Foo\n").is_empty());
        assert!(run("type RoundingMode = Foo\n").is_empty());
        assert!(run("type AdamWConfig = Foo\n").is_empty());
        assert!(run("type Hamt[a] = Foo\n").is_empty());
        assert!(run("type Json = Foo\ntype JsonObject = Foo\n").is_empty());
    }

    #[test]
    fn rejects_lowercase_type() {
        let v = run("type frame = Foo\n");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].rule_id, "surf-type-pascal-case");
        assert_eq!(v[0].line, Some(1));
        assert!(v[0].message.contains("frame"));
    }

    #[test]
    fn rejects_snake_case_type() {
        let v = run("type rounding_mode = Foo\n");
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("rounding_mode"));
    }

    #[test]
    fn rejects_screaming_snake_type() {
        let v = run("type ROUNDING_MODE = Foo\n");
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn rejects_pascal_with_underscore() {
        // PascalCase forbids underscores per §3.1; `Adam_W` is not PascalCase.
        let v = run("type Adam_W = Foo\n");
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("Adam_W"));
    }

    #[test]
    fn reports_line_numbers() {
        let src = "// header\nmodule Foo\n\ntype bad_one = Foo\n";
        let v = run(src);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].line, Some(4));
    }

    #[test]
    fn ignores_def_lines() {
        // `def` is unrelated; rule should not fire.
        assert!(run("def add_one(x: i32) -> i32 = x + 1\n").is_empty());
    }

    #[test]
    fn ignores_keyword_substring() {
        // `type` only matches at line start; inline mentions are skipped.
        assert!(run("// the type annotation is foo\n").is_empty());
    }

    #[test]
    fn flags_multiple_violations() {
        let src = "type good = Foo\ntype Good = Foo\ntype bad_two = Foo\n";
        let v = run(src);
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn pascal_case_predicate_examples() {
        assert!(is_pascal_case("Frame"));
        assert!(is_pascal_case("Json"));
        assert!(is_pascal_case("AdamWConfig"));
        assert!(is_pascal_case("A"));
        assert!(!is_pascal_case(""));
        assert!(!is_pascal_case("frame"));
        assert!(!is_pascal_case("Frame_T"));
        assert!(!is_pascal_case("_Frame"));
        assert!(!is_pascal_case("Frame!"));
    }
}
