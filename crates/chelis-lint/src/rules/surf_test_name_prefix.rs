//! Rule `surf-test-name-prefix` — Surf test functions are named `test_*`
//! or `example_*`.
//!
//! Spec authority: `spec/01-nomenclature.md` §10.1 (Surf test functions).
//!
//! A "test function" here matches the exact discovery contract used by
//! `chelis test` (see `enumerate_test_fns` in
//! `crates/chelis-cli/src/main.rs`):
//!
//! 1. The function has `! { Test }` (alone or in a comma-separated
//!    effect list) in its signature, and
//! 2. The function is **nullary** — its parameter list is empty.
//!
//! Helper assertion wrappers that take parameters (e.g., `def
//! check_field(rows: ..., ...) -> unit ! { Test } = ...`) are not
//! candidates for `chelis test` discovery — they're internal
//! domain-specific assertion combinators. §10.1 governs the
//! discovery-eligible nullary form; helpers are out of scope.
//!
//! Carve-out: functions declared inside the canonical test-assertion
//! library module `Std.Test` are exempt regardless of arity. That
//! module *introduces* the `Test` effect — its `assert_true`,
//! `assert_eq`, `fail`, etc. carry the effect specifically so the
//! type system can prove only test code calls them. Renaming them to
//! `test_assert_true`, `test_assert_eq`, … would defeat the
//! "primitive vs. user-test" distinction §10.1 is trying to draw. The
//! carve-out is keyed on the `module Std.Test` declaration; user code
//! that re-uses the module name to publish its own assertion helpers
//! will hit the same carve-out, which is the right behavior.

use crate::{Context, Rule, Surface, Violation};
use regex::Regex;
use std::sync::OnceLock;

static TEST_DEF_RE: OnceLock<Regex> = OnceLock::new();

fn test_def_re() -> &'static Regex {
    // `def NAME(PARAMS) -> ...  ! { ... Test ... }` up to `=`. `(?m)`
    // anchors `^` per line so each `def` start re-matches; `(?s)` lets
    // `[^=]*?` span newlines for parameter lists or return types that
    // wrap across lines.
    //
    // Captures:
    //   1: the function name
    //   2: the parameter list contents (everything between `(` and `)`)
    //   3: the effect row contents (everything between `{` and `}`)
    //
    // Capture 2 is examined for emptiness to decide whether the def is
    // nullary (the only shape `chelis test` discovers); capture 3 is
    // scanned for the `Test` token by `effect_row_has_test`.
    TEST_DEF_RE.get_or_init(|| {
        Regex::new(
            r"(?ms)^[ \t]*def[ \t]+([A-Za-z_][A-Za-z0-9_]*)(?:\[[^\]]*\])?[ \t]*\(([^)]*)\)[^=]*?![ \t]*\{([^}]*)\}",
        )
        .unwrap()
    })
}

pub struct SurfTestNamePrefix;

impl Rule for SurfTestNamePrefix {
    fn id(&self) -> &'static str {
        "surf-test-name-prefix"
    }

    fn spec_ref(&self) -> &'static str {
        "§10.1"
    }

    fn applies_to(&self) -> &[Surface] {
        &[Surface::SurfSource]
    }

    fn summary(&self) -> &'static str {
        "Surf functions carrying the `Test` effect are named `test_*` for unit tests or `example_*` for illustrative examples"
    }

    fn check(&self, ctx: &Context<'_>) -> Vec<Violation> {
        let Some(source) = ctx.source else {
            return Vec::new();
        };
        if is_std_test_module(source) {
            return Vec::new();
        }
        // The (?m)^ in the regex anchors per-line; we re-anchor each
        // capture using its byte offset to compute a 1-indexed line.
        let mut out = Vec::new();
        for caps in test_def_re().captures_iter(source) {
            let name = caps.get(1).unwrap().as_str();
            let params = caps.get(2).unwrap().as_str();
            let effect_row = caps.get(3).unwrap().as_str();
            if !effect_row_has_test(effect_row) {
                continue;
            }
            // Helper functions that take parameters are not discoverable by
            // `chelis test` (it scans for nullary `test_*` defs only) and
            // therefore aren't user-test functions §10.1 governs. Skip.
            if !params.trim().is_empty() {
                continue;
            }
            if name.starts_with("test_") || name.starts_with("example_") {
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
                    "function `{name}` carries the `Test` effect but is not named `test_*` or `example_*` (§10.1)"
                ),
            });
        }
        out
    }
}

fn effect_row_has_test(row: &str) -> bool {
    // Effect rows are comma-separated capability tokens.
    row.split(',').any(|t| t.trim() == "Test")
}

/// True when the source declares `module Std.Test`. The first such
/// declaration on its own line is sufficient — Surf only allows one
/// `module` decl per file in practice, and we don't want to scan the
/// rest of the file.
fn is_std_test_module(source: &str) -> bool {
    source.lines().map(str::trim_start).any(|line| {
        line.starts_with("module Std.Test") && {
            // Guard against `module Std.TestRig`, `module Std.TestKit`, etc.
            let rest = &line["module Std.Test".len()..];
            rest.is_empty() || rest.starts_with(char::is_whitespace)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn run(src: &str) -> Vec<Violation> {
        let path = Path::new("tests/foo.ch");
        let ctx = Context {
            root: Path::new("/"),
            path,
            source: Some(src),
            surface: Surface::SurfSource,
        };
        SurfTestNamePrefix.check(&ctx)
    }

    #[test]
    fn accepts_test_prefixed() {
        let src = "def test_add() -> unit ! { Test } = assert_true(true, \"ok\")\n";
        assert!(run(src).is_empty());
    }

    #[test]
    fn accepts_example_prefixed() {
        let src = "def example_demo() -> unit ! { Test } = assert_true(true, \"ok\")\n";
        assert!(run(src).is_empty());
    }

    #[test]
    fn rejects_bare_name_with_test_effect() {
        let src = "def some_thing() -> unit ! { Test } = assert_true(true, \"ok\")\n";
        let v = run(src);
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("some_thing"));
        assert_eq!(v[0].line, Some(1));
    }

    #[test]
    fn rejects_camel_name_with_test_effect() {
        let src = "def someThing() -> unit ! { Test } = assert_true(true, \"ok\")\n";
        let v = run(src);
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("someThing"));
    }

    #[test]
    fn ignores_helpers_without_test_effect() {
        // Test-file helpers are out of scope: no `! { Test }`, no fire.
        let src = "def expected_shape() -> List[i32] = list_lit(1, 2, 3)\n";
        assert!(run(src).is_empty());
    }

    #[test]
    fn ignores_pure_def_without_effect_row() {
        let src = "def add_one(x: i32) -> i32 = x + 1\n";
        assert!(run(src).is_empty());
    }

    #[test]
    fn handles_multi_capability_effect_row() {
        // `! { IO, Test }` still counts as Test-carrying.
        let src = "def some_thing() -> unit ! { IO, Test } = assert_true(true, \"ok\")\n";
        let v = run(src);
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("some_thing"));
    }

    #[test]
    fn ignores_non_test_capability() {
        // `! { IO }` is not a test; rule does not fire even if the name
        // is unconventional.
        let src = "def some_thing() -> unit ! { IO } = assert_true(true, \"ok\")\n";
        assert!(run(src).is_empty());
    }

    #[test]
    fn handles_wrapped_signature() {
        // Multi-line signature where the return-type wraps across
        // lines. `def NAME` itself stays on one line. Nullary so the
        // rule still applies.
        let src = "def some_thing()\n  -> unit ! { Test } = todo()\n";
        let v = run(src);
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("some_thing"));
    }

    #[test]
    fn flags_each_violation() {
        let src = "def test_a() -> unit ! { Test } = todo()\ndef bad_b() -> unit ! { Test } = todo()\ndef example_c() -> unit ! { Test } = todo()\ndef bad_d() -> unit ! { Test } = todo()\n";
        let v = run(src);
        assert_eq!(v.len(), 2);
        assert!(v.iter().any(|x| x.message.contains("bad_b")));
        assert!(v.iter().any(|x| x.message.contains("bad_d")));
    }

    #[test]
    fn std_test_module_is_exempt() {
        // The canonical assertion-library module introduces the `Test`
        // effect; renaming `assert_true` would defeat §10.1's intent.
        let src = "module Std.Test\ndef assert_true(cond: bool) -> unit ! { Test } = todo()\ndef fail(msg: string) -> unit ! { Test } = todo()\n";
        let v = run(src);
        assert!(v.is_empty(), "Std.Test module is exempt; got: {v:?}");
    }

    #[test]
    fn ignores_test_effect_helpers_with_parameters() {
        // Helper combinators that take args are not discoverable by
        // `chelis test`, so §10.1 doesn't apply. Even with the `Test`
        // effect, the rule must NOT fire.
        let src = "def check_field(label: string, expected: i32, actual: i32) -> unit ! { Test } = todo()\n";
        assert!(run(src).is_empty());
    }

    #[test]
    fn rejects_nullary_test_effect_def_with_wrong_name() {
        // Nullary + Test effect + wrong prefix = real violation.
        let src = "def my_check() -> unit ! { Test } = todo()\n";
        let v = run(src);
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("my_check"));
    }

    #[test]
    fn similar_module_names_are_not_exempt() {
        // `Std.TestKit`, `Std.TestRig`, etc. are not the canonical
        // assertion module; the carve-out must not silently extend.
        let src = "module Std.TestKit\ndef helper() -> unit ! { Test } = todo()\n";
        let v = run(src);
        assert_eq!(v.len(), 1, "Std.TestKit must not be exempt");
        assert!(v[0].message.contains("helper"));
    }
}
