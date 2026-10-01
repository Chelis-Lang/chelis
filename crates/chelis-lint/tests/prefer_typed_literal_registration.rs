//! `prefer-typed-literal` ships as a non-blocking warning (§12.6).
//!
//! The rule must run through the lint driver over a tree, and it must stay out
//! of the blocking registry: downstream programs that still spell a literal
//! cast keep compiling under the built-in style gate.

use chelis_lint::{Severity, Violation};
use std::fs;

const RULE: &str = "prefer-typed-literal";

fn violations(rules: &[Box<dyn chelis_lint::Rule>], body: &str) -> Vec<Violation> {
    let tmp = tempfile::tempdir().expect("tempdir");
    fs::write(tmp.path().join("main.ch"), body).expect("write");
    chelis_lint::lint(tmp.path(), rules)
        .expect("lint run")
        .into_iter()
        .filter(|violation| violation.rule_id == RULE)
        .collect()
}

#[test]
fn registered_as_a_non_blocking_warning() {
    let rules = chelis_lint::registry::non_blocking_rules();
    let rule = rules
        .iter()
        .find(|rule| rule.id() == RULE)
        .expect("prefer-typed-literal is a registered non-blocking rule");
    assert_eq!(rule.severity(), Severity::Warning);
    assert_eq!(rule.spec_ref(), "§12.6");
    assert!(
        chelis_lint::registry::selectable_rules()
            .iter()
            .any(|rule| rule.id() == RULE),
        "`chelis lint --rule prefer-typed-literal` must be able to select it"
    );
}

#[test]
fn absent_from_the_blocking_registry() {
    assert!(
        !chelis_lint::registry::all_rules()
            .iter()
            .any(|rule| rule.id() == RULE),
        "a literal cast must not fail the built-in style gate"
    );
    let found = violations(&chelis_lint::registry::all_rules(), "x = cast(1.0, f32)\n");
    assert!(found.is_empty(), "{found:?}");
}

#[test]
fn driver_reports_literal_casts_and_ignores_the_rest() {
    let rules = chelis_lint::registry::non_blocking_rules();
    let found = violations(
        &rules,
        "a = cast(1.0, f32)\nb = cast(-1.0, f32)\nc = cast(a, f64)\nd = 2i64\n",
    );
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(found[0].line, Some(1));
    assert_eq!(found[0].col, Some(5));
}
