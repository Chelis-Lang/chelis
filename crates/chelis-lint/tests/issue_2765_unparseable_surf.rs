//! Regression tests for chelis#2765.
//!
//! Every lint rule that needs an AST takes the same deferral —
//! `let Ok(decls) = parser::parse_str(source) else { return Vec::new() }`.
//! Per rule that is correct: a rule must not invent a diagnosis from a broken
//! parse. In aggregate it was wrong, because silence from every rule is
//! indistinguishable from a clean file, so `chelis lint --check` reported
//! success over a file no rule could read.
//!
//! Measured consequence before the fix: a reef package whose source declares
//! two modules got `reef build` exit 1 and `chelis lint --check .` exit 0.
//! `chelis fmt --check` rejects that file, but it has no directory form, so it
//! could not stand in for the lint over a tree.
//!
//! §12.5 puts the parse verdict on exactly one rule. These tests pin that the
//! rule reports what cannot be parsed, stays silent on what can, and does not
//! start second-guessing files that parse.

use chelis_lint::{Severity, Violation};
use std::fs;
use std::path::Path;

const RULE: &str = "surf-parses";

fn write(path: &Path, body: &str) {
    fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    fs::write(path, body).expect("write");
}

fn violations_for(files: &[(&str, &str)]) -> Vec<Violation> {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();
    for (rel, body) in files {
        write(&root.join(rel), body);
    }
    let rules = chelis_lint::registry::all_rules();
    chelis_lint::lint(root, &rules)
        .expect("lint run")
        .into_iter()
        .filter(|violation| violation.rule_id == RULE)
        .collect()
}

// --- the shapes §12.5 says must be reported ---

#[test]
fn reports_two_module_declarations() {
    // The shape from chelis#2116 that lint could not see.
    let found = violations_for(&[(
        "src/data.ch",
        "module Ub10.Data\nmodule Ub10.Other\ndef value(x: i32) -> i32 = x\n",
    )]);
    assert_eq!(found.len(), 1, "expected one violation, got {found:?}");
    assert_eq!(found[0].spec_ref, "§12.5");
}

#[test]
fn reports_a_stray_declaration_before_the_module() {
    let found = violations_for(&[(
        "src/data.ch",
        "import Foo.Bar\nmodule Ub10.Data\ndef value(x: i32) -> i32 = x\n",
    )]);
    assert_eq!(found.len(), 1, "expected one violation, got {found:?}");
}

#[test]
fn reports_outright_garbage() {
    let found = violations_for(&[("src/data.ch", "@@@ this is not Surf at all (((\n")]);
    assert_eq!(found.len(), 1, "expected one violation, got {found:?}");
}

#[test]
fn reports_a_byte_order_mark() {
    let found = violations_for(&[(
        "src/data.ch",
        "\u{feff}module Ub10.Data\ndef value(x: i32) -> i32 = x\n",
    )]);
    assert_eq!(found.len(), 1, "a BOM is a lex failure, got {found:?}");
}

#[test]
fn carries_the_parser_message_so_the_diagnostic_is_actionable() {
    let found = violations_for(&[(
        "src/data.ch",
        "module Ub10.Data\nmodule Ub10.Other\ndef value(x: i32) -> i32 = x\n",
    )]);
    assert_eq!(found.len(), 1);
    assert!(
        found[0].message.contains("module declaration"),
        "message should carry the parser's reason, got {:?}",
        found[0].message
    );
}

// --- negative parity: the rule must not fire on source that parses ---

#[test]
fn stays_silent_on_source_that_parses() {
    let found = violations_for(&[
        (
            "src/data.ch",
            "module Ub10.Data\ndef value(x: i32) -> i32 = x\n",
        ),
        ("src/other.ch", "def bare(x: i32) -> i32 = x\n"),
        ("src/empty.ch", ""),
    ]);
    assert!(found.is_empty(), "expected no violations, got {found:?}");
}

#[test]
fn does_not_judge_non_surf_files() {
    // A `.dp` file is Deep, not Surf. Deep has the same hole and it is tracked
    // separately (the hull corpus carries 50 deliberate `reject_*.dp`
    // fixtures, so widening this rule there needs its own analysis).
    let found = violations_for(&[("fixtures/x.dp", "(this is not ( valid deep\n")]);
    assert!(found.is_empty(), "expected no violations, got {found:?}");
}

#[test]
fn a_file_that_parses_but_violates_other_rules_is_not_a_parse_failure() {
    // `myValue` parses and violates `surf-value-snake-case` (§3.2). This rule
    // must not become a catch-all for "something is wrong with this file".
    //
    // Finding this case took three attempts, which is itself worth recording:
    // Surf's case split (§1.1) is enforced in the PARSER, so `def Value(...)`
    // and `module data` are parse errors rather than naming violations. A
    // lowercase-leading camelCase name is one of the few spellings that
    // reaches the naming rules at all.
    let found = violations_for(&[(
        "src/data.ch",
        "module Ub10.Data\ndef myValue(x: i32) -> i32 = x\n",
    )]);
    assert!(found.is_empty(), "expected no violations, got {found:?}");
}

// --- the rule blocks, which is the whole point ---

#[test]
fn rule_is_blocking_and_registered() {
    let rule = chelis_lint::registry::all_rules()
        .into_iter()
        .find(|rule| rule.id() == RULE)
        .expect("surf-parses must be in all_rules()");
    assert_eq!(rule.severity(), Severity::Error);
    assert!(rule.severity().blocks_check());
    assert_eq!(rule.spec_ref(), "§12.5");
}
