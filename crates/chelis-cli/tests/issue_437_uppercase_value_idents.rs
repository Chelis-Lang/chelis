//! Issue #437: the front end accepts finance-standard single-letter
//! uppercase VALUE identifiers (`S`, `K`, `T`, `N`, `P`).
//!
//! Before this change the lexer's §1.1 case-split classified a bare
//! uppercase identifier as a `TypeIdent` unconditionally, so a value
//! binding `S = ...` or an uppercase parameter `def f(S, K) = ...`
//! failed at parse with "expected end of declaration expression, found
//! Eq" / "expected identifier, found TypeIdent". The repro in the issue
//! is `S = f(cast(2.0, f32))`.
//!
//! `spec/01-nomenclature.md` §1.1/§3.2/§3.3 now bless single-letter
//! uppercase names in value-binding positions (the value mirror of the
//! `[..]`-clause type-variable override in `spec/02-surf-syntax.md`
//! §P4a). These tests pin the acceptance surface at the CLI: canonically
//! formatted source using `S`/`K`/`T`/`N`/`P` passes `chelis check`
//! cleanly WITHOUT the style-gate bypass, so the formatter and lint also
//! accept the new value names; multi-letter PascalCase value bindings
//! stay rejected.

use assert_cmd::Command;
use serde_json::Value;
use std::io::Write;

/// `chelis check` exit code with a non-empty errors array (#207).
const CHECK_ERRORS_EXIT_CODE: i32 = 2;

fn check_json(src: &str) -> (Option<i32>, Value) {
    let mut tmp = tempfile::Builder::new()
        .prefix("issue437-")
        .suffix(".ch")
        .tempfile()
        .expect("create tempfile");
    tmp.write_all(src.as_bytes()).expect("write tempfile");
    tmp.flush().expect("flush tempfile");

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", tmp.path().to_str().expect("path utf8")])
        .output()
        .expect("run chelis check");
    let stdout = String::from_utf8(output.stdout).expect("utf8");
    let json: Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|err| panic!("chelis check stdout must be valid JSON: {err}\n{stdout}"));
    (output.status.code(), json)
}

fn errors(json: &Value) -> &Vec<Value> {
    json.get("errors")
        .and_then(Value::as_array)
        .expect("errors array present")
}

#[test]
fn uppercase_single_letter_value_binding_checks_clean() {
    // The #437 binding form `S = ...`, canonically formatted so it clears
    // the fmt + lint gates `chelis check` runs before the front end.
    // `T = S` exercises the value-*reference* path (S used as a value).
    let src = "S = cast(2.0, f32)\nT = S\n";
    let (code, json) = check_json(src);
    assert_eq!(code, Some(0), "S = ... must check clean; json={json}");
    assert!(errors(&json).is_empty(), "expected no errors; json={json}");
}

#[test]
fn uppercase_single_letter_params_and_references_check_clean() {
    // Finance notation: spot/strike as uppercase parameters, referenced
    // in the body. The body references resolve via the type checker's
    // environment lookup (constructor-vs-value disambiguation). Canonical
    // form parenthesizes the body, so this also proves fmt accepts it.
    let src = "def payoff(S: f32, K: f32) -> f32 = (S - K)\n";
    let (code, json) = check_json(src);
    assert_eq!(
        code,
        Some(0),
        "uppercase params must check clean; json={json}"
    );
    assert!(errors(&json).is_empty(), "expected no errors; json={json}");
}

#[test]
fn each_finance_letter_binds_and_resolves() {
    // S, K, T, N, P each bind as a value and resolve when referenced
    // downstream — `T = S` uses S as a value, not a constructor.
    let src = "S = cast(2.0, f32)\n\
               K = cast(3.0, f32)\n\
               T = S\n\
               N = K\n\
               P = T\n";
    let (code, json) = check_json(src);
    assert_eq!(
        code,
        Some(0),
        "S/K/T/N/P chain must check clean; json={json}"
    );
    assert!(errors(&json).is_empty(), "expected no errors; json={json}");
}

#[test]
fn multi_letter_uppercase_value_binding_still_rejected() {
    // Negative parity: `Foo = ...` is NOT a value binding. The §1.1
    // case-split is preserved for multi-letter PascalCase, so this stays
    // a parse error (non-zero exit, non-empty errors). If this ever
    // checks clean the override has leaked past single-letter names.
    let src = "Foo = cast(2.0, f32)\n";
    let (code, json) = check_json(src);
    assert_eq!(
        code,
        Some(CHECK_ERRORS_EXIT_CODE),
        "multi-letter uppercase binding must be rejected; json={json}"
    );
    assert!(
        !errors(&json).is_empty(),
        "expected a parse error for `Foo = ...`; json={json}"
    );
}
