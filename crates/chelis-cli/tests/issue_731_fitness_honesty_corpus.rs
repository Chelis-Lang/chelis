//! chelis#731 §C4.4 — the fitness-honesty corpus.
//!
//! A continuous, corpus-level enforcement of [04-TOT-2]
//! (spec/04-type-system.md §10 / `spec/design/checker_totality.md`): a suite of
//! KNOWN-ill-typed / malformed programs, each asserted to score strictly below
//! 1.0 through `chelis check`. Any member that ever scores 1.0 fails the build.
//! This stands even after Phase 2's `ErrorWitness` migration makes a silent
//! `Type::Error` unconstructible: it is the behavioral proof that the checker's
//! score cannot read "perfect" on a program that does not type-check.
//!
//! Per Jeff's review (chelis#731) the invariant that matters is score < 1.0 for
//! every known-bad program; the exact margin (e.g. < 0.9) is calibration tied to
//! open question 4's severity weights and is deliberately NOT hard-coded here.
//!
//! Membership (Phase 1): the wrapper battery's ill-typed variants, the four
//! Phase 0 holes (chelis#709 `with seed`/`with device` bodies, chelis#710
//! `(def)`/`(cast)` malformed forms), the seed-suffix and unknown-effect-kind
//! diagnostics, and the chelis#710 census-extension malformed `.dp` family. The
//! Surf-reachable chelis#755 (field access) and chelis#756 (deep-type
//! conversion) silent holes are NOT yet members: they are unfixed at Phase 1
//! (global restoration over every census site is Phase 2's witness migration),
//! so adding them here would assert a property the tree does not yet have. They
//! join when their fix lands.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

const MASKED_ERROR: &str = "add(cast(1.0, f32), cast(2, int64))";

/// `chelis check` score for a program written with the given extension.
fn check_score(program: &str, ext: &str) -> f64 {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("p{ext}"));
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("chelis check should run");
    let parsed: serde_json::Value =
        serde_json::from_slice(&out.stdout).unwrap_or_else(|e| panic!("check must emit JSON: {e}"));
    parsed["score"].as_f64().expect("numeric score")
}

/// Every member of the corpus must score strictly below 1.0. A member that
/// scores 1.0 is a fitness-honesty violation (the score lies about a program
/// that does not type-check) and fails the build.
fn assert_below_one(members: &[(&str, String, &str)]) {
    let mut offenders = Vec::new();
    for (name, program, ext) in members {
        let score = check_score(program, ext);
        if score >= 1.0 {
            offenders.push(format!("{name}: score {score} (must be < 1.0)"));
        }
    }
    assert!(
        offenders.is_empty(),
        "[04-TOT-2] fitness-honesty violation: known-bad programs scored 1.0:\n  {}",
        offenders.join("\n  ")
    );
}

/// The wrapper battery (chelis#709 canary) with an ill-typed body, plus the two
/// effect-handler holes and the seed-suffix / return-type diagnostics. Every
/// one is Surf-reachable and must score below 1.0.
#[test]
fn surf_known_bad_programs_score_below_one() {
    let cases: Vec<(&str, String, &str)> = vec![
        ("bare", format!("def f() -> f32 = {MASKED_ERROR}\n"), ".ch"),
        (
            "let_body",
            format!("def f() -> f32 = let x = {MASKED_ERROR} in x\n"),
            ".ch",
        ),
        (
            "if_then",
            format!("def f(c: bool) -> f32 = if c then {MASKED_ERROR} else 1.0\n"),
            ".ch",
        ),
        (
            "lambda_body",
            format!("def f() -> f32 = (fn (v: f32) -> f32 = {MASKED_ERROR})(1.0)\n"),
            ".ch",
        ),
        (
            "pipe_stage",
            format!("def f() -> f32 = 1.0 |> fn (v: f32) -> f32 = {MASKED_ERROR}\n"),
            ".ch",
        ),
        (
            "tuple_elem",
            format!("def f() -> (f32, f32) = ({MASKED_ERROR}, 1.0)\n"),
            ".ch",
        ),
        (
            "list_elem",
            format!("def f() -> [f32] = [{MASKED_ERROR}, 1.0]\n"),
            ".ch",
        ),
        (
            "match_arm",
            format!("def f(c: bool) -> f32 = match c {{ true => {MASKED_ERROR}, false => 1.0 }}\n"),
            ".ch",
        ),
        (
            "grad_callee",
            format!("def g(x: f32) -> f32 = {MASKED_ERROR}\ndef f(x: f32) -> f32 = grad(g)(x)\n"),
            ".ch",
        ),
        (
            "jit_callee",
            format!("def g(x: f32) -> f32 = {MASKED_ERROR}\ndef f(x: f32) -> f32 = jit(g)(x)\n"),
            ".ch",
        ),
        // chelis#709: the two effect-handler bodies (now checked).
        (
            "with_seed_body",
            format!("def f() -> f32 = with seed(42i64) {{ {MASKED_ERROR} }}\n"),
            ".ch",
        ),
        (
            "with_device_body",
            format!("def f() -> f32 = with device(\"gpu:0\") {{ {MASKED_ERROR} }}\n"),
            ".ch",
        ),
        // chelis#709 escalation: an int64 body from an `-> f32` fn.
        (
            "masked_return_type",
            "def f() -> f32 = with seed(42i64) { cast(5, int64) }\n".to_string(),
            ".ch",
        ),
        // chelis#731 §C1.5 / chelis#771: an unsuffixed seed literal.
        (
            "unsuffixed_seed",
            "def f() -> f32 = with seed(42) { add(cast(1.0, f32), cast(2.0, f32)) }\n".to_string(),
            ".ch",
        ),
    ];
    assert_below_one(&cases);
}

/// The chelis#710 malformed `.dp` family (the two Phase 0 holes plus the
/// census-extension forms), each module-wrapped, plus the bogus-effect-kind
/// handle-effect. Every one reaches the checker malformed and must score below
/// 1.0 via a pushed `MalformedForm`.
#[test]
fn malformed_dp_forms_score_below_one() {
    let wrap = |form: &str| format!("(module {{}} m.main (def {{}} out {form}))\n");
    let cases: Vec<(&str, String, &str)> = vec![
        (
            "dp_def_no_body",
            "(module {} m.main (def {} orphan))\n".to_string(),
            ".dp",
        ),
        (
            "dp_cast_no_target",
            wrap("(cast {} (lit {type: (t-prim {} f32)} 42.0))"),
            ".dp",
        ),
        ("dp_jit_no_child", wrap("(jit {})"), ".dp"),
        ("dp_realize_no_child", wrap("(realize {})"), ".dp"),
        ("dp_copy_no_child", wrap("(copy {})"), ".dp"),
        ("dp_borrow_no_child", wrap("(borrow {})"), ".dp"),
        ("dp_var_no_name", wrap("(var {})"), ".dp"),
        ("dp_lit_no_value", wrap("(lit {})"), ".dp"),
        ("dp_match_no_kids", wrap("(match {})"), ".dp"),
        ("dp_pipe_no_kids", wrap("(pipe {})"), ".dp"),
        (
            "dp_tuple_get_one_kid",
            wrap("(tuple-get {} (lit {type: (t-prim {} int32)} 0))"),
            ".dp",
        ),
        (
            "dp_record_non_symbol_head",
            wrap("(record {} (lit {type: (t-prim {} int32)} 1))"),
            ".dp",
        ),
        (
            "dp_access_one_kid",
            wrap("(access {} (lit {type: (t-prim {} int32)} 1))"),
            ".dp",
        ),
        (
            "dp_record_update_no_kids",
            wrap("(record-update {})"),
            ".dp",
        ),
        ("dp_grad_no_kids", wrap("(grad {})"), ".dp"),
        ("dp_vmap_no_kids", wrap("(vmap {})"), ".dp"),
        // chelis#709 / §I1: an unknown effect kind in a handle-effect.
        (
            "dp_unknown_effect_kind",
            wrap(
                "(handle-effect {effect: teleport} (lit {type: (t-prim {} int64)} 42) \
                 (lit {type: (t-prim {} f32)} 2.5))",
            ),
            ".dp",
        ),
    ];
    assert_below_one(&cases);
}
