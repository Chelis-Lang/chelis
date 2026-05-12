//! RT-3a adversarial coverage for WS-A5 precision polymorphism.
//!
//! This file pins observed behavioral gaps in the WS-A5 type-system
//! refactor, exercising the contextual desugar rule
//! (spec/02-surf-syntax.md P4b, spec/04-type-system.md 5.8.1) and
//! the surrounding error-handling surfaces. Tests here describe the
//! ACTUAL behavior at commit 4c5e21d so future fixes flip them.
//!
//! Findings legend (see rt3a-redteam-findings final report):
//!   * Test names tagged "expected_to_fail_*" pin a current bug; once
//!     the underlying issue is fixed, the test should be inverted.
//!   * Test names tagged "baseline_*" pin behavior that is correct
//!     today and should stay correct.
//!
//! All literal strings here are ASCII; em-dash (U+2014) is forbidden
//! by chelis lint 8.6 and is therefore avoided.
//!
//! See spec/04-type-system.md 5.8 and 5.8.1 for the contract under test.
//!
//! Convention: each test creates an isolated tempdir for its source
//! file so concurrent test execution does not collide.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

fn run_json_check(path: &Path) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    serde_json::from_slice(&output).expect("check output should be json")
}

fn run_build(path: &Path, target: &str, out_dir: &Path) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            target,
            "-o",
            out_dir.to_str().unwrap(),
        ])
        .unwrap()
}

// ---------------------------------------------------------------------------
// A. WS-C blocker reproducer (positive control). This is the headline
// reason WS-A5 exists; if this regresses, A blocks WS-C dispatch.
// ---------------------------------------------------------------------------

#[test]
fn baseline_wsc_blocker_reproducer_errors_without_unbound_wrapping() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("wsc_blocker.ch");
    write_file(
        &path,
        "sig poly_id: tensor[d, p] -> tensor[d, p]\n\
         def poly_id(x) = x\n\
         def use_mismatch(x: tensor[3, int32]) -> tensor[3, f32] = poly_id(x)\n",
    );

    let json = run_json_check(&path);
    let errors = json["errors"]
        .as_array()
        .expect("errors should be a json array");
    assert!(
        !errors.is_empty(),
        "WS-A5 headline contract: poly_id with mismatched precision must \
         produce a check error, got clean output {json}"
    );
    let any_mentions_both_precisions = errors.iter().any(|e| {
        let msg = e.get("message").and_then(|m| m.as_str()).unwrap_or("");
        msg.contains("f32") && msg.contains("int32")
    });
    assert!(
        any_mentions_both_precisions,
        "at least one error should mention both f32 and int32 so the user \
         sees the mismatched precisions, got {errors:?}"
    );
}

// ---------------------------------------------------------------------------
// B. Type::Error masks downstream PrecisionMismatch.
//
// `crates/chelis-types/src/unify.rs:370` keeps the permissive
// `(Type::Error, _) | (_, Type::Error) => Ok(())` rule. The agent
// flagged this as deferred. Below we PIN that an unbound function
// inside a polymorphic-precision sig swallows the otherwise-detectable
// precision mismatch on the declared return type.
// ---------------------------------------------------------------------------

#[test]
fn expected_to_fail_unbound_function_masks_downstream_precision_mismatch() {
    // BASELINE (no unbound wrapper): precision mismatch IS reported.
    let dir = tempdir().expect("tempdir");
    let baseline_path = dir.path().join("baseline.ch");
    write_file(
        &baseline_path,
        "sig poly_id: tensor[d, p] -> tensor[d, p]\n\
         def poly_id(x) = x\n\
         def use_mix(x: tensor[3, int32]) -> tensor[3, f32] = poly_id(x)\n",
    );
    let baseline_errors = run_json_check(&baseline_path)["errors"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let baseline_has_precision_error = baseline_errors.iter().any(|e| {
        let msg = e.get("message").and_then(|m| m.as_str()).unwrap_or("");
        msg.contains("int32") && msg.contains("f32")
    });
    assert!(
        baseline_has_precision_error,
        "baseline (no unbound wrapper) must report precision mismatch; \
         got {baseline_errors:?}"
    );

    // ATTACK: wrap the arg in an unbound function call. The precision
    // mismatch on the declared return type should still surface, but
    // currently does not because Type::Error swallows it.
    let attack_path = dir.path().join("attack.ch");
    write_file(
        &attack_path,
        "sig poly_id: tensor[d, p] -> tensor[d, p]\n\
         def poly_id(x) = x\n\
         def use_mix(x: tensor[3, int32]) -> tensor[3, f32] = \
         poly_id(nonexistent_function(x))\n",
    );
    let attack_errors = run_json_check(&attack_path)["errors"]
        .as_array()
        .cloned()
        .unwrap_or_default();

    let has_unbound = attack_errors.iter().any(|e| {
        let kind = e.get("kind").and_then(|k| k.as_str()).unwrap_or("");
        kind == "UnboundVariable"
    });
    assert!(
        has_unbound,
        "attack: unbound variable must still be reported; got {attack_errors:?}"
    );

    // The bug: the precision mismatch on the declared return type is
    // SILENTLY ABSORBED by Type::Error. Pin the silent absorption.
    let attack_has_precision_error = attack_errors.iter().any(|e| {
        let msg = e.get("message").and_then(|m| m.as_str()).unwrap_or("");
        let kind = e.get("kind").and_then(|k| k.as_str()).unwrap_or("");
        kind != "UnboundVariable" && msg.contains("int32") && msg.contains("f32")
    });
    // Currently fails (the bug is present). When the deferred Type::Error
    // narrowing is implemented, this assertion flips.
    assert!(
        !attack_has_precision_error,
        "RT-3a pin: today, Type::Error masks the downstream precision \
         mismatch. If this assertion newly fails, that means the \
         deferred Type::Error narrowing landed; flip the assertion. \
         Got {attack_errors:?}"
    );
}

#[test]
fn expected_to_fail_unbound_function_in_polymorphic_body_masks_return_mismatch() {
    // Even simpler: a polymorphic-precision sig whose body is just a
    // call to an unbound function. The declared sig says return is
    // tensor[d, f32]; nothing in the body could possibly produce that
    // because `nonexistent_function` does not exist. Yet only the
    // unbound-variable error is reported; the type mismatch on the
    // returned-vs-declared shape is not flagged at all.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("body_error.ch");
    write_file(
        &path,
        "sig f: tensor[d, p] -> tensor[d, f32]\n\
         def f(x) = nonexistent_function(x)\n",
    );
    let errors = run_json_check(&path)["errors"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let has_unbound = errors
        .iter()
        .any(|e| e.get("kind").and_then(|k| k.as_str()) == Some("UnboundVariable"));
    assert!(has_unbound, "must report unbound function, got {errors:?}");
    // Pin: only one error today. With Type::Error narrowing, additional
    // shape/precision errors should fire; flip when the fix lands.
    assert_eq!(
        errors.len(),
        1,
        "RT-3a pin: today only the unbound error fires; Type::Error \
         masks any downstream complaint. Got {errors:?}"
    );
}

// ---------------------------------------------------------------------------
// C. Unbound type-variable in a value-position annotation is silently
// accepted. Spec P4b says outside a sig the existing rule applies;
// the existing rule should reject `(t-prim {} weirdname)` (cf. the
// f8e4m3 rejection contract in spec 1.1.1). Today it does not, because
// `Prim::parse_name` returns None and the tensor type collapses to
// Type::Error which then unifies with anything.
// ---------------------------------------------------------------------------

#[test]
fn expected_to_fail_unbound_precision_name_in_let_silently_accepted() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("unbound_p_in_let.ch");
    write_file(
        &path,
        "def main() -> tensor[3, f32] = {\n  \
           xs: tensor[3, p] = [1.0, 2.0, 3.0]\n  \
           xs\n\
         }\n",
    );
    let json = run_json_check(&path);
    let errors = json["errors"].as_array().cloned().unwrap_or_default();
    // RT-3a pin: ZERO errors today. Spec implies an unbound precision
    // name in a value-position annotation should be rejected (it is not
    // a quantifier outside a sig, and it is not a known primitive).
    assert!(
        errors.is_empty(),
        "RT-3a pin: today an unbound precision name in a let-binding \
         passes silently with score 1.0. If this newly fails, the bug \
         is fixed; flip the assertion. Got {errors:?}"
    );
    let score = json["score"].as_f64().unwrap_or(-1.0);
    assert!(
        (score - 1.0).abs() < f64::EPSILON,
        "RT-3a pin: score is 1.0 today for a clearly-bogus precision \
         name in a let-binding. Got {score}"
    );
}

#[test]
fn expected_to_fail_unbound_precision_in_def_param_silently_accepted() {
    // `def f(x: tensor[3, weirdname]) -> tensor[3, weirdname] = x`
    // The desugarer treats this as an implicit sig, so `weirdname`
    // gets collected as an implicit quantifier and the SIG ends up
    // polymorphic. The DEF param annotation, however, is desugared
    // with an empty quantifier scope, so it becomes
    // `(t-prim {} weirdname)`. The infer path collapses that to
    // Type::Error and accepts anything. Net effect: a totally bogus
    // precision name passes clean.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("unbound_p_in_def.ch");
    write_file(
        &path,
        "def f(x: tensor[3, weirdname]) -> tensor[3, weirdname] = x\n",
    );
    let json = run_json_check(&path);
    let errors = json["errors"].as_array().cloned().unwrap_or_default();
    assert!(
        errors.is_empty(),
        "RT-3a pin: today an unknown precision name in a def-param \
         annotation passes silently. Got {errors:?}"
    );
}

#[test]
fn baseline_two_independent_precision_tvars_accepted() {
    // Two distinct precision variables in a sig: this is well-formed
    // per spec (each is implicitly quantified independently).
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("two_tvars.ch");
    write_file(
        &path,
        "sig f: tensor[d, p] -> tensor[d, q] -> tensor[d, p]\n\
         def f(a, b) = a\n\
         def use_it(x: tensor[3, f32], y: tensor[3, int32]) \
         -> tensor[3, f32] = f(x, y)\n",
    );
    let errors = run_json_check(&path)["errors"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(
        errors.is_empty(),
        "two independent precision tvars should accept distinct \
         instantiations, got {errors:?}"
    );
}

#[test]
fn baseline_same_precision_tvar_reused_rejects_mismatch() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("same_tvar.ch");
    write_file(
        &path,
        "sig f: tensor[d, p] -> tensor[d, p] -> tensor[d, p]\n\
         def f(a, b) = a\n\
         def break_it(x: tensor[3, f32], y: tensor[3, int32]) \
         -> tensor[3, f32] = f(x, y)\n",
    );
    let errors = run_json_check(&path)["errors"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(
        errors
            .iter()
            .any(|e| { e.get("kind").and_then(|k| k.as_str()) == Some("PrecisionMismatch") }),
        "sig with same `p` reused must reject inconsistent call, \
         got {errors:?}"
    );
}

#[test]
fn baseline_concrete_primitive_in_sig_stays_concrete() {
    // A primitive name in a sig precision slot must NEVER be promoted
    // to a type variable, even if it could be parsed as a lowercase
    // identifier.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("concrete_in_sig.ch");
    write_file(
        &path,
        "sig f: tensor[d, f32] -> tensor[d, f32]\n\
         def f(x) = x\n\
         def break_it(x: tensor[3, int32]) -> tensor[3, int32] = f(x)\n",
    );
    let errors = run_json_check(&path)["errors"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(
        errors.iter().any(|e| {
            let msg = e.get("message").and_then(|m| m.as_str()).unwrap_or("");
            msg.contains("f32") && msg.contains("int32")
        }),
        "concrete `f32` in sig must NOT be treated as a tvar; mismatched \
         call must error. Got {errors:?}"
    );
}

#[test]
fn baseline_unsigned_alias_in_precision_rejected_with_diagnostic() {
    // Spec 1.1.2 unsigned aliases are explicitly excluded from the
    // implicit quantifier collection so they reach the type-checker
    // 1.1.2 rejection path, not get silently absorbed as a quantifier.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("unsigned_alias.ch");
    write_file(
        &path,
        "sig f: tensor[d, u8] -> tensor[d, u8]\n\
         def f(x) = x\n",
    );
    let errors = run_json_check(&path)["errors"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(
        !errors.is_empty(),
        "unsigned alias `u8` must be rejected with a precise diagnostic, \
         got clean output"
    );
}

// ---------------------------------------------------------------------------
// D. Backend reachability of `TensorPrec::Var`. Spec 5.8.1 says
// "After monomorphization, every reachable tensor type at lowering
// time must carry `TensorPrec::Concrete(_)`. Backends assert this
// invariant at the lowering match arm; a `TensorPrec::Var(_)`
// reaching a backend is a monomorphization bug, not user error."
//
// The agent claimed backends would naturally fail at the conversion
// boundary. They do not: each backend silently emits code as if the
// polymorphic tensor were rank-0 / some-default-precision.
// ---------------------------------------------------------------------------

#[test]
fn expected_to_fail_polymorphic_sig_no_call_emits_c_silently() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("poly_no_call.ch");
    let out = dir.path().join("c_out");
    fs::create_dir_all(&out).unwrap();
    write_file(
        &src,
        "sig poly_id: tensor[d, p] -> tensor[d, p]\n\
         def poly_id(x) = x\n",
    );
    let result = run_build(&src, "c", &out);
    let stderr = String::from_utf8_lossy(&result.stderr);
    let stdout = String::from_utf8_lossy(&result.stdout);
    // Today: build succeeds and emits C that pretends the input is rank-0.
    // Spec 5.8.1 says reaching a backend with `TensorPrec::Var(_)`
    // is a monomorphization bug; the build should refuse, not silently
    // emit code. Pin the silent acceptance.
    assert!(
        result.status.success(),
        "RT-3a pin: today the C backend silently builds polymorphic sig \
         with no instantiation. Once the spec invariant is enforced, the \
         build should fail. stderr={stderr} stdout={stdout}"
    );
    let c_path = out.join("poly_no_call.c");
    let c_text = fs::read_to_string(&c_path).expect("C output");
    assert!(
        c_text.contains("expected rank 0"),
        "RT-3a pin: today the polymorphic input is degraded to rank 0 \
         in the emitted C. C output: {c_text}"
    );
}

#[test]
fn expected_to_fail_polymorphic_sig_no_call_emits_hip_silently() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("poly_no_call.ch");
    let out = dir.path().join("hip_out");
    fs::create_dir_all(&out).unwrap();
    write_file(
        &src,
        "sig poly_id: tensor[d, p] -> tensor[d, p]\n\
         def poly_id(x) = x\n",
    );
    let result = run_build(&src, "hip", &out);
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        result.status.success(),
        "RT-3a pin: today HIP backend silently builds polymorphic sig. \
         stderr={stderr}"
    );
}

#[test]
fn expected_to_fail_polymorphic_sig_no_call_emits_metal_silently() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("poly_no_call.ch");
    let out = dir.path().join("metal_out");
    fs::create_dir_all(&out).unwrap();
    write_file(
        &src,
        "sig poly_id: tensor[d, p] -> tensor[d, p]\n\
         def poly_id(x) = x\n",
    );
    let result = run_build(&src, "metal", &out);
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        result.status.success(),
        "RT-3a pin: today Metal backend silently builds polymorphic sig. \
         stderr={stderr}"
    );
}

// ---------------------------------------------------------------------------
// E. Cross-cutting: WS-A5 + WS-B2 (contextual tensor literal inference).
// `let xs: tensor[3, p] = [1.0, 2.0, 3.0]` outside a sig. The `p`
// is unbound but accepted; the literals get tagged with `(t-prim {} p)`
// which collapses to Type::Error. See test C above for the wider
// silent-acceptance story.
// ---------------------------------------------------------------------------

#[test]
fn expected_to_fail_unbound_p_in_let_with_tensor_literal_silently_accepted() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("let_lit_p.ch");
    write_file(
        &path,
        "def main() -> tensor[3, f32] = {\n  \
           xs: tensor[3, p] = [1.0, 2.0, 3.0]\n  \
           xs\n\
         }\n",
    );
    let json = run_json_check(&path);
    let errors = json["errors"].as_array().cloned().unwrap_or_default();
    assert!(
        errors.is_empty(),
        "RT-3a pin: today an unbound `p` in a let-binding precision slot \
         is silently accepted by the WS-B2 contextual literal inference \
         path. Got {errors:?}"
    );
}

// ---------------------------------------------------------------------------
// G. Whole-tensor type variable builtins still enforce precision
// consistency post-WS-A5.
// ---------------------------------------------------------------------------

#[test]
fn baseline_builtin_add_rejects_mixed_precision_args() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("builtin_add_mix.ch");
    write_file(
        &path,
        "def use_add(x: tensor[3, f32], y: tensor[3, int32]) \
         -> tensor[3, f32] = add(x, y)\n",
    );
    let errors = run_json_check(&path)["errors"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(
        errors
            .iter()
            .any(|e| e.get("kind").and_then(|k| k.as_str()) == Some("PrecisionMismatch")),
        "builtin add must reject mixed precisions, got {errors:?}"
    );
}

#[test]
fn baseline_builtin_add_accepts_matching_precision_args() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("builtin_add_match.ch");
    write_file(
        &path,
        "def use_add(x: tensor[3, f32], y: tensor[3, f32]) \
         -> tensor[3, f32] = add(x, y)\n",
    );
    let errors = run_json_check(&path)["errors"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(
        errors.is_empty(),
        "builtin add with matching precisions must accept, got {errors:?}"
    );
}

#[test]
fn baseline_builtin_mul_rejects_mixed_precision_args() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("builtin_mul_mix.ch");
    write_file(
        &path,
        "def use_mul(x: tensor[3, f32], y: tensor[3, bf16]) \
         -> tensor[3, f32] = mul(x, y)\n",
    );
    let errors = run_json_check(&path)["errors"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(
        errors
            .iter()
            .any(|e| e.get("kind").and_then(|k| k.as_str()) == Some("PrecisionMismatch")),
        "builtin mul must reject mixed precisions, got {errors:?}"
    );
}

#[test]
fn baseline_builtin_sub_rejects_mixed_precision_args() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("builtin_sub_mix.ch");
    write_file(
        &path,
        "def use_sub(x: tensor[3, int32], y: tensor[3, int64]) \
         -> tensor[3, int32] = sub(x, y)\n",
    );
    let errors = run_json_check(&path)["errors"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(
        errors
            .iter()
            .any(|e| e.get("kind").and_then(|k| k.as_str()) == Some("PrecisionMismatch")),
        "builtin sub must reject mixed precisions, got {errors:?}"
    );
}
