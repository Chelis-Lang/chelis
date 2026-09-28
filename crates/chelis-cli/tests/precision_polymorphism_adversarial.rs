//! RT-3a adversarial coverage for WS-A5 precision polymorphism.
//!
//! This file originally pinned observed behavioral gaps in the WS-A5
//! type-system refactor, exercising the contextual desugar rule
//! (spec/02-surf-syntax.md P4b, spec/04-type-system.md 5.8.1) and
//! the surrounding error-handling surfaces. The WS-A5 RT-3a fixups
//! (F1: surgical Type::Error masking detector at the def-body vs
//! declared-sig unify site; F2: backend tripwire at the
//! `Type::Tensor` -> `HostType::Tensor` conversion boundary; F3:
//! validator fall-through for unbound precision names in value
//! position; F4: `f8e4m3` no longer in the surf desugar PRIMITIVES
//! list) closed all four bugs. The previously-named
//! `expected_to_fail_*` tests have been renamed to drop the prefix
//! and inverted to assert the fixed behavior.
//!
//! WS-A8 supersedes the F2 backend-rejection contract. Section D's
//! original tests asserted that backends MUST refuse to emit a
//! polymorphic sig with no call site. After WS-A8 implemented true
//! monomorphization, the contract is: a polymorphic sig with no
//! concrete call site is silently elided from emission (no caller
//! needs the symbol; no concrete type to instantiate). The backend
//! tripwire panic remains as a safety net but is unreachable from
//! properly-typed source. The Section D tests now pin the
//! silent-elision contract and add positive cases that exercise
//! monomorphization with a concrete call site.
//!
//! Findings legend:
//!   * Test names tagged "baseline_*" pin behavior that was correct
//!     before the fixups and must stay correct.
//!   * Other tests assert the post-fix behavior (the bugs are closed).
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

// Issue #207: `chelis check` now exits non-zero when the JSON
// `errors` array is non-empty. This helper is shared across
// adversarial cases that DO expect errors, so it must not assert on
// exit status.
fn run_json_check(path: &Path) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    serde_json::from_slice(&output.stdout).expect("check output should be json")
}

fn run_build(path: &Path, target: &str, out_dir: &Path) -> std::process::Output {
    // Use `output()` (does not panic on non-zero exit) so callers can
    // assert on `result.status.success()` for both success and failure
    // cases. The F2 polymorphic-sig tests deliberately expect failure
    // and need the structured `Output` value, not the assert-on-success
    // shape `assert_cmd::Command::unwrap` provides.
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
        .output()
        .expect("build subprocess")
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
        "sig poly_id[d, p]: tensor[d, p] -> tensor[d, p]\n\
         def poly_id(x) = x\n\
         def use_mismatch(x: tensor[3, i32]) -> tensor[3, f32] = poly_id(x)\n",
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
        msg.contains("f32") && msg.contains("i32")
    });
    assert!(
        any_mentions_both_precisions,
        "at least one error should mention both f32 and i32 so the user \
         sees the mismatched precisions, got {errors:?}"
    );
}

// ---------------------------------------------------------------------------
// B. Type::Error no longer masks downstream declared-sig mismatches.
//
// Pre-F1, `crates/chelis-types/src/unify.rs:370`'s permissive
// `(Type::Error, _) | (_, Type::Error) => Ok(())` rule absorbed the
// silent passthrough at the def-body vs declared-sig unify site. The
// F1 fix kept the permissive unify rule (so a single upstream error
// does not fan out a cascade of secondary diagnostics) and added a
// surgical detector at `infer.rs`'s def-body call site that
// re-surfaces the masked declared-shape constraint when the body
// collapses to `Type::Error` and the body's inference also produced
// an `UnboundVariable` diagnostic.
//
// chelis#773 interaction (measured, intentional). #773 removed
// `infer_app`'s arg-Error short-circuit: an Error-typed ARGUMENT no
// longer collapses the whole call. This sharpens F1's real contract
// ("an unbound function must not mask a REAL downstream precision
// mismatch") into two honest cases, distinguished by whether the
// downstream op's return precision is free or concrete:
//
//   * unbound wrapper feeding a PRECISION-POLYMORPHIC downstream op
//     (`poly_id(nonexistent_function(x))`): the erased arg leaves
//     `poly_id`'s output precision genuinely FREE, so the body
//     resolves to the declared return with no conflict. The old F1
//     re-surface fired only because the arg-Error short-circuit
//     collapsed the body to `Type::Error`; that report was
//     speculative (it assumed the unknown wrapper is
//     precision-preserving). Post-#773 the body no longer collapses,
//     so ONLY the (sound, actionable) unbound-var error is reported.
//   * unbound wrapper feeding a CONCRETE-precision downstream op
//     (`force_f64(nonexistent_function(x))`): the body resolves to a
//     concrete `f64` that genuinely conflicts with the declared
//     `f32`, so the real mismatch surfaces alongside the unbound-var
//     error WITHOUT any Error-collapse hack. This is F1's
//     masking-prevention contract, preserved by honest resolution.
//
// The directly-unbound-CALLEE case (Section B2 below,
// `def f(x) = nonexistent_function(x)`) is unchanged: #773 kept the
// callee-Error early-out, so a body whose callee is unbound still
// collapses to `Type::Error` and F1 still re-surfaces the declared
// return.
// ---------------------------------------------------------------------------

#[test]
fn unbound_wrapper_into_polymorphic_downstream_reports_only_unbound() {
    // BASELINE (no unbound wrapper): precision mismatch IS reported.
    // `poly_id(x)` with `x: tensor[3, i32]` resolves to
    // `tensor[3, i32]`, which conflicts with the declared
    // `tensor[3, f32]` return. This must stay caught.
    let dir = tempdir().expect("tempdir");
    let baseline_path = dir.path().join("baseline.ch");
    write_file(
        &baseline_path,
        "sig poly_id[d, p]: tensor[d, p] -> tensor[d, p]\n\
         def poly_id(x) = x\n\
         def use_mix(x: tensor[3, i32]) -> tensor[3, f32] = poly_id(x)\n",
    );
    let baseline_errors = run_json_check(&baseline_path)["errors"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let baseline_has_precision_error = baseline_errors.iter().any(|e| {
        let msg = e.get("message").and_then(|m| m.as_str()).unwrap_or("");
        msg.contains("i32") && msg.contains("f32")
    });
    assert!(
        baseline_has_precision_error,
        "baseline (no unbound wrapper) must report precision mismatch; \
         got {baseline_errors:?}"
    );

    // ATTACK: wrap the arg in an unbound function call, feeding a
    // PRECISION-POLYMORPHIC downstream op (`poly_id`). chelis#773: the
    // erased arg (`nonexistent_function(x)` is `Type::Error`) leaves
    // `poly_id`'s output precision genuinely FREE, so the body resolves
    // to the declared `tensor[3, f32]` with no conflict. The pre-#773 F1
    // path re-surfaced an i32/f32 "declared-shape" error ONLY because
    // the arg-Error short-circuit collapsed the body to `Type::Error`;
    // that report was speculative (it assumed the unknown wrapper is
    // precision-preserving). Post-#773 the sound result is exactly the
    // unbound-var error — the program still rejects, and no fabricated
    // precision mismatch is emitted. The concrete-downstream companion
    // below pins that a REAL mismatch still surfaces.
    let attack_path = dir.path().join("attack.ch");
    write_file(
        &attack_path,
        "sig poly_id[d, p]: tensor[d, p] -> tensor[d, p]\n\
         def poly_id(x) = x\n\
         def use_mix(x: tensor[3, i32]) -> tensor[3, f32] = \
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

    // chelis#773: no speculative precision-mismatch error. Because the
    // polymorphic body is genuinely free, the ONLY diagnostic is the
    // unbound var — assert exactly that (both the count and that no
    // non-UnboundVariable precision error was fabricated).
    assert_eq!(
        attack_errors.len(),
        1,
        "attack: the polymorphic body resolves freely, so exactly one \
         diagnostic (the unbound var) is sound; got {attack_errors:?}"
    );
    let fabricated_precision_error = attack_errors.iter().any(|e| {
        let kind = e.get("kind").and_then(|k| k.as_str()).unwrap_or("");
        let msg = e.get("message").and_then(|m| m.as_str()).unwrap_or("");
        kind != "UnboundVariable" && msg.contains("f32")
    });
    assert!(
        !fabricated_precision_error,
        "attack: a precision-polymorphic downstream op must NOT fabricate \
         a speculative f32 mismatch off an erased (Error) argument; \
         got {attack_errors:?}"
    );
}

#[test]
fn unbound_wrapper_into_concrete_downstream_still_surfaces_real_mismatch() {
    // F1's masking-prevention contract, PRESERVED by honest resolution
    // (chelis#773). When the unbound wrapper feeds a CONCRETE-precision
    // downstream op, the body resolves to that concrete precision, which
    // genuinely conflicts with the declared return — so the real
    // mismatch surfaces ALONGSIDE the unbound-var error, with no
    // `Type::Error`-collapse hack. `force_f64` returns `tensor[3, f64]`
    // regardless of its (erased) argument; the declared `use_mix` return
    // is `tensor[3, f32]`.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("concrete_attack.ch");
    write_file(
        &path,
        "sig poly_id[d, p]: tensor[d, p] -> tensor[d, p]\n\
         def poly_id(x) = x\n\
         def force_f64(y: tensor[3, f64]) -> tensor[3, f64] = y\n\
         def use_mix(x: tensor[3, i32]) -> tensor[3, f32] = \
         force_f64(nonexistent_function(x))\n",
    );
    let errors = run_json_check(&path)["errors"]
        .as_array()
        .cloned()
        .unwrap_or_default();

    let has_unbound = errors
        .iter()
        .any(|e| e.get("kind").and_then(|k| k.as_str()) == Some("UnboundVariable"));
    assert!(
        has_unbound,
        "concrete attack: unbound variable must still be reported; got {errors:?}"
    );

    // The real f64-vs-f32 mismatch surfaces without any Error collapse:
    // the body honestly resolves to concrete `tensor[3, f64]`.
    let has_real_mismatch = errors.iter().any(|e| {
        let kind = e.get("kind").and_then(|k| k.as_str()).unwrap_or("");
        let msg = e.get("message").and_then(|m| m.as_str()).unwrap_or("");
        kind == "TypeMismatch" && msg.contains("f64") && msg.contains("f32")
    });
    assert!(
        has_real_mismatch,
        "F1 preserved: a concrete-precision downstream op must surface the \
         real f64-vs-f32 mismatch alongside the unbound error (honest \
         resolution, no Type::Error collapse); got {errors:?}"
    );
}

#[test]
fn unbound_function_in_polymorphic_body_no_longer_masks_return_mismatch() {
    // Even simpler: a polymorphic-precision sig whose body is just a
    // call to an unbound function. The declared sig says return is
    // tensor[d, f32]; nothing in the body could possibly produce that
    // because `nonexistent_function` does not exist. Post-F1 both the
    // unbound-variable error AND the def-body-vs-declared-sig
    // mismatch surface.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("body_error.ch");
    write_file(
        &path,
        "sig f[d, p]: tensor[d, p] -> tensor[d, f32]\n\
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
    // F1 fix: the def-body diagnostic also fires, naming the declared
    // return type so the user sees the constraint that was previously
    // masked by the permissive Type::Error unify rule.
    let has_body_mismatch = errors.iter().any(|e| {
        let msg = e.get("message").and_then(|m| m.as_str()).unwrap_or("");
        let kind = e.get("kind").and_then(|k| k.as_str()).unwrap_or("");
        kind == "TypeMismatch" && msg.contains("f32") && msg.contains("declared")
    });
    assert!(
        has_body_mismatch,
        "F1 fix: a TypeMismatch naming the declared `f32` return must \
         fire alongside the unbound-variable error. Got {errors:?}"
    );
}

// ---------------------------------------------------------------------------
// C. Unbound type-variable in a value-position annotation is now rejected.
//
// Spec P4b says outside a sig the existing rule applies, and the
// existing rule must reject `(t-prim {} weirdname)`. The shared type
// resolver owns that closed-primitive rejection as a `TypeMismatch`;
// the legacy tensor-precision validator remains only as a fallback for
// type metadata no semantic resolver reached.
// ---------------------------------------------------------------------------

/// An unbound precision name in a let-binding precision slot must be rejected
/// once by the shared type resolver, and the score must drop below 1.0.
///
/// This fixture also exercises the cross-cutting WS-A5 + WS-B2
/// (contextual tensor literal inference) path: the `[1.0, 2.0, 3.0]`
/// literal in `xs: tensor[3, p] = [1.0, 2.0, 3.0]` outside a sig had
/// its element type driven by the annotation's unbound `p`. The declaration
/// owns one located unknown-primitive witness; the generated literal copies
/// must reuse it rather than emit one diagnostic per element.
#[test]
fn unbound_precision_name_in_let_now_rejected() {
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
    assert_eq!(
        errors.len(),
        1,
        "the declaration must own one diagnostic regardless of contextual literal length: \
         {errors:?}"
    );
    let error = &errors[0];
    assert_eq!(error["kind"], "TypeMismatch", "{errors:?}");
    assert!(
        error["message"]
            .as_str()
            .is_some_and(|message| message.contains("unknown primitive type `p`")),
        "{errors:?}"
    );
    assert_eq!(error["span"]["offset"], 49, "{errors:?}");
    assert_eq!(error["span_id"], "source:49..50", "{errors:?}");
    assert!(
        error["suggestions"].as_array().is_some_and(|suggestions| {
            suggestions.iter().any(|suggestion| {
                suggestion.as_str().is_some_and(|suggestion| {
                    suggestion.contains("nearest active dtype")
                        && suggestion.contains("declare `p`")
                })
            })
        }),
        "{errors:?}"
    );
    let score = json["score"].as_f64().unwrap_or(-1.0);
    assert!(
        score < 1.0,
        "score must drop below 1.0 once the unbound precision name surfaces a real error. \
         Got {score}"
    );
}

#[test]
fn explicit_unusual_precision_in_def_param_is_bound() {
    // Surf P4b applies the complete explicit list to the synthesized
    // signature, including inline parameter types. The parameter and result
    // therefore share one `weirdname` binder. F3 still rejects unbound
    // precision names in value-local annotations.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("implicit_p_in_def.ch");
    write_file(
        &path,
        "def f[weirdname](x: tensor[3, weirdname]) -> tensor[3, weirdname] = x\n",
    );
    let json = run_json_check(&path);
    let errors = json["errors"].as_array().cloned().unwrap_or_default();
    assert!(errors.is_empty(), "explicit binder must check: {json:#?}");
    assert_eq!(json["score"].as_f64(), Some(1.0), "{json:#?}");
}

#[test]
fn baseline_two_independent_precision_tvars_accepted() {
    // Two distinct precision variables in a sig are explicitly quantified
    // independently.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("two_tvars.ch");
    write_file(
        &path,
        "sig f[d, p, q]: tensor[d, p] -> tensor[d, q] -> tensor[d, p]\n\
         def f(a, b) = a\n\
         def use_it(x: tensor[3, f32], y: tensor[3, i32]) \
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
        "sig f[d, p]: tensor[d, p] -> tensor[d, p] -> tensor[d, p]\n\
         def f(a, b) = a\n\
         def break_it(x: tensor[3, f32], y: tensor[3, i32]) \
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
        "sig f[d]: tensor[d, f32] -> tensor[d, f32]\n\
         def f(x) = x\n\
         def break_it(x: tensor[3, i32]) -> tensor[3, i32] = f(x)\n",
    );
    let errors = run_json_check(&path)["errors"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(
        errors.iter().any(|e| {
            let msg = e.get("message").and_then(|m| m.as_str()).unwrap_or("");
            msg.contains("f32") && msg.contains("i32")
        }),
        "concrete `f32` in sig must NOT be treated as a tvar; mismatched \
         call must error. Got {errors:?}"
    );
}

#[test]
fn baseline_unsigned_alias_in_precision_rejected_with_diagnostic() {
    // Spec 1.1.2 unsigned aliases are explicitly excluded from the
    // binder admission so they reach the type-checker 1.1.2 rejection path,
    // not get silently absorbed as a quantifier.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("unsigned_alias.ch");
    write_file(
        &path,
        "sig f[d]: tensor[d, u8] -> tensor[d, u8]\n\
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
// WS-A8 supersedes the WS-A5 RT-3a F2 contract: monomorphization
// itself was implemented (call-site precision substitution into the
// def body, host emission of polymorphic-sig defs is skipped). The
// backend tripwire panic in `try_extract_tensor_type` is retained as
// a safety net, but properly-typed source never reaches it.
//
// Post-WS-A8 contract: a polymorphic sig with no concrete call site
// is silently elided from emission (no caller is asking for it; no
// concrete type to monomorphize against). The build succeeds with no
// emitted symbols for that sig. The F2 backend tripwire only fires
// for genuine internal monomorphization gaps and is unreachable from
// well-typed source.
// ---------------------------------------------------------------------------

fn polymorphic_sig_no_call_now_skips_emit_silently(target: &str, dir_name: &str) {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("poly_no_call.ch");
    let out = dir.path().join(dir_name);
    fs::create_dir_all(&out).unwrap();
    write_file(
        &src,
        "sig poly_id[d, p]: tensor[d, p] -> tensor[d, p]\n\
         def poly_id(x) = x\n",
    );
    let result = run_build(&src, target, &out);
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        result.status.success(),
        "WS-A8: target `{target}` must accept a polymorphic sig with \
         no concrete instantiation by silently eliding the emit per \
         spec/04-type-system.md \u{00a7}5.8.1. stderr={stderr}"
    );
    assert!(
        !stderr.contains("monomorphization"),
        "WS-A8: stderr must not surface the monomorphization tripwire \
         for properly-typed source. stderr={stderr}"
    );
}

#[test]
fn polymorphic_sig_no_call_skips_emit_silently_c_backend() {
    polymorphic_sig_no_call_now_skips_emit_silently("c", "c_out");
}

#[test]
fn polymorphic_sig_no_call_skips_emit_silently_hip_backend() {
    polymorphic_sig_no_call_now_skips_emit_silently("hip", "hip_out");
}

#[test]
fn polymorphic_sig_no_call_skips_emit_silently_metal_backend() {
    polymorphic_sig_no_call_now_skips_emit_silently("metal", "metal_out");
}

// WS-A8 positive cases: a polymorphic sig with a concrete call site
// must build successfully, with the call-site instantiation supplying
// the concrete precision through monomorphization.
fn polymorphic_sig_with_concrete_call_now_builds(target: &str, dir_name: &str) {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("poly_with_call.ch");
    let out = dir.path().join(dir_name);
    fs::create_dir_all(&out).unwrap();
    write_file(
        &src,
        "sig poly_id[n, p]: tensor[n, p] -> tensor[n, p]\n\
         def poly_id(x) = x\n\
         def call(x: tensor[3, f32]) -> tensor[3, f32] = poly_id(x)\n",
    );
    let result = run_build(&src, target, &out);
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        result.status.success(),
        "WS-A8: target `{target}` must build a polymorphic sig + concrete \
         call site successfully. stderr={stderr}"
    );
}

#[test]
fn polymorphic_sig_with_concrete_call_builds_c_backend() {
    polymorphic_sig_with_concrete_call_now_builds("c", "c_out");
}

#[test]
fn polymorphic_sig_with_concrete_call_builds_hip_backend() {
    polymorphic_sig_with_concrete_call_now_builds("hip", "hip_out");
}

#[test]
fn polymorphic_sig_with_concrete_call_builds_metal_backend() {
    polymorphic_sig_with_concrete_call_now_builds("metal", "metal_out");
}

// ---------------------------------------------------------------------------
// E. Cross-cutting: WS-A5 + WS-B2 (contextual tensor literal
// inference). The section-E test that pinned `let xs: tensor[3, p] =
// [1.0, 2.0, 3.0]` outside a sig wrote the identical fixture and a
// strict subset of the assertions of section C's
// `unbound_precision_name_in_let_now_rejected`; the two were merged
// into that single test in the e2e parsimony pass. See the doc comment
// on `unbound_precision_name_in_let_now_rejected` above for the WS-B2
// contextual-literal angle.
// ---------------------------------------------------------------------------

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
        "def use_add(x: tensor[3, f32], y: tensor[3, i32]) \
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
        "def use_sub(x: tensor[3, i32], y: tensor[3, i64]) \
         -> tensor[3, i32] = sub(x, y)\n",
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
