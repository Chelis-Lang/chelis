//! chelis#285 regression: an inline annotation on a `def` that also has an
//! explicit standalone `sig` must NOT bypass body-vs-signature checking.
//!
//! Before the fix, `crates/chelis-surf/src/desugar.rs::desugar_fun_def`
//! synthesized a second `(defsig ...)` from the def's inline annotations,
//! filling every un-annotated position with a wildcard `(t-var {} _)`. The
//! type checker's `defsig` binding is last-write-wins
//! (`chelis-types::collect_declarations`), so the wildcard signature
//! silently overwrote the concrete explicit `sig`, dropping the
//! body-vs-signature contract on the un-annotated positions. A body could
//! then transpose, change rank, or collapse rigid dims while every caller
//! was still checked against the real `sig`.
//!
//! Fix: when an explicit `sig` already declares the name, the redundant
//! synthesized signature is suppressed so the explicit sig drives body
//! validation.
//!
//! Each lie below passes **clean** before the fix and is **rejected** after
//! it; the honest positives must remain clean both ways. The controls
//! (same lie without the inline annotation) were already rejected and are
//! kept here so a future regression that re-introduces the bypass is
//! distinguishable from a general body-check regression.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use tempfile::tempdir;

fn check_json(src: &str) -> Value {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("m.ch");
    fs::write(&path, src).expect("write file");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", "--allow-style-violations", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    serde_json::from_slice(&output.stdout).expect("check output should be json")
}

fn deep_text(src: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("m.ch");
    fs::write(&path, src).expect("write file");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["deep", path.to_str().unwrap()])
        .output()
        .expect("run chelis deep");
    String::from_utf8(output.stdout).expect("deep output should be utf8")
}

fn assert_body_sig_rejected(json: &Value, label: &str) {
    let errors = json["errors"]
        .as_array()
        .unwrap_or_else(|| panic!("{label}: errors should be a json array, got {json}"));
    assert!(
        !errors.is_empty(),
        "{label}: expected a body-vs-signature error, got a clean check ({json})"
    );
    let has_sig_err = errors.iter().any(|e| {
        e["message"]
            .as_str()
            .is_some_and(|m| m.contains("body doesn't match declared signature"))
    });
    assert!(
        has_sig_err,
        "{label}: expected a \"body doesn't match declared signature\" error, got {errors:?}"
    );
    let score = json["score"].as_f64().unwrap_or(1.0);
    assert!(
        score < 1.0,
        "{label}: a rejected body must score < 1.0, got {score} ({json})"
    );
}

fn assert_clean(json: &Value, label: &str) {
    let errors = json["errors"]
        .as_array()
        .unwrap_or_else(|| panic!("{label}: errors should be a json array, got {json}"));
    assert!(
        errors.is_empty(),
        "{label}: expected no check errors, got {errors:?}"
    );
    let score = json["score"].as_f64().unwrap_or(0.0);
    assert!(
        (score - 1.0).abs() < 1e-9,
        "{label}: expected score 1.0, got {score} ({json})"
    );
}

fn assert_effect_rejected(json: &Value, label: &str) {
    let errors = json["errors"]
        .as_array()
        .unwrap_or_else(|| panic!("{label}: errors should be a json array, got {json}"));
    assert!(
        !errors.is_empty(),
        "{label}: expected an undeclared-effect error, got a clean check ({json})"
    );
    let has_effect_err = errors.iter().any(|e| {
        e["message"]
            .as_str()
            .is_some_and(|m| m.contains("its body performs effects") && m.contains("not declared"))
    });
    assert!(
        has_effect_err,
        "{label}: expected an \"its body performs effects ... that were not declared\" error, got {errors:?}"
    );
    let score = json["score"].as_f64().unwrap_or(1.0);
    assert!(
        score < 1.0,
        "{label}: a rejected body must score < 1.0, got {score} ({json})"
    );
}

// ── Negatives: the three lie classes must be caught ──────────────────

/// (a) Transpose lie: bare param seeded from the sig, inline return
/// annotation. Body is `tensor[seq, batch]`; sig declares `tensor[batch,
/// seq]`. §4.2 transposition safety.
#[test]
fn transpose_lie_with_inline_return_is_rejected() {
    let json = check_json(
        "sig g: &tensor[batch, seq, f32] -> tensor[batch, seq, f32]\n\
         def g(x) -> tensor[batch, seq, f32] = permute(x, 1, 0)\n",
    );
    assert_body_sig_rejected(&json, "transpose lie (sig + inline return)");
}

/// (b) Return-contract drop: inline *param* annotation, no inline return.
/// Body reduces rank to `tensor[seq]`; sig declares rank-2 return. This is
/// the idiomatic `sig` + inline-param form (cf. `chelis-std .../generate.ch`).
#[test]
fn return_contract_drop_with_inline_param_is_rejected() {
    let json = check_json(
        "sig h: &tensor[batch, seq, f32] -> tensor[batch, seq, f32]\n\
         def h(x: &tensor[batch, seq, f32]) = sum(x, cast(0, int32))\n",
    );
    assert_body_sig_rejected(&json, "return drop (sig + inline param, no ret_ty)");
}

/// (c) §4.4 rigid-dim lie: bare params, inline return annotation. Body
/// returns `tensor[m]`; sig declares distinct rigid dims `n`, `m` with
/// return `tensor[n]`.
#[test]
fn rigid_dim_lie_with_inline_return_is_rejected() {
    let json = check_json(
        "sig f: &tensor[n, f32] -> &tensor[m, f32] -> tensor[n, f32]\n\
         def f(x, y) -> tensor[n, f32] = y\n",
    );
    assert_body_sig_rejected(&json, "rigid-dim lie (sig + inline return)");
}

// ── Controls: the same lies without the inline annotation stay caught ──

#[test]
fn transpose_lie_bare_param_no_annotation_control_still_rejected() {
    let json = check_json(
        "sig g: &tensor[batch, seq, f32] -> tensor[batch, seq, f32]\n\
         def g(x) = permute(x, 1, 0)\n",
    );
    assert_body_sig_rejected(&json, "transpose lie control (no inline annotation)");
}

// ── Positives: honest defs in the same surface forms must pass ──────────

/// Honest identity body under the exact `sig` + inline-return form that the
/// transpose lie abuses. The fix must not over-reject this.
#[test]
fn honest_identity_with_sig_and_inline_return_type_checks() {
    let json = check_json(
        "sig g: &tensor[batch, seq, f32] -> tensor[batch, seq, f32]\n\
         def g(x) -> tensor[batch, seq, f32] = relu(x)\n",
    );
    assert_clean(&json, "honest identity (sig + inline return)");
}

/// Honest body with an inline param annotation and no inline return, under
/// an explicit sig — the `generate.ch`-style form whose return contract was
/// previously dropped. Must pass when the body honors the sig.
#[test]
fn honest_inline_param_no_ret_with_sig_type_checks() {
    let json = check_json(
        "sig h: &tensor[batch, seq, f32] -> tensor[batch, seq, f32]\n\
         def h(x: &tensor[batch, seq, f32]) = relu(x)\n",
    );
    assert_clean(&json, "honest inline-param (sig, no ret_ty)");
}

/// Inline annotations with NO standalone sig: the synthesized signature is
/// the only one, so it must still be emitted and still validate the body.
/// (Suppression is keyed on an explicit sig existing; this case has none.)
#[test]
fn honest_inline_annotations_without_sig_still_validated() {
    // Honest body passes.
    let ok =
        check_json("def g(x: &tensor[batch, seq, f32]) -> tensor[batch, seq, f32] = relu(x)\n");
    assert_clean(&ok, "honest inline-only (no sig)");
    // Dishonest body without a sig is still caught by the synthesized sig.
    let bad = check_json(
        "def g(x: &tensor[batch, seq, f32]) -> tensor[batch, seq, f32] = permute(x, 1, 0)\n",
    );
    assert_body_sig_rejected(&bad, "transpose lie inline-only (no sig)");
}

// ── Structural lock: with an explicit sig, no redundant defsig is synthesized ──

/// The mechanism lock: when an explicit `sig g:` exists, desugaring an
/// annotated `def g` must emit exactly ONE `(defsig` for `g` (the explicit
/// one), not the wildcard-filled synthesized second copy that caused #285.
#[test]
fn explicit_sig_suppresses_synthesized_defsig() {
    let deep = deep_text(
        "sig g: &tensor[batch, seq, f32] -> tensor[batch, seq, f32]\n\
         def g(x) -> tensor[batch, seq, f32] = relu(x)\n",
    );
    let defsig_count = deep.matches("(defsig").count();
    assert_eq!(
        defsig_count, 1,
        "expected exactly one defsig for `g` (explicit sig only), got {defsig_count}:\n{deep}"
    );
    // And the surviving defsig must be the concrete explicit one, not a
    // wildcard `(t-var {} _)` parameter copy.
    assert!(
        !deep.contains("(t-var {} _)"),
        "the surviving signature must be the concrete explicit sig, found a wildcard t-var:\n{deep}"
    );
}

/// Counterpart: with NO explicit sig, the synthesized defsig IS still
/// emitted (a single one), so inline-only annotated defs keep their
/// signature. Guards against the suppression firing too broadly.
#[test]
fn no_explicit_sig_keeps_synthesized_defsig() {
    let deep =
        deep_text("def g(x: &tensor[batch, seq, f32]) -> tensor[batch, seq, f32] = relu(x)\n");
    let defsig_count = deep.matches("(defsig").count();
    assert_eq!(
        defsig_count, 1,
        "inline-only annotated def must still synthesize its defsig, got {defsig_count}:\n{deep}"
    );
}

// ── Effect-row parity: suppressing the synthesized defsig must not drop ──
// the def's `! { ... }` effect contract ──────────────────────────────────
//
// The synthesized defsig is also the carrier of a `def`'s inline effect
// clause (`apply_effect_metadata` attaches `eff` to the `t-fn`), and the
// effect upper-bound check reads the declared effect set only from a
// `defsig` (`chelis-effects::declared_effects_from_defsig`). So when an
// explicit `sig` carries no effect clause, naively suppressing the
// synthesized defsig would silently delete the def's effect bound and let
// the body leak effects unchecked — the effect-row analogue of #285. The
// fix makes `Decl::Sig` desugaring inherit the same-named def's clause when
// the sig declares none. These lock that in.

/// Regression: an explicit `sig` with NO effect clause must still enforce
/// the `def`'s own `! {}` bound. The body performs `IO` via `print` while
/// the def declares `! {}` (pure), so the effect upper-bound check must
/// fire. Passed clean before the effect-row fix.
#[test]
fn explicit_eff_less_sig_preserves_def_empty_effect_bound() {
    let json = check_json(
        "sig f: &tensor[n, f32] -> unit\n\
         def f(x) -> unit ! {} = print(x)\n",
    );
    assert_effect_rejected(&json, "def !{} effect bound under eff-less sig");
}

/// Same form but with no inline param/return annotation at all — so
/// `desugar_fun_def` never enters the synthesized-defsig branch and the
/// inherited-onto-the-sig path is the *only* carrier of the def's `! {}`.
/// Locks that the effect bound survives independently of the type-shape path.
#[test]
fn explicit_eff_less_sig_preserves_bare_def_empty_effect_bound() {
    let json = check_json(
        "sig f: &tensor[n, f32] -> unit\n\
         def f(x) ! {} = print(x)\n",
    );
    assert_effect_rejected(&json, "bare def !{} effect bound under eff-less sig");
}

/// Positive: the inherited bound is the *def's* clause, so an honest
/// `! { io }` def whose body performs exactly `IO` under an eff-less sig
/// must stay clean. The fix must not over-reject by inheriting a wrong
/// (e.g. empty) bound.
#[test]
fn honest_def_io_effect_clause_under_eff_less_sig_type_checks() {
    let json = check_json(
        "sig f: &tensor[n, f32] -> unit\n\
         def f(x) -> unit ! { io } = print(x)\n",
    );
    assert_clean(&json, "honest def !{io} under eff-less sig");
}

/// Control: the same effect lie WITHOUT an explicit sig is caught by the
/// surviving synthesized defsig, so a future regression that re-drops the
/// inherited bound is distinguishable from a general effect-check regression.
#[test]
fn effect_lie_inline_only_no_sig_control_still_rejected() {
    let json = check_json("def f(x: &tensor[n, f32]) -> unit ! {} = print(x)\n");
    assert_effect_rejected(&json, "effect lie control (no inline sig)");
}

/// Structural lock: an eff-less explicit `sig` plus a `def` with an effect
/// clause must emit exactly ONE `(defsig`, and that surviving defsig must
/// carry the inherited `eff` metadata so the effect checker can read it.
#[test]
fn eff_less_sig_inherits_def_effect_metadata_into_single_defsig() {
    let deep = deep_text(
        "sig f: &tensor[n, f32] -> unit\n\
         def f(x) -> unit ! {} = print(x)\n",
    );
    let defsig_count = deep.matches("(defsig").count();
    assert_eq!(
        defsig_count, 1,
        "expected exactly one defsig for `f` (explicit sig only), got {defsig_count}:\n{deep}"
    );
    assert!(
        deep.contains("eff:"),
        "the surviving defsig must carry the def's inherited effect metadata:\n{deep}"
    );
}

// ── Additional surface-form coverage the first pass missed ──────────────

/// An inline annotation that directly CONTRADICTS the explicit sig (rank-1
/// param/return vs the sig's rank-2) is the most literal statement of the
/// #285 bug class. After the fix the explicit sig drives validation and the
/// contradiction surfaces; before it, the synthesized sig (carrying the
/// inline rank-1 types) won and the lie passed clean.
#[test]
fn contradictory_inline_annotation_vs_sig_is_rejected() {
    let json = check_json(
        "sig h: &tensor[batch, seq, f32] -> tensor[batch, seq, f32]\n\
         def h(x: &tensor[batch, f32]) -> tensor[batch, f32] = relu(x)\n",
    );
    assert_body_sig_rejected(
        &json,
        "contradictory inline annotation (rank-1 vs sig rank-2)",
    );
}

/// Order independence: the `def` written BEFORE its `sig` must still suppress
/// the synthesized defsig and reject the lie. Suppression is keyed on a
/// pre-pass over all decls (`explicit_sig_names`), so it cannot depend on
/// source order; a regression making it order-dependent would slip past
/// every sig-first test above.
#[test]
fn def_before_sig_transpose_lie_is_rejected() {
    let src = "def g(x) -> tensor[batch, seq, f32] = permute(x, 1, 0)\n\
               sig g: &tensor[batch, seq, f32] -> tensor[batch, seq, f32]\n";
    assert_body_sig_rejected(&check_json(src), "transpose lie (def before sig)");
    let deep = deep_text(src);
    let defsig_count = deep.matches("(defsig").count();
    assert_eq!(
        defsig_count, 1,
        "def-before-sig must still emit exactly one defsig, got {defsig_count}:\n{deep}"
    );
}
