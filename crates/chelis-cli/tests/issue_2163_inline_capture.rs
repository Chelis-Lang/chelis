//! chelis#2163: host-lane inlining must not capture a caller's variable under
//! a same-named binder in the callee.
//!
//! `substitute_expr` replaces a callee's parameter names with the caller's
//! argument expressions. It tracks the callee's own `fn` and `let` binders, so
//! a binder is never itself replaced. It did NOT check the other direction:
//! whether a binder it substitutes UNDER captures a free variable of the
//! argument. With `k` replaced by the caller's `x` inside
//! `fn (x: p) -> add(x, k)`, the body became `fn (x) -> add(x, x)` and the
//! compiled lane computed a different answer from `eval`.
//!
//! Every test compares `eval` against the built, linked and executed C, which
//! is the only lane that showed the defect. [04-DTYPE-1] and the checker are
//! not involved: these programs are well typed and `eval` was always right.
//!
//! Negative parity is the shadowing direction, which must not regress: a
//! callee binder that shadows the callee's OWN parameter still shadows it, and
//! renaming a captured binder must not change a program that never captured.
use assert_cmd::Command;
use std::fs;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

fn program(body: &str) -> String {
    format!("module Bounded.Main\nexport (main)\n{body}\n")
}

fn eval(source: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("capture.ch");
    fs::write(&path, source).expect("source");
    let output = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("eval");
    assert!(
        output.status.success(),
        "eval rejected: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("UTF-8")
}

/// The compiled lane must agree with `eval`, and both must equal `expected`.
fn assert_lanes_agree(body: &str, name: &str, expected: &[f64]) {
    let source = program(body);
    let interpreted = eval(&source);
    assert_eq!(
        common::parse_tensor_data(&interpreted, "main"),
        expected,
        "{name}: eval value"
    );
    let native = common::build_and_run(&source, name);
    assert_eq!(native.trim(), interpreted.trim(), "{name}: eval vs C");
}

const SHIFT_ALL: &str = "def shift_all[n, p: Float](v: &tensor[n, p], k: p) -> tensor[n, p] = \
     to_tensor(map(fn (x: p) -> add(x, k), to_list(v)))\n";

#[test]
fn a_lambda_binder_does_not_capture_the_callers_argument() {
    // The callee's lambda binder `x` has the same name as the caller's `x`,
    // which is substituted in for `k`. Captured, the body doubles each element
    // ([2, 4]) instead of shifting it by 100.
    assert_lanes_agree(
        &format!(
            "{SHIFT_ALL}def main() -> tensor[2, f64] = {{\n  x = 100.0f64\n  \
             shift_all(to_tensor([1.0f64, 2.0f64]), x)\n}}"
        ),
        "capture_generic_f64",
        &[101.0, 102.0],
    );
}

#[test]
fn a_callable_parameter_callee_does_not_capture_either() {
    // The same defect without generics: a callee with a function-typed
    // parameter is inlined too. Captured, this gives [3, 6] instead of
    // [102, 104].
    assert_lanes_agree(
        "def apply_shift(f: (f32) -> f32, k: f32) -> tensor[2, f32] = \
             to_tensor(map(fn (x: f32) -> add(f(x), k), to_list(to_tensor([1.0f32, 2.0f32]))))\n\
         def dbl(a: f32) -> f32 = mul(a, 2.0f32)\n\
         def main() -> tensor[2, f32] = {\n  x = 100.0f32\n  apply_shift(dbl, x)\n}",
        "capture_callable_param",
        &[102.0, 104.0],
    );
}

#[test]
fn a_caller_variable_with_a_different_name_is_unaffected() {
    // The control for the fix: no collision, so no renaming, and the result is
    // the one this program always had in eval.
    assert_lanes_agree(
        &format!(
            "{SHIFT_ALL}def main() -> tensor[2, f64] = {{\n  y = 100.0f64\n  \
             shift_all(to_tensor([1.0f64, 2.0f64]), y)\n}}"
        ),
        "capture_control_renamed",
        &[101.0, 102.0],
    );
}

#[test]
fn a_callee_binder_still_shadows_the_callees_own_parameter() {
    // Negative parity. `fn (k: p) -> add(k, k)` binds `k` itself, so the
    // parameter `k` must NOT be substituted inside it: the result doubles each
    // element and does not mention the caller's value at all. Capture
    // avoidance must not weaken that.
    assert_lanes_agree(
        "def shadow_k[n, p: Float](v: &tensor[n, p], k: p) -> tensor[n, p] = \
             to_tensor(map(fn (k: p) -> add(k, k), to_list(v)))\n\
         def main() -> tensor[2, f64] = {\n  x = 100.0f64\n  \
         shadow_k(to_tensor([1.0f64, 2.0f64]), x)\n}",
        "shadow_own_parameter",
        &[2.0, 4.0],
    );
}

#[test]
fn a_let_value_slot_is_outside_the_binders_scope() {
    // `x = k` substitutes into the let's VALUE, which the binder does not
    // scope over, so nothing is captured. This one already produced the right
    // answer before the fix; it is here so the `let` arm's renaming cannot
    // regress it.
    assert_lanes_agree(
        "def shift_let[n, p: Float](v: &tensor[n, p], k: p) -> tensor[n, p] = {\n  x = k\n  \
             to_tensor(map(fn (e: p) -> add(e, x), to_list(v)))\n}\n\
         def main() -> tensor[2, f64] = {\n  x = 100.0f64\n  \
         shift_let(to_tensor([1.0f64, 2.0f64]), x)\n}",
        "capture_let_binder",
        &[101.0, 102.0],
    );
}

#[test]
fn a_let_bound_name_in_the_callee_does_not_capture_the_argument() {
    // The `let` binder capturing for real: `k` is substituted inside the let
    // BODY, which `x` does scope over. Captured, each element is shifted by
    // the let's own 7 twice ([15, 16]) instead of by the caller's 100 and the
    // 7 ([108, 109]).
    assert_lanes_agree(
        "def shift_body[n, p: Float](v: &tensor[n, p], k: p, m: p) -> tensor[n, p] = {\n  \
             x = m\n  to_tensor(map(fn (e: p) -> add(add(e, k), x), to_list(v)))\n}\n\
         def main() -> tensor[2, f64] = {\n  x = 100.0f64\n  \
         shift_body(to_tensor([1.0f64, 2.0f64]), x, 7.0f64)\n}",
        "capture_let_body",
        &[108.0, 109.0],
    );
}

#[test]
fn a_fresh_name_avoids_an_existing_unreferenced_inl_binder() {
    // The fresh name must collide with nothing, including a binder that the
    // body never reads. `avoid` built from `var` references alone does not see
    // the `x__inl1` binder here, so the outer `x` renames onto it and the
    // inner lambda's `x` reads the wrong value: [103, 106] instead of
    // [102, 104].
    assert_lanes_agree(
        "def cap[n, p: Float](v: &tensor[n, p], k: p) -> tensor[n, p] = \
             to_tensor(map(fn (x: p) -> add(x, (fn (x__inl1: p) -> add(x, k))(add(x, x))), \
             to_list(v)))\n\
         def main() -> tensor[2, f64] = {\n  x = 100.0f64\n  \
         cap(to_tensor([1.0f64, 2.0f64]), x)\n}",
        "fresh_avoids_existing_inl",
        &[102.0, 104.0],
    );
}

#[test]
fn a_shadowed_parameter_does_not_trigger_a_rename() {
    // A replacement whose parameter is shadowed here can never be inserted
    // below, so no binder underneath needs renaming. Renaming anyway is not
    // free: the fresh name can land on an existing binder and capture. With
    // `live` widened to every replacement, the inner `fn (x: p)` renames onto
    // the existing `x__inl1` and this gives [5, 10] instead of [3, 6].
    assert_lanes_agree(
        "def f[n, p: Float](v: &tensor[n, p], k: p) -> tensor[n, p] = \
             to_tensor(map(fn (k: p) -> \
             (fn (x: p) -> add(x, (fn (x__inl1: p) -> add(x, x))(add(x, x))))(k), \
             to_list(v)))\n\
         def main() -> tensor[2, f64] = {\n  x = 100.0f64\n  \
         f(to_tensor([1.0f64, 2.0f64]), x)\n}",
        "shadowed_parameter_no_rename",
        &[3.0, 6.0],
    );
}

#[test]
fn a_let_binding_rename_does_not_reach_its_own_value() {
    // `x = add(x, m)` reads the OUTER `x` (the callee's parameter): the
    // binding's own name is not in scope in its own value. Applying the
    // rename there produces `x__inl1 = add(x__inl1, m)`, a self-reference that
    // fails ownership lowering with "references unbound name `x__inl1`".
    assert_lanes_agree(
        "def bump[n, p: Float](v: &tensor[n, p], x: p, m: p) -> tensor[n, p] = {\n  \
             x = add(x, m)\n  to_tensor(map(fn (e: p) -> add(e, x), to_list(v)))\n}\n\
         def main() -> tensor[2, f64] = {\n  x = 100.0f64\n  \
         bump(to_tensor([1.0f64, 2.0f64]), x, 7.0f64)\n}",
        "let_rename_skips_own_value",
        &[108.0, 109.0],
    );
}

#[test]
fn renaming_a_captured_binder_stops_at_an_inner_binder_of_the_same_name() {
    // The renaming is itself scope-aware. The outer lambda's `x` is renamed
    // because it would capture the caller's `x`, but the inner lambda binds
    // `x` again over a different value (`add(x, x)`), so the inner body's `x`
    // must keep referring to the inner binder. A rename that ignored that
    // shadowing would read the outer element instead and give [5, 18].
    assert_lanes_agree(
        "def nest[n, p: Float](v: &tensor[n, p], k: p) -> tensor[n, p] = \
             to_tensor(map(fn (x: p) -> add(x, (fn (x: p) -> mul(x, k))(add(x, x))), \
             to_list(v)))\n\
         def main() -> tensor[2, f64] = {\n  x = 3.0f64\n  \
         nest(to_tensor([1.0f64, 2.0f64]), x)\n}",
        "capture_nested_shadow",
        &[7.0, 14.0],
    );
}
