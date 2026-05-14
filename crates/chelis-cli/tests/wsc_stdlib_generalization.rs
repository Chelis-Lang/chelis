//! WS-C acceptance tests: stdlib tensor-op signatures generalized over
//! a precision type variable per `spec/04-type-system.md` sec 5.8 and
//! the `spec/02-surf-syntax.md` sec P4b contextual-precision-polymorphism
//! rule shipped in WS-A5.
//!
//! Coverage focus:
//!
//! * The original WS-C blocker reproducer (a polymorphic-precision sig
//!   silently accepting a precision mismatch) now errors at type-check
//!   time, demonstrating that WS-A5 actually delivered the required
//!   infrastructure.
//! * Each generalized stdlib stub-sig shape (matching the production
//!   shape in `packages/chelis-std/src/`) accepts every backend-supported
//!   dtype in its precision slot per spec sec 1.1 and sec 5.4. We
//!   replicate the sig in a single-file test rather than importing from
//!   chelis-std so the test does not depend on the package staging
//!   infrastructure used by the chelis-std self-test suite.
//! * Spec sec 5.7.2 integer-matmul rejection fires at the type-check
//!   entry with a precise diagnostic citing the section, matching the
//!   stdlib's float-only matmul shape requirement.
//! * Polymorphic precision pass-through across two distinct concrete
//!   instantiations in the same module continues to type-check.
//!
//! The 13 pre-existing chelis-std test failures noted in the WS-C
//! brief (8 sig-only-without-def reduce ops; 5 optimizer evaluator
//! failures) are out of scope for WS-C and are not exercised here.
//!
//! This file does NOT cover the call-site rejection of transcendental
//! ops (`exp`, `log`, `sin`, etc.) on integer tensors. Spec sec 5.4
//! requires that rejection at the type-check entry, but the current
//! type-check infrastructure routes transcendentals through the
//! polymorphic `tensor_unop` scheme without a kind restriction; the
//! IR lowering's float-only enforcement (lower_transcendental) silently
//! produces a zero-constant placeholder for integer tensors instead of
//! emitting a diagnostic. Closing this gap is a compiler-side change
//! (kind-restriction syntax in Surf sigs OR a transcendental-only
//! scheme in chelis-types/builtins.rs that rejects integer operands
//! at type-check time) and therefore out of scope for the stdlib-only
//! WS-C dispatch.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

fn run_check(path: &Path) -> Value {
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

fn errors(json: &Value) -> Vec<&Value> {
    json["errors"]
        .as_array()
        .expect("errors should be a json array")
        .iter()
        .collect()
}

fn expect_clean(json: &Value, label: &str) {
    let errs = errors(json);
    assert!(
        errs.is_empty(),
        "{label}: expected clean check, got {errs:?}"
    );
    let score = json["score"].as_f64().unwrap_or(0.0);
    assert!(
        (score - 1.0).abs() < f64::EPSILON,
        "{label}: expected score 1.0, got {score}"
    );
}

fn expect_error_mentioning(json: &Value, fragment: &str, label: &str) {
    let errs = errors(json);
    assert!(
        !errs.is_empty(),
        "{label}: expected at least one error, got clean output {json}"
    );
    let any_match = errs.iter().any(|e| {
        e.get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("")
            .contains(fragment)
    });
    assert!(
        any_match,
        "{label}: expected error containing `{fragment}`, got {errs:?}"
    );
}

fn expect_any_error(json: &Value, label: &str) {
    let errs = errors(json);
    assert!(
        !errs.is_empty(),
        "{label}: expected at least one error, got clean output {json}"
    );
}

// Active arithmetic dtypes per spec sec 1.1, intersected with sec 5.4
// arithmetic row.
const ARITHMETIC_DTYPES: &[&str] = &[
    "f32", "f64", "bf16", "f16", "int8", "int16", "int32", "int64",
];
const FLOAT_DTYPES: &[&str] = &["f32", "f64", "bf16", "f16"];
const INTEGER_DTYPES: &[&str] = &["int8", "int16", "int32", "int64"];

// ---------------------------------------------------------------
// 1. WS-C blocker reproducer (sanity check WS-A5 still delivers).
// ---------------------------------------------------------------

/// Spec sec 5.8 requires that a polymorphic-precision sig actually
/// constrain the precision across a single call. Before WS-A5 the
/// precision slot was a wildcard and this case silently passed; the
/// WS-C dispatch turns on the type-checker enforcement.
#[test]
fn wsc_blocker_reproducer_polymorphic_precision_rejects_mismatch() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("wsc_blocker.ch");
    write_file(
        &path,
        r#"sig poly_id: tensor[d, p] -> tensor[d, p]
def poly_id(x) = x
def use_mismatch(x: tensor[3, int32]) -> tensor[3, f32] = poly_id(x)
"#,
    );
    let json = run_check(&path);
    expect_error_mentioning(&json, "f32", "wsc-blocker");
    expect_error_mentioning(&json, "int32", "wsc-blocker");
}

// ---------------------------------------------------------------
// 2. Stub-sig shape coverage matrix; replicate the production sig
//    shapes from packages/chelis-std/src/ in a single-file context
//    and exercise each at every active dtype.
// ---------------------------------------------------------------

/// Std.Tensor.Reduce.min: arithmetic reduction; precision tvar admits
/// every active arithmetic dtype. The stub sig has no def in
/// production stdlib; we replicate that here and exercise it through a
/// caller. Calling a stub would fail at runtime but type-check
/// succeeds, which is what this test pins.
#[test]
fn stub_sig_min_shape_accepts_all_arithmetic_dtypes() {
    for dtype in ARITHMETIC_DTYPES {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("min.ch");
        let src = format!(
            r#"sig min: &tensor[a, b, p] -> int32 -> tensor[b, p]
def call_min(xs: &tensor[2, 3, {dtype}]) -> tensor[3, {dtype}] =
  min(xs, cast(0, int32))
"#
        );
        write_file(&path, &src);
        let json = run_check(&path);
        expect_clean(&json, &format!("min[{dtype}]"));
    }
}

/// Std.Tensor.Reduce.prod: same shape as min.
#[test]
fn stub_sig_prod_shape_accepts_all_arithmetic_dtypes() {
    for dtype in ARITHMETIC_DTYPES {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("prod.ch");
        let src = format!(
            r#"sig prod: &tensor[a, b, p] -> int32 -> tensor[b, p]
def call_prod(xs: &tensor[2, 3, {dtype}]) -> tensor[3, {dtype}] =
  prod(xs, cast(0, int32))
"#
        );
        write_file(&path, &src);
        let json = run_check(&path);
        expect_clean(&json, &format!("prod[{dtype}]"));
    }
}

/// Std.Tensor.Reduce.argmax: input precision is generalized; output is
/// always int64 indices (changed in WS-C from spec-misaligned f32).
#[test]
fn stub_sig_argmax_shape_returns_int64_indices_at_all_arithmetic_input_dtypes() {
    for dtype in ARITHMETIC_DTYPES {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("argmax.ch");
        let src = format!(
            r#"sig argmax: &tensor[a, b, p] -> int32 -> tensor[b, int64]
def call_argmax(xs: &tensor[2, 3, {dtype}]) -> tensor[3, int64] =
  argmax(xs, cast(0, int32))
"#
        );
        write_file(&path, &src);
        let json = run_check(&path);
        expect_clean(&json, &format!("argmax[{dtype}]"));
    }
}

/// Std.Tensor.Reduce.argmin mirror of argmax.
#[test]
fn stub_sig_argmin_shape_returns_int64_indices_at_all_arithmetic_input_dtypes() {
    for dtype in ARITHMETIC_DTYPES {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("argmin.ch");
        let src = format!(
            r#"sig argmin: &tensor[a, b, p] -> int32 -> tensor[b, int64]
def call_argmin(xs: &tensor[2, 3, {dtype}]) -> tensor[3, int64] =
  argmin(xs, cast(0, int32))
"#
        );
        write_file(&path, &src);
        let json = run_check(&path);
        expect_clean(&json, &format!("argmin[{dtype}]"));
    }
}

/// Std.Nn.Conv.conv1d: production sig shape with concrete dims; the
/// precision tvar admits every dtype the sig itself does not restrict.
/// Surf has no kind-restriction syntax in this cycle, so the test
/// covers every active dtype rather than only floats. The runtime
/// surface for conv ops is an HIP backend gap and is not exercised
/// here.
#[test]
fn stub_sig_conv1d_shape_accepts_all_dtypes_at_sig_level() {
    for dtype in ARITHMETIC_DTYPES {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("conv1d.ch");
        let src = format!(
            r#"sig conv1d: &tensor[1, 4, 1, 16, p] -> &tensor[8, 4, 1, 3, p] -> tensor[1, 8, 1, 14, p]
def call_conv1d(x: &tensor[1, 4, 1, 16, {dtype}], w: &tensor[8, 4, 1, 3, {dtype}]) -> tensor[1, 8, 1, 14, {dtype}] = conv1d(x, w)
"#
        );
        write_file(&path, &src);
        let json = run_check(&path);
        expect_clean(&json, &format!("conv1d[{dtype}]"));
    }
}

/// Std.Nn.Conv.conv2d_small: mirror of conv1d.
#[test]
fn stub_sig_conv2d_small_shape_accepts_all_dtypes_at_sig_level() {
    for dtype in ARITHMETIC_DTYPES {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("conv2d.ch");
        let src = format!(
            r#"sig conv2d_small: &tensor[1, 3, 8, 8, p] -> &tensor[8, 3, 3, 3, p] -> tensor[1, 8, 6, 6, p]
def call_conv2d(x: &tensor[1, 3, 8, 8, {dtype}], w: &tensor[8, 3, 3, 3, {dtype}]) -> tensor[1, 8, 6, 6, {dtype}] = conv2d_small(x, w)
"#
        );
        write_file(&path, &src);
        let json = run_check(&path);
        expect_clean(&json, &format!("conv2d_small[{dtype}]"));
    }
}

/// Std.Init.Xavier.sample: stub sig with Random effect; the precision
/// tvar appears in both the tensor slot and the scalar gain argument.
#[test]
fn stub_sig_xavier_sample_shape_accepts_all_dtypes_at_sig_level() {
    for dtype in ARITHMETIC_DTYPES {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("xavier.ch");
        let src = format!(
            r#"sig sample: tensor[32, 128, p] -> p -> tensor[32, 128, p] ! {{ Random }}
def call_xavier(t: tensor[32, 128, {dtype}], gain: {dtype}) -> tensor[32, 128, {dtype}] ! {{ Random }} = sample(t, gain)
"#
        );
        write_file(&path, &src);
        let json = run_check(&path);
        expect_clean(&json, &format!("xavier-sample[{dtype}]"));
    }
}

// ---------------------------------------------------------------
// 3. Negative coverage: sig-level precision mismatch within a single
//    call must surface, not silently accept.
// ---------------------------------------------------------------

/// Calling `min` with a return-precision that differs from the input
/// precision must fail because the precision tvar `p` must be the same
/// for input and output.
#[test]
fn neg_min_rejects_mismatched_input_output_precision() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("neg_min.ch");
    write_file(
        &path,
        r#"sig min: &tensor[a, b, p] -> int32 -> tensor[b, p]
def bad(xs: &tensor[2, 3, int32]) -> tensor[3, f32] =
  min(xs, cast(0, int32))
"#,
    );
    let json = run_check(&path);
    expect_any_error(&json, "min mismatched in/out precision");
}

/// Calling `prod` with mismatched return-precision must fail.
#[test]
fn neg_prod_rejects_mismatched_input_output_precision() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("neg_prod.ch");
    write_file(
        &path,
        r#"sig prod: &tensor[a, b, p] -> int32 -> tensor[b, p]
def bad(xs: &tensor[2, 3, f64]) -> tensor[3, f32] =
  prod(xs, cast(0, int32))
"#,
    );
    let json = run_check(&path);
    expect_any_error(&json, "prod mismatched in/out precision");
}

/// Calling `conv1d` with mismatched input/weight precision must fail
/// because both share the same precision tvar `p`.
#[test]
fn neg_conv1d_rejects_mismatched_input_weight_precision() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("neg_conv1d.ch");
    write_file(
        &path,
        r#"sig conv1d: &tensor[1, 4, 1, 16, p] -> &tensor[8, 4, 1, 3, p] -> tensor[1, 8, 1, 14, p]
def bad(x: &tensor[1, 4, 1, 16, f32], w: &tensor[8, 4, 1, 3, bf16]) -> tensor[1, 8, 1, 14, f32] = conv1d(x, w)
"#,
    );
    let json = run_check(&path);
    expect_any_error(&json, "conv1d mismatched input/weight precision");
}

/// Calling `argmax` and asserting an f32 result must fail because the
/// generalized sig pins the index dtype to int64.
#[test]
fn neg_argmax_return_must_be_int64_not_input_precision() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("neg_argmax.ch");
    write_file(
        &path,
        r#"sig argmax: &tensor[a, b, p] -> int32 -> tensor[b, int64]
def bad(xs: &tensor[2, 3, f32]) -> tensor[3, f32] =
  argmax(xs, cast(0, int32))
"#,
    );
    let json = run_check(&path);
    expect_any_error(&json, "argmax must return int64");
}

/// Calling xavier sample with a non-matching gain dtype must fail
/// because tensor and scalar share the same precision tvar.
#[test]
fn neg_xavier_sample_rejects_mismatched_gain_precision() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("neg_xavier.ch");
    write_file(
        &path,
        r#"sig sample: tensor[32, 128, p] -> p -> tensor[32, 128, p] ! { Random }
def bad(t: tensor[32, 128, f32], gain: f64) -> tensor[32, 128, f32] ! { Random } = sample(t, gain)
"#,
    );
    let json = run_check(&path);
    expect_any_error(&json, "xavier sample mismatched gain precision");
}

// ---------------------------------------------------------------
// 4. Spec sec 5.7.2: integer matmul rejection at the type-check entry.
// ---------------------------------------------------------------

#[test]
fn integer_matmul_rejected_at_type_check_per_spec_5_7_2() {
    for dtype in INTEGER_DTYPES {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("int_matmul.ch");
        let src = format!(
            r#"def call_matmul(a: tensor[3, 4, {dtype}], b: tensor[4, 5, {dtype}]) -> tensor[3, 5, {dtype}] =
  matmul(a, b)
"#
        );
        write_file(&path, &src);
        let json = run_check(&path);
        expect_error_mentioning(
            &json,
            "5.7.2",
            &format!("integer matmul rejection ({dtype})"),
        );
    }
}

/// Float matmul must continue to type-check across all float dtypes,
/// proving the rejection is precision-targeted, not blanket.
#[test]
fn float_matmul_accepted_at_all_float_dtypes() {
    for dtype in FLOAT_DTYPES {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("float_matmul.ch");
        let src = format!(
            r#"def call_matmul(a: tensor[3, 4, {dtype}], b: tensor[4, 5, {dtype}]) -> tensor[3, 5, {dtype}] =
  matmul(a, b)
"#
        );
        write_file(&path, &src);
        let json = run_check(&path);
        expect_clean(&json, &format!("float matmul ({dtype})"));
    }
}

// ---------------------------------------------------------------
// 5. Polymorphic precision pass-through: a single polymorphic stub
//    used at two different concrete dtypes across two call sites
//    type-checks cleanly.
// ---------------------------------------------------------------

#[test]
fn min_used_at_two_distinct_dtypes_in_same_module() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("two_dtypes.ch");
    write_file(
        &path,
        r#"sig min: &tensor[a, b, p] -> int32 -> tensor[b, p]
def call_f32(xs: &tensor[2, 3, f32]) -> tensor[3, f32] = min(xs, cast(0, int32))
def call_int64(xs: &tensor[2, 3, int64]) -> tensor[3, int64] = min(xs, cast(0, int32))
"#,
    );
    let json = run_check(&path);
    expect_clean(&json, "min used at f32 and int64 in same module");
}

#[test]
fn argmax_used_at_two_distinct_input_dtypes_in_same_module() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("two_argmax.ch");
    write_file(
        &path,
        r#"sig argmax: &tensor[a, b, p] -> int32 -> tensor[b, int64]
def call_f32(xs: &tensor[2, 3, f32]) -> tensor[3, int64] = argmax(xs, cast(0, int32))
def call_int8(xs: &tensor[2, 3, int8]) -> tensor[3, int64] = argmax(xs, cast(0, int32))
"#,
    );
    let json = run_check(&path);
    expect_clean(&json, "argmax at f32 and int8 in same module");
}
