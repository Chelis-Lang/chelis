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

// Issue #207: `chelis check` now exits non-zero when the JSON
// `errors` array is non-empty. This helper is shared between
// `expect_clean` and `expect_any_error` test sites, so it
// captures stdout regardless of exit status. The dedicated
// invariant test covers the exit-code surface.
fn run_check(path: &Path) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    serde_json::from_slice(&output.stdout).expect("check output should be json")
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

// ---------------------------------------------------------------
// 1. WS-C blocker reproducer.
//
// The WS-C-blocker reproducer (polymorphic-precision sig must reject a
// precision mismatch across a single call) is pinned by the
// keep-by-default regression lock
// `precision_polymorphism_adversarial.rs::baseline_wsc_blocker_reproducer_errors_without_unbound_wrapping`
// and by `precision_polymorphism.rs::ws_c_blocker_polymorphic_precision_does_not_silently_accept_mismatch`.
// Both survivors write the identical fixture and assert the f32+int32
// mismatch error, so the copy that previously lived here was removed in
// the e2e parsimony pass.
// ---------------------------------------------------------------

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

/// Std.Tensor.Reduce.argmax / argmin: input precision is generalized;
/// output is always int64 indices (changed in WS-C from spec-misaligned
/// f32). The two ops share an identical sig shape, so they are
/// exercised by one table-driven test over `[argmax, argmin]`
/// (consolidated in the e2e parsimony pass).
#[test]
fn stub_sig_argmax_argmin_shape_returns_int64_indices_at_all_arithmetic_input_dtypes() {
    for op in ["argmax", "argmin"] {
        for dtype in ARITHMETIC_DTYPES {
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("argreduce.ch");
            let src = format!(
                "sig {op}: &tensor[a, b, p] -> int32 -> tensor[b, int64]\n\
                 def call_{op}(xs: &tensor[2, 3, {dtype}]) -> tensor[3, int64] =\n  \
                 {op}(xs, cast(0, int32))\n"
            );
            write_file(&path, &src);
            let json = run_check(&path);
            expect_clean(&json, &format!("{op}[{dtype}]"));
        }
    }
}

/// Std.Nn.Conv.conv1d / conv2d_small: production sig shapes with
/// concrete dims; the precision tvar admits every dtype the sig itself
/// does not restrict. Surf has no kind-restriction syntax in this
/// cycle, so the test covers every active dtype rather than only
/// floats. The runtime surface for conv ops is an HIP backend gap and
/// is not exercised here. The two conv shapes share the identical
/// "concrete-dim sig generalized over a precision tvar" structure, so
/// they are exercised by one table-driven test over each conv
/// fixture (consolidated in the e2e parsimony pass).
#[test]
fn stub_sig_conv_shapes_accept_all_dtypes_at_sig_level() {
    let fixtures: &[(&str, &str)] = &[
        (
            "conv1d",
            "sig conv1d: &tensor[1, 4, 1, 16, p] -> &tensor[8, 4, 1, 3, p] -> tensor[1, 8, 1, 14, p]\n\
             def call_conv1d(x: &tensor[1, 4, 1, 16, {dtype}], w: &tensor[8, 4, 1, 3, {dtype}]) -> tensor[1, 8, 1, 14, {dtype}] = conv1d(x, w)\n",
        ),
        (
            "conv2d_small",
            "sig conv2d_small: &tensor[1, 3, 8, 8, p] -> &tensor[8, 3, 3, 3, p] -> tensor[1, 8, 6, 6, p]\n\
             def call_conv2d(x: &tensor[1, 3, 8, 8, {dtype}], w: &tensor[8, 3, 3, 3, {dtype}]) -> tensor[1, 8, 6, 6, {dtype}] = conv2d_small(x, w)\n",
        ),
    ];
    for (label, template) in fixtures {
        for dtype in ARITHMETIC_DTYPES {
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("conv.ch");
            let src = template.replace("{dtype}", dtype);
            write_file(&path, &src);
            let json = run_check(&path);
            expect_clean(&json, &format!("{label}[{dtype}]"));
        }
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

/// Calling `min` / `prod` with a return-precision that differs from
/// the input precision must fail because the precision tvar `p` must
/// be the same for input and output. The two reduce ops share the
/// identical sig shape, so they are exercised by one table-driven test
/// over each `(op, input_dtype)` pair (consolidated in the e2e
/// parsimony pass; the distinct input dtypes of the original pair are
/// preserved as the per-op table entries).
#[test]
fn neg_reduce_rejects_mismatched_input_output_precision() {
    let cases: &[(&str, &str)] = &[("min", "int32"), ("prod", "f64")];
    for (op, input_dtype) in cases {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("neg_reduce.ch");
        let src = format!(
            "sig {op}: &tensor[a, b, p] -> int32 -> tensor[b, p]\n\
             def bad(xs: &tensor[2, 3, {input_dtype}]) -> tensor[3, f32] =\n  \
             {op}(xs, cast(0, int32))\n"
        );
        write_file(&path, &src);
        let json = run_check(&path);
        expect_any_error(&json, &format!("{op} mismatched in/out precision"));
    }
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
//
// The integer-matmul rejection invariant is pinned by the
// keep-by-default regression lock
// `numeric_dtype_adversarial.rs::rt4_invariant_int_matmul_rejected_for_every_int_dtype`,
// which loops every integer dtype and asserts the 5.7.2 citation. The
// copy that previously lived here was removed in the e2e parsimony
// pass. The positive parity case (float matmul accepted) stays here
// because no other file pins the float-matmul-accepted side.
// ---------------------------------------------------------------

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

/// A single polymorphic stub used at two distinct concrete dtypes
/// across two call sites in the same module type-checks cleanly. The
/// `min` (precision-passthrough sig) and `argmax` (int64-indices sig)
/// cases share the identical "one polymorphic stub, two concrete call
/// sites" structure, so they are exercised by one table-driven test
/// over each stub fixture (consolidated in the e2e parsimony pass).
#[test]
fn polymorphic_stub_used_at_two_distinct_dtypes_in_same_module() {
    let fixtures: &[(&str, &str)] = &[
        (
            "min used at f32 and int64 in same module",
            "sig min: &tensor[a, b, p] -> int32 -> tensor[b, p]\n\
             def call_f32(xs: &tensor[2, 3, f32]) -> tensor[3, f32] = min(xs, cast(0, int32))\n\
             def call_int64(xs: &tensor[2, 3, int64]) -> tensor[3, int64] = min(xs, cast(0, int32))\n",
        ),
        (
            "argmax at f32 and int8 in same module",
            "sig argmax: &tensor[a, b, p] -> int32 -> tensor[b, int64]\n\
             def call_f32(xs: &tensor[2, 3, f32]) -> tensor[3, int64] = argmax(xs, cast(0, int32))\n\
             def call_int8(xs: &tensor[2, 3, int8]) -> tensor[3, int64] = argmax(xs, cast(0, int32))\n",
        ),
    ];
    for (label, src) in fixtures {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("two_dtypes.ch");
        write_file(&path, src);
        let json = run_check(&path);
        expect_clean(&json, label);
    }
}
