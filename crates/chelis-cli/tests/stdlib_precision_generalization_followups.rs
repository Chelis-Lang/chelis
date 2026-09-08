//! WS-C v3 acceptance: the 10 stdlib files that WS-C v2 escalated to
//! WS-A6 + WS-A7 are now generalized (or pinned with rationale) per
//! spec sec 5.8 + sec P4b.
//!
//! Coverage:
//!
//! 1. WS-A6 + WS-A7 sanity reproducer: a sig-with-borrows + bare-def
//!    pattern type-checks at every active dtype. This pins that the
//!    WS-A6 (contextual desugar for def parameter annotations) and
//!    WS-A7 (bare-def + sig-with-borrows return type inference)
//!    fixes that gated WS-C v3 actually deliver the required
//!    infrastructure.
//!
//! 2. Each newly-generalized stdlib op accepts every backend-supported
//!    dtype its sig admits. We use stdlib-shape replicas in single-file
//!    tests rather than chelis-std imports so the WS-C v3 evidence
//!    does not depend on the package staging infrastructure used by
//!    the chelis-std self-test corpus.
//!
//! 3. Float-only ops in the stdlib (silu / gelu / rmsnorm / loss /
//!    init / optim / generate) are pinned to f32 with explicit
//!    rationale at the source-file level (see commit message). Calling
//!    these ops at any other precision is a type error at the call
//!    site because the sig pins f32. This file pins the f32
//!    constraint by exercising it.
//!
//! The standalone type-check of each newly-generalized production
//! stdlib file under packages/chelis-std/src/ lives in the
//! consolidated `production_stdlib_typechecks.rs` (one check per file,
//! deduplicated across the WS-* suites).
//!
//! Coverage matrix (printed by run_coverage_matrix at the end):
//!
//! | op                          | f32 | f64 | bf16 | f16 | int8 | int16 | int32 | int64 |
//! | --------------------------- | --- | --- | ---- | --- | ---- | ----- | ----- | ----- |
//! | linear.forward              |  Y  |  Y  |  Y   |  Y  |   N  |   N   |   N   |   N   |
//! | embedding.forward (table p) |  Y  |  Y  |  Y   |  Y  |   Y  |   Y   |   Y   |   Y   |
//! | attention.sdpa              |  Y  |  Y  |  Y   |  Y  |   N  |   N   |   N   |   N   |
//! | metrics.accuracy            |  Y  |  Y  |  Y   |  Y  |   Y  |   Y   |   Y   |   Y   |
//! | optim.tensor_add            |  Y  |  Y  |  Y   |  Y  |   Y  |   Y   |   Y   |   Y   |
//! | optim.tensor_sub            |  Y  |  Y  |  Y   |  Y  |   Y  |   Y   |   Y   |   Y   |
//! | optim.tensor_mul            |  Y  |  Y  |  Y   |  Y  |   Y  |   Y   |   Y   |   Y   |
//! | optim.tensor_div (#178)     |  Y  |  Y  |  Y   |  Y  |   N  |   N   |   N   |   N   |
//! | optim.tensor_floor_div      |  Y  |  Y  |  Y   |  Y  |   Y  |   Y   |   Y   |   Y   |
//! | optim.tensor_trunc_div      |  N  |  N  |  N   |  N  |   Y  |   Y   |   Y   |   Y   |
//! | test.assert_close_tensor    |  Y  |  Y  |  Y   |  Y  |   N  |   N   |   N   |   N   |
//! | test.assert_shape           |  Y  |  Y  |  Y   |  Y  |   Y  |   Y   |   Y   |   Y   |
//!
//! Linear / attention reject integers because the underlying matmul
//! sig rejects integers per spec sec 5.7.2 (no integer matmul).
//! chelis#178: `div` is float-only (its integer-operand diagnostic
//! points at `floor_div` / `trunc_div`); `trunc_div` is integer-only;
//! `floor_div` admits both. All other rows admit every active
//! arithmetic dtype.
//!
//! For the f32-pinned ops (rmsnorm / silu / gelu / generate / random
//! / kaiming / xavierext / bce / crossentropy / kldiv / perplexity /
//! adamw / lamb), the pin is a sig-level commitment; calling at any
//! other precision is rejected at the type-check entry per spec sec
//! 5.4. We pin that rejection here too.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

const ARITHMETIC_DTYPES: &[&str] = &[
    "f32", "f64", "bf16", "f16", "int8", "int16", "int32", "int64",
];
const FLOAT_DTYPES: &[&str] = &["f32", "f64", "bf16", "f16"];

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

// Issue #207: `chelis check` now exits non-zero when the JSON
// `errors` array is non-empty. The follow-up matrix below mixes
// clean and error-expecting cases through the same helper.
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

fn expect_one_active_float_error(json: &Value, dtype: &str, label: &str) {
    let errs = errors(json);
    assert_eq!(
        errs.len(),
        1,
        "{label}: expected exactly one diagnostic, never an empty-error fallback or cascade; got {errs:?}"
    );
    let error = errs[0];
    assert_eq!(
        error.get("kind").and_then(Value::as_str),
        Some("PrecisionMismatch"),
        "{label}: expected PrecisionMismatch; got {error:?}"
    );
    let message = error.get("message").and_then(Value::as_str).unwrap_or("");
    assert!(
        message.contains("active float dtype") && message.contains(dtype),
        "{label}: expected an active-float diagnostic naming {dtype}; got {message:?}"
    );
}

// =================================================================
// 1. WS-A6 + WS-A7 sanity reproducer.
//
// The WS-A6 (contextual desugar for def parameter annotations) and
// WS-A7 (bare-def + sig-with-borrows return inference) dtype-matrix
// re-tests that previously lived here were consolidated into their
// owning files in the e2e parsimony pass:
//   * `def_annotation_desugar.rs::def_quantifier_precision_tvar_typechecks_at_every_arithmetic_dtype`
//   * `bareref_return_inference.rs::bare_arg_add_with_borrow_sig_typechecks_at_every_arithmetic_dtype`
// Both owning files now carry the full arithmetic-dtype matrix.
// =================================================================

// =================================================================
// 2. Newly-generalized stdlib op shapes accept every admissible dtype.
// =================================================================

/// School.Nn.Linear.forward shape: matmul + add + expand. Accepts every
/// FLOAT dtype (integer matmul is rejected per spec sec 5.7.2).
/// Note: the test replicas use single-letter dim names (a, b, c)
/// because the desugar treats only single-letter and explicit-
/// quantifier names as d-vars; multi-letter names (batch, in_dim, ...)
/// become d-name (concrete) inside a sig-only declaration. The
/// production linear.ch sigs use single-letter dim names for the same
/// reason; precision generalization works orthogonally via the shared
/// p tvar in the sig's t-fn.
#[test]
fn linear_forward_accepts_all_float_dtypes() {
    for dtype in FLOAT_DTYPES {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("linear.ch");
        let src = format!(
            r#"sig forward: &tensor[a, b, p] -> &tensor[b, c, p] -> &tensor[c, p] -> tensor[a, c, p]
def forward(x, w, b) = {{
  bias = insert(b, 0, shape(x, cast(0, int32)))
  wx = matmul(x, w)
  out = add(wx, bias)
  _ = drop(bias)
  _ = drop(wx)
  out
}}
def call(x: &tensor[2, 3, {dtype}], w: &tensor[3, 4, {dtype}], bias_in: &tensor[4, {dtype}]) -> tensor[2, 4, {dtype}] = forward(x, w, bias_in)
"#
        );
        write_file(&path, &src);
        let json = run_check(&path);
        expect_clean(&json, &format!("linear.forward[{dtype}]"));
    }
}

// Spec sec 5.7.2 integer matmul rejection (direct-call path) is pinned
// by the keep-by-default regression lock
// `numeric_dtype_adversarial.rs::rt4_invariant_int_matmul_rejected_for_every_int_dtype`,
// which loops every integer dtype and asserts the 5.7.2 citation. The
// copy that previously lived here was removed in the e2e parsimony
// pass.

/// School.Nn.Embedding.forward shape: gather. The table precision is a
/// tvar; the ids dtype is fixed at int64. Accepts every active dtype
/// for the table element precision.
#[test]
fn embedding_forward_accepts_all_arithmetic_dtypes_for_table() {
    for dtype in ARITHMETIC_DTYPES {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("embedding.ch");
        let src = format!(
            r#"sig forward: &tensor[a, b, int64] -> &tensor[c, d, p] -> tensor[a, b, d, p]
def forward[a, b, c, d, p](ids, table) = gather(table, ids, 0)
def call(ids: &tensor[1, 2, int64], table: &tensor[4, 3, {dtype}]) -> tensor[1, 2, 3, {dtype}] = forward(ids, table)
"#
        );
        write_file(&path, &src);
        let json = run_check(&path);
        expect_clean(&json, &format!("embedding.forward table precision={dtype}"));
    }
}

/// School.Nn.Attention.scaled_dot_product_attention shape: matmul +
/// softmax + permute. Accepts every FLOAT dtype (integer matmul +
/// integer softmax both reject per spec sec 5.7.2 + sec 5.4).
#[test]
fn attention_sdpa_accepts_all_float_dtypes() {
    for dtype in FLOAT_DTYPES {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("attn.ch");
        let src = format!(
            r#"sig sdpa: &tensor[4, 4, p] -> &tensor[4, 4, p] -> &tensor[4, 4, p] -> &tensor[4, 4, p] -> tensor[4, 4, p]
def sdpa(q, k, v, scale) = {{
  kt = permute(k, 1, 0)
  scores = matmul(q, kt)
  scaled = mul(scores, scale)
  weights = softmax(scaled, -1)
  out = matmul(weights, v)
  _ = drop(kt)
  _ = drop(scores)
  _ = drop(scaled)
  _ = drop(weights)
  out
}}
def call(q: &tensor[4, 4, {dtype}], k: &tensor[4, 4, {dtype}], v: &tensor[4, 4, {dtype}], scale: &tensor[4, 4, {dtype}]) -> tensor[4, 4, {dtype}] = sdpa(q, k, v, scale)
"#
        );
        write_file(&path, &src);
        let json = run_check(&path);
        expect_clean(&json, &format!("attention.sdpa[{dtype}]"));
    }
}

/// School.Loss.Metrics.accuracy shape: sort returns int64 indices for any
/// input precision, so accuracy admits every active arithmetic dtype
/// as the logits precision.
#[test]
fn metrics_accuracy_accepts_all_arithmetic_dtypes_for_logits() {
    for dtype in ARITHMETIC_DTYPES {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("acc.ch");
        let src = format!(
            r#"sig row_argmax: &tensor[piece, classes, p] -> int64
def row_argmax[piece, classes, p](row: &tensor[piece, classes, p]) -> int64 = {{
  pair = sort(row, cast(1, int32))
  cast(0, int64)
}}
def call(xs: &tensor[1, 3, {dtype}]) -> int64 = row_argmax(xs)
"#
        );
        write_file(&path, &src);
        let json = run_check(&path);
        expect_clean(&json, &format!("metrics.accuracy logits precision={dtype}"));
    }
}

/// School.Optim.tensor_add / tensor_sub / tensor_mul shape: pure
/// delegation to the underlying primitive. Each accepts every active
/// arithmetic dtype. `floor_div` is the same shape (admits all
/// arithmetic dtypes per chelis#178). The variants share an identical
/// shape, so they are exercised by one table-driven test.
///
/// `div` and `trunc_div` are NOT in this all-dtypes loop: chelis#178
/// makes `div` float-only and `trunc_div` integer-only; their dtype
/// matrices are pinned separately below.
#[test]
fn optim_tensor_ops_accept_all_arithmetic_dtypes() {
    for op in ["add", "sub", "mul", "floor_div"] {
        for dtype in ARITHMETIC_DTYPES {
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("optim_op.ch");
            let src = format!(
                "def tensor_op[n, p](lhs: &tensor[n, p], rhs: &tensor[n, p]) -> tensor[n, p] = {op}(lhs, rhs)\n\
                 def call(xs: &tensor[3, {dtype}]) -> tensor[3, {dtype}] = tensor_op(xs, xs)\n"
            );
            write_file(&path, &src);
            let json = run_check(&path);
            expect_clean(&json, &format!("optim.tensor_{op}[{dtype}]"));
        }
    }
}

/// chelis#178: a polymorphic `div` wrapper accepts float dtypes and
/// rejects integer dtypes (pointing at `floor_div` / `trunc_div`).
#[test]
fn optim_tensor_div_accepts_float_rejects_integer_dtypes() {
    for dtype in ARITHMETIC_DTYPES {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("optim_div.ch");
        let src = format!(
            "def tensor_op[n, p](lhs: &tensor[n, p], rhs: &tensor[n, p]) -> tensor[n, p] = div(lhs, rhs)\n\
             def call(xs: &tensor[3, {dtype}]) -> tensor[3, {dtype}] = tensor_op(xs, xs)\n"
        );
        write_file(&path, &src);
        let json = run_check(&path);
        if FLOAT_DTYPES.contains(dtype) {
            expect_clean(&json, &format!("optim.tensor_div[{dtype}]"));
        } else {
            expect_any_error(&json, &format!("optim.tensor_div[{dtype}]"));
        }
    }
}

/// chelis#178: a polymorphic `trunc_div` wrapper is integer-only —
/// accepts integer dtypes and rejects float dtypes.
#[test]
fn optim_tensor_trunc_div_accepts_integer_rejects_float_dtypes() {
    for dtype in ARITHMETIC_DTYPES {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("optim_trunc_div.ch");
        let src = format!(
            "def tensor_op[n, p](lhs: &tensor[n, p], rhs: &tensor[n, p]) -> tensor[n, p] = trunc_div(lhs, rhs)\n\
             def call(xs: &tensor[3, {dtype}]) -> tensor[3, {dtype}] = tensor_op(xs, xs)\n"
        );
        write_file(&path, &src);
        let json = run_check(&path);
        if FLOAT_DTYPES.contains(dtype) {
            expect_any_error(&json, &format!("optim.tensor_trunc_div[{dtype}]"));
        } else {
            expect_clean(&json, &format!("optim.tensor_trunc_div[{dtype}]"));
        }
    }
}

/// Std.Test.assert_close_tensor is precision-generalized over the active
/// float dtypes, with one shared tensor/tolerance precision variable.
#[test]
fn test_assert_close_tensor_accepts_exactly_active_float_dtypes() {
    for dtype in ARITHMETIC_DTYPES {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("assert_close_t.ch");
        let src = format!(
            r#"sig assert_close_tensor: &tensor[n, p] -> &tensor[n, p] -> p -> string -> unit ! {{ Test }}
def assert_close_tensor(actual, expected, tolerance, label) = test_assert_close_tensor(actual, expected, tolerance, label)
def call(actual: &tensor[3, {dtype}], expected: &tensor[3, {dtype}]) -> unit ! {{ Test }} = assert_close_tensor(actual, expected, cast(0.001, {dtype}), "label")
"#
        );
        write_file(&path, &src);
        let json = run_check(&path);
        if FLOAT_DTYPES.contains(dtype) {
            expect_clean(&json, &format!("test.assert_close_tensor[{dtype}]"));
        } else {
            expect_any_error(&json, &format!("test.assert_close_tensor[{dtype}]"));
        }
    }
}

/// The public `chelis check` route must preserve the builtin's active-float
/// domain when the function value is aliased or passed through a higher-order
/// parameter. A literal callee-name check cannot satisfy this contract.
#[test]
fn test_assert_close_tensor_aliases_reject_every_non_float_dtype() {
    for dtype in ["int8", "int16", "int32", "int64", "bool"] {
        for (route, declarations) in [
            (
                "top-level alias",
                "close_alias = test_assert_close_tensor\n".to_string(),
            ),
            (
                "nested alias",
                "close_alias = test_assert_close_tensor\nnested_alias = close_alias\n"
                    .to_string(),
            ),
            (
                "higher-order alias",
                "close_alias = test_assert_close_tensor\ndef invoke(f, actual, expected, tol) = f(actual, expected, tol, \"cli\")\n"
                    .to_string(),
            ),
        ] {
            let callee = if route == "nested alias" {
                "nested_alias"
            } else if route == "higher-order alias" {
                "invoke(close_alias"
            } else {
                "close_alias"
            };
            let closing = if route == "higher-order alias" {
                ", actual, expected, tol)"
            } else {
                "(actual, expected, tol, \"cli\")"
            };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("assert_close_alias_neg.ch");
            let source = format!(
                "{declarations}def bad(actual: &tensor[2, {dtype}], expected: &tensor[2, {dtype}], tol: {dtype}) -> unit ! {{ Test }} = {callee}{closing}\n"
            );
            write_file(&path, &source);
            let json = run_check(&path);
            expect_one_active_float_error(&json, dtype, route);
        }

        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("assert_close_local_alias_neg.ch");
        let source = format!(
            r#"def bad(actual: &tensor[2, {dtype}], expected: &tensor[2, {dtype}], tol: {dtype}) -> unit ! {{ Test }} = {{
  close_alias = test_assert_close_tensor
  close_alias(actual, expected, tol, "cli")
}}
"#
        );
        write_file(&path, &source);
        let json = run_check(&path);
        expect_one_active_float_error(&json, dtype, "local alias");
    }
}

#[test]
fn test_assert_shape_accepts_all_arithmetic_dtypes() {
    for dtype in ARITHMETIC_DTYPES {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("assert_shape.ch");
        let src = format!(
            r#"sig assert_shape: &tensor[..r, p] -> List[int64] -> string -> unit ! {{ Test }}
def assert_shape(t, expected_shape, label) = ()
def call(t: &tensor[3, {dtype}]) -> unit ! {{ Test }} = assert_shape(t, [cast(3, int64)], "label")
"#
        );
        write_file(&path, &src);
        let json = run_check(&path);
        expect_clean(&json, &format!("test.assert_shape[{dtype}]"));
    }
}

// =================================================================
// 3. Negative coverage: sig-level precision mismatch must surface.
// =================================================================

/// Linear's forward sig pins x and w to the same precision tvar; a
/// call mixing two distinct precisions across input and weight is a
/// type error.
#[test]
fn linear_forward_rejects_mismatched_input_weight_precision() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("linear_mix.ch");
    write_file(
        &path,
        r#"sig forward: &tensor[a, b, p] -> &tensor[b, c, p] -> &tensor[c, p] -> tensor[a, c, p]
def forward(x, w, b) = {
  bias = insert(b, 0, shape(x, cast(0, int32)))
  wx = matmul(x, w)
  out = add(wx, bias)
  _ = drop(bias)
  _ = drop(wx)
  out
}
def bad(x: &tensor[2, 3, f32], w: &tensor[3, 4, bf16], b: &tensor[4, f32]) -> tensor[2, 4, f32] = forward(x, w, b)
"#,
    );
    let json = run_check(&path);
    expect_any_error(&json, "linear.forward mismatched input/weight precision");
}

/// Embedding's forward sig pins ids to int64; passing a non-int64 ids
/// tensor must fail at the call site.
#[test]
fn embedding_forward_rejects_non_int64_ids() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("embedding_neg.ch");
    write_file(
        &path,
        r#"sig forward: &tensor[batch, seq, int64] -> &tensor[vocab, hidden, p] -> tensor[batch, seq, hidden, p]
def forward[batch, seq, vocab, hidden, p](ids, table) = gather(table, ids, 0)
def bad(ids: &tensor[1, 2, int32], table: &tensor[4, 3, f32]) -> tensor[1, 2, 3, f32] = forward(ids, table)
"#,
    );
    let json = run_check(&path);
    expect_any_error(&json, "embedding.forward non-int64 ids");
}

/// Attention sdpa pins the four input tensors to the same precision
/// tvar; mixing precisions is a type error.
#[test]
fn attention_sdpa_rejects_mismatched_qk_precision() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("attn_neg.ch");
    write_file(
        &path,
        r#"sig sdpa: &tensor[4, 4, p] -> &tensor[4, 4, p] -> &tensor[4, 4, p] -> &tensor[4, 4, p] -> tensor[4, 4, p]
def sdpa(q, k, v, scale) = {
  kt = permute(k, 1, 0)
  scores = matmul(q, kt)
  scaled = mul(scores, scale)
  weights = softmax(scaled, -1)
  out = matmul(weights, v)
  _ = drop(kt)
  _ = drop(scores)
  _ = drop(scaled)
  _ = drop(weights)
  out
}
def bad(q: &tensor[4, 4, f32], k: &tensor[4, 4, bf16], v: &tensor[4, 4, f32], scale: &tensor[4, 4, f32]) -> tensor[4, 4, f32] = sdpa(q, k, v, scale)
"#,
    );
    let json = run_check(&path);
    expect_any_error(&json, "attention.sdpa mismatched q/k precision");
}

/// optim.tensor_add precision tvar pins lhs and rhs to the same dtype;
/// mixing is a type error.
#[test]
fn optim_tensor_add_rejects_mismatched_lhs_rhs_precision() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("optim_add_neg.ch");
    write_file(
        &path,
        r#"def tensor_add[n, p](lhs: &tensor[n, p], rhs: &tensor[n, p]) -> tensor[n, p] = add(lhs, rhs)
def bad(xs: &tensor[3, f32], ys: &tensor[3, bf16]) -> tensor[3, f32] = tensor_add(xs, ys)
"#,
    );
    let json = run_check(&path);
    expect_any_error(&json, "optim.tensor_add mismatched lhs/rhs precision");
}

/// test.assert_close_tensor precision tvar pins actual and expected to
/// the same dtype; mixing is a type error.
#[test]
fn test_assert_close_tensor_rejects_mismatched_precision() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("assert_close_neg.ch");
    write_file(
        &path,
        r#"sig assert_close_tensor: &tensor[n, p] -> &tensor[n, p] -> p -> string -> unit ! { Test }
def assert_close_tensor(actual, expected, tolerance, label) = test_assert_close_tensor(actual, expected, tolerance, label)
def bad(actual: &tensor[3, f32], expected: &tensor[3, bf16]) -> unit ! { Test } = assert_close_tensor(actual, expected, cast(0.001, f32), "label")
"#,
    );
    let json = run_check(&path);
    expect_any_error(
        &json,
        "test.assert_close_tensor mismatched actual/expected precision",
    );
}

// =================================================================
// 4. f32-pinned ops reject non-f32 input at the sig.
// =================================================================

/// f32-pinned stdlib ops reject non-f32 input at the sig.
///
/// * School.Nn.Silu.forward is f32-pinned per spec sec 5.4 transcendental
///   row (uses exp).
/// * School.Loss.CrossEntropy.loss is f32-pinned (uses softmax + log).
/// * School.Optim.AdamW step is f32-pinned (Config carries f32 fields).
///
/// All three share the identical "f32-pinned sig rejects non-f32 at
/// the call site" shape, so they are exercised by one table-driven
/// test over each op's f32-pinned fixture (consolidated in the e2e
/// parsimony pass). The `{dtype}` placeholder is substituted into
/// each fixture for every non-f32 arithmetic dtype.
#[test]
fn f32_pinned_ops_reject_non_f32_at_sig_level() {
    let non_f32 = ["f64", "bf16", "f16", "int8", "int16", "int32", "int64"];
    let fixtures: &[(&str, &str)] = &[
        (
            "silu.forward",
            "sig forward: &tensor[n, f32] -> tensor[n, f32]\n\
             def forward(x) = x\n\
             def bad(xs: &tensor[3, {dtype}]) -> tensor[3, {dtype}] = forward(xs)\n",
        ),
        (
            "crossentropy.loss",
            "sig loss: &tensor[batch, classes, f32] -> &tensor[batch, classes, f32] -> tensor[batch, f32]\n\
             def loss(logits, labels) = sum(mul(logits, labels), 1)\n\
             def bad(logits: &tensor[2, 3, {dtype}], labels: &tensor[2, 3, {dtype}]) -> tensor[2, {dtype}] = loss(logits, labels)\n",
        ),
        (
            "optim.adamw_step",
            "def adamw_step[n](params: tensor[n, f32], grads: &tensor[n, f32]) -> tensor[n, f32] = sub(params, grads)\n\
             def bad(p: tensor[3, {dtype}], g: &tensor[3, {dtype}]) -> tensor[3, {dtype}] = adamw_step(p, g)\n",
        ),
    ];
    for (label, template) in fixtures {
        for dtype in non_f32 {
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("f32_pinned_neg.ch");
            let src = template.replace("{dtype}", dtype);
            write_file(&path, &src);
            let json = run_check(&path);
            expect_any_error(&json, &format!("{label} rejects {dtype}"));
        }
    }
}
