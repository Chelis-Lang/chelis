//! chelis#3389: `pad`'s fill is an ordinary [05-OP-49] value operand, a
//! scalar of exactly the operand dtype, literal or computed at run time.
//!
//! A non-literal fill lowers as
//! `where(pad(false_like(x), P, true), broadcast(fill), pad(x, P, 0))`, so:
//!
//! * every lane places the fill's runtime bits in the padded cells, including
//!   `chelis build --target c` with no `grad` (it used to refuse the fill);
//! * `x` receives `shrink(g, inverse_padding)` whatever the fill;
//! * `fill` receives `g` with every moved-`x` cell replaced by exact +0,
//!   summed highest axis first at the default accumulator, each tree
//!   finalizing at the operand dtype.
//!
//! Every positive case runs on the `eval` lane and on a built C executable,
//! and the two must print identical lines. The order locks pin the masked
//! tree against the alternatives the spec rejects: a padded-only tree, a flat
//! row-major tree, and JAX's difference of totals.

#[path = "common/mod.rs"]
mod common;

use assert_cmd::Command;
use common::{build_and_run, gcc_available, write_file};
use serde_json::Value;
use tempfile::tempdir;

/// Run `chelis eval --file` and return stdout, failing on a nonzero exit.
fn eval_stdout(source: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("pad_fill.ch");
    write_file(&path, source);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    assert!(
        out.status.success(),
        "chelis eval failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("utf-8 stdout")
}

/// The printed value of top-level binding `name`.
fn printed<'a>(stdout: &'a str, name: &str) -> &'a str {
    let prefix = format!("{name} = ");
    stdout
        .lines()
        .find_map(|line| line.strip_prefix(prefix.as_str()))
        .unwrap_or_else(|| panic!("no `{name}` line in:\n{stdout}"))
}

/// Assert every `(binding, printed value)` pair on the eval lane and, when a
/// host C compiler exists, on the built executable. Printed floats are
/// shortest round-trip spellings, so equal text is equal bits.
fn assert_lanes(source: &str, stem: &str, expected: &[(&str, &str)]) {
    let eval = eval_stdout(source);
    for (name, value) in expected {
        assert_eq!(printed(&eval, name), *value, "eval `{name}`");
    }
    if !gcc_available() {
        eprintln!("skipping the C lane: no host C compiler");
        return;
    }
    let native = build_and_run(source, stem);
    for (name, value) in expected {
        assert_eq!(printed(&native, name), *value, "C `{name}`");
    }
}

/// `chelis check` error messages for `source`.
fn check_errors(source: &str) -> Vec<String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("pad_fill.ch");
    write_file(&path, source);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("chelis check should run");
    let report: Value = serde_json::from_slice(&out.stdout).expect("check emits JSON");
    report["errors"]
        .as_array()
        .expect("errors array")
        .iter()
        .map(|error| error.to_string())
        .collect()
}

const PADDED_SUM: &str = "def padded_sum(x: tensor[2, f32], fill: f32) -> f32 = tensor_to_scalar(sum(mul(pad(x, [[0i64, 1i64]], fill), to_tensor([1.0f32, 2.0f32, 3.0f32])), 0i32))\n";

#[test]
fn issue_reproducer_differentiates_a_runtime_fill_in_both_lanes() {
    let source = format!(
        "{PADDED_SUM}\
         def padded(x: tensor[2, f32], fill: f32) -> tensor[3, f32] = pad(x, [[0i64, 1i64]], fill)\n\
         def through(x: tensor[2, f32]) -> f32 = padded_sum(x, 0.5f32)\n\
         padded_value = padded(to_tensor([1.0f32, 1.0f32]), 0.5f32)\n\
         forward = padded_sum(to_tensor([1.0f32, 1.0f32]), 0.5f32)\n\
         dx = grad(padded_sum, wrt=x)(to_tensor([1.0f32, 1.0f32]), 0.5f32)\n\
         dfill = grad(padded_sum, wrt=fill)(to_tensor([1.0f32, 1.0f32]), 0.5f32)\n\
         joint = grad(padded_sum)(to_tensor([1.0f32, 1.0f32]), 0.5f32)\n\
         dthrough = grad(through)(to_tensor([1.0f32, 1.0f32]))\n"
    );
    assert_lanes(
        &source,
        "reproducer",
        &[
            // The forward pass with no grad: the C lane used to refuse it.
            ("padded_value", "tensor(shape=[3], data=[1.0, 1.0, 0.5])"),
            ("forward", "4.5"),
            ("dx", "tensor(shape=[2], data=[1.0, 2.0])"),
            ("dfill", "3.0"),
            ("joint.0", "tensor(shape=[2], data=[1.0, 2.0])"),
            ("joint.1", "3.0"),
            // A literal passed through the fill parameter.
            ("dthrough", "tensor(shape=[2], data=[1.0, 2.0])"),
        ],
    );
}

/// Each loss is written twice: with the `pad` primitive and with the
/// hand-written composition the primitive lowers to. Both must print the
/// same bits, and those bits are the masked tree's.
#[test]
fn fill_cotangent_is_the_masked_tree_and_matches_the_composition() {
    let cases = [
        // [a, 0, b, c] masked tree gives 1; a padded-only tree gives 0.
        (
            "order1",
            "x: tensor[1, f32]",
            "[[1i64, 2i64]]",
            "insert(scalar_to_tensor(fill), 0i32, 4i64)",
            "to_tensor([1.0f32, 5.0f32, 100000000.0f32, -100000000.0f32])",
            "tensor_to_scalar(sum(mul(PADDED, WEIGHTS), 0i32))",
            "to_tensor([0.0f32])",
        ),
        // Axis 1 then axis 0 gives 0; a flat row-major tree gives 2.
        (
            "order2",
            "x: tensor[1, 1, f32]",
            "[[1i64, 1i64], [1i64, 1i64]]",
            "insert(insert(scalar_to_tensor(fill), 0i32, 3i64), 1i32, 3i64)",
            "reshape(to_tensor([1.0f32, 1.0f32, 100000000.0f32, -100000000.0f32, 7.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32]), [3i64, 3i64])",
            "tensor_to_scalar(sum(sum(mul(PADDED, WEIGHTS), 1i32), 0i32))",
            "reshape(to_tensor([0.0f32]), [1i64, 1i64])",
        ),
        // Interior cotangents never enter: the exact 2, where a difference of
        // totals gives 0.
        (
            "cancel",
            "x: tensor[2, f32]",
            "[[0i64, 2i64]]",
            "insert(scalar_to_tensor(fill), 0i32, 4i64)",
            "to_tensor([100000000.0f32, 100000000.0f32, 1.0f32, 1.0f32])",
            "tensor_to_scalar(sum(mul(PADDED, WEIGHTS), 0i32))",
            "to_tensor([0.0f32, 0.0f32])",
        ),
        // An interior inf stays out of the fill's cotangent.
        (
            "interior_inf",
            "x: tensor[2, f32]",
            "[[0i64, 2i64]]",
            "insert(scalar_to_tensor(fill), 0i32, 4i64)",
            "mul(exp(to_tensor([100.0f32, 0.0f32, 0.0f32, 0.0f32])), to_tensor([1.0f32, 0.0f32, 1.0f32, 1.0f32]))",
            "tensor_to_scalar(sum(mul(PADDED, WEIGHTS), 0i32))",
            "to_tensor([0.0f32, 0.0f32])",
        ),
        // A NaN in a padded cell propagates.
        (
            "padded_nan",
            "x: tensor[2, f32]",
            "[[0i64, 2i64]]",
            "insert(scalar_to_tensor(fill), 0i32, 4i64)",
            "sqrt(to_tensor([1.0f32, 1.0f32, -1.0f32, 1.0f32]))",
            "tensor_to_scalar(sum(mul(PADDED, WEIGHTS), 0i32))",
            "to_tensor([0.0f32, 0.0f32])",
        ),
        // Rank 2 with padding on both sides of different axes: 0+1+2+5+8.
        (
            "rank2",
            "x: tensor[2, 2, f32]",
            "[[1i64, 0i64], [0i64, 1i64]]",
            "insert(insert(scalar_to_tensor(fill), 0i32, 3i64), 1i32, 3i64)",
            "reshape(to_tensor([0.0f32, 1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32]), [3i64, 3i64])",
            "tensor_to_scalar(sum(sum(mul(PADDED, WEIGHTS), 1i32), 0i32))",
            "reshape(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]), [2i64, 2i64])",
        ),
    ];
    let mut source = String::new();
    for (name, param, padding, broadcast, weights, body, x) in cases {
        let primitive = format!("pad(x, {padding}, fill)");
        let composed = format!(
            "where(pad(lt(x, x), {padding}, true), {broadcast}, pad(x, {padding}, 0.0f32))"
        );
        for (suffix, padded) in [("prim", primitive), ("comp", composed)] {
            let body = body.replace("PADDED", &padded).replace("WEIGHTS", weights);
            source.push_str(&format!(
                "def {name}_{suffix}({param}, fill: f32) -> f32 = {body}\n\
                 {name}_{suffix}_grad = grad({name}_{suffix})({x}, 0.0f32)\n"
            ));
        }
    }
    let expected = [
        ("order1", "tensor(shape=[1], data=[5.0])", "1.0"),
        ("order2", "tensor(shape=[1, 1], data=[7.0])", "0.0"),
        (
            "cancel",
            "tensor(shape=[2], data=[100000000.0, 100000000.0])",
            "2.0",
        ),
        ("interior_inf", "tensor(shape=[2], data=[inf, 0.0])", "2.0"),
        ("padded_nan", "tensor(shape=[2], data=[1.0, 1.0])", "NaN"),
        (
            "rank2",
            "tensor(shape=[2, 2], data=[3.0, 4.0, 6.0, 7.0])",
            "16.0",
        ),
    ];
    let mut lines = Vec::new();
    for (name, dx, dfill) in expected {
        for suffix in ["prim", "comp"] {
            lines.push((format!("{name}_{suffix}_grad.0"), dx));
            lines.push((format!("{name}_{suffix}_grad.1"), dfill));
        }
    }
    let lines = lines
        .iter()
        .map(|(name, value)| (name.as_str(), *value))
        .collect::<Vec<_>>();
    assert_lanes(&source, "order_locks", &lines);
}

#[test]
fn fill_cotangent_is_exact_positive_zero_and_accumulates_f16_at_f32() {
    let source = "def unpadded(x: tensor[2, f32], fill: f32) -> f32 = tensor_to_scalar(sum(mul(pad(x, [[0i64, 0i64]], fill), to_tensor([-1.0f32, -2.0f32])), 0i32))\n\
         def half(x: tensor[1, f16], fill: f16) -> f16 = tensor_to_scalar(sum(mul(pad(x, [[2i64, 1i64]], fill), to_tensor([2048.0f16, 1.0f16, 3.0f16, 1.0f16])), 0i32))\n\
         zero_grad = grad(unpadded, wrt=fill)(to_tensor([1.0f32, 1.0f32]), 0.5f32)\n\
         zero_sign = div(1.0f32, grad(unpadded, wrt=fill)(to_tensor([1.0f32, 1.0f32]), 0.5f32))\n\
         half_grad = grad(half, wrt=fill)(to_tensor([0.0f16]), 0.0f16)\n";
    assert_lanes(
        source,
        "zero_and_half",
        &[
            // Zero padding: every cell holds a moved `x`, so the fill's
            // cotangent is exact +0 even though every `g` is negative.
            ("zero_grad", "0.0"),
            ("zero_sign", "inf"),
            // [2048, 1, +0, 1]: f32 accumulation gives 2050; an f16 running
            // sum rounds 2049 to 2048 twice.
            ("half_grad", "2050.0"),
        ],
    );
}

#[test]
fn a_batched_fill_pads_and_differentiates_under_vmap() {
    let source = "def row(x: tensor[2, f32], fill: tensor[f32]) -> tensor[3, f32] = pad(x, [[0i64, 1i64]], tensor_to_scalar(fill))\n\
         def row_loss(x: tensor[2, f32], fill: tensor[f32]) -> f32 = tensor_to_scalar(sum(mul(row(x, fill), to_tensor([1.0f32, 2.0f32, 3.0f32])), 0i32))\n\
         batched = vmap(row)(to_tensor([[1.0f32, 1.0f32], [2.0f32, 2.0f32]]), to_tensor([0.5f32, -3.0f32]))\n\
         batched_grad = vmap(grad(row_loss, wrt=fill))(to_tensor([[1.0f32, 1.0f32], [2.0f32, 2.0f32]]), to_tensor([0.5f32, -3.0f32]))\n";
    assert_lanes(
        source,
        "batched",
        &[
            (
                "batched",
                "tensor(shape=[2, 3], data=[1.0, 1.0, 0.5, 2.0, 2.0, -3.0])",
            ),
            ("batched_grad", "tensor(shape=[2], data=[3.0, 3.0])"),
        ],
    );
    // Negative: only the fill is a value operand. A padding amount read from
    // batched elements is still a batch-varying extent.
    let errors = check_errors(
        "def grow[n, m](x: tensor[n, f32], k: tensor[i64]) -> tensor[m, f32] = pad(x, [[0i64, tensor_to_scalar(k)]], 0.0f32)\n\
         out = vmap(grow)(to_tensor([[1.0f32, 1.0f32], [2.0f32, 2.0f32]]), to_tensor([1i64, 2i64]))\n",
    );
    assert!(
        errors
            .iter()
            .any(|error| error.contains("batch_varying_extent")
                && error.contains("pad")
                && error.contains("vmapped argument 'k'")),
        "a batched padding amount must stay rejected: {errors:?}"
    );
}

/// A padded axis whose extent comes from a parameter: the `[n]` operand and a
/// runtime padding amount `k`, with respect to `x`, the fill, and both.
#[test]
fn symbolic_extents_differentiate_in_both_lanes() {
    let source = "def sym1[n](x: tensor[n, f32], f: f32) -> f32 = tensor_to_scalar(sum(pad(x, [[1i64, 3i64]], f), 0i32))\n\
         def dyn1[n](x: tensor[n, f32], k: i64, f: f32) -> f32 = tensor_to_scalar(sum(pad(x, [[1i64, k]], f), 0i32))\n\
         sym_dx = grad(sym1, wrt=x)(to_tensor([1.0f32, 2.0f32]), 0.5f32)\n\
         sym_dfill = grad(sym1, wrt=f)(to_tensor([1.0f32, 2.0f32]), 0.5f32)\n\
         sym_joint = grad(sym1)(to_tensor([1.0f32, 2.0f32]), 0.5f32)\n\
         dyn_dx = grad(dyn1, wrt=x)(to_tensor([1.0f32, 2.0f32]), 3i64, 0.5f32)\n\
         dyn_dfill = grad(dyn1, wrt=f)(to_tensor([1.0f32, 2.0f32]), 3i64, 0.5f32)\n\
         dyn_joint = grad(dyn1, wrt=(x, f))(to_tensor([1.0f32, 2.0f32]), 3i64, 0.5f32)\n";
    let ones = "tensor(shape=[2], data=[1.0, 1.0])";
    assert_lanes(
        source,
        "symbolic",
        &[
            ("sym_dx", ones),
            ("sym_dfill", "4.0"),
            ("sym_joint.0", ones),
            ("sym_joint.1", "4.0"),
            ("dyn_dx", ones),
            ("dyn_dfill", "4.0"),
            ("dyn_joint.0", ones),
            ("dyn_joint.1", "4.0"),
        ],
    );
}

/// The issue's attention-with-sink shape, `[q, k]` scores padded with a
/// learned sink column. The C lane does not yet build these gradients: the
/// sink's leaves a runtime extent undeclared (chelis#3412), and the scores'
/// stops at a rank-2 shrink emission panic a literal fill reaches too
/// (chelis#3511).
#[test]
fn a_rank_two_symbolic_sink_differentiates_on_eval() {
    let source = "def att[q, k](s: tensor[q, k, f32], sink: f32) -> f32 = tensor_to_scalar(sum(sum(pad(s, [[0i64, 0i64], [0i64, 1i64]], sink), 1i32), 0i32))\n\
         ds = grad(att, wrt=s)(reshape(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32]), [2i64, 3i64]), 0.5f32)\n\
         dsink = grad(att, wrt=sink)(reshape(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32]), [2i64, 3i64]), 0.5f32)\n\
         joint = grad(att)(reshape(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32]), [2i64, 3i64]), 0.5f32)\n";
    let ones = "tensor(shape=[2, 3], data=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0])";
    let eval = eval_stdout(source);
    for (name, value) in [
        ("ds", ones),
        ("dsink", "2.0"),
        ("joint.0", ones),
        ("joint.1", "2.0"),
    ] {
        assert_eq!(printed(&eval, name), value, "eval `{name}`");
    }
}

/// The cause was not the runtime fill: a hand-written Bool mask padded after
/// the operand made the Bool cotangent's `Shrink` the first operation to
/// produce the parameter's extent `n`, and the backward graph kept that
/// never-read Bool chain alive as `n`'s declarer.
#[test]
fn a_hand_written_mask_over_a_parameter_extent_differentiates() {
    let source = "def masked[n](x: tensor[n, f32]) -> f32 = {\n    p = pad(x, [[1i64, 3i64]], 0.0f32)\n    q = pad(mul(x, x), [[1i64, 3i64]], 0.5f32)\n    mask = pad(lt(x, x), [[1i64, 3i64]], true)\n    tensor_to_scalar(sum(where(mask, q, p), 0i32))\n}\n\
         dx = grad(masked)(to_tensor([1.0f32, 2.0f32]))\n";
    assert_lanes(
        source,
        "masked",
        &[("dx", "tensor(shape=[2], data=[1.0, 1.0])")],
    );
}

#[test]
fn grad_of_grad_differentiates_the_fill_cotangent() {
    // sum(w * pad(x, P, fill)^2) over the one padded cell is 3 * fill^2:
    // first derivative 6 * fill = 3 at 0.5, second derivative 6.
    let source = "def curvature(fill: f32) -> f32 = {\n    p = pad(to_tensor([1.0f32, 1.0f32]), [[0i64, 1i64]], fill)\n    tensor_to_scalar(sum(mul(mul(p, p), to_tensor([1.0f32, 2.0f32, 3.0f32])), 0i32))\n}\n\
         def slope(fill: f32) -> f32 = grad(curvature)(fill)\n\
         first = slope(0.5f32)\n\
         second = grad(slope)(0.5f32)\n";
    assert_lanes(source, "second", &[("first", "3.0"), ("second", "6.0")]);
}

#[test]
fn an_integer_runtime_fill_is_forward_only() {
    let int_pad = "def int_pad(x: tensor[3, f32], n: i32) -> f32 = tensor_to_scalar(sum(mul(cast(pad(to_tensor([1i32, 2i32]), [[0i64, 1i64]], n), f32), x), 0i32))\n";
    // Positive: the integer fill pads at run time, and `x` differentiates.
    assert_lanes(
        &format!("{int_pad}dx = grad(int_pad, wrt=x)(to_tensor([1.0f32, 1.0f32, 1.0f32]), 3i32)\n"),
        "int_fill",
        &[("dx", "tensor(shape=[3], data=[1.0, 2.0, 3.0])")],
    );
    // Negative: an integer fill is not a differentiation target.
    let errors = check_errors(&format!(
        "{int_pad}bad = grad(int_pad, wrt=n)(to_tensor([1.0f32, 1.0f32, 1.0f32]), 3i32)\n"
    ));
    assert!(
        errors
            .iter()
            .any(|error| error.contains("grad `wrt` index 1 is not differentiable")),
        "an integer `wrt` fill must be non_differentiable: {errors:?}"
    );
}

#[test]
fn a_fill_of_another_dtype_is_a_type_error() {
    for (fill_ty, argument) in [("f64", "0.5f64"), ("i32", "1i32"), ("f16", "0.5f16")] {
        let errors = check_errors(&format!(
            "def padded(x: tensor[2, f32], fill: {fill_ty}) -> tensor[3, f32] = pad(x, [[0i64, 1i64]], fill)\n\
             out = padded(to_tensor([1.0f32, 1.0f32]), {argument})\n"
        ));
        assert!(
            errors
                .iter()
                .any(|error| error.contains("pad argument 3 (fill)")),
            "a {fill_ty} fill for an f32 pad must be a type error: {errors:?}"
        );
    }
}
