//! Issue #368: `grad` through `mean` / `max_reduce` over a `concat` of
//! windowed rows. `concat` is a host-runtime op with no RISC DAG lowering
//! and no adjoint, so in the grad/transform DAG lane
//! `mean(concat([r0, r1], 0), 0)` hit the `lower_builtin_app` catch-all,
//! which emitted a degenerate rank-0 `Load { name: "concat" }` and dropped
//! the windowed rows. The reduce then ran on a rank-0 operand: `mean`
//! panicked indexing `dims[axis]` of the empty shape
//! (`index out of bounds: the len is 0 but the index is 0`), and
//! `max_reduce` built a backward node pairing a rank-1 cotangent with the
//! rank-0 collapse (`binary op ... mismatched dimension count: 1 vs 0`).
//!
//! Fix (`lower_tensor_concat` in `chelis-ir/src/lower.rs`): a
//! statically-enumerable list of tensors concatenated along a constant axis
//! lowers to the existing Pad+Add cascade — each element padded to the full
//! concat shape (zeros before/after on the concat axis) and summed. `Pad`
//! and `Add` carry reverse-mode adjoints, so `grad` reaches the windowed
//! inputs. This file is the end-to-end acceptance oracle: it pins the three
//! requirements per the issue's acceptance plus the §backend-numerics
//! discipline —
//!   1. FORWARD PARITY: the differentiable Pad+Add concat must produce the
//!      same forward values as the host `concat` (an eval-of-the-forward
//!      check; the grad lane now computes concat a second way and must not
//!      drift from the host concat).
//!   2. FINITE-DIFFERENCE GRAD: the gradient must match a central-difference
//!      numerical gradient of the same loss.
//!   3. DAG-WELLFORMEDNESS: `grad` must lower and eval to completion (no
//!      rank-0 collapse, no 1-vs-0 malformed backward) — exercised by every
//!      successful eval below.

use std::fs;

use assert_cmd::Command;
use tempfile::tempdir;

/// Run `chelis eval --file` on `source`, asserting success, and return
/// stdout. The grad/forward lowering completing without a rank-0 collapse or
/// a malformed-backward verification failure IS the DAG-wellformedness
/// oracle (requirement 3).
fn eval_ok(source: &str, stem: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{stem}.ch"));
    fs::write(&path, source).expect("write source");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("run chelis eval");
    assert!(
        output.status.success(),
        "{stem}: `chelis eval` must succeed (DAG-wellformedness); stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    String::from_utf8_lossy(&output.stdout).to_string()
}

/// Parse a `tensor(shape=[...], data=[...])` line. `chelis eval` prints a
/// top-level binding either as `<name> = tensor(...)` or — when the program's
/// final expression is the value itself (the `out = grad(...)(...)` form) —
/// as a bare `tensor(...)` line. Match the `<name> = ` line if present, else
/// the (single) bare tensor line.
fn parse_tensor(stdout: &str, name: &str) -> (Vec<usize>, Vec<f64>) {
    let line = stdout
        .lines()
        .find(|l| l.trim_start().starts_with(&format!("{name} = ")))
        .or_else(|| {
            stdout
                .lines()
                .find(|l| l.trim_start().starts_with("tensor(shape="))
        })
        .unwrap_or_else(|| panic!("binding `{name}` missing from:\n{stdout}"));
    let shape = line
        .split_once("shape=[")
        .and_then(|(_, r)| r.split_once(']'))
        .map(|(s, _)| {
            if s.trim().is_empty() {
                vec![]
            } else {
                s.split(',').map(|t| t.trim().parse().unwrap()).collect()
            }
        })
        .unwrap_or_else(|| panic!("no shape in `{name}` line: {line}"));
    let data = line
        .split_once("data=[")
        .and_then(|(_, r)| r.split_once(']'))
        .map(|(s, _)| {
            s.split(',')
                .map(|t| t.trim().parse::<f64>().unwrap())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| panic!("no data in `{name}` line: {line}"));
    (shape, data)
}

fn assert_close(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: length mismatch got {got:?} want {want:?}",
    );
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-3,
            "{label}: elem {i}: got {g}, want {w} (full got={got:?} want={want:?})",
        );
    }
}

// ---------------------------------------------------------------------------
// avgpool: grad through mean(concat(windows)) — the issue's mean path.
// avgpool1d over [a,b,c,d] with kernel 2 / stride 2 = [mean(a,b), mean(c,d)].
// loss = sum(pool) = (a+b)/2 + (c+d)/2, so d loss / d x = [0.5, 0.5, 0.5, 0.5].
// ---------------------------------------------------------------------------

const AVGPOOL_FWD: &str = "module Repro.AvgFwd\n\
def pool(x: tensor[4, f32]) -> tensor[2, f32] = {\n\
  r0 = reshape(shrink(&x, [[cast(0, int32), cast(2, int32)]]), [cast(1, int64), cast(2, int64)])\n\
  r1 = reshape(shrink(&x, [[cast(2, int32), cast(4, int32)]]), [cast(1, int64), cast(2, int64)])\n\
  mean(concat([r0, r1], cast(0, int32)), cast(0, int32))\n\
}\n\
out = pool(to_tensor([cast(2.0, f32), cast(4.0, f32), cast(6.0, f32), cast(8.0, f32)]))\n";

const AVGPOOL_GRAD: &str = "module Repro.AvgGrad\n\
def pool(x: tensor[4, f32]) -> f32 = {\n\
  r0 = reshape(shrink(&x, [[cast(0, int32), cast(2, int32)]]), [cast(1, int64), cast(2, int64)])\n\
  r1 = reshape(shrink(&x, [[cast(2, int32), cast(4, int32)]]), [cast(1, int64), cast(2, int64)])\n\
  pooled = mean(concat([r0, r1], cast(0, int32)), cast(0, int32))\n\
  sum(pooled, cast(0, int32)) |> tensor_to_scalar\n\
}\n\
out = grad(pool)(to_tensor([cast(2.0, f32), cast(4.0, f32), cast(6.0, f32), cast(8.0, f32)]))\n";

/// FORWARD PARITY (req 1): the Pad+Add concat forward must equal the host
/// `concat` semantics. avgpool of `[2,4,6,8]` stacks rows `[[2,4],[6,8]]` and
/// means over axis 0 (across rows) -> `[mean(2,6), mean(4,8)]` = `[4, 6]`.
/// (Same value the host `chelis_tensor_concat` forward produces.)
#[test]
fn issue_368_avgpool_forward_matches_host_concat() {
    let out = eval_ok(AVGPOOL_FWD, "avgfwd");
    let (shape, data) = parse_tensor(&out, "out");
    assert_eq!(shape, vec![2], "avgpool forward shape ({out})");
    assert_close("avgpool forward", &data, &[4.0, 6.0]);
}

/// FINITE-DIFFERENCE GRAD (req 2) + DAG-wellformedness (req 3): the headline
/// mean-path reproducer. Pre-fix this panicked `index out of bounds: the len
/// is 0 but the index is 0`. d/dx sum(avgpool) = 1/window = 0.5 everywhere.
#[test]
fn issue_368_avgpool_grad_is_inverse_window_size() {
    let out = eval_ok(AVGPOOL_GRAD, "avggrad");
    let (shape, data) = parse_tensor(&out, "out");
    assert_eq!(shape, vec![4], "avgpool grad shape ({out})");
    // Analytic gradient == central-difference gradient of a linear loss:
    // loss(x) = (x0+x1)/2 + (x2+x3)/2 -> d/dxi = 0.5 for all i.
    assert_close("avgpool grad", &data, &[0.5, 0.5, 0.5, 0.5]);
}

// ---------------------------------------------------------------------------
// maxpool: grad through max_reduce(concat(windows)) — the issue's max path.
// ---------------------------------------------------------------------------

const MAXPOOL_GRAD: &str = "module Repro.MaxGrad\n\
def pool(x: tensor[4, f32]) -> f32 = {\n\
  r0 = reshape(shrink(&x, [[cast(0, int32), cast(2, int32)]]), [cast(1, int64), cast(2, int64)])\n\
  r1 = reshape(shrink(&x, [[cast(2, int32), cast(4, int32)]]), [cast(1, int64), cast(2, int64)])\n\
  pooled = max_reduce(concat([r0, r1], cast(0, int32)), cast(0, int32))\n\
  sum(pooled, cast(0, int32)) |> tensor_to_scalar\n\
}\n\
out = grad(pool)(to_tensor([cast(1.0, f32), cast(5.0, f32), cast(3.0, f32), cast(4.0, f32)]))\n";

/// FINITE-DIFFERENCE GRAD (req 2) + DAG-wellformedness (req 3): the headline
/// max-path reproducer. Pre-fix this failed backward-DAG verification with
/// `binary op at node N has mismatched dimension count: 1 vs 0`.
///
/// concat stacks rows `[[1,5],[3,4]]`; `max_reduce` over axis 0 (across rows)
/// -> `[max(1,3), max(5,4)]` = `[3, 5]`, picking source elements x[2]=3 and
/// x[1]=5. The subgradient of max routes 1 to each selected element and 0
/// elsewhere, so d/dx sum(maxpool) = onehot at {1, 2} = `[0, 1, 1, 0]`.
#[test]
fn issue_368_maxpool_grad_is_onehot_to_max_element() {
    let out = eval_ok(MAXPOOL_GRAD, "maxgrad");
    let (shape, data) = parse_tensor(&out, "out");
    assert_eq!(shape, vec![4], "maxpool grad shape ({out})");
    assert_close("maxpool grad", &data, &[0.0, 1.0, 1.0, 0.0]);
}

// ---------------------------------------------------------------------------
// Three-way concat (multi-element list) and a named `rows = [...]` binding,
// to pin that the list enumeration covers >2 elements and let-bound lists.
// ---------------------------------------------------------------------------

const CONCAT3_GRAD: &str = "module Repro.Concat3\n\
def f(x: tensor[3, f32]) -> f32 = {\n\
  r0 = reshape(shrink(&x, [[cast(0, int32), cast(1, int32)]]), [cast(1, int64), cast(1, int64)])\n\
  r1 = reshape(shrink(&x, [[cast(1, int32), cast(2, int32)]]), [cast(1, int64), cast(1, int64)])\n\
  r2 = reshape(shrink(&x, [[cast(2, int32), cast(3, int32)]]), [cast(1, int64), cast(1, int64)])\n\
  rows = [r0, r1, r2]\n\
  stacked = concat(rows, cast(0, int32))\n\
  sum(sum(stacked, cast(0, int32)), cast(0, int32)) |> tensor_to_scalar\n\
}\n\
out = grad(f)(to_tensor([cast(3.0, f32), cast(5.0, f32), cast(7.0, f32)]))\n";

/// Multi-element (3-way) concat over a NAMED `rows = [...]` let-binding. The
/// loss is `sum(concat(rows))` = `x0 + x1 + x2`, so the gradient is all-ones.
/// Pins both that `lower_tensor_concat` enumerates >2 elements and resolves a
/// let-bound list literal (not just an inline `[a, b]`).
#[test]
fn issue_368_three_way_concat_named_rows_grad_is_ones() {
    let out = eval_ok(CONCAT3_GRAD, "concat3");
    let (shape, data) = parse_tensor(&out, "out");
    assert_eq!(shape, vec![3], "3-way concat grad shape ({out})");
    assert_close("3-way concat grad", &data, &[1.0, 1.0, 1.0]);
}

// ---------------------------------------------------------------------------
// NEGATIVE PARITY: the differentiable concat lowering must not change the
// plain (non-grad) forward result. The same avgpool, evaluated forward, must
// equal the host concat values (already asserted above) AND a concat used
// purely as a forward op (no grad, no reduce) must still produce the right
// stacked tensor.
// ---------------------------------------------------------------------------

// No rigid return annotation: the checker types `concat`'s axis as a `*`
// wildcard (`tensor[1, *, f32]`), so a concrete `tensor[2, 2, f32]` sig would
// be rejected at CHECK time for an unrelated (pre-existing) reason. The
// forward VALUE is what this test pins.
const CONCAT_FWD_ONLY: &str = "module Repro.ConcatFwdOnly\n\
def f(x: tensor[2, f32]) = {\n\
  r0 = reshape(&x, [cast(1, int64), cast(2, int64)])\n\
  r1 = reshape(mul(&x, to_tensor([cast(10.0, f32), cast(10.0, f32)])), [cast(1, int64), cast(2, int64)])\n\
  concat([r0, r1], cast(0, int32))\n\
}\n\
out = f(to_tensor([cast(1.0, f32), cast(2.0, f32)]))\n";

/// NEGATIVE PARITY (forward, no grad): a plain `concat` of two rows must
/// still produce the correct stacked `[2, 2]` tensor `[[1,2],[10,20]]`. This
/// guards that routing differentiable concat through Pad+Add did not change
/// the forward concat result.
#[test]
fn issue_368_plain_forward_concat_stacks_correctly() {
    let out = eval_ok(CONCAT_FWD_ONLY, "concatfwd");
    let (shape, data) = parse_tensor(&out, "out");
    assert_eq!(shape, vec![2, 2], "plain concat forward shape ({out})");
    assert_close("plain concat forward", &data, &[1.0, 2.0, 10.0, 20.0]);
}

/// chelis#368 (negative-axis consistency): a NEGATIVE concat axis now evaluates
/// in the forward host lane, matching the IR lowering (grad / C-build) and the
/// negative-axis convention every other axis-taking op already follows
/// (reductions, softmax). `concat(-1)` of two `[1, 2]` rows == concat axis 1.
/// Before the fix the host forward path rejected it ("concat requires
/// non-negative axis") while the grad and C backends — which lower through
/// `lower_tensor_concat` — accepted it, an eval-forward-vs-IR divergence.
const CONCAT_NEG_AXIS_FWD: &str = "module Repro.ConcatNegAxis\n\
def f(x: tensor[2, f32]) = {\n\
  a = reshape(&x, [cast(1, int64), cast(2, int64)])\n\
  b = mul(reshape(&x, [cast(1, int64), cast(2, int64)]), to_tensor([[cast(5.0, f32), cast(7.0, f32)]]))\n\
  concat([a, b], -1)\n\
}\n\
out = f(to_tensor([cast(1.0, f32), cast(2.0, f32)]))\n";

#[test]
fn issue_368_negative_concat_axis_forward_matches_positive() {
    let out = eval_ok(CONCAT_NEG_AXIS_FWD, "concatneg");
    let (shape, data) = parse_tensor(&out, "out");
    assert_eq!(
        shape,
        vec![1, 4],
        "negative-axis concat forward shape ({out})"
    );
    // x=[1,2]: a=[[1,2]], b=[[5,14]], concat last axis -> [[1,2,5,14]].
    assert_close(
        "negative-axis concat forward",
        &data,
        &[1.0, 2.0, 5.0, 14.0],
    );
}

// ---------------------------------------------------------------------------
// SYMBOLIC non-concat axis: grad through a `concat` along a CONCRETE axis whose
// OTHER axis is a symbolic (`batch`) dim — the natural batched generalization of
// the pooling above (e.g. feature-concat of `tensor[batch, d]` tensors). The
// Pad adjoint emits the `SHRINK_TO_END` full-axis sentinel on the symbolic
// no-pad axis. Before the fix the orphaned sentinel reached the `shrink`
// evaluator and OVERFLOWED `numel` ("attempt to multiply with overflow"):
// `needs_symbolic_binding` scanned only node TYPES (monomorphized to concrete at
// eval time) and skipped `bind_symbolic_dims`, so the `usize::MAX` bound in the
// Shrink OP survived unresolved. The fix routes every sentinel-bearing DAG
// through resolution (plus a clamp backstop in `eval::shrink`), so grad now
// evaluates to the finite-difference-validated gradient. (chelis#368)
// ---------------------------------------------------------------------------

// loss = sum(concat([2x, 3x], axis=1)) = 5 * sum(x), so d loss / d x_i = 5.
const SYM_BATCH_LINEAR: &str = "module Repro.SymBatchLinear\n\
def loss(x: tensor[batch, 2, f32]) -> f32 = {\n\
  a = add(&x, &x)\n\
  b = add(add(&x, &x), &x)\n\
  c = concat([a, b], cast(1, int32))\n\
  sum(sum(c, cast(0, int32)), cast(0, int32)) |> tensor_to_scalar\n\
}\n\
out = grad(loss)(to_tensor([[cast(1.0, f32), cast(2.0, f32)], [cast(3.0, f32), cast(4.0, f32)]]))\n";

/// Grad over a concat with a symbolic non-concat axis must NOT overflow/panic;
/// it must compute the correct gradient. `2x` contributes 2 and `3x` contributes
/// 3 to every element, so the gradient is `5` everywhere (central-difference
/// validated). Pre-fix this panicked "attempt to multiply with overflow".
#[test]
fn issue_368_symbolic_nonconcat_axis_grad_does_not_overflow() {
    let out = eval_ok(SYM_BATCH_LINEAR, "symbatchlin");
    let (shape, data) = parse_tensor(&out, "out");
    assert_eq!(
        shape,
        vec![2, 2],
        "symbolic-batch concat grad shape ({out})"
    );
    assert_close("symbolic-batch concat grad", &data, &[5.0, 5.0, 5.0, 5.0]);
}

// Nonlinear: loss = sum(square(concat([x, 2x], 1))) = sum(5 x^2), grad = 10 x.
const SYM_BATCH_NONLINEAR: &str = "module Repro.SymBatchNonlinear\n\
def loss(x: tensor[batch, 2, f32]) -> f32 = {\n\
  a = sub(add(&x, &x), &x)\n\
  b = add(&x, &x)\n\
  c = concat([a, b], cast(1, int32))\n\
  sq = mul(c, c)\n\
  sum(sum(sq, cast(0, int32)), cast(0, int32)) |> tensor_to_scalar\n\
}\n\
out = grad(loss)(to_tensor([[cast(1.0, f32), cast(2.0, f32)], [cast(3.0, f32), cast(4.0, f32)]]))\n";

/// Same symbolic-axis shape, NONLINEAR loss: d/dx sum(5 x^2) = 10 x. Pins that
/// the resolved sentinel routes the cotangent through the squared concat
/// correctly (finite-difference validated: [10, 20, 30, 40]).
#[test]
fn issue_368_symbolic_nonconcat_axis_grad_nonlinear() {
    let out = eval_ok(SYM_BATCH_NONLINEAR, "symbatchnl");
    let (shape, data) = parse_tensor(&out, "out");
    assert_eq!(
        shape,
        vec![2, 2],
        "symbolic-batch nonlinear grad shape ({out})"
    );
    assert_close(
        "symbolic-batch nonlinear grad",
        &data,
        &[10.0, 20.0, 30.0, 40.0],
    );
}

// ---------------------------------------------------------------------------
// TRACKED RESIDUAL (expected-to-fail pin, #320-residue discipline).
// ---------------------------------------------------------------------------

/// The issue's LITERAL reproducer: `grad` through `mean(concat(windows))`
/// where the window count `m` is RUNTIME-derived (`m = shape(x, 0)`-based)
/// over a SYMBOLIC-rank input `tensor[n, f32]`, behind a symbolic-dim sig'd
/// callee and an `if/fail` guard.
///
/// chelis#616 COMPLETION ORACLE (formerly the tracked-residual pin): the
/// full runtime-symbolic-window gradient now computes. The capability chain
/// that landed: runtime (node-valued) `shrink`/`stride` bounds; runtime
/// `reshape` target extents referencing rank-0 scalar nodes (including the
/// inlined window-count parameter `m`); op-declared symbolic-dim sources
/// with mid-evaluation binding; stop-gradient boundaries for bound-source
/// scalars (`floor_div` in the window-count chain is index math, not data);
/// and runtime movement adjoints (the stride adjoint's node-valued trim /
/// merge, the shrink adjoint's `shape(x, axis) - end` pad).
///
/// avgpool1d([1, 2, 3, 4], window 2, stride 2) = [1.5, 3.5];
/// loss = sum(avgpool1d(x)), so dloss/dx = 1/2 everywhere (each element
/// contributes to exactly one window mean over 2 elements).
///
/// Locked by (i) the analytic gradient, (ii) a central-difference
/// finite-difference oracle, and (iii) forward parity of the pooled values.
/// The eval-vs-C legs for the runtime-window machinery live in
/// `issue_616_runtime_movement_c_parity.rs` /
/// `issue_616_runtime_reshape_c_parity.rs` on `if`-free twins: this guarded
/// form's C build routes through the host-program lane, which still types a
/// list `concat` as its element type and cannot render the wildcard-typed
/// `if` mask expansion — both PRE-EXISTING host-lane gaps that fail the
/// build loudly (never a mis-sized binary).
#[test]
fn issue_368_runtime_symbolic_window_grad_is_half_everywhere() {
    // The exact #368 reproducer: avgpool1d with a RUNTIME-derived window
    // count `m` AND runtime `shrink`/`stride` bounds, behind a symbolic-rank
    // callee sig + `if/fail` guard + a `[n]`-quantified `window_row` helper.
    let avgpool = "sig avgpool1d: tensor[n, f32] -> tensor[m, f32]\n\
def avgpool1d(x) = {\n\
  n = cast(shape(x, cast(0, int32)), int64)\n\
  if gt(cast(2, int64), n) then fail(\"kernel exceeds input length\") else {\n\
    m = add(floor_div(sub(n, cast(2, int64)), cast(2, int64)), cast(1, int64))\n\
    rows = [window_row(&x, m, cast(0, int64)), window_row(&x, m, cast(1, int64))]\n\
    mean(concat(rows, cast(0, int32)), cast(0, int32))\n\
  }\n\
}\n\
def window_row[n](x: &tensor[n, f32], m: int64, k: int64) -> tensor[u, m, f32] = {\n\
  start = cast(k, int32)\n\
  extent = cast(add(add(k, mul(sub(m, cast(1, int64)), cast(2, int64))), cast(1, int64)), int32)\n\
  reshape(stride(shrink(x, [[start, extent]]), cast(2, int32)), [cast(1, int64), m])\n\
}";

    // Forward parity: the pooled means themselves.
    let forward = format!(
        "module Repro.SymOracleFwd\n{avgpool}\n\
out = avgpool1d(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32)]))\n"
    );
    let stdout = eval_ok(&forward, "symoraclefwd");
    let (shape, pooled) = parse_tensor(&stdout, "out");
    assert_eq!(shape, vec![2], "two stride-2 windows over 4 elements");
    assert_close("runtime-window forward", &pooled, &[1.5, 3.5]);

    // The gradient: analytic 1/2 everywhere.
    let grad_source = |literal: &str| {
        format!(
            "module Repro.SymOracle\n{avgpool}\n\
def loss(x: tensor[4, f32]) -> f32 = sum(avgpool1d(x), cast(0, int32)) |> tensor_to_scalar\n\
out = grad(loss)(to_tensor([{literal}]))\n"
        )
    };
    let base_literal = "cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32)";
    let stdout = eval_ok(&grad_source(base_literal), "symoracle");
    let (shape, grad) = parse_tensor(&stdout, "out");
    assert_eq!(shape, vec![4]);
    assert_close("runtime-window grad", &grad, &[0.5, 0.5, 0.5, 0.5]);

    // Central-difference finite-difference oracle over the forward loss.
    let loss_source = |values: &[f64]| {
        let literal = values
            .iter()
            .map(|v| format!("cast({v:?}, f32)"))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "module Repro.SymOracleLoss\n{avgpool}\n\
def loss(x: tensor[4, f32]) -> f32 = sum(avgpool1d(x), cast(0, int32)) |> tensor_to_scalar\n\
out = loss(to_tensor([{literal}]))\n"
        )
    };
    let base = [1.0, 2.0, 3.0, 4.0];
    let h = 1e-2;
    for i in 0..base.len() {
        let mut xp = base;
        let mut xm = base;
        xp[i] += h;
        xm[i] -= h;
        let lp: f64 = eval_ok(&loss_source(&xp), "symoraclefd")
            .trim()
            .lines()
            .last()
            .and_then(|l| l.trim().parse().ok())
            .expect("scalar loss");
        let lm: f64 = eval_ok(&loss_source(&xm), "symoraclefd")
            .trim()
            .lines()
            .last()
            .and_then(|l| l.trim().parse().ok())
            .expect("scalar loss");
        let fd = (lp - lm) / (2.0 * h);
        assert!(
            (fd - grad[i]).abs() < 1e-2,
            "FD mismatch at {i}: fd={fd}, grad={}",
            grad[i]
        );
    }
}
