//! Issue #368 (end-to-end CLI oracle): `grad` through the windowed-pool
//! idiom — `mean` / `max_reduce` over a `concat`'d window stack behind a
//! symbolic-dim callee with a `fail(...)` guard.
//!
//! On 0.10.0 these probes failed inside host-runtime `grad` lowering:
//!   * mean → PANIC "index out of bounds: the len is 0 but the index is 0".
//!   * max  → malformed backward DAG, "binary op ... mismatched dimension
//!     count: 1 vs 0".
//!
//! After the fix, the STATICALLY-bounded windowed pool grads correctly
//! end to end (concat lowered to Pad+Add, the mean collapse recovered, the
//! `if`/`fail` blend rank-reconciled). The RUNTIME-bounded window (a shrink
//! extent derived from `shape(x, ...)`) is not yet differentiable in the IR
//! lane and must produce a CLEAN, SPECIFIC rejection — never the old panic.
//!
//! The IR-level FD oracle is the sibling `issue_368_grad_windowed_pool.rs`
//! in `chelis-ir`.

use std::fs;

use assert_cmd::Command;
use tempfile::tempdir;

/// Static-bound windowed AVG pool behind the symbolic-dim callee + `fail`
/// guard. `loss(x) = sum(mean_rows([[x0,x1],[x2,x3]])) = (x0+x1+x2+x3)/2`,
/// so `df/dx = [0.5, 0.5, 0.5, 0.5]`.
const AVGPOOL_STATIC: &str = "module Repro.Avgpool368\n\
sig pool: tensor[n, f32] -> tensor[m, f32]\n\
def pool(x) = {\n\
  n = cast(shape(x, cast(0, int32)), int64)\n\
  if gt(cast(2, int64), n) then fail(\"kernel exceeds input length\") else {\n\
    rows = [row0(&x), row1(&x)]\n\
    mean(concat(rows, cast(0, int32)), cast(0, int32))\n\
  }\n\
}\n\
def row0[n](x: &tensor[n, f32]) -> tensor[u, m, f32] = {\n\
  reshape(shrink(x, [[cast(0, int32), cast(2, int32)]]), [cast(1, int64), cast(2, int64)])\n\
}\n\
def row1[n](x: &tensor[n, f32]) -> tensor[u, m, f32] = {\n\
  reshape(shrink(x, [[cast(2, int32), cast(4, int32)]]), [cast(1, int64), cast(2, int64)])\n\
}\n\
def loss(x: tensor[4, f32]) -> f32 = { sum(pool(x), cast(0, int32)) |> tensor_to_scalar }\n\
def df(x: tensor[4, f32]) -> tensor[4, f32] = grad(loss)(x)\n\
out = df(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32)]))\n";

/// Static-bound windowed MAX pool, same structure. With x = [1,2,3,4] the
/// per-column maxima are x2, x3, so `df/dx = [0, 0, 1, 1]`.
const MAXPOOL_STATIC: &str = "module Repro.Maxpool368\n\
sig pool: tensor[n, f32] -> tensor[m, f32]\n\
def pool(x) = {\n\
  n = cast(shape(x, cast(0, int32)), int64)\n\
  if gt(cast(2, int64), n) then fail(\"kernel exceeds input length\") else {\n\
    rows = [row0(&x), row1(&x)]\n\
    max_reduce(concat(rows, cast(0, int32)), cast(0, int32))\n\
  }\n\
}\n\
def row0[n](x: &tensor[n, f32]) -> tensor[u, m, f32] = {\n\
  reshape(shrink(x, [[cast(0, int32), cast(2, int32)]]), [cast(1, int64), cast(2, int64)])\n\
}\n\
def row1[n](x: &tensor[n, f32]) -> tensor[u, m, f32] = {\n\
  reshape(shrink(x, [[cast(2, int32), cast(4, int32)]]), [cast(1, int64), cast(2, int64)])\n\
}\n\
def loss(x: tensor[4, f32]) -> f32 = { sum(pool(x), cast(0, int32)) |> tensor_to_scalar }\n\
def df(x: tensor[4, f32]) -> tensor[4, f32] = grad(loss)(x)\n\
out = df(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32)]))\n";

/// RUNTIME-bounded window: the shrink extent is derived from a runtime `m`
/// (computed from `shape(x, 0)`), so the differentiable IR lane cannot lower
/// the `shrink` to static bounds. This is the issue's verbatim headline
/// reproducer. It must reject CLEANLY (not panic / not malformed DAG).
const AVGPOOL_RUNTIME: &str = "module Repro.AvgpoolRuntime368\n\
sig avgpool1d: tensor[n, f32] -> tensor[m, f32]\n\
def avgpool1d(x) = {\n\
  n = cast(shape(x, cast(0, int32)), int64)\n\
  if gt(cast(2, int64), n) then fail(\"kernel exceeds input length\") else {\n\
    m = add(div(sub(n, cast(2, int64)), cast(2, int64)), cast(1, int64))\n\
    rows = [window_row(&x, m, cast(0, int64)), window_row(&x, m, cast(1, int64))]\n\
    mean(concat(rows, cast(0, int32)), cast(0, int32))\n\
  }\n\
}\n\
def window_row[n](x: &tensor[n, f32], m: int64, k: int64) -> tensor[u, m, f32] = {\n\
  start = cast(k, int32)\n\
  extent = cast(add(add(k, mul(sub(m, cast(1, int64)), cast(2, int64))), cast(1, int64)), int32)\n\
  reshape(stride(shrink(x, [[start, extent]]), cast(2, int32)), [cast(1, int64), m])\n\
}\n\
def loss_avg(x: tensor[4, f32]) -> f32 = { sum(avgpool1d(x), cast(0, int32)) |> tensor_to_scalar }\n\
def df(x: tensor[4, f32]) -> tensor[4, f32] = grad(loss_avg)(x)\n\
out = df(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32)]))\n";

fn run_eval(source: &str, stem: &str) -> std::process::Output {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{stem}.ch"));
    fs::write(&path, source).expect("write source");
    Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("run chelis eval")
}

/// Assert `chelis eval` of `source` succeeds and prints a `tensor[4]` whose
/// data equals `want` (within float-print tolerance).
fn assert_eval_grad(source: &str, stem: &str, want: &[f64], label: &str) {
    let output = run_eval(source, stem);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "{label}: `chelis eval` must succeed; stdout={stdout} stderr={stderr}",
    );
    assert!(
        stdout.contains("shape=[4]"),
        "{label}: gradient must be tensor[4]; got stdout={stdout}",
    );
    // The renderer prints integral floats without a decimal point, so match
    // both forms per element.
    for w in want {
        let int_form = format!("{}", *w as i64);
        let float_form = format!("{w:.1}");
        assert!(
            stdout.contains(&int_form) || stdout.contains(&float_form),
            "{label}: gradient must contain {w}; got stdout={stdout}",
        );
    }
}

/// Headline acceptance: static-bound windowed MEAN pool grads to
/// `[0.5, 0.5, 0.5, 0.5]` end to end. Before the fix this PANICKED in
/// host-runtime grad lowering.
#[test]
fn issue_368_avgpool_static_grad_is_correct() {
    assert_eval_grad(
        AVGPOOL_STATIC,
        "avgpool",
        &[0.5, 0.5, 0.5, 0.5],
        "issue #368 static avgpool grad",
    );
}

/// Headline acceptance: static-bound windowed MAX pool grads to
/// `[0, 0, 1, 1]` end to end. Before the fix this built a malformed backward
/// DAG that failed verification ("dimension count 1 vs 0").
#[test]
fn issue_368_maxpool_static_grad_is_correct() {
    assert_eval_grad(
        MAXPOOL_STATIC,
        "maxpool",
        &[0.0, 0.0, 1.0, 1.0],
        "issue #368 static maxpool grad",
    );
}

/// The runtime-bounded window (the issue's verbatim reproducer) must reject
/// CLEANLY: exit non-zero with a specific, actionable diagnostic naming the
/// runtime `shrink` bounds limitation — NOT the old "index out of bounds"
/// panic, and NOT a confusing internal "bounds len 0" verify failure.
#[test]
fn issue_368_runtime_bounded_window_rejects_cleanly() {
    let output = run_eval(AVGPOOL_RUNTIME, "avgpool_runtime");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}{stderr}");
    assert!(
        !output.status.success(),
        "runtime-bounded window grad is not yet supported and must not succeed; \
         stdout={stdout} stderr={stderr}",
    );
    // Must be the clean, specific rejection — not the panic / opaque verify
    // failure the issue reported.
    assert!(
        combined.contains("statically-known bounds") && combined.contains("chelis#368"),
        "rejection must be the specific runtime-`shrink`-bounds diagnostic; got: {combined}",
    );
    assert!(
        !combined.contains("index out of bounds: the len is 0"),
        "must NOT be the old #368 panic; got: {combined}",
    );
    assert!(
        !combined.contains("bounds len 0 != input rank"),
        "must NOT be the opaque internal verify failure; got: {combined}",
    );
}
