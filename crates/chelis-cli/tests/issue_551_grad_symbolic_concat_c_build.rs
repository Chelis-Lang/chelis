//! Issue #551 (+ #340 host-lane witness): `chelis build --target c` over a
//! `grad` through a `concat` whose NON-concat axis is a symbolic (`batch`)
//! dim used to ICE in `chelis-ir/src/dag.rs::symbolic_occurrences`
//! ("symbolic dim `_anon_dim_1_0` ... referenced by a non-Load node ... but
//! no Load input declares it"). Two independent build-lane gaps caused it:
//!
//!   1. A single-axis reduction (`sum`/`max_reduce`/...) whose operand is a
//!      host-lane `concat` (or grad-backward concat cascade) carries a
//!      `*`-wildcard non-concat axis. The C backend renames the operand's
//!      wildcard and the reduction's SURVIVING wildcard to DIFFERENT
//!      `_anon_dim_*` names, so the reduction output's symbol appears in no
//!      Load. `shape_source_for_axis` had no reduction arm, so it could not
//!      trace the kept axis back to the declaring Load and the guard panicked.
//!      Fixed by a reduction arm that maps the kept output axis back through
//!      the removed reduce axis to the source Load (`dag.rs`).
//!   2. The grad `Pad` adjoint emits a `SHRINK_TO_END` full-axis sentinel on
//!      the symbolic no-pad axis. The eval lane resolves it via
//!      `bind_symbolic_dims`; the C build lane never binds, so the sentinel
//!      reached codegen and an over-broad assert failed loud. The C shrink
//!      loop is output-shape-driven and never reads the `hi` bound, so the
//!      sentinel is a correct full-axis identity — the assert is now
//!      per-axis and only rejects a MALFORMED sentinel (a concrete-axis or
//!      nonzero-start sentinel, a genuine producing-pass bug).
//!
//! This is the end-to-end C-build acceptance oracle: each case
//!   * `chelis build --target c` (must not ICE),
//!   * `gcc`-compiles + links the emitted kernel against the emitted
//!     `libchelis_runtime.a` and runs it,
//!   * checks the runtime output against the analytic value AND (for the
//!     grad cases) a self-contained central-difference finite-difference
//!     gradient computed in C from the emitted forward `loss`.
//!
//! Negative parity lives in the `chelis-ir` unit tests
//! (`symbolic_occurrences_reduction_arm_still_fails_loud_without_load`): a
//! reduction whose kept-axis symbol has NO declaring Load must still panic.
//! The reduction arm must not launder an unbound symbolic dim green.

use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;

use assert_cmd::Command;
use tempfile::{TempDir, tempdir};

/// `chelis build --target c` the source into a fresh build directory,
/// returning the (tempdir, build-dir) pair. `chelis build -o <dir>` writes
/// `<stem>.c`, the runtime headers, and `libchelis_runtime.a` into `<dir>`.
fn build_c(source: &str, stem: &str) -> (TempDir, std::path::PathBuf) {
    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join(format!("{stem}.ch"));
    fs::write(&src_path, source).expect("write .ch source");
    let build_dir = dir.path().join("build");
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            src_path.to_str().unwrap(),
            "--target",
            "c",
            "-o",
            build_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    (dir, build_dir)
}

/// gcc-compile `<stem>.c + driver.c` against the emitted
/// `libchelis_runtime.a`, run the binary, and return stdout. A gcc or
/// run-time failure fails the test with the captured diagnostics.
fn compile_and_run(build_dir: &Path, stem: &str, driver_src: &str) -> String {
    let driver = build_dir.join("driver.c");
    fs::write(&driver, driver_src).expect("write driver.c");
    let kernel = build_dir.join(format!("{stem}.c"));
    let runtime = build_dir.join("libchelis_runtime.a");
    let bin = build_dir.join("test_bin");
    let compile = StdCommand::new("gcc")
        .args([
            "-O0",
            "-std=c11",
            "-I",
            build_dir.to_str().unwrap(),
            kernel.to_str().unwrap(),
            driver.to_str().unwrap(),
            "-o",
            bin.to_str().unwrap(),
            runtime.to_str().unwrap(),
            "-lm",
            "-lpthread",
            "-ldl",
        ])
        .output()
        .expect("invoke gcc");
    assert!(
        compile.status.success(),
        "gcc compile failed: stderr={}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let run = StdCommand::new(&bin).output().expect("run test binary");
    assert!(
        run.status.success(),
        "binary exited non-zero: stdout={} stderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8_lossy(&run.stdout).into_owned()
}

// --- Grad, linear loss: loss = sum(concat([2x, 3x], axis=1)) = 5*sum(x). ---
// d loss / d x_i = 5 everywhere. Symbolic non-concat axis (`batch`).
const LINEAR_GRAD: &str = "module Repro.Sym551Linear\n\
def loss(x: tensor[batch, 2, f32]) -> f32 = {\n\
  a = add(&x, &x)\n\
  b = add(add(&x, &x), &x)\n\
  c = concat([a, b], cast(1, int32))\n\
  sum(sum(c, cast(0, int32)), cast(0, int32)) |> tensor_to_scalar\n\
}\n\
out = grad(loss)\n";

/// The headline #551 repro. Pre-fix `chelis build --target c` ICE'd at
/// `dag.rs` ("symbolic dim `_anon_dim_1_0` ... no Load input declares it").
/// The emitted grad kernel must compile, run, and (batch=2, x=[[1,2],[3,4]])
/// produce the analytic gradient 5 everywhere, matching the eval lane
/// (`issue_368_symbolic_nonconcat_axis_grad_does_not_overflow`).
#[test]
fn issue_551_grad_symbolic_concat_c_build_linear() {
    let (_dir, build_dir) = build_c(LINEAR_GRAD, "sym551lin");
    let driver = r#"
#include <stdio.h>
#include <string.h>
#include "chelis_runtime.h"
extern chelis_tensor* out(chelis_tensor* arg0);
int main(void) {
    int64_t shape[2] = {2, 2};
    chelis_tensor* x = chelis_alloc(2, shape, CHELIS_F32);
    float xd[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    memcpy(x->data, xd, sizeof(xd));
    chelis_tensor* g = out(x);
    if (g->size != 4) { printf("FAIL_SIZE %lld\n", (long long)g->size); return 1; }
    for (int i = 0; i < 4; i++) printf("%.6f\n", g->data[i]);
    return 0;
}
"#;
    let stdout = compile_and_run(&build_dir, "sym551lin", driver);
    let got: Vec<f64> = stdout
        .lines()
        .map(|l| l.trim().parse::<f64>().expect("grad element"))
        .collect();
    assert_eq!(got.len(), 4, "grad shape (stdout={stdout})");
    for (i, g) in got.iter().enumerate() {
        assert!(
            (g - 5.0).abs() < 1e-3,
            "linear grad elem {i}: got {g}, want 5.0 (all={got:?})"
        );
    }
}

// --- Grad, nonlinear loss: loss = sum(square(concat([x, 2x], 1))) = sum(5 x^2).
// d loss / d x_i = 10 x_i. FD-validated in-C against the emitted forward loss.
const NONLINEAR_GRAD: &str = "module Repro.Sym551Nonlinear\n\
def loss(x: tensor[batch, 2, f32]) -> f32 = {\n\
  a = sub(add(&x, &x), &x)\n\
  b = add(&x, &x)\n\
  c = concat([a, b], cast(1, int32))\n\
  sq = mul(c, c)\n\
  sum(sum(sq, cast(0, int32)), cast(0, int32)) |> tensor_to_scalar\n\
}\n\
out = grad(loss)\n";

/// Nonlinear symbolic-axis concat grad, C build lane. The emitted grad
/// kernel `out` must match BOTH the analytic gradient `10 x` AND a
/// central-difference finite-difference gradient computed in C from the
/// emitted forward scalar `loss` (a genuine executed FD check, RT-2
/// discipline). This is the eval-vs-C-backend + FD oracle for #551's
/// nonlinear path (eval reference: [10, 20, 30, 40]).
#[test]
fn issue_551_grad_symbolic_concat_c_build_nonlinear_fd() {
    let (_dir, build_dir) = build_c(NONLINEAR_GRAD, "sym551nl");
    let driver = r#"
#include <stdio.h>
#include <string.h>
#include "chelis_runtime.h"
extern chelis_tensor* out(chelis_tensor* arg0);
extern float loss(chelis_tensor* x);

static float call_loss(const float* xd) {
    int64_t shape[2] = {2, 2};
    chelis_tensor* x = chelis_alloc(2, shape, CHELIS_F32);
    memcpy(x->data, xd, sizeof(float) * 4);
    return loss(x);
}

int main(void) {
    float base[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    int64_t shape[2] = {2, 2};
    chelis_tensor* x = chelis_alloc(2, shape, CHELIS_F32);
    memcpy(x->data, base, sizeof(base));
    chelis_tensor* g = out(x);
    if (g->size != 4) { printf("FAIL_SIZE %lld\n", (long long)g->size); return 1; }

    const float h = 1e-2f;
    for (int i = 0; i < 4; i++) {
        float xp[4], xm[4];
        memcpy(xp, base, sizeof(base));
        memcpy(xm, base, sizeof(base));
        xp[i] += h;
        xm[i] -= h;
        float fd = (call_loss(xp) - call_loss(xm)) / (2.0f * h);
        /* analytic (10 x), C grad, and finite-difference all on one line */
        printf("%.6f %.6f %.6f\n", 10.0f * base[i], g->data[i], fd);
    }
    return 0;
}
"#;
    let stdout = compile_and_run(&build_dir, "sym551nl", driver);
    let rows: Vec<Vec<f64>> = stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split_whitespace()
                .map(|t| t.parse::<f64>().expect("number"))
                .collect()
        })
        .collect();
    assert_eq!(rows.len(), 4, "expected 4 grad rows (stdout={stdout})");
    for (i, row) in rows.iter().enumerate() {
        let (analytic, c_grad, fd) = (row[0], row[1], row[2]);
        assert!(
            (c_grad - analytic).abs() < 1e-2,
            "elem {i}: C grad {c_grad} != analytic {analytic}"
        );
        assert!(
            (c_grad - fd).abs() < 5e-2,
            "elem {i}: C grad {c_grad} != finite-difference {fd}"
        );
    }
}

// --- #340 host-lane witness (no grad): a reduction whose operand is a
// host-lane concat over a symbolic axis. Same guard, non-grad trigger. ---
const HOST_LANE_CONCAT_REDUCE: &str = "module Repro.Sym551HostLane\n\
def f(x: tensor[batch, 2, f32]) -> tensor[four, f32] = {\n\
  a = add(&x, &x)\n\
  b = add(add(&x, &x), &x)\n\
  c = concat([a, b], cast(1, int32))\n\
  sum(c, cast(0, int32))\n\
}\n\
out = f\n";

/// #340 witness folded into #551's oracle: a FORWARD (no grad) reduction
/// sourced from a host-lane `concat` over a symbolic batch axis. Pre-fix
/// this hit the SAME `dag.rs` guard ("symbolic dim ... referenced by a
/// non-Load node"). The emitted kernel must compile, run, and produce the
/// eval-lane value. x=[[1,2],[3,4]]: a=2x=[[2,4],[6,8]], b=3x=[[3,6],[9,12]],
/// concat axis 1 -> [[2,4,3,6],[6,8,9,12]], sum axis 0 -> [8,12,12,18].
#[test]
fn issue_551_host_lane_concat_reduce_c_build() {
    let (_dir, build_dir) = build_c(HOST_LANE_CONCAT_REDUCE, "sym551host");
    let driver = r#"
#include <stdio.h>
#include <string.h>
#include "chelis_runtime.h"
extern chelis_tensor* out(chelis_tensor* arg0);
int main(void) {
    int64_t shape[2] = {2, 2};
    chelis_tensor* x = chelis_alloc(2, shape, CHELIS_F32);
    float xd[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    memcpy(x->data, xd, sizeof(xd));
    chelis_tensor* r = out(x);
    if (r->size != 4) { printf("FAIL_SIZE %lld\n", (long long)r->size); return 1; }
    for (int i = 0; i < 4; i++) printf("%.6f\n", r->data[i]);
    return 0;
}
"#;
    let stdout = compile_and_run(&build_dir, "sym551host", driver);
    let got: Vec<f64> = stdout
        .lines()
        .map(|l| l.trim().parse::<f64>().expect("element"))
        .collect();
    assert_eq!(got, vec![8.0, 12.0, 12.0, 18.0], "host-lane concat->reduce");
}

// ---------------------------------------------------------------------------
// chelis#593 (RT-2 finding), CLOSED by the chelis#616 renaming fix. The
// original hole: the C-backend `rename_anonymous_dims` resolved an anon
// trailing dim by copying the operand's dims wholesale, clobbering the
// correctly-sized CONCRETE padded axis (`Lit(4)`) back to the operand extent
// (`Lit(2)`) — a heap OOB write and a silently wrong forward result.
// `validate_pad_output_sizing` was the fail-closed floor that kept it loud.
// chelis#616 replaced the wholesale copy: any extent-altering movement op
// renames per-axis instead, and the fresh runtime dims are declared from the
// op's own bound formulas (with runtime equality guards across sites), so
// the leading-axis symbolic concat is now correctly sized end to end. The
// tests below flipped from fail-closed rejects to build-and-run oracles;
// `validate_pad_output_sizing` remains as a defensive backstop.
// ---------------------------------------------------------------------------

// The RT-2 reproducer: reduce over a LEADING-axis concat, symbolic trailing dim.
const RT2_REDUCE_LEADING_CONCAT: &str = "module Repro.RT2Reduce\n\
def loss(x: tensor[2, batch, f32]) -> tensor[batch, f32] = {\n\
  a = add(&x, &x)\n\
  b = add(add(&x, &x), &x)\n\
  c = concat([a, b], cast(0, int32))\n\
  sum(c, cast(0, int32))\n\
}\n\
out = loss\n";

/// chelis#593 CLOSED by the chelis#616 anon-dim renaming fix: the
/// copy-first-input shortcut no longer clobbers an extent-altering Pad's
/// concrete padded axis (per-axis fresh anon dims + op-declared runtime-dim
/// sources replaced it), so the RT-2 reproducer now builds AND runs
/// CORRECTLY instead of tripping the fail-closed `validate_pad_output_sizing`
/// floor. x=[[1,2,3],[4,5,6]]: a=2x, b=3x, concat axis 0, sum axis 0 ->
/// [25, 35, 45] — the value eval computes, with the second concat pad's
/// runtime equality guard validating the shared `four` extent.
#[test]
fn issue_593_leading_axis_symbolic_concat_reduce_builds_and_runs() {
    let (_dir, build_dir) = build_c(RT2_REDUCE_LEADING_CONCAT, "rt2reduce");
    let driver = r#"
#include <stdio.h>
#include <string.h>
#include "chelis_runtime.h"
extern chelis_tensor* out(chelis_tensor* arg0);
int main(void) {
    int64_t shape[2] = {2, 3};
    chelis_tensor* x = chelis_alloc(2, shape, CHELIS_F32);
    float xd[6] = {1,2,3,4,5,6};
    memcpy(x->data, xd, sizeof(xd));
    chelis_tensor* r = out(x);
    if (r->size != 3) { printf("FAIL_SIZE %lld\n", (long long)r->size); return 1; }
    for (int i = 0; i < 3; i++) printf("%.1f\n", r->data[i]);
    return 0;
}
"#;
    let stdout = compile_and_run(&build_dir, "rt2reduce", driver);
    let got: Vec<f64> = stdout
        .lines()
        .map(|l| l.trim().parse::<f64>().expect("element"))
        .collect();
    assert_eq!(
        got,
        vec![25.0, 35.0, 45.0],
        "leading-axis symbolic concat reduce (the RT-2 reproducer)"
    );
}

// The bare-concat path (no reduce) — main's pre-existing memory hole. The #551
// arm is not involved here; the guard must cover it too.
const BARE_LEADING_CONCAT: &str = "module Repro.BareLeading\n\
def f(x: tensor[2, batch, f32]) -> tensor[four, batch, f32] = {\n\
  a = add(&x, &x)\n\
  b = add(add(&x, &x), &x)\n\
  concat([a, b], cast(0, int32))\n\
}\n\
out = f\n";

/// chelis#593 CLOSED (see the reduce twin above): the BARE leading-axis
/// concat over a symbolic trailing dim now builds and runs with the
/// correctly-sized `[four, batch]` output — the concat pads keep their
/// concrete padded axis and the sig-named `four` extent is declared from
/// the first pad's runtime formula (equality-guarded at the second).
#[test]
fn issue_593_bare_leading_axis_symbolic_concat_builds_and_runs() {
    let (_dir, build_dir) = build_c(BARE_LEADING_CONCAT, "bareleading");
    let driver = r#"
#include <stdio.h>
#include <string.h>
#include "chelis_runtime.h"
extern chelis_tensor* out(chelis_tensor* arg0);
int main(void) {
    int64_t shape[2] = {2, 3};
    chelis_tensor* x = chelis_alloc(2, shape, CHELIS_F32);
    float xd[6] = {1,2,3,4,5,6};
    memcpy(x->data, xd, sizeof(xd));
    chelis_tensor* c = out(x);
    if (c->ndim != 2 || c->shape[0] != 4 || c->shape[1] != 3) {
        printf("FAIL_SHAPE %d %lld %lld\n", c->ndim, (long long)c->shape[0], (long long)c->shape[1]);
        return 1;
    }
    for (int i = 0; i < c->size; i++) printf("%.1f\n", c->data[i]);
    return 0;
}
"#;
    let stdout = compile_and_run(&build_dir, "bareleading", driver);
    let got: Vec<f64> = stdout
        .lines()
        .map(|l| l.trim().parse::<f64>().expect("element"))
        .collect();
    assert_eq!(
        got,
        vec![
            2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0
        ],
        "bare leading-axis symbolic concat (a = 2x rows, then b = 3x rows)"
    );
}

// POSITIVE control: a CONCRETE-dim leading-axis concat is correctly sized (no
// anon dim to trigger the wrapper clobber), so it must still build AND run.
const CONCRETE_LEADING_CONCAT: &str = "module Repro.ConcreteLeading\n\
def f(x: tensor[2, 3, f32]) -> tensor[3, f32] = {\n\
  a = add(&x, &x)\n\
  b = add(add(&x, &x), &x)\n\
  c = concat([a, b], cast(0, int32))\n\
  sum(c, cast(0, int32))\n\
}\n\
out = f\n";

/// POSITIVE (no over-rejection): a concrete-dim leading-axis concat is
/// well-sized and must still build + run correctly. x=[[1,2,3],[4,5,6]]:
/// a=2x, b=3x, concat axis 0 -> rows [2,4,6],[8,10,12],[3,6,9],[12,15,18];
/// sum axis 0 -> [25, 35, 45].
#[test]
fn issue_593_concrete_leading_axis_concat_still_builds_and_runs() {
    let (_dir, build_dir) = build_c(CONCRETE_LEADING_CONCAT, "concleading");
    let driver = r#"
#include <stdio.h>
#include <string.h>
#include "chelis_runtime.h"
extern chelis_tensor* out(chelis_tensor* arg0);
int main(void) {
    int64_t shape[2] = {2, 3};
    chelis_tensor* x = chelis_alloc(2, shape, CHELIS_F32);
    float xd[6] = {1,2,3,4,5,6};
    memcpy(x->data, xd, sizeof(xd));
    chelis_tensor* r = out(x);
    if (r->size != 3) { printf("FAIL_SIZE %lld\n", (long long)r->size); return 1; }
    for (int i = 0; i < 3; i++) printf("%.1f\n", r->data[i]);
    return 0;
}
"#;
    let stdout = compile_and_run(&build_dir, "concleading", driver);
    let got: Vec<f64> = stdout
        .lines()
        .map(|l| l.trim().parse::<f64>().expect("element"))
        .collect();
    assert_eq!(got, vec![25.0, 35.0, 45.0], "concrete leading-axis concat");
}

/// POSITIVE (no over-rejection): a symbolic LAST-axis concat grad must still
/// build + run correctly — the working #551 path (reuses SYM_BATCH_LINEAR,
/// concat axis 1 over `tensor[batch, 2]`), guarding that the #593 floor does
/// not reject the last-axis case. d/dx sum(concat([2x,3x],axis=1)) = 5.
#[test]
fn issue_593_last_axis_symbolic_concat_grad_still_builds_and_runs() {
    let (_dir, build_dir) = build_c(LINEAR_GRAD, "lastaxisok");
    let driver = r#"
#include <stdio.h>
#include <string.h>
#include "chelis_runtime.h"
extern chelis_tensor* out(chelis_tensor* arg0);
int main(void) {
    int64_t shape[2] = {2, 2};
    chelis_tensor* x = chelis_alloc(2, shape, CHELIS_F32);
    float xd[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    memcpy(x->data, xd, sizeof(xd));
    chelis_tensor* g = out(x);
    if (g->size != 4) { printf("FAIL_SIZE %lld\n", (long long)g->size); return 1; }
    for (int i = 0; i < 4; i++) printf("%.1f\n", g->data[i]);
    return 0;
}
"#;
    let stdout = compile_and_run(&build_dir, "lastaxisok", driver);
    let got: Vec<f64> = stdout
        .lines()
        .map(|l| l.trim().parse::<f64>().expect("element"))
        .collect();
    assert_eq!(
        got,
        vec![5.0, 5.0, 5.0, 5.0],
        "last-axis symbolic concat grad"
    );
}
