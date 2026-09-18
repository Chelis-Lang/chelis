//! chelis#549: `grad` by-position named-axis recovery must not be silently
//! wrong after an intervening axis-reorder (`permute`/transpose).
//!
//! ## The bug this file locks
//!
//! Rank monomorphization records a named anchor's offset in a function's
//! *formal parameter* (`dim_axis_positions`) and consults that index at the
//! *monomorphized operand* at the reduce/split site. These coincide only if no
//! axis-reordering op intervenes. A `permute` between the recording site and the
//! use site leaves the recorded index valid-but-stale — in range, but pointing
//! at the wrong axis — so the reduction/split lands on the wrong axis: a SILENT
//! WRONG GRADIENT on a program that type-checks cleanly and whose forward pass
//! is correct. The pre-#515 in-range guard (`idx < dims.len()`) caught
//! out-of-range/unrecorded anchors but NOT a stale in-range index.
//!
//! The fix (crates/chelis-ir/src/lower.rs, `recover_anchor_axis`) pairs each
//! recorded position with the anchor's concrete extent and locates the anchor
//! purely by that extent in the monomorphized operand — the stale recorded
//! *index* is never trusted. It takes the axis when the extent is unique
//! (relocating through any intervening reorder), and FAILS LOUD when the extent
//! appears at more than one axis — never reducing/splitting a possibly-wrong
//! axis. Spans the three by-position use sites (#388 reduce, #373/#515 rank
//! spread, #339 expand) through the shared `recover_anchor_axis`.
//!
//! ## RT-1 square/equal-extent residual (closed to fail-loud)
//!
//! When the reduced axis's extent collides with another axis — the all-equal
//! (square) operand, e.g. `seq x seq` attention — value re-validation cannot
//! tell the anchor from its twin. An earlier version "confirmed by extent" at
//! the recorded position, which succeeds *vacuously* on a square operand and
//! silently returned the wrong axis. This is now an `AmbiguousAfterReorder`
//! loud failure. The soundness FLOOR over-rejects even a *non-permuted* square
//! reduce (the recorded index was in fact correct, but value alone cannot prove
//! no reorder happened); see `square_non_permuted_named_reduce_overrejected_*`.
//! Recovering those instead of rejecting needs the deeper fix: track the anchor
//! identity through the reorder during lowering. No executable example or
//! existing test hits the square by-position path, so the floor breaks no
//! legitimate corpus program.
//!
//! ## Oracle design (positive + negative parity, CLAUDE.md)
//!
//! The headline reduction `loss = sum_a (sum_seq x)^2` row-sums `r = [6, 15]`,
//! so `dL/dx_ij = 2*r_i = [[12,12,12],[30,30,30]]` — the value central finite
//! differences (h=1e-4, documented in `issue_364_grad_axis_fd.rs`) confirm. The
//! WRONG axis (column-sum) gradient is `[[10,14,18],[10,14,18]]` (`2*c_j`,
//! c=[5,7,9]); that is the exact pre-fix silent value the issue cites, pinned
//! here as the distinct control so the positive assertions are non-vacuous.
//! Correctness is cross-checked three ways: hardcoded FD value, a live in-test
//! central finite-difference of the forward loss, and the C backend's own grad
//! output (eval-vs-backend agreement). The negatives feed an *ambiguous* permute
//! (the anchor's extent appears at two axes after the reorder) through both the
//! reduce path and the rank-spread path and require a LOUD failure citing
//! chelis#549 in both the eval and the C-build lanes.

use std::fs;
use std::process::Command as StdCommand;

use assert_cmd::Command;
use tempfile::tempdir;

// ── Fixtures ────────────────────────────────────────────────────────────────

/// The forward `loss` for Repro 1: a `permute` sits between the bridge formal
/// (`[a, seq]`, recording `seq -> axis 1`) and the named reduce `sum(_, seq)`,
/// so by the use site the operand is `[Lit(3), Lit(2)]` and the recorded index 1
/// is stale. Scalar-returning so it doubles as the finite-difference target.
const REPRO1_FWD_PRELUDE: &str = "\
def bridge[a](x: &tensor[a, seq, f32]) -> tensor[a, f32] = sum(permute(x, 1, 0), seq)\n\
def loss(x: tensor[2, 3, f32]) -> f32 = {\n\
  r = bridge(&x)\n\
  sq = r * r\n\
  tensor_to_scalar(sum(sq, 0))\n\
}\n";

/// Repro 2: the same staleness through the #373/#515 rank-spread path — the
/// permuted operand flows into a Tier-3 `sum_seq[..pre, seq, ..post]` callee, so
/// the by-position fallback in `extract_rank_var_bindings` is exercised.
const REPRO2_FWD_PRELUDE: &str = "\
def sum_seq[pre, post](x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\n\
def bridge[a](x: &tensor[a, seq, f32]) -> tensor[a, f32] = sum_seq(permute(x, 1, 0))\n\
def loss(x: tensor[2, 3, f32]) -> f32 = {\n\
  r = bridge(&x)\n\
  sq = r * r\n\
  tensor_to_scalar(sum(sq, 0))\n\
}\n";

/// Control: the SAME loss WITHOUT a permute. The by-position recovery must still
/// resolve `seq` (an anti-over-rejection guard: the #549 re-validation must not
/// break the legitimate #388/#373 by-position path).
const CONTROL_FWD_PRELUDE: &str = "\
def bridge[a](x: &tensor[a, seq, f32]) -> tensor[a, f32] = sum(x, seq)\n\
def loss(x: tensor[2, 3, f32]) -> f32 = {\n\
  r = bridge(&x)\n\
  sq = r * r\n\
  tensor_to_scalar(sum(sq, 0))\n\
}\n";

const REPRO1_INPUT: &str = "to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])";
/// `[[1,2,3],[4,5,6]]` flattened, for the live finite-difference.
const REPRO1_INPUT_FLAT: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
const REPRO1_INPUT_SHAPE: [usize; 2] = [2, 3];

/// The correct (row-sum) gradient `2*r`, `r=[6,15]` — the finite-difference
/// value documented in `issue_364_grad_axis_fd.rs`.
const CORRECT_GRAD: [f64; 6] = [12.0, 12.0, 12.0, 30.0, 30.0, 30.0];
/// The pre-fix SILENT WRONG (column-sum) gradient the issue cites: `2*c`,
/// c=[5,7,9]. Pinned as the distinct control so the positive tests are not
/// vacuously equal to the wrong answer.
const WRONG_COLUMN_GRAD: [f64; 6] = [10.0, 14.0, 18.0, 10.0, 14.0, 18.0];

/// Reduce-path ambiguous negative: formal `[a, b, seq]` with `a == seq` extent
/// (both 2). `permute(x, 2, 0, 1)` reorders to `[seq=2, a=2, b=3]`, so after
/// monomorphization the recorded `seq` extent (2) appears at TWO axes and the
/// recorded index 2 holds `b` (3). The anchor cannot be located by value — must
/// fail loud rather than reduce a possibly-wrong axis.
const NEG_REDUCE: &str = "\
def bridge[a, b](x: &tensor[a, b, seq, f32]) -> tensor[a, b, f32] = sum(permute(x, 2, 0, 1), seq)\n\
def loss(x: tensor[2, 3, 2, f32]) -> f32 = {\n\
  r = bridge(&x)\n\
  sq = r * r\n\
  tensor_to_scalar(sum(sum(sq, 0), 0))\n\
}\n\
out = grad(loss)(to_tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]]))\n";

/// Spread-path ambiguous negative: same ambiguity routed through a Tier-3
/// `sum_seq[..pre, seq, ..post]` callee, exercising the `extract_rank_var_bindings`
/// by-position fallback.
const NEG_SPREAD: &str = "\
def sum_seq[pre, post](x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\n\
def bridge[a, b](x: &tensor[a, b, seq, f32]) -> tensor[a, b, f32] = sum_seq(permute(x, 2, 0, 1))\n\
def loss(x: tensor[2, 3, 2, f32]) -> f32 = {\n\
  r = bridge(&x)\n\
  sq = r * r\n\
  tensor_to_scalar(sum(sum(sq, 0), 0))\n\
}\n\
out = grad(loss)(to_tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]]))\n";

/// Rank-3 UNIQUE-extent permute positive: formal `[a, b, seq]` with distinct
/// extents (a=2, b=1, seq=3). `permute(x, 2, 0, 1)` -> `[seq=3, a=2, b=1]`; the
/// recorded `seq` extent (3) is unique after the reorder, so the anchor is
/// relocated correctly (recovery, not rejection). Same `2*r` gradient shape as
/// Repro 1.
const POS_RANK3_PRELUDE: &str = "\
def bridge[a, b](x: &tensor[a, b, seq, f32]) -> tensor[a, b, f32] = sum(permute(x, 2, 0, 1), seq)\n\
def loss(x: tensor[2, 1, 3, f32]) -> f32 = {\n\
  r = bridge(&x)\n\
  sq = r * r\n\
  tensor_to_scalar(sum(sum(sq, 0), 0))\n\
}\n";
const POS_RANK3_INPUT: &str = "to_tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])";

/// RT-1 square/equal-extent negative: a `3x3` operand where the reduced axis's
/// extent collides with the other axis. After `permute(x, 1, 0)` the recorded
/// `seq` position is stale AND its extent (3) appears at BOTH axes, so value
/// re-validation cannot tell the anchor from its twin — the fast "confirm by
/// extent" that an earlier version did would have trusted the stale position and
/// silently returned the column-sum `[[24,30,36]x3]` (correct row-sum is
/// `[[12,12,12],[30,30,30],[48,48,48]]`). Must fail LOUD instead.
const SQUARE_PERMUTE_PRELUDE: &str = "\
def bridge[a](x: &tensor[a, seq, f32]) -> tensor[a, f32] = sum(permute(x, 1, 0), seq)\n\
def loss(x: tensor[3, 3, f32]) -> f32 = {\n\
  r = bridge(&x)\n\
  sq = r * r\n\
  tensor_to_scalar(sum(sq, 0))\n\
}\n";

/// The SAME square reduce WITHOUT a permute. The recorded position is in fact
/// correct here, but value re-validation cannot distinguish that from the
/// permuted case (a `3x3` is `3x3` either way), so the soundness FLOOR
/// over-rejects it too. Documented as a known limitation; the deeper fix
/// (tracking the anchor through the reorder during lowering) is what recovers
/// these instead of rejecting.
const SQUARE_NOPERMUTE_PRELUDE: &str = "\
def bridge[a](x: &tensor[a, seq, f32]) -> tensor[a, f32] = sum(x, seq)\n\
def loss(x: tensor[3, 3, f32]) -> f32 = {\n\
  r = bridge(&x)\n\
  sq = r * r\n\
  tensor_to_scalar(sum(sq, 0))\n\
}\n";

const SQUARE_INPUT: &str = "to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])";

// ── Harness ──────────────────────────────────────────────────────────────────

fn grad_source(prelude: &str, input: &str) -> String {
    format!("{prelude}out = grad(loss)({input})\n")
}

fn forward_source(prelude: &str, input: &str) -> String {
    format!("{prelude}out = loss({input})\n")
}

/// Run `chelis eval --file` on `source`, asserting success, returning stdout.
fn eval_ok(source: &str, stem: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join(format!("{stem}.ch"));
    fs::write(&file, source).expect("write source");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", file.to_str().unwrap()])
        .output()
        .expect("run chelis eval");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "{stem}: `chelis eval` must succeed; stdout={stdout} stderr={stderr}",
    );
    stdout
}

/// Run `chelis eval --file` expecting FAILURE; return stderr for message pins.
fn eval_fail(source: &str, stem: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join(format!("{stem}.ch"));
    fs::write(&file, source).expect("write source");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", file.to_str().unwrap()])
        .output()
        .expect("run chelis eval");
    assert!(
        !output.status.success(),
        "{stem}: `chelis eval` must FAIL loud (not return a silent wrong gradient); \
         stdout={}",
        String::from_utf8_lossy(&output.stdout),
    );
    String::from_utf8_lossy(&output.stderr).to_string()
}

/// Parse the single-root `tensor(shape=[..], data=[..])` line eval/backend
/// prints into `(shape, data)`.
fn parse_tensor(stdout: &str) -> (Vec<usize>, Vec<f64>) {
    let line = stdout
        .lines()
        .find(|l| l.contains("tensor(shape="))
        .unwrap_or_else(|| panic!("no `tensor(shape=...)` line in stdout: {stdout}"));
    assert!(
        !line.contains("+ ..."),
        "tensor render is truncated (`+ ...`); fixture exceeds the 32-element cap: {line}",
    );
    let shape_str = line
        .split("shape=[")
        .nth(1)
        .and_then(|s| s.split(']').next())
        .unwrap_or_else(|| panic!("no shape in: {line}"));
    let shape: Vec<usize> = if shape_str.trim().is_empty() {
        vec![]
    } else {
        shape_str
            .split(',')
            .map(|t| t.trim().parse::<usize>().expect("shape int"))
            .collect()
    };
    let data_str = line
        .split("data=[")
        .nth(1)
        .and_then(|s| s.split(']').next())
        .unwrap_or_else(|| panic!("no data in: {line}"));
    let data: Vec<f64> = if data_str.trim().is_empty() {
        vec![]
    } else {
        data_str
            .split(',')
            .map(|t| t.trim().parse::<f64>().expect("data float"))
            .collect()
    };
    (shape, data)
}

/// Eval a grad program and parse its `out` tensor.
fn eval_grad(prelude: &str, input: &str, stem: &str) -> (Vec<usize>, Vec<f64>) {
    parse_tensor(&eval_ok(&grad_source(prelude, input), stem))
}

/// Eval a scalar-returning forward program; parse the bare numeric line.
fn eval_scalar(source: &str, stem: &str) -> f64 {
    let stdout = eval_ok(source, stem);
    stdout
        .lines()
        .find_map(|l| {
            l.trim()
                .trim_start_matches("out = ")
                .trim()
                .parse::<f64>()
                .ok()
        })
        .unwrap_or_else(|| panic!("no scalar line in eval stdout: {stdout}"))
}

/// Format a flat `data` slice (row-major) into a nested Surf tensor literal for
/// the given `shape`, e.g. `([2,3], [1..6]) -> "[[1.0, 2.0, 3.0], [4.0, ..]]"`.
fn nested_literal(data: &[f64], shape: &[usize]) -> String {
    if shape.is_empty() {
        let text = data[0].to_string();
        return if text.contains('.') || text.contains('e') {
            text
        } else {
            format!("{text}.0")
        };
    }
    let inner: usize = shape[1..].iter().product::<usize>().max(1);
    let parts: Vec<String> = (0..shape[0])
        .map(|i| nested_literal(&data[i * inner..(i + 1) * inner], &shape[1..]))
        .collect();
    format!("[{}]", parts.join(", "))
}

/// Central finite-difference of the forward `loss` defined by `prelude`,
/// evaluated about `base` (h=1e-2; the loss is exactly quadratic so the central
/// difference has no truncation error and matches the analytic gradient to
/// float rounding).
fn central_finite_difference(prelude: &str, base: &[f64], shape: &[usize], stem: &str) -> Vec<f64> {
    let h = 1e-2;
    (0..base.len())
        .map(|i| {
            let mut up = base.to_vec();
            let mut dn = base.to_vec();
            up[i] += h;
            dn[i] -= h;
            let lit_up = format!("to_tensor({})", nested_literal(&up, shape));
            let lit_dn = format!("to_tensor({})", nested_literal(&dn, shape));
            let f_up = eval_scalar(&forward_source(prelude, &lit_up), &format!("{stem}_up{i}"));
            let f_dn = eval_scalar(&forward_source(prelude, &lit_dn), &format!("{stem}_dn{i}"));
            (f_up - f_dn) / (2.0 * h)
        })
        .collect()
}

/// Build `source` with the C backend, compile + link + run it, return stdout.
/// Mirrors `rank_poly_tier3.rs::build_compile_run` (the eval-vs-backend oracle).
fn build_compile_run_grad(source: &str, name: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    fs::write(&src, source).expect("write source");

    let build = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            src.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("run chelis build");
    assert!(
        build.status.success(),
        "{name}: grad build must succeed; stderr: {}",
        String::from_utf8_lossy(&build.stderr)
    );

    let c_source = format!("{name}.c");
    let needs_blas = fs::read_to_string(out_dir.join(&c_source))
        .map(|t| t.contains("cblas_sgemm(") || t.contains("\"chelis_blas.h\""))
        .unwrap_or(false);
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: true,
            needs_blas,
        },
    );
    let bin = out_dir.join(name);
    let mut cc = StdCommand::new(&toolchain.compiler);
    cc.current_dir(&out_dir)
        .arg("-O2")
        .args(&toolchain.compile_flags)
        .arg(&c_source)
        .args(["-L.", "-lchelis_runtime"])
        .args(&toolchain.link_flags)
        .args(["-o", bin.to_str().unwrap()]);
    let link = cc.status().expect("host compiler runs");
    assert!(
        link.success(),
        "{name}: link of grad C must succeed: {link}"
    );

    let run = StdCommand::new(&bin).output().expect("binary runs");
    assert!(
        run.status.success(),
        "{name}: grad binary must run: {}\nstderr: {}",
        run.status,
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8(run.stdout).expect("utf-8 stdout")
}

/// Build `source` with the C backend expecting FAILURE; return stderr.
fn build_fail(source: &str, name: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    fs::write(&src, source).expect("write source");
    let build = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            src.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("run chelis build");
    assert!(
        !build.status.success(),
        "{name}: grad build must FAIL loud (no silent wrong C); stdout: {}",
        String::from_utf8_lossy(&build.stdout),
    );
    String::from_utf8_lossy(&build.stderr).to_string()
}

fn assert_close(label: &str, got: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{label}: length mismatch got={got:?} expected={expected:?}",
    );
    for (i, (g, e)) in got.iter().zip(expected).enumerate() {
        assert!(
            (g - e).abs() < tol,
            "{label}: [{i}] = {g} != {e} (tol {tol}); full got={got:?} expected={expected:?}",
        );
    }
}

// ── Positives ────────────────────────────────────────────────────────────────

/// Repro 1 (#388 reduce path): `grad` recovers the correct axis after the
/// permute. The gradient is the row-sum `[[12,12,12],[30,30,30]]`, NOT the
/// pre-fix silent column-sum `[[10,14,18],[10,14,18]]`.
#[test]
fn repro1_reduce_path_grad_recovers_correct_axis() {
    let (shape, grad) = eval_grad(REPRO1_FWD_PRELUDE, REPRO1_INPUT, "repro1");
    assert_eq!(shape, vec![2, 3], "gradient shape");
    assert_close("repro1 grad", &grad, &CORRECT_GRAD, 1e-3);
    assert_ne!(
        grad, WRONG_COLUMN_GRAD,
        "repro1 grad must NOT be the pre-fix silent column gradient (the discriminator)",
    );
}

/// Repro 2 (#373/#515 rank-spread path): identical staleness through a Tier-3
/// callee recovers the same correct gradient.
#[test]
fn repro2_spread_path_grad_recovers_correct_axis() {
    let (shape, grad) = eval_grad(REPRO2_FWD_PRELUDE, REPRO1_INPUT, "repro2");
    assert_eq!(shape, vec![2, 3], "gradient shape");
    assert_close("repro2 grad", &grad, &CORRECT_GRAD, 1e-3);
    assert_ne!(
        grad, WRONG_COLUMN_GRAD,
        "repro2 grad must NOT be the pre-fix silent column gradient",
    );
}

/// Eval-vs-backend agreement: the C backend builds and runs the Repro 1 grad
/// program and produces the SAME correct gradient as the evaluator (both lanes
/// read the recovered axis from the shared lowering; before the fix both agreed
/// on the WRONG value, so agreement alone is not enough — both must equal the
/// finite-difference value).
#[test]
fn repro1_grad_eval_matches_c_backend() {
    let eval = eval_grad(REPRO1_FWD_PRELUDE, REPRO1_INPUT, "repro1_eval").1;
    let (be_shape, be_grad) = parse_tensor(&build_compile_run_grad(
        &grad_source(REPRO1_FWD_PRELUDE, REPRO1_INPUT),
        "repro1_be",
    ));
    assert_eq!(be_shape, vec![2, 3], "backend gradient shape");
    assert_close("eval-vs-backend", &be_grad, &eval, 1e-5);
    assert_close("backend grad vs FD value", &be_grad, &CORRECT_GRAD, 1e-3);
}

/// Independent correctness oracle: a live central finite-difference of the
/// forward loss matches the evaluated gradient (CLAUDE.md backend-numerics
/// discipline; the loss is quadratic so the central difference is exact to
/// rounding).
#[test]
fn repro1_grad_matches_central_finite_difference() {
    let grad = eval_grad(REPRO1_FWD_PRELUDE, REPRO1_INPUT, "repro1_fd_grad").1;
    let fd = central_finite_difference(
        REPRO1_FWD_PRELUDE,
        &REPRO1_INPUT_FLAT,
        &REPRO1_INPUT_SHAPE,
        "repro1_fd",
    );
    assert_close("grad vs central FD", &grad, &fd, 5e-2);
    assert_close("FD vs documented value", &fd, &CORRECT_GRAD, 5e-2);
}

/// Rank-3 unique-extent permute: the recovery RELOCATES the anchor (does not
/// merely reject) when the reorder leaves the recorded extent unambiguous.
/// Cross-checked against a live finite-difference.
#[test]
fn rank3_unique_extent_permute_relocates_anchor() {
    let (shape, grad) = eval_grad(POS_RANK3_PRELUDE, POS_RANK3_INPUT, "pos3");
    assert_eq!(shape, vec![2, 1, 3], "gradient shape");
    assert_close("rank3 grad", &grad, &CORRECT_GRAD, 1e-3);
    let fd =
        central_finite_difference(POS_RANK3_PRELUDE, &REPRO1_INPUT_FLAT, &[2, 1, 3], "pos3_fd");
    assert_close("rank3 grad vs central FD", &grad, &fd, 5e-2);
}

/// Anti-over-rejection: the legitimate #388/#373 by-position recovery (no
/// permute) still resolves `seq` and gives the correct gradient. The #549
/// re-validation must not reject a non-reordered operand.
#[test]
fn non_permuted_named_reduce_still_recovers() {
    let (shape, grad) = eval_grad(CONTROL_FWD_PRELUDE, REPRO1_INPUT, "control");
    assert_eq!(shape, vec![2, 3], "gradient shape");
    assert_close("control grad", &grad, &CORRECT_GRAD, 1e-3);
}

// ── Negative parity (fail-closed) ─────────────────────────────────────────────

/// Reduce path: a permute that moves the anchor to an out-of-position but
/// in-range axis whose extent is AMBIGUOUS must fail LOUD in the eval lane —
/// never silently reduce a possibly-wrong axis. Pins the chelis#549 reason and
/// the offending axis name so a wrong-reason regression is caught.
#[test]
fn reduce_path_ambiguous_permute_fails_loud_eval() {
    let stderr = eval_fail(NEG_REDUCE, "neg_reduce");
    assert!(
        stderr.contains("chelis#549"),
        "eval must cite the chelis#549 axis-reorder soundness rule; stderr={stderr}",
    );
    assert!(
        stderr.contains("seq") && stderr.contains("unambiguously"),
        "eval must name the unresolvable axis `seq`; stderr={stderr}",
    );
}

/// Reduce path, C-build lane: the same ambiguous case must also fail LOUD when
/// built to C (a soundness rejection, not an absorbed generic message), so the
/// backend never emits a wrong-axis kernel.
#[test]
fn reduce_path_ambiguous_permute_fails_loud_build() {
    let stderr = build_fail(NEG_REDUCE, "neg_reduce_build");
    assert!(
        stderr.contains("chelis#549"),
        "C build must cite chelis#549 (not an absorbed generic grad message); stderr={stderr}",
    );
}

/// Rank-spread path: the same ambiguity through a Tier-3 callee fails LOUD in
/// the eval lane (the `extract_rank_var_bindings` fallback is fail-closed).
#[test]
fn spread_path_ambiguous_permute_fails_loud_eval() {
    let stderr = eval_fail(NEG_SPREAD, "neg_spread");
    assert!(
        stderr.contains("chelis#549") && stderr.contains("rank-spread anchor `seq`"),
        "eval must cite chelis#549 for the rank-spread path; stderr={stderr}",
    );
}

/// Rank-spread path, C-build lane: also fails LOUD with the precise reason.
#[test]
fn spread_path_ambiguous_permute_fails_loud_build() {
    let stderr = build_fail(NEG_SPREAD, "neg_spread_build");
    assert!(
        stderr.contains("chelis#549"),
        "C build must cite chelis#549 for the rank-spread path; stderr={stderr}",
    );
}

/// RT-1 regression: the all-equal-extent (square) operand. After `permute` the
/// recorded `seq` position is stale AND its extent collides with the other axis,
/// so an earlier "confirm by recorded extent" fast-path trusted the stale
/// position and SILENTLY returned the column-sum gradient (`[[24,30,36]x3]`,
/// exit 0). The soundness floor must instead fail LOUD: value re-validation
/// cannot tell the anchor from its equal-extent twin. Eval lane.
#[test]
fn square_permute_ambiguous_extent_fails_loud_eval() {
    let stderr = eval_fail(
        &grad_source(SQUARE_PERMUTE_PRELUDE, SQUARE_INPUT),
        "sq_eval",
    );
    assert!(
        stderr.contains("chelis#549") && stderr.contains("seq"),
        "the square-extent reorder must fail loud citing chelis#549 and `seq`; stderr={stderr}",
    );
    // The pre-fix silent value must never appear on stdout — eval exits non-zero
    // (asserted in `eval_fail`), so there is no `tensor(...)` line at all.
    assert!(
        !stderr.contains("24") || stderr.contains("chelis#549"),
        "must not silently emit the column-sum gradient; stderr={stderr}",
    );
}

/// RT-1, C-build lane: the same square case must also fail LOUD when built to C
/// (a soundness rejection, surfaced — not absorbed into a generic message).
#[test]
fn square_permute_ambiguous_extent_fails_loud_build() {
    let stderr = build_fail(
        &grad_source(SQUARE_PERMUTE_PRELUDE, SQUARE_INPUT),
        "sq_build",
    );
    assert!(
        stderr.contains("chelis#549"),
        "the square-extent reorder C build must cite chelis#549; stderr={stderr}",
    );
}

/// Documented soundness-floor limitation (NOT a desired end state). The SAME
/// square reduce WITHOUT a permute is a legitimate, correct program — the
/// recorded position is genuinely the anchor — but value re-validation cannot
/// distinguish a non-reordered `3x3` from a reordered one, so the floor
/// over-rejects it (fails loud) rather than risk the permuted twin's silent
/// wrong gradient. This test pins that current behavior AND the gradient the
/// program SHOULD yield (`[[12,12,12],[30,30,30],[48,48,48]]`, the value a
/// non-square `[a, seq]` reduce produces). When the deeper fix lands — tracking
/// the anchor identity through the reorder during lowering, so non-permuted
/// square recovers and permuted square stays correct — flip this to a positive
/// assertion against the documented value.
///
/// Corpus check (RT-1 item 3): no executable example or existing test hits this
/// path on a square operand, so the floor breaks no legitimate corpus program;
/// it is reachable only by hand-written square literal inputs like this one.
#[test]
fn square_non_permuted_named_reduce_overrejected_pending_deeper_fix() {
    // The value the deeper fix should eventually produce here (row-sum `2*r`,
    // r = [6, 15, 24]); pinned for the future positive flip.
    let _expected_when_recovered: [f64; 9] = [12.0, 12.0, 12.0, 30.0, 30.0, 30.0, 48.0, 48.0, 48.0];
    let stderr = eval_fail(
        &grad_source(SQUARE_NOPERMUTE_PRELUDE, SQUARE_INPUT),
        "sq_noperm",
    );
    assert!(
        stderr.contains("chelis#549"),
        "the soundness floor currently over-rejects the legitimate non-permuted square \
         reduce (documented limitation; deeper permute-tracking fix recovers it); stderr={stderr}",
    );
}

/// Provenance guard: the wrong-axis (column-sum) gradient and the correct
/// (row-sum) gradient are genuinely DIFFERENT vectors, so the positive
/// assertions are non-vacuous. (The wrong vector is what the pre-fix silent
/// recovery produced; here it is constructed directly by reducing axis 0.)
#[test]
fn correct_and_wrong_axis_gradients_are_distinct() {
    let column_prelude = "\
def bridge[a](x: &tensor[seq, a, f32]) -> tensor[a, f32] = sum(x, seq)\n\
def loss(x: tensor[2, 3, f32]) -> f32 = {\n\
  r = bridge(&x)\n\
  sq = r * r\n\
  tensor_to_scalar(sum(sq, 0))\n\
}\n";
    let (_, column_grad) = eval_grad(column_prelude, REPRO1_INPUT, "column");
    assert_close("column control", &column_grad, &WRONG_COLUMN_GRAD, 1e-3);
    assert_ne!(
        column_grad.to_vec(),
        CORRECT_GRAD.to_vec(),
        "the correct and wrong-axis gradients must differ, else the repros are vacuous",
    );
}
