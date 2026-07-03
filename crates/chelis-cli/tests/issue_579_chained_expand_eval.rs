//! chelis#579 (residue of chelis#397): chained `expand` rank-1 -> rank-N
//! broadcasts (the batchnorm pattern) failed host eval at chelis 0.12.0 with
//! the rank-monomorphization ICE `tensor rank mismatch: 1 dims vs 2 dims` /
//! `1 dims vs 4 dims` even though the program was check-clean.
//!
//! The fix shipped after 0.12.0 (PR #596, the Form-3 §4.7.2 runtime
//! expand-size resolution + fail-closed reject). This corpus pins BOTH sides
//! of that resolution so the #579 failure mode cannot silently return:
//!
//! - POSITIVE: the supported spelling of the batchnorm broadcasts, where each
//!   runtime extent is sourced from an in-scope tensor via `shape(x, axis)`,
//!   EVALUATES correctly at rank-1 -> rank-2 (batchnorm1d) and through the
//!   full chained rank-1 -> rank-4 `broadcast_to_achw` (batchnorm2d),
//!   including the two-broadcast affine composition, and the C backend agrees
//!   with the evaluator value-for-value.
//! - NEGATIVE: the sourceless spelling from the issue (a bare runtime
//!   `int64` dim parameter with no tensor source) is rejected LOUDLY at
//!   check and eval with the #469 sourceless-size diagnostic, and the reject
//!   never regresses into the original `rank mismatch` ICE. Genuine misuse
//!   (wrong-axis broadcast, out-of-bounds insert axis) also still fails with
//!   targeted reasons.
//!
//! Sibling coverage: `rank_poly_tier3.rs` pins the check/build/eval reject of
//! the rank-1 -> rank-4 chain and the single-expand `bias_broadcast`
//! positives. This file adds the missing CHAINED positive lane (eval + C
//! parity) plus the batchnorm1d-flavor reject pins.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;
use tempfile::tempdir;

// ── Surf sources ─────────────────────────────────────────────────────────

/// batchnorm1d-style single shape-sourced broadcast: rank-1 `g` scaled over
/// the batch axis of rank-2 `x`. The #579 discriminator case (passed even at
/// 0.12.0); pinned so the working baseline is explicit.
const BN1D_SOURCE: &str = "def bn1d_scale(x: &tensor[a, n, f32], g: &tensor[n, f32]) -> tensor[a, n, f32] = {\n\
    \x20 gb: tensor[a, n, f32] = expand(g, 0, shape(x, cast(0, int32)))\n\
    \x20 mul(x, gb)\n\
    }\n\
    xs = to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\n\
    gs = to_tensor([10.0, 20.0, 30.0])\n\
    out = bn1d_scale(xs, gs)\n";

/// The supported spelling of school's `broadcast_to_achw` (batchnorm2d):
/// three CHAINED Form-3 expands, every extent read from the in-scope source
/// tensor `x` via `shape(x, axis)`. `out` is the raw rank-1 -> rank-4
/// broadcast; `out_affine` is the batchnorm2d-forward affine composition
/// (two broadcasts through the same helper, then mul + add). `x` carries
/// sequential data so the affine values encode flat position: any axis
/// permutation in evaluation changes them and cannot hide.
fn chained_achw_source(shape: &[usize; 4]) -> String {
    let mut next = 1.0f64;
    let xs = nested_literal(shape, &mut next);
    format!(
        "def broadcast_to_achw(g: &tensor[c, f32], x: &tensor[a, c, h, w, f32]) -> tensor[a, c, h, w, f32] = {{\n\
        \x20 step1: tensor[c, h, f32] = expand(g, 1, shape(x, cast(2, int32)))\n\
        \x20 step2: tensor[c, h, w, f32] = expand(step1, 2, shape(x, cast(3, int32)))\n\
        \x20 step3: tensor[a, c, h, w, f32] = expand(step2, 0, shape(x, cast(0, int32)))\n\
        \x20 step3\n\
        }}\n\
        def bn2d_affine(x: &tensor[a, c, h, w, f32], g: &tensor[c, f32], b: &tensor[c, f32]) -> tensor[a, c, h, w, f32] = {{\n\
        \x20 gb = broadcast_to_achw(g, x)\n\
        \x20 bb = broadcast_to_achw(b, x)\n\
        \x20 add(mul(x, gb), bb)\n\
        }}\n\
        xs = to_tensor({xs})\n\
        gs = to_tensor([10.0, 20.0, 30.0])\n\
        bs = to_tensor([1.0, 2.0, 3.0])\n\
        out = broadcast_to_achw(gs, xs)\n\
        out_affine = bn2d_affine(xs, gs, bs)\n"
    )
}

/// The EXACT sourceless spelling from the issue, batchnorm1d flavor: the
/// expand size is a bare runtime `int64` parameter with no tensor source.
/// At 0.12.0 this was check-clean and died at eval with
/// `tensor rank mismatch: 1 dims vs 2 dims`.
const SOURCELESS_1D_SOURCE: &str = "def bcast_1d_to_2d[a, n](g: tensor[n, f32], a_dim: int64) -> tensor[a, n, f32] = expand(g, 0, a_dim)\n\
    out = bcast_1d_to_2d(to_tensor([1.0, 2.0, 3.0]), cast(2, int64))\n";

/// The sourceless chained rank-1 -> rank-4 spelling (school's original
/// `broadcast_to_achw`). At 0.12.0: check-clean, eval died with
/// `tensor rank mismatch: 1 dims vs 4 dims`.
const SOURCELESS_ACHW_SOURCE: &str = "def broadcast_to_achw[c, h, w, a](v: &tensor[c, f32], h_dim: int64, w_dim: int64, a_dim: int64) -> tensor[a, c, h, w, f32] = {\n\
    \x20 step1: tensor[c, h, f32] = expand(v, 1, h_dim)\n\
    \x20 step2: tensor[c, h, w, f32] = expand(step1, 2, w_dim)\n\
    \x20 step3: tensor[a, c, h, w, f32] = expand(step2, 0, a_dim)\n\
    \x20 step3\n\
    }\n\
    out = broadcast_to_achw(to_tensor([1.0, 2.0]), cast(3, int64), cast(4, int64), cast(5, int64))\n";

/// Build a nested Surf tensor literal of `shape` with sequential f32 data.
fn nested_literal(shape: &[usize], next: &mut f64) -> String {
    match shape.split_first() {
        None => {
            let v = *next;
            *next += 1.0;
            format!("{v:?}")
        }
        Some((head, rest)) => {
            let inner: Vec<String> = (0..*head).map(|_| nested_literal(rest, next)).collect();
            format!("[{}]", inner.join(", "))
        }
    }
}

// ── Helpers (rank_poly_tier3.rs conventions) ─────────────────────────────

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

fn check_errors(json: &Value, label: &str) -> Vec<String> {
    json["errors"]
        .as_array()
        .unwrap_or_else(|| panic!("{label}: errors should be a json array, got {json}"))
        .iter()
        .map(|e| e["message"].as_str().unwrap_or_default().to_string())
        .collect()
}

fn assert_clean(json: &Value, label: &str) {
    let errors = check_errors(json, label);
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

fn eval_stdout(dir: &Path, source: &str, name: &str) -> String {
    let src = dir.join(format!("{name}.ch"));
    fs::write(&src, source).expect("write source");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "eval",
            "--file",
            src.to_str().unwrap(),
            "--allow-style-violations",
        ])
        .output()
        .expect("run chelis eval");
    assert!(
        output.status.success(),
        "chained-expand eval must succeed; stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("utf-8 stdout")
}

fn eval_stderr_expecting_failure(dir: &Path, source: &str, name: &str) -> String {
    let src = dir.join(format!("{name}.ch"));
    fs::write(&src, source).expect("write source");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "eval",
            "--file",
            src.to_str().unwrap(),
            "--allow-style-violations",
        ])
        .output()
        .expect("run chelis eval");
    assert!(
        !output.status.success(),
        "eval was expected to fail; stdout: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    String::from_utf8(output.stderr).expect("utf-8 stderr")
}

fn build_compile_run(source: &str, name: &str) -> String {
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
        "chained-expand C build must succeed; stderr: {}",
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
        "link of chained-expand C must succeed: {link}"
    );

    let run = StdCommand::new(&bin).output().expect("binary runs");
    assert!(
        run.status.success(),
        "chained-expand binary must run: {}\nstderr: {}",
        run.status,
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8(run.stdout).expect("utf-8 stdout")
}

fn parse_printed_tensors(stdout: &str) -> Vec<(String, Vec<usize>, Vec<f64>)> {
    let mut out = Vec::new();
    for line in stdout.lines() {
        let Some((name, rest)) = line.split_once(" = tensor(") else {
            continue;
        };
        let shape = rest
            .split_once("shape=[")
            .and_then(|(_, s)| s.split_once(']'))
            .map(|(s, _)| {
                s.split(',')
                    .filter_map(|p| p.trim().parse::<usize>().ok())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let data = rest
            .split_once("data=[")
            .and_then(|(_, s)| s.split_once(']'))
            .map(|(s, _)| {
                s.split(',')
                    .filter_map(|p| p.trim().parse::<f64>().ok())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        out.push((name.trim().to_string(), shape, data));
    }
    out
}

fn find_tensor<'a>(
    tensors: &'a [(String, Vec<usize>, Vec<f64>)],
    name: &str,
    label: &str,
) -> &'a (String, Vec<usize>, Vec<f64>) {
    tensors
        .iter()
        .find(|(n, _, _)| n == name)
        .unwrap_or_else(|| panic!("{label}: output missing `{name}`: {tensors:?}"))
}

/// Every backend-printed tensor must have an eval twin, shape-for-shape and
/// value-for-value (the eval-vs-backend agreement oracle).
fn assert_eval_agrees_with_backend(source: &str, name: &str, backend: &str) {
    let backend_tensors = parse_printed_tensors(backend);
    assert!(
        !backend_tensors.is_empty(),
        "{name}: backend printed no tensors: {backend}"
    );
    let dir = tempdir().expect("tempdir");
    let eval = eval_stdout(dir.path(), source, name);
    let eval_tensors = parse_printed_tensors(&eval);
    for (tensor_name, shape, data) in &backend_tensors {
        let e = find_tensor(&eval_tensors, tensor_name, name);
        assert_eq!(
            shape, &e.1,
            "{tensor_name}: eval-vs-backend shape disagreement"
        );
        assert_eq!(
            data.len(),
            e.2.len(),
            "{tensor_name}: eval-vs-backend length"
        );
        for (i, (b, ev)) in data.iter().zip(e.2.iter()).enumerate() {
            assert!(
                (b - ev).abs() < 1e-6,
                "{tensor_name}[{i}]: backend {b} vs eval {ev}"
            );
        }
    }
}

// Expected values for the chained achw corpora: x carries sequential data
// 1..=numel over shape [a, c, h, w]; g = [10, 20, 30]; b = [1, 2, 3]. The
// channel of flat index i is (i / (h*w)) % c.
//
// SMALL corpus [2, 3, 2, 2] (24 elements): fits under the evaluator's
// 32-element print budget, so every element is asserted; the sequential
// affine values encode flat position, catching axis permutations in VALUES.
// BIG corpus [2, 3, 4, 5] (120 elements): all four dims DISTINCT, so any
// axis or shape-source mixup changes the output SHAPE and cannot hide; the
// printer truncates data to a prefix, which is asserted element-for-element.
const SMALL_SHAPE: [usize; 4] = [2, 3, 2, 2];
const BIG_SHAPE: [usize; 4] = [2, 3, 4, 5];
const G: [f64; 3] = [10.0, 20.0, 30.0];
const B: [f64; 3] = [1.0, 2.0, 3.0];

fn channel_of(i: usize, shape: &[usize; 4]) -> usize {
    (i / (shape[2] * shape[3])) % shape[1]
}

// ── Positives: the #579 patterns in the supported shape-sourced form ─────

/// batchnorm1d flavor: a single shape-sourced rank-1 -> rank-2 expand plus
/// `mul` checks clean, evaluates, and produces the exact per-column scaling.
#[test]
fn issue_579_bn1d_single_shape_sourced_expand_evals() {
    assert_clean(
        &check_json(BN1D_SOURCE),
        "bn1d shape-sourced expand checks clean",
    );
    let dir = tempdir().expect("tempdir");
    let eval = eval_stdout(dir.path(), BN1D_SOURCE, "issue_579_bn1d");
    let tensors = parse_printed_tensors(&eval);
    let out = find_tensor(&tensors, "out", "bn1d eval");
    assert_eq!(out.1, vec![2, 3], "bn1d broadcast shape");
    let expected = [10.0, 40.0, 90.0, 40.0, 100.0, 180.0];
    for (i, e) in expected.iter().enumerate() {
        assert!(
            (out.2[i] - e).abs() < 1e-6,
            "bn1d out[{i}]: eval {} != {e}",
            out.2[i]
        );
    }
}

/// Assert the raw broadcast (`out`) and the affine composition
/// (`out_affine`) of a chained achw corpus against the analytic values.
/// `require_full` demands complete (untruncated) data, used for the small
/// corpus that fits the evaluator's print budget; the big corpus asserts
/// whatever prefix is printed, element-for-element.
fn assert_achw_values(
    tensors: &[(String, Vec<usize>, Vec<f64>)],
    shape: &[usize; 4],
    require_full: bool,
    label: &str,
) {
    let numel: usize = shape.iter().product();
    let expect_shape: Vec<usize> = shape.to_vec();

    let out = find_tensor(tensors, "out", label);
    assert_eq!(out.1, expect_shape, "{label}: chained broadcast shape");
    if require_full {
        assert_eq!(out.2.len(), numel, "{label}: broadcast element count");
    } else {
        assert!(!out.2.is_empty(), "{label}: broadcast printed no data");
    }
    for (i, v) in out.2.iter().enumerate() {
        let e = G[channel_of(i, shape)];
        assert!((v - e).abs() < 1e-6, "{label}: out[{i}]: {v} != {e}");
    }

    let affine = find_tensor(tensors, "out_affine", label);
    assert_eq!(affine.1, expect_shape, "{label}: affine shape");
    if require_full {
        assert_eq!(affine.2.len(), numel, "{label}: affine element count");
    } else {
        assert!(!affine.2.is_empty(), "{label}: affine printed no data");
    }
    for (i, v) in affine.2.iter().enumerate() {
        let c = channel_of(i, shape);
        let e = (i as f64 + 1.0) * G[c] + B[c];
        assert!((v - e).abs() < 1e-6, "{label}: out_affine[{i}]: {v} != {e}");
    }
}

/// The #579 headline: the chained rank-1 -> rank-4 `broadcast_to_achw` in
/// its supported shape-sourced spelling checks clean and EVALUATES, with the
/// exact per-channel broadcast values, and the two-broadcast batchnorm2d
/// affine composition on top of it evaluates too. At 0.12.0 the chained
/// pattern died in eval lowering with `tensor rank mismatch: 1 dims vs 4
/// dims`. Small corpus: complete value coverage. Big corpus: distinct dims
/// pin the shape against axis/source mixups.
#[test]
fn issue_579_chained_rank4_shape_sourced_expand_evals() {
    for (shape, require_full, label) in [
        (&SMALL_SHAPE, true, "small achw eval"),
        (&BIG_SHAPE, false, "big achw eval"),
    ] {
        let source = chained_achw_source(shape);
        assert_clean(
            &check_json(&source),
            "chained rank-1 -> rank-4 shape-sourced expand checks clean",
        );
        let dir = tempdir().expect("tempdir");
        let eval = eval_stdout(dir.path(), &source, "issue_579_achw");
        let tensors = parse_printed_tensors(&eval);
        assert_achw_values(&tensors, shape, require_full, label);
    }
}

/// eval-vs-C-backend agreement on the chained corpus: `chelis build --target
/// c` must succeed, the compiled binary must produce the analytic values, and
/// every printed tensor must match the evaluator shape-for-shape and
/// value-for-value (on the printed prefix; the printers truncate long data).
#[test]
fn issue_579_chained_rank4_expand_c_backend_agrees() {
    let source = chained_achw_source(&BIG_SHAPE);
    let backend = build_compile_run(&source, "issue_579_achw_c");
    let tensors = parse_printed_tensors(&backend);
    assert_achw_values(&tensors, &BIG_SHAPE, false, "big achw backend");
    assert_eval_agrees_with_backend(&source, "issue_579_achw_c", &backend);
}

// ── Negatives: the sourceless issue spelling and genuine misuse ──────────

/// The batchnorm1d-flavor sourceless spelling from the issue must be rejected
/// at CHECK with the #469 sourceless-size diagnostic, and no error may carry
/// the original `rank mismatch` ICE text. (The rank-1 -> rank-4 chain's check
/// reject is pinned in rank_poly_tier3.rs; this pins the 1d flavor whose
/// 0.12.0 signature was `1 dims vs 2 dims`.)
#[test]
fn issue_579_sourceless_bn1d_expand_rejected_at_check_without_rank_ice() {
    let json = check_json(SOURCELESS_1D_SOURCE);
    let errors = check_errors(&json, "sourceless bn1d check");
    assert!(
        !errors.is_empty(),
        "sourceless bn1d expand must be rejected at check, got a clean check"
    );
    assert!(
        errors
            .iter()
            .any(|m| m.contains("no tensor in scope carries it") && m.contains("chelis#469")),
        "check rejection must cite the #469 sourceless-size rule, got {errors:?}"
    );
    assert!(
        errors.iter().all(|m| !m.contains("rank mismatch")),
        "the #579 rank-mismatch ICE text must not reappear in check errors: {errors:?}"
    );
}

/// The eval lane rejects both sourceless spellings (1d and the chained
/// rank-1 -> rank-4) with the same #469 diagnostic and NEVER the 0.12.0
/// `tensor rank mismatch: N dims vs M dims` ICE or an internal compiler
/// error. This is the exact #579 symptom pinned extinct.
#[test]
fn issue_579_sourceless_expand_rejects_in_eval_without_rank_ice() {
    for (source, name) in [
        (SOURCELESS_1D_SOURCE, "issue_579_sourceless_1d"),
        (SOURCELESS_ACHW_SOURCE, "issue_579_sourceless_achw"),
    ] {
        let dir = tempdir().expect("tempdir");
        let stderr = eval_stderr_expecting_failure(dir.path(), source, name);
        assert!(
            stderr.contains("no tensor in scope carries it") && stderr.contains("chelis#469"),
            "{name}: eval must reject with the #469 sourceless-size diagnostic, got: {stderr}"
        );
        assert!(
            !stderr.contains("rank mismatch"),
            "{name}: the 0.12.0 rank-mismatch ICE must not reappear: {stderr}"
        );
        assert!(
            !stderr.contains("internal compiler error"),
            "{name}: the reject must be a clean diagnostic, not an ICE: {stderr}"
        );
    }
}

/// Negative parity for the positive bn1d case: broadcasting over the WRONG
/// axis (extent read from `x` axis 0 but inserted at axis 1, ascribed
/// `[3, 2]`) makes the following `mul` shape-invalid and must be rejected at
/// check with a dimension-mismatch reason.
#[test]
fn issue_579_wrong_axis_broadcast_rejected_at_check() {
    let source = "def bad(x: &tensor[2, 3, f32], g: &tensor[3, f32]) -> tensor[2, 3, f32] = {\n\
        \x20 gb: tensor[3, 2, f32] = expand(g, 1, shape(x, cast(0, int32)))\n\
        \x20 mul(x, gb)\n\
        }\n\
        out = bad(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), to_tensor([10.0, 20.0, 30.0]))\n";
    let json = check_json(source);
    let errors = check_errors(&json, "wrong-axis broadcast check");
    assert!(
        errors.iter().any(|m| m.contains("dimension mismatch")),
        "wrong-axis broadcast must fail check with a dimension mismatch, got {errors:?}"
    );
}

/// Negative parity for the chained case: an out-of-bounds insert axis on a
/// genuinely rank-1 operand must fail eval with the targeted expand
/// diagnostic, never the #579 rank-monomorphization ICE and never a silent
/// wrong-shape success.
#[test]
fn issue_579_out_of_bounds_insert_axis_fails_eval_with_targeted_reason() {
    let source = "def bad(g: &tensor[3, f32]) -> tensor[3, 2, 2, f32] = expand(g, 3, 2)\n\
        out = bad(to_tensor([1.0, 2.0, 3.0]))\n";
    let dir = tempdir().expect("tempdir");
    let stderr = eval_stderr_expecting_failure(dir.path(), source, "issue_579_axis_oob");
    assert!(
        stderr.contains("expand") && stderr.contains("out of bounds"),
        "out-of-bounds insert axis must fail with the targeted expand reason, got: {stderr}"
    );
    assert!(
        !stderr.contains("internal compiler error"),
        "out-of-bounds insert axis must not ICE: {stderr}"
    );
}
