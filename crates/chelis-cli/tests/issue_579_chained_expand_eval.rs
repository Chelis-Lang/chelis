//! chelis#579 (residue of chelis#397): school's BatchNorm broadcasts
//! (batchnorm1d rank-1 -> rank-2, batchnorm2d `broadcast_to_achw` rank-1 ->
//! rank-4) were reported check-clean and `reef build`-clean at chelis 0.12.0
//! yet failing `chelis test`/eval with the rank-monomorphization ICE
//! `tensor rank mismatch: 1 dims vs N dims`.
//!
//! Empirical v0.12.0 baseline (established by building the v0.12.0 tag and
//! running this corpus against it during review):
//!
//! - School's REAL batchnorm extent spelling, a LET-BOUND
//!   `cast(shape(x, axis), i64)` read passed to `expand`, was check-clean
//!   AND eval-clean at v0.12.0 but REJECTED at `chelis build` as sourceless:
//!   a §4.7.2 Form-3 check-accept/build-reject asymmetry. PR #596 closed it
//!   by following let-bound `shape` reads to their source tensor. The
//!   `issue_579_let_bound_*` tests below FAIL on a pre-#596 compiler and are
//!   this file's #596 discriminator.
//! - The bare-runtime-scalar spellings (school's pre-0.12-bump
//!   `broadcast_to_achw(v, h_dim, w_dim, a_dim)` signature) were rejected at
//!   check from v0.12.0 by the #494 source-tracking predicate, which is what
//!   forced school's §4.7.2 witness rewrite at its 0.12.0 pin bump.
//!   chelis#469 removed that predicate (section 4.7.2 admits any `i64` size),
//!   so they are pinned here executing on both lanes, never regressing into
//!   the reported rank-mismatch ICE.
//! - The inline shape-sourced chain (`expand(g, 1, shape(x, cast(2,
//!   i32)))`) already checked, evaluated, and built correctly at v0.12.0.
//!   Its chained positive lane (eval + C parity) was previously untested and
//!   is pinned here as the explicit working baseline.
//!
//! The issue's own symptom, `chelis test` dying with the rank-mismatch ICE
//! inside the school repo at 0.12.0, has NOT been reproduced upstream: at
//! the v0.12.0 tag, school's exact spellings pass check, eval, and a minimal
//! `chelis test` reef in isolation. The school-context trigger remains
//! unisolated, so school's `tests_blocked/nn/batchnorm.ch` probe at its next
//! pin bump is the closing oracle for chelis#579 itself; this corpus pins
//! the adjacent upstream lanes so the failure class cannot silently return.
//!
//! Sibling coverage: `rank_poly_tier3.rs` pins the scalar-parameter rank-1 ->
//! rank-4 chain and the single-expand `bias_broadcast` positives. This file
//! adds the let-bound (school-real) spelling across check/eval/build/C-parity,
//! the missing CHAINED positive lane (eval + C parity), and the
//! batchnorm1d-flavor scalar-parameter pins.

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
const BN1D_SOURCE: &str = "def bn1d_scale[a, n](x: &tensor[a, n, f32], g: &tensor[n, f32]) -> tensor[a, n, f32] = {\n\
    \x20 gb: tensor[a, n, f32] = insert(g, 0, shape(x, cast(0, i32)))\n\
    \x20 mul(x, gb)\n\
    }\n\
    xs = to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\n\
    gs = to_tensor([10.0, 20.0, 30.0])\n\
    out = bn1d_scale(xs, gs)\n";

/// School's REAL batchnorm1d extent spelling (src/nn/batchnorm.ch):
/// `a_dim = cast(shape(x, cast(0, i32)), i64)` LET-BOUND, then
/// `expand(g, 0, a_dim)`. At the v0.12.0 tag this exact program was
/// check-clean AND eval-clean but `chelis build` rejected the extent as
/// sourceless (the §4.7.2 check-accept/build-reject asymmetry #596 closed
/// by following let-bound `shape` reads to their source tensor). The
/// let-bound tests below fail on a pre-#596 compiler.
const BN1D_LET_BOUND_SOURCE: &str = "def bn1d_scale[a, n](x: &tensor[a, n, f32], g: &tensor[n, f32]) -> tensor[a, n, f32] = {\n\
    \x20 a_dim = cast(shape(x, cast(0, i32)), i64)\n\
    \x20 gb: tensor[a, n, f32] = insert(g, 0, a_dim)\n\
    \x20 mul(x, gb)\n\
    }\n\
    xs = to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\n\
    gs = to_tensor([10.0, 20.0, 30.0])\n\
    out = bn1d_scale(xs, gs)\n";

/// How the chained corpus spells its runtime extents.
#[derive(Clone, Copy)]
enum ExtentSpelling {
    /// `expand(g, 1, shape(x, cast(2, i32)))`: the `bias_broadcast` form.
    /// Already worked end-to-end at v0.12.0; pinned as the working baseline.
    Inline,
    /// School's batchnorm form: `h_dim = cast(shape(x, cast(2, i32)),
    /// i64)` let-bound, then `expand(g, 1, h_dim)`. Check/eval-clean but
    /// build-REJECTED as sourceless at v0.12.0; fixed by #596.
    LetBound,
}

/// The supported spelling of school's `broadcast_to_achw` (batchnorm2d):
/// three CHAINED Form-3 expands, every extent read from the in-scope source
/// tensor `x` via `shape(x, axis)` (inline or let-bound per `spelling`).
/// `out` is the raw rank-1 -> rank-4 broadcast; `out_affine` is the
/// batchnorm2d-forward affine composition (two broadcasts through the same
/// helper, then mul + add). `x` carries sequential data so the affine values
/// encode flat position: any axis permutation in evaluation changes them and
/// cannot hide.
fn chained_achw_source(shape: &[usize; 4], spelling: ExtentSpelling) -> String {
    let mut next = 1.0f64;
    let xs = nested_literal(shape, &mut next);
    let (lets, h_size, w_size, a_size) = match spelling {
        ExtentSpelling::Inline => (
            "",
            "shape(x, cast(2, i32))",
            "shape(x, cast(3, i32))",
            "shape(x, cast(0, i32))",
        ),
        ExtentSpelling::LetBound => (
            "\x20 h_dim = cast(shape(x, cast(2, i32)), i64)\n\
             \x20 w_dim = cast(shape(x, cast(3, i32)), i64)\n\
             \x20 a_dim = cast(shape(x, cast(0, i32)), i64)\n",
            "h_dim",
            "w_dim",
            "a_dim",
        ),
    };
    format!(
        "def broadcast_to_achw[c, a, h, w](g: &tensor[c, f32], x: &tensor[a, c, h, w, f32]) -> tensor[a, c, h, w, f32] = {{\n\
        {lets}\
        \x20 step1: tensor[c, h, f32] = insert(g, 1, {h_size})\n\
        \x20 step2: tensor[c, h, w, f32] = insert(step1, 2, {w_size})\n\
        \x20 step3: tensor[a, c, h, w, f32] = insert(step2, 0, {a_size})\n\
        \x20 step3\n\
        }}\n\
        def bn2d_affine[a, c, h, w](x: &tensor[a, c, h, w, f32], g: &tensor[c, f32], b: &tensor[c, f32]) -> tensor[a, c, h, w, f32] = {{\n\
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

/// Bare-runtime-scalar spelling, batchnorm1d flavor: the expand size is a
/// bare `i64` parameter with no tensor source. NOT school's shipped spelling
/// (that one is `BN1D_LET_BOUND_SOURCE`); check-rejected from v0.12.0 by the
/// #494 source-tracking predicate until chelis#469 removed it.
const SCALAR_PARAM_1D_SOURCE: &str = "def bcast_1d_to_2d[a, n](g: tensor[n, f32], a_dim: i64) -> tensor[a, n, f32] = insert(g, 0, a_dim)\n\
    out = bcast_1d_to_2d(to_tensor([1.0, 2.0, 3.0]), cast(2, i64))\n";

/// The chained rank-1 -> rank-4 scalar-parameter spelling: school's
/// PRE-0.12-bump `broadcast_to_achw` signature (bare `i64` dim params),
/// retired in school's §4.7.2 witness rewrite when the 0.12.0 pin bump brought
/// in the #494 check reject, and admissible again since chelis#469. School's
/// intermediate `step1: tensor[c, h, f32]` ascriptions are left out: a local
/// ascription naming a binder that only the declared result introduces (`h`
/// here) is refused at lowering on both lanes ("cannot resolve authored
/// extent"), a local-ascription gap separate from the size rule this file pins.
const SCALAR_PARAM_ACHW_SOURCE: &str = "def broadcast_to_achw[c, h, w, a](v: &tensor[c, f32], h_dim: i64, w_dim: i64, a_dim: i64) -> tensor[a, c, h, w, f32] = {\n\
    \x20 step1 = insert(v, 1, h_dim)\n\
    \x20 step2 = insert(step1, 2, w_dim)\n\
    \x20 insert(step2, 0, a_dim)\n\
    }\n\
    out = broadcast_to_achw(to_tensor([1.0, 2.0]), cast(3, i64), cast(4, i64), cast(5, i64))\n";

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
            "--emit-c",
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
        .arg("libchelis_runtime.a")
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

/// Assert the bn1d corpus `out` tensor: xs `[[1,2,3],[4,5,6]]` scaled
/// per-column by g `[10,20,30]`.
fn assert_bn1d_out(tensors: &[(String, Vec<usize>, Vec<f64>)], label: &str) {
    let out = find_tensor(tensors, "out", label);
    assert_eq!(out.1, vec![2, 3], "{label}: bn1d broadcast shape");
    let expected = [10.0, 40.0, 90.0, 40.0, 100.0, 180.0];
    assert_eq!(out.2.len(), expected.len(), "{label}: bn1d element count");
    for (i, e) in expected.iter().enumerate() {
        assert!(
            (out.2[i] - e).abs() < 1e-6,
            "{label}: bn1d out[{i}]: {} != {e}",
            out.2[i]
        );
    }
}

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
    assert_bn1d_out(&parse_printed_tensors(&eval), "bn1d eval");
}

/// School's REAL bn1d spelling (let-bound `cast(shape(x, 0), i64)` extent)
/// through EVERY lane: checks clean, evaluates to the exact values, C build
/// succeeds, the compiled binary produces the same values, and eval agrees
/// with the backend. At the v0.12.0 tag this exact program was check-clean
/// and eval-clean but `chelis build` REJECTED the extent as sourceless; #596
/// taught lowering to follow let-bound `shape` reads to their source. This
/// test FAILS on a pre-#596 compiler (verified against the v0.12.0 tag).
#[test]
fn issue_579_let_bound_shape_extent_bn1d_builds_and_evals() {
    assert_clean(
        &check_json(BN1D_LET_BOUND_SOURCE),
        "let-bound bn1d checks clean",
    );
    let dir = tempdir().expect("tempdir");
    let eval = eval_stdout(
        dir.path(),
        BN1D_LET_BOUND_SOURCE,
        "issue_579_bn1d_let_bound",
    );
    assert_bn1d_out(&parse_printed_tensors(&eval), "let-bound bn1d eval");
    let backend = build_compile_run(BN1D_LET_BOUND_SOURCE, "issue_579_bn1d_let_bound_c");
    assert_bn1d_out(&parse_printed_tensors(&backend), "let-bound bn1d backend");
    assert_eval_agrees_with_backend(
        BN1D_LET_BOUND_SOURCE,
        "issue_579_bn1d_let_bound_c",
        &backend,
    );
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

/// The chained rank-1 -> rank-4 `broadcast_to_achw` in the inline
/// shape-sourced spelling checks clean and EVALUATES, with the exact
/// per-channel broadcast values, and the two-broadcast batchnorm2d affine
/// composition on top of it evaluates too. This spelling already worked at
/// v0.12.0; it is the working baseline the #579 failure class is pinned
/// against (the issue reported the chain dying in eval at 0.12.0 in school's
/// module context). Small corpus: complete value coverage. Big corpus:
/// distinct dims pin the shape against axis/source mixups.
#[test]
fn issue_579_chained_rank4_shape_sourced_expand_evals() {
    for (shape, require_full, label) in [
        (&SMALL_SHAPE, true, "small achw eval"),
        (&BIG_SHAPE, false, "big achw eval"),
    ] {
        let source = chained_achw_source(shape, ExtentSpelling::Inline);
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

/// The chained corpus in school's LET-BOUND extent spelling (the batchnorm2d
/// train-mode form) checks clean and evaluates with the exact per-channel
/// values on both corpora. Check and eval were already clean at v0.12.0 for
/// this spelling; the build-lane discriminator is the companion
/// `issue_579_let_bound_shape_extent_chained_c_backend_agrees`.
#[test]
fn issue_579_let_bound_shape_extent_chained_evals() {
    for (shape, require_full, label) in [
        (&SMALL_SHAPE, true, "small let-bound achw eval"),
        (&BIG_SHAPE, false, "big let-bound achw eval"),
    ] {
        let source = chained_achw_source(shape, ExtentSpelling::LetBound);
        assert_clean(
            &check_json(&source),
            "chained let-bound shape-extent expand checks clean",
        );
        let dir = tempdir().expect("tempdir");
        let eval = eval_stdout(dir.path(), &source, "issue_579_achw_let_bound");
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
    let source = chained_achw_source(&BIG_SHAPE, ExtentSpelling::Inline);
    let backend = build_compile_run(&source, "issue_579_achw_c");
    let tensors = parse_printed_tensors(&backend);
    assert_achw_values(&tensors, &BIG_SHAPE, false, "big achw backend");
    assert_eval_agrees_with_backend(&source, "issue_579_achw_c", &backend);
}

/// The #596 discriminator, chained flavor: the LET-BOUND spelling must BUILD
/// on the C backend (at v0.12.0 it was check/eval-clean but build-rejected
/// as sourceless), the compiled binary must produce the analytic values, and
/// eval must agree with the backend value-for-value.
#[test]
fn issue_579_let_bound_shape_extent_chained_c_backend_agrees() {
    let source = chained_achw_source(&BIG_SHAPE, ExtentSpelling::LetBound);
    let backend = build_compile_run(&source, "issue_579_achw_let_bound_c");
    let tensors = parse_printed_tensors(&backend);
    assert_achw_values(&tensors, &BIG_SHAPE, false, "big let-bound achw backend");
    assert_eval_agrees_with_backend(&source, "issue_579_achw_let_bound_c", &backend);
}

// ── The bare-scalar spellings (chelis#469), and genuine misuse ───────────

/// The batchnorm1d-flavor bare-scalar spelling checks clean (chelis#469), and
/// no diagnostic carries the `rank mismatch` ICE text the issue reported.
#[test]
fn issue_579_scalar_param_bn1d_expand_checks_clean_without_rank_ice() {
    let json = check_json(SCALAR_PARAM_1D_SOURCE);
    let errors = check_errors(&json, "scalar-parameter bn1d check");
    assert!(
        errors.is_empty(),
        "a scalar-parameter bn1d expand size is admissible under section 4.7.2, got {errors:?}"
    );
}

/// Both bare-scalar spellings (1d and the chained rank-1 -> rank-4) build,
/// and the evaluator agrees with the compiled C, with NEVER the `tensor rank
/// mismatch: N dims vs M dims` ICE the issue reported (chelis#469).
#[test]
fn issue_579_scalar_param_expand_agrees_in_eval_without_rank_ice() {
    for (source, name, shape) in [
        (
            SCALAR_PARAM_1D_SOURCE,
            "issue_579_scalar_param_1d",
            vec![2, 3],
        ),
        (
            SCALAR_PARAM_ACHW_SOURCE,
            "issue_579_scalar_param_achw",
            vec![5, 2, 3, 4],
        ),
    ] {
        let backend = build_compile_run(source, name);
        let tensors = parse_printed_tensors(&backend);
        let out = tensors
            .iter()
            .find(|(n, _, _)| n == "out")
            .unwrap_or_else(|| panic!("{name}: backend output missing `out`: {backend}"));
        assert_eq!(out.1, shape, "{name}: the runtime extents size the result");
        assert_eval_agrees_with_backend(source, name, &backend);
    }
}

/// Negative parity for the positive bn1d case: broadcasting over the WRONG
/// axis (extent read from `x` axis 0 but inserted at axis 1, ascribed
/// `[3, 2]`) makes the following `mul` shape-invalid and must be rejected at
/// check with a dimension-mismatch reason.
#[test]
fn issue_579_wrong_axis_broadcast_rejected_at_check() {
    let source = "def bad(x: &tensor[2, 3, f32], g: &tensor[3, f32]) -> tensor[2, 3, f32] = {\n\
        \x20 gb: tensor[3, 2, f32] = insert(g, 1, shape(x, cast(0, i32)))\n\
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
/// genuinely rank-1 operand must fail eval with the targeted insert
/// diagnostic, never the #579 rank-monomorphization ICE and never a silent
/// wrong-shape success.
#[test]
fn issue_579_out_of_bounds_insert_axis_fails_eval_with_targeted_reason() {
    let source = "def bad(g: &tensor[3, f32]) -> tensor[3, 2, 2, f32] = insert(g, 3, 2i64)\n\
        out = bad(to_tensor([1.0, 2.0, 3.0]))\n";
    let dir = tempdir().expect("tempdir");
    let stderr = eval_stderr_expecting_failure(dir.path(), source, "issue_579_axis_oob");
    assert!(
        stderr.contains("insert") && stderr.contains("out of bounds"),
        "out-of-bounds insert axis must fail with the targeted insert reason, got: {stderr}"
    );
    assert!(
        !stderr.contains("internal compiler error"),
        "out-of-bounds insert axis must not ICE: {stderr}"
    );
}
