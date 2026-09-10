//! chelis#616 step 3a: forward eval-vs-C parity for runtime (node-valued)
//! MOVEMENT bounds — `shrink`/`stride` whose bounds are `shape()`-derived
//! integer arithmetic, with no runtime reshape target in the chain.
//!
//! Before the op-declared symbolic-dim source landed, `chelis build
//! --target c` of such a program ICE'd in `symbolic_occurrences` ("no Load
//! input declares it"): the movement op's runtime output extent had no
//! declaring Load, and the C emitter had no other way to declare the output
//! dim. Now the owning op declares it inline (`int <dim> = <extent>;`) from
//! the same bound scalars the eval lane resolves, so one compiled binary
//! handles every input length and the two lanes must agree exactly.
//!
//! The verb: `shrink(x, [[1i64, n - 1]])` keeps `[2, ..., n-1]` for input
//! `[1, 2, ..., n]` (with `n = shape(x, 0)` read at run time); `stride(_, 2i64)`
//! then keeps every other element: `[2, 4, 6, ...]`.

use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;

use assert_cmd::Command;
use tempfile::{TempDir, tempdir};

/// Runtime-shrink-only verb; its single movement output dim is the sig's `u`.
const SHRINK_BODY: &str = "\
  extent = cast(sub(cast(shape(x, cast(0, int32)), int64), cast(1, int64)), int64)\n\
  shrink(x, [[cast(1, int64), extent]])";

/// The full runtime shrink -> stride chain, anchored by a static reshape so
/// each movement output keeps its own (anonymous) dim. This is the
//  movement-op shape of the chelis#616 avgpool oracle's `window_row`.
const CHAIN_ANCHORED_BODY: &str = "\
  extent = cast(sub(cast(shape(x, cast(0, int32)), int64), cast(1, int64)), int64)\n\
  w = stride(shrink(x, [[cast(1, int64), extent]]), cast(2, int64))\n\
  reshape(w, [cast(2, int64)])";

fn f32_literal(values: &[f64]) -> String {
    values
        .iter()
        .map(|v| format!("cast({v:?}, f32)"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn run_eval(source: &str, stem: &str) -> std::process::Output {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{stem}.ch"));
    fs::write(&path, source).expect("write");
    Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("run chelis eval")
}

/// Parse the `tensor(shape=[...], data=[...])` line from eval / driver output.
fn parse_tensor_data(stdout: &str) -> Vec<f64> {
    let line = stdout
        .lines()
        .find(|l| l.contains("data=["))
        .unwrap_or_else(|| panic!("no tensor in output: {stdout}"));
    line.split_once("data=[")
        .and_then(|(_, r)| r.split_once(']'))
        .map(|(s, _)| {
            s.split(',')
                .filter(|t| !t.trim().is_empty())
                .map(|t| t.trim().parse::<f64>().expect("tensor element"))
                .collect()
        })
        .unwrap()
}

fn assert_close(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: length mismatch got {got:?} want {want:?}"
    );
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-3,
            "{label}: elem {i}: got {g}, want {w} (full got={got:?} want={want:?})"
        );
    }
}

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

fn gcc(
    build_dir: &Path,
    stem: &str,
    extra_src: Option<&Path>,
    bin_name: &str,
) -> std::path::PathBuf {
    let kernel = build_dir.join(format!("{stem}.c"));
    let runtime = build_dir.join("libchelis_runtime.a");
    let bin = build_dir.join(bin_name);
    let mut args = vec![
        "-O0".to_string(),
        "-std=c11".to_string(),
        "-I".to_string(),
        build_dir.to_str().unwrap().to_string(),
        kernel.to_str().unwrap().to_string(),
    ];
    if let Some(extra) = extra_src {
        args.push(extra.to_str().unwrap().to_string());
    }
    args.extend([
        "-o".to_string(),
        bin.to_str().unwrap().to_string(),
        runtime.to_str().unwrap().to_string(),
        "-lm".to_string(),
        "-lpthread".to_string(),
        "-ldl".to_string(),
    ]);
    let compile = StdCommand::new("gcc")
        .args(&args)
        .output()
        .expect("invoke gcc");
    assert!(
        compile.status.success(),
        "gcc compile failed: stderr={}",
        String::from_utf8_lossy(&compile.stderr)
    );
    bin
}

/// eval-vs-C forward parity for the anchored shrink -> stride chain at
/// n = 6: both lanes compute `[2, 4]` by resolving the shrink end (`n - 1`)
/// and the stride extent at run time. Each movement op declares its own
/// fresh runtime dim in the emitted C (the copy-first-input dim shortcut
/// must not propagate the shrink's extent onto the stride).
#[test]
fn issue_616_runtime_shrink_stride_forward_eval_matches_c() {
    let input: Vec<f64> = (1..=6).map(|v| v as f64).collect();
    let source = format!(
        "module Repro.RtChain\nsig f: tensor[n, f32] -> tensor[2, f32]\ndef f(x) = {{\n{CHAIN_ANCHORED_BODY}\n}}\nout = f(to_tensor([{}]))\n",
        f32_literal(&input)
    );

    let eval_out = run_eval(&source, "rtchain");
    assert!(
        eval_out.status.success(),
        "forward eval failed: {}",
        String::from_utf8_lossy(&eval_out.stderr)
    );
    let eval_values = parse_tensor_data(&String::from_utf8_lossy(&eval_out.stdout));
    assert_close("eval forward", &eval_values, &[2.0, 4.0]);

    let (_dir, build_dir) = build_c(&source, "rtchain");
    let bin = gcc(&build_dir, "rtchain", None, "self_bin");
    let run = StdCommand::new(&bin).output().expect("run emitted program");
    assert!(
        run.status.success(),
        "emitted program exited non-zero: stdout={} stderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let c_values = parse_tensor_data(&String::from_utf8_lossy(&run.stdout));
    assert_close("eval-vs-C n=6", &c_values, &eval_values);
}

/// The definitive runtime-extent oracle: ONE compiled binary (the exported
/// `def out(x)`, runtime shrink only), fed n = 4, 5, 9, sizes its output
/// from the input's runtime extent each call — nothing about the movement
/// bounds is codegen-baked. The sig-named `u` output dim is declared inline
/// at the shrink from its own bound scalars.
#[test]
fn issue_616_runtime_shrink_c_binary_handles_multiple_lengths() {
    let source = format!(
        "module Repro.RtShrink\nsig out: tensor[n, f32] -> tensor[u, f32]\ndef out(x) = {{\n{SHRINK_BODY}\n}}\n"
    );
    let (_dir, build_dir) = build_c(&source, "rtshrink");

    let lengths = [4usize, 5, 9];
    let runs = lengths
        .iter()
        .map(|n| {
            format!(
                "    {{ int64_t shape[1] = {{{n}}}; chelis_tensor* x = chelis_alloc(1, shape, CHELIS_DTYPE_F32); \
                 chelis_tensor_write* x_guard = chelis_tensor_begin_write(x); chelis_write_view x_view = chelis_tensor_write_view(x_guard); \
                 for (int i = 0; i < {n}; i++) ((float *)x_view.data)[i] = (float)(i + 1); chelis_tensor_end_write(x_guard); \
                 chelis_tensor* w = out(x); chelis_read_view w_view = chelis_tensor_read_view(w); \
                 for (int64_t i = 0; i < w_view.count; i++) printf(\"%.6f\\n\", ((const float *)w_view.data)[i]); \
                 printf(\"---\\n\"); chelis_tensor_release(w); chelis_tensor_release(x); }}"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let driver_src = format!(
        r#"
#include <stdio.h>
#include "chelis_runtime.h"
extern chelis_tensor* out(chelis_tensor* arg0);
int main(void) {{
{runs}
    return 0;
}}
"#
    );
    let driver = build_dir.join("driver.c");
    fs::write(&driver, driver_src).expect("write driver.c");
    let bin = gcc(&build_dir, "rtshrink", Some(&driver), "driver_bin");
    let run = StdCommand::new(&bin).output().expect("run driver binary");
    assert!(
        run.status.success(),
        "driver binary exited non-zero: stdout={} stderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let blocks: Vec<Vec<f64>> = String::from_utf8_lossy(&run.stdout)
        .split("---")
        .filter(|blk| !blk.trim().is_empty())
        .map(|blk| {
            blk.lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| l.trim().parse::<f64>().expect("window element"))
                .collect()
        })
        .collect();
    assert_eq!(blocks.len(), lengths.len(), "one window per input length");
    for (n, got) in lengths.iter().zip(blocks.iter()) {
        // shrink [[1, n-1]] over [1..n] keeps the values 2..=n-1.
        let want: Vec<f64> = (2..*n).map(|v| v as f64).collect();
        assert_close(&format!("C shrink window n={n}"), got, &want);
    }
}

/// The formerly-guarded degenerate (chelis#632, flipped by the chelis#631
/// change set): a shrink -> stride chain returned DIRECTLY under a
/// sig-named output dim. Through v0.16.x the sig symbol `u` was painted
/// onto BOTH movement outputs by the wildcard-KEYED dim substitution
/// (`tensor_dim_substitutions` recorded `"*" -> u` and repainted every
/// wildcard-typed node), so the C binary declared `u` at the shrink
/// (`n-2`) and equality-guarded it at the stride (`ceil((n-2)/2)`) — a
/// loud abort on a well-formed program while eval computed the correct
/// window. Anon dims are no longer substitution keys, and the helper
/// root is retyped POSITIONALLY on op-declarable axes only: the inner
/// shrink keeps its own per-node anon extent, the stride root declares
/// `u` from its own extent, and the chain has full eval-vs-C parity.
/// The checker-side residue of chelis#632 (movement typing passes a
/// symbolic dim through non-identity axes — the annotation-level lie)
/// is tracked separately on that issue.
#[test]
fn issue_632_direct_return_movement_chain_eval_matches_c() {
    let input: Vec<f64> = (1..=6).map(|v| v as f64).collect();
    let body = "\
  extent = cast(sub(cast(shape(x, cast(0, int32)), int64), cast(1, int64)), int64)\n\
  stride(shrink(x, [[cast(1, int64), extent]]), cast(2, int64))";
    let source = format!(
        "module Repro.RtDegenerate\nsig f: tensor[n, f32] -> tensor[u, f32]\ndef f(x) = {{\n{body}\n}}\nout = f(to_tensor([{}]))\n",
        f32_literal(&input)
    );

    let eval_out = run_eval(&source, "rtdegen");
    assert!(
        eval_out.status.success(),
        "eval computes the true window: {}",
        String::from_utf8_lossy(&eval_out.stderr)
    );
    let eval_values = parse_tensor_data(&String::from_utf8_lossy(&eval_out.stdout));
    assert_close("eval degenerate chain", &eval_values, &[2.0, 4.0]);

    let (_dir, build_dir) = build_c(&source, "rtdegen");
    let bin = gcc(&build_dir, "rtdegen", None, "self_bin");
    let run = StdCommand::new(&bin).output().expect("run emitted program");
    assert!(
        run.status.success(),
        "C binary must run the direct-return movement chain to success; stderr={}",
        String::from_utf8_lossy(&run.stderr)
    );
    let c_values = parse_tensor_data(&String::from_utf8_lossy(&run.stdout));
    assert_close("C degenerate chain", &c_values, &eval_values);
}

/// chelis#632 (checker side): a LITERAL non-identity stride returned
/// directly under distinct sig symbols. Before the fresh-extent fix the
/// checker's pass-through arm unified `u := n` and the §4.4.1 return-dim
/// rigidity guard REJECTED this well-formed program at check time; now
/// axis 0 mints a fresh extent, the program checks, and both lanes agree
/// on `[1, 3, 5]`.
#[test]
fn issue_632_literal_stride_under_sig_symbols_matches_c() {
    let input: Vec<f64> = (1..=6).map(|v| v as f64).collect();
    let source = format!(
        "module Repro.LitStrideSig\nsig f: tensor[n, f32] -> tensor[u, f32]\ndef f(x) = stride(x, cast(2, int64))\nout = f(to_tensor([{}]))\n",
        f32_literal(&input)
    );

    let eval_out = run_eval(&source, "litstride");
    assert!(
        eval_out.status.success(),
        "eval must accept the sig-symbol literal stride: {}",
        String::from_utf8_lossy(&eval_out.stderr)
    );
    let eval_values = parse_tensor_data(&String::from_utf8_lossy(&eval_out.stdout));
    assert_close("eval literal stride", &eval_values, &[1.0, 3.0, 5.0]);

    let (_dir, build_dir) = build_c(&source, "litstride");
    let bin = gcc(&build_dir, "litstride", None, "self_bin");
    let run = StdCommand::new(&bin).output().expect("run emitted program");
    assert!(
        run.status.success(),
        "C binary must run the sig-symbol literal stride; stderr={}",
        String::from_utf8_lossy(&run.stderr)
    );
    let c_values = parse_tensor_data(&String::from_utf8_lossy(&run.stdout));
    assert_close("C literal stride", &c_values, &eval_values);
}

/// RED-TEAM FINDING 1 (chelis#616): a MULTI-AXIS shrink mixing a runtime
/// axis with a literal-bounded axis. Type inference used to collapse EVERY
/// axis to a wildcard when any bound was non-literal; downstream
/// unification then filled the runtime axis's extent from the sibling
/// literal axis and the C backend baked the wrong extent unguarded — a
/// SILENT mis-size (C returned shape [3,3] with fabricated values where
/// eval computed [2,3]). Per-axis inference keeps the literal axis precise
/// and only the runtime axis symbolic; both lanes must now agree exactly.
#[test]
fn issue_616_multi_axis_runtime_shrink_matches_c() {
    let source = "module Repro.MatSlice\n\
sig f: tensor[rows, 5, f32] -> tensor[rows, 3, f32]\n\
def f(x) = {\n\
  rows = cast(shape(x, cast(0, int32)), int64)\n\
  shrink(x, [[cast(0, int64), rows], [cast(1, int64), cast(4, int64)]])\n\
}\n\
out = f(to_tensor([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32), cast(5.0, f32)], [cast(6.0, f32), cast(7.0, f32), cast(8.0, f32), cast(9.0, f32), cast(10.0, f32)]]))\n";

    let eval_out = run_eval(source, "matslice");
    assert!(
        eval_out.status.success(),
        "matslice eval failed: {}",
        String::from_utf8_lossy(&eval_out.stderr)
    );
    let eval_values = parse_tensor_data(&String::from_utf8_lossy(&eval_out.stdout));
    let want = [2.0, 3.0, 4.0, 7.0, 8.0, 9.0];
    assert_close("eval matslice", &eval_values, &want);

    let (_dir, build_dir) = build_c(source, "matslice");
    let bin = gcc(&build_dir, "matslice", None, "self_bin");
    let run = StdCommand::new(&bin).output().expect("run emitted program");
    assert!(
        run.status.success(),
        "matslice C binary exited non-zero: stdout={} stderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(
        stdout.contains("shape=[2, 3]"),
        "C matslice must keep the true [2, 3] shape, never a baked sibling \
         extent; stdout={stdout}"
    );
    let c_values = parse_tensor_data(&stdout);
    assert_close("eval-vs-C matslice", &c_values, &want);
}

/// RED-TEAM FINDING 1 sibling: the same per-axis mixing through a
/// multi-axis PAD (runtime after-pad on axis 0, literal pads on axis 1).
#[test]
fn issue_616_multi_axis_runtime_pad_matches_c() {
    let source = "module Repro.MatPad\n\
sig f: tensor[rows, 3, f32] -> tensor[u, 5, f32]\n\
def f(x) = {\n\
  k = cast(shape(x, cast(0, int32)), int64)\n\
  pad(x, [[cast(0, int64), k], [cast(1, int64), cast(1, int64)]], cast(0.0, f32))\n\
}\n\
out = f(to_tensor([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)], [cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)]]))\n";

    let eval_out = run_eval(source, "matpad");
    assert!(
        eval_out.status.success(),
        "matpad eval failed: {}",
        String::from_utf8_lossy(&eval_out.stderr)
    );
    let eval_values = parse_tensor_data(&String::from_utf8_lossy(&eval_out.stdout));

    let (_dir, build_dir) = build_c(source, "matpad");
    let bin = gcc(&build_dir, "matpad", None, "self_bin");
    let run = StdCommand::new(&bin).output().expect("run emitted program");
    assert!(
        run.status.success(),
        "matpad C binary exited non-zero: stdout={} stderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(
        stdout.contains("shape=[4, 5]"),
        "C matpad must produce the true [4, 5] shape; stdout={stdout}"
    );
    let c_values = parse_tensor_data(&stdout);
    assert_close("eval-vs-C matpad", &c_values, &eval_values);
}

/// Negative parity for the per-axis fix: a LITERAL pair alongside a runtime
/// pair keeps its precise infer-time validation — an out-of-range literal
/// bound is rejected at `chelis check`, not deferred to run time just
/// because a sibling axis is runtime.
#[test]
fn issue_616_literal_axis_still_checked_beside_runtime_axis() {
    let source = "module Repro.MatSliceBad\n\
sig f: tensor[rows, 5, f32] -> tensor[rows, 3, f32]\n\
def f(x) = {\n\
  rows = cast(shape(x, cast(0, int32)), int64)\n\
  shrink(x, [[cast(0, int64), rows], [cast(1, int64), cast(9, int64)]])\n\
}\n\
out = f(to_tensor([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32), cast(5.0, f32)], [cast(6.0, f32), cast(7.0, f32), cast(8.0, f32), cast(9.0, f32), cast(10.0, f32)]]))\n";
    let output = run_eval(source, "matslicebad");
    assert!(
        !output.status.success(),
        "the out-of-range literal bound [1, 9] on the 5-wide axis must be \
         rejected even with a runtime sibling axis; stdout={}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("out of range"),
        "rejection must name the out-of-range literal bound; stderr={stderr}"
    );
}

/// RED-TEAM FINDING 2 (chelis#616): grad of a runtime shrink feeding a
/// reduction DIRECTLY (no interposed reshape). The backward Sum-adjoint
/// restore is an `Expand` over the runtime axis; the C lane used to ICE
/// ("no Load input declares it") because the expanded-axis extent had no
/// declaration source. It is now op-declared from the Expand's shape-dep
/// (the forward input's actual shape) and both lanes agree.
#[test]
fn issue_616_runtime_shrink_grad_through_reduction_matches_c() {
    let source = "module Repro.RtShrinkGrad\n\
sig f: tensor[5, f32] -> f32\n\
def f(x) = {\n\
  e = cast(sub(cast(shape(x, cast(0, int32)), int64), cast(1, int64)), int64)\n\
  w = shrink(x, [[cast(1, int64), e]])\n\
  sum(w, cast(0, int32)) |> tensor_to_scalar\n\
}\n\
out = grad(f)(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32), cast(5.0, f32)]))\n";

    let eval_out = run_eval(source, "rtshrinkgrad");
    assert!(
        eval_out.status.success(),
        "runtime shrink grad eval failed: {}",
        String::from_utf8_lossy(&eval_out.stderr)
    );
    let eval_grad = parse_tensor_data(&String::from_utf8_lossy(&eval_out.stdout));
    assert_close("eval shrink grad", &eval_grad, &[0.0, 1.0, 1.0, 1.0, 0.0]);

    let (_dir, build_dir) = build_c(source, "rtshrinkgrad");
    let bin = gcc(&build_dir, "rtshrinkgrad", None, "self_bin");
    let run = StdCommand::new(&bin).output().expect("run emitted program");
    assert!(
        run.status.success(),
        "runtime shrink grad C binary exited non-zero: stdout={} stderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let c_grad = parse_tensor_data(&String::from_utf8_lossy(&run.stdout));
    assert_close("eval-vs-C shrink grad", &c_grad, &eval_grad);
}

/// Error-path parity (soundness): a runtime shrink whose bounds resolve to a
/// ZERO-SIZE axis (`start == end`) must fail LOUDLY in both lanes — the eval
/// lane rejects an empty-or-inverted bound, and the C runtime guard mirrors
/// it exactly (`end <= start` aborts). Before this guard alignment the C
/// lane silently produced an empty tensor where eval errored.
#[test]
fn issue_616_runtime_shrink_zero_size_axis_errs_in_both_lanes() {
    // k = n - 4 == 0 for the 4-element input: bounds [0, 0).
    let body = "\
  k = cast(sub(cast(shape(x, cast(0, int32)), int64), cast(4, int64)), int64)\n\
  shrink(x, [[cast(0, int64), k]])";
    let input = f32_literal(&[1.0, 2.0, 3.0, 4.0]);
    let source = format!(
        "module Repro.RtZeroSize\nsig f: tensor[n, f32] -> tensor[u, f32]\ndef f(x) = {{\n{body}\n}}\nout = f(to_tensor([{input}]))\n"
    );

    let eval_out = run_eval(&source, "rtzero");
    assert!(
        !eval_out.status.success(),
        "eval must reject the zero-size shrink axis; stdout={}",
        String::from_utf8_lossy(&eval_out.stdout)
    );
    let eval_err = String::from_utf8_lossy(&eval_out.stderr).into_owned();
    assert!(
        eval_err.contains("empty or inverted"),
        "eval must name the empty-bound rejection; stderr={eval_err}"
    );

    let (_dir, build_dir) = build_c(&source, "rtzero");
    let bin = gcc(&build_dir, "rtzero", None, "self_bin");
    let run = StdCommand::new(&bin).output().expect("run emitted program");
    assert!(
        !run.status.success(),
        "C binary must abort on the zero-size shrink axis; stdout={}",
        String::from_utf8_lossy(&run.stdout)
    );
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        stderr.lines().last() == Some("numeric trap: domain in shrink at int64"),
        "C rejection must use the canonical shrink domain diagnostic; stderr={stderr}"
    );
}

/// Error-path parity (soundness): a runtime shrink END that overshoots the
/// input extent must fail LOUDLY in both lanes — the eval lane with a
/// range-guard error, the C lane with the emitted runtime abort — never a
/// silently mis-sized window.
#[test]
fn issue_616_runtime_shrink_overshoot_errs_in_both_lanes() {
    // end = n + 1 > n: out of range for every input.
    let body = "\
  extent = cast(add(cast(shape(x, cast(0, int32)), int64), cast(1, int64)), int64)\n\
  shrink(x, [[cast(1, int64), extent]])";
    let input = f32_literal(&[1.0, 2.0, 3.0, 4.0]);
    let source = format!(
        "module Repro.RtOvershoot\nsig f: tensor[n, f32] -> tensor[u, f32]\ndef f(x) = {{\n{body}\n}}\nout = f(to_tensor([{input}]))\n"
    );

    let eval_out = run_eval(&source, "rtover");
    assert!(
        !eval_out.status.success(),
        "eval must reject the overshooting shrink bound; stdout={}",
        String::from_utf8_lossy(&eval_out.stdout)
    );

    let (_dir, build_dir) = build_c(&source, "rtover");
    let bin = gcc(&build_dir, "rtover", None, "self_bin");
    let run = StdCommand::new(&bin).output().expect("run emitted program");
    assert!(
        !run.status.success(),
        "C binary must abort on the overshooting shrink bound; stdout={}",
        String::from_utf8_lossy(&run.stdout)
    );
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        stderr.lines().last() == Some("numeric trap: domain in shrink at int64"),
        "C rejection must use the canonical shrink domain diagnostic; stderr={stderr}"
    );
}

#[test]
fn checked_movement_expansion_guards_preserve_expand_and_insert_identity() {
    for (op, result_shape) in [("expand", "3"), ("insert", "3, 1")] {
        for (size, valid) in [(4, true), (3, false)] {
            let source = format!(
                "module Repro.MovementIdentity\nsig f: tensor[1, f32] -> tensor[n, f32] -> tensor[{result_shape}, f32]\ndef f(x, y) = {{\n  small = shrink(y, [[0i64, sub(shape(y, 0i32), 1i64)]])\n  {op}(x, 0i32, shape(small, 0i32))\n}}\nout = f(to_tensor([1.0f32]), to_tensor([{}]))\n",
                vec!["1.0f32"; size].join(", ")
            );
            let stem = format!("movement_{op}_{size}");
            let eval = run_eval(&source, &stem);
            let (_dir, build_dir) = build_c(&source, &stem);
            let bin = gcc(&build_dir, &stem, None, "identity_bin");
            let compiled = StdCommand::new(bin).output().unwrap();
            for (lane, output) in [("eval", eval), ("C", compiled)] {
                if valid {
                    assert!(
                        output.status.success(),
                        "{lane} {op}: {}",
                        String::from_utf8_lossy(&output.stderr)
                    );
                    assert_eq!(
                        parse_tensor_data(&String::from_utf8_lossy(&output.stdout)),
                        vec![1.0; 3]
                    );
                } else {
                    assert!(!output.status.success(), "{lane} {op} accepted bad extent");
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    assert!(
                        stderr
                            .lines()
                            .any(|line| line == format!("numeric trap: domain in {op} at int64")),
                        "{lane} {op}: {stderr}"
                    );
                }
            }
        }
    }
}
