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
//! The verb: `shrink(x, [[1, n-1]])` keeps `[2, ..., n-1]` for input
//! `[1, 2, ..., n]` (with `n = shape(x, 0)` read at run time); `stride(_, 2)`
//! then keeps every other element: `[2, 4, 6, ...]`.

use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;

use assert_cmd::Command;
use tempfile::{TempDir, tempdir};

/// Runtime-shrink-only verb; its single movement output dim is the sig's `u`.
const SHRINK_BODY: &str = "\
  extent = cast(sub(cast(shape(x, cast(0, int32)), int64), cast(1, int64)), int32)\n\
  shrink(x, [[cast(1, int32), extent]])";

/// The full runtime shrink -> stride chain, anchored by a static reshape so
/// each movement output keeps its own (anonymous) dim. This is the
//  movement-op shape of the chelis#616 avgpool oracle's `window_row`.
const CHAIN_ANCHORED_BODY: &str = "\
  extent = cast(sub(cast(shape(x, cast(0, int32)), int64), cast(1, int64)), int32)\n\
  w = stride(shrink(x, [[cast(1, int32), extent]]), cast(2, int32))\n\
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
                "    {{ int shape[1] = {{{n}}}; chelis_tensor* x = chelis_alloc(1, shape, CHELIS_F32); \
                 for (int i = 0; i < {n}; i++) x->data[i] = (float)(i + 1); \
                 chelis_tensor* w = out(x); \
                 for (int i = 0; i < w->size; i++) printf(\"%.6f\\n\", w->data[i]); \
                 printf(\"---\\n\"); }}"
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

/// The guarded degenerate (chelis#616 over-unification): a shrink -> stride
/// chain returned DIRECTLY under a sig-named output dim gets both movement
/// outputs unified to the one symbol `u` by the checker, though their
/// runtime extents genuinely differ (`n-2` vs `ceil((n-2)/2)`). The C lane
/// must fail LOUDLY at run time (the second site's equality guard), never
/// allocate a mis-sized tensor. The eval lane, which computes shapes from
/// actual values and never consults the over-unified type, accepts and
/// computes the correct window — the asymmetry is the checker's typing
/// imprecision, tracked with the movement typing work, and the C guard is
/// the soundness floor under it.
#[test]
fn issue_616_over_unified_movement_chain_fails_loud_not_mis_sized() {
    let input: Vec<f64> = (1..=6).map(|v| v as f64).collect();
    let body = "\
  extent = cast(sub(cast(shape(x, cast(0, int32)), int64), cast(1, int64)), int32)\n\
  stride(shrink(x, [[cast(1, int32), extent]]), cast(2, int32))";
    let source = format!(
        "module Repro.RtDegenerate\nsig f: tensor[n, f32] -> tensor[u, f32]\ndef f(x) = {{\n{body}\n}}\nout = f(to_tensor([{}]))\n",
        f32_literal(&input)
    );

    let eval_out = run_eval(&source, "rtdegen");
    assert!(
        eval_out.status.success(),
        "eval computes the true window regardless of the over-unified type: {}",
        String::from_utf8_lossy(&eval_out.stderr)
    );
    let eval_values = parse_tensor_data(&String::from_utf8_lossy(&eval_out.stdout));
    assert_close("eval degenerate chain", &eval_values, &[2.0, 4.0]);

    let (_dir, build_dir) = build_c(&source, "rtdegen");
    let bin = gcc(&build_dir, "rtdegen", None, "self_bin");
    let run = StdCommand::new(&bin).output().expect("run emitted program");
    assert!(
        !run.status.success(),
        "C binary must abort on the over-unified runtime dim; stdout={}",
        String::from_utf8_lossy(&run.stdout)
    );
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        stderr.contains("runtime dim `u` mismatch"),
        "C abort must name the runtime-dim equality guard; stderr={stderr}"
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
  extent = cast(add(cast(shape(x, cast(0, int32)), int64), cast(1, int64)), int32)\n\
  shrink(x, [[cast(1, int32), extent]])";
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
        stderr.contains("shrink bound out of range"),
        "C abort must name the shrink range guard; stderr={stderr}"
    );
}
