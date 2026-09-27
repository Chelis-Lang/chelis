//! C-backend coverage for `numel(to_tensor([]))` returning the wrong value
//! via the runtime path (Runtime-EmptyTensorNumel-F1, pre-0.7.8 release
//! blocker, sibling of `eval_empty_tensor_numel`).
//!
//! ## Background
//!
//! The host evaluator and the C runtime each had their own clamp that
//! inflated a zero-element shape product back to one. `eval_empty_tensor_numel`
//! covers the host path; this file covers the C runtime path. The bug surface
//! lives in `crates/chelis-runtime/src/lib.rs::chelis_alloc_tensor` (and
//! `chelis_alloc_view`):
//!
//! ```rust
//! if tensor.size == 0 {
//!     tensor.size = 1;
//! }
//! ```
//!
//! After multiplying shape components, the runtime clamped a 0-size tensor
//! back to size 1 "for allocation safety", but allocation safety is already
//! handled by `bytes.max(1)` on the `posix_memalign` call. The clamp only
//! corrupted `chelis_tensor_numel`'s return value.
//!
//! ## Fixture
//!
//! `cbackend_numel_empty_tensor_matches_eval`:
//!   1. Eval `numel(to_tensor(empty_values))` with `empty_values: List[f32]` and
//!      assert stdout is `0`.
//!   2. Build the same program with `chelis build --target c`, compile with
//!      gcc, run the binary; assert stdout is `result = 0`.
//!   3. Assert the numeric values agree across lanes.
//!
//! Eval and the C runtime use different print formats (eval prints the bare
//! value, the C emitter prefixes the binding name), so the fixture compares
//! the extracted numeric value rather than raw stdout. The bug surface is a
//! runtime clamp shared by both lanes through their respective numel
//! reporting paths (eval reads TensorValue.shape; the C emitter calls
//! `chelis_tensor_numel` on the runtime struct). A fix that only addressed
//! one lane would show as a value mismatch here.

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
use tempfile::tempdir;

fn chelis_eval(source: &str, name: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join(format!("{name}.ch"));
    fs::write(&src_path, source).expect("write .ch source");
    let assert = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", src_path.to_str().unwrap()])
        .assert()
        .success();
    let out = assert.get_output();
    String::from_utf8_lossy(&out.stdout).trim_end().to_string()
}

fn chelis_build_c(source: &str, name: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join(format!("{name}.ch"));
    fs::write(&src_path, source).expect("write .ch source");
    let kernel_c = dir.path().join(format!("{name}.c"));
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            src_path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            kernel_c.to_str().unwrap(),
        ])
        .assert()
        .success();
    (dir, kernel_c)
}

fn gcc_compile_and_run(build_dir: &Path, kernel_c: &Path, name: &str) -> String {
    let runtime = build_dir.join("libchelis_runtime.a");

    let bin = build_dir.join(name);
    let compile = StdCommand::new("gcc")
        .args([
            "-O0",
            "-std=c11",
            "-I",
            build_dir.to_str().unwrap(),
            kernel_c.to_str().unwrap(),
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

    let run = StdCommand::new(&bin).output().expect("run binary");
    assert!(
        run.status.success(),
        "binary exited non-zero: stdout={} stderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8_lossy(&run.stdout).trim_end().to_string()
}

#[test]
fn cbackend_numel_empty_tensor_matches_eval() {
    let source =
        "result = {\n  empty_values: List[f32] = []\n  numel(to_tensor(empty_values))\n}\n";
    let name = "numel_empty_agreement";

    let eval_out = chelis_eval(source, name);
    let (build_dir, kernel_c) = chelis_build_c(source, name);
    let c_out = gcc_compile_and_run(build_dir.path(), &kernel_c, name);

    // Issue #912 [05-OBS-6]: eval now labels single roots too.
    assert_eq!(
        eval_out.trim(),
        "result = 0",
        "eval stdout for numel(empty) must be 0 not 1 (Runtime-EmptyTensorNumel-F1); got {eval_out}"
    );
    assert_eq!(
        c_out.trim(),
        "result = 0",
        "C-backend stdout for numel(empty) must be `result = 0` not `result = 1` (Runtime-EmptyTensorNumel-F1); got {c_out}"
    );
}
