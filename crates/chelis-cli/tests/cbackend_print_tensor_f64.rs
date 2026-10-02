//! C-backend `chelis_print_tensor_stdout` precision-read pin.
//!
//! CBackend-PrintTensorF64. Third instance of the C-backend
//! precision-lookup bug class closed in this branch.  PR #64
//! (CBackend-CastMemcpy) and PR #67 (CBackend-ReshapeMemcpy) fixed
//! the storage side -- `emit_cast` now performs element-wise
//! precision conversion, and `chelis_host_reshape_tensor` now uses
//! the source-dtype element size in its `memcpy`.  Both storage-side
//! fixes were masked in normal Chelis-level testing by the print
//! routine emitted from `host_emit.rs::append_tensor_print_helper`
//! (around line 207), which reads
//!
//! ```c
//! double value = t->data[i];
//! ```
//!
//! against `t->data` declared as `float *` in
//! `crates/chelis-runtime/include/chelis_runtime.h` line 20.  For
//! every tensor whose dtype has element size > 4 (`CHELIS_DTYPE_F64`,
//! `CHELIS_DTYPE_I64`), the printer reads 4-byte chunks and widens; an
//! 8-byte element renders as two unrelated 4-byte halves and the
//! tail of the buffer is dropped entirely.  The two dtype-ignoring
//! bugs cancelled each other on small hand-authored examples (e.g.
//! reshape of `[1.5, 2.5, 3.5, 4.5]` happens to print "1.5 2.5 3.5
//! 4.5" because the low f32 halves of those f64 bit patterns are
//! exactly 0.0 and the print routine elides ".0" -- they look
//! correct by accident).  This file pins the print routine
//! directly.
//!
//! Diagnosis: `docs/archive/investigations/cbackend_print_tensor_f64_diagnosis.md`.
//!
//! Each fixture:
//!   1. Writes a `.ch` program that produces a top-level result tensor
//!      with the test precision (f64, i64, or f32 control) via a
//!      typed kernel function -- this forces the storage-side
//!      precision into the emitted C, post-PR-#64.
//!   2. Runs `chelis eval --file` on the source.  This goes through
//!      the host evaluator, which is correct on every supported
//!      precision (PR #59 fixed `eval_cast`), so the eval stdout is
//!      the ground truth.
//!   3. Runs `chelis build --target c` on the same source, compiles
//!      the emitted main with `gcc`, and runs the binary.  The
//!      emitted main calls `chelis_print_tensor_stdout` for each
//!      top-level binding.
//!   4. Asserts the C-build stdout matches the eval stdout exactly,
//!      character for character.
//!
//! Fixtures:
//!   * `cbackend_print_tensor_f64`  -- detects 4-byte-chunk read of an
//!     f64 buffer.
//!   * `cbackend_print_tensor_int64` -- detects 4-byte-chunk read of an
//!     i64 buffer.
//!   * `cbackend_print_tensor_f32_control` -- f32 baseline; the print
//!     routine's `float *` read matches the f32 element size, so this
//!     fixture passes on the buggy code and must keep passing after
//!     the fix.
//!
//! Originally gated `#[ignore]` in the failing-test commit; flipped
//! to running in the fix commit on this branch.

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
use tempfile::tempdir;

/// Run `chelis eval --file` on the source program.  Returns stdout
/// trimmed of trailing whitespace.  Asserts the eval call succeeds.
fn chelis_eval(source: &str, fn_name: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join(format!("{fn_name}.ch"));
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

/// Run `chelis build --target c` on the source program and return
/// the build output directory plus the emitted kernel file path.
fn chelis_build_c(source: &str, fn_name: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join(format!("{fn_name}.ch"));
    fs::write(&src_path, source).expect("write .ch source");
    let kernel_c = dir.path().join(format!("{fn_name}.c"));
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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

/// gcc-compile the emitted C and run the resulting binary, returning
/// stdout trimmed of trailing whitespace.
fn gcc_compile_and_run(build_dir: &Path, kernel_c: &Path, fn_name: &str) -> String {
    let runtime = build_dir.join("libchelis_runtime.a");

    let bin = build_dir.join(fn_name);
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

    let run = StdCommand::new(&bin).output().expect("run print binary");
    assert!(
        run.status.success(),
        "binary exited non-zero: stdout={} stderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8_lossy(&run.stdout).trim_end().to_string()
}

/// f64 print fixture.  The kernel function casts an f32 input to f64,
/// producing an output tensor whose `data` buffer holds 8-byte
/// elements.  The print routine reads `data` as `float *`, so an
/// 8-byte element splits into two unrelated 4-byte halves.  Bug
/// observable today as `[0.0, 1.9375, 0.0, 2.0625]` against the
/// ground-truth `[1.5, 2.5, 3.5, 4.5]`.
#[test]
fn cbackend_print_tensor_f64() {
    let source = "def to_f64(x: tensor[4, f32]) -> tensor[4, f64] = cast(x, f64)\n\
                  src = to_tensor([1.5, 2.5, 3.5, 4.5])\n\
                  result = to_f64(src)\n";
    let eval_out = chelis_eval(source, "print_f64");
    let (build, kernel_c) = chelis_build_c(source, "print_f64");
    let cbuild_out = gcc_compile_and_run(build.path(), &kernel_c, "print_f64");
    // Lock the ground truth so a regression that breaks both eval and
    // build still fails the test.  Eval is correct against PR #59.
    assert_eq!(
        eval_out,
        "src = tensor(shape=[4], data=[1.5, 2.5, 3.5, 4.5])\n\
         result = tensor(shape=[4], data=[1.5, 2.5, 3.5, 4.5])",
        "eval ground truth changed; update fixture"
    );
    assert_eq!(
        cbuild_out, eval_out,
        "chelis build --target c print routine must agree with eval for f64"
    );
}

/// i64 print fixture.  Same shape of bug as the f64 case: the print
/// routine reads `data` as `float *` and decodes 8-byte i64
/// elements at a 4-byte stride.
///
/// Source path is `f32 -> f64 -> i64` rather than `i32 -> i64`
/// because the host runtime stores i32 tensors as f32 bit patterns
/// (see `crates/chelis-runtime/src/lib.rs` line 1559); a cast from
/// i32 would read those f32 bit patterns through `(int32_t*)
/// src->data` and produce i64 elements that hold f32 bit patterns
/// in their low 4 bytes.  That is a downstream same-class bug
/// flagged in the diagnosis sibling sweep and out of scope for this
/// PR.  Casting from f64 reads `(double*)src->data` correctly post-
/// PR-#64 and yields the canonical 8-byte i64 bit pattern in the
/// destination buffer, which is the input this fixture needs to
/// exercise the print routine in isolation.
#[test]
fn cbackend_print_tensor_int64() {
    let source = "def f32_to_f64(x: tensor[4, f32]) -> tensor[4, f64] = cast(x, f64)\n\
                  def f64_to_i64(y: tensor[4, f64]) -> tensor[4, i64] = cast(y, i64)\n\
                  src = to_tensor([100000.0, 200000.0, 300000.0, 400000.0])\n\
                  mid = f32_to_f64(src)\n\
                  result = f64_to_i64(mid)\n";
    let eval_out = chelis_eval(source, "print_i64");
    let (build, kernel_c) = chelis_build_c(source, "print_i64");
    let cbuild_out = gcc_compile_and_run(build.path(), &kernel_c, "print_i64");
    // Ground-truth eval: since chelis#732 P1, i64 tensor elements print
    // as integers ([05-OBS-2]) while f64 elements keep the `.0` form. Pin
    // the eval output explicitly so the C-build comparison cannot
    // silently agree on garbage.
    assert_eq!(
        eval_out,
        "src = tensor(shape=[4], data=[100000.0, 200000.0, 300000.0, 400000.0])\n\
         mid = tensor(shape=[4], data=[100000.0, 200000.0, 300000.0, 400000.0])\n\
         result = tensor(shape=[4], data=[100000, 200000, 300000, 400000])",
        "eval ground truth changed; update fixture"
    );
    // chelis#732 Phase 2: the generated printer renders i64 elements
    // as exact integers ([05-OBS-2]), byte-identical to eval.
    assert_eq!(
        cbuild_out,
        "src = tensor(shape=[4], data=[100000.0, 200000.0, 300000.0, 400000.0])\n\
         mid = tensor(shape=[4], data=[100000.0, 200000.0, 300000.0, 400000.0])\n\
         result = tensor(shape=[4], data=[100000, 200000, 300000, 400000])",
        "chelis build --target c print routine must decode i64 elements \
         at the correct stride (byte parity with eval returns at chelis#732 \
         Phase 2)"
    );
}

/// f32 control.  The print routine's `float *` read matches the f32
/// element size, so the buggy code is correct here.  Locks the
/// invariant that the fix does not regress the f32 path.  The source
/// has two top-level bindings so the `chelis eval --file` output
/// uses the labelled `name = tensor(...)` form, matching the form
/// the emitted C `main()` always prints.
#[test]
fn cbackend_print_tensor_f32_control() {
    let source = "src = to_tensor([1.5, 2.5, 3.5, 4.5])\n\
                  result = to_tensor([10.0, 20.0, 30.0, 40.0])\n";
    let eval_out = chelis_eval(source, "print_f32");
    let (build, kernel_c) = chelis_build_c(source, "print_f32");
    let cbuild_out = gcc_compile_and_run(build.path(), &kernel_c, "print_f32");
    assert_eq!(
        eval_out,
        "src = tensor(shape=[4], data=[1.5, 2.5, 3.5, 4.5])\n\
         result = tensor(shape=[4], data=[10.0, 20.0, 30.0, 40.0])",
        "eval ground truth changed; update fixture"
    );
    assert_eq!(
        cbuild_out, eval_out,
        "chelis build --target c print routine must agree with eval for f32"
    );
}
