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
//! every tensor whose dtype has element size > 4 (`CHELIS_F64`,
//! `CHELIS_I64`), the printer reads 4-byte chunks and widens; an
//! 8-byte element renders as two unrelated 4-byte halves and the
//! tail of the buffer is dropped entirely.  The two dtype-ignoring
//! bugs cancelled each other on small hand-authored examples (e.g.
//! reshape of `[1.5, 2.5, 3.5, 4.5]` happens to print "1.5 2.5 3.5
//! 4.5" because the low f32 halves of those f64 bit patterns are
//! exactly 0.0 and the print routine elides ".0" -- they look
//! correct by accident).  This file pins the print routine
//! directly.
//!
//! Diagnosis: `docs/investigations/cbackend_print_tensor_f64_diagnosis.md`.
//!
//! Each fixture:
//!   1. Writes a `.ch` program that produces a top-level result tensor
//!      with the test precision (f64, int64, or f32 control) via a
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
//!     int64 buffer.
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

/// Locate `target/debug/` from the test binary's path.  Mirrors the
/// helper in `cbackend_cast_memcpy.rs` and `cbackend_reshape_memcpy.rs`.
fn target_debug_dir() -> PathBuf {
    let exe = std::env::current_exe().expect("current_exe failed");
    exe.parent()
        .and_then(Path::parent)
        .map(PathBuf::from)
        .expect("could not resolve target/debug dir from current_exe")
}

/// Mirror of `cbackend_reshape_memcpy::ensure_runtime_static_lib`.
/// `chelis-runtime` is built as a dev-dependency, so cargo only emits
/// the hashed staticlib in `target/debug/deps/`; the gcc-link step
/// expects `target/debug/libchelis_runtime.a`.
fn ensure_runtime_static_lib(canonical: &Path) -> std::io::Result<()> {
    if canonical.exists() {
        return Ok(());
    }
    let deps_dir = canonical
        .parent()
        .expect("canonical lib path has no parent")
        .join("deps");
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in fs::read_dir(&deps_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("libchelis_runtime-") && name.ends_with(".a") {
            let meta = entry.metadata()?;
            let mtime = meta.modified()?;
            match &newest {
                Some((cur, _)) if *cur >= mtime => {}
                _ => newest = Some((mtime, entry.path())),
            }
        }
    }
    let Some((_, hashed)) = newest else {
        return Err(std::io::Error::other(format!(
            "no libchelis_runtime-*.a found in {}",
            deps_dir.display()
        )));
    };
    let tmp = canonical.with_extension("a.tmp");
    fs::copy(&hashed, &tmp)?;
    fs::rename(&tmp, canonical)?;
    Ok(())
}

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
    let canonical = target_debug_dir().join("libchelis_runtime.a");
    ensure_runtime_static_lib(&canonical).expect("materialize libchelis_runtime.a");

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
            canonical.to_str().unwrap(),
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
#[ignore = "chelis_print_tensor_stdout f64 misread; see commit <pending>"]
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

/// int64 print fixture.  The kernel function casts an int32 input to
/// int64.  Same shape of bug as the f64 case: `data` typed as
/// `float *` reads only 4 bytes per element.  Values are chosen large
/// enough that the upper 4 bytes are not all zero; the values also
/// must survive the int32 input stage (so they fit in int32) but the
/// observation is on the int64 output tensor's print.
#[test]
#[ignore = "chelis_print_tensor_stdout f64 misread; see commit <pending>"]
fn cbackend_print_tensor_int64() {
    let source = "def to_i64(x: tensor[4, int32]) -> tensor[4, int64] = cast(x, int64)\n\
                  src = to_tensor([cast(100000, int32), cast(200000, int32), \
                  cast(300000, int32), cast(400000, int32)])\n\
                  result = to_i64(src)\n";
    let eval_out = chelis_eval(source, "print_i64");
    let (build, kernel_c) = chelis_build_c(source, "print_i64");
    let cbuild_out = gcc_compile_and_run(build.path(), &kernel_c, "print_i64");
    // Ground-truth eval renders int64 tensor elements through the
    // shared formatter, which prints them with a `.0` suffix.  Pin
    // the eval output explicitly so the C-build comparison cannot
    // silently agree on garbage.
    assert_eq!(
        eval_out,
        "src = tensor(shape=[4], data=[100000.0, 200000.0, 300000.0, 400000.0])\n\
         result = tensor(shape=[4], data=[100000.0, 200000.0, 300000.0, 400000.0])",
        "eval ground truth changed; update fixture"
    );
    assert_eq!(
        cbuild_out, eval_out,
        "chelis build --target c print routine must agree with eval for int64"
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
