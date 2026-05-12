//! C-backend `Cast` op coverage: pin the rule that `chelis build --target c`
//! emits an element-wise precision conversion, not a bit-preserving memcpy.
//!
//! CBackend-CastMemcpy (0.7.6 red-team v2 closeout). Parallel to PR #59,
//! which fixed `eval_cast` in the host runtime
//! (`crates/chelis-compiler-api/src/runtime.rs::eval_cast`). The C backend
//! has the same shape of gap at `crates/chelis-backend-c/src/emit.rs`
//! around line 2858: `emit_cast` writes
//! `memcpy(dst, src, n * sizeof(float))` regardless of source and target
//! precision. Result: `cast(tensor[f32], f64)` produces output bytes that
//! are the f32 bit pattern reinterpreted as f64 -- silent data corruption.
//!
//! Diagnosis: `docs/investigations/cbackend_cast_memcpy_diagnosis.md`.
//!
//! Each test:
//!   1. Writes a tiny `.ch` program that casts a typed-parameter tensor.
//!   2. Runs `chelis build --target c` and writes the generated header,
//!      runtime header, and static lib.
//!   3. Writes a `main.c` harness that allocates an input tensor with
//!      known values, calls the exported function, and prints the output
//!      elements as the target precision dictates.
//!   4. Compiles with `gcc` and runs the binary.
//!   5. Asserts the printed output equals the IR/runtime-eval result
//!      (element-wise precision-converted), not the bit-preserving memcpy
//!      result.
//!
//! The four fixtures cover the four conversion shapes:
//!   * `cbackend_cast_tensor_f32_to_f64` -- widening; requires zero-padding
//!     of mantissa bits and re-encoding the exponent. memcpy puts f32
//!     bit patterns into an f64 buffer, producing garbage.
//!   * `cbackend_cast_tensor_f64_to_f32` -- narrowing; requires rounding.
//!     memcpy copies the low 4 bytes of each f64, also garbage.
//!   * `cbackend_cast_tensor_f32_to_int32` -- truncating-toward-zero
//!     conversion. memcpy reinterprets f32 bit patterns as int32s.
//!   * `cbackend_cast_tensor_int32_to_f32` -- integer widening to float.
//!     memcpy reinterprets int32 bit patterns as f32s.
//!
//! Originally gated `#[ignore]` in the failing-test commit; flipped to
//! running in the fix commit on this branch.

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
use tempfile::tempdir;

/// Locate `target/debug/` from the test binary's path. The test binary
/// lives at `<target>/debug/deps/<binary>`, so `..` twice yields the
/// debug dir.
fn target_debug_dir() -> PathBuf {
    let exe = std::env::current_exe().expect("current_exe failed");
    exe.parent()
        .and_then(Path::parent)
        .map(PathBuf::from)
        .expect("could not resolve target/debug dir from current_exe")
}

/// Mirror of `redteam_exec_compile.rs::ensure_runtime_static_lib`. When
/// `chelis-runtime` is built as a dev-dependency, cargo only emits the
/// hashed staticlib in `target/debug/deps/`; the test gcc invocation
/// links against the conventional `target/debug/libchelis_runtime.a`.
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

/// Run `chelis build --target c` on the source program. Returns the
/// build output directory (which contains the emitted C, header,
/// runtime header, and static lib).
fn chelis_build_c(source: &str, fn_name: &str) -> tempfile::TempDir {
    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join(format!("{fn_name}.ch"));
    fs::write(&src_path, source).expect("write .ch source");

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
            dir.path().join(format!("{fn_name}.c")).to_str().unwrap(),
        ])
        .assert()
        .success();

    dir
}

/// Compile `kernel.c + main.c + libchelis_runtime.a` and run the
/// binary, returning stdout on success.
fn gcc_compile_and_run(build_dir: &Path, kernel_c: &Path, main_c: &Path) -> String {
    let canonical = target_debug_dir().join("libchelis_runtime.a");
    ensure_runtime_static_lib(&canonical).expect("materialize libchelis_runtime.a");

    let bin = build_dir.join("test_bin");
    let compile = StdCommand::new("gcc")
        .args([
            "-O0",
            "-std=c11",
            "-I",
            build_dir.to_str().unwrap(),
            kernel_c.to_str().unwrap(),
            main_c.to_str().unwrap(),
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

    let run = StdCommand::new(&bin).output().expect("run cast binary");
    assert!(
        run.status.success(),
        "binary exited non-zero: stdout={} stderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8_lossy(&run.stdout).into_owned()
}

const HARNESS_INCLUDES: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "chelis_runtime.h"
"#;

/// f32 -> f64 widening. Input values are exactly representable in both
/// formats, so the expected output is element-wise identical. With the
/// memcpy bug, the output is garbage (f32 bit patterns interpreted as
/// f64s).
#[test]
fn cbackend_cast_tensor_f32_to_f64() {
    let build = chelis_build_c(
        "def cast_demo(x: tensor[3, f32]) -> tensor[3, f64] = {\n  \
         cast(x, f64)\n\
         }\n",
        "cast_demo",
    );
    let kernel_c = build.path().join("cast_demo.c");
    let main_c = build.path().join("main.c");
    fs::write(
        &main_c,
        format!(
            r#"{HARNESS_INCLUDES}
extern chelis_tensor* cast_demo(chelis_tensor* x);

int main(void) {{
    float in_data[3] = {{1.5f, 2.5f, 3.5f}};
    chelis_tensor t;
    memset(&t, 0, sizeof(t));
    t.data = in_data;
    t.shape[0] = 3;
    t.strides[0] = 1;
    t.ndim = 1;
    t.dtype = CHELIS_F32;
    t.size = 3;
    t.owns_data = 0;

    chelis_tensor* out = cast_demo(&t);
    if (out->dtype != CHELIS_F64) {{ printf("FAIL_DTYPE %d\n", out->dtype); return 1; }}
    double* d = (double*)out->data;
    printf("%.17g %.17g %.17g\n", d[0], d[1], d[2]);
    return 0;
}}
"#
        ),
    )
    .expect("write main.c");

    let stdout = gcc_compile_and_run(build.path(), &kernel_c, &main_c);
    let trimmed = stdout.trim();
    assert_eq!(
        trimmed, "1.5 2.5 3.5",
        "expected element-wise f32->f64 widening; got stdout={trimmed:?}"
    );
}

/// f64 -> f32 narrowing. Input values are exactly representable in both
/// formats. With the memcpy bug the output reads only the low 4 bytes
/// of each f64, which is a different bit pattern from the rounded f32.
#[test]
fn cbackend_cast_tensor_f64_to_f32() {
    let build = chelis_build_c(
        "def cast_demo(x: tensor[3, f64]) -> tensor[3, f32] = {\n  \
         cast(x, f32)\n\
         }\n",
        "cast_demo",
    );
    let kernel_c = build.path().join("cast_demo.c");
    let main_c = build.path().join("main.c");
    fs::write(
        &main_c,
        format!(
            r#"{HARNESS_INCLUDES}
extern chelis_tensor* cast_demo(chelis_tensor* x);

int main(void) {{
    double in_data[3] = {{1.5, 2.5, 3.5}};
    chelis_tensor t;
    memset(&t, 0, sizeof(t));
    t.data = (float*)in_data;
    t.shape[0] = 3;
    t.strides[0] = 1;
    t.ndim = 1;
    t.dtype = CHELIS_F64;
    t.size = 3;
    t.owns_data = 0;

    chelis_tensor* out = cast_demo(&t);
    if (out->dtype != CHELIS_F32) {{ printf("FAIL_DTYPE %d\n", out->dtype); return 1; }}
    float* d = (float*)out->data;
    printf("%.9g %.9g %.9g\n", d[0], d[1], d[2]);
    return 0;
}}
"#
        ),
    )
    .expect("write main.c");

    let stdout = gcc_compile_and_run(build.path(), &kernel_c, &main_c);
    let trimmed = stdout.trim();
    assert_eq!(
        trimmed, "1.5 2.5 3.5",
        "expected element-wise f64->f32 narrowing; got stdout={trimmed:?}"
    );
}

/// f32 -> int32 truncation toward zero. Input `[1.5, 2.5, 3.5]`
/// truncates to `[1, 2, 3]`. With the memcpy bug, the int32 buffer
/// contains the raw f32 bit patterns (1.5f -> 0x3FC00000 -> 1069547520).
#[test]
fn cbackend_cast_tensor_f32_to_int32() {
    let build = chelis_build_c(
        "def cast_demo(x: tensor[3, f32]) -> tensor[3, int32] = {\n  \
         cast(x, int32)\n\
         }\n",
        "cast_demo",
    );
    let kernel_c = build.path().join("cast_demo.c");
    let main_c = build.path().join("main.c");
    fs::write(
        &main_c,
        format!(
            r#"{HARNESS_INCLUDES}
extern chelis_tensor* cast_demo(chelis_tensor* x);

int main(void) {{
    float in_data[3] = {{1.5f, 2.5f, 3.5f}};
    chelis_tensor t;
    memset(&t, 0, sizeof(t));
    t.data = in_data;
    t.shape[0] = 3;
    t.strides[0] = 1;
    t.ndim = 1;
    t.dtype = CHELIS_F32;
    t.size = 3;
    t.owns_data = 0;

    chelis_tensor* out = cast_demo(&t);
    if (out->dtype != CHELIS_I32) {{ printf("FAIL_DTYPE %d\n", out->dtype); return 1; }}
    int32_t* d = (int32_t*)out->data;
    printf("%d %d %d\n", d[0], d[1], d[2]);
    return 0;
}}
"#
        ),
    )
    .expect("write main.c");

    let stdout = gcc_compile_and_run(build.path(), &kernel_c, &main_c);
    let trimmed = stdout.trim();
    assert_eq!(
        trimmed, "1 2 3",
        "expected truncate-toward-zero f32->int32; got stdout={trimmed:?}"
    );
}

/// int32 -> f32 conversion. Input `[1, 2, 3]` converts to `[1.0, 2.0,
/// 3.0]`. With the memcpy bug, the f32 buffer contains the raw int32
/// bit patterns (1 -> 0x00000001 -> ~1.4e-45 denormal).
#[test]
fn cbackend_cast_tensor_int32_to_f32() {
    let build = chelis_build_c(
        "def cast_demo(x: tensor[3, int32]) -> tensor[3, f32] = {\n  \
         cast(x, f32)\n\
         }\n",
        "cast_demo",
    );
    let kernel_c = build.path().join("cast_demo.c");
    let main_c = build.path().join("main.c");
    fs::write(
        &main_c,
        format!(
            r#"{HARNESS_INCLUDES}
extern chelis_tensor* cast_demo(chelis_tensor* x);

int main(void) {{
    int32_t in_data[3] = {{1, 2, 3}};
    chelis_tensor t;
    memset(&t, 0, sizeof(t));
    t.data = (float*)in_data;
    t.shape[0] = 3;
    t.strides[0] = 1;
    t.ndim = 1;
    t.dtype = CHELIS_I32;
    t.size = 3;
    t.owns_data = 0;

    chelis_tensor* out = cast_demo(&t);
    if (out->dtype != CHELIS_F32) {{ printf("FAIL_DTYPE %d\n", out->dtype); return 1; }}
    float* d = (float*)out->data;
    printf("%.9g %.9g %.9g\n", d[0], d[1], d[2]);
    return 0;
}}
"#
        ),
    )
    .expect("write main.c");

    let stdout = gcc_compile_and_run(build.path(), &kernel_c, &main_c);
    let trimmed = stdout.trim();
    assert_eq!(
        trimmed, "1 2 3",
        "expected int32->f32 element-wise conversion; got stdout={trimmed:?}"
    );
}
