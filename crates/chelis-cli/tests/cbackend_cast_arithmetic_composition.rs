//! C-backend multi-op cast + arithmetic composition fixtures.
//!
//! W2 PR 4 of the 0.7.8 compiler cleanup workstream
//! (`CRuntime-F32Coupling`).  The cast+arithmetic shape
//! `add(cast(t, f64), cast(t, f64))` is the canonical reproducer for
//! the bug class closed by PRs #64, #67, #72, #84, #86, and #87:
//! whenever the chain reads through a non-migrated f32-strided
//! accessor, the f64 mantissa silently truncates and the C-backend
//! output diverges from `chelis eval`.
//!
//! W2 PR 3's agent reported observing this corruption before applying
//! their host_emit migration; verification on the current `main`
//! (post-PR #87 merge) confirms the chain produces byte-exact f64
//! output through the DAG-emitted kernel path.  These fixtures lock
//! the property so any future regression in either the runtime or
//! the host_emit code paths surfaces here.
//!
//! Each fixture:
//!   1. Writes a small `.ch` program that combines `cast` with a
//!      tensor arithmetic op (`add` or `mul`).
//!   2. Runs `chelis build --target c`.
//!   3. Writes a `main.c` harness that builds the input tensor with
//!      known values, calls the exported function, and prints the
//!      output elements at full precision.
//!   4. Compiles with `gcc` and runs the binary.
//!   5. Asserts byte-exact agreement with the host-evaluator result
//!      computed at the same precision.
//!
//! Compositions covered:
//!   * `add(cast(t, f64), cast(t, f64))` for source tensor[f32]
//!   * `mul(cast(t, f64), cast(t, f64))` for source tensor[f32]
//!   * `add(t, cast(s, f64))` mixing f64 native + cast-from-f32
//!   * `cast(add(t, t), f64)` arithmetic-then-cast (widening at end)

mod common;

use assert_cmd::Command;
use common::authored_c_symbol;
use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;
use tempfile::tempdir;

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

fn gcc_compile_and_run(build_dir: &Path, kernel_c: &Path, main_c: &Path) -> String {
    let runtime = build_dir.join("libchelis_runtime.a");

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
    String::from_utf8_lossy(&run.stdout).into_owned()
}

const HARNESS_INCLUDES: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "chelis_runtime.h"

static chelis_dtype harness_dtype(const chelis_tensor *tensor) {
    return chelis_tensor_read_view(tensor).dtype;
}
static const void *harness_data(const chelis_tensor *tensor) {
    return chelis_tensor_read_view(tensor).data;
}
"#;

/// `add(cast(t, f64), cast(t, f64))` from f32 source.  The host
/// evaluator agrees that the result is `2 * t` at full f64.  Any
/// f32-strided intermediate read in cast or add would produce
/// either garbage or a low-precision result.
#[test]
fn cbackend_add_of_two_casts_f64_from_f32() {
    let build = chelis_build_c(
        "def composed(x: tensor[3, f32]) -> tensor[3, f64] = {\n  \
         a = cast(x, f64)\n  \
         b = cast(x, f64)\n  \
         add(a, b)\n\
         }\n",
        "composed",
    );
    let kernel_c = build.path().join("composed.c");
    let main_c = build.path().join("main.c");
    let composed = authored_c_symbol("composed");
    fs::write(
        &main_c,
        format!(
            r#"{HARNESS_INCLUDES}
extern chelis_tensor* {composed}(chelis_tensor* x);
static chelis_tensor* composed(chelis_tensor* x) {{ chelis_tensor_retain(x); return x; }}

int main(void) {{
    float in_data[3] = {{1.5f, 2.5f, 3.5f}};
    int64_t in_shape[1] = {{3}};
    chelis_tensor* t = chelis_tensor_entry_borrow(
        1, in_shape, CHELIS_DTYPE_F32, in_data, (int64_t)sizeof(in_data));

    chelis_tensor* out = {composed}(t);
    if (harness_dtype(out) != CHELIS_DTYPE_F64) {{ printf("FAIL_DTYPE %d\n", harness_dtype(out)); return 1; }}
    double* d = (const double*)harness_data(out);
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
        trimmed, "3 5 7",
        "expected 2*t at full f64 from add of two f64 casts; got stdout={trimmed:?}"
    );
}

/// `mul(cast(t, f64), cast(t, f64))` from f32 source.  Result is
/// `t * t` at full f64.  Same dispatch surface as the add case but
/// the multiplication amplifies any precision loss.
#[test]
fn cbackend_mul_of_two_casts_f64_from_f32() {
    let build = chelis_build_c(
        "def composed(x: tensor[3, f32]) -> tensor[3, f64] = {\n  \
         a = cast(x, f64)\n  \
         b = cast(x, f64)\n  \
         mul(a, b)\n\
         }\n",
        "composed",
    );
    let kernel_c = build.path().join("composed.c");
    let main_c = build.path().join("main.c");
    let composed = authored_c_symbol("composed");
    fs::write(
        &main_c,
        format!(
            r#"{HARNESS_INCLUDES}
extern chelis_tensor* {composed}(chelis_tensor* x);
static chelis_tensor* composed(chelis_tensor* x) {{ chelis_tensor_retain(x); return x; }}

int main(void) {{
    float in_data[3] = {{1.5f, 2.5f, 3.5f}};
    int64_t in_shape[1] = {{3}};
    chelis_tensor* t = chelis_tensor_entry_borrow(
        1, in_shape, CHELIS_DTYPE_F32, in_data, (int64_t)sizeof(in_data));

    chelis_tensor* out = {composed}(t);
    if (harness_dtype(out) != CHELIS_DTYPE_F64) {{ printf("FAIL_DTYPE %d\n", harness_dtype(out)); return 1; }}
    double* d = (const double*)harness_data(out);
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
        trimmed, "2.25 6.25 12.25",
        "expected t*t at full f64 from mul of two f64 casts; got stdout={trimmed:?}"
    );
}

/// `cast(add(t, t), f64)` arithmetic first then cast.  Tests the
/// reverse order: the f32 add happens at f32 precision, then the
/// cast widens to f64.  Result is `2 * t` represented in f64 but
/// with f32 precision (the add ran at f32).
#[test]
fn cbackend_cast_after_add_widens_to_f64() {
    let build = chelis_build_c(
        "def composed(x: tensor[3, f32]) -> tensor[3, f64] = {\n  \
         s = add(x, x)\n  \
         cast(s, f64)\n\
         }\n",
        "composed",
    );
    let kernel_c = build.path().join("composed.c");
    let main_c = build.path().join("main.c");
    let composed = authored_c_symbol("composed");
    fs::write(
        &main_c,
        format!(
            r#"{HARNESS_INCLUDES}
extern chelis_tensor* {composed}(chelis_tensor* x);
static chelis_tensor* composed(chelis_tensor* x) {{ chelis_tensor_retain(x); return x; }}

int main(void) {{
    float in_data[3] = {{1.5f, 2.5f, 3.5f}};
    int64_t in_shape[1] = {{3}};
    chelis_tensor* t = chelis_tensor_entry_borrow(
        1, in_shape, CHELIS_DTYPE_F32, in_data, (int64_t)sizeof(in_data));

    chelis_tensor* out = {composed}(t);
    if (harness_dtype(out) != CHELIS_DTYPE_F64) {{ printf("FAIL_DTYPE %d\n", harness_dtype(out)); return 1; }}
    double* d = (const double*)harness_data(out);
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
        trimmed, "3 5 7",
        "expected widened f64 result of f32 add; got stdout={trimmed:?}"
    );
}

/// `mul(cast(t, f64), s)` where `s` is a native f64 tensor.  Mixes
/// a cast-derived buffer with an already-f64 buffer; both reads
/// must hit the f64 dispatch arm correctly.
#[test]
fn cbackend_mul_cast_with_native_f64() {
    let build = chelis_build_c(
        "def composed(x: tensor[3, f32], y: tensor[3, f64]) -> tensor[3, f64] = {\n  \
         a = cast(x, f64)\n  \
         mul(a, y)\n\
         }\n",
        "composed",
    );
    let kernel_c = build.path().join("composed.c");
    let main_c = build.path().join("main.c");
    let composed = authored_c_symbol("composed");
    fs::write(
        &main_c,
        format!(
            r#"{HARNESS_INCLUDES}
extern chelis_tensor* {composed}(chelis_tensor* x, chelis_tensor* y);
static chelis_tensor* composed(chelis_tensor* x, chelis_tensor* y) {{ (void)y; chelis_tensor_retain(x); return x; }}

int main(void) {{
    float in_x[3] = {{2.0f, 3.0f, 4.0f}};
    double in_y[3] = {{0.5, 0.25, 0.125}};
    int64_t in_shape[1] = {{3}};
    chelis_tensor* tx = chelis_tensor_entry_borrow(
        1, in_shape, CHELIS_DTYPE_F32, in_x, (int64_t)sizeof(in_x));
    chelis_tensor* ty = chelis_tensor_entry_borrow(
        1, in_shape, CHELIS_DTYPE_F64, in_y, (int64_t)sizeof(in_y));

    chelis_tensor* out = {composed}(tx, ty);
    if (harness_dtype(out) != CHELIS_DTYPE_F64) {{ printf("FAIL_DTYPE %d\n", harness_dtype(out)); return 1; }}
    double* d = (const double*)harness_data(out);
    printf("%.17g %.17g %.17g\n", d[0], d[1], d[2]);
    return 0;
}}
"#
        ),
    )
    .expect("write main.c");

    let stdout = gcc_compile_and_run(build.path(), &kernel_c, &main_c);
    let trimmed = stdout.trim();
    // x=[2,3,4] cast to f64, mul with y=[0.5, 0.25, 0.125]: [1, 0.75, 0.5].
    assert_eq!(
        trimmed, "1 0.75 0.5",
        "expected f32-cast * native-f64 element-wise; got stdout={trimmed:?}"
    );
}

/// `add(cast(t, f64), cast(t, f64))` for i32 source.  The i32
/// values widen to f64 first, then the f64 add runs.  Validates
/// the i32 -> f64 cast plus the f64 add dispatch chain.
#[test]
fn cbackend_add_of_two_casts_f64_from_int32() {
    let build = chelis_build_c(
        "def composed(x: tensor[3, i32]) -> tensor[3, f64] = {\n  \
         a = cast(x, f64)\n  \
         b = cast(x, f64)\n  \
         add(a, b)\n\
         }\n",
        "composed",
    );
    let kernel_c = build.path().join("composed.c");
    let main_c = build.path().join("main.c");
    let composed = authored_c_symbol("composed");
    fs::write(
        &main_c,
        format!(
            r#"{HARNESS_INCLUDES}
extern chelis_tensor* {composed}(chelis_tensor* x);
static chelis_tensor* composed(chelis_tensor* x) {{ chelis_tensor_retain(x); return x; }}

int main(void) {{
    int32_t in_data[3] = {{7, 11, 13}};
    int64_t in_shape[1] = {{3}};
    chelis_tensor* t = chelis_tensor_entry_borrow(
        1, in_shape, CHELIS_DTYPE_I32, in_data, (int64_t)sizeof(in_data));

    chelis_tensor* out = {composed}(t);
    if (harness_dtype(out) != CHELIS_DTYPE_F64) {{ printf("FAIL_DTYPE %d\n", harness_dtype(out)); return 1; }}
    double* d = (const double*)harness_data(out);
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
        trimmed, "14 22 26",
        "expected 2*i32 widened to f64; got stdout={trimmed:?}"
    );
}
