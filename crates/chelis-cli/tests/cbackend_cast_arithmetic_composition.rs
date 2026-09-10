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

#[path = "../../../tests/support/runtime_archive.rs"]
mod runtime_archive;

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
use tempfile::tempdir;

fn target_debug_dir() -> PathBuf {
    let exe = std::env::current_exe().expect("current_exe failed");
    exe.parent()
        .and_then(Path::parent)
        .map(PathBuf::from)
        .expect("could not resolve target/debug dir from current_exe")
}

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
    static NEXT_TEMP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let tmp = canonical.with_extension(format!(
        "a.tmp.{}.{}",
        std::process::id(),
        NEXT_TEMP.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    fs::copy(&hashed, &tmp)?;
    fs::rename(&tmp, canonical)?;
    Ok(())
}

#[test]
fn runtime_archive_materialization_is_safe_within_one_process() {
    let source_root = target_debug_dir();
    let source_deps = source_root.join("deps");
    let source = fs::read_dir(&source_deps)
        .expect("read source deps")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| {
            path.file_name().is_some_and(|name| {
                let name = name.to_string_lossy();
                name.starts_with("libchelis_runtime-") && name.ends_with(".a")
            })
        })
        .expect("hashed runtime archive");

    let sandbox = tempdir().expect("runtime materialization sandbox");
    let deps = sandbox.path().join("deps");
    fs::create_dir(&deps).expect("create sandbox deps");
    fs::copy(&source, deps.join(source.file_name().unwrap())).expect("seed hashed archive");
    let canonical = sandbox.path().join("libchelis_runtime.a");

    std::thread::scope(|scope| {
        let mut threads = Vec::new();
        for _ in 0..16 {
            threads.push(scope.spawn(|| ensure_runtime_static_lib(&canonical)));
        }
        for thread in threads {
            thread.join().expect("materialization thread").unwrap();
        }
    });

    assert!(
        canonical.is_file(),
        "canonical runtime archive was not created"
    );
}

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
    let canonical = runtime_archive::explicit().unwrap_or_else(|| {
        let archive = target_debug_dir().join("libchelis_runtime.a");
        ensure_runtime_static_lib(&archive).expect("materialize libchelis_runtime.a");
        archive
    });

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
    fs::write(
        &main_c,
        format!(
            r#"{HARNESS_INCLUDES}
extern chelis_tensor* composed(chelis_tensor* x);

int main(void) {{
    float in_data[3] = {{1.5f, 2.5f, 3.5f}};
    int64_t in_shape[1] = {{3}};
    chelis_tensor* t = chelis_tensor_entry_borrow(
        1, in_shape, CHELIS_DTYPE_F32, in_data, (int64_t)sizeof(in_data));

    chelis_tensor* out = composed(t);
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
    fs::write(
        &main_c,
        format!(
            r#"{HARNESS_INCLUDES}
extern chelis_tensor* composed(chelis_tensor* x);

int main(void) {{
    float in_data[3] = {{1.5f, 2.5f, 3.5f}};
    int64_t in_shape[1] = {{3}};
    chelis_tensor* t = chelis_tensor_entry_borrow(
        1, in_shape, CHELIS_DTYPE_F32, in_data, (int64_t)sizeof(in_data));

    chelis_tensor* out = composed(t);
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
    fs::write(
        &main_c,
        format!(
            r#"{HARNESS_INCLUDES}
extern chelis_tensor* composed(chelis_tensor* x);

int main(void) {{
    float in_data[3] = {{1.5f, 2.5f, 3.5f}};
    int64_t in_shape[1] = {{3}};
    chelis_tensor* t = chelis_tensor_entry_borrow(
        1, in_shape, CHELIS_DTYPE_F32, in_data, (int64_t)sizeof(in_data));

    chelis_tensor* out = composed(t);
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
    fs::write(
        &main_c,
        format!(
            r#"{HARNESS_INCLUDES}
extern chelis_tensor* composed(chelis_tensor* x, chelis_tensor* y);

int main(void) {{
    float in_x[3] = {{2.0f, 3.0f, 4.0f}};
    double in_y[3] = {{0.5, 0.25, 0.125}};
    int64_t in_shape[1] = {{3}};
    chelis_tensor* tx = chelis_tensor_entry_borrow(
        1, in_shape, CHELIS_DTYPE_F32, in_x, (int64_t)sizeof(in_x));
    chelis_tensor* ty = chelis_tensor_entry_borrow(
        1, in_shape, CHELIS_DTYPE_F64, in_y, (int64_t)sizeof(in_y));

    chelis_tensor* out = composed(tx, ty);
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

/// `add(cast(t, f64), cast(t, f64))` for int32 source.  The int32
/// values widen to f64 first, then the f64 add runs.  Validates
/// the int32 -> f64 cast plus the f64 add dispatch chain.
#[test]
fn cbackend_add_of_two_casts_f64_from_int32() {
    let build = chelis_build_c(
        "def composed(x: tensor[3, int32]) -> tensor[3, f64] = {\n  \
         a = cast(x, f64)\n  \
         b = cast(x, f64)\n  \
         add(a, b)\n\
         }\n",
        "composed",
    );
    let kernel_c = build.path().join("composed.c");
    let main_c = build.path().join("main.c");
    fs::write(
        &main_c,
        format!(
            r#"{HARNESS_INCLUDES}
extern chelis_tensor* composed(chelis_tensor* x);

int main(void) {{
    int32_t in_data[3] = {{7, 11, 13}};
    int64_t in_shape[1] = {{3}};
    chelis_tensor* t = chelis_tensor_entry_borrow(
        1, in_shape, CHELIS_DTYPE_I32, in_data, (int64_t)sizeof(in_data));

    chelis_tensor* out = composed(t);
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
        "expected 2*int32 widened to f64; got stdout={trimmed:?}"
    );
}
