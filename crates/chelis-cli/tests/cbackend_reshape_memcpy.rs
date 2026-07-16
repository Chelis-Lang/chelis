//! C-backend host reshape helper precision-preservation pin.
//!
//! CBackend-ReshapeMemcpy (sibling-sweep finding from PR #64,
//! `docs/investigations/cbackend_cast_memcpy_diagnosis.md` section
//! "Sibling sweep"). The host-side reshape helper at
//! `crates/chelis-backend-c/src/host_emit.rs::append_tensor_reshape_helper`
//! (around line 276) emits a `memcpy(out, in, n * sizeof(float))` that
//! drops the upper 4 bytes of every element for any tensor whose dtype
//! is f64 or int64 (silent data corruption). The destination tensor is
//! allocated by `chelis_alloc` with the correct dtype, so the upper
//! halves of each element are left zero-initialised; the lower halves
//! hold half of the source element bits.
//!
//! Diagnosis: `docs/investigations/cbackend_reshape_memcpy_diagnosis.md`.
//!
//! Each fixture:
//!   1. Writes a minimal `.ch` program whose only operation is a
//!      host-level `reshape` of a tensor of the test precision.
//!   2. Runs `chelis build --target c`. The emitted `.c` contains the
//!      `static chelis_host_reshape_tensor` helper plus an
//!      auto-generated `int main(void)` that prints the result via the
//!      buggy `chelis_print_tensor_stdout` (which reads as `float*`
//!      regardless of dtype and therefore masks the bug). We do not
//!      use that main.
//!   3. Post-processes the emitted `.c` to:
//!        * strip `static` from `chelis_host_reshape_tensor` so the
//!          helper is callable from a separate translation unit, and
//!        * rename the emitted `int main(void)` to a stub so the
//!          linker picks up our custom main.
//!   4. Writes a custom `main.c` harness that constructs a
//!      `chelis_tensor` with raw f64 / int64 / f32 backing storage and
//!      a `dtype` field of the right precision, then calls
//!      `chelis_host_reshape_tensor` directly and prints every output
//!      element by reading the destination buffer at the correct C
//!      element type.
//!   5. Compiles and runs the binary; asserts the printed elements
//!      match the input values exactly (reshape is a dtype-preserving,
//!      element-preserving operation -- spec
//!      `spec/03-deep-syntax.md` line 383 "Movement: reshape, ...").
//!
//! Fixtures:
//!   * `cbackend_reshape_tensor_f64`  -- detects upper-4-bytes drop.
//!   * `cbackend_reshape_tensor_int64` -- detects upper-4-bytes drop.
//!   * `cbackend_reshape_tensor_f32`  -- control: should pass today
//!     because `sizeof(float)` is the correct element size for f32.
//!
//! Originally gated `#[ignore]` in the failing-test commit; flipped to
//! running in the fix commit on this branch.

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
use tempfile::tempdir;

/// Locate `target/debug/` from the test binary's path.
fn target_debug_dir() -> PathBuf {
    let exe = std::env::current_exe().expect("current_exe failed");
    exe.parent()
        .and_then(Path::parent)
        .map(PathBuf::from)
        .expect("could not resolve target/debug dir from current_exe")
}

/// Mirror of `cbackend_cast_memcpy.rs::ensure_runtime_static_lib`. When
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
/// build output directory.
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

/// Post-process the emitted kernel so:
///   - `chelis_host_reshape_tensor` is non-static (callable from main.c)
///   - the auto-generated `int main(void)` is renamed away so our
///     harness `main` wins at link time.
fn patch_emitted_kernel(kernel_c: &Path) {
    let original = fs::read_to_string(kernel_c).expect("read kernel.c");
    let after_static = original.replacen(
        "static chelis_tensor* chelis_host_reshape_tensor(",
        "chelis_tensor* chelis_host_reshape_tensor(",
        1,
    );
    assert!(
        after_static != original,
        "expected to strip `static` from chelis_host_reshape_tensor signature"
    );
    let after_main = after_static.replacen(
        "int main(void) {",
        "static int chelis_emitted_main_unused(void) {",
        1,
    );
    assert!(
        after_main != after_static,
        "expected to rename emitted `int main(void)`"
    );
    fs::write(kernel_c, after_main).expect("write patched kernel.c");
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

    let run = StdCommand::new(&bin).output().expect("run reshape binary");
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

extern chelis_tensor* chelis_host_reshape_tensor(
    chelis_tensor* input, const chelis_list* shape_values);
"#;

/// Build a `chelis_list` of int64 shape values in the harness.
const BUILD_SHAPE_LIST_HELPER: &str = r"
static chelis_list* build_shape_list_i64(const int64_t* dims, int64_t len) {
    chelis_value* items = (chelis_value*)malloc(sizeof(chelis_value) * (size_t)len);
    for (int64_t i = 0; i < len; ++i) {
        items[i] = chelis_value_from_int64(dims[i]);
    }
    chelis_list* list = chelis_list_from_values(items, len);
    free(items);
    return list;
}
";

/// f64 reshape. Source buffer is 4 f64 elements; reshape to [2, 2]
/// must preserve all 8 bytes of each element. With the
/// `sizeof(float)` memcpy bug, only the low 4 bytes of each source
/// f64 are copied; reading the destination as f64 yields garbage.
#[test]
fn cbackend_reshape_tensor_f64() {
    let build = chelis_build_c(
        "module Demo\n\
         src = cast(to_tensor([1.5, 2.5, 3.5, 4.5]), f64)\n\
         result = reshape(src, [cast(2, int64), cast(2, int64)])\n",
        "reshape_demo",
    );
    let kernel_c = build.path().join("reshape_demo.c");
    patch_emitted_kernel(&kernel_c);
    let main_c = build.path().join("main.c");
    fs::write(
        &main_c,
        format!(
            r#"{HARNESS_INCLUDES}
{BUILD_SHAPE_LIST_HELPER}

int main(void) {{
    double in_data[4] = {{1.5, 2.5, 3.5, 4.5}};
    chelis_tensor t;
    memset(&t, 0, sizeof(t));
    t.data = (float*)in_data;
    t.shape[0] = 4;
    t.strides[0] = 1;
    t.ndim = 1;
    t.dtype = CHELIS_F64;
    t.size = 4;
    t.owns_data = 0;

    int64_t dims[2] = {{2, 2}};
    chelis_list* shape = build_shape_list_i64(dims, 2);
    chelis_tensor* out = chelis_host_reshape_tensor(&t, shape);
    if (out->dtype != CHELIS_F64) {{ printf("FAIL_DTYPE %d\n", out->dtype); return 1; }}
    if (out->size != 4) {{ printf("FAIL_SIZE %d\n", out->size); return 1; }}
    double* d = (double*)out->data;
    printf("%.17g %.17g %.17g %.17g\n", d[0], d[1], d[2], d[3]);
    return 0;
}}
"#
        ),
    )
    .expect("write main.c");

    let stdout = gcc_compile_and_run(build.path(), &kernel_c, &main_c);
    let trimmed = stdout.trim();
    assert_eq!(
        trimmed, "1.5 2.5 3.5 4.5",
        "expected dtype-preserving f64 reshape; got stdout={trimmed:?}"
    );
}

/// int64 reshape. Source buffer is 4 int64 elements (1, 2, 3, 4);
/// reshape to [2, 2] must preserve all 8 bytes of each element. With
/// the `sizeof(float)` memcpy bug, only the low 4 bytes are copied;
/// reading as int64 yields the original value because the upper half
/// of small positive int64s is zero, but the buffer is *short* by
/// 16 bytes -- the upper halves of the last two elements are stale
/// or zero. We make the bug observable by using values whose upper
/// 4 bytes are non-zero.
#[test]
fn cbackend_reshape_tensor_int64() {
    let build = chelis_build_c(
        "module Demo\n\
         src = to_tensor([cast(1, int64), cast(2, int64), cast(3, int64), cast(4, int64)])\n\
         result = reshape(src, [cast(2, int64), cast(2, int64)])\n",
        "reshape_demo",
    );
    let kernel_c = build.path().join("reshape_demo.c");
    patch_emitted_kernel(&kernel_c);
    let main_c = build.path().join("main.c");
    // Use int64 values with non-zero upper 4 bytes so the byte-drop
    // bug is observable. 0x0123456789ABCDEFLL etc.
    fs::write(
        &main_c,
        format!(
            r#"{HARNESS_INCLUDES}
{BUILD_SHAPE_LIST_HELPER}

int main(void) {{
    int64_t in_data[4] = {{
        0x0123456789ABCDEFLL,
        0x1122334455667788LL,
        0x7FEDCBA987654321LL,
        0x0011223344556677LL
    }};
    chelis_tensor t;
    memset(&t, 0, sizeof(t));
    t.data = (float*)in_data;
    t.shape[0] = 4;
    t.strides[0] = 1;
    t.ndim = 1;
    t.dtype = CHELIS_I64;
    t.size = 4;
    t.owns_data = 0;

    int64_t dims[2] = {{2, 2}};
    chelis_list* shape = build_shape_list_i64(dims, 2);
    chelis_tensor* out = chelis_host_reshape_tensor(&t, shape);
    if (out->dtype != CHELIS_I64) {{ printf("FAIL_DTYPE %d\n", out->dtype); return 1; }}
    if (out->size != 4) {{ printf("FAIL_SIZE %d\n", out->size); return 1; }}
    int64_t* d = (int64_t*)out->data;
    printf("%llx %llx %llx %llx\n",
        (long long)d[0], (long long)d[1], (long long)d[2], (long long)d[3]);
    return 0;
}}
"#
        ),
    )
    .expect("write main.c");

    let stdout = gcc_compile_and_run(build.path(), &kernel_c, &main_c);
    let trimmed = stdout.trim();
    assert_eq!(
        trimmed, "123456789abcdef 1122334455667788 7fedcba987654321 11223344556677",
        "expected dtype-preserving int64 reshape; got stdout={trimmed:?}"
    );
}

/// f32 reshape control: with `sizeof(float)` matching the element
/// width for f32 tensors, this fixture is expected to pass today
/// even on the buggy code path. Locks the invariant that the fix
/// does not regress the f32 case.
#[test]
fn cbackend_reshape_tensor_f32_control() {
    let build = chelis_build_c(
        "module Demo\n\
         src = to_tensor([1.5, 2.5, 3.5, 4.5])\n\
         result = reshape(src, [cast(2, int64), cast(2, int64)])\n",
        "reshape_demo",
    );
    let kernel_c = build.path().join("reshape_demo.c");
    patch_emitted_kernel(&kernel_c);
    let main_c = build.path().join("main.c");
    fs::write(
        &main_c,
        format!(
            r#"{HARNESS_INCLUDES}
{BUILD_SHAPE_LIST_HELPER}

int main(void) {{
    float in_data[4] = {{1.5f, 2.5f, 3.5f, 4.5f}};
    chelis_tensor t;
    memset(&t, 0, sizeof(t));
    t.data = in_data;
    t.shape[0] = 4;
    t.strides[0] = 1;
    t.ndim = 1;
    t.dtype = CHELIS_F32;
    t.size = 4;
    t.owns_data = 0;

    int64_t dims[2] = {{2, 2}};
    chelis_list* shape = build_shape_list_i64(dims, 2);
    chelis_tensor* out = chelis_host_reshape_tensor(&t, shape);
    if (out->dtype != CHELIS_F32) {{ printf("FAIL_DTYPE %d\n", out->dtype); return 1; }}
    if (out->size != 4) {{ printf("FAIL_SIZE %d\n", out->size); return 1; }}
    float* d = (float*)out->data;
    printf("%.9g %.9g %.9g %.9g\n", d[0], d[1], d[2], d[3]);
    return 0;
}}
"#
        ),
    )
    .expect("write main.c");

    let stdout = gcc_compile_and_run(build.path(), &kernel_c, &main_c);
    let trimmed = stdout.trim();
    assert_eq!(
        trimmed, "1.5 2.5 3.5 4.5",
        "expected f32 reshape control to pass; got stdout={trimmed:?}"
    );
}
