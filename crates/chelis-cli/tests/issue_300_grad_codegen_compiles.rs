//! Issue #300: the C backend EMITS C for a `grad` over the
//! constant-broadcast idiom `expand(scalar_to_tensor(c), axis, n)`, but the
//! emitted kernel did not gcc-compile:
//!
//! ```text
//! error: pointer value used where a floating-point was expected
//! error: pointer cannot be cast to type 'float'
//! ```
//!
//! Root cause (a C-backend host-lowering defect, NOT grad-specific): a
//! scalar-returning function body is lowered statement-by-statement in the
//! host lane. A `let k = expand(scalar_to_tensor(cast(2.5, f32)),
//! cast(0, int32), cast(2, int32))` binding lost its tensor type because:
//!
//!   * `expr_int_literal` did not see through `cast(0, int32)` /
//!     `cast(2, int32)`, so `infer_app_expr_host_type`'s `expand` shape
//!     handler bailed and the result type degraded to `Unknown`;
//!   * `infer_builtin_host_type_from_arg_tys` had no `expand` arm.
//!
//! With no tensor type, `expand` fell through to a host-lane `Builtin
//! "expand"` that the C emitter renders as `void* k = /* unsupported
//! builtin expand */ 0`. The consuming `sum(mul(x, k), 0)` tensor helper
//! then took `k` as a "scalar" input and emitted
//! `((float*)tmp->data)[0] = (float)(void* k);` -- the pointer-to-float
//! cast gcc rejects.
//!
//! The grad in the headline reproducer is incidental: the same defect
//! reproduces on the pure forward control (scalar return + constant
//! `expand`, no `grad`) -- see `issue_300_forward_scalar_const_expand_*`.
//!
//! This file is the closing oracle for the "emit but never compile"
//! coverage gap called out in #300 and in the #288 test's note: it EMITS
//! the C *and* invokes the native C compiler, asserting the emitted kernel
//! compiles cleanly and runs to the correct gradient `[2.5, 2.5]`.

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
use tempfile::tempdir;

/// Locate `target/debug/` from the test binary's path. The test binary
/// lives at `<target>/debug/deps/<binary>`, so `..` twice yields the
/// debug dir. (Mirror of `cbackend_cast_memcpy.rs`.)
fn target_debug_dir() -> PathBuf {
    let exe = std::env::current_exe().expect("current_exe failed");
    exe.parent()
        .and_then(Path::parent)
        .map(PathBuf::from)
        .expect("could not resolve target/debug dir from current_exe")
}

/// Mirror of `cbackend_cast_memcpy.rs::ensure_runtime_static_lib`. When
/// `chelis-runtime` is built as a dev-dependency, cargo only emits the
/// hashed staticlib in `target/debug/deps/`; the test cc invocation links
/// against the conventional `target/debug/libchelis_runtime.a`.
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
    let tmp = canonical.with_extension(format!("a.tmp.{}", std::process::id()));
    fs::copy(&hashed, &tmp)?;
    match fs::rename(&tmp, canonical) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && canonical.exists() => Ok(()),
        Err(e) => {
            let _ = fs::remove_file(&tmp);
            Err(e)
        }
    }
}

/// Resolve the C compiler the way the existing C-backend exec tests do:
/// honor `$CC`, else `cc`. CI runners provide one.
fn c_compiler() -> String {
    std::env::var("CC").unwrap_or_else(|_| "cc".to_string())
}

/// Run `chelis build --target c` on `source`, returning the build dir
/// (containing the emitted `<stem>.c`, headers, and runtime static lib).
fn chelis_build_c(source: &str, stem: &str) -> tempfile::TempDir {
    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join(format!("{stem}.ch"));
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
            dir.path().join(format!("{stem}.c")).to_str().unwrap(),
        ])
        .assert()
        .success();

    dir
}

/// Compile the emitted `<stem>.c` (which carries its own `main()` driving
/// the top-level `out = ...` binding) against the runtime static lib, run
/// it, and return stdout. Asserts both the compile and the run succeed --
/// the compile assertion is the #300 regression guard.
fn compile_and_run_emitted(build_dir: &Path, kernel_c: &Path) -> String {
    let canonical = target_debug_dir().join("libchelis_runtime.a");
    ensure_runtime_static_lib(&canonical).expect("materialize libchelis_runtime.a");

    let bin = build_dir.join("issue_300_bin");
    let compile = StdCommand::new(c_compiler())
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
        .expect("invoke C compiler");
    assert!(
        compile.status.success(),
        "emitted grad/expand kernel must compile cleanly (issue #300); \
         compiler stderr=\n{}",
        String::from_utf8_lossy(&compile.stderr),
    );

    let run = StdCommand::new(&bin).output().expect("run emitted binary");
    assert!(
        run.status.success(),
        "emitted binary exited non-zero: stdout={} stderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr),
    );
    String::from_utf8_lossy(&run.stdout).into_owned()
}

/// The headline reproducer (verbatim from issue #300): `grad(f)(x)` over
/// the constant-broadcast idiom. The emitted C must compile cleanly and
/// run to `df(x) = [2.5, 2.5]` (the host evaluator's numeric oracle).
#[test]
fn issue_300_grad_const_expand_kernel_compiles_and_runs() {
    let source = "module Repro.GradExpandConst\n\
def f(x: tensor[2, f32]) -> f32 = {\n  \
  k = expand(scalar_to_tensor(cast(2.5, f32)), cast(0, int32), cast(2, int32))\n  \
  tensor_to_scalar(sum(mul(x, k), cast(0, int32)))\n\
}\n\
def df(x: tensor[2, f32]) -> tensor[2, f32] = grad(f)(x)\n\
out = df(to_tensor([3.0, 4.0]))\n";

    let build = chelis_build_c(source, "repro");
    let kernel_c = build.path().join("repro.c");
    let stdout = compile_and_run_emitted(build.path(), &kernel_c);
    assert!(
        stdout.contains("shape=[2]") && stdout.contains("data=[2.5, 2.5]"),
        "grad(f)(x) must print [2.5, 2.5]; got stdout={stdout:?}",
    );
}

/// Minimal non-grad control: the same defect reproduces on the pure
/// forward path. A scalar-returning function with a constant `expand`
/// binding consumed by a tensor reduction. `f(x) = sum(x * 2.5)` for
/// `x = [3, 4]` is `2.5 * 7 = 17.5`.
#[test]
fn issue_300_forward_scalar_const_expand_compiles_and_runs() {
    let source = "module Repro.ForwardConstExpand\n\
def h(x: tensor[2, f32]) -> f32 = {\n  \
  k = expand(scalar_to_tensor(cast(2.5, f32)), cast(0, int32), cast(2, int32))\n  \
  tensor_to_scalar(sum(mul(x, k), cast(0, int32)))\n\
}\n\
out = h(to_tensor([3.0, 4.0]))\n";

    let build = chelis_build_c(source, "fwd");
    let kernel_c = build.path().join("fwd.c");
    let stdout = compile_and_run_emitted(build.path(), &kernel_c);
    let trimmed = stdout.trim();
    assert!(
        trimmed.contains("17.5"),
        "h(x) = sum(x * 2.5) for x=[3,4] must be 17.5; got stdout={trimmed:?}",
    );
}

/// A tensor-returning function with a constant-`expand` binding consumed
/// by a tensor helper, driven through a host call (`out = scale(...)`).
/// This pins that the constant value is actually materialized into the
/// broadcast tensor: `scale(x) = x * 2.5` for `x = [3, 4]` is
/// `[7.5, 10.0]`. The pre-fix `scalar_to_tensor` f64-vs-f32 storage
/// mismatch zeroed the constant, so `mul(x, k)` collapsed to `x * 0`.
#[test]
fn issue_300_scale_const_expand_materializes_value() {
    let source = "module Repro.ScaleConstExpand\n\
def scale(x: tensor[2, f32]) -> tensor[2, f32] = {\n  \
  k = expand(scalar_to_tensor(cast(2.5, f32)), cast(0, int32), cast(2, int32))\n  \
  mul(x, k)\n\
}\n\
out = scale(to_tensor([3.0, 4.0]))\n";

    let build = chelis_build_c(source, "scale");
    let kernel_c = build.path().join("scale.c");
    let stdout = compile_and_run_emitted(build.path(), &kernel_c);
    assert!(
        stdout.contains("shape=[2]") && stdout.contains("data=[7.5, 10.0]"),
        "scale(x) = x * 2.5 for x=[3,4] must print [7.5, 10.0]; got stdout={stdout:?}",
    );
}

/// The f64 sibling of the scale test, isolating the f64 const-broadcast
/// materialization with zero tensor inputs. The #300 emit fix dispatches
/// `scalar_to_tensor` on the result precision: an f64 const-broadcast must
/// route through `chelis_scalar_tensor_from_f64` (8-byte storage), NOT the
/// new f32 default. `1.1` is the canonical f32-truncation smoking gun: an
/// f32 round-trip of 1.1 widens back to `1.100000023841858`, so if the
/// constant were materialized at f32 the printed f64 tensor would carry
/// that signature instead of `1.1`. `expand(scalar_to_tensor(1.1), 0, 2)`
/// at f64 is the rank-1 tensor `[1.1, 1.1]` at full f64 precision; the
/// helper has zero tensor inputs, so it also exercises the NULL-inputs /
/// count-0 path on the f64 surface.
#[test]
fn issue_300_const_expand_f64_keeps_full_precision() {
    let source = "module Repro.ConstExpandF64\n\
def make_const() -> tensor[2, f64] =\n  \
  expand(scalar_to_tensor(cast(1.1, f64)), cast(0, int32), cast(2, int32))\n\
out = make_const()\n";

    let build = chelis_build_c(source, "const64");
    let kernel_c = build.path().join("const64.c");
    let stdout = compile_and_run_emitted(build.path(), &kernel_c);
    assert!(
        stdout.contains("shape=[2]"),
        "make_const must print a rank-1 size-2 tensor; got stdout={stdout:?}",
    );
    assert!(
        !stdout.contains("1.100000023841858"),
        "f64 const-broadcast must not collapse to the f32-truncated value \
         (issue #300 f64 dispatch); got stdout={stdout:?}",
    );
    assert!(
        stdout.contains("data=[1.1, 1.1]"),
        "expand(scalar_to_tensor(1.1), 0, 2) at f64 must print [1.1, 1.1]; \
         got stdout={stdout:?}",
    );
}
