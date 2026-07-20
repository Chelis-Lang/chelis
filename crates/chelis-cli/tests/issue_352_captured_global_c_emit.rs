//! Issue #352: a def whose body references a free top-level tensor binding
//! built to UNCOMPILABLE C: the emitted host function referenced the binding
//! by name (`__tensor_arg1_1 = w;`) with no declaration in scope, so the
//! native compiler rejected the translation unit with `use of undeclared
//! identifier 'w'`. `chelis eval` computed the same program correctly (the
//! #346 load-closure serves the binding's value), making this an
//! eval-vs-backend parity gap on the backend side.
//!
//! Root cause: `emit_host_program` declares every top-level binding as a
//! LOCAL inside the generated `main()`, while compiled host functions are
//! emitted as file-scope functions whose only in-scope names are their own
//! parameters. The tensor helper already threads the captured binding as an
//! input slot (with shape validation); only the host-level name reference
//! dangled.
//!
//! Fix shape: top-level bindings referenced by any compiled host function
//! are hoisted to file-scope `static` declarations (one per binding, before
//! the function definitions); `main()` assigns them in binding order instead
//! of declaring locals. Both the function bodies and `main()` then resolve
//! the same object, mirroring eval's call-time load-closure semantics.
//! Check-time name resolution already rejects forward references from a
//! call site to a later binding, so every captured binding is initialized
//! before any user call runs.
//!
//! This file is the closing oracle for #352: build --target c, compile with
//! the native cc, run, and assert exact output plus `chelis eval` agreement.
//! The `--target hip` host lane shares `emit_host_program` via
//! `codegen_host_program`, pinned by an emit-shape assertion (no hipcc run).

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
use tempfile::tempdir;

/// Locate `target/debug/` from the test binary's path. The test binary
/// lives at `<target>/debug/deps/<binary>`, so `..` twice yields the
/// debug dir. (Mirror of `issue_300_grad_codegen_compiles.rs`.)
fn target_debug_dir() -> PathBuf {
    let exe = std::env::current_exe().expect("current_exe failed");
    exe.parent()
        .and_then(Path::parent)
        .map(PathBuf::from)
        .expect("could not resolve target/debug dir from current_exe")
}

/// Mirror of `issue_300_grad_codegen_compiles.rs::ensure_runtime_static_lib`.
/// When `chelis-runtime` is built as a dev-dependency, cargo only emits the
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

/// Compile the emitted `<stem>.c` against the runtime static lib without
/// asserting on the outcome. Returns the compiler `Output` and the path the
/// binary lands at on success. Split out of `compile_and_run_emitted` so
/// the known-gap pins below can assert that compilation FAILS with the
/// pinned diagnostic instead of panicking inside the helper.
fn compile_emitted(build_dir: &Path, kernel_c: &Path) -> (std::process::Output, PathBuf) {
    let canonical = target_debug_dir().join("libchelis_runtime.a");
    ensure_runtime_static_lib(&canonical).expect("materialize libchelis_runtime.a");

    let bin = build_dir.join("issue_352_bin");
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
    (compile, bin)
}

/// Compile the emitted `<stem>.c` (which carries its own `main()` driving
/// the top-level bindings) against the runtime static lib, run it, and
/// return stdout. The compile assertion (cc exit 0, default flags) is the
/// #352 contract invariant: emitted C must compile cleanly.
fn compile_and_run_emitted(build_dir: &Path, kernel_c: &Path) -> String {
    let (compile, bin) = compile_emitted(build_dir, kernel_c);
    assert!(
        compile.status.success(),
        "emitted C for a def capturing a top-level binding must compile \
         cleanly (issue #352); compiler stderr=\n{}",
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

/// Run `chelis eval --file` on `source` and return trimmed stdout.
/// Asserts the eval call succeeds. (Mirror of
/// `issue_300_grad_codegen_compiles.rs::chelis_eval`.)
fn chelis_eval(source: &str, stem: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join(format!("{stem}.ch"));
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

/// Extract the `<name> = ...` display line from a stdout dump. Both the
/// compiled binary and `chelis eval` print top-level bindings in this
/// shape, so exact line equality is the agreement oracle.
fn binding_line<'a>(stdout: &'a str, name: &str) -> &'a str {
    let prefix = format!("{name} = ");
    stdout
        .lines()
        .find(|line| line.starts_with(&prefix))
        .unwrap_or_else(|| panic!("no `{name} = ...` line in stdout: {stdout:?}"))
        .trim_end()
}

/// Headline reproducer (verbatim from issue #352): free top-level tensor
/// binding `w` captured by a def whose body also uses a named reduce.
/// Pre-fix: clang failed with `use of undeclared identifier 'w'`.
/// Acceptance: compiles, runs, prints exactly
/// `out = tensor(shape=[2], data=[39.0, 57.0])`, and eval agrees.
#[test]
fn issue_352_captured_global_named_reduce_compiles_runs_and_evals() {
    let source = "w = to_tensor([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]])\n\
def f(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = sum(add(x, w), seq)\n\
out = f(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n";

    let build = chelis_build_c(source, "repro");
    let stdout = compile_and_run_emitted(build.path(), &build.path().join("repro.c"));
    assert_eq!(
        binding_line(&stdout, "out"),
        "out = tensor(shape=[2], data=[39.0, 57.0])",
        "issue #352 acceptance output; full stdout={stdout:?}",
    );

    let eval_out = chelis_eval(source, "repro");
    assert_eq!(
        binding_line(&stdout, "out"),
        binding_line(&eval_out, "out"),
        "eval and compiled C must agree on `out` (issue #352)",
    );
}

/// Capture without the named reduce: plain `add(x, w)` then a concrete-axis
/// `sum`. Confirms the fix is about the captured name, not reduce-specific.
#[test]
fn issue_352_captured_global_concrete_sum_compiles_and_runs() {
    let source = "w = to_tensor([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]])\n\
def f(x: tensor[2, 3, f32]) -> tensor[2, f32] = sum(add(x, w), 1)\n\
out = f(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n";

    let build = chelis_build_c(source, "concrete");
    let stdout = compile_and_run_emitted(build.path(), &build.path().join("concrete.c"));
    assert_eq!(
        binding_line(&stdout, "out"),
        "out = tensor(shape=[2], data=[39.0, 57.0])",
        "concrete-axis capture; full stdout={stdout:?}",
    );

    let eval_out = chelis_eval(source, "concrete");
    assert_eq!(binding_line(&stdout, "out"), binding_line(&eval_out, "out"));
}

/// TWO captured bindings in one def body.
#[test]
fn issue_352_two_captured_globals_in_one_def() {
    let source = "w1 = to_tensor([10.0, 20.0])\n\
w2 = to_tensor([100.0, 200.0])\n\
def f(x: tensor[2, f32]) -> tensor[2, f32] = add(add(x, w1), w2)\n\
out = f(to_tensor([1.0, 2.0]))\n";

    let build = chelis_build_c(source, "twocaps");
    let stdout = compile_and_run_emitted(build.path(), &build.path().join("twocaps.c"));
    assert_eq!(
        binding_line(&stdout, "out"),
        "out = tensor(shape=[2], data=[111.0, 222.0])",
        "two captures in one def; full stdout={stdout:?}",
    );

    let eval_out = chelis_eval(source, "twocaps");
    assert_eq!(binding_line(&stdout, "out"), binding_line(&eval_out, "out"));
}

/// The SAME binding captured by TWO defs: must be declared exactly once at
/// file scope (no redefinition / duplicate symbol), and both defs must see
/// the same value.
#[test]
fn issue_352_same_global_captured_by_two_defs_declared_once() {
    let source = "w = to_tensor([10.0, 20.0])\n\
def f(x: tensor[2, f32]) -> tensor[2, f32] = add(x, w)\n\
def g(x: tensor[2, f32]) -> tensor[2, f32] = mul(x, w)\n\
a = f(to_tensor([1.0, 2.0]))\n\
b = g(to_tensor([1.0, 2.0]))\n";

    let build = chelis_build_c(source, "twodefs");
    let kernel_c = build.path().join("twodefs.c");

    // Emit-shape invariant: exactly one file-scope declaration of `w`.
    let c_source = fs::read_to_string(&kernel_c).expect("read emitted C");
    let decl_count = c_source
        .lines()
        .filter(|line| line.trim() == "static chelis_tensor* w;")
        .count();
    assert_eq!(
        decl_count, 1,
        "captured binding `w` must be hoisted to exactly one file-scope \
         static declaration (issue #352); emitted C=\n{c_source}",
    );

    let stdout = compile_and_run_emitted(build.path(), &kernel_c);
    assert_eq!(
        binding_line(&stdout, "a"),
        "a = tensor(shape=[2], data=[11.0, 22.0])",
        "full stdout={stdout:?}",
    );
    assert_eq!(
        binding_line(&stdout, "b"),
        "b = tensor(shape=[2], data=[10.0, 40.0])",
        "full stdout={stdout:?}",
    );

    let eval_out = chelis_eval(source, "twodefs");
    assert_eq!(binding_line(&stdout, "a"), binding_line(&eval_out, "a"));
    assert_eq!(binding_line(&stdout, "b"), binding_line(&eval_out, "b"));
}

/// Transitive: def `g` calls def `f`, and `f` captures `w`. Only `f`
/// references the binding; the call chain must still compile and run.
#[test]
fn issue_352_transitive_capture_through_def_call() {
    let source = "w = to_tensor([10.0, 20.0])\n\
def f(x: tensor[2, f32]) -> tensor[2, f32] = add(x, w)\n\
def g(x: tensor[2, f32]) -> tensor[2, f32] = neg(f(x))\n\
out = g(to_tensor([1.0, 2.0]))\n";

    let build = chelis_build_c(source, "transitive");
    let stdout = compile_and_run_emitted(build.path(), &build.path().join("transitive.c"));
    assert_eq!(
        binding_line(&stdout, "out"),
        "out = tensor(shape=[2], data=[-11.0, -22.0])",
        "transitive capture; full stdout={stdout:?}",
    );

    let eval_out = chelis_eval(source, "transitive");
    assert_eq!(binding_line(&stdout, "out"), binding_line(&eval_out, "out"));
}

/// A captured f64 binding: interacts with the #372 `cast(literal, p)`
/// precision rule. `0.1` and `0.2` are not f32-representable, so any
/// silent f64->f32 truncation in the captured value shows up as the
/// `1.100000...` signature instead of the exact `1.1` / `1.2`.
#[test]
fn issue_352_captured_f64_global_exact_precision() {
    let source = "w64 = to_tensor([cast(0.1, f64), cast(0.2, f64)])\n\
def f(x: tensor[2, f64]) -> tensor[2, f64] = add(x, w64)\n\
out = f(cast(to_tensor([1.0, 1.0]), f64))\n";

    let build = chelis_build_c(source, "f64cap");
    let stdout = compile_and_run_emitted(build.path(), &build.path().join("f64cap.c"));
    assert_eq!(
        binding_line(&stdout, "out"),
        "out = tensor(shape=[2], data=[1.1, 1.2])",
        "captured f64 binding must stay exact (issue #352 x #372); \
         full stdout={stdout:?}",
    );
    assert!(
        !stdout.contains("1.100000023841858"),
        "f32-truncation signature must not appear; got stdout={stdout:?}",
    );

    let eval_out = chelis_eval(source, "f64cap");
    assert_eq!(
        binding_line(&stdout, "out"),
        binding_line(&eval_out, "out"),
        "eval and compiled C must agree on the exact f64 values",
    );
}

/// grad over a def that captures a top-level binding: gradient w.r.t. the
/// parameter, constant in `w`. d/dx sum(x * w) = w = [10, 20]. The C
/// backend pre-fix emitted the same dangling `w` in BOTH the forward
/// function and the grad-generated function; post-fix the program must
/// compile and run to the correct gradient.
///
/// KNOWN GAP (pinned by `issue_352_grad_over_capturing_def_eval_gap`):
/// `chelis eval` fails on this program for a DIFFERENT, pre-existing
/// reason -- the host-runtime grad lane does not serve captured top-level
/// bindings ("missing required input `w`"). Backend-only assertion here;
/// no eval agreement until that eval-side gap is fixed.
#[test]
fn issue_352_grad_over_capturing_def_compiles_and_runs() {
    let source = "w = to_tensor([10.0, 20.0])\n\
def f(x: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(mul(x, w), 0))\n\
def df(x: tensor[2, f32]) -> tensor[2, f32] = grad(f)(x)\n\
out = df(to_tensor([3.0, 4.0]))\n";

    let build = chelis_build_c(source, "gradcap");
    let stdout = compile_and_run_emitted(build.path(), &build.path().join("gradcap.c"));
    assert_eq!(
        binding_line(&stdout, "out"),
        "out = tensor(shape=[2], data=[10.0, 20.0])",
        "grad w.r.t. the parameter of f(x) = sum(x * w) is w; \
         full stdout={stdout:?}",
    );
}

/// chelis#377 (FIXED, was the pinned eval-side grad-capture gap): the host
/// runtime grad lane now SERVES a captured top-level binding instead of
/// failing with `missing required input \`w\``. `grad(f)(x)` for
/// `f(x) = sum(x * w)` is `w` (constant in `x`), so `df(x) = [10, 20]`.
/// This is the promoted form of the former `..._eval_gap` pin (its own
/// instruction): the eval value must now agree EXACTLY with the C backend,
/// which already compiled and ran this. Root cause was that the transform
/// load callback only served `placeholder_tensors` + `tensor_bindings`, not
/// the captured top-level binding delivered via `captured_env` (the C
/// backend hoists the capture into a file-scope global; eval lacked the
/// parity path). See `crates/chelis-compiler-api/src/runtime/transforms.rs`.
#[test]
fn issue_377_grad_over_capturing_def_evals_and_agrees_with_backend() {
    let source = "w = to_tensor([10.0, 20.0])\n\
def f(x: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(mul(x, w), 0))\n\
def df(x: tensor[2, f32]) -> tensor[2, f32] = grad(f)(x)\n\
out = df(to_tensor([3.0, 4.0]))\n";

    // Eval lane now succeeds and computes the gradient.
    let eval_out = chelis_eval(source, "gradcap");
    assert_eq!(
        binding_line(&eval_out, "out"),
        "out = tensor(shape=[2], data=[10.0, 20.0])",
        "grad w.r.t. the parameter of f(x) = sum(x * w) is w (chelis#377); \
         full eval stdout={eval_out:?}",
    );

    // Eval-vs-backend agreement: the C backend already ran this (see
    // `issue_352_grad_over_capturing_def_compiles_and_runs`); the eval value
    // must match exactly now that the eval-side gap is closed.
    let build = chelis_build_c(source, "gradcap");
    let backend_out = compile_and_run_emitted(build.path(), &build.path().join("gradcap.c"));
    assert_eq!(
        binding_line(&eval_out, "out"),
        binding_line(&backend_out, "out"),
        "eval and compiled C must agree on the captured-binding gradient (chelis#377)",
    );
}

/// NEGATIVE parity: a genuinely-undefined name in a def body must still
/// fail at CHECK time with `UnboundVariable` and never reach codegen. Pins
/// that the capture fix did not loosen name resolution.
#[test]
fn issue_352_undefined_name_in_def_body_still_fails_at_check() {
    let source = "def f(x: &tensor[batch, seq, f32]) -> tensor[batch, f32] = \
sum(add(x, nosuchname), seq)\n\
out = f(to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))\n";

    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join("undef.ch");
    fs::write(&src_path, source).expect("write .ch source");

    let check = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", src_path.to_str().unwrap()])
        .output()
        .expect("invoke chelis check");
    assert!(
        !check.status.success(),
        "check must reject an unbound name in a def body",
    );
    let check_stdout = String::from_utf8_lossy(&check.stdout);
    assert!(
        check_stdout.contains("UnboundVariable") && check_stdout.contains("nosuchname"),
        "check must report UnboundVariable for `nosuchname`; \
         stdout={check_stdout:?}",
    );

    // Build must fail at the check stage and emit no C artifact.
    let out_c = dir.path().join("undef.c");
    let build = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            src_path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_c.to_str().unwrap(),
        ])
        .output()
        .expect("invoke chelis build");
    assert!(
        !build.status.success(),
        "build must fail at check for an unbound name",
    );
    let build_stderr = String::from_utf8_lossy(&build.stderr);
    assert!(
        build_stderr.contains("unbound variable") && build_stderr.contains("nosuchname"),
        "build must surface the unbound-variable error; stderr={build_stderr:?}",
    );
    assert!(
        !out_c.exists(),
        "no C artifact may be emitted for a program that fails check",
    );
}

/// `--target hip` host lane: a scalar-returning def (no tensor-only entry
/// signature) routes through `cmd_build_hip_host`, which reuses the same
/// `emit_host_program` C host emission. Emit-shape assertion only -- the
/// generated host source must declare the captured binding at file scope.
/// hipcc is deliberately not invoked (manual HIP gates own execution).
#[test]
fn issue_352_hip_host_lane_declares_captured_global() {
    let source = "w = to_tensor([10.0, 20.0])\n\
def f(x: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(mul(x, w), 0))\n\
out = f(to_tensor([3.0, 4.0]))\n";

    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join("hipcap.ch");
    fs::write(&src_path, source).expect("write .ch source");
    let out_cpp = dir.path().join("hipcap.cpp");

    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            src_path.to_str().unwrap(),
            "--target",
            "hip",
            "--output",
            out_cpp.to_str().unwrap(),
        ])
        .assert()
        .success();

    let host_source = fs::read_to_string(&out_cpp).expect("read emitted HIP host source");
    assert!(
        host_source
            .lines()
            .any(|line| line.trim() == "static chelis_tensor* w;"),
        "HIP host lane shares the host emit path and must declare the \
         captured binding at file scope (issue #352); emitted=\n{host_source}",
    );
    assert!(
        host_source.contains("= w;"),
        "sanity: the host function still references the captured binding; \
         emitted=\n{host_source}",
    );
}

/// chelis#378 (was a pinned #352-residue gap; fixed in the WS-2A lowering
/// pass): a host-lane def capturing a top-level SCALAR binding now emits
/// compilable C that runs and agrees with eval. Root cause was that the DAG
/// lane claimed the scalar binding (`skip_for_lowered` in `chelis-ir`'s host
/// lowering, because a `(lit ...)` body is DAG-lowerable), so it never
/// reached `HostProgram::globals` and the host function body referenced an
/// undeclared `c`. The fix keeps a value binding captured by a host-lane
/// function in `host.globals` (the C emitter's `captured_global_names` then
/// declares it at file scope). `f(x) = x + c` with `c = 2.5` and `x = 1.0`
/// yields `3.5`, matching eval.
#[test]
fn issue_378_scalar_capture_compiles_runs_and_evals() {
    let source = "c = 2.5\n\
def f(x: f32) -> f32 = (x + c)\n\
out = f(1.0)\n";

    let build = chelis_build_c(source, "scalarcap");
    let stdout = compile_and_run_emitted(build.path(), &build.path().join("scalarcap.c"));
    // Both lanes print the scalar root bare since chelis#732 Phase 1's
    // [05-OBS-4] (the chelis#775 decision): the evaluator's old
    // `tensor(shape=[], ...)` wrapper was an internal realization artifact
    // and is no longer an exit form, so the lanes now agree byte-for-byte.
    assert_eq!(
        binding_line(&stdout, "out"),
        "out = 3.5",
        "issue #378 acceptance: scalar capture compiles, runs, and the C \
         backend prints the captured-binding result; full stdout={stdout:?}",
    );

    let eval_out = chelis_eval(source, "scalarcap");
    assert_eq!(
        binding_line(&eval_out, "out"),
        "out = 3.5",
        "evaluator must compute the same scalar-capture value (issue #378)",
    );
}

/// KNOWN GAP (RESIDUAL, advanced by chelis#377): `vmap` over a def that
/// captures a top-level binding. chelis#377 closed the eval-side
/// missing-input half — the transform lane now SERVES the captured `w` (by
/// resolving top-level bindings referenced by the inlined body's `Load`s) —
/// so the eval failure has moved one layer deeper, to the SAME vmap-lane
/// batch-typing defect the backend already exhibits: `vectorize_axis0`
/// prepends the batch axis to the captured `w`'s node too, typing it `[3, 2]`
/// while the actual binding is rank-1 `[2]`. Correct vmap-with-captures must
/// BROADCAST the capture across the batch axis, not batch it; that vmap-lane
/// capability is the remaining residual (tracked under chelis#377). The grad
/// half of the capture gap is fully fixed and promoted in
/// `issue_377_grad_over_capturing_def_evals_and_agrees_with_backend`.
///
/// FAILURE-MODE PINS. The backend side fails the COMPILED binary at runtime
/// with `expected rank 2, got 1`. The eval side fails with a CLEAN transform-
/// lane diagnostic naming the capture (chelis#377) — NOT the uncontrolled
/// `binary_map` shape-assertion PANIC it produced before the fix (serving the
/// rank-1 capture into a vmap-batched rank-2 `Load`). This test asserts the
/// eval failure is clean (never a panic) so the residual cannot regress into
/// an evaluator crash, and surfaces loudly when the broadcast fix lands.
#[test]
fn issue_352_vmap_over_capturing_def_gap() {
    let source = "w = to_tensor([10.0, 20.0])\n\
def dot_w(x: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(mul(x, w), 0))\n\
def fv(xs: tensor[3, 2, f32]) -> tensor[3, f32] = xs |> vmap(dot_w, axis=0)\n\
out = fv(to_tensor([[1.0, 1.0], [2.0, 2.0], [0.0, 1.0]]))\n";

    let build = chelis_build_c(source, "vmapcap");
    let (compile, bin) = compile_emitted(build.path(), &build.path().join("vmapcap.c"));
    assert!(
        compile.status.success(),
        "the #376 hoist must keep the vmap-over-capture program COMPILING; \
         compiler stderr=\n{}",
        String::from_utf8_lossy(&compile.stderr),
    );

    let run = StdCommand::new(&bin).output().expect("run emitted binary");
    assert!(
        !run.status.success(),
        "pinned gap unexpectedly fixed: vmap-over-capture binary now runs; \
         promote this pin to an exact-output + eval agreement assertion",
    );
    let run_stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run_stderr.contains("expected rank 2, got 1"),
        "pinned gap changed shape: the vmap helper batched the captured \
         input (rank-2 validation vs the rank-1 binding); stderr={run_stderr:?}",
    );

    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join("vmapcap.ch");
    fs::write(&src_path, source).expect("write .ch source");
    let eval = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", src_path.to_str().unwrap()])
        .output()
        .expect("invoke chelis eval");
    assert!(
        !eval.status.success(),
        "eval side of the gap is also broken"
    );
    let eval_stderr = String::from_utf8_lossy(&eval.stderr);
    // chelis#377 review fix: the eval side now fails with a CLEAN transform-
    // lane diagnostic, NOT the uncontrolled `binary_map` shape-assertion panic
    // it used to produce (the served rank-1 capture reaching an elementwise op
    // against the vmap-batched rank-2 `Load`). The transform lane rejects the
    // capture rank mismatch before eval. Pin BOTH that it stays a failure (the
    // broadcast capability is still a residual) AND that it is never a panic.
    assert!(
        !eval_stderr.contains("panicked"),
        "vmap-over-capture must fail with a clean diagnostic, never an evaluator \
         panic (chelis#377); stderr={eval_stderr:?}",
    );
    assert!(
        eval_stderr.contains("capturing top-level binding `w`")
            && eval_stderr.contains("chelis#377"),
        "pinned eval-side vmap-capture residual must report the clean \
         capture-batching diagnostic citing chelis#377; stderr={eval_stderr:?}",
    );
}

/// chelis#377 review: `vmap(grad(f))` (per-sample gradients) where `f` captures
/// a top-level binding hits the SAME vmap-lane capture-batching residual as a
/// plain `vmap` over a capturing def, and must likewise fail with a CLEAN
/// diagnostic — never the `binary_map` shape-assertion PANIC it produced before
/// the fix. Pinned so the per-sample-grad composition cannot silently regress
/// into an evaluator crash; promote to a value oracle when vmap-with-captures
/// broadcasts the capture across the batch axis (chelis#377).
#[test]
fn issue_377_vmap_of_grad_over_capturing_def_fails_clean() {
    let source = "w = to_tensor([10.0, 20.0])\n\
def f(x: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(mul(x, w), 0))\n\
def gradf(x: tensor[2, f32]) -> tensor[2, f32] = grad(f)(x)\n\
def batched(xs: tensor[3, 2, f32]) -> tensor[3, 2, f32] = xs |> vmap(gradf, axis=0)\n\
out = batched(to_tensor([[1.0, 1.0], [2.0, 2.0], [0.0, 1.0]]))\n";

    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join("vmapgradcap.ch");
    fs::write(&src_path, source).expect("write .ch source");
    let eval = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", src_path.to_str().unwrap()])
        .output()
        .expect("invoke chelis eval");
    assert!(
        !eval.status.success(),
        "vmap(grad(capturing def)) is still a residual (chelis#377)",
    );
    let eval_stderr = String::from_utf8_lossy(&eval.stderr);
    assert!(
        !eval_stderr.contains("panicked"),
        "vmap(grad(capture)) must fail clean, never panic; stderr={eval_stderr:?}",
    );
    assert!(
        eval_stderr.contains("capturing top-level binding `w`")
            && eval_stderr.contains("chelis#377"),
        "must report the clean capture-batching diagnostic; stderr={eval_stderr:?}",
    );
}

/// FIXED (#379): top-level binding names spelled like a C keyword or the
/// generated entry point are mangled at C-emit time (`c_ident` in
/// `chelis-backend-c/src/host_emit.rs`), so a binding named `main` captured
/// by a def no longer collides with `int main(void)` — the emitted C
/// compiles, runs, and agrees with `chelis eval`. This was the promoted
/// form of the former `..._emits_illegal_c_gap` pin (the pin's own
/// instruction). The full compile-run-eval agreement oracle lives in
/// `ws2b_numeric_identifier_divergence::issue_379_binding_named_main_compiles_and_matches_eval`;
/// this arm keeps the regression guard local to the #352 capture corpus.
#[test]
fn issue_352_captured_binding_named_main_compiles_and_runs() {
    let source = "main = to_tensor([10.0, 20.0])\n\
def f(x: tensor[2, f32]) -> tensor[2, f32] = add(x, main)\n\
out = f(to_tensor([1.0, 2.0]))\n";

    let build = chelis_build_c(source, "mainname");
    let stdout = compile_and_run_emitted(build.path(), &build.path().join("mainname.c"));
    assert_eq!(
        binding_line(&stdout, "out"),
        "out = tensor(shape=[2], data=[11.0, 22.0])",
        "a captured binding named `main` must mangle and run (#379); \
         full stdout={stdout:?}",
    );

    let eval_out = chelis_eval(source, "mainname");
    assert_eq!(
        binding_line(&stdout, "out"),
        binding_line(&eval_out, "out"),
        "eval and compiled C must agree for a binding named `main` (#379)",
    );
}
