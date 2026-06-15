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

/// Pin the eval-side grad gap so its eventual fix surfaces here: the host
/// runtime grad lane errors with `missing required input \`w\`` instead of
/// serving the captured binding (pre-existing, NOT introduced or fixed by
/// the #352 backend change). When this starts passing eval, replace this
/// pin with an eval-vs-backend agreement assertion in
/// `issue_352_grad_over_capturing_def_compiles_and_runs`.
#[test]
fn issue_352_grad_over_capturing_def_eval_gap() {
    let source = "w = to_tensor([10.0, 20.0])\n\
def f(x: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(mul(x, w), 0))\n\
def df(x: tensor[2, f32]) -> tensor[2, f32] = grad(f)(x)\n\
out = df(to_tensor([3.0, 4.0]))\n";

    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join("gradcap.ch");
    fs::write(&src_path, source).expect("write .ch source");
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", src_path.to_str().unwrap()])
        .output()
        .expect("invoke chelis eval");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("missing required input `w`"),
        "pinned eval-side grad-capture gap changed shape: stderr={stderr:?} \
         stdout={:?}",
        String::from_utf8_lossy(&output.stdout),
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

/// KNOWN GAP (pre-existing #352 residue, found by the #376 review; NOT
/// introduced or fixed by #376): a host-lane def capturing a top-level
/// SCALAR binding still emits uncompilable C. Root cause differs from the
/// tensor case #376 fixed: the DAG lane claims the binding
/// (`skip_for_lowered` in `chelis-ir`'s host lowering), so it never reaches
/// `HostProgram::globals` and the hoist pass never sees it -- generated
/// `main()` contains no binding for `c` at all and the def body's bare name
/// dangles. `chelis eval` computes the same program correctly, so this is
/// the remaining backend-side eval-vs-backend parity hole of the #352
/// class. When the lowering fix lands this pin fails; replace it with a
/// compile-run-eval agreement assertion like the tensor arms above.
#[test]
fn issue_352_scalar_capture_still_uncompilable_gap() {
    let source = "c = 2.5\n\
def f(x: f32) -> f32 = (x + c)\n\
out = f(1.0)\n";

    let build = chelis_build_c(source, "scalarcap");
    let (compile, _) = compile_emitted(build.path(), &build.path().join("scalarcap.c"));
    assert!(
        !compile.status.success(),
        "pinned gap unexpectedly fixed: scalar-capture C now compiles; \
         promote this pin to a compile-run-eval agreement assertion",
    );
    let stderr = String::from_utf8_lossy(&compile.stderr);
    assert!(
        stderr.contains("undeclared"),
        "pinned gap changed shape: expected an undeclared-identifier \
         diagnostic for the captured scalar `c`; compiler stderr={stderr:?}",
    );

    // The eval side of the parity gap is already correct.
    let eval_out = chelis_eval(source, "scalarcap");
    assert_eq!(
        binding_line(&eval_out, "out"),
        "out = tensor(shape=[], data=[3.5])",
        "eval must keep computing the scalar-capture program correctly",
    );
}

/// KNOWN GAP (pre-existing class, found by the #376 review; NOT introduced
/// or fixed by #376): `vmap` over a def that captures a top-level binding.
/// The #376 hoist makes the emitted C COMPILE (pre-fix it failed with the
/// same undeclared-identifier break as the headline), but the vmap-lane
/// helper types the captured input as BATCHED -- it validates `w` at rank 2
/// while `main()` passes the rank-1 binding, so the binary aborts at
/// runtime. `chelis eval` fails on the same program for the eval-side
/// reason already pinned by `issue_352_grad_over_capturing_def_eval_gap`
/// (the host-runtime transform lane does not serve captured bindings).
/// Both failure shapes are pinned so either side's fix surfaces here.
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
    assert!(
        eval_stderr.contains("missing required input `w`"),
        "pinned eval-side vmap-capture gap changed shape: stderr={eval_stderr:?}",
    );
}

/// KNOWN GAP (pre-existing class, found by the #376 review; NOT introduced
/// by #376): top-level binding names are emitted verbatim into C with no
/// sanitization layer. A binding named `main` passes `chelis check`, and
/// when captured by a def the #376 hoist emits `static chelis_tensor* main;`
/// which collides with the generated `int main(void)`. This is NOT a
/// regression: pre-#376 the same program failed cc with the undeclared-
/// identifier break instead, and a binding named a C keyword (`register`)
/// breaks even uncaptured on both sides of #376 (`main()` locals also use
/// verbatim names). Pinned so a future identifier-sanitization or
/// check-time rejection surfaces here.
#[test]
fn issue_352_captured_binding_named_main_emits_illegal_c_gap() {
    let source = "main = to_tensor([10.0, 20.0])\n\
def f(x: tensor[2, f32]) -> tensor[2, f32] = add(x, main)\n\
out = f(to_tensor([1.0, 2.0]))\n";

    let build = chelis_build_c(source, "mainname");
    let (compile, _) = compile_emitted(build.path(), &build.path().join("mainname.c"));
    assert!(
        !compile.status.success(),
        "pinned gap unexpectedly fixed: a captured binding named `main` now \
         compiles; promote this pin to a compile-run-eval agreement \
         assertion (or to a check-time rejection assertion)",
    );
    assert!(
        String::from_utf8_lossy(&compile.stderr).contains("main"),
        "pinned gap changed shape: expected the `main` symbol collision in \
         the compiler diagnostic; stderr={:?}",
        String::from_utf8_lossy(&compile.stderr),
    );
}
