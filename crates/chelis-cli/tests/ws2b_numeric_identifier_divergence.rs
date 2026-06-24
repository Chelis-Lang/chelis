//! WS-2B: eval-vs-C-backend agreement for the numeric-semantics and
//! C-identifier-hygiene divergence family. The evaluator is the reference;
//! the C backend must agree (byte-identical on integer dtypes). No silent
//! wrong answers — a wrong value is worse than a trap.
//!
//! Covers four issues, each with a POSITIVE test and a NEGATIVE / parity
//! test:
//!
//! * #387 — integer `div`/`mod` by zero must TRAP, not return a finite
//!   wrong value. The evaluator halts with one clean diagnostic shared
//!   between `div` and `mod`; the C backend follows the platform's SIGFPE.
//!   Float `div` keeps IEEE-754 (`1.0 / 0.0 == inf`).
//! * #381 — `scalar_to_tensor` of a top-level scalar binding that a def
//!   captures (and that the DAG lane materializes as a rank-0 tensor) must
//!   evaluate, matching the C backend / the DAG pass-through.
//! * #347 — the C backend must print `argmax_reduce`/`argmin_reduce` int64
//!   results as the integer indices, not the reinterpreted f32 bit pattern.
//! * #379 — top-level bindings spelled like C keywords or the emitted
//!   helper scheme must produce compilable C (identifier mangling).
//!
//! #378 (a captured *scalar* top-level binding still emits uncompilable C)
//! is a chelis-ir host-lowering gap outside this change's surface: the
//! scalar binding is dropped by `skip_for_lowered` before the C backend
//! receives it, so the backend cannot emit a value it never gets. The
//! eval-vs-C parity arm for #381 is gated on that lowering fix and is
//! therefore an eval-only oracle here; the #347/#379 arms exercise full
//! eval-vs-C agreement.

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
use tempfile::tempdir;

// -----------------------------------------------------------------------------
// Harness (mirrors issue_352_captured_global_c_emit.rs)
// -----------------------------------------------------------------------------

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

fn c_compiler() -> String {
    std::env::var("CC").unwrap_or_else(|_| "cc".to_string())
}

/// Run `chelis build --target c` on `source`, returning the build dir.
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
/// asserting on the outcome; returns the compiler `Output` and the path the
/// binary lands at on success.
fn compile_emitted(build_dir: &Path, kernel_c: &Path) -> (std::process::Output, PathBuf) {
    let canonical = target_debug_dir().join("libchelis_runtime.a");
    ensure_runtime_static_lib(&canonical).expect("materialize libchelis_runtime.a");

    let bin = build_dir.join("ws2b_bin");
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

/// Compile + run the emitted C, asserting clean compilation and a zero
/// exit. Returns stdout.
fn compile_and_run_emitted(build_dir: &Path, kernel_c: &Path) -> String {
    let (compile, bin) = compile_emitted(build_dir, kernel_c);
    assert!(
        compile.status.success(),
        "emitted C must compile cleanly; compiler stderr=\n{}",
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

/// Run `chelis eval --file` and return trimmed stdout; asserts success.
fn chelis_eval_ok(source: &str, stem: &str) -> String {
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

/// Run an inline `chelis eval EXPR`, returning the full `Output` so callers
/// can assert on exit status, stdout, and stderr.
fn chelis_eval_expr(expr: &str) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", expr])
        .output()
        .expect("invoke chelis eval")
}

/// Extract the `<name> = ...` display line from a stdout dump.
fn binding_line<'a>(stdout: &'a str, name: &str) -> &'a str {
    let prefix = format!("{name} = ");
    stdout
        .lines()
        .find(|line| line.starts_with(&prefix))
        .unwrap_or_else(|| panic!("no `{name} = ...` line in stdout: {stdout:?}"))
        .trim_end()
}

/// The rendered `tensor(...)` value of a root, with any `<name> = ` label
/// stripped. `chelis eval` prints a SINGLE-root program's value WITHOUT a
/// label (`tensor(...)`), while the C backend always labels it
/// (`out = tensor(...)`); both render the value identically. This helper
/// normalizes that asymmetry so eval-vs-C parity compares the values.
fn tensor_value<'a>(stdout: &'a str, name: &str) -> &'a str {
    let labeled = format!("{name} = ");
    stdout
        .lines()
        .map(str::trim_end)
        .find_map(|line| {
            if let Some(rest) = line.strip_prefix(&labeled) {
                Some(rest)
            } else if line.starts_with("tensor(") || line.starts_with("error") {
                Some(line)
            } else {
                None
            }
        })
        .unwrap_or_else(|| {
            panic!("no tensor value (labeled `{name}` or bare) in stdout: {stdout:?}")
        })
}

// -----------------------------------------------------------------------------
// #387 — integer division / remainder by zero must TRAP
// -----------------------------------------------------------------------------

/// The single canonical evaluator diagnostic shared by integer `div` and
/// `mod` (the consistency requirement: the two must agree).
const INT_DIV_ZERO_DIAGNOSTIC: &str = "integer division or remainder by zero";

/// POSITIVE: valid integer division is truncating (round toward zero) and
/// agrees byte-for-byte between eval and the C backend. `7/2 == 3`,
/// `-7/2 == -3` per spec/05-risc-primitives.md.
#[test]
fn issue_387_integer_div_truncates_eval_matches_backend() {
    let source = "def d(x: tensor[2, int64], y: tensor[2, int64]) -> tensor[2, int64] = div(x, y)\n\
out = d(cast(to_tensor([7, -7]), int64), cast(to_tensor([2, 2]), int64))\n";

    let build = chelis_build_c(source, "intdiv");
    let stdout = compile_and_run_emitted(build.path(), &build.path().join("intdiv.c"));
    assert_eq!(
        binding_line(&stdout, "out"),
        "out = tensor(shape=[2], data=[3.0, -3.0])",
        "integer div must truncate toward zero (7/2=3, -7/2=-3); stdout={stdout:?}",
    );

    let eval_out = chelis_eval_ok(source, "intdiv");
    assert_eq!(
        tensor_value(&stdout, "out"),
        tensor_value(&eval_out, "out"),
        "eval and C backend must agree byte-for-byte on integer division",
    );
}

/// POSITIVE: float division keeps IEEE-754 semantics — `1.0 / 0.0 == inf`,
/// NOT a trap. The integer trap must not leak into the float lane.
#[test]
fn issue_387_float_div_by_zero_is_ieee_inf() {
    let out = chelis_eval_expr("div(1.0, 0.0)");
    assert!(
        out.status.success(),
        "float div by zero must succeed (IEEE), not trap; stderr={}",
        String::from_utf8_lossy(&out.stderr),
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim_end(),
        "inf",
        "float 1.0/0.0 must be +inf per IEEE-754",
    );
}

/// NEGATIVE: integer `div` by zero traps with the clean diagnostic instead
/// of returning a finite wrong value (pre-fix: `i64::MAX`). Exit 1, not a
/// silently-wrong scalar.
#[test]
fn issue_387_integer_div_by_zero_traps_in_eval() {
    let out = chelis_eval_expr("div(cast(7, int64), cast(0, int64))");
    assert!(
        !out.status.success(),
        "integer div by zero must trap, not return a value; stdout={}",
        String::from_utf8_lossy(&out.stdout),
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains(INT_DIV_ZERO_DIAGNOSTIC),
        "integer div by zero must emit the canonical diagnostic; stderr={stderr:?}",
    );
    // The pre-fix silently-wrong value must never appear.
    assert!(
        !String::from_utf8_lossy(&out.stdout).contains("9223372036854775807"),
        "the pre-fix i64::MAX wrong value must not be produced",
    );
}

/// NEGATIVE / consistency: integer `mod` by zero traps with the SAME clean
/// diagnostic as `div` (pre-fix: a raw Rust remainder panic). `div` and
/// `mod` must agree.
#[test]
fn issue_387_integer_mod_by_zero_traps_with_same_diagnostic() {
    let out = chelis_eval_expr("mod(cast(7, int64), cast(0, int64))");
    assert!(
        !out.status.success(),
        "integer mod by zero must trap; stdout={}",
        String::from_utf8_lossy(&out.stdout),
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains(INT_DIV_ZERO_DIAGNOSTIC),
        "integer mod by zero must emit the SAME canonical diagnostic as div \
         (div/mod consistency); stderr={stderr:?}",
    );
    assert!(
        !stderr.contains("attempt to calculate the remainder"),
        "the raw Rust remainder panic must be replaced by the clean diagnostic; \
         stderr={stderr:?}",
    );
}

/// NEGATIVE parity: the C backend follows the platform signal (SIGFPE) for
/// integer division by zero — it must NOT print a finite wrong value. Both
/// lanes trap; neither yields a silently-wrong integer.
#[test]
fn issue_387_integer_div_by_zero_traps_in_backend() {
    let source = "def d(x: tensor[2, int64], y: tensor[2, int64]) -> tensor[2, int64] = div(x, y)\n\
out = d(cast(to_tensor([7, 8]), int64), cast(to_tensor([2, 0]), int64))\n";

    let build = chelis_build_c(source, "intdivtrap");
    let (compile, bin) = compile_emitted(build.path(), &build.path().join("intdivtrap.c"));
    assert!(
        compile.status.success(),
        "the program must compile; the trap is a runtime signal, not a compile \
         error; compiler stderr=\n{}",
        String::from_utf8_lossy(&compile.stderr),
    );
    let run = StdCommand::new(&bin).output().expect("run emitted binary");
    assert!(
        !run.status.success(),
        "C backend integer div by zero must trap (platform SIGFPE), not print \
         a finite value; stdout={}",
        String::from_utf8_lossy(&run.stdout),
    );
    // The result tensor must never have been printed.
    assert!(
        !String::from_utf8_lossy(&run.stdout).contains("out ="),
        "no `out = ...` line may be printed when the program traps; stdout={}",
        String::from_utf8_lossy(&run.stdout),
    );
}

// -----------------------------------------------------------------------------
// #381 — scalar_to_tensor of a captured top-level scalar (eval lane)
// -----------------------------------------------------------------------------

/// POSITIVE: a top-level f64 scalar binding captured by a def and fed to
/// `scalar_to_tensor` evaluates correctly. The DAG lane materializes the
/// captured scalar as a rank-0 tensor; the host runtime's
/// `scalar_to_tensor` must accept that rank-0 tensor as the identity rather
/// than erroring "expects scalar input". Pre-fix this failed at runtime.
///
/// (eval-only oracle: the eval-vs-C parity arm is gated on #378, the
/// chelis-ir host-lowering gap that drops a captured *scalar* binding
/// before the C backend can emit it.)
#[test]
fn issue_381_scalar_to_tensor_on_captured_scalar_evals() {
    let source = "c = cast(1.1, f64)\n\
def make(n: tensor[2, f64]) -> tensor[2, f64] = \
add(n, expand(scalar_to_tensor(c), cast(0, int32), cast(2, int32)))\n\
out = make(to_tensor([cast(1.0, f64), cast(2.0, f64)]))\n";

    let eval_out = chelis_eval_ok(source, "s2t_capture");
    assert_eq!(
        binding_line(&eval_out, "out"),
        "out = tensor(shape=[2], data=[2.1, 3.1])",
        "1.1 broadcast-added to [1.0, 2.0] is [2.1, 3.1]; eval stdout={eval_out:?}",
    );
}

/// NEGATIVE: `scalar_to_tensor` of a rank>=1 tensor must still be rejected.
/// The fix only relaxed the rank-0 (scalar-equivalent) case; a genuine
/// multi-element tensor is not a scalar and must fail.
#[test]
fn issue_381_scalar_to_tensor_on_rank1_tensor_still_rejected() {
    let out = chelis_eval_expr("scalar_to_tensor(to_tensor([1.0, 2.0]))");
    assert!(
        !out.status.success(),
        "scalar_to_tensor of a rank-1 tensor must be rejected; stdout={}",
        String::from_utf8_lossy(&out.stdout),
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("scalar_to_tensor"),
        "rejection must name scalar_to_tensor; stderr={stderr:?}",
    );
}

// -----------------------------------------------------------------------------
// #347 — C backend prints int64 argmax/argmin indices, not f32 bit patterns
// -----------------------------------------------------------------------------

/// POSITIVE + parity: `argmax_reduce` over an integer axis prints the
/// integer indices in the C backend, matching eval. Pre-fix the C backend
/// stored the index as a float into the int64 buffer and the print path
/// read it back as the reinterpreted bit pattern (`0x3F800000` =
/// `1065353216` for `1.0f`).
#[test]
fn issue_347_argmax_int_axis_backend_matches_eval() {
    let source = "def am(x: &tensor[batch, seq, f32]) -> tensor[batch, int64] = argmax_reduce(x, 1)\n\
out = am(to_tensor([[1.0, 9.0, 3.0], [7.0, 5.0, 6.0]]))\n";

    let build = chelis_build_c(source, "argmax");
    let stdout = compile_and_run_emitted(build.path(), &build.path().join("argmax.c"));
    assert_eq!(
        binding_line(&stdout, "out"),
        "out = tensor(shape=[2], data=[1.0, 0.0])",
        "argmax of row 0 is index 1, row 1 is index 0; stdout={stdout:?}",
    );
    // The reinterpreted-f32-bits signature must never appear.
    assert!(
        !stdout.contains("1065353216"),
        "the pre-fix reinterpreted-f32-bits value must not appear; stdout={stdout:?}",
    );

    let eval_out = chelis_eval_ok(source, "argmax");
    assert_eq!(
        tensor_value(&stdout, "out"),
        tensor_value(&eval_out, "out"),
        "eval and C backend must agree on argmax int64 indices (#347)",
    );
}

/// POSITIVE + parity: the same fix covers `argmin_reduce` and a named axis.
#[test]
fn issue_347_argmin_named_axis_backend_matches_eval() {
    let source = "def am(x: &tensor[batch, seq, f32]) -> tensor[batch, int64] = argmin_reduce(x, seq)\n\
out = am(to_tensor([[1.0, 9.0, 3.0], [7.0, 5.0, 6.0]]))\n";

    let build = chelis_build_c(source, "argmin");
    let stdout = compile_and_run_emitted(build.path(), &build.path().join("argmin.c"));
    assert_eq!(
        binding_line(&stdout, "out"),
        "out = tensor(shape=[2], data=[0.0, 1.0])",
        "argmin of row 0 is index 0, row 1 is index 1; stdout={stdout:?}",
    );
    assert!(
        !stdout.contains("1065353216"),
        "the pre-fix reinterpreted-f32-bits value must not appear; stdout={stdout:?}",
    );

    let eval_out = chelis_eval_ok(source, "argmin");
    assert_eq!(
        tensor_value(&stdout, "out"),
        tensor_value(&eval_out, "out"),
        "eval and C backend must agree on argmin int64 indices (#347)",
    );
}

// -----------------------------------------------------------------------------
// #379 — C-identifier hygiene for keyword / collision-named bindings
// -----------------------------------------------------------------------------

/// POSITIVE + parity: a top-level binding named exactly a C keyword
/// (`register`) emits compilable C, runs, and agrees with eval. Pre-fix the
/// binding was emitted verbatim as `chelis_tensor* register = ...;`, which
/// is illegal C. The display label must stay the user-facing `register`.
#[test]
fn issue_379_c_keyword_binding_name_compiles_and_matches_eval() {
    let source = "register = to_tensor([10.0, 20.0])\n\
out = add(register, to_tensor([1.0, 2.0]))\n";

    let build = chelis_build_c(source, "kw");
    let kernel_c = build.path().join("kw.c");

    // Emit-shape invariant: the C keyword must be mangled in the declaration.
    let c_source = fs::read_to_string(&kernel_c).expect("read emitted C");
    assert!(
        !c_source.lines().any(
            |line| line.trim_start().starts_with("chelis_tensor* register ")
                || line.trim_start().starts_with("chelis_tensor *register ")
        ),
        "the C keyword `register` must not be emitted as a raw identifier; \
         emitted C=\n{c_source}",
    );
    // The user-facing print label stays raw.
    assert!(
        c_source.contains("\"register\""),
        "the printed display label must remain the user name `register`; \
         emitted C=\n{c_source}",
    );

    let stdout = compile_and_run_emitted(build.path(), &kernel_c);
    assert_eq!(
        binding_line(&stdout, "register"),
        "register = tensor(shape=[2], data=[10.0, 20.0])",
        "stdout={stdout:?}",
    );
    assert_eq!(
        binding_line(&stdout, "out"),
        "out = tensor(shape=[2], data=[11.0, 22.0])",
        "stdout={stdout:?}",
    );

    let eval_out = chelis_eval_ok(source, "kw");
    assert_eq!(
        binding_line(&stdout, "register"),
        binding_line(&eval_out, "register"),
    );
    assert_eq!(binding_line(&stdout, "out"), binding_line(&eval_out, "out"));
}

/// POSITIVE + parity: a binding named `main` (which collides with the
/// generated `int main(void)`) captured by a def emits compilable C, runs,
/// and agrees with eval. This is a TENSOR binding, so it reaches
/// `HostProgram::globals` and is in scope for this fix (the SCALAR-capture
/// analogue is the separate #378 lowering gap).
#[test]
fn issue_379_binding_named_main_compiles_and_matches_eval() {
    let source = "main = to_tensor([10.0, 20.0])\n\
def f(x: tensor[2, f32]) -> tensor[2, f32] = add(x, main)\n\
out = f(to_tensor([1.0, 2.0]))\n";

    let build = chelis_build_c(source, "mainname");
    let kernel_c = build.path().join("mainname.c");

    // Emit-shape invariant: the `main` binding must not collide with the
    // generated entry point `int main(void)`.
    let c_source = fs::read_to_string(&kernel_c).expect("read emitted C");
    assert!(
        c_source.contains("int main(void)"),
        "the generated entry point must still be `int main(void)`; \
         emitted C=\n{c_source}",
    );
    assert!(
        !c_source
            .lines()
            .any(|line| line.trim() == "static chelis_tensor* main;"),
        "the user binding `main` must be mangled, not collide with the entry \
         point; emitted C=\n{c_source}",
    );

    let stdout = compile_and_run_emitted(build.path(), &kernel_c);
    assert_eq!(
        binding_line(&stdout, "out"),
        "out = tensor(shape=[2], data=[11.0, 22.0])",
        "stdout={stdout:?}",
    );

    let eval_out = chelis_eval_ok(source, "mainname");
    assert_eq!(binding_line(&stdout, "out"), binding_line(&eval_out, "out"));
}

/// NEGATIVE: a normal (non-keyword, non-colliding) binding name must be
/// emitted byte-identical — the mangle pass must not disturb ordinary
/// identifiers. Pins that `c_ident` only rewrites the problematic cases.
#[test]
fn issue_379_ordinary_binding_name_is_not_mangled() {
    let source = "w = to_tensor([10.0, 20.0])\n\
out = add(w, to_tensor([1.0, 2.0]))\n";

    let build = chelis_build_c(source, "ordinary");
    let c_source = fs::read_to_string(build.path().join("ordinary.c")).expect("read emitted C");
    assert!(
        !c_source.contains("chelis_user__w"),
        "an ordinary binding name `w` must not be mangled; emitted C=\n{c_source}",
    );
    let stdout = compile_and_run_emitted(build.path(), &build.path().join("ordinary.c"));
    assert_eq!(
        binding_line(&stdout, "out"),
        "out = tensor(shape=[2], data=[11.0, 22.0])",
        "stdout={stdout:?}",
    );
}
