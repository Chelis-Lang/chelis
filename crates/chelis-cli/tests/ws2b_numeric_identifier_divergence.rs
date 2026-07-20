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
//!   between `div` and `mod`; the C backend emits an EXPLICIT, PORTABLE
//!   `chelis_int_div_guard` (`abort()` with the same diagnostic) rather than
//!   relying on a hardware fault — x86 raises SIGFPE on integer #DE but ARM64
//!   does not fault, so a SIGFPE-dependent trap silently returned a wrong
//!   value on macOS arm64. Float `div` keeps IEEE-754 (`1.0 / 0.0 == inf`).
//! * #381 — `scalar_to_tensor` of a top-level scalar binding that a def
//!   captures (and that the DAG lane materializes as a rank-0 tensor) must
//!   evaluate, matching the C backend / the DAG pass-through.
//! * #347 — the C backend must print `argmax_reduce`/`argmin_reduce` int64
//!   results as the integer indices, not the reinterpreted f32 bit pattern.
//! * #379 — top-level bindings spelled like C keywords or the emitted
//!   helper scheme must produce compilable C (identifier mangling).
//! * #365 — a `Bool` comparison-mask const (max-reduce / softmax backward)
//!   must fill through the dtype-correct `chelis_fill_bool_bits`, not
//!   `chelis_fill_f32_bits`, so a debug-runtime build does not abort on the
//!   dtype assertion. (The test links the debug `libchelis_runtime.a`, whose
//!   `debug_assert` is active.)
//!
//! #378 (route a captured top-level scalar binding into `HostProgram::globals`)
//! landed in chelis-ir, so the #381 program now compiles on the C backend and
//! its arm is a full eval-vs-C parity oracle (the captured f64 scalar is packed
//! into a `CHELIS_F64` rank-0 tensor for the tensor-helper input). All arms
//! here exercise eval-vs-C agreement.

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

    // Resolve the host toolchain the way `chelis build` and the other CLI test
    // harnesses (rank_poly_tier3.rs, parity.rs) do, so the link command carries
    // every platform-required flag. On macOS the transcendental kernels route
    // through Accelerate's vForce (`vvexpf`/`vvlogf`/...), so
    // `toolchain.link_flags` includes `-framework Accelerate`; a hand-rolled
    // `-lm -lpthread -ldl` link omits it and `ld` fails with
    // `Undefined symbols ... _vvexpf` on arm64 (the softmax-backward program
    // emits `vvexpf`). `needs_blas` is read from the emitted C so a BLAS kernel
    // links cblas too.
    let needs_blas = fs::read_to_string(kernel_c)
        .map(|t| t.contains("cblas_sgemm(") || t.contains("\"chelis_blas.h\""))
        .unwrap_or(false);
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: true,
            needs_blas,
        },
    );

    let bin = build_dir.join("ws2b_bin");
    let mut cmd = StdCommand::new(&toolchain.compiler);
    cmd.arg("-O0")
        .arg("-std=c11")
        .args(&toolchain.compile_flags)
        .arg("-I")
        .arg(build_dir.to_str().unwrap())
        .arg(kernel_c.to_str().unwrap())
        .arg(canonical.to_str().unwrap())
        .args(&toolchain.link_flags)
        .arg("-o")
        .arg(bin.to_str().unwrap());
    let compile = cmd.output().expect("invoke C compiler");
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
/// Integer decode of a printed tensor payload for cross-lane VALUE
/// comparison (chelis#732 P1): eval renders int64 tensor elements as
/// integers ([05-OBS-2]) while the compiled lane keeps its float-formatted
/// pre-contract form until Phase 2, so byte comparison of these lines is
/// per-lane and the cross-lane assertion decodes both.
fn tensor_ints(stdout: &str, name: &str) -> Vec<i64> {
    let payload = tensor_value(stdout, name);
    let start = payload.find("data=[").map(|i| i + "data=[".len());
    let (Some(start), Some(end)) = (start, payload.rfind(']')) else {
        panic!("no data payload in `{payload}`");
    };
    payload[start..end]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            let v: f64 = s.parse().unwrap_or_else(|e| panic!("`{s}`: {e}"));
            assert!(
                v.fract() == 0.0,
                "non-integral element `{s}` in `{payload}`"
            );
            // Above 2^53 the f64 parse collapses distinct int64s (the
            // PR #792 red-team F3 class); this helper's rows are small
            // indices/quotients, so refuse loudly rather than compare.
            assert!(
                v.abs() < 9007199254740992.0,
                "element `{s}` at or above 2^53 cannot be decoded through \
                 f64 in `{payload}`"
            );
            v as i64
        })
        .collect()
}

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

/// POSITIVE: integer `trunc_div` is truncating (round toward zero) and
/// agrees byte-for-byte between eval and the C backend. `7/2 == 3`,
/// `-7/2 == -3` per spec/05-risc-primitives.md §2.1. (chelis#178: integer
/// `div` is now a type error; `trunc_div` carries the old C-truncating
/// semantics.)
#[test]
fn issue_387_integer_trunc_div_truncates_eval_matches_backend() {
    let source = "def d(x: tensor[2, int64], y: tensor[2, int64]) -> tensor[2, int64] = trunc_div(x, y)\n\
out = d(cast(to_tensor([7, -7]), int64), cast(to_tensor([2, 2]), int64))\n";

    let build = chelis_build_c(source, "inttruncdiv");
    let stdout = compile_and_run_emitted(build.path(), &build.path().join("inttruncdiv.c"));
    assert_eq!(
        binding_line(&stdout, "out"),
        "out = tensor(shape=[2], data=[3.0, -3.0])",
        "integer trunc_div must truncate toward zero (7/2=3, -7/2=-3); stdout={stdout:?}",
    );

    let eval_out = chelis_eval_ok(source, "inttruncdiv");
    assert_eq!(
        tensor_ints(&stdout, "out"),
        tensor_ints(&eval_out, "out"),
        "eval and C backend must agree on the VALUES of integer trunc_div (values; byte parity returns at chelis#732 Phase 2)",
    );
}

/// POSITIVE: integer `floor_div` rounds the quotient toward −∞ and agrees
/// byte-for-byte between eval and the C backend. It differs from
/// `trunc_div` on the mixed-sign case: `7 floor_div 2 == 3` but
/// `-7 floor_div 2 == -4` (truncate gives `-3`). Per
/// spec/05-risc-primitives.md §2.1; matches Python `//`.
#[test]
fn chelis_178_integer_floor_div_rounds_toward_neg_inf_eval_matches_backend() {
    let source = "def d(x: tensor[2, int64], y: tensor[2, int64]) -> tensor[2, int64] = floor_div(x, y)\n\
out = d(cast(to_tensor([7, -7]), int64), cast(to_tensor([2, 2]), int64))\n";

    let build = chelis_build_c(source, "intfloordiv");
    let stdout = compile_and_run_emitted(build.path(), &build.path().join("intfloordiv.c"));
    assert_eq!(
        binding_line(&stdout, "out"),
        "out = tensor(shape=[2], data=[3.0, -4.0])",
        "integer floor_div must round toward -inf (7/2=3, -7/2=-4); stdout={stdout:?}",
    );

    let eval_out = chelis_eval_ok(source, "intfloordiv");
    assert_eq!(
        tensor_ints(&stdout, "out"),
        tensor_ints(&eval_out, "out"),
        "eval and C backend must agree on the VALUES of integer floor_div (values; byte parity returns at chelis#732 Phase 2)",
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

/// chelis#178 POSITIVE: scalar (host-lane) `floor_div` / `trunc_div` produce
/// the exact sign-rounding values per spec/05-risc-primitives.md §2.1. The
/// two ops differ on the mixed-sign exact-fraction cases — floor rounds
/// toward −∞, truncate toward zero — so every sign combination is asserted
/// with an exact expected value. This is the acceptance-oracle assertion that
/// the migration is *exercised*, not merely compiled.
#[test]
fn chelis_178_scalar_floor_trunc_div_exact_sign_rounding() {
    // (op, lhs, rhs, expected)
    let cases = &[
        // trunc_div: round toward zero
        ("trunc_div", 7, 2, "3"),
        ("trunc_div", -7, 2, "-3"),
        ("trunc_div", 7, -2, "-3"),
        ("trunc_div", -7, -2, "3"),
        ("trunc_div", -8, 2, "-4"),
        // floor_div: round toward -inf (differs on mixed sign)
        ("floor_div", 7, 2, "3"),
        ("floor_div", -7, 2, "-4"),
        ("floor_div", 7, -2, "-4"),
        ("floor_div", -7, -2, "3"),
        ("floor_div", -8, 2, "-4"),
    ];
    for (op, lhs, rhs, expected) in cases {
        let expr = format!("{op}(cast({lhs}, int64), cast({rhs}, int64))");
        let out = chelis_eval_expr(&expr);
        assert!(
            out.status.success(),
            "`{expr}` must evaluate; stderr={}",
            String::from_utf8_lossy(&out.stderr),
        );
        assert_eq!(
            String::from_utf8_lossy(&out.stdout).trim_end(),
            *expected,
            "`{expr}` must equal {expected} (spec/05 §2.1 sign-rounding)",
        );
    }
}

/// NEGATIVE: integer `trunc_div` by zero traps with the clean diagnostic
/// instead of returning a finite wrong value (pre-fix: `i64::MAX`). Exit 1,
/// not a silently-wrong scalar. (chelis#178: `div` on ints is a type error;
/// the zero-divisor trap moved to `trunc_div` / `floor_div`.)
#[test]
fn issue_387_integer_trunc_div_by_zero_traps_in_eval() {
    let out = chelis_eval_expr("trunc_div(cast(7, int64), cast(0, int64))");
    assert!(
        !out.status.success(),
        "integer trunc_div by zero must trap, not return a value; stdout={}",
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
/// diagnostic as `trunc_div` (pre-fix: a raw Rust remainder panic). The
/// integer division/remainder family (`trunc_div`, `floor_div`, `mod`) must
/// agree.
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
        "integer mod by zero must emit the SAME canonical diagnostic as trunc_div \
         (integer division/remainder consistency); stderr={stderr:?}",
    );
    assert!(
        !stderr.contains("attempt to calculate the remainder"),
        "the raw Rust remainder panic must be replaced by the clean diagnostic; \
         stderr={stderr:?}",
    );
}

/// NEGATIVE parity: the C backend must trap integer division by zero with
/// an EXPLICIT, PORTABLE guard (`chelis_int_div_guard` -> `abort()` with the
/// clean diagnostic), not by relying on a hardware fault. The divisor is
/// computed at RUNTIME (`sub(y, z)`), so the compiler cannot constant-fold it
/// to a literal `0` and elide the division. This is the regression that
/// macOS arm64 surfaced: ARM64 does not fault on integer div-by-zero, so a
/// SIGFPE-dependent trap silently returned a wrong value there.
#[test]
fn issue_387_integer_div_by_zero_traps_in_backend() {
    // y - z = [2, 0]: the second divisor is zero, computed at runtime.
    // chelis#178: integer division is `trunc_div`; the zero-divisor guard
    // applies to it identically.
    let source = "def d(x: tensor[2, int64], y: tensor[2, int64], z: tensor[2, int64]) -> tensor[2, int64] = trunc_div(x, sub(y, z))\n\
out = d(cast(to_tensor([7, 8]), int64), cast(to_tensor([3, 5]), int64), cast(to_tensor([1, 5]), int64))\n";

    let build = chelis_build_c(source, "intdivtrap");
    let kernel_c = build.path().join("intdivtrap.c");

    // Emit-shape: the integer divisor must be wrapped in the portable guard.
    let c_source = fs::read_to_string(&kernel_c).expect("read emitted C");
    assert!(
        c_source.contains("chelis_int_div_guard("),
        "integer trunc_div must emit the portable zero-divisor guard (#387); \
         emitted C=\n{c_source}",
    );

    let (compile, bin) = compile_emitted(build.path(), &kernel_c);
    assert!(
        compile.status.success(),
        "the program must compile; the trap is a runtime abort, not a compile \
         error; compiler stderr=\n{}",
        String::from_utf8_lossy(&compile.stderr),
    );
    let run = StdCommand::new(&bin).output().expect("run emitted binary");
    assert!(
        !run.status.success(),
        "C backend integer div by zero must trap (portable abort), not print \
         a finite value; stdout={}",
        String::from_utf8_lossy(&run.stdout),
    );
    // The trap emits the SAME clean diagnostic the evaluator does (stronger
    // than the old SIGFPE silent exit).
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        stderr.contains(INT_DIV_ZERO_DIAGNOSTIC),
        "C backend trap must emit the canonical diagnostic on stderr (#387); \
         stderr={stderr:?}",
    );
    // The result tensor must never have been printed.
    assert!(
        !String::from_utf8_lossy(&run.stdout).contains("out ="),
        "no `out = ...` line may be printed when the program traps; stdout={}",
        String::from_utf8_lossy(&run.stdout),
    );
}

/// NEGATIVE parity: a SCALAR (host-lane) integer division by zero traps via
/// the same portable guard. The divisor is runtime-computed (`sub(b, b)`).
/// Covers the host-emit `trunc_div` path (distinct from the DAG tensor path
/// above). chelis#178: scalar integer division is `trunc_div`.
#[test]
fn issue_387_scalar_integer_div_by_zero_traps_in_backend() {
    let source = "def d(a: int64, b: int64) -> int64 = trunc_div(a, sub(b, b))\n\
out = d(cast(7, int64), cast(5, int64))\n";

    let build = chelis_build_c(source, "scalardivtrap");
    let kernel_c = build.path().join("scalardivtrap.c");
    let c_source = fs::read_to_string(&kernel_c).expect("read emitted C");
    assert!(
        c_source.contains("chelis_int_div_guard("),
        "scalar integer trunc_div must emit the portable guard (#387); emitted C=\n{c_source}",
    );

    let (compile, bin) = compile_emitted(build.path(), &kernel_c);
    assert!(
        compile.status.success(),
        "must compile: {}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let run = StdCommand::new(&bin).output().expect("run emitted binary");
    assert!(
        !run.status.success(),
        "scalar integer div by zero must trap; stdout={}",
        String::from_utf8_lossy(&run.stdout),
    );
    assert!(
        String::from_utf8_lossy(&run.stderr).contains(INT_DIV_ZERO_DIAGNOSTIC),
        "scalar trap must emit the canonical diagnostic; stderr={:?}",
        String::from_utf8_lossy(&run.stderr),
    );
}

/// NEGATIVE parity: a SCALAR (host-lane) integer remainder by zero traps via
/// the same portable guard and diagnostic — `div` and `mod` stay consistent
/// on the backend as well as in eval.
#[test]
fn issue_387_scalar_integer_mod_by_zero_traps_in_backend() {
    let source = "def d(a: int64, b: int64) -> int64 = mod(a, sub(b, b))\n\
out = d(cast(7, int64), cast(5, int64))\n";

    let build = chelis_build_c(source, "scalarmodtrap");
    let kernel_c = build.path().join("scalarmodtrap.c");
    let (compile, bin) = compile_emitted(build.path(), &kernel_c);
    assert!(
        compile.status.success(),
        "must compile: {}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let run = StdCommand::new(&bin).output().expect("run emitted binary");
    assert!(
        !run.status.success(),
        "scalar integer mod by zero must trap; stdout={}",
        String::from_utf8_lossy(&run.stdout),
    );
    assert!(
        String::from_utf8_lossy(&run.stderr).contains(INT_DIV_ZERO_DIAGNOSTIC),
        "scalar mod trap must emit the canonical diagnostic; stderr={:?}",
        String::from_utf8_lossy(&run.stderr),
    );
}

/// POSITIVE parity: a SCALAR (host-lane) FLOAT division by zero must NOT trap
/// — it stays IEEE-754 (`1.0 / 0.0 == inf`). Guards the integer trap from
/// leaking into the float lane on the backend side.
#[test]
fn issue_387_scalar_float_div_by_zero_is_ieee_in_backend() {
    let source = "def d(a: f32, b: f32) -> f32 = div(a, sub(b, b))\n\
out = d(1.0, 5.0)\n";

    let build = chelis_build_c(source, "scalarfdiv");
    let kernel_c = build.path().join("scalarfdiv.c");
    let c_source = fs::read_to_string(&kernel_c).expect("read emitted C");
    assert!(
        !c_source.contains("chelis_int_div_guard("),
        "float div must NOT emit the integer trap guard (#387); emitted C=\n{c_source}",
    );
    let stdout = compile_and_run_emitted(build.path(), &kernel_c);
    assert_eq!(
        binding_line(&stdout, "out"),
        "out = inf",
        "scalar float 1.0/0.0 must be +inf, not a trap; stdout={stdout:?}",
    );
}

// -----------------------------------------------------------------------------
// #381 — scalar_to_tensor of a captured top-level scalar (eval lane)
// -----------------------------------------------------------------------------

/// POSITIVE + eval-vs-C parity: a top-level f64 scalar binding captured by a
/// def and fed to `scalar_to_tensor` evaluates correctly AND agrees with the
/// C backend. The DAG lane materializes the captured scalar as a rank-0
/// tensor; the host runtime's `scalar_to_tensor` accepts that rank-0 tensor
/// as the identity rather than erroring "expects scalar input".
///
/// The C-backend arm is now live: #378 (chelis-ir, merged) routes the
/// captured scalar binding into `HostProgram::globals` so the C emitter
/// declares it, and #381 (this PR) packs that captured f64 scalar into a
/// `CHELIS_F64` rank-0 tensor for the tensor-helper input (the pre-fix
/// catch-all packed it as `CHELIS_F32`, storing only the low 4 bytes, so the
/// f64 kernel read garbage and silently dropped the value -- the eval-vs-C
/// divergence this arm exists to lock). `out` is a rank-1 f64 tensor, so
/// both lanes render it identically; the comparison uses the value to stay
/// robust to the pre-existing rank-0-scalar bare-vs-`tensor(shape=[])`
/// display divergence noted by WS-2A.
#[test]
fn issue_381_scalar_to_tensor_on_captured_scalar_evals_and_matches_backend() {
    let source = "c = cast(1.1, f64)\n\
def make(n: tensor[2, f64]) -> tensor[2, f64] = \
add(n, expand(scalar_to_tensor(c), cast(0, int32), cast(2, int32)))\n\
out = make(to_tensor([cast(1.0, f64), cast(2.0, f64)]))\n";

    // Eval lane (the reference): 1.1 broadcast-added to [1.0, 2.0].
    let eval_out = chelis_eval_ok(source, "s2t_capture");
    assert_eq!(
        binding_line(&eval_out, "out"),
        "out = tensor(shape=[2], data=[2.1, 3.1])",
        "eval: 1.1 + [1.0, 2.0] = [2.1, 3.1]; stdout={eval_out:?}",
    );

    // C-backend lane: must compile (needs #378's captured-binding global),
    // run, and produce the same value. Pre-#381 the f64 scalar was packed as
    // f32 and the C output was [1.0, 2.0] (the captured 1.1 dropped to ~0).
    let build = chelis_build_c(source, "s2t_capture");
    let kernel_c = build.path().join("s2t_capture.c");
    // Emit-shape: the captured f64 scalar packs into a CHELIS_F64 rank-0
    // tensor through a double*, not CHELIS_F32.
    let c_source = fs::read_to_string(&kernel_c).expect("read emitted C");
    assert!(
        c_source.contains("chelis_alloc(0, NULL, CHELIS_F64)") && c_source.contains("((double*)"),
        "captured f64 scalar must pack into a CHELIS_F64 rank-0 tensor (#381); \
         emitted C=\n{c_source}",
    );
    let stdout = compile_and_run_emitted(build.path(), &kernel_c);
    assert_eq!(
        tensor_value(&stdout, "out"),
        tensor_value(&eval_out, "out"),
        "eval and C backend must agree on the captured-f64-scalar program \
         (#381 x #378); C stdout={stdout:?}",
    );
    // Pin the exact value too, so a both-lanes-wrong regression can't pass.
    assert_eq!(
        binding_line(&stdout, "out"),
        "out = tensor(shape=[2], data=[2.1, 3.1])",
        "C: captured 1.1 must survive as f64; stdout={stdout:?}",
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
        tensor_ints(&stdout, "out"),
        tensor_ints(&eval_out, "out"),
        "eval and C backend must agree on the VALUES of argmax int64 indices (#347) (values; byte parity returns at chelis#732 Phase 2)",
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
        tensor_ints(&stdout, "out"),
        tensor_ints(&eval_out, "out"),
        "eval and C backend must agree on the VALUES of argmin int64 indices (#347) (values; byte parity returns at chelis#732 Phase 2)",
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

// -----------------------------------------------------------------------------
// #365 — Bool comparison-mask const fills through the dtype-correct helper
// -----------------------------------------------------------------------------

/// POSITIVE + emit-shape: a `max_reduce` backward materializes a `Bool`
/// comparison mask. The mask const must fill through `chelis_fill_bool_bits`
/// (dtype-correct for CHELIS_BOOL), NOT `chelis_fill_f32_bits` (which asserts
/// CHELIS_F32). The build links the debug `libchelis_runtime.a`, so the
/// debug-build dtype assertion is active: a regression aborts the run.
/// Pre-fix the Bool const used `chelis_fill_f32_bits` and aborted here.
#[test]
fn issue_365_max_reduce_backward_bool_mask_fill_is_dtype_correct() {
    let source = "def f(x: tensor[3, f32]) -> f32 = tensor_to_scalar(max_reduce(x, 0))\n\
def df(x: tensor[3, f32]) -> tensor[3, f32] = grad(f)(x)\n\
out = df(to_tensor([1.0, 5.0, 3.0]))\n";

    let build = chelis_build_c(source, "maxback");
    let kernel_c = build.path().join("maxback.c");

    // Emit-shape: the Bool mask const must use the dtype-correct fill.
    let c_source = fs::read_to_string(&kernel_c).expect("read emitted C");
    assert!(
        c_source.contains("chelis_fill_bool_bits("),
        "a Bool mask const must fill through chelis_fill_bool_bits (#365); \
         emitted C=\n{c_source}",
    );

    // Compile + run against the (debug) runtime; a dtype-assert abort would
    // make the binary exit non-zero. d/dx max([1,5,3]) routes the gradient
    // to the max element (index 1): [0, 1, 0].
    let stdout = compile_and_run_emitted(build.path(), &kernel_c);
    assert_eq!(
        binding_line(&stdout, "out"),
        "out = tensor(shape=[3], data=[0.0, 1.0, 0.0])",
        "max-reduce backward gradient flows to the max element; stdout={stdout:?}",
    );

    let eval_out = chelis_eval_ok(source, "maxback");
    assert_eq!(
        tensor_value(&stdout, "out"),
        tensor_value(&eval_out, "out"),
        "eval and C backend must agree on the max-reduce backward (#365)",
    );
}

/// POSITIVE: a softmax backward composition (reachable Bool mask) also
/// builds and runs to completion under the debug runtime without a dtype
/// abort. d/dx sum(softmax(x)) is mathematically zero (softmax sums to 1);
/// assert the run completes and the values are near zero rather than pinning
/// an exact f32-noise vector.
#[test]
fn issue_365_softmax_backward_runs_under_debug_runtime() {
    let source = "def f(x: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(softmax(x, 0), 0))\n\
def df(x: tensor[3, f32]) -> tensor[3, f32] = grad(f)(x)\n\
out = df(to_tensor([1.0, 2.0, 3.0]))\n";

    let build = chelis_build_c(source, "softmaxback");
    let stdout = compile_and_run_emitted(build.path(), &build.path().join("softmaxback.c"));
    // The run completing (compile_and_run_emitted asserts a zero exit) is the
    // #365 acceptance: no dtype-assert abort. sum(softmax) is constant, so
    // every gradient element is ~0 (f32 rounding noise around zero).
    let line = binding_line(&stdout, "out");
    let data = line
        .split_once("data=[")
        .and_then(|(_, rest)| rest.strip_suffix("])"))
        .unwrap_or_else(|| panic!("could not parse data from {line:?}"));
    for elem in data.split(", ") {
        let v: f64 = elem
            .parse()
            .unwrap_or_else(|_| panic!("bad element {elem:?}"));
        assert!(
            v.abs() < 1e-4,
            "d/dx sum(softmax(x)) must be ~0; got {v} in {line:?}",
        );
    }
}

// -----------------------------------------------------------------------------
// #476 — inline sparse gather/scatter read int32 indices through the
// dtype-correct pointer, not `(int)t->data[i]`. Same #347 class as the
// argmax/argmin index prints above: int tensors bit-pack their values into
// the float-typed `->data`, so `(int)t->data[i]` on a CHELIS_I32 index
// `(int)`-truncates the FLOAT reinterpretation of the int32 bits (index `2`
// → `(int)2.8e-45f` → `0`), silently gathering the WRONG row. The user
// surface defaults integer literals to int32 (`to_tensor([2, 0, 1])` is a
// CHELIS_I32 tensor), so this fires on ordinary index code; the pre-fix
// corpus never reproduced it because every gather fixture cast indices to
// int64 (`cast(_, int64)`), which took the always-correct CHELIS_I64 branch.
//
// The acceptance oracle is BIT-IDENTITY eval-vs-C on the integer/index path
// (no float summation here — indices are exact), PLUS a negative assertion
// that the pre-fix reinterpreted-float corruption (every index → row 0) is
// rejected. An int64-index control proves the i64 branch is untouched.
// -----------------------------------------------------------------------------

/// POSITIVE + parity: `gather` with the DEFAULT int32 index dtype agrees
/// byte-for-byte between eval and the C backend. The gather is the program
/// root, so it lowers to the inline WireDag tensor-lane emit
/// (`emit_sparse_gather`), which is the buggy path — NOT the runtime helper
/// `chelis_tensor_gather` (that path always read indices at the correct
/// width via `read_index_slot`).
#[test]
fn issue_476_gather_int32_indices_backend_matches_eval() {
    // indices [2, 0, 1] over a 3x2 table → rows [30,31],[10,11],[20,21].
    let source = "table = to_tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])\n\
embed = gather(table, to_tensor([2, 0, 1]), 0)\n";

    let build = chelis_build_c(source, "gatheri32");
    let stdout = compile_and_run_emitted(build.path(), &build.path().join("gatheri32.c"));
    assert_eq!(
        binding_line(&stdout, "embed"),
        "embed = tensor(shape=[3, 2], data=[30.0, 31.0, 10.0, 11.0, 20.0, 21.0])",
        "int32-index gather must select rows 2,0,1; stdout={stdout:?}",
    );
    // NEGATIVE: the pre-fix corruption read every int32 index as 0 (the
    // float reinterpretation of small ints rounds toward zero), so every
    // output row was row 0 (`[10,11]`). That signature must never appear.
    assert_ne!(
        binding_line(&stdout, "embed"),
        "embed = tensor(shape=[3, 2], data=[10.0, 11.0, 10.0, 11.0, 10.0, 11.0])",
        "pre-fix #476 read every int32 index as row 0; that corruption must \
         not recur; stdout={stdout:?}",
    );

    let eval_out = chelis_eval_ok(source, "gatheri32");
    assert_eq!(
        binding_line(&stdout, "embed"),
        binding_line(&eval_out, "embed"),
        "eval and C backend must agree byte-for-byte on int32-index gather (#476)",
    );
}

/// POSITIVE / control: the same gather with indices cast to int64 still
/// agrees. The int64 branch was always correct; this proves the fix did not
/// regress it.
#[test]
fn issue_476_gather_int64_indices_unchanged() {
    let source = "table = to_tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])\n\
embed = gather(table, cast(to_tensor([2, 0, 1]), int64), 0)\n";

    let build = chelis_build_c(source, "gatheri64");
    let stdout = compile_and_run_emitted(build.path(), &build.path().join("gatheri64.c"));
    assert_eq!(
        binding_line(&stdout, "embed"),
        "embed = tensor(shape=[3, 2], data=[30.0, 31.0, 10.0, 11.0, 20.0, 21.0])",
        "int64-index gather must select rows 2,0,1; stdout={stdout:?}",
    );

    let eval_out = chelis_eval_ok(source, "gatheri64");
    assert_eq!(
        binding_line(&stdout, "embed"),
        binding_line(&eval_out, "embed"),
        "eval and C backend must agree on int64-index gather (control)",
    );
}

/// POSITIVE + parity: `scatter(..., "replace")` over an int32-index path
/// agrees byte-for-byte. Routed through the runtime helper at the program
/// root, but the int32 read there is the same #347 class; the
/// `read_index_slot` width-correct path keeps it honest. This guards the
/// user-facing scatter surface in addition to the inline emit.
#[test]
fn issue_476_scatter_replace_int32_indices_backend_matches_eval() {
    // base 3x2 zeros; updates rows written at indices [2,0,1].
    let source = "base = to_tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])\n\
out = scatter(base, to_tensor([2, 0, 1]), to_tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]), 0, \"replace\")\n";

    let build = chelis_build_c(source, "scatteri32");
    let stdout = compile_and_run_emitted(build.path(), &build.path().join("scatteri32.c"));
    assert_eq!(
        binding_line(&stdout, "out"),
        "out = tensor(shape=[3, 2], data=[2.0, 2.0, 3.0, 3.0, 1.0, 1.0])",
        "scatter_replace at int32 indices [2,0,1] places update row 0→pos2, \
         1→pos0, 2→pos1; stdout={stdout:?}",
    );

    let eval_out = chelis_eval_ok(source, "scatteri32");
    assert_eq!(
        binding_line(&stdout, "out"),
        binding_line(&eval_out, "out"),
        "eval and C backend must agree byte-for-byte on int32 scatter_replace (#476)",
    );
}

/// POSITIVE + parity: the inline `emit_sparse_scatter_add` path — reached as
/// the gather ADJOINT (grad of gather scatter-adds the upstream grad back to
/// the gathered rows) — agrees byte-for-byte at int32 indices. This exercises
/// BOTH inline sparse emits in one program: `emit_sparse_gather` (forward)
/// and `emit_sparse_scatter_add` (backward). Index 2 appears twice → its row
/// accumulates grad 2; index 0 once → grad 1; index 1 never → grad 0.
#[test]
fn issue_476_gather_grad_scatter_add_int32_backend_matches_eval() {
    let source = "def f(table: tensor[3, 2, f32]) -> f32 = tensor_to_scalar(sum(sum(gather(table, to_tensor([2, 0, 2]), 0), 0), 0))\n\
def df(table: tensor[3, 2, f32]) -> tensor[3, 2, f32] = grad(f)(table)\n\
out = df(to_tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]))\n";

    let build = chelis_build_c(source, "gathergradi32");
    let stdout = compile_and_run_emitted(build.path(), &build.path().join("gathergradi32.c"));
    assert_eq!(
        binding_line(&stdout, "out"),
        "out = tensor(shape=[3, 2], data=[1.0, 1.0, 0.0, 0.0, 2.0, 2.0])",
        "grad of gather([2,0,2]) accumulates 1 at row0, 0 at row1, 2 at row2; \
         stdout={stdout:?}",
    );
    // NEGATIVE: the pre-fix corruption read every index as 0, so the ENTIRE
    // gradient (3 gather positions) would pile onto row 0 (`[3,3]`) and rows
    // 1,2 would be zero. That signature must never appear.
    assert_ne!(
        binding_line(&stdout, "out"),
        "out = tensor(shape=[3, 2], data=[3.0, 3.0, 0.0, 0.0, 0.0, 0.0])",
        "pre-fix #476 piled the whole gather grad onto row 0; that corruption \
         must not recur; stdout={stdout:?}",
    );

    // `chelis eval` prints a single-root program's value WITHOUT a label;
    // the C backend labels it `out = ...`. `tensor_value` normalizes that
    // asymmetry so the comparison is value-vs-value.
    let eval_out = chelis_eval_ok(source, "gathergradi32");
    assert_eq!(
        tensor_value(&stdout, "out"),
        tensor_value(&eval_out, "out"),
        "eval and C backend must agree byte-for-byte on int32 gather-grad \
         (scatter_add adjoint) (#476)",
    );
}

// -----------------------------------------------------------------------------
// #172 — max/min reductions PROPAGATE NaN (torch parity), consistently across
// eval and the C backend (contiguous SIMD + strided). Pre-fix, the SIMD
// `chelis_max_f32` dropped NaN position-dependently and the `value > best`
// reductions silently dropped it everywhere, so `max_reduce([NaN,..]) = a
// finite value` on both lanes — diverging from `torch.max` (always NaN).
// -----------------------------------------------------------------------------

/// POSITIVE + parity: `max_reduce` of a slice containing a runtime NaN
/// (`0.0 / 0.0`) yields NaN on the C backend, matching eval and torch. The
/// divisor is runtime-derived so the compiler cannot constant-fold the NaN
/// away. Uses `is_nan` semantics, not bit-identity (NaN has many encodings).
#[test]
fn issue_172_max_reduce_propagates_nan_backend_matches_eval() {
    // Row 0: a/b = [0/0, 1/1] = [NaN, 1] -> max NaN. Row 1: [2/1, 3/1] =
    // [2, 3] -> max 3. The output is a `tensor[2]` (both lanes label it),
    // and the non-NaN row proves the fix doesn't blanket-NaN the result.
    let source = "def f(a: tensor[2, 2, f32], b: tensor[2, 2, f32]) -> tensor[2, f32] = max_reduce(div(a, b), 1)\n\
out = f(to_tensor([[0.0, 1.0], [2.0, 3.0]]), to_tensor([[0.0, 1.0], [1.0, 1.0]]))\n";

    let build = chelis_build_c(source, "maxnan");
    let stdout = compile_and_run_emitted(build.path(), &build.path().join("maxnan.c"));
    let val = binding_line(&stdout, "out");
    assert!(
        val.to_ascii_lowercase().contains("nan"),
        "max_reduce of a NaN row must be NaN on the C backend (#172 torch parity), \
         not a finite value; got {val:?}",
    );
    // The non-NaN row must still reduce to 3 — the fix propagates NaN only
    // for the slice that actually contains one.
    assert!(
        val.contains("3.0"),
        "the NaN-free row must still reduce to 3 (#172 must not blanket-NaN); \
         got {val:?}",
    );

    let eval_out = chelis_eval_ok(source, "maxnan");
    assert!(
        tensor_value(&eval_out, "out")
            .to_ascii_lowercase()
            .contains("nan"),
        "eval must also propagate NaN through max_reduce (#172); got {eval_out:?}",
    );
}

/// POSITIVE + parity: `min_reduce` propagates NaN identically.
#[test]
fn issue_172_min_reduce_propagates_nan_backend_matches_eval() {
    let source = "def f(a: tensor[2, 2, f32], b: tensor[2, 2, f32]) -> tensor[2, f32] = min_reduce(div(a, b), 1)\n\
out = f(to_tensor([[0.0, 1.0], [2.0, 3.0]]), to_tensor([[0.0, 1.0], [1.0, 1.0]]))\n";

    let build = chelis_build_c(source, "minnan");
    let stdout = compile_and_run_emitted(build.path(), &build.path().join("minnan.c"));
    let val = binding_line(&stdout, "out");
    assert!(
        val.to_ascii_lowercase().contains("nan"),
        "min_reduce of a NaN row must be NaN on the C backend (#172); got {val:?}",
    );
    assert!(
        val.contains("2.0"),
        "the NaN-free row min must still be 2 (#172 must not blanket-NaN); got {val:?}",
    );

    let eval_out = chelis_eval_ok(source, "minnan");
    assert!(
        tensor_value(&eval_out, "out")
            .to_ascii_lowercase()
            .contains("nan"),
        "eval must propagate NaN through min_reduce (#172)",
    );
}

/// POSITIVE control: a NaN-FREE max_reduce still agrees byte-for-byte; the
/// NaN-propagation fix must not perturb ordinary reductions.
#[test]
fn issue_172_max_reduce_no_nan_unchanged_backend_matches_eval() {
    let source = "def f(x: tensor[2, 3, f32]) -> tensor[2, f32] = max_reduce(x, 1)\n\
out = f(to_tensor([[1.0, 3.0, 2.0], [6.0, 4.0, 5.0]]))\n";

    let build = chelis_build_c(source, "maxok");
    let stdout = compile_and_run_emitted(build.path(), &build.path().join("maxok.c"));
    assert_eq!(
        binding_line(&stdout, "out"),
        "out = tensor(shape=[2], data=[3.0, 6.0])",
        "NaN-free max_reduce must be unchanged; stdout={stdout:?}",
    );
    let eval_out = chelis_eval_ok(source, "maxok");
    assert_eq!(
        tensor_value(&stdout, "out"),
        tensor_value(&eval_out, "out"),
        "eval and C backend must agree on NaN-free max_reduce (#172 control)",
    );
}

/// FORWARD-through-grad: a `max_reduce` whose forward value is NaN keeps that
/// NaN when the function is differentiated (the traced forward still
/// propagates). This guards the forward half of the grad path; the exact
/// NaN gradient ROUTING (torch sends grad to the NaN slot) is a separate
/// grad-NaN-semantics question tracked as a #172 follow-up.
#[test]
fn issue_172_max_reduce_grad_forward_propagates_nan() {
    // `df` differentiates `f` (a `max_reduce` over a NaN slice); `fout` is
    // the forward value of the SAME function. The grad def must lower/run
    // (no internal error), and the forward of a NaN slice must be NaN — the
    // traced forward propagates the NaN exactly like the standalone forward.
    // Two roots so eval labels both with a `name = ` prefix.
    let source = "def f(a: tensor[3, f32]) -> f32 = tensor_to_scalar(max_reduce(div(a, to_tensor([1.0, 0.0, 1.0])), 0))\n\
def df(a: tensor[3, f32]) -> tensor[3, f32] = grad(f)(a)\n\
gout = df(to_tensor([1.0, 0.0, 2.0]))\n\
fout = f(to_tensor([1.0, 0.0, 2.0]))\n";

    let eval_out = chelis_eval_ok(source, "maxgradfwd");
    assert!(
        binding_line(&eval_out, "fout")
            .to_ascii_lowercase()
            .contains("nan"),
        "the forward of a max_reduce over a NaN slice must be NaN (#172); \
         eval={eval_out:?}",
    );
    // The grad def must have produced a value (it ran without an internal
    // error); we don't pin the exact NaN gradient routing here (separate
    // grad-NaN-semantics follow-up).
    assert!(
        eval_out.contains("gout ="),
        "grad of a max_reduce over a NaN slice must still produce a gradient \
         tensor (no internal error); eval={eval_out:?}",
    );
}

/// REGRESSION-LOCK (#172 tanh parity): Chelis `tanh` (lowered as
/// `2*sigmoid(2x) - 1`) is bit-identical to `torch.tanh` on the probe set.
/// Pinning the exact f32 values guards against the lowering drifting away
/// from torch parity. Reference values from torch 2.x CPU:
/// `tanh([0.5, -0.5, 1.0, -2.0]) = [0.46211717, -0.46211717, 0.7615942,
/// -0.96402758]`. The values render through the shared f32 printer.
#[test]
fn issue_172_tanh_matches_torch_reference() {
    let source = "def f(x: tensor[4, f32]) -> tensor[4, f32] = tanh(x)\n\
out = f(to_tensor([0.5, -0.5, 1.0, -2.0]))\n";

    let eval_out = chelis_eval_ok(source, "tanhparity");
    let val = tensor_value(&eval_out, "out");
    let data = val
        .split_once("data=[")
        .and_then(|(_, rest)| rest.strip_suffix("])"))
        .unwrap_or_else(|| panic!("could not parse tanh data from {val:?}"));
    let got: Vec<f64> = data
        .split(", ")
        .map(|e| e.parse().unwrap_or_else(|_| panic!("bad element {e:?}")))
        .collect();
    let torch_ref = [
        0.462_117_165_327_072_14,
        -0.462_117_165_327_072_14,
        0.761_594_176_292_419_4,
        -0.964_027_583_599_090_6,
    ];
    assert_eq!(got.len(), 4, "tanh must return 4 elements; got {got:?}");
    for (i, (&g, &r)) in got.iter().zip(torch_ref.iter()).enumerate() {
        assert!(
            (g - r).abs() < 1e-6,
            "tanh[{i}] = {g} must match torch reference {r} (#172 parity)",
        );
    }
}
