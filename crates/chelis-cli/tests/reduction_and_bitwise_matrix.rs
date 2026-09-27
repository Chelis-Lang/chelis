//! Sweep 3 of the numeric audit brief: bitwise ops beyond i64, the
//! reduction family across dtypes, and the Bool dtype - eval vs compiled C.
//!
//! Findings (each confirmed by execution):
//!
//! | issue | one line |
//! |---|---|
//! | chelis#682 | bitand/shifts now agree across eval/C at every integer width, including boundary counts |
//! | chelis#692 | max/min/prod/argmax/argmin reduce on an i64 tensor PANIC the compiler (emit.rs:4238/4406/4777) |
//! | chelis#723 | the C print helper renders i64 elements through double, hiding an EXACT sum of 2^53 + 1 |
//! | chelis#724 | mean of an i64 tensor: eval 187.5 (fractional in an i64 tensor), C 187.0 |
//! | chelis#684 | eval's i64 tensor sum is genuinely lossy at 2^53 (Vec<f64> storage) |
//!
//! And the clean surfaces, locked: the Bool dtype is correct end-to-end in
//! both lanes (the lib.rs:144-152 "f32-encoded" comment notwithstanding -
//! probed, not believed), all seven f32 reductions agree across lanes, and
//! the C lane's i64 `sum` is EXACT at 2^53 + 1 (provable only through
//! `to_list`, which is what #723 is about).
//!
//! Shift counts are explicitly width-bounded: bits shifted beyond the declared
//! integer width are discarded in both lanes.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

fn c_toolchain_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn eval_first_line(program: &str) -> Result<String, String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("p.ch");
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string())
}

/// Build to C; distinguish a compiler PANIC from a clean rejection from a
/// runnable binary. Returns (built_ok, build_stderr, Option<run_stdout>).
fn c_build_outcome(program: &str, name: &str) -> (bool, String, Option<String>) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, program);
    let built = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("chelis build should run");
    let stderr = String::from_utf8_lossy(&built.stderr).into_owned();
    if !built.status.success() {
        return (false, stderr, None);
    }
    let status = common::link_generated(&out_dir, &format!("{name}.c"), name);
    assert!(status.success(), "link failed for {name}");
    let run = std::process::Command::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    (
        true,
        stderr,
        Some(
            String::from_utf8_lossy(&run.stdout)
                .lines()
                .next()
                .unwrap_or("")
                .trim()
                .to_string(),
        ),
    )
}

fn c_first_line(program: &str, name: &str) -> String {
    let (ok, stderr, stdout) = c_build_outcome(program, name);
    assert!(ok, "{name}: build failed: {stderr}");
    stdout.expect("ran")
}

/// Compile and run generated C with undefined-behavior sanitization enabled.
/// The returned output lets callers assert either successful value parity or
/// the language-defined negative-count trap without hiding a sanitizer report.
fn c_ubsan_run(program: &str, name: &str) -> std::process::Output {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, program);
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: false,
            needs_blas: false,
        },
    );
    // Prefer clang's compiler-rt UBSan when available. Some GCC
    // installations provide the compiler but omit the separately packaged
    // libubsan shared object; that is an environment gap, not a reason to
    // weaken this executable oracle.
    let clang_available = std::process::Command::new("clang")
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success());
    let sanitizer_compiler = if clang_available {
        "clang"
    } else {
        toolchain.compiler.as_str()
    };
    let mut compiler = std::process::Command::new(sanitizer_compiler);
    compiler
        .current_dir(&out_dir)
        .args([
            "-O1",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=undefined",
        ])
        .args(&toolchain.compile_flags)
        .arg(format!("{name}.c"))
        .arg("libchelis_runtime.a")
        .args(&toolchain.link_flags)
        .args(["-fsanitize=undefined", "-o", name]);
    let linked = compiler.output().expect("invoke UBSan C compiler");
    assert!(
        linked.status.success(),
        "UBSan link failed for {name}: {}",
        String::from_utf8_lossy(&linked.stderr)
    );
    std::process::Command::new(out_dir.join(name))
        .env("UBSAN_OPTIONS", "halt_on_error=1:print_stacktrace=1")
        .output()
        .expect("run UBSan binary")
}

fn scalar_program(expr: &str, ret_ty: &str) -> String {
    format!("module M.Main\ndef run() -> {ret_ty} = {expr}\nout = print(run())\n")
}

// ===========================================================================
// chelis#682 at every width - eval/C parity locks
// ===========================================================================

/// Value-parity row promoted when chelis#682 gained closed C-expression arms.
#[test]
fn bitwise_ops_agree_across_lanes_at_every_width() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    for ty in ["i8", "i16", "i32", "i64"] {
        for (label, expr, expected) in [
            (
                "bitand",
                format!("bitand(cast(12, {ty}), cast(10, {ty}))"),
                "8",
            ),
            (
                "bitor",
                format!("bitor(cast(12, {ty}), cast(10, {ty}))"),
                "14",
            ),
            (
                "bitxor",
                format!("bitxor(cast(12, {ty}), cast(10, {ty}))"),
                "6",
            ),
            ("shl", format!("shl(cast(1, {ty}), cast(4, {ty}))"), "16"),
            ("shr", format!("shr(cast(-16, {ty}), cast(2, {ty}))"), "-4"),
        ] {
            let program = scalar_program(&expr, ty);
            assert_eq!(eval_first_line(&program).expect("eval"), expected);
            let c_got = c_first_line(&program, &format!("bw_{ty}_{label}"));
            // Only the C lane is domain-wired here by design: it is the
            // #718 width-escape suspect, and a domain diagnostic before
            // the generic LANE DIVERGENCE assert names the failure class.
            // The eval side's exact-string assert_eq above subsumes its
            // own domain check (in-domain expected value).
            common::assert_elements_in_domain(ty, &c_got, &expr);
            assert_eq!(
                c_got, expected,
                "LANE DIVERGENCE for `{expr}`: C printed {c_got}"
            );
        }
    }
}

/// [04-NUM-13]: shifts use declared-width two's-complement semantics.
#[test]
fn eval_shift_width_semantics_are_locked() {
    assert_eq!(
        eval_first_line(&scalar_program("shl(cast(1, i8), cast(7, i8))", "i8")).expect("eval"),
        "-128"
    );
    assert_eq!(
        eval_first_line(&scalar_program("shr(cast(-16, i64), cast(2, i64))", "i64")).expect("eval"),
        "-4",
        "arithmetic (sign-preserving) right shift"
    );
}

/// Both lanes apply the declared i8 width after the shift.
#[test]
fn shl_past_width_is_defined_as_fully_shifted_out() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let program = scalar_program("shl(cast(1, i8), cast(9, i8))", "i8");
    assert_eq!(eval_first_line(&program).expect("eval"), "0");
    assert_eq!(c_first_line(&program, "shl_past_width"), "0");
}

/// Red-team regression for chelis#682: signed C `<<` is undefined for a
/// negative lhs and for results entering the sign bit, while any count at
/// least the promoted width is undefined. Generated code must instead use the
/// unsigned width-aware helpers and remain clean under UBSan.
#[test]
fn shift_boundaries_agree_across_lanes_and_are_ubsan_clean() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain with UBSan");
    }
    for (ty, width, min) in [
        ("i8", 8, "-128"),
        ("i16", 16, "-32768"),
        ("i32", 32, "-2147483648"),
        ("i64", 64, "-9223372036854775808"),
    ] {
        for (label, expr, expected) in [
            (
                "sign_bit",
                format!("shl(cast(1, {ty}), cast({}, {ty}))", width - 1),
                min,
            ),
            (
                "negative_lhs",
                format!("shl(cast(-1, {ty}), cast(1, {ty}))"),
                "-2",
            ),
            (
                "count_at_width",
                format!("shl(cast(1, {ty}), cast({width}, {ty}))"),
                "0",
            ),
            (
                "negative_shr_at_width",
                format!("shr(cast(-1, {ty}), cast({width}, {ty}))"),
                "-1",
            ),
        ] {
            let program = scalar_program(&expr, ty);
            assert_eq!(
                eval_first_line(&program).expect("eval boundary shift"),
                expected
            );
            let run = c_ubsan_run(&program, &format!("shift_{ty}_{label}"));
            assert!(
                run.status.success(),
                "{ty}/{label}: UBSan binary failed: {}",
                String::from_utf8_lossy(&run.stderr)
            );
            assert!(
                !String::from_utf8_lossy(&run.stderr).contains("runtime error:"),
                "{ty}/{label}: UBSan reported undefined behavior: {}",
                String::from_utf8_lossy(&run.stderr)
            );
            assert_eq!(
                String::from_utf8_lossy(&run.stdout)
                    .lines()
                    .next()
                    .unwrap_or_default(),
                expected,
                "{ty}/{label}: eval/C shift divergence"
            );
        }
    }
}

#[test]
fn negative_shift_counts_trap_before_ubsan_at_every_width() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain with UBSan");
    }
    for ty in ["i8", "i16", "i32", "i64"] {
        for op in ["shl", "shr"] {
            let program = scalar_program(&format!("{op}(cast(1, {ty}), cast(-1, {ty}))"), ty);
            let eval_err = eval_first_line(&program).expect_err("negative eval shift must fail");
            assert!(
                eval_err.contains("shift amount must be non-negative, got -1"),
                "{ty}/{op}: wrong eval diagnostic: {eval_err}"
            );
            let run = c_ubsan_run(&program, &format!("shift_negative_count_{ty}_{op}"));
            assert!(
                !run.status.success(),
                "{ty}/{op}: negative compiled shift must trap"
            );
            let stderr = String::from_utf8_lossy(&run.stderr);
            assert!(
                stderr.contains("shift amount must be non-negative, got -1"),
                "{ty}/{op}: wrong C diagnostic: {stderr}"
            );
            assert!(
                !stderr.contains("runtime error:"),
                "{ty}/{op}: language trap must precede any undefined shift: {stderr}"
            );
        }
    }
}

/// Phase 3 checked-accumulator row: integer reduction overflow must use the
/// same declared-width trap in eval and generated C, and the language trap
/// must fire before UBSan can observe signed-overflow undefined behavior.
#[test]
fn integer_reduce_sum_overflow_traps_before_ubsan() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain with UBSan");
    }
    for (ty, max) in [("i32", "2147483647"), ("i64", "9223372036854775807")] {
        let program = format!(
            "module M.Main\ndef run(x: tensor[2, {ty}]) -> tensor[{ty}] = sum(x, cast(0, i32))\n\
             out = print(run(to_tensor([cast({max}, {ty}), cast(1, {ty})])))\n"
        );
        let expected = format!("numeric trap: overflow in sum at {ty}");
        let eval_err = eval_first_line(&program).expect_err("eval reduction overflow must trap");
        assert!(
            eval_err.contains(&expected),
            "{ty}: wrong eval reduction diagnostic: {eval_err}"
        );
        let run = c_ubsan_run(&program, &format!("sum_overflow_{ty}"));
        assert!(
            !run.status.success(),
            "{ty}: compiled reduction overflow must trap"
        );
        let stderr = String::from_utf8_lossy(&run.stderr);
        assert!(
            stderr.contains(&expected),
            "{ty}: wrong C reduction diagnostic: {stderr}"
        );
        assert!(
            !stderr.contains("runtime error:"),
            "{ty}: language trap must precede signed-overflow UB: {stderr}"
        );
    }
}

// ===========================================================================
// chelis#692 - i64 reduce ops panic the compiler
// ===========================================================================

/// Before chelis#730 Phase 1, `chelis build` panicked instead of either
/// compiling correctly or rejecting with a diagnostic. This row asserts the
/// non-panic contract: any outcome except a Rust backtrace. The reduce-family
/// panics are now section C2 diagnostics through the emitter channel;
/// the build fails cleanly and this test accepts the rejection arm.
#[test]
fn int64_max_reduce_does_not_panic_the_compiler() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let program = "module M.Main\n\
         def f(x: tensor[4, i64]) -> tensor[i64] = max_reduce(x, 0)\n\
         out = print(f(to_tensor([cast(100, i64), cast(400, i64), \
         cast(200, i64), cast(50, i64)])))\n";
    let (ok, stderr, stdout) = c_build_outcome(program, "i64_maxred");
    assert!(
        !stderr.contains("panicked"),
        "the compiler must not panic; stderr: {stderr}"
    );
    if ok {
        // chelis#732 Phase 2: rank-0 renders bare ([05-OBS-4]) and i64
        // prints exact integers ([05-OBS-2]) in the compiled lane too.
        assert_eq!(
            stdout.as_deref(),
            Some("400"),
            "if it builds, it must be correct"
        );
    }
}

// ===========================================================================
// chelis#723 / #684 - i64 sum exactness vs what gets printed
// ===========================================================================

const I64_SUM_2P53: &str = "module M.Main\n\
     def f(x: tensor[1, 2, i64]) -> tensor[1, i64] = sum(x, 1)\n\
     out = print(to_list(f(to_tensor([[cast(9007199254740992, i64), cast(1, i64)]]))))\n";

/// **The C lane's i64 tensor sum is EXACT at 2^53 + 1.** This `to_list`
/// route originally proved that the computation was correct while the tensor
/// printer still laundered the stored value through double (#723).
#[test]
fn c_int64_tensor_sum_is_exact_via_to_list() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    assert_eq!(
        c_first_line(I64_SUM_2P53, "i64_sum_exact"),
        "[9007199254740993]",
        "the compiled i64 sum accumulates exactly"
    );
}

/// Before typed eval storage, this printed `[9007199254740992]` because the
/// element collapsed through `Vec<f64>` before the sum ran (#684).
#[test]
fn eval_int64_tensor_sum_is_exact_at_2p53() {
    assert_eq!(
        eval_first_line(I64_SUM_2P53).expect("eval"),
        "[9007199254740993]"
    );
}

/// Green since chelis#732 Phase 2 (un-ignored per the plan's B2.3): the
/// generated print helper prints i64 elements through `long long`
/// printf, so the direct tensor print carries the SAME exact sum `to_list`
/// proved. (The pre-fix helper's `(double)` cast printed ...992.0,
/// chelis#723.)
#[test]
fn c_int64_tensor_print_is_exact_above_2p53() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let program = "module M.Main\n\
         def f(x: tensor[1, 2, i64]) -> tensor[1, i64] = sum(x, 1)\n\
         out = print(f(to_tensor([[cast(9007199254740992, i64), cast(1, i64)]])))\n";
    let line = c_first_line(program, "i64_sum_print");
    common::assert_elements_in_domain("i64", &line, "i64_sum_print");
    assert!(
        line.contains("9007199254740993"),
        "the printed tensor must carry the exact i64 the runtime holds; got: {line}"
    );
}

// ===========================================================================
// chelis#724 - integer mean: three stages, three answers
// ===========================================================================

/// Historical row (kept for the archaeology): eval printed 187.5 (a
/// fractional value inside an i64-typed tensor) and compiled C printed
/// 187.0 - three stages, three answers. RE-AUTHORED at the chelis#729
/// rework per the DECIDED chelis#724 disposition (reject integer mean):
/// both lanes now agree by construction because `chelis check` rejects
/// the program before either lane runs, with the chelis#724 capability
/// diagnostic pointing at the cast-first idiom. Lane agreement is the
/// row's original assertion, now satisfied as agreement-in-rejection.
#[test]
fn int64_tensor_mean_lanes_agree() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let program = "module M.Main\n\
         def f(x: tensor[4, i64]) -> tensor[i64] = mean(x, 0)\n\
         out = print(f(to_tensor([cast(100, i64), cast(400, i64), \
         cast(200, i64), cast(50, i64)])))\n";
    let eval_err = eval_first_line(program)
        .expect_err("integer mean must be rejected by the checker (chelis#724)");
    assert!(
        eval_err.contains("chelis#724") && !eval_err.contains("numeric trap"),
        "eval-lane rejection must be the check-time chelis#724 capability \
         diagnostic, not a runtime trap; got: {eval_err}"
    );
    let (ok, c_stderr, _) = c_build_outcome(program, "i64_mean");
    assert!(
        !ok && c_stderr.contains("chelis#724"),
        "compiled-lane build must reject with the same chelis#724 \
         diagnostic (the check precedes codegen); got ok={ok}, stderr: {c_stderr}"
    );
}

// ===========================================================================
// CONTROLS: the clean surfaces
// ===========================================================================

/// All seven f32 reductions agree across lanes on exactly-representable
/// values - byte-identically since chelis#732 Phase 2 (§C2.3): both lanes
/// render the rank-0 result bare ([05-OBS-4]) and integers as integers
/// ([05-OBS-2]). (The Phase 1 interim carried per-lane expected strings
/// while the compiled lane kept its pre-contract wrapper.)
#[test]
fn f32_reductions_agree_across_lanes() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    for (op, ret, expected) in [
        ("sum", "f32", "9.0"),
        ("mean", "f32", "2.25"),
        ("max_reduce", "f32", "4.5"),
        ("min_reduce", "f32", "0.5"),
        ("prod_reduce", "f32", "8.4375"),
        ("argmax_reduce", "i64", "1"),
        ("argmin_reduce", "i64", "3"),
    ] {
        let program = format!(
            "module M.Main\n\
             def f(x: tensor[4, f32]) -> tensor[{ret}] = {op}(x, 0)\n\
             out = print(f(to_tensor([1.5, 4.5, 2.5, 0.5])))\n"
        );
        let eval_got = eval_first_line(&program).expect("eval");
        common::assert_elements_in_domain(ret, &eval_got, op);
        assert_eq!(eval_got, expected, "{op} (eval)");
        let c_got = c_first_line(&program, &format!("red_{op}"));
        common::assert_elements_in_domain(ret, &c_got, op);
        assert_eq!(c_got, expected, "{op} (C)");
    }
}

/// i64 `sum` agrees across lanes on in-range values (the one i64
/// reduction that neither panics nor diverges).
#[test]
fn int64_sum_agrees_across_lanes_in_range() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let program = "module M.Main\n\
         def f(x: tensor[4, i64]) -> tensor[i64] = sum(x, 0)\n\
         out = print(f(to_tensor([cast(100, i64), cast(400, i64), \
         cast(200, i64), cast(50, i64)])))\n";
    // chelis#732 Phase 2: BOTH lanes render the rank-0 i64 result bare
    // ([05-OBS-4]) as an exact integer ([05-OBS-2]) - byte-identical.
    assert_eq!(eval_first_line(program).expect("eval"), "750");
    assert_eq!(c_first_line(program, "i64_sum_range"), "750");
}

/// **The Bool dtype is clean end-to-end in both lanes** - probed rather than
/// believing either reading of the "bool storage today is 4-byte
/// f32-encoded" comment (chelis-runtime lib.rs:144-152). Whatever the
/// storage is, the values are right. Since chelis#732 Phase 2 BOTH lanes
/// print bool tensor elements as `true`/`false` ([05-OBS-2], chelis#726
/// closed), so each row carries ONE expected string, byte-identical
/// across lanes.
#[test]
fn bool_dtype_is_clean_in_both_lanes() {
    let rows: &[(&str, &str)] = &[
        (
            "module M.Main\ndef run() -> bool = and(true, not(false))\nout = print(run())\n",
            "true",
        ),
        (
            "module M.Main\ndef f() -> tensor[3, bool] = to_tensor([true, false, true])\nout = print(f())\n",
            "tensor(shape=[3], data=[true, false, true])",
        ),
        (
            "module M.Main\nout = print(to_list(to_tensor([true, false, true])))\n",
            "[true, false, true]",
        ),
        (
            "module M.Main\ndef f(x: tensor[3, bool]) -> tensor[3, bool] = not(x)\nout = print(f(to_tensor([true, false, true])))\n",
            "tensor(shape=[3], data=[false, true, false])",
        ),
        (
            "module M.Main\ndef run() -> i64 = cast(true, i64)\nout = print(run())\n",
            "1",
        ),
        (
            "module M.Main\ndef f(x: tensor[2, f32], y: tensor[2, f32]) -> tensor[2, bool] = cmplt(x, y)\nout = print(f(to_tensor([1.0, 3.0]), to_tensor([2.0, 2.0])))\n",
            "tensor(shape=[2], data=[true, false])",
        ),
    ];
    let have_cc = c_toolchain_available();
    for (i, (program, expected)) in rows.iter().enumerate() {
        assert_eq!(
            eval_first_line(program).expect("eval"),
            *expected,
            "bool row {i} (eval)"
        );
        if have_cc {
            assert_eq!(
                c_first_line(program, &format!("bool_row_{i}")),
                *expected,
                "bool row {i} (C)"
            );
        }
    }
}
