//! Sweep 3 of the numeric audit brief: bitwise ops beyond int64, the
//! reduction family across dtypes, and the Bool dtype - eval vs compiled C.
//!
//! Findings (each confirmed by execution):
//!
//! | issue | one line |
//! |---|---|
//! | chelis#682 | bitand/shl stub to 0 in C at EVERY width (int8/16/32/64); eval is width-correct |
//! | chelis#692 | max/min/prod/argmax/argmin reduce on an int64 tensor PANIC the compiler (emit.rs:4238/4406/4777) |
//! | chelis#723 | the C print helper renders int64 elements through double, hiding an EXACT sum of 2^53 + 1 |
//! | chelis#724 | mean of an int64 tensor: eval 187.5 (fractional in an int64 tensor), C 187.0 |
//! | chelis#684 | eval's int64 tensor sum is genuinely lossy at 2^53 (Vec<f64> storage) |
//!
//! And the clean surfaces, locked: the Bool dtype is correct end-to-end in
//! both lanes (the lib.rs:144-152 "f32-encoded" comment notwithstanding -
//! probed, not believed), all seven f32 reductions agree across lanes, and
//! the C lane's int64 `sum` is EXACT at 2^53 + 1 (provable only through
//! `to_list`, which is what #723 is about).
//!
//! One lucky green is documented so nobody cites it: `shl(1i8, 9)` prints 0
//! in both lanes - eval because the shift leaves the 8-bit window, C
//! because of the #682 stub. Two unrelated mechanisms colliding.

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

fn scalar_program(expr: &str, ret_ty: &str) -> String {
    format!("module M.Main\ndef run() -> {ret_ty} = {expr}\nout = print(run())\n")
}

// ===========================================================================
// chelis#682 at every width - eval width-correct, C stubs to 0
// ===========================================================================

/// Observed today: C prints 0 for every row (the host_emit.rs:2300 stub).
#[test]
#[ignore = "chelis#682: bitwise ops stub to 0 in compiled C at every integer width; eval is \
            correct (bitand 8, shl 16 at int8/int16/int32/int64). Run with \
            `cargo test -p chelis-cli --test reduction_and_bitwise_matrix -- --ignored`."]
fn bitwise_ops_agree_across_lanes_at_every_width() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    for ty in ["int8", "int16", "int32", "int64"] {
        for (expr, expected) in [
            (format!("bitand(cast(12, {ty}), cast(10, {ty}))"), "8"),
            (format!("shl(cast(1, {ty}), cast(4, {ty}))"), "16"),
        ] {
            let program = scalar_program(&expr, ty);
            assert_eq!(eval_first_line(&program).expect("eval"), expected);
            let c_got = c_first_line(&program, &format!("bw_{ty}_{}", &expr[..4]));
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

/// eval's width semantics for shifts, locked eval-only: shl(1i8, 7) lands on
/// the sign bit and wraps to -128 (today's wrap behavior; if #680's trap
/// contract is extended to shifts this row gets re-authored consciously).
#[test]
fn eval_shift_width_semantics_are_locked() {
    assert_eq!(
        eval_first_line(&scalar_program("shl(cast(1, int8), cast(7, int8))", "int8"))
            .expect("eval"),
        "-128"
    );
    assert_eq!(
        eval_first_line(&scalar_program(
            "shr(cast(-16, int64), cast(2, int64))",
            "int64"
        ))
        .expect("eval"),
        "-4",
        "arithmetic (sign-preserving) right shift"
    );
}

/// The lucky green, documented: shl(1i8, 9) prints 0 in BOTH lanes for two
/// unrelated reasons (eval: the bit left the 8-bit window; C: the #682
/// stub). Never cite this row as cross-lane evidence.
#[test]
fn shl_past_width_agreement_is_a_coincidence_not_evidence() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let program = scalar_program("shl(cast(1, int8), cast(9, int8))", "int8");
    assert_eq!(eval_first_line(&program).expect("eval"), "0");
    assert_eq!(c_first_line(&program, "shl_past_width"), "0");
}

// ===========================================================================
// chelis#692 - int64 reduce ops panic the compiler
// ===========================================================================

/// Observed today: `chelis build` PANICS (emit.rs:4238) instead of either
/// compiling correctly or rejecting with a diagnostic. This row asserts the
/// non-panic contract: any outcome except a Rust backtrace.
#[test]
#[ignore = "chelis#692: max_reduce on an int64 tensor panics the compiler at emit.rs:4238 \
            (min_reduce at 4406, argmax/argmin at 4777). Must become a clean diagnostic or \
            a correct build. Run with \
            `cargo test -p chelis-cli --test reduction_and_bitwise_matrix -- --ignored`."]
fn int64_max_reduce_does_not_panic_the_compiler() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let program = "module M.Main\n\
         def f(x: tensor[4, int64]) -> tensor[int64] = max_reduce(x, 0)\n\
         out = print(f(to_tensor([cast(100, int64), cast(400, int64), \
         cast(200, int64), cast(50, int64)])))\n";
    let (ok, stderr, stdout) = c_build_outcome(program, "i64_maxred");
    assert!(
        !stderr.contains("panicked"),
        "the compiler must not panic; stderr: {stderr}"
    );
    if ok {
        assert_eq!(
            stdout.as_deref(),
            Some("tensor(shape=[], data=[400.0])"),
            "if it builds, it must be correct"
        );
    }
}

// ===========================================================================
// chelis#723 / #684 - int64 sum exactness vs what gets printed
// ===========================================================================

const I64_SUM_2P53: &str = "module M.Main\n\
     def f(x: tensor[1, 2, int64]) -> tensor[1, int64] = sum(x, 1)\n\
     out = print(to_list(f(to_tensor([[cast(9007199254740992, int64), cast(1, int64)]]))))\n";

/// **The C lane's int64 tensor sum is EXACT at 2^53 + 1** - provable only
/// through `to_list`, because the tensor print helper launders int64 through
/// double (#723). This lock keeps the correct computation correct while
/// #723/#684 land.
#[test]
fn c_int64_tensor_sum_is_exact_via_to_list() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    assert_eq!(
        c_first_line(I64_SUM_2P53, "i64_sum_exact"),
        "[9007199254740993]",
        "the compiled int64 sum accumulates exactly"
    );
}

/// Observed today: eval prints [9007199254740992] - the Vec<f64> storage
/// (#684) collapses the element before the sum ever runs.
#[test]
#[ignore = "chelis#684: eval's int64 tensor sum returns 9007199254740992 (f64 storage); the \
            exact answer 9007199254740993 is what the C lane already produces. Run with \
            `cargo test -p chelis-cli --test reduction_and_bitwise_matrix -- --ignored`."]
fn eval_int64_tensor_sum_is_exact_at_2p53() {
    assert_eq!(
        eval_first_line(I64_SUM_2P53).expect("eval"),
        "[9007199254740993]"
    );
}

/// Observed today: the direct tensor print of the SAME exact sum shows
/// ...992.0 - `case CHELIS_I64: value = (double)...` in the emitted print
/// helper (host_emit.rs) cannot carry the value `to_list` just proved.
#[test]
#[ignore = "chelis#723: the C tensor print helper renders int64 through double; the exact \
            sum 9007199254740993 prints as 9007199254740992.0. Run with \
            `cargo test -p chelis-cli --test reduction_and_bitwise_matrix -- --ignored`."]
fn c_int64_tensor_print_is_exact_above_2p53() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let program = "module M.Main\n\
         def f(x: tensor[1, 2, int64]) -> tensor[1, int64] = sum(x, 1)\n\
         out = print(f(to_tensor([[cast(9007199254740992, int64), cast(1, int64)]])))\n";
    let line = c_first_line(program, "i64_sum_print");
    common::assert_elements_in_domain("int64", &line, "i64_sum_print");
    assert!(
        line.contains("9007199254740993"),
        "the printed tensor must carry the exact int64 the runtime holds; got: {line}"
    );
}

// ===========================================================================
// chelis#724 - integer mean: three stages, three answers
// ===========================================================================

/// Observed today: eval prints 187.5 (a fractional value inside an
/// int64-typed tensor), compiled C prints 187.0. This row asserts only lane
/// AGREEMENT, not which of the three defensible semantics lands.
#[test]
#[ignore = "chelis#724: mean of an int64 tensor - eval 187.5, compiled C 187.0, checker \
            blessed both. Lanes must agree once the semantics are authored. Run with \
            `cargo test -p chelis-cli --test reduction_and_bitwise_matrix -- --ignored`."]
fn int64_tensor_mean_lanes_agree() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let program = "module M.Main\n\
         def f(x: tensor[4, int64]) -> tensor[int64] = mean(x, 0)\n\
         out = print(f(to_tensor([cast(100, int64), cast(400, int64), \
         cast(200, int64), cast(50, int64)])))\n";
    let eval_got = eval_first_line(program).expect("eval");
    let c_got = c_first_line(program, "i64_mean");
    // chelis#729 Phase 0: agreement alone is not enough for this row.
    // 187.5 in an int64 tensor is a domain violation even if both lanes
    // were to agree on it; the checker keeps this red until the #724
    // semantics are authored, not merely until the lanes coincide.
    common::assert_elements_in_domain("int64", &eval_got, "i64_mean eval");
    common::assert_elements_in_domain("int64", &c_got, "i64_mean C");
    assert_eq!(
        eval_got, c_got,
        "mean of an integer tensor must mean ONE thing"
    );
}

// ===========================================================================
// CONTROLS: the clean surfaces
// ===========================================================================

/// All seven f32 reductions agree across lanes on exactly-representable
/// values. Per-lane expected strings since chelis#732 Phase 1: eval
/// renders the rank-0 result bare ([05-OBS-4]) and integers as integers
/// ([05-OBS-2]) while the compiled lane keeps its pre-contract wrapper
/// until the Phase 2 migration - the VALUES are asserted equal across
/// lanes, and byte-identical rendering returns at Phase 2 (§C2.3).
#[test]
fn f32_reductions_agree_across_lanes() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    for (op, ret, eval_expected, c_expected) in [
        ("sum", "f32", "9.0", "tensor(shape=[], data=[9.0])"),
        ("mean", "f32", "2.25", "tensor(shape=[], data=[2.25])"),
        ("max_reduce", "f32", "4.5", "tensor(shape=[], data=[4.5])"),
        ("min_reduce", "f32", "0.5", "tensor(shape=[], data=[0.5])"),
        (
            "prod_reduce",
            "f32",
            "8.4375",
            "tensor(shape=[], data=[8.4375])",
        ),
        (
            "argmax_reduce",
            "int64",
            "1",
            "tensor(shape=[], data=[1.0])",
        ),
        (
            "argmin_reduce",
            "int64",
            "3",
            "tensor(shape=[], data=[3.0])",
        ),
    ] {
        let program = format!(
            "module M.Main\n\
             def f(x: tensor[4, f32]) -> tensor[{ret}] = {op}(x, 0)\n\
             out = print(f(to_tensor([1.5, 4.5, 2.5, 0.5])))\n"
        );
        let eval_got = eval_first_line(&program).expect("eval");
        common::assert_elements_in_domain(ret, &eval_got, op);
        assert_eq!(eval_got, eval_expected, "{op} (eval)");
        let c_got = c_first_line(&program, &format!("red_{op}"));
        common::assert_elements_in_domain(ret, &c_got, op);
        assert_eq!(c_got, c_expected, "{op} (C)");
    }
}

/// int64 `sum` agrees across lanes on in-range values (the one int64
/// reduction that neither panics nor diverges).
#[test]
fn int64_sum_agrees_across_lanes_in_range() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let program = "module M.Main\n\
         def f(x: tensor[4, int64]) -> tensor[int64] = sum(x, 0)\n\
         out = print(f(to_tensor([cast(100, int64), cast(400, int64), \
         cast(200, int64), cast(50, int64)])))\n";
    // chelis#732 P1 migration: eval renders the rank-0 int64 result bare
    // ([05-OBS-4]) as an exact integer ([05-OBS-2]); the compiled lane
    // keeps its pre-contract form until Phase 2. Same VALUE, 750, both.
    assert_eq!(eval_first_line(program).expect("eval"), "750");
    assert_eq!(
        c_first_line(program, "i64_sum_range"),
        "tensor(shape=[], data=[750.0])"
    );
}

/// **The Bool dtype is clean end-to-end in both lanes** - probed rather than
/// believing either reading of the "bool storage today is 4-byte
/// f32-encoded" comment (chelis-runtime lib.rs:144-152). Whatever the
/// storage is, the values are right.
/// Per-lane expected strings since chelis#732 Phase 1: eval prints bool
/// tensor elements as true/false ([05-OBS-2], chelis#726's eval half)
/// while the compiled lane keeps its pre-contract 1.0/0.0 form until the
/// Phase 2 generated printer (its red cell is the observation harness's
/// `c_bool_tensor_print_matches_to_list_exit`). Values agree; byte parity
/// returns at Phase 2.
#[test]
fn bool_dtype_is_clean_in_both_lanes() {
    let rows: &[(&str, &str, &str)] = &[
        (
            "module M.Main\ndef run() -> bool = and(true, not(false))\nout = print(run())\n",
            "true",
            "true",
        ),
        (
            "module M.Main\ndef f() -> tensor[3, bool] = to_tensor([true, false, true])\nout = print(f())\n",
            "tensor(shape=[3], data=[true, false, true])",
            "tensor(shape=[3], data=[1.0, 0.0, 1.0])",
        ),
        (
            "module M.Main\nout = print(to_list(to_tensor([true, false, true])))\n",
            "[true, false, true]",
            "[true, false, true]",
        ),
        (
            "module M.Main\ndef f(x: tensor[3, bool]) -> tensor[3, bool] = not(x)\nout = print(f(to_tensor([true, false, true])))\n",
            "tensor(shape=[3], data=[false, true, false])",
            "tensor(shape=[3], data=[0.0, 1.0, 0.0])",
        ),
        (
            "module M.Main\ndef run() -> int64 = cast(true, int64)\nout = print(run())\n",
            "1",
            "1",
        ),
        (
            "module M.Main\ndef f(x: tensor[2, f32], y: tensor[2, f32]) -> tensor[2, bool] = cmplt(x, y)\nout = print(f(to_tensor([1.0, 3.0]), to_tensor([2.0, 2.0])))\n",
            "tensor(shape=[2], data=[true, false])",
            "tensor(shape=[2], data=[1.0, 0.0])",
        ),
    ];
    let have_cc = c_toolchain_available();
    for (i, (program, eval_expected, c_expected)) in rows.iter().enumerate() {
        assert_eq!(
            eval_first_line(program).expect("eval"),
            *eval_expected,
            "bool row {i} (eval)"
        );
        if have_cc {
            assert_eq!(
                c_first_line(program, &format!("bool_row_{i}")),
                *c_expected,
                "bool row {i} (C)"
            );
        }
    }
}
