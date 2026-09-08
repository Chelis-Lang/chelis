//! chelis#730 Phase 1 - the branded-rejection parity locks for the
//! section C5 live-site conversions of `spec/design/loud_unsupported.md`.
//!
//! Part II's law: every conversion lands with a test that the formerly
//! substituted case either fails loudly with the branded section C2 shape
//! (`unsupported: <what> on <context> (<stage>); <authority>: <hint>`) or computes the
//! authored value, plus a control that the supported neighbor still works.
//! Phase 3 turns the narrow-scalar and scalar-math rows below from rejection
//! locks into exact positive controls while preserving the other negatives.
//!
//! Rows covered here: 2 (the builtin stub, chelis#682/#704/#705/#715),
//! 3 (to_string, chelis#734), 6/7 (narrow scalars, chelis#714/#718).
//! Rows with pre-existing acceptance files: 1 (issue_703 +
//! grad_zero_placeholder_matrix), 5 (loud_unsupported_census_canaries),
//! 8/12 (reduce_window_nonliteral_matrix), 9/20 (issue_709), 10
//! (narrow_dtype_matrix), 11 (reduction_and_bitwise_matrix), 13
//! (loud_unsupported_census_canaries).

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

/// `chelis build --target c`; (build_ok, stderr, concatenated emission).
fn c_build(program: &str, name: &str) -> (bool, String, String) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
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
    let mut emitted = String::new();
    if out_dir.is_dir() {
        for entry in std::fs::read_dir(&out_dir).expect("read out dir") {
            let p = entry.expect("entry").path();
            if p.is_file()
                && let Ok(text) = std::fs::read_to_string(&p)
            {
                emitted.push_str(&text);
            }
        }
    }
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        emitted,
    )
}

/// Build + link + run; `Ok(first stdout line)` or the failing stage's text.
fn c_run_first_line(program: &str, name: &str) -> Result<String, String> {
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
    if !built.status.success() {
        return Err(String::from_utf8_lossy(&built.stderr).into_owned());
    }
    let status = common::link_generated(&out_dir, &format!("{name}.c"), name);
    if !status.success() {
        return Err(format!("link failed: {status}"));
    }
    let run = std::process::Command::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    if !run.status.success() {
        return Err(String::from_utf8_lossy(&run.stderr).into_owned());
    }
    Ok(String::from_utf8_lossy(&run.stdout)
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string())
}

/// Assert a build rejection carries the frozen section C2 shape: the
/// `error:` surfacing, the literal `unsupported: ` brand, the ` on `
/// context clause, a parenthesized stage, and a typed authority clause.
fn assert_branded_rejection(stderr: &str, what_fragment: &str, ctx: &str) {
    assert!(
        stderr.contains("error:"),
        "{ctx}: a build rejection surfaces as an `error:` line; got: {stderr}"
    );
    let line = stderr
        .lines()
        .find(|l| l.contains("unsupported: "))
        .unwrap_or_else(|| panic!("{ctx}: no branded `unsupported: ` line in: {stderr}"));
    assert!(
        line.contains(what_fragment),
        "{ctx}: the diagnostic must name `{what_fragment}`; got: {line}"
    );
    assert!(
        line.contains(" on ")
            && (line.contains("); deliberate [") || line.contains("); unimplemented chelis#")),
        "{ctx}: the diagnostic must follow the authority-bearing \
         `unsupported: <what> on <context> (<stage>); <authority>: <hint>` shape; got: {line}"
    );
}

// ===========================================================================
// Row 2 (chelis#704/#705/#715): the remaining host-only stub arm, branded.
// ===========================================================================

/// The remaining host-only builtin that used to hit the stub fails the build
/// with the branded diagnostic naming the builtin and leaves no stub marker
/// behind. Scalar math moved to exact compiled support in chelis#729 Phase 3.
///
/// The former bitwise case was promoted to supported C lowering in
/// chelis#682 and is covered by the positive precision matrix.
#[test]
fn stubbed_builtins_are_rejected_with_the_branded_shape() {
    let cases: &[(&str, &str, &str)] = &[(
        "tensor_scan_705",
        "def gen() -> tensor[5, f32] = \
             tensor_scan(0.0, fn (prev: f32, i: int64) -> add(prev, 1.0), cast(5, int64))\n\
             out = gen()\n",
        "tensor_scan",
    )];
    for (name, program, builtin) in cases {
        let (ok, stderr, emitted) = c_build(program, name);
        assert!(
            !ok,
            "{name}: an unimplemented builtin must fail the build (census row 2)"
        );
        assert_branded_rejection(&stderr, builtin, name);
        assert!(
            !emitted.contains("unsupported builtin"),
            "{name}: no stub marker may be left in any emitted artifact"
        );
    }
}

/// Control (B2.4): the supported neighbors still build and run - scalar
/// sqrt/exp and tensor relu were correct before the conversion and must
/// stay correct after it.
#[test]
fn working_neighbors_still_build_and_run() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let got = c_run_first_line(
        "module M.Main\ndef f(x: f32) -> f32 = sqrt(x)\nout = print(f(2.25))\n",
        "ctl_sqrt_p1",
    )
    .expect("scalar sqrt must keep building and running");
    assert_eq!(got, "1.5");
    let got = c_run_first_line(
        "module M.Main\n\
         def f(x: tensor[4, f32]) -> tensor[4, f32] = relu(x)\n\
         out = print(f(to_tensor([-1.0, 2.0, -3.0, 4.0])))\n",
        "ctl_relu_p1",
    )
    .expect("tensor relu must keep building and running");
    assert!(got.contains("data=[0.0, 2.0, 0.0, 4.0]"), "got: {got}");
}

// ===========================================================================
// Row 3 (chelis#734): to_string of tensors/lists, branded.
// ===========================================================================

/// The formerly `<value>`-substituting cases fail the build with the
/// branded diagnostic; eval keeps stringifying correctly (the lane that
/// worked stays working).
#[test]
fn to_string_of_tensor_and_list_is_rejected_branded() {
    let tensor_program = "module M.Main\n\
         def f(x: tensor[2, f32]) -> string = to_string(x)\n\
         out = print(f(to_tensor([1.5, 2.5])))\n";
    assert_eq!(
        eval_first_line(tensor_program).expect("eval stringifies tensors"),
        "tensor(shape=[2], data=[1.5, 2.5])"
    );
    let (ok, stderr, emitted) = c_build(tensor_program, "ts_tensor_p1");
    assert!(!ok, "to_string(tensor) must fail the build (census row 3)");
    assert_branded_rejection(&stderr, "to_string", "ts_tensor_p1");
    assert!(
        !emitted.contains("<value>"),
        "no `<value>` placeholder may be left in any emitted artifact"
    );

    let list_program = "module M.Main\n\
         def f(xs: List[int64]) -> string = to_string(xs)\n\
         out = print(f([cast(1, int64), cast(2, int64)]))\n";
    assert_eq!(
        eval_first_line(list_program).expect("eval stringifies lists"),
        "[1, 2]"
    );
    let (ok, stderr, _) = c_build(list_program, "ts_list_p1");
    assert!(!ok, "to_string(list) must fail the build (census row 3)");
    assert_branded_rejection(&stderr, "to_string", "ts_list_p1");
}

/// Control: the scalar to_string arms keep working in the compiled lane.
#[test]
fn to_string_scalar_arms_still_work() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let got = c_run_first_line(
        "module M.Main\ndef f() -> string = to_string(cast(7, int64))\nout = print(f())\n",
        "ts_ctl_p1",
    )
    .expect("to_string of an int64 scalar must keep working");
    assert_eq!(got, "7");
}

// ===========================================================================
// Rows 6/7 (chelis#714/#718): narrow scalars, branded at the baking point.
// ===========================================================================

/// Rows 6/7, adjudicated by execution at Phase 1 (B2.5):
///
/// Both parameter and cast-literal shapes now cross the same exact typed
/// target boundary. The old cast-literal "success" was a widened-double
/// accident, not proof of an f16 ABI; Phase 3 locks both shapes to the real
/// representation and own-width arithmetic.
#[test]
fn narrow_scalar_arithmetic_computes_at_declared_width() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    // Param-typed shape: exact f16 multiplication.
    let f16_param_program = "module M.Main\n\
         def f(x: f16) -> f16 = mul(x, cast(2.0, f16))\n\
         out = print(f(cast(0.25, f16)))\n";
    assert_eq!(
        c_run_first_line(f16_param_program, "f16_param_p1")
            .expect("param-typed f16 arithmetic must compile and run"),
        "0.5"
    );

    // Cast-literal shape reaches the identical exact target decision.
    let f16_program = "module M.Main\n\
         def f() -> f16 = add(cast(0.5, f16), cast(0.25, f16))\n\
         out = print(f())\n";
    assert_eq!(
        eval_first_line(f16_program).expect("eval computes f16 scalars"),
        "0.75"
    );
    assert_eq!(
        c_run_first_line(f16_program, "f16_add_p1")
            .expect("cast-literal f16 arithmetic must compile and run"),
        "0.75"
    );
}

/// Control: f32/f64/int64 scalar arithmetic - the resolved host types -
/// keeps building and running.
#[test]
fn resolved_scalar_arithmetic_still_works() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    for (program, expected, name) in [
        (
            "module M.Main\ndef f() -> f32 = add(cast(1.5, f32), cast(0.25, f32))\n\
             out = print(f())\n",
            "1.75",
            "ctl_f32_add_p1",
        ),
        (
            "module M.Main\ndef f() -> int64 = add(cast(126, int64), cast(1, int64))\n\
             out = print(f())\n",
            "127",
            "ctl_i64_add_p1",
        ),
    ] {
        let got = c_run_first_line(program, name).expect("resolved scalars must keep working");
        assert_eq!(got, expected, "{name}");
    }
}
