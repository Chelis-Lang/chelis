//! Core C/eval parity controls for chelis#1951, #1957, #1966, and #1967.
//!
//! These tests distinguish supported direct calls from unsupported
//! first-class callable values, require an injective private C namespace for
//! source definitions, and lock left-to-right eager evaluation across a
//! lambda-application boundary.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

#[derive(Debug)]
struct CArtifacts {
    _dir: tempfile::TempDir,
    out_dir: std::path::PathBuf,
    source: String,
}

fn build_c(program: &str, name: &str) -> Result<CArtifacts, (String, Vec<String>)> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, program);
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().expect("utf-8 path"),
            "--target",
            "c",
            "--output",
            out_dir.to_str().expect("utf-8 path"),
        ])
        .output()
        .expect("chelis build should run");
    let artifacts = if out_dir.is_dir() {
        std::fs::read_dir(&out_dir)
            .expect("read output directory")
            .map(|entry| {
                entry
                    .expect("output entry")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect()
    } else {
        Vec::new()
    };
    if !output.status.success() {
        return Err((
            String::from_utf8_lossy(&output.stderr).into_owned(),
            artifacts,
        ));
    }
    let source = std::fs::read_to_string(out_dir.join(format!("{name}.c")))
        .expect("a successful C build emits its translation unit");
    Ok(CArtifacts {
        _dir: dir,
        out_dir,
        source,
    })
}

fn eval(program: &str, name: &str) -> std::process::Output {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    write_file(&path, program);
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().expect("utf-8 path")])
        .output()
        .expect("chelis eval should run")
}

fn link_and_run(artifacts: &CArtifacts, name: &str) -> std::process::Output {
    let status = common::link_generated(&artifacts.out_dir, &format!("{name}.c"), name);
    assert!(status.success(), "generated C must link: {status}");
    std::process::Command::new(artifacts.out_dir.join(name))
        .output()
        .expect("compiled binary should run")
}

/// #1951: C has no general first-class-function ABI.  An application result
/// used as a function must reject before artifact emission rather than return
/// the argument unchanged.
#[test]
fn returned_function_application_rejects_before_c_artifact_emission() {
    let program = "def g[n, k](x: tensor[n, f32]) -> tensor[k, f32] = shrink(x, [[1i64, shape(x, 0i32)]])\n\
                   def h[k](y: tensor[k, f32]) -> tensor[k, f32] = add(y, y)\n\
                   def pick[p](f: (tensor[p, f32]) -> tensor[p, f32]) -> (tensor[p, f32]) -> tensor[p, f32] = f\n\
                   def main() = (pick(h))(g(to_tensor([1.0f32, 2.0f32, 3.0f32])))\n\
                   out = print(main())\n";
    let eval_output = eval(program, "returned_function_eval");
    assert!(
        eval_output.status.success(),
        "eval remains the supported reference lane: {}",
        String::from_utf8_lossy(&eval_output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&eval_output.stdout).contains("data=[4.0, 6.0]"),
        "eval must apply the returned callable: {}",
        String::from_utf8_lossy(&eval_output.stdout)
    );

    let (stderr, artifacts) = build_c(program, "returned_function_c")
        .expect_err("C must reject an unrepresentable returned function value");
    assert!(
        stderr.contains("unsupported:") && stderr.contains("function value"),
        "C rejection must be typed and identify the unsupported ABI boundary:\n{stderr}"
    );
    assert!(
        artifacts.is_empty(),
        "a fenced program must not emit partial C artifacts: {artifacts:?}"
    );
}

/// #1957: a source def is never emitted under the public spelling, so a legal
/// Chelis name cannot collide with a libc function such as `read`.
#[test]
fn source_def_read_uses_private_c_namespace_and_runs() {
    let artifacts = build_c(
        "def read(x: tensor[2, f32]) -> tensor[2, f32] = mul(x, to_tensor([3.0f32, 5.0f32]))\n\
         out = print(read(to_tensor([1.0f32, 2.0f32])))\n",
        "libc_read",
    )
    .expect("a legal source def named read must build");
    assert!(
        artifacts.source.contains("chelis_fn_"),
        "source defs must use the private emitted-function namespace:\n{}",
        artifacts.source
    );
    assert!(
        !artifacts.source.contains(" read("),
        "the raw libc spelling must not appear as a generated C call or declarator:\n{}",
        artifacts.source
    );
    if common::gcc_available() {
        let run = link_and_run(&artifacts, "libc_read");
        assert!(
            run.status.success(),
            "a private generated symbol must link and run: {}",
            String::from_utf8_lossy(&run.stderr)
        );
        assert!(
            String::from_utf8_lossy(&run.stdout).contains("data=[3.0, 10.0]"),
            "the C result must remain correct: {}",
            String::from_utf8_lossy(&run.stdout)
        );
    }
}

/// #1966: eager argument evaluation crosses an ordinary lambda application,
/// even if the parameter is unused in the selected body branch.
#[test]
fn lambda_application_evaluates_argument_before_unused_body_parameter() {
    let program = "def f(z: i64) -> i64 = (fn (x) -> if false then x else 7i64)(trunc_div(1i64, z))\n\
                   out = f(0i64)\n";
    let eval_output = eval(program, "lambda_argument_eval");
    assert!(
        !eval_output.status.success(),
        "eval must trap on the argument"
    );
    assert!(
        String::from_utf8_lossy(&eval_output.stderr)
            .contains("numeric trap: division by zero in trunc_div at i64"),
        "eval must report the argument trap: {}",
        String::from_utf8_lossy(&eval_output.stderr)
    );

    let artifacts = build_c(program, "lambda_argument_c").expect("C build must succeed");
    if common::gcc_available() {
        let run = link_and_run(&artifacts, "lambda_argument_c");
        assert!(!run.status.success(), "C must not skip the argument trap");
        assert!(
            String::from_utf8_lossy(&run.stderr)
                .contains("numeric trap: division by zero in trunc_div at i64"),
            "C must report the same first trap as eval: {}",
            String::from_utf8_lossy(&run.stderr)
        );
    }
}

/// #1966 negative parity: when both the argument and body can trap, the
/// argument's trap is observably first.
#[test]
fn lambda_application_argument_trap_precedes_body_trap() {
    let program = "def f(z: i64) -> i64 = (fn (x) -> add(trunc_div(-9223372036854775808i64, -1i64), x))(trunc_div(1i64, z))\n\
                   out = f(0i64)\n";
    let artifacts = build_c(program, "lambda_order_c").expect("C build must succeed");
    if common::gcc_available() {
        let run = link_and_run(&artifacts, "lambda_order_c");
        let stderr = String::from_utf8_lossy(&run.stderr);
        assert!(!run.status.success(), "C must trap");
        assert!(
            stderr.contains("division by zero") && !stderr.contains("overflow"),
            "the argument trap must precede the body trap:\n{stderr}"
        );
    }
}

/// #1967: the caller's `y` is materialized before the lambda body introduces
/// its own `y`; no C name capture may rewrite the argument.
#[test]
fn lambda_body_binding_cannot_capture_caller_argument_expression() {
    let program = "def f(y: i64) -> i64 = (fn (x) -> {\n\
                     y = 9i64\n\
                     x\n\
                   })(y)\n\
                   out = f(3i64)\n";
    let eval_output = eval(program, "lambda_shadow_eval");
    assert!(
        eval_output.status.success() && String::from_utf8_lossy(&eval_output.stdout).contains("3"),
        "eval control must return the caller value: stdout={} stderr={}",
        String::from_utf8_lossy(&eval_output.stdout),
        String::from_utf8_lossy(&eval_output.stderr)
    );

    let artifacts = build_c(program, "lambda_shadow_c").expect("C build must succeed");
    if common::gcc_available() {
        let run = link_and_run(&artifacts, "lambda_shadow_c");
        assert!(
            run.status.success() && String::from_utf8_lossy(&run.stdout).contains("3"),
            "C must preserve the caller value rather than capture it: stdout={} stderr={}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        );
    }
}

/// #1967 positive neighbor: a body binding of the lambda parameter itself is
/// ordinary local rebinding, not a caller-capture hazard.
#[test]
fn lambda_body_can_bind_its_own_parameter() {
    let artifacts = build_c(
        "def f(y: i64) -> i64 = (fn (x) -> {\n\
           x = 9i64\n\
           x\n\
         })(y)\n\
         out = f(3i64)\n",
        "lambda_own_binding_c",
    )
    .expect("C build must succeed");
    if common::gcc_available() {
        let run = link_and_run(&artifacts, "lambda_own_binding_c");
        assert!(
            run.status.success() && String::from_utf8_lossy(&run.stdout).contains("9"),
            "a lambda's local parameter rebinding remains supported: stdout={} stderr={}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        );
    }
}
