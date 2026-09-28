//! [06] §7.5: structural rejection beneath a comparison keeps the atom's
//! exact AD reason even though the comparison contributes zero cotangents.

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

fn eval_program(program: &str) -> Result<String, String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("comparison_grad.ch");
    write_file(&path, program);
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().expect("utf8 path")])
        .output()
        .expect("run chelis eval");
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).into_owned())
    }
}

fn comparison_program(operand: &str) -> String {
    format!(
        "module M.Main\n\
         def f(x: tensor[2, f32], m: tensor[2, f32]) -> f32 = \
           tensor_to_scalar(sum(where(cmplt({operand}, to_tensor([1.0f32, 1.0f32])), \
             mul(x, x), x), 0i32))\n\
         out = print(grad(f, wrt=x)(to_tensor([3.0f32, 4.0f32]), \
           to_tensor([0.0f32, 2.0f32])))\n"
    )
}

#[test]
fn comparison_operand_casts_report_the_exact_structural_reason() {
    for (operand, op) in [
        ("cast(cast(m, i32), f32)", "cast"),
        ("cast(cast_trunc(m, i32), f32)", "cast_trunc"),
    ] {
        let stderr = eval_program(&comparison_program(operand))
            .expect_err("a piecewise constant comparison operand must reject grad");
        assert!(
            stderr.contains(&format!(
                "grad: {op} is non-differentiable (piecewise constant)"
            )),
            "expected the atom's reason for {op}: {stderr}"
        );
        assert!(
            !stderr.contains("no reverse-mode adjoint") && !stderr.contains("node "),
            "the diagnostic must not leak an IR node id: {stderr}"
        );
    }
}

#[test]
fn ordinary_comparison_still_differentiates_the_selected_values() {
    let output = eval_program(&comparison_program("m"))
        .expect("a plain comparison contributes zeros without rejecting grad");
    assert!(
        output.contains("tensor(shape=[2], data=[6.0, 1.0])"),
        "expected exact branch gradients [6, 1]: {output}"
    );
}
