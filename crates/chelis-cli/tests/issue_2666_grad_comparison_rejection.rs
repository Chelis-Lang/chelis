//! [06] §7.5: a comparison operand is a control slot. A piecewise-constant
//! conversion read only there executes forward and contributes nothing
//! (chelis#3464), while the same conversion with a data path keeps the atom's
//! exact AD reason.

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

/// `where(cmplt(operand, 1), x * x, x)` summed, at `x = [3, 4]` and
/// `m = [0, 2]`. With `data_use`, the sum also adds `x * operand`.
fn comparison_program(operand: &str, data_use: bool) -> String {
    let selected = format!("where(cmplt({operand}, to_tensor([1.0f32, 1.0f32])), mul(x, x), x)");
    let body = if data_use {
        format!("add({selected}, mul(x, {operand}))")
    } else {
        selected
    };
    format!(
        "module M.Main\n\
         def f(x: tensor[2, f32], m: tensor[2, f32]) -> f32 = \
           tensor_to_scalar(sum({body}, 0i32))\n\
         out = print(grad(f, wrt=x)(to_tensor([3.0f32, 4.0f32]), \
           to_tensor([0.0f32, 2.0f32])))\n"
    )
}

/// chelis#3464: `cast(cast(m, i32), f32)` is `[0, 2]`, so the first element
/// takes `x * x` and the second `x`: exact branch gradients `[6, 1]`.
#[test]
fn comparison_operand_casts_are_control_slot_reads() {
    for operand in ["cast(cast(m, i32), f32)", "cast(cast_trunc(m, i32), f32)"] {
        let output = eval_program(&comparison_program(operand, false))
            .unwrap_or_else(|stderr| panic!("{operand} read by a comparison: {stderr}"));
        assert!(
            output.contains("tensor(shape=[2], data=[6.0, 1.0])"),
            "expected exact branch gradients [6, 1] for {operand}: {output}"
        );
    }
}

#[test]
fn comparison_operand_casts_with_a_data_path_report_the_exact_structural_reason() {
    for (operand, op) in [
        ("cast(cast(x, i32), f32)", "cast"),
        ("cast(cast_trunc(x, i32), f32)", "cast_trunc"),
    ] {
        let stderr = eval_program(&comparison_program(operand, true))
            .expect_err("a piecewise constant conversion on a data path must reject grad");
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
    let output = eval_program(&comparison_program("m", false))
        .expect("a plain comparison contributes zeros without rejecting grad");
    assert!(
        output.contains("tensor(shape=[2], data=[6.0, 1.0])"),
        "expected exact branch gradients [6, 1]: {output}"
    );
}
