//! [05-OP-35]: scalar and tensor closeness share the comparison width and
//! exceptional-value rules. A storage-width subtraction must never certify
//! a difference that is greater than the tolerance at the comparison width.
use assert_cmd::Command;

#[path = "common/mod.rs"]
mod common;
use common::{make_app, write_file};

fn verdict(
    dtype: &str,
    actual: &str,
    expected: &str,
    tol: &str,
    tensor: bool,
) -> std::process::Output {
    let expression = |value: &str| {
        match value {
            "inf" => "div(1.0, 0.0)",
            "-inf" => "neg(div(1.0, 0.0))",
            "NaN" => "div(0.0, 0.0)",
            _ => value,
        }
        .to_string()
    };
    let actual = expression(actual);
    let expected = expression(expected);
    let tol = expression(tol);
    let (_dir, reef_home, app) = make_app("issue-2988-close");
    let assertion = if tensor {
        "assert_close_tensor(scalar_to_tensor(actual), scalar_to_tensor(expected), tol, \"width-witness\")"
    } else {
        "assert_close(actual, expected, tol, \"width-witness\")"
    };
    write_file(
        &app.join("src/main.ch"),
        &format!(
            "module Demo.Main\nimport Std.Test (assert_close, assert_close_tensor)\ndef test_case() -> unit ! {{ Test }} = {{\n  actual = cast({actual}, {dtype})\n  expected = cast({expected}, {dtype})\n  tol = cast({tol}, {dtype})\n  {assertion}\n}}\nran = test_case()\n",
        ),
    );
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(&app)
        .args(["eval", "--file", app.join("src/main.ch").to_str().unwrap()])
        .output()
        .unwrap()
}

#[test]
fn scalar_and_tensor_closeness_agree_at_the_specified_comparison_width() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        let small = if dtype == "bf16" {
            "-0.00390625"
        } else {
            "-0.00048828125"
        };
        for (actual, expected, tol, pass, invalid) in [
            ("1.0", small, "1.0", false, false),
            ("1.0", "0.0", "1.0", true, false),
            ("1.0", "1.0", "0.0", true, false),
            ("0.0", "-0.0", "0.0", true, false),
            ("1.0", "2.0", "0.0", false, false),
            ("inf", "inf", "0.1", true, false),
            ("-inf", "-inf", "0.1", true, false),
            ("inf", "-inf", "0.1", false, false),
            ("inf", "1.0", "0.1", false, false),
            ("NaN", "NaN", "0.0", false, false),
            ("NaN", "1.0", "1.0", false, false),
            ("1.0", "NaN", "1.0", false, false),
            ("1.0", "1.0", "inf", false, true),
            ("1.0", "1.0", "-inf", false, true),
            ("1.0", "1.0", "NaN", false, true),
            ("1.0", "1.0", "-1.0", false, true),
        ] {
            for tensor in [false, true] {
                let output = verdict(dtype, actual, expected, tol, tensor);
                let stderr = String::from_utf8_lossy(&output.stderr);
                assert_eq!(
                    output.status.success(),
                    pass,
                    "{dtype} {actual}/{expected} tol={tol} tensor={tensor}: {stderr}"
                );
                if !pass {
                    assert!(stderr.contains("width-witness"), "{stderr}");
                    assert_eq!(stderr.contains("invalid tolerance"), invalid, "{stderr}");
                }
            }
        }
    }
}
