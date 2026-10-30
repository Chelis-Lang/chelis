//! [05-OP-12]/[05-OP-13], spec/04 §4.7.3-.5 and [04-TOT-2]:
//! checked extrema execute after an authored runtime numel reshape.

use serde_json::Value;
use std::{
    path::PathBuf,
    process::{Command, Output},
};
use tempfile::TempDir;

fn cli(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_chelis"))
        .args(arguments)
        .output()
        .expect("execute chelis")
}

fn text(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn prepare(directory: &TempDir, name: &str, source: &str) -> PathBuf {
    let path = directory.path().join(format!("{name}.ch"));
    let program = chelis_surf::parser::parse_str(source, &chelis_syntax_host::NativeStack).unwrap();
    std::fs::write(&path, chelis_surf::format::format_program(&program)).unwrap();
    for command in ["fmt", "lint"] {
        let output = cli(&[command, "--check", path.to_str().unwrap()]);
        assert!(output.status.success(), "{name}: {}", text(&output));
    }
    path
}

fn check(path: &std::path::Path) {
    let output = cli(&["check", path.to_str().unwrap()]);
    assert!(output.status.success(), "{}", text(&output));
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["score"].as_f64(), Some(1.0));
    assert_eq!(report["errors"], serde_json::json!([]));
    assert_eq!(report["unresolved_names"], serde_json::json!([]));
    assert_eq!(report["untyped_nodes"], 0);
}

fn execute(directory: &TempDir, name: &str, path: &std::path::Path) -> (Output, Output) {
    check(path);
    let evaluated = cli(&["eval", "--file", path.to_str().unwrap()]);
    let output_directory = directory.path().join(format!("{name}-native"));
    let built = cli(&[
        "build",
        path.to_str().unwrap(),
        "--target",
        "c",
        "-o",
        output_directory.to_str().unwrap(),
    ]);
    assert!(built.status.success(), "{name}: {}", text(&built));
    let native = Command::new(output_directory.join(name)).output().unwrap();
    (evaluated, native)
}

fn success(directory: &TempDir, name: &str, source: &str, dtype: &str, bits: &str, value: &str) {
    let path = prepare(directory, name, source);
    let json = cli(&["eval", "--file", path.to_str().unwrap(), "--json"]);
    assert!(json.status.success(), "{name}: {}", text(&json));
    let report: Value = serde_json::from_slice(&json.stdout).unwrap();
    let result = report["roots"]
        .as_array()
        .unwrap()
        .iter()
        .find(|root| root["name"] == "result")
        .unwrap();
    assert_eq!(result["value"]["type"], "scalar");
    assert_eq!(result["value"]["value"]["dtype"], dtype);
    assert_eq!(result["value"]["value"]["bits"], bits);
    let (evaluated, native) = execute(directory, name, &path);
    for output in [evaluated, native] {
        assert!(output.status.success(), "{name}: {}", text(&output));
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            format!("result = {value}\n")
        );
    }
}

fn program(
    op: &str,
    dtype: &str,
    rank: usize,
    symbolic: bool,
    guarded: bool,
    reshaped: bool,
) -> String {
    let binders = if symbolic {
        if rank == 1 { "[n]" } else { "[n, m]" }
    } else {
        ""
    };
    let dims = if symbolic {
        if rank == 1 { "n" } else { "n, m" }
    } else if rank == 1 {
        "3"
    } else {
        "2, 3"
    };
    let operand = if reshaped {
        "reshape(x, [numel(x)])"
    } else {
        "x"
    };
    let body = format!("tensor_to_scalar({op}({operand}, 0i32))");
    let body = if guarded && rank == 2 {
        format!(
            "if gt(shape(x, 0i32), 0i64) then if gt(shape(x, 1i32), 0i64) then {body} else 0.0{dtype} else 0.0{dtype}"
        )
    } else if guarded {
        format!("if gt(numel(x), 0i64) then {body} else 0.0{dtype}")
    } else {
        body
    };
    let values = if rank == 1 {
        format!("[1.0{dtype}, -2.0{dtype}, 0.0{dtype}]")
    } else {
        format!("[[1.0{dtype}, -2.0{dtype}, 0.0{dtype}], [4.0{dtype}, -3.0{dtype}, 2.0{dtype}]]")
    };
    format!(
        "module Extrema\ndef extreme{binders}(x: tensor[{dims}, {dtype}]) -> {dtype} = {body}\nresult = extreme(to_tensor({values}))\n"
    )
}

fn matrix(dtype: &str, bits: [&str; 4]) {
    let directory = tempfile::tempdir().unwrap();
    for (op_index, op) in ["min_reduce", "max_reduce"].into_iter().enumerate() {
        for rank in [1, 2] {
            let value = match (op_index, rank) {
                (0, 1) => "-2.0",
                (0, 2) => "-3.0",
                (1, 1) => "1.0",
                _ => "4.0",
            };
            for symbolic in [false, true] {
                for guarded in [false, true] {
                    let name = format!("{op}_{dtype}_{rank}_{symbolic}_{guarded}");
                    success(
                        &directory,
                        &name,
                        &program(op, dtype, rank, symbolic, guarded, true),
                        dtype,
                        bits[op_index * 2 + rank - 1],
                        value,
                    );
                }
            }
        }
    }
}

#[test]
fn f16_extrema_after_numel_reshape_preserve_stored_values() {
    matrix("f16", ["c000", "c200", "3c00", "4400"]);
}

#[test]
fn f32_extrema_after_numel_reshape_preserve_stored_values() {
    matrix("f32", ["c0000000", "c0400000", "3f800000", "40800000"]);
}

#[test]
fn f64_extrema_after_numel_reshape_preserve_stored_values() {
    matrix(
        "f64",
        [
            "c000000000000000",
            "c008000000000000",
            "3ff0000000000000",
            "4010000000000000",
        ],
    );
}

#[test]
fn direct_extrema_and_sum_controls_keep_their_known_results() {
    let directory = tempfile::tempdir().unwrap();
    for (op, reshaped, bits, value) in [
        ("min_reduce", false, "c0000000", "-2.0"),
        ("max_reduce", false, "3f800000", "1.0"),
        ("sum", true, "bf800000", "-1.0"),
    ] {
        success(
            &directory,
            &format!("control_{op}"),
            &program(op, "f32", 1, true, false, reshaped),
            "f32",
            bits,
            value,
        );
    }
}

#[test]
fn runtime_empty_axes_trap_while_untaken_extrema_do_not_execute() {
    let directory = tempfile::tempdir().unwrap();
    for dtype in ["f16", "f32", "f64"] {
        for op in ["min_reduce", "max_reduce"] {
            for guarded in [false, true] {
                let reduction = format!("tensor_to_scalar({op}(reshape(x, [numel(x)]), 0i32))");
                let body = if guarded {
                    format!("if gt(numel(x), 0i64) then {reduction} else 0.0{dtype}")
                } else {
                    reduction
                };
                let source = format!(
                    "def extreme[n](x: tensor[n, {dtype}]) -> {dtype} = {body}\ndef empty(size: i64) -> tensor[*, {dtype}] = insert(scalar_to_tensor(0.0{dtype}), 0i32, size)\nresult = extreme(empty(0i64))\n"
                );
                let name = format!("empty_{op}_{dtype}_{guarded}");
                let path = prepare(&directory, &name, &source);
                let (evaluated, native) = execute(&directory, &name, &path);
                for output in [evaluated, native] {
                    if guarded {
                        assert!(output.status.success(), "{name}: {}", text(&output));
                        assert_eq!(output.stdout, b"result = 0.0\n");
                    } else {
                        assert_eq!(output.status.code(), Some(1), "{name}: {}", text(&output));
                        assert!(output.stdout.is_empty(), "{name}: {}", text(&output));
                        assert!(
                            String::from_utf8_lossy(&output.stderr)
                                .contains(&format!("numeric trap: domain in {op} at {dtype}")),
                            "{name}: {}",
                            text(&output)
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn invalid_shape_products_trap_before_extrema_and_later_effects() {
    let directory = tempfile::tempdir().unwrap();
    for op in ["min_reduce", "max_reduce"] {
        for extra in [0, 1, -4] {
            let source = format!(
                "def extreme[n](x: tensor[n, f32], extra: i64) -> f32 ! {{ IO }} = {{\n _ = print(\"before\")\n part = reshape(x, [add(numel(x), extra)])\n _ = print(\"after\")\n tensor_to_scalar({op}(part, 0i32))\n}}\nresult = extreme(to_tensor([1.0f32, -2.0f32, 0.0f32]), {extra}i64)\n"
            );
            let name = format!(
                "product_{op}_{}",
                extra.to_string().replace('-', "negative_")
            );
            let path = prepare(&directory, &name, &source);
            let (evaluated, native) = execute(&directory, &name, &path);
            for output in [evaluated, native] {
                if extra == 0 {
                    assert!(output.status.success(), "{name}: {}", text(&output));
                    assert_eq!(
                        output.stdout,
                        format!(
                            "before\nafter\nresult = {}\n",
                            if op == "min_reduce" { "-2.0" } else { "1.0" }
                        )
                        .as_bytes()
                    );
                } else {
                    assert_eq!(output.status.code(), Some(1), "{name}: {}", text(&output));
                    assert_eq!(output.stdout, b"before\n");
                    assert!(
                        String::from_utf8_lossy(&output.stderr)
                            .contains("numeric trap: domain in reshape at i64"),
                        "{name}: {}",
                        text(&output)
                    );
                }
            }
        }
    }
}

#[test]
fn invalid_axis_dtype_precision_and_effect_contracts_stay_rejected() {
    let directory = tempfile::tempdir().unwrap();
    for op in ["min_reduce", "max_reduce"] {
        for (case, source) in [
            (
                "axis",
                format!(
                    "def extreme[n](x: tensor[n, f32]) -> f32 = tensor_to_scalar({op}(reshape(x, [numel(x)]), 1i32))\n"
                ),
            ),
            (
                "dtype",
                format!(
                    "def extreme[n](x: tensor[n, bool]) -> bool = tensor_to_scalar({op}(reshape(x, [numel(x)]), 0i32))\n"
                ),
            ),
            (
                "precision",
                format!(
                    "def extreme[n](x: tensor[n, f32]) -> f64 = tensor_to_scalar({op}(reshape(x, [numel(x)]), 0i32))\n"
                ),
            ),
            (
                "shape_precision",
                format!(
                    "def extreme[n](x: tensor[n, f32]) -> f32 = tensor_to_scalar({op}(reshape(x, [cast(numel(x), i32)]), 0i32))\n"
                ),
            ),
            (
                "effect",
                format!(
                    "def extreme[n](x: tensor[n, f32]) -> f32 ! {{}} = {{\n _ = print(\"forbidden\")\n tensor_to_scalar({op}(reshape(x, [numel(x)]), 0i32))\n}}\n"
                ),
            ),
        ] {
            let path = prepare(&directory, &format!("invalid_{op}_{case}"), &source);
            let output = cli(&["check", path.to_str().unwrap()]);
            assert!(!output.status.success(), "{op}/{case}: {}", text(&output));
            let report: Value = serde_json::from_slice(&output.stdout).unwrap();
            assert!(!report["errors"].as_array().unwrap().is_empty());
            assert_eq!(report["unresolved_names"], serde_json::json!([]));
            assert!(report["score"].as_f64().unwrap() < 1.0);
        }
    }
}
