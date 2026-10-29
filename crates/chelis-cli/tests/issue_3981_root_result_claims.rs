//! Named result claims survive shape-preserving host/tensor composition.
//! Authority: spec/04-type-system.md §4.7.2, §4.7.3 and §4.7.6.

use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
};
use tempfile::TempDir;

fn source(cols: usize, matched: bool, returned: &str, helper: &str) -> String {
    let values = if cols == 0 {
        "to_tensor([], f32)".to_string()
    } else {
        "to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32, 9.0f32, 10.0f32, 11.0f32, 12.0f32, 13.0f32, 14.0f32, 15.0f32])".to_string()
    };
    let (start, end) = if matched { (0, 5) } else { (1, 4) };
    format!(
        "module SliceClaim\nexport (forward)\ntype Inputs[rows, cols] =\n  | Inputs {{ x: tensor[rows, cols, f32], start: i64, end: i64 }}\n{helper}\nsig forward[rows, cols]: Inputs[rows, cols] -> tensor[rows, cols, f32]\ndef forward(inputs: Inputs[rows, cols]) = {{\n  part = {{\n    indices = to_tensor(range(inputs.start, inputs.end), i64)\n    selected = gather(inputs.x, indices, 0i32)\n    result = reshape(selected, [sub(inputs.end, inputs.start), shape(inputs.x, 1i32)])\n    result\n  }}\n  {returned}\n}}\nx = reshape({values}, [5i64, {cols}i64])\nout = shape(forward(Inputs {{ x, start: {start}i64, end: {end}i64 }}), 0i32)\n"
    )
}

fn command(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_chelis"))
        .args(arguments)
        .output()
        .expect("execute CLI")
}

fn text(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn prepare_source(dir: &TempDir, stem: &str, source: &str) -> PathBuf {
    let path = dir.path().join(format!("{stem}.ch"));
    let decls = chelis_surf::parser::parse_str(source, &chelis_syntax_host::NativeStack)
        .expect("fixture parses");
    fs::write(&path, chelis_surf::format::format_program(&decls)).unwrap();
    let filename = path.to_str().unwrap();
    let fmt = command(&["fmt", "--check", filename]);
    assert!(fmt.status.success(), "{}", text(&fmt));
    let lint = command(&["lint", "--check", filename]);
    assert!(lint.status.success(), "{}", text(&lint));
    let checked = command(&["check", filename]);
    assert!(checked.status.success(), "{}", text(&checked));
    let report: serde_json::Value = serde_json::from_slice(&checked.stdout).expect("check JSON");
    assert_eq!(report["score"].as_f64(), Some(1.0));
    assert_eq!(report["errors"], serde_json::json!([]));
    assert_eq!(report["unresolved_names"], serde_json::json!([]));
    assert_eq!(report["untyped_nodes"], serde_json::json!(0));
    path
}

fn execute(dir: &TempDir, stem: &str, source: &str) -> (Output, Output) {
    let path = prepare_source(dir, stem, source);
    let path = path.to_str().unwrap();
    let evaluated = command(&["eval", "--file", path]);
    let built = dir.path().join(format!("{stem}-built"));
    let output = command(&["build", path, "-o", built.to_str().unwrap()]);
    assert!(
        output.status.success(),
        "build must succeed: {}",
        text(&output)
    );
    let native = Command::new(built.join(stem))
        .output()
        .expect("execute built C binary");
    (evaluated, native)
}

fn assert_claim_rejection(output: &Output, operation: &str) {
    assert_eq!(output.status.code(), Some(1), "{}", text(output));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("extent `rows`: inputs.x axis 0 = 5"),
        "{stderr}"
    );
    assert!(
        stderr.contains(&format!("{operation} axis 0 = 3")),
        "{stderr}"
    );
    assert!(
        stderr
            .lines()
            .any(|line| line == format!("numeric trap: domain in {operation} at i64")),
        "{stderr}"
    );
    assert!(
        !String::from_utf8_lossy(&output.stdout)
            .lines()
            .any(|line| line.starts_with("out = ")),
        "a false result never reaches out: {}",
        text(output)
    );
}

fn assert_success(output: &Output) {
    assert!(output.status.success(), "{}", text(output));
    assert!(
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .any(|line| line == "out = 5"),
        "{}",
        text(output)
    );
}

fn where_source() -> &'static str {
    "where(cmplt(part, part), part, part)"
}

#[test]
fn where_root_rejects_mismatched_named_rows() {
    let dir = tempfile::tempdir().unwrap();
    let source = source(3, false, where_source(), "");
    let (eval, native) = execute(&dir, "where_bad", &source);
    assert_claim_rejection(&eval, "where");
    assert_claim_rejection(&native, "where");
}

#[test]
fn where_root_rejects_mismatched_named_rows_with_zero_columns() {
    let dir = tempfile::tempdir().unwrap();
    let source = source(0, false, where_source(), "");
    let (eval, native) = execute(&dir, "where_empty_bad", &source);
    assert_claim_rejection(&eval, "where");
    assert_claim_rejection(&native, "where");
}

#[test]
fn where_root_accepts_matching_named_rows_with_populated_or_empty_data() {
    let dir = tempfile::tempdir().unwrap();
    for cols in [3, 0] {
        let source = source(cols, true, where_source(), "");
        let (eval, native) = execute(&dir, &format!("where_good_{cols}"), &source);
        assert_success(&eval);
        assert_success(&native);
    }
}

#[test]
fn direct_computed_return_preserves_its_named_guard_and_success_twin() {
    let dir = tempfile::tempdir().unwrap();
    for cols in [3, 0] {
        for matched in [false, true] {
            let source = source(cols, matched, "part", "");
            let (eval, native) = execute(&dir, &format!("direct_{cols}_{matched}"), &source);
            if matched {
                assert_success(&eval);
                assert_success(&native);
            } else {
                assert_claim_rejection(&eval, "reshape");
                assert_claim_rejection(&native, "reshape");
            }
        }
    }
}

#[test]
fn aliases_and_shape_preserving_helpers_keep_the_root_claim() {
    let dir = tempfile::tempdir().unwrap();
    for (name, returned, helper, operation) in [
        (
            "alias",
            "{ chosen = where(cmplt(part, part), part, part)\n alias = chosen\n alias }",
            "",
            "where",
        ),
        (
            "nested",
            "where(cmplt(part, part), neg(part), part)",
            "",
            "where",
        ),
        ("neg", "neg(part)", "", "neg"),
        (
            "helper",
            "select(part)",
            "def select[n, m](x: tensor[n, m, f32]) -> tensor[n, m, f32] = where(cmplt(x, x), x, x)",
            "where",
        ),
        // The identity activation receives an already produced value: its
        // formal ingress owns this call-boundary check, attributed to load.
        (
            "identity",
            "identity(where(cmplt(part, part), part, part))",
            "def identity[n, m](x: tensor[n, m, f32]) -> tensor[n, m, f32] = x",
            "load",
        ),
    ] {
        for cols in [3, 0] {
            for matched in [false, true] {
                let source = source(cols, matched, returned, helper);
                let (eval, native) = execute(&dir, &format!("{name}_{cols}_{matched}"), &source);
                if matched {
                    assert_success(&eval);
                    assert_success(&native);
                } else {
                    assert_claim_rejection(&eval, operation);
                    assert_claim_rejection(&native, operation);
                }
            }
        }
    }
}

#[test]
fn a_nested_tensor_result_uses_the_same_declaring_entry_witnesses() {
    let dir = tempfile::tempdir().unwrap();
    for cols in [3, 0] {
        for matched in [false, true] {
            let returned = format!("({}, 9i64)", where_source());
            let source = source(cols, matched, &returned, "")
                .replacen(
                    "-> tensor[rows, cols, f32]",
                    "-> (tensor[rows, cols, f32], i64)",
                    1,
                )
                .replace("}), 0i32)", "}).0, 0i32)");
            let (eval, native) = execute(&dir, &format!("tuple_{cols}_{matched}"), &source);
            if matched {
                assert_success(&eval);
                assert_success(&native);
            } else {
                assert_claim_rejection(&eval, "where");
                assert_claim_rejection(&native, "where");
            }
        }
    }
}

#[test]
fn named_root_claim_executes_between_the_producers_surrounding_effects() {
    let dir = tempfile::tempdir().unwrap();
    let returned = "{ _ = print(\"before-where\")\n chosen = where(cmplt(part, part), part, part)\n _ = print(\"after-where\")\n chosen }";
    for cols in [3, 0] {
        for matched in [false, true] {
            let source = source(cols, matched, returned, "").replacen(
                "-> tensor[rows, cols, f32]\n",
                "-> tensor[rows, cols, f32] ! { IO }\n",
                1,
            );
            let (eval, native) = execute(&dir, &format!("effects_{cols}_{matched}"), &source);
            for output in [&eval, &native] {
                if matched {
                    assert_success(output);
                } else {
                    assert_claim_rejection(output, "where");
                }
                let stdout = String::from_utf8_lossy(&output.stdout);
                assert_eq!(stdout.matches("before-where").count(), 1, "{stdout}");
                assert_eq!(
                    stdout.matches("after-where").count(),
                    usize::from(matched),
                    "{stdout}"
                );
            }
        }
    }
}

fn runtime_target_source(target: i64, nested: bool, wildcard: bool, effects: bool) -> String {
    let tensor = if wildcard {
        "tensor[*, *, f32]"
    } else {
        "tensor[rows, cols, f32]"
    };
    let result = if nested {
        format!("({tensor}, i64)")
    } else {
        tensor.into()
    };
    let effect = if effects { " ! { IO }" } else { "" };
    let before = if effects {
        "_ = print(\"before-target\")"
    } else {
        ""
    };
    let after_reshape = if effects {
        "_ = print(\"after-reshape\")"
    } else {
        ""
    };
    let after_where = if effects {
        "_ = print(\"after-where\")"
    } else {
        ""
    };
    let returned = if nested { "(chosen, 9i64)" } else { "chosen" };
    let projection = if nested { ".0" } else { "" };
    format!(
        "module RuntimeTarget\nexport (forward)\ntype Inputs[rows, cols] =\n  | Inputs {{ x: tensor[rows, cols, f32], target: i64 }}\ndef forward[rows, cols](inputs: Inputs[rows, cols], direct: tensor[cols, f32], samples: List[tensor[cols, f32]]) -> {result}{effect} = {{\n  {before}\n  part = reshape(inputs.x, [shape(inputs.x, 0i32), inputs.target])\n  {after_reshape}\n  chosen = where(cmplt(part, part), part, part)\n  {after_where}\n  {returned}\n}}\nx = reshape(to_tensor([], f32), [0i64, 5i64])\nrecord = Inputs {{ x, target: {target}i64 }}\nout = shape(forward(record, to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32]), [to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32])]){projection}, 1i32)\n"
    )
}

#[test]
fn opaque_runtime_reshape_targets_reach_named_result_guards() {
    let dir = tempfile::tempdir().unwrap();
    // Aggregate-result effect timing is a separate #1277 obligation.
    for (nested, effects) in [(false, false), (true, false), (false, true)] {
        for target in [3, 5] {
            let source = runtime_target_source(target, nested, false, effects);
            let (eval, native) = execute(
                &dir,
                &format!("opaque_claim_{nested}_{effects}_{target}"),
                &source,
            );
            for (lane, output) in [("eval", &eval), ("native", &native)] {
                if target == 5 {
                    assert_success(output);
                } else {
                    assert_eq!(output.status.code(), Some(1), "{}", text(output));
                    assert!(
                        text(output)
                            .contains("extent `cols`: inputs.x axis 1 = 5, where axis 1 = 3"),
                        "{lane}, nested={nested}, effects={effects}: {}",
                        text(output)
                    );
                    assert!(
                        text(output).contains("numeric trap: domain in where at i64"),
                        "{}",
                        text(output)
                    );
                    assert!(
                        !String::from_utf8_lossy(&output.stdout)
                            .lines()
                            .any(|line| line.starts_with("out = ")),
                        "{}",
                        text(output)
                    );
                }
                if effects {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    assert_eq!(stdout.matches("before-target").count(), 1, "{stdout}");
                    assert_eq!(stdout.matches("after-reshape").count(), 1, "{stdout}");
                    assert_eq!(
                        stdout.matches("after-where").count(),
                        usize::from(target == 5),
                        "{lane}, nested={nested}, effects={effects}, target={target}: {stdout}"
                    );
                }
            }
        }
    }
}

#[test]
fn opaque_runtime_reshape_targets_preserve_values_and_validation_without_result_claims() {
    let dir = tempfile::tempdir().unwrap();
    for target in [-1, 3, 5] {
        let source = runtime_target_source(target, false, true, true);
        let (eval, native) = execute(&dir, &format!("opaque_value_{target}"), &source);
        for output in [&eval, &native] {
            if target < 0 {
                assert_eq!(output.status.code(), Some(1), "{}", text(output));
                assert!(text(output).contains("reshape"), "{}", text(output));
            } else {
                assert!(output.status.success(), "{}", text(output));
                assert!(
                    String::from_utf8_lossy(&output.stdout)
                        .lines()
                        .any(|line| line == format!("out = {target}")),
                    "{}",
                    text(output)
                );
            }
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert_eq!(stdout.matches("before-target").count(), 1, "{stdout}");
            assert_eq!(
                stdout.matches("after-reshape").count(),
                usize::from(target >= 0),
                "{stdout}"
            );
            assert_eq!(
                stdout.matches("after-where").count(),
                usize::from(target >= 0),
                "{stdout}"
            );
        }
    }
}

fn rank_zero_source(gradient: bool, invalid: bool) -> String {
    let name = if gradient { "loss" } else { "value" };
    let parameter = if invalid {
        "[n](x: tensor[n, f32])"
    } else {
        "(x: tensor[1, f32])"
    };
    let callee = if gradient {
        format!("grad({name})")
    } else {
        name.into()
    };
    let values = if invalid {
        "[3.0f32, 4.0f32]"
    } else {
        "[3.0f32]"
    };
    format!(
        "module RankZero\ndef {name}{parameter} -> f32 = tensor_to_scalar(reshape(x, []))\nout = {callee}(to_tensor({values}))\n"
    )
}

#[test]
fn represented_rank_zero_targets_preserve_scalar_and_gradient_execution() {
    let dir = tempfile::tempdir().unwrap();
    for gradient in [false, true] {
        let source = rank_zero_source(gradient, false);
        let (eval, native) = execute(&dir, &format!("rank_zero_{gradient}"), &source);
        let expected = if gradient {
            "out = tensor(shape=[1], data=[1.0])"
        } else {
            "out = 3.0"
        };
        for output in [&eval, &native] {
            assert!(output.status.success(), "{}", text(output));
            assert!(
                String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .any(|line| line == expected),
                "{}",
                text(output)
            );
        }
    }
}

fn assert_rank_zero_product_rejection(output: &Output) {
    assert_eq!(output.status.code(), Some(1), "{}", text(output));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("reshape target has 1 elements but the tensor has 2"),
        "{stderr}"
    );
    assert!(
        stderr.contains("numeric trap: domain in reshape at i64"),
        "{stderr}"
    );
    assert!(
        !String::from_utf8_lossy(&output.stdout)
            .lines()
            .any(|line| line.starts_with("out = ")),
        "{}",
        text(output)
    );
}

#[test]
fn represented_rank_zero_targets_reject_nonunit_element_counts() {
    let dir = tempfile::tempdir().unwrap();
    let (eval, native) = execute(&dir, "rank_zero_bad", &rank_zero_source(false, true));
    for output in [&eval, &native] {
        assert_rank_zero_product_rejection(output);
    }
}

#[test]
fn represented_rank_zero_gradient_rejects_nonunit_element_count() {
    let dir = tempfile::tempdir().unwrap();
    let stem = "rank_zero_grad_bad";
    let path = prepare_source(&dir, stem, &rank_zero_source(true, true));
    let path = path.to_str().unwrap();
    assert_rank_zero_product_rejection(&command(&["eval", "--file", path]));
    let built = dir.path().join("built");
    let output = command(&["build", path, "-o", built.to_str().unwrap()]);
    if output.status.success() {
        let native = Command::new(built.join(stem))
            .output()
            .expect("execute native gradient rejection");
        assert_rank_zero_product_rejection(&native);
    } else {
        assert!(
            text(&output).contains("reshape") && text(&output).contains("product mismatch"),
            "{}",
            text(&output)
        );
    }
}
