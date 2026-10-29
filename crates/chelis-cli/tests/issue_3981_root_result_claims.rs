//! Named result claims survive shape-preserving host/tensor composition.
//! Authority: spec/04-type-system.md §4.7.2, §4.7.3 and §4.7.6.

use std::{
    fs,
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

fn execute(dir: &TempDir, stem: &str, source: &str) -> (Output, Output) {
    let path = dir.path().join(format!("{stem}.ch"));
    let decls = chelis_surf::parser::parse_str(source, &chelis_syntax_host::NativeStack)
        .expect("fixture parses");
    fs::write(&path, chelis_surf::format::format_program(&decls)).unwrap();
    let path = path.to_str().unwrap();
    let fmt = command(&["fmt", "--check", path]);
    assert!(fmt.status.success(), "{}", text(&fmt));
    let lint = command(&["lint", "--check", path]);
    assert!(lint.status.success(), "{}", text(&lint));
    let checked = command(&["check", path]);
    assert!(checked.status.success(), "{}", text(&checked));
    let report: serde_json::Value = serde_json::from_slice(&checked.stdout).expect("check JSON");
    assert_eq!(report["score"].as_f64(), Some(1.0));
    assert_eq!(report["errors"], serde_json::json!([]));
    assert_eq!(report["unresolved_names"], serde_json::json!([]));
    assert_eq!(report["untyped_nodes"], serde_json::json!(0));
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
