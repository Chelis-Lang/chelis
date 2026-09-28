//! #2373: linked C and Eval execute the same checked computed concat.

mod common;

use common::{build_and_run, gcc_available, link_generated, parse_tensor_data, write_file};
use std::process::Command as StdCommand;
use tempfile::tempdir;

use assert_cmd::Command;

fn assert_only_trap_line(output: &std::process::Output, expected: &str) {
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let traps = text
        .lines()
        .filter(|line| line.contains("numeric trap:"))
        .collect::<Vec<_>>();
    assert_eq!(traps, [expected], "{text}");
}

fn zero_element_concat_source(width: i64) -> String {
    format!(
        "def empty() -> tensor[0, f32] = to_tensor([])\ndef join[n](x: tensor[0, n, 2, f32]) -> tensor[0, *, 2, f32] = concat([x, x], 1i32)\nbase = insert(insert(empty(), 1i32, 2i64), 1i32, {width}i64)\noutput = join(base)\n"
    )
}

#[test]
fn user_authored_trap_line_is_an_ordinary_eval_failure() {
    for source in [
        "fail(\"numeric trap: domain in concat at i64\")",
        "fail(\"ordinary failure\\nnumeric trap: domain in concat at i64\")",
        "test_assert(false, \"ordinary assertion\\nnumeric trap: domain in concat at i64\")",
    ] {
        for json in [false, true] {
            let mut command = Command::cargo_bin("chelis").expect("binary");
            command.env("CHELIS_STYLE_GATE_DISABLE", "1").arg("eval");
            if json {
                command.arg("--json");
            }
            let output = command.arg(source).output().expect("eval");
            assert!(!output.status.success());
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                stderr.starts_with("error: "),
                "source={source:?}, json={json}, stderr={stderr:?}"
            );
        }
    }
}

#[test]
fn registered_numeric_builtins_keep_canonical_eval_trap_lines() {
    for (source, expected) in [
        (
            "add(9223372036854775807i64, 1i64)",
            "numeric trap: overflow in add at i64\n",
        ),
        (
            "trunc_div(-9223372036854775808i64, -1i64)",
            "numeric trap: overflow in trunc_div at i64\n",
        ),
    ] {
        for json in [false, true] {
            let mut command = Command::cargo_bin("chelis").expect("binary");
            command.env("CHELIS_STYLE_GATE_DISABLE", "1").arg("eval");
            if json {
                command.arg("--json");
            }
            let output = command.arg(source).output().expect("eval");
            assert!(!output.status.success());
            assert_eq!(
                String::from_utf8_lossy(&output.stderr),
                expected,
                "source={source:?}, json={json}"
            );
        }
    }
}

fn program(producer: &str) -> String {
    let binding = match producer {
        "direct" => "scores = x",
        "copy" => "scores = copy(x)",
        "add" => "scores = add(x, x)",
        "mul" => "scores = mul(x, x)",
        _ => panic!("unknown producer"),
    };
    format!(
        "module Repro.ComputedConcat\ndef probabilities[s](x: tensor[s, s, f32]) -> tensor[s, *, f32] = {{\n  {binding}\n  softmax(concat([scores, scores], cast(1, i32)), -1)\n}}\noutput = probabilities(to_tensor([[0.0, 1.0], [2.0, 0.0]]))\n"
    )
}

#[test]
fn direct_copy_and_arithmetic_linked_c_values() {
    assert!(gcc_available(), "this oracle requires a linked C binary");
    for producer in ["direct", "copy", "add", "mul"] {
        let stdout = build_and_run(&program(producer), &format!("computed_concat_{producer}"));
        let line = stdout
            .lines()
            .find(|line| line.starts_with("output = tensor("))
            .unwrap_or_else(|| panic!("missing tensor output: {stdout}"));
        assert!(line.contains("shape=[2, 4]"), "{producer}: {line}");
        let actual = parse_tensor_data(&stdout, "output");
        let input = if producer == "add" {
            [[0.0_f64, 2.0], [4.0, 0.0]]
        } else if producer == "mul" {
            [[0.0_f64, 1.0], [4.0, 0.0]]
        } else {
            [[0.0_f64, 1.0], [2.0, 0.0]]
        };
        let expected: Vec<f64> = input
            .iter()
            .flat_map(|row| {
                let denominator = 2.0 * row.iter().map(|x| x.exp()).sum::<f64>();
                row.iter()
                    .chain(row.iter())
                    .map(move |x| x.exp() / denominator)
            })
            .collect();
        assert_eq!(actual.len(), expected.len(), "{producer}");
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "{producer} element {index}: {actual} != {expected}"
            );
        }
    }
}

#[test]
fn shape_changing_matmul_producer_keeps_eval_c_values() {
    assert!(gcc_available(), "this oracle requires a linked C binary");
    let source = "def join[s](x: tensor[s, s, f32]) -> tensor[s, *, f32] = {\n  product = matmul(x, x)\n  concat([product, product], 1i32)\n}\noutput = join(to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32]]))\n";
    let stdout = build_and_run(source, "computed_concat_matmul");
    assert_eq!(
        parse_tensor_data(&stdout, "output"),
        vec![7.0, 10.0, 7.0, 10.0, 15.0, 22.0, 15.0, 22.0]
    );
}

#[test]
fn non_axis_mismatch_reaches_concat_in_both_lanes() {
    assert!(gcc_available(), "this oracle requires a linked C binary");
    let source =
        "output = concat([to_tensor([[1.0f32], [2.0f32]]), to_tensor([[3.0f32]])], 1i32)\n";
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("non_axis_mismatch.ch");
    let out_dir = dir.path().join("non_axis_mismatch-out");
    write_file(&path, source);
    let eval = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("eval");
    assert!(!eval.status.success());
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
    assert!(link_generated(&out_dir, "non_axis_mismatch.c", "non_axis_mismatch").success());
    let c = StdCommand::new(out_dir.join("non_axis_mismatch"))
        .output()
        .expect("linked C run");
    assert!(!c.status.success());
    for output in [&eval, &c] {
        assert_only_trap_line(output, "numeric trap: domain in concat at i64");
    }
}

#[test]
fn zero_element_concat_output_metadata_overflow_matches_eval_and_c() {
    assert!(gcc_available(), "this oracle requires a linked C binary");
    let source = zero_element_concat_source(1_i64 << 61);
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("concat_metadata_overflow.ch");
    let out_dir = dir.path().join("concat_metadata_overflow-out");
    write_file(&path, &source);

    let eval = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("eval");
    assert!(!eval.status.success());
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
    assert!(
        link_generated(
            &out_dir,
            "concat_metadata_overflow.c",
            "concat_metadata_overflow"
        )
        .success()
    );
    let c = StdCommand::new(out_dir.join("concat_metadata_overflow"))
        .output()
        .expect("linked C run");
    assert!(!c.status.success());
    for output in [&c, &eval] {
        assert_only_trap_line(output, "numeric trap: overflow in concat at i64");
    }
}

#[test]
fn zero_element_concat_output_metadata_within_i64_succeeds_on_both_lanes() {
    assert!(gcc_available(), "this oracle requires a linked C binary");
    let source = zero_element_concat_source(1_i64 << 60);
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("concat_metadata_fits.ch");
    let out_dir = dir.path().join("concat_metadata_fits-out");
    write_file(&path, &source);

    let eval = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("eval");
    assert!(
        eval.status.success(),
        "{}",
        String::from_utf8_lossy(&eval.stderr)
    );
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
    assert!(link_generated(&out_dir, "concat_metadata_fits.c", "concat_metadata_fits").success());
    let c = StdCommand::new(out_dir.join("concat_metadata_fits"))
        .output()
        .expect("linked C run");
    assert!(c.status.success(), "{}", String::from_utf8_lossy(&c.stderr));
    for output in [&eval, &c] {
        let text = String::from_utf8_lossy(&output.stdout);
        assert!(text.contains("shape=[0, 2305843009213693952, 2]"), "{text}");
        assert!(text.contains("data=[]"), "{text}");
    }
}

#[test]
fn copied_extent_keeps_result_claim_on_eval_and_linked_c() {
    assert!(gcc_available(), "this oracle requires a linked C binary");
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("copied_claim.ch");
    let out_dir = dir.path().join("copied_claim-out");
    let source = "def join[n, m](x: tensor[n, m, f32]) -> tensor[n, 3, f32] = concat([copy(x), copy(x)], 1i32)\noutput = join(to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32]]))\n";
    write_file(&path, source);

    let eval = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("eval");
    assert!(!eval.status.success());
    let eval_json = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--json", "--file", path.to_str().unwrap()])
        .output()
        .expect("eval json");
    assert!(!eval_json.status.success());
    for output in [&eval, &eval_json] {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!stderr.starts_with("error: "), "{stderr}");
    }

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
    let status = link_generated(&out_dir, "copied_claim.c", "copied_claim");
    assert!(status.success(), "link failed: {status}");
    let c = StdCommand::new(out_dir.join("copied_claim"))
        .output()
        .expect("linked C run");
    assert!(!c.status.success());
    for (lane, output) in [("Eval", eval), ("C", c)] {
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            text.contains("numeric trap: domain in concat at i64")
                && text.contains("claimed = 3")
                && text.contains("axis 1 = 4"),
            "{lane}: {text}"
        );
    }
}

#[test]
fn helper_aliases_keep_distinct_widths_in_linked_c() {
    assert!(gcc_available(), "this oracle requires a linked C binary");
    let source = "def join[s](x: tensor[s, *, f32]) -> tensor[s, *, f32] = {\n  copied = copy(x)\n  alias = copied\n  parts = [alias, alias]\n  concat(parts, 1i32)\n}\none = join(to_tensor([[1.0f32], [2.0f32]]))\ntwo = join(to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32]]))\n";
    let stdout = build_and_run(source, "computed_concat_aliases");
    for (name, shape, values) in [
        ("one", "shape=[2, 2]", vec![1.0, 1.0, 2.0, 2.0]),
        (
            "two",
            "shape=[2, 4]",
            vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0],
        ),
    ] {
        let line = stdout
            .lines()
            .find(|line| line.starts_with(&format!("{name} = tensor(")))
            .unwrap_or_else(|| panic!("missing {name}: {stdout}"));
        assert!(line.contains(shape), "{name}: {line}");
        assert_eq!(parse_tensor_data(&stdout, name), values, "{name}");
    }
}

#[test]
fn dynamic_axis_and_bound_list_linked_c_values() {
    assert!(gcc_available(), "this oracle requires a linked C binary");
    for (list, binding, stem) in [
        ("[scores, scores]", "", "computed_concat_dynamic_direct"),
        (
            "parts",
            "parts = [scores, scores]\n  ",
            "computed_concat_dynamic_bound",
        ),
    ] {
        let source = format!(
            "def join[s](x: tensor[s, s, f32], axis: i32) -> tensor[*, *, f32] = {{\n  scores = copy(x)\n  {binding}concat({list}, axis)\n}}\noutput = join(to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32]]), cast(1, i32))\n"
        );
        let stdout = build_and_run(&source, stem);
        let line = stdout
            .lines()
            .find(|line| line.starts_with("output = tensor("))
            .unwrap_or_else(|| panic!("missing output: {stdout}"));
        assert!(line.contains("shape=[2, 4]"), "{list}: {line}");
        assert_eq!(
            parse_tensor_data(&stdout, "output"),
            vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0],
            "{list}"
        );
    }
}

#[test]
fn empty_list_fails_in_linked_c() {
    assert!(gcc_available(), "this oracle requires a linked C binary");
    let stem = "computed_concat_empty";
    let source = "parts: List[tensor[2, 2, f32]] = []\noutput = concat(parts, 1i32)\n";
    let expected = "concat expects at least one tensor part";
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{stem}.ch"));
    let out_dir = dir.path().join(format!("{stem}-out"));
    write_file(&path, source);
    let eval = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("eval");
    assert!(!eval.status.success());
    assert_eq!(
        String::from_utf8_lossy(&eval.stderr),
        "error: concat expects at least one tensor part\n"
    );
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
    let status = link_generated(&out_dir, &format!("{stem}.c"), stem);
    assert!(status.success(), "link failed: {status}");
    let run = StdCommand::new(out_dir.join(stem))
        .output()
        .expect("linked C run");
    assert!(!run.status.success(), "{stem} must reject");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(text.contains(expected), "{stem}: {text}");
}
