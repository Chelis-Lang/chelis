//! #1650: the JSON dtype is the oracle; an empty printed payload is insufficient.
use serde_json::{Value, json};
use std::{fs, process::Command};

#[test]
fn empty_tensor_eval_json_retains_all_checked_leaf_dtypes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.ch");
    for (dtype, wire_dtype) in [
        ("f16", "f16"),
        ("bf16", "bf16"),
        ("f32", "f32"),
        ("f64", "f64"),
        ("i8", "int8"),
        ("i16", "int16"),
        ("i32", "int32"),
        ("i64", "int64"),
        ("bool", "bool"),
    ] {
        for (ty, input, shape) in [
            (format!("List[{dtype}]"), "[]", json!([0])),
            (format!("List[List[{dtype}]]"), "[[], []]", json!([2, 0])),
        ] {
            fs::write(&path, format!("xs: {ty} = {input}\nout = to_tensor(xs)\n")).unwrap();
            let output = Command::new(assert_cmd::cargo_bin!("chelis"))
                .env("CHELIS_STYLE_GATE_DISABLE", "1")
                .args(["eval", "--file", path.to_str().unwrap(), "--json"])
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let result: Value = serde_json::from_slice(&output.stdout).unwrap();
            let root = result["roots"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["name"] == "out")
                .unwrap();
            assert_eq!(root["value"]["value"]["shape"], shape, "{result}");
            assert_eq!(
                root["value"]["value"]["data"]["dtype"], wire_dtype,
                "{result}"
            );
            let payload: chelis_compiler_api::schema::TensorElements =
                serde_json::from_value(root["value"]["value"]["data"].clone()).unwrap();
            assert_eq!(payload.len(), 0, "{result}");
        }
    }
}

#[test]
fn invalid_list_element_types_remain_checker_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.ch");
    for source in [
        "out = to_tensor([1i32, 1.0f32])\n",
        "out = to_tensor([\"bad\"])\n",
    ] {
        fs::write(&path, source).unwrap();
        let output = Command::new(assert_cmd::cargo_bin!("chelis"))
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["check", path.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(!output.status.success(), "{output:?}");
    }
}

mod common;

fn run_native_observation(
    source_text: &str,
    c_dtype: &str,
    rank: usize,
    shape: &[usize],
    count: usize,
    wrong_dtype: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("empty.ch");
    let output_dir = dir.path().join("generated");
    fs::write(&source, source_text).unwrap();
    let output = Command::new(assert_cmd::cargo_bin!("chelis"))
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            source.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            output_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let generated = output_dir.join("empty.c");
    let text = fs::read_to_string(&generated).unwrap();
    // Add a typed assertion at the emitted observation point. No producer,
    // allocation, dtype or shape expression is changed by this harness.
    let needle = "static void chelis_print_tensor_stdout(const chelis_tensor* t) {";
    assert_eq!(
        text.matches(needle).count(),
        1,
        "expected one typed observation helper"
    );
    let expected = if wrong_dtype {
        "CHELIS_DTYPE_F32"
    } else {
        c_dtype
    };
    let shape_checks = shape
        .iter()
        .enumerate()
        .map(|(axis, extent)| format!(" || chelis_tensor_shape(t, {axis}) != {extent}"))
        .collect::<String>();
    let check = format!(
        "\nchelis_read_view observed = chelis_tensor_read_view(t);\nif (observed.dtype != {expected} || observed.count != {count} || chelis_tensor_rank(t) != {rank}{shape_checks}) exit(91);\n"
    );
    fs::write(
        &generated,
        text.replacen(needle, &format!("{needle}{check}"), 1),
    )
    .unwrap();
    assert!(common::link_generated(&output_dir, "empty.c", "probe").success());
    let run = Command::new(output_dir.join("probe")).output().unwrap();
    if wrong_dtype {
        assert_eq!(
            run.status.code(),
            Some(91),
            "typed observation must detect an incorrect dtype expectation"
        );
    } else {
        assert!(run.status.success(), "{source_text}: {run:?}");
    }
}

fn native_observation(dtype: &str, c_dtype: &str, nested: bool, empty: bool, wrong_dtype: bool) {
    let leaf = if dtype == "bool" {
        "true".to_string()
    } else {
        format!("cast(1, {dtype})")
    };
    let (ty, input, rank, first, count) = match (nested, empty) {
        (true, true) => (
            format!("List[List[{dtype}]]"),
            "[[], []]".to_string(),
            2,
            2,
            0,
        ),
        (false, true) => (format!("List[{dtype}]"), "[]".to_string(), 1, 0, 0),
        (true, false) => (
            format!("List[List[{dtype}]]"),
            format!("[[{leaf}], [{leaf}]]"),
            2,
            2,
            2,
        ),
        (false, false) => (format!("List[{dtype}]"), format!("[{leaf}]"), 1, 1, 1),
    };
    let shape = if nested {
        vec![first, if empty { 0 } else { 1 }]
    } else {
        vec![first]
    };
    run_native_observation(
        &format!("xs: {ty} = {input}\nout = to_tensor(xs)\n"),
        c_dtype,
        rank,
        &shape,
        count,
        wrong_dtype,
    );
}

#[test]
fn generated_c_empty_tensor_metadata_matches_every_checked_dtype() {
    for (dtype, c_dtype) in [
        ("f16", "CHELIS_DTYPE_F16"),
        ("bf16", "CHELIS_DTYPE_BF16"),
        ("f32", "CHELIS_DTYPE_F32"),
        ("f64", "CHELIS_DTYPE_F64"),
        ("i8", "CHELIS_DTYPE_I8"),
        ("i16", "CHELIS_DTYPE_I16"),
        ("i32", "CHELIS_DTYPE_I32"),
        ("i64", "CHELIS_DTYPE_I64"),
        ("bool", "CHELIS_DTYPE_BOOL"),
    ] {
        for nested in [false, true] {
            for empty in [true, false] {
                native_observation(dtype, c_dtype, nested, empty, false);
            }
        }
    }
}

#[test]
fn generated_c_typed_observation_rejects_an_incorrect_empty_dtype() {
    native_observation("f64", "CHELIS_DTYPE_F64", false, true, true);
}

#[test]
fn generated_c_contextual_empty_tensor_preserves_the_checked_dtype() {
    run_native_observation(
        "out: tensor[0, f64] = to_tensor([])\n",
        "CHELIS_DTYPE_F64",
        1,
        &[0],
        0,
        false,
    );
}

#[test]
fn generated_c_rejects_an_unresolved_empty_tensor_dtype() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("unresolved.ch");
    let output_dir = dir.path().join("generated");
    fs::write(&source, "out = to_tensor([])\n").unwrap();
    let output = Command::new(assert_cmd::cargo_bin!("chelis"))
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            source.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            output_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success(), "{output:?}");
    let message = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    // chelis#2518: nothing determines the empty list's element type, so the
    // checker rejects the declaration before code generation is reached.
    assert!(
        message.contains("`to_tensor` admits only some operand types")
            && message.contains("never determined within `out`"),
        "{message}"
    );
    assert!(!output_dir.join("unresolved.c").exists(), "{message}");
    assert!(!output_dir.join("unresolved.h").exists(), "{message}");
}

/// [05-OP-57]: an unconstrained empty payload cannot authorize f32. Since
/// chelis#2518 the checker rejects it before `eval` runs.
#[test]
fn unresolved_empty_list_dtype_rejects_in_eval() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unresolved.ch");
    fs::write(&path, "out = to_tensor([])\n").unwrap();
    let output = Command::new(assert_cmd::cargo_bin!("chelis"))
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let message = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        message.contains("`to_tensor` admits only some operand types"),
        "{message}"
    );
}
