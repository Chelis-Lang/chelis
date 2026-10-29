//! Exact stored-word qualification for concat movement and arithmetic sqrt.

mod common;

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::process::Command as NativeCommand;
use tempfile::tempdir;

struct Case {
    dtype: &'static str,
    c_dtype: &'static str,
    width: usize,
    words: [u64; 10],
    canonical: u64,
    two: u64,
}

const CASES: [Case; 4] = [
    Case {
        dtype: "f16",
        c_dtype: "CHELIS_DTYPE_F16",
        width: 2,
        words: [
            0xfc01, 0xfe11, 0x7c01, 0x7e11, 0xbc00, 0xfc00, 0, 0x8000, 0x7c00, 0x4400,
        ],
        canonical: 0x7e00,
        two: 0x4000,
    },
    Case {
        dtype: "bf16",
        c_dtype: "CHELIS_DTYPE_BF16",
        width: 2,
        words: [
            0xff81, 0xffc1, 0x7f81, 0x7fc1, 0xbf80, 0xff80, 0, 0x8000, 0x7f80, 0x4080,
        ],
        canonical: 0x7fc0,
        two: 0x4000,
    },
    Case {
        dtype: "f32",
        c_dtype: "CHELIS_DTYPE_F32",
        width: 4,
        words: [
            0xff80_0001,
            0xffc0_0011,
            0x7f80_0001,
            0x7fc0_0011,
            0xbf80_0000,
            0xff80_0000,
            0,
            0x8000_0000,
            0x7f80_0000,
            0x4080_0000,
        ],
        canonical: 0x7fc0_0000,
        two: 0x4000_0000,
    },
    Case {
        dtype: "f64",
        c_dtype: "CHELIS_DTYPE_F64",
        width: 8,
        words: [
            0xfff0_0000_0000_0001,
            0xfff8_0000_0000_0011,
            0x7ff0_0000_0000_0001,
            0x7ff8_0000_0000_0011,
            0xbff0_0000_0000_0000,
            0xfff0_0000_0000_0000,
            0,
            0x8000_0000_0000_0000,
            0x7ff0_0000_0000_0000,
            0x4010_0000_0000_0000,
        ],
        canonical: 0x7ff8_0000_0000_0000,
        two: 0x4000_0000_0000_0000,
    },
];

fn canonical_source(path: &Path, source: &str) {
    fs::write(path, source).unwrap();
    Command::cargo_bin("chelis")
        .unwrap()
        .args(["fmt", "--inplace"])
        .arg(path)
        .assert()
        .success();
}

fn eval_words(
    directory: &Path,
    source: &str,
    case: &Case,
    shape: &[usize],
    output_shape: &[usize],
) -> Vec<u64> {
    let raw = directory.join("input.bin");
    let bytes: Vec<_> = case
        .words
        .iter()
        .flat_map(|word| word.to_le_bytes()[..case.width].to_vec())
        .collect();
    fs::write(&raw, bytes).unwrap();
    let dimensions = shape
        .iter()
        .map(|n| format!("{n}i64"))
        .collect::<Vec<_>>()
        .join(", ");
    let path = directory.join("eval.ch");
    canonical_source(
        &path,
        &format!(
            "{source}out = {{\n  mapped = mmap_file({raw})\n  flat = mmap_tensor(mapped, 0i64, 10i64, {dtype})\n  x = reshape(flat, [{dimensions}])\n  compute(x)\n}}\n",
            raw = serde_json::to_string(raw.to_str().unwrap()).unwrap(),
            dtype = case.dtype,
        ),
    );
    let checked = Command::cargo_bin("chelis")
        .unwrap()
        .arg("check")
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        checked.status.success(),
        "{}",
        String::from_utf8_lossy(&checked.stderr)
    );
    let report: Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert_eq!(report["score"].as_f64(), Some(1.0));
    assert_eq!(report["errors"], serde_json::json!([]));
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .args(["eval", "--file"])
        .arg(&path)
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let document: Value = serde_json::from_slice(&output.stdout).unwrap();
    let result = &document["roots"]
        .as_array()
        .unwrap()
        .iter()
        .find(|root| root["name"] == "out")
        .unwrap()["value"]["value"];
    assert_eq!(result["data"]["dtype"], case.dtype);
    assert_eq!(result["shape"], serde_json::json!(output_shape));
    result["data"]["bits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|word| u64::from_str_radix(word.as_str().unwrap(), 16).unwrap())
        .collect()
}

fn native_words(
    directory: &Path,
    source: &str,
    case: &Case,
    input_shape: &[usize],
    output_shape: &[usize],
) -> Vec<u64> {
    let path = directory.join("compute.ch");
    canonical_source(&path, source);
    let generated = directory.join("generated");
    Command::cargo_bin("chelis")
        .unwrap()
        .args(["build", "--emit-c"])
        .arg(&path)
        .args(["--target", "c", "--output"])
        .arg(&generated)
        .assert()
        .success();
    let numbers = |values: &[usize]| {
        values
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    };
    let words = case
        .words
        .iter()
        .map(|word| format!("UINT64_C(0x{word:016x})"))
        .collect::<Vec<_>>()
        .join(", ");
    let header = fs::read_to_string(generated.join("compute.h")).unwrap();
    let symbol = common::authored_c_symbol("compute");
    let invoke = if header.lines().any(|line| line.starts_with("void compute(")) {
        "chelis_tensor *inputs[1] = {x};\nchelis_tensor *outputs[1] = {0};\ncompute(inputs, 1, outputs, 1);\nchelis_tensor *out = outputs[0];".to_string()
    } else {
        assert!(
            header.contains(&format!("{symbol}(")),
            "no declared tensor entry: {header}"
        );
        format!("chelis_tensor *out = {symbol}(x);")
    };
    let driver = directory.join("driver.c");
    fs::write(&driver, format!(r#"
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include "chelis_runtime.h"
#include "compute.h"
int main(void) {{
    uint64_t words[] = {{ {words} }};
    uint64_t aligned[10] = {{0}};
    unsigned char *input = (unsigned char *)aligned;
    for (int i = 0; i < 10; ++i)
        for (int b = 0; b < {width}; ++b)
            input[i * {width} + b] = (unsigned char)(words[i] >> (8 * b));
    int64_t shape[] = {{ {input_dimensions} }};
    int64_t expected[] = {{ {output_dimensions} }};
    chelis_tensor *x = chelis_tensor_entry_borrow({input_rank}, shape, {dtype}, input, 10 * {width});
    {invoke}
    chelis_read_view view = chelis_tensor_read_view(out);
    if (view.dtype != {dtype} || view.count != {count}) return 2;
    if (chelis_tensor_rank(out) != {output_rank}) return 3;
    for (int axis = 0; axis < {output_rank}; ++axis)
        if (chelis_tensor_shape(out, axis) != expected[axis]) return 4;
    const unsigned char *data = (const unsigned char *)view.data;
    for (int i = 0; i < {count}; ++i) {{
        uint64_t bits = 0;
        for (int b = 0; b < {width}; ++b)
            bits |= (uint64_t)data[i * {width} + b] << (8 * b);
        printf("%" PRIx64 "\n", bits);
    }}
    chelis_tensor_release(out);
    chelis_tensor_release(x);
    return 0;
}}
"#,
        width = case.width, dtype = case.c_dtype,
        input_rank = input_shape.len(), input_dimensions = numbers(input_shape),
        output_rank = output_shape.len(), output_dimensions = numbers(output_shape),
        count = output_shape.iter().product::<usize>(),
    )).unwrap();
    let executable = directory.join("native_bits");
    let compiled = NativeCommand::new(chelis_backend_c::toolchain::c_compiler())
        .args(["-O2", "-std=c11", "-I"])
        .arg(&generated)
        .arg(generated.join("compute.c"))
        .arg(driver)
        .arg(generated.join("libchelis_runtime.a"))
        .args(["-lm", "-lpthread", "-ldl", "-o"])
        .arg(&executable)
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let output = NativeCommand::new(executable).output().unwrap();
    assert!(
        output.status.success(),
        "native status {:?}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| u64::from_str_radix(line, 16).unwrap())
        .collect()
}

#[test]
fn ingress_selection_concat_and_arithmetic_match_exact_words_on_eval_and_c() {
    for case in &CASES {
        for operation in [
            "identity",
            "relu",
            "sqrt",
            "concat",
            "vmap_concat",
            "vmap_runtime_concat",
        ] {
            let (source, input_shape, output_shape, expected) = match operation {
                "vmap_concat" | "vmap_runtime_concat" => (
                    format!(
                        "def item(x: tensor[{input}, {p}]) -> tensor[{output}, {p}] = concat([x, x], 0i32)\ndef compute(x: tensor[2, {input}, {p}]) -> tensor[2, {output}, {p}] = (vmap(item))(x)\n",
                        p = case.dtype,
                        input = if operation == "vmap_runtime_concat" {
                            "*"
                        } else {
                            "5"
                        },
                        output = if operation == "vmap_runtime_concat" {
                            "*"
                        } else {
                            "10"
                        }
                    ),
                    vec![2, 5],
                    vec![2, 10],
                    [case.words[..5].repeat(2), case.words[5..].repeat(2)].concat(),
                ),
                _ => {
                    let body = match operation {
                        "identity" => "x",
                        "concat" => "concat([x, x], 0i32)",
                        "relu" => "relu(x)",
                        "sqrt" => "sqrt(x)",
                        _ => unreachable!(),
                    };
                    let size = if operation == "concat" { 20 } else { 10 };
                    let mut expected = case.words.to_vec();
                    if operation == "sqrt" {
                        expected[..6].fill(case.canonical);
                        expected[9] = case.two;
                    } else if operation == "relu" {
                        expected[4..6].fill(0);
                    } else if operation == "concat" {
                        expected = expected.repeat(2);
                    }
                    (
                        format!(
                            "def compute(x: tensor[10, {p}]) -> tensor[{size}, {p}] = {body}\n",
                            p = case.dtype
                        ),
                        vec![10],
                        vec![size],
                        expected,
                    )
                }
            };
            let directory = tempdir().unwrap();
            assert_eq!(
                eval_words(directory.path(), &source, case, &input_shape, &output_shape),
                expected,
                "Eval {} {operation}",
                case.dtype
            );
            assert_eq!(
                native_words(directory.path(), &source, case, &input_shape, &output_shape),
                expected,
                "C {} {operation}",
                case.dtype
            );
        }
    }
}

#[test]
fn original_concat_reciprocal_preserves_negative_zero_on_eval_and_c() {
    let source = "def main() = {\n x = to_tensor([cast(-0.0, f32), cast(-1.0, f32)])\n y = concat([copy(x), x], 0i32)\n div(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]), y)\n}\n";
    let directory = tempdir().unwrap();
    let path = directory.path().join("negative_zero.ch");
    canonical_source(&path, source);
    let eval = Command::cargo_bin("chelis")
        .unwrap()
        .args(["eval", "--file"])
        .arg(path)
        .output()
        .unwrap();
    assert!(
        eval.status.success(),
        "{}",
        String::from_utf8_lossy(&eval.stderr)
    );
    assert!(
        String::from_utf8(eval.stdout)
            .unwrap()
            .contains("data=[-inf, -1.0, -inf, -1.0]")
    );
    let native = common::build_and_run(source, "negative_zero");
    assert!(native.contains("data=[-inf, -1.0, -inf, -1.0]"), "{native}");
}
