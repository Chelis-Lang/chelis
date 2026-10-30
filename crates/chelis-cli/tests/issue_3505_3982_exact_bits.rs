//! Exact stored-word qualification for concat movement and arithmetic sqrt.

mod common;

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
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
    let executable = compile_native_driver(directory, &generated, &driver);
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

fn compile_native_driver(directory: &Path, generated: &Path, driver: &Path) -> PathBuf {
    let executable = directory.join("native_bits");
    let compiled = NativeCommand::new(chelis_backend_c::toolchain::c_compiler())
        .args(["-O2", "-std=c11", "-I"])
        .arg(generated)
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
    executable
}

#[test]
fn ingress_selection_concat_and_arithmetic_match_exact_words_on_eval_and_c() {
    for case in &CASES {
        for operation in [
            "identity",
            "relu",
            "sqrt",
            "floor",
            "ceil",
            "round",
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
                        "floor" => "floor(x)",
                        "ceil" => "ceil(x)",
                        "round" => "round(x)",
                        _ => unreachable!(),
                    };
                    let size = if operation == "concat" { 20 } else { 10 };
                    let mut expected = case.words.to_vec();
                    if operation == "sqrt" {
                        expected[..6].fill(case.canonical);
                        expected[9] = case.two;
                    } else if matches!(operation, "floor" | "ceil" | "round") {
                        expected[..4].fill(case.canonical);
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

#[test]
fn metal_concat_reaches_the_typed_nonnumeric_gate() {
    let source = "def compute(x: tensor[2, f32]) -> tensor[4, f32] = concat([x, x], 0i32)\n";
    let directory = tempdir().unwrap();
    let path = directory.path().join("concat_metal.ch");
    canonical_source(&path, source);
    let checked = Command::cargo_bin("chelis")
        .unwrap()
        .arg("check")
        .arg(&path)
        .output()
        .unwrap();
    assert!(checked.status.success());
    let report: Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert_eq!(report["score"].as_f64(), Some(1.0), "{report}");
    assert_eq!(report["errors"], serde_json::json!([]), "{report}");
    let emitted = Command::cargo_bin("chelis")
        .unwrap()
        .arg("build")
        .arg(&path)
        .args(["--target", "metal", "--output"])
        .arg(directory.path().join("metal"))
        .output()
        .unwrap();
    assert_eq!(emitted.status.code(), Some(1));
    assert!(emitted.stdout.is_empty());
    let stderr = String::from_utf8_lossy(&emitted.stderr);
    assert!(stderr.contains("chelis#2266"), "{stderr}");
    assert!(stderr.contains("codegen:metal"), "{stderr}");
    let native = common::build_and_run(
        &format!(
            "{source}out = div(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]), compute(to_tensor([cast(-0.0, f32), cast(-1.0, f32)])))\n"
        ),
        "concat_c_admission",
    );
    assert!(native.contains("data=[-inf, -1.0, -inf, -1.0]"), "{native}");
}

#[test]
fn concat_checks_nonconcat_extents_independently_of_selected_elements() {
    let tensor = |shape: [usize; 2]| {
        let count = shape.iter().product::<usize>();
        let data = if count == 0 {
            "to_tensor([], f32)".to_string()
        } else {
            format!("to_tensor([{}])", vec!["1.0f32"; count].join(", "))
        };
        format!("reshape({data}, [{}i64, {}i64])", shape[0], shape[1])
    };
    for fixed_axis in [false, true] {
        for (first, second, matching) in [
            ([1, 2], [0, 3], false),
            ([0, 2], [0, 3], false),
            ([1, 0], [1, 3], false),
            ([0, 2], [1, 3], false),
            ([1, 2], [0, 2], true),
            ([1, 0], [1, 0], true),
        ] {
            let first_width = if fixed_axis {
                first[0].to_string()
            } else {
                "*".into()
            };
            let second_width = if fixed_axis {
                second[0].to_string()
            } else {
                "*".into()
            };
            for consumer in ["grad", "discarded", "untaken"] {
                let (parameters, body, call) = match consumer {
                    "grad" => (
                        String::new(),
                        "tensor_to_scalar(sum(sum(concat([a, b], 0i32), 1i32), 0i32))",
                        "grad(loss, wrt=a)(a0(), b0())",
                    ),
                    "discarded" => (
                        String::new(),
                        "{ _ = concat([a, b], 0i32)\n  1.0f32 }",
                        "loss(a0(), b0())",
                    ),
                    "untaken" => (
                        "flag: bool, ".to_string(),
                        "if flag then tensor_to_scalar(sum(sum(concat([a, b], 0i32), 1i32), 0i32)) else tensor_to_scalar(sum(sum(a, 1i32), 0i32))",
                        "grad(loss, wrt=a)(false, a0(), b0())",
                    ),
                    _ => unreachable!(),
                };
                let source = format!(
                    "def loss({parameters}a: tensor[{first_width}, *, f32], b: tensor[{second_width}, *, f32]) -> f32 = {body}\ndef a0() -> tensor[{}, {}, f32] = {}\ndef b0() -> tensor[{}, {}, f32] = {}\nout = {call}\n",
                    first[0],
                    first[1],
                    tensor(first),
                    second[0],
                    second[1],
                    tensor(second),
                );
                let directory = tempdir().unwrap();
                let path = directory.path().join("concat_metadata.ch");
                canonical_source(&path, &source);
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
                assert_eq!(report["score"].as_f64(), Some(1.0), "{report}");
                assert_eq!(report["errors"], serde_json::json!([]), "{report}");
                assert_eq!(report["untyped_nodes"], 0, "{report}");
                assert_eq!(
                    report["unresolved_names"],
                    serde_json::json!([]),
                    "{report}"
                );
                let evaluated = Command::cargo_bin("chelis")
                    .unwrap()
                    .args(["eval", "--file"])
                    .arg(&path)
                    .arg("--json")
                    .output()
                    .unwrap();
                let context = format!(
                    "{consumer}, fixed={fixed_axis}, {first:?}/{second:?}: {}",
                    String::from_utf8_lossy(&evaluated.stderr)
                );
                if matching || consumer == "untaken" {
                    assert!(evaluated.status.success(), "{context}");
                } else {
                    assert_eq!(
                        evaluated.status.code(),
                        Some(1),
                        "{context}; unexpected success: {}",
                        String::from_utf8_lossy(&evaluated.stdout)
                    );
                    assert!(
                        String::from_utf8_lossy(&evaluated.stderr).contains("numeric trap: domain"),
                        "{context}"
                    );
                }
            }
        }
    }
}

#[test]
fn native_concat_checks_empty_and_zero_bystander_metadata() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("compute.ch");
    canonical_source(
        &path,
        "def loss(a: tensor[*, *, f32], b: tensor[*, *, f32]) -> f32 = tensor_to_scalar(sum(sum(concat([a, b], 0i32), 1i32), 0i32))\ndef compute(a: tensor[*, *, f32], b: tensor[*, *, f32]) -> tensor[*, *, f32] = grad(loss, wrt=a)(a, b)\n",
    );
    let generated = directory.path().join("generated");
    Command::cargo_bin("chelis")
        .unwrap()
        .args(["build", "--emit-c"])
        .arg(&path)
        .args(["--target", "c", "--output"])
        .arg(&generated)
        .assert()
        .success();
    let header = fs::read_to_string(generated.join("compute.h")).unwrap();
    let symbol = common::authored_c_symbol("compute");
    let invoke = if header.lines().any(|line| line.starts_with("void compute(")) {
        "chelis_tensor *inputs[2] = {a, b};\nchelis_tensor *outputs[1] = {0};\ncompute(inputs, 2, outputs, 1);\nchelis_tensor *out = outputs[0];".to_string()
    } else {
        assert!(
            header.contains(&format!("{symbol}(")),
            "no declared tensor entry: {header}"
        );
        format!("chelis_tensor *out = {symbol}(a, b);")
    };
    let driver = directory.path().join("metadata_driver.c");
    fs::write(&driver, format!(r#"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "chelis_runtime.h"
#include "compute.h"
int main(int argc, char **argv) {{
    if (argc != 5) return 2;
    int64_t ashape[2] = {{strtoll(argv[1], 0, 10), strtoll(argv[2], 0, 10)}};
    int64_t bshape[2] = {{strtoll(argv[3], 0, 10), strtoll(argv[4], 0, 10)}};
    float av[16], bv[16];
    for (int i = 0; i < 16; ++i) {{ av[i] = 1.0f; bv[i] = 1.0f; }}
    int64_t count = ashape[0] * ashape[1];
    chelis_tensor *a = chelis_tensor_entry_borrow(2, ashape, CHELIS_DTYPE_F32, av, count * 4);
    chelis_tensor *b = chelis_tensor_entry_borrow(2, bshape, CHELIS_DTYPE_F32, bv, bshape[0] * bshape[1] * 4);
    {invoke}
    chelis_read_view view = chelis_tensor_read_view(out);
    if (view.dtype != CHELIS_DTYPE_F32 || view.count != count || chelis_tensor_rank(out) != 2) return 3;
    if (chelis_tensor_shape(out, 0) != ashape[0] || chelis_tensor_shape(out, 1) != ashape[1]) return 4;
    const float *values = (const float *)view.data;
    for (int64_t i = 0; i < count; ++i) if (values[i] != 1.0f) return 5;
    chelis_tensor_release(out);
    chelis_tensor_release(a);
    chelis_tensor_release(b);
    puts("ok");
    return 0;
}}
"#)).unwrap();
    let executable = compile_native_driver(directory.path(), &generated, &driver);
    for (first, second, matching) in [
        ([1, 2], [0, 3], false),
        ([0, 2], [0, 3], false),
        ([1, 0], [1, 3], false),
        ([0, 2], [1, 3], false),
        ([1, 2], [0, 2], true),
        ([1, 0], [1, 0], true),
    ] {
        let output = NativeCommand::new(&executable)
            .args(first.iter().chain(second.iter()).map(ToString::to_string))
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(
            output.status.code(),
            Some(if matching { 0 } else { 1 }),
            "{first:?}/{second:?}: {stderr}"
        );
        if matching {
            assert_eq!(output.stdout, b"ok\n");
        } else {
            assert!(
                stderr.contains("numeric trap: domain in concat at i64"),
                "{stderr}"
            );
        }
    }
}

#[test]
fn discarded_concat_uses_original_metadata_in_untaken_grad_branches() {
    for (rank, axis) in [(2, 0), (3, -2)] {
        for parts in 2..=4 {
            let names = &['a', 'b', 'c', 'd'][..parts];
            let dimensions = vec!["*"; rank].join(", ");
            let parameters = names
                .iter()
                .map(|name| format!("{name}: tensor[{dimensions}, f32]"))
                .collect::<Vec<_>>()
                .join(", ");
            let arguments = names
                .iter()
                .map(char::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            let mut total = "a".to_string();
            for _ in 0..rank {
                total = format!("sum({total}, 0i32)");
            }
            let total = format!("tensor_to_scalar({total})");
            let source = format!(
                "def loss(flag: bool, {parameters}) -> f32 = if flag then {{\n _ = concat([{arguments}], {axis}i32)\n {total}\n}} else {total}\ndef compute(flag: bool, {parameters}) -> tensor[{dimensions}, f32] = grad(loss, wrt=a)(flag, {arguments})\n"
            );
            let directory = tempdir().unwrap();
            let path = directory.path().join("compute.ch");
            canonical_source(&path, &source);
            let generated = directory.path().join("generated");
            Command::cargo_bin("chelis")
                .unwrap()
                .args(["build", "--emit-c"])
                .arg(&path)
                .args(["--target", "c", "--output"])
                .arg(&generated)
                .assert()
                .success();
            let symbol = common::authored_c_symbol("compute");
            let header = fs::read_to_string(generated.join("compute.h")).unwrap();
            assert!(header.contains(&format!("{symbol}(")), "{header}");
            let c_arguments = (0..parts)
                .map(|index| format!("inputs[{index}]"))
                .collect::<Vec<_>>()
                .join(", ");
            let driver = directory.path().join("activation_driver.c");
            fs::write(&driver, format!(r#"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "chelis_runtime.h"
#include "compute.h"
int main(int argc, char **argv) {{
    if (argc != 3) return 2;
    int kind = atoi(argv[2]);
    bool flag = atoi(argv[1]) != 0;
    int64_t shapes[{parts}][{rank}], counts[{parts}];
    float values[{parts}][16] = {{0}};
    chelis_tensor *inputs[{parts}];
    for (int part = 0; part < {parts}; ++part) {{
        for (int axis = 0; axis < {rank}; ++axis) shapes[part][axis] = 1;
        shapes[part][{rank} - 2] = part == 0 ? 1 : 2;
        shapes[part][{rank} - 1] = part == 1 && kind >= 2 ? 3 : 2;
        if (part == 1 && kind % 2 == 1) shapes[part][{rank} - 2] = 0;
        counts[part] = 1;
        for (int axis = 0; axis < {rank}; ++axis) counts[part] *= shapes[part][axis];
        inputs[part] = chelis_tensor_entry_borrow({rank}, shapes[part], CHELIS_DTYPE_F32, values[part], counts[part] * 4);
    }}
    chelis_tensor *out = {symbol}(flag, {c_arguments});
    chelis_read_view view = chelis_tensor_read_view(out);
    if (view.dtype != CHELIS_DTYPE_F32 || view.count != counts[0] || chelis_tensor_rank(out) != {rank}) return 3;
    for (int axis = 0; axis < {rank}; ++axis)
        if (chelis_tensor_shape(out, axis) != shapes[0][axis]) return 4;
    const float *data = (const float *)view.data;
    for (int64_t i = 0; i < view.count; ++i) if (data[i] != 1.0f) return 5;
    chelis_tensor_release(out);
    for (int part = 0; part < {parts}; ++part) chelis_tensor_release(inputs[part]);
    puts("ok");
    return 0;
}}
"#)).unwrap();
            let executable = compile_native_driver(directory.path(), &generated, &driver);
            for kind in 0..4 {
                for flag in [false, true] {
                    let should_succeed = !flag || kind < 2;
                    let context = format!("rank={rank}, parts={parts}, kind={kind}, flag={flag}");
                    let native = NativeCommand::new(&executable)
                        .args([u8::from(flag).to_string(), kind.to_string()])
                        .output()
                        .unwrap();
                    let native_stderr = String::from_utf8_lossy(&native.stderr);
                    assert_eq!(
                        native.status.code(),
                        Some(if should_succeed { 0 } else { 1 }),
                        "{context}: {native_stderr}"
                    );
                    if should_succeed {
                        assert_eq!(native.stdout, b"ok\n", "{context}");
                    } else {
                        assert!(
                            native_stderr.contains("numeric trap: domain in concat at i64"),
                            "{context}: {native_stderr}"
                        );
                    }
                    let arguments = (0..parts)
                        .map(|part| {
                            let mut shape = vec![1; rank];
                            shape[rank - 2] = if part == 0 { 1 } else { 2 };
                            shape[rank - 1] = if part == 1 && kind >= 2 { 3 } else { 2 };
                            if part == 1 && kind % 2 == 1 {
                                shape[rank - 2] = 0;
                            }
                            let count = shape.iter().product::<usize>();
                            let flat = if count == 0 {
                                "to_tensor([], f32)".to_string()
                            } else {
                                format!("to_tensor([{}])", vec!["0.0f32"; count].join(", "))
                            };
                            let shape = shape
                                .iter()
                                .map(|size| format!("{size}i64"))
                                .collect::<Vec<_>>()
                                .join(", ");
                            format!("reshape({flat}, [{shape}])")
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    let path = directory.path().join("evaluate.ch");
                    canonical_source(
                        &path,
                        &format!("{source}out = compute({flag}, {arguments})\n"),
                    );
                    let evaluated = Command::cargo_bin("chelis")
                        .unwrap()
                        .args(["eval", "--file"])
                        .arg(path)
                        .output()
                        .unwrap();
                    assert_eq!(
                        evaluated.status.success(),
                        should_succeed,
                        "{context}: {}",
                        String::from_utf8_lossy(&evaluated.stderr)
                    );
                    if should_succeed {
                        let text = String::from_utf8(evaluated.stdout).unwrap();
                        assert_eq!(
                            common::parse_tensor_data(&text, "out"),
                            [1.0, 1.0],
                            "{context}: {text}"
                        );
                    } else {
                        assert!(
                            String::from_utf8_lossy(&evaluated.stderr)
                                .contains("numeric trap: domain in concat at i64"),
                            "{context}"
                        );
                    }
                }
            }
        }
    }
}
