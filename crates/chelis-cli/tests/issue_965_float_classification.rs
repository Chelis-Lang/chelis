//! [05-OP-20..22]: float classification preserves surface and stored width.
mod common;
use assert_cmd::Command;
use common::build_and_run;

fn evaluate(source: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("classification.ch");
    std::fs::write(&path, source).unwrap();
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file"])
        .arg(path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn scalar_and_tensor_classification_agree_in_eval_and_native_c() {
    for (dtype, subnormal) in [
        ("f16", "0.000000059604644775390625"),
        ("bf16", "9.183549615799121e-41"),
        ("f32", "1.401298464324817e-45"),
        ("f64", "5e-324"),
    ] {
        let source = format!(
            r#"
def nan_scalar(x: {dtype}) -> bool = is_nan(x)
def finite_scalar(x: {dtype}) -> bool = is_finite(x)
def infinite_scalar(x: {dtype}) -> bool = is_infinite(x)
def nan_tensor(x: tensor[8,{dtype}]) -> tensor[8,bool] = is_nan(x)
def finite_tensor(x: tensor[8,{dtype}]) -> tensor[8,bool] = is_finite(x)
def infinite_tensor(x: tensor[8,{dtype}]) -> tensor[8,bool] = is_infinite(x)
pos_inf = div(1.0{dtype},0.0{dtype})
neg_inf = div(-1.0{dtype},0.0{dtype})
nan = div(0.0{dtype},0.0{dtype})
x: tensor[8,{dtype}] = to_tensor([pos_inf,neg_inf,nan,0.0{dtype},-0.0{dtype},{subnormal}{dtype},1.0{dtype},-1.0{dtype}])
nan_values = nan_tensor(x)
finite_values = finite_tensor(x)
infinite_values = infinite_tensor(x)
nan_yes = nan_scalar(nan)
nan_no = nan_scalar(pos_inf)
finite_yes = finite_scalar({subnormal}{dtype})
finite_no = finite_scalar(neg_inf)
infinite_yes = infinite_scalar(neg_inf)
infinite_no = infinite_scalar(nan)
"#
        );
        for actual in [
            evaluate(&source),
            build_and_run(&source, &format!("classification_{dtype}")),
        ] {
            for expected in [
                "nan_values = tensor(shape=[8], data=[false, false, true, false, false, false, false, false])",
                "finite_values = tensor(shape=[8], data=[false, false, false, true, true, true, true, true])",
                "infinite_values = tensor(shape=[8], data=[true, true, false, false, false, false, false, false])",
                "nan_yes = true",
                "nan_no = false",
                "finite_yes = true",
                "finite_no = false",
                "infinite_yes = true",
                "infinite_no = false",
            ] {
                assert!(
                    actual.contains(expected),
                    "{dtype}: missing {expected}: {actual}"
                );
            }
        }
    }
}

#[test]
fn nonfloat_operands_wrong_shapes_and_wrong_arity_are_type_errors() {
    for operation in ["is_nan", "is_finite", "is_infinite"] {
        for source in [
            format!("result = {operation}(1i8)\n"),
            format!("result = {operation}(1i16)\n"),
            format!("result = {operation}(1i32)\n"),
            format!("result = {operation}(1i64)\n"),
            format!("f = {operation}\nresult = f(1i64)\n"),
            format!(
                "def bad(f: (&i32 -> bool), x: i32) -> bool = f(x)\nresult = bad({operation},1i32)\n"
            ),
            format!("result = {operation}(true)\n"),
            format!("result = {operation}(\"x\")\n"),
            format!("def f(x: tensor[2,f32]) -> tensor[3,bool] = {operation}(x)\n"),
            format!("def f(x: f32) -> f32 = {operation}(x)\n"),
            format!("def f(x: tensor[2,i64]) -> tensor[2,bool] = {operation}(x)\n"),
            format!("def f[p](x: p) -> bool = {operation}(x)\n"),
            format!("result = {operation}()\n"),
            format!("result = {operation}(1.0f32,2.0f32)\n"),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("rejected.ch");
            std::fs::write(&path, source).unwrap();
            Command::cargo_bin("chelis")
                .unwrap()
                .env("CHELIS_STYLE_GATE_DISABLE", "1")
                .args(["check", "--file"])
                .arg(path)
                .assert()
                .failure();
        }
    }
}

#[test]
fn raw_scalar_payload_classification_agrees_in_eval_and_native_c() {
    for (dtype, width, sign, exponent, quiet) in [
        ("f16", 2, 0x8000u64, 0x7c00u64, 0x0200u64),
        ("bf16", 2, 0x8000, 0x7f80, 0x0040),
        ("f32", 4, 0x80000000, 0x7f800000, 0x00400000),
        (
            "f64",
            8,
            0x8000000000000000,
            0x7ff0000000000000,
            0x0008000000000000,
        ),
    ] {
        let patterns = [
            0,
            sign,
            1,
            sign | 1,
            exponent - 1,
            sign | (exponent - 1),
            exponent,
            sign | exponent,
            exponent | 1,
            sign | exponent | 1,
            exponent | quiet,
            sign | exponent | quiet | 3,
        ];
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("bits.bin");
        std::fs::write(
            &data,
            patterns
                .iter()
                .flat_map(|bits| bits.to_le_bytes()[..width].to_vec())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let mut source = format!(
            "def main() -> unit ! {{ IO }} = {{\n  m = mmap_file({:?})\n",
            data.to_str().unwrap()
        );
        let mut expected = String::new();
        for i in 0..patterns.len() {
            source.push_str(&format!("  x{i} = tensor_to_scalar(reshape(mmap_tensor(m, {}i64, 1i64, {dtype}), []))\n  _ = print((is_nan(x{i}), is_finite(x{i}), is_infinite(x{i})))\n", i * width));
            expected.push_str(&format!("({}, {}, {})\n", i >= 8, i <= 5, i == 6 || i == 7));
        }
        source.push_str("  ()\n}\nrun = main()\n");
        for actual in [
            evaluate(&source),
            build_and_run(&source, &format!("scalar_bits_{dtype}")),
        ] {
            assert!(
                actual.starts_with(&expected),
                "{dtype}: expected {expected}, got {actual}"
            );
        }
    }
}

#[test]
fn predicates_compose_through_float_generics_aliases_shapes_and_ad() {
    let source = r#"
def finite[p: Float](x: p) -> bool = is_finite(x)
def masked(x: tensor[3,f32]) -> tensor[3,f32] = where(is_finite(x), x, to_tensor([0.0f32,0.0f32,0.0f32]))
def loss(x: tensor[3,f32]) -> f32 = tensor_to_scalar(sum(masked(x),0))
f = is_finite
alias_result = f(1.0f16)
generic_result = finite(1.0bf16)
x: tensor[3,f32] = to_tensor([1.0f32,-2.0f32,0.0f32])
derivative = grad(loss)(x)
def symbolic[n,p: Float](x: tensor[n,p]) -> tensor[n,bool] = is_finite(x)
shape_result = symbolic(x)
"#;
    for actual in [
        evaluate(source),
        build_and_run(source, "classification_composition"),
    ] {
        for expected in [
            "alias_result = true",
            "generic_result = true",
            "derivative = tensor(shape=[3], data=[1.0, 1.0, 1.0])",
            "shape_result = tensor(shape=[3], data=[true, true, true])",
        ] {
            assert!(actual.contains(expected), "missing {expected}: {actual}");
        }
    }
}

#[test]
fn device_targets_report_the_open_classification_issue() {
    for target in ["hip", "metal"] {
        for dtype in ["f16", "bf16", "f32", "f64"] {
            for predicate in ["is_nan", "is_finite", "is_infinite"] {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("classification.ch");
                std::fs::write(
                    &path,
                    format!("def main(x: tensor[2,{dtype}]) -> tensor[2,bool] = {predicate}(x)\n"),
                )
                .unwrap();
                let output = Command::cargo_bin("chelis")
                    .unwrap()
                    .env("CHELIS_STYLE_GATE_DISABLE", "1")
                    .args(["build", "--emit-c", "--target", target])
                    .arg(path)
                    .output()
                    .unwrap();
                assert!(!output.status.success());
                let stderr = String::from_utf8_lossy(&output.stderr);
                assert!(
                    stderr.contains(if target == "metal" && dtype == "f64" {
                        "[04-TGT-1]"
                    } else {
                        "965"
                    }),
                    "{target} {dtype} {predicate}: {stderr}"
                );
            }
        }
    }
}

#[test]
fn predicate_values_select_exact_scalar_and_tensor_callback_signatures() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        let source = format!(
            r#"
def apply_scalar(f: (&{dtype} -> bool), x: {dtype}) -> bool = f(x)
def apply_tensor(f: (&tensor[2,{dtype}] -> tensor[2,bool]), x: tensor[2,{dtype}]) -> tensor[2,bool] = f(x)
x: tensor[2,{dtype}] = to_tensor([1.0{dtype},div(0.0{dtype},0.0{dtype})])
scalar_nan = apply_scalar(is_nan,div(0.0{dtype},0.0{dtype}))
scalar_finite = apply_scalar(is_finite,1.0{dtype})
scalar_infinite = apply_scalar(is_infinite,div(-1.0{dtype},0.0{dtype}))
tensor_nan = apply_tensor(is_nan,x)
tensor_finite = apply_tensor(is_finite,x)
tensor_infinite = apply_tensor(is_infinite,x)
"#
        );
        for actual in [
            evaluate(&source),
            build_and_run(&source, &format!("predicate_values_{dtype}")),
        ] {
            for expected in [
                "scalar_nan = true",
                "scalar_finite = true",
                "scalar_infinite = true",
                "tensor_nan = tensor(shape=[2], data=[false, true])",
                "tensor_finite = tensor(shape=[2], data=[true, false])",
                "tensor_infinite = tensor(shape=[2], data=[false, false])",
            ] {
                assert!(
                    actual.contains(expected),
                    "{dtype}: missing {expected}: {actual}"
                );
            }
        }
    }
}
