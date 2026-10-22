//! [05-OP-46] producers retain their actual carrier before [05-OP-20..22]
//! classification, including rank-polymorphic exported tuple wrappers.
mod common;

use assert_cmd::Command;

const PRODUCERS: &[&str] = &[
    "abs", "cos", "tan", "atan", "floor", "ceil", "round", "recip",
];

fn evaluate(source: &str) -> (String, serde_json::Value) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("classification.ch");
    common::write_file(&path, source);
    let checked = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("check")
        .arg(&path)
        .output()
        .unwrap();
    let report: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert!(checked.status.success(), "{report}");
    assert_eq!(report["score"], 1.0, "{report}");
    assert_eq!(report["errors"], serde_json::json!([]), "{report}");
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--json", "--file"])
        .arg(path)
        .output()
        .unwrap();
    assert!(
        json.status.success(),
        "{}",
        String::from_utf8_lossy(&json.stderr)
    );
    (
        String::from_utf8(output.stdout).unwrap(),
        serde_json::from_slice(&json.stdout).unwrap(),
    )
}

#[test]
fn unary_producers_classify_through_runtime_rank_rows_and_dtype_binders() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        let mut source = format!(
            "finite = is_finite\nnan = is_nan\ninfinite = is_infinite\n\
             def scalar(x: {dtype}) -> bool = is_finite(abs(x))\n\
             def fixed(x: tensor[4,{dtype}]) -> tensor[4,bool] = finite(abs(x))\n\
             def bounded[r,p: Float](x: tensor[..r,p]) -> tensor[..r,bool] = {{\n\
               magnitude = abs(x)\n is_finite(magnitude)\n }}\n\
             scalar_result = scalar(-1.0{dtype})\n\
             fixed_result = fixed(to_tensor([-1.0{dtype},0.0{dtype},div(1.0{dtype},0.0{dtype}),div(0.0{dtype},0.0{dtype})]))\n\
             bounded_result = bounded(to_tensor([-1.0{dtype},0.0{dtype},div(1.0{dtype},0.0{dtype}),div(0.0{dtype},0.0{dtype})]))\n\
             rank_zero_result = bounded(scalar_to_tensor(-1.0{dtype}))\n\
             matrix_result = bounded(reshape(to_tensor([-1.0{dtype},0.0{dtype},div(1.0{dtype},0.0{dtype}),div(0.0{dtype},0.0{dtype})]),[2i64,2i64]))\n\
             empty: tensor[0,{dtype}] = to_tensor([])\n\
             empty_result = bounded(empty)\n"
        );
        for producer in PRODUCERS {
            source.push_str(&format!(
                "def classify_{producer}[r](x: tensor[..r,{dtype}]) -> (tensor[..r,bool],tensor[..r,bool],tensor[..r,bool]) = {{\n\
                 produced = {producer}(x)\n\
                 (is_nan(produced),is_finite(produced),is_infinite(produced))\n }}\n\
                 def wrapped_{producer}(args: (tensor[4,{dtype}],)) -> (tensor[4,bool],tensor[4,bool],tensor[4,bool]) = classify_{producer}(args.0)\n\
                 result_{producer} = wrapped_{producer}((to_tensor([-1.0{dtype},0.0{dtype},div(1.0{dtype},0.0{dtype}),div(0.0{dtype},0.0{dtype})]),))\n"
            ));
        }
        let (evaluated, json) = evaluate(&source);
        assert!(evaluated.contains("scalar_result = true"), "{evaluated}");
        for name in ["fixed_result", "bounded_result"] {
            assert!(
                evaluated.contains(&format!(
                    "{name} = tensor(shape=[4], data=[true, true, false, false])"
                )),
                "{evaluated}"
            );
        }
        for expected in [
            "rank_zero_result = true",
            "matrix_result = tensor(shape=[2, 2], data=[true, true, false, false])",
            "empty_result = tensor(shape=[0], data=[])",
        ] {
            assert!(evaluated.contains(expected), "{evaluated}");
        }
        for (root, shape) in [
            ("rank_zero_result", serde_json::json!([])),
            ("matrix_result", serde_json::json!([2, 2])),
            ("empty_result", serde_json::json!([0])),
        ] {
            let value = &json["roots"]
                .as_array()
                .unwrap()
                .iter()
                .find(|value| value["name"] == root)
                .unwrap()["value"];
            assert_eq!(value["type"], "tensor", "{root}: {value}");
            assert_eq!(value["value"]["shape"], shape, "{root}: {value}");
            assert_eq!(value["value"]["data"]["dtype"], "bool", "{root}: {value}");
        }
        assert_eq!(
            common::build_and_run(&source, &format!("producers_{dtype}")),
            evaluated
        );
    }
}

#[test]
fn producer_classification_rejects_nonfloat_operands_and_result_shape_invention() {
    for source in [
        "bad = is_finite(abs(to_tensor([1i32,2i32])))\n",
        "bad = is_nan(abs(true))\n",
        "bad = is_infinite(cos(1i64))\n",
        "def bad[r](x: tensor[..r,f32]) -> bool = is_finite(abs(x))\n",
        "def bad(x: tensor[2,f32]) -> tensor[3,bool] = is_finite(abs(x))\n",
        "def bad[r](x: tensor[..r,f32]) -> tensor[..r,f32] = is_finite(abs(x))\n",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("invalid.ch");
        common::write_file(&path, source);
        let output = Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .arg("check")
            .arg(path)
            .output()
            .unwrap();
        let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert!(
            !output.status.success(),
            "invalid source accepted: {source}\n{report}"
        );
        assert!(!report["errors"].as_array().unwrap().is_empty(), "{report}");
    }
}

#[test]
fn imported_exported_tuple_wrapper_preserves_producer_type() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    common::write_file(
        &root.join("reef.toml"),
        &format!(
            "schema = \"1\"\n[package]\nname = \"classification\"\nversion = \"0.1.0\"\ncompiler = \"={}\"\nmodule_prefix = \"Probe\"\n",
            common::COMPILER_VERSION,
        ),
    );
    common::write_file(
        &root.join("src/kernels.ch"),
        "module Probe.Kernels\nexport (wrapped)\n\
         def classify[r,p: Float](x: tensor[..r,p]) -> tensor[..r,bool] = {\n\
           magnitude = abs(x)\n is_finite(magnitude)\n }\n\
         def wrapped(args: (tensor[3,f32],)) -> (tensor[3,bool],) = (classify(args.0),)\n",
    );
    let path = root.join("src/main.ch");
    common::write_file(
        &path,
        "module Probe.Main\nimport Probe.Kernels (wrapped)\n\
         result = wrapped((to_tensor([0.0f32,-1.0f32,div(1.0f32,0.0f32)]),))\n",
    );
    let evaluated = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        evaluated.status.success(),
        "{}",
        String::from_utf8_lossy(&evaluated.stderr)
    );
    let evaluated = String::from_utf8(evaluated.stdout).unwrap();
    assert!(
        evaluated.contains("data=[true, true, false]"),
        "{evaluated}"
    );
    let output = root.join("native");
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("build")
        .arg(&path)
        .args(["--target", "c", "--output"])
        .arg(&output)
        .assert()
        .success();
    let native = std::process::Command::new(output.join("main"))
        .output()
        .unwrap();
    assert!(
        native.status.success(),
        "{}",
        String::from_utf8_lossy(&native.stderr)
    );
    assert_eq!(String::from_utf8(native.stdout).unwrap(), evaluated);
}
