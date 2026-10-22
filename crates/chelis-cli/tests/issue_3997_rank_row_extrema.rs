//! [05-OP-40] and spec/04 §4.5.3: extrema retain activated rank and dtype.
mod common;
use assert_cmd::Command;
use serde_json::{Value, json};

fn check(path: &std::path::Path) -> std::process::Output {
    Command::cargo_bin("chelis").unwrap().env("CHELIS_STYLE_GATE_DISABLE", "1").arg("check").arg(path).output().unwrap()
}

fn evaluate(source: &str) -> (String, Value) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("extrema.ch");
    common::write_file(&path, source);
    let checked = check(&path);
    let report: Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert!(checked.status.success(), "{source}\n{report}");
    assert_eq!(report["score"], 1.0, "{report}");
    assert_eq!(report["errors"], json!([]), "{report}");
    let eval = |json: bool| {
        let mut cmd = Command::cargo_bin("chelis").unwrap();
        cmd.env("CHELIS_STYLE_GATE_DISABLE", "1").arg("eval");
        if json { cmd.arg("--json"); }
        let output = cmd.arg("--file").arg(&path).output().unwrap();
        assert!(output.status.success(), "{source}\n{}", String::from_utf8_lossy(&output.stderr));
        output.stdout
    };
    (String::from_utf8(eval(false)).unwrap(), serde_json::from_slice(&eval(true)).unwrap())
}

fn root<'a>(report: &'a Value, name: &str) -> &'a Value {
    &report["roots"].as_array().unwrap().iter().find(|root| root["name"] == name).unwrap()["value"]
}

fn assert_tensor(report: &Value, name: &str, dtype: &str, shape: Value) {
    let value = root(report, name);
    assert_eq!(value["type"], "tensor", "{name}: {value}");
    assert_eq!(value["value"]["shape"], shape, "{name}: {value}");
    let wire_dtype = match dtype { "i8" => "int8", "i16" => "int16", "i32" => "int32", "i64" => "int64", other => other };
    assert_eq!(value["value"]["data"]["dtype"], wire_dtype, "{name}: {value}");
}

fn direct_matrix(dtypes: &[&str], name: &str) {
    let mut source = String::from(
        "def minimum[r,p: Numeric](x: tensor[..r,p], y: tensor[..r,p]) -> tensor[..r,p] = min_elem(x,y)\n\
         def maximum[r,p: Numeric](x: tensor[..r,p], y: tensor[..r,p]) -> tensor[..r,p] = max_elem(x,y)\n\
         def scalar_min[p: Numeric](x: p,y: p) -> p = min_elem(x,y)\n\
         def scalar_max[p: Numeric](x: p,y: p) -> p = max_elem(x,y)\n\
         def named[rows,cols,p: Numeric](x: tensor[rows,cols,p],y: tensor[rows,cols,p]) -> tensor[rows,cols,p] = minimum(x,y)\n"
    );
    for dtype in dtypes {
        source.push_str(&format!("def minimum_{dtype}[r](x: tensor[..r,{dtype}],y: tensor[..r,{dtype}]) -> tensor[..r,{dtype}] = min_elem(x,y)\n\
         def maximum_{dtype}[r](x: tensor[..r,{dtype}],y: tensor[..r,{dtype}]) -> tensor[..r,{dtype}] = max_elem(x,y)\n"));
        let literal = |value: i32| if dtype.starts_with('i') { format!("{value}{dtype}") } else { format!("{value}.0{dtype}") };
        let a = [4,1,6,3,2,5].map(literal).join(",");
        let b = [2,2,2,2,2,2].map(literal).join(",");
        source.push_str(&format!("empty_{dtype}: tensor[0,{dtype}] = to_tensor([])\n"));
        for (op, function) in [("min","minimum"),("max","maximum")] {
            source.push_str(&format!(
                "{op}_scalar_{dtype} = scalar_{op}({four},{two})\n\
                 {op}_zero_{dtype} = {function}_{dtype}(scalar_to_tensor({four}),scalar_to_tensor({two}))\n\
                 {op}_one_{dtype} = {function}_{dtype}(to_tensor([{a}]),to_tensor([{b}]))\n\
                 {op}_two_{dtype} = {function}_{dtype}(reshape(to_tensor([{a}]),[2i64,3i64]),reshape(to_tensor([{b}]),[2i64,3i64]))\n\
                 {op}_empty_{dtype} = {function}_{dtype}(reshape(empty_{dtype},[2i64,0i64]),reshape(empty_{dtype},[2i64,0i64]))\n",
                four=literal(4), two=literal(2)
            ));
        }
        source.push_str(&format!("named_{dtype} = named(reshape(to_tensor([{a}]),[2i64,3i64]),reshape(to_tensor([{b}]),[2i64,3i64]))\n"));
    }
    let (evaluated, report) = evaluate(&source);
    for dtype in dtypes {
        for op in ["min","max"] {
            for (rank, shape) in [("zero",json!([])),("one",json!([6])),("two",json!([2,3])),("empty",json!([2,0]))] {
                assert_tensor(&report, &format!("{op}_{rank}_{dtype}"), dtype, shape);
                let data = &root(&report,&format!("{op}_{rank}_{dtype}"))["value"]["data"];
                let expected = match (op, rank) { (_,"empty") => vec![], ("min","zero") => vec![2], ("max","zero") => vec![4], ("min",_) => vec![2,1,2,2,2,2], _ => vec![4,2,6,3,2,5] };
                if dtype.starts_with('i') {
                    assert_eq!(data["values"], json!(expected), "{op}_{rank}_{dtype}");
                } else {
                    let bits: Vec<_> = expected.iter().map(|value| match *dtype {
                        "f16" => format!("{:04x}", half::f16::from_f32(*value as f32).to_bits()),
                        "bf16" => format!("{:04x}", half::bf16::from_f32(*value as f32).to_bits()),
                        "f32" => format!("{:08x}", (*value as f32).to_bits()),
                        "f64" => format!("{:016x}", (*value as f64).to_bits()),
                        _ => unreachable!(),
                    }).collect();
                    assert_eq!(data["bits"], json!(bits), "{op}_{rank}_{dtype}");
                }
            }
            let expected = if op == "min" {2} else {4};
            let rendered = if dtype.starts_with('i') { format!("{expected}") } else { format!("{expected}.0") };
            assert!(evaluated.contains(&format!("{op}_scalar_{dtype} = {rendered}\n")), "{evaluated}");
        }
        assert_tensor(&report,&format!("named_{dtype}"),dtype,json!([2,3]));
    }
    assert_eq!(common::build_and_run(&source,name), evaluated);
}

#[test]
fn integer_extrema_keep_exact_dtype_at_independent_ranks() {
    direct_matrix(&["i8","i16","i32","i64"], "integer_extrema");
}

#[test]
fn float_extrema_keep_exact_dtype_at_independent_ranks() {
    direct_matrix(&["f16","bf16","f32","f64"], "float_extrema");
}

#[test]
fn minimum_classification_preserves_rank_rows_and_nonfinite_values() {
    let mut source = String::from("def classify[r,p: Float](x: tensor[..r,p], y: tensor[..r,p]) -> (tensor[..r,p],tensor[..r,bool],tensor[..r,bool],tensor[..r,bool]) = {\n selected = min_elem(x,y)\n (selected,is_nan(selected),is_finite(selected),is_infinite(selected))\n}\n");
    for dtype in ["f16","bf16","f32","f64"] {
        for predicate in ["is_nan","is_finite","is_infinite"] {
            source.push_str(&format!("def {predicate}_{dtype}[r](x: tensor[..r,{dtype}],y: tensor[..r,{dtype}]) -> tensor[..r,bool] = {{\n selected = min_elem(x,y)\n {predicate}(selected)\n}}\n\
             {predicate}_one_{dtype} = {predicate}_{dtype}(to_tensor([1.0{dtype},div(1.0{dtype},0.0{dtype}),div(0.0{dtype},0.0{dtype})]),to_tensor([2.0{dtype},div(1.0{dtype},0.0{dtype}),0.0{dtype}]))\n\
             {predicate}_two_{dtype} = {predicate}_{dtype}(to_tensor([[1.0{dtype},2.0{dtype},3.0{dtype}],[4.0{dtype},5.0{dtype},6.0{dtype}]]),to_tensor([[2.0{dtype},2.0{dtype},2.0{dtype}],[2.0{dtype},2.0{dtype},2.0{dtype}]]))\n"));
        }
        let values = format!("1.0{dtype},div(1.0{dtype},0.0{dtype}),neg(div(1.0{dtype},0.0{dtype})),div(0.0{dtype},0.0{dtype}),-0.0{dtype},3.0{dtype}");
        source.push_str(&format!(
            "empty_{dtype}: tensor[0,{dtype}] = to_tensor([])\n\
             one_{dtype} = classify(to_tensor([{values}]),to_tensor([{values}]))\n\
             two_{dtype} = classify(reshape(to_tensor([{values}]),[2i64,3i64]),reshape(to_tensor([{values}]),[2i64,3i64]))\n\
             zero_{dtype} = classify(scalar_to_tensor(div(0.0{dtype},0.0{dtype})),scalar_to_tensor(1.0{dtype}))\n\
             vacant_{dtype} = classify(reshape(empty_{dtype},[2i64,0i64]),reshape(empty_{dtype},[2i64,0i64]))\n"
        ));
    }
    let (evaluated, _) = evaluate(&source);
    assert_eq!(evaluated.matches("data=[false, false, false, true, false, false]").count(), 8, "{evaluated}");
    assert_eq!(evaluated.matches("data=[true, false, false, false, true, true]").count(), 8, "{evaluated}");
    assert_eq!(evaluated.matches("data=[false, true, true, false, false, false]").count(), 8, "{evaluated}");
    assert_eq!(common::build_and_run(&source,"classified_extrema"),evaluated);
}

#[test]
fn reported_matrix_minimum_classifier_executes_at_each_float_width() {
    for dtype in ["f16","bf16","f32","f64"] {
        let source = format!("def classify[r](x: tensor[..r,{dtype}], y: tensor[..r,{dtype}]) -> tensor[..r,bool] = is_finite(min_elem(x,y))\nresult = classify(reshape(to_tensor([1.0{dtype},2.0{dtype},3.0{dtype},4.0{dtype}]),[2i64,2i64]),reshape(to_tensor([2.0{dtype},2.0{dtype},2.0{dtype},2.0{dtype}]),[2i64,2i64]))\n");
        let (evaluated,report) = evaluate(&source);
        assert_tensor(&report,"result","bool",json!([2,2]));
        assert!(evaluated.contains("shape=[2, 2], data=[true, true, true, true]"),"{evaluated}");
        assert_eq!(common::build_and_run(&source,&format!("reported_min_{dtype}")),evaluated);
    }
}

#[test]
fn extrema_reject_incompatible_inputs_and_false_output_claims() {
    for op in ["min_elem","max_elem"] {
        for source in [
            format!("bad = {op}(to_tensor([1i32,2i32]),to_tensor([1i32]))\n"),
            format!("bad = {op}(1i32,1i64)\n"),
            format!("bad = {op}(to_tensor([1.0f32]),to_tensor([1.0f64]))\n"),
            format!("bad = {op}(1.0f32,to_tensor([1.0f32]))\n"),
            format!("bad = {op}(true,false)\n"),
            format!("def bad[r](x: tensor[..r,f32],y: tensor[..r,f32]) -> f32 = {op}(x,y)\n"),
            format!("def bad(x: tensor[2,f32],y: tensor[2,f32]) -> tensor[3,f32] = {op}(x,y)\n"),
            format!("def bad[r](x: tensor[..r,f32],y: tensor[..r,f32]) -> tensor[..r,f64] = {op}(x,y)\n"),
            format!("bad = is_finite({op}(to_tensor([1i32]),to_tensor([2i32])))\n"),
            format!("bad = is_nan({op}(1i64,2i64))\n"),
            format!("bad = is_infinite({op}(1i8,2i8))\n"),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("invalid.ch");
            common::write_file(&path,&source);
            let output = check(&path);
            let report: Value = serde_json::from_slice(&output.stdout).unwrap();
            assert!(!output.status.success(), "accepted {source}\n{report}");
            assert!(!report["errors"].as_array().unwrap().is_empty(), "{report}");
        }
    }
}

#[test]
fn imported_exported_f32_tuple_wrapper_retains_minimum_and_mask() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path();
    common::write_file(&home.join("reef.toml"), &format!("schema = \"1\"\n[package]\nname = \"extrema\"\nversion = \"0.1.0\"\ncompiler = \"={}\"\nmodule_prefix = \"Probe\"\n",common::COMPILER_VERSION));
    common::write_file(&home.join("src/kernels.ch"),
        "module Probe.Kernels\nexport (wrapped)\n\
         def minimum[r,p: Float](x: tensor[..r,p], y: tensor[..r,p]) -> tensor[..r,p] = min_elem(x,y)\n\
         def wrapped(args: (tensor[2,3,f32],tensor[2,3,f32])) -> (tensor[2,3,f32],tensor[2,3,bool]) = {\n\
           selected = minimum(args.0,args.1)\n (selected,is_finite(selected))\n}\n");
    let path = home.join("src/main.ch");
    common::write_file(&path,"module Probe.Main\nimport Probe.Kernels (wrapped)\nresult = wrapped((to_tensor([[4.0f32,1.0f32,6.0f32],[3.0f32,2.0f32,5.0f32]]),to_tensor([[2.0f32,2.0f32,2.0f32],[2.0f32,2.0f32,2.0f32]])))\n");
    let checked = check(&path);
    let report: Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert!(checked.status.success(), "{report}");
    assert_eq!(report["errors"],json!([]));
    let eval = Command::cargo_bin("chelis").unwrap().env("CHELIS_STYLE_GATE_DISABLE", "1").args(["eval","--file"]).arg(&path).output().unwrap();
    assert!(eval.status.success(),"{}",String::from_utf8_lossy(&eval.stderr));
    let evaluated = String::from_utf8(eval.stdout).unwrap();
    assert!(evaluated.contains("shape=[2, 3], data=[2.0, 1.0, 2.0, 2.0, 2.0, 2.0]"),"{evaluated}");
    assert!(evaluated.contains("shape=[2, 3], data=[true, true, true, true, true, true]"),"{evaluated}");
    let output = home.join("native");
    Command::cargo_bin("chelis").unwrap().env("CHELIS_STYLE_GATE_DISABLE", "1").arg("build").arg(&path).args(["--target","c","--output"]).arg(&output).assert().success();
    let native = std::process::Command::new(output.join("main")).output().unwrap();
    assert!(native.status.success(),"{}",String::from_utf8_lossy(&native.stderr));
    assert_eq!(String::from_utf8(native.stdout).unwrap(),evaluated);
}

#[test]
fn authored_extent_claims_and_dynamic_input_agreement_remain_enforced() {
    for op in ["min_elem","max_elem"] {
        for (left,right,success) in [("1.0f32,2.0f32","3.0f32,4.0f32",true),("1.0f32,2.0f32,3.0f32","4.0f32,5.0f32,6.0f32",false),("1.0f32,2.0f32","3.0f32,4.0f32,5.0f32",false)] {
            let source = format!("def claimed(x: tensor[*,f32],y: tensor[*,f32]) -> tensor[2,f32] = {op}(x,y)\nresult = claimed(to_tensor([{left}]),to_tensor([{right}]))\n");
            if success {
                let (evaluated,report) = evaluate(&source);
                assert_tensor(&report,"result","f32",json!([2]));
                assert!(evaluated.contains(if op == "min_elem" {"data=[1.0, 2.0]"} else {"data=[3.0, 4.0]"}),"{evaluated}");
                assert_eq!(common::build_and_run(&source,"agreeing_claim"),evaluated);
                continue;
            }
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("claim.ch");
            common::write_file(&path,&source);
            let checked = check(&path);
            let report: Value = serde_json::from_slice(&checked.stdout).unwrap();
            assert!(checked.status.success(),"{report}");
            assert_eq!(report["errors"],json!([]));
            let eval = Command::cargo_bin("chelis").unwrap().env("CHELIS_STYLE_GATE_DISABLE", "1").args(["eval","--file"]).arg(&path).output().unwrap();
            assert!(!eval.status.success(),"false extent claim evaluated: {source}");
            assert!(String::from_utf8_lossy(&eval.stderr).to_ascii_lowercase().contains("domain"),"{}",String::from_utf8_lossy(&eval.stderr));
            let output = dir.path().join("native");
            Command::cargo_bin("chelis").unwrap().env("CHELIS_STYLE_GATE_DISABLE", "1").arg("build").arg(&path).args(["--target","c","--output"]).arg(&output).assert().success();
            let native = std::process::Command::new(output.join("claim")).output().unwrap();
            assert!(!native.status.success(),"false extent claim executed: {source}");
            assert!(String::from_utf8_lossy(&native.stderr).to_ascii_lowercase().contains("domain"),"{}",String::from_utf8_lossy(&native.stderr));
            assert!(!String::from_utf8_lossy(&native.stdout).contains("result ="));
        }
    }
}
