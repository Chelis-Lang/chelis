//! Spec/04 tuple projection and §4.7.2: declared tensor results do not
//! erase structured actual parameters or restrict the origin of an i64 extent.
mod common;

use assert_cmd::Command;

fn check(source: &str, accepted: bool) {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("structured.ch");
    common::write_file(&file, source);
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("check")
        .arg(&file)
        .output()
        .unwrap();
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(output.status.success(), accepted, "{report}");
    if accepted {
        assert_eq!(report["score"], 1.0, "{report}");
        assert_eq!(report["errors"], serde_json::json!([]), "{report}");
    } else {
        assert!(!report["errors"].as_array().unwrap().is_empty(), "{report}");
    }
}

fn parity(source: &str, expected: &str) {
    check(source, true);
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("structured.ch");
    common::write_file(&file, source);
    let eval = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file"])
        .arg(&file)
        .output()
        .unwrap();
    assert!(
        eval.status.success(),
        "{}",
        String::from_utf8_lossy(&eval.stderr)
    );
    assert_eq!(String::from_utf8(eval.stdout).unwrap(), expected);
    assert_eq!(common::build_and_run(source, "structured"), expected);
}

#[test]
fn declared_tensor_results_preserve_projected_and_destructured_parameters() {
    for (parameter, body, argument) in [
        (
            "(tensor[2,f32],tensor[2,f32])",
            "p.0",
            "(to_tensor([1.0f32,2.0f32]),to_tensor([8.0f32,9.0f32]))",
        ),
        (
            "(tensor[2,f32],tensor[2,f32])",
            "{\n (a,b) = p\n a\n }",
            "(to_tensor([1.0f32,2.0f32]),to_tensor([8.0f32,9.0f32]))",
        ),
        (
            "(tensor[2,f32],f32)",
            "p.0",
            "(to_tensor([1.0f32,2.0f32]),3.0f32)",
        ),
        (
            "(tensor[*,f32],f32)",
            "p.0",
            "(to_tensor([1.0f32,2.0f32]),3.0f32)",
        ),
        (
            "((i64,tensor[2,f32]),f32)",
            "(p.0).1",
            "((7i64,to_tensor([1.0f32,2.0f32])),3.0f32)",
        ),
    ] {
        parity(
            &format!(
                "def first(p: {parameter}) -> tensor[2,f32] = {body}\nout = first({argument})\n"
            ),
            "out = tensor(shape=[2], data=[1.0, 2.0])\n",
        );
    }
}

#[test]
fn tuple_runtime_extents_execute_under_dimension_binders() {
    for size in [
        "t.0",
        "add(t.0,0i64)",
        "cast(t.0,i64)",
        "{\n (a,b) = t\n a\n }",
    ] {
        parity(
            &format!(
                "def rows[a,n](b: tensor[n,f32], t: (i64,i64)) -> tensor[a,n,f32] = insert(b,0i32,{size})\nout = rows(to_tensor([1.0f32,2.0f32]),(3i64,9i64))\n"
            ),
            "out = tensor(shape=[3, 2], data=[1.0, 2.0, 1.0, 2.0, 1.0, 2.0])\n",
        );
    }
    parity(
        "def rows[a,n](b: tensor[1,n,f32], t: (i64,i64)) -> tensor[a,n,f32] = expand(b,0i32,t.0)\nout = rows(reshape(to_tensor([1.0f32,2.0f32]),[1i64,2i64]),(3i64,9i64))\n",
        "out = tensor(shape=[3, 2], data=[1.0, 2.0, 1.0, 2.0, 1.0, 2.0])\n",
    );
}

#[test]
fn tuple_helpers_compose_through_wrappers_and_callbacks() {
    parity(
        "def first(p: (tensor[2,f32],f32)) -> tensor[2,f32] = p.0\ndef wrapped(p: (tensor[2,f32],f32)) -> tensor[2,f32] = first(p)\nout = wrapped((to_tensor([1.0f32,2.0f32]),7.0f32))\n",
        "out = tensor(shape=[2], data=[1.0, 2.0])\n",
    );
    parity(
        "def first(p: (tensor[2,f32],f32)) -> tensor[2,f32] = p.0\nout = index(map(fn (p: (tensor[2,f32],f32)) -> first(p),[(to_tensor([1.0f32,2.0f32]),7.0f32)]),0i64)\n",
        "out = tensor(shape=[2], data=[1.0, 2.0])\n",
    );
    parity(
        "def first(p: (tensor[2,f32],f32)) -> tensor[2,f32] = p.0\nout = fold(fn (acc: tensor[2,f32], p: (tensor[2,f32],f32)) -> add(acc,first(p)),to_tensor([0.0f32,0.0f32]),[(to_tensor([1.0f32,2.0f32]),7.0f32)])\n",
        "out = tensor(shape=[2], data=[1.0, 2.0])\n",
    );
}

#[test]
fn tuple_parameters_retain_named_shapes_and_scalar_leaf_types() {
    parity(
        "def first[n](p: (tensor[n,f32],f32)) -> tensor[n,f32] = p.0\na = first((to_tensor([1.0f32,2.0f32]),7.0f32))\nb = first((to_tensor([3.0f32]),9.0f32))\n",
        "a = tensor(shape=[2], data=[1.0, 2.0])\nb = tensor(shape=[1], data=[3.0])\n",
    );
    parity(
        "def scaled(p: (tensor[2,f32],f32)) -> tensor[2,f32] = mul(p.0,expand(scalar_to_tensor(p.1) |> insert(0i32,1i64),0i32,2i64))\nout = scaled((to_tensor([1.0f32,2.0f32]),3.0f32))\n",
        "out = tensor(shape=[2], data=[3.0, 6.0])\n",
    );
}

#[test]
#[allow(deprecated)]
fn prepared_calls_preserve_internal_mixed_tuples_and_existing_admission() {
    use chelis_compiler_api::compiler::prepare_eval;
    use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind, TensorValue};
    use std::collections::BTreeMap;

    let prepared = prepare_eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: "def first(p: (tensor[2,f32],f32)) -> tensor[2,f32] = p.0\ndef wrapped(t: tensor[2,f32], k: f32) -> tensor[2,f32] = first((t,k))\n".into(),
        bindings: BTreeMap::new(),
    }).expect("declared tuple entry compiles");
    for bits in [["3f800000", "40000000"], ["40400000", "40800000"]] {
        let tensor = serde_json::json!({"type":"tensor", "value":{"shape":[2],"data":{"dtype":"f32","bits":bits}}});
        let scalar = serde_json::json!({"type":"scalar","value":{"dtype":"f32","bits":"3f800000"}});
        let binding: TensorValue = serde_json::from_value(tensor["value"].clone()).unwrap();
        let argument: ExecutionValue = serde_json::from_value(scalar.clone()).unwrap();
        let result = prepared
            .call_root(
                "wrapped",
                BTreeMap::from([("k".into(), argument)]),
                BTreeMap::from([("t".into(), binding.clone())]),
            )
            .expect("typed wrapper call evaluates");
        assert_eq!(result.roots.len(), 1);
        assert_eq!(
            serde_json::to_value(&result.roots[0].value).unwrap(),
            tensor
        );
        for mutation in ["missing", "dtype", "shape"] {
            let mut input = binding.clone();
            let mut arg = scalar.clone();
            match mutation {
                "missing" => {}
                "dtype" => {
                    arg = serde_json::json!({"type":"scalar","value":{"dtype":"f64","bits":"3ff0000000000000"}});
                }
                "shape" => {
                    input.shape = vec![1, 2];
                }
                _ => unreachable!(),
            }
            let args = if mutation == "missing" {
                BTreeMap::new()
            } else {
                BTreeMap::from([("k".into(), serde_json::from_value(arg).unwrap())])
            };
            assert!(
                prepared
                    .call_root("wrapped", args, BTreeMap::from([("t".into(), input)]))
                    .is_err(),
                "{mutation}"
            );
        }
        // #3804 owns aggregate API admission. #2938's direct call_root
        // witness came from the earlier #3827 candidate, not merged main.
        let tuple: ExecutionValue =
            serde_json::from_value(serde_json::json!({"type":"tuple","value":[tensor,scalar]}))
                .unwrap();
        let error = prepared
            .call_root(
                "first",
                BTreeMap::from([("p".into(), tuple)]),
                BTreeMap::new(),
            )
            .expect_err("aggregate API admission remains separately owned");
        assert!(
            error.errors.iter().any(|error| error.kind()
                == chelis_vocab::DiagnosticKind::UnsupportedFeature
                && error.message.contains("chelis#3804")),
            "{error:?}"
        );
    }
}

#[test]
fn structured_parameters_keep_type_index_and_static_shape_rejections() {
    for source in [
        "def bad(p: (tensor[2,f32],f32)) -> tensor[2,f32] = p.2\n",
        "def bad(p: (tensor[2,f32],f32)) -> tensor[2,f32] = p.1\n",
        "def bad(p: (tensor[3,f32],f32)) -> tensor[2,f32] = p.0\n",
        "def bad[a,n](b: tensor[n,f32], t: (i32,i64)) -> tensor[a,n,f32] = insert(b,0i32,t.0)\n",
    ] {
        check(source, false);
    }
}

#[test]
fn structured_parameters_keep_runtime_extent_guards() {
    for (source, operation) in [
        (
            "def first(p: (tensor[*,f32],f32)) -> tensor[2,f32] = p.0\nout = first((to_tensor([1.0f32,2.0f32,3.0f32]),7.0f32))\n",
            "load",
        ),
        (
            "def rows[a,n](b: tensor[n,f32], t: (i64,i64)) -> tensor[a,n,f32] = insert(b,0i32,t.0)\ndef size(n: i64) -> i64 = sub(n,4i64)\nout = rows(to_tensor([1.0f32,2.0f32]),(size(3i64),9i64))\n",
            "insert",
        ),
    ] {
        check(source, true);
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("bad_structured.ch");
        common::write_file(&file, source);
        let eval = Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["eval", "--file"])
            .arg(&file)
            .output()
            .unwrap();
        let out = dir.path().join("out");
        let build = Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .arg("build")
            .arg(&file)
            .args(["--emit-c", "--target", "c", "--output"])
            .arg(&out)
            .output()
            .unwrap();
        assert!(
            build.status.success(),
            "{}",
            String::from_utf8_lossy(&build.stderr)
        );
        assert!(common::link_generated(&out, "bad_structured.c", "bad_structured").success());
        let native = std::process::Command::new(out.join("bad_structured"))
            .output()
            .unwrap();
        for (lane, output) in [("Eval", eval), ("C", native)] {
            assert!(!output.status.success(), "{lane} accepted invalid extent");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                stderr.contains(&format!("numeric trap: domain in {operation} at i64")),
                "{lane}: {stderr}"
            );
        }
    }
}

/// §4.7: entry obligations precede the body even for unused nested fields.
fn trap_parity(source: &str, trap: &str, context: &str) {
    check(source, true);
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("entry.ch");
    common::write_file(&file, source);
    let eval = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file"])
        .arg(&file)
        .output()
        .unwrap();
    let out = dir.path().join("out");
    let build = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("build")
        .arg(&file)
        .args(["--emit-c", "--target", "c", "--output"])
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    assert!(common::link_generated(&out, "entry.c", "entry").success());
    let native = std::process::Command::new(out.join("entry"))
        .output()
        .unwrap();
    for (lane, output) in [("Eval", eval), ("C", native)] {
        assert!(!output.status.success(), "{lane} accepted invalid extent");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(&format!("numeric trap: {trap} at i64")),
            "{lane}: {stderr}"
        );
        assert!(stderr.contains(context), "{lane}: {stderr}");
    }
}

#[test]
fn staged_entry_checks_unused_nested_parameters_in_signature_order() {
    let dynamic =
        "def dynamic(n: i64) -> tensor[*,f32] = insert(scalar_to_tensor(1.0f32),0i32,n)\n";
    for (signature, argument, context) in [
        (
            "first[n](p: (tensor[n,f32],tensor[n,f32])) -> tensor[n,f32]",
            "(dynamic(2i64),dynamic(SIZEi64))",
            "p.1",
        ),
        (
            "first(p: (tensor[2,f32],tensor[2,f32])) -> tensor[2,f32]",
            "(dynamic(2i64),dynamic(SIZEi64))",
            "p.1",
        ),
        (
            "first[n](p: (tensor[n,f32],(f32,tensor[n,f32]))) -> tensor[n,f32]",
            "(dynamic(2i64),(7.0f32,dynamic(SIZEi64)))",
            "p.1.1",
        ),
        // z precedes a in the signature, contrary to lexical map order.
        (
            "first[n](p: (tensor[n,f32],f32), z: (f32,tensor[n,f32]), a: (tensor[n,f32],f32)) -> tensor[n,f32]",
            "(dynamic(2i64),7.0f32),(7.0f32,dynamic(SIZEi64)),(dynamic(SIZEi64),7.0f32)",
            "z.1",
        ),
    ] {
        let source = format!("{dynamic}def {signature} = p.0\nout = first({argument})\n");
        parity(
            &source.replace("SIZE", "2"),
            "out = tensor(shape=[2], data=[1.0, 1.0])\n",
        );
        trap_parity(&source.replace("SIZE", "3"), "domain in load", context);
    }
}

#[test]
fn staged_entry_guards_precede_body_traps() {
    let source = "def dynamic(n: i64) -> tensor[*,f32] = insert(scalar_to_tensor(1.0f32),0i32,n)\ndef first[n](p: (tensor[n,f32],tensor[n,f32]), d: i64) -> tensor[*,n,f32] = insert(p.0,0i32,floor_div(1i64,d))\nout = first((dynamic(2i64),dynamic(SIZEi64)),DIVISORi64)\n";
    parity(
        &source.replace("SIZE", "2").replace("DIVISOR", "1"),
        "out = tensor(shape=[1, 2], data=[1.0, 1.0])\n",
    );
    trap_parity(
        &source.replace("SIZE", "3").replace("DIVISOR", "0"),
        "domain in load",
        "p.1",
    );
    trap_parity(
        &source.replace("SIZE", "2").replace("DIVISOR", "0"),
        "division by zero in floor_div",
        "",
    );
}

#[test]
fn projected_local_ascriptions_keep_physical_carriers_and_runtime_claims() {
    for (parameter, projection, argument) in [
        ("(tensor[*,f32],f32)", "p.0", "(to_tensor([VALUES]),7.0f32)"),
        (
            "((f32,tensor[*,f32]),i64)",
            "(p.0).1",
            "((7.0f32,to_tensor([VALUES])),9i64)",
        ),
    ] {
        // A local alias has the same physical source and its own claim owner.
        for body in [
            format!("x: tensor[2,f32] = {projection}\n x"),
            format!("original = {projection}\n x: tensor[2,f32] = original\n x"),
        ] {
            let source = format!(
                "def first(p: {parameter}) -> tensor[2,f32] = {{\n {body}\n}}\nout = first({argument})\n"
            );
            parity(
                &source.replace("VALUES", "1.0f32,2.0f32"),
                "out = tensor(shape=[2], data=[1.0, 2.0])\n",
            );
            trap_parity(
                &source.replace("VALUES", "1.0f32,2.0f32,3.0f32"),
                "domain in load",
                "",
            );
        }
    }
}

#[test]
fn projected_local_ascriptions_retain_named_and_multiaxis_claims() {
    // Wildcard results leave the local annotation as the only shape claim.
    for (source, expected) in [
        (
            "def first[n](q: tensor[n,f32], p: (tensor[*,f32],f32)) -> tensor[*,f32] = {\n x: tensor[n,f32] = p.0\n x\n}\ndef dynamic(n: i64) -> tensor[*,f32] = insert(scalar_to_tensor(1.0f32),0i32,n)\nout = first(dynamic(2i64),(dynamic(SIZEi64),7.0f32))\n",
            "out = tensor(shape=[2], data=[1.0, 1.0])\n",
        ),
        (
            "def first(p: (tensor[*,*,f32],f32)) -> tensor[*,*,f32] = {\n x: tensor[2,2,f32] = p.0\n x\n}\ndef dynamic(n: i64) -> tensor[*,*,f32] = insert(insert(scalar_to_tensor(1.0f32),0i32,n),0i32,2i64)\nout = first((dynamic(SIZEi64),7.0f32))\n",
            "out = tensor(shape=[2, 2], data=[1.0, 1.0, 1.0, 1.0])\n",
        ),
    ] {
        parity(&source.replace("SIZE", "2"), expected);
        trap_parity(&source.replace("SIZE", "3"), "domain in load", "");
    }
}

fn ones(extents: &[usize]) -> String {
    format!(
        "tensor(shape={extents:?}, data=[{}])",
        vec!["1.0"; extents.iter().product()].join(", ")
    )
}

fn aggregate_witness_claim_matrix(local: bool) {
    let dynamic = "def dynamic(n: i64) -> tensor[*,f32] = insert(scalar_to_tensor(1.0f32),0i32,tensor_to_scalar(scalar_to_tensor(n)))\n";
    for (params, projection, argument) in [
        (
            "p: (tensor[n,f32],tensor[*,f32])",
            "p.1",
            "(dynamic(WITNESSi64),dynamic(ACTUALi64))",
        ),
        (
            "p: (tensor[*,f32],tensor[n,f32])",
            "p.0",
            "(dynamic(ACTUALi64),dynamic(WITNESSi64))",
        ),
        (
            "p: ((i64,tensor[n,f32]),(tensor[*,f32],bool))",
            "(p.1).0",
            "((7i64,dynamic(WITNESSi64)),(dynamic(ACTUALi64),true))",
        ),
        (
            "z: (tensor[n,f32],f32), a: (tensor[*,f32],i64)",
            "a.0",
            "(dynamic(WITNESSi64),7.0f32),(dynamic(ACTUALi64),9i64)",
        ),
    ] {
        let (result, body) = if local {
            (
                "*",
                format!("{{\n alias = {projection}\n x: tensor[n,f32] = alias\n x\n}}"),
            )
        } else {
            ("n", projection.to_owned())
        };
        let template = format!(
            "{dynamic}def first[n]({params}) -> tensor[{result},f32] = {body}\nout = first({argument})\n"
        );
        for extent in [0, 2, 3] {
            let source = template.replace("WITNESS", &extent.to_string());
            parity(
                &source.replace("ACTUAL", &extent.to_string()),
                &format!("out = {}\n", ones(&[extent])),
            );
            // The returned wildcard field must not declare its own expected n.
            trap_parity(&source.replace("ACTUAL", "4"), "domain in load", "");
        }
    }
}

#[test]
fn aggregate_interface_witnesses_govern_declared_results() {
    aggregate_witness_claim_matrix(false);
}

#[test]
fn aggregate_interface_witnesses_govern_local_claims() {
    aggregate_witness_claim_matrix(true);
}

#[test]
fn aggregate_interface_witnesses_keep_axes_and_activations() {
    let dynamic = "def matrix(m: i64, n: i64) -> tensor[*,*,f32] = insert(insert(scalar_to_tensor(1.0f32),0i32,tensor_to_scalar(scalar_to_tensor(n))),0i32,tensor_to_scalar(scalar_to_tensor(m)))\n";
    for local in [false, true] {
        let (result, body) = if local {
            (
                "*,*",
                "{\n alias = p\n x: tensor[m,n,f32] = (alias.1).0\n x\n}",
            )
        } else {
            ("m,n", "(p.1).0")
        };
        let declaration = format!(
            "{dynamic}def first[m,n](p: ((i64,tensor[m,n,f32]),(tensor[*,*,f32],bool))) -> tensor[{result},f32] = {body}\n"
        );
        // Reusing one declaration must not reuse a prior invocation's extents.
        let mut source = declaration.clone();
        let mut expected = String::new();
        for (index, (m, n)) in [(0, 3), (2, 3), (3, 2)].into_iter().enumerate() {
            source.push_str(&format!(
                "out{index} = first(((7i64,matrix({m}i64,{n}i64)),(matrix({m}i64,{n}i64),true)))\n"
            ));
            expected.push_str(&format!("out{index} = {}\n", ones(&[m, n])));
        }
        parity(&source, &expected);
        for (m, n) in [(3, 2), (2, 4), (0, 3)] {
            let source = format!(
                "{declaration}out = first(((7i64,matrix(2i64,3i64)),(matrix({m}i64,{n}i64),true)))\n"
            );
            trap_parity(&source, "domain in load", "");
        }
    }
}

#[test]
fn aggregate_witness_claims_keep_entry_and_local_placement_separate() {
    let dynamic = "def dynamic(n: i64) -> tensor[*,f32] = insert(scalar_to_tensor(1.0f32),0i32,tensor_to_scalar(scalar_to_tensor(n)))\n";
    for local in [false, true] {
        let (result, body) = if local {
            ("*", "q = floor_div(1i64,d)\n x: tensor[n,f32] = p.1\n x")
        } else {
            ("n", "q = floor_div(1i64,d)\n p.1")
        };
        let template = format!(
            "{dynamic}def first[n](p: (tensor[n,f32],tensor[*,f32]), d: i64) -> tensor[{result},f32] = {{\n {body}\n}}\nout = first((dynamic(2i64),dynamic(SIZEi64)),DIVISORi64)\n"
        );
        parity(
            &template.replace("SIZE", "2").replace("DIVISOR", "1"),
            &format!("out = {}\n", ones(&[2])),
        );
        trap_parity(
            &template.replace("SIZE", "3").replace("DIVISOR", "0"),
            if local {
                "division by zero in floor_div"
            } else {
                "domain in load"
            },
            "",
        );
        trap_parity(
            &template.replace("SIZE", "2").replace("DIVISOR", "0"),
            "division by zero in floor_div",
            "",
        );
    }
}

#[test]
fn aggregate_result_counterexample_keeps_static_refutation() {
    // The review's exact source remains a control alongside executable runtime
    // mismatches. Whole-program lowering can prove these constant extents false.
    let source = "def dynamic(n: i64) -> tensor[*,f32] = insert(scalar_to_tensor(1.0f32),0i32,n)\ndef first[n](p: (tensor[n,f32],tensor[*,f32])) -> tensor[n,f32] = p.1\nout = first((dynamic(2i64),dynamic(SIZEi64)))\n";
    parity(
        &source.replace("SIZE", "2"),
        &format!("out = {}\n", ones(&[2])),
    );
    let source = source.replace("SIZE", "3");
    check(&source, true);
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("original.ch");
    common::write_file(&file, &source);
    let eval = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file"])
        .arg(&file)
        .output()
        .unwrap();
    assert!(!eval.status.success());
    let error = String::from_utf8_lossy(&eval.stderr);
    assert!(
        error.contains("dimension mismatch")
            && error.contains("`first` declares extent `n` = 2")
            && error.contains("body produces 3"),
        "{error}"
    );
    let build = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("build")
        .arg(&file)
        .args(["--target", "c", "--emit-c", "--output"])
        .arg(dir.path().join("out"))
        .output()
        .unwrap();
    assert!(!build.status.success());
    let error = String::from_utf8_lossy(&build.stderr);
    assert!(
        error.contains("dimension mismatch")
            && error.contains("`first` declares extent `n` = 2")
            && error.contains("body produces 3"),
        "{error}"
    );
}

#[test]
fn aggregate_host_control_results_keep_fixed_interface_witnesses() {
    for (params, selected, other, args) in [
        (
            "p: (tensor[n,f32],tensor[*,f32])",
            "p.1",
            "p.0",
            "(dynamic(2i64),dynamic(SIZEi64))",
        ),
        (
            "p: ((i64,tensor[n,f32]),tensor[*,f32])",
            "p.1",
            "(p.0).1",
            "((7i64,dynamic(2i64)),dynamic(SIZEi64))",
        ),
        (
            "z: (tensor[n,f32],f32), a: (tensor[*,f32],i64)",
            "a.0",
            "z.0",
            "(dynamic(2i64),7.0f32),(dynamic(SIZEi64),9i64)",
        ),
    ] {
        let source = format!(
            "def dynamic(n: i64) -> tensor[*,f32] = insert(scalar_to_tensor(1.0f32),0i32,tensor_to_scalar(scalar_to_tensor(n)))\ndef first[n]({params}, choose: bool, d: i64) -> tensor[n,f32] = {{\n q = floor_div(1i64,d)\n if choose then {selected} else {other}\n}}\nout = first({args},CHOOSE,DIVISORi64)\n"
        );
        for (extent, choose) in [(2, "true"), (3, "false")] {
            parity(
                &source
                    .replace("SIZE", &extent.to_string())
                    .replace("CHOOSE", choose)
                    .replace("DIVISOR", "1"),
                &format!("out = {}\n", ones(&[2])),
            );
        }
        let mismatch = source.replace("SIZE", "3").replace("CHOOSE", "true");
        trap_parity(&mismatch.replace("DIVISOR", "1"), "domain in load", "");
        // The selected result owns its claim. An earlier independent body
        // trap must run before the result comparison, even when it fails.
        trap_parity(
            &mismatch.replace("DIVISOR", "0"),
            "division by zero in floor_div",
            "",
        );
    }
}

#[test]
fn aggregate_host_control_results_preserve_each_axis() {
    let source = "def matrix(m: i64, n: i64) -> tensor[*,*,f32] = insert(insert(scalar_to_tensor(1.0f32),0i32,tensor_to_scalar(scalar_to_tensor(n))),0i32,tensor_to_scalar(scalar_to_tensor(m)))\ndef first[m,n](p: (tensor[m,n,f32],tensor[*,*,f32]), choose: bool) -> tensor[m,n,f32] = if choose then p.1 else p.0\nout = first((matrix(2i64,3i64),matrix(Mi64,Ni64)),true)\n";
    parity(
        &source.replace("Mi64", "2i64").replace("Ni64", "3i64"),
        &format!("out = {}\n", ones(&[2, 3])),
    );
    for (m, n) in [(3, 2), (2, 4), (0, 3)] {
        trap_parity(
            &source
                .replace("Mi64", &format!("{m}i64"))
                .replace("Ni64", &format!("{n}i64")),
            "domain in load",
            "",
        );
    }
}
