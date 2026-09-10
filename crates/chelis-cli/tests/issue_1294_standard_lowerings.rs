//! Executable section-4 recipes with independent results and adversarial values.
mod common;
use assert_cmd::Command;
use common::{build_and_run, parse_tensor_data};

fn evaluate(source: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("conv.ch");
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
fn explicit_epsilon_changes_values_at_all_float_widths() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        for (epsilon, expected) in [(0, vec![-1., 6.]), (3, vec![0., 4.])] {
            let source = format!(
                "def f(x: tensor[1,2,{dtype}], g: tensor[2,{dtype}], b: tensor[2,{dtype}], e: {dtype}) -> tensor[1,2,{dtype}] = layer_norm(x,g,b,e)\nx: tensor[1,2,{dtype}] = reshape(to_tensor([0.0{dtype},2.0{dtype}]),[1i64,2i64])\ng: tensor[2,{dtype}] = to_tensor([2.0{dtype},4.0{dtype}])\nb: tensor[2,{dtype}] = to_tensor([1.0{dtype},2.0{dtype}])\nresult = f(x,g,b,{epsilon}.0{dtype})\n"
            );
            assert_eq!(parse_tensor_data(&evaluate(&source), "result"), expected);
            assert_eq!(
                parse_tensor_data(
                    &build_and_run(&source, &format!("epsilon_{dtype}_{epsilon}")),
                    "result"
                ),
                expected
            );
        }
    }
}

#[test]
fn epsilon_adjoint_reverses_the_supplied_parameter() {
    let source = "x: tensor[1,2,f32] = reshape(to_tensor([0.0f32,2.0f32]),[1i64,2i64])\ng: tensor[2,f32] = to_tensor([2.0f32,4.0f32])\nb: tensor[2,f32] = to_tensor([1.0f32,2.0f32])\ndef loss(e: tensor[f32]) -> f32 = tensor_to_scalar(sum(reshape(layer_norm(x,g,b,tensor_to_scalar(e)),[2i64]),0))\nresult = grad(loss)(scalar_to_tensor(3.0f32))\n";
    for actual in [evaluate(source), build_and_run(source, "epsilon_adjoint")] {
        assert!(actual.contains("result = -0.125"), "{actual}");
    }
}

#[test]
fn batch_label_selection_uses_the_matching_diagonal() {
    let source = "def paired_select(x: tensor[2,3,f32], labels: tensor[2,int32]) -> tensor[2,f32] = diagonal(gather(x,labels,1),0,1)\nx: tensor[2,3,f32] = reshape(to_tensor([1.0f32,2.0f32,3.0f32,4.0f32,5.0f32,6.0f32]),[2i64,3i64])\nlabels: tensor[2,int32] = to_tensor([0,2])\nresult = paired_select(x,labels)\n";
    assert_eq!(parse_tensor_data(&evaluate(source), "result"), vec![1., 6.]);
    assert_eq!(
        parse_tensor_data(&build_and_run(source, "paired_labels"), "result"),
        vec![1., 6.]
    );
}

#[test]
fn embedding_and_masking_do_not_compute_with_unselected_nonfinite_values() {
    let source = "def embed(t: tensor[2,1,f32], i: tensor[1,int32]) -> tensor[1,1,f32] = gather(t,i,0)\ndef masked(m: tensor[1,bool], x: tensor[1,f32], fill: tensor[1,f32]) -> tensor[1,f32] = where(m,x,fill)\nt: tensor[2,1,f32] = reshape(to_tensor([3.0f32,div(0.0f32,0.0f32)]),[2i64,1i64])\ni: tensor[1,int32] = to_tensor([0])\nm: tensor[1,bool] = to_tensor([true])\nx: tensor[1,f32] = to_tensor([1.0f32])\nfill: tensor[1,f32] = to_tensor([div(-1.0f32,0.0f32)])\nselected = embed(t,i)\nresult = masked(m,x,fill)\n";
    for actual in [
        evaluate(source),
        build_and_run(source, "selection_nonfinite"),
    ] {
        assert_eq!(parse_tensor_data(&actual, "selected"), vec![3.]);
        assert_eq!(parse_tensor_data(&actual, "result"), vec![1.]);
    }
}

#[test]
fn softmax_nontrailing_axis_values_and_explicit_backend_limits() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        let source = format!(
            "def f(x: tensor[2,3,{dtype}]) -> tensor[2,3,{dtype}] = softmax(x,0)\nx: tensor[2,3,{dtype}] = reshape(to_tensor([0.0{dtype},0.0{dtype},0.0{dtype},0.0{dtype},0.0{dtype},0.0{dtype}]),[2i64,3i64])\nresult = f(x)\n"
        );
        assert_eq!(
            parse_tensor_data(&evaluate(&source), "result"),
            vec![0.5; 6]
        );
        if dtype == "f64" {
            // Existing #729 C max_reduce capability gap: execute and assert
            // the loud rejection, without narrowing the normative signature.
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("softmax_f64.ch");
            std::fs::write(&path, &source).unwrap();
            let output = Command::cargo_bin("chelis")
                .unwrap()
                .env("CHELIS_STYLE_GATE_DISABLE", "1")
                .arg("build")
                .arg(path)
                .args(["--target", "c", "--output"])
                .arg(dir.path().join("out"))
                .output()
                .unwrap();
            assert!(!output.status.success());
            let diagnostic = String::from_utf8(output.stderr).unwrap();
            assert!(
                diagnostic.contains("op `max_reduce` on `f64` tensors"),
                "{diagnostic}"
            );
            assert!(
                diagnostic.contains("unimplemented chelis#729"),
                "{diagnostic}"
            );
        } else {
            assert_eq!(
                parse_tensor_data(
                    &build_and_run(&source, &format!("softmax_axis0_{dtype}")),
                    "result"
                ),
                vec![0.5; 6]
            );
        }
    }
}

#[test]
fn scalar_kernel_inputs_keep_exact_integer_storage() {
    for (dtype, suffix, value) in [
        ("int8", "i8", "101"),
        ("int16", "i16", "30001"),
        ("int32", "i32", "16777217"),
        ("int64", "i64", "9007199254740993"),
    ] {
        let source = format!(
            "def f(x: tensor[1,{dtype}], s: {dtype}) -> tensor[1,{dtype}] = add(x,insert(scalar_to_tensor(s),0,1i64))\nx: tensor[1,{dtype}] = to_tensor([0{suffix}])\nresult = f(x,{value}{suffix})\n"
        );
        for actual in [
            evaluate(&source),
            build_and_run(&source, &format!("scalar_input_{dtype}")),
        ] {
            assert!(
                actual.contains(&format!("result = tensor(shape=[1], data=[{value}])")),
                "{actual}"
            );
        }
    }
}

#[test]
fn attention_recipe_has_independent_query_key_and_value_axes() {
    let source = r#"def attention(q: tensor[2,1,f32], k: tensor[3,1,f32], v: tensor[3,2,f32], mask: tensor[2,3,bool], scale: f32) -> tensor[2,2,f32] = matmul(softmax(where(mask,mul(matmul(q,permute(k,1,0)),insert(insert(scalar_to_tensor(scale),0,2i64),1,3i64)),insert(insert(scalar_to_tensor(div(-1.0f32,0.0f32)),0,2i64),1,3i64)),1),v)
q: tensor[2,1,f32] = reshape(to_tensor([0.0f32,0.0f32]),[2i64,1i64])
k: tensor[3,1,f32] = reshape(to_tensor([1.0f32,2.0f32,3.0f32]),[3i64,1i64])
v: tensor[3,2,f32] = reshape(to_tensor([1.0f32,2.0f32,3.0f32,4.0f32,5.0f32,6.0f32]),[3i64,2i64])
mask: tensor[2,3,bool] = reshape(to_tensor([true,false,false,false,false,true]),[2i64,3i64])
result = attention(q,k,v,mask,0.5f32)
"#;
    for actual in [evaluate(source), build_and_run(source, "attention_axes")] {
        assert_eq!(parse_tensor_data(&actual, "result"), vec![1., 2., 5., 6.]);
    }
}

#[test]
fn attention_recipe_executes_batched_heads() {
    let source = r#"def attention(q: tensor[1,2,2,1,f32], k: tensor[1,2,3,1,f32], v: tensor[1,2,3,2,f32], mask: tensor[1,2,2,3,bool], scale: f32) -> tensor[1,2,2,2,f32] = matmul(softmax(where(mask,mul(matmul(q,permute(k,0,1,3,2)),insert(insert(insert(insert(scalar_to_tensor(scale),0,1i64),1,2i64),2,2i64),3,3i64)),insert(insert(insert(insert(scalar_to_tensor(div(-1.0f32,0.0f32)),0,1i64),1,2i64),2,2i64),3,3i64)),3),v)
q: tensor[1,2,2,1,f32] = reshape(to_tensor([0.0f32,0.0f32,0.0f32,0.0f32]),[1i64,2i64,2i64,1i64])
k: tensor[1,2,3,1,f32] = reshape(to_tensor([1.0f32,2.0f32,3.0f32,4.0f32,5.0f32,6.0f32]),[1i64,2i64,3i64,1i64])
v: tensor[1,2,3,2,f32] = reshape(to_tensor([1.0f32,2.0f32,3.0f32,4.0f32,5.0f32,6.0f32,7.0f32,8.0f32,9.0f32,10.0f32,11.0f32,12.0f32]),[1i64,2i64,3i64,2i64])
mask: tensor[1,2,2,3,bool] = reshape(to_tensor([true,false,false,false,false,true,false,true,false,true,false,false]),[1i64,2i64,2i64,3i64])
result = attention(q,k,v,mask,0.5f32)
"#;
    for actual in [evaluate(source), build_and_run(source, "attention_batched")] {
        assert_eq!(
            parse_tensor_data(&actual, "result"),
            vec![1., 2., 5., 6., 9., 10., 7., 8.]
        );
    }
}

#[test]
fn hosted_matmul_broadcasts_batch_axes_at_every_float_width() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        // where keeps the input on the host path; reshape around it checks
        // the boundary under a DAG-capable parent, as in attention.
        let source = format!(
            "def f(a: tensor[2,1,1,2,{dtype}], b: tensor[3,2,1,{dtype}], m: tensor[2,1,1,2,bool]) -> tensor[2,3,1,1,{dtype}] = matmul(reshape(where(m,a,a),[2i64,1i64,1i64,2i64]),b)\na: tensor[2,1,1,2,{dtype}] = reshape(to_tensor([1.0{dtype},2.0{dtype},3.0{dtype},4.0{dtype}]),[2i64,1i64,1i64,2i64])\nb: tensor[3,2,1,{dtype}] = reshape(to_tensor([1.0{dtype},0.0{dtype},0.0{dtype},1.0{dtype},1.0{dtype},1.0{dtype}]),[3i64,2i64,1i64])\nm: tensor[2,1,1,2,bool] = reshape(to_tensor([true,true,true,true]),[2i64,1i64,1i64,2i64])\nresult = f(a,b,m)\n"
        );
        for actual in [
            evaluate(&source),
            build_and_run(&source, &format!("hosted_matmul_{dtype}")),
        ] {
            assert_eq!(
                parse_tensor_data(&actual, "result"),
                vec![1., 2., 3., 3., 4., 7.]
            );
        }
    }
}

#[test]
fn hosted_matmul_empty_reductions_and_batches_keep_dtype_and_shape() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        for (left, right, output, count, empty) in [
            ("2,1,0", "0,2", "2,1,2", 0, false),
            ("0,1,2", "1,2,1", "0,1,1", 2, true),
        ] {
            let shape_list = |shape: &str| {
                shape
                    .split(',')
                    .map(|n| format!("{n}i64"))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            let values = (0..count)
                .map(|_| format!("1.0{dtype}"))
                .collect::<Vec<_>>()
                .join(",");
            let source = format!(
                "def f(a: tensor[{left},{dtype}], b: tensor[{right},{dtype}], mask: tensor[{left},bool]) -> tensor[{output},{dtype}] = matmul(reshape(where(mask,a,a),[{}]),b)\nxs: List[{dtype}] = []\nys: List[{dtype}] = [{values}]\nms: List[bool] = []\na: tensor[{left},{dtype}] = reshape(to_tensor(xs),[{}])\nb: tensor[{right},{dtype}] = reshape(to_tensor(ys),[{}])\nmask: tensor[{left},bool] = reshape(to_tensor(ms),[{}])\nresult = f(a,b,mask)\n",
                shape_list(left),
                shape_list(left),
                shape_list(right),
                shape_list(left)
            );
            for actual in [
                evaluate(&source),
                build_and_run(&source, &format!("matmul_empty_{dtype}_{empty}")),
            ] {
                assert!(
                    actual.contains(&format!(
                        "result = tensor(shape=[{}],",
                        output.replace(',', ", ")
                    )),
                    "{actual}"
                );
                if empty {
                    assert!(actual.contains("data=[]"), "{actual}");
                } else {
                    assert_eq!(parse_tensor_data(&actual, "result"), vec![0.; 4]);
                }
            }
        }
    }
}

#[test]
fn hosted_matmul_preserves_the_canonical_reduction_tree() {
    // The four- and eight-leaf cases independently distinguish the canonical
    // tree from vendor GEMM and the old stride-four reduction. An odd tail is
    // carried unchanged; a singleton must not acquire a positive-zero seed.
    for (dtype, large, small) in [
        ("f16", "65504.0", "0.0001"),
        ("bf16", "1e20", "1.0"),
        ("f32", "1e20", "1.0"),
        ("f64", "1e20", "1.0"),
    ] {
        for (batch_dims, batch_shape) in [("", ""), ("1,", "1i64,")] {
            for (values, expected) in [
                (vec![large, small, &format!("-{large}"), small], 0.0_f64),
                (
                    vec![
                        large,
                        small,
                        &format!("-{large}"),
                        small,
                        &format!("-{large}"),
                        small,
                        large,
                        small,
                    ],
                    0.0,
                ),
                (
                    vec![
                        large,
                        small,
                        &format!("-{large}"),
                        small,
                        &format!("-{large}"),
                        small,
                        large,
                        small,
                        "3.0",
                    ],
                    3.0,
                ),
                (vec!["-0.0"], -0.0),
            ] {
                let n = values.len();
                let lhs = values
                    .iter()
                    .map(|v| format!("{v}{dtype}"))
                    .collect::<Vec<_>>()
                    .join(",");
                let rhs = vec![format!("1.0{dtype}"); n].join(",");
                let mask = vec!["true"; n].join(",");
                let source = format!(
                    "def f(a: tensor[{batch_dims}1,{n},{dtype}], b: tensor[{batch_dims}{n},1,{dtype}], m: tensor[{batch_dims}1,{n},bool]) -> tensor[{batch_dims}1,1,{dtype}] = matmul(reshape(where(m,a,a),[{batch_shape}1i64,{n}i64]),b)\na: tensor[{batch_dims}1,{n},{dtype}] = reshape(to_tensor([{lhs}]),[{batch_shape}1i64,{n}i64])\nb: tensor[{batch_dims}{n},1,{dtype}] = reshape(to_tensor([{rhs}]),[{batch_shape}{n}i64,1i64])\nm: tensor[{batch_dims}1,{n},bool] = reshape(to_tensor([{mask}]),[{batch_shape}1i64,{n}i64])\nresult = f(a,b,m)\n"
                );
                for actual in [
                    evaluate(&source),
                    build_and_run(&source, "matmul_canonical_tree"),
                ] {
                    let data = parse_tensor_data(&actual, "result");
                    assert_eq!(data.len(), 1, "{actual}");
                    assert_eq!(
                        data[0].to_bits(),
                        expected.to_bits(),
                        "{dtype} rank prefix {batch_dims} n={n}: {actual}"
                    );
                }
            }
        }
    }
}

#[test]
fn hosted_matmul_evaluates_effectful_operands_once_in_source_order() {
    let source = r#"def lhs() -> tensor[1,1,f32] ! { IO } = {
    _ = print("LHS")
    reshape(to_tensor([2.0f32]),[1i64,1i64])
}
def rhs() -> tensor[1,1,f32] ! { IO } = {
    _ = print("RHS")
    reshape(to_tensor([3.0f32]),[1i64,1i64])
}
result = matmul(lhs(),rhs())
"#;
    for actual in [
        evaluate(source),
        build_and_run(source, "matmul_operand_order"),
    ] {
        let effects: Vec<_> = actual
            .lines()
            .filter(|line| *line == "LHS" || *line == "RHS")
            .collect();
        assert_eq!(effects, vec!["LHS", "RHS"], "{actual}");
        assert_eq!(parse_tensor_data(&actual, "result"), vec![6.]);
    }
}

#[test]
fn bool_scalar_kernel_inputs_keep_exact_bool_storage() {
    for value in ["true", "false"] {
        let source = format!(
            "def f(s: bool) -> tensor[1,bool] = insert(scalar_to_tensor(s),0,1i64)\nresult = f({value})\n"
        );
        for actual in [
            evaluate(&source),
            build_and_run(&source, &format!("bool_scalar_input_{value}")),
        ] {
            assert!(
                actual.contains(&format!("result = tensor(shape=[1], data=[{value}])")),
                "{actual}"
            );
        }
    }
}

#[test]
fn canonical_sum_preserves_integer_trap_occurrence_in_eval_and_c() {
    for (dtype, suffix, max) in [
        ("int32", "", i64::from(i32::MAX)),
        ("int64", "i64", i64::MAX),
    ] {
        for trapping in [false, true] {
            let values = if trapping {
                vec![max, 1, 0, 0, -max, -1, 0, 0]
            } else {
                vec![max, -max, 0, 0, 1, -1, 0, 0]
            };
            let values = values
                .iter()
                .map(|v| format!("{v}{suffix}"))
                .collect::<Vec<_>>()
                .join(",");
            let source = format!(
                "def f(x: tensor[8,{dtype}]) -> tensor[1,{dtype}] = insert(sum(x,0),0,1i64)\nx: tensor[8,{dtype}] = to_tensor([{values}])\nresult = f(x)\n"
            );
            if !trapping {
                for actual in [
                    evaluate(&source),
                    build_and_run(&source, "integer_sum_tree"),
                ] {
                    assert_eq!(parse_tensor_data(&actual, "result"), vec![0.]);
                }
                continue;
            }
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("integer_sum_trap.ch");
            std::fs::write(&path, &source).unwrap();
            let output = Command::cargo_bin("chelis")
                .unwrap()
                .env("CHELIS_STYLE_GATE_DISABLE", "1")
                .args(["eval", "--file"])
                .arg(&path)
                .output()
                .unwrap();
            assert!(!output.status.success(), "canonical pair must overflow");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                stderr.contains("sum") && stderr.contains("overflow"),
                "{stderr}"
            );
            let out_dir = dir.path().join("out");
            Command::cargo_bin("chelis")
                .unwrap()
                .env("CHELIS_STYLE_GATE_DISABLE", "1")
                .arg("build")
                .arg(&path)
                .args(["--target", "c", "--output"])
                .arg(&out_dir)
                .assert()
                .success();
            assert!(
                common::link_generated(&out_dir, "integer_sum_trap.c", "integer_sum_trap")
                    .success()
            );
            let output = std::process::Command::new(out_dir.join("integer_sum_trap"))
                .output()
                .unwrap();
            assert!(!output.status.success(), "canonical C pair must overflow");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                stderr.contains("sum") && stderr.contains("overflow"),
                "{stderr}"
            );
        }
    }
}

#[test]
fn scalar_and_fused_sums_preserve_the_same_canonical_tree() {
    for expression in ["sum(x,0)", "sum(neg(x),0)"] {
        let source = format!(
            "def f(x: tensor[8,f32]) -> tensor[1,f32] = insert({expression},0,1i64)\nx: tensor[8,f32] = to_tensor([1e20f32,1.0f32,-1e20f32,1.0f32,-1e20f32,1.0f32,1e20f32,1.0f32])\nresult = f(x)\n"
        );
        for actual in [evaluate(&source), build_and_run(&source, "scalar_sum_tree")] {
            assert_eq!(
                parse_tensor_data(&actual, "result"),
                vec![0.],
                "{expression}: {actual}"
            );
        }
    }
}
