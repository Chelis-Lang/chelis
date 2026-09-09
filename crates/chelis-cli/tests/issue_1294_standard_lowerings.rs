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
