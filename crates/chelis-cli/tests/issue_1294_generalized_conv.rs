//! [05-OP-51]: execute one-, two-, and three-dimensional cross-correlation
//! through evaluation and generated C, with independent hand-computed values.
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
fn spatial_ranks_and_float_dtypes_execute_with_exact_results() {
    let cases = [
        (
            "[1i64,1i64,4i64]",
            "[1i64,1i64,2i64]",
            "[1i64]",
            "[(0i64,0i64)]",
            4,
            vec![12., 23., 34.],
        ),
        (
            "[1i64,1i64,2i64,3i64]",
            "[1i64,1i64,1i64,2i64]",
            "[1i64,2i64]",
            "[(0i64,0i64),(0i64,1i64)]",
            6,
            vec![12., 30., 45., 60.],
        ),
        (
            "[1i64,1i64,2i64,2i64,2i64]",
            "[1i64,1i64,2i64,1i64,1i64]",
            "[1i64,1i64,1i64]",
            "[(0i64,0i64),(0i64,0i64),(0i64,0i64)]",
            8,
            vec![15., 26., 37., 48.],
        ),
    ];
    for dtype in ["f16", "bf16", "f32", "f64"] {
        for (index, (shape, kernel_shape, strides, padding, count, expected)) in
            cases.iter().enumerate()
        {
            let values = (1..=*count)
                .map(|n| format!("{n}.0{dtype}"))
                .collect::<Vec<_>>()
                .join(",");
            let input_dims = shape.replace("i64", "").replace(['[', ']'], "");
            let kernel_dims = kernel_shape.replace("i64", "").replace(['[', ']'], "");
            let output_dims = ["1,1,3", "1,1,2,2", "1,1,1,2,2"][index];
            let source = format!(
                "def convolve(x: tensor[{input_dims},{dtype}], k: tensor[{kernel_dims},{dtype}]) -> tensor[{output_dims},{dtype}] = conv(x,k,{strides},{padding})\nx: tensor[{input_dims},{dtype}] = reshape(to_tensor([{values}]), {shape})\nk: tensor[{kernel_dims},{dtype}] = reshape(to_tensor([10.0{dtype},1.0{dtype}]), {kernel_shape})\nresult = convolve(x,k)\n"
            );
            let interpreted = evaluate(&source);
            assert_eq!(
                &parse_tensor_data(&interpreted, "result"),
                expected,
                "eval {dtype} rank {}",
                index + 1
            );
            let compiled = build_and_run(&source, &format!("conv_{dtype}_{index}"));
            assert_eq!(
                &parse_tensor_data(&compiled, "result"),
                expected,
                "compiled {dtype} rank {}",
                index + 1
            );
        }
    }
}

#[test]
fn batches_and_channels_preserve_the_declared_output_order() {
    // Each batch has two input channels. The first output channel selects
    // x[channel 0, position] + x[channel 1, position + 1]; the second uses
    // weights [[2,1],[1,2]]. Expected values are in batch/output-channel order.
    let source = "def convolve(x: tensor[2,2,3,f32], k: tensor[2,2,2,f32]) -> tensor[2,2,2,f32] = conv(x,k,[1i64],[(0i64,0i64)])\nx: tensor[2,2,3,f32] = reshape(to_tensor([1.0f32,2.0f32,3.0f32,4.0f32,5.0f32,6.0f32,7.0f32,8.0f32,9.0f32,10.0f32,11.0f32,12.0f32]),[2i64,2i64,3i64])\nk: tensor[2,2,2,f32] = reshape(to_tensor([1.0f32,0.0f32,0.0f32,1.0f32,2.0f32,1.0f32,1.0f32,2.0f32]),[2i64,2i64,2i64])\nresult = convolve(x,k)\n";
    let expected = vec![6., 8., 18., 24., 18., 20., 54., 60.];
    assert_eq!(parse_tensor_data(&evaluate(source), "result"), expected);
    assert_eq!(
        parse_tensor_data(&build_and_run(source, "conv_batch_channels"), "result"),
        expected
    );
}

#[test]
fn input_adjoint_reverses_overlapping_windows() {
    let source = "def convolve(x: tensor[1,1,4,f32], k: tensor[1,1,2,f32]) -> tensor[1,1,3,f32] = conv(x,k,[1i64],[(0i64,0i64)])\nk: tensor[1,1,2,f32] = reshape(to_tensor([10.0f32,1.0f32]), [1i64,1i64,2i64])\ndef loss(x: tensor[1,1,4,f32]) -> f32 = tensor_to_scalar(sum(reshape(convolve(x,k),[3i64]),0))\nx: tensor[1,1,4,f32] = reshape(to_tensor([1.0f32,2.0f32,3.0f32,4.0f32]),[1i64,1i64,4i64])\nresult = grad(loss)(x)\n";
    let expected = vec![10., 11., 11., 1.];
    assert_eq!(parse_tensor_data(&evaluate(source), "result"), expected);
    assert_eq!(
        parse_tensor_data(&build_and_run(source, "conv_adjoint"), "result"),
        expected
    );
}

#[test]
fn empty_convolution_dimensions_preserve_float_dtype_and_spatial_shape() {
    // [05-OP-51]: zero input channels produce dtype zero, whereas zero
    // batch/output channels produce empty tensors. Spatial validation still
    // applies (negative controls live in issue_1294_generalized_conv checker tests).
    for dtype in ["f16", "bf16", "f32", "f64"] {
        for (name, input, kernel, output, input_count, kernel_count, expected) in [
            (
                "input_channels",
                "1,0,3",
                "1,0,1",
                "1,1,3",
                0,
                0,
                vec![0., 0., 0.],
            ),
            ("batch", "0,1,3", "1,1,1", "0,1,3", 0, 1, vec![]),
            ("output_channels", "1,1,3", "0,1,1", "1,0,3", 3, 0, vec![]),
        ] {
            let shape_list = |dims: &str| {
                dims.split(',')
                    .map(|d| format!("{d}i64"))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            let values = |count| {
                (0..count)
                    .map(|_| format!("1.0{dtype}"))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            let source = format!(
                "def convolve(x: tensor[{input},{dtype}], k: tensor[{kernel},{dtype}]) -> tensor[{output},{dtype}] = conv(x,k,[1i64],[(0i64,0i64)])\nxs: List[{dtype}] = [{}]\nks: List[{dtype}] = [{}]\nx: tensor[{input},{dtype}] = reshape(to_tensor(xs),[{}])\nk: tensor[{kernel},{dtype}] = reshape(to_tensor(ks),[{}])\nresult = convolve(x,k)\n",
                values(input_count),
                values(kernel_count),
                shape_list(input),
                shape_list(kernel)
            );
            for actual in [
                evaluate(&source),
                build_and_run(&source, &format!("conv_empty_{name}_{dtype}")),
            ] {
                if expected.is_empty() {
                    assert!(actual.contains("data=[]"), "{name} {dtype}: {actual}");
                } else {
                    assert_eq!(
                        parse_tensor_data(&actual, "result"),
                        expected,
                        "{name} {dtype}: {actual}"
                    );
                }
                let shape = output.replace(',', ", ");
                assert!(
                    actual.contains(&format!("result = tensor(shape=[{shape}],")),
                    "{name} {dtype}: {actual}"
                );
            }
        }
    }
}

#[test]
fn kernel_adjoint_sums_corresponding_window_entries() {
    let source = "def convolve(x: tensor[1,1,4,f32], k: tensor[1,1,2,f32]) -> tensor[1,1,3,f32] = conv(x,k,[1i64],[(0i64,0i64)])\nx: tensor[1,1,4,f32] = reshape(to_tensor([1.0f32,2.0f32,3.0f32,4.0f32]),[1i64,1i64,4i64])\ndef loss(k: tensor[1,1,2,f32]) -> f32 = tensor_to_scalar(sum(reshape(convolve(x,k),[3i64]),0))\nk: tensor[1,1,2,f32] = reshape(to_tensor([10.0f32,1.0f32]), [1i64,1i64,2i64])\nresult = grad(loss)(k)\n";
    assert_eq!(parse_tensor_data(&evaluate(source), "result"), vec![6., 9.]);
    assert_eq!(
        parse_tensor_data(&build_and_run(source, "conv_kernel_adjoint"), "result"),
        vec![6., 9.]
    );
}

#[test]
fn generalized_convolution_preserves_the_canonical_reduction_tree() {
    for dtype in ["f32", "f64"] {
        for n in [4, 8, 9] {
            let mut values = vec!["1e20", "1.0", "-1e20", "1.0"];
            if n >= 8 {
                values.extend(["-1e20", "1.0", "1e20", "1.0"]);
            }
            if n == 9 {
                values.push("3.0");
            }
            let lhs = values
                .iter()
                .map(|v| format!("{v}{dtype}"))
                .collect::<Vec<_>>()
                .join(",");
            let rhs = vec![format!("1.0{dtype}"); n].join(",");
            let source = format!(
                "def f(a: tensor[1,1,{n},{dtype}], b: tensor[1,1,{n},{dtype}]) -> tensor[1,1,1,{dtype}] = conv(a,b,[1i64],[(0i64,0i64)])\na: tensor[1,1,{n},{dtype}] = reshape(to_tensor([{lhs}]),[1i64,1i64,{n}i64])\nb: tensor[1,1,{n},{dtype}] = reshape(to_tensor([{rhs}]),[1i64,1i64,{n}i64])\nresult = f(a,b)\n"
            );
            for actual in [
                evaluate(&source),
                build_and_run(&source, "conv_canonical_tree"),
            ] {
                assert_eq!(
                    parse_tensor_data(&actual, "result"),
                    vec![if n == 9 { 3.0 } else { 0.0 }],
                    "{dtype} n={n}: {actual}"
                );
            }
        }
    }
}
