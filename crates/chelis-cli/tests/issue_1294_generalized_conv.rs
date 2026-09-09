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
fn empty_channel_contraction_returns_zeros() {
    let source = "def convolve(x: tensor[1,0,3,f32], k: tensor[1,0,1,f32]) -> tensor[1,1,3,f32] = conv(x,k,[1i64],[(0i64,0i64)])\nempty: List[f32] = []\nx: tensor[1,0,3,f32] = reshape(to_tensor(empty),[1i64,0i64,3i64])\nk: tensor[1,0,1,f32] = reshape(to_tensor(empty),[1i64,0i64,1i64])\nresult = convolve(x,k)\n";
    assert_eq!(
        parse_tensor_data(&evaluate(source), "result"),
        vec![0., 0., 0.]
    );
    assert_eq!(
        parse_tensor_data(&build_and_run(source, "conv_empty_channels"), "result"),
        vec![0., 0., 0.]
    );
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
