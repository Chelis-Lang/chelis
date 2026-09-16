//! spec/06 §§2.1–2.3 and §5.2: absent adjoints are typed zeros, not absent results.
use chelis_compiler_api::{
    compiler::eval,
    schema::{EvalRequest, ExecutionValue, SourceKind},
};

fn request(source: &str) -> EvalRequest {
    EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        bindings: Default::default(),
    }
}

fn roots(source: &str, dtype: &str, shape: &[i64], expected: &[Vec<f64>]) {
    let result = eval(request(source)).unwrap_or_else(|error| panic!("{source}\n{error:?}"));
    assert_eq!(result.roots.len(), expected.len(), "{source}\n{result:?}");
    for (root, expected) in result.roots.iter().zip(expected) {
        let ExecutionValue::Tensor { value } = &root.value else {
            panic!("wrong carrier: {root:?}")
        };
        assert_eq!(value.shape, shape);
        assert_eq!(value.data.prim().name(), dtype);
        assert_eq!(
            value
                .data
                .to_f64_lossy_vec()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}

#[test]
fn fused_unused_slot_default_reverse_subset_and_repetition() {
    for (selection, expected) in [
        ("", vec![vec![4.0, 14.0], vec![0.0, 0.0]]),
        (", wrt=(unused, x)", vec![vec![0.0, 0.0], vec![4.0, 14.0]]),
        (", wrt=unused", vec![vec![0.0, 0.0]]),
        (", wrt=x", vec![vec![4.0, 14.0]]),
        (
            ", wrt=(unused, unused, x)",
            vec![vec![0.0, 0.0], vec![0.0, 0.0], vec![4.0, 14.0]],
        ),
    ] {
        roots(
            &format!(
                "def loss(x: tensor[f32], unused: tensor[f32]) -> f32 = tensor_to_scalar(mul(x, x))\nout = vmap(grad(loss{selection}))(to_tensor([2.0f32, 7.0f32]), to_tensor([3.0f32, 11.0f32]))\n"
            ),
            "f32",
            &[2],
            &expected,
        );
    }
}

#[test]
fn fused_constant_rank_two_both_axes_and_precisions() {
    for dtype in ["f32", "f64"] {
        for axis in [0, 1] {
            let inner = if axis == 0 { 3 } else { 2 };
            let axis_arg = if axis == 0 { "" } else { ", axis=1" };
            let source = format!(
                "def loss(x: tensor[{inner}, {dtype}]) -> {dtype} = 3.0{dtype}\nout = vmap(grad(loss){axis_arg})(to_tensor([[2.0{dtype}, 7.0{dtype}, 11.0{dtype}], [3.0{dtype}, 5.0{dtype}, 13.0{dtype}]]))\n"
            );
            roots(&source, dtype, &[2, 3], &[vec![0.0; 6]]);
        }
    }
}

#[test]
fn fused_named_batch_zero_retains_caller_shape() {
    roots(
        "def loss(x: tensor[f32]) -> f32 = 3.0f32\ndef mapped(xs: tensor[batch, f32]) -> tensor[batch, f32] = vmap(grad(loss))(xs)\nout = mapped(to_tensor([2.0f32, 7.0f32]))\n",
        "f32",
        &[2],
        &[vec![0.0; 2]],
    );
}

#[test]
fn fused_constant_primal_range_trap_is_not_erased_by_zero() {
    let source = |literal| {
        format!(
            "def loss(x: tensor[f32]) -> f32 = cast(cast({literal}f64, i32), f32)\nout = vmap(grad(loss))(to_tensor([2.0f32, 7.0f32]))\n"
        )
    };
    // Check the negative first: no-roots is not an overflow diagnostic.
    let error = eval(request(&source("2147483648.0"))).unwrap_err();
    assert!(
        format!("{error:?}").contains("numeric trap: overflow in cast at i32"),
        "{error:?}"
    );
    roots(&source("7.0"), "f32", &[2], &[vec![0.0; 2]]);
}

#[test]
fn fused_empty_batch_does_not_execute_constant_primal_cast() {
    for transform in ["loss", "grad(loss)"] {
        roots(
            &format!(
                "def loss(x: tensor[f32]) -> f32 = cast(cast(2147483648.0f64, i32), f32)\nout = {{ empty: tensor[0, f32] = to_tensor([])\n vmap({transform})(empty) }}\n"
            ),
            "f32",
            &[0],
            &[vec![]],
        );
    }
}

#[test]
fn fused_live_callable_specialization_remains_nonzero() {
    for call in ["f(x)", "apply(f, x)", "x |> f"] {
        roots(
            &format!(
                "def apply(f: tensor[f32] -> f32, x: tensor[f32]) -> f32 = f(x)\ndef model(x: tensor[f32]) -> f32 = tensor_to_scalar(mul(x, x))\ndef mapped(f: tensor[f32] -> f32, xs: tensor[2, f32]) -> tensor[2, f32] = {{\n target = fn (x: tensor[f32]) -> {call}\n vmap(grad(target))(xs)\n}}\nout = mapped(model, to_tensor([2.0f32, 7.0f32]))\n"
            ),
            "f32",
            &[2],
            &[vec![4.0, 14.0]],
        );
    }
}

#[test]
fn fused_live_recursion_is_not_an_absent_adjoint() {
    let error = eval(request("def loss(x: tensor[f32]) -> f32 = mul(tensor_to_scalar(x), loss(x))\nout = vmap(grad(loss))(to_tensor([2.0f32, 7.0f32]))\n")).unwrap_err();
    assert_eq!(error.stage, "eval", "{error:?}");
    assert!(
        format!("{error:?}").contains("recursive inlining"),
        "{error:?}"
    );
}

#[test]
fn fused_runtime_extents_keep_distinct_gradient_values() {
    roots(
        "def loss(x: tensor[rows, f32], y: tensor[rows, f32]) -> f32 = {\n a = insert(sum(x, 0), 0, shape(x, 0))\n b = insert(sum(y, 0), 0, add(shape(y, 0), 2i64))\n tensor_to_scalar(add(sum(a, 0), sum(b, 0)))\n}\nout = vmap(grad(loss))(to_tensor([[2.0f32, 7.0f32, 11.0f32], [3.0f32, 5.0f32, 13.0f32]]), to_tensor([[17.0f32, 19.0f32, 23.0f32], [29.0f32, 31.0f32, 37.0f32]]))\n",
        "f32",
        &[2, 3],
        &[vec![3.0; 6], vec![5.0; 6]],
    );
}

#[test]
fn fused_zero_keeps_nested_named_extent_claim() {
    for extent in [2, 3] {
        let row = if extent == 2 {
            "[2.0f32, 7.0f32]"
        } else {
            "[2.0f32, 7.0f32, 11.0f32]"
        };
        let source = format!(
            "def claim(x: tensor[n, f32]) -> tensor[2, f32] = shrink(x, [[0i64, shape(x, 0)]])\ndef loss(x: tensor[{extent}, f32]) -> f32 = cast(shape(claim(x), 0), f32)\nout = vmap(grad(loss))(to_tensor([{row}, {row}]))\n"
        );
        if extent == 2 {
            roots(&source, "f32", &[2, 2], &[vec![0.0; 4]]);
        } else {
            let error = eval(request(&source)).unwrap_err();
            assert_eq!(error.stage, "eval", "{error:?}");
            let rendered = format!("{error:?}");
            assert!(rendered.contains("claimed = 2"), "{error:?}");
            assert!(rendered.contains("shrink axis 1 = 3"), "{error:?}");
            assert!(
                rendered.contains("numeric trap: domain in shrink at i64"),
                "{error:?}"
            );
            assert!(!rendered.contains("extent source(s)"), "{error:?}");
        }
    }
}

#[test]
fn fused_unsupported_ad_does_not_become_zero() {
    let error = eval(request("def loss(x: tensor[f32]) -> f32 = tensor_to_scalar(round(x))\nout = vmap(grad(loss))(to_tensor([2.0f32, 7.0f32]))\n")).unwrap_err();
    assert_eq!(error.stage, "eval", "{error:?}");
    assert!(
        format!("{error:?}").contains("grad: round is non-differentiable (piecewise constant)"),
        "{error:?}"
    );
}
