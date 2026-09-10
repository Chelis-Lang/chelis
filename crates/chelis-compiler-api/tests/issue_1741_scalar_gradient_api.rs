//! spec/06 §2.1: scalar cotangents are scalars; tensor rank and pytree shape survive.
use chelis_compiler_api::{
    compiler::eval,
    schema::{EvalRequest, ExecutionValue, SourceKind},
};

fn evaluate(source: &str) -> chelis_compiler_api::schema::EvalResult {
    eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        bindings: Default::default(),
    })
    .unwrap_or_else(|error| panic!("{source}: {error:?}"))
}

fn assert_true(source: &str) {
    let result = evaluate(source);
    let root = result
        .roots
        .iter()
        .find(|root| root.name.as_deref() == Some("out"))
        .expect("out root");
    assert!(
        matches!(root.value, ExecutionValue::Bool { value: true }),
        "{result:?}"
    );
}

#[test]
fn scalar_gradients_retain_scalar_comparison_and_exact_dtype() {
    for dtype in ["f32", "f64"] {
        assert_true(&format!(
            "def d(x: {dtype}) -> {dtype} = grad(fn (xx: {dtype}) -> exp(xx), wrt=xx)(x)\nout = gt(d(1.0{dtype}), 0.0{dtype})\n"
        ));
        let result = evaluate(&format!(
            "def d(x: {dtype}) -> {dtype} = grad(fn (xx: {dtype}) -> xx * xx, wrt=xx)(x)\nout = d(3.0{dtype})\n"
        ));
        let root = result
            .roots
            .iter()
            .find(|r| r.name.as_deref() == Some("out"))
            .unwrap();
        let ExecutionValue::Scalar { value } = root.value else {
            panic!("{root:?}")
        };
        assert_eq!(value.get().prim().name(), dtype);
        assert_eq!(value.get().as_f64_lossy(), 6.0);
    }
}

#[test]
fn multi_target_scalar_gradients_remain_independent_scalar_leaves() {
    for dtype in ["f32", "f64"] {
        assert_true(&format!(
            "def loss(x: {dtype}, y: {dtype}) -> {dtype} = x * y\ndef positive(x: {dtype}, y: {dtype}) -> bool = {{\n  ds = grad(loss)(x, y)\n  and(eq(ds.0, y), eq(ds.1, x))\n}}\nout = positive(2.0{dtype}, 3.0{dtype})\n"
        ));
    }
}

#[test]
fn structured_scalar_gradients_keep_their_scalar_fields() {
    for dtype in ["f32", "f64"] {
        assert_true(&format!(
            "type Box = | Box {{ value: {dtype} }}\ndef loss(box: Box) -> {dtype} = match box with {{ | Box {{ value }} => value * value }}\ndef positive(x: {dtype}) -> bool = match grad(loss)(Box {{ value: x }}) with {{ | Box {{ value }} => eq(value, 2.0{dtype} * x) }}\nout = positive(3.0{dtype})\n"
        ));
        assert_true(&format!(
            "def loss(xs: List[{dtype}]) -> {dtype} = index(xs, 0i64) * index(xs, 0i64)\ndef positive(x: {dtype}) -> bool = eq(index(grad(loss)([x]), 0i64), 2.0{dtype} * x)\nout = positive(3.0{dtype})\n"
        ));
    }
}

#[test]
fn rank_zero_gradient_stays_a_tensor() {
    for dtype in ["f32", "f64"] {
        assert_true(&format!(
            "def loss(x: tensor[{dtype}]) -> {dtype} = tensor_to_scalar(x * x)\ndef positive(x: tensor[{dtype}]) -> bool = eq(tensor_to_scalar(grad(loss)(x)), 6.0{dtype})\nout = positive(scalar_to_tensor(3.0{dtype}))\n"
        ));
    }
}

#[test]
fn mixed_gradient_targets_preserve_scalar_and_tensor_positions() {
    for dtype in ["f32", "f64"] {
        assert_true(&format!(
            "def loss(x: {dtype}, y: tensor[{dtype}]) -> {dtype} = x * tensor_to_scalar(y)\ndef positive(x: {dtype}, y: tensor[{dtype}]) -> bool = {{\n  ds = grad(loss)(x, y)\n  and(eq(ds.0, tensor_to_scalar(y)), eq(tensor_to_scalar(ds.1), x))\n}}\nout = positive(2.0{dtype}, scalar_to_tensor(3.0{dtype}))\n"
        ));
        assert_true(&format!(
            "def loss(pair: ({dtype}, tensor[{dtype}])) -> {dtype} = pair.0 * tensor_to_scalar(pair.1)\ndef positive(x: {dtype}) -> bool = {{\n  ds = grad(loss)((x, scalar_to_tensor(3.0{dtype})))\n  and(eq(ds.0, 3.0{dtype}), eq(tensor_to_scalar(ds.1), x))\n}}\nout = positive(2.0{dtype})\n"
        ));
    }
}

#[test]
fn genuine_mixed_scalar_tensor_comparisons_still_reject() {
    for tensor in ["scalar_to_tensor(1.0f32)", "to_tensor([1.0f32])"] {
        let error = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: format!("out = gt({tensor}, 0.0f32)\n"),
            bindings: Default::default(),
        })
        .expect_err("mixed comparison is a type error");
        assert!(format!("{error:?}").contains("scalar"), "{error:?}");
    }
}
