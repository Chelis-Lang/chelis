//! spec/06 §§2.1–2.2: select cotangent groups in written `wrt` order.
//! Exact polynomial derivatives are independent oracles, including carriers.
use chelis_compiler_api::{
    compiler::eval,
    schema::{EvalRequest, ExecutionValue, SourceKind},
};

type Leaf<'a> = (&'a str, Option<&'a [i64]>, &'a [f64]);

fn assert_roots(source: &str, dtype: &str, expected: &[Leaf<'_>]) {
    let result = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        bindings: Default::default(),
    })
    .unwrap_or_else(|error| panic!("{source}\n{error:?}"));
    assert_eq!(result.roots.len(), expected.len(), "{source}\n{result:?}");
    for (root, (name, shape, values)) in result.roots.iter().zip(expected) {
        assert_eq!(root.name.as_deref(), Some(*name), "{result:?}");
        match (&root.value, shape) {
            (ExecutionValue::Scalar { value }, None) => {
                assert_eq!(value.get().prim().name(), dtype);
                assert_eq!([value.get().as_f64_lossy()].as_slice(), *values, "{source}");
            }
            (ExecutionValue::Tensor { value }, Some(shape)) => {
                assert_eq!(value.data.prim().name(), dtype);
                assert_eq!(value.shape.as_slice(), *shape, "{source}");
                assert_eq!(value.data.to_f64_lossy_vec(), *values, "{source}");
            }
            _ => panic!("wrong cotangent carrier for {name}: {root:?}\n{source}"),
        }
    }
}

#[test]
fn direct_and_inline_scalar_selections_keep_written_order() {
    for dtype in ["f32", "f64"] {
        for inline in [false, true] {
            let (declaration, target) = if inline {
                (
                    String::new(),
                    format!("fn (x: {dtype}, w: {dtype}) -> mul(x, w)"),
                )
            } else {
                (
                    format!("def pair(x: {dtype}, w: {dtype}) -> {dtype} = mul(x, w)\n"),
                    "pair".into(),
                )
            };
            let source =
                format!("{declaration}out = grad({target}, wrt=(w, x))(2.0{dtype}, 3.0{dtype})\n");
            assert_roots(
                &source,
                dtype,
                &[("out.0", None, &[2.0]), ("out.1", None, &[3.0])],
            );
        }
    }
}

#[test]
fn default_order_explicit_parameter_order_and_single_target_are_controls() {
    for (selection, expected) in [
        (
            "",
            vec![("out.0", None, &[3.0][..]), ("out.1", None, &[2.0][..])],
        ),
        (
            ", wrt=(x, w)",
            vec![("out.0", None, &[3.0][..]), ("out.1", None, &[2.0][..])],
        ),
        (", wrt=w", vec![("out", None, &[2.0][..])]),
    ] {
        let source = format!(
            "def pair(x: f32, w: f32) -> f32 = mul(x, w)\nout = grad(pair{selection})(2.0f32, 3.0f32)\n"
        );
        assert_roots(&source, "f32", &expected);
    }
}

#[test]
fn reversed_subset_is_not_a_reversal_of_all_arguments() {
    // x*w*z at (2,3,5): dz=6, dx=15; dw=10 is not selected.
    assert_roots(
        "def triple(x: f32, w: f32, z: f32) -> f32 = mul(mul(x, w), z)\nout = grad(triple, wrt=(z, x))(2.0f32, 3.0f32, 5.0f32)\n",
        "f32",
        &[("out.0", None, &[6.0]), ("out.1", None, &[15.0])],
    );
}

#[test]
fn tensor_targets_move_as_whole_shaped_values() {
    assert_roots(
        "def pair(x: tensor[2, f32], w: tensor[2, f32]) -> tensor[f32] = sum(mul(x, w), 0i32)\nout = grad(pair, wrt=(w, x))(to_tensor([2.0f32, 7.0f32]), to_tensor([3.0f32, 11.0f32]))\n",
        "f32",
        &[
            ("out.0", Some(&[2]), &[2.0, 7.0]),
            ("out.1", Some(&[2]), &[3.0, 11.0]),
        ],
    );
}

#[test]
fn rank_zero_tensor_and_scalar_carriers_follow_the_selected_groups() {
    for dtype in ["f32", "f64"] {
        assert_roots(
            &format!(
                "def pair(x: {dtype}, w: tensor[{dtype}]) -> {dtype} = mul(x, tensor_to_scalar(w))\nout = grad(pair, wrt=(w, x))(2.0{dtype}, scalar_to_tensor(3.0{dtype}))\n"
            ),
            dtype,
            &[("out.0", Some(&[]), &[2.0]), ("out.1", None, &[3.0])],
        );
    }
}

#[test]
fn selected_zero_keeps_shape_and_does_not_shift_the_next_group() {
    assert_roots(
        "def loss(x: f32, unused: tensor[2, f32]) -> f32 = mul(x, x)\nout = grad(loss, wrt=(unused, x))(3.0f32, to_tensor([2.0f32, 7.0f32]))\n",
        "f32",
        &[("out.0", Some(&[2]), &[0.0, 0.0]), ("out.1", None, &[6.0])],
    );
}

#[test]
fn structured_target_moves_without_reversing_its_leaf_order() {
    // (2*p0+5*p1)*scale at ((2,7),3): dscale=39, dp=(6,15).
    assert_roots(
        "def loss(pair: (f32, f32), scale: f32) -> f32 = mul(add(mul(2.0f32, pair.0), mul(5.0f32, pair.1)), scale)\nout = grad(loss, wrt=(scale, pair))((2.0f32, 7.0f32), 3.0f32)\n",
        "f32",
        &[
            ("out.0", None, &[39.0]),
            ("out.1.0", None, &[6.0]),
            ("out.1.1", None, &[15.0]),
        ],
    );
}

#[test]
fn existing_fused_vmap_grad_preserves_selected_group_order() {
    assert_roots(
        "def pair(x: tensor[f32], w: tensor[f32]) -> f32 = tensor_to_scalar(mul(x, w))\nout = vmap(grad(pair, wrt=(w, x)))(to_tensor([2.0f32, 7.0f32]), to_tensor([3.0f32, 11.0f32]))\n",
        "f32",
        &[
            ("out.0", Some(&[2]), &[2.0, 7.0]),
            ("out.1", Some(&[2]), &[3.0, 11.0]),
        ],
    );
}

#[test]
fn fused_default_single_and_reversed_subset_keep_their_target_identity() {
    for (selection, expected) in [
        (
            "",
            vec![
                ("out.0", Some(&[2][..]), &[3.0, 11.0][..]),
                ("out.1", Some(&[2][..]), &[2.0, 7.0][..]),
            ],
        ),
        (", wrt=w", vec![("out", Some(&[2][..]), &[2.0, 7.0][..])]),
    ] {
        assert_roots(
            &format!(
                "def pair(x: tensor[f32], w: tensor[f32]) -> f32 = tensor_to_scalar(mul(x, w))\nout = vmap(grad(pair{selection}))(to_tensor([2.0f32, 7.0f32]), to_tensor([3.0f32, 11.0f32]))\n"
            ),
            "f32",
            &expected,
        );
    }
    assert_roots(
        "def loss(x: tensor[f32], w: tensor[f32], z: tensor[f32]) -> f32 = tensor_to_scalar(mul(mul(x, w), z))\nout = vmap(grad(loss, wrt=(z, x)))(to_tensor([2.0f32, 7.0f32]), to_tensor([3.0f32, 11.0f32]), to_tensor([5.0f32, 13.0f32]))\n",
        "f32",
        &[
            ("out.0", Some(&[2]), &[6.0, 77.0]),
            ("out.1", Some(&[2]), &[15.0, 143.0]),
        ],
    );
}

#[test]
fn non_differentiable_explicit_target_still_rejects() {
    let error = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: "def loss(count: int32, x: f32) -> f32 = mul(x, x)\nout = grad(loss, wrt=count)(2i32, 3.0f32)\n".into(),
        bindings: Default::default(),
    }).expect_err("ordering must not admit a discrete cotangent target");
    assert!(
        format!("{error:?}").contains("not differentiable"),
        "{error:?}"
    );
}

#[test]
fn repeated_selected_targets_retain_each_requested_tuple_slot() {
    // The checker already preserves each written selector, including repeats.
    assert_roots(
        "def pair(x: f32, w: f32) -> f32 = mul(x, w)\nout = grad(pair, wrt=(w, w, x))(2.0f32, 3.0f32)\n",
        "f32",
        &[
            ("out.0", None, &[2.0]),
            ("out.1", None, &[2.0]),
            ("out.2", None, &[3.0]),
        ],
    );
}
