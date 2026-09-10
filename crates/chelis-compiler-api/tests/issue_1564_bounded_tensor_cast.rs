//! Bounded cast targets retain their actual dtype at the host evaluator boundary.
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
    .unwrap_or_else(|e| panic!("{source}: {e:?}"))
}

#[test]
fn each_bounded_tensor_call_keeps_its_dtype_shape_and_exact_values() {
    for (family, dtypes) in [
        ("Float", ["f32", "f64"]),
        ("Float", ["f16", "bf16"]),
        ("Int", ["int16", "int64"]),
        ("Int", ["int8", "int32"]),
        ("Numeric", ["f32", "int64"]),
    ] {
        let source = format!(
            "def recast[p: {family}](x: tensor[2, p]) -> tensor[2, p] = cast(x, p)\ndef convert[p: {family}](x: tensor[2, int32], witness: p) -> tensor[2, p] = cast(x, p)\ndef scalar[p: {family}](x: p) -> p = cast(x, p)\na = recast(cast(to_tensor([1, 2]), {}))\nb = recast(cast(to_tensor([3, 4]), {}))\nc = convert(to_tensor([5, 6]), cast(0, {}))\nd = convert(to_tensor([7, 8]), cast(0, {}))\ns = scalar(cast(9, {}))\n",
            dtypes[0], dtypes[1], dtypes[0], dtypes[1], dtypes[1]
        );
        let result = evaluate(&source);
        let scalar = result
            .roots
            .iter()
            .find(|r| r.name.as_deref() == Some("s"))
            .expect("scalar root");
        let ExecutionValue::Scalar { value } = scalar.value else {
            panic!("{scalar:?}")
        };
        assert_eq!(value.get().prim().name(), dtypes[1]);
        assert_eq!(value.get().as_f64_lossy(), 9.0);
        for (name, dtype, values) in [
            ("a", dtypes[0], vec![1.0, 2.0]),
            ("b", dtypes[1], vec![3.0, 4.0]),
            ("c", dtypes[0], vec![5.0, 6.0]),
            ("d", dtypes[1], vec![7.0, 8.0]),
        ] {
            let root = result
                .roots
                .iter()
                .find(|r| r.name.as_deref() == Some(name))
                .expect(name);
            let ExecutionValue::Tensor { value } = &root.value else {
                panic!("{root:?}")
            };
            assert_eq!(value.shape, vec![2]);
            assert_eq!(value.data.prim().name(), dtype);
            assert_eq!(value.data.to_f64_lossy_vec(), values);
        }
    }
}

#[test]
fn integer_tensor_cast_preserves_values_beyond_float_exactness_and_checks_range() {
    let definition =
        "def convert[p: Int](x: tensor[2, int64], witness: p) -> tensor[2, p] = cast(x, p)\n";
    let result = evaluate(&format!(
        "{definition}out = convert(to_tensor([9007199254740993i64, -9007199254740993i64]), 0i64)\n"
    ));
    let ExecutionValue::Tensor { value } = &result.roots[0].value else {
        panic!("{result:?}")
    };
    assert_eq!(
        value.data.to_i64_exact_vec(),
        Some(vec![9007199254740993, -9007199254740993])
    );
    let error = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: format!("{definition}out = convert(to_tensor([128i64, 0i64]), 0i8)\n"),
        bindings: Default::default(),
    })
    .expect_err("target range is checked");
    assert!(format!("{error:?}").to_lowercase().contains("overflow"));
}

#[test]
fn checked_cast_does_not_become_a_truncating_cast() {
    let declarations =
        "def convert[p: Int](x: tensor[2, f32], witness: p) -> tensor[2, p] = cast(x, p)\n";
    let source = format!("{declarations}out = convert(to_tensor([1.5f32, -2.5f32]), 0i64)\n");
    let error = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source,
        bindings: Default::default(),
    })
    .expect_err("fractional checked cast traps");
    assert!(format!("{error:?}").to_lowercase().contains("domain"));
    let result = evaluate(&format!(
        "{}out = convert(to_tensor([1.5f32, -2.5f32]), 0i64)\n",
        declarations.replace("= cast(", "= cast_trunc(")
    ));
    let ExecutionValue::Tensor { value } = &result.roots[0].value else {
        panic!("{result:?}")
    };
    assert_eq!(value.data.prim().name(), "int64");
    assert_eq!(value.shape, vec![2]);
    assert_eq!(value.data.to_f64_lossy_vec(), vec![1.0, -2.0]);
}

#[test]
fn result_only_constraints_keep_independent_precisions() {
    let result = evaluate(
        "def convert[p: Float](x: tensor[2, int32]) -> tensor[2, p] = cast(x, p)\na: tensor[2, f32] = convert(to_tensor([16777217, -12]))\nb: tensor[2, f64] = convert(to_tensor([16777217, -12]))\n",
    );
    for (name, dtype, expected) in [("a", "f32", 16777216.0), ("b", "f64", 16777217.0)] {
        let root = result
            .roots
            .iter()
            .find(|r| r.name.as_deref() == Some(name))
            .unwrap();
        let ExecutionValue::Tensor { value } = &root.value else {
            panic!("{root:?}")
        };
        assert_eq!(value.shape, vec![2]);
        assert_eq!(value.data.prim().name(), dtype);
        assert_eq!(value.data.to_f64_lossy_vec(), vec![expected, -12.0]);
    }
}
