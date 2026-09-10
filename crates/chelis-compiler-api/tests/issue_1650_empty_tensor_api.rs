//! [05-OP-57]/[04-NUM-1]: empty Lists retain their checked scalar leaf dtype.
use chelis_compiler_api::{
    compiler::eval,
    schema::{EvalRequest, ExecutionValue, SourceKind},
};

const DTYPES: [&str; 9] = [
    "f16", "bf16", "f32", "f64", "int8", "int16", "int32", "int64", "bool",
];

fn evaluate(source: &str) -> chelis_compiler_api::schema::EvalResult {
    eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        bindings: Default::default(),
    })
    .unwrap_or_else(|e| panic!("{source}: {e:?}"))
}

fn tensor(source: &str, dtype: &str, shape: &[i64], values: &[f64]) {
    let result = evaluate(source);
    let root = result
        .roots
        .iter()
        .find(|r| r.name.as_deref() == Some("out"))
        .expect("out root");
    let ExecutionValue::Tensor { value } = &root.value else {
        panic!("{root:?}")
    };
    assert_eq!(value.data.prim().name(), dtype, "{source}");
    assert_eq!(value.shape, shape, "{source}");
    assert_eq!(value.data.to_f64_lossy_vec(), values, "{source}");
}

#[test]
fn all_empty_scalar_list_dtypes_survive_direct_and_local_calls() {
    for dtype in DTYPES {
        tensor(
            &format!("xs: List[{dtype}] = []\nout = to_tensor(xs)\n"),
            dtype,
            &[0],
            &[],
        );
        tensor(
            &format!(
                "def make(xs: List[{dtype}]) -> tensor[0, {dtype}] = to_tensor(xs)\nout = make([])\n"
            ),
            dtype,
            &[0],
            &[],
        );
    }
}

#[test]
fn nested_empty_lists_preserve_dtype_and_observable_shape() {
    for dtype in DTYPES {
        tensor(
            &format!("xs: List[List[{dtype}]] = [[], []]\nout = to_tensor(xs)\n"),
            dtype,
            &[2, 0],
            &[],
        );
    }
}

#[test]
fn generic_empty_calls_use_their_independent_checked_precision() {
    let source = "def make[p: Numeric](xs: List[p]) -> tensor[0, p] = to_tensor(xs)\na: List[f64] = []\nb: List[int64] = []\nfirst = make(a)\nout = make(b)\n";
    tensor(source, "int64", &[0], &[]);
    tensor(
        &source
            .replace("first =", "out =")
            .replace("out = make(b)", "second = make(b)"),
        "f64",
        &[0],
        &[],
    );
}

#[test]
fn nonempty_controls_keep_dtype_and_values() {
    for dtype in DTYPES {
        let leaf = if dtype == "bool" {
            "true".to_string()
        } else {
            format!("cast(1, {dtype})")
        };
        tensor(
            &format!("xs: List[{dtype}] = [{leaf}]\nout = to_tensor(xs)\n"),
            dtype,
            &[1],
            &[1.0],
        );
        tensor(
            &format!("xs: List[List[{dtype}]] = [[{leaf}], [{leaf}]]\nout = to_tensor(xs)\n"),
            dtype,
            &[2, 1],
            &[1.0, 1.0],
        );
    }
}

#[test]
fn malformed_heterogeneous_and_unresolved_nested_shapes_reject() {
    for source in [
        "out = to_tensor([])\n",
        "out = to_tensor([1i32, 1.0f32])\n",
        "out = to_tensor([true, 1i32])\n",
        "out = to_tensor([\"bad\"])\n",
        "out = to_tensor([[1i32], [2i32, 3i32]])\n",
        "xs: List[List[f64]] = []\nout = to_tensor(xs)\n",
    ] {
        assert!(
            eval(EvalRequest {
                source_kind: SourceKind::Surf,
                source: source.into(),
                bindings: Default::default()
            })
            .is_err(),
            "must reject: {source}"
        );
    }
}
