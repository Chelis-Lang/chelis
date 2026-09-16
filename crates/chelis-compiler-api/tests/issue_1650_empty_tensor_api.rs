//! [05-OP-57]/[04-NUM-1]: empty Lists retain their checked scalar leaf dtype.
use chelis_compiler_api::{
    compiler::eval,
    schema::{EvalRequest, ExecutionValue, SourceKind},
};

const DTYPES: [&str; 9] = [
    "f16", "bf16", "f32", "f64", "i8", "i16", "i32", "i64", "bool",
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
    let source = "def make[p: Numeric](xs: List[p]) -> tensor[0, p] = to_tensor(xs)\na: List[f64] = []\nb: List[i64] = []\nfirst = make(a)\nout = make(b)\n";
    tensor(source, "i64", &[0], &[]);
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

/// A separate signature carries the same checked precision evidence as
/// annotated parameters, even when the List is computed in the body.
#[test]
fn separate_signatures_preserve_computed_list_element_dtypes() {
    for dtype in ["i32", "i64", "f32", "f64"] {
        let definitions =
            "sig make[p: Numeric]: p -> tensor[n, p]\ndef make(x) = to_tensor(drop([x], 1i64))\n";
        tensor(
            &format!("{definitions}out = make(cast(1, {dtype}))\n"),
            dtype,
            &[0],
            &[],
        );
        tensor(
            &format!(
                "{}out = make(cast(1, {dtype}))\n",
                definitions.replace("drop([x], 1i64)", "[x]")
            ),
            dtype,
            &[1],
            &[1.0],
        );
    }
}

#[test]
fn separate_signature_calls_keep_declared_and_checked_binders_independent() {
    tensor(
        "sig make[p: Numeric]: p -> tensor[n, p]\ndef make(x) = to_tensor([add(x, cast(1, p))])\nfirst = make(1.0f64)\nout = make(1i64)\n",
        "i64",
        &[1],
        &[2.0],
    );
    tensor(
        "sig make[p: Numeric]: p -> tensor[n, p]\ndef make(x) = to_tensor(drop([x], 1i64))\nfirst = make(1i64)\nout = make(1.0f64)\n",
        "f64",
        &[0],
        &[],
    );
}

#[test]
fn separate_signatures_do_not_supply_unrelated_or_conflicting_dtypes() {
    let error = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: "sig make[p: Numeric]: p -> p\ndef make(x) = {\n unused = to_tensor([])\n x\n}\nout = make(1i64)\n".into(),
        bindings: Default::default(),
    }).expect_err("the argument dtype does not constrain the unrelated empty List");
    assert_eq!(error.stage, "eval");
    assert!(
        error.errors.iter().any(|diagnostic| {
            diagnostic.kind() == chelis_vocab::DiagnosticKind::EvalError
                && diagnostic
                    .message
                    .contains("to_tensor requires a resolved checked element dtype [05-OP-57]")
        }),
        "{error:?}"
    );

    let error = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: "sig make[p: Numeric]: p -> tensor[n, p]\ndef make(x) = to_tensor([x, true])\nout = make(1i64)\n".into(),
        bindings: Default::default(),
    }).expect_err("a checked signature does not permit heterogeneous List elements");
    assert!(
        error.errors.iter().any(|diagnostic| {
            diagnostic.kind() == chelis_vocab::DiagnosticKind::PrecisionMismatch
                && diagnostic.message.contains("Numeric")
                && diagnostic.message.contains("bool")
        }),
        "{error:?}"
    );
}
