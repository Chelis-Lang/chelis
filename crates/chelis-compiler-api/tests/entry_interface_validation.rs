//! #2530: the public selected-evaluation binding route reaches DAG admission.

use chelis_compiler_api::compiler::eval_selected;
use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind, TensorValue};
use chelis_types::{RawTensor, finalize_tensor, types::Prim};
use std::collections::BTreeMap;

const SOURCE: &str = "def sample(x: tensor[f32]) -> tensor[f32] = x + x\n";

fn input(shape: Vec<i64>, count: usize) -> TensorValue {
    TensorValue {
        shape,
        data: finalize_tensor("load", Prim::F32, RawTensor::Float(vec![1.0; count])).unwrap(),
    }
}

fn request(value: TensorValue) -> EvalRequest {
    EvalRequest {
        source_kind: SourceKind::Surf,
        source: SOURCE.into(),
        bindings: BTreeMap::from([("x".into(), value)]),
    }
}

#[test]
fn selected_scalar_binding_rejects_rank_one_and_executes_rank_zero() {
    let error = eval_selected(request(input(vec![4], 4)), &["sample".into()]).unwrap_err();
    let messages = error
        .errors
        .iter()
        .map(|error| error.message.as_str())
        .collect::<Vec<_>>();
    assert!(
        messages.iter().any(
            |message| message.contains("input `x` expected rank 0, got 1")
                && message
                    .lines()
                    .any(|line| line == "numeric trap: domain in load at i64")
        ),
        "{messages:?}"
    );

    let result = eval_selected(request(input(vec![], 1)), &["sample".into()]).unwrap();
    let root = result
        .roots
        .iter()
        .find(|root| root.name.as_deref() == Some("sample"))
        .expect("sample root");
    let ExecutionValue::Tensor { value } = &root.value else {
        panic!("expected tensor root: {:?}", root.value)
    };
    assert!(value.shape.is_empty());
    assert_eq!(value.data.to_f64_lossy_vec(), [2.0]);
}

#[test]
fn checked_value_declaration_actualizes_its_free_tensor_input() {
    let source = "x: tensor[2, f32] = x\n";
    let request = |shape: Vec<i64>, count| EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        bindings: BTreeMap::from([("x".into(), input(shape, count))]),
    };

    let good = eval_selected(request(vec![2], 2), &["x".into()]).unwrap();
    let root = good
        .roots
        .iter()
        .find(|root| root.name.as_deref() == Some("x"))
        .expect("declared value root");
    let ExecutionValue::Tensor { value } = &root.value else {
        panic!("expected tensor root: {:?}", root.value)
    };
    assert_eq!(value.shape, [2]);
    assert_eq!(value.data.to_f64_lossy_vec(), [1.0, 1.0]);

    let error = eval_selected(request(vec![], 1), &["x".into()]).unwrap_err();
    assert!(
        error.errors.iter().any(|error| {
            error.message == "input `x` expected rank 1, got 0\nnumeric trap: domain in load at i64"
        }),
        "{error:?}"
    );

    let error = eval_selected(request(vec![3], 3), &["x".into()]).unwrap_err();
    assert!(
        error.errors.iter().any(|error| {
            error.message
                == "extent `2`: claimed = 2, x axis 0 = 3\nnumeric trap: domain in load at i64"
        }),
        "{error:?}"
    );
}
