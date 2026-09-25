//! #2373: a computed tensor retains its checked concat geometry at admission.

use chelis_compiler_api::compiler::{check, eval};
use chelis_compiler_api::schema::{CheckRequest, EvalRequest, ExecutionValue, SourceKind};
use std::collections::BTreeMap;

fn program(producer: &str) -> String {
    let binding = match producer {
        "direct" => "scores = x",
        "copy" => "scores = copy(x)",
        "add" => "scores = add(x, x)",
        "mul" => "scores = mul(x, x)",
        _ => panic!("unknown producer"),
    };
    format!(
        "def probabilities[s](x: tensor[s, s, f32]) -> tensor[s, *, f32] = {{\n  {binding}\n  softmax(concat([scores, scores], cast(1, i32)), -1)\n}}\noutput = probabilities(to_tensor([[0.0, 1.0], [2.0, 0.0]]))\n"
    )
}

#[test]
fn direct_copy_and_arithmetic_producers_keep_full_result() {
    for producer in ["direct", "copy", "add", "mul"] {
        let source = program(producer);
        let checked = check(CheckRequest {
            source_kind: SourceKind::Surf,
            source: source.clone(),
        })
        .expect("check report");
        assert_eq!(checked.score.get(), 1.0, "{producer}: {checked:?}");
        assert!(checked.errors.is_empty(), "{producer}: {checked:?}");
        let evaluated = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source,
            bindings: BTreeMap::new(),
        })
        .unwrap_or_else(|error| panic!("{producer}: {error:?}"));
        let output = evaluated
            .roots
            .iter()
            .find(|root| root.name.as_deref() == Some("output"))
            .expect("output root");
        let ExecutionValue::Tensor { value } = &output.value else {
            panic!("{producer}: tensor output");
        };
        assert_eq!(value.shape, [2, 4], "{producer}");
        assert_eq!(serde_json::to_value(&value.data).unwrap()["dtype"], "f32");
        let input = if producer == "add" {
            [[0.0_f64, 2.0], [4.0, 0.0]]
        } else if producer == "mul" {
            [[0.0_f64, 1.0], [4.0, 0.0]]
        } else {
            [[0.0_f64, 1.0], [2.0, 0.0]]
        };
        let expected: Vec<f64> = input
            .iter()
            .flat_map(|row| {
                let denominator = 2.0 * row.iter().map(|x| x.exp()).sum::<f64>();
                row.iter()
                    .chain(row.iter())
                    .map(move |x| x.exp() / denominator)
            })
            .collect();
        let actual = value.data.to_f64_lossy_vec();
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "{producer} element {index}: {actual} != {expected}"
            );
        }
    }
}

#[test]
fn computed_producer_does_not_admit_bad_axis_or_shape() {
    for (source, expected) in [
        (
            program("copy").replace("cast(1, i32)", "cast(2, i32)"),
            "axis",
        ),
        (
            program("add").replace("add(x, x)", "add(x, to_tensor([[1.0f32]]))"),
            "dimension",
        ),
    ] {
        let checked = check(CheckRequest {
            source_kind: SourceKind::Surf,
            source,
        })
        .expect("check report");
        let messages = checked
            .errors
            .iter()
            .map(|error| error.message.to_lowercase())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            messages.contains(expected),
            "expected {expected}: {messages}"
        );
    }
}

#[test]
fn dynamic_axis_and_bound_list_keep_full_values() {
    for list in ["[scores, scores]", "parts"] {
        let binding = if list == "parts" {
            "parts = [scores, scores]\n  "
        } else {
            ""
        };
        let source = format!(
            "def join[s](x: tensor[s, s, f32], axis: i32) -> tensor[*, *, f32] = {{\n  scores = copy(x)\n  {binding}concat({list}, axis)\n}}\noutput = join(to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32]]), cast(1, i32))\n"
        );
        let checked = check(CheckRequest {
            source_kind: SourceKind::Surf,
            source: source.clone(),
        })
        .expect("check report");
        assert_eq!(checked.score.get(), 1.0, "{checked:?}");
        assert!(checked.errors.is_empty(), "{checked:?}");
        let evaluated = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source,
            bindings: BTreeMap::new(),
        })
        .unwrap_or_else(|error| panic!("{list}: {error:?}"));
        let root = evaluated
            .roots
            .iter()
            .find(|root| root.name.as_deref() == Some("output"))
            .expect("output root");
        assert_eq!(
            serde_json::to_value(&root.value).unwrap(),
            serde_json::json!({"type":"tensor","value":{"shape":[2,4],"data":{"dtype":"f32","bits":["3f800000","40000000","3f800000","40000000","40400000","40800000","40400000","40800000"]}}}),
            "{list}"
        );
    }
}

#[test]
fn checked_static_concat_preserves_values() {
    let source = "def join(x: tensor[2, 2, f32]) -> tensor[2, 4, f32] = concat([copy(x), copy(x)], cast(1, i32))\n";
    let checked = check(CheckRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
    })
    .expect("check report");
    assert_eq!(checked.score.get(), 1.0, "{checked:?}");
    assert!(checked.errors.is_empty(), "{checked:?}");
    let evaluated = chelis_compiler_api::compiler::eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.into(),
            bindings: BTreeMap::from([(
                "x".into(),
                chelis_compiler_api::schema::TensorValue {
                    shape: vec![2, 2],
                    data: serde_json::from_value(serde_json::json!({"dtype":"f32","bits":["3f800000","40000000","40400000","40800000"]})).unwrap(),
                },
            )]),
        },
        &["join".into()],
    )
    .expect("static concat");
    assert_eq!(evaluated.roots.len(), 1);
    assert_eq!(
        serde_json::to_value(&evaluated.roots[0].value).unwrap(),
        serde_json::json!({"type":"tensor","value":{"shape":[2,4],"data":{"dtype":"f32","bits":["3f800000","40000000","3f800000","40000000","40400000","40800000","40400000","40800000"]}}})
    );
}

#[test]
fn copied_runtime_extent_keeps_independent_result_claim() {
    let source = "def join[n, m](x: tensor[n, m, f32]) -> tensor[n, 3, f32] = concat([copy(x), copy(x)], 1i32)\noutput = join(to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32]]))\n";
    let checked = check(CheckRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
    })
    .expect("check report");
    assert_eq!(checked.score.get(), 1.0, "{checked:?}");
    assert!(checked.errors.is_empty(), "{checked:?}");
    let error = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        bindings: BTreeMap::new(),
    })
    .expect_err("the independent result claim must fail");
    assert_eq!(error.stage, "eval");
    assert_eq!(error.errors.len(), 1);
    assert!(
        error.errors[0]
            .message
            .contains("numeric trap: domain in concat at i64"),
        "{error:?}"
    );
    assert!(
        error.errors[0].message.contains("claimed = 3")
            && error.errors[0].message.contains("axis 1 = 4"),
        "{error:?}"
    );
}

#[test]
fn helper_aliases_keep_two_activations_separate() {
    let source = "def join[s](x: tensor[s, *, f32]) -> tensor[s, *, f32] = {\n  copied = copy(x)\n  alias = copied\n  parts = [alias, alias]\n  concat(parts, 1i32)\n}\none = join(to_tensor([[1.0f32], [2.0f32]]))\ntwo = join(to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32]]))\n";
    let checked = check(CheckRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
    })
    .expect("check report");
    assert_eq!(checked.score.get(), 1.0, "{checked:?}");
    assert!(checked.errors.is_empty(), "{checked:?}");
    let evaluated = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        bindings: BTreeMap::new(),
    })
    .expect("helper activations");
    for (name, shape, bits) in [
        (
            "one",
            vec![2, 2],
            vec!["3f800000", "3f800000", "40000000", "40000000"],
        ),
        (
            "two",
            vec![2, 4],
            vec![
                "3f800000", "40000000", "3f800000", "40000000", "40400000", "40800000", "40400000",
                "40800000",
            ],
        ),
    ] {
        let root = evaluated
            .roots
            .iter()
            .find(|root| root.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("missing {name}: {evaluated:?}"));
        assert_eq!(
            serde_json::to_value(&root.value).unwrap(),
            serde_json::json!({"type":"tensor","value":{"shape":shape,"data":{"dtype":"f32","bits":bits}}}),
            "{name}"
        );
    }
}

#[test]
fn invalid_concat_list_rank_dtype_and_non_axis_extent_stay_loud() {
    for (source, expected) in [
        (
            "output = concat([to_tensor([1.0f32]), to_tensor([2i32])], 0i32)\n",
            "tensor precision mismatch: f32 vs i32",
        ),
        (
            "output = concat([to_tensor([1.0f32]), to_tensor([[2.0f32]])], 0i32)\n",
            "list element rank mismatch: 1 dims vs 2 dims",
        ),
    ] {
        let checked = check(CheckRequest {
            source_kind: SourceKind::Surf,
            source: source.into(),
        })
        .expect("check report");
        let messages = checked
            .errors
            .iter()
            .map(|error| error.message.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(messages.contains(expected), "{source}: {messages}");
    }
    for (source, expected) in [
        (
            "parts: List[tensor[2, 2, f32]] = []\noutput = concat(parts, 1i32)\n",
            "concat expects at least one tensor part",
        ),
        (
            "output = concat([to_tensor([[1.0f32], [2.0f32]]), to_tensor([[3.0f32]])], 1i32)\n",
            "concat expects matching non-concatenated axes; axis 0 differed",
        ),
    ] {
        let checked = check(CheckRequest {
            source_kind: SourceKind::Surf,
            source: source.into(),
        })
        .expect("check report");
        assert_eq!(checked.score.get(), 1.0, "{source}: {checked:?}");
        assert!(checked.errors.is_empty(), "{source}: {checked:?}");
        let error = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.into(),
            bindings: BTreeMap::new(),
        })
        .expect_err("runtime concat domain error");
        assert_eq!(error.stage, "eval");
        assert_eq!(error.errors.len(), 1);
        assert!(
            error.errors[0].message.contains(expected),
            "{source}: {error:?}"
        );
    }
}
