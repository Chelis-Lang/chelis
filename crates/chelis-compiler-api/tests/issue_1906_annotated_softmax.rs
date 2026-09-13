//! [05-OP-62] concat order and §4.2 softmax values survive an explicit return.
use chelis_compiler_api::compiler::{check, eval, eval_selected};
use chelis_compiler_api::schema::{CheckRequest, EvalRequest, ExecutionValue, SourceKind};
use std::collections::BTreeMap;

const JOIN: &str =
    "def join_columns[s](x: tensor[s, *, f32], y: tensor[s, *, f32]) = concat([x, y], 1i32)\n";

fn program(annotated: bool, helper: bool, unequal: bool) -> String {
    let result = if annotated {
        " -> tensor[s, *, f32]"
    } else {
        ""
    };
    let body = if helper {
        "join_columns(x, y)"
    } else {
        "concat([x, y], 1i32)"
    };
    let right = if unequal {
        "[[2.0], [-1.0]]"
    } else {
        "[[2.0, 3.0], [-1.0, 2.0]]"
    };
    format!(
        "{JOIN}def probabilities[s](x: tensor[s, *, f32], y: tensor[s, *, f32]){result} = softmax({body}, -1)\noutput = probabilities(to_tensor([[0.0, 1.0], [2.0, 0.0]]), to_tensor({right}))\n"
    )
}

fn values(annotated: bool, helper: bool, unequal: bool) {
    let source = program(annotated, helper, unequal);
    assert!(
        check(CheckRequest {
            source_kind: SourceKind::Surf,
            source: source.clone()
        })
        .expect("check request")
        .errors
        .is_empty(),
        "lawful source"
    );
    let evaluated = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source,
        bindings: BTreeMap::new(),
    })
    .expect("evaluate full body");
    let output = evaluated
        .roots
        .iter()
        .find(|root| root.name.as_deref() == Some("output"))
        .expect("output root");
    let ExecutionValue::Tensor { value } = &output.value else {
        panic!("tensor output")
    };
    let width = if unequal { 3 } else { 4 };
    assert_eq!(value.shape, vec![2, width]);
    assert_eq!(serde_json::to_value(&value.data).unwrap()["dtype"], "f32");
    let rows: Vec<Vec<f64>> = if unequal {
        vec![vec![0.0, 1.0, 2.0], vec![2.0, 0.0, -1.0]]
    } else {
        vec![vec![0.0, 1.0, 2.0, 3.0], vec![2.0, 0.0, -1.0, 2.0]]
    };
    let expected: Vec<f64> = rows
        .iter()
        .flat_map(|row| {
            let denominator: f64 = row.iter().map(|item| item.exp()).sum();
            row.iter().map(move |item| item.exp() / denominator)
        })
        .collect();
    let actual = value.data.to_f64_lossy_vec();
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-6,
            "element {index}: {actual} != {expected}"
        );
    }
}

#[test]
fn annotated_helper_equal() {
    values(true, true, false);
}
#[test]
fn annotated_helper_unequal() {
    values(true, true, true);
}
#[test]
fn inferred_helper_equal() {
    values(false, true, false);
}
#[test]
fn inferred_helper_unequal() {
    values(false, true, true);
}
#[test]
fn annotated_direct_equal() {
    values(true, false, false);
}
#[test]
fn annotated_direct_unequal() {
    values(true, false, true);
}
#[test]
fn inferred_direct_equal() {
    values(false, false, false);
}
#[test]
fn inferred_direct_unequal() {
    values(false, false, true);
}

#[test]
fn wrong_row_extent_still_rejects() {
    let source = program(true, true, true).replace("[[2.0], [-1.0]]", "[[2.0], [-1.0], [3.0]]");
    rejected(source, "dimension");
}

#[test]
fn wrong_rank_still_rejects() {
    let source = program(true, true, true).replace("[[2.0], [-1.0]]", "[2.0, -1.0]");
    rejected(source, "tensor rank mismatch: 2 dims vs 1 dims");
}

#[test]
fn invalid_softmax_axis_still_rejects() {
    let source = program(true, true, true).replace(", -1)", ", 2)");
    rejected(source, "axis");
}

#[test]
fn invalid_concat_axis_still_rejects() {
    let source = program(true, true, true).replace("1i32)", "2i32)");
    rejected(source, "axis");
}

#[test]
fn host_admission_preserves_shared_runtime_row_guard() {
    let source = "def probabilities(x: tensor[extent, *, f32], y: tensor[extent, *, f32]) -> tensor[extent, *, f32] = softmax(concat([x, y], 1i32), -1)\noutput = probabilities(to_tensor([[0.0, 1.0], [2.0, 0.0]]), to_tensor([[2.0]]))\n";
    let error = eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.to_string(),
            bindings: BTreeMap::new(),
        },
        &["output".to_string()],
    )
    .expect_err("a Host decision must retain the input relationship");
    assert_eq!(error.stage, "eval");
    assert_eq!(error.errors.len(), 1);
    // chelis#1788 re-rendered this verdict as the frozen [04-NUM-9] pair. What
    // this row pins is unchanged: a Host decision still retains the input
    // relationship and refuses the disagreeing call.
    assert_eq!(
        error.errors[0].message,
        "extent `extent`: x axis 0 = 2, y axis 0 = 1\nnumeric trap: domain in load at int64"
    );
}

fn rejected(source: String, expected: &str) {
    let report = check(CheckRequest {
        source_kind: SourceKind::Surf,
        source,
    })
    .expect("check report");
    let errors = report
        .errors
        .iter()
        .map(|error| error.message.to_lowercase())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(errors.contains(expected), "expected {expected}: {errors}");
}

#[test]
fn host_admission_preserves_random_draw_order() {
    let source = "def probabilities[s](x: tensor[s, *, f32]) -> tensor[s, *, f32] ! { Random } = {\n _ = uniform_like(x, 0.0f32, 1.0f32)\n softmax(concat([x, x], 1i32), -1)\n}\ndef run() = with seed(42i64) {\n x = to_tensor([[0.0, 1.0], [2.0, 0.0]])\n first = probabilities(copy(x))\n next = uniform_like(x, 0.0f32, 1.0f32)\n (first, next)\n}\noutput = run()\n";
    let execute = |source: String| {
        eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source,
            bindings: BTreeMap::new(),
        })
        .expect("seeded execution")
    };
    let annotated = execute(source.to_string());
    let inferred = execute(source.replace(" -> tensor[s, *, f32]", ""));
    assert_eq!(
        serde_json::to_value(&annotated.roots).unwrap(),
        serde_json::to_value(&inferred.roots).unwrap()
    );
    let omitted_draw =
        execute(source.replace(" _ = uniform_like(x, 0.0f32, 1.0f32)\n", " _ = ()\n"));
    assert_ne!(
        serde_json::to_value(&annotated.roots).unwrap(),
        serde_json::to_value(&omitted_draw.roots).unwrap(),
        "the discarded draw must consume its ordinal"
    );
}
