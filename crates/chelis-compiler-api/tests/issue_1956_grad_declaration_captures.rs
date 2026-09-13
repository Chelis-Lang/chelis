//! #1956: a named gradient's free values retain declaration scope.
//! spec/02 value scope, spec/04 [04-LIN-1/2], spec/06 sections 2.1-2.2.
//! The derivative of sum(x*w) is exactly w; anonymous captures stay lexical.
use chelis_compiler_api::{
    compiler::{check, eval, eval_selected},
    schema::{CheckRequest, EvalRequest, EvalResult, SourceKind},
};
use serde_json::json;

const LOSS: &str = "w = to_tensor([3.0f32, 5.0f32])\n\
def loss(x: tensor[2, f32]) -> tensor[f32] = sum(mul(x, w), 0i32)\n";
const DECLARATION: [&str; 2] = ["40400000", "40a00000"];
const LEXICAL: [&str; 2] = ["40e00000", "41300000"];

fn checked_request(source: &str) -> EvalRequest {
    let checked = check(CheckRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
    })
    .unwrap_or_else(|error| panic!("check failed before runtime: {error:?}\n{source}"));
    assert!(
        checked.errors.is_empty(),
        "admission, not semantics: {checked:?}\n{source}"
    );
    assert!(checked.unresolved_names.is_empty(), "{checked:?}");
    EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        bindings: Default::default(),
    }
}

fn evaluate(source: &str, selected: bool) -> EvalResult {
    let request = checked_request(source);
    let result = if selected {
        eval_selected(request, &["out".into()])
    } else {
        eval(request)
    };
    result.unwrap_or_else(|error| panic!("evaluation failed after check: {error:?}\n{source}"))
}

fn assert_tensor(result: &EvalResult, name: &str, bits: &[&str]) {
    let out = result
        .roots
        .iter()
        .find(|root| root.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("missing {name}: {result:?}"));
    assert_eq!(
        serde_json::to_value(&out.value).unwrap(),
        json!({"type":"tensor","value":{"shape":[bits.len()],"data":{"dtype":"f32","bits":bits}}}),
        "root {name}"
    );
}

fn assert_observation(result: &EvalResult, selected: bool, roots: &[&str], events: &[&str]) {
    // Selection filters returned observations, not the checked-program manifest.
    let mut manifest_names = vec!["w"];
    manifest_names.extend_from_slice(roots);
    let names = if selected { roots } else { &manifest_names };
    assert_eq!(
        result
            .roots
            .iter()
            .map(|root| root.name.as_deref().unwrap())
            .collect::<Vec<_>>(),
        names
    );
    assert_eq!(
        result
            .manifest
            .entries
            .iter()
            .map(|entry| entry.name.as_str())
            .collect::<Vec<_>>(),
        manifest_names
    );
    assert!(
        result
            .manifest
            .entries
            .iter()
            .all(|entry| entry.required_inputs.is_empty())
    );
    assert_eq!(result.transcript, events);
}

#[test]
fn direct_named_gradient_keeps_declaration_capture_under_caller_shadow() {
    let source = format!(
        "{LOSS}out = {{ w = to_tensor([7.0f32, 11.0f32])\n grad(loss)(to_tensor([1.0f32, 2.0f32])) }}\n"
    );
    let result = evaluate(&source, false);
    assert_observation(&result, false, &["out"], &[]);
    assert_tensor(&result, "out", &DECLARATION);
}

#[test]
fn unshadowed_named_gradient_is_a_positive_control() {
    let source = format!("{LOSS}out = grad(loss)(to_tensor([1.0f32, 2.0f32]))\n");
    for selected in [false, true] {
        let result = evaluate(&source, selected);
        assert_observation(&result, selected, &["out"], &[]);
        assert_tensor(&result, "out", &DECLARATION);
    }
}

#[test]
fn inline_anonymous_gradient_keeps_its_lexical_shadow() {
    let source = format!(
        "{LOSS}out = {{ w = to_tensor([7.0f32, 11.0f32])\n grad(fn (x: tensor[2, f32]) -> sum(mul(x, w), 0i32))(to_tensor([1.0f32, 2.0f32])) }}\n"
    );
    for selected in [false, true] {
        let result = evaluate(&source, selected);
        assert_observation(&result, selected, &["out"], &[]);
        assert_tensor(&result, "out", &LEXICAL);
    }
}

#[test]
fn selected_named_gradient_keeps_declaration_capture_under_caller_shadow() {
    let source = format!(
        "{LOSS}out = {{ w = to_tensor([7.0f32, 11.0f32])\n grad(loss)(to_tensor([1.0f32, 2.0f32])) }}\n"
    );
    let result = evaluate(&source, true);
    assert_observation(&result, true, &["out"], &[]);
    assert_tensor(&result, "out", &DECLARATION);
}

#[test]
fn frozen_named_transform_does_not_follow_later_target_or_value_shadows() {
    // The grad object is created before either local shadow. Its original
    // target is the named loss, not the subsequently installed anonymous fn.
    let source = format!(
        "{LOSS}out = {{ derivative = grad(loss)\n loss = fn (x: tensor[2, f32]) -> sum(x, 0i32)\n w = to_tensor([7.0f32, 11.0f32])\n derivative(to_tensor([1.0f32, 2.0f32])) }}\n"
    );
    for selected in [false, true] {
        let result = evaluate(&source, selected);
        assert_observation(&result, selected, &["out"], &[]);
        assert_tensor(&result, "out", &DECLARATION);
    }
}

#[test]
fn frozen_anonymous_transform_keeps_the_creation_time_value() {
    let source = format!(
        "{LOSS}out = {{ w = to_tensor([7.0f32, 11.0f32])\n derivative = grad(fn (x: tensor[2, f32]) -> sum(mul(x, w), 0i32))\n w = to_tensor([13.0f32, 17.0f32])\n derivative(to_tensor([1.0f32, 2.0f32])) }}\n"
    );
    for selected in [false, true] {
        let result = evaluate(&source, selected);
        assert_observation(&result, selected, &["out"], &[]);
        assert_tensor(&result, "out", &LEXICAL);
    }
}

#[test]
fn actuals_run_once_in_primal_order_before_written_wrt_result_order() {
    let source = "w = to_tensor([3.0f32, 5.0f32])\ndef pair(x: tensor[2, f32], y: tensor[2, f32]) -> tensor[f32] = sum(add(mul(x, w), y), 0i32)\nout = grad(pair, wrt=(y, x))({ _ = print(\"left\")\n to_tensor([1.0f32, 2.0f32]) }, { _ = print(\"right\")\n to_tensor([7.0f32, 11.0f32]) })\n";
    for selected in [false, true] {
        let result = evaluate(source, selected);
        assert_observation(&result, selected, &["out.0", "out.1"], &["left", "right"]);
        assert_tensor(&result, "out.0", &["3f800000", "3f800000"]);
        assert_tensor(&result, "out.1", &DECLARATION);
    }
}

const DRAW: &str = "uniform_like(to_tensor([0.0f32, 0.0f32]), 0.0f32, 1.0f32)";
const INITIALIZED: [&str; 2] = ["3e40ae19", "3f5a27eb"];
const PARENT_FIRST: [&str; 2] = ["3f2759ea", "3f3dd732"];

fn initialized_loss(failing: bool) -> String {
    let tail = if failing {
        "to_tensor([cast(floor_div(1i32, 0i32), f32), 0.0f32])".to_string()
    } else {
        "value".to_string()
    };
    format!(
        "w = with seed(17i64) {{ _ = print(\"initialize\")\n value = {DRAW}\n {tail} }}\ndef loss(x: tensor[2, f32]) -> tensor[f32] = sum(mul(x, w), 0i32)\n"
    )
}

fn assert_reused_declaration(selected: bool, shadow: bool) {
    let binding = if shadow {
        "w = to_tensor([7.0f32, 11.0f32])\n"
    } else {
        ""
    };
    let source = format!(
        "{}out = with seed(42i64) {{ _ = print(\"entry\")\n {binding} first = grad(loss)(to_tensor([1.0f32, 2.0f32]))\n second = grad(loss)(to_tensor([3.0f32, 4.0f32]))\n (first, second, {DRAW}) }}\n",
        initialized_loss(false)
    );
    let result = evaluate(&source, selected);
    // Independent parent-stream observation is checked even on the capture RED.
    assert_tensor(&result, "out.2", &PARENT_FIRST);
    let events = if selected {
        ["entry", "initialize"]
    } else {
        ["initialize", "entry"]
    };
    assert_observation(&result, selected, &["out.0", "out.1", "out.2"], &events);
    if !selected {
        assert_tensor(&result, "w", &INITIALIZED);
    }
    assert_tensor(&result, "out.0", &INITIALIZED);
    assert_tensor(&result, "out.1", &INITIALIZED);
}

#[test]
fn eager_and_selected_gradients_reuse_one_initialization_without_a_parent_draw() {
    for selected in [false, true] {
        assert_reused_declaration(selected, false);
    }
}

#[test]
fn eager_initialized_declaration_wins_over_the_gradient_callers_shadow() {
    assert_reused_declaration(false, true);
}

#[test]
fn selected_gradient_initializes_its_declaration_despite_the_callers_shadow() {
    assert_reused_declaration(true, true);
}

#[test]
fn eager_and_selected_initializer_errors_keep_the_entered_transcript() {
    let source = format!(
        "{}out = {{ _ = print(\"entry\")\n grad(loss)({{ _ = print(\"actual\")\n to_tensor([1.0f32, 2.0f32]) }}) }}\n",
        initialized_loss(true)
    );
    for selected in [false, true] {
        let request = checked_request(&source);
        let error = if selected {
            eval_selected(request, &["out".into()])
        } else {
            eval(request)
        }
        .expect_err("the declaration initializer must fail");
        assert_eq!(error.stage, "eval");
        assert_eq!(
            error.transcript,
            if selected {
                vec!["entry", "actual", "initialize"]
            } else {
                vec!["initialize"]
            }
        );
        assert_eq!(error.errors.len(), 1);
        assert_eq!(
            error.errors[0].message,
            "numeric trap: division by zero in floor_div at int32"
        );
    }
}

#[test]
fn a_caller_shadow_cannot_hide_the_selected_declarations_initializer_error() {
    let source = format!(
        "{}out = {{ w = to_tensor([7.0f32, 11.0f32])\n _ = print(\"entry\")\n grad(loss)({{ _ = print(\"actual\")\n to_tensor([1.0f32, 2.0f32]) }}) }}\n",
        initialized_loss(true)
    );
    let error = eval_selected(checked_request(&source), &["out".into()])
        .expect_err("caller capture must not bypass a required declaration failure");
    assert_eq!(error.stage, "eval");
    assert_eq!(error.transcript, ["entry", "actual", "initialize"]);
    assert_eq!(error.errors.len(), 1);
    assert_eq!(
        error.errors[0].message,
        "numeric trap: division by zero in floor_div at int32"
    );
}

#[test]
fn a_failing_actual_does_not_enter_the_selected_capture_initializer() {
    let source = format!(
        "{}out = {{ _ = print(\"entry\")\n grad(loss)({{ _ = print(\"actual\")\n to_tensor([cast(floor_div(1i32, 0i32), f32), 0.0f32]) }}) }}\n",
        initialized_loss(false)
    );
    let error = eval_selected(checked_request(&source), &["out".into()])
        .expect_err("the actual fails before transform capture servicing");
    assert_eq!(error.stage, "eval");
    assert_eq!(error.transcript, ["entry", "actual"]);
    assert_eq!(error.errors.len(), 1);
    assert_eq!(
        error.errors[0].message,
        "numeric trap: division by zero in floor_div at int32"
    );
}
