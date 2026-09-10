//! Effects preceding a failed evaluation survive its error transport (#1585).
//! spec/04 §4.7 fixes source-ordered effects against a trapping operation.

use chelis_compiler_api::compiler::{eval, eval_selected};
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use std::collections::BTreeMap;

fn request(body: &str) -> EvalRequest {
    EvalRequest {
        source_kind: SourceKind::Surf,
        source: format!("def run() -> int64 ! {{ IO }} = {{\n{body}\n}}\nout = run()\n"),
        bindings: BTreeMap::new(),
    }
}

#[test]
fn failure_retains_only_effects_that_precede_the_trap() {
    let error = eval(request(
        "_ = print(\"first\")\n_ = debug(\"second\")\nvalue = floor_div(1i64, 0i64)\n_ = print(\"after\")\nvalue",
    ))
    .expect_err("division must fail");
    assert_eq!(error.stage, "eval");
    assert_eq!(error.transcript, ["first", "second"]);
    assert!(
        error
            .errors
            .iter()
            .any(|d| d.message.contains("division by zero"))
    );
}

#[test]
fn failing_before_any_effect_has_an_empty_transcript() {
    let error = eval(request(
        "value = floor_div(1i64, 0i64)\n_ = print(\"after\")\nvalue",
    ))
    .expect_err("division must fail");
    assert!(error.transcript.is_empty());
}

#[test]
fn successful_evaluation_keeps_each_effect_once() {
    let result =
        eval(request("_ = print(\"first\")\n_ = debug(\"second\")\n7i64")).expect("success");
    assert_eq!(result.transcript, ["first", "second"]);
    assert_eq!(result.roots.len(), 1);
}

#[test]
fn compile_failure_does_not_claim_effects_ran() {
    let error = eval(request("_ = print(\"never\")\nmissing_name")).expect_err("unknown name");
    assert!(error.transcript.is_empty());
}

#[test]
fn nested_authored_failure_retains_debug_output_and_exact_message() {
    let mut request = request("value = nested()\nvalue");
    request.source.insert_str(0, "def nested() -> int64 ! { IO } = {\n_ = debug(\"nested\")\nfail(\"authored failure\")\n}\n");
    let error = eval(request).expect_err("authored failure");
    assert_eq!(error.transcript, ["nested"], "{error:?}");
    assert_eq!(error.errors.len(), 1);
    assert_eq!(error.errors[0].message, "authored failure");
}

#[test]
fn selected_evaluations_do_not_leak_transcripts_between_roots() {
    let source = "def bad() -> int64 ! { IO } = {\n_ = print(\"bad\")\nfail(\"stop\")\n}\ndef good() -> int64 ! { IO } = {\n_ = print(\"good\")\n7i64\n}\na = bad()\nb = good()\n";
    let request = || EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    };
    let error = eval_selected(request(), &["a".to_string()]).expect_err("bad root");
    assert_eq!(error.transcript, ["bad"]);
    let result = eval_selected(request(), &["b".to_string()]).expect("good root");
    assert_eq!(result.transcript, ["good"]);
}
