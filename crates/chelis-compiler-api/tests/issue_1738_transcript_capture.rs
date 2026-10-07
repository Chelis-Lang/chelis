//! Scoped completed-effect capture for forced timeout reporting (#1738).

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use chelis_compiler_api::{TranscriptCapture, install_transcript_capture};
use std::collections::BTreeMap;

#[test]
fn scoped_capture_retains_the_same_completed_effects_as_the_error() {
    let capture = TranscriptCapture::new();
    let _guard = install_transcript_capture(capture.clone());
    let error = eval(request(
        "_ = print(\"first\")\n_ = debug(\"second\")\nfail(\"stop\")",
    ))
    .expect_err("authored failure");
    assert_eq!(capture.finish(), Some(error.transcript));
}

#[test]
fn scoped_capture_does_not_replace_the_success_transcript() {
    let capture = TranscriptCapture::new();
    let _guard = install_transcript_capture(capture.clone());
    let result = eval(request("_ = print(\"first\")\n7i64")).unwrap();
    assert_eq!(result.transcript, ["first"]);
    assert_eq!(capture.finish(), Some(result.transcript.clone()));
}

#[test]
fn compile_failure_leaves_scoped_capture_empty() {
    let capture = TranscriptCapture::new();
    let _guard = install_transcript_capture(capture.clone());
    eval(request("_ = print(\"never\")\nmissing_name")).expect_err("unknown name");
    assert_eq!(capture.finish(), Some(vec![]));
}

#[test]
fn closed_capture_does_not_suppress_the_library_result() {
    let capture = TranscriptCapture::new();
    let _guard = install_transcript_capture(capture.clone());
    assert_eq!(capture.finish(), Some(vec![]));
    let result = eval(request("_ = print(\"late\")\n7i64")).unwrap();
    assert_eq!(result.transcript, ["late"]);
    assert_eq!(capture.finish(), None);
}

fn request(body: &str) -> EvalRequest {
    EvalRequest {
        source_kind: SourceKind::Surf,
        source: format!("def run() -> i64 ! {{ IO }} = {{\n{body}\n}}\nout = run()\n"),
        bindings: BTreeMap::new(),
    }
}
