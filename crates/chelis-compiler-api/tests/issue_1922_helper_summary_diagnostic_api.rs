//! [05-UNS-1..3]: source-reachable fatal lowering uses the diagnostic channel,
//! not an unwind or a fallback value. This is not insert/concat support.

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use std::collections::BTreeMap;

const SOURCE: &str = include_str!("../../../tests/support/helper_summary_fatal.ch");
const MESSAGE: &str = "`insert` size resolves to `seq`, but no in-scope tensor axis supplies that extent. Use an int64 literal or a shape(tensor, int32-axis) read. Tracked by Chelis-Lang/chelis#469";

fn request(source: &str) -> EvalRequest {
    EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    }
}

#[test]
fn fatal_summary_diagnostic_returns_and_later_eval_recovers() {
    // Repeat on this same thread: an escaped probe must not leave an inlining
    // marker or cached false summary which changes the second disposition.
    for _ in 0..2 {
        let caught = std::panic::catch_unwind(|| eval(request(SOURCE)));
        if let Err(payload) = &caught {
            eprintln!(
                "escaped diagnostic: {:?}",
                payload.downcast_ref::<chelis_ir::lower::LowerDiagnostic>()
            );
        }
        let error = caught
            .expect("public eval must return its fatal lowering diagnostic")
            .expect_err("unsupported insert must not fall back to a value");
        assert_eq!(error.stage, "eval");
        assert!(error.transcript.is_empty());
        assert_eq!(error.errors.len(), 1);
        // The existing runtime boundary carries the producer's source ID in
        // the diagnostic text; this repair does not change the wire schema.
        assert_eq!(
            error.errors[0].message,
            format!("{MESSAGE} at source span `surf:467..507`")
        );
        let valid = eval(request(
            "def sink_output(x: f32) -> f32 = add(x, 2.0)\noutput = sink_output(3.0)\n",
        ))
        .expect("later same-name valid evaluation must recover");
        let output = valid
            .roots
            .iter()
            .find(|root| root.name.as_deref() == Some("output"))
            .expect("output root");
        assert_eq!(
            serde_json::to_value(&output.value).expect("stored scalar"),
            serde_json::json!({"type": "scalar", "value": {"dtype": "f32", "bits": "40a00000"}})
        );
    }
}
