use effect_data::{EffectError, EffectErrorKind};
use chelis_compiler_api::schema::{CheckRequest, Diagnostic, SourceKind};
use chelis_compiler_api::schema::numbers::UnitInterval;
use serde_json::{Value, json};

fn diagnostic(value: &Diagnostic) -> Value {
    json!({"wire": value, "kind": value.kind().as_str(), "message": value.message,
        "severity_bits": value.severity.get().to_bits(), "expected": value.expected,
        "got": value.got, "suggestions": value.suggestions, "span": value.span,
        "deep_path": value.deep_path, "span_id": value.span_id,
        "unsupported_identity": format!("{:?}", value.unsupported_identity())})
}

fn snapshot() {
    let mut conversions = Vec::new();
    for kind in [EffectErrorKind::UnhandledEffect, EffectErrorKind::InvalidHandler,
        EffectErrorKind::BuildTargetMismatch, EffectErrorKind::TypeTotality] {
        for severity in [0.0, 0.25, 0.8, 1.0] {
            let error = EffectError { kind: kind.clone(), message: "effect diagnostic detail".to_string(),
                suggestions: vec!["first repair".to_string(), "second repair".to_string()] };
            let converted = Diagnostic::from_effect_error(&error, UnitInterval::new(severity).unwrap());
            conversions.push(json!({"kind": format!("{:?}", kind),
                "severity_bits": severity.to_bits(), "diagnostic": diagnostic(&converted)}));
        }
    }
    let corpus: Value = serde_json::from_str(CORPUS_JSON).unwrap();
    let mut outcomes = serde_json::Map::new();
    for entry in corpus["cases"].as_array().unwrap() {
        let request = CheckRequest {
            source_kind: if entry["format"] == "deep" { SourceKind::Deep } else { SourceKind::Surf },
            source: entry["source"].as_str().unwrap().to_string(),
        };
        let outcome = match chelis_compiler_api::compiler::check(request) {
            Ok(result) => json!({"status": "reported", "score": result.score,
                "components": result.components, "typed_nodes": result.typed_nodes,
                "untyped_nodes": result.untyped_nodes, "total_nodes": result.total_nodes,
                "unresolved_names": result.unresolved_names,
                "inferred_signatures": result.inferred_signatures,
                "diagnostics": result.errors.iter().map(diagnostic).collect::<Vec<_>>()}),
            Err(error) => json!({"status": "rejected", "stage": error.stage,
                "diagnostics": error.errors.iter().map(diagnostic).collect::<Vec<_>>(),
                "transcript": error.transcript}),
        };
        assert!(outcomes.insert(entry["id"].as_str().unwrap().to_string(), outcome).is_none());
    }
    println!("CHECKER_DIAGNOSTIC_SNAPSHOT {}", json!({"conversions": conversions, "outcomes": outcomes}));
}
fn main() {
    std::thread::Builder::new().stack_size(64 * 1024 * 1024).spawn(snapshot).unwrap()
        .join().expect("the diagnostic snapshot must succeed");
}
