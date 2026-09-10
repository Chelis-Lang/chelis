//! Report domains and cross-field consistency under [04-FIT-18].

use chelis_compiler_api::schema::{WireCheckResult, WireDiagnostic};
use serde_json::json;

fn report() -> serde_json::Value {
    json!({"score":-0.0,"components":{"parse":1.0,"structure":0.5,"names":0.0,"types":1.0},
        "typed_nodes":9007199254740993_i64,"untyped_nodes":1,"total_nodes":9007199254740994_i64,
        "unresolved_names":[],"errors":[]})
}

fn fitness_report() -> chelis_types::FitnessReport {
    chelis_types::FitnessReport {
        score: -0.0,
        components: chelis_types::fitness::FitnessComponents {
            parse: 1.0,
            structure: 1.0,
            names: 1.0,
            types: 1.0,
        },
        errors: vec![],
        typed_nodes: 2,
        untyped_nodes: 1,
        total_nodes: 3,
        unresolved_names: vec![],
    }
}

#[test]
fn report_document_producer_and_consumer_preserve_typed_values() {
    use chelis_compiler_api::schema::{CheckResult, numbers::NonnegativeCount};

    for bits in [(-0.0_f64).to_bits(), 0x3fb9_52b7_4674_9bc0, 1] {
        let mut fitness = fitness_report();
        fitness.score = f64::from_bits(bits);
        fitness.components.parse = fitness.score;
        let mut producer = CheckResult::try_from_fitness(&fitness).unwrap();
        producer.typed_nodes = NonnegativeCount::new(9_007_199_254_740_993).unwrap();
        producer.total_nodes = NonnegativeCount::new(9_007_199_254_740_994).unwrap();

        for requested in [false, true] {
            producer.inferred_signatures = requested.then(Vec::new);
            let document = producer.to_report_json().unwrap();
            let consumer: WireCheckResult = serde_json::from_str(&document).unwrap();
            assert_eq!(consumer.score.get().to_bits(), bits);
            assert_eq!(consumer.components.parse.get().to_bits(), bits);
            assert_eq!(consumer.typed_nodes.get(), 9_007_199_254_740_993);
            assert_eq!(consumer.untyped_nodes.get(), 1);
            assert_eq!(consumer.total_nodes.get(), 9_007_199_254_740_994);
            assert_eq!(consumer.inferred_signatures.is_some(), requested);
            if requested {
                assert!(consumer.inferred_signatures.unwrap().is_empty());
                assert!(document.contains("\"inferred_signatures\": [],\n  \"errors\": []"));
            } else {
                assert!(!document.contains("inferred_signatures"));
            }
        }
    }
}

#[test]
fn report_document_producer_rejects_inconsistent_counts() {
    use chelis_compiler_api::schema::{CheckResult, numbers::NonnegativeCount};

    let accepted = CheckResult::try_from_fitness(&fitness_report()).unwrap();
    for (typed, untyped) in [(4, 1), (2, 2)] {
        let mut producer = accepted.clone();
        producer.typed_nodes = NonnegativeCount::new(typed).unwrap();
        producer.untyped_nodes = NonnegativeCount::new(untyped).unwrap();
        let error = producer.to_report_json().unwrap_err();
        assert!(error.to_string().contains("inconsistent report counts"));
    }
}

#[test]
fn report_accepts_exact_counts_and_boundary_scores() {
    assert!(serde_json::from_value::<WireCheckResult>(report()).is_ok());
    for score in [0.0, 0.5, 1.0] {
        let mut input = report();
        input["score"] = json!(score);
        assert!(serde_json::from_value::<WireCheckResult>(input).is_ok());
    }
}

#[test]
fn report_rejects_invalid_scores_counts_and_inconsistent_totals() {
    for (field, value) in [
        ("score", json!(-0.1)),
        ("score", json!(1.1)),
        ("score", json!(null)),
        ("typed_nodes", json!(-1)),
        ("total_nodes", json!(u64::MAX)),
        ("untyped_nodes", json!(1.0)),
        ("typed_nodes", json!(9007199254740995_i64)),
        ("untyped_nodes", json!(2)),
    ] {
        let mut input = report();
        input[field] = value;
        assert!(
            serde_json::from_value::<WireCheckResult>(input.clone()).is_err(),
            "accepted {input}"
        );
    }
    for component in ["parse", "structure", "names", "types"] {
        let mut input = report();
        input["components"][component] = json!(1.1);
        assert!(serde_json::from_value::<WireCheckResult>(input).is_err());
    }
    for missing in [
        "score",
        "components",
        "typed_nodes",
        "untyped_nodes",
        "total_nodes",
    ] {
        let mut input = report();
        input.as_object_mut().unwrap().remove(missing);
        assert!(serde_json::from_value::<WireCheckResult>(input).is_err());
    }
}

#[test]
fn diagnostic_severity_uses_the_same_bounded_domain() {
    for severity in [json!(-0.0), json!(0.0), json!(1.0)] {
        assert!(
            serde_json::from_value::<WireDiagnostic>(
                json!({"kind":"type_mismatch","message":"bad type","severity":severity})
            )
            .is_ok()
        );
    }
    for severity in [json!(-0.1), json!(1.1), json!(null), json!("1.0")] {
        assert!(
            serde_json::from_value::<WireDiagnostic>(
                json!({"kind":"type_mismatch","message":"bad type","severity":severity})
            )
            .is_err()
        );
    }
}

#[test]
fn inferred_parameter_references_are_scoped_to_the_owning_ordered_signature() {
    let scalar = json!({"kind":"prim","name":"f64"});
    let signature = json!({"kind":"fn","args":[scalar.clone()],"ret":scalar.clone()});
    let mut input = report();
    input["inferred_signatures"] = json!([{
        "function":"identity","recursive_cycle":false,
        "checked_signature":"f64 -> f64","display_signature":"f64 -> f64",
        "checked_signature_structured":signature.clone(),"display_signature_structured":signature,
        "effect_row":[],"effect_row_display":[],
        "params":[{"index":0,"name":"x","written":false,"inferred_read_only":false,
            "checked_type":"f64","display_type":"f64",
            "checked_type_structured":scalar.clone(),"display_type_structured":scalar}]
    }]);
    assert!(serde_json::from_value::<WireCheckResult>(input.clone()).is_ok());
    for index in [json!(-1), json!(1), json!(u64::MAX), json!(0.0)] {
        input["inferred_signatures"][0]["params"][0]["index"] = index;
        assert!(
            serde_json::from_value::<WireCheckResult>(input.clone()).is_err(),
            "accepted {input}"
        );
    }
}

#[test]
fn producer_rejects_invalid_measurements_instead_of_emitting_repaired_reports() {
    use chelis_compiler_api::schema::{CheckResult, Diagnostic};
    let valid = fitness_report();
    let accepted = CheckResult::try_from_fitness(&valid).unwrap();
    let encoded = serde_json::to_string(&accepted).unwrap();
    let decoded: WireCheckResult = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded.score.get().to_bits(), (-0.0_f64).to_bits());
    for bad in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        let mut input = valid.clone();
        input.score = bad;
        assert!(CheckResult::try_from_fitness(&input).is_err());
        let mut error = chelis_types::errors::CheckError::new(
            chelis_types::errors::CheckErrorKind::Other,
            "bad".into(),
            vec![],
        );
        error.severity = bad;
        assert!(Diagnostic::try_from_check_error(&error).is_err());
    }
    let mut inconsistent = valid.clone();
    inconsistent.untyped_nodes = 2;
    assert!(CheckResult::try_from_fitness(&inconsistent).is_err());
    let mut post_mutation = accepted.clone();
    post_mutation.untyped_nodes =
        chelis_compiler_api::schema::numbers::NonnegativeCount::new(2).unwrap();
    assert!(serde_json::to_string(&post_mutation).is_err());
    if usize::BITS == 64 {
        let mut overflow = valid;
        overflow.total_nodes = usize::MAX;
        assert!(CheckResult::try_from_fitness(&overflow).is_err());
    }
}
