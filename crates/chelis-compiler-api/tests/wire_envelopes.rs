//! spec/10 §3: exact versions gate typed payload decoding without erasing keys.

use chelis_compiler_api::schema::{
    EXECUTION_VALUE_SCHEMA_VERSION, EvalResult, WIRE_DAG_SCHEMA_VERSION, WireDag,
};

#[test]
fn execution_version_is_checked_before_values_in_either_field_order() {
    for version in ["", "\"schema_version\":2,", "\"schema_version\":4,"] {
        for json in [
            format!("{{{version}\"roots\":[{{\"value\":{{\"type\":\"unknown\"}}}}]}}"),
            format!(
                "{{\"roots\":[{{\"value\":{{\"type\":\"unknown\"}}}}],{version}\"transcript\":[]}}"
            ),
        ] {
            let error = serde_json::from_str::<EvalResult>(&json).unwrap_err();
            assert!(
                error.to_string().contains("schema_version"),
                "{json}: {error}"
            );
        }
    }
}

#[test]
fn current_envelopes_accept_reordered_fields_and_reject_duplicate_versions() {
    let eval = format!("{{\"roots\":[],\"schema_version\":{EXECUTION_VALUE_SCHEMA_VERSION}}}");
    let dag = format!("{{\"roots\":[],\"nodes\":[],\"schema_version\":{WIRE_DAG_SCHEMA_VERSION}}}");
    assert!(serde_json::from_str::<EvalResult>(&eval).is_ok());
    assert!(serde_json::from_str::<WireDag>(&dag).is_ok());
    assert!(WireDag::from_validated_json(&dag).is_ok());
    let duplicate_eval = eval.replacen(
        '{',
        &format!("{{\"schema_version\":{EXECUTION_VALUE_SCHEMA_VERSION},"),
        1,
    );
    let duplicate_dag = dag.replacen(
        '{',
        &format!("{{\"schema_version\":{WIRE_DAG_SCHEMA_VERSION},"),
        1,
    );
    assert!(serde_json::from_str::<EvalResult>(&duplicate_eval).is_err());
    assert!(serde_json::from_str::<WireDag>(&duplicate_dag).is_err());
    assert!(WireDag::from_validated_json(&duplicate_dag).is_err());
}

#[test]
fn dag_envelope_preserves_duplicate_payload_fields_for_rejection() {
    let payload = |value: &str| {
        format!(
            r#"{{"schema_version":{WIRE_DAG_SCHEMA_VERSION},"nodes":[{{"shape_deps":[],"span_id":null,"merged_spans":[],"id":0,"op":{{"kind":"const","value":{value}}},"inputs":[],"output_type":{{"dims":[],"precision":"f64"}}}}],"roots":[0]}}"#
        )
    };
    let valid = payload(r#"{"dtype":"f64","bits":"8000000000000000"}"#);
    assert!(serde_json::from_str::<WireDag>(&valid).is_ok());
    for value in [
        r#"{"dtype":"f64","dtype":"f64","bits":"8000000000000000"}"#,
        r#"{"dtype":"f64","bits":"8000000000000000","bits":"8000000000000000"}"#,
    ] {
        let input = payload(value);
        assert!(
            serde_json::from_str::<WireDag>(&input).is_err(),
            "accepted {input}"
        );
        assert!(
            WireDag::from_validated_json(&input).is_err(),
            "validated {input}"
        );
    }
}

#[test]
fn producers_cannot_emit_wrong_version_execution_envelopes() {
    let mut result: EvalResult = serde_json::from_str(&format!(
        "{{\"schema_version\":{EXECUTION_VALUE_SCHEMA_VERSION},\"roots\":[]}}"
    ))
    .unwrap();
    assert!(serde_json::to_string(&result).is_ok());
    result.schema_version = EXECUTION_VALUE_SCHEMA_VERSION - 1;
    assert!(serde_json::to_string(&result).is_err());
}

#[test]
fn batch_envelopes_execute_the_same_version_and_duplicate_controls() {
    use chelis_compiler_api::schema::WireBatchResultEnvelope;
    let batch =
        |eval: &str| format!(r#"{{"results":[{{"kind":"eval","ok":true,"result":{eval}}}]}}"#);
    let valid = batch(r#"{"schema_version":3,"roots":[]}"#);
    let result = serde_json::from_str::<WireBatchResultEnvelope>(&valid);
    assert!(result.is_ok(), "{valid}: {result:?}");
    for eval in [
        r#"{"roots":[],"schema_version":2}"#,
        r#"{"roots":[],"schema_version":3,"schema_version":3}"#,
    ] {
        assert!(serde_json::from_str::<WireBatchResultEnvelope>(&batch(eval)).is_err());
    }
}
