//! Execution-wire version migration mechanics (chelis#729 Phase 1, section C3's wire
//! layer; the rt857 F5 negative-parity suite). Locks the versioning
//! mechanics recorded at `schema::EXECUTION_VALUE_SCHEMA_VERSION` and
//! the tagged per-dtype payload's exactness both directions.

#[path = "../../../tests/support/wire_values.rs"]
mod wire_values;

use chelis_compiler_api::schema::{EvalResult, ExecutionValue, TensorValue};

/// A version-less result payload is REJECTED naming the field (the v1
/// compat default was deleted at the chelis#729 rework: every
/// reader/writer is in-repo, and per the chelis#730 closed-vocabulary
/// doctrine a closed surface has no `Default`).
#[test]
fn versionless_result_is_rejected_naming_the_field() {
    let err = serde_json::from_str::<EvalResult>(r#"{"roots":[]}"#)
        .expect_err("a version-less payload must not parse");
    assert!(
        err.to_string().contains("schema_version"),
        "the rejection must name the missing field; got: {err}"
    );
}

/// A stale `schema_version` (1, or anything not the current constant) is
/// rejected with a message naming the field and both versions.
#[test]
fn stale_or_unknown_schema_version_is_rejected_naming_the_field() {
    for version in [1u32, 2, 3, 999] {
        let payload = format!(r#"{{"schema_version":{version},"roots":[]}}"#);
        let err = serde_json::from_str::<EvalResult>(&payload)
            .expect_err("a non-current schema_version must not parse");
        let msg = err.to_string();
        assert!(
            msg.contains("schema_version")
                && msg.contains(&version.to_string())
                && msg.contains('4'),
            "the rejection must name the field, the stale version, and the \
             supported version; got: {msg}"
        );
    }
}

/// A freshly produced result stamps the current version (negative parity
/// for the default above: the field is present on the wire, not elided).
#[test]
fn produced_result_stamps_v3() {
    let result = EvalResult {
        schema_version: chelis_compiler_api::schema::EXECUTION_VALUE_SCHEMA_VERSION,
        roots: vec![],
        manifest: Default::default(),
        transcript: vec![],
    };
    let json = serde_json::to_string(&result).expect("serialize");
    assert!(
        json.contains(r#""schema_version":4"#),
        "the version stamp must be on the wire; got: {json}"
    );
}

/// Exact int64 above 2^53 survives the v3 payload in BOTH directions
/// (the chelis#686 capacity fix this wire break exists for).
#[test]
fn v3_int64_payload_is_exact_above_2p53_both_directions() {
    let tensor = ExecutionValue::Tensor {
        value: TensorValue {
            shape: vec![2],
            data: wire_values::storage_i64(vec![9007199254740993, -9007199254740993]),
        },
    };
    let json = serde_json::to_string(&tensor).expect("serialize");
    assert!(
        json.contains("9007199254740993"),
        "the exact digits must be on the wire; got: {json}"
    );
    let back: ExecutionValue = serde_json::from_str(&json).expect("parse");
    match back {
        ExecutionValue::Tensor { value } => {
            assert_eq!(
                value.data,
                wire_values::storage_i64(vec![9007199254740993, -9007199254740993])
            );
        }
        other => panic!("round-trip changed the variant: {other:?}"),
    }
}

/// An unknown dtype tag is rejected at deserialization, never guessed.
#[test]
fn unknown_dtype_tag_is_rejected() {
    let err = serde_json::from_str::<TensorValue>(
        r#"{"shape":[1],"data":{"dtype":"u128","values":[1]}}"#,
    )
    .expect_err("unknown dtype tag must be rejected");
    assert!(
        err.to_string().contains("u128") || err.to_string().contains("variant"),
        "the rejection should point at the bad tag; got: {err}"
    );
}

/// The v1 bare-array `data` shape fails LOUDLY (a serde type error at the
/// payload position, per the mechanics note on
/// EXECUTION_VALUE_SCHEMA_VERSION), and is never reinterpreted as some
/// dtype's values.
#[test]
fn v1_bare_array_data_fails_loudly() {
    let err = serde_json::from_str::<TensorValue>(r#"{"shape":[4],"data":[1.0,2.0,3.0,4.0]}"#)
        .expect_err("the v1 bare-array payload must not parse as v3");
    let msg = err.to_string();
    assert!(
        msg.contains("invalid type") || msg.contains("expected"),
        "the failure must be a loud serde type error; got: {msg}"
    );
}

/// bool and half payloads keep their tags and values through the wire
/// (f16/bf16 use their exact stored 16-bit payloads).
#[test]
fn v3_bool_and_half_payloads_round_trip() {
    for data in [
        serde_json::from_value::<chelis_types::TensorStorage>(
            serde_json::json!({"dtype":"bool","values":[true,false]}),
        )
        .unwrap(),
        serde_json::from_value(serde_json::json!({"dtype":"f16","bits":["6800","2e66"]})).unwrap(),
        serde_json::from_value(serde_json::json!({"dtype":"bf16","bits":["4380","3f40"]})).unwrap(),
    ] {
        let tensor = ExecutionValue::Tensor {
            value: TensorValue {
                shape: vec![2],
                data: data.clone(),
            },
        };
        let json = serde_json::to_string(&tensor).expect("serialize");
        let back: ExecutionValue = serde_json::from_str(&json).expect("parse");
        match back {
            ExecutionValue::Tensor { value } => assert_eq!(value.data, data),
            other => panic!("round-trip changed the variant: {other:?}"),
        }
    }
}

/// Numeric scalar leaves are exact-width tagged carriers as well as tensor
/// elements. This locks the serde surface independently of evaluator
/// construction, including the full int64 digits above 2^53.
#[test]
fn v3_numeric_scalar_variants_round_trip_at_every_dtype() {
    let payload = ExecutionValue::List {
        value: vec![
            wire_values::scalar_integer(chelis_types::types::Prim::Int8, -8),
            wire_values::scalar_integer(chelis_types::types::Prim::Int16, -16),
            wire_values::scalar_integer(chelis_types::types::Prim::Int32, -32),
            wire_values::scalar_integer(chelis_types::types::Prim::Int64, 9_007_199_254_740_993),
            serde_json::from_value(
                serde_json::json!({"type":"scalar","value":{"dtype":"f16","bits":"3e00"}}),
            )
            .unwrap(),
            serde_json::from_value(
                serde_json::json!({"type":"scalar","value":{"dtype":"bf16","bits":"3fc0"}}),
            )
            .unwrap(),
            wire_values::scalar_f32(0.25),
            wire_values::scalar_f64(1e100),
        ],
    };
    let json = serde_json::to_string(&payload).expect("serialize scalar carriers");
    assert!(
        json.contains("9007199254740993"),
        "int64 scalar digits must remain exact: {json}"
    );
    let decoded: ExecutionValue = serde_json::from_str(&json).expect("decode scalar carriers");
    assert_eq!(
        serde_json::to_value(decoded).expect("serialize decoded value"),
        serde_json::to_value(payload).expect("serialize original value")
    );
}

/// Negative parity: the type vocabulary is closed. A generic `float` tag is
/// not guessed as f32 or f64.
#[test]
fn v3_unknown_numeric_scalar_tag_is_rejected() {
    let err = serde_json::from_str::<ExecutionValue>(r#"{"type":"float","value":0.5}"#)
        .expect_err("unknown scalar dtype tag must be rejected");
    assert!(
        err.to_string().contains("variant") || err.to_string().contains("float"),
        "the rejection should point at the unknown scalar tag: {err}"
    );
}
