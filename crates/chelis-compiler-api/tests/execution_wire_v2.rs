//! Execution-wire v2 mechanics (chelis#729 Phase 1, section C3's wire
//! layer; the rt857 F5 negative-parity suite). Locks the versioning
//! mechanics recorded at `schema::EXECUTION_VALUE_SCHEMA_VERSION` and
//! the tagged per-dtype payload's exactness both directions.

use chelis_compiler_api::schema::{EvalResult, ExecutionValue, TensorElements, TensorValue};

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
    for version in [1u32, 3, 999] {
        let payload = format!(r#"{{"schema_version":{version},"roots":[]}}"#);
        let err = serde_json::from_str::<EvalResult>(&payload)
            .expect_err("a non-current schema_version must not parse");
        let msg = err.to_string();
        assert!(
            msg.contains("schema_version") && msg.contains(&version.to_string()) && msg.contains('2'),
            "the rejection must name the field, the stale version, and the \
             supported version; got: {msg}"
        );
    }
}

/// A freshly produced result stamps the current version (negative parity
/// for the default above: the field is present on the wire, not elided).
#[test]
fn produced_result_stamps_v2() {
    let result = EvalResult {
        schema_version: chelis_compiler_api::schema::EXECUTION_VALUE_SCHEMA_VERSION,
        roots: vec![],
        transcript: vec![],
    };
    let json = serde_json::to_string(&result).expect("serialize");
    assert!(
        json.contains(r#""schema_version":2"#),
        "the version stamp must be on the wire; got: {json}"
    );
}

/// Exact int64 above 2^53 survives the v2 payload in BOTH directions
/// (the chelis#686 capacity fix this wire break exists for).
#[test]
fn v2_int64_payload_is_exact_above_2p53_both_directions() {
    let tensor = ExecutionValue::Tensor {
        value: TensorValue {
            shape: vec![2],
            data: TensorElements::Int64(vec![9007199254740993, -9007199254740993]),
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
                TensorElements::Int64(vec![9007199254740993, -9007199254740993])
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
        .expect_err("the v1 bare-array payload must not parse as v2");
    let msg = err.to_string();
    assert!(
        msg.contains("invalid type") || msg.contains("expected"),
        "the failure must be a loud serde type error; got: {msg}"
    );
}

/// bool and half payloads keep their tags and values through the wire
/// (f16/bf16 carry exact f64 images; every half value is exactly
/// representable in f64).
#[test]
fn v2_bool_and_half_payloads_round_trip() {
    for data in [
        TensorElements::Bool(vec![true, false]),
        TensorElements::F16(vec![2048.0, 0.0999755859375]),
        TensorElements::Bf16(vec![256.0, 0.75]),
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
