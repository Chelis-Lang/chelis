//! Spec-first callable ABI cutover controls; spec/11 §1.4.
//! The descriptor cutover, version codec, producers and consumers must land
//! together. Execution-value, WireDag and DLPack versions are separate.

use chelis_compiler_api::schema::CompiledArtifactManifest;
use serde_json::json;

#[test]
fn callable_v2_preserves_exact_int64_extents() {
    for extent in [0_i64, 1, i64::from(i32::MAX) + 1, i64::MAX] {
        let value = json!({
            "abi_version": 2,
            "target": "c",
            "host_entry_name": "chelis_main",
            "inputs": [{"name": "x", "dtype": "f32", "dims": [{"size": extent}]}],
            "outputs": [],
            "source_path": "absent.ch",
            "source_hash": "fixture"
        });
        let manifest: CompiledArtifactManifest =
            serde_json::from_value(value).expect("valid V2 metadata");
        let encoded = serde_json::to_value(manifest).unwrap();
        assert_eq!(encoded["abi_version"], 2);
        assert_eq!(encoded["inputs"][0]["dims"][0]["size"], extent);
    }
}

#[test]
fn callable_v1_rejects_before_malformed_payload_decoding() {
    let error = serde_json::from_str::<CompiledArtifactManifest>(
        r#"{"abi_version":1,"inputs":"invalid tensor metadata","source_path":42}"#,
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("artifact ABI version"),
        "{error}"
    );
    assert!(error.to_string().contains("unsupported"), "{error}");
}

#[test]
fn callable_v2_rejects_missing_duplicate_and_noninteger_discriminants() {
    for version_fields in [
        "",
        "\"abi_version\":0,",
        "\"abi_version\":3,",
        "\"abi_version\":-1,",
        "\"abi_version\":2.0,",
        "\"abi_version\":true,",
        "\"abi_version\":\"2\",",
        "\"abi_version\":null,",
        "\"abi_version\":4294967296,",
        "\"abi_version\":2,\"abi_version\":2,",
        "\"abi_version\":2,\"abi_version\":1,",
    ] {
        let source = format!("{{{version_fields}\"inputs\":\"invalid tensor metadata\"}}");
        let error = serde_json::from_str::<CompiledArtifactManifest>(&source).unwrap_err();
        assert!(
            error.to_string().contains("artifact ABI version"),
            "{error}"
        );
    }
}
