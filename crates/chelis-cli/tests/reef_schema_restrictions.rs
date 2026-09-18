use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("fixture parent must be created");
    }
    fs::write(path, contents).expect("fixture file must be written");
}

#[test]
fn reef_schema_json_preserves_the_canonical_active_float_restriction() {
    let directory = tempdir().expect("fixture directory must be created");
    let root = directory.path().join("schema-restrictions");
    // Category-1 auto-sync (see `scripts/bump_compiler_pins.py`): derive the
    // `compiler` pin from `COMPILER_VERSION` rather than hand-pinning it, so a
    // release bump does not leave this fixture rejected by `validate_manifest`.
    write(
        &root.join("reef.toml"),
        &format!(
            r#"[package]
name = "schema-restrictions"
version = "1.0.0"
compiler = "={}"
module_prefix = "Restriction"
"#,
            chelis_compiler_api::COMPILER_VERSION
        ),
    );
    write(
        &root.join("src/main.ch"),
        r#"module Restriction.Main
export (restricted_close)
def restricted_close[n, p_float: Float](actual: &tensor[n, p_float], expected: &tensor[n, p_float], tolerance: p_float) -> unit ! { Test } = test_assert_close_tensor(actual, expected, tolerance, "restricted")
"#,
    );

    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["reef", "schema", root.to_str().unwrap()])
        .output()
        .expect("reef schema must run");
    assert!(
        output.status.success(),
        "reef schema failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let schema: Value = serde_json::from_slice(&output.stdout).expect("schema JSON");
    // chelis#1654 bumped the public JSON schema to 3 for the obligations.
    assert_eq!(schema["format_version"], 3);
    assert_eq!(
        schema["modules"][0]["functions"][0]["type_variable_restrictions"],
        serde_json::json!([{"variable": "t0", "domain": "active_float"}])
    );
}

#[test]
fn reef_schema_json_preserves_a_checked_collection_relation() {
    let directory = tempdir().expect("fixture directory must be created");
    let root = directory.path().join("schema-collection-contract");
    write(
        &root.join("reef.toml"),
        &format!(
            r#"[package]
name = "schema-collection-contract"
version = "1.0.0"
compiler = "={}"
module_prefix = "Relation"
"#,
            chelis_compiler_api::COMPILER_VERSION
        ),
    );
    write(
        &root.join("src/main.ch"),
        r#"module Relation.Main
export (measure)
measure = len
"#,
    );

    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["reef", "schema", root.to_str().unwrap()])
        .output()
        .expect("reef schema must run");
    assert!(
        output.status.success(),
        "reef schema failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let schema: Value = serde_json::from_slice(&output.stdout).expect("schema JSON");
    let function = &schema["modules"][0]["functions"][0];
    assert_eq!(schema["format_version"], 3);
    assert_eq!(
        function["collection_obligations"],
        serde_json::json!([{
            "len": {
                "operand": "(t-var {} t0)",
                "result": "(t-var {} t1)"
            }
        }])
    );
}
