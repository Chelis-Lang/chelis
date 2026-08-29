use chelis_reef::{BuildOptions, PackageSchema, build_package_with_options, package_schema};
use chelis_shell::read_shell;
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

fn package_fixture() -> (tempfile::TempDir, std::path::PathBuf) {
    let directory = tempdir().expect("fixture directory must be created");
    let root = directory.path().join("schema-restrictions");
    // Category-1 auto-sync (see `scripts/bump_compiler_pins.py`): a synthesized
    // fixture derives its `compiler` pin from the running crate version rather
    // than hand-pinning it, because `validate_manifest` rejects any pin but the
    // running binary's and a release bump would otherwise leave this test red.
    // Every workspace member inherits `workspace.package.version`, so this is
    // the same string `chelis_compiler_api::COMPILER_VERSION` expands to.
    write(
        &root.join("reef.toml"),
        &format!(
            r#"[package]
name = "schema-restrictions"
version = "1.0.0"
compiler = "={}"
module_prefix = "Restriction"
"#,
            env!("CARGO_PKG_VERSION")
        ),
    );
    write(
        &root.join("src/main.ch"),
        r#"module Restriction.Main
export (restricted_close, unrestricted_identity, integer_identity)
def restricted_close[p_float](actual: &tensor[n, p_float], expected: &tensor[n, p_float], tolerance: p_float) -> unit ! { Test } = test_assert_close_tensor(actual, expected, tolerance, "restricted")
def unrestricted_identity[p](value: &tensor[n, p]) -> &tensor[n, p] = value
def integer_identity(value: int32) -> int32 = value
"#,
    );
    (directory, root)
}

fn exported<'a>(value: &'a Value, name: &str, collection: &str) -> &'a Value {
    value["modules"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|module| module[collection].as_array().unwrap())
        .find(|symbol| symbol["name"] == name)
        .unwrap_or_else(|| panic!("missing exported {collection} entry `{name}`"))
}

fn expected_active_float() -> Value {
    serde_json::json!([{"variable": "t0", "domain": "active_float"}])
}

#[test]
fn public_schema_and_decoded_chb_preserve_exact_scheme_restrictions() {
    let (_directory, root) = package_fixture();
    let schema = package_schema(&root).expect("schema package must check");
    let schema_json = serde_json::to_value(schema).expect("schema must serialize");

    assert_eq!(schema_json["format_version"], 2);
    assert_eq!(
        exported(&schema_json, "restricted_close", "functions")["type_variable_restrictions"],
        expected_active_float()
    );
    for name in ["unrestricted_identity", "integer_identity"] {
        assert_eq!(
            exported(&schema_json, name, "functions")["type_variable_restrictions"],
            serde_json::json!([]),
            "{name} must not acquire a restriction"
        );
    }
    let mut missing_required_field = schema_json.clone();
    missing_required_field["modules"][0]["functions"][0]
        .as_object_mut()
        .unwrap()
        .remove("type_variable_restrictions");
    assert!(
        serde_json::from_value::<PackageSchema>(missing_required_field).is_err(),
        "schema v2 consumers must not silently default a missing restriction ledger"
    );

    let artifacts = build_package_with_options(&root, &BuildOptions { auto_fetch: false })
        .expect("schema package must build");
    let shell = read_shell(&artifacts.shell_path).expect("CHB must decode");
    let shell_json = serde_json::to_value(shell).expect("CHB model must serialize");

    assert_eq!(shell_json["format_version"], 2);
    assert_eq!(
        exported(&shell_json, "restricted_close", "exports")["type_variable_restrictions"],
        expected_active_float()
    );
    for name in ["unrestricted_identity", "integer_identity"] {
        assert_eq!(
            exported(&shell_json, name, "exports")["type_variable_restrictions"],
            serde_json::json!([]),
            "{name} must not acquire a restriction"
        );
    }
}
