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
export (restricted_close, unrestricted_identity, integer_identity, int_bounded, numeric_bounded, set_bounded, measure)
def restricted_close[n, p_float: Float](actual: &tensor[n, p_float], expected: &tensor[n, p_float], tolerance: p_float) -> unit ! { Test } = test_assert_close_tensor(actual, expected, tolerance, "restricted")
def unrestricted_identity[n, p](value: &tensor[n, p]) -> &tensor[n, p] = value
def integer_identity(value: i32) -> i32 = value
def int_bounded[q: Int](value: q) -> q = value
def numeric_bounded[q: Numeric](value: q) -> q = value
def set_bounded[q: {f32, f64}](value: q) -> q = value
measure = len
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

/// An authored `spec/04-type-system.md` §5.9 bound reaches the published
/// package surface as its own domain, not as the one pre-existing domain.
fn expected_domain(domain: &str) -> Value {
    serde_json::json!([{"variable": "t0", "domain": domain}])
}

/// spec/04-type-system.md §5.9's explicit set publishes the members it admits,
/// in §1.1 declaration order, rather than a family name it does not have
/// (chelis#2443). A family would be a bare string here.
fn expected_set_domain() -> Value {
    serde_json::json!([{"variable": "t0", "domain": {"active_set": ["f32", "f64"]}}])
}

#[test]
fn public_schema_and_decoded_chb_preserve_exact_scheme_restrictions() {
    let (_directory, root) = package_fixture();
    let schema = package_schema(&root, &chelis_std_bundle::EMBEDDED_RUNTIME)
        .expect("schema package must check");
    let schema_json = serde_json::to_value(schema).expect("schema must serialize");

    // chelis#2443 moved this to 4: the `domain` field widened from a string
    // enum to a string-or-object union for §5.9's explicit dtype set.
    assert_eq!(schema_json["format_version"], 4);
    assert_eq!(
        exported(&schema_json, "restricted_close", "functions")["type_variable_restrictions"],
        expected_active_float()
    );
    for (name, domain) in [
        ("int_bounded", "active_int"),
        ("numeric_bounded", "active_numeric"),
    ] {
        assert_eq!(
            exported(&schema_json, name, "functions")["type_variable_restrictions"],
            expected_domain(domain),
            "{name} must publish its authored dtype-family bound"
        );
    }
    assert_eq!(
        exported(&schema_json, "set_bounded", "functions")["type_variable_restrictions"],
        expected_set_domain(),
        "an explicit dtype set must publish its members, not a family name"
    );
    for name in ["unrestricted_identity", "integer_identity"] {
        assert_eq!(
            exported(&schema_json, name, "functions")["type_variable_restrictions"],
            serde_json::json!([]),
            "{name} must not acquire a restriction"
        );
    }
    assert_eq!(
        exported(&schema_json, "measure", "functions")["collection_obligations"],
        serde_json::json!([{
            "len": {
                "operand": "(t-var {} t0)",
                "result": "(t-var {} t1)"
            }
        }])
    );
    let decoded_current = serde_json::from_value::<PackageSchema>(schema_json.clone())
        .expect("current package schema version must decode");
    assert_eq!(
        serde_json::to_value(decoded_current).expect("decoded current schema must reserialize"),
        schema_json,
        "current package schema must round-trip without identity drift"
    );
    // chelis#2443: 3 joins the rejected set now that `domain` may be an object.
    for version in [2, 3, 99] {
        let mut incompatible = schema_json.clone();
        incompatible["format_version"] = version.into();
        let error = serde_json::from_value::<PackageSchema>(incompatible)
            .expect_err("non-current package schema versions must be rejected");
        assert!(
            error.to_string().contains("unsupported") && error.to_string().contains("expected 4"),
            "unexpected version {version} diagnostic: {error}"
        );
    }
    let mut missing_required_field = schema_json.clone();
    missing_required_field["modules"][0]["functions"][0]
        .as_object_mut()
        .unwrap()
        .remove("type_variable_restrictions");
    assert!(
        serde_json::from_value::<PackageSchema>(missing_required_field).is_err(),
        "schema v3 consumers must not silently default a missing restriction ledger"
    );
    let mut missing_collection_ledger = schema_json.clone();
    missing_collection_ledger["modules"][0]["functions"][0]
        .as_object_mut()
        .unwrap()
        .remove("collection_obligations");
    assert!(
        serde_json::from_value::<PackageSchema>(missing_collection_ledger).is_err(),
        "schema v3 consumers must not silently default a missing collection ledger"
    );

    let canonical_append = serde_json::json!({
        "append": {
            "list": "(t-var {} t0)",
            "value": "(t-var {} t1)",
            "result": "(t-var {} t0)"
        }
    });
    let canonical_len = serde_json::json!({
        "len": {
            "operand": "(t-var {} t0)",
            "result": "(t-var {} t1)"
        }
    });
    let mut canonical_multi_relation = schema_json.clone();
    let measure = canonical_multi_relation["modules"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .flat_map(|module| module["functions"].as_array_mut().unwrap())
        .find(|function| function["name"] == "measure")
        .unwrap();
    measure["collection_obligations"] =
        serde_json::json!([canonical_append.clone(), canonical_len.clone()]);
    serde_json::from_value::<PackageSchema>(canonical_multi_relation.clone())
        .expect("a sorted unique canonical relation ledger must deserialize");

    for (label, obligations) in [
        (
            "duplicate",
            serde_json::json!([canonical_len.clone(), canonical_len.clone()]),
        ),
        (
            "noncanonical order",
            serde_json::json!([canonical_len.clone(), canonical_append.clone()]),
        ),
        (
            "noncanonical Deep",
            serde_json::json!([{
                "len": {
                    "operand": "(t-var   {} t0)",
                    "result": "(t-var {} t1)"
                }
            }]),
        ),
    ] {
        let mut invalid = schema_json.clone();
        let measure = invalid["modules"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .flat_map(|module| module["functions"].as_array_mut().unwrap())
            .find(|function| function["name"] == "measure")
            .unwrap();
        measure["collection_obligations"] = obligations;
        let error = serde_json::from_value::<PackageSchema>(invalid)
            .expect_err("noncanonical schema obligation ledger must be rejected");
        assert!(
            error.to_string().contains("collection obligations"),
            "unexpected {label} diagnostic: {error}"
        );
    }

    let hidden = "(t-var {} t99)";
    let t0 = "(t-var {} t0)";
    let t1 = "(t-var {} t1)";
    for (label, obligation) in [
        (
            "len operand",
            serde_json::json!({"len": {"operand": hidden, "result": t1}}),
        ),
        (
            "len result",
            serde_json::json!({"len": {"operand": t0, "result": hidden}}),
        ),
        (
            "index list",
            serde_json::json!({"index": {"list": hidden, "index": t1, "result": t0}}),
        ),
        (
            "index index",
            serde_json::json!({"index": {"list": t0, "index": hidden, "result": t1}}),
        ),
        (
            "index result",
            serde_json::json!({"index": {"list": t0, "index": t1, "result": hidden}}),
        ),
        (
            "append list",
            serde_json::json!({"append": {"list": hidden, "value": t1, "result": t0}}),
        ),
        (
            "append value",
            serde_json::json!({"append": {"list": t0, "value": hidden, "result": t0}}),
        ),
        (
            "append result",
            serde_json::json!({"append": {"list": t0, "value": t1, "result": hidden}}),
        ),
        (
            "concat lhs",
            serde_json::json!({"concat": {"lhs": hidden, "rhs": t1, "result": t0}}),
        ),
        (
            "concat rhs",
            serde_json::json!({"concat": {"lhs": t0, "rhs": hidden, "result": t0}}),
        ),
        (
            "concat result",
            serde_json::json!({"concat": {"lhs": t0, "rhs": t1, "result": hidden}}),
        ),
    ] {
        let mut invalid = schema_json.clone();
        let measure = invalid["modules"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .flat_map(|module| module["functions"].as_array_mut().unwrap())
            .find(|function| function["name"] == "measure")
            .unwrap();
        measure["collection_obligations"] = serde_json::json!([obligation]);
        let error = serde_json::from_value::<PackageSchema>(invalid)
            .expect_err("a hidden collection-contract variable must be rejected");
        assert!(
            error
                .to_string()
                .contains("absent from its type representation"),
            "unexpected {label} diagnostic: {error}"
        );
    }

    for (label, type_repr) in [("malformed", "not Deep"), ("nonfunction", "(t-var {} t0)")] {
        let mut invalid = schema_json.clone();
        let measure = invalid["modules"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .flat_map(|module| module["functions"].as_array_mut().unwrap())
            .find(|function| function["name"] == "measure")
            .unwrap();
        measure["type_repr"] = type_repr.into();
        let error = serde_json::from_value::<PackageSchema>(invalid)
            .expect_err("collection obligations require a valid callable type");
        assert!(
            error.to_string().contains("function type representation"),
            "unexpected {label} diagnostic: {error}"
        );
    }

    let artifacts = build_package_with_options(
        &root,
        &BuildOptions { auto_fetch: false },
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .expect("schema package must build");
    let shell = read_shell(&artifacts.shell_path).expect("CHB must decode");
    let shell_json = serde_json::to_value(shell).expect("CHB model must serialize");

    assert_eq!(
        shell_json["format_version"],
        chelis_shell::SHELL_FORMAT_VERSION
    );
    assert_eq!(
        exported(&shell_json, "set_bounded", "exports")["type_variable_restrictions"],
        expected_set_domain(),
        "a decoded CHB must preserve an explicit dtype set exactly"
    );
    assert_eq!(
        exported(&shell_json, "restricted_close", "exports")["type_variable_restrictions"],
        expected_active_float()
    );
    for (name, domain) in [
        ("int_bounded", "active_int"),
        ("numeric_bounded", "active_numeric"),
    ] {
        assert_eq!(
            exported(&shell_json, name, "exports")["type_variable_restrictions"],
            expected_domain(domain),
            "{name} must round-trip its bound through the published CHB"
        );
    }
    for name in ["unrestricted_identity", "integer_identity"] {
        assert_eq!(
            exported(&shell_json, name, "exports")["type_variable_restrictions"],
            serde_json::json!([]),
            "{name} must not acquire a restriction"
        );
    }
    assert_eq!(
        exported(&shell_json, "measure", "exports")["collection_obligations"],
        serde_json::json!([{
            "len": {
                "operand": "(t-var {} t0)",
                "result": "(t-var {} t1)"
            }
        }])
    );
}
