use std::fs;
use std::path::Path;

use chelis_repr_inventory::discover_production_rust_sources;
use tempfile::tempdir;

fn write(root: &Path, relative: &str, source: &str) {
    let path = root.join(relative);
    fs::create_dir_all(path.parent().expect("fixture parent")).expect("create fixture parent");
    fs::write(path, source).expect("write fixture");
}

#[test]
fn production_graph_excludes_cfg_test_items_and_test_only_source() {
    let fixture = tempdir().expect("temporary fixture");
    write(
        fixture.path(),
        "crates/example/src/lib.rs",
        r#"
mod live;

#[cfg(test)]
mod tests;

#[cfg(test)]
fn test_only_item() {
    unimplemented_rejection!(600, "test-only item");
}

struct ProductionCarrier {
    #[cfg(test)]
    test_only_field: unimplemented_rejection!(424244, "test-only field"),
    live: u32,
}

fn production_item() {
    #[cfg(test)]
    unimplemented_rejection!(705, "test-only statement");
    #[cfg(test)]
    {
        unimplemented_rejection!(424242, "test-only expression statement");
    }
    let _ = match 0 {
        #[cfg(test)]
        0 => unimplemented_rejection!(424243, "test-only match arm"),
        _ => 1,
    };
    unimplemented_rejection!(879, "production item");
}
"#,
    );
    write(
        fixture.path(),
        "crates/example/src/live.rs",
        "fn live() { unimplemented_rejection!(912, \"live module\"); }\n",
    );
    write(
        fixture.path(),
        "crates/example/src/tests.rs",
        "fn test_only() { unimplemented_rejection!(689, \"test-only module\"); }\n",
    );

    let sources =
        discover_production_rust_sources(fixture.path(), &["crates/example/src/lib.rs".into()])
            .expect("discover production graph");
    let paths = sources
        .iter()
        .map(|source| source.path.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        paths,
        ["crates/example/src/lib.rs", "crates/example/src/live.rs"]
    );
    let root = &sources[0].source;
    assert!(!root.contains("test-only item"));
    assert!(!root.contains("test-only statement"));
    assert!(!root.contains("test-only expression statement"));
    assert!(!root.contains("test-only match arm"));
    assert!(!root.contains("test-only field"));
    assert!(root.contains("production item"));
}

#[test]
fn cfg_test_include_and_macro_wiring_are_filtered_before_validation() {
    let fixture = tempdir().expect("temporary fixture");
    write(
        fixture.path(),
        "crates/example/src/lib.rs",
        r#"
#[cfg(test)]
mod tests {
    include!("missing_test_only.rs");

    macro_rules! hidden_module {
        () => { mod hidden; };
    }
    hidden_module!();
}

fn production_item() {
    #[cfg(test)]
    include!("missing_expression_only.rs");
    unimplemented_rejection!(879, "production item");
}
"#,
    );

    let sources =
        discover_production_rust_sources(fixture.path(), &["crates/example/src/lib.rs".into()])
            .expect("test-only wiring must be absent from the production graph");
    assert_eq!(sources.len(), 1);
    assert!(!sources[0].source.contains("missing_test_only.rs"));
    assert!(!sources[0].source.contains("missing_expression_only.rs"));
    assert!(!sources[0].source.contains("hidden_module"));
    assert!(sources[0].source.contains("production item"));
}

#[test]
fn production_path_module_outside_target_directory_is_included() {
    let fixture = tempdir().expect("temporary fixture");
    write(
        fixture.path(),
        "crates/example/src/lib.rs",
        r#"
#[path = "../../../tests/support/outside.rs"]
mod outside;
"#,
    );
    write(
        fixture.path(),
        "tests/support/outside.rs",
        "fn outside(issue: u32) {\n    chelis_types::unsupported::__build_unimplemented_rejection(issue, \"dynamic\");\n}\n",
    );

    let sources =
        discover_production_rust_sources(fixture.path(), &["crates/example/src/lib.rs".into()])
            .expect("discover production graph");
    assert_eq!(
        sources
            .iter()
            .map(|source| source.path.as_str())
            .collect::<Vec<_>>(),
        ["crates/example/src/lib.rs", "tests/support/outside.rs",]
    );
    assert!(
        sources[1]
            .source
            .contains("__build_unimplemented_rejection")
    );
}

#[test]
fn ambiguous_standard_module_resolution_fails_closed() {
    let fixture = tempdir().expect("temporary fixture");
    write(
        fixture.path(),
        "crates/example/src/lib.rs",
        "mod doubled;\n",
    );
    write(fixture.path(), "crates/example/src/doubled.rs", "");
    write(fixture.path(), "crates/example/src/doubled/mod.rs", "");

    let error =
        discover_production_rust_sources(fixture.path(), &["crates/example/src/lib.rs".into()])
            .expect_err("ambiguous module must fail");
    assert!(error.message.contains("ambiguous"), "{}", error.message);
}

#[test]
fn malformed_or_conditional_path_resolution_fails_closed() {
    for source in [
        "#[path = concat!(\"outside\", \".rs\")]\nmod outside;\n",
        "#[cfg_attr(feature = \"alternate\", path = \"alternate.rs\")]\nmod outside;\n",
    ] {
        let fixture = tempdir().expect("temporary fixture");
        write(fixture.path(), "crates/example/src/lib.rs", source);
        let error =
            discover_production_rust_sources(fixture.path(), &["crates/example/src/lib.rs".into()])
                .expect_err("unsupported path wiring must fail");
        assert!(
            error.message.contains("path") || error.message.contains("parse"),
            "{}",
            error.message
        );
    }
}

#[test]
fn macro_generated_module_wiring_fails_closed() {
    let fixture = tempdir().expect("temporary fixture");
    write(
        fixture.path(),
        "crates/example/src/lib.rs",
        r#"
macro_rules! hidden_module {
    () => { mod hidden; };
}
hidden_module!();
"#,
    );
    write(fixture.path(), "crates/example/src/hidden.rs", "");

    let error =
        discover_production_rust_sources(fixture.path(), &["crates/example/src/lib.rs".into()])
            .expect_err("macro-generated module wiring must fail");
    assert!(
        error.message.contains("macro_rules!") && error.message.contains("module wiring"),
        "{}",
        error.message
    );
}
