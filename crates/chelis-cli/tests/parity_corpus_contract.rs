//! chelis#1868: source-derived parity input membership, not execution proof.

#[path = "common/parity_corpus.rs"]
mod parity_corpus;

use std::collections::BTreeSet;
use std::fs;
use tempfile::tempdir;

const CASE: &str = r#"
#[test]
fn example_case() {
    drive_parity(&examples_root().join("example.ch"), true);
}
"#;

fn names(values: &[&str]) -> BTreeSet<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

#[test]
fn discovery_keeps_the_executable_root_and_excludes_illustrative_subtrees() {
    let directory = tempdir().unwrap();
    fs::write(directory.path().join("example.ch"), "").unwrap();
    fs::write(directory.path().join("README.md"), "").unwrap();
    fs::create_dir(directory.path().join("illustrative")).unwrap();
    fs::write(directory.path().join("illustrative/future.ch"), "").unwrap();
    fs::create_dir(directory.path().join("directory.ch")).unwrap();
    assert_eq!(
        parity_corpus::discover(directory.path()).unwrap(),
        names(&["example.ch"])
    );
    parity_corpus::validate(directory.path(), CASE).unwrap();
}

#[test]
fn adding_an_example_requires_its_real_test_input_without_updating_a_list() {
    let directory = tempdir().unwrap();
    fs::write(directory.path().join("example.ch"), "").unwrap();
    parity_corpus::validate(directory.path(), CASE).unwrap();
    fs::write(directory.path().join("added.ch"), "").unwrap();
    assert!(
        parity_corpus::validate(directory.path(), CASE)
            .unwrap_err()
            .contains("added.ch")
    );
    let added = CASE
        .replace("example_case", "added_case")
        .replace("example.ch", "added.ch");
    parity_corpus::validate(directory.path(), &format!("{CASE}\n{added}")).unwrap();
}

#[test]
fn missing_example_empty_directory_and_unreadable_root_fail() {
    let directory = tempdir().unwrap();
    assert!(parity_corpus::validate(directory.path(), CASE).is_err());
    assert!(parity_corpus::validate(&directory.path().join("absent"), CASE).is_err());
    fs::write(directory.path().join("different.ch"), "").unwrap();
    let error = parity_corpus::validate(directory.path(), CASE).unwrap_err();
    assert!(
        error.contains("example.ch") && error.contains("different.ch"),
        "{error}"
    );
}

#[test]
fn comments_literals_helpers_and_removed_test_attributes_cannot_supply_inputs() {
    for source in [
        format!("/* outer /* nested */ {CASE} */"),
        format!("const TEXT: &str = r###\"{CASE}\"###;"),
        CASE.replace("#[test]", ""),
        CASE.replace(
            "drive_parity(&examples_root().join(\"example.ch\"), true);",
            "",
        ),
        "#[test] fn empty() { let _ = \"example.ch\"; }".to_owned(),
        "#[test] fn unused() { let path = examples_root().join(\"example.ch\"); }".to_owned(),
    ] {
        assert!(
            parity_corpus::declared_inputs(&source).unwrap().is_empty(),
            "{source}"
        );
    }
}

#[test]
fn immutable_paths_and_existing_rejection_input_forms_are_derived() {
    let source = r#"
        #[test] fn parity() {
            let path = examples_root().join("value.ch");
            drive_parity(&path, true);
        }
        #[test] fn library() { drive_parity(&examples_root().join("library.ch"), false); }
        #[test] fn softmax() {
            let path = examples_root().join("softmax.ch");
            assert_check_clean(&path);
            assert_eq!(run_eval(&path), b"expected");
        }
        #[test] fn dropout() { check_dropout_eval_and_c_rejection("dropout.ch", b"expected"); }
    "#;
    assert_eq!(
        parity_corpus::declared_inputs(source).unwrap(),
        names(&["value.ch", "library.ch", "softmax.ch", "dropout.ch"])
    );
}

#[test]
fn conditional_declarations_and_nested_calls_do_not_supply_unconditional_inputs() {
    for attribute in ["#[cfg(any())]", "#[cfg_attr(any(), ignore)]"] {
        assert!(parity_corpus::declared_inputs(&format!("{attribute}\n{CASE}")).is_err());
    }
    assert!(parity_corpus::declared_inputs(&format!("#![cfg(any())]\n{CASE}")).is_err());
    for source in [
        "#[test] fn conditional() { if false { drive_parity(&examples_root().join(\"example.ch\"), true); } }",
        "#[test] fn deferred() { let _ = || drive_parity(&examples_root().join(\"example.ch\"), true); }",
    ] {
        assert!(parity_corpus::declared_inputs(source).unwrap().is_empty());
    }
}

#[test]
fn ignored_membership_retains_the_separate_ignore_ledger_obligation() {
    let source = format!("#[ignore = \"requires system cblas.h\"]\n{CASE}");
    assert_eq!(
        parity_corpus::declared_inputs(&source).unwrap(),
        names(&["example.ch"])
    );
    assert_eq!(
        parity_corpus::declared_inputs(&format!("/// A documented case.\n{CASE}")).unwrap(),
        names(&["example.ch"])
    );
}

#[test]
fn unresolved_mutable_reassigned_and_shadowed_inputs_fail_closed() {
    for body in [
        "drive_parity(&unknown, true);",
        "let mut path = examples_root().join(\"example.ch\"); drive_parity(&path, true);",
        "let path = examples_root().join(\"example.ch\"); path = other(); drive_parity(&path, true);",
        "let drive_parity = other; drive_parity(&examples_root().join(\"example.ch\"), true);",
        "fn drive_parity(_: &std::path::Path, _: bool) {} drive_parity(&examples_root().join(\"example.ch\"), true);",
        "drive_parity(&examples_root().join(\"example.ch\"), true); fn drive_parity(_: &std::path::Path, _: bool) {}",
        "let examples_root = other; drive_parity(&examples_root().join(\"example.ch\"), true);",
        "drive_parity(&examples_root().join(\"example.ch\"), true); fn examples_root() -> std::path::PathBuf { other() }",
    ] {
        let source = format!("#[test] fn invalid() {{ {body} }}");
        assert!(parity_corpus::declared_inputs(&source).is_err(), "{source}");
    }
}

#[test]
fn malformed_source_nonliteral_and_nonroot_paths_fail_closed() {
    for source in [
        "#[test] fn broken( {",
        "#[test] fn repeated() {} #[test] fn repeated() {}",
    ] {
        assert!(parity_corpus::declared_inputs(source).is_err());
    }
    for filename in [
        "../outside.ch",
        "illustrative/future.ch",
        "/outside.ch",
        "notes.md",
        "",
    ] {
        let source = CASE.replace("example.ch", filename);
        assert!(parity_corpus::declared_inputs(&source).is_err(), "{source}");
    }
    assert!(parity_corpus::declared_inputs(&CASE.replace("\"example.ch\"", "NAME")).is_err());
}

#[test]
fn shipped_corpus_is_derived_and_each_removed_input_is_detected() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples");
    let source = include_str!("parity.rs");
    parity_corpus::validate(&root, source).unwrap();
    let actual = parity_corpus::discover(&root).unwrap();
    assert_eq!(parity_corpus::declared_inputs(source).unwrap(), actual);
    for filename in actual {
        let missing = source.replace(&format!("\"{filename}\""), "\"removed-input.ch\"");
        assert!(
            parity_corpus::validate(&root, &missing)
                .unwrap_err()
                .contains(&filename),
            "{filename}"
        );
    }
}

// APFS rejects the fixture before the scanner can observe it; required Linux CI owns it.
#[cfg(target_os = "linux")]
#[test]
fn non_utf8_executable_filename_is_an_error_instead_of_an_omission() {
    use std::os::unix::ffi::OsStrExt;
    let directory = tempdir().unwrap();
    fs::write(
        directory
            .path()
            .join(std::ffi::OsStr::from_bytes(b"invalid\xff.ch")),
        "",
    )
    .unwrap();
    assert!(
        parity_corpus::discover(directory.path())
            .unwrap_err()
            .contains("UTF-8")
    );
}
