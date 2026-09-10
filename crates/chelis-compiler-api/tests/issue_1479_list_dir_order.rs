//! chelis#1479 row 1: `list_dir` returns entries in `[05-HOST-4]` order.
//!
//! `spec/05-risc-primitives.md` [05-HOST-4] orders the result by the byte
//! sequence of the entry name the host reports, fixed before any conversion to
//! `string`. Before this, both lanes returned raw `fs::read_dir` order, which
//! differs between filesystems, so the same directory gave a different value on
//! two machines holding identical files.
//!
//! Each fixture below is created in an order that is not the required one, so a
//! lane that forwards host enumeration order cannot pass by coincidence. The
//! sibling runtime-lane test is
//! `crates/chelis-runtime/tests/issue_1479_list_dir_order.rs`, and the
//! cross-lane byte-equality gate is
//! `crates/chelis-cli/tests/issue_1479_list_dir_lane_parity.rs`.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind};
use tempfile::tempdir;

/// Escape an absolute host path into a Surf string literal.
fn surf_string_literal(value: &str) -> String {
    let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

fn list_dir_names(dir: &Path) -> Vec<String> {
    let source = format!(
        "names = list_dir({})\n",
        surf_string_literal(dir.to_str().expect("UTF-8 fixture path"))
    );
    let result = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source,
        bindings: BTreeMap::new(),
    })
    .unwrap_or_else(|err| panic!("eval failed: {err:?}"));

    let root = result
        .roots
        .iter()
        .find(|root| root.name.as_deref() == Some("names"))
        .unwrap_or_else(|| panic!("missing root `names` in {:?}", result.roots));

    match &root.value {
        ExecutionValue::List { value } => value
            .iter()
            .map(|element| match element {
                ExecutionValue::String { value } => value.clone(),
                other => panic!("list_dir element must be a string, got {other:?}"),
            })
            .collect(),
        other => panic!("list_dir must return a List, got {other:?}"),
    }
}

#[test]
fn list_dir_orders_entries_by_name_not_creation_order() {
    let dir = tempdir().expect("tempdir");
    // Written in reverse of the required order, so forwarding the host's
    // enumeration order cannot produce the expected result by accident on a
    // filesystem that happens to enumerate in creation order.
    for name in ["zulu.txt", "mike.txt", "delta.txt", "alpha.txt"] {
        fs::write(dir.path().join(name), b"x").expect("write fixture entry");
    }

    assert_eq!(
        list_dir_names(dir.path()),
        vec!["alpha.txt", "delta.txt", "mike.txt", "zulu.txt"],
    );
}

#[test]
fn list_dir_orders_by_byte_sequence_not_case_insensitively() {
    let dir = tempdir().expect("tempdir");
    // 'M' is 0x4D and 'Z' is 0x5A, both below 'a' at 0x61, so byte order puts
    // every capitalized name first. A case-folding or locale-aware comparison
    // would interleave them as apple, Mango, Zebra.
    //
    // These deliberately are not case variants of one another: the default
    // macOS filesystem is case-insensitive, so a fixture containing both
    // "Banana" and "banana" creates one file, not two.
    for name in ["apple", "Mango", "Zebra"] {
        fs::write(dir.path().join(name), b"x").expect("write fixture entry");
    }

    assert_eq!(list_dir_names(dir.path()), vec!["Mango", "Zebra", "apple"]);
}

#[test]
fn list_dir_orders_dotfiles_and_digits_by_byte_sequence() {
    let dir = tempdir().expect("tempdir");
    // '.' is 0x2E, digits are 0x30..0x39, uppercase 0x41.., lowercase 0x61...
    for name in ["b", "A", "10", "2", ".hidden"] {
        fs::write(dir.path().join(name), b"x").expect("write fixture entry");
    }

    // "10" precedes "2" because this is byte order, not numeric order.
    assert_eq!(
        list_dir_names(dir.path()),
        vec![".hidden", "10", "2", "A", "b"],
    );
}

/// Actual invalid-name filesystem coverage runs on Linux; the pure conversion
/// controls also exercise these cases on macOS without creating host files.
#[cfg(target_os = "linux")]
#[test]
fn list_dir_rejects_non_utf8_names_in_host_order() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;

    let dir = tempdir().expect("tempdir");
    for raw in [
        b"a\xff".as_slice(),
        b"a\xfe".as_slice(),
        b"0-valid".as_slice(),
    ] {
        fs::write(dir.path().join(OsStr::from_bytes(raw)), b"x").expect("write fixture entry");
    }
    let path = dir.path().to_str().unwrap();
    let error = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: format!("names = list_dir({})\n", surf_string_literal(path)),
        bindings: BTreeMap::new(),
    })
    .expect_err("invalid names must reject the entire result");
    let expected = format!(
        r#"IO trap in list_dir: directory b"{}", entry b"a\xfe": name is not valid UTF-8"#,
        path.as_bytes().escape_ascii()
    );
    assert!(
        error
            .errors
            .iter()
            .any(|diagnostic| diagnostic.message == expected),
        "unexpected failure: {error:?}"
    );
}

#[test]
fn list_dir_on_an_empty_directory_returns_the_empty_list() {
    let dir = tempdir().expect("tempdir");
    let empty = dir.path().join("empty");
    fs::create_dir(&empty).expect("create empty fixture directory");

    assert!(list_dir_names(&empty).is_empty());
}

#[test]
fn list_dir_excludes_the_current_and_parent_directory_links() {
    let dir = tempdir().expect("tempdir");
    fs::create_dir(dir.path().join("child")).expect("create child directory");
    fs::write(dir.path().join("file"), b"x").expect("write fixture entry");

    let names = list_dir_names(dir.path());
    assert_eq!(names, vec!["child", "file"]);
    assert!(!names.iter().any(|name| name == "." || name == ".."));
}

#[test]
fn list_dir_returns_entry_names_rather_than_paths() {
    let dir = tempdir().expect("tempdir");
    fs::write(dir.path().join("leaf.txt"), b"x").expect("write fixture entry");

    let names = list_dir_names(dir.path());
    assert_eq!(names, vec!["leaf.txt"]);
    assert!(
        !names[0].contains(std::path::MAIN_SEPARATOR),
        "entry must be a bare name, got {}",
        names[0]
    );
}
