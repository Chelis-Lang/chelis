//! chelis#1479 row 1: `list_dir`'s `[05-HOST-4]` order agrees across lanes.
//!
//! `crates/chelis-compiler-api/tests/issue_1479_list_dir_order.rs` and
//! `crates/chelis-runtime/tests/issue_1479_list_dir_order.rs` each pin one
//! lane against the expected list. Neither can catch the two agreeing on the
//! wrong order, or drifting apart while both stay internally consistent, which
//! is what [05-HOST-1] forbids: a host-runtime operation preserves its value
//! result in every language execution mode.
//!
//! This file is the end-to-end half. Each shape is evaluated by the
//! interpreter AND compiled to C, linked, and run, and the two lanes' stdout
//! must be byte-equal as well as equal to the expected rendering.
//!
//! The rendering is an inlined `fold` over `string_concat`, the same shape
//! `Std.Text.join` uses, rather than `to_string`: the compiled lane rejects
//! `to_string` of a `List(String)` with a typed `Unsupported` receipt that
//! chelis#1059 owns. Prefixing with `names:` keeps the empty-directory case
//! from asserting a vacuously empty string.
//!
//! The fixture path is absolute in both lanes, so the two lanes' differing
//! working directories cannot affect the result. That is also why this does
//! not live in `crates/chelis-cli/tests/parity.rs`, whose eval lane inherits
//! the test process's cwd while its build lane runs from the binary's parent.

#[path = "common/mod.rs"]
mod common;

use assert_cmd::Command;
use common::{build_and_run, gcc_available, write_file};
use std::fs;
use std::path::Path;
use tempfile::tempdir;

/// Escape an absolute host path into a Surf string literal.
fn surf_string_literal(value: &str) -> String {
    format!(
        "\"{}\"",
        value
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t")
    )
}

/// A program that prints `list_dir`'s result for one absolute fixture path,
/// separator-joined so the printed line encodes the order rather than the set.
fn source_for(fixture: &Path) -> String {
    include_str!("../../../examples/io/list_directory.ch").replace(
        "list_dir(\".\")",
        &format!(
            "list_dir({})",
            surf_string_literal(fixture.to_str().expect("UTF-8 fixture path"))
        ),
    )
}

/// Evaluate `source` through the interpreter lane and return its stdout.
fn eval_stdout(source: &str, name: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    write_file(&path, source);
    Command::cargo_bin("chelis")
        .expect("binary")
        .arg("fmt")
        .arg("--inplace")
        .arg(&path)
        .assert()
        .success();
    Command::cargo_bin("chelis")
        .expect("binary")
        .arg("check")
        .arg(&path)
        .assert()
        .success();
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    assert!(
        out.status.success(),
        "eval failed for `{name}`: {}\nstderr:\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr),
    );
    String::from_utf8(out.stdout).expect("utf-8 stdout")
}

/// Create `entries` in the given order, then assert both lanes print
/// `expected_rendering` and agree byte for byte.
fn assert_lane_parity(name: &str, entries: &[&str], expected_rendering: &str) {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("fixture");
    fs::create_dir(&fixture).expect("create fixture directory");
    for entry in entries {
        fs::write(fixture.join(entry), b"x").expect("write fixture entry");
    }

    let source = source_for(&fixture);
    let eval = eval_stdout(&source, name);
    // Exact line equality, not `contains`: the empty-directory row's expected
    // rendering is the bare `names:` prefix, which every non-empty listing also
    // contains, so a substring check would let `[".", ".."]` through.
    assert!(
        eval.lines().any(|line| line == expected_rendering),
        "{name}: eval lane did not print the [05-HOST-4] order\nexpected the exact line: \
         {expected_rendering}\nfull stdout:\n{eval}"
    );

    assert!(
        gcc_available(),
        "this cross-lane oracle requires a C compiler"
    );
    let built = build_and_run(&source, name);
    assert_eq!(
        built, eval,
        "{name}: build lane disagrees with eval lane\nbuild stdout:\n{built}\neval stdout:\n{eval}"
    );
}

/// The entries are created in reverse of the required order, so a lane that
/// forwards the host's enumeration order cannot pass by coincidence on a
/// filesystem that happens to enumerate in creation order.
#[test]
fn both_lanes_order_list_dir_entries_by_name_not_creation_order() {
    assert_lane_parity(
        "issue1479_creation_order",
        &["zulu.txt", "mike.txt", "delta.txt", "alpha.txt"],
        "names:/alpha.txt/delta.txt/mike.txt/zulu.txt",
    );
}

/// '.' is 0x2E and digits are 0x30..0x39, so "10" precedes "2": this is byte
/// order, not numeric order, and not a locale collation.
#[test]
fn both_lanes_order_list_dir_dotfiles_and_digits_by_byte_sequence() {
    assert_lane_parity(
        "issue1479_byte_order",
        &["b", "A", "10", "2", ".hidden"],
        "names:/.hidden/10/2/A/b",
    );
}

/// An empty directory yields the empty `List` on both lanes. The expected line
/// is the bare prefix, which is why `assert_lane_parity` matches a whole line
/// rather than a substring: `contains("names:")` would also accept a listing
/// that wrongly included the `.` and `..` links.
#[test]
fn both_lanes_render_an_empty_directory_as_the_empty_list() {
    assert_lane_parity("issue1479_empty", &[], "names:");
}

#[test]
fn both_lanes_preserve_unicode_list_dir_names() {
    assert_lane_parity(
        "issue1479_unicode",
        &["\u{fffd}", "替", "λ", "ASCII"],
        "names:/ASCII/λ/替/\u{fffd}",
    );
}

/// Invalid host names require a supporting filesystem, not a weaker contract.
/// Linux CI executes these complete eval/build/link/run failure cases.
#[cfg(target_os = "linux")]
#[test]
fn both_lanes_reject_invalid_list_dir_names_without_a_partial_list() {
    use common::link_generated;
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;

    assert!(
        gcc_available(),
        "this cross-lane oracle requires a C compiler"
    );
    let cases: &[(&[&[u8]], &str)] = &[
        (&[b"a\xff", b"a\xfe"], r#"a\xfe"#),
        (&[b"a\xfe", b"a\xff"], r#"a\xfe"#),
        (&[b"\x81a", b"\x80z"], r#"\x80z"#),
        (&[b"bad\n\r\t\\\"'\xff"], r#"bad\n\r\t\\\"\'\xff"#),
    ];
    for (invalid_names, escaped_entry) in cases {
        let dir = tempdir().expect("tempdir");
        let fixture = dir.path().join("directory\n\"\\λ");
        fs::create_dir(&fixture).unwrap();
        fs::write(fixture.join("0-valid"), b"x").unwrap();
        for name in *invalid_names {
            fs::write(fixture.join(OsStr::from_bytes(name)), b"x").unwrap();
        }
        let source = source_for(&fixture);
        let path = dir.path().join("invalid.ch");
        write_file(&path, &source);
        let expected = format!(
            "IO trap in list_dir: directory b\"{}\", entry b\"{escaped_entry}\": name is not valid UTF-8",
            fixture.as_os_str().as_bytes().escape_ascii(),
        );
        let eval = Command::cargo_bin("chelis")
            .unwrap()
            .arg("eval")
            .arg("--file")
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            !eval.status.success(),
            "invalid names returned successfully: {eval:?}"
        );
        assert!(
            eval.stdout.is_empty(),
            "a failed listing must not publish a result: {eval:?}"
        );
        let eval_stderr = String::from_utf8(eval.stderr).unwrap();
        assert!(
            eval_stderr.lines().any(|line| line.ends_with(&expected)),
            "wrong evaluator diagnostic: {eval_stderr}"
        );

        let out = dir.path().join("out");
        Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .arg("build")
            .arg("--emit-c")
            .arg(&path)
            .args(["--target", "c", "--output"])
            .arg(&out)
            .assert()
            .success();
        assert!(link_generated(&out, "invalid.c", "invalid").success());
        let compiled = std::process::Command::new(out.join("invalid"))
            .output()
            .unwrap();
        assert_eq!(compiled.status.code(), Some(1), "{compiled:?}");
        assert!(
            compiled.stdout.is_empty(),
            "compiled listing published a partial result"
        );
        assert_eq!(
            String::from_utf8(compiled.stderr).unwrap(),
            format!("{expected}\n")
        );
    }
}
