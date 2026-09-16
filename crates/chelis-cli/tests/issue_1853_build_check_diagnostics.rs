//! chelis#1853: `chelis build` on a program the checker rejects rendered the
//! rejection as a Rust `Debug` dump of `Vec<CheckError>`.
//!
//! spec/04 [04-FIT-26] requires a textual rendering to carry each diagnostic's
//! projected `kind`, `message` and location, one line per diagnostic, and
//! never a debug rendering. The oracle for "projected" is `chelis check`'s own
//! report for the same file: every rendered line is compared with the
//! matching element of its `errors` array.

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

/// The issue's reproducer: one checker diagnostic.
const ONE_ERROR: &str = "def f(x: tensor[3, f32]) -> tensor[4, f32] = x\n\
                         out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n";

/// The control: several diagnostics, some located and carrying suggestions.
const SEVERAL_ERRORS: &str = "def f() -> int32 = missing_first\n\
                              def g() -> int32 = missing_second\n\
                              out = f()\n";

/// The debug-rendering markers [04-FIT-26] keeps off the textual surface.
const DEBUG_MARKERS: [&str; 6] = [
    "CheckError {",
    "span_offset",
    "severity:",
    "None",
    "Some(",
    "suggestions: [",
];

struct Run {
    ok: bool,
    stdout: String,
    stderr: String,
}

fn chelis(args: &[&str]) -> Run {
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .args(args)
        .output()
        .expect("chelis should run");
    Run {
        ok: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// `chelis check`'s published `errors` for `program`.
fn check_errors(path: &str) -> Vec<serde_json::Value> {
    let run = chelis(&["check", path]);
    assert!(
        !run.ok,
        "the fixture must be rejected by check:\n{}",
        run.stdout
    );
    let report: serde_json::Value = serde_json::from_str(&run.stdout).expect("check report JSON");
    report["errors"].as_array().expect("errors array").clone()
}

/// The line check's projection determines for one diagnostic.
fn expected_line(error: &serde_json::Value) -> String {
    let mut line = format!(
        "  {}: {}",
        error["kind"].as_str().expect("kind"),
        error["message"].as_str().expect("message")
    );
    if let Some(span) = error.get("span") {
        let offset = span["offset"].as_u64().expect("offset");
        match span.get("len").and_then(serde_json::Value::as_u64) {
            Some(len) => line.push_str(&format!(" at byte {offset} (length {len})")),
            None => line.push_str(&format!(" at byte {offset}")),
        }
    }
    if let Some(span_id) = error.get("span_id").and_then(serde_json::Value::as_str) {
        line.push_str(&format!(" [{span_id}]"));
    }
    for suggestion in error
        .get("suggestions")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
    {
        line.push_str(&format!(
            " (suggestion: {})",
            suggestion.as_str().expect("suggestion")
        ));
    }
    line
}

fn assert_build_renders_check_diagnostics(program: &str, name: &str, expected_count: usize) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join("out");
    write_file(&path, program);
    let path = path.to_str().unwrap();

    let errors = check_errors(path);
    assert_eq!(
        errors.len(),
        expected_count,
        "check's diagnostics: {errors:?}"
    );

    let build = chelis(&[
        "build",
        path,
        "--target",
        "c",
        "-o",
        out_dir.to_str().unwrap(),
    ]);
    assert!(!build.ok, "a rejected program must fail the build");
    assert!(build.stdout.is_empty(), "stdout: {}", build.stdout);
    for marker in DEBUG_MARKERS {
        assert!(
            !build.stderr.contains(marker),
            "debug marker {marker:?} in build stderr:\n{}",
            build.stderr
        );
    }
    let expected = errors
        .iter()
        .map(expected_line)
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(
        build.stderr,
        format!("error: Check errors: Type errors:\n{expected}\n"),
        "build stderr must be check's diagnostics, one line each"
    );
}

#[test]
fn a_rejected_build_renders_one_projected_line_for_its_diagnostic() {
    assert_build_renders_check_diagnostics(ONE_ERROR, "one_error", 1);
}

#[test]
fn a_rejected_build_renders_one_line_per_diagnostic_in_check_order() {
    assert_build_renders_check_diagnostics(SEVERAL_ERRORS, "several_errors", 4);
}

/// Negative parity: an accepted program emits no diagnostic at all.
#[test]
fn an_accepted_build_renders_no_diagnostics() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("accepted.ch");
    let out_dir = dir.path().join("out");
    write_file(
        &path,
        "def f(x: tensor[3, f32]) -> tensor[3, f32] = x\n\
         out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n",
    );
    let build = chelis(&[
        "build",
        path.to_str().unwrap(),
        "--target",
        "c",
        "-o",
        out_dir.to_str().unwrap(),
    ]);
    assert!(build.ok, "the control must build:\n{}", build.stderr);
    assert!(!build.stderr.contains("Type errors"), "{}", build.stderr);
}

/// `chelis deep --annotate` stops on the same rejection and renders it the
/// same way.
#[test]
fn deep_annotate_renders_the_rejection_without_a_debug_dump() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("annotate.ch");
    write_file(&path, SEVERAL_ERRORS);
    let path = path.to_str().unwrap();
    let errors = check_errors(path);

    let run = chelis(&["deep", "--annotate", path]);
    assert!(!run.ok, "a rejected program cannot be annotated");
    for marker in DEBUG_MARKERS {
        assert!(
            !run.stderr.contains(marker),
            "{marker:?} in:\n{}",
            run.stderr
        );
    }
    for error in &errors {
        assert!(
            run.stderr.contains(&format!("{}\n", expected_line(error))),
            "missing {:?} in:\n{}",
            expected_line(error),
            run.stderr
        );
    }
}
