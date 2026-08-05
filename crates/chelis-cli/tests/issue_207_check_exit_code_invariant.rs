//! Issue #207: `chelis check` exit-code invariant.
//!
//! Invariant: `exit_code != 0` iff `json.errors.len() > 0`.
//!
//! Empty errors array implies exit 0; any non-empty errors array implies
//! exit non-zero. The previous (RT-205 F7) contract intentionally
//! exited 0 even with type errors in the JSON, which let downstream CI
//! gates accept programs that `chelis test` later rejected with exit 2.
//! Issue #207 reversed that decision: machine-facing JSON stays the
//! same, but the exit code now matches the contents of `errors[]` so
//! shell-script consumers do not have to parse the JSON to detect a
//! type error.
//!
//! Exit code on errors: `2`, matching the `chelis test` convention for
//! "compile test context" failures (the same kind of error surfaced by
//! the same input on a different surface).
//!
//! Owning code: `cmd_check` and `cmd_check_one` in
//! `crates/chelis-cli/src/main.rs`.

use assert_cmd::Command;
use serde_json::Value;
use std::io::Write;

/// Exit code for `chelis check` when the JSON `errors` array is
/// non-empty. Matches `chelis test`'s exit 2 for "compile test context"
/// failures (the surface that previously caught what `chelis check`
/// missed).
const CHECK_ERRORS_EXIT_CODE: i32 = 2;

fn write_tempfile(prefix: &str, src: &str) -> tempfile::NamedTempFile {
    let mut tmp = tempfile::Builder::new()
        .prefix(prefix)
        .suffix(".ch")
        .tempfile()
        .expect("create tempfile");
    tmp.write_all(src.as_bytes()).expect("write tempfile");
    tmp.flush().expect("flush tempfile");
    tmp
}

fn run_check_capture(path: &std::path::Path) -> (Option<i32>, String) {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().expect("path utf8")])
        .output()
        .expect("run chelis check");
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    (output.status.code(), stdout)
}

fn parse_errors_array(stdout: &str) -> Vec<Value> {
    let json: Value = serde_json::from_str(stdout)
        .unwrap_or_else(|err| panic!("chelis check stdout must be valid JSON: {err}\n{stdout}"));
    json.get("errors")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_else(|| panic!("missing errors array; stdout={stdout}"))
}

/// Issue #207 reproducer: a def body that does not match its declared
/// signature must trip a TypeMismatch and the process must exit
/// non-zero. Pre-fix this exited 0.
#[test]
fn issue_207_check_exits_nonzero_on_type_mismatch() {
    let src = "module Probe.Mismatch\n\
               export (mismatch)\n\
               \n\
               sig mismatch: &tensor[n, 4, f32] -> tensor[n, 4, f32]\n\
               def mismatch(x) = cast(0, f32)\n";
    let tmp = write_tempfile("issue207-tm-", src);
    let (code, stdout) = run_check_capture(tmp.path());
    let errors = parse_errors_array(&stdout);
    assert!(
        !errors.is_empty(),
        "TypeMismatch fixture must produce a non-empty errors array; stdout={stdout}"
    );
    let has_type_mismatch = errors.iter().any(|e| {
        e.get("kind")
            .and_then(Value::as_str)
            .is_some_and(|k| k == "TypeMismatch")
    });
    assert!(
        has_type_mismatch,
        "expected a TypeMismatch entry; stdout={stdout}"
    );
    assert_eq!(
        code,
        Some(CHECK_ERRORS_EXIT_CODE),
        "issue #207 invariant: errors non-empty implies exit {CHECK_ERRORS_EXIT_CODE}; stdout={stdout}"
    );
}

/// Positive control: a clean program must keep exit 0 and an empty
/// errors array. Pins the other side of the iff invariant.
#[test]
fn issue_207_check_exits_zero_on_clean_program() {
    let src = "def answer() -> int32 = cast(7, int32)\n";
    let tmp = write_tempfile("issue207-clean-", src);
    let (code, stdout) = run_check_capture(tmp.path());
    let errors = parse_errors_array(&stdout);
    assert!(
        errors.is_empty(),
        "clean fixture must produce empty errors array; stdout={stdout}"
    );
    assert_eq!(
        code,
        Some(0),
        "issue #207 invariant: errors empty implies exit 0; stdout={stdout}"
    );
}

#[test]
fn pipeline_artifact_semantic_reports_stay_exact() {
    let fixtures = [
        (
            "clean",
            "def answer() -> int32 = cast(7, int32)\n",
            serde_json::json!({
                "score": 1,
                "components": { "parse": 1, "structure": 1, "names": 1, "types": 1 },
                "typed_nodes": 2,
                "untyped_nodes": 0,
                "total_nodes": 2,
                "unresolved_names": [],
                "errors": [],
            }),
            Some(0),
        ),
        (
            "effect",
            "def noisy(x: tensor[4, f32]) -> tensor[4, f32] ! { } = dropout(x, 0.5)\n",
            serde_json::json!({
                "score": 0.8,
                "components": { "parse": 1, "structure": 1, "names": 1, "types": 1 },
                "typed_nodes": 4,
                "untyped_nodes": 0,
                "total_nodes": 4,
                "unresolved_names": [],
                "errors": [{
                    "kind": "UnhandledEffect",
                    "message": "Function `noisy` is declared with effects `{}` but its body performs effects `{Random}` that were not declared",
                    "severity": 0.8,
                }],
            }),
            Some(CHECK_ERRORS_EXIT_CODE),
        ),
        (
            "linearity",
            "def broken(x: tensor[4, f32]) -> tensor[4, f32] = { y = realize(x)\n add(x, y) }\n",
            serde_json::json!({
                "score": 0.8,
                "components": { "parse": 1, "structure": 1, "names": 1, "types": 1 },
                "typed_nodes": 7,
                "untyped_nodes": 0,
                "total_nodes": 7,
                "unresolved_names": [],
                "errors": [{
                    "kind": "UseAfterConsume",
                    "message": "variable `x` was already consumed by realize at surf:56..66; later use at surf:72..73 is invalid",
                    "severity": 0.9,
                }],
            }),
            Some(CHECK_ERRORS_EXIT_CODE),
        ),
    ];

    for (name, source, expected, expected_code) in fixtures {
        let file = write_tempfile(&format!("pipeline-artifact-{name}-"), source);
        let (code, stdout) = run_check_capture(file.path());
        let actual: Value = serde_json::from_str(&stdout).expect("check JSON");
        assert_eq!(actual, expected, "{name} report changed");
        assert_eq!(code, expected_code, "{name} exit code changed");
    }
}

/// Invariant sweep across three distinct error categories. For each
/// fixture the helper asserts `errors_non_empty <=> exit != 0`.
///
/// Categories:
/// * TypeMismatch (def body vs declared sig)
/// * DimensionMismatch (concrete dim literal vs sig)
/// * Validator rejection (conv2d stride 0; same shape RT-205 F7 used,
///   but now exits non-zero per the inverted contract)
#[test]
fn issue_207_invariant_holds_across_error_categories() {
    let fixtures: &[(&str, &str, &str)] = &[
        (
            "tm",
            "module Probe.Mismatch\n\
             sig mismatch: &tensor[n, 4, f32] -> tensor[n, 4, f32]\n\
             def mismatch(x) = cast(0, f32)\n",
            "TypeMismatch",
        ),
        (
            "dm",
            "def want_2x2(a: tensor[2, 2, f32]) -> f32 = trace(a, 0, 1)\n\
             def main(a: tensor[3, 3, f32]) -> f32 = want_2x2(a)\n",
            "DimensionMismatch",
        ),
        (
            "validator",
            "module Probe.Validator\n\
             def f(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] = conv2d(&x, &k, 0, 0)\n",
            "",
        ),
    ];
    for (tag, src, expected_kind) in fixtures {
        let tmp = write_tempfile(&format!("issue207-inv-{tag}-"), src);
        let (code, stdout) = run_check_capture(tmp.path());
        let errors = parse_errors_array(&stdout);
        let exit_nonzero = code != Some(0);
        let errors_nonempty = !errors.is_empty();
        assert_eq!(
            exit_nonzero, errors_nonempty,
            "issue #207 invariant violated for category {tag}: exit_nonzero={exit_nonzero}, errors_nonempty={errors_nonempty}; stdout={stdout}"
        );
        assert!(
            errors_nonempty,
            "category {tag} fixture must produce errors; stdout={stdout}"
        );
        if !expected_kind.is_empty() {
            let has_kind = errors.iter().any(|e| {
                e.get("kind")
                    .and_then(Value::as_str)
                    .is_some_and(|k| k == *expected_kind)
            });
            assert!(
                has_kind,
                "category {tag} expected kind {expected_kind}; stdout={stdout}"
            );
        }
        assert_eq!(
            code,
            Some(CHECK_ERRORS_EXIT_CODE),
            "category {tag} exit code mismatch; stdout={stdout}"
        );
    }
}

/// Clean-program parity for the invariant sweep: a fixture with no
/// errors must satisfy the empty-errors-implies-exit-0 half of the
/// iff. Without this companion test the invariant collapses to a
/// one-sided implication.
#[test]
fn issue_207_invariant_holds_for_clean_program() {
    let src = "def answer() -> int32 = cast(7, int32)\n";
    let tmp = write_tempfile("issue207-inv-clean-", src);
    let (code, stdout) = run_check_capture(tmp.path());
    let errors = parse_errors_array(&stdout);
    assert!(
        errors.is_empty(),
        "clean fixture must have empty errors; stdout={stdout}"
    );
    assert_eq!(
        code,
        Some(0),
        "issue #207 invariant: clean program must exit 0; stdout={stdout}"
    );
}

/// Wave-1 red-team finding M2: an empty (or whitespace-only) `.ch` file
/// previously reported `{score: 1, errors: []}` and exited 0, while
/// `chelis build` accepted it and produced a no-op C function. Both
/// surfaces now reject a zero-declaration program with the same message
/// (`empty program: no declarations found`). The `check` surface raises
/// it via the JSON errors array (kind `Other`) and exits with
/// `CHECK_ERRORS_EXIT_CODE`. The `build` surface raises it via the
/// process error arm.
#[test]
fn rt_wave1_207_empty_file_check_rejects_with_error() {
    let src = "";
    let tmp = write_tempfile("rt207-empty-", src);
    let (code, stdout) = run_check_capture(tmp.path());
    let errors = parse_errors_array(&stdout);
    assert!(
        !errors.is_empty(),
        "empty .ch must produce a non-empty errors array; stdout={stdout}"
    );
    let has_empty_program = errors.iter().any(|e| {
        e.get("message")
            .and_then(Value::as_str)
            .is_some_and(|m| m.contains("empty program"))
    });
    assert!(
        has_empty_program,
        "empty file must produce an 'empty program' error; stdout={stdout}"
    );
    assert_eq!(
        code,
        Some(CHECK_ERRORS_EXIT_CODE),
        "empty file must exit {CHECK_ERRORS_EXIT_CODE}; stdout={stdout}"
    );
}

/// Wave-1 red-team finding M2: `chelis build` on a zero-declaration
/// file must reject with the same canonical message that `chelis check`
/// surfaces. Previously build emitted a no-op C function; the check
/// vs. build disagreement was the footgun.
#[test]
fn rt_wave1_207_empty_file_build_rejects_with_same_message() {
    let src = "";
    let tmp = write_tempfile("rt207-empty-build-", src);
    // Build emits header / object / runtime artifacts into the working
    // directory by default; sandbox them in a tempdir so the test does
    // not pollute the crate directory when run via cargo.
    let outdir = tempfile::tempdir().expect("create build outdir");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(outdir.path())
        .args(["build", tmp.path().to_str().expect("path utf8")])
        .output()
        .expect("run chelis build");
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !output.status.success(),
        "build on empty file must fail; combined output={combined}"
    );
    assert!(
        combined.contains("empty program"),
        "build on empty file must mention 'empty program'; combined output={combined}"
    );
}
