//! Checked package calls reach the in-process native solver only when the
//! optional provider feature is selected.

use assert_cmd::Command;
use std::path::PathBuf;

fn package() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../packages/chelis-clarabel")
}

fn example() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/illustrative/clarabel_qp")
}

fn eval(file: &str) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(package())
        .args(["eval", "--file", file])
        .output()
        .expect("run Chelis evaluator")
}

#[test]
fn checked_unconstrained_and_inequality_calls_solve() {
    for file in ["tests/solve.ch", "tests/nonnegative.ch"] {
        let result = eval(file);
        assert!(
            result.status.success(),
            "{file}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(result.stdout, b"main = true\n", "{file}");
    }
}

#[test]
fn path_dependency_calls_the_native_solver() {
    let result = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(example())
        .args(["eval", "--file", "src/main.ch"])
        .output()
        .expect("evaluate example");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(result.stdout, b"main = true\n");
}

#[test]
fn stopped_status_does_not_carry_solved_fields() {
    let result = eval("tests/stopped.ch");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(result.stdout, b"main = true\n");
}

#[test]
fn invalid_cone_partition_fails_before_solver() {
    let result = eval("tests/invalid_cone.ch");
    assert!(!result.status.success());
    assert!(
        String::from_utf8_lossy(&result.stderr)
            .contains("cone dimensions sum to 2, but b and A have 1 rows")
    );
}

#[test]
fn wrong_tensor_dtype_is_rejected_by_checker() {
    let result = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(package())
        .args(["check", "tests/wrong_dtype.ch"])
        .output()
        .expect("check Chelis source");
    assert!(!result.status.success());
    let report = String::from_utf8_lossy(&result.stdout);
    assert!(report.contains("PrecisionMismatch"), "{report}");
}

#[test]
fn package_with_matching_linked_name_and_changed_source_runs_its_own_body() {
    let directory = tempfile::tempdir().expect("temporary package");
    std::fs::create_dir(directory.path().join("src")).expect("source directory");
    std::fs::create_dir(directory.path().join("tests")).expect("test directory");
    std::fs::copy(
        package().join("reef.toml"),
        directory.path().join("reef.toml"),
    )
    .expect("copy manifest");
    std::fs::copy(
        package().join("tests/solve.ch"),
        directory.path().join("tests/solve.ch"),
    )
    .expect("copy call");
    let source = std::fs::read_to_string(package().join("src/qp.ch")).expect("provider source");
    let changed = source.replace(
        "Clarabel native provider unavailable",
        "lookalike package body executed",
    );
    assert_ne!(changed, source);
    std::fs::write(directory.path().join("src/qp.ch"), changed).expect("write lookalike");
    let result = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(directory.path())
        .args(["eval", "--file", "tests/solve.ch"])
        .output()
        .expect("evaluate lookalike");
    assert!(
        !result.status.success(),
        "the lookalike must not invoke Clarabel"
    );
    let report = String::from_utf8_lossy(&result.stderr);
    assert!(
        report.contains("lookalike package body executed"),
        "{report}"
    );
}
