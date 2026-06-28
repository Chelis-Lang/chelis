use assert_cmd::Command;
use tempfile::tempdir;

const HELLO_TENSOR: &str = include_str!("../../../examples/hello_tensor.ch");

#[test]
fn surf_output_from_deep_is_canonical_and_checks() {
    let dir = tempdir().expect("tempdir");
    let source_path = dir.path().join("hello_tensor.ch");
    let deep_path = dir.path().join("hello_tensor.dp");
    let surf_path = dir.path().join("roundtrip.ch");
    std::fs::write(&source_path, HELLO_TENSOR).expect("write source");

    let deep = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["deep", source_path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    std::fs::write(&deep_path, deep).expect("write deep");

    let surf = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["surf", deep_path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    std::fs::write(&surf_path, surf).expect("write surf");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", "--check", surf_path.to_str().unwrap()])
        .assert()
        .success();
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["lint", "--check", surf_path.to_str().unwrap()])
        .assert()
        .success();
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", surf_path.to_str().unwrap()])
        .assert()
        .success();
}
