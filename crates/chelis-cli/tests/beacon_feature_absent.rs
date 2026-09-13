#![cfg(not(feature = "chelis-prove"))]

use assert_cmd::Command;

#[test]
fn explicit_beacon_without_runner_rejects_but_explicit_fuzz_still_runs() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("property.ch");
    std::fs::write(&source, "@property reflexive forall(x: f32): x == x\n").unwrap();
    let mut beacon = Command::cargo_bin("chelis").unwrap();
    let result = beacon
        .current_dir(directory.path())
        .args([
            "prove",
            "property.ch",
            "--tier",
            "beacon-only",
            "--samples",
            "3",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        !result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stdout)
    );
    assert!(String::from_utf8_lossy(&result.stderr).contains("requires the chelis-prove feature"));
    let mut fuzz = Command::cargo_bin("chelis").unwrap();
    fuzz.current_dir(directory.path())
        .args([
            "prove",
            "property.ch",
            "--tier",
            "fuzz-only",
            "--samples",
            "3",
            "--json",
        ])
        .assert()
        .success();
}
