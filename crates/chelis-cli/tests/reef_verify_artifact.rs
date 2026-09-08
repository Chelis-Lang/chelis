use assert_cmd::Command;
use chelis_shell::{PackageId, SHELL_FORMAT_VERSION, ShellPackage, encode_shell};
use predicates::prelude::*;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use tempfile::tempdir;

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn fixture() -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
    let dir = tempdir().expect("tempdir");
    let archive = b"release archive";
    let archive_path = dir.path().join("demo-1.2.3.tar.zst");
    let shell_path = dir.path().join("demo-1.2.3.chb");
    fs::write(&archive_path, archive).expect("write archive");
    let shell = ShellPackage {
        format_version: SHELL_FORMAT_VERSION,
        package: PackageId {
            name: "demo".to_string(),
            version: "1.2.3".to_string(),
        },
        compiler: "=0.17.4".to_string(),
        modules: Vec::new(),
        dependencies: Vec::new(),
        archive_sha256: sha256(archive),
    };
    fs::write(&shell_path, encode_shell(&shell).expect("encode shell")).expect("write shell");
    (dir, archive_path, shell_path)
}

fn command(archive: &std::path::Path, shell: &std::path::Path) -> Command {
    let mut command = Command::cargo_bin("chelis").expect("binary");
    command.args([
        "reef",
        "verify-artifact",
        "--archive",
        archive.to_str().unwrap(),
        "--shell",
        shell.to_str().unwrap(),
    ]);
    command
}

#[test]
fn verify_artifact_accepts_valid_pair_in_human_and_machine_modes() {
    let (_dir, archive, shell) = fixture();

    command(&archive, &shell)
        .assert()
        .success()
        .stdout(predicate::str::contains("Verified demo 1.2.3"))
        .stderr(predicate::str::is_empty());

    let output = command(&archive, &shell)
        .arg("--json")
        .output()
        .expect("run verifier");
    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    let report: Value = serde_json::from_slice(&output.stdout).expect("JSON-only stdout");
    assert_eq!(report["valid"], true);
    assert_eq!(report["errors"], serde_json::json!([]));
    assert_eq!(report["package"]["name"], "demo");
    assert_eq!(report["package"]["version"], "1.2.3");
}

#[test]
fn verify_artifact_machine_failure_is_json_only_and_obeys_invariant() {
    let (_dir, archive, shell) = fixture();
    let mut bytes = fs::read(&shell).expect("read shell");
    bytes.extend_from_slice(b"junk");
    fs::write(&shell, bytes).expect("append junk");

    let output = command(&archive, &shell)
        .arg("--json")
        .output()
        .expect("run verifier");

    assert!(!output.status.success());
    assert!(
        output.stderr.is_empty(),
        "stderr must stay empty in JSON mode"
    );
    let report: Value = serde_json::from_slice(&output.stdout).expect("JSON-only stdout");
    assert_eq!(report["valid"], false);
    assert!(
        report["errors"]
            .as_array()
            .is_some_and(|errors| !errors.is_empty()),
        "invalid reports must carry at least one error: {report}"
    );
}

#[test]
fn verify_artifact_rejects_truncation_mutation_and_archive_mismatch() {
    let (_dir, archive, shell) = fixture();
    let canonical = fs::read(&shell).expect("read shell");

    fs::write(&shell, &canonical[..canonical.len() - 1]).expect("truncate shell");
    command(&archive, &shell)
        .assert()
        .failure()
        .stderr(predicate::str::contains("decode"));

    fs::write(&shell, &canonical).expect("restore shell");
    let mut mutation = canonical.clone();
    let offset = mutation
        .windows(b"=0.17.4".len())
        .position(|window| window == b"=0.17.4")
        .expect("compiler bytes");
    mutation[offset] = b'?';
    fs::write(&shell, mutation).expect("mutate shell");
    command(&archive, &shell)
        .assert()
        .failure()
        .stderr(predicate::str::contains("compiler"));

    fs::write(&shell, canonical).expect("restore shell");
    fs::write(&archive, b"other archive").expect("replace archive");
    command(&archive, &shell)
        .assert()
        .failure()
        .stderr(predicate::str::contains("archive_sha256"));
}
