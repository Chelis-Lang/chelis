//! Exercise the actual effectful adapter and core-backed receipt validation.
use std::{path::Path, process::Command};

#[test]
fn failed_observations_cannot_publish_identity() {
    let scripts = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts");
    let output = Command::new("python3")
        .args(["-m", "unittest", "test_runtime_identity_observer"])
        .current_dir(scripts)
        .env(
            "CHELIS_IDENTITY_TEST_HELPER",
            env!("CARGO_BIN_EXE_chelis-runtime-identity-build"),
        )
        .output()
        .expect("launch adapter failure scenarios");
    assert!(
        output.status.success(),
        "adapter failure scenarios failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
