//! Public CLI device rejection for first-class Count (chelis#1287).

use std::path::Path;

use assert_cmd::Command;
use tempfile::tempdir;

fn assert_device_target_rejects_count(target: &str) {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("CLI crate lives under <workspace>/crates");
    let source = workspace.join("examples/count_bool_axes.ch");
    let temp = tempdir().expect("temporary output directory");
    let output_dir = temp.path().join(format!("count-{target}"));
    let result = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            source.to_str().expect("UTF-8 example path"),
            "--target",
            target,
            "--output",
            output_dir.to_str().expect("UTF-8 output path"),
        ])
        .output()
        .expect("chelis build runs");

    assert!(
        !result.status.success(),
        "{target} Count build silently succeeded and emitted {:?}",
        std::fs::read_dir(&output_dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .collect::<Vec<_>>()
    );
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("unimplemented chelis#1291:"), "{stderr}");
    assert!(
        stderr.contains("count") && stderr.contains("--target c"),
        "{stderr}"
    );
}

#[test]
fn hip_cli_rejects_host_helper_count_with_issue_1291_receipt() {
    assert_device_target_rejects_count("hip");
}

#[test]
fn metal_cli_rejects_host_helper_count_with_issue_1291_receipt() {
    assert_device_target_rejects_count("metal");
}
