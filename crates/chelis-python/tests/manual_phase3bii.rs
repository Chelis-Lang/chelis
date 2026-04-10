use std::path::PathBuf;
use std::process::Command;

#[test]
#[ignore = "manual phase 3b-ii acceptance gate: requires py/.venv with torch/numpy and installs bindings/python into that venv"]
fn phase3bii_python_manual_acceptance_oracle() {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let python = repo_root.join("py/.venv/bin/python");
    assert!(python.exists(), "expected {}", python.display());

    let status = Command::new("uv")
        .arg("pip")
        .arg("install")
        .arg("--python")
        .arg(&python)
        .arg("-e")
        .arg(repo_root.join("bindings/python"))
        .current_dir(&repo_root)
        .status()
        .expect("install chelis package");
    assert!(status.success(), "uv pip install failed");

    let status = Command::new(&python)
        .arg(repo_root.join("bindings/python/tests/manual_phase3bii.py"))
        .current_dir(&repo_root)
        .status()
        .expect("run manual python direct-execution acceptance");

    assert!(status.success(), "manual phase 3b-ii acceptance failed");
}
