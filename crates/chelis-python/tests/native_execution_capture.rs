//! Retain actual test artifacts only after successful native case completion.
//! These support controls do not substitute for the 37 native boundary cases.
#[path = "support/capture.rs"]
mod capture;

use capture::ArtifactDirectory;
use std::process::Command;

fn fixture(directory: &ArtifactDirectory) {
    let library = directory.path().join("fixture.so");
    std::fs::write(&library, b"capture support fixture; not an executable receipt").unwrap();
    std::fs::write(
        directory.path().join("loaded-model.json"),
        serde_json::to_vec(&serde_json::json!({"path":library,"library":"fixture.so"})).unwrap(),
    )
    .unwrap();
}

#[test]
fn ordinary_direct_test_directory_is_removed_on_drop() {
    let path = {
        let directory = ArtifactDirectory::temporary().unwrap();
        let path = directory.path().to_path_buf();
        fixture(&directory);
        directory.finish().unwrap();
        path
    };
    assert!(!path.exists());
}

#[test]
fn capture_retains_exact_case_files_only_after_explicit_completion() {
    let root = tempfile::tempdir().unwrap();
    let path = {
        let directory = ArtifactDirectory::for_capture(root.path(), "suite", "exact_case").unwrap();
        fixture(&directory);
        let path = directory.path().to_path_buf();
        assert!(!path.join("completion.json").exists());
        directory.finish().unwrap();
        path
    };
    let record: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("completion.json")).unwrap()).unwrap();
    assert_eq!(record["suite"], "suite");
    assert_eq!(record["test"], "exact_case");
    assert_eq!(record["completed"], true);
    assert_eq!(record["files"].as_array().unwrap().len(), 2);
    assert!(path.join("fixture.so").is_file());
}

#[test]
fn incomplete_or_missing_loaded_library_keeps_files_without_success_record() {
    let root = tempfile::tempdir().unwrap();
    let path = {
        let directory = ArtifactDirectory::for_capture(root.path(), "suite", "exact_case").unwrap();
        std::fs::write(directory.path().join("partial.c"), b"partial").unwrap();
        assert!(directory.finish().is_err());
        directory.path().to_path_buf()
    };
    assert!(path.join("partial.c").is_file());
    assert!(!path.join("completion.json").exists());
}

#[test]
fn repeated_case_instances_are_distinct_and_never_overwrite_prior_artifacts() {
    let root = tempfile::tempdir().unwrap();
    let first = ArtifactDirectory::for_capture(root.path(), "suite", "exact_case").unwrap();
    let second = ArtifactDirectory::for_capture(root.path(), "suite", "exact_case").unwrap();
    assert_ne!(first.path(), second.path());
    fixture(&first);
    fixture(&second);
    first.finish().unwrap();
    second.finish().unwrap();
    assert!(first.path().join("completion.json").is_file());
    assert!(second.path().join("completion.json").is_file());
    assert!(first.finish().is_err(), "completion must not overwrite a receipt");
}

#[test]
fn foreign_loaded_library_and_path_traversal_are_rejected() {
    let root = tempfile::tempdir().unwrap();
    assert!(ArtifactDirectory::for_capture(root.path(), "suite", "../escape").is_err());
    let directory = ArtifactDirectory::for_capture(root.path(), "suite", "exact_case").unwrap();
    let foreign = root.path().join("foreign.so");
    std::fs::write(&foreign, b"foreign").unwrap();
    std::fs::write(
        directory.path().join("loaded-model.json"),
        serde_json::to_vec(&serde_json::json!({"path":foreign,"library":foreign})).unwrap(),
    )
    .unwrap();
    assert!(directory.finish().is_err());
}

#[test]
fn actual_child_failure_preserves_command_status_and_both_streams() {
    let root = tempfile::tempdir().unwrap();
    let directory = ArtifactDirectory::for_capture(root.path(), "suite", "exact_case").unwrap();
    let executable = pyo3_build_config::get().executable.as_ref().unwrap();
    let mut command = Command::new(executable);
    command.args(["-I", "-c", "import sys; print('out'); print('err', file=sys.stderr); sys.exit(7)"]);
    let output = directory.command_output("compiler", &mut command).unwrap();
    assert_eq!(output.status.code(), Some(7));
    assert_eq!(output.stdout, b"out\n");
    assert_eq!(output.stderr, b"err\n");
    let record: serde_json::Value = serde_json::from_slice(
        &std::fs::read(directory.path().join("compiler.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(record["returncode"], 7);
    assert_eq!(record["command"][0], executable);
    assert!(!directory.path().join("completion.json").exists());
}

#[test]
fn snapshot_retains_original_bytes_without_extending_original_directory_lifetime() {
    use pyo3::{prelude::*, types::PyDict};
    let root = tempfile::tempdir().unwrap();
    let directory = ArtifactDirectory::for_capture(root.path(), "suite", "exact_case").unwrap();
    let original = tempfile::tempdir().unwrap();
    let library = original.path().join("model.so");
    std::fs::write(&library, b"snapshot helper control").unwrap();
    Python::with_gil(|py| {
        let globals = PyDict::new(py);
        directory.install(py, &globals).unwrap();
        globals.set_item("actual_path", library.to_str().unwrap()).unwrap();
        py.run(c"import types\nmodel = types.SimpleNamespace(path=actual_path)\n_capture_native_model(model)\ndel model", Some(&globals), None).unwrap();
        globals.clear();
    });
    drop(original);
    assert!(!library.exists(), "capture must not preserve the original TempDir");
    assert_eq!(std::fs::read(directory.path().join("loaded-artifacts/model.so")).unwrap(), b"snapshot helper control");
    directory.finish().unwrap();
    let record: serde_json::Value = serde_json::from_slice(
        &std::fs::read(directory.path().join("loaded-model.json")).unwrap(),
    ).unwrap();
    assert_eq!(record["path"], library.to_str().unwrap());
    assert_eq!(record["files"][0]["original"], library.to_str().unwrap());
}
