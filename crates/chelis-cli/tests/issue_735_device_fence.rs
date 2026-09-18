//! chelis#735: the core C build must not erase a device request into host C.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

const CUDA_DIAGNOSTIC: &str = "error: `chelis build --target c` cannot satisfy resource \
region `cuda:0`: host C accepts only `cpu` or `cpu:<label>`\n";

fn write(path: &Path, source: &str) {
    fs::write(path, source).expect("write fixture");
}

#[test]
fn surf_c_build_rejects_device_requests_at_exit_one_without_artifacts() {
    for device in [
        "cuda:0",
        "metal",
        "rocm",
        "xpu:1",
        "Gpu:0",
        "gpu:0",
        "host",
        "",
        "cpu:",
        "cpu:two words",
        "cpu:/0",
    ] {
        let dir = tempdir().expect("tempdir");
        let source = dir.path().join("device.ch");
        let output = dir.path().join("out");
        write(
            &source,
            &format!("value: i32 = with device(\"{device}\") {{ 1 }}\n"),
        );
        let expected = format!(
            "error: `chelis build --target c` cannot satisfy resource region `{device}`: \
             host C accepts only `cpu` or `cpu:<label>`\n"
        );

        Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args([
                "build",
                source.to_str().unwrap(),
                "--target",
                "c",
                "--output",
                output.to_str().unwrap(),
            ])
            .assert()
            .code(1)
            .stdout(predicate::eq(""))
            .stderr(predicate::eq(expected));

        assert!(
            !output.exists(),
            "{device}: rejection must precede output-directory creation"
        );
    }
}

#[test]
fn deep_c_build_has_the_same_rejection_and_artifact_absence() {
    let dir = tempdir().expect("tempdir");
    let source = dir.path().join("device.dp");
    let output = dir.path().join("out");
    write(
        &source,
        r#"(def {} value
  (handle-effect {effect: resource}
    (lit {type: (t-prim {} string)} "cuda:0")
    (lit {type: (t-prim {} i32)} 1)))
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            source.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            output.to_str().unwrap(),
        ])
        .assert()
        .code(1)
        .stdout(predicate::eq(""))
        .stderr(predicate::eq(CUDA_DIAGNOSTIC));

    assert!(!output.exists());
}

#[test]
fn nested_device_requests_cannot_hide_behind_host_regions() {
    for source_text in [
        "value: i32 = with device(\"cpu\") { with device(\"cuda:0\") { 1 } }\n",
        "value: i32 = with device(\"cuda:0\") { with device(\"cpu\") { 1 } }\n",
    ] {
        let dir = tempdir().expect("tempdir");
        let source = dir.path().join("nested.ch");
        let output = dir.path().join("out");
        write(&source, source_text);

        Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args([
                "build",
                source.to_str().unwrap(),
                "--target",
                "c",
                "--output",
                output.to_str().unwrap(),
            ])
            .assert()
            .code(1)
            .stdout(predicate::eq(""))
            .stderr(predicate::eq(CUDA_DIAGNOSTIC));

        assert!(!output.exists());
    }
}

#[test]
fn unpinned_and_explicit_host_c_programs_still_emit_artifacts() {
    for (name, source_text) in [
        ("plain", "value: i32 = 1\n"),
        ("cpu", "value: i32 = with device(\"cpu\") { 1 }\n"),
        (
            "named_cpu",
            "value: i32 = with device(\"cpu:author-device\") { 1 }\n",
        ),
        (
            "nested_cpu",
            "value: i32 = with device(\"cpu:outer\") { with device(\"cpu:inner\") { 1 } }\n",
        ),
    ] {
        let dir = tempdir().expect("tempdir");
        let source = dir.path().join(format!("{name}.ch"));
        let output = dir.path().join("out");
        write(&source, source_text);

        Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args([
                "build",
                source.to_str().unwrap(),
                "--target",
                "c",
                "--output",
                output.to_str().unwrap(),
            ])
            .assert()
            .success()
            .stderr(predicate::eq(""));

        assert!(output.join(format!("{name}.c")).is_file(), "{name}");
        assert!(output.join(format!("{name}.h")).is_file(), "{name}");
        assert!(output.join("chelis_runtime.h").is_file(), "{name}");
        assert!(output.join("libchelis_runtime.a").is_file(), "{name}");
    }
}
