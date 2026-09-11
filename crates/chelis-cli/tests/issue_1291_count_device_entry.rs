//! Public CLI tensor-entry coverage for dedicated Count device kernels.

use std::fs;
use std::path::Path;

use assert_cmd::Command;
use tempfile::tempdir;

fn build_count_entry(target: &str, generated_suffix: &str) -> String {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("CLI crate lives under <workspace>/crates");
    let source = workspace.join("examples/count_bool_device_entry.ch");
    let temp = tempdir().expect("temporary output directory");
    let output_dir = temp.path().join(format!("count-entry-{target}"));

    Command::cargo_bin("chelis")
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
        .assert()
        .success();

    fs::read_to_string(output_dir.join(format!("count_bool_device_entry{generated_suffix}")))
        .expect("generated device source")
}

#[test]
fn hip_cli_emits_dedicated_count_tensor_entry() {
    let source = build_count_entry("hip", "_hip.cpp");
    assert!(source.contains("kernel_count_"), "{source}");
    assert!(source.contains("_device("), "{source}");
    for forbidden in ["kernel_sum", "kernel_cast", "count_host_fallback"] {
        assert!(
            !source.contains(forbidden),
            "HIP Count entry emitted forbidden fallback {forbidden:?}:\n{source}"
        );
    }
}

#[test]
fn metal_cli_emits_dedicated_count_tensor_entry_without_stub() {
    let source = build_count_entry("metal", "_metal.mm");
    assert!(source.contains("kernel void k_count_"), "{source}");
    assert!(!source.contains("M1 fallback stub"), "{source}");
    for forbidden in ["k_reduce_sum", "k_cast", "count_host_fallback"] {
        assert!(
            !source.contains(forbidden),
            "Metal Count entry emitted forbidden fallback {forbidden:?}:\n{source}"
        );
    }
}
