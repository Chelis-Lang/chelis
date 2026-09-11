//! Public CLI device host-helper support for first-class Count (chelis#1291).

use std::path::Path;

use assert_cmd::Command;
use tempfile::tempdir;

const TWO_COUNT_HELPERS: &str = "\
module Example.TwoCounts\n\
mask: tensor[2, 3, bool] = [[true, false, true], [false, true, true]]\n\
rows: tensor[2, int64] = count(&mask, 1)\n\
columns: tensor[3, int64] = count(&mask, 0)\n";

fn assert_device_target_emits_count_helper(target: &str) {
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
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );

    let emitted = std::fs::read_dir(&output_dir)
        .expect("device output directory")
        .map(|entry| entry.expect("generated artifact").path())
        .collect::<Vec<_>>();
    let helper_path = emitted
        .iter()
        .find(|path| {
            path.file_name().is_some_and(|name| {
                let name = name.to_string_lossy();
                name.contains("global__tensor_0")
                    && match target {
                        "hip" => name.ends_with("_hip.cpp"),
                        "metal" => name.ends_with("_metal.mm"),
                        _ => false,
                    }
            })
        })
        .unwrap_or_else(|| panic!("{target} output lacks a device Count helper: {emitted:?}"));
    let helper = std::fs::read_to_string(helper_path).expect("device helper is readable");
    assert!(helper.contains("count_"), "{helper}");
    for forbidden in [
        "kernel_sum",
        "kernel_cast",
        "count_host_fallback",
        "unimplemented stub",
    ] {
        assert!(
            !helper.contains(forbidden),
            "{target} Count helper emitted forbidden fallback {forbidden:?}:\n{helper}"
        );
    }
}

#[test]
fn hip_cli_emits_host_helper_count_on_the_device() {
    assert_device_target_emits_count_helper("hip");
}

#[test]
fn metal_cli_emits_host_helper_count_on_the_device() {
    assert_device_target_emits_count_helper("metal");
}

fn assert_multiple_count_helpers_use_separate_device_translation_units(target: &str) {
    let temp = tempdir().expect("temporary source and output directory");
    let source = temp.path().join("two_counts.ch");
    std::fs::write(&source, TWO_COUNT_HELPERS).expect("write two-Count fixture");
    let output_dir = temp.path().join(format!("two-counts-{target}"));
    let result = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            source.to_str().expect("UTF-8 fixture path"),
            "--target",
            target,
            "--output",
            output_dir.to_str().expect("UTF-8 output path"),
        ])
        .output()
        .expect("chelis build runs");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );

    let suffix = match target {
        "hip" => "_hip.cpp",
        "metal" => "_metal.mm",
        _ => panic!("unexpected target {target}"),
    };
    let helpers = std::fs::read_dir(&output_dir)
        .expect("device output directory")
        .map(|entry| entry.expect("generated artifact").path())
        .filter(|path| {
            let name = path
                .file_name()
                .expect("artifact file name")
                .to_string_lossy();
            name.contains("global__tensor_") && name.ends_with(suffix)
        })
        .collect::<Vec<_>>();
    assert_eq!(helpers.len(), 2, "{target} helper files: {helpers:?}");
}

#[test]
fn hip_cli_isolates_multiple_host_count_helpers() {
    assert_multiple_count_helpers_use_separate_device_translation_units("hip");
}

#[test]
fn metal_cli_isolates_multiple_host_count_helpers() {
    assert_multiple_count_helpers_use_separate_device_translation_units("metal");
}

/// A Count-bearing helper is emitted as device code, so it receives the full
/// device capability policy at the shared gate before any backend runs. On
/// Metal, direct `sub` is an unimplemented chelis#1306 cell: the build must
/// stop at the gate with that typed receipt, not fall through to the
/// emitter's backstop.
#[test]
fn metal_cli_gates_a_count_helper_with_the_full_device_policy() {
    let temp = tempdir().expect("temporary source and output directory");
    let source = temp.path().join("count_sub.ch");
    std::fs::write(
        &source,
        "mask: tensor[2, 3, bool] = [[true, false, true], [false, true, true]]\n\
         rows: tensor[2, int64] = sub(count(&mask, 1), count(&mask, 1))\n",
    )
    .expect("write Count-plus-sub fixture");
    let output_dir = temp.path().join("count-sub-metal");
    let result = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            source.to_str().expect("UTF-8 fixture path"),
            "--target",
            "metal",
            "--output",
            output_dir.to_str().expect("UTF-8 output path"),
        ])
        .output()
        .expect("chelis build runs");
    assert!(
        !result.status.success(),
        "a Count helper carrying direct `sub` must not build for Metal:\n{}",
        String::from_utf8_lossy(&result.stdout)
    );
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("unimplemented chelis#1306:"), "{stderr}");
    assert!(
        stderr.contains("does not yet support exact `sub`"),
        "the shared gate, not the emitter backstop, must reject:\n{stderr}"
    );
    assert!(
        !stderr.contains("reached emission"),
        "the emitter backstop must never be the first line of defense:\n{stderr}"
    );
}

/// The HIP lane runs the same gate call. Checked `int64` subtraction is an
/// unimplemented chelis#1306 cell on HIP, so the same program stops at the
/// early capability gate with that typed receipt rather than reaching the
/// HIP emitter.
#[test]
fn hip_cli_gates_a_count_helper_with_the_full_device_policy() {
    let temp = tempdir().expect("temporary source and output directory");
    let source = temp.path().join("count_sub.ch");
    std::fs::write(
        &source,
        "mask: tensor[2, 3, bool] = [[true, false, true], [false, true, true]]\n\
         rows: tensor[2, int64] = sub(count(&mask, 1), count(&mask, 1))\n",
    )
    .expect("write Count-plus-sub fixture");
    let output_dir = temp.path().join("count-sub-hip");
    let result = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            source.to_str().expect("UTF-8 fixture path"),
            "--target",
            "hip",
            "--output",
            output_dir.to_str().expect("UTF-8 output path"),
        ])
        .output()
        .expect("chelis build runs");
    assert!(
        !result.status.success(),
        "a Count helper carrying checked int64 `sub` must not build for HIP:\n{}",
        String::from_utf8_lossy(&result.stdout)
    );
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("unimplemented chelis#1306:"), "{stderr}");
    assert!(
        stderr.contains("early capability gate"),
        "the shared gate, not the emitter, must reject:\n{stderr}"
    );
    assert!(!stderr.contains("reached emission"), "{stderr}");
}

/// The `build-deep` lane runs the same four gate call sites as `build`. The
/// fixture is desugared with `chelis deep` and built from the `.dp`, so both
/// `cmd_build_deep` sites (Metal and HIP) stop at the gate with the typed
/// chelis#1306 receipt instead of reaching an emitter.
#[test]
fn build_deep_gates_a_count_helper_with_the_full_device_policy() {
    let temp = tempdir().expect("temporary source and output directory");
    let surf = temp.path().join("count_sub.ch");
    std::fs::write(
        &surf,
        "mask: tensor[2, 3, bool] = [[true, false, true], [false, true, true]]\n\
         rows: tensor[2, int64] = sub(count(&mask, 1), count(&mask, 1))\n",
    )
    .expect("write Count-plus-sub fixture");
    let desugared = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["deep", surf.to_str().expect("UTF-8 fixture path")])
        .output()
        .expect("chelis deep runs");
    assert!(
        desugared.status.success(),
        "{}",
        String::from_utf8_lossy(&desugared.stderr)
    );
    let deep = temp.path().join("count_sub.dp");
    std::fs::write(&deep, &desugared.stdout).expect("write desugared fixture");

    for (target, gate_marker) in [
        ("metal", "does not yet support exact `sub`"),
        ("hip", "early capability gate"),
    ] {
        let output_dir = temp.path().join(format!("count-sub-deep-{target}"));
        let result = Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args([
                "build",
                deep.to_str().expect("UTF-8 fixture path"),
                "--target",
                target,
                "--output",
                output_dir.to_str().expect("UTF-8 output path"),
            ])
            .output()
            .expect("chelis build runs");
        assert!(
            !result.status.success(),
            "{target} build-deep of a Count helper carrying `sub` must not build:\n{}",
            String::from_utf8_lossy(&result.stdout)
        );
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(
            stderr.contains("unimplemented chelis#1306:"),
            "{target}: {stderr}"
        );
        assert!(stderr.contains(gate_marker), "{target}: {stderr}");
        assert!(!stderr.contains("reached emission"), "{target}: {stderr}");
    }
}
