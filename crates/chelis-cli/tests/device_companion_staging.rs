//! C3: actual build staging includes the independent device implementation.
use assert_cmd::cargo::CommandCargoExt;
use std::{fs, process::Command};
#[test]
fn hip_build_stages_companion_once_and_c_build_excludes_it() {
    for (target, include_device) in [("hip", true), ("c", false)] {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("companion.ch");
        let output = directory.path().join("out");
        fs::write(&source, "x = [1.0, 2.0]\ny = mul(x, x)\n").unwrap();
        let result = Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .arg("build")
            .arg(&source)
            .args(["--target", target, "--output"])
            .arg(&output)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{target}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        for (name, expected) in [
            (
                "chelis_device_owner.cpp",
                include_str!("../../chelis-backend-hip/runtime/chelis_device_owner.cpp"),
            ),
            (
                "chelis_device_owner.h",
                include_str!("../../chelis-backend-hip/runtime/chelis_device_owner.h"),
            ),
            (
                "chelis_device_descriptor.h",
                include_str!("../../chelis-backend-hip/runtime/chelis_device_descriptor.h"),
            ),
        ] {
            assert_eq!(
                output.join(name).exists(),
                include_device,
                "{target}: {name}"
            );
            if include_device {
                assert_eq!(fs::read_to_string(output.join(name)).unwrap(), expected);
            }
        }
        if include_device {
            let stdout = String::from_utf8_lossy(&result.stdout);
            let compile = stdout
                .lines()
                .find(|line| line.starts_with("Compile:") || line.starts_with("Compile object:"))
                .expect("HIP compile instruction");
            assert_eq!(compile.matches("chelis_device_owner.cpp").count(), 1);
        }
    }
}
