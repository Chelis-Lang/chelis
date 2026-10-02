//! chelis#761 / chelis#729 Phase 3: compiled-C literal ingress must preserve
//! f32 subnormals instead of decimalizing them through an untyped C literal.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

const PROGRAM: &str = "module M.Main\n\
    def id(x: tensor[1, f32]) -> tensor[1, f32] = x\n\
    out = print(id(to_tensor([1.0e-40])))\n";

fn c_toolchain_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn rendered_element(stdout: &str) -> &str {
    stdout
        .split("data=[")
        .nth(1)
        .and_then(|rest| rest.split(']').next())
        .map(str::trim)
        .unwrap_or_else(|| panic!("missing tensor element in `{stdout}`"))
}

fn assert_min_subnormal_survives(stdout: &str, lane: &str) {
    let token = rendered_element(stdout);
    let parsed: f32 = token
        .parse()
        .unwrap_or_else(|error| panic!("{lane}: `{token}` is not an f32: {error}"));
    assert_ne!(
        parsed.to_bits(),
        0,
        "{lane}: f32 subnormal ingress silently flushed to zero: {stdout}"
    );
    assert_eq!(
        parsed.to_bits(),
        1.0e-40_f32.to_bits(),
        "{lane}: ingress changed the stored f32 subnormal bits: {stdout}"
    );
}

#[test]
fn eval_preserves_f32_subnormal_ingress_bits() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("subnormal.ch");
    write_file(&path, PROGRAM);
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("eval should run");
    assert!(
        output.status.success(),
        "eval failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_min_subnormal_survives(&String::from_utf8_lossy(&output.stdout), "eval");
}

#[test]
fn c_preserves_f32_subnormal_ingress_bits() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("subnormal.ch");
    let out_dir = dir.path().join("subnormal-out");
    write_file(&path, PROGRAM);
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    let status = common::link_generated(&out_dir, "subnormal.c", "subnormal");
    assert!(status.success(), "link failed: {status}");
    let output = std::process::Command::new(out_dir.join("subnormal"))
        .output()
        .expect("compiled program should run");
    assert!(
        output.status.success(),
        "compiled program failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_min_subnormal_survives(&String::from_utf8_lossy(&output.stdout), "C");
}
