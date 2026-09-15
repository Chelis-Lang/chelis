use super::common;

use assert_cmd::Command;
use std::fs;

pub fn run(source: &str, native: bool) -> (bool, String) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("completion.ch");
    fs::write(&path, source).expect("fixture");
    let output = if native {
        assert!(
            common::gcc_available(),
            "the compiled C receipt must execute"
        );
        let destination = dir.path().join("c");
        let built = Command::cargo_bin("chelis")
            .expect("chelis")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["build", "--allow-style-violations"])
            .arg(&path)
            .args(["--target", "c", "-o"])
            .arg(&destination)
            .output()
            .expect("build");
        assert!(built.status.success(), "{source}\n{built:?}");
        assert!(common::link_generated(&destination, "completion.c", "completion").success());
        std::process::Command::new(destination.join("completion"))
            .output()
            .expect("execute native fixture")
    } else {
        Command::cargo_bin("chelis")
            .expect("chelis")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["eval", "--allow-style-violations", "--file"])
            .arg(&path)
            .output()
            .expect("eval")
    };
    (
        output.status.success(),
        format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
    )
}

pub fn assert_claim(output: &str, op: &str, actual: usize) {
    assert!(
        output.contains(&format!("extent `3`: claimed = 3, {op} axis 0 = {actual}")),
        "{output}"
    );
    assert!(
        output
            .lines()
            .any(|line| line == format!("numeric trap: domain in {op} at int64")),
        "{output}"
    );
    assert!(!output.contains("out ="), "{output}");
}
