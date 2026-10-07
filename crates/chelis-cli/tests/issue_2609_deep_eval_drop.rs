//! The evaluator's text path releases its nested execution value after output.

use std::fs;
use std::process::Command;

use tempfile::tempdir;

const DEPTH: usize = 200_000;

#[test]
fn a_very_deep_root_renders_and_returns_without_overflowing_on_cleanup() {
    let directory = tempdir().expect("temporary source directory");
    let source = directory.path().join("deep_root.ch");
    fs::write(
        &source,
        format!(
            "type Chain =\n  | Link(Chain)\n  | End\n\
             def case(n: i64) -> List[Chain] = [fold(fn (acc: Chain, x: i64) -> Link(acc), End, range(0i64, n))]\n\
             a = case({DEPTH}i64)\n"
        ),
    )
    .expect("write source");

    let output = Command::new(env!("CARGO_BIN_EXE_chelis"))
        .args(["eval", "--file", source.to_str().expect("UTF-8 path")])
        .output()
        .expect("run evaluator");
    assert!(
        output.status.success(),
        "eval aborted: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).expect("UTF-8 output"),
        format!("a = [{}End{}]\n", "Link(".repeat(DEPTH), ")".repeat(DEPTH))
    );
}
