//! Eval-versus-compiled parity for host-effect programs ([05-HOST-1],
//! [05-HOST-2], chelis#1297).
//!
//! Each lane runs the same source in its own fresh working directory, so a
//! program that writes files observes only its own effects. Two lanes agree
//! when they print the same stdout, exit with the same status, and report the
//! same failure message and the context line before it. The comparison is
//! modulo exactly one presentation difference: `chelis eval` prefixes a failure with `error: `, while a
//! compiled executable prints the message alone.

// Each including test binary uses only the helpers its lane needs.
#![allow(dead_code)]

use assert_cmd::Command;
use std::path::Path;
use std::process::Command as StdCommand;
use tempfile::tempdir;

/// The eval CLI's presentation prefix on a failure message.
pub const EVAL_FAILURE_PREFIX: &str = "error: ";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaneRun {
    pub status: Option<i32>,
    pub stdout: String,
    /// The failure message with lane presentation removed; empty on success.
    pub failure: String,
    /// The stderr line before the failure message, which carries a trap's
    /// context (spec/04-type-system.md section 4.7), with lane presentation
    /// removed; empty on success or when the failure is one line.
    pub context: String,
}

fn chelis() -> Command {
    let mut command = Command::cargo_bin("chelis").expect("binary");
    command.env("CHELIS_STYLE_GATE_DISABLE", "1");
    command
}

fn write_source(dir: &Path, name: &str, source: &str) -> std::path::PathBuf {
    let path = dir.join(format!("{name}.ch"));
    std::fs::write(&path, source).expect("write source");
    path
}

/// The failure body of eval's stderr: its last line, without the
/// presentation prefix that eval adds to a runtime failure (a numeric trap
/// is printed without it).
pub fn eval_failure_body(stderr: &str) -> String {
    let last = stderr.lines().last().unwrap_or_default();
    last.strip_prefix(EVAL_FAILURE_PREFIX)
        .unwrap_or(last)
        .to_string()
}

/// The failure body of a compiled executable's stderr: its last line.
pub fn compiled_failure_body(stderr: &str) -> String {
    stderr.lines().last().unwrap_or_default().to_string()
}

/// The context line of a failing lane's stderr: the line before its last,
/// without eval's presentation prefix.
pub fn failure_context(stderr: &str) -> String {
    let lines = stderr.lines().collect::<Vec<_>>();
    let Some(line) = lines.len().checked_sub(2).map(|index| lines[index]) else {
        return String::new();
    };
    line.strip_prefix(EVAL_FAILURE_PREFIX)
        .unwrap_or(line)
        .to_string()
}

/// Runs `source` under `chelis eval --file` in a fresh directory.
pub fn eval_lane(source: &str, name: &str) -> LaneRun {
    let dir = tempdir().expect("tempdir");
    let path = write_source(dir.path(), name, source);
    let output = chelis()
        .current_dir(dir.path())
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval runs");
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    LaneRun {
        status: output.status.code(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        failure: if output.status.success() {
            String::new()
        } else {
            eval_failure_body(&stderr)
        },
        context: if output.status.success() {
            String::new()
        } else {
            failure_context(&stderr)
        },
    }
}

/// Builds `source` for C, which must succeed, then runs the executable in a
/// fresh directory.
pub fn compiled_lane(source: &str, name: &str) -> LaneRun {
    let dir = tempdir().expect("tempdir");
    let path = write_source(dir.path(), name, source);
    let out = dir.path().join("out");
    let build = chelis()
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("chelis build runs");
    assert!(
        build.status.success(),
        "{name}: a legal host-effect program must build: stderr={}",
        String::from_utf8_lossy(&build.stderr)
    );
    let run = tempdir().expect("run dir");
    let output = StdCommand::new(out.join(name))
        .current_dir(run.path())
        .output()
        .expect("the built executable runs");
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    LaneRun {
        status: output.status.code(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        failure: if output.status.success() {
            String::new()
        } else {
            compiled_failure_body(&stderr)
        },
        context: if output.status.success() {
            String::new()
        } else {
            failure_context(&stderr)
        },
    }
}

/// The parity rule: the same stdout, exit status, failure body, and context.
pub fn lanes_agree(eval: &LaneRun, compiled: &LaneRun) -> bool {
    eval == compiled
}

/// Runs both lanes and asserts they agree; returns the agreed run.
pub fn assert_lanes_agree(source: &str, name: &str) -> LaneRun {
    let eval = eval_lane(source, name);
    let compiled = compiled_lane(source, name);
    assert!(
        lanes_agree(&eval, &compiled),
        "{name}: eval and compiled C disagree\neval: {eval:?}\ncompiled: {compiled:?}\nsource:\n{source}"
    );
    assert_eq!(
        eval, compiled,
        "{name}: eval and compiled C disagree\nsource:\n{source}"
    );
    eval
}
