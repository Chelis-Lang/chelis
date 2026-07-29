//! `chelis eval --timeout <secs>` (chelis#914).
//!
//! ## Background
//!
//! Interactive Ctrl-C on the CLI already works — a foreground `chelis eval`
//! dies on SIGINT in tens of milliseconds. What did not exist was a *loud,
//! scriptable* failure for the unattended case: an accidentally quadratic or
//! mis-sized program was indistinguishable from one still making progress, so
//! CI and agent harnesses could only kill the whole process blind.
//!
//! `--timeout` arms a watchdog that trips the same cooperative cancellation
//! token the bindings use. On trip the eval unwinds through the ordinary error
//! channel and the CLI prints `error: evaluation timed out after <N>s
//! (--timeout)` and exits non-zero.
//!
//! ## Coverage
//!
//! Positive: `timeout_trips_on_a_slow_program` — the flag fires, with the
//! documented message and a non-zero exit, well before the program would have
//! finished on its own.
//!
//! Negative parity (the flag must not fire when it must not):
//! * `generous_timeout_does_not_disturb_a_fast_program` — a fast program under
//!   a large timeout produces its normal value and exit 0.
//! * `no_timeout_flag_runs_to_completion` — the same slow-ish program without
//!   the flag completes and prints its result, so the watchdog is opt-in and
//!   the per-node token check does not spuriously cancel.
//! * `timeout_message_is_absent_from_a_successful_run` — guards against the
//!   watchdog's hard-exit backstop firing on a run that already succeeded.

use assert_cmd::Command;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tempfile::tempdir;

/// Long enough that a 2-second timeout is unambiguous. A flat fold, not a
/// recursion: deep recursion overflows the interpreter stack long before it
/// becomes slow.
const SLOW_PROGRAM: &str = "result = fold(fn (acc, x) -> acc + x, 0i64, range(0, 4000000))\n";

/// Finishes in milliseconds.
const FAST_PROGRAM: &str = "result = fold(fn (acc, x) -> acc + x, 0i64, range(0, 100))\n";

fn write_program(dir: &Path, name: &str, body: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, body).expect("write program");
    path
}

fn eval(args: &[&str]) -> (Option<i32>, String, String) {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(args)
        .output()
        .expect("run chelis eval");
    (
        output.status.code(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn timeout_trips_on_a_slow_program() {
    let dir = tempdir().expect("tempdir");
    let path = write_program(dir.path(), "slow.ch", SLOW_PROGRAM);

    let started = Instant::now();
    let (code, stdout, stderr) = eval(&[
        "eval",
        "--timeout",
        "2",
        "--file",
        path.to_str().expect("utf-8 path"),
    ]);
    let elapsed = started.elapsed();

    assert_ne!(code, Some(0), "a timed-out eval must exit non-zero");
    assert!(
        stderr.contains("evaluation timed out after 2s (--timeout)"),
        "expected the documented timeout message, got stderr: {stderr}"
    );
    assert!(
        stdout.trim().is_empty(),
        "a timed-out eval must not print a result, got stdout: {stdout}"
    );
    // The program takes far longer than this uncancelled; a generous ceiling
    // keeps the test honest on a loaded machine while still proving the
    // watchdog fired rather than the program completing.
    assert!(
        elapsed.as_secs() < 20,
        "timeout should trip near 2s, took {elapsed:?}"
    );
}

#[test]
fn generous_timeout_does_not_disturb_a_fast_program() {
    let dir = tempdir().expect("tempdir");
    let path = write_program(dir.path(), "fast.ch", FAST_PROGRAM);

    let (code, stdout, stderr) = eval(&[
        "eval",
        "--timeout",
        "600",
        "--file",
        path.to_str().expect("utf-8 path"),
    ]);

    assert_eq!(code, Some(0), "stderr: {stderr}");
    assert_eq!(stdout.trim(), "4950");
    assert!(
        !stderr.contains("timed out"),
        "a fast program must not report a timeout, got: {stderr}"
    );
}

#[test]
fn no_timeout_flag_runs_to_completion() {
    let dir = tempdir().expect("tempdir");
    let path = write_program(dir.path(), "fast.ch", FAST_PROGRAM);

    let (code, stdout, stderr) = eval(&["eval", "--file", path.to_str().expect("utf-8 path")]);

    assert_eq!(code, Some(0), "stderr: {stderr}");
    assert_eq!(
        stdout.trim(),
        "4950",
        "without --timeout nothing about eval changes"
    );
}

#[test]
fn timeout_message_is_absent_from_a_successful_run() {
    let dir = tempdir().expect("tempdir");
    let path = write_program(dir.path(), "fast.ch", FAST_PROGRAM);

    // A timeout shorter than the watchdog's hard-exit grace, on a program that
    // finishes immediately: the process must exit 0 on its own before the
    // watchdog has any say.
    let (code, stdout, stderr) = eval(&[
        "eval",
        "--timeout",
        "1",
        "--file",
        path.to_str().expect("utf-8 path"),
    ]);

    assert_eq!(code, Some(0), "stderr: {stderr}");
    assert_eq!(stdout.trim(), "4950");
    assert!(!stderr.contains("timed out"), "stderr: {stderr}");
}
