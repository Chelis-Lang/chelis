//! CLI surface coverage for the [05-OP-75] host clock reads,
//! `clock_wall_read()` and `clock_monotonic_read()`.
//!
//! Each returns `(seconds, nanoseconds)` and carries `IO`. `chelis eval` and
//! compiled C both run them through the runtime's one [05-HOST-2] definition
//! (chelis#1297). Every accepted program has a rejected twin.

use assert_cmd::Command;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};
use tempfile::tempdir;

const CLOCKS: [&str; 2] = ["clock_wall_read", "clock_monotonic_read"];

fn chelis() -> Command {
    let mut command = Command::cargo_bin("chelis").expect("binary");
    command.env("CHELIS_STYLE_GATE_DISABLE", "1");
    command
}

/// The `(seconds, nanoseconds)` halves printed under `prefix`.
fn printed_pair(stdout: &str, prefix: &str) -> (i64, i64) {
    let half = |index: usize| {
        let key = format!("{prefix}.{index} = ");
        let line = stdout
            .lines()
            .find_map(|line| line.strip_prefix(&key))
            .unwrap_or_else(|| panic!("no `{key}` line in stdout={stdout}"));
        line.trim()
            .parse::<i64>()
            .unwrap_or_else(|error| panic!("`{key}{line}` is not an i64: {error}"))
    };
    (half(0), half(1))
}

fn host_now() -> (i64, i64) {
    let since = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("the host clock is after 1970");
    (
        i64::try_from(since.as_secs()).expect("fits"),
        i64::from(since.subsec_nanos()),
    )
}

/// `chelis eval` reads the host wall clock: the reading lies between two host
/// reads taken around the command, and its nanoseconds lie in `[0, 10^9)`.
#[test]
fn eval_wall_read_lies_between_host_reads() {
    let lower = host_now();
    let output = chelis()
        .args(["eval", "clock_wall_read()"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let upper = host_now();
    let stdout = String::from_utf8_lossy(&output);
    let reading = printed_pair(&stdout, "eval_result");
    assert!(
        lower <= reading && reading <= upper,
        "{lower:?} <= {reading:?} <= {upper:?}"
    );
    assert!((0..1_000_000_000).contains(&reading.1), "{reading:?}");
}

/// Two monotonic reads in sequence never decrease, in `(seconds,
/// nanoseconds)` order.
#[test]
fn eval_monotonic_reads_never_decrease() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("clocks.ch");
    fs::write(
        &src,
        "first = clock_monotonic_read()\nsecond = clock_monotonic_read()\n",
    )
    .expect("write source");
    let output = chelis()
        .args(["eval", "--file", src.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8_lossy(&output);
    let first = printed_pair(&stdout, "first");
    let second = printed_pair(&stdout, "second");
    assert!(first <= second, "{first:?} then {second:?}");
    for reading in [first, second] {
        assert!((0..1_000_000_000).contains(&reading.1), "{reading:?}");
    }
}

/// `chelis check` accepts a clock read in a function that declares `IO`.
#[test]
fn check_accepts_a_declared_io_clock_read() {
    for clock in CLOCKS {
        let dir = tempdir().expect("tempdir");
        let src = dir.path().join("reader.ch");
        fs::write(
            &src,
            format!("def read() -> (i64, i64) ! {{IO}} = {clock}()\n"),
        )
        .expect("write source");
        let output = chelis()
            .args(["check", src.to_str().unwrap()])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let json: serde_json::Value =
            serde_json::from_slice(&output).expect("check output should be json");
        assert_eq!(json["score"].as_f64(), Some(1.0), "{clock}: {json}");
        assert_eq!(
            json["errors"].as_array().map(Vec::len),
            Some(0),
            "{clock}: {json}"
        );
    }
}

/// The rejected twin: a function declared pure cannot read a clock, and
/// `chelis check` names the undeclared `IO` effect.
#[test]
fn check_rejects_a_clock_read_in_a_pure_function() {
    for clock in CLOCKS {
        let dir = tempdir().expect("tempdir");
        let src = dir.path().join("reader.ch");
        fs::write(
            &src,
            format!("def read() -> (i64, i64) ! {{}} = {clock}()\n"),
        )
        .expect("write source");
        let output = chelis()
            .args(["check", src.to_str().unwrap()])
            .assert()
            .failure()
            .get_output()
            .clone();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("UnhandledEffect") && stdout.contains("IO"),
            "{clock}: expected an UnhandledEffect naming IO, got stdout={stdout} stderr={}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

/// Builds `source` for C in a fresh directory, runs the executable, and
/// returns its stdout.
fn build_and_run(source: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("reader.ch");
    let out = dir.path().join("out");
    fs::write(&src, source).expect("write source");
    chelis()
        .args([
            "build",
            src.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out.to_str().unwrap(),
        ])
        .assert()
        .success();
    let output = std::process::Command::new(out.join("reader"))
        .output()
        .expect("the built executable runs");
    assert!(
        output.status.success(),
        "compiled clock program failed: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// Compiled C reads the host wall clock: the reading lies between two host
/// reads taken around the executable's run, with nanoseconds in `[0, 10^9)`.
#[test]
fn compiled_wall_read_lies_between_host_reads() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("reader.ch");
    let out = dir.path().join("out");
    fs::write(&src, "reading = clock_wall_read()\n").expect("write source");
    chelis()
        .args([
            "build",
            src.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out.to_str().unwrap(),
        ])
        .assert()
        .success();
    let lower = host_now();
    let output = std::process::Command::new(out.join("reader"))
        .output()
        .expect("the built executable runs");
    let upper = host_now();
    assert!(output.status.success(), "{output:?}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let reading = printed_pair(&stdout, "reading");
    assert!(
        lower <= reading && reading <= upper,
        "{lower:?} <= {reading:?} <= {upper:?}"
    );
    assert!((0..1_000_000_000).contains(&reading.1), "{reading:?}");
}

/// Compiled C reads a monotonic clock: two reads in sequence never decrease,
/// and a read inside an `IO` function runs once per call.
#[test]
fn compiled_monotonic_reads_never_decrease() {
    let stdout = build_and_run(
        "def read() -> (i64, i64) ! {IO} = clock_monotonic_read()\n\
         first = read()\n\
         second = clock_monotonic_read()\n\
         third = read()\n",
    );
    let first = printed_pair(&stdout, "first");
    let second = printed_pair(&stdout, "second");
    let third = printed_pair(&stdout, "third");
    assert!(
        first <= second && second <= third,
        "{first:?} {second:?} {third:?}"
    );
    for reading in [first, second, third] {
        assert!((0..1_000_000_000).contains(&reading.1), "{reading:?}");
    }
}

/// An uncalled definition that reads a clock is ordinary compiled code: the
/// program builds, and the executable never reads the clock.
#[test]
fn an_uncalled_clock_read_definition_builds() {
    for clock in CLOCKS {
        let stdout = build_and_run(&format!(
            "def dead() -> (i64, i64) ! {{IO}} = {clock}()\nout = 7i32\n"
        ));
        assert_eq!(stdout, "out = 7\n", "{clock}");
    }
}

/// The rejected twin: building a pure function that reads a clock fails for
/// the semantic reason, the undeclared `IO` effect, not a target gate.
#[test]
fn build_rejects_a_clock_read_in_a_pure_function_for_its_effect() {
    for clock in CLOCKS {
        let dir = tempdir().expect("tempdir");
        let src = dir.path().join("reader.ch");
        let out = dir.path().join("out");
        fs::write(
            &src,
            format!("def read() -> (i64, i64) ! {{}} = {clock}()\nreading = read()\n"),
        )
        .expect("write source");
        let output = chelis()
            .args([
                "build",
                src.to_str().unwrap(),
                "--target",
                "c",
                "--output",
                out.to_str().unwrap(),
            ])
            .assert()
            .failure()
            .get_output()
            .clone();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("declared with effects `{}`")
                && stderr.contains("performs effects `{IO}`"),
            "{clock}: expected the undeclared IO effect, got stderr={stderr}"
        );
        assert!(
            !stderr.contains("unsupported:"),
            "{clock}: a target gate must not reject a legal clock read: {stderr}"
        );
    }
}
