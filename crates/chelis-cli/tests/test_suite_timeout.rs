//! chelis#927 — the public test command owns a hard whole-suite deadline.
//!
//! These probes deliberately hang at lifecycle points where a per-test
//! timeout cannot help. The public supervisor must bound the command, keep
//! NDJSON honest, and reap the complete suite process group.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write fixture");
}

fn make_reef_package(name: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let pkg = dir.path().join(name);
    fs::create_dir_all(pkg.join("src")).expect("mkdir src");
    fs::create_dir_all(pkg.join("tests")).expect("mkdir tests");
    write_file(
        &pkg.join("reef.toml"),
        &format!(
            r#"schema = "1"

[package]
name = "{name}"
version = "0.1.0"
compiler = "={version}"
module_prefix = "Deadline"
"#,
            version = chelis_compiler_api::COMPILER_VERSION,
        ),
    );
    write_file(
        &pkg.join("src/main.ch"),
        "module Deadline.Main\n\ndef noop() -> unit = test_assert(true, \"noop\")\n",
    );
    write_file(
        &pkg.join("tests/smoke.ch"),
        r#"module Deadline.Tests.Smoke

def test_ok() -> unit = test_assert(true, "ok")
"#,
    );
    (dir, pkg)
}

fn run_json_hang(pkg: &Path, hook: &str, timeout_seconds: &str) -> std::process::Output {
    let started = Instant::now();
    let mut command = Command::cargo_bin("chelis").expect("binary");
    command
        .timeout(Duration::from_secs(10))
        .current_dir(pkg)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_TEST_INTERNAL_TESTING", "1")
        .env(hook, "1")
        .args([
            "test",
            "tests/",
            "--json",
            "--suite-timeout",
            timeout_seconds,
        ]);
    if hook == "CHELIS_TEST_HANG_AFTER_SUITE" {
        command.env("CHELIS_TEST_EMIT_FINALIZED_SUITE", "1");
    }
    let output = command.output().expect("run");
    assert!(
        started.elapsed() < Duration::from_secs(8),
        "public suite deadline did not bound {hook}"
    );
    output
}

fn json_lines(output: &std::process::Output) -> Vec<Value> {
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stdout.trim().is_empty(), "missing NDJSON timeout report");
    stdout
        .lines()
        .map(|line| {
            serde_json::from_str(line)
                .unwrap_or_else(|err| panic!("invalid NDJSON line {line:?}: {err}"))
        })
        .collect()
}

#[test]
fn preparation_hang_is_bounded_and_json_is_explicitly_incomplete() {
    let (_dir, pkg) = make_reef_package("suite-timeout-preparation");
    let output = run_json_hang(&pkg, "CHELIS_TEST_HANG_BEFORE_SUITE", "1");
    assert_eq!(
        output.status.code(),
        Some(1),
        "timeout is a failed suite, not success or a runner parse error; stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );

    let lines = json_lines(&output);
    let timeout = lines
        .iter()
        .find_map(|line| line.get("suite"))
        .expect("explicit suite timeout record");
    assert_eq!(timeout["status"], "timeout");
    assert_eq!(timeout["incomplete"], true);
    let summary = lines
        .last()
        .and_then(|line| line.get("summary"))
        .expect("final summary");
    assert_eq!(summary["incomplete"], true);
    assert_eq!(summary["passed"], 0);
    assert_eq!(summary["failed"], 1);
}

#[test]
fn preparation_hang_plain_output_is_explicitly_incomplete() {
    let (_dir, pkg) = make_reef_package("suite-timeout-preparation-plain");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .timeout(Duration::from_secs(10))
        .current_dir(&pkg)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_TEST_INTERNAL_TESTING", "1")
        .env("CHELIS_TEST_HANG_BEFORE_SUITE", "1")
        .args(["test", "tests/", "--suite-timeout", "1"])
        .output()
        .expect("run");
    assert_eq!(output.status.code(), Some(1));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("<suite>"));
    assert!(stdout.contains("suite timeout after 1s"));
    assert!(stdout.contains("1 failed (suite incomplete)"));
}

#[test]
fn hang_hook_requires_internal_testing_gate() {
    let (_dir, pkg) = make_reef_package("suite-timeout-hook-gate");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .timeout(Duration::from_secs(10))
        .current_dir(&pkg)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env_remove("CHELIS_TEST_INTERNAL_TESTING")
        .env("CHELIS_TEST_EMIT_FINALIZED_SUITE", "1")
        .env("CHELIS_TEST_HANG_BEFORE_SUITE", "1")
        .args(["test", "tests/", "--suite-timeout", "3"])
        .output()
        .expect("run");
    assert_eq!(
        output.status.code(),
        Some(0),
        "a production environment variable must not activate a test hook; stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn former_suite_worker_environment_marker_cannot_bypass_deadline() {
    let (_dir, pkg) = make_reef_package("suite-timeout-worker-marker");
    let started = Instant::now();
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .timeout(Duration::from_secs(10))
        .current_dir(&pkg)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_INTERNAL_TEST_SUITE_WORKER", "1")
        .env("CHELIS_TEST_INTERNAL_TESTING", "1")
        .env("CHELIS_TEST_HANG_BEFORE_SUITE", "1")
        .args(["test", "tests/", "--json", "--suite-timeout", "1"])
        .output()
        .expect("run");
    assert!(started.elapsed() < Duration::from_secs(5));
    assert_eq!(output.status.code(), Some(1));
    let lines = json_lines(&output);
    assert_eq!(
        lines.last().expect("summary")["summary"]["incomplete"],
        true
    );
}

#[test]
fn timeout_exit_is_bounded_when_stdout_consumer_stops_reading() {
    let (_dir, pkg) = make_reef_package("suite-timeout-stdout-backpressure");
    let started = Instant::now();
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_chelis"))
        .current_dir(&pkg)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_TEST_INTERNAL_TESTING", "1")
        .env("CHELIS_TEST_PROGRESS_ROWS", "6000")
        .env("CHELIS_TEST_HANG_BEFORE_SUITE", "1")
        .args(["test", "tests/", "--json", "--suite-timeout", "1"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn");

    let status = loop {
        if let Some(status) = child.try_wait().expect("poll") {
            break status;
        }
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "timeout reporter blocked indefinitely on an unread stdout pipe"
        );
        std::thread::sleep(Duration::from_millis(20));
    };
    assert_eq!(status.code(), Some(1));
    assert!(started.elapsed() < Duration::from_secs(4));
    let mut stderr = Vec::new();
    child
        .stderr
        .take()
        .expect("stderr pipe")
        .read_to_end(&mut stderr)
        .expect("read stderr");
    assert!(
        String::from_utf8_lossy(&stderr).contains("suite incomplete"),
        "writable stderr lacked incomplete timeout diagnostic"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn normal_output_forwarding_is_part_of_whole_command_deadline() {
    use std::os::fd::AsRawFd;

    let (_dir, pkg) = make_reef_package("suite-timeout-normal-backpressure");
    let mut source = String::from("module Deadline.Tests.Backpressure\n\n");
    for index in 0..100 {
        source.push_str(&format!(
            "def test_{index}() -> unit = test_assert(true, \"ok\")\n"
        ));
    }
    write_file(&pkg.join("tests/smoke.ch"), &source);

    let started = Instant::now();
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_chelis"))
        .current_dir(&pkg)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["test", "tests/", "--json", "--suite-timeout", "3"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn");
    let stdout_fd = child.stdout.as_ref().expect("stdout pipe").as_raw_fd();
    let pipe_size = unsafe { libc::fcntl(stdout_fd, libc::F_SETPIPE_SZ, 4096) };
    assert!(pipe_size >= 4096, "shrink stdout pipe");

    let status = loop {
        if let Some(status) = child.try_wait().expect("poll") {
            break status;
        }
        assert!(
            started.elapsed() < Duration::from_secs(6),
            "normal output forwarding escaped the whole-command deadline"
        );
        std::thread::sleep(Duration::from_millis(20));
    };
    assert_eq!(status.code(), Some(1));
    assert!(started.elapsed() < Duration::from_secs(5));
    let mut stderr = Vec::new();
    child
        .stderr
        .take()
        .expect("stderr pipe")
        .read_to_end(&mut stderr)
        .expect("read stderr");
    assert!(
        String::from_utf8_lossy(&stderr).contains("output forwarding exceeded"),
        "bounded failure must identify incomplete output: {}",
        String::from_utf8_lossy(&stderr)
    );
}

#[cfg(target_os = "linux")]
#[test]
fn blocked_stderr_cannot_publish_a_perfect_stdout_summary() {
    use std::os::fd::AsRawFd;

    let (_dir, pkg) = make_reef_package("suite-timeout-stderr-backpressure");
    let stderr_file = pkg.join("large-stderr.bin");
    fs::write(&stderr_file, vec![b'x'; 128 * 1024]).expect("large stderr probe");
    let started = Instant::now();
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_chelis"))
        .current_dir(&pkg)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_TEST_INTERNAL_TESTING", "1")
        .env("CHELIS_TEST_BATCH_STDERR_FILE", &stderr_file)
        .args(["test", "tests/", "--json", "--suite-timeout", "3"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn");
    let stderr_fd = child.stderr.as_ref().expect("stderr pipe").as_raw_fd();
    let pipe_size = unsafe { libc::fcntl(stderr_fd, libc::F_SETPIPE_SZ, 4096) };
    assert!(pipe_size >= 4096, "shrink stderr pipe");

    let status = loop {
        if let Some(status) = child.try_wait().expect("poll") {
            break status;
        }
        assert!(
            started.elapsed() < Duration::from_secs(6),
            "stderr forwarding escaped the whole-command deadline"
        );
        std::thread::sleep(Duration::from_millis(20));
    };
    assert_eq!(status.code(), Some(1));
    assert!(started.elapsed() < Duration::from_secs(5));

    let mut stdout = Vec::new();
    child
        .stdout
        .take()
        .expect("stdout pipe")
        .read_to_end(&mut stdout)
        .expect("read stdout");
    let lines = stdout
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice::<Value>(line).expect("valid NDJSON"))
        .collect::<Vec<_>>();
    assert!(
        lines.iter().any(|line| line["suite"]["incomplete"] == true),
        "writable stdout lacked incomplete failure: {lines:#?}"
    );
    let summary = &lines.last().expect("summary")["summary"];
    assert_eq!(summary["incomplete"], true);
    assert_eq!(summary["failed"], 1);
    assert!(
        !lines.iter().any(|line| line["summary"]["failed"] == 0),
        "stderr failure left a false-perfect stdout summary: {lines:#?}"
    );
}

#[test]
fn maximum_suite_timeout_does_not_overflow_instant() {
    let (_dir, pkg) = make_reef_package("suite-timeout-u64-max");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .timeout(Duration::from_secs(10))
        .current_dir(&pkg)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "test",
            "missing-tests/",
            "--suite-timeout",
            "18446744073709551615",
        ])
        .output()
        .expect("run");
    assert_eq!(
        output.status.code(),
        Some(2),
        "invalid test path should fail normally, not panic from deadline overflow"
    );
    assert!(!String::from_utf8_lossy(&output.stderr).contains("panicked"));
}

#[test]
fn zero_suite_timeout_is_rejected_instead_of_coerced() {
    let (_dir, pkg) = make_reef_package("suite-timeout-zero");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&pkg)
        .args(["test", "tests/", "--suite-timeout", "0"])
        .output()
        .expect("run");
    assert_eq!(output.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("must be at least 1 second"),
        "unexpected diagnostic: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn suite_worker_has_no_externally_addressable_cli_route() {
    let (_dir, pkg) = make_reef_package("suite-timeout-no-hidden-route");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&pkg)
        .arg("__test_suite")
        .output()
        .expect("run");
    assert_eq!(output.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("unrecognized subcommand"),
        "removed suite route remained addressable: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[cfg(unix)]
#[test]
fn fork_failure_is_loud_bounded_and_cleans_progress_file() {
    let (dir, pkg) = make_reef_package("suite-timeout-fork-failure");
    let temp_path = dir.path().join("progress-tmp");
    fs::create_dir(&temp_path).expect("mkdir progress temp");
    let started = Instant::now();
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&pkg)
        .env("TMPDIR", &temp_path)
        .env("CHELIS_TEST_INTERNAL_TESTING", "1")
        .env("CHELIS_TEST_FORCE_SUITE_FORK_FAILURE", "1")
        .args(["test", "tests/", "--suite-timeout", "30"])
        .output()
        .expect("run");
    assert_eq!(output.status.code(), Some(2));
    assert!(started.elapsed() < Duration::from_secs(2));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("forced suite fork failure"),
        "fork failure was not reported: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_no_progress_tempfiles(&temp_path);
}

fn assert_no_progress_tempfiles(directory: &Path) {
    let leaked = fs::read_dir(directory)
        .expect("read temp directory")
        .filter_map(Result::ok)
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|name| name.starts_with("chelis-test-progress-") && name.ends_with(".ndjson"))
        .collect::<Vec<_>>();
    assert!(leaked.is_empty(), "leaked progress files: {leaked:?}");
}

#[cfg(target_os = "linux")]
fn wait_for_suite_child(public_pid: u32) -> i32 {
    let children_path = format!("/proc/{public_pid}/task/{public_pid}/children");
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Ok(children) = fs::read_to_string(&children_path)
            && let Some(pid) = children.split_whitespace().next()
        {
            return pid.parse::<i32>().expect("numeric suite pid");
        }
        assert!(
            Instant::now() < deadline,
            "suite child never appeared under public supervisor"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[cfg(target_os = "linux")]
fn assert_process_disappears(pid: i32) {
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let exists = unsafe { libc::kill(pid, 0) } == 0
            || std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH);
        if !exists {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "process {pid} survived suite-group cleanup"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[cfg(unix)]
#[test]
fn sigterm_grace_is_inside_the_advertised_suite_deadline() {
    let (_dir, pkg) = make_reef_package("suite-timeout-term-inclusive");
    let started = Instant::now();
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .timeout(Duration::from_secs(4))
        .current_dir(&pkg)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_TEST_INTERNAL_TESTING", "1")
        .env("CHELIS_TEST_IGNORE_SIGTERM", "1")
        .env("CHELIS_TEST_HANG_BEFORE_SUITE", "1")
        .args(["test", "tests/", "--json", "--suite-timeout", "1"])
        .output()
        .expect("run");
    let elapsed = started.elapsed();
    assert_eq!(output.status.code(), Some(1));
    assert!(
        elapsed < Duration::from_millis(1800),
        "TERM-resistant suite exceeded its advertised deadline: {elapsed:?}"
    );
    assert_eq!(
        json_lines(&output).last().expect("summary")["summary"]["incomplete"],
        true
    );
}

#[test]
fn finalization_hang_retains_rows_but_replaces_false_green_summary() {
    let (_dir, pkg) = make_reef_package("suite-timeout-finalization");
    let output = run_json_hang(&pkg, "CHELIS_TEST_HANG_AFTER_SUITE", "3");
    assert_eq!(output.status.code(), Some(1));

    let lines = json_lines(&output);
    assert!(
        lines.iter().any(|line| {
            line.get("test").and_then(Value::as_str) == Some("test_ok")
                && line.get("status").and_then(Value::as_str) == Some("pass")
        }),
        "completed test row was discarded: {lines:#?}"
    );
    assert_eq!(
        lines
            .iter()
            .filter(|line| line.get("summary").is_some())
            .count(),
        1,
        "child's perfect summary must be replaced, not retained"
    );
    let summary = &lines.last().expect("summary")["summary"];
    assert_eq!(summary["incomplete"], true);
    assert_eq!(summary["passed"], 1);
    assert_eq!(summary["failed"], 1);
}

#[test]
fn plain_timeout_counts_rows_not_fail_text_in_filename() {
    let (_dir, pkg) = make_reef_package("suite-timeout-plain-counts");
    fs::rename(pkg.join("tests/smoke.ch"), pkg.join("tests/FAIL_cases.ch"))
        .expect("rename fixture");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .timeout(Duration::from_secs(10))
        .current_dir(&pkg)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_TEST_INTERNAL_TESTING", "1")
        .env("CHELIS_TEST_HANG_AFTER_SUITE", "1")
        .args(["test", "tests/", "--suite-timeout", "3"])
        .output()
        .expect("run");
    assert_eq!(output.status.code(), Some(1));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("tests/FAIL_cases.ch"));
    assert!(
        stdout.contains("1 passed, 1 failed (suite incomplete)"),
        "filename text corrupted the structural counts: {stdout}"
    );
}

#[test]
fn normal_batch_stderr_is_byte_exact_and_separate_from_progress() {
    let (_dir, pkg) = make_reef_package("suite-timeout-batch-stderr");
    let stderr_probe =
        "\u{1e}chelis-test-row:{\"file\":\"forged\",\"test\":\"row\",\"status\":\"pass\"}";
    let binary_stderr = pkg.join("batch-stderr.bin");
    write_file(&binary_stderr, "");
    fs::write(&binary_stderr, [0xff, 0xfe, b'\n']).expect("write non-UTF-8 stderr probe");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .timeout(Duration::from_secs(10))
        .current_dir(&pkg)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_TEST_INTERNAL_TESTING", "1")
        .env("CHELIS_TEST_BATCH_STDERR", stderr_probe)
        .env("CHELIS_TEST_BATCH_STDERR_FILE", &binary_stderr)
        .args(["test", "tests/", "--json", "--batch-mode", "auto"])
        .output()
        .expect("run");
    assert_eq!(output.status.code(), Some(0));
    let mut expected_stderr = format!("{stderr_probe}\n").into_bytes();
    expected_stderr.extend_from_slice(&[0xff, 0xfe, b'\n']);
    assert_eq!(
        output.stderr, expected_stderr,
        "ordinary worker stderr must survive byte-for-byte"
    );
    let lines = json_lines(&output);
    assert_eq!(
        lines
            .iter()
            .filter(|line| line.get("test").and_then(Value::as_str) == Some("test_ok"))
            .count(),
        1
    );
    assert!(
        !lines
            .iter()
            .any(|line| line.get("file").and_then(Value::as_str) == Some("forged")),
        "worker stderr was misclassified as a completed row"
    );
}

#[test]
fn batch_stderr_survives_fallback_without_duplication() {
    let (_dir, pkg) = make_reef_package("suite-timeout-batch-stderr-fallback");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .timeout(Duration::from_secs(10))
        .current_dir(&pkg)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_TEST_INTERNAL_TESTING", "1")
        .env("CHELIS_TEST_BATCH_STDERR", "batch diagnostic before abort")
        .env("CHELIS_TEST_FORCE_BATCH_ABORT", "1")
        .args(["test", "tests/", "--json", "--batch-mode", "auto"])
        .output()
        .expect("run");
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(output.stderr, b"batch diagnostic before abort\n");
    let lines = json_lines(&output);
    assert_eq!(
        lines
            .iter()
            .filter(|line| line.get("test").and_then(Value::as_str) == Some("test_ok"))
            .count(),
        1,
        "fallback duplicated or lost the completed row: {lines:#?}"
    );
}

#[test]
fn expect_finalization_hang_retains_verdict_and_mode_summary() {
    let (_dir, pkg) = make_reef_package("suite-timeout-expect-finalization");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .timeout(Duration::from_secs(10))
        .current_dir(&pkg)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_TEST_INTERNAL_TESTING", "1")
        .env("CHELIS_TEST_EMIT_FINALIZED_SUITE", "1")
        .env("CHELIS_TEST_HANG_AFTER_SUITE", "1")
        .args([
            "test",
            "tests/",
            "--json",
            "--expect",
            "neg",
            "--suite-timeout",
            "3",
        ])
        .output()
        .expect("run");
    assert_eq!(output.status.code(), Some(1));

    let lines = json_lines(&output);
    assert!(
        lines.iter().any(|line| {
            line.get("file").and_then(Value::as_str) == Some("tests/smoke.ch")
                && line.get("expect").and_then(Value::as_str) == Some("neg")
                && line.get("verdict").and_then(Value::as_str) == Some("config-error")
        }),
        "completed expected-failure verdict was discarded: {lines:#?}"
    );
    assert_eq!(
        lines
            .iter()
            .filter(|line| line.get("summary").is_some())
            .count(),
        1,
        "child summary must be replaced, not retained"
    );
    let summary = &lines.last().expect("summary")["summary"];
    assert_eq!(summary["mode"], "neg");
    assert_eq!(summary["incomplete"], true);
    assert_eq!(summary["ok"], 0);
    assert_eq!(summary["failed"], 2);
}

#[test]
fn expect_finalization_hang_plain_output_retains_mode_verdict() {
    let (_dir, pkg) = make_reef_package("suite-timeout-expect-finalization-plain");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .timeout(Duration::from_secs(10))
        .current_dir(&pkg)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_TEST_INTERNAL_TESTING", "1")
        .env("CHELIS_TEST_EMIT_FINALIZED_SUITE", "1")
        .env("CHELIS_TEST_HANG_AFTER_SUITE", "1")
        .args(["test", "tests/", "--expect", "neg", "--suite-timeout", "3"])
        .output()
        .expect("run");
    assert_eq!(output.status.code(), Some(1));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("CONFIG-ERROR") && stdout.contains("tests/smoke.ch"),
        "completed verdict was discarded: {stdout}"
    );
    assert!(
        stdout.contains("0 ok, 2 failing (neg mode, suite incomplete)"),
        "missing mode-correct incomplete summary: {stdout}"
    );
}

#[cfg(unix)]
#[test]
fn timeout_retains_completed_batch_rows_and_reaps_hung_descendant() {
    let (_dir, pkg) = make_reef_package("suite-timeout-descendant");
    write_file(
        &pkg.join("tests/smoke.ch"),
        r#"module Deadline.Tests.Smoke

def test_one() -> unit = test_assert(true, "one")
def test_two() -> unit = test_assert(true, "two")
def test_three() -> unit = test_assert(true, "three")
def test_four() -> unit = test_assert(true, "four")
def test_five() -> unit = test_assert(true, "five")
"#,
    );
    let pid_file = pkg.join("hung-batch.pid");
    let started = Instant::now();
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .timeout(Duration::from_secs(10))
        .current_dir(&pkg)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_TEST_INTERNAL_TESTING", "1")
        .env("CHELIS_TEST_BATCH_STDERR", "diagnostic before timeout")
        .env("CHELIS_TEST_HANG_AFTER_BATCH", "1")
        .env("CHELIS_TEST_HANG_PID_FILE", &pid_file)
        .args([
            "test",
            "tests/",
            "--json",
            "--suite-timeout",
            "3",
            "--batch-mode",
            "auto",
        ])
        .output()
        .expect("run");
    assert!(started.elapsed() < Duration::from_secs(8));
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(output.stderr, b"diagnostic before timeout\n");
    let lines = json_lines(&output);
    let completed = lines
        .iter()
        .filter(|line| {
            line.get("test").is_some() && line.get("status").and_then(Value::as_str) == Some("pass")
        })
        .count();
    assert_eq!(
        completed, 5,
        "completed batch rows were discarded: {lines:#?}"
    );
    assert_eq!(
        lines.last().expect("summary")["summary"]["incomplete"],
        true
    );
    assert_eq!(lines.last().expect("summary")["summary"]["passed"], 5);
    assert_eq!(lines.last().expect("summary")["summary"]["failed"], 1);

    let pid: i32 = fs::read_to_string(&pid_file)
        .expect("batch hook wrote pid")
        .trim()
        .parse()
        .expect("numeric pid");
    let exists = unsafe { libc::kill(pid, 0) } == 0
        || std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH);
    assert!(
        !exists,
        "hung batch descendant {pid} survived suite timeout"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn killing_public_supervisor_reaps_suite_and_batch_descendants() {
    let (dir, pkg) = make_reef_package("suite-timeout-parent-cancellation");
    let temp_path = dir.path().join("progress-tmp");
    fs::create_dir(&temp_path).expect("mkdir progress temp");
    let pid_file = pkg.join("hung-batch.pid");
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_chelis"))
        .current_dir(&pkg)
        .env("TMPDIR", &temp_path)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_TEST_INTERNAL_TESTING", "1")
        .env("CHELIS_TEST_HANG_AFTER_BATCH", "1")
        .env("CHELIS_TEST_HANG_PID_FILE", &pid_file)
        .args([
            "test",
            "tests/",
            "--suite-timeout",
            "60",
            "--batch-mode",
            "auto",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn public supervisor");

    let suite_pid = wait_for_suite_child(child.id());
    let batch_pid = {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Ok(pid) = fs::read_to_string(&pid_file) {
                break pid.trim().parse::<i32>().expect("numeric batch pid");
            }
            if let Some(status) = child.try_wait().expect("poll public supervisor") {
                let mut stdout = String::new();
                child
                    .stdout
                    .take()
                    .expect("stdout")
                    .read_to_string(&mut stdout)
                    .expect("read stdout");
                let mut stderr = String::new();
                child
                    .stderr
                    .take()
                    .expect("stderr")
                    .read_to_string(&mut stderr)
                    .expect("read stderr");
                panic!(
                    "public supervisor exited before batch hook: {status}; stdout={stdout:?}; stderr={stderr:?}"
                );
            }
            assert!(
                Instant::now() < deadline,
                "batch descendant never reached cancellation hook"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    };

    child.kill().expect("kill public supervisor");
    child.wait().expect("reap public supervisor");

    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let suite_exists = unsafe { libc::kill(suite_pid, 0) } == 0
            || std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH);
        let batch_exists = unsafe { libc::kill(batch_pid, 0) } == 0
            || std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH);
        if !suite_exists && !batch_exists {
            assert_no_progress_tempfiles(&temp_path);
            break;
        }
        assert!(
            Instant::now() < deadline,
            "cancellation orphaned suite={suite_pid} (exists={suite_exists}) or batch={batch_pid} (exists={batch_exists})"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[cfg(target_os = "linux")]
#[test]
fn killed_suite_leader_with_hung_batch_is_bounded_and_incomplete() {
    let (dir, pkg) = make_reef_package("suite-leader-killed-with-batch");
    let temp_path = dir.path().join("progress-tmp");
    fs::create_dir(&temp_path).expect("mkdir progress temp");
    let pid_file = pkg.join("hung-batch.pid");
    let started = Instant::now();
    let child = std::process::Command::new(env!("CARGO_BIN_EXE_chelis"))
        .current_dir(&pkg)
        .env("TMPDIR", &temp_path)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_TEST_INTERNAL_TESTING", "1")
        .env("CHELIS_TEST_HANG_AFTER_BATCH", "1")
        .env("CHELIS_TEST_HANG_PID_FILE", &pid_file)
        .args([
            "test",
            "tests/",
            "--json",
            "--suite-timeout",
            "10",
            "--batch-mode",
            "auto",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn public supervisor");
    let suite_pid = wait_for_suite_child(child.id());
    let batch_pid = {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Ok(pid) = fs::read_to_string(&pid_file) {
                break pid.trim().parse::<i32>().expect("numeric batch pid");
            }
            assert!(Instant::now() < deadline, "batch hook was not reached");
            std::thread::sleep(Duration::from_millis(20));
        }
    };

    assert_eq!(unsafe { libc::kill(suite_pid, libc::SIGKILL) }, 0);
    let output = child.wait_with_output().expect("wait public supervisor");
    assert_eq!(output.status.code(), Some(1));
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "reader joins escaped the suite wall after leader loss"
    );
    let lines = json_lines(&output);
    assert!(
        lines.iter().any(|line| {
            line.get("suite")
                .and_then(|suite| suite.get("status"))
                .and_then(Value::as_str)
                == Some("abnormal-exit")
        }),
        "leader loss lacked an explicit incomplete record: {lines:#?}"
    );
    assert_eq!(
        lines.last().expect("summary")["summary"]["incomplete"],
        true
    );
    assert_process_disappears(batch_pid);
    assert_process_disappears(suite_pid);
    assert_no_progress_tempfiles(&temp_path);
}

#[cfg(target_os = "linux")]
#[test]
fn killed_suite_leader_before_preparation_is_bounded_and_nonempty() {
    let (dir, pkg) = make_reef_package("suite-leader-killed-before-preparation");
    let temp_path = dir.path().join("progress-tmp");
    fs::create_dir(&temp_path).expect("mkdir progress temp");
    let started = Instant::now();
    let child = std::process::Command::new(env!("CARGO_BIN_EXE_chelis"))
        .current_dir(&pkg)
        .env("TMPDIR", &temp_path)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_TEST_INTERNAL_TESTING", "1")
        .env("CHELIS_TEST_HANG_BEFORE_SUITE", "1")
        .args(["test", "tests/", "--suite-timeout", "10"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn public supervisor");
    let suite_pid = wait_for_suite_child(child.id());
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(unsafe { libc::kill(suite_pid, libc::SIGKILL) }, 0);
    let output = child.wait_with_output().expect("wait public supervisor");
    assert_eq!(output.status.code(), Some(1));
    assert!(started.elapsed() < Duration::from_secs(3));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("suite process exited abnormally"),
        "leader loss produced no plain incomplete record: {stdout:?}"
    );
    assert!(stdout.contains("suite incomplete"));
    assert_process_disappears(suite_pid);
    assert_no_progress_tempfiles(&temp_path);
}

#[cfg(not(unix))]
#[test]
fn public_test_fails_closed_without_process_group_cleanup() {
    let (_dir, pkg) = make_reef_package("suite-timeout-non-unix");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&pkg)
        .args(["test", "tests/"])
        .output()
        .expect("run");
    assert_eq!(output.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("requires Unix process-group semantics")
    );
}
