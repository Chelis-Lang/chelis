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
//! documented message and a non-zero exit, and does so through *cooperative*
//! cancellation rather than the watchdog's hard-exit backstop. The two are
//! indistinguishable by output (same message, same exit code), so the
//! wall-clock bound is what separates them; see the assertion's comment.
//!
//! Negative parity (the flag must not fire when it must not):
//! * `generous_timeout_does_not_disturb_a_fast_program` — a fast program under
//!   a large timeout produces its normal value and exit 0.
//! * `no_timeout_flag_runs_to_completion` — the same slow-ish program without
//!   the flag completes and prints its result, so the watchdog is opt-in and
//!   the per-node token check does not spuriously cancel.
//! * `timeout_message_is_absent_from_a_successful_run` — guards against the
//!   watchdog's hard-exit backstop firing on a run that already succeeded.
//!
//! ## The second eval lane
//!
//! `chelis eval --file` has two dispatches, and the tests above only reach
//! one. A file that resolves into a reef package routes through
//! `run_eval_in_context` → `eval_in_context` → `eval_compiled`; everything
//! else falls back to the legacy `try_eval` path. The two differ in how the
//! error reaches `cmd_eval`: the reef lane flattens a `CompilerError` to a
//! `String` in `join_eval_error` and re-wraps it in `EvalInContextError`,
//! so the cancellation sentinel survives only because that flattening
//! special-cases `is_cancellation`. `timeout_trips_inside_a_reef_package`
//! and `generous_timeout_does_not_disturb_a_reef_package_eval` cover that
//! lane, with the same positive/negative pairing.

use assert_cmd::Command;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tempfile::{TempDir, tempdir};

/// Long enough that a 2-second timeout is unambiguous. A flat fold, not a
/// recursion: deep recursion overflows the interpreter stack long before it
/// becomes slow.
const SLOW_PROGRAM: &str = "result = fold(fn (acc, x) -> acc + x, 0i64, range(0, 4000000))\n";

/// Finishes in milliseconds.
const FAST_PROGRAM: &str = "result = fold(fn (acc, x) -> acc + x, 0i64, range(0, 100))\n";

/// What `FAST_PROGRAM` (and its reef-package twin) prints on success.
///
/// `[05-OBS-6]` in `spec/05-risc-primitives.md` removed the
/// bare-when-single root form: every root renders `name = value` at every
/// exit in both lanes, so a lone root is `result = 4950`, not `4950`.
const EXPECTED_FAST_STDOUT: &str = "result = 4950";

fn write_program(dir: &Path, name: &str, body: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, body).expect("write program");
    path
}

/// `TIMEOUT_HARD_EXIT_GRACE` in `chelis-cli`'s `main.rs`: how long the
/// watchdog waits after the deadline for cooperative cancellation to unwind
/// before it hard-exits the process itself. Mirrored rather than exported
/// because it is an internal policy constant, not a CLI contract.
const HARD_EXIT_GRACE: Duration = Duration::from_secs(5);

/// A dep-free reef package exporting one helper, plus the package root.
///
/// The helper exists to make the routing self-proving. A snippet that
/// `import`s it can only evaluate through the reef-context lane: if
/// `detect_eval_package_root` failed to find the package and the CLI fell
/// back to the legacy path, the import would not resolve and the run would
/// fail with a name error instead of a timeout. So "stderr is the timeout
/// message" carries "the reef lane ran" with it.
fn reef_package() -> (TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join("demo");
    std::fs::create_dir_all(root.join("src")).expect("mkdir src");
    std::fs::write(
        root.join("reef.toml"),
        format!(
            r#"schema = "1"

[package]
name = "demo"
version = "0.1.0"
compiler = "={}"
module_prefix = "Demo"
"#,
            env!("CARGO_PKG_VERSION")
        ),
    )
    .expect("write reef.toml");
    std::fs::write(
        root.join("src/slow.ch"),
        "module Demo.Slow\n\ndef zero() -> i64 = 0i64\n",
    )
    .expect("write module");
    (dir, root)
}

/// The reef-lane counterparts of `SLOW_PROGRAM` / `FAST_PROGRAM`. The fold
/// seed comes from the imported `zero()` so the import is genuinely
/// consumed and cannot be optimized away as dead.
fn reef_snippet(iterations: u32) -> String {
    format!(
        "import Demo.Slow (zero)\n\n\
         result = fold(fn (acc, x) -> acc + x, zero(), range(0, {iterations}))\n"
    )
}

fn eval(args: &[&str]) -> (Option<i32>, String, String) {
    eval_impl(None, args)
}

/// `eval` with the working directory set, which is what puts a loose
/// snippet inside a reef package: the Phase H detector resolves a
/// non-`module` file's package root from the cwd.
fn eval_in(cwd: &Path, args: &[&str]) -> (Option<i32>, String, String) {
    eval_impl(Some(cwd), args)
}

fn eval_impl(cwd: Option<&Path>, args: &[&str]) -> (Option<i32>, String, String) {
    let mut command = Command::cargo_bin("chelis").expect("binary");
    command.env("CHELIS_STYLE_GATE_DISABLE", "1").args(args);
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    let output = command.output().expect("run chelis eval");
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
    // The ceiling is the hard-exit backstop boundary, not a generous
    // round number. Measured: with the per-node token checks deleted from
    // both eval lanes, this test still passed at 7.08s under the old
    // `< 20` bound — the watchdog's `process::exit` prints a byte-identical
    // message from a detached thread, so every other assertion here is
    // satisfied by the backstop alone. Anything at or past
    // `timeout + HARD_EXIT_GRACE` means cooperative cancellation did not
    // fire, which is the thing this test exists to prove.
    assert!(
        elapsed < Duration::from_secs(2) + HARD_EXIT_GRACE,
        "cancellation must unwind cooperatively before the watchdog's \
         hard-exit backstop; took {elapsed:?}"
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
    assert_eq!(stdout.trim(), EXPECTED_FAST_STDOUT);
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
        EXPECTED_FAST_STDOUT,
        "without --timeout nothing about eval changes"
    );
}

#[test]
fn timeout_message_is_absent_from_a_successful_run() {
    let dir = tempdir().expect("tempdir");
    let path = write_program(dir.path(), "fast.ch", FAST_PROGRAM);

    // A program that finishes immediately, under a timeout wide enough that
    // even a contended CI box completes the whole subprocess (compile phases
    // included) inside it: the process must exit 0 on its own before the
    // watchdog has any say. Review flagged `--timeout 1` here as a measured
    // flake risk under contention (2,434s vs 24s observed on rank_poly_tier3);
    // 10s keeps the assertion meaningful without racing the scheduler.
    let (code, stdout, stderr) = eval(&[
        "eval",
        "--timeout",
        "10",
        "--file",
        path.to_str().expect("utf-8 path"),
    ]);

    assert_eq!(code, Some(0), "stderr: {stderr}");
    assert_eq!(stdout.trim(), EXPECTED_FAST_STDOUT);
    assert!(!stderr.contains("timed out"), "stderr: {stderr}");
}

/// Positive, reef lane: `--timeout` trips on a slow program evaluated
/// against a reef package's compiled context, not just on the legacy path.
///
/// Three things are asserted together because each one alone can pass for
/// the wrong reason:
///
/// * The documented message and a non-zero exit — the user-facing contract.
/// * The raw sentinel is absent. The reef lane is the only path that
///   round-trips cancellation through a `String` (`join_eval_error` →
///   `EvalInContextError::Compile` → `Box<dyn Error>`), so if that
///   flattening stopped special-casing `is_cancellation` the user would see
///   `chelis::eval::cancelled` instead of a timeout.
/// * The run ended before the hard-exit backstop could fire. Without this
///   the test is a tautology: the watchdog's `process::exit` prints a
///   byte-identical message from a detached thread and would satisfy the
///   first assertion even if the token never reached this eval lane at all.
///   Cooperative cancellation returns at the first node visit after the
///   deadline, so anything at or past `timeout + HARD_EXIT_GRACE` means the
///   cooperative path did not fire. (A failure here on a heavily contended
///   machine is the CPU-starvation diagnostic in `CLAUDE.md`, not a
///   regression — re-run on a quiet box before believing it.)
#[test]
fn timeout_trips_inside_a_reef_package() {
    let (_dir, root) = reef_package();
    let snippet_dir = tempdir().expect("snippet tempdir");
    // Outside the package, so the cwd is the only thing that can put this
    // snippet in reef context.
    let path = write_program(snippet_dir.path(), "slow.ch", &reef_snippet(4_000_000));

    let started = Instant::now();
    let (code, stdout, stderr) = eval_in(
        &root,
        &[
            "eval",
            "--timeout",
            "2",
            "--file",
            path.to_str().expect("utf-8 path"),
        ],
    );
    let elapsed = started.elapsed();

    assert_ne!(code, Some(0), "a timed-out eval must exit non-zero");
    assert!(
        stderr.contains("evaluation timed out after 2s (--timeout)"),
        "expected the documented timeout message, got stderr: {stderr}"
    );
    assert!(
        !stderr.contains("chelis::eval::cancelled"),
        "the internal cancellation sentinel must not reach the user: {stderr}"
    );
    assert!(
        stdout.trim().is_empty(),
        "a timed-out eval must not print a result, got stdout: {stdout}"
    );
    assert!(
        elapsed < Duration::from_secs(2) + HARD_EXIT_GRACE,
        "cancellation must unwind cooperatively before the watchdog's \
         hard-exit backstop; took {elapsed:?}"
    );
}

/// Negative parity for the reef lane: the per-node token check added to
/// `eval_compiled` must not cancel a run that has no reason to be
/// cancelled, and the import must still resolve.
///
/// The stdout assertion does double duty. `4950` is only reachable if
/// `Demo.Slow.zero` resolved, which only happens on the reef-context path —
/// so this also proves the sibling test above is exercising that path
/// rather than silently falling back to the legacy one and timing out
/// there.
#[test]
fn generous_timeout_does_not_disturb_a_reef_package_eval() {
    let (_dir, root) = reef_package();
    let snippet_dir = tempdir().expect("snippet tempdir");
    let path = write_program(snippet_dir.path(), "fast.ch", &reef_snippet(100));

    let (code, stdout, stderr) = eval_in(
        &root,
        &[
            "eval",
            "--timeout",
            "600",
            "--file",
            path.to_str().expect("utf-8 path"),
        ],
    );

    assert_eq!(code, Some(0), "stderr: {stderr}");
    assert_eq!(stdout.trim(), EXPECTED_FAST_STDOUT);
    assert!(
        !stderr.contains("timed out"),
        "a fast reef-package eval must not report a timeout, got: {stderr}"
    );
}
