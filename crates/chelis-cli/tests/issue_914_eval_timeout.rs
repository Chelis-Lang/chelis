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
//! documented message prefix and a non-zero exit.
//!
//! Which of the two timeout paths ran is deliberately NOT asserted here.
//! Cooperative unwinding is CPU-bound work racing a fixed wall-clock grace, so
//! on a contended machine the backstop legitimately wins and the process is
//! force-killed; that is the design, and the user-visible outcome is the same
//! failure. The rows below therefore assert only what `--timeout` promises,
//! and the ordering property lives in the two `#[ignore]`d manual-gate rows at
//! the bottom of this file, which are the ones with a precondition on load.
//!
//! Until chelis#1607 the two paths were indistinguishable by output and a
//! wall-clock bound was the only thing separating them. The hard-exit path now
//! says which it is, so they are separable by reading stderr, not timing it.
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
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
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

/// The suffix `chelis-cli` appends when the watchdog gives up on cooperation
/// and kills the process (chelis#1607). It is the only thing that distinguishes
/// the hard-exit path from the cooperative one: both print the same prefix and
/// both exit 1.
const BACKSTOP_SUFFIX: &str = "forced exit";

/// Slack over the backstop's own guarantee, for the default-suite rows.
///
/// The backstop terminates at `timeout + HARD_EXIT_GRACE`, and that is a
/// wall-clock sleep in a detached thread, so even a starved runner delivers it
/// within seconds of the deadline. Four times the entire budget leaves any
/// scheduling delay far below the bound.
///
/// What it catches is a process that exits far too late. It does NOT catch a
/// process that never exits: `Command::output` blocks until the child does, so
/// a true hang stops this test rather than failing this assertion, and the test
/// runner's slow-timeout is the oracle for that case.
const TERMINATION_SLACK: Duration = Duration::from_secs(30);

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
    // Termination, not ordering. `--timeout` promises the run stops; the
    // backstop promises it stops by `timeout + HARD_EXIT_GRACE` whatever the
    // program does. Exceeding this means the process exited far too late, not
    // that it hung: see `TERMINATION_SLACK`. Which path ran is the ignored
    // pair's business (chelis#1607).
    assert!(
        elapsed < Duration::from_secs(2) + HARD_EXIT_GRACE + TERMINATION_SLACK,
        "a timed-out eval must terminate on its own; took {elapsed:?}"
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
    // Shape check on the chelis#1607 line: the forced-exit wording is reserved
    // for a run the watchdog actually killed, and must never appear on a run
    // that finished by itself.
    assert!(
        !stderr.contains(BACKSTOP_SUFFIX),
        "a successful run must not claim a forced exit, got: {stderr}"
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
    assert!(
        !stderr.contains(BACKSTOP_SUFFIX),
        "a successful run must not claim a forced exit, got: {stderr}"
    );
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
/// * The run terminated on its own rather than hanging.
///
/// It no longer claims the cooperative path ran. That claim was carried by a
/// wall-clock bound, which cannot separate a forced exit from a slow one, and
/// the property itself does not hold under contention. `BACKSTOP_SUFFIX` makes
/// the paths separable by reading stderr, and the ignored rows at the bottom of
/// this file assert both polarities of the ordering under their stated load
/// preconditions (chelis#1607).
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
    // Termination, not ordering; see the sibling row above.
    assert!(
        elapsed < Duration::from_secs(2) + HARD_EXIT_GRACE + TERMINATION_SLACK,
        "a timed-out eval must terminate on its own; took {elapsed:?}"
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
    assert!(
        !stderr.contains(BACKSTOP_SUFFIX),
        "a successful run must not claim a forced exit, got: {stderr}"
    );
}

// The ordering property, as a manual gate in both polarities (chelis#1607).
// Measured unwind cost after `cancel()` on a 10-core macOS box: ~0.20 s idle,
// ~0.68 s at 24 spinner threads, 2.0-3.15 s at 50, past the 5 s grace at 100.
// `docs/manual_gates.md` carries the commands and the preconditions.

/// Spin `threads` busy loops until the returned guard is dropped.
///
/// Threads rather than child processes: no binary to find, nothing to reap if
/// the test panics, and the contention is identical because it is the same
/// scheduler deciding who runs.
struct Spinners {
    stop: Arc<AtomicBool>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl Spinners {
    fn start(threads: usize) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let handles = (0..threads)
            .map(|_| {
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || while !stop.load(Ordering::Relaxed) {})
            })
            .collect();
        Self { stop, handles }
    }
}

impl Drop for Spinners {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

fn slow_run_stderr() -> String {
    let dir = tempdir().expect("tempdir");
    let path = write_program(dir.path(), "slow.ch", SLOW_PROGRAM);
    let (_code, _stdout, stderr) = eval(&[
        "eval",
        "--timeout",
        "2",
        "--file",
        path.to_str().expect("utf-8 path"),
    ]);
    stderr
}

/// MANUAL GATE, ignored by default. Positive polarity of the ordering claim:
/// on an idle machine cancellation unwinds cooperatively and the watchdog never
/// forces the exit.
///
/// ```text
/// cargo nextest run -p chelis-cli --test issue_914_eval_timeout \
///     cooperative_unwind_precedes_the_backstop_on_an_idle_box -- --ignored
/// ```
///
/// Precondition: an otherwise idle box, one-minute load average below the core
/// count. Under contention this row fails truthfully rather than spuriously,
/// which is the whole reason it is not in the default suite.
#[test]
#[ignore = "load-sensitive by construction; manual gate, see docs/manual_gates.md"]
fn cooperative_unwind_precedes_the_backstop_on_an_idle_box() {
    let stderr = slow_run_stderr();
    assert!(
        stderr.contains("evaluation timed out after 2s (--timeout)"),
        "expected the documented timeout message, got stderr: {stderr}"
    );
    assert!(
        !stderr.contains(BACKSTOP_SUFFIX),
        "on an idle box cancellation must unwind before the watchdog forces \
         the exit; got stderr: {stderr}"
    );
}

/// MANUAL GATE, ignored by default. Negative polarity, and the twin that keeps
/// the row above honest: starve the machine and the backstop must fire and say
/// so. Without this, deleting every cancellation poll would leave the row above
/// green on a fast box and nothing would notice.
///
/// ```text
/// cargo nextest run -p chelis-cli --test issue_914_eval_timeout \
///     a_starved_box_falls_back_to_the_forced_exit --test-threads=1 -- --ignored
/// ```
///
/// Precondition: an otherwise idle box to start with, and `--test-threads=1`,
/// because this row deliberately oversubscribes every core tenfold for the few
/// seconds it runs. Expect it to make the machine unresponsive briefly.
#[test]
#[ignore = "oversubscribes the machine; manual gate, see docs/manual_gates.md"]
fn a_starved_box_falls_back_to_the_forced_exit() {
    let cores = std::thread::available_parallelism().map_or(8, std::num::NonZero::get);
    // Ten runnable threads per core is the ratio measured to push the unwind
    // past the 5 s grace; five was not enough (2.0-3.15 s of unwind).
    let _spinners = Spinners::start(cores * 10);
    let stderr = slow_run_stderr();
    assert!(
        stderr.contains(BACKSTOP_SUFFIX),
        "a starved box must reach the watchdog's forced exit, and the line \
         must say so; got stderr: {stderr}"
    );
    assert!(
        stderr.contains("evaluation timed out after 2s (--timeout)"),
        "the forced-exit line must still carry the documented prefix: {stderr}"
    );
}
