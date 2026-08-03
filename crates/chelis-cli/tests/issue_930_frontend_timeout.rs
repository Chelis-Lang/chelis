//! `chelis eval --timeout` during a FRONT-END-bound compile (chelis#930).
//!
//! ## Background
//!
//! `--timeout` (chelis#914) is two-stage: a watchdog trips the cooperative
//! cancellation token, and — because cancellation used to be observed only at
//! evaluation node visits — hard-exits the process a grace period later if
//! nothing unwound. For a compile-bound program the cooperative stage could
//! never fire, so the backstop was not a backstop at all: it was the normal
//! path, and the reported deadline was systematically `<N> + grace`.
//!
//! chelis#930 made the front end poll the same token, so a compile-bound
//! program now unwinds cooperatively. The backstop stays (see
//! `install_eval_timeout`), but it is now genuinely a last resort.
//!
//! ## Coverage
//!
//! Positive: `timeout_trips_cooperatively_during_the_front_end` — the flag
//! fires with the documented message and, decisively, does so *before* the
//! hard-exit grace could have elapsed. Elapsed time is the only thing that
//! distinguishes the two stages, since they print the same text by design.
//!
//! Negative parity: `front_end_polling_does_not_disturb_an_untimed_compile`
//! and `generous_timeout_does_not_disturb_a_front_end_heavy_program` — the
//! per-declaration polling must not cancel a program nobody asked to cancel.

use assert_cmd::Command;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tempfile::tempdir;

/// Mirrors `TIMEOUT_HARD_EXIT_GRACE` in `chelis-cli`'s `main.rs`. Kept as a
/// local constant rather than imported because the binary crate exposes
/// nothing; if the grace changes, this test's margin needs revisiting, which
/// is the point of naming it here.
const HARD_EXIT_GRACE: Duration = Duration::from_secs(5);

const TIMEOUT_SECS: u64 = 2;

/// Enough declarations that the front end runs for far longer than
/// `TIMEOUT_SECS` in a debug build, with an evaluation of one multiply-add so
/// nothing but compilation can be responsible for the wall-clock. Matches
/// chelis#930's repro (1500 declarations, ~70 KB).
const FRONT_END_HEAVY_DEFS: usize = 1500;

/// Small enough to compile and evaluate in well under a second.
const SMALL_DEFS: usize = 20;

/// [05-OBS-6] labels every named root, including a program with exactly one.
const EXPECTED_SMALL_STDOUT: &str = "result = 1";

fn front_end_heavy_program(defs: usize) -> String {
    let mut source = String::with_capacity(defs * 48);
    for index in 0..defs {
        source.push_str(&format!(
            "def f{index}(x: i64) -> i64 = x * {index}i64 + 1i64\n"
        ));
    }
    source.push_str("result = f0(1i64)\n");
    source
}

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

/// Manual gate (chelis#930): this builds a 1500-declaration program,
/// subprocesses the real binary, and asserts a wall-clock bound
/// (`elapsed + 1s < 7s`). Wall-clock bounds do not hold on a contended
/// box -- the repo has measured 2,434s vs 24s for the same suite under
/// load -- so per the review it runs as a documented manual gate, not in
/// the default suite. Command + expected success condition:
/// `docs/manual_gates.md` ("frontend timeout trips cooperatively").
#[test]
#[ignore = "wall-clock bound; run manually per docs/manual_gates.md (chelis#930)"]
fn timeout_trips_cooperatively_during_the_front_end() {
    let dir = tempdir().expect("tempdir");
    let path = write_program(
        dir.path(),
        "front_end_heavy.ch",
        &front_end_heavy_program(FRONT_END_HEAVY_DEFS),
    );

    let started = Instant::now();
    let (code, stdout, stderr) = eval(&[
        "eval",
        "--timeout",
        &TIMEOUT_SECS.to_string(),
        "--file",
        path.to_str().expect("utf-8 path"),
    ]);
    let elapsed = started.elapsed();

    assert_ne!(code, Some(0), "a timed-out compile must exit non-zero");
    assert!(
        stderr.contains(&format!(
            "evaluation timed out after {TIMEOUT_SECS}s (--timeout)"
        )),
        "expected the documented timeout message, got stderr: {stderr}"
    );
    assert!(
        stdout.trim().is_empty(),
        "a timed-out compile must not print a result, got stdout: {stdout}"
    );
    // The discriminator. Before chelis#930 the front end could not observe the
    // token, so this run could only end via the hard-exit backstop at
    // `TIMEOUT_SECS + HARD_EXIT_GRACE`. Landing meaningfully earlier proves the
    // cooperative path ran. One second of margin absorbs process startup and a
    // loaded machine without weakening the distinction.
    let backstop_at = Duration::from_secs(TIMEOUT_SECS) + HARD_EXIT_GRACE;
    assert!(
        elapsed + Duration::from_secs(1) < backstop_at,
        "the compile should have been cancelled cooperatively near \
         {TIMEOUT_SECS}s; took {elapsed:?}, which is the hard-exit backstop \
         at {backstop_at:?}"
    );
}

#[test]
fn generous_timeout_does_not_disturb_a_front_end_heavy_program() {
    let dir = tempdir().expect("tempdir");
    let path = write_program(dir.path(), "small.ch", &front_end_heavy_program(SMALL_DEFS));

    let (code, stdout, stderr) = eval(&[
        "eval",
        "--timeout",
        "600",
        "--file",
        path.to_str().expect("utf-8 path"),
    ]);

    assert_eq!(code, Some(0), "stderr: {stderr}");
    assert_eq!(stdout.trim(), EXPECTED_SMALL_STDOUT);
    assert!(
        !stderr.contains("timed out"),
        "an armed but untripped token must not cancel a compile: {stderr}"
    );
}

#[test]
fn front_end_polling_does_not_disturb_an_untimed_compile() {
    let dir = tempdir().expect("tempdir");
    let path = write_program(dir.path(), "small.ch", &front_end_heavy_program(SMALL_DEFS));

    let (code, stdout, stderr) = eval(&["eval", "--file", path.to_str().expect("utf-8 path")]);

    assert_eq!(code, Some(0), "stderr: {stderr}");
    assert_eq!(
        stdout.trim(),
        EXPECTED_SMALL_STDOUT,
        "with no token installed the front end must behave exactly as before"
    );
}
