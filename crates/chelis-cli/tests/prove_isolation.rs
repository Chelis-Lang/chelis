//! Tier B process-isolation acceptance (RFC residual: cvc5 fails by process
//! abort, so the only way to GUARANTEE a Tier B solve never takes `chelis`
//! down is to run cvc5 in a child process whose crash maps to a clean Tier C
//! result).
//!
//! This is the end-to-end oracle for `chelis_prove::worker`: it runs the real
//! `chelis prove` against the flagship opaque-invariant example, both
//! normally (obligations must reach the SMT tier through the worker) and with
//! the worker forced to crash on every solve (the parent must SURVIVE and the
//! obligations must fall to Tier C, not crash the process).
//!
//! Run:
//!   cargo nextest run -p chelis-cli --features smt --test prove_isolation
//! with `LD_LIBRARY_PATH` set to the uv python lib (AGENTS.md).
#![cfg(feature = "smt")]

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crate is two levels below the repo root")
        .to_path_buf()
}

fn chelis_bin() -> PathBuf {
    assert_cmd::cargo_bin!("chelis").to_path_buf()
}

fn flagship_example() -> PathBuf {
    repo_root().join("examples/opaque_invariants.ch")
}

/// Run `chelis prove <flagship>` with an optional worker-crash sentinel.
/// Returns (success, stdout).
fn run_prove(crash_mode: Option<&str>) -> (bool, String) {
    let mut cmd = Command::new(chelis_bin());
    cmd.arg("prove").arg(flagship_example());
    if let Some(mode) = crash_mode {
        cmd.env("CHELIS_PROVE_WORKER_CRASH", mode);
    }
    let out = cmd
        .output()
        .expect("running `chelis prove` must not fail to spawn");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
    )
}

/// Baseline: with isolation on (the `chelis` binary enables it) and no crash,
/// the producer obligations are proved at the SMT tier -- i.e. the worker
/// child really did run cvc5 and return its result to the parent. If this
/// said `fuzz` instead, isolation would be silently routing everything to
/// Tier C and the crash-recovery test below would be vacuous.
#[test]
fn isolated_worker_proves_obligations_at_smt_tier() {
    let (ok, stdout) = run_prove(None);
    assert!(
        ok,
        "baseline `chelis prove` must succeed; stdout:\n{stdout}"
    );
    assert!(
        stdout.contains("proved (smt)"),
        "obligations must reach the SMT tier through the worker; stdout:\n{stdout}"
    );
}

/// Test-worker transport is selected by the explicit test-support entry,
/// never by an ambient marker in a production host's environment.
#[test]
fn production_worker_keeps_stdout_transport_with_a_libtest_marker() {
    let output = Command::new(chelis_bin())
        .arg("prove")
        .arg(flagship_example())
        .env("CHELIS_PROVE_TEST_WORKER", "support::solver_worker")
        .output()
        .expect("spawn chelis prove");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{output:?}");
    assert!(stdout.contains("proved (smt)"), "{stdout}");
}

/// The point of isolation: a worker that ABORTS on every solve (cvc5's actual
/// failure mode -- an uncatchable C++ process abort) must NOT take `chelis`
/// down. The parent must exit successfully and the obligations must fall to
/// Tier C (`fuzz`), never `smt` (the worker never returned an smt verdict).
#[test]
fn parent_survives_a_worker_that_aborts_on_every_solve() {
    let (ok, stdout) = run_prove(Some("abort"));
    assert!(
        ok,
        "the parent must survive a worker abort (not die by signal); stdout:\n{stdout}"
    );
    assert!(
        stdout.contains("proved (fuzz)"),
        "obligations must fall to Tier C when the worker crashes; stdout:\n{stdout}"
    );
    assert!(
        !stdout.contains("proved (smt)"),
        "a crashing worker can never return an smt verdict; stdout:\n{stdout}"
    );
}

/// A worker that PANICS (a Rust-level failure, e.g. an unwrap deep in a
/// dependency) is recovered the same way -- the parent survives and falls to
/// Tier C.
#[test]
fn parent_survives_a_worker_that_panics_on_every_solve() {
    let (ok, stdout) = run_prove(Some("panic"));
    assert!(
        ok,
        "the parent must survive a worker panic; stdout:\n{stdout}"
    );
    assert!(
        stdout.contains("proved (fuzz)"),
        "obligations must fall to Tier C when the worker panics; stdout:\n{stdout}"
    );
}

/// A worker that overflows its STACK (SIGABRT, also uncatchable in-process)
/// is recovered the same way.
#[test]
fn parent_survives_a_worker_that_overflows_its_stack() {
    let (ok, stdout) = run_prove(Some("overflow"));
    assert!(
        ok,
        "the parent must survive a worker stack overflow; stdout:\n{stdout}"
    );
    assert!(
        stdout.contains("proved (fuzz)"),
        "obligations must fall to Tier C when the worker overflows; stdout:\n{stdout}"
    );
}
