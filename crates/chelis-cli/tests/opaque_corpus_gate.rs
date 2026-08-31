//! W7 opaque-invariants corpus: the DEFAULT-GATE oracle (RFC D-CORPUS).
//!
//! This is the non-smt half of the W7 oracle and runs in the default
//! `cargo nextest` gate. It drives two Python runners (the no-shell scripting
//! policy: the corpus logic is Python, invoked here, not reimplemented in
//! Rust) against the FRESHLY BUILT non-smt `chelis` binary:
//!
//!   1. SOLVER-FREE (the load-bearing W7 deliverable): asserts the default
//!      (non-smt) binary links ZERO cvc5 symbols and `chelis check`s every
//!      check-lane corpus program to its pinned exit code. This is the
//!      executable form of the RFC's "no solver in the check loop" guarantee.
//!      The cross-build IDENTITY arm (non-smt vs smt check verdicts) needs
//!      both binaries and is a documented MANUAL gate (see the corpus README)
//!      plus the `--features smt` companion test.
//!
//!   2. CHECK-LANE COVERAGE: asserts every check-lane target token
//!      (CheckErrorKind / WF-message marker / clean-positive) is hit by >= 1
//!      generated program. The prove-lane coverage is the `--features smt`
//!      companion (the obligation surface only compiles under `smt`).
//!
//! The corpus is committed; this test runs it against the live binary, so a
//! regression in the shipped diagnostics fails the gate. Python selection
//! follows the repository's managed-interpreter contract: an explicit
//! `PYO3_PYTHON` is authoritative, otherwise `.venv/bin/python` is used.

use std::path::PathBuf;
use std::process::Command;

#[path = "../../../tests/support/managed_python.rs"]
mod managed_python;

/// Repo root = crates/chelis-cli/../.. .
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crate is two levels below the repo root")
        .to_path_buf()
}

fn corpus_dir() -> PathBuf {
    repo_root().join("tests/corpus/opaque_invariants")
}

/// The freshly built `chelis` binary for THIS test run's feature set. In the
/// default `cargo nextest` gate this is the non-smt build.
fn chelis_bin() -> PathBuf {
    assert_cmd::cargo_bin!("chelis").to_path_buf()
}

fn require_managed_python() -> PathBuf {
    managed_python::managed_python(&repo_root()).unwrap_or_else(|error| panic!("{error}"))
}

/// Run a corpus Python runner, returning (success, stdout, stderr).
fn run_runner(script: &str, args: &[&str]) -> (bool, String, String) {
    let py = require_managed_python();
    let out = Command::new(py)
        .arg(corpus_dir().join(script))
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("spawn {script}: {e}"));
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

// The solver-free assertion is only meaningful for the DEFAULT (non-smt)
// binary: under `--features smt` the binary deliberately links cvc5, so
// "zero cvc5 symbols" is false by design. This is the load-bearing
// solver-free gate for the default build, so do NOT weaken the assertion;
// instead skip it under `--features smt` (the smt build asserts the
// opposite -- that cvc5 IS linked -- through the smt companion suite).
#[test]
#[cfg_attr(
    feature = "smt",
    ignore = "solver-free gate applies only to the default (non-smt) binary; the smt binary links cvc5 by design"
)]
fn check_is_solver_free_on_the_corpus() {
    // The non-smt binary: zero cvc5 symbols (assertion A) + every check-lane
    // program checks to its pinned exit (assertion C). The smt CONTROL +
    // cross-build IDENTITY arm live in the manual gate / smt companion, so
    // --require-control is NOT passed here.
    let bin = chelis_bin();
    let (ok, stdout, stderr) =
        run_runner("solver_free.py", &["--nonsmt-bin", bin.to_str().unwrap()]);
    assert!(
        ok,
        "solver-free gate failed.\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("cvc5 symbols = 0 (PASS)"),
        "the non-smt binary must link zero cvc5 symbols: {stdout}"
    );
    assert!(
        stdout.contains("SOLVER-FREE GATE: PASS"),
        "missing pass line: {stdout}"
    );
}

#[test]
fn check_lane_targets_are_all_covered() {
    // Measure check-lane coverage against the freshly built binary. The
    // prove-lane is owned by the smt companion; the runner drops prove-only
    // targets when --prove-bin is absent, so a green here means EVERY
    // check-lane target token is hit by >= 1 generated program.
    let bin = chelis_bin();
    let (ok, stdout, stderr) = run_runner(
        "coverage_runner.py",
        &["--check-bin", bin.to_str().unwrap()],
    );
    assert!(
        ok,
        "check-lane coverage gate failed.\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("COVERAGE GATE: PASS"),
        "missing coverage pass line: {stdout}"
    );
    // The skipped prove lane must be reported, never silently dropped.
    assert!(
        stdout.contains("SKIPPED LANES (no binary): prove"),
        "the prove lane must be reported as skipped here: {stdout}"
    );
}

#[test]
fn corpus_is_in_sync_with_the_generator() {
    // Regenerating must produce a byte-identical manifest (the committed
    // corpus is mechanically generated, never hand-patched). A drift here
    // means the committed programs/manifest are stale vs generate_corpus.py.
    // Generated into a TEMP dir so this test never mutates the committed tree
    // (and cannot race other tests reading the corpus).
    let py = require_managed_python();
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = Command::new(&py)
        .arg(corpus_dir().join("generate_corpus.py"))
        .arg("--out-dir")
        .arg(tmp.path())
        .output()
        .expect("run generator");
    assert!(out.status.success(), "generator failed: {out:?}");
    let committed = std::fs::read_to_string(corpus_dir().join("manifest.json"))
        .expect("read committed manifest");
    let regenerated = std::fs::read_to_string(tmp.path().join("manifest.json"))
        .expect("read regenerated manifest");
    assert_eq!(
        committed, regenerated,
        "committed manifest.json is stale vs generate_corpus.py; regenerate the corpus"
    );
    // Every committed program file must be byte-identical to its regenerated
    // twin (the manifest equality alone does not pin the program bytes).
    let committed_programs = corpus_dir().join("programs");
    let regen_programs = tmp.path().join("programs");
    for entry in std::fs::read_dir(&committed_programs).expect("read committed programs") {
        let name = entry.expect("dir entry").file_name();
        let a = std::fs::read_to_string(committed_programs.join(&name)).expect("committed prog");
        let b = std::fs::read_to_string(regen_programs.join(&name))
            .unwrap_or_else(|_| panic!("regenerated program missing: {name:?}"));
        assert_eq!(
            a, b,
            "committed program {name:?} is stale vs generate_corpus.py"
        );
    }
    // ...and no EXTRA regenerated file exists that the committed tree lacks.
    let committed_count = std::fs::read_dir(&committed_programs).unwrap().count();
    let regen_count = std::fs::read_dir(&regen_programs).unwrap().count();
    assert_eq!(
        committed_count, regen_count,
        "committed programs/ has a different file count than the regenerated tree"
    );
}
