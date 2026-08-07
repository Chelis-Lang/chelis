//! W7 opaque-invariants corpus: the `--features smt` half of the oracle
//! (RFC D-CORPUS).
//!
//! The prove-lane obligation surface only compiles under the `smt` feature
//! (the obligation collection + Tier B lowering live in the optional
//! `chelis-prove` dependency), mirroring the existing
//! `prove_invariant_obligations.rs` gating. This companion runs the FULL
//! coverage gate (both check + prove lanes) against the freshly built `smt`
//! binary, and the prove performance-sanity check.
//!
//! Run:
//!   cargo nextest run -p chelis-cli --features smt --test opaque_corpus_smt
//! with `LD_LIBRARY_PATH` set to the uv python lib (AGENTS.md).
#![cfg(feature = "smt")]

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crate is two levels below the repo root")
        .to_path_buf()
}

fn venv_python() -> PathBuf {
    std::env::var_os("PYO3_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join(".venv/bin/python"))
}

fn corpus_dir() -> PathBuf {
    repo_root().join("tests/corpus/opaque_invariants")
}

/// The freshly built `chelis` binary for THIS test run -- here the smt build.
fn smt_bin() -> PathBuf {
    assert_cmd::cargo_bin!("chelis").to_path_buf()
}

fn require_venv() -> PathBuf {
    let py = venv_python();
    assert!(
        py.exists(),
        "the configured managed Python is a documented build prerequisite (AGENTS.md); \
         set valid `PYO3_PYTHON` or create `.venv` with `uv venv --python 3.11`. Missing: {}",
        py.display()
    );
    py
}

fn run_runner(script: &str, args: &[&str]) -> (bool, String, String) {
    let py = require_venv();
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

#[test]
fn full_corpus_coverage_is_complete() {
    // Both lanes. The smt binary serves the prove lane (obligations) AND can
    // `chelis check` the check lane, so we point both --check-bin and
    // --prove-bin at it: a green here MEASURES that every targeted token --
    // every CheckErrorKind / WF marker / obligation status / injection /
    // starvation / producer position / perf -- is hit by >= 1 generated
    // program. Zero uncovered, with --require-all (no skipped lane allowed).
    let bin = smt_bin();
    let (ok, stdout, stderr) = run_runner(
        "coverage_runner.py",
        &[
            "--check-bin",
            bin.to_str().unwrap(),
            "--prove-bin",
            bin.to_str().unwrap(),
            "--require-all",
        ],
    );
    assert!(
        ok,
        "full coverage gate failed.\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("COVERAGE GATE: PASS"),
        "missing coverage pass line: {stdout}"
    );
    assert!(
        stdout.contains("uncovered:        0"),
        "every target token must be covered: {stdout}"
    );
}

#[test]
fn prove_performance_sanity_and_obligation_accounting() {
    // RFC D-CORPUS prove-performance sanity: a module with MANY exported
    // producers (the prove_perf_many_producers fixture, 40 SMT-provable
    // clamping constructors) must complete within a sane bound AND account
    // for every obligation: the obligation-record count equals the producer
    // count and the summary `obligations` field, with the sum of
    // pass/fail/unsupported/error == producer count.
    let bin = smt_bin();
    let prog = corpus_dir().join("programs/prove_perf_many_producers.ch");
    let start = Instant::now();
    let out = Command::new(&bin)
        .args(["prove", prog.to_str().unwrap(), "--json"])
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .output()
        .expect("run prove");
    let elapsed = start.elapsed();
    // Sane bound: the 40-producer SMT prove runs in well under a second
    // locally; 60s is a generous CI ceiling that still catches a runaway
    // (e.g. an exponential obligation blow-up).
    assert!(
        elapsed.as_secs() < 60,
        "prove on the many-producer module took {elapsed:?} (> 60s bound)"
    );
    let exit = out.status.code().unwrap_or(-1);
    assert_eq!(
        exit, 0,
        "all clamping-constructor obligations must pass (exit 0)"
    );

    // Tally obligation records + the summary.
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut obligation_count = 0usize;
    let mut accounted = 0usize;
    let mut summary_obligations: Option<i64> = None;
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        match v.get("kind").and_then(|k| k.as_str()) {
            Some("obligation") => {
                obligation_count += 1;
                if matches!(
                    v.get("status").and_then(|s| s.as_str()),
                    Some("passed" | "failed" | "unsupported" | "error")
                ) {
                    accounted += 1;
                }
            }
            Some("summary") => {
                summary_obligations = v.get("obligations").and_then(|o| o.as_i64());
            }
            _ => {}
        }
    }
    // The perf fixture has exactly 40 exported producers (PERF_PRODUCER_COUNT).
    assert_eq!(
        obligation_count, 40,
        "every exported producer must yield exactly one obligation record"
    );
    assert_eq!(
        accounted, obligation_count,
        "every obligation must carry a definite status (no silently-dropped producer)"
    );
    assert_eq!(
        summary_obligations,
        Some(obligation_count as i64),
        "the summary obligations count must equal the obligation-record count"
    );
}
