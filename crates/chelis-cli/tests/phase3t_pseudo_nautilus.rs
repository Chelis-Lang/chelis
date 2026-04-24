//! Phase 3t.5 — pseudo-nautilus fixture integration tests.
//!
//! The fixture in `tests/fixtures/pseudo_nautilus/` exists to demonstrate the
//! Phase 3t "hard rule" pattern end-to-end: all tests that verify internal
//! correctness are written in Chelis and run via `chelis test`; Python is
//! reserved for external-oracle parity (scipy, pandas, sympy, QuantLib) and
//! lives in a sibling `parity/` directory that the test runner never touches.
//!
//! Coverage here:
//! 1. `chelis test tests/` on the fixture exits 0 with every enumerated test
//!    reporting PASS.
//! 2. `chelis test --json tests/` emits one NDJSON record per test plus a
//!    summary, with no `"status":"fail"` rows.
//! 3. `parity/run_parity.py` runs when scipy is available and exits 0;
//!    otherwise the scipy-parity assertion is skipped (but the runner itself
//!    must still be syntactically valid Python that imports cleanly, which
//!    the test verifies via `python3 -c` before invoking the script).
//!
//! The fixture is copied from the checked-in source tree into a tempdir so
//! the reef graph resolver sees a self-contained package root, matching the
//! pattern used by `phase3t_test_std.rs::make_app`.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
use tempfile::tempdir;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pseudo_nautilus")
        .canonicalize()
        .expect("pseudo_nautilus fixture must exist on disk")
}

fn copy_dir_recursive(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).expect("create dir");
    for entry in fs::read_dir(src).expect("read dir") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        let target = dst.join(entry.file_name());
        if path.is_dir() {
            copy_dir_recursive(&path, &target);
        } else {
            fs::copy(&path, &target).expect("copy file");
        }
    }
}

fn package_std() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../packages/chelis-std")
        .canonicalize()
        .expect("chelis-std package must exist")
}

/// Stage the fixture into a tempdir and publish `chelis-std` into a local
/// reef home so the fixture's `chelis-std` dep resolves. Returns
/// `(tempdir guard, package root, reef home)`. Callers keep the guard alive
/// for the duration of the test.
fn stage_fixture(scratch_name: &str) -> (tempfile::TempDir, PathBuf, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let pkg = dir.path().join(scratch_name);
    copy_dir_recursive(&fixture_path(), &pkg);
    // Publish chelis-std into a scoped reef home so `import Std.Test` in
    // `tests/*.ch` resolves. Without this, module-init pre-checks would fail
    // with `unresolved import Std.Test`.
    let reef_home = dir.path().join("reef-home");
    let std_staging = dir.path().join("chelis-std-src");
    copy_dir_recursive(&package_std(), &std_staging);
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", std_staging.to_str().unwrap()])
        .assert()
        .success();
    (dir, pkg, reef_home)
}

#[test]
fn pseudo_nautilus_chelis_test_all_green() {
    // The fixture's Chelis-native tests must pass as a unit — this is the
    // "internal correctness, no external oracle needed" half of the hard rule.
    let (_dir, pkg, reef_home) = stage_fixture("pseudo-nautilus-green");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&pkg)
        .args(["test", "tests/"])
        .assert()
        .success()
        .code(0)
        .stdout(predicate::str::contains("test_erf_zero"))
        .stdout(predicate::str::contains("test_erf_symmetry"))
        .stdout(predicate::str::contains("test_erf_bounded"))
        .stdout(predicate::str::contains("test_erf_monotonic"))
        .stdout(predicate::str::contains("4 passed, 0 failed"))
        .stdout(predicate::str::contains("FAIL").not());
}

#[test]
fn pseudo_nautilus_chelis_test_json_all_pass() {
    // Every NDJSON row must carry `"status":"pass"`; the summary row must
    // carry `"failed":0`. This locks the machine-facing output shape so a
    // future regression that silently downgrades assertions to skips (or
    // drops tests entirely) fails this test.
    let (_dir, pkg, reef_home) = stage_fixture("pseudo-nautilus-json");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&pkg)
        .args(["test", "--json", "tests/"])
        .assert()
        .success()
        .code(0)
        .get_output()
        .stdout
        .clone();
    let text = String::from_utf8(output).expect("utf-8 stdout");
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(
        lines.len(),
        5,
        "expected 4 test rows + 1 summary row, got {lines:?}"
    );
    for (idx, line) in lines.iter().enumerate().take(4) {
        let record: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("row {idx} not valid JSON: {line:?} ({e})"));
        assert_eq!(
            record["status"], "pass",
            "row {idx} should be pass: {record}"
        );
        assert!(
            record.get("message").is_none(),
            "passing row {idx} should not carry a message: {record}"
        );
        let test_name = record["test"].as_str().expect("test name");
        assert!(
            test_name.starts_with("test_erf_"),
            "row {idx} test name does not match the fixture: {test_name}"
        );
    }
    let summary: serde_json::Value =
        serde_json::from_str(lines[4]).expect("summary row parses as JSON");
    assert_eq!(summary["summary"]["passed"], 4);
    assert_eq!(summary["summary"]["failed"], 0);
}

#[test]
fn pseudo_nautilus_chelis_test_filter_picks_single_test() {
    // `--filter` must be substring-matched against `<file>::<test>` so
    // `erf_symmetry` narrows to exactly the symmetry test. This guards
    // against a filter that silently runs all tests when it can't parse the
    // needle — a bug that would make the flag useless in CI.
    let (_dir, pkg, reef_home) = stage_fixture("pseudo-nautilus-filter");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&pkg)
        .args(["test", "--filter", "erf_symmetry", "tests/"])
        .assert()
        .success()
        .code(0)
        .stdout(predicate::str::contains("test_erf_symmetry"))
        .stdout(predicate::str::contains("1 passed, 0 failed"))
        .stdout(predicate::str::contains("test_erf_zero").not())
        .stdout(predicate::str::contains("test_erf_bounded").not())
        .stdout(predicate::str::contains("test_erf_monotonic").not());
}

#[test]
fn pseudo_nautilus_parity_script_imports_cleanly() {
    // The `parity/` half of the fixture must exist and be valid Python 3. We
    // don't force a full scipy run in the default `cargo test` gate because
    // scipy isn't guaranteed on every dev machine, but the script's syntax
    // and import structure are part of the checked-in contract — a broken
    // `run_parity.py` would silently undermine the "external oracle lives in
    // Python" half of the hard rule. `python3 -c "import ast; ast.parse(...)"`
    // catches syntactic regressions without needing scipy installed.
    let (_dir, pkg, _reef_home) = stage_fixture("pseudo-nautilus-parity-syntax");
    let script = pkg.join("parity/run_parity.py");
    assert!(
        script.exists(),
        "parity/run_parity.py missing from fixture: {}",
        script.display()
    );
    let source = fs::read_to_string(&script).expect("read parity script");
    // Quick sanity: the script must reference scipy.special.erf somewhere
    // (the whole point of the parity pattern is that the external oracle is
    // scipy, not a second Chelis implementation).
    assert!(
        source.contains("scipy.special"),
        "parity script must reference scipy.special: {}",
        script.display()
    );
    // Parse via python3's `ast` module — catches broken syntax without
    // importing scipy itself.
    let probe = StdCommand::new("python3")
        .arg("-c")
        .arg(format!(
            "import ast, pathlib; ast.parse(pathlib.Path(r'{}').read_text())",
            script.display()
        ))
        .output();
    match probe {
        Ok(out) if out.status.success() => {}
        Ok(out) => panic!(
            "parity/run_parity.py failed ast.parse:\nstderr={}",
            String::from_utf8_lossy(&out.stderr)
        ),
        Err(e) => {
            // python3 missing is a legitimate CI-environment state — we log
            // and skip rather than failing the suite. A truly Python-less
            // host can't run the parity half at all, and the `parity/`
            // directory existing is still enough to lock the contract.
            eprintln!("note: python3 not available ({e}); skipping parity ast.parse check");
        }
    }
}

/// Full scipy-parity gate. Ignored by default because scipy is not on every
/// dev machine and the script can take several seconds when it shells out to
/// `chelis eval` for every sample point. Run manually with:
///
///   cargo test -p chelis-cli --test phase3t_pseudo_nautilus \
///     -- --ignored pseudo_nautilus_parity_script_runs_with_scipy
///
/// Expected: exit 0, printed parity table with max |diff| below 1e-5.
#[test]
#[ignore]
fn pseudo_nautilus_parity_script_runs_with_scipy() {
    let (_dir, pkg, _reef_home) = stage_fixture("pseudo-nautilus-parity-run");
    // Put the freshly-built chelis binary on PATH so the parity script's
    // `chelis eval` subprocesses see it. `assert_cmd::cargo_bin` yields the
    // path; PATH prepend gives the script plain `chelis` resolution.
    let chelis_bin = assert_cmd::cargo_bin!("chelis");
    let chelis_dir = chelis_bin
        .parent()
        .expect("cargo_bin has a parent directory");
    let orig_path = std::env::var("PATH").unwrap_or_default();
    let new_path = format!("{}:{orig_path}", chelis_dir.display());
    let output = StdCommand::new("python3")
        .arg(pkg.join("parity/run_parity.py"))
        .arg("--strict")
        .env("PATH", new_path)
        .current_dir(&pkg)
        .output()
        .expect("spawn python3");
    if !output.status.success() {
        panic!(
            "parity script exited nonzero:\nstdout={}\nstderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
