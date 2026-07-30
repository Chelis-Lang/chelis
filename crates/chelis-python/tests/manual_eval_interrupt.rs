//! Manual acceptance gate for chelis#914 + chelis#930: bindings interrupt
//! latency, in both the evaluation phase and the front end.
//!
//! `#[ignore]` because it installs the bindings into the repo's `.venv` (a
//! maturin build of `chelis-python`) and then spends minutes measuring real
//! SIGINTs against running work. Neither belongs in the inner loop.
//!
//! **Manual command:**
//!
//! ```text
//! cargo nextest run -p chelis-python --test manual_eval_interrupt -- --ignored
//! ```
//!
//! **Expected success condition:** the probe prints `ALL PASS: 4/4`. The key
//! lines are SIGINT-to-`KeyboardInterrupt` under the 250 ms budget during
//! evaluation (chelis#914) and under 1 s during a front-end-dominated compile
//! (chelis#930). Measured on an M-series workstation: 49.1 ms and 62.1 ms
//! respectively, the latter abandoning 2.9 s of a 3.9 s compile. The pre-fix
//! baselines are "never, until the evaluation completes" and "the remaining
//! compile time" — 17.3 s on the chelis#930 repro.
//!
//! Note the absolute compile time depends on how the extension was built:
//! `uv pip install` produces a release build, so the same 1500-declaration
//! source compiles in seconds here and in ~20 s under a debug build. The
//! probe derives its interrupt point from a measured baseline for exactly
//! that reason, and asserts a ratio as well as a budget.
//!
//! **Prerequisite:** a repo-root `.venv` (`uv venv --python 3.11`) — the same
//! one `.cargo/config.toml` points `PYO3_PYTHON` at. The test skips with a
//! clear message rather than failing if it or `uv` is absent, so a checkout
//! without the Python toolchain does not report a spurious failure.
//!
//! The assertions live in `bindings/python/tests/manual_eval_interrupt.py`;
//! this wrapper only provisions and runs it, so the probe stays runnable on
//! its own during development.

use std::path::PathBuf;
use std::process::Command;

#[test]
#[ignore = "manual chelis#914 / chelis#930 acceptance gate: installs bindings into .venv and measures real SIGINT latency"]
fn eval_interrupt_latency_manual_acceptance_oracle() {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let python = repo_root.join(".venv/bin/python");
    if !python.exists() {
        eprintln!(
            "skipping: {} not found; create it with `uv venv --python 3.11`",
            python.display()
        );
        return;
    }
    if Command::new("uv").arg("--version").output().is_err() {
        eprintln!("skipping: `uv` not on PATH");
        return;
    }

    // Build and install the extension module under test. Without this the
    // probe would silently measure whatever `chelis` build happened to be
    // installed already — which is exactly how a green run could mean nothing.
    let status = Command::new("uv")
        .args(["pip", "install", "--python"])
        .arg(&python)
        .arg("--force-reinstall")
        .arg(repo_root.join("bindings/python"))
        .current_dir(&repo_root)
        .status()
        .expect("run uv pip install");
    assert!(status.success(), "installing bindings/python failed");

    let output = Command::new(&python)
        .arg(repo_root.join("bindings/python/tests/manual_eval_interrupt.py"))
        .current_dir(&repo_root)
        .output()
        .expect("run the interrupt probe");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    println!("{stdout}");
    assert!(
        output.status.success(),
        "interrupt-latency probe failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("ALL PASS"),
        "probe did not report ALL PASS\nstdout:\n{stdout}"
    );
}
