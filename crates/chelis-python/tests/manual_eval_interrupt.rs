//! Manual acceptance gate for chelis#914: bindings interrupt latency.
//!
//! `#[ignore]` because it installs the bindings into the repo's `.venv` (a
//! maturin build of `chelis-python`) and then spends ~10 s measuring a real
//! SIGINT against a running evaluation. Neither belongs in the inner loop.
//!
//! **Manual command:**
//!
//! ```text
//! cargo nextest run -p chelis-python --test manual_eval_interrupt -- --ignored
//! ```
//!
//! **Expected success condition:** the probe prints `ALL PASS: 3/3`, the key
//! line being SIGINT-to-`KeyboardInterrupt` under the 250 ms budget. Measured
//! 50.5 ms on an M-series workstation, debug build; the pre-fix baseline is
//! "never, until the evaluation completes on its own".
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
#[ignore = "manual chelis#914 acceptance gate: installs bindings into .venv and measures real SIGINT latency"]
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
