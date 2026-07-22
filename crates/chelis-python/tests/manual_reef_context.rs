//! Manual acceptance gate for reef-context dependency resolution (issue #816).
//!
//! Drives `bindings/python/tests/manual_reef_context.py`, which builds its own
//! temp reef project (depending on the bundled `chelis-std` package plus a
//! sibling library module) and asserts that `chelis.eval(..., project_root=)`
//! and `chelis.compile_and_load(..., project_root=)` resolve reef-declared
//! dependencies — including the in-context entry-DAG trap (an entry that calls
//! a library function must still compile to a callable, correctly-scoped
//! kernel) — plus auto-discovery, the no-root regression, and scalar-entry
//! rejection.
//!
//! Prerequisites (this workstation): the chelis toolchain `0.16.1` installed
//! (`~/.chelis`) with its reef registry populated, `uv`, and a built
//! `libchelis_runtime.a` discoverable via `CHELIS_RUNTIME_DIR`.
//!
//! Manual command (from the repo root; the first temp-project context compile
//! takes tens of seconds):
//!
//! ```sh
//! # bindings installed into py/.venv, runtime staticlib on CHELIS_RUNTIME_DIR
//! cargo build -p chelis-runtime --target-dir target/agents/<name>
//! export CHELIS_RUNTIME_DIR="$PWD/target/agents/<name>/debug"
//! export CHELIS_TOOLCHAIN=0.16.1
//! export DYLD_LIBRARY_PATH="$(py/.venv/bin/python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
//! cargo test -p chelis-python --test manual_reef_context -- --ignored
//! ```
//!
//! Expected success condition: the child Python process exits 0 after printing
//! "All reef-context acceptance checks passed."
//!
//! NOTE (Shoals — chelis#825): the plan's original acceptance imported
//! `Shoals.Pricing` (Black-Scholes → 10.4506). On this branch the post-0.16.1
//! compiler tightened the `with seed(...)` int64 rule, which the published
//! Shoals 0.23.1 dependency graph (built for the `=0.16.1` toolchain) trips
//! during library-context compilation — reproducible with the branch's own CLI
//! (`chelis eval --file`), independent of these bindings. Tracked as chelis#825
//! (the ecosystem-drift canary checks Shoals `main`, not the published 0.23.1
//! artifact). The acceptance therefore uses `chelis-std` + a sibling module,
//! which is dev-compiler-clean and exercises the identical resolution
//! machinery. The scalar eval path DOES produce 10.450575828552246 through the
//! shipped `0.16.1` CLI.

use std::path::PathBuf;
use std::process::Command;

#[test]
#[ignore = "manual acceptance gate (#816): needs the 0.16.1 toolchain + reef registry, uv, and CHELIS_RUNTIME_DIR (libchelis_runtime.a); builds a temp reef project (tens of seconds)"]
fn reef_context_manual_acceptance_oracle() {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let python = repo_root.join("py/.venv/bin/python");
    assert!(python.exists(), "expected {}", python.display());

    let status = Command::new("uv")
        .arg("pip")
        .arg("install")
        .arg("--python")
        .arg(&python)
        .arg("-e")
        .arg(repo_root.join("bindings/python"))
        .current_dir(&repo_root)
        .status()
        .expect("install chelis package");
    assert!(status.success(), "uv pip install failed");

    let status = Command::new(&python)
        .arg(repo_root.join("bindings/python/tests/manual_reef_context.py"))
        .current_dir(&repo_root)
        .status()
        .expect("run manual reef-context acceptance");

    assert!(status.success(), "manual reef-context acceptance failed");
}
