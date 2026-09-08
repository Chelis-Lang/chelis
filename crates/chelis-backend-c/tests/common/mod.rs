//! Shared probe-directory helper for the `chelis-backend-c` test suite.
//!
//! Every test in this crate that compiles C writes sources, objects, and
//! executables into a directory under `std::env::temp_dir()`. Twenty such
//! sites had grown by copy-paste and eight had drifted to a fixed literal
//! name with no entropy at all (chelis#1492), so two tests - or two processes
//! running the same test - shared one directory and could read each other's
//! fixtures. The same class produced real intermittent `macOS Workspace`
//! failures in `chelis-runtime`, fixed in chelis#1490.
//!
//! Stamping a process id into each of the twenty sites, which is the shape
//! chelis#1490 used, would not have fixed all of them. The
//! `backend-sanitizers` job runs `cargo test -p chelis-backend-c`, where the
//! tests in one binary are threads of ONE process: `cblas_available` here and
//! in `rt_cleanup_redteam.rs` is called by two tests apiece and is not
//! memoized, so both would still race in a single pid's directory.
//! `tempfile` creates with `O_EXCL` and retries on collision, which makes the
//! defect unrepresentable rather than improbable - and one helper, rather
//! than twenty call sites, is what stops the convention drifting again.

// Each integration-test binary compiles this whole module, so a helper an
// individual binary does not call is dead code from its point of view.
#![allow(dead_code)]

use tempfile::{Builder, TempDir};

/// A unique scratch directory for one probe, removed when the returned guard
/// is dropped.
///
/// `label` survives into the path (`chelis_<label>_a1b2c3`), so a directory
/// observed while a test is running is still traceable to the test that
/// wrote it.
///
/// Bind the guard for as long as the directory is needed. Both
/// `let _ = probe_dir("x");` and `let dir = probe_dir("x").path().to_path_buf();`
/// compile and then delete the directory before the test can use it; the
/// latter is a one-line "tidy" of the two-line form every call site uses, so
/// it is the likelier mistake. It fails loudly at the first write rather than
/// silently, and it is the same hazard `tempfile::TempDir` itself carries.
pub fn probe_dir(label: &str) -> TempDir {
    Builder::new()
        .prefix(&format!("chelis_{label}_"))
        .tempdir()
        .unwrap_or_else(|err| panic!("could not create a probe directory for `{label}`: {err}"))
}
