//! chelis#1492: no test in this crate derives a probe path from `temp_dir()`
//! by hand.
//!
//! The defect was not that any single path was wrong. It was that twenty
//! call sites each built their own path, and with no shared helper to drift
//! against, eight had reached a fixed literal with no entropy at all. The
//! scan below is therefore as much the fix as `probe_dir` is: it is what
//! stops the twenty-first site reintroducing the class.
//!
//! Two limits, stated so the scan is not read as more than it is. It enforces
//! "no raw `temp_dir()`", NOT "everything goes through `probe_dir`":
//! `tempfile` used directly is not flagged, and `tests/fused_in_place_exec.rs`
//! already does that - unique and self-cleaning, so not the defect. And it
//! matches text, so `env::var("TMPDIR")` or a hard-coded `/tmp` would evade
//! it. Spellings that differ only in whitespace do not evade it in practice,
//! because the gate runs `cargo fmt --check` and rustfmt collapses them into
//! the matched form.

mod common;

use std::path::PathBuf;

use common::probe_dir;

/// Flags a source file that derives a probe path from `temp_dir()` itself
/// rather than going through the helper.
///
/// Shared by the scan and its negative control so the control cannot pass
/// against logic the scan does not use.
fn offending_lines(source: &str) -> Vec<String> {
    source
        .lines()
        .filter(|line| line.contains("temp_dir()"))
        .filter(|line| !line.trim_start().starts_with("//"))
        .map(|line| line.trim().to_string())
        .collect()
}

fn test_sources() -> Vec<(String, String)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut found = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("the crate's tests directory is readable") {
        let path = entry.expect("a readable directory entry").path();
        // This file names the call in its own scan logic and in its negative
        // control's fixture, so it is the one source that cannot be scanned.
        let is_this_file = path
            .file_stem()
            .is_some_and(|s| s == "probe_dir_discipline");
        if path.extension().is_some_and(|ext| ext == "rs") && !is_this_file {
            let name = path
                .file_name()
                .expect("a file with an extension has a name")
                .to_string_lossy()
                .to_string();
            found.push((
                name,
                std::fs::read_to_string(&path).expect("readable source"),
            ));
        }
    }
    assert!(
        found.len() > 15,
        "expected to scan the whole suite, found only {} files - has the layout moved?",
        found.len()
    );
    found
}

#[test]
fn no_test_builds_a_probe_path_from_temp_dir_directly() {
    let mut offenders = Vec::new();
    for (name, source) in test_sources() {
        for line in offending_lines(&source) {
            offenders.push(format!("{name}: {line}"));
        }
    }
    assert!(
        offenders.is_empty(),
        "these sites build a probe path themselves instead of using \
         `common::probe_dir`, which is how chelis#1492's collision class \
         returned once already:\n  {}",
        offenders.join("\n  ")
    );
}

#[test]
fn the_scan_detects_a_violation() {
    // Negative control: an all-clear from the scan above means nothing unless
    // the scan can fail.
    let violation = r#"    let dir = std::env::temp_dir().join("chelis_probe");"#;
    assert_eq!(offending_lines(violation).len(), 1);
    // A comment mentioning the call is not a violation.
    assert!(offending_lines("    // never call temp_dir() here").is_empty());
}

#[test]
fn two_probe_directories_never_share_a_path() {
    let first = probe_dir("collision_check");
    let second = probe_dir("collision_check");
    assert_ne!(
        first.path(),
        second.path(),
        "two probes with the same label must still get distinct directories - \
         this is the property a process id alone does not give: the \
         sanitizers job runs `cargo test`, where two tests calling one helper \
         are threads of a single pid"
    );
    assert!(first.path().is_dir() && second.path().is_dir());
}

#[test]
fn a_probe_directory_is_removed_when_it_is_dropped() {
    let path = {
        let probe = probe_dir("cleanup_check");
        std::fs::write(probe.path().join("probe.c"), "int main(void){return 0;}")
            .expect("the probe directory exists and is writable");
        probe.path().to_path_buf()
    };
    assert!(
        !path.exists(),
        "the probe directory outlived its guard at {}; on `main` this is \
         what left 170 directories in the temp dir per suite run",
        path.display()
    );
}
