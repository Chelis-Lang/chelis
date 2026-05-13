//! Wave 2 cascade survey: enumerate every tuple-destructure linearity
//! violation surfaced by W1 PR 1 (`LinearityInfo::warnings`) across the
//! chelis-std + examples corpus.
//!
//! Output drives `docs/investigations/linearity_destructure_cleanup_survey.md`
//! and the threshold-rule disposition. The test asserts the survey
//! result is what the docs pin (zero corpus warnings as of W1's land);
//! later cleanup commits update the assertion if true positives are
//! fixed.
//!
//! The W1 warning channel is internal to `chelis-types`; the CLI's
//! `check` command does not surface it on stderr or in the JSON
//! report. This test is the only Rust-side caller that inspects
//! `LinearityInfo::warnings()` against the production corpus.
//!
//! The test removes itself when the W2-cascade flip commit drops
//! `LinearityInfo::warnings` entirely.

use chelis_reef::prepare_program_for_file;
use chelis_surf::desugar::desugar_program;
use chelis_types::{check_linearity, check_typed_program};
use std::path::{Path, PathBuf};

/// Corpus roots to walk. Mirrors the F3 PR 1 sweep set
/// (`examples/`, `examples/illustrative/`, `packages/`,
/// `crates/*/tests/`). `crates/*/tests/` is excluded here because
/// those `.ch` fixtures are intentionally adversarial (linearity
/// negative controls) and would not represent a production-corpus
/// signal.
fn corpus_roots() -> Vec<PathBuf> {
    let workspace_root = workspace_root();
    vec![
        workspace_root.join("examples"),
        workspace_root.join("packages"),
    ]
}

/// Resolve the workspace root from this test's manifest dir.
fn workspace_root() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    // crates/chelis-cli -> workspace root
    manifest
        .parent()
        .and_then(|p| p.parent())
        .map(PathBuf::from)
        .expect("workspace root resolves from CARGO_MANIFEST_DIR")
}

/// Walk a corpus root and collect every `.ch` file.
fn collect_ch_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if !root.exists() {
        return files;
    }
    let walker = walkdir::WalkDir::new(root)
        .sort_by_file_name()
        .into_iter()
        .filter_entry(|entry| {
            // Skip dot-prefixed entries and `target/` build outputs.
            let name = entry.file_name().to_string_lossy();
            if entry.depth() > 0 && name.starts_with('.') {
                return false;
            }
            if entry.file_type().is_dir() && name == "target" {
                return false;
            }
            true
        });
    for entry in walker {
        let Ok(entry) = entry else {
            continue;
        };
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("ch") {
            continue;
        }
        files.push(path.to_path_buf());
    }
    files
}

/// Run the front-end pipeline through linearity and return the
/// surfaced `LinearityInfo::warnings()`. Returns `None` if the file
/// cannot be parsed, desugared, or type-checked (the survey only
/// cares about linearity-stage warnings on programs that reach the
/// linearity check). Reef-aware import resolution is used when the
/// file sits inside a reef package; otherwise the raw parser is
/// used.
fn linearity_warnings(file: &Path) -> Option<Vec<String>> {
    let source = std::fs::read_to_string(file).ok()?;

    let decls = match prepare_program_for_file(file).ok().flatten() {
        Some(prepared) => prepared.decls,
        None => chelis_surf::parser::parse_str(&source).ok()?,
    };
    let deep = desugar_program(&decls);
    let checked = check_typed_program(&deep).ok()?;
    let checked = chelis_effects::check_program(&checked).ok()?;
    let linearity = check_linearity(&checked).ok()?;
    let info = linearity.linearity();
    Some(
        info.warnings()
            .iter()
            .map(|w| format!("{:?}: {}", w.kind, w.message))
            .collect(),
    )
}

/// Aggregate warning summary across the corpus. Survey artifact.
struct SurveyReport {
    /// File-path-relative -> warning messages.
    by_file: Vec<(PathBuf, Vec<String>)>,
    /// Files that errored out before the linearity check ran
    /// (parse, desugar, or type-check failure). Reported in the
    /// survey markdown so coverage gaps are visible.
    skipped: Vec<PathBuf>,
}

fn run_survey() -> SurveyReport {
    let mut by_file = Vec::new();
    let mut skipped = Vec::new();
    // Snapshot lockfile state under example reef packages before
    // the walk so the test does not leave generated `reef.lock`
    // files lying around in the source tree afterwards. The chelis
    // -std package keeps its own committed lockfile; everything
    // else is a transient build artifact when materialized by
    // `prepare_program_for_file`.
    let workspace_root = workspace_root();
    let example_lockfile_dirs = collect_example_reef_packages(&workspace_root.join("examples"));
    let pre_state: Vec<(PathBuf, Option<Vec<u8>>)> = example_lockfile_dirs
        .iter()
        .map(|dir| {
            let lock = dir.join("reef.lock");
            (lock.clone(), std::fs::read(&lock).ok())
        })
        .collect();

    for root in corpus_roots() {
        for file in collect_ch_files(&root) {
            match linearity_warnings(&file) {
                Some(warnings) => {
                    if !warnings.is_empty() {
                        by_file.push((file, warnings));
                    }
                }
                None => skipped.push(file),
            }
        }
    }

    // Restore lockfile state under example reef packages.
    for (path, before) in pre_state {
        match before {
            Some(bytes) => {
                let _ = std::fs::write(&path, bytes);
            }
            None => {
                let _ = std::fs::remove_file(&path);
            }
        }
    }

    SurveyReport { by_file, skipped }
}

/// Find every directory under `root` that contains a `reef.toml`
/// manifest. Used to restore any `reef.lock` files the survey
/// inadvertently writes during reef-aware import resolution.
fn collect_example_reef_packages(root: &Path) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if !root.exists() {
        return dirs;
    }
    for entry in walkdir::WalkDir::new(root).into_iter().flatten() {
        if entry.file_name() == "reef.toml"
            && entry.file_type().is_file()
            && let Some(parent) = entry.path().parent()
        {
            dirs.push(parent.to_path_buf());
        }
    }
    dirs
}

/// Wave 2 cascade survey: assert the W1 destructure-warning channel
/// surfaces zero violations across the corpus. This pins the survey
/// result documented in
/// `docs/investigations/linearity_destructure_cleanup_survey.md` so
/// later corpus drift does not silently change the threshold-rule
/// disposition.
///
/// Marked `#[ignore]` because the survey walks every `.ch` file
/// under `examples/` and `packages/` through reef-aware import
/// resolution + the full front-end pipeline; the wall-clock cost
/// runs past the inner-loop budget. Manual gate:
/// `cargo test --package chelis-cli --test linearity_destructure_corpus_survey -- --ignored --nocapture`.
/// Re-run after any change to `LinearityInfo::warnings` plumbing,
/// destructure desugar, or corpus additions to confirm the
/// threshold-rule disposition still holds.
#[test]
#[ignore = "manual gate: corpus survey for Wave 2 cascade; see survey doc for command"]
fn corpus_emits_zero_destructure_warnings() {
    let report = run_survey();
    if !report.by_file.is_empty() {
        let lines: Vec<String> = report
            .by_file
            .iter()
            .flat_map(|(file, warnings)| {
                warnings
                    .iter()
                    .map(move |w| format!("{}: {w}", file.display()))
            })
            .collect();
        panic!(
            "expected zero LinearityInfo::warnings across the corpus; got {} warning(s):\n{}",
            lines.len(),
            lines.join("\n")
        );
    }
    // Skipped files are normal (some examples deliberately fail
    // type-check as negative controls). Print them via stderr only
    // when run with `--nocapture` so the survey artifact stays
    // tractable on the default `cargo test` path.
    if !report.skipped.is_empty() {
        eprintln!(
            "linearity_destructure_corpus_survey: {} file(s) skipped (parse / type-check failure before linearity)",
            report.skipped.len()
        );
    }
}
