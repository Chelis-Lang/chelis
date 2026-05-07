//! Style gate: enforces `chelis fmt --check` and `chelis lint --check`
//! on every Surf and Deep file ingested by `chelis build`, `chelis
//! check`, `chelis validate`, and `chelis eval --file`.
//!
//! The gate is invoked once per user-supplied source file, before the
//! normal compile/check/eval pipeline runs. On any deviation from
//! canonical formatting or any lint violation, the gate returns an
//! error with a one-line per-issue diagnostic the user can act on. The
//! `--allow-style-violations` flag downgrades the failure to a stderr
//! warning and lets the build proceed; CI should never pass that flag.
//!
//! The shared exception list `STYLE_GATE_EXCEPTIONS` is the single
//! source of truth for both `cmd_lint` (the standalone subcommand) and
//! the gate. Each entry pairs a path glob with a rule id and a spec
//! cross-reference (`§N.M` of `spec/01-nomenclature.md`) explaining
//! why the case is exempt — free-form prose is not part of the schema.

use chelis_lint::{Exception, Violation};
use std::path::Path;

/// Path globs (relative to the lint root) that are exempt from specific
/// lint rules, with a mandatory cross-reference to the spec section
/// that explains why. New entries must cite a `§` ref or update the
/// spec to add one — opaque exemptions are not permitted.
///
/// The `Std.Test` carve-out for `surf-test-name-prefix` is implemented
/// inside the rule itself (it short-circuits when the source declares
/// `module Std.Test`) and does not need a path-glob entry here.
///
/// `crates/chelis-surf/tests/fixtures/*.ch`: the Surf parser test
/// corpus deliberately exercises legacy syntactic shapes — including
/// the `def name(...) : T = ...` colon form — to verify the parser
/// still accepts them so existing files compile after `chelis fmt`
/// rewrites them to canonical arrow form. Renaming or rewriting
/// these fixtures would defeat their purpose. §3.5 explicitly notes
/// the formatter rewrites colon to arrow, which is what the parser
/// must still accept on input.
pub fn exceptions() -> Vec<Exception> {
    vec![Exception {
        pattern: "crates/chelis-surf/tests/fixtures/*.ch".to_string(),
        rule_id: "surf-def-arrow-form".to_string(),
        cross_ref: "§3.5".to_string(),
    }]
}

/// Outcome of one style-gate run on one file.
pub struct GateOutcome {
    pub fmt_diff: Option<FmtDiff>,
    pub lint_violations: Vec<Violation>,
}

impl GateOutcome {
    pub fn is_clean(&self) -> bool {
        self.fmt_diff.is_none() && self.lint_violations.is_empty()
    }

    pub fn issue_count(&self) -> usize {
        self.lint_violations.len() + usize::from(self.fmt_diff.is_some())
    }
}

/// Captures a `fmt --check` failure: the file is not byte-identical to
/// its canonical re-print.
pub struct FmtDiff {
    pub path: std::path::PathBuf,
}

/// Environment variable that, when set to a non-empty value, disables
/// the style gate on every Surf and Deep ingestion path. Intended for
/// the integration-test corpus only — many tests synthesize ad-hoc
/// Surf to exercise type/effect/linearity behavior and do not care
/// about canonical formatting. Production CI must not set this; the
/// dedicated `style_gate.rs` integration test asserts that the gate
/// fires when the variable is unset.
const STYLE_GATE_DISABLE_ENV: &str = "CHELIS_STYLE_GATE_DISABLE";

fn env_disables_gate() -> bool {
    std::env::var(STYLE_GATE_DISABLE_ENV)
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false)
}

/// Run the style gate on `file`. Returns `Ok(())` if clean (or if
/// `allow_violations` is true, even when issues were found — a warning
/// is emitted to `stderr` in that case). Otherwise returns the joined
/// diagnostics as a `Box<dyn Error>` so callers can propagate it as the
/// existing CLI error type.
///
/// `allow_violations` plumbs the `--allow-style-violations` CLI flag.
/// CI must never pass it; it is intended for local emergency builds
/// and one-off migration scripts.
pub fn enforce_style_gate(
    file: &Path,
    source: &str,
    allow_violations: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if env_disables_gate() {
        return Ok(());
    }
    let outcome = run_gate(file, source);
    if outcome.is_clean() {
        return Ok(());
    }
    let report = format_outcome(file, &outcome);
    if allow_violations {
        eprintln!(
            "warning: style gate bypassed (--allow-style-violations): {} issue(s)",
            outcome.issue_count()
        );
        eprintln!("{report}");
        return Ok(());
    }
    Err(report.into())
}

/// Run the gate without printing or escape-hatching — used by tests
/// and any future caller that wants the structured outcome.
pub fn run_gate(file: &Path, source: &str) -> GateOutcome {
    let fmt_diff = check_fmt(file, source);
    let lint_violations = run_lint_for_single_file(file);
    GateOutcome {
        fmt_diff,
        lint_violations,
    }
}

fn check_fmt(file: &Path, source: &str) -> Option<FmtDiff> {
    let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("");
    let canonical = if ext == "dp" {
        match chelis_deep::parser::parse_str_strict(source) {
            Ok(exprs) => chelis_deep::printer::print_canonical(&exprs),
            // If the file doesn't parse, the regular compile path will
            // surface that error with a better message; we don't
            // double-report here.
            Err(_) => return None,
        }
    } else {
        match chelis_surf::parser::parse_str(source) {
            Ok(decls) => chelis_surf::format::format_program(&decls),
            Err(_) => return None,
        }
    };
    if canonical == source {
        None
    } else {
        Some(FmtDiff {
            path: file.to_path_buf(),
        })
    }
}

fn run_lint_for_single_file(file: &Path) -> Vec<Violation> {
    let rules = chelis_lint::registry::all_rules();
    // Lint the file's parent directory and filter to violations that
    // belong to this file. The walker requires a directory root.
    let parent = file.parent().unwrap_or_else(|| Path::new("."));
    let raw = match chelis_lint::lint(parent, &rules) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let mine: Vec<Violation> = raw.into_iter().filter(|v| v.path == file).collect();
    let exceptions_list = exceptions();
    chelis_lint::exceptions::apply_exceptions(&mine, &exceptions_list, parent)
}

fn format_outcome(file: &Path, outcome: &GateOutcome) -> String {
    let mut buf = String::new();
    if let Some(diff) = &outcome.fmt_diff {
        buf.push_str(&format!(
            "{}: not canonically formatted; run `chelis fmt --inplace {}` to fix\n",
            diff.path.display(),
            file.display()
        ));
    }
    for v in &outcome.lint_violations {
        buf.push_str(&format!("{v}\n"));
    }
    buf.push_str(&format!(
        "{} issue(s) blocked the build; pass `--allow-style-violations` to bypass (CI must not).",
        outcome.issue_count()
    ));
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_surf_file_passes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("clean.ch");
        // Canonical Surf — empty program is the simplest canonical form.
        let canonical = chelis_surf::format::format_program(&[]);
        std::fs::write(&path, &canonical).unwrap();
        let res = enforce_style_gate(&path, &canonical, false);
        assert!(res.is_ok(), "expected clean canonical pass; got {res:?}");
    }

    #[test]
    fn non_canonical_surf_file_fails() {
        // Source with extra trailing whitespace that won't survive fmt.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("noncanonical.ch");
        let bad_source = "def foo() -> i32 = 1   \n";
        std::fs::write(&path, bad_source).unwrap();
        let res = enforce_style_gate(&path, bad_source, false);
        assert!(res.is_err(), "expected fmt diff to fail the gate");
    }

    #[test]
    fn allow_violations_bypasses_failure() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("noncanonical.ch");
        let bad_source = "def foo() -> i32 = 1   \n";
        std::fs::write(&path, bad_source).unwrap();
        let res = enforce_style_gate(&path, bad_source, true);
        assert!(res.is_ok(), "--allow-style-violations should pass");
    }

    #[test]
    fn lint_violation_fails_clean_file() {
        // A file whose source IS canonically formatted but violates a
        // lint rule must still fail the gate. `_internal` parses as a
        // valid Surf identifier (the lexer accepts a leading
        // underscore) but `surf-value-snake-case` rejects leading
        // underscores per §3.2.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("leading_underscore.ch");
        let src = "def _internal() -> i32 = 1\n";
        // Write the canonical form so fmt-check passes.
        let decls = chelis_surf::parser::parse_str(src).unwrap();
        let canonical = chelis_surf::format::format_program(&decls);
        std::fs::write(&path, &canonical).unwrap();
        let res = enforce_style_gate(&path, &canonical, false);
        assert!(res.is_err(), "expected lint violation to fail the gate");
    }
}
