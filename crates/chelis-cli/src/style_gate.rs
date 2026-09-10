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
use std::path::{Path, PathBuf};
use std::process::Command;

/// Detect the Cargo workspace root that exception globs are anchored
/// against, probing from `probe_dir`.
///
/// Exception patterns in [`exceptions`] are authored workspace-root
/// relative (e.g. `crates/chelisup/bootstrap/chelisup.sh`), so
/// exception matching must strip the workspace-root prefix from each
/// violation's absolute path. The workspace root is detected with
/// `cargo locate-project --workspace --message-format plain` run with
/// its working directory set to `probe_dir`; cargo reports the
/// absolute path of the workspace root `Cargo.toml`, and the workspace
/// root is its parent directory.
///
/// `probe_dir` is the directory the lint operates on (a walk target's
/// directory, or a single file's parent) rather than the process CWD,
/// so `chelis lint --check /abs/workspace/crates` issued from an
/// unrelated directory still resolves the correct workspace root.
///
/// This is Cargo's own canonical workspace-locating mechanism, not a
/// hand-rolled filesystem walk-up: it is the same probe `cargo` itself
/// uses and it correctly resolves the workspace root from any
/// subdirectory of the workspace. Per
/// `feedback_no_walkup_filesystem_detection.md`, a hand-rolled walk-up
/// search for `Cargo.toml` / `.git` is rejected (brittle under
/// symlinks, mounts, permission edges); deferring to Cargo's probe is
/// the durable alternative.
///
/// If `cargo` is unavailable or the command fails (`probe_dir` is not
/// inside any Cargo workspace), this returns `Err` with a clear
/// diagnostic. No fallback path with different semantics is added: a
/// single detection mechanism, callers surface the failure as a clean
/// CLI error.
pub fn detect_lint_workspace_root(probe_dir: &Path) -> Result<PathBuf, String> {
    let output = Command::new("cargo")
        .args(["locate-project", "--workspace", "--message-format", "plain"])
        .current_dir(probe_dir)
        .output()
        .map_err(|err| {
            format!(
                "could not detect workspace root; failed to run `cargo locate-project`: {err}; run from within a Cargo workspace"
            )
        })?;
    if !output.status.success() {
        return Err(
            "could not detect workspace root; run from within a Cargo workspace".to_string(),
        );
    }
    let manifest = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if manifest.is_empty() {
        return Err(
            "could not detect workspace root; `cargo locate-project` reported no manifest"
                .to_string(),
        );
    }
    let manifest_path = PathBuf::from(&manifest);
    let root = manifest_path.parent().ok_or_else(|| {
        format!("could not detect workspace root; `{manifest}` has no parent directory")
    })?;
    std::fs::canonicalize(root).map_err(|err| {
        format!(
            "could not detect workspace root; failed to canonicalize `{}`: {err}",
            root.display()
        )
    })
}

/// Path globs (relative to the lint root) that are exempt from specific
/// lint rules, with a mandatory cross-reference to the spec section
/// that explains why. New entries must cite a `§` ref or update the
/// spec to add one — opaque exemptions are not permitted.
///
/// The `Std.Test` carve-out for `surf-test-name-prefix` is implemented
/// inside the rule itself (it short-circuits when the source declares
/// `module Std.Test`) and does not need a path-glob entry here.
///
/// `crates/chelisup/bootstrap/chelisup.sh`: the single shell carve-out
/// to the `no-shell-scripts` rule (§2.9). The chelisup bootstrap
/// one-liner runs on a bare machine before any chelis, cargo, or Python
/// exists, so it cannot be any of those; it is minimal POSIX `sh`,
/// shellcheck-clean, and test-covered. Every other script remains
/// Python.
pub fn exceptions() -> Vec<Exception> {
    vec![Exception {
        pattern: "crates/chelisup/bootstrap/chelisup.sh".to_string(),
        rule_id: "no-shell-scripts".to_string(),
        cross_ref: "§2.9".to_string(),
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

pub fn disabled_by_env() -> bool {
    env_disables_gate()
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
    enforce_style_gate_structured(file, source, allow_violations)
        .map_err(|rejection| rejection.report.into())
}

/// A style-gate rejection, in both the renderings a caller may need.
///
/// One gate run produces both. That is the point: `chelis check` needs the
/// human report for stderr AND one machine-facing message per issue for the
/// check report, and running the gate twice to get them made the two
/// disagree. `run_lint_for_single_file` re-reads the file, so a rewrite
/// landing between two runs could leave the caller having decided to fail
/// from run A while building its diagnostics from run B -- shipping exit 2
/// with an empty `errors` array, the one state `spec/04` § Gating says
/// cannot happen (chelis#886).
pub struct StyleRejection {
    /// The joined human report, byte-identical to what the gate has always
    /// written to stderr.
    pub report: String,
    /// One message per issue, for a machine-facing carrier. Non-empty by
    /// construction: this type is only built from a non-clean outcome, and a
    /// non-clean outcome has at least one issue.
    pub issues: Vec<String>,
}

/// [`enforce_style_gate`], handing back the rejection's structure rather
/// than only its rendered text.
pub fn enforce_style_gate_structured(
    file: &Path,
    source: &str,
    allow_violations: bool,
) -> Result<(), StyleRejection> {
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
    let issues = issue_messages(file, &outcome);
    debug_assert_eq!(
        issues.len(),
        outcome.issue_count(),
        "one message per issue, or the report's error count lies"
    );
    Err(StyleRejection { report, issues })
}

/// One single-line message per issue, from the same outcome
/// [`format_outcome`] renders for the terminal.
///
/// Deliberately not `format_outcome`'s output split on newlines: that string
/// ends with a summary line that is not an issue, and a lint violation's own
/// `Display` may not be one line forever. Building from the structured
/// outcome keeps the count honest.
fn issue_messages(file: &Path, outcome: &GateOutcome) -> Vec<String> {
    let mut issues = Vec::with_capacity(outcome.issue_count());
    if let Some(diff) = &outcome.fmt_diff {
        issues.push(format!(
            "{}: not canonically formatted; run `chelis fmt --inplace {}` to fix",
            diff.path.display(),
            file.display()
        ));
    }
    for violation in &outcome.lint_violations {
        issues.push(violation.to_string());
    }
    issues
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
        let format_source = strip_deep_lint_directive_lines(source);
        // Canonical Deep always uses LF and ends non-empty programs with a
        // newline. Check these byte-level invariants before parsing: malformed
        // or role-invalid Deep may be rejected by the stamped parser, but that
        // must not make CRLF or a missing final newline silently pass the
        // formatting gate.
        if format_source.contains('\r')
            || (!format_source.is_empty() && !format_source.ends_with('\n'))
        {
            return Some(FmtDiff {
                path: file.to_path_buf(),
            });
        }
        match chelis_deep::parser::parse_and_stamp_file(&format_source) {
            Ok(exprs) => chelis_deep::printer::print_canonical(&exprs),
            // The stamped ingress is the only Deep ingress (chelis#1088):
            // there is no weaker second parse to fall back to. A `.dp` it
            // rejects is malformed or role-invalid, and the regular compile
            // path surfaces that with a better message than a formatting
            // diff would; we don't double-report here. The byte-level LF and
            // final-newline invariants above already ran, so the line-ending
            // cases the old fallback covered are still reported.
            Err(_) => return None,
        }
    } else {
        match chelis_surf::format::format_source(source) {
            Ok(canonical) => canonical,
            // If the file doesn't lex/parse, the regular compile path
            // surfaces that error with a better message.
            Err(_) => return None,
        }
    };
    let compare_source = if ext == "dp" {
        strip_deep_lint_directive_lines(source)
    } else {
        source.to_string()
    };
    if canonical == compare_source {
        None
    } else {
        Some(FmtDiff {
            path: file.to_path_buf(),
        })
    }
}

pub fn strip_deep_lint_directive_lines(source: &str) -> String {
    source
        .split_inclusive('\n')
        .filter(|line| {
            let trimmed = line.trim_start();
            !(trimmed.starts_with(';') && trimmed.contains("chelis-lint:"))
        })
        .collect()
}

fn run_lint_for_single_file(file: &Path) -> Vec<Violation> {
    let rules = chelis_lint::registry::all_rules();
    // Lint the file directly. The canonical walker admits explicit file
    // roots, which avoids path-shape mismatches between `message.ch` and
    // `./message.ch`.
    let raw = match chelis_lint::lint(file, &rules) {
        Ok(v) => v,
        Err(error) => {
            return vec![Violation {
                rule_id: "lint-traversal-policy".to_string(),
                spec_ref: "§12.2".to_string(),
                path: file.to_path_buf(),
                line: None,
                col: None,
                message: error.to_string(),
            }];
        }
    };
    let exceptions_list = exceptions();
    // Anchor exception matching against the detected Cargo workspace
    // root, mirroring the standalone `chelis lint` subcommand. Closes
    // `Lint-ExceptionPathRoot-F1` for the build-time style gate as
    // well, and `Lint-WorkspaceRootCwdAssumption-F1`: single-file
    // builds (`chelis check foo.ch`, `chelis build foo.ch`) now anchor
    // exceptions against the real workspace root regardless of which
    // directory the CLI was invoked from.
    //
    // If detection fails the file is not inside a Cargo workspace, so
    // no workspace-rooted exception glob can legitimately apply. The
    // raw violations pass through unfiltered: that is the correct
    // behavior (a file outside the workspace is not, e.g.,
    // `crates/chelisup/bootstrap/chelisup.sh`), not a second
    // detection mechanism with different semantics.
    let probe_dir = file
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    match detect_lint_workspace_root(probe_dir) {
        Ok(workspace_root) => {
            chelis_lint::exceptions::apply_exceptions(&raw, &exceptions_list, &workspace_root)
        }
        Err(_) => raw,
    }
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
    fn bootstrap_sh_is_the_only_no_shell_scripts_exception() {
        // The chelisup bootstrap installer is the single shell carve-out
        // (§2.9). Its exception must be registered, cite §2.9, and
        // actually filter a `no-shell-scripts` violation at its path,
        // while any other `.sh` path still fires.
        let excs = exceptions();
        let entry = excs
            .iter()
            .find(|e| e.pattern == "crates/chelisup/bootstrap/chelisup.sh")
            .expect("bootstrap exception must be registered");
        assert_eq!(entry.rule_id, "no-shell-scripts");
        assert_eq!(entry.cross_ref, "§2.9");

        let root = std::path::Path::new("/repo");
        let blessed = Violation {
            rule_id: "no-shell-scripts".to_string(),
            spec_ref: "§2.9".to_string(),
            path: root.join("crates/chelisup/bootstrap/chelisup.sh"),
            line: None,
            col: None,
            message: "shell".to_string(),
        };
        let kept = chelis_lint::exceptions::apply_exceptions([&blessed], &excs, root);
        assert!(
            kept.is_empty(),
            "bootstrap .sh must be excepted; kept={kept:?}"
        );

        let other = Violation {
            path: root.join("scripts/whatever.sh"),
            ..blessed.clone()
        };
        let kept_other = chelis_lint::exceptions::apply_exceptions([&other], &excs, root);
        assert_eq!(
            kept_other.len(),
            1,
            "only the one carve-out path is excepted"
        );
    }

    #[test]
    fn all_exception_cross_refs_resolve_against_the_spec() {
        // Every exception's `cross_ref` must point at a real section of
        // the nomenclature spec (closes the opaque-exemption loophole).
        let spec = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../spec/01-nomenclature.md"
        ));
        let res = chelis_lint::exceptions::verify_cross_refs(&exceptions(), spec);
        assert!(res.is_ok(), "unresolvable cross_refs: {:?}", res.err());
    }

    #[cfg(unix)]
    #[test]
    fn inadmissible_lint_root_fails_closed() {
        use std::os::unix::net::UnixListener;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("socket.ch");
        let _socket = UnixListener::bind(&path).unwrap();
        let canonical = chelis_surf::format::format_program(&[]);

        let outcome = run_gate(&path, &canonical);
        assert_eq!(outcome.lint_violations.len(), 1);
        assert_eq!(outcome.lint_violations[0].rule_id, "lint-traversal-policy");
        assert!(
            outcome.lint_violations[0]
                .message
                .contains("not a regular file or directory"),
            "the gate must surface the depth-zero rejection reason: {}",
            outcome.lint_violations[0].message
        );
    }

    #[test]
    fn malformed_traversal_policy_fails_closed() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("spec")).unwrap();
        std::fs::write(
            dir.path().join("spec/lint.md"),
            "# Lint\n\n### 12.2 Traversal exclusions\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("chelis-lint.toml"),
            "version = 2\nspec = \"spec/lint.md\"\n",
        )
        .unwrap();
        let path = dir.path().join("clean.ch");
        let canonical = chelis_surf::format::format_program(&[]);
        std::fs::write(&path, &canonical).unwrap();

        let outcome = run_gate(&path, &canonical);
        assert_eq!(outcome.lint_violations.len(), 1);
        assert_eq!(outcome.lint_violations[0].rule_id, "lint-traversal-policy");
        assert!(outcome.lint_violations[0].message.contains("unsupported"));
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
