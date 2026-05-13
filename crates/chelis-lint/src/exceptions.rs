//! Exception-list infrastructure for `chelis lint` per §12.
//!
//! Every exception entry carries a mandatory `cross_ref` field pointing at a
//! section of `chelis/spec/01-nomenclature.md` that explains why the case is
//! exempt. Free-form prose is deliberately not part of the schema. Entries
//! whose `cross_ref` doesn't resolve to a real spec section are rejected at
//! load time as build-time errors, closing the loophole that produced the
//! original `SKILL.md` carve-out.
//!
//! Matching: an exception applies to a violation when (a) the path-relative-
//! to-`workspace_root` matches the exception's glob pattern and (b) the rule
//! id matches.
//!
//! Path anchoring: exception patterns in
//! `crates/chelis-cli/src/style_gate.rs::exceptions` are written relative to
//! the workspace root (e.g., `crates/chelis-surf/tests/fixtures/*.ch`,
//! `docs/book/src/*.md`). They must therefore be matched against
//! workspace-relative paths, not against paths relative to the per-target
//! lint walk root. When the CLI walks a sub-directory (e.g., `chelis lint
//! --check crates`), the walk root is a sub-directory of the workspace
//! root; using it for prefix-stripping drops the leading workspace-relative
//! segments and the exception silently fails to match. Callers therefore
//! pass an explicit `workspace_root` argument distinct from the walk root.
//! Closes `Lint-ExceptionPathRoot-F1`.

use crate::{Exception, Violation};
use regex::Regex;
use std::collections::HashSet;
use std::sync::OnceLock;

/// Validate an exception list against the recorded style guide.
///
/// Reads `spec_source`, extracts the set of section identifiers it defines
/// (e.g. `§1.4`, `§8.5`, `§11.1`), and returns `Ok` if every exception's
/// `cross_ref` matches one of those identifiers. Otherwise returns an
/// `Err` with one message per unresolvable cross-ref.
///
/// The match is exact: an exception with `cross_ref = "§8.5"` is accepted
/// only if the spec has a `### 8.5 Foo` heading. `§8` is treated as a
/// distinct identifier from `§8.5`, so a top-level reference doesn't
/// silently match a sub-section.
pub fn verify_cross_refs(exceptions: &[Exception], spec_source: &str) -> Result<(), Vec<String>> {
    let known = section_ids_from_spec(spec_source);
    let mut errors = Vec::new();
    for exc in exceptions {
        if !known.contains(&exc.cross_ref) {
            errors.push(format!(
                "exception for rule `{}` (pattern `{}`) has cross_ref `{}` that does not resolve to a section in the recorded style guide; either fix the cross_ref or document the rule in spec/01-nomenclature.md",
                exc.rule_id, exc.pattern, exc.cross_ref
            ));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Extract every `§<number>` section identifier from a markdown spec file
/// by scanning for `### N.M Title` and `### N.M.K Title` headings.
fn section_ids_from_spec(spec_source: &str) -> HashSet<String> {
    let heading = heading_re();
    let mut out = HashSet::new();
    for caps in heading.captures_iter(spec_source) {
        let id = caps.get(1).unwrap().as_str();
        out.insert(format!("§{id}"));
        // Also accept the parent identifier — a `§8.5.1` heading implies
        // the existence of a top-level `§8` section even if that exact
        // numeric isn't its own heading line.
        for parent in section_parents(id) {
            out.insert(format!("§{parent}"));
        }
    }
    out
}

static HEADING_RE: OnceLock<Regex> = OnceLock::new();
fn heading_re() -> &'static Regex {
    // `## 1.4 Title` or `### 8.5 Title` or `#### 8.5.1 Title`.
    HEADING_RE.get_or_init(|| Regex::new(r"(?m)^#+\s+([0-9]+(?:\.[0-9]+)*)\s").unwrap())
}

fn section_parents(id: &str) -> Vec<String> {
    let parts: Vec<&str> = id.split('.').collect();
    (1..parts.len()).map(|n| parts[..n].join(".")).collect()
}

/// Filter `violations` against the exception list. A violation is dropped
/// if any exception's `rule_id` and `pattern` (glob) both match.
///
/// `workspace_root` is the path against which exception globs are anchored
/// (typically the canonical path of the workspace root). It is distinct
/// from the per-target lint walk root because exception patterns are
/// written workspace-relative; using the walk root for prefix-stripping
/// silently breaks workspace-rooted patterns under sub-directory walks
/// (see `Lint-ExceptionPathRoot-F1`).
pub fn apply_exceptions<'a>(
    violations: impl IntoIterator<Item = &'a Violation>,
    exceptions: &[Exception],
    workspace_root: &std::path::Path,
) -> Vec<Violation> {
    violations
        .into_iter()
        .filter(|v| !is_excepted(v, exceptions, workspace_root))
        .cloned()
        .collect()
}

fn is_excepted(
    violation: &Violation,
    exceptions: &[Exception],
    workspace_root: &std::path::Path,
) -> bool {
    let rel = violation
        .path
        .strip_prefix(workspace_root)
        .unwrap_or(&violation.path)
        .to_string_lossy()
        .to_string();
    for exc in exceptions {
        if exc.rule_id != violation.rule_id {
            continue;
        }
        if glob_match(&exc.pattern, &rel) {
            return true;
        }
    }
    false
}

/// Minimal glob: `*` matches any single path segment without `/`; `**`
/// matches any sequence of segments. Anchored from the start of the
/// path-relative-to-root.
fn glob_match(pattern: &str, path: &str) -> bool {
    let pattern_parts: Vec<&str> = pattern.split('/').collect();
    let path_parts: Vec<&str> = path.split('/').collect();
    glob_match_parts(&pattern_parts, &path_parts)
}

fn glob_match_parts(pattern: &[&str], path: &[&str]) -> bool {
    match (pattern.first(), path.first()) {
        (None, None) => true,
        (None, Some(_)) => false,
        (Some(&"**"), _) => {
            // `**` matches any number of segments.
            for n in 0..=path.len() {
                if glob_match_parts(&pattern[1..], &path[n..]) {
                    return true;
                }
            }
            false
        }
        (Some(&p), Some(&pp)) => {
            if !match_segment(p, pp) {
                return false;
            }
            glob_match_parts(&pattern[1..], &path[1..])
        }
        (Some(_), None) => false,
    }
}

fn match_segment(pattern: &str, segment: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if !pattern.contains('*') {
        return pattern == segment;
    }
    // Single `*` in the middle: split and check prefix/suffix.
    let mut parts = pattern.splitn(2, '*');
    let prefix = parts.next().unwrap_or("");
    let suffix = parts.next().unwrap_or("");
    segment.starts_with(prefix)
        && segment.ends_with(suffix)
        && segment.len() >= prefix.len() + suffix.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn extracts_section_ids_from_headings() {
        let spec = "# Title\n\n## 1. Foo\n\n### 1.4 Bar\n\n#### 8.5 Baz\n#### 8.5.1 Quux\n";
        let ids = section_ids_from_spec(spec);
        assert!(ids.contains("§1"));
        assert!(ids.contains("§1.4"));
        assert!(ids.contains("§8"));
        assert!(ids.contains("§8.5"));
        assert!(ids.contains("§8.5.1"));
    }

    #[test]
    fn verify_accepts_resolvable_cross_ref() {
        let spec = "### 8.5 Foo\n";
        let exc = Exception {
            pattern: "docs/book/src/*.md".into(),
            rule_id: "doc-filename-convention".into(),
            cross_ref: "§8.5".into(),
        };
        assert!(verify_cross_refs(&[exc], spec).is_ok());
    }

    #[test]
    fn verify_rejects_unresolvable_cross_ref() {
        // §99 doesn't exist in the spec — must fail.
        let spec = "### 8.5 Foo\n";
        let exc = Exception {
            pattern: "anything".into(),
            rule_id: "no-shell-scripts".into(),
            cross_ref: "§99".into(),
        };
        let res = verify_cross_refs(&[exc], spec);
        assert!(res.is_err());
        let errors = res.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("§99"));
        assert!(errors[0].contains("does not resolve"));
    }

    #[test]
    fn verify_reports_every_unresolvable() {
        let spec = "### 8.5 Foo\n";
        let exc1 = Exception {
            pattern: "a".into(),
            rule_id: "x".into(),
            cross_ref: "§99".into(),
        };
        let exc2 = Exception {
            pattern: "b".into(),
            rule_id: "y".into(),
            cross_ref: "§42".into(),
        };
        let res = verify_cross_refs(&[exc1, exc2], spec);
        assert_eq!(res.unwrap_err().len(), 2);
    }

    #[test]
    fn glob_match_basic() {
        assert!(glob_match("docs/foo.md", "docs/foo.md"));
        assert!(!glob_match("docs/foo.md", "docs/bar.md"));
    }

    #[test]
    fn glob_match_star_segment() {
        assert!(glob_match("docs/*.md", "docs/foo.md"));
        assert!(glob_match("docs/*.md", "docs/bar.md"));
        assert!(!glob_match("docs/*.md", "docs/sub/foo.md"));
    }

    #[test]
    fn glob_match_double_star() {
        assert!(glob_match("docs/**/*.md", "docs/foo.md"));
        assert!(glob_match("docs/**/*.md", "docs/sub/foo.md"));
        assert!(glob_match("docs/**/*.md", "docs/a/b/c.md"));
        assert!(!glob_match("docs/**/*.md", "src/foo.md"));
    }

    #[test]
    fn apply_exceptions_drops_matched() {
        let v = Violation {
            rule_id: "no-shell-scripts".into(),
            spec_ref: "§2.9".into(),
            path: PathBuf::from("/repo/scripts/install.sh"),
            line: None,
            col: None,
            message: "shell".into(),
        };
        let exc = Exception {
            pattern: "scripts/install.sh".into(),
            rule_id: "no-shell-scripts".into(),
            cross_ref: "§2.9".into(),
        };
        let kept = apply_exceptions([&v], &[exc], std::path::Path::new("/repo"));
        assert!(kept.is_empty());
    }

    #[test]
    fn apply_exceptions_keeps_unmatched() {
        let v = Violation {
            rule_id: "no-shell-scripts".into(),
            spec_ref: "§2.9".into(),
            path: PathBuf::from("/repo/scripts/install.sh"),
            line: None,
            col: None,
            message: "shell".into(),
        };
        let exc = Exception {
            pattern: "scripts/install.sh".into(),
            rule_id: "different-rule".into(),
            cross_ref: "§2.9".into(),
        };
        let kept = apply_exceptions([&v], &[exc], std::path::Path::new("/repo"));
        assert_eq!(kept.len(), 1);
    }

    /// Anchor invariant: exception matching must be relative to the
    /// workspace root, not the per-target walk root. A workspace-rooted
    /// exception pattern like `crates/x/fixtures/*.ch` must match a
    /// violation at `<workspace_root>/crates/x/fixtures/foo.ch`
    /// regardless of which sub-tree the CLI is walking. Pins
    /// `Lint-ExceptionPathRoot-F1`.
    #[test]
    fn apply_exceptions_anchors_on_workspace_root_not_walk_target() {
        let v = Violation {
            rule_id: "surf-def-arrow-form".into(),
            spec_ref: "§3.5".into(),
            path: PathBuf::from("/repo/crates/chelis-surf/tests/fixtures/foo.ch"),
            line: None,
            col: None,
            message: "colon-form".into(),
        };
        let exc = Exception {
            pattern: "crates/chelis-surf/tests/fixtures/*.ch".into(),
            rule_id: "surf-def-arrow-form".into(),
            cross_ref: "§3.5".into(),
        };
        // The CLI walks `<repo>/crates` as the target. The exception
        // must still match because `workspace_root` is `/repo`, not
        // `/repo/crates`.
        let kept = apply_exceptions([&v], &[exc], std::path::Path::new("/repo"));
        assert!(
            kept.is_empty(),
            "workspace-rooted exception must match under subtree walks; kept={kept:?}"
        );
    }

    #[test]
    fn apply_exceptions_glob_pattern() {
        let v = Violation {
            rule_id: "doc-filename-convention".into(),
            spec_ref: "§8.5".into(),
            path: PathBuf::from("/repo/docs/book/src/SUMMARY.md"),
            line: None,
            col: None,
            message: "doc".into(),
        };
        // mdBook tool-required exception.
        let exc = Exception {
            pattern: "docs/book/src/SUMMARY.md".into(),
            rule_id: "doc-filename-convention".into(),
            cross_ref: "§8.5".into(),
        };
        let kept = apply_exceptions([&v], &[exc], std::path::Path::new("/repo"));
        assert!(kept.is_empty());
    }
}
