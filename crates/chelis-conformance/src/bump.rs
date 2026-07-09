//! Version-propagation arm: the mechanical half of `conform bump`.
//!
//! A pin bump is a *de-narrowing event*, not a version edit (contract §7). This
//! module does the deterministic, offline part — rewrite every pin location in
//! lockstep — which `conform bump` then follows with `sync` (restamp the pointer
//! blocks + re-materialize skills) and the executable gates (blocked probes,
//! negatives) run by the CLI. The whole point of the arm is that a bump that
//! skips the restamp leaves the managed-block stamp behind the pin, so
//! `conform audit` (and therefore `conform bump-check`) fails — the exact
//! failure mode that raw direct-to-`main` cascades used to slip past.

use std::fs;
use std::path::{Path, PathBuf};

use crate::audit::{is_installable_version, parse_compiler_pin};

/// Rewrite every pin location under `root` to `new_version`, in lockstep:
/// `reef.toml`'s `compiler = "=X.Y.Z"` and every workflow's
/// `CHELIS_TAG`/`CHELIS_VERSION` env plus `chelisup install` line. Returns the
/// files changed (empty if already at `new_version`).
pub fn rewrite_pins(root: &Path, new_version: &str) -> Result<Vec<PathBuf>, String> {
    // Validate up front so a bump can only ever write a pin the chelisup shim
    // can actually install — otherwise `conform bump garbage` would leave a
    // `compiler = "=garbage"` that audits green but the toolchain cannot resolve.
    if !is_installable_version(new_version) {
        return Err(format!(
            "{new_version:?} is not an installable X.Y.Z version (no `v`, no pre-release)"
        ));
    }

    let mut changed = Vec::new();

    let reef_path = root.join("reef.toml");
    let reef =
        fs::read_to_string(&reef_path).map_err(|e| format!("read {}: {e}", reef_path.display()))?;
    let old_pin = parse_compiler_pin(&reef)
        .ok_or_else(|| "reef.toml has no `compiler = \"=X.Y.Z\"` pin to bump".to_string())?;
    let old = old_pin.trim_start_matches('=').to_string();

    if old != new_version {
        // Rewrite only the `compiler` line under the root/`[package]` table —
        // never a coincidental `"=X.Y.Z"` elsewhere (e.g. a lockstep-versioned
        // `chelis-std = "=X.Y.Z"` dependency pinned to the same version).
        let new_reef = rewrite_compiler_pin(&reef, &old, new_version);
        if new_reef != reef {
            fs::write(&reef_path, new_reef)
                .map_err(|e| format!("write {}: {e}", reef_path.display()))?;
            changed.push(reef_path);
        }
    }

    let wf_dir = root.join(".github/workflows");
    if let Ok(entries) = fs::read_dir(&wf_dir) {
        for e in entries.flatten() {
            let p = e.path();
            let is_yaml = p
                .extension()
                .and_then(|s| s.to_str())
                .is_some_and(|e| e == "yml" || e == "yaml");
            if !is_yaml {
                continue;
            }
            let Ok(body) = fs::read_to_string(&p) else {
                continue;
            };
            let updated = rewrite_workflow_pins(&body, &old, new_version);
            if updated != body {
                fs::write(&p, updated).map_err(|e| format!("write {}: {e}", p.display()))?;
                changed.push(p);
            }
        }
    }

    Ok(changed)
}

/// Rewrite the pin tokens in one workflow body.
fn rewrite_workflow_pins(body: &str, old: &str, new: &str) -> String {
    body.replace(
        &format!("CHELIS_VERSION: {old}"),
        &format!("CHELIS_VERSION: {new}"),
    )
    .replace(
        &format!("CHELIS_VERSION: \"{old}\""),
        &format!("CHELIS_VERSION: \"{new}\""),
    )
    .replace(
        &format!("CHELIS_TAG: v{old}"),
        &format!("CHELIS_TAG: v{new}"),
    )
    .replace(
        &format!("CHELIS_TAG: \"v{old}\""),
        &format!("CHELIS_TAG: \"v{new}\""),
    )
    .replace(
        &format!("chelisup install {old}"),
        &format!("chelisup install {new}"),
    )
}

/// Rewrite the `compiler = "=old"` value to `"=new"` on the `compiler` line
/// under the root or `[package]` table only, leaving every other line — and any
/// coincidental `"=old"` token elsewhere — untouched.
fn rewrite_compiler_pin(reef: &str, old: &str, new: &str) -> String {
    let old_tok = format!("\"={old}\"");
    let new_tok = format!("\"={new}\"");
    let mut in_scope = true;
    let mut out = String::with_capacity(reef.len());
    for line in reef.split_inclusive('\n') {
        let body = line.strip_suffix('\n').unwrap_or(line);
        let trimmed = body.trim();
        if let Some(header) = trimmed.strip_prefix('[').and_then(|h| h.strip_suffix(']')) {
            in_scope = header.trim() == "package";
            out.push_str(line);
            continue;
        }
        let is_compiler_key = trimmed
            .strip_prefix("compiler")
            .is_some_and(|r| r.trim_start().starts_with('='));
        if in_scope && is_compiler_key {
            out.push_str(&line.replace(&old_tok, &new_tok));
        } else {
            out.push_str(line);
        }
    }
    out
}

/// The bare pin (`X.Y.Z`, no leading `=`) recorded in a `reef.toml` text.
pub fn bare_pin(reef_toml: &str) -> Option<String> {
    parse_compiler_pin(reef_toml).map(|p| p.trim_start_matches('=').to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workflow_pin_rewrite() {
        let body = "env:\n  CHELIS_TAG: v0.14.0\n  CHELIS_VERSION: 0.14.0\n  run: chelisup install 0.14.0\n";
        let out = rewrite_workflow_pins(body, "0.14.0", "0.15.0");
        assert!(out.contains("CHELIS_TAG: v0.15.0"));
        assert!(out.contains("CHELIS_VERSION: 0.15.0"));
        assert!(out.contains("chelisup install 0.15.0"));
        assert!(!out.contains("0.14.0"));
    }

    #[test]
    fn bare_pin_extraction() {
        assert_eq!(
            bare_pin("compiler = \"=0.14.0\"\n").as_deref(),
            Some("0.14.0")
        );
        assert_eq!(bare_pin("no pin here"), None);
    }

    #[test]
    fn compiler_pin_rewrite_leaves_lockstep_dependency_alone() {
        let reef =
            "[package]\ncompiler = \"=0.14.0\"\n\n[dependencies]\nchelis-std = \"=0.14.0\"\n";
        let out = rewrite_compiler_pin(reef, "0.14.0", "0.15.0");
        assert!(out.contains("compiler = \"=0.15.0\""));
        assert!(
            out.contains("chelis-std = \"=0.14.0\""),
            "a same-version dependency pin must not be rewritten"
        );
    }

    #[test]
    fn rewrite_pins_rejects_non_installable_version() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("reef.toml"), "compiler = \"=0.14.0\"\n").unwrap();
        for bad in ["garbage", "=0.15.0", "v0.15.0", "0.15", "0.15.0-rc1"] {
            assert!(
                rewrite_pins(dir.path(), bad).is_err(),
                "bump to {bad:?} must be rejected"
            );
        }
        // reef.toml is untouched by a rejected bump.
        assert_eq!(
            std::fs::read_to_string(dir.path().join("reef.toml")).unwrap(),
            "compiler = \"=0.14.0\"\n"
        );
    }
}
