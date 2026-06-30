//! Version-string validation.
//!
//! Two distinct checks, deliberately kept separate:
//!
//! - [`validate_install_version`] is strict `X.Y.Z` (digits only),
//!   because `install` builds a release-asset name from it
//!   (`chelis-vX.Y.Z-<slug>.tar.gz`) and a non-version there is a user
//!   typo we should reject loudly, not paper over.
//! - [`is_safe_path_component`] is the security gate applied to every
//!   version that gets joined into a filesystem path during resolution.
//!   A resolved version can come from an attacker-influenceable source
//!   (a `reef.toml` pin, a `chelis-toolchain` file, the `CHELIS_TOOLCHAIN`
//!   env), so it must never be allowed to contain a path separator or a
//!   `..` traversal before it is joined under `~/.chelis/toolchains/`.

/// True iff `v` is exactly `X.Y.Z` where each component is one or more
/// ASCII digits. Used by `install` and `default`.
pub fn validate_install_version(v: &str) -> Result<(), String> {
    let parts: Vec<&str> = v.split('.').collect();
    let ok = parts.len() == 3
        && parts
            .iter()
            .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()));
    if ok {
        Ok(())
    } else {
        Err(format!(
            "malformed version {v:?}; expected X.Y.Z (for example 0.12.0)"
        ))
    }
}

/// True iff `v` is safe to use as a single path component under the
/// store root: non-empty, no path separators, no NUL, and not a
/// `.`/`..` traversal. Intentionally broader than [`validate_install_version`]
/// so a future channel-style name (for example a date-stamped nightly)
/// could resolve, while still refusing anything that escapes the store.
pub fn is_safe_path_component(v: &str) -> bool {
    if v.is_empty() || v == "." || v == ".." {
        return false;
    }
    !v.chars()
        .any(|c| c == '/' || c == '\\' || c == '\0' || c.is_control())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_version_accepts_xyz() {
        assert!(validate_install_version("0.12.0").is_ok());
        assert!(validate_install_version("10.0.255").is_ok());
    }

    #[test]
    fn install_version_rejects_non_xyz() {
        for bad in ["0.13", "1.2.3.4", "v0.1.0", "0.1.0-rc1", "", "abc", "0..0"] {
            assert!(
                validate_install_version(bad).is_err(),
                "should reject {bad:?}"
            );
        }
    }

    #[test]
    fn path_component_rejects_traversal_and_separators() {
        for bad in ["", ".", "..", "a/b", "a\\b", "../etc", "x\0y", "a\nb"] {
            assert!(!is_safe_path_component(bad), "should reject {bad:?}");
        }
    }

    #[test]
    fn path_component_accepts_versions_and_channels() {
        for good in ["0.12.0", "0.13.0-rc1", "nightly-2026-06-30", "0.13"] {
            assert!(is_safe_path_component(good), "should accept {good:?}");
        }
    }
}
