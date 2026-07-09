//! Version-string validation.
//!
//! The two underlying rules — strict `X.Y.Z` and safe-path-component — live in
//! the shared [`chelis_version`] leaf crate so the installer and the
//! `chelis-conformance` auditor cannot diverge on what a valid pin is. This
//! module keeps the installer-facing wrapper (a `Result` with an actionable
//! error) and re-exports the security gate:
//!
//! - [`validate_install_version`] is strict `X.Y.Z` (digits only), because
//!   `install` builds a release-asset name from it (`chelis-vX.Y.Z-<slug>.tar.gz`)
//!   and a non-version there is a user typo we should reject loudly.
//! - [`is_safe_path_component`] is the security gate applied to every version
//!   joined into a filesystem path during resolution. A resolved version can
//!   come from an attacker-influenceable source (a `reef.toml` pin, a
//!   `chelis-toolchain` file, the `CHELIS_TOOLCHAIN` env), so it must never
//!   contain a path separator or a `..` traversal before it is joined under
//!   `~/.chelis/toolchains/`.

pub use chelis_version::is_safe_path_component;

/// True iff `v` is exactly `X.Y.Z` where each component is one or more
/// ASCII digits. Used by `install` and `default`.
pub fn validate_install_version(v: &str) -> Result<(), String> {
    if chelis_version::is_strict_semver(v) {
        Ok(())
    } else {
        Err(format!(
            "malformed version {v:?}; expected X.Y.Z (for example 0.12.0)"
        ))
    }
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
