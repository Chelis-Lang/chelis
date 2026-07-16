//! Shared, zero-dependency version + path-component primitives for the chelis
//! toolchain.
//!
//! This is the single source of truth for "what is a valid pin/version" so the
//! installer ([`chelisup`]) and the downstream-conformance auditor
//! ([`chelis-conformance`]) cannot silently diverge. Before this crate the
//! strict-`X.Y.Z` rule was mirrored independently in both — and a divergence
//! would let the auditor bless a pin the installer rejects (or vice versa),
//! producing a shell that audits green but whose toolchain cannot resolve.
//!
//! [`chelisup`]: https://docs.rs/chelisup
//! [`chelis-conformance`]: https://docs.rs/chelis-conformance

/// True iff `v` is a strict `X.Y.Z` release version: exactly three non-empty
/// components, each one or more ASCII digits. No `v` prefix, no pre-release
/// suffix, no fourth component.
///
/// This is the format the installer builds a release-asset name from
/// (`chelis-vX.Y.Z-<slug>.tar.gz`), so anything else is a user typo to reject
/// loudly rather than paper over. It is also what conformance requires of a
/// reef pin, so that a pinned toolchain the auditor accepts is one the
/// installer can actually fetch.
#[must_use]
pub fn is_strict_semver(v: &str) -> bool {
    let mut parts = v.split('.');
    matches!(
        (parts.next(), parts.next(), parts.next(), parts.next()),
        (Some(a), Some(b), Some(c), None)
            if [a, b, c]
                .iter()
                .all(|p| !p.is_empty() && p.bytes().all(|x| x.is_ascii_digit()))
    )
}

/// True iff `v` is safe to use as a single path component under the toolchain
/// store root: non-empty, not `.`/`..`, and free of path separators, NUL, or
/// control characters.
///
/// Deliberately broader than [`is_strict_semver`] so a future channel-style
/// name (for example a date-stamped nightly) could still resolve, while
/// refusing anything that escapes the store. A resolved version can come from
/// an attacker-influenceable source (a `reef.toml` pin, a `chelis-toolchain`
/// file, `$CHELIS_TOOLCHAIN`), so it must never contain a separator or a `..`
/// traversal before it is joined under `~/.chelis/toolchains/`.
#[must_use]
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
    fn strict_semver_accepts_xyz() {
        for good in ["0.12.0", "10.0.255", "0.14.0"] {
            assert!(is_strict_semver(good), "should accept {good:?}");
        }
    }

    #[test]
    fn strict_semver_rejects_non_xyz() {
        for bad in ["0.13", "1.2.3.4", "v0.1.0", "0.1.0-rc1", "", "abc", "0..0"] {
            assert!(!is_strict_semver(bad), "should reject {bad:?}");
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
