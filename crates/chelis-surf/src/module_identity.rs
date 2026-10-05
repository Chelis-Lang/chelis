//! §6.5 module identity: the correspondence between a `.ch` file's path
//! inside a reef package and the module it is required to declare.
//!
//! This rule has two consumers that must never disagree: the reef package
//! loader, which rejects a mismatch at load time and makes the package
//! uncompilable, and the `reef-module-identity` lint rule, which reports the
//! same mismatch from the style surface. A divergence between them is the
//! defect chelis#2116 filed — a green style gate on a package that cannot
//! load — so the rule lives in one place and both call it.
//!
//! It sits here, rather than in `chelis-reef`, because `chelis-lint` is
//! deliberately dependency-pure, and pulling the package manager into the
//! lint crate to reach 30 lines of string arithmetic would undo that.
//! `chelis-surf` already owns module declarations, and both `chelis-reef` and
//! `chelis-lint` already depend on it, so the shared home costs no new edge in
//! either direction.

use std::path::Path;

/// Validate a declared module name against the package's `module_prefix` and
/// the file's path beneath its source root (§6.5).
///
/// `rel` is the file path relative to the source root, already carrying the
/// root's own name as its first component when the root is not `src`, so
/// `properties/laws.ch` in a `Demo` package derives `demo.properties.laws`.
///
/// The comparison ignores case and nothing else: a path segment contributes
/// itself, underscores included, so `src/cross_entropy.ch` derives
/// `demo.cross_entropy`.
pub fn validate_module_path(prefix: &str, module: &str, rel: &Path) -> Result<(), String> {
    let prefix_lower = prefix.to_lowercase();
    let module_lower = module.to_lowercase();
    if module_lower == prefix_lower || !module_lower.starts_with(&(prefix_lower.clone() + ".")) {
        return Err(format!(
            "module `{module}` does not belong to module_prefix `{prefix}`"
        ));
    }
    let rel_no_ext = rel.with_extension("");
    let rel_module = rel_no_ext
        .iter()
        .map(|seg| seg.to_string_lossy().to_string())
        .collect::<Vec<_>>()
        .join(".");
    let expected = format!("{prefix_lower}.{rel_module}");
    if module_lower != expected {
        // `rel` arrives already prefixed with the source-root name for a
        // non-`src` root (see `chelis_reef::load_package_modules`), so the
        // displayed path is the package-root-relative form ("src/foo.ch" or
        // "properties/foo.ch") and the expected module name reflects the
        // same rule.
        return Err(format!(
            "module `{module}` does not match file path {} (expected `{}`)",
            rel.display(),
            expected
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- positive: the shapes §6.5's table says must load ---

    #[test]
    fn accepts_single_segment_under_src() {
        assert!(validate_module_path("Demo", "Demo.Linalg", Path::new("linalg.ch")).is_ok());
    }

    #[test]
    fn accepts_nested_segments() {
        assert!(validate_module_path("Demo", "Demo.Nn.Linear", Path::new("nn/linear.ch")).is_ok());
        assert!(validate_module_path("Demo", "Demo.Io.Json", Path::new("io/json.ch")).is_ok());
    }

    #[test]
    fn accepts_additional_source_root_as_first_component() {
        assert!(
            validate_module_path(
                "Demo",
                "Demo.Properties.Laws",
                Path::new("properties/laws.ch")
            )
            .is_ok()
        );
    }

    #[test]
    fn comparison_ignores_case_only() {
        // Case is free on both sides.
        assert!(validate_module_path("demo", "DEMO.LINALG", Path::new("linalg.ch")).is_ok());
        // Underscores are not: a path segment contributes itself verbatim, so
        // the PascalCase compaction §6.3 would suggest is a mismatch here.
        assert!(
            validate_module_path("Demo", "Demo.CrossEntropy", Path::new("cross_entropy.ch"))
                .is_err()
        );
        assert!(
            validate_module_path("Demo", "Demo.Cross_entropy", Path::new("cross_entropy.ch"))
                .is_ok()
        );
    }

    // --- negative: §6.5's first two violation bullets ---

    #[test]
    fn rejects_module_not_rooted_at_prefix() {
        let err = validate_module_path("Ub10", "Data", Path::new("data.ch"))
            .expect_err("unrooted module must be rejected");
        assert_eq!(err, "module `Data` does not belong to module_prefix `Ub10`");
    }

    #[test]
    fn rejects_module_equal_to_prefix() {
        // The prefix alone is not a member of its own ladder.
        let err = validate_module_path("Demo", "Demo", Path::new("demo.ch"))
            .expect_err("bare prefix must be rejected");
        assert!(err.contains("does not belong to module_prefix"), "{err}");
    }

    #[test]
    fn rejects_rooted_module_that_does_not_match_the_path() {
        let err = validate_module_path("Ub10", "Ub10.Other", Path::new("data.ch"))
            .expect_err("path mismatch must be rejected");
        assert_eq!(
            err,
            "module `Ub10.Other` does not match file path data.ch (expected `ub10.data`)"
        );
    }

    #[test]
    fn rejects_right_leaf_in_wrong_directory() {
        let err = validate_module_path("Demo", "Demo.Linear", Path::new("nn/linear.ch"))
            .expect_err("dropped directory component must be rejected");
        assert!(err.contains("does not match file path"), "{err}");
    }

    #[test]
    fn rejects_prefix_that_only_shares_a_leading_substring() {
        // `Demonstration` starts with `Demo` textually but is not rooted at it:
        // the rule is component-wise, not a raw string prefix.
        let err = validate_module_path("Demo", "Demonstration.Linalg", Path::new("linalg.ch"))
            .expect_err("substring-only prefix must be rejected");
        assert!(err.contains("does not belong to module_prefix"), "{err}");
    }
}
