//! Source architecture guard for the type-inference module tree
//! (openspec `modularize-type-inference`).
//!
//! Inference used to live in one 25,000-line `src/infer.rs`. The split into
//! `src/infer/` only stays split if something checks; review alone did not
//! stop the file growing the first time. This module is that check.
//!
//! It rejects three states:
//!
//! 1. the legacy monolith `src/infer.rs` is back,
//! 2. a production file under `src/infer/` exceeds [`MAX_INFER_FILE_LINES`],
//! 3. a required role module from [`REQUIRED_INFER_ROLES`] is missing.
//!
//! The core ([`check_inference_sources`]) is a pure function over a list of
//! `(path, physical line count)` pairs, so the unit fixtures below cover both
//! polarities without creating or deleting repository files. Only
//! [`tests::inference_source_tree_satisfies_architecture`] touches the disk.
//!
//! The line limit is a floor under the design, not the design itself: a file
//! can sit at 2,999 lines and still own three unrelated responsibilities. The
//! required-role list is what carries the actual boundaries.

/// Maximum physical lines for a production Rust file under `src/infer/`.
pub(crate) const MAX_INFER_FILE_LINES: usize = 3_000;

/// The legacy path whose return would undo the split.
pub(crate) const LEGACY_INFER_PATH: &str = "infer.rs";

/// Role modules the inference tree must keep, as `(role, module path)`.
///
/// These mirror the `type-inference-architecture` capability: separate
/// modules own program orchestration, checked-program construction,
/// annotation ownership, validation, expression forms, and application
/// inference, and the application dispatcher is separate from the numeric,
/// tensor, shape, and collection rules.
pub(crate) const REQUIRED_INFER_ROLES: &[(&str, &str)] = &[
    ("stack and recursion protection", "infer/stack.rs"),
    ("checked-program construction", "infer/checked.rs"),
    ("annotation ownership", "infer/annotate.rs"),
    ("program orchestration", "infer/program.rs"),
    ("declared type resolution", "infer/declared_type.rs"),
    ("validation", "infer/validate.rs"),
    ("expression forms", "infer/expr.rs"),
    ("application dispatch", "infer/app.rs"),
    ("numeric operation rules", "infer/app_numeric.rs"),
    ("tensor operation rules", "infer/app_tensor.rs"),
    ("shape operation rules", "infer/app_shape.rs"),
    ("collection operation rules", "infer/app_collection.rs"),
];

/// A way the inference source tree violates its architecture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SourceArchViolation {
    /// `src/infer.rs` exists again.
    LegacyMonolith { path: String },
    /// A file under `src/infer/` is longer than the limit.
    FileTooLong {
        path: String,
        lines: usize,
        limit: usize,
    },
    /// A required role module is absent.
    MissingRole { role: String, path: String },
}

impl std::fmt::Display for SourceArchViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SourceArchViolation::LegacyMonolith { path } => write!(
                f,
                "the legacy inference monolith `src/{path}` is back; inference belongs in the \
                 `src/infer/` module tree"
            ),
            SourceArchViolation::FileTooLong { path, lines, limit } => write!(
                f,
                "`src/{path}` has {lines} physical lines, over the {limit}-line limit; split it \
                 by responsibility rather than trimming it to fit"
            ),
            SourceArchViolation::MissingRole { role, path } => write!(
                f,
                "the `{role}` role module `src/{path}` is missing; the inference tree must keep \
                 one module per role"
            ),
        }
    }
}

/// Is this a production Rust source file inside the inference tree?
///
/// All `.rs` files under `infer/` count. A non-Rust file is not production
/// source and is not measured.
fn is_inference_source(path: &str) -> bool {
    path.starts_with("infer/") && path.ends_with(".rs")
}

/// The guard core.
///
/// `files` is every source path in the crate's `src/`, relative to `src/`,
/// paired with its physical line count. Returns every violation found, in a
/// stable order: legacy monolith, then oversized files in input order, then
/// missing roles in [`REQUIRED_INFER_ROLES`] order.
pub(crate) fn check_inference_sources(files: &[(String, usize)]) -> Vec<SourceArchViolation> {
    let mut violations = Vec::new();

    if files.iter().any(|(path, _)| path == LEGACY_INFER_PATH) {
        violations.push(SourceArchViolation::LegacyMonolith {
            path: LEGACY_INFER_PATH.to_string(),
        });
    }

    for (path, lines) in files {
        if is_inference_source(path) && *lines > MAX_INFER_FILE_LINES {
            violations.push(SourceArchViolation::FileTooLong {
                path: path.clone(),
                lines: *lines,
                limit: MAX_INFER_FILE_LINES,
            });
        }
    }

    for (role, required) in REQUIRED_INFER_ROLES {
        if !files.iter().any(|(path, _)| path == required) {
            violations.push(SourceArchViolation::MissingRole {
                role: (*role).to_string(),
                path: (*required).to_string(),
            });
        }
    }

    violations
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    /// A tree that satisfies every rule: no legacy file, every role present,
    /// all sizes under the limit.
    fn compliant_tree() -> Vec<(String, usize)> {
        let mut files: Vec<(String, usize)> = REQUIRED_INFER_ROLES
            .iter()
            .map(|(_, path)| ((*path).to_string(), 1_200))
            .collect();
        files.push(("infer/mod.rs".to_string(), 400));
        files.push(("lib.rs".to_string(), 40));
        files
    }

    // ── Positive fixtures ───────────────────────────────────────────────

    #[test]
    fn compliant_tree_has_no_violations() {
        assert_eq!(
            check_inference_sources(&compliant_tree()),
            Vec::new(),
            "a tree with every role, no legacy file, and no oversized file must pass"
        );
    }

    #[test]
    fn file_exactly_at_the_limit_is_accepted() {
        let mut files = compliant_tree();
        files.push((
            "infer/app_tensor_extra.rs".to_string(),
            MAX_INFER_FILE_LINES,
        ));
        assert_eq!(
            check_inference_sources(&files),
            Vec::new(),
            "{MAX_INFER_FILE_LINES} lines is the limit, not one line over it"
        );
    }

    #[test]
    fn oversized_file_outside_the_inference_tree_is_not_measured() {
        let mut files = compliant_tree();
        files.push(("builtins.rs".to_string(), 40_000));
        assert_eq!(
            check_inference_sources(&files),
            Vec::new(),
            "this guard governs `src/infer/` only; other modules have their own history"
        );
    }

    #[test]
    fn non_rust_file_in_the_inference_tree_is_not_measured() {
        let mut files = compliant_tree();
        files.push(("infer/notes.md".to_string(), 9_000));
        assert_eq!(
            check_inference_sources(&files),
            Vec::new(),
            "only production Rust source is measured"
        );
    }

    // ── Negative fixtures ───────────────────────────────────────────────

    #[test]
    fn legacy_monolith_is_rejected_and_named() {
        let mut files = compliant_tree();
        files.push((LEGACY_INFER_PATH.to_string(), 25_207));

        let violations = check_inference_sources(&files);
        assert_eq!(
            violations,
            vec![SourceArchViolation::LegacyMonolith {
                path: LEGACY_INFER_PATH.to_string(),
            }],
            "a returned `src/infer.rs` must be rejected on its own"
        );
        assert!(
            violations[0].to_string().contains("src/infer.rs"),
            "the diagnostic must name the legacy path, got: {}",
            violations[0]
        );
    }

    #[test]
    fn oversized_inference_file_is_rejected_with_its_line_count() {
        let mut files = compliant_tree();
        files.push(("infer/app_collection_huge.rs".to_string(), 3_001));

        let violations = check_inference_sources(&files);
        assert_eq!(
            violations,
            vec![SourceArchViolation::FileTooLong {
                path: "infer/app_collection_huge.rs".to_string(),
                lines: 3_001,
                limit: MAX_INFER_FILE_LINES,
            }],
            "one line over the limit must be rejected"
        );
        let rendered = violations[0].to_string();
        assert!(
            rendered.contains("infer/app_collection_huge.rs") && rendered.contains("3001"),
            "the diagnostic must name the file and its line count, got: {rendered}"
        );
    }

    #[test]
    fn missing_role_module_is_rejected_and_named() {
        let files: Vec<(String, usize)> = compliant_tree()
            .into_iter()
            .filter(|(path, _)| path != "infer/app_numeric.rs")
            .collect();

        let violations = check_inference_sources(&files);
        assert_eq!(
            violations,
            vec![SourceArchViolation::MissingRole {
                role: "numeric operation rules".to_string(),
                path: "infer/app_numeric.rs".to_string(),
            }],
            "dropping a role module must be rejected"
        );
        assert!(
            violations[0].to_string().contains("infer/app_numeric.rs"),
            "the diagnostic must name the missing module, got: {}",
            violations[0]
        );
    }

    #[test]
    fn every_required_role_is_individually_enforced() {
        for (role, required) in REQUIRED_INFER_ROLES {
            let files: Vec<(String, usize)> = compliant_tree()
                .into_iter()
                .filter(|(path, _)| path != required)
                .collect();
            assert_eq!(
                check_inference_sources(&files),
                vec![SourceArchViolation::MissingRole {
                    role: (*role).to_string(),
                    path: (*required).to_string(),
                }],
                "removing the `{role}` module must be rejected"
            );
        }
    }

    #[test]
    fn violations_accumulate_rather_than_stopping_at_the_first() {
        let mut files: Vec<(String, usize)> = compliant_tree()
            .into_iter()
            .filter(|(path, _)| path != "infer/expr.rs")
            .collect();
        files.push((LEGACY_INFER_PATH.to_string(), 25_207));
        files.push(("infer/app_huge.rs".to_string(), 5_000));

        let violations = check_inference_sources(&files);
        assert_eq!(
            violations.len(),
            3,
            "a legacy file, an oversized file, and a missing role are three separate \
             findings; a reader should see all of them at once, got: {violations:#?}"
        );
    }

    // ── The repository itself ───────────────────────────────────────────

    /// Physical line count: the number of newline-separated lines, matching
    /// what `wc -l` reports for a file with a trailing newline.
    fn physical_lines(contents: &str) -> usize {
        contents.lines().count()
    }

    fn collect_src_files(root: &Path, prefix: &str, out: &mut Vec<(String, usize)>) {
        let entries = std::fs::read_dir(root).expect("read src dir");
        for entry in entries {
            let entry = entry.expect("read dir entry");
            let name = entry.file_name().to_string_lossy().into_owned();
            let relative = if prefix.is_empty() {
                name.clone()
            } else {
                format!("{prefix}/{name}")
            };
            let path = entry.path();
            if path.is_dir() {
                collect_src_files(&path, &relative, out);
            } else {
                let contents = std::fs::read_to_string(&path)
                    .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
                out.push((relative, physical_lines(&contents)));
            }
        }
    }

    /// The live check against this crate's own `src/`.
    #[test]
    fn inference_source_tree_satisfies_architecture() {
        let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        collect_src_files(&src, "", &mut files);
        files.sort();

        let violations = check_inference_sources(&files);
        assert!(
            violations.is_empty(),
            "the inference source tree violates its architecture:\n{}",
            violations
                .iter()
                .map(|v| format!("  - {v}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}
