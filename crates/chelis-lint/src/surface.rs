//! Surface kinds — the file/entry classification that rules dispatch on.

use std::path::Path;

/// What kind of entry the lint is looking at. Each rule names the surfaces it
/// applies to; the driver only invokes the rule when the entry matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Surface {
    /// `.ch` — Surf source.
    SurfSource,
    /// `.dp` — Deep source (test fixtures, translator output).
    DeepSource,
    /// `.rs` — Rust source.
    RustSource,
    /// `.py` — Python source.
    PythonSource,
    /// `.sh` — shell script (project policy: don't write these).
    ShellScript,
    /// `Cargo.toml` or `reef.toml` — manifest.
    ManifestToml,
    /// `.md` documentation file.
    DocFile,
    /// `.snap` — insta snapshot test file.
    SnapshotFile,
    /// `.yml` under `.github/workflows/`.
    WorkflowFile,
    /// A directory entry — used for rules that check directory naming
    /// (Rust crate dirs, top-level repo dirs, etc.).
    Directory,
}

impl Surface {
    /// Classify a filesystem path by extension and location. Returns `None`
    /// for entries the lint doesn't care about (binary blobs, build outputs,
    /// other unknown file types).
    #[must_use]
    pub fn classify(path: &Path, is_dir: bool) -> Option<Self> {
        if is_dir {
            return Some(Surface::Directory);
        }
        let name = path.file_name()?.to_str()?;
        if name == "Cargo.toml" || name == "reef.toml" {
            return Some(Surface::ManifestToml);
        }
        let ext = path.extension()?.to_str()?;
        match ext {
            "ch" => Some(Surface::SurfSource),
            "dp" => Some(Surface::DeepSource),
            "rs" => Some(Surface::RustSource),
            "py" => Some(Surface::PythonSource),
            "sh" => Some(Surface::ShellScript),
            "md" => Some(Surface::DocFile),
            "snap" => Some(Surface::SnapshotFile),
            "yml" | "yaml" => {
                // Only treat as workflow file if under .github/workflows/.
                // Walk the path's parents looking for the `.github/workflows`
                // pair so we match both absolute and relative paths.
                let parent = path.parent()?;
                if parent.file_name().and_then(|n| n.to_str()) == Some("workflows")
                    && parent
                        .parent()
                        .and_then(|p| p.file_name())
                        .and_then(|n| n.to_str())
                        == Some(".github")
                {
                    Some(Surface::WorkflowFile)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Whether the rule driver should read the file's text content for this
    /// surface. Source/text surfaces yes; directory entries and snapshot
    /// binaries no.
    #[must_use]
    pub fn needs_source(self) -> bool {
        !matches!(self, Surface::Directory)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn classify(path: &str, is_dir: bool) -> Option<Surface> {
        Surface::classify(&PathBuf::from(path), is_dir)
    }

    #[test]
    fn classifies_surf() {
        assert_eq!(classify("src/linalg.ch", false), Some(Surface::SurfSource));
    }

    #[test]
    fn classifies_deep_fixture() {
        assert_eq!(
            classify("tests/fixtures/simple_def.dp", false),
            Some(Surface::DeepSource)
        );
    }

    #[test]
    fn classifies_rust_module() {
        assert_eq!(
            classify("crates/foo/src/lib.rs", false),
            Some(Surface::RustSource)
        );
    }

    #[test]
    fn classifies_python() {
        assert_eq!(
            classify("scripts/build.py", false),
            Some(Surface::PythonSource)
        );
    }

    #[test]
    fn classifies_shell_script() {
        assert_eq!(
            classify("scripts/install.sh", false),
            Some(Surface::ShellScript)
        );
    }

    #[test]
    fn classifies_cargo_manifest() {
        assert_eq!(classify("Cargo.toml", false), Some(Surface::ManifestToml));
        assert_eq!(
            classify("crates/foo/Cargo.toml", false),
            Some(Surface::ManifestToml)
        );
    }

    #[test]
    fn classifies_reef_manifest() {
        assert_eq!(classify("reef.toml", false), Some(Surface::ManifestToml));
    }

    #[test]
    fn classifies_doc() {
        assert_eq!(classify("docs/note.md", false), Some(Surface::DocFile));
    }

    #[test]
    fn classifies_snapshot() {
        assert_eq!(
            classify("tests/snapshots/cli__error.snap", false),
            Some(Surface::SnapshotFile)
        );
    }

    #[test]
    fn classifies_workflow_only_under_github_workflows() {
        assert_eq!(
            classify(".github/workflows/ci.yml", false),
            Some(Surface::WorkflowFile)
        );
        // A yml file outside .github/workflows/ is not a workflow file.
        assert_eq!(classify("config/foo.yml", false), None);
    }

    #[test]
    fn classifies_directory() {
        assert_eq!(
            classify("crates/chelis-lint", true),
            Some(Surface::Directory)
        );
    }

    #[test]
    fn unknown_extension_is_none() {
        assert_eq!(classify("data/blob.bin", false), None);
        assert_eq!(classify("README", false), None);
    }
}
