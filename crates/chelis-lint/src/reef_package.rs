//! Minimal reef-package layout reading, shared by the traversal policy and
//! `reef-module-identity`.
//!
//! Both need the same two facts from a `reef.toml` — the package's
//! `module_prefix` and which directories are source roots — and they must not
//! disagree about them. Two readers would be free to drift, and a drift here
//! is invisible: the policy would prune a directory the rule believes it is
//! judging, or the rule would judge a file the policy believes is excluded.
//!
//! This is deliberately not a reef manifest parser. It reads two keys and
//! declines on anything else, because `chelis-lint` is dependency-pure and
//! reaching `chelis-reef` for 30 lines of TOML would undo that. A manifest
//! reef itself rejects is not this crate's to diagnose.

use serde::Deserialize;
use std::path::Path;

/// The package facts the lint needs.
#[derive(Debug, Clone)]
pub struct PackageLayout {
    /// `package.module_prefix`, non-empty.
    pub module_prefix: String,
    /// Source roots relative to the package root, `src` first.
    pub source_roots: Vec<String>,
}

#[derive(Deserialize)]
struct ManifestFile {
    package: Option<ManifestPackage>,
}

#[derive(Deserialize)]
struct ManifestPackage {
    module_prefix: Option<String>,
    #[serde(default)]
    additional_sources: Vec<String>,
}

/// Read a `reef.toml`'s layout, or `None` if it is not a package manifest this
/// crate can read: unreadable, not TOML, or declaring no `module_prefix`.
pub fn read_layout(manifest: &Path) -> Option<PackageLayout> {
    let text = std::fs::read_to_string(manifest).ok()?;
    let parsed: ManifestFile = toml::from_str(&text).ok()?;
    let package = parsed.package?;
    let module_prefix = package.module_prefix?;
    if module_prefix.trim().is_empty() {
        return None;
    }
    let mut source_roots = vec!["src".to_string()];
    source_roots.extend(package.additional_sources);
    Some(PackageLayout {
        module_prefix,
        source_roots,
    })
}

/// Whether `path` lies inside a declared source root of a reef package rooted
/// at or below `boundary`.
///
/// The climb stops at `boundary` so a manifest above the lint's policy root
/// cannot change what the lint walks (§12.2: machine-local ancestors above the
/// policy root cannot grant lint exceptions). The innermost package wins, as
/// it does for the rule.
pub fn inside_package_source_root(path: &Path, boundary: &Path) -> bool {
    let mut cursor = path.parent();
    while let Some(directory) = cursor {
        let manifest = directory.join("reef.toml");
        if manifest.is_file()
            && let Some(layout) = read_layout(&manifest)
            && let Ok(relative) = path.strip_prefix(directory)
            && let Some(first) = relative.components().next()
            && let Some(first) = first.as_os_str().to_str()
        {
            return layout.source_roots.iter().any(|root| root == first);
        }
        if directory == boundary {
            break;
        }
        cursor = directory.parent();
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn pkg(root: &Path, extra: &str) -> PathBuf {
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::write(
            root.join("reef.toml"),
            format!(
                "[package]\nname = \"d\"\nversion = \"0.1.0\"\nmodule_prefix = \"Demo\"\n{extra}"
            ),
        )
        .expect("write");
        root.to_path_buf()
    }

    #[test]
    fn reads_prefix_and_default_source_root() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = pkg(tmp.path(), "");
        let layout = read_layout(&root.join("reef.toml")).expect("layout");
        assert_eq!(layout.module_prefix, "Demo");
        assert_eq!(layout.source_roots, vec!["src".to_string()]);
    }

    #[test]
    fn reads_additional_sources_after_src() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = pkg(tmp.path(), "additional_sources = [\"properties\"]\n");
        let layout = read_layout(&root.join("reef.toml")).expect("layout");
        assert_eq!(
            layout.source_roots,
            vec!["src".to_string(), "properties".to_string()]
        );
    }

    #[test]
    fn declines_a_manifest_with_no_module_prefix() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        fs::write(root.join("reef.toml"), "[package]\nname = \"d\"\n").expect("write");
        assert!(read_layout(&root.join("reef.toml")).is_none());
    }

    #[test]
    fn declines_unreadable_and_non_toml() {
        let tmp = tempfile::tempdir().expect("tempdir");
        assert!(read_layout(&tmp.path().join("absent.toml")).is_none());
        fs::write(tmp.path().join("reef.toml"), "not : toml : at all\n").expect("write");
        assert!(read_layout(&tmp.path().join("reef.toml")).is_none());
    }

    #[test]
    fn recognises_a_path_inside_a_source_root() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = pkg(tmp.path(), "");
        assert!(inside_package_source_root(
            &root.join("src/target/x.ch"),
            &root
        ));
        assert!(inside_package_source_root(
            &root.join("src/a/b/x.ch"),
            &root
        ));
    }

    #[test]
    fn rejects_a_path_outside_every_source_root() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = pkg(tmp.path(), "");
        assert!(!inside_package_source_root(
            &root.join("target/x.ch"),
            &root
        ));
        assert!(!inside_package_source_root(
            &root.join("docs/target/x.ch"),
            &root
        ));
    }

    #[test]
    fn rejects_when_no_manifest_is_reachable() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        std::fs::create_dir_all(root.join("src")).expect("mkdir");
        assert!(!inside_package_source_root(&root.join("src/x.ch"), root));
    }

    #[test]
    fn does_not_climb_above_the_boundary() {
        // The manifest is at `outer`, the boundary is `outer/inner`. A file
        // under `outer/inner/src` must not be granted by a manifest the lint's
        // policy root does not cover.
        let tmp = tempfile::tempdir().expect("tempdir");
        let outer = pkg(tmp.path(), "");
        let inner = outer.join("inner");
        fs::create_dir_all(inner.join("src")).expect("mkdir");
        assert!(!inside_package_source_root(&inner.join("src/x.ch"), &inner));
    }
}
