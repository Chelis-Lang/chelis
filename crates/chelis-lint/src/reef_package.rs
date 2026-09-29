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

/// Read a `reef.toml`'s layout, or `None` when it cannot be read as one.
///
/// This is deliberately not a manifest validator. It mirrors exactly one of
/// reef's checks — `additional_sources`, because those values name directories
/// the traversal policy will stop pruning, so a value reef rejects must not
/// widen the lint corpus. Everything else reef validates (`name`, `version`,
/// the compiler pin, schema selection, dependencies) is not checked here, so a
/// manifest that fails those can still yield a layout. Mirroring them would
/// make this a second manifest parser, which is the drift this module exists
/// to avoid.
///
/// Whether a manifest is allowed to speak for the lint at all is a traversal
/// question, not a parsing one: see `inside_package_source_root`.
pub fn read_layout(manifest: &Path) -> Option<PackageLayout> {
    let text = std::fs::read_to_string(manifest).ok()?;
    let parsed: ManifestFile = toml::from_str(&text).ok()?;
    let package = parsed.package?;
    let module_prefix = package.module_prefix?;
    if module_prefix.trim().is_empty() {
        return None;
    }
    let mut source_roots = vec!["src".to_string()];
    for entry in package.additional_sources {
        if !is_valid_additional_source(&entry) || source_roots.contains(&entry) {
            return None;
        }
        source_roots.push(entry);
    }
    Some(PackageLayout {
        module_prefix,
        source_roots,
    })
}

/// Reserved source-root names, mirroring `chelis_reef`.
const RESERVED_ADDITIONAL_SOURCE_DIRS: [&str; 2] = ["src", "tests"];

/// Whether `entry` is an `additional_sources` value reef would accept.
fn is_valid_additional_source(entry: &str) -> bool {
    !entry.trim().is_empty()
        && !RESERVED_ADDITIONAL_SOURCE_DIRS.contains(&entry)
        && !entry.contains('/')
        && !entry.contains('\\')
        && entry
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Whether `path` lies inside a declared source root of a reef package rooted
/// at or below `boundary`.
///
/// The climb stops at `boundary` so a manifest above the lint's policy root
/// cannot change what the lint walks (§12.2: machine-local ancestors above the
/// policy root cannot grant lint exceptions). The innermost package wins, as
/// it does for the rule.
///
/// `admit_manifest` decides whether a located `reef.toml` may speak for the
/// lint. The traversal policy owns that question and passes its own
/// `is_admitted_ancillary`, which is the same admission
/// `reef-module-identity` applies to the same file. One reader and one
/// admission rule: an earlier version declined every symlinked manifest here
/// instead, which refused the internal links §12.2 explicitly admits
/// ("internal links to admitted regular files remain visible") and which the
/// reef loader follows, so the lint went silent on a package that does not
/// build.
pub fn inside_package_source_root(
    path: &Path,
    boundary: &Path,
    admit_manifest: &dyn Fn(&Path) -> bool,
) -> bool {
    let mut cursor = path.parent();
    while let Some(directory) = cursor {
        let manifest = directory.join("reef.toml");
        if manifest.is_file()
            && admit_manifest(&manifest)
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

    /// Admit every manifest: these unit tests exercise the layout and climb
    /// rules, not the policy's admission, which has its own tests.
    const ADMIT_ALL: fn(&Path) -> bool = |_| true;
    /// Refuse every manifest, to prove admission is load-bearing here.
    const REFUSE_ALL: fn(&Path) -> bool = |_| false;

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
            &root,
            &ADMIT_ALL
        ));
        assert!(inside_package_source_root(
            &root.join("src/a/b/x.ch"),
            &root,
            &ADMIT_ALL
        ));
    }

    #[test]
    fn rejects_a_path_outside_every_source_root() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = pkg(tmp.path(), "");
        assert!(!inside_package_source_root(
            &root.join("target/x.ch"),
            &root,
            &ADMIT_ALL
        ));
        assert!(!inside_package_source_root(
            &root.join("docs/target/x.ch"),
            &root,
            &ADMIT_ALL
        ));
    }

    #[test]
    fn rejects_when_no_manifest_is_reachable() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        std::fs::create_dir_all(root.join("src")).expect("mkdir");
        assert!(!inside_package_source_root(
            &root.join("src/x.ch"),
            root,
            &ADMIT_ALL
        ));
    }

    #[test]
    fn does_not_climb_above_the_boundary() {
        // An earlier version of this test put the boundary at `outer/inner`
        // and asserted a file under `outer/inner/src` was not granted. It
        // passed whether or not the boundary existed: without the break the
        // climb reached `outer`, and the path relative to `outer` starts with
        // `inner`, which is not a source root, so the answer was `false`
        // either way. Deleting the boundary killed 0 of 411 tests.
        //
        // The knob only moves when the intermediate directory is itself named
        // like the outer package's source root, so a climb past the boundary
        // WOULD find a granting manifest.
        let tmp = tempfile::tempdir().expect("tempdir");
        let outer = pkg(tmp.path(), "");
        let boundary = outer.join("src");
        fs::create_dir_all(boundary.join("src")).expect("mkdir");
        assert!(
            !inside_package_source_root(&boundary.join("src/x.ch"), &boundary, &ADMIT_ALL),
            "a manifest above the policy root must not grant a lint exception"
        );
    }

    #[test]
    fn declines_a_manifest_reef_would_reject() {
        for bad in [
            "additional_sources = [\".git\"]\n",
            "additional_sources = [\"a/b\"]\n",
            "additional_sources = [\"..\"]\n",
            "additional_sources = [\"src\"]\n",
            "additional_sources = [\"p\", \"p\"]\n",
            "additional_sources = [\"tests\"]\n",
            "additional_sources = [\"\"]\n",
            "additional_sources = [\"   \"]\n",
        ] {
            let tmp = tempfile::tempdir().expect("tempdir");
            let root = pkg(tmp.path(), bad);
            assert!(
                read_layout(&root.join("reef.toml")).is_none(),
                "reef rejects {bad:?}, so the lint must grant nothing from it"
            );
        }
    }

    #[test]
    fn a_symlinked_manifest_is_read_and_admission_is_the_callers_call() {
        // An earlier version declined every symlinked `reef.toml` here. That
        // refused the internal links §12.2 explicitly admits ("internal links
        // to admitted regular files remain visible") and that the reef loader
        // follows, so `chelis lint --check` went silent on a package
        // `chelis reef build` rejects -- reintroducing the very false green
        // this change exists to remove. Reading is unconditional; whether a
        // manifest may speak for the lint belongs to the traversal policy,
        // which applies the same admission to it as `reef-module-identity`.
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path().join("pkg");
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::create_dir_all(root.join("shared")).expect("mkdir");
        fs::write(
            root.join("shared/pkg.toml"),
            "[package]\nname = \"d\"\nversion = \"0.1.0\"\nmodule_prefix = \"Demo\"\n",
        )
        .expect("write");
        std::os::unix::fs::symlink(root.join("shared/pkg.toml"), root.join("reef.toml"))
            .expect("symlink");

        assert!(
            read_layout(&root.join("reef.toml")).is_some(),
            "an internal symlinked manifest is still a manifest"
        );
        assert!(
            inside_package_source_root(&root.join("src/target/x.ch"), &root, &ADMIT_ALL),
            "an admitted symlinked manifest grants its source roots"
        );
        assert!(
            !inside_package_source_root(&root.join("src/target/x.ch"), &root, &REFUSE_ALL),
            "a manifest the caller refuses grants nothing"
        );
    }
}
