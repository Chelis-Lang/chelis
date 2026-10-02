//! The chelis-std runtime each Chelis binary embeds.
//!
//! chelis-std is the language runtime: every Chelis program depends on it the
//! way a Rust program depends on `core`. The build script packs
//! `packages/chelis-std` into this crate's `OUT_DIR` through the packing
//! step `chelis reef build` runs, and this crate embeds the resulting archive
//! and shell with `include_bytes!`. Nothing generated is committed: editing a
//! `.ch` file under `packages/chelis-std/src` and rebuilding is the whole
//! workflow.
//!
//! Only binaries and test harnesses depend on this crate. Libraries such as
//! `chelis-reef` and `chelis-compiler-api` take the runtime as an explicit
//! `&'static EmbeddedRuntime` parameter, and a binary passes
//! [`EMBEDDED_RUNTIME`].

use chelis_reef::EmbeddedRuntime;

/// The chelis-std version `packages/chelis-std/reef.toml` names.
pub const BUNDLED_CHELIS_STD_VERSION: &str =
    include_str!(concat!(env!("OUT_DIR"), "/chelis-std.version"));

/// The zstd-compressed tar archive of the chelis-std sources, as
/// `chelis reef build` writes `dist/chelis-std-<version>.tar.zst`.
pub const CHELIS_STD_ARCHIVE: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/chelis-std.tar.zst"));

/// The chelis-std shell, as `chelis reef build` writes
/// `dist/chelis-std-<version>.chb`.
pub const CHELIS_STD_SHELL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/chelis-std.chb"));

/// The runtime a binary passes to every graph entry point.
pub static EMBEDDED_RUNTIME: EmbeddedRuntime = EmbeddedRuntime::new(
    BUNDLED_CHELIS_STD_VERSION,
    CHELIS_STD_ARCHIVE,
    CHELIS_STD_SHELL,
);

#[cfg(test)]
#[path = "../build/stage.rs"]
mod stage;

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_shell::{
        SHELL_FORMAT_VERSION, TypeVariableDomain, TypeVariableRestriction, decode_shell,
    };
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};
    use tempfile::tempdir;

    fn std_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packages/chelis-std")
    }

    /// A copy of the runtime inputs in `dir`, the tree a test edits.
    fn std_copy(dir: &Path) -> PathBuf {
        let copy = dir.join("chelis-std");
        stage::stage(&std_root(), &copy).expect("copy the chelis-std inputs");
        copy
    }

    /// Pack `root` exactly as the build script packs `packages/chelis-std`.
    fn pack_as_build_script(root: &Path, dir: &Path) -> chelis_reef::PackedPackage {
        let staged = dir.join("staged");
        stage::stage(root, &staged).expect("stage the runtime inputs");
        stage::pack(&staged).expect("pack the staged runtime")
    }

    fn assert_embedded(packed: &chelis_reef::PackedPackage) {
        assert!(
            packed.archive == CHELIS_STD_ARCHIVE,
            "packed archive differs from the embedded archive"
        );
        assert!(
            packed.shell == CHELIS_STD_SHELL,
            "packed shell differs from the embedded shell"
        );
    }

    /// The embedded shell names the embedded archive and the manifest
    /// version, the invariant the installer checks for any artifact pair.
    #[test]
    fn archive_self_consistency() {
        let shell = decode_shell(CHELIS_STD_SHELL).expect("decode embedded shell");
        assert_eq!(shell.package.name, "chelis-std");
        assert_eq!(shell.package.version, BUNDLED_CHELIS_STD_VERSION);
        assert_eq!(shell.archive_sha256, EMBEDDED_RUNTIME.archive_sha256());
    }

    #[test]
    fn version_is_the_std_manifest_version() {
        let manifest = chelis_reef::read_manifest_for_src(&std_root()).expect("read manifest");
        assert_eq!(BUNDLED_CHELIS_STD_VERSION, manifest.package.version);
        assert_eq!(EMBEDDED_RUNTIME.version(), manifest.package.version);
    }

    /// The build-time pair is byte-identical to the pair committed before the
    /// runtime was packed at build time.
    #[test]
    fn build_time_pair_matches_the_committed_pair() {
        for dist in [
            std_root().join("dist"),
            Path::new(env!("CARGO_MANIFEST_DIR")).join("dist"),
        ] {
            let stem = format!("chelis-std-{BUNDLED_CHELIS_STD_VERSION}");
            let archive = std::fs::read(dist.join(format!("{stem}.tar.zst"))).expect("archive");
            let shell = std::fs::read(dist.join(format!("{stem}.chb"))).expect("shell");
            assert!(archive == CHELIS_STD_ARCHIVE, "{} archive", dist.display());
            assert!(shell == CHELIS_STD_SHELL, "{} shell", dist.display());
        }
    }

    #[test]
    fn embedded_shell_preserves_std_test_active_float_scheme() {
        let shell = decode_shell(CHELIS_STD_SHELL).expect("decode embedded shell");
        assert_eq!(shell.format_version, SHELL_FORMAT_VERSION);
        let symbol = shell
            .modules
            .iter()
            .find(|module| module.module == "Std.Test")
            .and_then(|module| {
                module
                    .exports
                    .iter()
                    .find(|symbol| symbol.name == "assert_close_tensor")
            })
            .expect("Std.Test.assert_close_tensor must be shipped");
        assert_eq!(
            symbol.type_variable_restrictions,
            vec![TypeVariableRestriction {
                variable: "t0".to_string(),
                domain: TypeVariableDomain::ActiveFloat,
            }]
        );
        let type_repr = symbol.type_repr.as_deref().expect("function type");
        assert!(type_repr.contains("(d-rank {} r0)"));
        assert!(type_repr.matches("(t-var {} t0)").count() >= 3);
    }

    /// An extraction is a reef package tree whose `compiler =` pin is this
    /// build's version, the pin `validate_manifest` requires when the runtime
    /// loads.
    #[test]
    fn extract_yields_reef_package_layout() {
        let dir = tempdir().expect("tempdir");
        EMBEDDED_RUNTIME
            .extract_into(dir.path())
            .expect("extract bundle");

        let manifest = std::fs::read_to_string(dir.path().join("reef.toml")).expect("reef.toml");
        assert!(manifest.contains("name = \"chelis-std\""), "{manifest}");
        let expected_compiler_line = format!("compiler = \"={}\"", env!("CARGO_PKG_VERSION"));
        assert!(manifest.contains(&expected_compiler_line), "{manifest}");

        let any_ch = std::fs::read_dir(dir.path().join("src"))
            .expect("read src/")
            .filter_map(|e| e.ok())
            .any(|e| e.path().extension().and_then(|s| s.to_str()) == Some("ch"));
        assert!(any_ch, "src/ must contain at least one .ch file");
    }

    /// The in-memory view the reef loader reads (chelis#2616) is exactly the
    /// tree an extraction writes: the same package-relative paths and bytes.
    #[test]
    fn archive_files_match_an_extracted_tree() {
        let dir = tempdir().expect("tempdir");
        EMBEDDED_RUNTIME
            .extract_into(dir.path())
            .expect("extract bundle");
        let mut on_disk = BTreeMap::new();
        let mut pending = vec![dir.path().to_path_buf()];
        while let Some(current) = pending.pop() {
            for entry in std::fs::read_dir(&current).expect("read extracted dir") {
                let path = entry.expect("read extracted entry").path();
                if path.is_dir() {
                    pending.push(path);
                } else {
                    let relative = path.strip_prefix(dir.path()).expect("relative");
                    on_disk.insert(relative.to_path_buf(), std::fs::read(&path).expect("read"));
                }
            }
        }
        let in_memory = EMBEDDED_RUNTIME
            .archive_files()
            .expect("read archive in memory");
        assert!(in_memory.contains_key(Path::new("reef.toml")));
        assert_eq!(in_memory, &on_disk);
    }

    /// The archive holds exactly the staged inputs: the manifest and the
    /// `.ch` sources, nothing else from the source tree.
    #[test]
    fn archive_members_are_the_runtime_inputs() {
        let inputs = stage::runtime_inputs(&std_root()).expect("runtime inputs");
        let members = EMBEDDED_RUNTIME
            .archive_files()
            .expect("archive files")
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(members, inputs);
    }

    /// The build script asks cargo to rerun when a watched path changes, so
    /// every input must lie under one.
    #[test]
    fn every_runtime_input_is_watched() {
        let watched = stage::watched_paths(&std_root()).expect("watched paths");
        for input in stage::runtime_inputs(&std_root()).expect("runtime inputs") {
            assert!(
                watched.iter().any(|path| input.starts_with(path)),
                "{} is not under a watched path",
                input.display()
            );
        }
    }

    /// Selection keeps the manifest, `.ch` files under every declared source
    /// root, and declared metadata files, and drops everything else.
    #[test]
    fn staging_selects_sources_manifest_and_declared_metadata() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let write = |relative: &str, text: &str| {
            let path = root.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        };
        write(
            "reef.toml",
            &format!(
                "schema = \"3\"\n[package]\nname = \"chelis-std\"\nversion = \"0.0.1\"\n\
                 compiler = \"={}\"\nmodule_prefix = \"Std\"\nresolver = \"2\"\n\
                 additional_sources = [\"extra\"]\nreadme = \"README.md\"\n",
                env!("CARGO_PKG_VERSION")
            ),
        );
        write("src/a.ch", "module Std.A\n");
        write("src/nested/b.ch", "module Std.Nested.B\n");
        write("src/.DS_Store", "litter");
        write("src/notes.txt", "litter");
        write("extra/c.ch", "module Std.C\n");
        write("extra/c.ch.swp", "litter");
        write("README.md", "readme");
        write("tests/t.ch", "module Std.T\n");
        let inputs = stage::runtime_inputs(root).expect("runtime inputs");
        let expected = [
            "README.md",
            "extra/c.ch",
            "reef.toml",
            "src/a.ch",
            "src/nested/b.ch",
        ]
        .map(PathBuf::from);
        assert_eq!(inputs, expected);
    }

    /// A file that is not a runtime input, such as the `.DS_Store` Finder
    /// writes, does not change the bundle.
    #[test]
    fn a_stray_file_in_std_src_does_not_change_the_bundle() {
        let dir = tempdir().expect("tempdir");
        let copy = std_copy(dir.path());
        std::fs::write(copy.join("src/.DS_Store"), b"\0\0\0\x01Bud1").unwrap();
        std::fs::write(copy.join("src/io/notes.txt"), b"scratch").unwrap();
        assert_embedded(&pack_as_build_script(&copy, dir.path()));
    }

    /// The archive mtime is pinned, so the `SOURCE_DATE_EPOCH` nixpkgs and
    /// other reproducible-build drivers export does not change the bundle,
    /// although `chelis reef build` honors it.
    #[test]
    fn source_date_epoch_does_not_change_the_bundle() {
        // SAFETY: this test owns its process under the workspace test runner,
        // and no other thread reads the environment while it is set.
        unsafe { std::env::set_var("SOURCE_DATE_EPOCH", "1700000000") };
        let dir = tempdir().expect("tempdir");
        let packed = pack_as_build_script(&std_root(), dir.path());
        unsafe { std::env::remove_var("SOURCE_DATE_EPOCH") };
        assert_embedded(&packed);
    }

    /// Workspace packages and their declared normal and build dependencies on
    /// other workspace packages, from `cargo metadata`. Dev-dependencies are
    /// left out: they reach only test harnesses.
    fn workspace_library_edges() -> BTreeMap<String, Vec<String>> {
        let output = std::process::Command::new(env!("CARGO"))
            .args([
                "metadata",
                "--no-deps",
                "--offline",
                "--format-version",
                "1",
            ])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .expect("run cargo metadata");
        assert!(
            output.status.success(),
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let metadata: serde_json::Value =
            serde_json::from_slice(&output.stdout).expect("cargo metadata JSON");
        let packages = metadata["packages"].as_array().expect("packages");
        let names = packages
            .iter()
            .map(|package| package["name"].as_str().expect("name").to_string())
            .collect::<std::collections::BTreeSet<_>>();
        packages
            .iter()
            .map(|package| {
                let edges = package["dependencies"]
                    .as_array()
                    .expect("dependencies")
                    .iter()
                    .filter(|dependency| dependency["kind"].as_str() != Some("dev"))
                    .map(|dependency| dependency["name"].as_str().expect("name").to_string())
                    .filter(|name| names.contains(name))
                    .collect();
                (package["name"].as_str().unwrap().to_string(), edges)
            })
            .collect()
    }

    fn library_closure(edges: &BTreeMap<String, Vec<String>>, root: &str) -> Vec<String> {
        let mut seen = std::collections::BTreeSet::new();
        let mut pending = vec![root.to_string()];
        while let Some(package) = pending.pop() {
            if seen.insert(package.clone()) {
                pending.extend(edges.get(&package).into_iter().flatten().cloned());
            }
        }
        seen.into_iter().collect()
    }

    /// Libraries take the runtime as a parameter, so checking or building one
    /// never packs the std: the bundle is outside their dependency graphs.
    #[test]
    fn libraries_build_without_the_bundle() {
        let edges = workspace_library_edges();
        for library in ["chelis-types", "chelis-reef", "chelis-compiler-api"] {
            let closure = library_closure(&edges, library);
            assert!(
                !closure.iter().any(|package| package == "chelis-std-bundle"),
                "{library} reaches chelis-std-bundle through {closure:?}"
            );
        }
    }

    /// Only a final artifact (a binary or an extension module) depends on the
    /// bundle; no workspace package depends on such a package in turn.
    #[test]
    fn only_final_artifacts_depend_on_the_bundle() {
        let edges = workspace_library_edges();
        let dependents = edges
            .iter()
            .filter(|(_, deps)| deps.iter().any(|dep| dep == "chelis-std-bundle"))
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>();
        assert!(!dependents.is_empty(), "no binary embeds the runtime");
        for dependent in dependents {
            let users = edges
                .iter()
                .filter(|(_, deps)| deps.contains(&dependent))
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>();
            assert!(
                users.is_empty(),
                "{dependent} embeds the runtime but {users:?} depend on it"
            );
        }
    }

    /// Editing a std source changes both packed artifacts, and reverting the
    /// edit restores the embedded bytes exactly.
    #[test]
    fn a_std_source_edit_changes_the_bundle_and_reverting_restores_it() {
        let dir = tempdir().expect("tempdir");
        let copy = std_copy(dir.path());
        let edited = copy.join("src/text.ch");
        let original = std::fs::read(&edited).expect("read std source");
        let mut changed = original.clone();
        changed.extend_from_slice(b"\n-- an edit\n");
        std::fs::write(&edited, &changed).unwrap();
        let packed = pack_as_build_script(&copy, &dir.path().join("edited"));
        assert!(packed.archive != CHELIS_STD_ARCHIVE, "archive unchanged");
        assert!(packed.shell != CHELIS_STD_SHELL, "shell unchanged");

        std::fs::write(&edited, &original).unwrap();
        assert_embedded(&pack_as_build_script(&copy, &dir.path().join("reverted")));
    }
}
