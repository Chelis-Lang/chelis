//! Compile-time-embedded chelis-std runtime artifacts.
//!
//! The chelis-std runtime ships with the chelis compiler — every chelis
//! program implicitly depends on it the same way a Rust program implicitly
//! depends on `core`/`std`. Distribution-wise, the runtime's `.tar.zst`
//! archive and `.chb` shell are baked into the chelis binary at compile
//! time via `include_bytes!()`. The reef loader checks for chelis-std
//! specifically and serves bytes from this crate; everything else
//! continues to flow through the local-registry path.
//!
//! ## Why a separate crate
//!
//! Keeping the embedded bytes in their own crate:
//! - leaves `chelis-runtime` (the C-runtime FFI crate) untouched;
//! - lets `chelis-reef` declare a clean dependency edge to the bundle
//!   without inheriting the bytes' crate-internal layout;
//! - confines the build.rs / `include_bytes!` surface to one place;
//! - makes regeneration explicit (run `scripts/regenerate_chelis_std_bundle.py`,
//!   then commit every output listed below).
//!
//! ## Artifact regeneration
//!
//! 1. Edit `packages/chelis-std/src/**.ch` as needed.
//! 2. Run `python3 scripts/regenerate_chelis_std_bundle.py`. The script:
//!    - builds the chelis CLI in release mode;
//!    - runs `chelis reef build` against `packages/chelis-std/`;
//!    - writes the resulting `chelis-std-<version>.tar.zst` and `.chb`
//!      under `packages/chelis-std/dist/`;
//!    - copies that pair into `crates/chelis-std-bundle/dist/`; and
//!    - regenerates `packages/chelis-std/reef.lock` from those final bytes.
//! 3. Commit both dist pairs and `packages/chelis-std/reef.lock`.
//! 4. Run `python3 scripts/regenerate_chelis_std_bundle.py --check`. The
//!    check regenerates twice, requires the committed five-output set to
//!    equal the first pass and both passes to be byte-identical, and restores
//!    the committed inputs without changing the worktree.
//! 5. Rebuilding any crate that depends on this one (`chelis-reef`,
//!    `chelis-cli`) picks up the new bytes.
//!
//! ## Loader integration
//!
//! The reef loader reads the runtime through [`archive_files`], which
//! decompresses the embedded archive in memory into package-relative paths
//! (`reef.toml`, `src/...`), so the bundled package has no filesystem
//! location (chelis#2616). [`extract_into`] writes the same tree to a
//! caller-supplied directory for callers that need the files on disk.

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

/// The version of chelis-std these embedded bytes provide. Hand-
/// maintained in lockstep with `crates/chelis-reef/src/lib.rs`'s
/// `BUNDLED_CHELIS_STD_VERSION` constant and with
/// `packages/chelis-std/reef.toml`'s `[package].version`. The
/// `bundled_chelis_std_version_matches_packages_manifest` test in
/// chelis-reef and `archive_self_consistency` here assert agreement.
pub const BUNDLED_CHELIS_STD_VERSION: &str = "0.4.0";

/// The compile-time-embedded zstd-compressed tar archive of the
/// chelis-std source tree. Layout inside the archive mirrors the
/// `packages/chelis-std/` source dir: `reef.toml`, `src/*.ch`, etc.
/// Same shape as the on-disk artifact `chelis reef build` produces
/// for any reef package.
pub const CHELIS_STD_ARCHIVE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/dist/chelis-std-0.4.0.tar.zst"
));

/// The compile-time-embedded `ShellPackage` (bincode) for chelis-std,
/// covering its public surface (modules, exports, types). Same shape
/// as `<name>-<version>.chb` produced by `chelis reef build`.
pub const CHELIS_STD_SHELL: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/dist/chelis-std-0.4.0.chb"
));

/// SHA256 of [`CHELIS_STD_ARCHIVE`], computed at runtime on first
/// access. Used by the reef loader to fill `LockedDependency.archive_sha256`
/// for synthesized `Bundled` lockfile entries.
pub fn archive_sha256() -> String {
    let mut hasher = Sha256::new();
    hasher.update(CHELIS_STD_ARCHIVE);
    hex_encode(&hasher.finalize())
}

/// SHA256 of [`CHELIS_STD_SHELL`], computed at runtime on first
/// access. Used by the reef loader to fill `LockedDependency.shell_sha256`
/// for synthesized `Bundled` lockfile entries.
pub fn shell_sha256() -> String {
    let mut hasher = Sha256::new();
    hasher.update(CHELIS_STD_SHELL);
    hex_encode(&hasher.finalize())
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

/// Decompress and untar [`CHELIS_STD_ARCHIVE`] into `dest`. The
/// directory is created if it does not already exist. After this
/// returns successfully, `dest` will contain a chelis-std package
/// tree (`reef.toml`, `src/`, etc.) suitable for
/// `load_package_modules` and `read_manifest`.
///
/// Errors propagate as plain strings to match the chelis-reef error
/// idiom — the caller wraps in its richer error type.
pub fn extract_into(dest: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dest).map_err(|e| {
        format!(
            "failed to create chelis-std bundle dest {}: {e}",
            dest.display()
        )
    })?;
    let decoder = zstd::stream::read::Decoder::new(CHELIS_STD_ARCHIVE)
        .map_err(|e| format!("failed to start zstd decoder for chelis-std bundle: {e}"))?;
    let mut archive = tar::Archive::new(decoder);
    archive.unpack(dest).map_err(|e| {
        format!(
            "failed to unpack chelis-std bundle into {}: {e}",
            dest.display()
        )
    })?;
    Ok(())
}

/// Decompress [`CHELIS_STD_ARCHIVE`] into memory: every regular file keyed
/// by its package-relative path (`reef.toml`, `src/io/json.ch`, ...).
///
/// This is the loader's view of the runtime (chelis#2616). Reading the
/// archive in memory gives the bundled package no filesystem location, so
/// nothing is written to disk, nothing can leak, and no path can enter a
/// cache. An entry that is not a regular file, or whose path is absolute or
/// climbs out of the package, is an error rather than a skip.
pub fn archive_files() -> Result<BTreeMap<PathBuf, Vec<u8>>, String> {
    let decoder = zstd::stream::read::Decoder::new(CHELIS_STD_ARCHIVE)
        .map_err(|e| format!("failed to start zstd decoder for chelis-std bundle: {e}"))?;
    let mut archive = tar::Archive::new(decoder);
    let mut files = BTreeMap::new();
    for entry in archive
        .entries()
        .map_err(|e| format!("failed to read chelis-std bundle entries: {e}"))?
    {
        let mut entry =
            entry.map_err(|e| format!("failed to read chelis-std bundle entry: {e}"))?;
        let path = entry
            .path()
            .map_err(|e| format!("chelis-std bundle entry has an unreadable path: {e}"))?
            .into_owned();
        if !entry.header().entry_type().is_file() {
            return Err(format!(
                "chelis-std bundle entry `{}` is not a regular file",
                path.display()
            ));
        }
        let mut relative = PathBuf::new();
        for component in path.components() {
            match component {
                Component::Normal(part) => relative.push(part),
                Component::CurDir => {}
                Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                    return Err(format!(
                        "chelis-std bundle entry `{}` leaves the package root",
                        path.display()
                    ));
                }
            }
        }
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes).map_err(|e| {
            format!(
                "failed to read chelis-std bundle entry `{}`: {e}",
                path.display()
            )
        })?;
        if files.insert(relative, bytes).is_some() {
            return Err(format!(
                "chelis-std bundle entry `{}` appears twice",
                path.display()
            ));
        }
    }
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_shell::{
        SHELL_FORMAT_VERSION, TypeVariableDomain, TypeVariableRestriction, decode_shell,
    };
    use tempfile::tempdir;

    /// The embedded shell bincode must agree with the embedded
    /// archive: same `(name, version)` pair, same `archive_sha256`.
    /// This is the same invariant `install_validated_artifact_pair`
    /// asserts at install time; we lock it here at compile-time-bundle
    /// granularity so a stale dist/ in a commit cannot land green.
    #[test]
    fn archive_self_consistency() {
        let shell = decode_shell(CHELIS_STD_SHELL).expect("decode embedded shell");
        assert_eq!(shell.package.name, "chelis-std");
        assert_eq!(shell.package.version, BUNDLED_CHELIS_STD_VERSION);
        assert_eq!(
            shell.archive_sha256,
            archive_sha256(),
            "embedded shell's archive_sha256 must match the embedded archive bytes: \
             rerun scripts/regenerate_chelis_std_bundle.py and commit the result"
        );
    }

    #[test]
    fn package_and_embedded_shells_preserve_std_test_active_float_scheme() {
        let package_shell = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../packages/chelis-std/dist")
            .join(format!("chelis-std-{BUNDLED_CHELIS_STD_VERSION}.chb"));
        let package_bytes = std::fs::read(&package_shell).expect("read package CHB");
        assert_eq!(
            package_bytes, CHELIS_STD_SHELL,
            "package and compile-time embedded CHB bytes must agree"
        );

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

    /// The extract path must produce a tree that `chelis-reef` can
    /// read as a normal reef package: a `reef.toml` at the root, a
    /// `src/` subdirectory with at least one `.ch` file.
    #[test]
    fn extract_yields_reef_package_layout() {
        let dir = tempdir().expect("tempdir");
        extract_into(dir.path()).expect("extract bundle");

        let reef_toml = dir.path().join("reef.toml");
        assert!(reef_toml.is_file(), "reef.toml missing after extract");
        let manifest = std::fs::read_to_string(&reef_toml).expect("read reef.toml");
        assert!(
            manifest.contains("name = \"chelis-std\""),
            "reef.toml must declare name = chelis-std; got: {manifest}"
        );

        // The embedded archive carries chelis-std's own `reef.toml`, and
        // its `compiler =` pin is read through `validate_manifest` when the
        // bundled runtime is loaded. `validate_manifest` rejects any pin
        // other than `=<current compiler version>`, so a stale embedded
        // pin makes every chelis-std-importing program fail at load with
        // "package.compiler must be `=X.Y.Z`". That is exactly the failure
        // mode that broke the 0.9.0 release: the workspace version and
        // `packages/chelis-std/reef.toml` were bumped, but these embedded
        // bytes were never regenerated, so they still pinned the prior
        // version. The bundle crate's `CARGO_PKG_VERSION` marches with the
        // workspace, so assert the embedded pin equals it; a mismatch means
        // `scripts/regenerate_chelis_std_bundle.py` was not rerun after the
        // bump.
        let expected_compiler_line = format!("compiler = \"={}\"", env!("CARGO_PKG_VERSION"));
        assert!(
            manifest.contains(&expected_compiler_line),
            "embedded chelis-std reef.toml must pin {expected_compiler_line:?}; \
             rerun scripts/regenerate_chelis_std_bundle.py and commit \
             crates/chelis-std-bundle/dist/. got:\n{manifest}"
        );

        let src = dir.path().join("src");
        assert!(src.is_dir(), "src/ missing after extract");
        let any_ch = std::fs::read_dir(&src)
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
        extract_into(dir.path()).expect("extract bundle");
        let mut on_disk = BTreeMap::new();
        let mut pending = vec![dir.path().to_path_buf()];
        while let Some(current) = pending.pop() {
            for entry in std::fs::read_dir(&current).expect("read extracted dir") {
                let path = entry.expect("read extracted entry").path();
                if path.is_dir() {
                    pending.push(path);
                } else {
                    let relative = path
                        .strip_prefix(dir.path())
                        .expect("relative")
                        .to_path_buf();
                    on_disk.insert(relative, std::fs::read(&path).expect("read extracted file"));
                }
            }
        }
        let in_memory = archive_files().expect("read archive in memory");
        assert!(in_memory.contains_key(Path::new("reef.toml")));
        assert_eq!(in_memory, on_disk);
    }

    #[test]
    fn version_constant_matches_archive_path() {
        // The path-template hardcoded in include_bytes!() encodes the
        // version. Keep this assertion as a reminder that bumping
        // the constant requires updating both lib.rs (here) and the
        // build.rs file-existence check.
        assert_eq!(BUNDLED_CHELIS_STD_VERSION, "0.4.0");
    }
}
