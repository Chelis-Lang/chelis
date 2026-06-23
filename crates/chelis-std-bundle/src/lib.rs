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
//!   then commit the updated `dist/` files).
//!
//! ## Artifact regeneration
//!
//! 1. Edit `packages/chelis-std/src/**.ch` as needed.
//! 2. Run `python3 scripts/regenerate_chelis_std_bundle.py`. The script:
//!    - builds the chelis CLI in release mode;
//!    - runs `chelis reef build` against `packages/chelis-std/`;
//!    - copies the resulting `chelis-std-<version>.tar.zst` and `.chb`
//!      into `crates/chelis-std-bundle/dist/` (overwriting the
//!      committed bytes).
//! 3. `git add crates/chelis-std-bundle/dist/` and commit.
//! 4. Rebuilding any crate that depends on this one (`chelis-reef`,
//!    `chelis-cli`) picks up the new bytes.
//!
//! ## Loader integration
//!
//! [`extract_into`] decompresses the embedded archive into a caller-
//! supplied directory layout matching `load_registry_package`'s
//! expected on-disk shape: `<root>/reef.toml`, `<root>/src/...`. The
//! reef loader treats the result as if it had come from the local
//! registry's cache, with the bundled bytes substituting for the
//! filesystem-resident archive.

use sha2::{Digest, Sha256};
use std::path::Path;

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

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_shell::decode_shell;
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

    #[test]
    fn version_constant_matches_archive_path() {
        // The path-template hardcoded in include_bytes!() encodes the
        // version. Keep this assertion as a reminder that bumping
        // the constant requires updating both lib.rs (here) and the
        // build.rs file-existence check.
        assert_eq!(BUNDLED_CHELIS_STD_VERSION, "0.4.0");
    }
}
