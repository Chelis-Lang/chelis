//! The runtime a Chelis compiler carries (`spec/08-backends.md` §2.1).
//!
//! The CLI and the Python extension depend on this crate. It embeds the runtime
//! static archive produced by the `chelis-runtime` compilation that the same
//! build links, and stages that archive with the public headers of the same
//! compilation. Nothing here searches for a runtime: a runtime archive left in a
//! build directory by another configuration or commit is never read.
//!
//! A development build carries the path of the checkout it was compiled from
//! and, before staging, checks the declared sources of its runtime against that
//! checkout. A build with the `sealed` feature carries no build path and reads
//! no checkout.

#[cfg(not(feature = "sealed"))]
mod freshness;

use chelis_runtime_bundle_macro::runtime_archive;
use sha2::{Digest, Sha256};
use std::ffi::OsString;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

pub use chelis_runtime::public_headers::PUBLIC_HEADERS;

/// File name of the staged runtime archive.
pub const ARCHIVE_FILE_NAME: &str = "libchelis_runtime.a";
/// File name of the receipt that records what staging wrote.
pub const RECEIPT_FILE_NAME: &str = "chelis_runtime.receipt.json";
/// The runtime location variable that staging rejects instead of honoring.
pub const RUNTIME_DIR_VARIABLE: &str = "CHELIS_RUNTIME_DIR";
/// Schema identifier of the staging receipt.
pub const RECEIPT_SCHEMA: &str = "chelis-runtime-staging/1";
/// Build mode recorded in staging receipts.
pub const MODE: &str = if cfg!(feature = "sealed") {
    "sealed"
} else {
    "development"
};

/// The carried runtime archive and its SHA-256, or `None` when this crate was
/// compiled without linkable output (such a compilation produces no executable).
static CARRIED: Option<(&[u8], [u8; 32])> = runtime_archive!(chelis_runtime);

/// Why the carried runtime could not be staged.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error(
        "this chelis build carries no runtime archive: it was compiled without linkable \
         output. Build chelis with `cargo build`"
    )]
    Placeholder,
    #[error(
        "{RUNTIME_DIR_VARIABLE} is set ({value}), but chelis stages the runtime built into \
         it and never takes one from a directory. Unset {RUNTIME_DIR_VARIABLE}"
    )]
    RuntimeDirSet { value: String },
    #[error("the runtime archive at {path} has SHA-256 {actual}; chelis carries {expected}")]
    Mismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[error("{action} {path}: {source}")]
    Io {
        action: &'static str,
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("{0}")]
    StaleSources(StaleSources),
    #[error(
        "this development build of chelis checks its runtime against the checkout it was \
         built from, {}, which cannot be read: {source}. Rebuild chelis in a Chelis checkout, \
         or build a distribution with the `sealed-runtime` feature",
        .checkout.display()
    )]
    CheckoutUnavailable {
        checkout: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error(
        "this development build of chelis carries a runtime compiled without its declared \
         sources ({}), so it cannot check them against a checkout. Build chelis with Cargo \
         in a Chelis checkout, or build a distribution with the `sealed-runtime` feature",
        .missing.join(", ")
    )]
    UnrecordedSources { missing: Vec<String> },
    #[error("the carried runtime's build record has a malformed line `{line}`")]
    MalformedBuildRecord { line: String },
}

/// Declared runtime sources in a development checkout that differ from the
/// build record of the runtime the build carries, by path relative to the
/// checkout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaleSources {
    /// The checkout the build was compiled from.
    pub checkout: PathBuf,
    /// Recorded files whose bytes changed.
    pub changed: Vec<String>,
    /// Recorded files that no longer exist.
    pub removed: Vec<String>,
    /// Files in a declared directory that the record does not name.
    pub added: Vec<String>,
}

impl fmt::Display for StaleSources {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            formatter,
            "the runtime this development build of chelis carries is older than its sources \
             in {}:",
            self.checkout.display()
        )?;
        for (kind, paths) in [
            ("changed", &self.changed),
            ("removed", &self.removed),
            ("added", &self.added),
        ] {
            for path in paths {
                writeln!(formatter, "  {kind}: {path}")?;
            }
        }
        write!(
            formatter,
            "Rebuild chelis (`cargo build` for the CLI, `maturin develop` for the Python \
             extension). If Cargo reports nothing to rebuild, touch the changed and added files \
             and rebuild again"
        )
    }
}

/// The files one successful staging published.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagedRuntime {
    /// The staged archive, named [`ARCHIVE_FILE_NAME`].
    pub archive: PathBuf,
    /// Lowercase hexadecimal SHA-256 of the staged archive.
    pub archive_sha256: String,
    /// The staging receipt, named [`RECEIPT_FILE_NAME`].
    pub receipt: PathBuf,
}

/// Run the checks [`stage`] makes before it writes anything, so a caller can
/// fail before other work: a set [`RUNTIME_DIR_VARIABLE`] fails, and a
/// development build fails when the declared sources of the runtime it carries
/// changed in its checkout after the build (`spec/08-backends.md` §2.1). A
/// sealed build reads no checkout.
pub fn preflight() -> Result<(), RuntimeError> {
    reject_runtime_dir()?;
    #[cfg(not(feature = "sealed"))]
    freshness::check_carried_sources()?;
    Ok(())
}

/// Fail when [`RUNTIME_DIR_VARIABLE`] is set in this process's environment.
fn reject_runtime_dir() -> Result<(), RuntimeError> {
    reject_runtime_dir_value(std::env::var_os(RUNTIME_DIR_VARIABLE))
}

fn reject_runtime_dir_value(value: Option<OsString>) -> Result<(), RuntimeError> {
    match value {
        Some(value) => Err(RuntimeError::RuntimeDirSet {
            value: value.to_string_lossy().into_owned(),
        }),
        None => Ok(()),
    }
}

/// Lowercase hexadecimal SHA-256 of the carried runtime archive.
pub fn carried_sha256() -> Result<String, RuntimeError> {
    CARRIED
        .map(|(_, digest)| hex(&digest))
        .ok_or(RuntimeError::Placeholder)
}

/// Stage the carried runtime archive and the public runtime headers into `dir`.
///
/// [`preflight`] runs first. Each file is written beside its final name and
/// renamed into place; the archive is read back and verified against the
/// carried digest first. The receipt is written last, so a failed attempt
/// leaves no receipt behind.
pub fn stage(dir: &Path) -> Result<StagedRuntime, RuntimeError> {
    preflight()?;
    stage_carried(dir, CARRIED)
}

fn stage_carried(
    dir: &Path,
    carried: Option<(&[u8], [u8; 32])>,
) -> Result<StagedRuntime, RuntimeError> {
    let (archive, digest) = carried.ok_or(RuntimeError::Placeholder)?;
    let receipt = dir.join(RECEIPT_FILE_NAME);
    match fs::remove_file(&receipt) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(io_error("cannot remove the previous receipt", &receipt)(
                error,
            ));
        }
    }

    let mut headers = serde_json::Map::new();
    for (name, contents) in PUBLIC_HEADERS {
        publish(dir, name, contents.as_bytes(), None)?;
        headers.insert(
            (*name).to_owned(),
            hex(&Sha256::digest(contents.as_bytes()).into()).into(),
        );
    }
    let archive_path = publish(dir, ARCHIVE_FILE_NAME, archive, Some(&digest))?;
    let archive_sha256 = hex(&digest);
    let record = serde_json::json!({
        "schema": RECEIPT_SCHEMA,
        "archive": ARCHIVE_FILE_NAME,
        "archive_sha256": archive_sha256,
        "headers": headers,
        "mode": MODE,
        "chelis_version": env!("CARGO_PKG_VERSION"),
    });
    publish(
        dir,
        RECEIPT_FILE_NAME,
        format!("{record:#}\n").as_bytes(),
        None,
    )?;
    Ok(StagedRuntime {
        archive: archive_path,
        archive_sha256,
        receipt,
    })
}

/// Write `bytes` beside `dir/name`, verify them against `digest` when given,
/// and rename them into place.
fn publish(
    dir: &Path,
    name: &str,
    bytes: &[u8],
    digest: Option<&[u8; 32]>,
) -> Result<PathBuf, RuntimeError> {
    let destination = dir.join(name);
    let partial = dir.join(format!(".{name}.{}.partial", std::process::id()));
    let result = write_verified(&partial, bytes, digest, &destination).and_then(|()| {
        fs::rename(&partial, &destination).map_err(io_error("cannot publish", &destination))
    });
    if result.is_err() {
        // The attempt already failed; the error above is what the caller needs.
        let _ = fs::remove_file(&partial);
    }
    result.map(|()| destination)
}

fn write_verified(
    partial: &Path,
    bytes: &[u8],
    digest: Option<&[u8; 32]>,
    destination: &Path,
) -> Result<(), RuntimeError> {
    fs::write(partial, bytes).map_err(io_error("cannot write", partial))?;
    let Some(expected) = digest else {
        return Ok(());
    };
    let written = fs::read(partial).map_err(io_error("cannot read back", partial))?;
    let actual: [u8; 32] = Sha256::digest(&written).into();
    if &actual == expected {
        Ok(())
    } else {
        Err(RuntimeError::Mismatch {
            path: destination.to_path_buf(),
            expected: hex(expected),
            actual: hex(&actual),
        })
    }
}

fn io_error(action: &'static str, path: &Path) -> impl FnOnce(io::Error) -> RuntimeError {
    let path = path.to_path_buf();
    move |source| RuntimeError::Io {
        action,
        path,
        source,
    }
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest_of(bytes: &[u8]) -> [u8; 32] {
        Sha256::digest(bytes).into()
    }

    fn entries(dir: &Path) -> Vec<String> {
        let mut names = fs::read_dir(dir)
            .expect("staging directory")
            .map(|entry| {
                entry
                    .expect("entry")
                    .file_name()
                    .into_string()
                    .expect("UTF-8")
            })
            .collect::<Vec<_>>();
        names.sort();
        names
    }

    #[test]
    fn stages_the_carried_archive_headers_and_receipt() {
        let dir = tempfile::tempdir().expect("tempdir");
        let archive: &[u8] = b"!<arch>\ncarried runtime";
        let staged = stage_carried(dir.path(), Some((archive, digest_of(archive))))
            .expect("staging succeeds");

        assert_eq!(fs::read(&staged.archive).expect("archive"), archive);
        assert_eq!(staged.archive_sha256, hex(&digest_of(archive)));
        for (name, contents) in PUBLIC_HEADERS {
            assert_eq!(
                fs::read_to_string(dir.path().join(name)).expect("header"),
                *contents
            );
        }
        let receipt: serde_json::Value =
            serde_json::from_slice(&fs::read(&staged.receipt).expect("receipt")).expect("JSON");
        assert_eq!(receipt["schema"], RECEIPT_SCHEMA);
        assert_eq!(receipt["archive_sha256"], staged.archive_sha256.as_str());
        assert_eq!(receipt["mode"], MODE);
        assert_eq!(
            receipt["headers"]
                .as_object()
                .expect("header digests")
                .len(),
            PUBLIC_HEADERS.len()
        );
        assert!(
            entries(dir.path())
                .iter()
                .all(|name| !name.ends_with(".partial")),
            "no partial file remains"
        );
    }

    #[test]
    fn a_placeholder_runtime_stages_nothing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let error = stage_carried(dir.path(), None).unwrap_err();
        assert!(matches!(error, RuntimeError::Placeholder), "{error}");
        assert!(entries(dir.path()).is_empty());
    }

    #[test]
    fn bytes_that_do_not_match_the_carried_digest_are_not_published() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(dir.path().join(RECEIPT_FILE_NAME), "{}").expect("stale receipt");
        let wrong: &[u8] = b"runtime";
        let error = stage_carried(dir.path(), Some((wrong, [0; 32]))).unwrap_err();
        assert!(matches!(error, RuntimeError::Mismatch { .. }), "{error}");
        let names = entries(dir.path());
        assert!(!names.contains(&ARCHIVE_FILE_NAME.to_owned()), "{names:?}");
        assert!(!names.contains(&RECEIPT_FILE_NAME.to_owned()), "{names:?}");
        assert!(
            names.iter().all(|name| !name.ends_with(".partial")),
            "{names:?}"
        );
    }

    #[test]
    fn a_read_only_archive_from_an_earlier_staging_is_replaced() {
        let dir = tempfile::tempdir().expect("tempdir");
        let old = dir.path().join(ARCHIVE_FILE_NAME);
        fs::write(&old, b"old").expect("old archive");
        let mut permissions = fs::metadata(&old).expect("metadata").permissions();
        permissions.set_readonly(true);
        fs::set_permissions(&old, permissions).expect("read-only");

        let archive: &[u8] = b"!<arch>\nnew runtime";
        stage_carried(dir.path(), Some((archive, digest_of(archive)))).expect("staging succeeds");
        assert_eq!(fs::read(&old).expect("archive"), archive);
    }

    #[test]
    fn a_set_runtime_dir_is_rejected_even_when_empty() {
        assert!(reject_runtime_dir_value(None).is_ok());
        for value in ["/opt/chelis/lib", ""] {
            let error = reject_runtime_dir_value(Some(value.into())).unwrap_err();
            assert!(
                matches!(error, RuntimeError::RuntimeDirSet { .. }),
                "{error}"
            );
            assert!(
                error.to_string().contains("Unset CHELIS_RUNTIME_DIR"),
                "{error}"
            );
        }
    }
}
