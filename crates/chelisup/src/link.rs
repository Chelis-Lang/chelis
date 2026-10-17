//! `chelisup link <dir>`: register an external toolchain directory.
//!
//! The directory must contain an executable `bin/chelis`. Its
//! `--version` output names the version, and `<home>/toolchains/<ver>`
//! becomes a symbolic link to the canonical directory. The shim then runs
//! it like an installed toolchain. A real directory that `install`
//! placed is never replaced. The design is
//! `spec/design/chelis_packaging_and_install.md` §5.1.1.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::install::ensure_default_seeded;
use crate::paths::Store;
use crate::version::validate_install_version;

/// The result of a successful [`link`].
#[derive(Debug)]
pub struct LinkOutcome {
    /// The version that `bin/chelis --version` reported.
    pub version: String,
    /// The canonical directory that the store entry now names.
    pub target: PathBuf,
    /// The target of the link that this one replaced, if any.
    pub replaced: Option<PathBuf>,
}

/// Register `dir` as a toolchain. On any error the store is unchanged.
///
/// `pub(crate)`: chelisup owns the toolchain lifecycle. Other crates
/// drive it through the `chelisup` binary, never in-process.
pub(crate) fn link(store: &Store, dir: &Path) -> Result<LinkOutcome, String> {
    let target = dir
        .canonicalize()
        .map_err(|e| format!("cannot link {}: {e}", dir.display()))?;
    if !target.is_dir() {
        return Err(format!(
            "cannot link {}: it is not a directory",
            target.display()
        ));
    }
    // A directory in the chelis home holds the shim or another toolchain.
    // A link to it makes the shim run itself, or ties one entry to another.
    if let Ok(home) = store.home().canonicalize()
        && target.starts_with(&home)
    {
        return Err(format!(
            "cannot link {}: it is inside the chelis home {}",
            target.display(),
            home.display()
        ));
    }
    let binary = target.join("bin").join("chelis");
    if !binary.is_file() {
        return Err(format!(
            "cannot link {}: it has no bin/chelis",
            target.display()
        ));
    }
    if let (Ok(real), Ok(shim)) = (binary.canonicalize(), store.shim_path().canonicalize())
        && real == shim
    {
        return Err(format!(
            "cannot link {}: its bin/chelis is the chelis shim {}",
            target.display(),
            shim.display()
        ));
    }
    let version = reported_version(&binary)?;

    let entry = store.toolchain_dir(&version);
    let replaced = store.linked_target(&version);
    if replaced.is_none() && fs::symlink_metadata(&entry).is_ok() {
        return Err(format!(
            "chelis {version} is already installed at {}; \
             remove it first with: chelisup uninstall {version}",
            entry.display()
        ));
    }

    let toolchains = store.toolchains_dir();
    fs::create_dir_all(&toolchains)
        .map_err(|e| format!("could not create {}: {e}", toolchains.display()))?;
    // Make the new link under a scratch name in the home, outside
    // `toolchains/`, then rename it over the entry. This replaces an
    // existing link in one step.
    let scratch = store
        .home()
        .join(format!(".chelisup-link-{version}-{}", std::process::id()));
    let _ = fs::remove_file(&scratch);
    make_symlink(&target, &scratch)?;
    if let Err(e) = fs::rename(&scratch, &entry) {
        let _ = fs::remove_file(&scratch);
        return Err(format!("could not create {}: {e}", entry.display()));
    }
    ensure_default_seeded(store, &version)?;
    Ok(LinkOutcome {
        version,
        target,
        replaced,
    })
}

/// Run `binary --version` and accept only the exact output
/// `chelis X.Y.Z` followed by an optional newline.
fn reported_version(binary: &Path) -> Result<String, String> {
    let output = Command::new(binary)
        .arg("--version")
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("could not run {} --version: {e}", binary.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} --version failed ({}): {}",
            binary.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.strip_suffix('\n').unwrap_or(&stdout);
    match line.strip_prefix("chelis ") {
        Some(version) if validate_install_version(version).is_ok() => Ok(version.to_string()),
        _ => Err(format!(
            "{} --version printed {:?}; expected exactly `chelis X.Y.Z`",
            binary.display(),
            stdout
        )),
    }
}

#[cfg(unix)]
fn make_symlink(target: &Path, link: &Path) -> Result<(), String> {
    std::os::unix::fs::symlink(target, link)
        .map_err(|e| format!("could not create the link {}: {e}", link.display()))
}

#[cfg(not(unix))]
fn make_symlink(_target: &Path, _link: &Path) -> Result<(), String> {
    Err("chelisup link needs a unix host".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_directory_leaves_the_store_unchanged() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::at(tmp.path().join("home"));
        let err = link(&store, &tmp.path().join("absent")).unwrap_err();
        assert!(err.contains("absent"), "{err}");
        assert!(!store.home().exists());
    }
}
