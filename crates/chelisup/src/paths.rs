//! The chelisup store: path computation rooted at `$CHELIS_HOME`
//! (default `~/.chelis`).
//!
//! All path math is pure and lives behind [`Store`] so tests construct a
//! store over a tempdir with [`Store::at`] and never touch global env.
//! The single env-reading entry point is [`Store::from_env`]; its
//! precedence logic is factored into [`resolve_home`] so it can be
//! table-tested without mutating the process environment.

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

/// A chelisup store rooted at one directory.
#[derive(Debug, Clone)]
pub struct Store {
    home: PathBuf,
}

/// Pure home-root precedence: `$CHELIS_HOME` if set and non-empty, else
/// `$HOME/.chelis`. Errors only when neither is available.
pub fn resolve_home(chelis_home: Option<&OsStr>, home: Option<&OsStr>) -> Result<PathBuf, String> {
    if let Some(h) = chelis_home
        && !h.is_empty()
    {
        return Ok(PathBuf::from(h));
    }
    if let Some(h) = home
        && !h.is_empty()
    {
        return Ok(Path::new(h).join(".chelis"));
    }
    Err("could not determine the chelis home directory: \
         neither $CHELIS_HOME nor $HOME is set"
        .to_string())
}

impl Store {
    /// Build a store from the real environment.
    pub fn from_env() -> Result<Self, String> {
        let chelis_home = std::env::var_os("CHELIS_HOME");
        let home = std::env::var_os("HOME");
        let root = resolve_home(chelis_home.as_deref(), home.as_deref())?;
        Ok(Self { home: root })
    }

    /// Build a store rooted at an explicit directory. The test seam.
    pub fn at(home: impl Into<PathBuf>) -> Self {
        Self { home: home.into() }
    }

    /// The store root (`$CHELIS_HOME` or `~/.chelis`).
    pub fn home(&self) -> &Path {
        &self.home
    }

    /// `<home>/toolchains`.
    pub fn toolchains_dir(&self) -> PathBuf {
        self.home.join("toolchains")
    }

    /// `<home>/toolchains/<ver>`.
    pub fn toolchain_dir(&self, version: &str) -> PathBuf {
        self.toolchains_dir().join(version)
    }

    /// `<home>/toolchains/<ver>/bin/chelis`, the real compiler binary.
    pub fn toolchain_bin(&self, version: &str) -> PathBuf {
        self.toolchain_dir(version).join("bin").join("chelis")
    }

    /// `<home>/bin`, where the shim and installer copies live.
    pub fn bin_dir(&self) -> PathBuf {
        self.home.join("bin")
    }

    /// `<home>/bin/chelis`, the pin-resolving shim.
    pub fn shim_path(&self) -> PathBuf {
        self.bin_dir().join("chelis")
    }

    /// `<home>/bin/chelisup`, the installer copy.
    pub fn chelisup_path(&self) -> PathBuf {
        self.bin_dir().join("chelisup")
    }

    /// `<home>/nix-gcroots/chelisup`, the GC root for a Nix-packaged installer.
    pub fn nix_gc_root(&self) -> PathBuf {
        self.home.join("nix-gcroots").join("chelisup")
    }

    /// The temporary GC root that protects a new Nix installer during install.
    pub fn nix_gc_staging_root(&self) -> PathBuf {
        self.home.join("nix-gcroots").join("chelisup.next")
    }

    /// The GC root that protects a binary from an incomplete Nix install.
    pub fn nix_gc_partial_root(&self) -> PathBuf {
        self.home.join("nix-gcroots").join("chelisup.partial")
    }

    /// `<home>/default`, the recorded default-version file.
    pub fn default_file(&self) -> PathBuf {
        self.home.join("default")
    }

    /// The recorded default version, or `None` if unset/empty/unreadable.
    pub fn read_default(&self) -> Option<String> {
        let raw = fs::read_to_string(self.default_file()).ok()?;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    }

    /// Record `version` as the default. Caller validates installed-ness.
    pub fn write_default(&self, version: &str) -> Result<(), String> {
        fs::create_dir_all(&self.home)
            .map_err(|e| format!("could not create {}: {e}", self.home.display()))?;
        fs::write(self.default_file(), format!("{version}\n")).map_err(|e| {
            format!(
                "could not write default file {}: {e}",
                self.default_file().display()
            )
        })
    }

    /// True iff a toolchain with a real `bin/chelis` is installed.
    pub fn is_installed(&self, version: &str) -> bool {
        self.toolchain_bin(version).is_file()
    }

    /// Sorted list of installed toolchain versions (those with a real
    /// `bin/chelis`). Empty when none are installed.
    pub fn installed_versions(&self) -> Vec<String> {
        let mut out: Vec<String> = match fs::read_dir(self.toolchains_dir()) {
            Ok(rd) => rd
                .filter_map(|e| e.ok())
                .filter_map(|e| {
                    let name = e.file_name().to_string_lossy().into_owned();
                    if self.is_installed(&name) {
                        Some(name)
                    } else {
                        None
                    }
                })
                .collect(),
            Err(_) => Vec::new(),
        };
        out.sort();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    #[test]
    fn home_prefers_chelis_home() {
        let got =
            resolve_home(Some(OsStr::new("/tmp/store")), Some(OsStr::new("/home/u"))).unwrap();
        assert_eq!(got, PathBuf::from("/tmp/store"));
    }

    #[test]
    fn home_falls_back_to_home_dot_chelis() {
        let got = resolve_home(None, Some(OsStr::new("/home/u"))).unwrap();
        assert_eq!(got, PathBuf::from("/home/u/.chelis"));
    }

    #[test]
    fn home_empty_chelis_home_is_ignored() {
        let empty = OsString::new();
        let got = resolve_home(Some(empty.as_os_str()), Some(OsStr::new("/home/u"))).unwrap();
        assert_eq!(got, PathBuf::from("/home/u/.chelis"));
    }

    #[test]
    fn home_errors_without_either() {
        assert!(resolve_home(None, None).is_err());
    }

    #[test]
    fn store_paths_honor_root() {
        let s = Store::at("/r");
        assert_eq!(
            s.toolchain_bin("0.12.0"),
            PathBuf::from("/r/toolchains/0.12.0/bin/chelis")
        );
        assert_eq!(s.shim_path(), PathBuf::from("/r/bin/chelis"));
        assert_eq!(s.chelisup_path(), PathBuf::from("/r/bin/chelisup"));
        assert_eq!(s.nix_gc_root(), PathBuf::from("/r/nix-gcroots/chelisup"));
        assert_eq!(
            s.nix_gc_staging_root(),
            PathBuf::from("/r/nix-gcroots/chelisup.next")
        );
        assert_eq!(
            s.nix_gc_partial_root(),
            PathBuf::from("/r/nix-gcroots/chelisup.partial")
        );
        assert_eq!(s.default_file(), PathBuf::from("/r/default"));
    }

    #[test]
    fn read_default_round_trips_and_trims() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Store::at(tmp.path());
        assert_eq!(s.read_default(), None);
        s.write_default("0.12.0").unwrap();
        assert_eq!(s.read_default().as_deref(), Some("0.12.0"));
    }

    #[test]
    fn installed_versions_lists_only_real_toolchains() {
        let tmp = tempfile::tempdir().unwrap();
        let s = Store::at(tmp.path());
        // 0.1.0 has a real bin; 0.2.0 is an empty dir (a half-removed
        // install) and must not be reported as installed.
        fs::create_dir_all(s.toolchain_dir("0.1.0").join("bin")).unwrap();
        fs::write(s.toolchain_bin("0.1.0"), b"x").unwrap();
        fs::create_dir_all(s.toolchain_dir("0.2.0")).unwrap();
        assert_eq!(s.installed_versions(), vec!["0.1.0".to_string()]);
        assert!(s.is_installed("0.1.0"));
        assert!(!s.is_installed("0.2.0"));
    }
}
