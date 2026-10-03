//! The chelis-std runtime a binary embeds.
//!
//! chelis-std is the language runtime: every package depends on it, the way
//! a Rust crate depends on `core`, and it version-marches with the compiler.
//! A binary carries one runtime as the archive and shell `chelis reef build`
//! produces for `packages/chelis-std`. The `chelis-std-bundle` crate packs
//! that pair while the binary is compiled and exposes it as an
//! [`EmbeddedRuntime`]; every graph entry point that can reach the runtime
//! takes one as an explicit parameter. Reef never depends on the bundle, so a
//! library that forgets to pass the runtime fails to compile rather than at
//! run time.
//!
//! The runtime has no filesystem location (chelis#2616). Its modules are
//! parsed from the archive bytes in memory and its source digests are
//! computed from them, so a prepared graph carries no per-process path and
//! nothing is written to disk.

use crate::{
    CHELIS_STD_PACKAGE_NAME, LoadedPackage, LoadedSourceKind, ModuleSource, PackageSourceFile,
    ParsedManifest, ReefManifest, SourceDigest, declared_source_roots, inventory_path_hex,
    modules_from_source_files, package_versioning, parse_manifest_contents, sha256_bytes,
    shell_package_sha256,
};
use chelis_shell::{PackageId, ShellPackage};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::OnceLock;

/// Package-relative label the bundled runtime's diagnostics name in place
/// of a filesystem path.
pub(crate) const BUNDLED_RUNTIME_LABEL: &str = "<bundled chelis-std>";

/// The chelis-std runtime embedded in a binary: its version, source archive,
/// and shell. Decoded views are computed once per value on first use.
pub struct EmbeddedRuntime {
    version: &'static str,
    archive: &'static [u8],
    shell: &'static [u8],
    archive_sha256: OnceLock<String>,
    shell_sha256: OnceLock<String>,
    files: OnceLock<Result<BTreeMap<PathBuf, Vec<u8>>, String>>,
    loaded: OnceLock<Result<LoadedRuntime, String>>,
}

/// The runtime's package sources, parsed once.
struct LoadedRuntime {
    manifest: ReefManifest,
    resolver: package_versioning::ResolverVersion,
    modules: BTreeMap<String, ModuleSource>,
    shell: ShellPackage,
    shell_sha256: String,
}

impl fmt::Debug for EmbeddedRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EmbeddedRuntime")
            .field("version", &self.version)
            .field("archive_len", &self.archive.len())
            .field("shell_len", &self.shell.len())
            .finish_non_exhaustive()
    }
}

impl EmbeddedRuntime {
    /// The runtime `chelis-std` `version` whose packed source archive is
    /// `archive` and whose shell is `shell`. Nothing is validated here; the
    /// first load checks that the manifest, the shell, and `version` agree.
    pub const fn new(version: &'static str, archive: &'static [u8], shell: &'static [u8]) -> Self {
        Self {
            version,
            archive,
            shell,
            archive_sha256: OnceLock::new(),
            shell_sha256: OnceLock::new(),
            files: OnceLock::new(),
            loaded: OnceLock::new(),
        }
    }

    /// The chelis-std version this runtime provides.
    pub fn version(&self) -> &'static str {
        self.version
    }

    /// The zstd-compressed tar archive of the runtime sources.
    pub fn archive(&self) -> &'static [u8] {
        self.archive
    }

    /// The runtime's canonical shell encoding.
    pub fn shell(&self) -> &'static [u8] {
        self.shell
    }

    /// Lowercase hex SHA-256 of [`Self::archive`], the hash a lock records
    /// for the runtime.
    pub fn archive_sha256(&self) -> &str {
        self.archive_sha256
            .get_or_init(|| sha256_bytes(self.archive))
    }

    /// Lowercase hex SHA-256 of [`Self::shell`], the hash a lock records
    /// for the runtime.
    pub fn shell_sha256(&self) -> &str {
        self.shell_sha256.get_or_init(|| sha256_bytes(self.shell))
    }

    /// Whether `(name, version)` names this runtime.
    pub fn provides(&self, name: &str, version: &str) -> bool {
        name == CHELIS_STD_PACKAGE_NAME && version == self.version
    }

    /// Every regular file of the archive, keyed by its package-relative path
    /// (`reef.toml`, `src/io/json.ch`, ...), decompressed in memory once.
    ///
    /// An entry that is not a regular file, or whose path is absolute or
    /// climbs out of the package, is an error rather than a skip.
    pub fn archive_files(&self) -> Result<&BTreeMap<PathBuf, Vec<u8>>, String> {
        self.files
            .get_or_init(|| archive_files(self.archive))
            .as_ref()
            .map_err(Clone::clone)
    }

    /// Write the archive's files under `dest`, creating it when absent, for a
    /// caller that needs the runtime on disk.
    pub fn extract_into(&self, dest: &Path) -> Result<(), String> {
        std::fs::create_dir_all(dest).map_err(|e| {
            format!(
                "failed to create chelis-std bundle dest {}: {e}",
                dest.display()
            )
        })?;
        let decoder = zstd::stream::read::Decoder::new(self.archive)
            .map_err(|e| format!("failed to start zstd decoder for chelis-std bundle: {e}"))?;
        tar::Archive::new(decoder).unpack(dest).map_err(|e| {
            format!(
                "failed to unpack chelis-std bundle into {}: {e}",
                dest.display()
            )
        })
    }

    fn loaded(&self) -> Result<&LoadedRuntime, String> {
        self.loaded
            .get_or_init(|| self.load())
            .as_ref()
            .map_err(Clone::clone)
    }

    fn load(&self) -> Result<LoadedRuntime, String> {
        let files = self.archive_files()?;
        let parsed = runtime_manifest(files)?;
        let resolver = parsed.typed.resolver;
        let manifest = parsed.into_raw();
        if !self.provides(&manifest.package.name, &manifest.package.version) {
            return Err(format!(
                "the bundled runtime manifest names `{}` `{}`, expected `{CHELIS_STD_PACKAGE_NAME}` `{}`",
                manifest.package.name, manifest.package.version, self.version
            ));
        }
        let mut sources = Vec::new();
        for declared_root in declared_source_roots(&manifest) {
            for (path, bytes) in files {
                let Some(rel) = runtime_source_rel(path, declared_root) else {
                    continue;
                };
                let display = format!("{BUNDLED_RUNTIME_LABEL}/{}", path.display());
                let text = String::from_utf8(bytes.clone())
                    .map_err(|error| format!("{display} is not UTF-8: {error}"))?;
                sources.push(PackageSourceFile {
                    source_root: declared_root.to_string(),
                    rel: rel.to_path_buf(),
                    display,
                    text,
                });
            }
        }
        let modules = modules_from_source_files(BUNDLED_RUNTIME_LABEL, &manifest, sources)?;
        let shell = chelis_shell::decode_shell(self.shell)
            .map_err(|e| format!("failed to decode embedded chelis-std shell: {e}"))?;
        let expected = PackageId {
            name: CHELIS_STD_PACKAGE_NAME.to_string(),
            version: self.version.to_string(),
        };
        if shell.package != expected {
            return Err(format!(
                "the embedded chelis-std shell names `{}` `{}`, expected `{CHELIS_STD_PACKAGE_NAME}` `{}`",
                shell.package.name, shell.package.version, self.version
            ));
        }
        if shell.archive_sha256 != self.archive_sha256() {
            return Err(format!(
                "the embedded chelis-std shell names archive {}, but the embedded archive hashes to {}",
                shell.archive_sha256,
                self.archive_sha256()
            ));
        }
        let shell_sha256 = shell_package_sha256(&shell)?;
        Ok(LoadedRuntime {
            manifest,
            resolver,
            modules,
            shell,
            shell_sha256,
        })
    }

    /// The runtime manifest as reef parses it.
    pub(crate) fn manifest(&self) -> Result<ParsedManifest, String> {
        runtime_manifest(self.archive_files()?)
    }

    /// The runtime as a graph package. Its source is
    /// [`LoadedSourceKind::Bundled`], which carries no filesystem location.
    pub(crate) fn package(&self) -> Result<LoadedPackage, String> {
        let runtime = self.loaded()?;
        Ok(LoadedPackage {
            id: PackageId {
                name: runtime.manifest.package.name.clone(),
                version: runtime.manifest.package.version.clone(),
            },
            manifest: runtime.manifest.clone(),
            resolver: runtime.resolver,
            modules: runtime.modules.clone(),
            source: LoadedSourceKind::Bundled,
            archive_sha256: Some(runtime.shell.archive_sha256.clone()),
            shell_sha256: Some(runtime.shell_sha256.clone()),
            shell: Some(runtime.shell.clone()),
            remote_origin: None,
        })
    }

    /// Digest rows for the runtime `package` of a graph, identical in shape
    /// to the rows `PreparedReefGraph::push_filesystem_source_digests`
    /// produces for an extracted copy of the same archive: the manifest, then
    /// every `.ch` file under a declared source root keyed by its
    /// package-relative path. The bytes come from the embedded archive, never
    /// from disk, so the graph must have been loaded from this runtime.
    pub(crate) fn push_source_digests(
        &self,
        package: &LoadedPackage,
        digests: &mut Vec<SourceDigest>,
    ) -> Result<(), String> {
        if package.archive_sha256.as_deref() != Some(self.archive_sha256()) {
            return Err(format!(
                "the graph's bundled `{}` `{}` records archive {}, but this binary embeds `{CHELIS_STD_PACKAGE_NAME}` `{}` with archive {}",
                package.id.name,
                package.id.version,
                package.archive_sha256.as_deref().unwrap_or("<none>"),
                self.version,
                self.archive_sha256()
            ));
        }
        let files = self.archive_files()?;
        let manifest = files
            .get(Path::new("reef.toml"))
            .ok_or_else(|| format!("{BUNDLED_RUNTIME_LABEL} has no reef.toml"))?;
        digests.push(SourceDigest {
            package_name: package.id.name.clone(),
            package_version: package.id.version.clone(),
            module_name: "<manifest::reef.toml>".to_string(),
            sha256: Sha256::digest(manifest).into(),
        });
        for declared_root in declared_source_roots(&package.manifest) {
            for (path, bytes) in files {
                if runtime_source_rel(path, declared_root).is_none() {
                    continue;
                }
                digests.push(SourceDigest {
                    package_name: package.id.name.clone(),
                    package_version: package.id.version.clone(),
                    module_name: format!("<source::{}>", inventory_path_hex(path)),
                    sha256: Sha256::digest(bytes).into(),
                });
            }
        }
        Ok(())
    }
}

fn runtime_manifest(files: &BTreeMap<PathBuf, Vec<u8>>) -> Result<ParsedManifest, String> {
    let bytes = files
        .get(Path::new("reef.toml"))
        .ok_or_else(|| format!("{BUNDLED_RUNTIME_LABEL} has no reef.toml"))?;
    let text = std::str::from_utf8(bytes)
        .map_err(|error| format!("{BUNDLED_RUNTIME_LABEL}/reef.toml is not UTF-8: {error}"))?;
    parse_manifest_contents(
        Path::new(&format!("{BUNDLED_RUNTIME_LABEL}/reef.toml")),
        text,
    )
}

/// The path of a bundled `.ch` file relative to `declared_root`, or `None`
/// when the archive entry is not a source file under that root.
fn runtime_source_rel<'a>(path: &'a Path, declared_root: &str) -> Option<&'a Path> {
    if path.extension().and_then(|ext| ext.to_str()) != Some("ch") {
        return None;
    }
    path.strip_prefix(declared_root).ok()
}

fn archive_files(archive: &[u8]) -> Result<BTreeMap<PathBuf, Vec<u8>>, String> {
    let decoder = zstd::stream::read::Decoder::new(archive)
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
