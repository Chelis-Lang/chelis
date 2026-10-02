//! Select the chelis-std inputs the embedded runtime is packed from.
//!
//! Shared by the build script, which packs the staged copy, and by the
//! crate's tests, which check the selection. The runtime is a function of
//! `reef.toml`, every `.ch` file under the package's declared source roots,
//! and its declared metadata files. Nothing else in the source tree enters
//! the bundle, so editor and operating-system files (a `.DS_Store`, a swap
//! file) cannot change the bytes a binary embeds. A `.ch` file under a source
//! root is a module of the package whether or not it is committed, so it
//! enters the bundle as it enters the package.

use std::fs;
use std::path::{Path, PathBuf};

/// Every archive member of the embedded runtime is stamped with this mtime.
pub const ARCHIVE_MTIME: u64 = 0;

/// Pack a runtime tree that [`stage`] produced, as the embedded runtime is
/// packed.
pub fn pack(staged_root: &Path) -> Result<chelis_reef::PackedPackage, String> {
    chelis_reef::pack_runtime_package(staged_root, ARCHIVE_MTIME)
}

/// The package-relative paths whose change can change the runtime at
/// `std_root`: the manifest, each declared source root directory, and each
/// declared metadata file.
pub fn watched_paths(std_root: &Path) -> Result<Vec<PathBuf>, String> {
    let (roots, metadata) = declared_inputs(std_root)?;
    Ok(std::iter::once(PathBuf::from("reef.toml"))
        .chain(roots)
        .chain(metadata)
        .collect())
}

/// Every input of the runtime at `std_root`, as package-relative paths in
/// bytewise order.
pub fn runtime_inputs(std_root: &Path) -> Result<Vec<PathBuf>, String> {
    let (roots, metadata) = declared_inputs(std_root)?;
    let mut inputs = vec![PathBuf::from("reef.toml")];
    for root in &roots {
        collect_sources(std_root, root, &mut inputs)?;
    }
    inputs.extend(metadata);
    inputs.sort();
    inputs.dedup();
    Ok(inputs)
}

/// The declared source roots and metadata files of the package at
/// `std_root`.
fn declared_inputs(std_root: &Path) -> Result<(Vec<PathBuf>, Vec<PathBuf>), String> {
    let manifest = chelis_reef::read_manifest_for_src(std_root)?;
    let roots = std::iter::once("src")
        .chain(
            manifest
                .package
                .additional_sources
                .iter()
                .map(String::as_str),
        )
        .map(PathBuf::from)
        .collect();
    let metadata = &manifest.package.metadata;
    let metadata = [metadata.readme(), metadata.license_file()]
        .into_iter()
        .flatten()
        .map(|declared| PathBuf::from(declared.as_str()))
        .collect();
    Ok((roots, metadata))
}

/// Copy the runtime inputs at `std_root` into a fresh `dest` and return them
/// with their bytes, in the order [`runtime_inputs`] lists them.
pub fn stage(std_root: &Path, dest: &Path) -> Result<Vec<(PathBuf, Vec<u8>)>, String> {
    match fs::remove_dir_all(dest) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("failed to clear {}: {error}", dest.display())),
    }
    let mut staged = Vec::new();
    for relative in runtime_inputs(std_root)? {
        let source = std_root.join(&relative);
        let bytes =
            fs::read(&source).map_err(|e| format!("failed to read {}: {e}", source.display()))?;
        let target = dest.join(&relative);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
        }
        fs::write(&target, &bytes)
            .map_err(|e| format!("failed to write {}: {e}", target.display()))?;
        staged.push((relative, bytes));
    }
    Ok(staged)
}

fn collect_sources(
    std_root: &Path,
    relative: &Path,
    inputs: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let directory = std_root.join(relative);
    let entries = match fs::read_dir(&directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(format!("failed to read {}: {error}", directory.display())),
    };
    for entry in entries {
        let entry = entry.map_err(|e| format!("failed to read {}: {e}", directory.display()))?;
        let file_type = entry
            .file_type()
            .map_err(|e| format!("failed to stat {}: {e}", entry.path().display()))?;
        let child = relative.join(entry.file_name());
        if file_type.is_dir() {
            collect_sources(std_root, &child, inputs)?;
        } else if file_type.is_file()
            && child.extension().and_then(|ext| ext.to_str()) == Some("ch")
        {
            inputs.push(child);
        }
    }
    Ok(())
}
