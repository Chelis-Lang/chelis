//! Development source freshness (`spec/08-backends.md` §2.1).
//!
//! A development build carries the runtime its own compilation produced, and
//! that runtime's build record names the source files it was compiled from
//! (`chelis_runtime::build_record`). Before staging, the build compares the
//! record with the checkout it was compiled from, so it never stages a runtime
//! older than its sources. Sealed builds do not compile this module: they carry
//! no checkout path and read no checkout.

use crate::{RuntimeError, StaleSources, hex, io_error};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// The checkout this development build was compiled from: the workspace root
/// two levels above this crate's manifest directory, as Cargo named it.
pub(crate) fn checkout() -> Option<&'static Path> {
    Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2)
}

/// Check the carried runtime's build record against this build's checkout.
pub(crate) fn check_carried_sources() -> Result<(), RuntimeError> {
    let checkout = checkout().ok_or_else(|| RuntimeError::CheckoutUnavailable {
        checkout: PathBuf::from(env!("CARGO_MANIFEST_DIR")),
        source: io::Error::other("the manifest directory has no workspace root"),
    })?;
    check_sources(checkout, chelis_runtime::build_record::SOURCES)
}

/// A parsed build record, in the format `chelis_runtime::build_record::SOURCES`
/// documents.
struct Record<'a> {
    directories: Vec<&'a str>,
    files: BTreeMap<&'a str, &'a str>,
    unavailable: Vec<&'a str>,
}

fn parse(record: &str) -> Result<Record<'_>, RuntimeError> {
    let mut parsed = Record {
        directories: Vec::new(),
        files: BTreeMap::new(),
        unavailable: Vec::new(),
    };
    for line in record.lines() {
        let malformed = || RuntimeError::MalformedBuildRecord {
            line: line.to_owned(),
        };
        match line.split_once(' ') {
            Some(("dir", path)) if !path.is_empty() => parsed.directories.push(path),
            Some(("sha256", entry)) => match entry.split_once(' ') {
                Some((digest, path)) if !path.is_empty() => {
                    parsed.files.insert(path, digest);
                }
                _ => return Err(malformed()),
            },
            Some(("unavailable", path)) if !path.is_empty() => parsed.unavailable.push(path),
            _ => return Err(malformed()),
        }
    }
    Ok(parsed)
}

/// Compare `record` with the files below `checkout`: each recorded file must
/// hold the recorded bytes, and each declared directory must hold no other file
/// whose name does not begin with `.`.
pub(crate) fn check_sources(checkout: &Path, record: &str) -> Result<(), RuntimeError> {
    let record = parse(record)?;
    if !record.unavailable.is_empty() {
        return Err(RuntimeError::UnrecordedSources {
            missing: record
                .unavailable
                .iter()
                .map(|path| (*path).to_owned())
                .collect(),
        });
    }
    fs::read_dir(checkout).map_err(|source| RuntimeError::CheckoutUnavailable {
        checkout: checkout.to_path_buf(),
        source,
    })?;

    let mut stale = StaleSources {
        checkout: checkout.to_path_buf(),
        changed: Vec::new(),
        removed: Vec::new(),
        added: Vec::new(),
    };
    for (path, digest) in &record.files {
        let file = checkout.join(path);
        match fs::read(&file) {
            Ok(bytes) => {
                if hex(&Sha256::digest(&bytes).into()) != *digest {
                    stale.changed.push((*path).to_owned());
                }
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                stale.removed.push((*path).to_owned());
            }
            Err(error) => return Err(io_error("cannot read runtime source", &file)(error)),
        }
    }
    for directory in &record.directories {
        let mut present = Vec::new();
        list_files(checkout, directory, &mut present)?;
        stale.added.extend(
            present
                .into_iter()
                .filter(|path| !record.files.contains_key(path.as_str())),
        );
    }
    stale.added.sort();
    stale.added.dedup();

    if stale.changed.is_empty() && stale.removed.is_empty() && stale.added.is_empty() {
        Ok(())
    } else {
        Err(RuntimeError::StaleSources(stale))
    }
}

/// Append every file below `directory` whose name does not begin with `.`, as
/// the runtime's build script records them. An absent directory holds nothing;
/// its recorded files are reported as removed.
fn list_files(
    checkout: &Path,
    directory: &str,
    files: &mut Vec<String>,
) -> Result<(), RuntimeError> {
    let path = checkout.join(directory);
    let entries = match fs::read_dir(&path) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(io_error("cannot read runtime sources in", &path)(error)),
    };
    for entry in entries {
        let entry = entry.map_err(io_error("cannot read runtime sources in", &path))?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') {
            continue;
        }
        let relative = format!("{directory}/{name}");
        let metadata = fs::metadata(entry.path())
            .map_err(io_error("cannot read runtime source", &entry.path()))?;
        if metadata.is_dir() {
            list_files(checkout, &relative, files)?;
        } else {
            files.push(relative);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_runtime::build_record::SOURCES;
    use std::fs;

    /// The directory roots and recorded files of the build record of the runtime
    /// this test binary links, read independently of the checker's parser.
    fn declared() -> (Vec<String>, Vec<String>) {
        let mut directories = Vec::new();
        let mut files = Vec::new();
        for line in SOURCES.lines() {
            if let Some(path) = line.strip_prefix("dir ") {
                directories.push(path.to_owned());
            } else if let Some(entry) = line.strip_prefix("sha256 ") {
                let (_, path) = entry.split_once(' ').expect("a digest and a path");
                files.push(path.to_owned());
            } else {
                panic!("this test build's runtime recorded its sources, found `{line}`");
            }
        }
        (directories, files)
    }

    /// Every declared root with one recorded file it holds: a directory root and
    /// a file under it, or a file root and itself.
    fn roots() -> Vec<(String, String)> {
        let (directories, files) = declared();
        let mut roots = directories
            .iter()
            .map(|directory| {
                let prefix = format!("{directory}/");
                let file = files
                    .iter()
                    .find(|file| file.starts_with(&prefix))
                    .unwrap_or_else(|| panic!("the declared directory {directory} holds a file"));
                (directory.clone(), file.clone())
            })
            .collect::<Vec<_>>();
        for file in &files {
            if !directories
                .iter()
                .any(|directory| file.starts_with(&format!("{directory}/")))
            {
                roots.push((file.clone(), file.clone()));
            }
        }
        roots
    }

    /// A checkout holding exactly the files the record names, copied from the
    /// checkout this test build was compiled from.
    fn fixture() -> tempfile::TempDir {
        let source = checkout().expect("a development build records its checkout");
        let (directories, files) = declared();
        let copy = tempfile::tempdir().expect("tempdir");
        for directory in &directories {
            fs::create_dir_all(copy.path().join(directory)).expect("declared directory");
        }
        for file in &files {
            let destination = copy.path().join(file);
            fs::create_dir_all(destination.parent().expect("a parent directory"))
                .expect("parent directory");
            fs::copy(source.join(file), &destination).expect("recorded file");
        }
        copy
    }

    fn stale(result: Result<(), RuntimeError>) -> StaleSources {
        match result {
            Err(RuntimeError::StaleSources(stale)) => stale,
            other => panic!("expected stale sources, got {other:?}"),
        }
    }

    #[test]
    fn the_checkout_matches_the_record_this_runtime_was_built_from() {
        let checkout = checkout().expect("a development build records its checkout");
        check_sources(checkout, SOURCES).expect("the checkout is unchanged since this build");

        let (directories, files) = declared();
        for directory in [
            "crates/chelis-abi/src",
            "crates/chelis-runtime/include",
            "crates/chelis-runtime/src",
            "crates/chelis-vocab/src",
        ] {
            assert!(
                directories.iter().any(|declared| declared == directory),
                "{directory} is a declared root: {directories:?}"
            );
        }
        for file in [
            "Cargo.lock",
            "crates/chelis-abi/Cargo.toml",
            "crates/chelis-runtime/Cargo.toml",
            "crates/chelis-runtime/build.rs",
            "crates/chelis-runtime/src/lib.rs",
            "crates/chelis-vocab/Cargo.toml",
        ] {
            assert!(
                files.iter().any(|recorded| recorded == file),
                "{file} is recorded: {files:?}"
            );
        }
    }

    #[test]
    fn a_copy_of_the_recorded_files_is_fresh() {
        let copy = fixture();
        check_sources(copy.path(), SOURCES).expect("the copy holds exactly the recorded files");
    }

    #[test]
    fn an_edited_file_in_any_declared_root_fails_with_its_path() {
        for (root, file) in roots() {
            let copy = fixture();
            let path = copy.path().join(&file);
            let mut bytes = fs::read(&path).expect("recorded file");
            bytes.push(b'\n');
            fs::write(&path, bytes).expect("edit");

            let stale = stale(check_sources(copy.path(), SOURCES));
            assert_eq!(stale.checkout, copy.path(), "{root}");
            assert_eq!(stale.changed, std::slice::from_ref(&file), "{root}");
            assert!(
                stale.removed.is_empty() && stale.added.is_empty(),
                "{root}: {stale:?}"
            );
            let message = RuntimeError::StaleSources(stale).to_string();
            assert!(message.contains(&file), "{message}");
            assert!(message.contains("touch"), "{message}");
        }
    }

    #[test]
    fn a_removed_file_in_any_declared_root_fails_with_its_path() {
        for (root, file) in roots() {
            let copy = fixture();
            fs::remove_file(copy.path().join(&file)).expect("remove");

            let stale = stale(check_sources(copy.path(), SOURCES));
            assert_eq!(stale.removed, std::slice::from_ref(&file), "{root}");
            assert!(
                stale.changed.is_empty() && stale.added.is_empty(),
                "{root}: {stale:?}"
            );
            assert!(
                RuntimeError::StaleSources(stale)
                    .to_string()
                    .contains(&file),
                "{root}"
            );
        }
    }

    #[test]
    fn an_added_file_in_any_declared_directory_fails_with_its_path() {
        let (directories, _) = declared();
        for directory in directories {
            let copy = fixture();
            let added = [
                format!("{directory}/added_input.rs"),
                format!("{directory}/nested/added_input.h"),
            ];
            for file in &added {
                let path = copy.path().join(file);
                fs::create_dir_all(path.parent().expect("a parent directory")).expect("parent");
                fs::write(path, "// added after the build\n").expect("add");
            }

            let stale = stale(check_sources(copy.path(), SOURCES));
            assert_eq!(stale.added, added, "{directory}");
            assert!(
                stale.changed.is_empty() && stale.removed.is_empty(),
                "{directory}: {stale:?}"
            );
        }
    }

    #[test]
    fn hidden_names_in_a_declared_directory_are_not_inputs() {
        let copy = fixture();
        let source = copy.path().join("crates/chelis-runtime/src");
        fs::write(source.join(".DS_Store"), "finder").expect("hidden file");
        fs::create_dir_all(source.join(".cache")).expect("hidden directory");
        fs::write(source.join(".cache/lib.rs"), "editor state").expect("hidden tree");
        check_sources(copy.path(), SOURCES).expect("hidden names are not runtime inputs");
    }

    #[test]
    fn a_missing_development_checkout_fails() {
        let copy = fixture();
        let absent = copy.path().join("moved-away");
        match check_sources(&absent, SOURCES) {
            Err(error @ RuntimeError::CheckoutUnavailable { .. }) => {
                let message = error.to_string();
                assert!(message.contains(&absent.display().to_string()), "{message}");
                assert!(message.contains("sealed-runtime"), "{message}");
            }
            other => panic!("expected an unavailable checkout, got {other:?}"),
        }
    }

    #[test]
    fn a_runtime_compiled_without_its_sources_is_refused() {
        let copy = fixture();
        match check_sources(
            copy.path(),
            "unavailable Cargo.lock\nunavailable crates/chelis-abi/src\n",
        ) {
            Err(error @ RuntimeError::UnrecordedSources { .. }) => {
                let message = error.to_string();
                assert!(message.contains("crates/chelis-abi/src"), "{message}");
                assert!(message.contains("Cargo.lock"), "{message}");
            }
            other => panic!("expected unrecorded sources, got {other:?}"),
        }
    }

    #[test]
    fn a_record_line_of_unknown_kind_is_refused() {
        let copy = fixture();
        let error = check_sources(copy.path(), "file Cargo.lock\n").unwrap_err();
        assert!(
            matches!(error, RuntimeError::MalformedBuildRecord { .. }),
            "{error}"
        );
    }
}
