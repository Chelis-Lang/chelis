//! The runtime's build record declares every input of its compilation
//! (`spec/08-backends.md` §2.1; `crates/chelis-runtime/build.rs` writes it).
//!
//! For each runtime configuration, Cargo builds `chelis-runtime` alone in a
//! dedicated target directory. Cargo's dep-info for that build names every file
//! rustc read for the runtime and its workspace dependencies, and every path
//! Cargo watches to rerun the runtime's build script. Each such input must be a
//! recorded file, a declared directory or a watched path. Each declared root
//! must lie under a watched path, so that changing it reruns the script. The
//! declared crates must be exactly the workspace crates the build compiles, and
//! the record must name the lockfile and each of their manifests, which rustc
//! never reads. The script must also spell every watched path relative to its
//! package directory: a build-script execution cache relocates only that
//! directory, so an absolute path outside it would let the cache serve another
//! checkout's record.

use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

/// A runtime build record: its declared directories, its recorded files, and
/// the required roots its compilation could not find.
#[derive(Debug, Default)]
struct Record {
    directories: BTreeSet<String>,
    files: BTreeSet<String>,
    unavailable: Vec<String>,
}

fn parse_record(text: &str) -> Record {
    let mut record = Record::default();
    for line in text.lines() {
        if let Some(path) = line.strip_prefix("dir ") {
            record.directories.insert(path.to_owned());
        } else if let Some(entry) = line.strip_prefix("sha256 ") {
            let (_, path) = entry.split_once(' ').expect("a digest and a path");
            record.files.insert(path.to_owned());
        } else if let Some(path) = line.strip_prefix("unavailable ") {
            record.unavailable.push(path.to_owned());
        } else {
            panic!("unknown build record line `{line}`");
        }
    }
    record
}

/// The prerequisites of every rule in a Makefile-style dep-info file, with
/// escaped spaces restored.
fn dep_info_inputs(text: &str) -> Vec<PathBuf> {
    let mut inputs = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') {
            continue;
        }
        let Some((_, prerequisites)) = line.split_once(": ") else {
            continue;
        };
        let mut current = String::new();
        let mut characters = prerequisites.chars();
        while let Some(character) = characters.next() {
            match character {
                '\\' => current.extend(characters.next()),
                ' ' => {
                    if !current.is_empty() {
                        inputs.push(PathBuf::from(std::mem::take(&mut current)));
                    }
                }
                other => current.push(other),
            }
        }
        if !current.is_empty() {
            inputs.push(PathBuf::from(current));
        }
    }
    inputs
}

/// The `rerun-if-changed` paths in a build script's recorded output, as the
/// script spelled them.
fn watched_paths(output: &str) -> Vec<String> {
    output
        .lines()
        .filter_map(|line| {
            line.strip_prefix("cargo::")
                .or_else(|| line.strip_prefix("cargo:"))
        })
        .filter_map(|directive| directive.strip_prefix("rerun-if-changed="))
        .map(str::to_owned)
        .collect()
}

/// `path` with `.` and `..` folded lexically. Cargo writes a watched path as
/// the package directory joined with the path the script declared.
fn lexical(path: &Path) -> PathBuf {
    let mut folded = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                folded.pop();
            }
            other => folded.push(other),
        }
    }
    folded
}

/// One runtime build, as Cargo reported it.
struct Build {
    record: Record,
    /// The prerequisites Cargo's dep-info names, as Cargo wrote them.
    inputs: Vec<PathBuf>,
    /// The paths the build script told Cargo to watch, as the script spelled them.
    watched: Vec<String>,
    /// The runtime's package directory, against which Cargo resolves `watched`.
    package: PathBuf,
    workspace: PathBuf,
    /// The build record, which rustc reads from the build script's `OUT_DIR`.
    record_file: PathBuf,
    /// The workspace crate directories the build compiled.
    compiled: BTreeSet<String>,
}

impl Build {
    /// `path`, folded, relative to the workspace root, when it lies under it.
    fn relative(&self, path: &Path) -> Option<String> {
        let relative = lexical(path)
            .strip_prefix(&self.workspace)
            .ok()?
            .components()
            .map(|component| component.as_os_str().to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join("/");
        Some(relative)
    }
}

/// Everything that keeps `build`'s record from declaring its compilation.
fn findings(build: &Build) -> Vec<String> {
    let record = &build.record;
    let mut findings = record
        .unavailable
        .iter()
        .map(|path| format!("the runtime was compiled without its declared root {path}"))
        .collect::<Vec<_>>();

    let mut watched = BTreeSet::new();
    for path in &build.watched {
        if Path::new(path).is_absolute() {
            findings.push(format!(
                "watched path {path} is absolute, so a build-script cache can serve another \
                 checkout's record"
            ));
        }
        match build.relative(&build.package.join(path)) {
            Some(relative) => {
                watched.insert(relative);
            }
            None => findings.push(format!("watched path {path} is outside the workspace")),
        }
    }

    for input in &build.inputs {
        if lexical(input) == build.record_file {
            continue;
        }
        let Some(relative) = build.relative(input) else {
            findings.push(format!("undeclared input {}", input.display()));
            continue;
        };
        if !(record.files.contains(&relative)
            || record.directories.contains(&relative)
            || watched.contains(&relative))
        {
            findings.push(format!("undeclared input {relative}"));
        }
    }

    let under = |path: &str, directory: &str| {
        path == directory || path.starts_with(&format!("{directory}/"))
    };
    let file_roots = record.files.iter().filter(|file| {
        !record
            .directories
            .iter()
            .any(|directory| under(file, directory))
    });
    for root in record.directories.iter().chain(file_roots) {
        if !watched.iter().any(|path| under(root, path)) {
            findings.push(format!("declared root {root} is not watched by Cargo"));
        }
    }

    let declared = record
        .directories
        .iter()
        .filter_map(|directory| directory.strip_suffix("/src"))
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    for crate_directory in build.compiled.difference(&declared) {
        findings.push(format!(
            "workspace crate {crate_directory} is compiled but not declared"
        ));
    }
    for crate_directory in declared.difference(&build.compiled) {
        findings.push(format!("declared crate {crate_directory} is not compiled"));
    }

    // Rustc reads no manifest, and Cargo only watches the lockfile, so no
    // dep-info input requires them; the record must still name each one.
    let manifests = build
        .compiled
        .iter()
        .map(|crate_directory| format!("{crate_directory}/Cargo.toml"));
    for path in std::iter::once("Cargo.lock".to_owned()).chain(manifests) {
        if !record.files.contains(&path) {
            findings.push(format!("{path} is not recorded"));
        }
    }
    findings
}

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the bundle lies two levels below the workspace root")
        .to_path_buf()
}

#[test]
fn every_input_of_each_runtime_configuration_is_declared_and_watched() {
    let workspace = workspace();
    let target = workspace.join("target/runtime-declared-inputs");
    for features in [&[][..], &["--features", "ownership-ledger"][..]] {
        let output = Command::new(env!("CARGO"))
            .current_dir(&workspace)
            .env("CARGO_TARGET_DIR", &target)
            .args([
                "build",
                "--locked",
                "-p",
                "chelis-runtime",
                "--lib",
                "--message-format=json",
            ])
            .args(features)
            .output()
            .expect("run cargo");
        assert!(
            output.status.success(),
            "{features:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let rows = String::from_utf8(output.stdout)
            .expect("UTF-8 cargo messages")
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .collect::<Vec<_>>();

        let runtime = rows
            .iter()
            .filter(|row| {
                row["reason"] == "compiler-artifact" && row["target"]["name"] == "chelis_runtime"
            })
            .collect::<Vec<_>>();
        assert_eq!(runtime.len(), 1, "{features:?}: one runtime artifact");
        let rlibs = runtime[0]["filenames"]
            .as_array()
            .expect("artifact file names")
            .iter()
            .filter_map(Value::as_str)
            .filter(|path| path.ends_with(".rlib"))
            .collect::<Vec<_>>();
        assert_eq!(rlibs.len(), 1, "{features:?}: one runtime library");
        let dep_info = Path::new(rlibs[0]).with_extension("d");
        let inputs = dep_info_inputs(
            &fs::read_to_string(&dep_info)
                .unwrap_or_else(|error| panic!("{}: {error}", dep_info.display())),
        );

        let out_dirs = rows
            .iter()
            .filter(|row| {
                row["reason"] == "build-script-executed"
                    && row["package_id"] == runtime[0]["package_id"]
            })
            .filter_map(|row| row["out_dir"].as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            out_dirs.len(),
            1,
            "{features:?}: one runtime build script run"
        );
        let record_file = Path::new(out_dirs[0]).join("build_record.txt");
        let record = parse_record(&fs::read_to_string(&record_file).expect("build record"));
        let output_path = Path::new(out_dirs[0])
            .parent()
            .expect("a build script run directory")
            .join("output");
        let watched = watched_paths(
            &fs::read_to_string(&output_path)
                .unwrap_or_else(|error| panic!("{}: {error}", output_path.display())),
        );

        let compiled = rows
            .iter()
            .filter(|row| row["reason"] == "compiler-artifact")
            .filter_map(|row| row["manifest_path"].as_str())
            .filter_map(|manifest| {
                Path::new(manifest)
                    .parent()?
                    .strip_prefix(&workspace)
                    .ok()
                    .map(|directory| directory.to_string_lossy().into_owned())
            })
            .collect::<BTreeSet<_>>();

        let build = Build {
            record,
            inputs,
            watched,
            package: workspace.join("crates/chelis-runtime"),
            workspace: workspace.clone(),
            record_file,
            compiled,
        };
        let findings = findings(&build);
        assert!(findings.is_empty(), "{features:?}: {findings:#?}");
    }
}

/// A build whose record, dep-info, watched paths and compiled crates agree,
/// for the negative cases below.
fn agreeing() -> Build {
    let workspace = PathBuf::from("/checkout");
    let package = workspace.join("crates/runtime");
    let record_file = workspace.join("target/debug/build/runtime-1/out/build_record.txt");
    let inputs = vec![
        package.join("../../Cargo.lock"),
        package.join("../../crates/runtime"),
        workspace.join("crates/runtime/src/lib.rs"),
        record_file.clone(),
    ];
    Build {
        record: parse_record(
            "dir crates/runtime/src\n\
             sha256 00 Cargo.lock\n\
             sha256 01 crates/runtime/Cargo.toml\n\
             sha256 02 crates/runtime/src/lib.rs\n",
        ),
        inputs,
        watched: vec![
            "../../Cargo.lock".to_owned(),
            "../../crates/runtime".to_owned(),
        ],
        package,
        workspace,
        record_file,
        compiled: BTreeSet::from(["crates/runtime".to_owned()]),
    }
}

#[test]
fn an_agreeing_build_has_no_findings() {
    assert_eq!(findings(&agreeing()), Vec::<String>::new());
}

#[test]
fn an_input_outside_the_declared_roots_is_reported() {
    let mut build = agreeing();
    build
        .inputs
        .push(build.workspace.join("crates/other/src/table.rs"));
    build.inputs.push(PathBuf::from("/etc/runtime-table"));
    assert_eq!(
        findings(&build),
        [
            "undeclared input crates/other/src/table.rs",
            "undeclared input /etc/runtime-table",
        ]
    );
}

#[test]
fn a_declared_root_cargo_does_not_watch_is_reported() {
    let mut build = agreeing();
    build.watched.retain(|path| path != "../../Cargo.lock");
    assert_eq!(
        findings(&build),
        ["declared root Cargo.lock is not watched by Cargo"]
    );
}

#[test]
fn an_absolute_watched_path_is_reported() {
    let mut build = agreeing();
    build.watched = vec![
        "/checkout/Cargo.lock".to_owned(),
        "../../crates/runtime".to_owned(),
    ];
    assert_eq!(
        findings(&build),
        [
            "watched path /checkout/Cargo.lock is absolute, so a build-script cache can serve \
             another checkout's record"
        ]
    );
}

#[test]
fn a_compiled_workspace_crate_outside_the_record_is_reported() {
    let mut build = agreeing();
    build.compiled.insert("crates/unord".to_owned());
    assert_eq!(
        findings(&build),
        [
            "workspace crate crates/unord is compiled but not declared",
            "crates/unord/Cargo.toml is not recorded",
        ]
    );
}

#[test]
fn a_record_without_its_sources_is_reported() {
    let mut build = agreeing();
    build.record = parse_record("unavailable crates/runtime/src\n");
    let findings = findings(&build);
    assert!(
        findings.contains(
            &"the runtime was compiled without its declared root crates/runtime/src".to_owned()
        ),
        "{findings:#?}"
    );
}

#[test]
fn an_unrecorded_lockfile_or_manifest_is_reported() {
    let mut build = agreeing();
    build.record = parse_record(
        "dir crates/runtime/src\n\
         sha256 02 crates/runtime/src/lib.rs\n",
    );
    assert_eq!(
        findings(&build),
        [
            "Cargo.lock is not recorded",
            "crates/runtime/Cargo.toml is not recorded",
        ]
    );
}
