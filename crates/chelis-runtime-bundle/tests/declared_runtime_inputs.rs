//! The runtime's build record declares every input of its compilation
//! (`spec/08-backends.md` §2.1; `crates/chelis-runtime/build.rs` writes it).
//!
//! For each runtime configuration, Cargo builds `chelis-runtime` alone in a
//! dedicated target directory. Cargo's dep-info for that build names every file
//! rustc read for the runtime and its workspace dependencies, and every path
//! Cargo watches to rerun the runtime's build script. Each such input must be a
//! recorded file or a declared directory; each declared root must be watched,
//! so that changing it reruns the script; and the declared crates must be
//! exactly the workspace crates the build compiles. The script must also declare
//! every root relative to its package directory. A build-script execution cache
//! relocates only that directory, so an absolute path outside it would let the
//! cache serve another checkout's record.

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
fn declarations(output: &str) -> Vec<&str> {
    output
        .lines()
        .filter_map(|line| {
            line.strip_prefix("cargo::")
                .or_else(|| line.strip_prefix("cargo:"))
        })
        .filter_map(|directive| directive.strip_prefix("rerun-if-changed="))
        .collect()
}

/// `path` with `.` and `..` folded lexically. Cargo writes a declared input as
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

/// Everything that keeps `record` from describing one runtime build whose
/// dep-info names `inputs` and which compiled the workspace crate directories
/// `compiled`. `record_file` is the build script's output, which rustc reads.
fn findings(
    record: &Record,
    inputs: &[PathBuf],
    workspace: &Path,
    record_file: &Path,
    compiled: &BTreeSet<String>,
) -> Vec<String> {
    let mut findings = record
        .unavailable
        .iter()
        .map(|path| format!("the runtime was compiled without its declared root {path}"))
        .collect::<Vec<_>>();
    let mut watched = BTreeSet::new();
    for input in inputs {
        let input = lexical(input);
        if input == record_file {
            continue;
        }
        let Ok(relative) = input.strip_prefix(workspace) else {
            findings.push(format!("undeclared input {}", input.display()));
            continue;
        };
        let relative = relative
            .components()
            .map(|component| component.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        if record.files.contains(&relative) || record.directories.contains(&relative) {
            watched.insert(relative);
        } else {
            findings.push(format!("undeclared input {relative}"));
        }
    }
    let file_roots = record.files.iter().filter(|file| {
        !record
            .directories
            .iter()
            .any(|directory| file.starts_with(&format!("{directory}/")))
    });
    for root in record.directories.iter().chain(file_roots) {
        if !watched.contains(root) {
            findings.push(format!("declared root {root} is not watched by Cargo"));
        }
    }
    let declared = record
        .directories
        .iter()
        .filter_map(|directory| directory.strip_suffix("/src"))
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    for crate_directory in compiled.difference(&declared) {
        findings.push(format!(
            "workspace crate {crate_directory} is compiled but not declared"
        ));
    }
    for crate_directory in declared.difference(compiled) {
        findings.push(format!("declared crate {crate_directory} is not compiled"));
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
        let script_output = fs::read_to_string(&output_path)
            .unwrap_or_else(|error| panic!("{}: {error}", output_path.display()));
        let absolute = declarations(&script_output)
            .into_iter()
            .filter(|path| Path::new(path).is_absolute())
            .collect::<Vec<_>>();
        assert!(
            absolute.is_empty(),
            "{features:?}: declared inputs spelled as absolute paths: {absolute:#?}"
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

        let findings = findings(&record, &inputs, &workspace, &record_file, &compiled);
        assert!(findings.is_empty(), "{features:?}: {findings:#?}");
    }
}

/// A record, dep-info and crate set that agree, for the negative cases below.
fn agreeing() -> (Record, Vec<PathBuf>, PathBuf, PathBuf, BTreeSet<String>) {
    let workspace = PathBuf::from("/checkout");
    let record = parse_record(
        "dir crates/runtime/src\n\
         sha256 00 Cargo.lock\n\
         sha256 01 crates/runtime/Cargo.toml\n\
         sha256 02 crates/runtime/src/lib.rs\n",
    );
    let record_file = PathBuf::from("/checkout/target/debug/build/runtime-1/out/build_record.txt");
    // Cargo writes a watched root as the package directory joined with the
    // `../../<path>` the build script declared.
    let inputs = [
        "crates/runtime/../../Cargo.lock",
        "crates/runtime/../../crates/runtime/Cargo.toml",
        "crates/runtime/../../crates/runtime/src",
        "crates/runtime/src/lib.rs",
    ]
    .iter()
    .map(|path| workspace.join(path))
    .chain([record_file.clone()])
    .collect();
    let compiled = BTreeSet::from(["crates/runtime".to_owned()]);
    (record, inputs, workspace, record_file, compiled)
}

#[test]
fn an_agreeing_record_has_no_findings() {
    let (record, inputs, workspace, record_file, compiled) = agreeing();
    assert_eq!(
        findings(&record, &inputs, &workspace, &record_file, &compiled),
        Vec::<String>::new()
    );
}

#[test]
fn an_input_outside_the_declared_roots_is_reported() {
    let (record, mut inputs, workspace, record_file, compiled) = agreeing();
    inputs.push(workspace.join("crates/other/src/table.rs"));
    inputs.push(PathBuf::from("/etc/runtime-table"));
    assert_eq!(
        findings(&record, &inputs, &workspace, &record_file, &compiled),
        [
            "undeclared input crates/other/src/table.rs",
            "undeclared input /etc/runtime-table",
        ]
    );
}

#[test]
fn a_declared_root_cargo_does_not_watch_is_reported() {
    let (record, mut inputs, workspace, record_file, compiled) = agreeing();
    inputs.retain(|input| input != &workspace.join("crates/runtime/../../Cargo.lock"));
    assert_eq!(
        findings(&record, &inputs, &workspace, &record_file, &compiled),
        ["declared root Cargo.lock is not watched by Cargo"]
    );
}

#[test]
fn a_compiled_workspace_crate_outside_the_record_is_reported() {
    let (record, inputs, workspace, record_file, mut compiled) = agreeing();
    compiled.insert("crates/unord".to_owned());
    assert_eq!(
        findings(&record, &inputs, &workspace, &record_file, &compiled),
        ["workspace crate crates/unord is compiled but not declared"]
    );
}

#[test]
fn a_record_without_its_sources_is_reported() {
    let (_, inputs, workspace, record_file, compiled) = agreeing();
    let record = parse_record("unavailable crates/runtime/src\n");
    let findings = findings(&record, &inputs, &workspace, &record_file, &compiled);
    assert!(
        findings.contains(
            &"the runtime was compiled without its declared root crates/runtime/src".to_owned()
        ),
        "{findings:#?}"
    );
}
