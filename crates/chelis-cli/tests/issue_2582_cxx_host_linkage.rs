//! chelis#2582: the host program is emitted once and compiled as C (`--target
//! c`) or as C++ (the Metal `.mm`, the HIP `.cpp`). A prototype of a symbol a
//! C translation unit defines takes C++ linkage in the C++ unit unless it is
//! declared `extern "C"`, and then the link against `libchelis_runtime.a`
//! fails on the mangled name.
//!
//! `PROGRAM` reaches every emitter-private runtime entry (the accumulator,
//! consuming container, and consuming string entries), so the link tests
//! fail on any one of them. The structural test derives its set from the
//! emitted source rather than from a list: every file-scope prototype that is
//! not one of the program's own exports (the generated header's declarations)
//! must carry the C-linkage prefix, in every target's host source.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
use tempfile::{TempDir, tempdir};

#[path = "common/mod.rs"]
mod common;

use common::write_file;

const PROGRAM: &str = "\
def case(flag: bool) -> List[string] = {
  names = map(fn (x: string) -> string_concat(x, \"!\"), [\"ab\", \"c\"])
  more = append(names, string_concat(\"x\", \"y\"))
  both = concat(more, [\"z\"])
  tail = skip(both, 1i64)
  flat_map(fn (s: string) -> [s, s], tail)
}
def table(flag: bool) -> Dict[string, i64] = {
  d = dict_insert(dict_of([(\"a\", 1i64)]), \"b\", 2i64)
  e = dict_merge(d, dict_of([(\"c\", 3i64)]))
  dict_remove(e, \"a\")
}
a = case(true)
b = table(true)
";

const EXPECTED: &str = "a = [c!, c!, xy, xy, z, z]\nb = dict(b: 2, c: 3)\n";

const STEM: &str = "linkage";

struct Built {
    _dir: TempDir,
    out_dir: PathBuf,
}

fn build(target: &str) -> Built {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{STEM}.ch"));
    let out_dir = dir.path().join(format!("{target}-out"));
    write_file(&path, PROGRAM);
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            target,
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    Built { _dir: dir, out_dir }
}

fn host_source(built: &Built, file: &str) -> String {
    fs::read_to_string(built.out_dir.join(file))
        .unwrap_or_else(|error| panic!("missing emitted `{file}`: {error}"))
}

/// Compile `source` as C++ with `compiler`, link it against the staged
/// runtime archive, run it, and return its stdout.
fn link_and_run_as_cxx(compiler: &str, out_dir: &Path, source: &str) -> String {
    let object = out_dir.join("host.o");
    let compiled = StdCommand::new(compiler)
        .current_dir(out_dir)
        .args(["-O1", "-c", source, "-o"])
        .arg(&object)
        .output()
        .unwrap_or_else(|error| panic!("`{compiler}` must run: {error}"));
    assert!(
        compiled.status.success(),
        "`{source}` does not compile as C++:\n{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: false,
            needs_blas: false,
        },
    );
    let binary = out_dir.join("host");
    let linked = StdCommand::new(compiler)
        .current_dir(out_dir)
        .arg(&object)
        .arg("libchelis_runtime.a")
        .args(&toolchain.link_flags)
        .arg("-o")
        .arg(&binary)
        .output()
        .unwrap_or_else(|error| panic!("`{compiler}` must run: {error}"));
    assert!(
        linked.status.success(),
        "`{source}` does not link against libchelis_runtime.a:\n{}",
        String::from_utf8_lossy(&linked.stderr)
    );
    let run = StdCommand::new(&binary).output().expect("linked host runs");
    assert!(
        run.status.success(),
        "the linked host program failed: {}\n{}",
        run.status,
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8(run.stdout).expect("utf-8 stdout")
}

/// Every file-scope prototype in `source` that the generated `header` does
/// not declare, paired with whether the three lines before it are the
/// `#ifdef __cplusplus` / `extern "C"` / `#endif` prefix.
fn foreign_prototypes(source: &str, header: &str) -> Vec<(String, bool)> {
    let exported = header.lines().map(str::trim).collect::<Vec<_>>();
    let lines = source.lines().collect::<Vec<_>>();
    lines
        .iter()
        .enumerate()
        .filter(|(_, line)| {
            let starts_at_file_scope = line
                .chars()
                .next()
                .is_some_and(|first| first.is_ascii_alphabetic() || first == '_');
            starts_at_file_scope
                && line.ends_with(");")
                && line.contains('(')
                && !line.contains('=')
                && !["static ", "typedef ", "extern ", "return "]
                    .iter()
                    .any(|prefix| line.starts_with(prefix))
                && !exported.contains(&line.trim())
        })
        .map(|(index, line)| {
            let prefixed = index >= 3
                && lines[index - 3] == "#ifdef __cplusplus"
                && lines[index - 2] == "extern \"C\""
                && lines[index - 1] == "#endif";
            (line.to_string(), prefixed)
        })
        .collect()
}

fn assert_every_foreign_prototype_has_c_linkage(target: &str, source_file: &str, header: &str) {
    let built = build(target);
    let source = host_source(&built, source_file);
    let header = host_source(&built, header);
    let prototypes = foreign_prototypes(&source, &header);
    // The derived set must see the declarations this program reaches, or
    // the assertion below would pass over an empty set.
    for entry in ["chelis_list_push_moved", "chelis_string_concat_owned"] {
        assert!(
            prototypes
                .iter()
                .any(|(line, _)| line.contains(&format!("{entry}("))),
            "the {target} host source must declare `{entry}`:\n{source}"
        );
    }
    let bare = prototypes
        .iter()
        .filter(|(_, prefixed)| !prefixed)
        .map(|(line, _)| line.as_str())
        .collect::<Vec<_>>();
    assert!(
        bare.is_empty(),
        "{target} host source declares symbols without C linkage:\n{}",
        bare.join("\n")
    );
}

#[test]
fn eval_prints_the_expected_roots() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{STEM}.ch"));
    write_file(&path, PROGRAM);
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("eval runs");
    assert!(output.status.success(), "{output:?}");
    assert_eq!(String::from_utf8(output.stdout).unwrap(), EXPECTED);
}

#[test]
fn every_foreign_prototype_in_the_hip_host_source_has_c_linkage() {
    assert_every_foreign_prototype_has_c_linkage(
        "hip",
        &format!("{STEM}_hip.cpp"),
        &format!("{STEM}_hip.h"),
    );
}

#[test]
fn every_foreign_prototype_in_the_metal_host_source_has_c_linkage() {
    assert_every_foreign_prototype_has_c_linkage(
        "metal",
        &format!("{STEM}_metal.mm"),
        &format!("{STEM}_metal.h"),
    );
}

/// The C lane emits the same declarations; the guard leaves C unchanged.
#[test]
fn every_foreign_prototype_in_the_c_host_source_has_c_linkage() {
    assert_every_foreign_prototype_has_c_linkage("c", &format!("{STEM}.c"), &format!("{STEM}.h"));
}

/// The HIP host translation unit of a program with no device helper includes
/// no HIP header, so any C++ compiler links it: this runs on every platform.
#[test]
fn the_hip_host_source_links_as_cxx_and_agrees_with_eval() {
    let built = build("hip");
    let compiler = std::env::var("CXX").unwrap_or_else(|_| "c++".to_string());
    let stdout = link_and_run_as_cxx(&compiler, &built.out_dir, &format!("{STEM}_hip.cpp"));
    assert_eq!(stdout, EXPECTED);
}

/// The Metal host is Objective-C++, compiled by Apple clang.
#[cfg(target_os = "macos")]
#[test]
fn the_metal_host_source_links_as_objective_cxx_and_agrees_with_eval() {
    let built = build("metal");
    let stdout = link_and_run_as_cxx("clang++", &built.out_dir, &format!("{STEM}_metal.mm"));
    assert_eq!(stdout, EXPECTED);
}
