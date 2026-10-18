//! spec/08-backends.md §7: `chelis build` adds the C directories that
//! `CHELIS_C_INCLUDE_PATH` and `CHELIS_C_LIBRARY_PATH` declare to the C
//! compile and link, and refuses a malformed declaration.

#[path = "common/mod.rs"]
mod common;
use assert_cmd::Command;
use common::{gcc_available, write_file};
use std::path::Path;
use tempfile::tempdir;

/// A program whose generated C includes `chelis_blas.h`, so the build links
/// the target's BLAS provider: `-lopenblas` on Linux.
const BLAS_PROGRAM: &str = include_str!("../../../examples/recurrence_extent_scope.ch");
const BLAS_PROGRAM_OUTPUT: &str = "main.0 = 40.0\nmain.1 = Model([5.0, 5.0])\n";

/// Run `chelis build` for `source` with the two variables, or with neither
/// when `declared` is `None`.
fn build(
    dir: &Path,
    source: &str,
    declared: Option<(&str, &str)>,
    emit_c: bool,
) -> std::process::Output {
    let path = dir.join("program.ch");
    write_file(&path, source);
    let out_dir = dir.join("program-out");
    let mut command = Command::cargo_bin("chelis").unwrap();
    command
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env_remove("CHELIS_C_INCLUDE_PATH")
        .env_remove("CHELIS_C_LIBRARY_PATH")
        .args(["build", path.to_str().unwrap(), "--target", "c", "--output"])
        .arg(&out_dir);
    if emit_c {
        command.arg("--emit-c");
    }
    if let Some((include, library)) = declared {
        command
            .env("CHELIS_C_INCLUDE_PATH", include)
            .env("CHELIS_C_LIBRARY_PATH", library);
    }
    command.output().unwrap()
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[test]
fn the_compile_command_carries_the_declared_directories() {
    let dir = tempdir().unwrap();
    let include = dir.path().join("include");
    let lib = dir.path().join("lib");
    let declared = (include.to_str().unwrap(), lib.to_str().unwrap());
    let output = build(dir.path(), BLAS_PROGRAM, Some(declared), true);
    assert!(output.status.success(), "{}", text(&output.stderr));
    let stdout = text(&output.stdout);
    let compile = stdout
        .lines()
        .find(|line| line.starts_with("Compile:"))
        .unwrap_or_else(|| panic!("no Compile line: {stdout}"));
    let isystem = format!("-isystem{}", include.display());
    let search = format!("-L{}", lib.display());
    let rpath = format!("-Wl,-rpath,{}", lib.display());
    for flag in [&isystem, &search, &rpath] {
        assert!(compile.contains(flag.as_str()), "{flag} missing: {compile}");
    }
    let generated = std::fs::read_to_string(dir.path().join("program-out/program.c")).unwrap();
    assert!(generated.contains("#include \"chelis_blas.h\""));
    if cfg!(target_os = "linux") {
        let blas = compile
            .find("-lopenblas")
            .expect("the BLAS program links OpenBLAS");
        assert!(compile.find(&search).unwrap() < blas, "{compile}");
    }

    let undeclared = build(dir.path(), BLAS_PROGRAM, None, true);
    assert!(undeclared.status.success(), "{}", text(&undeclared.stderr));
    let undeclared = text(&undeclared.stdout);
    assert!(!undeclared.contains("-isystem"), "{undeclared}");
    assert!(!undeclared.contains("-Wl,-rpath"), "{undeclared}");
}

#[test]
fn a_build_with_declared_directories_passes_the_compiler_check_and_runs() {
    if !gcc_available() {
        eprintln!("skipping: no C compiler");
        return;
    }
    let dir = tempdir().unwrap();
    let include = dir.path().join("include");
    let lib = dir.path().join("lib");
    std::fs::create_dir_all(&include).unwrap();
    std::fs::create_dir_all(&lib).unwrap();
    let declared = (include.to_str().unwrap(), lib.to_str().unwrap());
    let output = build(dir.path(), BLAS_PROGRAM, Some(declared), false);
    assert!(output.status.success(), "{}", text(&output.stderr));
    let ran = std::process::Command::new(dir.path().join("program-out/program"))
        .output()
        .unwrap();
    assert!(ran.status.success(), "{}", text(&ran.stderr));
    assert_eq!(text(&ran.stdout), BLAS_PROGRAM_OUTPUT);
}

#[test]
fn a_relative_or_comma_declared_entry_fails_the_build() {
    for (include, library, variable, entry) in [
        (
            "include",
            "/abs/lib",
            "CHELIS_C_INCLUDE_PATH",
            "\"include\"",
        ),
        ("/abs/include:../x", "", "CHELIS_C_INCLUDE_PATH", "\"../x\""),
        ("/abs/include", "lib", "CHELIS_C_LIBRARY_PATH", "\"lib\""),
        (
            "/abs/include",
            "/abs/a,b/lib",
            "CHELIS_C_LIBRARY_PATH",
            "comma",
        ),
    ] {
        for emit_c in [true, false] {
            let dir = tempdir().unwrap();
            let output = build(dir.path(), BLAS_PROGRAM, Some((include, library)), emit_c);
            assert!(
                !output.status.success(),
                "{include:?} {library:?} built (emit_c {emit_c})"
            );
            let stderr = text(&output.stderr);
            assert!(
                stderr.contains(variable) && stderr.contains(entry),
                "{include:?} {library:?}: {stderr}"
            );
            assert!(!dir.path().join("program-out/program").exists());
        }
    }
}

/// A wrapper that prefixes a directory to an exported empty value leaves a
/// trailing empty entry. That entry declares nothing.
#[test]
fn an_empty_declared_entry_declares_nothing() {
    let dir = tempdir().unwrap();
    let output = build(dir.path(), BLAS_PROGRAM, Some(("/abs/include:", ":")), true);
    assert!(output.status.success(), "{}", text(&output.stderr));
    let stdout = text(&output.stdout);
    let compile = stdout
        .lines()
        .find(|line| line.starts_with("Compile:"))
        .unwrap_or_else(|| panic!("no Compile line: {stdout}"));
    assert!(compile.contains("-isystem/abs/include"), "{compile}");
    assert!(!compile.contains(" -L"), "{compile}");
    assert!(!compile.contains("-Wl,-rpath"), "{compile}");
}
