//! `CHELIS_C_INCLUDE_PATH` and `CHELIS_C_LIBRARY_PATH` (spec/08-backends.md
//! §7): a C program finds a header and a shared library only through the
//! declared directories, under the cleared tool environment.
#![cfg(unix)]

use chelis_backend_c::toolchain::{
    C_INCLUDE_PATH_VARIABLE, C_LIBRARY_PATH_VARIABLE, CodegenRequirements, runtime_toolchain,
    tool_command, verify_compiler,
};
use std::path::Path;
use std::process::Command;

const CHILD: &str = "CHELIS_DECLARED_DIRECTORIES_CHILD";
const TEST: &str = "a_program_finds_a_header_and_a_library_only_through_the_declared_directories";

const HEADER: &str = "int chelis_declared_stub_value(void);\n";
const LIBRARY: &str = "int chelis_declared_stub_value(void) { return 42; }\n";
const PROGRAM: &str = "#include <chelis_declared_stub.h>\n\
#include <stdio.h>\n\
int main(void) { printf(\"%d\\n\", chelis_declared_stub_value()); return 0; }\n";

fn library_file_name() -> &'static str {
    if cfg!(target_os = "macos") {
        "libchelisdeclaredstub.dylib"
    } else {
        "libchelisdeclaredstub.so"
    }
}

/// Write the header into `<root>/include` and build the shared library into
/// `<root>/lib`. On macOS the library names itself `@rpath/...`, so the
/// program finds it only through its run-time search path, as on Linux.
fn write_stub(root: &Path, compiler: &str) {
    let include = root.join("include");
    let lib = root.join("lib");
    std::fs::create_dir_all(&include).unwrap();
    std::fs::create_dir_all(&lib).unwrap();
    std::fs::write(include.join("chelis_declared_stub.h"), HEADER).unwrap();
    let source = root.join("stub.c");
    std::fs::write(&source, LIBRARY).unwrap();
    let mut build = tool_command(compiler);
    build
        .args(["-shared", "-fPIC"])
        .arg(&source)
        .arg("-o")
        .arg(lib.join(library_file_name()));
    if cfg!(target_os = "macos") {
        build.arg(format!("-Wl,-install_name,@rpath/{}", library_file_name()));
    }
    let output = build.output().unwrap();
    assert!(
        output.status.success(),
        "stub library build failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Compile and link the program with the runtime toolchain flags. Returns
/// the built executable, or the compiler's diagnostics.
fn build_program(root: &Path) -> Result<std::path::PathBuf, String> {
    let toolchain = runtime_toolchain(CodegenRequirements::default())?;
    let source = root.join("main.c");
    std::fs::write(&source, PROGRAM).unwrap();
    let program = root.join("main");
    let output = tool_command(&toolchain.compiler)
        .args(&toolchain.compile_flags)
        .arg(&source)
        .args(&toolchain.link_flags)
        .arg("-lchelisdeclaredstub")
        .arg("-o")
        .arg(&program)
        .output()
        .unwrap();
    if output.status.success() {
        Ok(program)
    } else {
        Err(String::from_utf8_lossy(&output.stderr).into_owned())
    }
}

#[test]
fn a_program_finds_a_header_and_a_library_only_through_the_declared_directories() {
    if let Ok(mode) = std::env::var(CHILD) {
        let root = Path::new(&std::env::var("CHELIS_DECLARED_ROOT").unwrap()).to_path_buf();
        match mode.as_str() {
            "declared" => {
                let toolchain = runtime_toolchain(CodegenRequirements::default()).unwrap();
                // The compiler check and its canary accept these flags.
                verify_compiler(
                    &toolchain.compiler,
                    &toolchain.compile_flags,
                    &toolchain.link_flags,
                    &[],
                )
                .unwrap_or_else(|error| panic!("{error}"));
                let program = build_program(&root).unwrap_or_else(|error| panic!("{error}"));
                let ran = tool_command(&program).output().unwrap();
                assert!(
                    ran.status.success(),
                    "{}",
                    String::from_utf8_lossy(&ran.stderr)
                );
                assert_eq!(String::from_utf8_lossy(&ran.stdout), "42\n");
            }
            "undeclared" => {
                let error = build_program(&root).unwrap_err();
                assert!(error.contains("chelis_declared_stub.h"), "{error}");
            }
            "include-only" => {
                // The header resolves, and the link still misses the library.
                let error = build_program(&root).unwrap_err();
                assert!(!error.contains("chelis_declared_stub.h"), "{error}");
                assert!(error.contains("chelisdeclaredstub"), "{error}");
            }
            other => panic!("unknown child mode {other}"),
        }
        return;
    }

    let root = tempfile::tempdir().unwrap();
    let compiler = runtime_toolchain(CodegenRequirements::default())
        .unwrap()
        .compiler;
    write_stub(root.path(), &compiler);
    let include = root.path().join("include");
    let lib = root.path().join("lib");
    for (mode, variables) in [
        (
            "declared",
            vec![
                (C_INCLUDE_PATH_VARIABLE, include.as_os_str()),
                (C_LIBRARY_PATH_VARIABLE, lib.as_os_str()),
            ],
        ),
        ("undeclared", vec![]),
        (
            "include-only",
            vec![(C_INCLUDE_PATH_VARIABLE, include.as_os_str())],
        ),
    ] {
        // The ambient variables a C driver reads are present, and the tool
        // environment clears them, so only the declared directories count.
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", TEST])
            .env(CHILD, mode)
            .env("CHELIS_DECLARED_ROOT", root.path())
            .env_remove(C_INCLUDE_PATH_VARIABLE)
            .env_remove(C_LIBRARY_PATH_VARIABLE)
            .env("CPATH", &include)
            .env("C_INCLUDE_PATH", &include)
            .env("LIBRARY_PATH", &lib)
            .env("LD_LIBRARY_PATH", &lib)
            .env("DYLD_LIBRARY_PATH", &lib)
            .envs(variables)
            .status()
            .unwrap();
        assert!(status.success(), "the {mode} child failed");
    }
}
