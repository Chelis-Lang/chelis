//! chelis#2107: native callers consume the generated declaration contract.

mod common;

use assert_cmd::Command;
use chelis_backend_c::GeneratedHeader;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as NativeCommand;
use tempfile::{TempDir, tempdir};

const QUALIFIED_MAIN: &str = "pkg__demo__Demo__Main__main";
const QUALIFIED_SUFFIX_LOOKALIKE: &str = "pkg__demo__Demo__Main__almost__main";

struct BuiltPackage {
    _dir: TempDir,
    out: PathBuf,
    source: String,
    header: String,
}

fn build_package() -> BuiltPackage {
    let dir = tempdir().expect("tempdir");
    let package = dir.path().join("demo");
    fs::create_dir_all(package.join("src")).expect("create package source");
    common::write_file(
        &package.join("reef.toml"),
        &format!(
            "[package]\n\
             name = \"demo\"\n\
             version = \"0.1.0\"\n\
             compiler = \"={}\"\n\
             module_prefix = \"Demo\"\n",
            common::COMPILER_VERSION
        ),
    );
    common::write_file(
        &package.join("src/main.ch"),
        "module Demo.Main\n\
         export (main, almost__main)\n\
         def main(x: i32) -> i32 = add(x, 1)\n\
         def almost__main(x: i32) -> i32 = add(x, 2)\n",
    );
    let out = package.join("out");
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(&package)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            package.join("src/main.ch").to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out.to_str().unwrap(),
        ])
        .assert()
        .success();
    BuiltPackage {
        source: fs::read_to_string(out.join("main.c")).expect("generated source"),
        header: fs::read_to_string(out.join("main.h")).expect("generated header"),
        _dir: dir,
        out,
    }
}

fn compile_driver(
    out: &Path,
    header_name: &str,
    source_name: &str,
    expected: i32,
) -> Result<std::process::Output, String> {
    let source = fs::read_to_string(out.join("main.c")).map_err(|error| error.to_string())?;
    let header = fs::read_to_string(out.join(header_name)).map_err(|error| error.to_string())?;
    let generated = GeneratedHeader::parse(&header).map_err(|error| error.to_string())?;
    generated
        .validate_source(&source)
        .map_err(|error| error.to_string())?;
    let symbol = generated
        .declaration(source_name)
        .ok_or_else(|| format!("generated header has no declaration for `{source_name}`"))?
        .symbol();
    common::write_file(
        &out.join("driver.c"),
        &format!(
            "#include <assert.h>\n\
             #include \"chelis_runtime.h\"\n\
             #include \"{header_name}\"\n\
             int main(void) {{\n\
             \x20   assert({symbol}(40) == {expected});\n\
             \x20   return 0;\n\
             }}\n"
        ),
    );
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: true,
            needs_blas: false,
        },
    );
    let mut command = NativeCommand::new(&toolchain.compiler);
    command.current_dir(out);
    command.arg("-O0");
    command.args(&toolchain.compile_flags);
    command.args(["main.c", "driver.c", "-L.", "-lchelis_runtime"]);
    command.args(&toolchain.link_flags);
    command.args(["-o", "probe"]);
    Ok(command.output().expect("run native compiler"))
}

#[test]
fn package_qualified_main_keeps_the_normative_module_abi_and_links() {
    let built = build_package();
    let generated = GeneratedHeader::parse(&built.header).expect("generated declaration metadata");
    generated
        .validate_source(&built.source)
        .expect("header and source agree");

    let main = generated
        .declaration(QUALIFIED_MAIN)
        .expect("package-qualified source main declaration");
    assert_eq!(main.symbol(), QUALIFIED_MAIN);
    assert_eq!(
        main.declaration(),
        "int32_t pkg__demo__Demo__Main__main(int32_t x);"
    );

    let lookalike = generated
        .declaration(QUALIFIED_SUFFIX_LOOKALIKE)
        .expect("ordinary suffix-lookalike declaration");
    assert!(
        lookalike.symbol().starts_with("chelis_fn_"),
        "an ordinary authored name ending in `__main` remains universally mangled"
    );
    assert_ne!(lookalike.symbol(), QUALIFIED_SUFFIX_LOOKALIKE);

    if common::gcc_available() {
        let compiled = compile_driver(&built.out, "main.h", QUALIFIED_MAIN, 41)
            .expect("generated declaration consumer");
        assert!(
            compiled.status.success(),
            "generated-header caller must compile and link:\n{}",
            String::from_utf8_lossy(&compiled.stderr)
        );
        let ran = NativeCommand::new(built.out.join("probe"))
            .output()
            .expect("run native probe");
        assert!(ran.status.success(), "native probe failed");
    }
}

#[test]
fn missing_stale_and_disagreeing_headers_are_rejected() {
    let built = build_package();
    let generated = GeneratedHeader::parse(&built.header).expect("generated declaration metadata");
    let main = generated
        .declaration(QUALIFIED_MAIN)
        .expect("package-qualified source main declaration");

    assert!(
        GeneratedHeader::parse("").is_err(),
        "an absent generated declaration surface must not be treated as usable metadata"
    );

    let stale_declaration = main
        .declaration()
        .replace(main.symbol(), "pkg__demo__Demo__Main__stale");
    let stale_header = built
        .header
        .replacen(main.declaration(), &stale_declaration, 1);
    let stale = GeneratedHeader::parse(&stale_header).expect("stale header remains syntactic");
    assert!(
        stale.validate_source(&built.source).is_err(),
        "a stale declaration symbol must disagree with the generated source"
    );
    common::write_file(&built.out.join("stale.h"), &stale_header);

    let disagreeing_header = built.header.replacen(
        main.declaration(),
        "double pkg__demo__Demo__Main__main(int32_t x);",
        1,
    );
    let disagreeing =
        GeneratedHeader::parse(&disagreeing_header).expect("disagreeing header remains syntactic");
    assert!(
        disagreeing.validate_source(&built.source).is_err(),
        "a declaration type disagreement must fail source validation"
    );
    common::write_file(&built.out.join("disagreeing.h"), &disagreeing_header);

    if common::gcc_available() {
        assert!(
            compile_driver(&built.out, "missing.h", QUALIFIED_MAIN, 41).is_err(),
            "a missing generated header must fail closed"
        );
        assert!(
            compile_driver(&built.out, "stale.h", QUALIFIED_MAIN, 41).is_err(),
            "a driver consuming a stale header must fail before compilation"
        );
        assert!(
            compile_driver(&built.out, "disagreeing.h", QUALIFIED_MAIN, 41).is_err(),
            "a header whose declaration type disagrees with the source must fail before compilation"
        );
    }
}
