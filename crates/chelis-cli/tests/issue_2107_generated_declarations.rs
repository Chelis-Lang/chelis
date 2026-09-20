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
const QUALIFIED_CALL: &str = "pkg__demo__Demo__Main__call";
const QUALIFIED_PREFIX_NAME: &str = "pkg__demo__Demo__Main__chelis_fn_63616c6c";

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
         export (main, almost__main, call, chelis_fn_63616c6c)\n\
         def main(x: i32) -> i32 = add(x, 1)\n\
         def almost__main(x: i32) -> i32 = add(x, 2)\n\
         def call(x: i32) -> i32 = add(x, 3)\n\
         def chelis_fn_63616c6c(x: i32) -> i32 = add(x, 4)\n",
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
    compile_driver_for_symbol(out, header_name, symbol, expected)
}

fn compile_driver_for_symbol(
    out: &Path,
    header_name: &str,
    symbol: &str,
    expected: i32,
) -> Result<std::process::Output, String> {
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

fn run_native_probe(out: &Path) {
    let ran = NativeCommand::new(out.join("probe"))
        .output()
        .expect("run native probe");
    assert!(
        ran.status.success(),
        "native probe failed:\n{}",
        String::from_utf8_lossy(&ran.stderr)
    );
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
        run_native_probe(&built.out);
    }
}

#[test]
fn authored_prefix_name_has_a_distinct_universal_symbol_and_runs() {
    let built = build_package();
    let generated = GeneratedHeader::parse(&built.header).expect("generated declaration metadata");
    generated
        .validate_source(&built.source)
        .expect("header and source agree");
    let call = generated
        .declaration(QUALIFIED_CALL)
        .expect("ordinary call declaration");
    let prefix = generated
        .declaration(QUALIFIED_PREFIX_NAME)
        .expect("prefix-shaped authored declaration");
    assert!(call.symbol().starts_with("chelis_fn_"));
    assert!(prefix.symbol().starts_with("chelis_fn_"));
    assert_ne!(call.symbol(), prefix.symbol());
    assert_ne!(
        prefix.symbol(),
        "chelis_fn_63616c6c",
        "an authored name resembling an encoded symbol must itself be universally mangled"
    );

    if common::gcc_available() {
        let call_compiled = compile_driver(&built.out, "main.h", QUALIFIED_CALL, 43)
            .expect("ordinary authored declaration consumer");
        assert!(
            call_compiled.status.success(),
            "ordinary authored caller must link:\n{}",
            String::from_utf8_lossy(&call_compiled.stderr)
        );
        run_native_probe(&built.out);

        let prefix_compiled = compile_driver(&built.out, "main.h", QUALIFIED_PREFIX_NAME, 44)
            .expect("prefix-shaped authored declaration consumer");
        assert!(
            prefix_compiled.status.success(),
            "prefix-shaped authored caller must link:\n{}",
            String::from_utf8_lossy(&prefix_compiled.stderr)
        );
        run_native_probe(&built.out);
    }
}

#[test]
fn missing_partial_stale_disagreeing_and_swapped_headers_are_rejected() {
    let built = build_package();
    let generated = GeneratedHeader::parse(&built.header).expect("generated declaration metadata");
    let main = generated
        .declaration(QUALIFIED_MAIN)
        .expect("package-qualified source main declaration");
    let lookalike = generated
        .declaration(QUALIFIED_SUFFIX_LOOKALIKE)
        .expect("ordinary suffix-lookalike declaration");

    assert!(
        GeneratedHeader::parse("").is_err(),
        "an absent generated declaration surface must not be treated as usable metadata"
    );

    let lookalike_block = format!(
        "/* chelis-source-name: {} */\n{}",
        lookalike.source_name(),
        lookalike.declaration()
    );
    let partial_header = built.header.replacen(&lookalike_block, "", 1);
    let partial = GeneratedHeader::parse(&partial_header).expect("partial header is syntactic");
    assert!(
        partial.validate_source(&built.source).is_err(),
        "a partially missing generated header must fail exact-set validation"
    );
    common::write_file(&built.out.join("partial.h"), &partial_header);

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

    let main_metadata = format!("/* chelis-source-name: {} */", main.source_name());
    let lookalike_metadata = format!("/* chelis-source-name: {} */", lookalike.source_name());
    let swapped_header = built
        .header
        .replacen(
            &main_metadata,
            "/* chelis-source-name: __swap_pending */",
            1,
        )
        .replacen(&lookalike_metadata, &main_metadata, 1)
        .replacen(
            "/* chelis-source-name: __swap_pending */",
            &lookalike_metadata,
            1,
        );
    let swapped =
        GeneratedHeader::parse(&swapped_header).expect("swapped metadata remains syntactic");
    assert!(
        swapped.validate_source(&built.source).is_err(),
        "swapped source-name metadata must fail before native execution"
    );
    common::write_file(&built.out.join("swapped.h"), &swapped_header);

    let extra_source = format!(
        "{}\n/* chelis-authored-export: synthetic_extra */\nint32_t chelis_fn_73796e7468657469635f6578747261(int32_t x) {{\n    return x;\n}}\n",
        built.source
    );
    assert!(
        generated.validate_source(&extra_source).is_err(),
        "an extra external source definition absent from the header must fail exact-set validation"
    );

    let source_with_private_helper = format!(
        "{}\nstatic int32_t chelis_private_test_helper(int32_t x) {{\n    return x;\n}}\n",
        built.source
    );
    generated
        .validate_source(&source_with_private_helper)
        .expect("static/private helpers are outside the published export set");

    if common::gcc_available() {
        assert!(
            compile_driver(&built.out, "missing.h", QUALIFIED_MAIN, 41).is_err(),
            "a missing generated header must fail closed"
        );
        assert!(
            compile_driver(&built.out, "partial.h", QUALIFIED_MAIN, 41).is_err(),
            "a partially missing generated header must fail before compilation"
        );
        assert!(
            compile_driver(&built.out, "stale.h", QUALIFIED_MAIN, 41).is_err(),
            "a driver consuming a stale header must fail before compilation"
        );
        assert!(
            compile_driver(&built.out, "disagreeing.h", QUALIFIED_MAIN, 41).is_err(),
            "a header whose declaration type disagrees with the source must fail before compilation"
        );
        assert!(
            compile_driver(&built.out, "swapped.h", QUALIFIED_MAIN, 41).is_err(),
            "a metadata-swapped header must fail before compilation"
        );

        let wrong_symbol = swapped
            .declaration(QUALIFIED_MAIN)
            .expect("swapped main metadata")
            .symbol();
        assert_eq!(
            wrong_symbol,
            lookalike.symbol(),
            "the bypass control must select the wrong exported function"
        );
        let bypassed = compile_driver_for_symbol(&built.out, "swapped.h", wrong_symbol, 42)
            .expect("compile bypass control");
        assert!(
            bypassed.status.success(),
            "the swapped header would compile if validation were bypassed:\n{}",
            String::from_utf8_lossy(&bypassed.stderr)
        );
        run_native_probe(&built.out);
    }
}
