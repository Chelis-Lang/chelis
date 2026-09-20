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

fn wrap_export_definition_with_macro(source: &str, symbol: &str, replacement: &str) -> String {
    let signature = format!("\nint32_t {symbol}(int32_t x) {{");
    let start = source
        .find(&signature)
        .unwrap_or_else(|| panic!("generated source has no wrapper definition for `{symbol}`"));
    let mut wrapped = source.to_string();
    wrapped.insert_str(start + 1, &format!("#define {symbol} {replacement}\n"));
    let end = wrapped[start..]
        .find("\n/* chelis-export-end: ")
        .map(|offset| start + offset + 1)
        .unwrap_or_else(|| panic!("generated export block for `{symbol}` has no end marker"));
    wrapped.insert_str(end, &format!("#undef {symbol}\n"));
    wrapped
}

fn hex_metadata(value: &str) -> String {
    value.bytes().map(|byte| format!("{byte:02x}")).collect()
}

fn declaration_record(source_name: &str, symbol: &str, declaration: &str) -> String {
    format!(
        "/* chelis-declaration: {} {} {} */\n{declaration}",
        hex_metadata(source_name),
        hex_metadata(symbol),
        hex_metadata(declaration)
    )
}

fn declaration_record_prefix(source_name: &str) -> String {
    format!("/* chelis-declaration: {} ", hex_metadata(source_name))
}

#[test]
fn package_qualified_main_keeps_the_normative_module_abi_and_links() {
    let built = build_package();
    let generated = GeneratedHeader::parse(&built.header).expect("generated declaration metadata");
    assert_eq!(generated.program_identity(), "chelis_file_6d61696e");
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
        "{}\n",
        declaration_record(
            lookalike.source_name(),
            lookalike.symbol(),
            lookalike.declaration()
        )
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
    let stale_header = built.header.replacen(
        &declaration_record(main.source_name(), main.symbol(), main.declaration()),
        &declaration_record(
            main.source_name(),
            "pkg__demo__Demo__Main__stale",
            &stale_declaration,
        ),
        1,
    );
    let stale = GeneratedHeader::parse(&stale_header).expect("stale header remains syntactic");
    assert!(
        stale.validate_source(&built.source).is_err(),
        "a stale declaration symbol must disagree with the generated source"
    );
    common::write_file(&built.out.join("stale.h"), &stale_header);

    let disagreeing_declaration = "double pkg__demo__Demo__Main__main(int32_t x);";
    let disagreeing_header = built.header.replacen(
        &declaration_record(main.source_name(), main.symbol(), main.declaration()),
        &declaration_record(main.source_name(), main.symbol(), disagreeing_declaration),
        1,
    );
    let disagreeing =
        GeneratedHeader::parse(&disagreeing_header).expect("disagreeing header remains syntactic");
    assert!(
        disagreeing.validate_source(&built.source).is_err(),
        "a declaration type disagreement must fail source validation"
    );
    common::write_file(&built.out.join("disagreeing.h"), &disagreeing_header);

    let main_metadata = declaration_record_prefix(main.source_name());
    let lookalike_metadata = declaration_record_prefix(lookalike.source_name());
    let swap_metadata = declaration_record_prefix("__swap_pending");
    let swapped_header = built
        .header
        .replacen(&main_metadata, &swap_metadata, 1)
        .replacen(&lookalike_metadata, &main_metadata, 1)
        .replacen(&swap_metadata, &lookalike_metadata, 1);
    let swapped =
        GeneratedHeader::parse(&swapped_header).expect("swapped metadata remains syntactic");
    assert!(
        swapped.validate_source(&built.source).is_err(),
        "swapped source-name metadata must fail before native execution"
    );
    common::write_file(&built.out.join("swapped.h"), &swapped_header);

    let extra_source = format!(
        "{}\nint32_t synthetic_extra(int32_t x)\n{{\n    return x;\n}}\n",
        built.source
    );
    assert!(
        generated.validate_source(&extra_source).is_err(),
        "an unmarked multiline external definition must fail exact-source validation"
    );

    let comment_spoofed_source = format!(
        "{}\nint32_t /* static */ synthetic_extra(int32_t x) {{\n    return x;\n}}\n",
        built.source
    );
    assert!(
        generated.validate_source(&comment_spoofed_source).is_err(),
        "a comment containing `static` must not hide an unmarked external definition"
    );

    generated
        .validate_source(&built.source)
        .expect("compiler-emitted static/private helpers remain outside the published export set");

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

#[test]
fn source_macro_alias_swap_is_rejected_before_native_wrong_call() {
    let built = build_package();
    let generated = GeneratedHeader::parse(&built.header).expect("generated declaration metadata");
    let call = generated
        .declaration(QUALIFIED_CALL)
        .expect("ordinary call declaration");
    let prefix = generated
        .declaration(QUALIFIED_PREFIX_NAME)
        .expect("prefix-shaped declaration");
    let swapped = wrap_export_definition_with_macro(
        &wrap_export_definition_with_macro(&built.source, call.symbol(), prefix.symbol()),
        prefix.symbol(),
        call.symbol(),
    );
    assert!(
        generated.validate_source(&swapped).is_err(),
        "a source-side macro reassociation must fail before native compilation"
    );

    if common::gcc_available() {
        common::write_file(&built.out.join("main.c"), &swapped);
        let bypassed = compile_driver_for_symbol(&built.out, "main.h", call.symbol(), 44)
            .expect("compile validation-bypass control");
        assert!(
            bypassed.status.success(),
            "the reassociated source should demonstrate wrong native selection when validation is bypassed:\n{}",
            String::from_utf8_lossy(&bypassed.stderr)
        );
        run_native_probe(&built.out);
    }
}
