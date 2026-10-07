use assert_cmd::Command;
use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;
use tempfile::tempdir;

fn write_source(dir: &Path) -> std::path::PathBuf {
    let valid_path = dir.join("valid.json");
    let mut document = String::from("{");
    for index in 0..25_000 {
        document.push_str(&format!("\"key{index:05}\":0,"));
    }
    document.push_str("\"last\":1}");
    fs::write(&valid_path, document).expect("valid JSON");
    let invalid_path = dir.join("invalid.json");
    fs::write(&invalid_path, "{\"last\":1,}").expect("invalid JSON");
    let valid = serde_json::to_string(&valid_path.to_string_lossy()).expect("valid path literal");
    let invalid =
        serde_json::to_string(&invalid_path.to_string_lossy()).expect("invalid path literal");
    let source = dir.join("main.ch");
    fs::write(
        &source,
        format!(
            "module Demo.Main\nimport Std.Io.Json (try_load_json, json_get, json_int)\nvalid = match try_load_json({valid}) with {{\n  | Some(doc) => match json_int(json_get(doc, \"last\")) with {{\n    | Some(value) => eq(value, 1i64)\n    | None => false\n  }}\n  | None => false\n}}\ninvalid = match try_load_json({invalid}) with {{\n  | Some(_) => false\n  | None => true\n}}\nshown_valid = print(valid)\nshown_invalid = print(invalid)\n"
        ),
    )
    .expect("Surf source");
    source
}

#[test]
fn compiled_c_parses_object_beyond_the_prior_stack_limit_and_rejects_trailing_comma() {
    let dir = tempdir().expect("tempdir");
    let source = write_source(dir.path());
    let output = dir.path().join("out");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", dir.path().join("reef"))
        .args([
            "build",
            "--emit-c",
            source.to_str().expect("UTF-8 path"),
            "--target",
            "c",
            "--output",
            output.to_str().expect("UTF-8 path"),
        ])
        .assert()
        .success();
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: true,
            needs_blas: false,
        },
    );
    let mut compile = StdCommand::new(&toolchain.compiler);
    compile.current_dir(&output);
    compile.arg("-O2");
    compile.args(&toolchain.compile_flags);
    compile.arg("main.c");
    compile.arg(output.join("libchelis_runtime.a"));
    compile.args(&toolchain.link_flags);
    compile.args(["-o", "main"]);
    assert!(compile.status().expect("C compiler").success());
    let run = StdCommand::new(output.join("main"))
        .output()
        .expect("compiled JSON program");
    assert!(
        run.status.success(),
        "compiled program: {:?}; stderr: {}",
        run.status,
        String::from_utf8_lossy(&run.stderr)
    );
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(stdout.contains("valid = true"), "{stdout}");
    assert!(stdout.contains("invalid = true"), "{stdout}");
}
