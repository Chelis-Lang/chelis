use assert_cmd::Command;
use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;
use tempfile::tempdir;

fn write_source(dir: &Path, field_len: usize, include_invalid: bool) -> std::path::PathBuf {
    let valid_path = dir.join("valid.csv");
    fs::write(&valid_path, format!("value\n{}\n", "x".repeat(field_len))).expect("valid CSV");
    let invalid_path = dir.join("invalid.csv");
    fs::write(
        &invalid_path,
        format!("value\n\"{}\n", "x".repeat(field_len)),
    )
    .expect("invalid CSV");
    let valid = serde_json::to_string(&valid_path.to_string_lossy()).expect("path literal");
    let invalid = serde_json::to_string(&invalid_path.to_string_lossy()).expect("path literal");
    let invalid_root = if include_invalid {
        format!(
            "invalid = match try_read_csv({invalid}) with {{\n  | Some(_) => false\n  | None => true\n}}\nshown_invalid = print(invalid)\n"
        )
    } else {
        String::new()
    };
    let source = dir.join("main.ch");
    fs::write(
        &source,
        format!(
            "module Demo.Main\nimport Std.Io.Csv (try_read_csv)\nvalid = match try_read_csv({valid}) with {{\n  | Some(rows) => match dict_get(index(rows, 0i64), \"value\") with {{\n    | Some(field) => eq(string_len(field), {field_len}i64)\n    | None => false\n  }}\n  | None => false\n}}\nshown_valid = print(valid)\n{invalid_root}"
        ),
    )
    .expect("source");
    source
}

#[test]
fn eval_reads_long_valid_line_and_rejects_long_unterminated_quote() {
    let dir = tempdir().expect("tempdir");
    let source = write_source(dir.path(), 8 * 1024, true);
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", dir.path().join("reef"))
        .args(["eval", "--file", source.to_str().expect("UTF-8 path")])
        .assert()
        .success()
        .stdout(predicates::str::contains("valid = true"))
        .stdout(predicates::str::contains("invalid = true"));
}

#[test]
fn compiled_c_reads_long_valid_line_without_a_recursive_frame_per_character() {
    let dir = tempdir().expect("tempdir");
    let source = write_source(dir.path(), 256 * 1024, false);
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
        .expect("compiled CSV program");
    assert!(run.status.success(), "compiled program: {:?}", run.status);
    assert!(
        String::from_utf8_lossy(&run.stdout).contains("true"),
        "compiled valid long line: {}",
        String::from_utf8_lossy(&run.stdout)
    );
}

#[test]
fn malformed_first_row_skips_a_later_expensive_valid_row() {
    let dir = tempdir().expect("tempdir");
    let long_row = format!("{},y", "x".repeat(4 * 1024 * 1024));
    let valid_csv = dir.path().join("valid-large.csv");
    let bad_first_csv = dir.path().join("bad-first-large.csv");
    fs::write(&valid_csv, format!("a,b\n{long_row}\n")).expect("valid large CSV");
    fs::write(&bad_first_csv, format!("a,b\n1\n{long_row}\n")).expect("malformed first row CSV");
    let source_for = |name: &str, csv: &Path| {
        let path = dir.path().join(format!("{name}.ch"));
        let csv = serde_json::to_string(&csv.to_string_lossy()).expect("CSV path literal");
        fs::write(
            &path,
            format!("module Demo.Main\nimport Std.Io.Csv (try_read_csv)\nv = match try_read_csv({csv}) with {{\n  | Some(rows) => len(rows)\n  | None => -1i64\n}}\n"),
        )
        .expect("Surf source");
        path
    };
    let valid_source = source_for("valid-large", &valid_csv);
    let bad_first_source = source_for("bad-first-large", &bad_first_csv);

    // The control proves the same later row needs more than the evaluation
    // budget when parsed; the malformed-first file can pass only by skipping it.
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", dir.path().join("reef"))
        .args([
            "eval",
            "--timeout",
            "3",
            "--file",
            valid_source.to_str().expect("UTF-8 path"),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("evaluation timed out after 3s"));
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", dir.path().join("reef"))
        .args([
            "eval",
            "--timeout",
            "3",
            "--file",
            bad_first_source.to_str().expect("UTF-8 path"),
        ])
        .assert()
        .success()
        .stdout(predicates::str::contains("v = -1"));
}
