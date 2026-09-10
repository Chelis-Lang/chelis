//! Source-path spelling must not turn a valid program into invalid C (#1710).
//! The oracle compiles and executes the emitted translation unit, including its header.
use assert_cmd::Command;
use std::{ffi::OsStr, fs, process::Command as NativeCommand};
use tempfile::tempdir;

const SOURCE: &str = "def main() = scalar_to_tensor(7.0f32)\n";

fn check_stem(stem: &str, deep: bool, expected_symbol: &str) {
    check_os_stem(OsStr::new(stem), deep, expected_symbol, false);
}

fn check_os_stem(stem: &OsStr, deep: bool, expected_symbol: &str, explicit_output: bool) {
    let dir = tempdir().unwrap();
    let mut filename = stem.to_os_string();
    filename.push(".ch");
    let surf = dir.path().join(filename);
    fs::write(&surf, SOURCE).unwrap();
    let source = if deep {
        let converted = Command::cargo_bin("chelis")
            .unwrap()
            .current_dir(dir.path())
            .arg("deep")
            .arg(&surf)
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let path = surf.with_extension("dp");
        fs::write(&path, converted).unwrap();
        path
    } else {
        surf
    };
    let output = dir.path().join("out");
    let mut c_filename = stem.to_os_string();
    c_filename.push(".c");
    let c = output.join(if explicit_output {
        "chosen-artifact.c".into()
    } else {
        c_filename
    });
    let output_arg = if explicit_output { &c } else { &output };
    let mut build = Command::cargo_bin("chelis").unwrap();
    build
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("build")
        .arg(&source)
        .args(["--target", "c", "--output"])
        .arg(output_arg);
    if deep {
        build.arg("--deep");
    }
    build.assert().success();
    let h = c.with_extension("h");
    assert!(
        c.exists() && h.exists(),
        "output paths retain the source stem"
    );
    let header = fs::read_to_string(&h).unwrap();
    assert!(
        header.contains(&format!("{expected_symbol}__main(")),
        "{header}"
    );
    let binary = output.join("program");
    let cc = NativeCommand::new(std::env::var("CC").unwrap_or_else(|_| "cc".into()))
        .args(["-std=c11", "-O0", "-include"])
        .arg(output.join("chelis_runtime.h"))
        .arg("-include")
        .arg(&h)
        .arg("-I")
        .arg(&output)
        .arg(&c)
        .arg(output.join("libchelis_runtime.a"))
        .args(["-lm", "-lpthread", "-ldl", "-o"])
        .arg(&binary)
        .output()
        .unwrap();
    assert!(
        cc.status.success(),
        "{stem:?}: {}",
        String::from_utf8_lossy(&cc.stderr)
    );
    let ran = NativeCommand::new(&binary).output().unwrap();
    assert!(
        ran.status.success(),
        "{}",
        String::from_utf8_lossy(&ran.stderr)
    );
    assert_eq!(ran.stdout, b"main = 7.0\n");
    let evaluated = Command::cargo_bin("chelis")
        .unwrap()
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file"])
        .arg(&source)
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    assert_eq!(ran.stdout, evaluated);
}

#[test]
fn hyphenated_source_builds_links_and_runs() {
    check_stem(
        "simple-shape",
        false,
        "chelis_file_73696d706c652d7368617065",
    );
}

#[test]
fn identifier_safe_source_keeps_its_symbol_and_output_paths() {
    check_stem("simple_shape", false, "simple_shape");
}

#[test]
fn deep_input_uses_the_same_source_name_encoding() {
    check_stem("deep-shape", true, "chelis_file_646565702d7368617065");
}

#[test]
fn punctuation_unicode_and_leading_digits_produce_legal_symbols() {
    for (stem, symbol) in [
        ("7shape", "chelis_file_377368617065"),
        ("caf\u{e9}", "chelis_file_636166c3a9"),
        ("shape.data", "chelis_file_73686170652e64617461"),
        ("double", "chelis_file_646f75626c65"),
    ] {
        check_stem(stem, false, symbol);
    }
}

#[test]
fn literal_escape_prefix_cannot_alias_an_encoded_filename() {
    // "a-b" and the legal literal spelling of its encoded name must differ.
    check_stem("a-b", false, "chelis_file_612d62");
    check_stem(
        "chelis_file_612d62",
        false,
        "chelis_file_6368656c69735f66696c655f363132643632",
    );
}

#[test]
fn explicit_output_path_does_not_change_the_module_symbol() {
    check_os_stem(OsStr::new("a-b"), false, "chelis_file_612d62", true);
}

#[cfg(target_os = "linux")]
#[test]
fn non_utf8_filenames_are_not_collapsed_to_a_shared_fallback() {
    use std::os::unix::ffi::OsStrExt;
    check_os_stem(
        OsStr::from_bytes(b"shape\xff"),
        false,
        "chelis_file_7368617065ff",
        false,
    );
    check_os_stem(
        OsStr::from_bytes(b"shape\xfe"),
        false,
        "chelis_file_7368617065fe",
        false,
    );
}

#[test]
fn tensor_dag_object_uses_the_encoded_header_symbol() {
    let dir = tempdir().unwrap();
    let source = dir.path().join("dag-shape.ch");
    fs::write(
        &source,
        "def f(x: tensor[1, f32]) -> tensor[1, f32] = x + x\n",
    )
    .unwrap();
    let out = dir.path().join("out");
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("build")
        .arg(&source)
        .args(["--target", "c", "--output"])
        .arg(&out)
        .assert()
        .success();
    let header = out.join("dag-shape.h");
    assert!(
        fs::read_to_string(&header)
            .unwrap()
            .contains("void chelis_file_6461672d7368617065(")
    );
    let compiled = NativeCommand::new(std::env::var("CC").unwrap_or_else(|_| "cc".into()))
        .args(["-std=c11", "-include"])
        .arg(out.join("chelis_runtime.h"))
        .arg("-include")
        .arg(header)
        .arg("-I")
        .arg(&out)
        .arg("-c")
        .arg(out.join("dag-shape.c"))
        .arg("-o")
        .arg(out.join("dag.o"))
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
}
