//! Executable #1585 channel oracle: preceding effects survive failure, while
//! JSON stdout remains a result document only (spec/08 §1.1).

use assert_cmd::Command;
use std::fs;
use tempfile::tempdir;

fn eval_source(source: &str, json: bool, deep: bool) -> std::process::Output {
    let dir = tempdir().expect("tempdir");
    let (name, source) = if deep {
        let declarations = chelis_surf::parser::parse_str(source).expect("parse Surf");
        let expressions = chelis_surf::desugar::desugar_program(&declarations)
            .expect("Surf fixture must desugar");
        (
            "failure.dp",
            chelis_deep::printer::print_canonical(&expressions),
        )
    } else {
        (
            "failure.ch",
            chelis_surf::format::format_source(source).expect("format"),
        )
    };
    let path = dir.path().join(name);
    fs::write(&path, source).expect("write fixture");
    let mut command = Command::cargo_bin("chelis").expect("binary");
    command
        .current_dir(dir.path())
        .args(["eval", "--file"])
        .arg(&path);
    if json {
        command.arg("--json");
    }
    command.output().expect("eval")
}

fn program(body: &str) -> String {
    format!("def run() -> i64 ! {{ IO }} = {{\n{body}\n}}\nout = run()\n")
}

#[test]
fn failure_emits_preceding_effects_once_in_text_and_json_modes() {
    let source = program(
        "_ = print(\"first\")\n_ = debug(\"second\")\nx = floor_div(1i64, 0i64)\n_ = print(\"after\")\nx",
    );
    for deep in [false, true] {
        for json in [false, true] {
            let output = eval_source(&source, json, deep);
            let stderr = String::from_utf8(output.stderr).expect("stderr");
            assert!(!output.status.success(), "{stderr}");
            assert!(stderr.contains("division by zero"), "{stderr}");
            assert!(!stderr.contains("after"), "{stderr}");
            if json {
                assert!(output.stdout.is_empty(), "no partial JSON or human output");
                assert!(
                    stderr.starts_with(
                        "first\nsecond\nnumeric trap: division by zero in floor_div at i64"
                    ),
                    "{stderr}"
                );
                assert_eq!(stderr.matches("first").count(), 1);
            } else {
                assert_eq!(output.stdout, b"first\nsecond\n");
                assert!(!stderr.contains("first"), "{stderr}");
            }
        }
    }
}

#[test]
fn failure_before_effect_emits_no_transcript() {
    let source = program("x = floor_div(1i64, 0i64)\n_ = print(\"after\")\nx");
    for json in [false, true] {
        let output = eval_source(&source, json, false);
        let stderr = String::from_utf8(output.stderr).expect("stderr");
        assert!(!output.status.success(), "{stderr}");
        assert!(output.stdout.is_empty());
        assert_eq!(
            stderr,
            "numeric trap: division by zero in floor_div at i64\n"
        );
        assert!(!stderr.contains("after"), "{stderr}");
    }
}

#[test]
fn success_keeps_transcript_in_its_existing_result_channel() {
    let source = program("_ = print(\"first\")\n7i64");
    for json in [false, true] {
        let output = eval_source(&source, json, false);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stderr.is_empty());
        if json {
            let result: serde_json::Value = serde_json::from_slice(&output.stdout).expect("JSON");
            assert_eq!(result["transcript"], serde_json::json!(["first"]));
            assert_eq!(result["roots"].as_array().expect("roots").len(), 1);
        } else {
            assert_eq!(output.stdout, b"first\nout = 7\n");
        }
    }
}

#[test]
fn text_transcript_reaches_a_shared_sink_before_the_diagnostic() {
    use std::io::{Read, Seek, SeekFrom};
    use std::process::Stdio;

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("ordered.ch");
    let source = program("_ = print(\"before\")\nfail(\"authored failure\")");
    fs::write(
        &path,
        chelis_surf::format::format_source(&source).expect("format"),
    )
    .expect("fixture");
    let mut sink = tempfile::tempfile().expect("sink");
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_chelis"))
        .current_dir(dir.path())
        .args(["eval", "--file"])
        .arg(path)
        .stdout(Stdio::from(sink.try_clone().expect("stdout")))
        .stderr(Stdio::from(sink.try_clone().expect("stderr")))
        .status()
        .expect("eval");
    assert!(!status.success());
    sink.seek(SeekFrom::Start(0)).expect("rewind");
    let mut text = String::new();
    sink.read_to_string(&mut text).expect("read output");
    assert_eq!(text, "before\nerror: authored failure\n");
}
