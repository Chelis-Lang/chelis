//! Executable acceptance for chelis#953 and chelis#1855.
//!
//! The public character operations, generated C literal transport, runtime
//! observation, and Std.Io.Json Unicode rules must agree byte-for-byte across
//! evaluator and compiled lanes. Negative cases pin both static rejection and
//! runtime Domain behavior.

use assert_cmd::Command;
use predicates::prelude::*;

#[path = "common/mod.rs"]
mod common;

use common::{build_and_run_app, link_generated, make_app, write_file};

fn eval_app(reef_home: &std::path::Path, app: &std::path::Path) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(app)
        .args(["eval", "--file", "src/main.ch"])
        .output()
        .expect("run evaluator")
}

fn build_app(reef_home: &std::path::Path, app: &std::path::Path, stem: &str) -> std::path::PathBuf {
    let out = app.join("out");
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(app)
        .args([
            "build",
            "--emit-c",
            "src/main.ch",
            "--target",
            "c",
            "--output",
        ])
        .arg(&out)
        .assert()
        .success();
    assert!(link_generated(&out, "main.c", stem).success());
    out.join(stem)
}

#[test]
fn character_builtins_and_all_string_literal_escape_classes_have_eval_c_parity() {
    let (_dir, reef_home, app) = make_app("issues-953-1855-char-parity");
    write_file(
        &app.join("src/main.ch"),
        r#"module Demo.Main

control = "a\u{000B}b"
nul = "a\u{0000}b"
adjacent = "\u{0041}1"
multibyte = "\u{00e9}\u{65e5}\u{1f600}"
code_ascii = char_code("A")
code_emoji = char_code("😀")
code_nul = char_code("\u{0000}")
from_ascii = char_from_code(cast(65, i64))
from_emoji = char_from_code(cast(128512, i64))
from_nul_code = char_code(char_from_code(cast(0, i64)))
nul_len = string_len(nul)
"#,
    );

    let eval = eval_app(&reef_home, &app);
    assert!(
        eval.status.success(),
        "{}",
        String::from_utf8_lossy(&eval.stderr)
    );
    let compiled = build_and_run_app(&reef_home, &app, "main");
    assert_eq!(eval.stdout, compiled.as_bytes(), "eval/C output diverged");
    let output = String::from_utf8(eval.stdout).expect("UTF-8 output");
    for expected in [
        "control = a\u{000b}b",
        "nul = a\0b",
        "adjacent = A1",
        "multibyte = é日😀",
        "code_ascii = 65",
        "code_emoji = 128512",
        "code_nul = 0",
        "from_ascii = A",
        "from_emoji = 😀",
        "from_nul_code = 0",
        "nul_len = 3",
    ] {
        assert!(
            output.contains(expected),
            "missing {expected:?} in {output:?}"
        );
    }
}

#[test]
fn character_builtin_domain_errors_match_in_eval_and_compiled_lanes() {
    for (stem, expression, expected) in [
        (
            "char_code_empty",
            "char_code(\"\")",
            "char_code operand has 0 Unicode scalar values, expected exactly one\nnumeric trap: domain in char_code at i64",
        ),
        (
            "char_code_many",
            "char_code(\"ab\")",
            "char_code operand has 2 Unicode scalar values, expected exactly one\nnumeric trap: domain in char_code at i64",
        ),
        (
            "char_from_code_negative",
            "char_from_code(cast(-1, i64))",
            "char_from_code code -1 is not a Unicode scalar value\nnumeric trap: domain in char_from_code at i64",
        ),
        (
            "char_from_code_surrogate",
            "char_from_code(cast(55296, i64))",
            "char_from_code code 55296 is not a Unicode scalar value\nnumeric trap: domain in char_from_code at i64",
        ),
        (
            "char_from_code_too_large",
            "char_from_code(cast(1114112, i64))",
            "char_from_code code 1114112 is not a Unicode scalar value\nnumeric trap: domain in char_from_code at i64",
        ),
    ] {
        let (_dir, reef_home, app) = make_app(stem);
        write_file(
            &app.join("src/main.ch"),
            &format!("module Demo.Main\n\nbad = {expression}\n"),
        );
        let eval = eval_app(&reef_home, &app);
        assert!(
            !eval.status.success(),
            "eval unexpectedly accepted {expression}"
        );
        assert!(
            String::from_utf8_lossy(&eval.stderr).contains(expected),
            "eval diagnostic mismatch: {}",
            String::from_utf8_lossy(&eval.stderr)
        );
        let binary = build_app(&reef_home, &app, stem);
        let compiled = std::process::Command::new(binary)
            .output()
            .expect("run compiled rejection");
        assert!(
            !compiled.status.success(),
            "compiled lane accepted {expression}"
        );
        assert!(
            String::from_utf8_lossy(&compiled.stderr).contains(expected),
            "compiled diagnostic mismatch: {}",
            String::from_utf8_lossy(&compiled.stderr)
        );
    }
}

#[test]
fn character_builtins_reject_wrong_static_types() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("wrong.ch");
    write_file(
        &path,
        "module Demo.Wrong\n\na = char_code(cast(1, i64))\nb = char_from_code(\"A\")\n",
    );
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .assert()
        .failure()
        .stdout(predicate::str::contains("expected string, got i64"))
        .stdout(predicate::str::contains("expected i64, got string"));
}

#[test]
fn std_json_unicode_is_rfc_interoperable_in_eval_and_c() {
    let (_dir, reef_home, app) = make_app("issue-953-json-unicode");
    write_file(
        &app.join("src/main.ch"),
        r#"module Demo.Main

import Std.Io.Json (JsonString, json_string, parse_json, to_json, try_parse_json)

def text_or(value: Option[string], fallback: string) -> string =
  match value with {
    | Some(text) => text
    | None => fallback
  }

encoded = to_json(JsonString("\u{0000}\u{0008}\u{000B}\u{000C}\u{001F}"))
decoded = text_or(json_string(Some(parse_json("\"\\u00e9\\ud83d\\ude00\""))), "BAD")
bad_hex = match try_parse_json("\"\\u12x4\"") with { | Some(_) => false | None => true }
lone_high = match try_parse_json("\"\\ud83d\"") with { | Some(_) => false | None => true }
lone_low = match try_parse_json("\"\\ude00\"") with { | Some(_) => false | None => true }
wrong_pair = match try_parse_json("\"\\ud83d\\u0041\"") with { | Some(_) => false | None => true }
raw_control = match try_parse_json("\"a\u{000B}b\"") with { | Some(_) => false | None => true }
"#,
    );

    let eval = eval_app(&reef_home, &app);
    assert!(
        eval.status.success(),
        "{}",
        String::from_utf8_lossy(&eval.stderr)
    );
    let compiled = build_and_run_app(&reef_home, &app, "main");
    assert_eq!(
        eval.stdout,
        compiled.as_bytes(),
        "JSON eval/C output diverged"
    );
    let output = String::from_utf8(eval.stdout).expect("UTF-8 output");
    let encoded = output
        .lines()
        .find_map(|line| line.strip_prefix("encoded = "))
        .expect("encoded root");
    let decoded_by_serde: String = serde_json::from_str(encoded).expect("RFC-valid JSON");
    assert_eq!(decoded_by_serde, "\0\u{8}\u{b}\u{c}\u{1f}");
    for expected in [
        "decoded = é😀",
        "bad_hex = true",
        "lone_high = true",
        "lone_low = true",
        "wrong_pair = true",
        "raw_control = true",
    ] {
        assert!(
            output.contains(expected),
            "missing {expected:?} in {output:?}"
        );
    }
}

#[test]
fn malformed_surf_unicode_escape_is_rejected_before_codegen() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("malformed.ch");
    write_file(&path, "module Demo.Bad\n\nbad = \"\\u{xyz}\"\n");
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["check", path.to_str().unwrap()])
        .assert()
        .failure();
}
