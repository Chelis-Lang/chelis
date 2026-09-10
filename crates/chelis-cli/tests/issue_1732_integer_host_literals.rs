//! [04-LIT-1], [05-OBS-7,11]: a host literal retains its checked width
//! through function returns, so the caller and callee ownership schemas agree.

mod common;

use std::{fs, process::Command};
use tempfile::TempDir;

fn check_eval_and_run(source: &str, expected: &str) {
    check_eval_and_run_kind(source, expected, "ch");
}

fn check_eval_and_run_kind(source: &str, expected: &str, extension: &str) {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join(format!("program.{extension}"));
    fs::write(&file, source).unwrap();
    let formatted = Command::new(env!("CARGO_BIN_EXE_chelis"))
        .args(["fmt", "--inplace"])
        .arg(&file)
        .output()
        .unwrap();
    assert!(formatted.status.success(), "{formatted:?}");
    let checked = Command::new(env!("CARGO_BIN_EXE_chelis"))
        .arg("check")
        .arg(&file)
        .output()
        .unwrap();
    assert!(checked.status.success(), "{checked:?}");
    let report: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert_eq!(report["score"], 1.0, "{report}");
    assert_eq!(report["errors"], serde_json::json!([]), "{report}");
    let evaluated = Command::new(env!("CARGO_BIN_EXE_chelis"))
        .args(["eval", "--file"])
        .arg(&file)
        .output()
        .unwrap();
    assert!(evaluated.status.success(), "{evaluated:?}");
    assert_eq!(String::from_utf8(evaluated.stdout).unwrap(), expected);
    let built = Command::new(env!("CARGO_BIN_EXE_chelis"))
        .arg("build")
        .arg(&file)
        .args(["--target", "c", "--output"])
        .arg(dir.path())
        .output()
        .unwrap();
    assert!(built.status.success(), "{built:?}");
    assert!(common::link_generated(dir.path(), "program.c", "program").success());
    let native = Command::new(dir.path().join("program")).output().unwrap();
    assert!(native.status.success(), "{native:?}");
    assert_eq!(String::from_utf8(native.stdout).unwrap(), expected);
}

#[test]
fn executable_nullary_integer_example_agrees_in_eval_and_c() {
    check_eval_and_run(
        include_str!("../../../examples/integer_functions.ch"),
        "anchor = 7\nuser = 7\n",
    );
}

#[test]
fn integer_literal_returns_preserve_every_integer_width_and_exact_int64() {
    for (dtype, literal, zero, expected) in [
        ("int8", "127i8", "0i8", "127"),
        ("int16", "32767i16", "0i16", "32767"),
        ("int32", "2147483647i32", "0i32", "2147483647"),
        ("int64", "9007199254740993i64", "0i64", "9007199254740993"),
    ] {
        let source = format!(
            "module Widths\n\
             def anchor() -> {dtype} = {literal}\n\
             def constant(unused: {dtype}) -> {dtype} = {literal}\n\
             def local() -> {dtype} = {{\nvalue = {literal}\nvalue\n}}\n\
             def user() -> {dtype} = anchor()\n\
             def argument_user() -> {dtype} = constant({zero})\n"
        );
        let expected = ["anchor", "local", "user", "argument_user"]
            .map(|name| format!("{name} = {expected}\n"))
            .concat();
        check_eval_and_run(&source, &expected);
    }
}

#[test]
fn integer_literal_return_remains_callable_through_aliases() {
    check_eval_and_run(
        "module Aliases\n\
         def anchor() -> int32 = 7\n\
         alias = anchor\n\
         second = alias\n\
         def direct() -> int32 = second()\n",
        "anchor = 7\ndirect = 7\n",
    );
}

#[test]
fn marked_integer_atom_finalizes_directly_at_its_float_width() {
    // One above an f32 midpoint beyond 2^53: going through f64 first
    // would round down to the midpoint and then choose the wrong f32.
    check_eval_and_run_kind(
        "(defsig {} answer (t-fn {} (t-prim {} int64)))\n\
         (def {} answer (fn {} (params {})\n\
           (cast {}\n\
             (lit {type: (t-prim {} f32), literal_source: integer} 18014399583223809)\n\
             (t-prim {} int64))))\n",
        "answer = 18014400656965632\n",
        "dp",
    );
}

#[test]
fn float_literal_finalization_and_nonnumeric_returns_are_unchanged() {
    for (dtype, literal, expected) in [
        ("f16", "0.1f16", "0.1"),
        ("bf16", "0.1bf16", "0.1"),
        ("f32", "0.1f32", "0.1"),
        ("f64", "0.10000000000000002f64", "0.10000000000000002"),
        ("bool", "true", "true"),
        ("string", "\"hello\"", "hello"),
    ] {
        check_eval_and_run(
            &format!(
                "module Controls\ndef anchor() -> {dtype} = {literal}\n\
                 def user() -> {dtype} = anchor()\n"
            ),
            &format!("anchor = {expected}\nuser = {expected}\n"),
        );
    }
}

#[test]
fn invalid_literal_width_and_return_type_are_rejected_before_codegen() {
    for (body, extension) in [
        ("def anchor() -> int8 = 128i8", "ch"),
        ("def anchor() -> int16 = 32768i16", "ch"),
        ("def anchor() -> int32 = 2147483648i32", "ch"),
        ("def anchor() -> int64 = 9223372036854775808i64", "ch"),
        ("def anchor() -> bool = 7", "ch"),
        ("def anchor() -> int8 = 7", "ch"),
        // Metadata alone may not reinterpret an integer atom as a float.
        ("(def {} answer (lit {type: (t-prim {} f32)} 7))", "dp"),
    ] {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join(format!("invalid.{extension}"));
        let source = if extension == "ch" {
            format!("module Invalid\n{body}\n")
        } else {
            body.to_string()
        };
        fs::write(&file, source).unwrap();
        for command in ["check", "eval", "build"] {
            let mut cmd = Command::new(env!("CARGO_BIN_EXE_chelis"));
            cmd.env("CHELIS_STYLE_GATE_DISABLE", "1").arg(command);
            if command == "eval" {
                cmd.arg("--file");
            }
            cmd.arg(&file);
            if command == "build" {
                cmd.args(["--target", "c", "--output"]).arg(dir.path());
            }
            let result = cmd.output().unwrap();
            assert!(!result.status.success(), "{body}: {result:?}");
            let errors = format!(
                "{}{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
            assert!(
                errors.contains("range")
                    || errors.contains("overflow")
                    || errors.contains("only valid after unary")
                    || errors.contains("integer atom")
                    || errors.contains("doesn't match declared signature"),
                "{body}: {errors}"
            );
            assert!(!dir.path().join("invalid.c").exists());
        }
    }
}
