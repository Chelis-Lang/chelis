//! chelis#1587 at the CLI, both ingresses. Issue #1592 later made the `iN`
//! names canonical while preserving the ecosystem-spelled execution wire tags.
//!
//! `def ident(x: i64) -> i64 = x` scored 1.0 on a tree without the alias
//! mapping and accepted `ident(1.5f64)` returning f64, because `i64` fell
//! through the desugarer's primitive test and became `forall a. a`. The
//! signature named a type and meant nothing.

use assert_cmd::Command;
use std::{fs, path::Path};
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

fn run(root: &Path, args: &[&str]) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(root)
        .args(args)
        .output()
        .expect("run")
}

fn text(output: &std::process::Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

const MISMATCH: &str = "module Alias.Main\nexport (main)\n\
     def ident(x: i64) -> i64 = x\n\
     def main() -> f64 = ident(1.5f64)\n";
const ACCEPTED: &str = "module Alias.Main\nexport (main)\n\
     def ident(x: i64) -> i64 = x\n\
     def main() -> i64 = ident(5i64)\n";

/// Regression test. Red on `4ad308501`, where both units scored 1.0 because
/// `i64` was a quantifier: the mismatch unit is now rejected as i64 against
/// f64, and the twin proves the alias still NAMES i64 rather than merely
/// being rejected everywhere.
#[test]
fn an_integer_signature_names_i64_at_both_ingresses() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();

    fs::write(root.join("mismatch.ch"), MISMATCH).expect("write");
    for args in [
        &["check", "mismatch.ch"][..],
        &["eval", "--file", "mismatch.ch"],
    ] {
        let rendered = text(&run(root, args));
        assert!(
            rendered.contains("i64"),
            "`chelis {}` must report the mismatch against i64, not accept a \
             quantifier: {rendered}",
            args[0]
        );
    }
    let check = text(&run(root, &["check", "mismatch.ch"]));
    assert!(
        !check.contains("\"score\": 1,"),
        "a signature saying i64 must not accept an f64 argument: {check}"
    );

    fs::write(root.join("ok.ch"), ACCEPTED).expect("write");
    let accepted = text(&run(root, &["check", "ok.ch"]));
    assert!(
        accepted.contains("\"score\": 1,"),
        "`i64` must still name i64 and accept an i64 argument: {accepted}"
    );
}

#[test]
fn integer_literal_adoption_preserves_exact_values_at_both_ingresses() {
    for (canonical, wire, value) in [
        ("i8", "int8", -128_i64),
        ("i16", "int16", -32768),
        ("i32", "int32", -2147483648),
        ("i64", "int64", 3000000000),
    ] {
        for (source, expected) in [
            (
                format!("out = cast({value}, {canonical})\n"),
                serde_json::json!({"type": "scalar", "value": {"dtype": wire, "value": value}}),
            ),
            (
                format!("out: tensor[2, {canonical}] = [{value}, 1]\n"),
                serde_json::json!({"type": "tensor", "value": {
                    "shape": [2], "data": {"dtype": wire, "values": [value, 1]}
                }}),
            ),
        ] {
            let dir = tempdir().expect("tempdir");
            let root = dir.path();
            fs::write(root.join("input.ch"), &source).expect("write");
            let deep = run(root, &["deep", "input.ch"]);
            assert!(deep.status.success(), "{}", text(&deep));
            fs::write(root.join("input.dp"), deep.stdout).expect("write Deep");
            for input in ["input.ch", "input.dp"] {
                let checked = run(root, &["check", input]);
                assert!(checked.status.success(), "{source}: {}", text(&checked));
                let check: serde_json::Value =
                    serde_json::from_slice(&checked.stdout).expect("JSON");
                assert_eq!(check["score"], 1.0, "{source}: {check}");
                assert_eq!(check["errors"], serde_json::json!([]));
                let evaluated = run(root, &["eval", "--json", "--file", input]);
                assert!(evaluated.status.success(), "{source}: {}", text(&evaluated));
                let eval: serde_json::Value =
                    serde_json::from_slice(&evaluated.stdout).expect("JSON");
                assert_eq!(eval["roots"].as_array().expect("roots").len(), 1);
                assert_eq!(eval["roots"][0]["value"], expected, "{source}: {eval}");
            }
        }
    }
}

#[test]
fn integer_literal_adoption_does_not_bypass_range_or_suffix_rejections() {
    for (canonical, outside) in [("i8", "128"), ("i16", "32768"), ("i32", "2147483648")] {
        for source in [
            format!("out = cast({outside}, {canonical})\n"),
            format!("out: tensor[1, {canonical}] = [{outside}]\n"),
            format!("out: tensor[1, {canonical}] = [1.0f64]\n"),
        ] {
            let dir = tempdir().expect("tempdir");
            fs::write(dir.path().join("bad.ch"), &source).expect("write");
            let deep = run(dir.path(), &["deep", "bad.ch"]);
            assert!(deep.status.success(), "{}", text(&deep));
            fs::write(dir.path().join("bad.dp"), deep.stdout).expect("write Deep");
            for input in ["bad.ch", "bad.dp"] {
                let checked = run(dir.path(), &["check", input]);
                assert!(!checked.status.success(), "{source}: {}", text(&checked));
                let result: serde_json::Value =
                    serde_json::from_slice(&checked.stdout).expect("JSON");
                let errors = result["errors"].as_array().expect("errors");
                assert!(!errors.is_empty(), "{source}: {result}");
                assert!(
                    errors
                        .iter()
                        .any(|error| error["message"].as_str().is_some_and(|message| {
                            message.contains(canonical)
                                && if source.contains("1.0f64") {
                                    message.contains("f64")
                                } else {
                                    message.contains("out of range")
                                }
                        })),
                    "{source}: {result}"
                );
                assert!(
                    !run(dir.path(), &["eval", "--file", input]).status.success(),
                    "{source}"
                );
                assert!(
                    !run(
                        dir.path(),
                        &["build", input, "--target", "c", "--output", "out"]
                    )
                    .status
                    .success(),
                    "{source}"
                );
                assert!(
                    !dir.path().join("out").exists(),
                    "a rejected input emitted artifacts: {source}"
                );
            }
        }
    }
}

#[test]
fn canonical_integer_literals_compile_and_run_at_the_declared_width() {
    for (name, source, expected) in [
        (
            "alias_i64_scalar",
            "out = cast(3000000000, i64)\n",
            "out = 3000000000",
        ),
        ("alias_i8_scalar", "out = cast(-128, i8)\n", "out = -128"),
        (
            "alias_i64_tensor",
            "out: tensor[2, i64] = [3000000000, 1]\n",
            "out = tensor(shape=[2], data=[3000000000, 1])",
        ),
    ] {
        assert_eq!(common::build_and_run(source, name).trim(), expected);
    }
}
