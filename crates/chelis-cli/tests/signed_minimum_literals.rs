//! [04-LIT-2]: actual formatter/checker acceptance and rejection at each signed
//! width; ordinary property operands and function arguments retain exact values.
use std::{
    fs,
    path::Path,
    process::{Command, Output},
};
use tempfile::TempDir;

fn run(command: &str, path: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_chelis"))
        .arg(command)
        .arg(path)
        .env_remove("CHELIS_STYLE_GATE_DISABLE")
        .output()
        .unwrap()
}
fn format_check(source: &str) -> (TempDir, std::path::PathBuf, Output) {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("minimum.ch");
    fs::write(&path, source).unwrap();
    let formatted = Command::new(env!("CARGO_BIN_EXE_chelis"))
        .args(["fmt", "--inplace"])
        .arg(&path)
        .output()
        .unwrap();
    if !formatted.status.success() {
        return (dir, path, formatted);
    }
    let checked = run("check", &path);
    (dir, path, checked)
}
#[test]
fn formatted_property_minimum_operands_and_calls_check_at_every_signed_width() {
    for (dtype, magnitude) in [
        ("i8", "128"),
        ("i16", "32768"),
        ("i32", "2147483648"),
        ("i64", "9223372036854775808"),
    ] {
        for literal in [
            format!("-{magnitude}{dtype}"),
            format!("-(({magnitude}{dtype}))"),
            format!("cast(-{magnitude}, {dtype})"),
            format!("cast(-{magnitude}{dtype}, {dtype})"),
        ] {
            let source = format!(
                "def value(x: {dtype}) -> {dtype} = x\nanswer = value({literal})\n@property prop_min forall(x: {dtype}):\n  value(x) >= {literal}\n"
            );
            let (_dir, path, checked) = format_check(&source);
            assert!(checked.status.success(), "{dtype}: {checked:?}");
            let json: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
            assert_eq!(json["score"], 1.0);
            assert_eq!(json["errors"], serde_json::json!([]));
            let before = fs::read(&path).unwrap();
            let second = Command::new(env!("CARGO_BIN_EXE_chelis"))
                .args(["fmt", "--inplace"])
                .arg(&path)
                .output()
                .unwrap();
            assert!(second.status.success());
            assert_eq!(before, fs::read(&path).unwrap());
            let evaluated = Command::new(env!("CARGO_BIN_EXE_chelis"))
                .args(["eval", "--file"])
                .arg(&path)
                .env_remove("CHELIS_STYLE_GATE_DISABLE")
                .output()
                .unwrap();
            assert!(evaluated.status.success(), "{evaluated:?}");
            assert!(
                String::from_utf8(evaluated.stdout)
                    .unwrap()
                    .contains(&format!("answer = -{magnitude}\n"))
            );
        }
    }
    let (_dir, _path, checked) = format_check(
        "def value(x: i32) -> i32 = x\n@property prop_min forall(x: i32):\n  value(x) >= -2147483648\n",
    );
    assert!(checked.status.success(), "{checked:?}");
}
#[test]
fn positive_overflow_below_minimum_and_explicit_neg_magnitude_remain_rejected() {
    for (dtype, magnitude, below) in [
        ("i8", "128", "129"),
        ("i16", "32768", "32769"),
        ("i32", "2147483648", "2147483649"),
        ("i64", "9223372036854775808", "9223372036854775809"),
    ] {
        for expression in [
            format!("{magnitude}{dtype}"),
            format!("-{below}{dtype}"),
            format!("neg(({magnitude}{dtype}))"),
            format!("-({magnitude}{dtype} + 0{dtype})"),
            format!("-({magnitude}{dtype} : {dtype})"),
            format!("-value({magnitude}{dtype})"),
        ] {
            let source =
                format!("def value(x: {dtype}) -> {dtype} = x\nanswer = value({expression})\n");
            let (_dir, _path, result) = format_check(&source);
            assert!(!result.status.success(), "{source}\n{result:?}");
        }
    }
    // A suffix still fixes the operand's dtype; admission of the default i32
    // minimum cannot turn a callee signature into an implicit widening context.
    let (_dir, _path, result) =
        format_check("def value(x: i64) -> i64 = x\nanswer = value(-2147483648)\n");
    assert!(!result.status.success(), "{result:?}");
}
