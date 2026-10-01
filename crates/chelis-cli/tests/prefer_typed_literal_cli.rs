//! `prefer-typed-literal` through the CLI (§12.6).
//!
//! The rule is a non-blocking warning with a lexical autofix. These tests pin
//! the public surface: `chelis lint` lists and reports it without failing
//! `--check`, `chelis check` still accepts the program, and `chelis lint
//! --fix` rewrites every literal cast it reports into a program that formats
//! canonically and evaluates to exactly the same values, while leaving the
//! casts the rule declines untouched.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

const RULE: &str = "prefer-typed-literal";

/// One binding per rewritable class: float and integer bodies, an integer
/// body under every float width, exponent and radix bodies, and the
/// value-sensitive cases (`1.1` at `f64`, `16777217` rounding at `f32`,
/// `257` rounding at `bf16`), each widened to `f64` so evaluation prints
/// the exact bound value.
const REWRITTEN: &str = "\
a = cast(1.1, f64)
b = cast(1.0, f32)
c = cast(0.1, f16)
wide_c = cast(c, f64)
d = cast(0.1, bf16)
wide_d = cast(d, f64)
e = cast(0.001, f64)
f = cast(3, i64)
g = cast(7, i8)
h = cast(1000, i16)
i = cast(5, i32)
j = cast(3000000000, i64)
k = cast(255, i64)
l = cast(16777217, f32)
wide_l = cast(l, f64)
m = cast(9007199254740993, f64)
n = cast(65504, f16)
wide_n = cast(n, f64)
o = cast(257, bf16)
wide_o = cast(o, f64)
t = to_tensor([cast(1.5, f64), cast(2, f64)])
";

const EXPECTED_AFTER_FIX: &str = "\
a = 1.1f64
b = 1.0f32
c = 0.1f16
wide_c = cast(c, f64)
d = 0.1bf16
wide_d = cast(d, f64)
e = 0.001f64
f = 3i64
g = 7i8
h = 1000i16
i = 5i32
j = 3000000000i64
k = 255i64
l = 16777217f32
wide_l = cast(l, f64)
m = 9007199254740993f64
n = 65504f16
wide_n = cast(n, f64)
o = 257bf16
wide_o = cast(o, f64)
t = to_tensor([1.5f64, 2f64])
";

/// Bodies the canonical printer respells. The rewrite keeps the authored
/// body, so these are evaluated past the formatter gate.
const NONCANONICAL: &str = "\
e = cast(1e-3, f64)
k = cast(0xFF, i64)
u = cast(1_000.25, f64)
";

const NONCANONICAL_AFTER_FIX: &str = "\
e = 1e-3f64
k = 0xFFi64
u = 1_000.25f64
";

/// Casts the rule must leave alone, each with its own reason (§12.6).
const DECLINED: &str = "\
negative = cast(-128, i8)
float_body = cast(3.0, i64)
radix_body = cast(0x10, f32)
bf16_zero = cast(0, bf16)
widened = cast(1.0f32, f64)
";

fn chelis() -> Command {
    Command::cargo_bin("chelis").expect("chelis binary")
}

fn eval(path: &Path, extra: &[&str]) -> String {
    let output = chelis()
        .args(["eval", "--file", path.to_str().unwrap()])
        .args(extra)
        .output()
        .expect("chelis eval");
    assert!(
        output.status.success(),
        "chelis eval failed for {}: stderr={}",
        path.display(),
        String::from_utf8_lossy(&output.stderr),
    );
    String::from_utf8(output.stdout).expect("utf8 stdout")
}

#[test]
fn lint_list_reports_prefer_typed_literal_as_a_warning() {
    chelis()
        .args(["lint", "--list"])
        .assert()
        .success()
        .stdout(predicate::str::contains(format!("{RULE}\twarning\t§12.6")));
}

#[test]
fn lint_check_warns_with_a_fix_marker_and_does_not_fail() {
    let dir = tempdir().expect("tempdir");
    fs::write(dir.path().join("main.ch"), "x = cast(1.0, f32)\n").expect("write");
    let output = chelis()
        .args(["lint", "--check", "--rule", RULE])
        .arg(dir.path())
        .output()
        .expect("chelis lint");
    assert!(output.status.success(), "a warning must not fail --check");
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    let lines: Vec<&str> = stdout.lines().filter(|line| line.contains(RULE)).collect();
    assert_eq!(lines.len(), 1, "one site, one report: {stdout}");
    let line = lines[0];
    assert!(line.starts_with("warning: "), "{line}");
    assert!(
        line.contains(&format!("main.ch:1:5: {RULE} (§12.6)")),
        "{line}"
    );
    assert!(line.contains("write `1.0f32`"), "{line}");
    assert!(line.ends_with("[fix]"), "{line}");
    assert!(!stdout.contains("blocking"), "{stdout}");
}

#[test]
fn check_accepts_a_literal_cast_and_prints_the_warning() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("main.ch");
    fs::write(&path, "x = cast(1.0, f32)\n").expect("write");
    chelis()
        .arg("check")
        .arg(&path)
        .assert()
        .success()
        .stderr(predicate::str::contains("warning: ").and(predicate::str::contains(RULE)));
}

#[test]
fn lint_fix_rewrites_every_reported_cast_and_preserves_every_value() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("main.ch");
    fs::write(&path, REWRITTEN).expect("write");
    let before = eval(&path, &[]);

    chelis()
        .args(["lint", "--fix", "--rule", RULE])
        .arg(dir.path())
        .assert()
        .success();
    let rewritten = fs::read_to_string(&path).expect("read rewritten");
    assert_eq!(rewritten, EXPECTED_AFTER_FIX);

    chelis()
        .args(["fmt", "--check"])
        .arg(&path)
        .assert()
        .success();
    assert_eq!(eval(&path, &[]), before, "the rewrite changed a value");
    chelis()
        .args(["lint", "--check", "--rule", RULE])
        .arg(dir.path())
        .assert()
        .success()
        .stdout(predicate::str::contains(RULE).not());
}

#[test]
fn lint_fix_leaves_declined_casts_untouched_and_silent() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("main.ch");
    fs::write(&path, DECLINED).expect("write");
    // `0x10` is not canonical; the formatter would respell it.
    let before = eval(&path, &["--allow-style-violations"]);

    chelis()
        .args(["lint", "--fix", "--rule", RULE])
        .arg(dir.path())
        .assert()
        .success()
        .stdout(predicate::str::contains(RULE).not());
    assert_eq!(fs::read_to_string(&path).expect("read"), DECLINED);
    assert_eq!(eval(&path, &["--allow-style-violations"]), before);
}

#[test]
fn lint_fix_keeps_noncanonical_bodies_and_their_values() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("main.ch");
    fs::write(&path, NONCANONICAL).expect("write");
    let before = eval(&path, &["--allow-style-violations"]);

    chelis()
        .args(["lint", "--fix", "--rule", RULE])
        .arg(dir.path())
        .assert()
        .success();
    assert_eq!(
        fs::read_to_string(&path).expect("read rewritten"),
        NONCANONICAL_AFTER_FIX
    );
    assert_eq!(eval(&path, &["--allow-style-violations"]), before);
}
