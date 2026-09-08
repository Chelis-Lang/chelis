//! chelis#1587 at the CLI, both ingresses.
//!
//! `def ident(x: i64) -> i64 = x` scored 1.0 on a tree without the alias
//! mapping and accepted `ident(1.5f64)` returning f64, because `i64` fell
//! through the desugarer's primitive test and became `forall a. a`. The
//! signature named a type and meant nothing.

use assert_cmd::Command;
use std::{fs, path::Path};
use tempfile::tempdir;

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
     def main() -> int64 = ident(5i64)\n";

/// Regression test. Red on `4ad308501`, where both units scored 1.0 because
/// `i64` was a quantifier: the mismatch unit is now rejected as int64 against
/// f64, and the twin proves the alias still NAMES int64 rather than merely
/// being rejected everywhere.
#[test]
fn a_short_integer_signature_names_int64_at_both_ingresses() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();

    fs::write(root.join("mismatch.ch"), MISMATCH).expect("write");
    for args in [
        &["check", "mismatch.ch"][..],
        &["eval", "--file", "mismatch.ch"],
    ] {
        let rendered = text(&run(root, args));
        assert!(
            rendered.contains("int64"),
            "`chelis {}` must report the mismatch against int64, not accept a \
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
        "`i64` must still name int64 and accept an int64 argument: {accepted}"
    );
}
