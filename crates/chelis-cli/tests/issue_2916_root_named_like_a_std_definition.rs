//! chelis#2916: a root whose name is also the last segment of a standard
//! library definition is its own declaration.
//!
//! Host lowering finds a declaration's signature by name, and a reference
//! spelled short may match a declaration by its last `__` segment. A root
//! named `period_text` exactly names its own declaration, which has no
//! authored function signature, so the search fell through to the short-name
//! match and found `Std.Datetime`'s two-parameter `period_text`. Compiled C
//! then refused the root as still requiring two generated parameters, while
//! `chelis eval` printed it. A second root, `overflow_failure`, collides
//! with another library definition the program reaches, so the rule is
//! tested on the name class, not on one name.
use assert_cmd::Command;

#[path = "common/mod.rs"]
mod common;
use common::{build_and_run_app, make_app, write_file};

fn chelis(
    reef_home: &std::path::Path,
    app: &std::path::Path,
    args: &[&str],
) -> std::process::Output {
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(app)
        .args(args)
        .output()
        .unwrap()
}

// REGRESSION TEST. On `4bb166024` the build refused the root with "callable
// `..period_text` still requires generated Host parameter(s) `arg0`, `arg1`".
#[test]
fn a_root_named_like_a_std_definition_builds_and_matches_eval() {
    let (_dir, reef_home, app) = make_app("issue-2916-root-name");
    let main = app.join("src/main.ch");
    write_file(
        &main,
        "module Demo.Main\nimport Std.Datetime (period, period_negate, period_to_string)\nperiod_text = period_to_string(period(0i64, 1i64))\noverflow_failure = period_to_string(period_negate(period(1i64, 2i64)))\n",
    );
    let evaluated = chelis(
        &reef_home,
        &app,
        &["eval", "--file", main.to_str().unwrap()],
    );
    assert!(
        evaluated.status.success(),
        "{}",
        String::from_utf8_lossy(&evaluated.stderr)
    );
    let expected = String::from_utf8(evaluated.stdout).unwrap();
    assert!(expected.contains("period_text = P1D"), "{expected}");
    assert!(expected.contains("overflow_failure = -P1M2D"), "{expected}");
    let compiled = build_and_run_app(&reef_home, &app, "main");
    assert_eq!(compiled, expected);
}

/// The negative twin: the root's own declaration still governs a call of it,
/// so applying the value to the std definition's arguments is refused on
/// both lanes rather than resolved to `Std.Datetime`'s function.
#[test]
fn a_root_named_like_a_std_definition_is_not_that_definition() {
    let (_dir, reef_home, app) = make_app("issue-2916-root-name-call");
    let main = app.join("src/main.ch");
    write_file(
        &main,
        "module Demo.Main\nimport Std.Datetime (period, period_to_string)\nperiod_text = 1i64\nbad = period_text(0i64, 1i64)\n",
    );
    let out = app.join("out");
    for args in [
        vec!["eval", "--file", main.to_str().unwrap()],
        vec![
            "build",
            main.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out.to_str().unwrap(),
        ],
    ] {
        let output = chelis(&reef_home, &app, &args);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "{args:?} accepted the call");
        assert!(
            stderr.contains("type mismatch: i64 vs (i64, i64)"),
            "{args:?}: {stderr}"
        );
    }
}
