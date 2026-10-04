//! chelis#2893: two uses of one top-level binding share one extent.
//!
//! A top-level tensor whose extent is known only at run time reaches each
//! root that reads it as a parameter `Load`. Compiled C gives every
//! anonymous extent an identity before emission; each `Load` of the binding
//! took a fresh one, so `neq(x, x)` compared two extents the C lane could not
//! prove equal and the build stopped at the ownership invariant, while
//! `chelis eval` ran the program.
//!
//! The positive oracle: the built executable's stdout equals `chelis eval`'s.
//! The negative twins combine two different bindings of different run-time
//! lengths, which must still fail on both lanes: a shared identity belongs to
//! one binding, never to two bindings that both have anonymous extents.
use assert_cmd::Command;
use std::process::Command as StdCommand;

#[path = "common/mod.rs"]
mod common;
use common::{build_and_run_app, make_app, write_file};

/// The issue's two witnesses, with the printed value as a root, and one
/// binding read by two roots.
const POSITIVE: &[(&str, &str)] = &[
    (
        "computed-element",
        "module Demo.Main\nx = to_tensor([div(1.0f32, 0.0f32), 2.0f32])\nb = neq(x, x)\n",
    ),
    (
        "run-time-range",
        "module Demo.Main\nx = cast(to_tensor(range(0i64, 4i64)), f16)\nb = add(copy(x), x)\n",
    ),
    (
        "two-roots",
        "module Demo.Main\nx = to_tensor(range(0i64, 3i64))\nb = add(x, x)\nc = mul(x, copy(x))\n",
    ),
];

/// Each twin, with the line `chelis eval` and the executable report.
const NEGATIVE: &[(&str, &str, &str, &str)] = &[
    (
        "two-run-time-ranges",
        "module Demo.Main\nx = cast(to_tensor(range(0i64, 4i64)), f16)\ny = cast(to_tensor(range(0i64, 3i64)), f16)\nb = add(copy(x), y)\n",
        "tensor shapes must match for elementwise op",
        "elementwise operand shape mismatch",
    ),
    (
        "two-computed-tensors",
        "module Demo.Main\nx = to_tensor([div(1.0f32, 0.0f32), 2.0f32])\ny = to_tensor([div(1.0f32, 0.0f32), 2.0f32, 3.0f32])\nb = neq(x, y)\n",
        "tensor comparison expects matching tensor shape",
        "elementwise operand shape mismatch",
    ),
];

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

// REGRESSION TEST. On `4bb166024` the build refused every positive case
// with "parameter `@chelis_global_78` of the top-level expression has
// inconsistent tensor types".
#[test]
fn two_uses_of_one_binding_build_and_match_eval() {
    for (name, source) in POSITIVE {
        let (_dir, reef_home, app) = make_app(&format!("issue-2893-{name}"));
        let main = app.join("src/main.ch");
        write_file(&main, source);
        let evaluated = chelis(
            &reef_home,
            &app,
            &["eval", "--file", main.to_str().unwrap()],
        );
        assert!(
            evaluated.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&evaluated.stderr)
        );
        let expected = String::from_utf8(evaluated.stdout).unwrap();
        let compiled = build_and_run_app(&reef_home, &app, "main");
        assert_eq!(compiled, expected, "{name}");
    }
}

#[test]
fn two_bindings_of_different_lengths_still_fail_in_both_lanes() {
    for (name, source, evaluator, executable) in NEGATIVE {
        let (_dir, reef_home, app) = make_app(&format!("issue-2893-{name}"));
        let main = app.join("src/main.ch");
        write_file(&main, source);
        let evaluated = chelis(
            &reef_home,
            &app,
            &["eval", "--file", main.to_str().unwrap()],
        );
        let stderr = String::from_utf8_lossy(&evaluated.stderr);
        assert!(!evaluated.status.success(), "{name}: eval accepted");
        assert!(stderr.contains(evaluator), "{name}: eval: {stderr}");
        let out = app.join("out");
        let built = chelis(
            &reef_home,
            &app,
            &[
                "build",
                main.to_str().unwrap(),
                "--target",
                "c",
                "--output",
                out.to_str().unwrap(),
            ],
        );
        assert!(
            built.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&built.stderr)
        );
        let ran = StdCommand::new(out.join("main")).output().unwrap();
        let stderr = String::from_utf8_lossy(&ran.stderr);
        assert!(!ran.status.success(), "{name}: the executable accepted");
        assert!(stderr.contains(executable), "{name}: executable: {stderr}");
    }
}
