//! Float patterns bind at the scrutinee dtype ([04-PAT-1], chelis#2438).
//! Each case runs through the evaluator and the generated C binary.

mod common;

use assert_cmd::Command;
use common::{build_and_run, write_file};

fn evaluate(source: &str, name: &str) -> String {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    write_file(&path, source);
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file"])
        .arg(&path)
        .output()
        .expect("eval runs");
    assert!(
        output.status.success(),
        "eval failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("UTF-8 output")
}

#[test]
fn decimal_float_patterns_match_at_the_scrutinee_width_in_both_lanes() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        let source = format!(
            "def choose(x: {dtype}) -> i32 =\n\
             \x20 match x with {{\n\
             \x20   | 0.1 => 1\n\
             \x20   | _ => 0\n\
             \x20 }}\n\
             def nested(x: {dtype}) -> i32 =\n\
             \x20 match (x, 1) with {{\n\
             \x20   | (0.1, 1) => 1\n\
             \x20   | _ => 0\n\
             \x20 }}\n\
             equal = choose(0.1{dtype})\n\
             other = choose(0.2{dtype})\n\
             casted = choose(cast(0.1f64, {dtype}))\n\
             nested_equal = nested(0.1{dtype})\n\
             nested_other = nested(0.2{dtype})\n"
        );
        let name = format!("float_pattern_{dtype}");
        let interpreted = evaluate(&source, &name);
        assert_eq!(
            interpreted, "equal = 1\nother = 0\ncasted = 1\nnested_equal = 1\nnested_other = 0\n",
            "wrong arm selected at {dtype}"
        );
        assert_eq!(build_and_run(&source, &name), interpreted, "C/eval {dtype}");
    }
}
