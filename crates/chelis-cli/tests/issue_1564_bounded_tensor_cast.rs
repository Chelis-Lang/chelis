//! Execute [05-OP-63]'s bounded-target tensor casts in eval and generated C.
use assert_cmd::Command;
use std::fs;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

fn eval(source: &str) -> std::process::Output {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("cast.ch");
    fs::write(&path, source).expect("source");
    Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("eval")
}

#[test]
fn bounded_tensor_casts_match_generated_c_at_each_target_dtype() {
    for (family, dtypes) in [
        ("Float", ["f32", "f64"]),
        ("Float", ["f16", "bf16"]),
        ("Int", ["int16", "int64"]),
        ("Int", ["int8", "int32"]),
        ("Numeric", ["f32", "int64"]),
    ] {
        for dtype in dtypes {
            let (input, expected) = match dtype {
                "f32" => (16777217, 16777216.0),
                "f64" => (16777217, 16777217.0),
                _ => (123, 123.0),
            };
            let source = format!(
                "module Bounded.Main\nexport (main)\ndef convert[p: {family}](x: tensor[2, int32], witness: p) -> tensor[2, p] = cast(x, p)\ndef recast[p: {family}](x: tensor[2, p]) -> tensor[2, p] = cast(x, p)\ndef main() -> tensor[2, {dtype}] = recast(convert(to_tensor([{input}, -12]), cast(0, {dtype})))\n"
            );
            let interpreted = eval(&source);
            assert!(
                interpreted.status.success(),
                "{}",
                String::from_utf8_lossy(&interpreted.stderr)
            );
            let interpreted = String::from_utf8(interpreted.stdout).expect("UTF-8");
            let native = common::build_and_run(&source, &format!("bounded_{family}_{dtype}"));
            assert_eq!(
                common::parse_tensor_data(&interpreted, "main"),
                vec![expected, -12.0]
            );
            assert_eq!(native.trim(), interpreted.trim(), "{family}/{dtype}");
        }
    }
}

#[test]
fn bounded_truncating_tensor_cast_matches_generated_c() {
    let source = "module Bounded.Main\nexport (main)\ndef convert[p: Float, q: Int](x: tensor[2, p], witness: q) -> tensor[2, q] = cast_trunc(x, q)\ndef main() -> tensor[2, int64] = convert(to_tensor([1.5f32, -2.5f32]), 0i64)\n";
    let interpreted = eval(source);
    assert!(
        interpreted.status.success(),
        "{}",
        String::from_utf8_lossy(&interpreted.stderr)
    );
    let interpreted = String::from_utf8(interpreted.stdout).unwrap();
    let native = common::build_and_run(source, "bounded_trunc");
    assert_eq!(
        common::parse_tensor_data(&interpreted, "main"),
        vec![1.0, -2.0]
    );
    assert_eq!(native.trim(), interpreted.trim());
}

#[test]
fn one_program_keeps_distinct_instantiations_of_the_same_cast() {
    let source = "module Bounded.Main\nexport (main)\ndef convert[p: Float](x: tensor[2, int32], witness: p) -> tensor[2, p] = cast(x, p)\ndef main() -> tensor[2, f64] = add(cast(convert(to_tensor([16777217, -12]), 0.0f32), f64), convert(to_tensor([16777217, -12]), 0.0f64))\n";
    let interpreted = eval(source);
    assert!(
        interpreted.status.success(),
        "{}",
        String::from_utf8_lossy(&interpreted.stderr)
    );
    let interpreted = String::from_utf8(interpreted.stdout).unwrap();
    let native = common::build_and_run(source, "distinct_cast_instantiations");
    assert_eq!(
        common::parse_tensor_data(&interpreted, "main"),
        vec![33554433.0, -24.0]
    );
    assert_eq!(native.trim(), interpreted.trim());
}

#[test]
fn invalid_bounded_tensor_calls_reject_before_either_execution_lane() {
    for body in [
        "def f[p](x: tensor[2, p]) -> tensor[2, p] = cast(x, p)\ndef main() -> tensor[2, int32] = f(to_tensor([1, 2]))\n",
        "def f[p: Float](x: tensor[2, p]) -> tensor[2, p] = cast(x, p)\ndef main() -> tensor[2, int32] = f(to_tensor([1, 2]))\n",
    ] {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("bad.ch");
        fs::write(&path, format!("module Bad.Main\nexport (main)\n{body}")).unwrap();
        for command in ["check", "eval", "build"] {
            let mut cli = Command::cargo_bin("chelis").unwrap();
            cli.env("CHELIS_STYLE_GATE_DISABLE", "1").arg(command);
            if command == "eval" {
                cli.arg("--file");
            }
            cli.arg(&path);
            if command == "build" {
                cli.args(["--target", "c", "--output"])
                    .arg(dir.path().join("out"));
            }
            let output = cli.output().unwrap();
            assert!(!output.status.success(), "{command} accepted {body}");
            let diagnostic = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(
                diagnostic.contains("04-DTYPE-1") || diagnostic.contains("Float"),
                "{diagnostic}"
            );
        }
    }
}
