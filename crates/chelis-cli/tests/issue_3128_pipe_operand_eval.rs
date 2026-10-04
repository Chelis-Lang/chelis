//! chelis#3128: `chelis fmt` and `chelis surf` keep the grouping a pipe or
//! ascription operand needs, so the printed program evaluates to the same
//! value as the authored one.
//!
//! Each case evaluates the authored program's Deep, the Deep of its
//! `chelis fmt` output, and the Deep of `chelis surf` output, and requires
//! the same value from all three. Evaluating Deep keeps the check independent
//! of how `eval --file` reads Surf.

use std::fs;
use std::path::Path;

use assert_cmd::Command;
use tempfile::tempdir;

const PRELUDE: &str = "def inner(x: f32) -> f32 = (x + 10.0)\n\
def twice(f: f32 -> f32, x: f32) -> f32 = f(f(x))\n\
def pick(c: bool) -> f32 -> f32 = if c then inner else fn (v: f32) -> v\n";

fn chelis(args: &[&str]) -> String {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(args)
        .output()
        .expect("run chelis");
    assert!(
        output.status.success(),
        "chelis {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("utf-8 stdout")
}

fn evaluate_deep(dir: &Path, name: &str, deep: &str) -> String {
    let path = dir.join(format!("{name}.dp"));
    fs::write(&path, deep).expect("write Deep");
    chelis(&["eval", "--file", path.to_str().unwrap()])
}

fn assert_same_value(name: &str, body: &str, expected: &str) {
    let dir = tempdir().expect("tempdir");
    let authored = dir.path().join(format!("{name}.ch"));
    fs::write(&authored, format!("{PRELUDE}def main() -> f32 = {body}\n")).expect("write");
    let authored_deep = chelis(&["deep", authored.to_str().unwrap()]);

    let formatted = dir.path().join(format!("{name}_fmt.ch"));
    fs::copy(&authored, &formatted).expect("copy");
    chelis(&["fmt", "--inplace", formatted.to_str().unwrap()]);
    let formatted_deep = chelis(&["deep", formatted.to_str().unwrap()]);

    let authored_dp = dir.path().join(format!("{name}_authored.dp"));
    fs::write(&authored_dp, &authored_deep).expect("write Deep");
    let resugared = dir.path().join(format!("{name}_surf.ch"));
    fs::write(&resugared, chelis(&["surf", authored_dp.to_str().unwrap()])).expect("write");
    let resugared_deep = chelis(&["deep", resugared.to_str().unwrap()]);

    let line = format!("main = {expected}");
    for (path, deep) in [
        ("authored", &authored_deep),
        ("fmt", &formatted_deep),
        ("surf", &resugared_deep),
    ] {
        let value = evaluate_deep(dir.path(), &format!("{name}_{path}_eval"), deep);
        assert!(
            value.contains(&line),
            "{name} via {path}: expected `{line}`, got `{value}`\nfmt: {}",
            fs::read_to_string(&formatted).unwrap_or_default()
        );
    }
}

#[test]
fn open_tailed_pipe_seeds_keep_their_value() {
    assert_same_value("if_seed", "(if true then 1.0 else 3.0) |> inner", "11.0");
    assert_same_value("fn_seed", "(fn (v: f32) -> v) |> twice(1.0)", "1.0");
    assert_same_value("pipe_seed", "(true |> pick)(1.0) |> inner", "21.0");
}

#[test]
fn open_tailed_pipe_stages_keep_their_value() {
    assert_same_value(
        "if_stage",
        "1.0 |> (if true then inner else fn (v: f32) -> v) |> inner",
        "21.0",
    );
    assert_same_value(
        "fn_stage",
        "1.0 |> (fn (v: f32) -> v * 2.0) |> inner",
        "12.0",
    );
    assert_same_value("pipe_stage", "1.0 |> (true |> pick) |> inner", "21.0");
    assert_same_value("pipe_stage_last", "1.0 |> (false |> pick)", "1.0");
}

#[test]
fn open_tailed_ascription_operands_keep_their_value() {
    assert_same_value("ann_if", "((if false then 1.0 else 2.0) : f32)", "2.0");
    assert_same_value("ann_fn", "((fn (v: f32) -> v) : f32 -> f32)(2.0)", "2.0");
}
