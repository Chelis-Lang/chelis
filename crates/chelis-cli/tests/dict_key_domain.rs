//! [05-OP-56]: dictionary keys are string, bool, or any active signed-integer
//! scalar dtype, and every one of them runs on both lanes.
//!
//! The checker admitted only `i64` and `string` keys, and the eval lane
//! refused `bool` keys at run time, although `main` reached that refusal
//! through an unbounded generic `dict_of`. Each row builds a dictionary with
//! a duplicate key, inserts, looks up, removes, and counts its keys, then
//! runs through `chelis eval --file` and compiled C, which must agree with
//! each other and with the atom's answer.
//!
//! The rows print no key order: the atom's canonical observation order is a
//! separate requirement. Unsigned keys have no row, since the `uint*` names
//! are reserved ([04-DTYPE-1]) and no unsigned type reaches `dict_of`.
//!
//! Evidentiary status: REGRESSION TESTS. The `i8`, `i16`, `i32` and `bool`
//! rows fail at `174d0568c`: the first three at the checker, `bool` there too
//! and, with the checker widened alone, in the eval lane.
#![allow(clippy::uninlined_format_args)]
use assert_cmd::Command;
use tempfile::tempdir;

mod common;

fn eval_stdout(program: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("p.ch");
    common::write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    assert!(
        out.status.success(),
        "eval failed:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("utf-8 stdout")
}

fn main_lines(stdout: &str) -> Vec<String> {
    stdout
        .lines()
        .filter(|line| line.starts_with("main"))
        .map(|line| line.trim().to_string())
        .collect()
}

/// `a` appears twice in `dict_of`, so the later entry wins, and `c` is then
/// inserted. For `bool`, `c` repeats `a`, so the insert replaces its value.
fn program(a: &str, b: &str, c: &str) -> String {
    format!(
        "module Keys.Main\n\
export (main)\n\
def found(o: Option[i32]) -> i32 = match o with {{\n\
  | Some(v) => v\n\
  | None => -1i32\n\
}}\n\
def main() -> (i32, i32, i32, bool, bool, i64, i64) = {{\n\
  d = dict_insert(dict_of([({a}, 10i32), ({b}, 20i32), ({a}, 30i32)]), {c}, 40i32)\n\
  (found(dict_get(d, {a})), found(dict_get(d, {b})), found(dict_get(dict_remove(d, {b}), {b})), dict_contains(d, {c}), dict_contains(dict_remove(d, {c}), {c}), len(d), len(dict_keys(d)))\n\
}}\n"
    )
}

fn expected(value_a: i32, count: i64) -> Vec<String> {
    vec![
        format!("main.0 = {value_a}"),
        "main.1 = 20".to_string(),
        "main.2 = -1".to_string(),
        "main.3 = true".to_string(),
        "main.4 = false".to_string(),
        format!("main.5 = {count}"),
        format!("main.6 = {count}"),
    ]
}

#[test]
fn every_admitted_key_dtype_runs_on_both_lanes() {
    if !common::gcc_available() {
        eprintln!("skipping: no C toolchain");
        return;
    }
    for (key, a, b, c, want) in [
        ("string", "\"b\"", "\"a\"", "\"c\"", expected(30, 3)),
        ("bool", "true", "false", "true", expected(40, 2)),
        ("i8", "5i8", "-3i8", "-127i8", expected(30, 3)),
        ("i16", "5i16", "-3i16", "-32767i16", expected(30, 3)),
        ("i32", "5i32", "-3i32", "-2147483647i32", expected(30, 3)),
        (
            "i64",
            "5i64",
            "-3i64",
            "-9223372036854775807i64",
            expected(30, 3),
        ),
    ] {
        let source = program(a, b, c);
        let eval = main_lines(&eval_stdout(&source));
        let compiled = main_lines(&common::build_and_run(&source, &format!("keys_{key}")));
        assert_eq!(eval, want, "eval lane, `{key}` keys:\n{source}");
        assert_eq!(compiled, want, "C lane, `{key}` keys:\n{source}");
    }
}
