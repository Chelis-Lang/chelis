//! A function's discarded trapping node traps in compiled C wherever it
//! traps in `chelis eval`, whether the program's roots are top-level values
//! or a `def main` (chelis#2413, chelis#2440).
//!
//! spec/06 §5.2 makes a potentially trapping node an observable root, and
//! spec/03 §4.4 evaluates a binding's initializer whether or not the binding
//! is read. So `dead = add(copy(v), copy(v))` in `g` overflows on
//! `2147483647i32` however `g` is reached. A `def main` inlines `g` into
//! its own kernel; a top-level value calls `g`'s exported kernel, whose
//! body returns the parameter it was given. That kernel used to be emitted
//! as a pass-through (an identity helper, or a sparse summary for a
//! `gather`), which ran none of the discarded work the evaluator runs.
//!
//! Each family below is one `g`, reached from a top-level value and from a
//! `def main`, with an overflowing and a non-overflowing argument. Every
//! program's eval verdict must equal its compiled C verdict, and the two
//! root forms must agree with each other.
mod common;

use assert_cmd::Command;
use std::path::Path;

const OVERFLOW: &str = "numeric trap: overflow in add at i32";
const OVERFLOWING: &str = "2147483647i32";
const IN_RANGE: &str = "5i32";

/// `g` returns its parameter; the overflow is only in a discarded `let`.
const CALL_TOP: &str = "def g(v: tensor[4, i32]) -> tensor[4, i32] = {
  dead = add(copy(v), copy(v))
  v
}
out = g(to_tensor([1i32, VALUE, 3i32, 4i32]))
";

const CALL_MAIN: &str = "def g(v: tensor[4, i32]) -> tensor[4, i32] = {
  dead = add(copy(v), copy(v))
  v
}
def main() -> tensor[4, i32] = g(to_tensor([1i32, VALUE, 3i32, 4i32]))
";

/// The same body mapped over rows by `vmap`.
const VMAP_TOP: &str = "def g(v: tensor[i32]) -> tensor[i32] = {
  dead = add(copy(v), copy(v))
  v
}
out = vmap(g)(to_tensor([1i32, VALUE, 3i32, 4i32]))
";

const VMAP_MAIN: &str = "def g(v: tensor[i32]) -> tensor[i32] = {
  dead = add(copy(v), copy(v))
  v
}
def main() -> tensor[4, i32] = vmap(g)(to_tensor([1i32, VALUE, 3i32, 4i32]))
";

/// `vmap(g)` reached through a parameter of another definition.
const VMAP_PARAM_TOP: &str = "def g(v: tensor[i32]) -> tensor[i32] = {
  dead = add(copy(v), copy(v))
  v
}
def h(xs: tensor[4, i32]) -> tensor[4, i32] = vmap(g)(xs)
out = h(to_tensor([1i32, VALUE, 3i32, 4i32]))
";

const VMAP_PARAM_MAIN: &str = "def g(v: tensor[i32]) -> tensor[i32] = {
  dead = add(copy(v), copy(v))
  v
}
def h(xs: tensor[4, i32]) -> tensor[4, i32] = vmap(g)(xs)
def main() -> tensor[4, i32] = h(to_tensor([1i32, VALUE, 3i32, 4i32]))
";

/// A `gather` result, which C summarizes as one sparse runtime call.
const GATHER_TOP: &str =
    "def g(v: tensor[3, f32], idx: tensor[2, i64], c: tensor[2, i32]) -> tensor[2, f32] = {
  dead = add(copy(c), copy(c))
  gather(v, idx, 0)
}
out = g(to_tensor([1.0f32, 2.0f32, 3.0f32]), to_tensor([0i64, 2i64]), to_tensor([1i32, VALUE]))
";

const GATHER_MAIN: &str = "def g(v: tensor[3, f32], idx: tensor[2, i64], c: tensor[2, i32]) -> tensor[2, f32] = {
  dead = add(copy(c), copy(c))
  gather(v, idx, 0)
}
def main() -> tensor[2, f32] = g(to_tensor([1.0f32, 2.0f32, 3.0f32]), to_tensor([0i64, 2i64]), to_tensor([1i32, VALUE]))
";

/// What one lane did with a program: the trap line it stopped with, or the
/// value of its last root with the root's name stripped, so a top-level
/// `out` and a `main` compare equal.
#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    Trap(String),
    Value(String),
}

fn chelis(directory: &Path, args: &[&str]) -> std::process::Output {
    Command::cargo_bin("chelis")
        .unwrap()
        .current_dir(directory)
        .args(args)
        .output()
        .unwrap()
}

fn text(output: &std::process::Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn verdict(output: &std::process::Output) -> Verdict {
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let last = stdout
            .lines()
            .last()
            .unwrap_or_else(|| panic!("no root was printed:\n{}", text(output)));
        let (_, value) = last
            .split_once(" = ")
            .unwrap_or_else(|| panic!("`{last}` is not a printed root"));
        Verdict::Value(value.to_string())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let trap = stderr
            .lines()
            .find(|line| line.starts_with("numeric trap:"))
            .unwrap_or_else(|| panic!("failed without a trap line:\n{}", text(output)));
        Verdict::Trap(trap.to_string())
    }
}

/// The file is canonical and lint-clean, so the style gate cannot be what
/// rejects it.
fn assert_canonical(directory: &Path, path: &str, source: &str) {
    let formatted = chelis(directory, &["fmt", "--inplace", path]);
    assert!(formatted.status.success(), "{}", text(&formatted));
    assert_eq!(
        std::fs::read_to_string(directory.join(path)).unwrap(),
        source,
        "the witness must already be canonical Surf"
    );
    let linted = chelis(directory, &["lint", "--check", path]);
    assert!(linted.status.success(), "{}", text(&linted));
}

fn eval_verdict(directory: &Path, path: &str) -> Verdict {
    verdict(&chelis(directory, &["eval", "--file", path]))
}

fn c_verdict(directory: &Path, path: &str, stem: &str) -> Verdict {
    let out = directory.join(format!("{stem}-out"));
    let built = chelis(
        directory,
        &[
            "build",
            path,
            "--target",
            "c",
            "--output",
            out.to_str().unwrap(),
        ],
    );
    assert!(built.status.success(), "{}", text(&built));
    assert!(common::link_generated(&out, &format!("{stem}.c"), stem).success());
    verdict(&std::process::Command::new(out.join(stem)).output().unwrap())
}

/// The eval and C verdicts of `template` with `value` in place of `VALUE`.
fn lane_verdicts(stem: &str, template: &str, value: &str) -> (Verdict, Verdict) {
    let source = template.replace("VALUE", value);
    let directory = tempfile::tempdir().unwrap();
    let path = format!("{stem}.ch");
    std::fs::write(directory.path().join(&path), &source).unwrap();
    assert_canonical(directory.path(), &path, &source);
    (
        eval_verdict(directory.path(), &path),
        c_verdict(directory.path(), &path, stem),
    )
}

/// Both root forms of one family, with both arguments: every program's C
/// verdict equals its eval verdict, the overflowing argument traps, the
/// in-range one returns `expected`, and the top-level and `def main` forms
/// agree.
fn assert_family(stem: &str, top: &str, main: &str, expected: &str) {
    for (value, want) in [
        (OVERFLOWING, Verdict::Trap(OVERFLOW.to_string())),
        (IN_RANGE, Verdict::Value(expected.to_string())),
    ] {
        let (top_eval, top_c) = lane_verdicts(&format!("{stem}_top"), top, value);
        let (main_eval, main_c) = lane_verdicts(&format!("{stem}_main"), main, value);
        assert_eq!(top_eval, want, "eval, top-level value, argument {value}");
        assert_eq!(
            top_c, top_eval,
            "C against eval, top-level value, argument {value}"
        );
        assert_eq!(main_eval, want, "eval, def main, argument {value}");
        assert_eq!(
            main_c, main_eval,
            "C against eval, def main, argument {value}"
        );
        assert_eq!(
            top_c, main_c,
            "C, top-level value against def main, argument {value}"
        );
    }
}

/// Evidentiary status: REGRESSION TEST for the top-level form, whose C
/// returned `[1, 2147483647, 3, 4]` at b47fdd7d3; the `def main` form is a
/// disposition lock.
#[test]
fn a_called_def_s_discarded_overflow_traps_in_c_with_and_without_main() {
    assert_family(
        "call",
        CALL_TOP,
        CALL_MAIN,
        "tensor(shape=[4], data=[1, 5, 3, 4])",
    );
}

/// Evidentiary status: REGRESSION TEST for the top-level form (the round's
/// witness); the `def main` form is a disposition lock.
#[test]
fn a_vmapped_def_s_discarded_overflow_traps_in_c_with_and_without_main() {
    assert_family(
        "vmap",
        VMAP_TOP,
        VMAP_MAIN,
        "tensor(shape=[4], data=[1, 5, 3, 4])",
    );
}

/// Evidentiary status: REGRESSION TEST for the top-level form; the
/// `def main` form is a disposition lock.
#[test]
fn a_vmapped_parameter_s_discarded_overflow_traps_in_c_with_and_without_main() {
    assert_family(
        "vmap_param",
        VMAP_PARAM_TOP,
        VMAP_PARAM_MAIN,
        "tensor(shape=[4], data=[1, 5, 3, 4])",
    );
}

/// Evidentiary status: REGRESSION TEST for the top-level form, whose C ran
/// the sparse summary and returned `[1.0, 3.0]` at b47fdd7d3; the `def main`
/// form is a disposition lock.
#[test]
fn a_gathering_def_s_discarded_overflow_traps_in_c_with_and_without_main() {
    assert_family(
        "gather",
        GATHER_TOP,
        GATHER_MAIN,
        "tensor(shape=[2], data=[1.0, 3.0])",
    );
}
