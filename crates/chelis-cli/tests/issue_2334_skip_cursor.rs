//! chelis#2334: a List whose last use is `skip` moves into the builtin
//! instead of being cloned, and never changes what a program means.
//!
//! Three halves, in the shape chelis#2205 established for `append`.
//!
//! The alias controls pin that every program in which the skipped list
//! is still observable somewhere else keeps its value on both lanes and
//! never aborts: a seed walked twice, a tuple-held alias skipped in one
//! branch, a nested list whose inner list is skipped, and an empty typed
//! seed skipped past its end. Those are the shapes on which a verified
//! Move meets a strong-owner count above one, which is why the consuming
//! entry point re-checks uniqueness at run time.
//!
//! The receipt is the number of cloning `chelis_list_drop(` calls a
//! let-bound skip chain emits, which is zero, against the consuming
//! calls that replaced them.
//!
//! The third half is the limit, and it is asserted rather than described.
//! The move needs the operand's terminal directly after the application,
//! so a cursor that binds its head before recursing gets the consuming
//! call and one that reads the head inside a later argument of the same
//! call does not. Both must compute the same answer; only one is cheap.
//!
//! Evidentiary status: REGRESSION TEST for the two receipt rows, which
//! count a symbol that does not exist before this change set. The
//! value-parity rows are LOCKS: they pass on `main` too, which is the
//! point.
#![allow(clippy::uninlined_format_args)]
use assert_cmd::Command;
use tempfile::tempdir;

mod common;

fn c_toolchain_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn write_file(path: &std::path::Path, contents: &str) {
    std::fs::write(path, contents).expect("write fixture");
}

fn eval_main(program: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("p.ch");
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    assert!(
        out.status.success(),
        "eval failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    stdout
        .lines()
        .find(|line| line.starts_with("main = "))
        .unwrap_or_else(|| panic!("no `main = ` line in:\n{stdout}"))
        .trim()
        .to_string()
}

/// Build through the C lane; returns the generated C and the binary's
/// `main = ` line. A runtime abort (the in-place guard firing on a
/// shared list) surfaces as a non-zero exit and fails here.
fn c_main(program: &str, name: &str) -> (String, String) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, program);
    let built = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("chelis build should run");
    assert!(
        built.status.success(),
        "build failed:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let generated =
        std::fs::read_to_string(out_dir.join(format!("{name}.c"))).expect("read generated C");
    let status = common::link_generated(&out_dir, &format!("{name}.c"), name);
    assert!(status.success(), "link failed: {status}");
    let run = std::process::Command::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    assert!(
        run.status.success(),
        "compiled `{name}` exited with {}; a fired exclusivity guard is a bug, not a lock:\n{}",
        run.status,
        String::from_utf8_lossy(&run.stderr)
    );
    let stdout = String::from_utf8_lossy(&run.stdout);
    let line = stdout
        .lines()
        .find(|line| line.starts_with("main = "))
        .unwrap_or_else(|| panic!("no `main = ` line in:\n{stdout}"))
        .trim()
        .to_string();
    (generated, line)
}

/// The cursor shape every measured caller has: the head is bound before
/// the recursive call, so the list's last use is the `skip`.
const CURSOR: &str = "def walk(xs: List[i64], acc: i64) -> i64 = \
if eq(len(xs), cast(0, i64)) then acc else {\n\
  head = index(xs, cast(0, i64))\n\
  walk(skip(xs, cast(1, i64)), add(acc, head))\n\
}\n";

/// The same walk with the head read inside a later argument of the same
/// call. The operand is still live when the skip runs, so the scheduler
/// cannot upgrade it and the cloning path stays.
const INLINE_CURSOR: &str = "def walk_inline(xs: List[i64], acc: i64) -> i64 = \
if eq(len(xs), cast(0, i64)) then acc else \
walk_inline(skip(xs, cast(1, i64)), add(acc, index(xs, cast(0, i64))))\n";

const SHARED_SEED: &str = "module Ac.Main\n\
export (main)\n\
def walk(xs: List[i64], acc: i64) -> i64 = \
if eq(len(xs), cast(0, i64)) then acc else {\n\
  head = index(xs, cast(0, i64))\n\
  walk(skip(xs, cast(1, i64)), add(acc, head))\n\
}\n\
def main() -> i64 = {\n\
  seed = [cast(1, i64), cast(2, i64), cast(3, i64)]\n\
  a = walk(seed, cast(0, i64))\n\
  b = walk(seed, cast(0, i64))\n\
  add(add(mul(a, cast(100, i64)), mul(b, cast(10, i64))), len(seed))\n\
}\n";

const BRANCHED_ALIAS: &str = "module Ac.Main\n\
export (main)\n\
def main() -> i64 = {\n\
  xs = [cast(1, i64), cast(2, i64), cast(3, i64)]\n\
  held = (xs, cast(0, i64))\n\
  ys = if gt(len(xs), cast(1, i64)) then skip(xs, cast(1, i64)) else xs\n\
  add(mul(len(ys), cast(10, i64)), len(held.0))\n\
}\n";

const NESTED_LIST: &str = "module Ac.Main\n\
export (main)\n\
def main() -> i64 = {\n\
  inner = [cast(1, i64), cast(2, i64), cast(3, i64)]\n\
  outer = [inner, inner]\n\
  tail = skip(inner, cast(1, i64))\n\
  add(add(mul(len(tail), cast(100, i64)), mul(len(outer), cast(10, i64))), len(inner))\n\
}\n";

const EMPTY_SEED: &str = "module Ac.Main\n\
export (main)\n\
def main() -> i64 = {\n\
  xs: List[i64] = []\n\
  ys = skip(xs, cast(4, i64))\n\
  zs = skip(ys, cast(1, i64))\n\
  add(mul(len(ys), cast(10, i64)), len(zs))\n\
}\n";

/// A skip whose count exceeds the length, on a list still held elsewhere.
const OVER_COUNT_HELD: &str = "module Ac.Main\n\
export (main)\n\
def main() -> i64 = {\n\
  xs = [cast(1, i64), cast(2, i64)]\n\
  ys = skip(xs, cast(9, i64))\n\
  add(mul(len(xs), cast(10, i64)), len(ys))\n\
}\n";

/// An N-step let-bound skip chain: every step's input is dead after it.
fn skip_chain(steps: usize) -> String {
    let elements = (0..=steps)
        .map(|value| format!("cast({value}, i64)"))
        .collect::<Vec<_>>()
        .join(", ");
    let mut source =
        format!("module Ac.Main\nexport (main)\ndef main() -> i64 = {{\n  x0 = [{elements}]\n");
    for step in 1..=steps {
        source.push_str(&format!("  x{step} = skip(x{}, cast(1, i64))\n", step - 1));
    }
    source.push_str(&format!("  len(x{steps})\n}}\n"));
    source
}

#[test]
fn alias_controls_keep_their_values_on_both_lanes() {
    if !c_toolchain_available() {
        eprintln!("skipping: no C toolchain");
        return;
    }
    for (name, program, expected) in [
        ("skip_shared_seed", SHARED_SEED, "main = 663"),
        ("skip_branched_alias", BRANCHED_ALIAS, "main = 23"),
        ("skip_nested_list", NESTED_LIST, "main = 223"),
        ("skip_empty_seed", EMPTY_SEED, "main = 0"),
        ("skip_over_count_held", OVER_COUNT_HELD, "main = 20"),
    ] {
        let evaluated = eval_main(program);
        assert_eq!(evaluated, expected, "{name}: eval value");
        let (_, compiled) = c_main(program, name);
        assert_eq!(compiled, expected, "{name}: compiled value");
    }
}

#[test]
fn a_let_bound_skip_chain_does_not_clone_per_step() {
    if !c_toolchain_available() {
        eprintln!("skipping: no C toolchain");
        return;
    }
    fn counts(steps: usize) -> (usize, usize) {
        let (generated, value) = c_main(&skip_chain(steps), &format!("skipchain{steps}"));
        assert_eq!(value, "main = 1", "chain of {steps} skips");
        (
            generated.matches(" = chelis_list_drop(").count(),
            generated.matches(" = chelis_list_drop_owned(").count(),
        )
    }
    let (small_cloning, small_owned) = counts(8);
    let (large_cloning, large_owned) = counts(16);
    eprintln!(
        "#2334 receipt: 8-step chain emits {small_cloning} cloning / {small_owned} consuming \
         skips, 16-step chain emits {large_cloning} cloning / {large_owned} consuming"
    );
    assert_eq!(
        (small_cloning, large_cloning),
        (0, 0),
        "#2334: a let-bound chain never clones; the exact count is a count of emitted calls \
         and carries no machine budget"
    );
    // The positive half. Without it, an emitter that dropped the call
    // entirely would satisfy the zero above.
    assert_eq!(
        (small_owned, large_owned),
        (8, 16),
        "#2334: every step of the chain emits the consuming call"
    );
}

/// The coverage boundary, executed rather than described.
///
/// The move needs the operand's `Drop` terminal directly after the
/// application. A cursor that binds its head first gets it; a cursor that
/// reads the head in a later argument of the same call does not, because
/// the operand is still live when the skip runs. Both answers must agree,
/// and they must be emitted differently, or the claim that this change
/// makes cursors linear would be unqualified and wrong.
#[test]
fn the_move_needs_the_terminal_after_the_application() {
    if !c_toolchain_available() {
        eprintln!("skipping: no C toolchain");
        return;
    }
    let bound = format!(
        "module Ac.Main\nexport (main)\n{CURSOR}\
         def main() -> i64 = walk([cast(1, i64), cast(2, i64), cast(3, i64)], cast(0, i64))\n"
    );
    let inline = format!(
        "module Ac.Main\nexport (main)\n{INLINE_CURSOR}\
         def main() -> i64 = walk_inline([cast(1, i64), cast(2, i64), cast(3, i64)], cast(0, i64))\n"
    );

    let (bound_c, bound_value) = c_main(&bound, "skip_cursor_bound");
    let (inline_c, inline_value) = c_main(&inline, "skip_cursor_inline");
    assert_eq!(bound_value, "main = 6", "the bound-head cursor's answer");
    assert_eq!(
        inline_value, bound_value,
        "both cursor spellings compute the same sum"
    );
    assert_eq!(
        eval_main(&bound),
        bound_value,
        "eval agrees with the C lane"
    );
    assert_eq!(
        eval_main(&inline),
        inline_value,
        "eval agrees on the inline"
    );

    // The call spelling, not the bare symbol. Every translation unit's
    // prelude declares `chelis_list *chelis_list_drop_owned(...)`
    // whether or not it calls it, so a bare-name search answers "is the
    // prototype present" and never "which entry point did the emitter
    // select". Leading ` = ` is what makes it a call site.
    assert!(
        bound_c.contains(" = chelis_list_drop_owned("),
        "the bound-head cursor is the shape the move reaches:\n{bound_c}"
    );
    assert!(
        !inline_c.contains(" = chelis_list_drop_owned("),
        "the operand is still live at the inline cursor's skip, so it stays on the \
         cloning path; this is the documented limit, not a regression:\n{inline_c}"
    );
    assert!(
        inline_c.contains(" = chelis_list_drop("),
        "the inline cursor still emits the cloning call:\n{inline_c}"
    );
}
