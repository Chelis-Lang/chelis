//! chelis#3229: host-lane inlining must respect the scope of match-arm
//! pattern binders.
//!
//! Inlining a callee substitutes its formals (and the callables passed for
//! them) into its body. A match arm whose pattern binds a formal's name
//! shadows that formal in the arm's guard and body, and an arm binder that
//! spells a name free in a substituted callable would capture it. Both are
//! lexical-scope rules the evaluator always honoured; the compiled C lane
//! broke them, so the two lanes printed different answers for one program.
//!
//! Every test runs `eval --file` and `build --target c`, executes the built
//! program, and requires both lanes to print exactly the expected bindings.
//! The controls are the negative direction: a formal that NO arm binder
//! shadows must still be substituted inside the arm, and a capture-free arm
//! binder must not change a program.

#[path = "common/mod.rs"]
mod common;

use assert_cmd::Command;
use std::fs;
use tempfile::tempdir;

const OPTION: &str = "type OptionInt =\n  | Some(i32)\n  | None\n";
const IDENTITY: &str = "def identity(z: i32) -> i32 = z\n";

fn eval(source: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("arm_binders.ch");
    fs::write(&path, source).expect("source");
    let output = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("eval runs");
    assert!(
        output.status.success(),
        "eval rejected: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("UTF-8")
}

/// Both lanes must print exactly `expected`, one `name = value` per line.
fn assert_lanes_print(source: &str, name: &str, expected: &[&str]) {
    let interpreted = eval(source);
    assert_eq!(
        interpreted.lines().collect::<Vec<_>>(),
        expected,
        "{name}: eval output"
    );
    let native = common::build_and_run(source, name);
    assert_eq!(
        native.lines().collect::<Vec<_>>(),
        expected,
        "{name}: compiled C output"
    );
}

#[test]
fn a_constructor_arm_binder_shadows_the_inlined_formal() {
    // The issue's witness: `Some(x)` re-binds the formal `x`, so `f(x)` in
    // that arm reads the payload 7, never the argument 100.
    let source = format!(
        "{OPTION}def pick(f: (i32) -> i32, x: i32, o: OptionInt) -> i32 = match o with {{\n  \
         | Some(x) => f(x)\n  | None => f(x)\n}}\n{IDENTITY}\
         a = pick(identity, 100i32, Some(7i32))\nb = pick(identity, 100i32, None)\n"
    );
    assert_lanes_print(&source, "ctor_shadow", &["a = 7", "b = 100"]);
}

#[test]
fn an_arm_binder_does_not_capture_a_substituted_callables_free_name() {
    // The callable passed for `g` reads the caller's `k`. Inlined under
    // `Some(k)`, the arm binder must be renamed, or `g(k)` computes
    // `1 + 1` instead of `1 + 10`.
    let source = format!(
        "{OPTION}def apply_some(g: (i32) -> i32, o: OptionInt) -> i32 = match o with {{\n  \
         | Some(k) => g(k)\n  | None => 0i32\n}}\n\
         def run(k: i32) -> i32 = apply_some(fn (z: i32) -> add(z, k), Some(1i32))\n\
         out = run(10i32)\n"
    );
    assert_lanes_print(&source, "arm_capture", &["out = 11"]);
}

#[test]
fn a_guard_reads_the_arm_binder_not_the_formal() {
    // The guard is inside the pattern's scope: `gt(x, 5)` tests the payload.
    // With 3 the guard fails and the second arm reads the formal `x` = 100.
    let source = format!(
        "{OPTION}def pick_guard(f: (i32) -> i32, x: i32, o: OptionInt) -> i32 = match o with {{\n  \
         | Some(x) if gt(x, 5i32) => f(x)\n  | Some(y) => f(add(x, y))\n  | None => f(x)\n}}\n\
         {IDENTITY}a = pick_guard(identity, 100i32, Some(7i32))\n\
         b = pick_guard(identity, 100i32, Some(3i32))\n"
    );
    assert_lanes_print(&source, "guard_shadow", &["a = 7", "b = 103"]);
}

#[test]
fn a_guard_does_not_capture_a_substituted_callables_free_name() {
    // The renamed arm binder must reach the guard as well as the body:
    // `g(k)` is `1 + 10` in both, so the first arm is taken.
    let source = format!(
        "{OPTION}def classify(g: (i32) -> i32, o: OptionInt) -> i32 = match o with {{\n  \
         | Some(k) if gt(g(k), 10i32) => g(k)\n  | Some(k) => 0i32\n  | None => -1i32\n}}\n\
         def run(k: i32) -> i32 = classify(fn (z: i32) -> add(z, k), Some(1i32))\n\
         out = run(10i32)\n"
    );
    assert_lanes_print(&source, "guard_capture", &["out = 11"]);
}

#[test]
fn a_nested_arm_binder_shadows_only_its_own_arm() {
    // The inner `Some(x)` shadows the formal; the inner `None` arm and the
    // outer `None` arm still read it.
    let source = format!(
        "{OPTION}def nest(f: (i32) -> i32, x: i32, o: OptionInt, p: OptionInt) -> i32 = \
         match o with {{\n  | Some(a) => match p with {{\n    | Some(x) => f(add(a, x))\n    \
         | None => f(x)\n  }}\n  | None => f(x)\n}}\n{IDENTITY}\
         a = nest(identity, 100i32, Some(7i32), Some(20i32))\n\
         b = nest(identity, 100i32, Some(7i32), None)\n\
         c = nest(identity, 100i32, None, Some(20i32))\n"
    );
    assert_lanes_print(&source, "nested_shadow", &["a = 27", "b = 100", "c = 100"]);
}

#[test]
fn record_tuple_and_as_pattern_binders_shadow_the_formal() {
    let source = format!(
        "type Pt =\n  | Pt {{ x: i32, y: i32 }}\n{IDENTITY}\
         def rec(f: (i32) -> i32, x: i32, p: Pt) -> i32 = match p with {{\n  \
         | Pt {{ x, y }} => f(sub(x, y))\n}}\n\
         def tup(f: (i32) -> i32, x: i32, t: (i32, i32)) -> i32 = match t with {{\n  \
         | (x, y) => f(sub(x, y))\n}}\n\
         def asp(f: (i32) -> i32, x: i32, n: i32) -> i32 = match n with {{\n  \
         | x @ 0 => f(sub(x, 1i32))\n  | x => f(x)\n}}\n\
         a = rec(identity, 100i32, Pt {{ x: 9i32, y: 2i32 }})\n\
         b = tup(identity, 100i32, (9i32, 2i32))\n\
         c = asp(identity, 100i32, 0i32)\n\
         d = asp(identity, 100i32, 5i32)\n"
    );
    assert_lanes_print(
        &source,
        "pattern_forms",
        &["a = 7", "b = 7", "c = -1", "d = 5"],
    );
}

#[test]
fn a_local_callable_inlined_under_an_arm_keeps_its_captured_name() {
    // Local-callable inlining substitutes `f`'s lambda into the arm, where
    // `Some(k)` would capture the lambda's `k` (the parameter, 10).
    let source = format!(
        "{OPTION}def run(o: OptionInt, k: i32) -> i32 = {{\n  \
         f = fn (z: i32) -> add(z, k)\n  match o with {{\n    | Some(k) => f(k)\n    \
         | None => f(0i32)\n  }}\n}}\n\
         a = run(Some(1i32), 10i32)\nb = run(None, 10i32)\n"
    );
    assert_lanes_print(&source, "local_callable_capture", &["a = 11", "b = 10"]);
}

#[test]
fn unshadowed_formals_and_capture_free_binders_are_still_substituted() {
    // Controls: no arm binder spells the formal `x` or the captured `k`, so
    // every occurrence must still be replaced by the caller's value.
    let source = format!(
        "{OPTION}{IDENTITY}def ctrl(f: (i32) -> i32, x: i32, o: OptionInt) -> i32 = \
         match o with {{\n  | Some(y) => f(add(x, y))\n  | None => f(x)\n}}\n\
         def apply_v(g: (i32) -> i32, o: OptionInt) -> i32 = match o with {{\n  \
         | Some(v) => g(v)\n  | None => 0i32\n}}\n\
         def run(k: i32) -> i32 = apply_v(fn (z: i32) -> add(z, k), Some(1i32))\n\
         a = ctrl(identity, 100i32, Some(7i32))\nb = ctrl(identity, 100i32, None)\n\
         c = run(10i32)\n"
    );
    assert_lanes_print(&source, "controls", &["a = 107", "b = 100", "c = 11"]);
}
