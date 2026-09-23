//! `match` arm selection in both execution lanes: [04-PAT-2].
//!
//! chelis#2445: a guard was never evaluated. `chelis eval` selected a guarded
//! arm whenever its pattern matched, and host lowering for C dropped guarded
//! and variable arms, let the last wildcard or `Some` arm win, or selected the
//! first constructor arm whose pattern matched.
//!
//! chelis#2446: host lowering sent only `i64`, `f64`, `f32`, `bool` and
//! `string` scrutinees to its literal path, so a match on any other scalar or
//! on a tuple fell into the Option path and was rejected as "an Option match
//! has no `Some` arm".
//!
//! chelis#2450: the Option and user-constructor path read only a top-level
//! variable under each constructor, so compiled C ignored a nested
//! sub-pattern and selected the arm anyway.
//!
//! Every program below runs in both lanes. Each parity test asserts that the
//! two outputs are byte-identical AND that eval prints the values the rule
//! gives, so a lane cannot agree with the other on a wrong answer. The claim is
//! the programs listed here, not every pattern a `match` can hold.

mod common;

use assert_cmd::Command;
use common::{authored_host_body_definition, build_and_run, link_generated, write_file};
use std::process::{Command as StdCommand, Output};

fn eval_output(source: &str, name: &str) -> Output {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    write_file(&path, source);
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file"])
        .arg(path)
        .output()
        .expect("eval should run")
}

fn evaluate(source: &str, name: &str) -> String {
    let output = eval_output(source, name);
    assert!(
        output.status.success(),
        "eval failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("UTF-8 stdout")
}

/// Build to C and return the build's output without asserting on it.
fn build_output(source: &str, name: &str) -> Output {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    write_file(&path, source);
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build", path.to_str().unwrap(), "--target", "c", "--output"])
        .arg(dir.path().join(format!("{name}-out")))
        .output()
        .expect("build should run")
}

/// Build to C and return the emitted body of the authored definition `def`.
fn emitted_body(source: &str, name: &str, def: &str) -> String {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, source);
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build", path.to_str().unwrap(), "--target", "c", "--output"])
        .arg(&out_dir)
        .assert()
        .success();
    let emitted = std::fs::read_to_string(out_dir.join(format!("{name}.c"))).expect("emitted C");
    let rest = authored_host_body_definition(&emitted, def);
    let end = rest.find("\n}\n").expect("the definition is terminated");
    rest[..end].to_string()
}

/// Build, link and run a program that is expected to trap: the build and
/// the link must succeed, and the binary's output is returned unasserted.
fn run_compiled(source: &str, name: &str) -> Output {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, source);
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build", path.to_str().unwrap(), "--target", "c", "--output"])
        .arg(&out_dir)
        .assert()
        .success();
    let status = link_generated(&out_dir, &format!("{name}.c"), name);
    assert!(status.success(), "link failed: {status}");
    StdCommand::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run")
}

/// Run `source` in both lanes, require byte-identical output, and require
/// every `expected` line in it.
fn assert_lanes_print(source: &str, name: &str, expected: &[&str]) {
    let interpreted = evaluate(source, name);
    let compiled = build_and_run(source, name);
    assert_eq!(
        compiled, interpreted,
        "eval/C output must be byte-identical for:\n{source}"
    );
    for line in expected {
        assert!(
            interpreted.lines().any(|printed| printed == *line),
            "expected `{line}` in:\n{interpreted}"
        );
    }
}

/// REGRESSION TEST (chelis#2445's scalar rows). On the base sha (53ecab5a6)
/// eval printed `b = true, e = 50, w2 = 1, w3 = 1, o2 = 1` and C printed
/// `a = false, d = 0, w1 = 3, w2 = 3, o1 = 0`. `bucket` also covers a first
/// guard that is `false` so that a later arm must win.
#[test]
fn guarded_scalar_arms_select_by_their_guard_in_eval_and_c() {
    let source = r#"
def is_zero(x: f32) -> bool =
  match x with {
    | v if eq(v, 0.0) => true
    | _ => false
  }
def is_two(x: i64) -> i64 =
  match x with {
    | 2 => 20i64
    | v if gt(v, 5i64) => 50i64
    | _ => 0i64
  }
def bucket(x: f32) -> i32 =
  match x with {
    | _ if gt(x, 10.0) => 1
    | _ if gt(x, 5.0) => 2
    | _ => 3
  }
def above(x: f32, limit: f32) -> i32 =
  match x with {
    | v if gt(v, limit) => 1
    | _ => 0
  }
a = is_zero(0.0)
b = is_zero(3.0)
c = is_two(2i64)
d = is_two(7i64)
e = is_two(3i64)
w1 = bucket(20.0)
w2 = bucket(7.0)
w3 = bucket(1.0)
o1 = above(9.0, 3.0)
o2 = above(1.0, 3.0)
"#;
    assert_lanes_print(
        source,
        "guarded_scalar_arms",
        &[
            "a = true",
            "b = false",
            "c = 20",
            "d = 50",
            "e = 0",
            "w1 = 1",
            "w2 = 2",
            "w3 = 3",
            "o1 = 1",
            "o2 = 0",
        ],
    );
}

/// REGRESSION TEST (chelis#2445's constructor, List and tuple rows). On the
/// base sha eval printed `opt_b = 1, adt_b = 1, list_b = 1, tuple_b = 1`, and
/// C rejected the program at its tuple match (chelis#2446).
#[test]
fn guarded_constructor_list_and_tuple_arms_select_by_their_guard_in_eval_and_c() {
    let source = r#"
type Shape =
  | Circle(f32)
  | Square(f32)
def big(o: Option[i64]) -> i32 =
  match o with {
    | Some(v) if gt(v, 10i64) => 1
    | Some(v) => 2
    | None => 3
  }
def classify(s: Shape) -> i32 =
  match s with {
    | Circle(r) if gt(r, 1.0) => 1
    | Circle(r) => 2
    | Square(w) => 3
  }
def head_class(xs: List[i64]) -> i32 =
  match xs with {
    | Cons(h, rest) if gt(h, 5i64) => 1
    | Cons(h, rest) => 2
    | Nil => 3
  }
def order(p: (f32, f32)) -> i32 =
  match p with {
    | (a, b) if gt(a, b) => 1
    | (a, b) => 2
  }
opt_a = big(Some(20i64))
opt_b = big(Some(5i64))
opt_c = big(None)
adt_a = classify(Circle(5.0))
adt_b = classify(Circle(0.5))
adt_c = classify(Square(2.0))
list_a = head_class([9i64, 1i64])
list_b = head_class([2i64, 1i64])
list_c = head_class([])
tuple_a = order((5.0, 1.0))
tuple_b = order((1.0, 5.0))
"#;
    assert_lanes_print(
        source,
        "guarded_constructor_arms",
        &[
            "opt_a = 1",
            "opt_b = 2",
            "opt_c = 3",
            "adt_a = 1",
            "adt_b = 2",
            "adt_c = 3",
            "list_a = 1",
            "list_b = 2",
            "list_c = 3",
            "tuple_a = 1",
            "tuple_b = 2",
        ],
    );
}

/// REGRESSION TEST (chelis#2445, chelis#2450). A guard reads bindings from
/// nested record, constructor, as- and List patterns, and a nested literal or
/// constructor sub-pattern constrains its arm. On the base sha eval printed
/// `b = 1, c = 1, f = 0, g = 4, j = 1, m = -5`, and C rejected the program.
#[test]
fn a_guard_reads_nested_pattern_bindings_in_eval_and_c() {
    let source = r#"
type Shape =
  | Circle(f32)
  | Rect { w: f32, h: f32 }
type Wrap =
  | Wrap(Option[i64])
def area_class(s: Shape) -> i32 =
  match s with {
    | Rect { w, h } if gt(mul(w, h), 10.0) => 1
    | Rect { w, h } if eq(w, h) => 2
    | Rect { w, h } => 3
    | Circle(r) => 4
  }
def unwrap_big(w: Wrap) -> i64 =
  match w with {
    | Wrap(Some(v)) if gt(v, 10i64) => v
    | Wrap(Some(0)) => -1i64
    | Wrap(Some(v)) => add(v, 100i64)
    | Wrap(None) => 0i64
  }
def as_guard(o: Option[i64]) -> i64 =
  match o with {
    | whole @ Some(v) if gt(v, 3i64) => v
    | whole @ Some(_) => 2i64
    | None => 3i64
  }
def list_pair(xs: List[i64]) -> i64 =
  match xs with {
    | Cons(a, Cons(b, _)) if gt(a, b) => sub(a, b)
    | Cons(a, Cons(b, _)) => sub(b, a)
    | _ => 0i64
  }
a = area_class(Rect { w: 5.0, h: 5.0 })
b = area_class(Rect { w: 2.0, h: 2.0 })
c = area_class(Rect { w: 1.0, h: 2.0 })
d = area_class(Circle(1.0))
e = unwrap_big(Wrap(Some(20i64)))
f = unwrap_big(Wrap(Some(0i64)))
g = unwrap_big(Wrap(Some(4i64)))
h = unwrap_big(Wrap(None))
i = as_guard(Some(9i64))
j = as_guard(Some(1i64))
k = as_guard(None)
l = list_pair([9i64, 4i64])
m = list_pair([4i64, 9i64])
n = list_pair([4i64])
"#;
    assert_lanes_print(
        source,
        "nested_guard_bindings",
        &[
            "a = 1", "b = 2", "c = 3", "d = 4", "e = 20", "f = -1", "g = 104", "h = 0", "i = 9",
            "j = 2", "k = 3", "l = 5", "m = 5", "n = 0",
        ],
    );
}

/// REGRESSION TEST (chelis#2450). No guard: a nested sub-pattern under `Some`
/// or a user constructor constrains its arm. On the base sha C printed
/// `nested = 0` and `wrapped = 0` while eval printed 20 and 1.
#[test]
fn a_nested_sub_pattern_constrains_its_arm_in_eval_and_c() {
    let source = r#"
type Wrap =
  | Wrap(Option[i64])
def inner(o: Option[Option[i64]]) -> i64 =
  match o with {
    | Some(Some(v)) => v
    | Some(None) => 0i64
    | None => -1i64
  }
def wrapped_none(w: Wrap) -> i64 =
  match w with {
    | Wrap(None) => 0i64
    | _ => 1i64
  }
nested = inner(Some(Some(20i64)))
nested_none = inner(Some(None))
absent = inner(None)
wrapped = wrapped_none(Wrap(Some(20i64)))
wrapped_empty = wrapped_none(Wrap(None))
"#;
    assert_lanes_print(
        source,
        "nested_sub_patterns",
        &[
            "nested = 20",
            "nested_none = 0",
            "absent = -1",
            "wrapped = 1",
            "wrapped_empty = 0",
        ],
    );
}

/// REGRESSION TEST (chelis#2446, chelis#2445). A match on each scalar dtype
/// the literal path did not list, with and without a guard, builds, runs and
/// agrees with eval. On the base sha every program was rejected at build with
/// "an Option match has no `Some` arm", and eval printed `f = 50` or `f = 30`
/// because it ignored the guard.
#[test]
fn every_scalar_dtype_scrutinee_lowers_in_c_and_agrees_with_eval() {
    for dtype in ["i8", "i16", "i32"] {
        let source = format!(
            r#"
def pick(x: {dtype}) -> i32 =
  match x with {{
    | 2 => 20
    | -3 => 30
    | _ => 0
  }}
def guarded(x: {dtype}) -> i32 =
  match x with {{
    | 2 => 20
    | v if gt(v, 5{dtype}) => 50
    | _ => 0
  }}
a = pick(2{dtype})
b = pick(-3{dtype})
c = pick(7{dtype})
d = guarded(2{dtype})
e = guarded(7{dtype})
f = guarded(3{dtype})
"#
        );
        assert_lanes_print(
            &source,
            &format!("scalar_{dtype}"),
            &["a = 20", "b = 30", "c = 0", "d = 20", "e = 50", "f = 0"],
        );
    }
    for dtype in ["f16", "bf16"] {
        let source = format!(
            r#"
def pick(x: {dtype}) -> i32 =
  match x with {{
    | 0.5 => 5
    | -2.0 => 20
    | _ => 0
  }}
def guarded(x: {dtype}) -> i32 =
  match x with {{
    | 0.5 => 5
    | v if gt(v, 3.0{dtype}) => 30
    | _ => 0
  }}
a = pick(0.5{dtype})
b = pick(-2.0{dtype})
c = pick(7.0{dtype})
d = guarded(0.5{dtype})
e = guarded(7.0{dtype})
f = guarded(1.0{dtype})
"#
        );
        assert_lanes_print(
            &source,
            &format!("scalar_{dtype}"),
            &["a = 5", "b = 20", "c = 0", "d = 5", "e = 30", "f = 0"],
        );
    }
}

/// REGRESSION TEST (chelis#2446). Tuple scrutinees, including a nested tuple
/// and literal sub-patterns, and a `bool` match with no wildcard arm. On the
/// base sha C rejected the tuple matches with "an Option match has no `Some`
/// arm", and the `bool` match alone with "a host-lowered literal match
/// requires an explicit default arm".
#[test]
fn tuple_and_bool_scrutinees_lower_in_c_and_agree_with_eval() {
    let source = r#"
def classify(p: (i32, bool)) -> i32 =
  match p with {
    | (0, true) => 1
    | (0, false) => 2
    | (n, true) => add(n, 10)
    | (_, _) => 4
  }
def outer_sum(p: (i64, (f32, i64))) -> i64 =
  match p with {
    | (a, (_, b)) => add(a, b)
  }
def flip(b: bool) -> i32 =
  match b with {
    | true => 1
    | false => 0
  }
def greet(s: string) -> i32 =
  match s with {
    | "hi" => 1
    | "yo" => 2
    | _ => 0
  }
a = classify((0, true))
b = classify((0, false))
c = classify((5, true))
d = classify((5, false))
e = outer_sum((3i64, (1.0, 4i64)))
f = flip(true)
g = flip(false)
h = greet("yo")
i = greet("zz")
"#;
    assert_lanes_print(
        source,
        "tuple_and_bool_scrutinees",
        &[
            "a = 1", "b = 2", "c = 15", "d = 4", "e = 7", "f = 1", "g = 0", "h = 2", "i = 0",
        ],
    );
}

/// REGRESSION TEST ([04-PAT-2]). No guard runs for an arm whose pattern did
/// not match, or for any arm after the selected one: each program would trap
/// on a division by zero if it did. On the base sha C rejected the program
/// (chelis#2446); eval printed the expected values only because it never ran
/// a guard at all.
#[test]
fn a_guard_runs_only_after_its_pattern_matches_and_before_later_arms() {
    let source = r#"
def not_reached(o: Option[i32]) -> i32 =
  match o with {
    | Some(v) if gt(trunc_div(10, v), 1) => 1
    | _ => 2
  }
def earlier_wins(x: i32, zero: i32) -> i32 =
  match x with {
    | _ if gt(x, 0) => 1
    | v if gt(trunc_div(10, zero), 1) => 2
    | _ => 3
  }
a = not_reached(None)
b = earlier_wins(5, 0)
"#;
    assert_lanes_print(source, "guard_evaluation_order", &["a = 2", "b = 1"]);
}

/// REGRESSION TEST, negative parity ([04-PAT-2]). A guard whose evaluation
/// traps makes the match trap in both lanes; it is not read as `false`. On
/// the base sha eval printed `a = 1` and C rejected the program
/// (chelis#2446).
#[test]
fn a_trapping_guard_traps_in_eval_and_c() {
    let source = r#"
def trap_guard(x: i32) -> i32 =
  match x with {
    | v if gt(trunc_div(10, v), 1) => 1
    | _ => 2
  }
a = trap_guard(0)
"#;
    let trap = "division by zero in trunc_div";
    let interpreted = eval_output(source, "trapping_guard");
    assert!(!interpreted.status.success(), "eval must trap");
    let eval_stderr = String::from_utf8_lossy(&interpreted.stderr);
    assert!(eval_stderr.contains(trap), "{eval_stderr}");
    let compiled = run_compiled(source, "trapping_guard");
    assert!(!compiled.status.success(), "the compiled binary must trap");
    let c_stderr = String::from_utf8_lossy(&compiled.stderr);
    assert!(c_stderr.contains(trap), "{c_stderr}");
    assert!(
        compiled.stdout.is_empty() && interpreted.stdout.is_empty(),
        "no value may be printed for a trapped root"
    );
}

/// REGRESSION TEST. A match whose every guard is `false` runs out of arms
/// and fails with the same message in both lanes. The checker does not check
/// coverage for a scalar scrutinee, so this reaches both lanes. On the base
/// sha eval printed `a = 1` and C rejected the program for lacking a default
/// arm.
#[test]
fn a_match_whose_guards_all_fail_fails_in_eval_and_c() {
    let source = r#"
def positive(x: f32) -> i32 =
  match x with {
    | v if gt(v, 0.0) => 1
  }
a = positive(-1.0)
"#;
    let message = "non-exhaustive runtime match";
    let interpreted = eval_output(source, "guards_all_fail");
    assert!(!interpreted.status.success(), "eval must fail");
    let eval_stderr = String::from_utf8_lossy(&interpreted.stderr);
    assert!(eval_stderr.contains(message), "{eval_stderr}");
    let compiled = run_compiled(source, "guards_all_fail");
    assert!(!compiled.status.success(), "the compiled binary must fail");
    let c_stderr = String::from_utf8_lossy(&compiled.stderr);
    assert!(c_stderr.contains(message), "{c_stderr}");
}

/// Disposition lock, negative parity. A guard that is not `bool` is rejected
/// before either lane runs; this passed on the base sha too.
#[test]
fn a_non_bool_guard_is_rejected_before_either_lane_runs() {
    let source = r#"
def bad(x: f32) -> bool =
  match x with {
    | v if v => true
    | _ => false
  }
a = bad(0.0)
"#;
    let message = "match arm guard must be bool";
    let interpreted = eval_output(source, "non_bool_guard");
    assert!(!interpreted.status.success(), "eval must reject");
    let eval_stderr = String::from_utf8_lossy(&interpreted.stderr);
    assert!(eval_stderr.contains(message), "{eval_stderr}");
    let built = build_output(source, "non_bool_guard");
    assert!(!built.status.success(), "build must reject");
    let build_stderr = String::from_utf8_lossy(&built.stderr);
    assert!(build_stderr.contains(message), "{build_stderr}");
}

/// Disposition lock (chelis#520 D1). Static arm selection, which `grad` needs,
/// cannot evaluate a guard, so a guarded static match under `grad` stays a
/// loud rejection in both lanes rather than selecting an arm; this passed on
/// the base sha too.
#[test]
fn a_guarded_static_match_under_grad_stays_rejected_in_both_lanes() {
    let source = r#"
type Mode =
  | ModeA
  | ModeB
def fwd_guard(x: tensor[2, f32]) -> f32 =
  match ModeA with {
    | ModeA if true => sum(x, cast(0, i32)) |> tensor_to_scalar
    | _ => cast(0.0, f32)
  }
out = grad(fwd_guard)(to_tensor([1.0f32, 2.0f32]))
"#;
    let message = "`match` arm guards are not supported by static arm selection";
    let interpreted = eval_output(source, "static_guard_grad");
    assert!(!interpreted.status.success(), "eval must reject");
    let eval_stderr = String::from_utf8_lossy(&interpreted.stderr);
    assert!(eval_stderr.contains(message), "{eval_stderr}");
    let built = build_output(source, "static_guard_grad");
    assert!(!built.status.success(), "build must reject");
    let build_stderr = String::from_utf8_lossy(&built.stderr);
    assert!(build_stderr.contains(message), "{build_stderr}");
}

/// `n` guarded `Some` arms in a row, then an unguarded `Some` and `None`.
fn guarded_chain(n: usize) -> String {
    let mut source = "def band(o: Option[i64]) -> i64 =\n  match o with {\n".to_string();
    for bound in 1..=n {
        source.push_str(&format!(
            "    | Some(v) if lt(v, {bound}i64) => {bound}i64\n"
        ));
    }
    source.push_str("    | Some(v) => 99i64\n    | None => 0i64\n  }\n");
    source
}

/// Disposition lock on the C lowering's size. Each guarded `Some` arm has two
/// failure exits, its pattern and its guard; placing the rest of the match at
/// both would double the emitted body with every arm. The arm is decided by a
/// `bool` test instead, so doubling the arms roughly doubles the body. Placing
/// the rest at both exits emits 8 copies of the tail for 3 arms and 64 for 6,
/// which this ratio rejects; at 10 arms that lowering did not finish in nine
/// minutes.
#[test]
fn a_chain_of_guarded_constructor_arms_lowers_at_linear_size() {
    let short = emitted_body(&guarded_chain(3), "guarded_chain_3", "band");
    let long = emitted_body(&guarded_chain(6), "guarded_chain_6", "band");
    assert!(
        long.len() < 3 * short.len(),
        "6 guarded arms emitted {} bytes against {} for 3",
        long.len(),
        short.len()
    );
    let source = format!(
        "{}a = band(Some(0i64))\nb = band(Some(4i64))\nc = band(Some(5i64))\nd = band(Some(40i64))\ne = band(None)\n",
        guarded_chain(6)
    );
    assert_lanes_print(
        &source,
        "guarded_chain_run",
        &["a = 1", "b = 5", "c = 6", "d = 99", "e = 0"],
    );
}
