//! Issue #3463: `grad` through a statically selected `match` arm whose
//! pattern nests constructor, record, as, or tuple sub-patterns.
//!
//! spec/06-transformations.md §2.10.1: `match` differentiates the arm the
//! forward program executes, pattern tests are discrete, and nested
//! destructuring preserves the field paths of §2.1's recursive cotangent
//! type. A nested pattern is ordinary destructuring of a value whose
//! constructors are all compile-time-known, so static arm selection decides
//! it at every depth: a nested constructor that differs from the value's
//! makes the arm not match ([04-PAT-2] then tries the next arm), a nested
//! string literal is compared with the string the value holds, and a nested
//! binder binds the field it matched.
//!
//! Each positive case runs the eval lane against an analytic gradient and,
//! where named, the compiled C lane against the eval lane's exact output; the
//! issue's reproducer is also checked against a central finite difference of
//! its forward loss and against the same pattern written as two single-level
//! matches.
//!
//! Negative parity:
//!   - a nested literal sub-pattern against a numeric field is a run-time
//!     test, so the arm is not compile-time-known: rejected in both lanes,
//!     citing the open chelis#618 (runtime arm selection);
//!   - the same literal test does NOT reject when a sibling nested
//!     constructor already fails, because the arm cannot match whatever the
//!     run-time value is (the positive half of the pair above);
//!   - the runtime-scrutinee and guarded-arm rejections cite the open
//!     chelis#618, never the closed chelis#520.

mod common;

use std::fs;

use assert_cmd::Command;
use tempfile::tempdir;

/// Run `chelis eval` on a full program source; return (stdout, stderr, ok).
fn eval_program(source: &str) -> (String, String, bool) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("prog.ch");
    fs::write(&path, source).expect("write");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("run");
    (
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
        output.status.success(),
    )
}

/// Run `chelis build --target c` on a full program source; return (stderr, ok).
fn build_program(source: &str) -> (String, bool) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("prog.ch");
    fs::write(&path, source).expect("write");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            dir.path().join("out").to_str().unwrap(),
        ])
        .output()
        .expect("run");
    (
        String::from_utf8_lossy(&output.stderr).into_owned(),
        output.status.success(),
    )
}

/// The printed `out = ...` line of an eval or compiled run.
fn out_line(stdout: &str) -> &str {
    stdout
        .lines()
        .find(|line| line.starts_with("out = "))
        .unwrap_or_else(|| panic!("no `out = ` line in output: {stdout}"))
}

/// Every `data=[...]` tensor payload of one printed value, in print order.
fn tensor_payloads(line: &str) -> Vec<Vec<f64>> {
    line.split("data=[")
        .skip(1)
        .map(|rest| {
            let (payload, _) = rest
                .split_once(']')
                .unwrap_or_else(|| panic!("unterminated payload in {line}"));
            payload
                .split(',')
                .map(|value| value.trim().parse::<f64>().expect("numeric element"))
                .collect()
        })
        .collect()
}

/// The eval lane succeeds and prints exactly `expected` as its `out` line;
/// the compiled C lane, when `compiled` is set, prints the same output.
fn assert_grad_prints(source: &str, expected: &str, compiled: bool) {
    let (stdout, stderr, ok) = eval_program(source);
    assert!(ok, "eval must differentiate the nested pattern: {stderr}");
    assert_eq!(out_line(&stdout), expected, "eval cotangent:\n{source}");
    if compiled {
        let compiled_stdout = common::build_and_run(source, "prog");
        assert_eq!(
            compiled_stdout, stdout,
            "the compiled C lane must print the eval lane's output:\n{source}"
        );
    }
}

const NESTED_TYPES: &str = "module Repro.Nested\n\n\
type Inner[d] =\n\
\x20 | Inner { w: tensor[d, f32] }\n\n\
type Outer[d] =\n\
\x20 | Outer { inner: Inner[d], s: tensor[d, f32] }\n\n";

/// The issue's reproducer, made nonlinear in `w` so the cotangent of each
/// field is distinguishable: d/dw = 2*w*s, d/ds = w*w.
const NESTED_LOSS: &str = "def loss_nested(p: Outer[2]) -> f32 =\n\
\x20 match p with {\n\
\x20   | Outer { inner: Inner { w }, s } => tensor_to_scalar(sum(mul(mul(w, w), s), 0i32))\n\
\x20 }\n\n";

/// The same loss with the nested pattern written as two single-level matches.
const FLAT_LOSS: &str = "def loss_nested(p: Outer[2]) -> f32 =\n\
\x20 match p with {\n\
\x20   | Outer { inner, s } => match inner with {\n\
\x20     | Inner { w } => tensor_to_scalar(sum(mul(mul(w, w), s), 0i32))\n\
\x20   }\n\
\x20 }\n\n";

fn outer_literal(w: [f64; 2], s: [f64; 2]) -> String {
    format!(
        "Outer {{ inner: Inner {{ w: to_tensor([{:?}f32, {:?}f32]) }}, s: to_tensor([{:?}f32, {:?}f32]) }}",
        w[0], w[1], s[0], s[1]
    )
}

const NESTED_EXPECTED: &str =
    "out = Outer(Inner(tensor(shape=[2], data=[6.0, 16.0])), tensor(shape=[2], data=[1.0, 4.0]))";

/// The issue's reproducer: the nested pattern differentiates to the same
/// cotangent as the single-level control, and both equal the analytic one.
#[test]
fn issue_3463_nested_record_pattern_grad_matches_flat_control() {
    let driver = format!(
        "out = grad(loss_nested)({})\n",
        outer_literal([1.0, 2.0], [3.0, 4.0])
    );
    assert_grad_prints(
        &format!("{NESTED_TYPES}{FLAT_LOSS}{driver}"),
        NESTED_EXPECTED,
        false,
    );
    assert_grad_prints(
        &format!("{NESTED_TYPES}{NESTED_LOSS}{driver}"),
        NESTED_EXPECTED,
        false,
    );
}

/// The nested-pattern cotangent agrees with a central finite difference of
/// the nested-pattern forward loss, leaf by leaf, in field-path order.
#[test]
fn issue_3463_nested_record_pattern_grad_matches_finite_differences() {
    let (w, s) = ([1.0, 2.0], [3.0, 4.0]);
    let (stdout, stderr, ok) = eval_program(&format!(
        "{NESTED_TYPES}{NESTED_LOSS}out = grad(loss_nested)({})\n",
        outer_literal(w, s)
    ));
    assert!(ok, "nested grad failed: {stderr}");
    let analytic = tensor_payloads(out_line(&stdout)).concat();
    assert_eq!(analytic.len(), 4, "w then s cotangent leaves: {stdout}");
    let forward = |w: [f64; 2], s: [f64; 2]| {
        let (stdout, stderr, ok) = eval_program(&format!(
            "{NESTED_TYPES}{NESTED_LOSS}out = loss_nested({})\n",
            outer_literal(w, s)
        ));
        assert!(ok, "nested forward failed: {stderr}");
        out_line(&stdout)
            .trim_start_matches("out = ")
            .parse::<f64>()
            .expect("scalar loss")
    };
    let h = 1e-2;
    for leaf in 0..4 {
        let shifted = |delta: f64| {
            let (mut w, mut s) = (w, s);
            if leaf < 2 {
                w[leaf] += delta;
            } else {
                s[leaf - 2] += delta;
            }
            forward(w, s)
        };
        let fd = (shifted(h) - shifted(-h)) / (2.0 * h);
        assert!(
            (analytic[leaf] - fd).abs() < 5e-2,
            "leaf {leaf}: analytic {} vs finite difference {fd}",
            analytic[leaf]
        );
    }
}

/// The compiled C lane differentiates the nested pattern and prints exactly
/// what the eval lane prints.
#[test]
fn issue_3463_nested_record_pattern_grad_c_lane_agrees_with_eval() {
    let source = format!(
        "{NESTED_TYPES}{NESTED_LOSS}out = grad(loss_nested)({})\n",
        outer_literal([1.0, 2.0], [3.0, 4.0])
    );
    assert_grad_prints(&source, NESTED_EXPECTED, true);
}

const LAYER_TYPES: &str = "module Repro.Layers\n\n\
type Layer =\n\
\x20 | Dense { w: tensor[2, f32] }\n\
\x20 | Scale { k: tensor[2, f32] }\n\n\
type Model =\n\
\x20 | Model { layer: Layer, b: tensor[2, f32] }\n\n\
def loss_layer(m: Model) -> f32 =\n\
\x20 match m with {\n\
\x20   | Model { layer: Dense { w }, b } => tensor_to_scalar(sum(mul(mul(w, w), b), 0i32))\n\
\x20   | Model { layer: Scale { k }, b } => tensor_to_scalar(sum(mul(k, b), 0i32))\n\
\x20 }\n\n";

/// A nested constructor that differs from the field's constructor makes the
/// arm not match, so selection continues; the executed arm alone is
/// differentiated and the cotangent keeps the primal's constructor. Both
/// outcomes, in both lanes.
#[test]
fn issue_3463_nested_constructor_test_selects_the_executed_arm() {
    assert_grad_prints(
        &format!(
            "{LAYER_TYPES}out = grad(loss_layer)(Model {{ layer: Scale {{ k: to_tensor([1.0f32, 2.0f32]) }}, b: to_tensor([3.0f32, 4.0f32]) }})\n"
        ),
        "out = Model(Scale(tensor(shape=[2], data=[3.0, 4.0])), tensor(shape=[2], data=[1.0, 2.0]))",
        true,
    );
    assert_grad_prints(
        &format!(
            "{LAYER_TYPES}out = grad(loss_layer)(Model {{ layer: Dense {{ w: to_tensor([1.0f32, 2.0f32]) }}, b: to_tensor([3.0f32, 4.0f32]) }})\n"
        ),
        "out = Model(Dense(tensor(shape=[2], data=[6.0, 16.0])), tensor(shape=[2], data=[1.0, 4.0]))",
        true,
    );
}

const DEEP_TYPES: &str = "module Repro.Deep\n\n\
type Pair =\n\
\x20 | Pair(tensor[2, f32], tensor[2, f32])\n\n\
type Mid =\n\
\x20 | Mid { pair: Pair }\n\n\
type Top =\n\
\x20 | Top { mid: Mid, t: tensor[2, f32] }\n\n";

/// Three levels: a record inside a record holding a positional constructor,
/// read through an as-pattern whose inner pattern is itself nested. The
/// as-bound value is the whole field, so its second match reads the same
/// leaves: d/da = c*t, d/dc = a*t, d/dt = a*c.
#[test]
fn issue_3463_positional_and_as_patterns_nest_three_levels() {
    let source = format!(
        "{DEEP_TYPES}def loss_deep(h: Top) -> f32 =\n\
         \x20 match h with {{\n\
         \x20   | Top {{ mid: Mid {{ pair: q @ Pair(_, _) }}, t }} => match q with {{\n\
         \x20     | Pair(a, c) => tensor_to_scalar(sum(mul(mul(a, c), t), 0i32))\n\
         \x20   }}\n\
         \x20 }}\n\n\
         out = grad(loss_deep)(Top {{ mid: Mid {{ pair: Pair(to_tensor([1.0f32, 2.0f32]), to_tensor([3.0f32, 4.0f32])) }}, t: to_tensor([5.0f32, 6.0f32]) }})\n"
    );
    assert_grad_prints(
        &source,
        "out = Top(Mid(Pair(tensor(shape=[2], data=[15.0, 24.0]), tensor(shape=[2], data=[5.0, 12.0]))), tensor(shape=[2], data=[3.0, 8.0]))",
        true,
    );
}

/// A tuple sub-pattern inside a record pattern destructures a tuple-typed
/// field; the cotangent keeps the tuple's shape: d/da = b*t, d/db = a*t,
/// d/dt = a*b.
#[test]
fn issue_3463_nested_tuple_sub_pattern_differentiates_each_component() {
    let source = "module Repro.TupleField\n\n\
type Tup =\n\
\x20 | Tup { pr: (tensor[2, f32], tensor[2, f32]), t: tensor[2, f32] }\n\n\
def loss_tup(p: Tup) -> f32 =\n\
\x20 match p with {\n\
\x20   | Tup { pr: (a, b), t } => tensor_to_scalar(sum(mul(mul(a, b), t), 0i32))\n\
\x20 }\n\n\
out = grad(loss_tup)(Tup { pr: (to_tensor([1.0f32, 2.0f32]), to_tensor([3.0f32, 4.0f32])), t: to_tensor([5.0f32, 6.0f32]) })\n";
    assert_grad_prints(
        source,
        "out = Tup((tensor(shape=[2], data=[15.0, 24.0]), tensor(shape=[2], data=[5.0, 12.0])), tensor(shape=[2], data=[3.0, 8.0]))",
        true,
    );
}

const NAMED_TYPES: &str = "module Repro.Named\n\n\
type Named =\n\
\x20 | Named { name: string, w: tensor[2, f32] }\n\n\
def loss_named(p: Named) -> f32 =\n\
\x20 match p with {\n\
\x20   | Named { name: \"a\", w } => tensor_to_scalar(sum(mul(w, w), 0i32))\n\
\x20   | Named { name: _, w } => tensor_to_scalar(sum(w, 0i32))\n\
\x20 }\n\n";

/// A string field is exact host data, so a nested string literal is decided
/// at lowering time: the matching literal takes its arm and a different one
/// falls through. The string field's cotangent is unit.
#[test]
fn issue_3463_nested_string_literal_is_decided_statically() {
    for (name, expected) in [
        ("a", "out = Named((), tensor(shape=[2], data=[2.0, 4.0]))"),
        ("b", "out = Named((), tensor(shape=[2], data=[1.0, 1.0]))"),
    ] {
        assert_grad_prints(
            &format!(
                "{NAMED_TYPES}out = grad(loss_named)(Named {{ name: \"{name}\", w: to_tensor([1.0f32, 2.0f32]) }})\n"
            ),
            expected,
            true,
        );
    }
}

const COUNTED_TYPES: &str = "module Repro.Counted\n\n\
type Layer =\n\
\x20 | Dense { w: tensor[2, f32] }\n\
\x20 | Scale { k: tensor[2, f32] }\n\n\
type Counted =\n\
\x20 | Counted { layer: Layer, n: i32 }\n\n\
def loss_counted(c: Counted) -> f32 =\n\
\x20 match c with {\n\
\x20   | Counted { layer: Dense { w }, n: 0 } => tensor_to_scalar(sum(w, 0i32))\n\
\x20   | Counted { layer: Dense { w }, n: _ } => tensor_to_scalar(sum(mul(w, w), 0i32))\n\
\x20   | Counted { layer: Scale { k }, n: _ } => tensor_to_scalar(sum(mul(k, k), 0i32))\n\
\x20 }\n\n";

/// A nested literal test against a numeric field needs the run-time value,
/// so the taken arm is not compile-time-known: both lanes reject loudly,
/// citing the open owner of runtime arm selection, never the closed #520.
#[test]
fn issue_3463_nested_literal_test_on_a_runtime_field_stays_rejected() {
    let source = format!(
        "{COUNTED_TYPES}out = grad(loss_counted)(Counted {{ layer: Dense {{ w: to_tensor([1.0f32, 2.0f32]) }}, n: 0i32 }})\n"
    );
    let (_stdout, stderr, ok) = eval_program(&source);
    assert!(!ok, "a run-time literal test must not select an arm");
    assert_cites_runtime_selection_owner(&stderr);
    assert!(
        stderr.contains("literal"),
        "names the literal test: {stderr}"
    );
    let (stderr, ok) = build_program(&source);
    assert!(!ok, "the compiled lane must reject the same match");
    assert_cites_runtime_selection_owner(&stderr);
}

/// The positive half of the pair above: when a sibling nested constructor
/// already fails, the arm cannot match whatever the run-time literal test
/// says, so selection skips it and differentiates the executed arm.
#[test]
fn issue_3463_failed_nested_constructor_skips_an_arm_with_a_runtime_literal() {
    assert_grad_prints(
        &format!(
            "{COUNTED_TYPES}out = grad(loss_counted)(Counted {{ layer: Scale {{ k: to_tensor([1.0f32, 2.0f32]) }}, n: 0i32 }})\n"
        ),
        "out = Counted(Scale(tensor(shape=[2], data=[2.0, 4.0])), ())",
        true,
    );
}

fn assert_cites_runtime_selection_owner(stderr: &str) {
    assert!(
        stderr.contains("chelis#618"),
        "the rejection must cite the open owner chelis#618: {stderr}"
    );
    assert!(
        !stderr.contains("chelis#520"),
        "the rejection must not cite the closed chelis#520: {stderr}"
    );
}

/// The runtime-scrutinee and guarded-arm rejections name their construct
/// and cite the open chelis#618 in both lanes.
#[test]
fn issue_3463_runtime_selection_rejections_cite_the_open_owner() {
    let runtime_scrutinee = "module Repro.RuntimeScrutinee\n\n\
def fwd_dyn(x: tensor[2, f32]) -> f32 =\n\
\x20 match tensor_to_scalar(sum(x, 0i32)) with {\n\
\x20   | 0.0 => 0.0f32\n\
\x20   | _ => tensor_to_scalar(sum(mul(x, x), 0i32))\n\
\x20 }\n\n\
out = grad(fwd_dyn)(to_tensor([1.0f32, 2.0f32]))\n";
    let guarded = "module Repro.Guarded\n\n\
type Mode =\n\
\x20 | ModeA\n\
\x20 | ModeB\n\n\
def fwd_guard(x: tensor[2, f32]) -> f32 =\n\
\x20 match ModeA with {\n\
\x20   | ModeA if true => tensor_to_scalar(sum(x, 0i32))\n\
\x20   | _ => 0.0f32\n\
\x20 }\n\n\
out = grad(fwd_guard)(to_tensor([1.0f32, 2.0f32]))\n";
    for (source, construct) in [(runtime_scrutinee, "runtime scrutinee"), (guarded, "guard")] {
        let (_stdout, stderr, ok) = eval_program(source);
        assert!(!ok, "eval must reject:\n{source}");
        assert!(
            stderr.contains(construct),
            "names the {construct}: {stderr}"
        );
        assert_cites_runtime_selection_owner(&stderr);
        let (stderr, ok) = build_program(source);
        assert!(!ok, "build must reject:\n{source}");
        assert!(
            stderr.contains(construct),
            "names the {construct}: {stderr}"
        );
        assert_cites_runtime_selection_owner(&stderr);
    }
}
