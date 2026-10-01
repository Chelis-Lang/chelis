//! chelis#2587: [05-OP-36]'s recursive equality agrees in the evaluator and
//! the compiled C lane.
//!
//! `eq` and `neq` over unit and two `List`, tuple, `Dict`, `Option`, or ADT
//! values compare structurally to one bool: lists and tuples by length and
//! then items in order, options and ADTs by constructor and then fields,
//! dictionaries as key/value sets independent of insertion order, and a tensor
//! field by dtype, dimensions and elements. Float leaves compare as IEEE
//! values at their own width, so a NaN is unequal and the signed zeros are
//! equal. The negative cases pin the front-end rejection of each operand the
//! atom does not admit.

use assert_cmd::Command;

#[path = "common/mod.rs"]
mod common;

use common::{build_and_run_app, make_app, write_file};

fn eval_app(reef_home: &std::path::Path, app: &std::path::Path) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(app)
        .args(["eval", "--file", "src/main.ch"])
        .output()
        .expect("run evaluator")
}

const PROGRAM: &str = r#"module Demo.Main

type Point =
  | Point { x: f64, y: f64 }
type Shape =
  | Circle(Point, f32)
  | Polygon(List[Point])
  | Empty
type Chain =
  | Link(i32, Chain)
  | End

def nan32(zero: f32) -> f32 = div(zero, zero)
def nan64(zero: f64) -> f64 = div(zero, zero)
def nan16(zero: f16) -> f16 = div(zero, zero)
def same_option(a: Option[i64], b: Option[i64]) -> bool = eq(a, b)
def other_option(a: Option[i64], b: Option[i64]) -> bool = neq(a, b)
def is_empty(xs: List[i32]) -> bool = eq(xs, [])

unit_eq = eq((), ())
unit_neq = neq((), ())
list_eq = eq([1i32, 2i32], [1i32, 2i32])
list_shorter = eq([1i32, 2i32], [1i32])
list_order = eq([1i32, 2i32], [2i32, 1i32])
list_empty = is_empty([])
list_not_empty = is_empty([1i32])
list_neq = neq([1i32], [2i32])
nested_eq = eq([[1i64], [2i64, 3i64]], [[1i64], [2i64, 3i64]])
nested_differs = eq([[1i64], [2i64, 3i64]], [[1i64], [2i64, 4i64]])
small_ints = eq([1i8, neg(2i8)], [1i8, neg(2i8)])
tuple_eq = eq((1i32, "a", true), (1i32, "a", true))
tuple_differs = eq((1i32, "a", true), (1i32, "b", true))
tuple_neq = neq((1i32, "a", true), (1i32, "a", false))
strings_eq = eq(["a", "bc"], ["a", "bc"])
bools_eq = eq([true, false], [lt(1i32, 2i32), false])
dict_order = eq(dict_of([("a", 1i64), ("b", 2i64)]), dict_of([("b", 2i64), ("a", 1i64)]))
dict_value = eq(dict_of([("a", 1i64), ("b", 2i64)]), dict_of([("a", 1i64), ("b", 3i64)]))
dict_key = eq(dict_of([("a", 1i64)]), dict_of([("c", 1i64)]))
dict_len = eq(dict_of([("a", 1i64)]), dict_of([("a", 1i64), ("b", 2i64)]))
dict_neq = neq(dict_of([(1i64, [1.5f32])]), dict_of([(1i64, [1.5f32])]))
some_eq = eq(Some([1i64]), Some([1i64]))
some_differs = eq(Some([1i64]), Some([2i64]))
some_none = eq(Some(1i64), None)
none_none = same_option(None, None)
none_neq = other_option(None, Some(1i64))
point_eq = eq(Point { x: 1.0f64, y: 2.0f64 }, Point { x: 1.0f64, y: 2.0f64 })
point_differs = eq(Point { x: 1.0f64, y: 2.0f64 }, Point { x: 1.0f64, y: 2.5f64 })
ctor_differs = eq(Circle(Point { x: 0.0f64, y: 0.0f64 }, 1.0f32), Empty)
shape_eq = eq(Polygon([Point { x: 0.0f64, y: 1.0f64 }]), Polygon([Point { x: 0.0f64, y: 1.0f64 }]))
empty_eq = eq(Empty, Empty)
chain_eq = eq(Link(1i32, Link(2i32, End)), Link(1i32, Link(2i32, End)))
chain_differs = eq(Link(1i32, Link(2i32, End)), Link(1i32, End))
nan_list = eq([nan32(0.0f32)], [nan32(0.0f32)])
nan_list_neq = neq([nan32(0.0f32)], [nan32(0.0f32)])
nan_point = eq(Point { x: nan64(0.0f64), y: 1.0f64 }, Point { x: nan64(0.0f64), y: 1.0f64 })
nan_half = eq([nan16(0.0f16)], [nan16(0.0f16)])
signed_zero = eq([0.0f64], [neg(0.0f64)])
signed_zero_point = eq(Point { x: neg(0.0f64), y: 1.0f64 }, Point { x: 0.0f64, y: 1.0f64 })
signed_zero_half = eq([neg(0.0f16)], [0.0f16])
signed_zero_bf16 = eq((neg(0.0bf16), 1i8), (0.0bf16, 1i8))
tensor_eq = eq([to_tensor([1.0f32, 2.0f32])], [to_tensor([1.0f32, 2.0f32])])
tensor_differs = eq([to_tensor([1.0f32, 2.0f32])], [to_tensor([1.0f32, 3.0f32])])
tensor_nan = eq([to_tensor([nan32(0.0f32)])], [to_tensor([nan32(0.0f32)])])
tensor_zero = eq((to_tensor([0.0f32]), 1i32), (to_tensor([neg(0.0f32)]), 1i32))
"#;

const EXPECTED: &[(&str, bool)] = &[
    ("unit_eq", true),
    ("unit_neq", false),
    ("list_eq", true),
    ("list_shorter", false),
    ("list_order", false),
    ("list_empty", true),
    ("list_not_empty", false),
    ("list_neq", true),
    ("nested_eq", true),
    ("nested_differs", false),
    ("small_ints", true),
    ("tuple_eq", true),
    ("tuple_differs", false),
    ("tuple_neq", true),
    ("strings_eq", true),
    ("bools_eq", true),
    ("dict_order", true),
    ("dict_value", false),
    ("dict_key", false),
    ("dict_len", false),
    ("dict_neq", false),
    ("some_eq", true),
    ("some_differs", false),
    ("some_none", false),
    ("none_none", true),
    ("none_neq", true),
    ("point_eq", true),
    ("point_differs", false),
    ("ctor_differs", false),
    ("shape_eq", true),
    ("empty_eq", true),
    ("chain_eq", true),
    ("chain_differs", false),
    ("nan_list", false),
    ("nan_list_neq", true),
    ("nan_point", false),
    ("nan_half", false),
    ("signed_zero", true),
    ("signed_zero_point", true),
    ("signed_zero_half", true),
    ("signed_zero_bf16", true),
    ("tensor_eq", true),
    ("tensor_differs", false),
    ("tensor_nan", false),
    ("tensor_zero", true),
];

#[test]
fn structural_equality_agrees_in_eval_and_compiled_c() {
    let (_dir, reef_home, app) = make_app("issue-2587-structural-eq");
    write_file(&app.join("src/main.ch"), PROGRAM);

    let eval = eval_app(&reef_home, &app);
    assert!(
        eval.status.success(),
        "{}",
        String::from_utf8_lossy(&eval.stderr)
    );
    let compiled = build_and_run_app(&reef_home, &app, "main");
    let output = String::from_utf8(eval.stdout).expect("UTF-8 output");
    assert_eq!(output, compiled, "eval/C output diverged");
    let lines: Vec<&str> = output.lines().collect();
    for (name, value) in EXPECTED {
        let expected = format!("{name} = {value}");
        assert!(
            lines.contains(&expected.as_str()),
            "missing {expected:?} in {output}"
        );
    }
    assert_eq!(
        lines.len(),
        EXPECTED.len(),
        "every root is pinned: {output}"
    );
}

/// Every operand [05-OP-36] does not admit is rejected before either lane
/// runs, with the rule that refuses it.
#[test]
fn inadmissible_equality_operands_are_rejected_by_check_eval_and_build() {
    for (stem, source, needle) in [
        (
            "function",
            "def inc(x: i32) -> i32 = add(x, 1i32)\nsame = eq([inc], [inc])\n",
            "[05-OP-36]",
        ),
        (
            "function_field",
            "type Callback =\n  | Callback { run: (i32) -> i32 }\ndef same(a: Callback, b: \
             Callback) -> bool = neq(a, b)\n",
            "[05-OP-36]",
        ),
        (
            "mapped_file",
            "def same(a: MappedFile, b: MappedFile) -> bool = eq(a, b)\n",
            "[05-OP-36]",
        ),
        (
            "key",
            "def same(a: List[key], b: List[key]) -> bool = eq(a, b)\n",
            "section 1.1",
        ),
        ("mismatched", "same = eq([1i32], [1i64])\n", "i64"),
        ("ordered", "less = lt([1i32], [2i32])\n", "lt"),
        (
            "unbounded_binder",
            "def same[a](x: List[a]) -> bool = eq(x, x)\n",
            "admits only some operand types",
        ),
    ] {
        let (_dir, reef_home, app) = make_app(&format!("issue-2587-reject-{stem}"));
        write_file(
            &app.join("src/main.ch"),
            &format!("module Demo.Main\n\n{source}"),
        );
        let eval = eval_app(&reef_home, &app);
        let stderr = String::from_utf8_lossy(&eval.stderr);
        assert!(!eval.status.success(), "{stem}: eval accepted\n{source}");
        assert!(stderr.contains(needle), "{stem}: eval said {stderr}");
        let build = Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef_home)
            .current_dir(&app)
            .args(["build", "src/main.ch", "--target", "c", "--output"])
            .arg(app.join("out"))
            .output()
            .expect("run build");
        let stderr = String::from_utf8_lossy(&build.stderr);
        assert!(!build.status.success(), "{stem}: build accepted\n{source}");
        assert!(stderr.contains(needle), "{stem}: build said {stderr}");
    }
}

/// chelis#1521: `chelis build` refuses an `eq` or `neq` operand pair the host
/// lane has no comparison for, such as two tensors an `IO` operand moves to
/// the host lane, rather than emitting a C address comparison. The evaluator
/// answers element-wise.
#[test]
fn host_lane_equality_without_a_comparison_is_refused_at_build() {
    for (operation, expected) in [("eq", "[true, false]"), ("neq", "[false, true]")] {
        let (_dir, reef_home, app) = make_app(&format!("issue-2587-host-tensor-{operation}"));
        write_file(
            &app.join("src/main.ch"),
            &format!(
                "module Demo.Main\n\ndef f(a: tensor[2, f32], b: tensor[2, f32]) -> tensor[2, \
                 bool] ! {{ IO }} = {operation}(debug(a), b)\nr = f(to_tensor([1.0f32, 2.0f32]), \
                 to_tensor([1.0f32, 3.0f32]))\n"
            ),
        );
        let eval = eval_app(&reef_home, &app);
        let stdout = String::from_utf8_lossy(&eval.stdout);
        assert!(
            eval.status.success(),
            "{operation}: {}",
            String::from_utf8_lossy(&eval.stderr)
        );
        assert!(
            stdout.contains(&format!("r = tensor(shape=[2], data={expected})")),
            "{operation}: {stdout}"
        );
        let build = Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef_home)
            .current_dir(&app)
            .args(["build", "src/main.ch", "--target", "c", "--output"])
            .arg(app.join("out"))
            .output()
            .expect("run build");
        let stderr = String::from_utf8_lossy(&build.stderr);
        assert!(!build.status.success(), "{operation}: build accepted");
        assert!(
            stderr.contains(&format!("builtin `{operation}`"))
                && stderr.contains("[04-TOT-2]")
                && stderr.contains("never compared by address"),
            "{operation}: {stderr}"
        );
    }
}
