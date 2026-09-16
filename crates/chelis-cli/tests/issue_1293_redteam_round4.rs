//! Permanent regressions for review 5037740563 of chelis#1293.
//!
//! The controlling contracts are spec/06 sections 2.1, 2.4, and 2.10.1
//! plus spec/05 [05-OP-35]. Runtime List selection must not arithmetically
//! mix unselected IEEE values, and recursive cotangents preserve every List,
//! tuple, and executed ADT-constructor layer in evaluator and generated C.

use assert_cmd::Command;
use predicates::prelude::*;

#[path = "common/mod.rs"]
mod common;

use common::{build_and_run_app, link_generated, make_app, write_file};

fn eval_app_stdout(reef_home: &std::path::Path, app_pkg: &std::path::Path) -> String {
    let assert = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .success();
    String::from_utf8(assert.get_output().stdout.clone()).expect("utf-8 stdout")
}

fn assert_eval_and_c_diagnostic(
    reef_home: &std::path::Path,
    app_pkg: &std::path::Path,
    expected: &str,
) {
    let eval = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .output()
        .expect("eval should run");
    assert!(
        !eval.status.success(),
        "invalid List control unexpectedly evaluated"
    );
    assert!(
        String::from_utf8_lossy(&eval.stderr).contains(expected),
        "unexpected eval diagnostic: {}",
        String::from_utf8_lossy(&eval.stderr)
    );

    let out_dir = app_pkg.join("main-out");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(app_pkg)
        .args([
            "build",
            app_pkg.join("src/main.ch").to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    assert!(
        link_generated(&out_dir, "main.c", "main").success(),
        "generated C must compile and link"
    );
    let compiled = std::process::Command::new(out_dir.join("main"))
        .output()
        .expect("compiled binary should run");
    assert!(
        !compiled.status.success(),
        "invalid List control unexpectedly compiled and ran"
    );
    assert!(
        String::from_utf8_lossy(&compiled.stderr).contains(expected),
        "unexpected compiled diagnostic: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
}

#[test]
fn runtime_list_selection_does_not_mix_unselected_non_finite_values() {
    let (_dir, reef_home, app_pkg) = make_app("issue-1293-list-non-finite-selection");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Index (drop_list, list_index, take_list)

def square_selected(xs: List[f32], index: i64) -> f32 = {
  selected = list_index(xs, index)
  mul(selected, selected)
}

def tensor_square_sum(value: tensor[2, f32]) -> f32 =
  tensor_to_scalar(sum(mul(&value, &value), cast(0, i32)))

def square_selected_tensor(xs: List[tensor[2, f32]], index: i64) -> f32 =
  tensor_square_sum(list_index(xs, index))

def nested_selected(
  xs: List[List[f32]],
  outer: i64,
  inner: i64
) -> f32 = {
  selected = list_index(list_index(xs, outer), inner)
  mul(selected, selected)
}

def composed_selected(xs: List[f32], count: i64) -> f32 = {
  window = drop_list(take_list(xs, add(count, count)), count)
  selected = list_index(window, sub(count, count))
  add(mul(selected, selected), mul(selected, selected))
}

zero: f32 = cast(0.0, f32)
one: f32 = cast(1.0, f32)
nan: f32 = div(zero, zero)
pos_inf: f32 = div(one, zero)
neg_inf: f32 = neg(pos_inf)
finite: f32 = cast(3.0, f32)
mask: tensor[2, bool] = [true, false]
runtime_one: i64 = tensor_to_scalar(count(&mask, 0))
runtime_zero: i64 = sub(runtime_one, runtime_one)
runtime_two: i64 = add(runtime_one, runtime_one)

unselected = grad(square_selected, wrt=xs)(
  [nan, finite, pos_inf, neg_inf],
  runtime_one
)
selected_nan = grad(square_selected, wrt=xs)(
  [nan, finite, pos_inf, neg_inf],
  runtime_zero
)
selected_pos_inf = grad(square_selected, wrt=xs)(
  [nan, finite, pos_inf, neg_inf],
  runtime_two
)
selected_neg_inf = grad(square_selected, wrt=xs)(
  [nan, finite, pos_inf, neg_inf],
  add(runtime_two, runtime_one)
)
shaped = grad(square_selected_tensor, wrt=xs)(
  [
    div(
      to_tensor([cast(0.0, f32), cast(1.0, f32)]),
      to_tensor([cast(0.0, f32), cast(0.0, f32)])
    ),
    to_tensor([cast(2.0, f32), cast(-3.0, f32)]),
    div(
      to_tensor([cast(-1.0, f32), cast(0.0, f32)]),
      to_tensor([cast(0.0, f32), cast(0.0, f32)])
    )
  ],
  runtime_one
)
nested = grad(nested_selected, wrt=xs)(
  [
    [nan, pos_inf],
    [cast(4.0, f32), neg_inf]
  ],
  runtime_one,
  runtime_zero
)
composed = grad(composed_selected, wrt=xs)(
  [nan, finite, pos_inf, neg_inf],
  runtime_one
)
"#,
    );

    let eval = eval_app_stdout(&reef_home, &app_pkg);
    let compiled = build_and_run_app(&reef_home, &app_pkg, "main");
    for expected in [
        "unselected = [0.0, 6.0, 0.0, 0.0]",
        "selected_nan = [NaN, 0.0, 0.0, 0.0]",
        "selected_pos_inf = [0.0, 0.0, inf, 0.0]",
        "selected_neg_inf = [0.0, 0.0, 0.0, -inf]",
        "shaped = [tensor(shape=[2], data=[0.0, 0.0]), tensor(shape=[2], data=[4.0, -6.0]), tensor(shape=[2], data=[0.0, 0.0])]",
        "nested = [[0.0, 0.0], [8.0, 0.0]]",
        "composed = [0.0, 12.0, 0.0, 0.0]",
    ] {
        assert!(
            eval.contains(expected),
            "eval missing `{expected}`:\n{eval}"
        );
        assert!(
            compiled.contains(expected),
            "compiled C missing `{expected}`:\n{compiled}"
        );
    }
}

#[test]
fn composed_runtime_list_bounds_are_guarded_before_internal_selection() {
    let (_dir, reef_home, app_pkg) = make_app("issue-1293-list-guard-before-selection");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Index (drop_list, list_index)

def loss(xs: List[f32], drop_count: i64, index: i64) -> f32 =
  list_index(drop_list(xs, drop_count), index)

mask: tensor[2, bool] = [true, false]
runtime_one: i64 = tensor_to_scalar(count(&mask, 0))
runtime_max: i64 = tensor_to_scalar(
  sum(to_tensor([9223372036854775807i64]), cast(0, i32))
)

out = grad(loss, wrt=xs)(
  [cast(2.0, f32), cast(3.0, f32), cast(5.0, f32)],
  runtime_one,
  runtime_max
)
"#,
    );

    assert_eval_and_c_diagnostic(
        &reef_home,
        &app_pkg,
        "index 9223372036854775807 out of bounds for list of len 2",
    );
}

#[test]
fn composed_runtime_list_negative_extreme_is_guarded_before_internal_selection() {
    let (_dir, reef_home, app_pkg) = make_app("issue-1293-list-negative-extreme-guard");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Index (drop_list, list_index, take_list)

def loss(xs: List[f32], drop_count: i64, take_count: i64) -> f32 =
  list_index(take_list(drop_list(xs, drop_count), take_count), cast(0, i64))

runtime_min: i64 = tensor_to_scalar(
  sum(to_tensor([-9223372036854775808i64]), cast(0, i32))
)

out = grad(loss, wrt=xs)(
  [cast(2.0, f32), cast(3.0, f32), cast(5.0, f32)],
  runtime_min,
  cast(9223372036854775807, i64)
)
"#,
    );

    assert_eval_and_c_diagnostic(&reef_home, &app_pkg, "drop requires non-negative count");
}

#[test]
fn recursive_list_tuple_and_adt_cotangents_match_in_eval_and_c() {
    let (_dir, reef_home, app_pkg) = make_app("issue-1293-list-recursive-cotangents");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Index (list_index)

type PairAlias = (f32, f32)

type Boxed =
  | Boxed { value: f32 }

type Wrapper[a] =
  | Wrapper { value: a }

type TensorBox =
  | TensorBox { value: tensor[2, f32], code: i32 }

type PairBox =
  | PairBox { pair: PairAlias, enabled: bool }

type Pick =
  | First { value: f32 }
  | Second { value: f32, scale: f32 }

type Conditional =
  | Live { value: f32 }
  | Frozen { tag: i64 }
  | Empty

type Tagged =
  | Tagged { value: f32, tag: i64 }

def tuple_loss(xs: List[PairAlias], index: i64) -> f32 = {
  selected = list_index(xs, index)
  add(mul(selected.0, selected.0), mul(selected.1, selected.1))
}

def nested_tuple_loss(
  xs: List[List[PairAlias]],
  outer: i64,
  inner: i64
) -> f32 = {
  selected = list_index(list_index(xs, outer), inner)
  add(selected.0, selected.1)
}

def boxed_loss(xs: List[Boxed], index: i64) -> f32 =
  match list_index(xs, index) with {
    | Boxed { value } => mul(value, value)
  }

def wrapper_loss(xs: List[Wrapper[f32]], index: i64) -> f32 =
  match list_index(xs, index) with {
    | Wrapper { value } => mul(value, value)
  }

def keep_wrapped(xs: List[Wrapper[f32]]) -> List[Wrapper[f32]] = xs

def tensor_box_loss(xs: List[TensorBox], index: i64) -> f32 =
  match list_index(xs, index) with {
    | TensorBox { value, code: _ } =>
      tensor_to_scalar(sum(mul(&value, &value), cast(0, i32)))
  }

def pair_box_loss(xs: List[PairBox], index: i64) -> f32 =
  match list_index(xs, index) with {
    | PairBox { pair, enabled: _ } =>
      add(mul(pair.0, pair.0), mul(pair.1, pair.1))
  }

def pick_loss(xs: List[Pick], index: i64) -> f32 =
  match list_index(xs, index) with {
    | First { value } => mul(value, value)
    | Second { value, scale } => mul(mul(value, value), scale)
  }

def conditional_loss(xs: List[Conditional], index: i64) -> f32 =
  match list_index(xs, index) with {
    | Live { value } => mul(value, value)
    | Frozen { tag: _ } => cast(0.0, f32)
    | Empty => cast(0.0, f32)
  }

def tagged_loss(xs: List[Tagged], index: i64) -> f32 =
  match list_index(xs, index) with {
    | Tagged { value, tag: _ } => mul(value, value)
  }

def tagged_value_loss(value: Tagged) -> f32 =
  match value with {
    | Tagged { value, tag: _ } => mul(value, value)
  }

def second_target_loss(
  ignored: tensor[2, f32],
  xs: List[PairAlias],
  index: i64
) -> f32 = {
  selected = list_index(xs, index)
  add(selected.0, selected.1)
}

def default_target_loss(
  metadata: (i64, bool),
  xs: List[PairAlias],
  index: i64
) -> f32 = {
  selected = list_index(xs, index)
  add(selected.0, selected.1)
}

def constant_pair_loss(xs: List[PairAlias]) -> f32 = cast(1.0, f32)

def collision_loss(
  xs: List[f32],
  __chelis_grad_arg_0_leaf_0_0: f32,
  index: i64
) -> f32 =
  mul(list_index(xs, index), __chelis_grad_arg_0_leaf_0_0)

mask: tensor[2, bool] = [true, false]
runtime_one: i64 = tensor_to_scalar(count(&mask, 0))
runtime_zero: i64 = sub(runtime_one, runtime_one)
pairs: List[PairAlias] = [
  (cast(2.0, f32), cast(3.0, f32)),
  (cast(5.0, f32), cast(7.0, f32))
]

tuple_grad = grad(tuple_loss, wrt=xs)(pairs, runtime_one)
nested_tuple_grad = grad(nested_tuple_loss, wrt=xs)(
  [[(cast(2.0, f32), cast(3.0, f32))], [(cast(5.0, f32), cast(7.0, f32))]],
  runtime_one,
  runtime_zero
)
empty_tuple_grad = grad(constant_pair_loss, wrt=xs)([])
singleton_tuple_grad = grad(tuple_loss, wrt=xs)(
  [(cast(11.0, f32), cast(13.0, f32))],
  runtime_zero
)
boxed_grad = grad(boxed_loss, wrt=xs)(
  [Boxed { value: cast(2.0, f32) }, Boxed { value: cast(5.0, f32) }],
  runtime_one
)
wrapper_grad = grad(wrapper_loss, wrt=xs)(
  keep_wrapped([
    Wrapper { value: cast(2.0, f32) },
    Wrapper { value: cast(5.0, f32) }
  ]),
  runtime_one
)
tensor_box_grad = grad(tensor_box_loss, wrt=xs)(
  [
    TensorBox {
      value: to_tensor([cast(2.0, f32), cast(3.0, f32)]),
      code: cast(31, i32)
    },
    TensorBox {
      value: to_tensor([cast(5.0, f32), cast(7.0, f32)]),
      code: cast(37, i32)
    }
  ],
  runtime_one
)
pair_box_grad = grad(pair_box_loss, wrt=xs)(
  [
    PairBox { pair: (cast(2.0, f32), cast(3.0, f32)), enabled: false },
    PairBox { pair: (cast(5.0, f32), cast(7.0, f32)), enabled: true }
  ],
  runtime_one
)
first_grad = grad(pick_loss, wrt=xs)(
  [First { value: cast(2.0, f32) }, First { value: cast(5.0, f32) }],
  runtime_one
)
second_grad = grad(pick_loss, wrt=xs)(
  [
    Second { value: cast(2.0, f32), scale: cast(3.0, f32) },
    Second { value: cast(5.0, f32), scale: cast(7.0, f32) }
  ],
  runtime_one
)
frozen_only_grad = grad(conditional_loss, wrt=xs)(
  [Frozen { tag: cast(41, i64) }],
  runtime_zero
)
empty_variant_grad = grad(conditional_loss, wrt=xs)(
  [Empty],
  runtime_zero
)
tagged_grad = grad(tagged_loss, wrt=xs)(
  [
    Tagged { value: cast(2.0, f32), tag: cast(17, i64) },
    Tagged { value: cast(5.0, f32), tag: cast(19, i64) }
  ],
  runtime_one
)
top_level_tagged_grad = grad(tagged_value_loss, wrt=value)(
  Tagged { value: cast(5.0, f32), tag: cast(23, i64) }
)
second_target = grad(second_target_loss, wrt=xs)(
  to_tensor([cast(100.0, f32), cast(200.0, f32)]),
  pairs,
  runtime_one
)
default_target = grad(default_target_loss)(
  (cast(29, i64), true),
  pairs,
  runtime_one
)
collision_grad = grad(collision_loss, wrt=xs)(
  [cast(2.0, f32), cast(5.0, f32)],
  cast(7.0, f32),
  runtime_one
)
"#,
    );

    let eval = eval_app_stdout(&reef_home, &app_pkg);
    let compiled = build_and_run_app(&reef_home, &app_pkg, "main");
    for expected in [
        "tuple_grad = [(0.0, 0.0), (10.0, 14.0)]",
        "nested_tuple_grad = [[(0.0, 0.0)], [(1.0, 1.0)]]",
        "empty_tuple_grad = []",
        "singleton_tuple_grad = [(22.0, 26.0)]",
        "second_target = [(0.0, 0.0), (1.0, 1.0)]",
        "default_target = [(0.0, 0.0), (1.0, 1.0)]",
        "collision_grad = [0.0, 7.0]",
    ] {
        assert!(
            eval.contains(expected),
            "eval missing `{expected}`:\n{eval}"
        );
        assert!(
            compiled.contains(expected),
            "compiled C missing `{expected}`:\n{compiled}"
        );
    }
    for output in [&eval, &compiled] {
        for expected in [
            "boxed_grad = [",
            "Boxed(0.0)",
            "Boxed(10.0)",
            "wrapper_grad = [",
            "Wrapper(0.0)",
            "Wrapper(10.0)",
            "tensor_box_grad = [",
            "TensorBox(tensor(shape=[2], data=[0.0, 0.0]), ())",
            "TensorBox(tensor(shape=[2], data=[10.0, 14.0]), ())",
            "pair_box_grad = [",
            "PairBox((0.0, 0.0), ())",
            "PairBox((10.0, 14.0), ())",
            "first_grad = [",
            "First(0.0)",
            "First(10.0)",
            "second_grad = [",
            "Second(0.0, 0.0)",
            "Second(70.0, 25.0)",
            "frozen_only_grad = [",
            "Frozen(())",
            "empty_variant_grad = [",
            "Empty]",
            "tagged_grad = [",
            "Tagged(0.0, ())",
            "Tagged(10.0, ())",
            "top_level_tagged_grad = ",
            "Tagged(10.0, ())",
        ] {
            assert!(
                output.contains(expected),
                "output missing `{expected}`:\n{output}"
            );
        }
    }
}

#[test]
fn recursively_all_discrete_list_shapes_are_rejected_at_the_checker() {
    let (_dir, reef_home, app_pkg) = make_app("issue-1293-list-recursive-discrete-negative");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

def loss(xs: List[(i64, bool)]) -> f32 = cast(1.0, f32)
bad = grad(loss, wrt=xs)([(cast(1, i64), true)])
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .failure()
        .stdout(predicate::str::contains(
            "grad `wrt` index 0 is not differentiable",
        ));
}
