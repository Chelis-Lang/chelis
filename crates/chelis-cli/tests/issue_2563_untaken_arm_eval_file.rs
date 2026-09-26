//! chelis#2563 in the host interpreter: an operation in an untaken `if` arm
//! checks nothing ([05-RNG-1] with spec/06 §5.2), whether its result is
//! discarded or consumed, whether it is a float `cast` out of `i32`'s range
//! or an integer `floor_div` by zero, with and without `grad`, and in the
//! body of a `vmap(grad(...))` applied in the arm. The shapes are
//! the #2466 reviewer's, respelled canonically; `chelis eval --file` runs each
//! whole file (no top-level value here traps, so the whole-file lane is a
//! witness). The DAG-evaluator and C rows of the same shapes are in
//! `chelis-compiler-api`'s `rule_d_entered_lanes`.
use assert_cmd::Command;
use std::path::Path;

const DISCARDED_CAST: &str = "def f(x: tensor[1, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  r = if gt(s, 5.0f32) then {
    dead = cast(mul(&x, to_tensor([1e30f32])), i32)
    sum(&x, 0i32)
  } else sum(&x, 0i32)
  sum(x, 0i32)
}
out = f(to_tensor([1.0f32]))
";

const CONSUMED_CAST: &str = "def f(x: tensor[1, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  r = if gt(s, 5.0f32) then cast(cast(mul(&x, to_tensor([1e30f32])), i32), f32) else x
  sum(r, 0i32)
}
out = f(to_tensor([1.0f32]))
";

const DISCARDED_CAST_UNDER_GRAD: &str = "def loss(x: tensor[1, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  r = if gt(s, 5.0f32) then {
    dead = cast(mul(&x, to_tensor([1e30f32])), i32)
    sum(&x, 0i32)
  } else sum(&x, 0i32)
  sum(x, 0i32)
}
out = grad(loss)(to_tensor([1.0f32]))
";

const CONSUMED_CAST_UNDER_GRAD: &str = "def loss(x: tensor[1, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  r = if gt(s, 5.0f32) then cast(cast(mul(&x, to_tensor([1e30f32])), i32), f32) else x
  sum(r, 0i32)
}
out = grad(loss)(to_tensor([1.0f32]))
";

const DISCARDED_FLOOR_DIV: &str = "def f(x: tensor[1, f32], d: tensor[1, i32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&d, 0i32))
  r = if gt(s, 0i32) then {
    q = floor_div(to_tensor([7i32]), &d)
    sum(&x, 0i32)
  } else sum(&x, 0i32)
  sum(x, 0i32)
}
out = f(to_tensor([1.0f32]), to_tensor([0i32]))
";

const CONSUMED_FLOOR_DIV: &str = "def f(x: tensor[1, f32], d: tensor[1, i32]) -> tensor[i32] = {
  s = tensor_to_scalar(sum(&d, 0i32))
  r = if gt(s, 0i32) then floor_div(to_tensor([7i32]), &d) else d
  sum(r, 0i32)
}
out = f(to_tensor([1.0f32]), to_tensor([0i32]))
";

const VMAP_GRAD_CAST: &str = "def f(x: tensor[f32]) -> tensor[f32] = {
  dead = cast(mul(&x, scalar_to_tensor(1e30f32)), i32)
  mul(&x, &x)
}
def h(xs: tensor[1, f32]) -> tensor[1, f32] = {
  s = tensor_to_scalar(sum(copy(xs), 0i32))
  if gt(s, 5.0f32) then vmap(grad(f))(xs) else xs
}
out = h(to_tensor([1.0f32]))
";

const VMAP_GRAD_FLOOR_DIV: &str = "def f(x: tensor[f32]) -> tensor[f32] = {
  dead = floor_div(scalar_to_tensor(7i32), cast(mul(&x, scalar_to_tensor(0.0f32)), i32))
  mul(&x, &x)
}
def h(xs: tensor[1, f32]) -> tensor[1, f32] = {
  s = tensor_to_scalar(sum(copy(xs), 0i32))
  if gt(s, 5.0f32) then vmap(grad(f))(xs) else xs
}
out = h(to_tensor([1.0f32]))
";

const CAST_OVERFLOW: &str = "numeric trap: overflow in cast at i32";
const FLOOR_DIV_ZERO: &str = "numeric trap: division by zero in floor_div at i32";

/// Every shape, with the trap its arm raises when taken.
const SHAPES: [(&str, &str, &str); 8] = [
    ("discarded cast", DISCARDED_CAST, CAST_OVERFLOW),
    ("consumed cast", CONSUMED_CAST, CAST_OVERFLOW),
    (
        "discarded cast under grad",
        DISCARDED_CAST_UNDER_GRAD,
        CAST_OVERFLOW,
    ),
    (
        "consumed cast under grad",
        CONSUMED_CAST_UNDER_GRAD,
        CAST_OVERFLOW,
    ),
    ("discarded floor_div", DISCARDED_FLOOR_DIV, FLOOR_DIV_ZERO),
    ("consumed floor_div", CONSUMED_FLOOR_DIV, FLOOR_DIV_ZERO),
    ("cast under vmap(grad)", VMAP_GRAD_CAST, CAST_OVERFLOW),
    (
        "floor_div under vmap(grad)",
        VMAP_GRAD_FLOOR_DIV,
        FLOOR_DIV_ZERO,
    ),
];

/// The shape with its arm taken: `x` is 1.0 and `d` is 0.
fn taken(source: &str) -> String {
    source
        .replace("gt(s, 5.0f32)", "gt(s, -5.0f32)")
        .replace("gt(s, 0i32)", "gt(s, -1i32)")
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

/// Write `source`, check that it is canonical and lint-clean (so the style
/// gate cannot be what decides the row), and evaluate the whole file.
fn eval_file(directory: &Path, stem: &str, source: &str) -> std::process::Output {
    let path = format!("{stem}.ch");
    std::fs::write(directory.join(&path), source).unwrap();
    let formatted = chelis(directory, &["fmt", "--inplace", &path]);
    assert!(formatted.status.success(), "{}", text(&formatted));
    assert_eq!(
        std::fs::read_to_string(directory.join(&path)).unwrap(),
        source,
        "{stem} must already be canonical Surf"
    );
    let linted = chelis(directory, &["lint", "--check", &path]);
    assert!(linted.status.success(), "{}", text(&linted));
    chelis(directory, &["eval", "--file", &path])
}

/// Evidentiary status: HELD RED at b003b1610 in every row (each traps with
/// its taken-arm trap); the expectation is Rule D's, and the fix is the
/// activation-carrying trap seed (phase 2). The two `vmap(grad)` rows are a
/// REGRESSION TEST: at 224414e1f each traps with its taken-arm trap, because
/// that call site spliced its body without the arm's activation.
#[test]
fn an_untaken_arms_operation_checks_nothing_in_eval_file() {
    let directory = tempfile::tempdir().unwrap();
    let mut failures = Vec::new();
    for (index, (shape, source, _)) in SHAPES.into_iter().enumerate() {
        let output = eval_file(directory.path(), &format!("untaken_{index}"), source);
        if !output.status.success() {
            failures.push(format!("{shape}: {}", text(&output).trim()));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// The positive control: taken, each arm's operation traps with its typed
/// trap, so the untaken rows cannot pass by dropping the check.
///
/// Evidentiary status: DISPOSITION LOCK (green at b003b1610).
#[test]
fn a_taken_arms_operation_traps_in_eval_file() {
    let directory = tempfile::tempdir().unwrap();
    let mut failures = Vec::new();
    for (index, (shape, source, trap)) in SHAPES.into_iter().enumerate() {
        let output = eval_file(directory.path(), &format!("taken_{index}"), &taken(source));
        if output.status.success() || !text(&output).contains(trap) {
            failures.push(format!("{shape}: {}", text(&output).trim()));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
