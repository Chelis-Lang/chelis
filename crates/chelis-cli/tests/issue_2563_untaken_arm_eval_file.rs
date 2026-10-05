//! chelis#2563 in the host interpreter: an operation in an untaken `if` arm
//! checks nothing ([05-RNG-1] with spec/06 §5.2), whether its result is
//! discarded or consumed, whether it is a float `cast` out of `i32`'s range
//! or an integer `floor_div` by zero, with and without `grad`, and in the
//! body of a `vmap(grad(...))` applied in the arm. The shapes are
//! the #2466 reviewer's, respelled canonically; `chelis eval --file` runs each
//! whole file (no top-level value here traps, so the whole-file lane is a
//! witness). The DAG-evaluator and C rows of the same shapes are in
//! `chelis-compiler-api`'s `rule_d_entered_lanes`.
//!
//! The same holds for every kind the activation gate gained beyond operand
//! values (an integer reduction, an empty reduced axis, runtime movement
//! bounds, a call's extent claim, a callee's result claims, a guarded abort
//! in a `grad` body), and a discarded `let` of each kind the trap seed
//! gained traps (decisions sections 6.2 and 11). `chelis-backend-c`'s
//! `exec_compile` has the DAG-evaluator and C rows of those.
//!
//! An untaken arm also contributes nothing to a gradient (spec/06 §2.10.1),
//! even where its values are not finite; a taken arm keeps its gradient.
use assert_cmd::Command;
use std::path::Path;

#[path = "common/mod.rs"]
mod common;

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

/// Under `grad` a float source cast to an integer on the gradient path is a
/// structural rejection ([04-NUM-14], chelis#2178), so this shape reaches the
/// same `overflow in cast at i32` trap through an integer source: a
/// comparison, which carries no cotangent, scaled past the `i32` range.
const CONSUMED_CAST_UNDER_GRAD: &str = "def loss(x: tensor[1, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  r = if gt(s, 5.0f32) then cast(cast(mul(cast(lt(&x, to_tensor([1e30f32])), i64), to_tensor([2147483648i64])), i32), f32) else x
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

const CONSUMED_INTEGER_SUM: &str = "def f(x: tensor[1, f32]) -> tensor[i32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  if gt(s, 5.0f32) then sum(to_tensor([2000000000i32, 2000000000i32]), 0i32) else scalar_to_tensor(7i32)
}
out = f(to_tensor([1.0f32]))
";
const DISCARDED_INTEGER_SUM: &str = "def f(x: tensor[1, f32], d: tensor[2, i32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  r = if gt(s, 5.0f32) then {
    dead = sum(&d, 0i32)
    sum(&x, 0i32)
  } else sum(&x, 0i32)
  sum(x, 0i32)
}
out = f(to_tensor([1.0f32]), to_tensor([2000000000i32, 2000000000i32]))
";
const EMPTY_MAX_REDUCE: &str = "def f(x: tensor[1, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  e = insert(scalar_to_tensor(2.0f32), 0i32, sub(shape(&x, 0i32), 1i64))
  if gt(s, 5.0f32) then max_reduce(e, 0i32) else sum(x, 0i32)
}
out = f(to_tensor([1.0f32]))
";
const EMPTY_ARGMAX_REDUCE: &str = "def f(x: tensor[1, f32]) -> tensor[i64] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  e = insert(scalar_to_tensor(2.0f32), 0i32, sub(shape(&x, 0i32), 1i64))
  if gt(s, 5.0f32) then argmax_reduce(e, 0i32) else scalar_to_tensor(7i64)
}
out = f(to_tensor([1.0f32]))
";
const SHRINK_PAST_THE_END: &str = "def f(x: tensor[4, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  if gt(s, 5.0f32) then sum(shrink(&x, [[1i64, add(shape(&x, 0i32), 3i64)]]), 0i32) else sum(x, 0i32)
}
out = f(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
";
const STRIDE_OF_ZERO: &str = "def f(x: tensor[4, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  if gt(s, 5.0f32) then sum(stride(&x, sub(shape(&x, 0i32), 4i64)), 0i32) else sum(x, 0i32)
}
out = f(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
";
const NEGATIVE_PAD: &str = "def f(x: tensor[4, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  if gt(s, 5.0f32) then sum(pad(&x, [[sub(shape(&x, 0i32), 5i64), 0i64]], 0.0f32), 0i32) else sum(x, 0i32)
}
out = f(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
";
const CALL_EXTENT_CLAIM: &str = "def g[n](a: tensor[n, f32], b: tensor[n, f32]) -> tensor[f32] = sum(a, 0i32)
def f(x: tensor[4, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  if gt(s, 5.0f32) then g(shrink(&x, [[0i64, sub(shape(&x, 0i32), 1i64)]]), copy(x)) else sum(x, 0i32)
}
out = f(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
";
const NAMED_RESULT_CLAIM: &str =
    "def g[n](x: tensor[n, f32]) -> tensor[n, f32] = shrink(x, [[1i64, shape(x, 0i32)]])
def f(x: tensor[4, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  if gt(s, 5.0f32) then sum(g(copy(x)), 0i32) else sum(x, 0i32)
}
out = f(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
";
const LITERAL_RESULT_CLAIM: &str = "def g[n](y: tensor[n, f32]) -> tensor[2, 2, f32] = reshape(y, [floor_div(shape(y, 0i32), 2i64), 2i64])
def f(x: tensor[6, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  if gt(s, 50.0f32) then sum(sum(g(copy(x)), 0i32), 0i32) else sum(x, 0i32)
}
out = f(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32]))
";
const GUARDED_FAIL_UNDER_GRAD: &str = "def loss(x: tensor[1, f32]) -> tensor[f32] = if lt(tensor_to_scalar(sum(&x, 0i32)), 5.0f32) then fail(\"guard tripped\") else sum(x, 0i32)
def h(x: tensor[1, f32]) -> tensor[1, f32] = {
  s = tensor_to_scalar(sum(copy(x), 0i32))
  if gt(s, 5.0f32) then grad(loss)(x) else x
}
out = h(to_tensor([1.0f32]))
";
const SUM_OVERFLOW: &str = "numeric trap: overflow in sum at i32";

/// Every kind the activation gate gained beyond operand values, as an
/// untaken arm, with the trap its arm raises when taken.
const GATED_SHAPES: [(&str, &str, &str); 11] = [
    ("consumed integer sum", CONSUMED_INTEGER_SUM, SUM_OVERFLOW),
    ("discarded integer sum", DISCARDED_INTEGER_SUM, SUM_OVERFLOW),
    (
        "empty max_reduce",
        EMPTY_MAX_REDUCE,
        "numeric trap: domain in max_reduce at f32",
    ),
    (
        "empty argmax_reduce",
        EMPTY_ARGMAX_REDUCE,
        "numeric trap: domain in argmax_reduce at i64",
    ),
    (
        "shrink past the end",
        SHRINK_PAST_THE_END,
        "numeric trap: domain in shrink at i64",
    ),
    (
        "stride of zero",
        STRIDE_OF_ZERO,
        "numeric trap: domain in stride at i64",
    ),
    (
        "negative pad",
        NEGATIVE_PAD,
        "numeric trap: domain in pad at i64",
    ),
    (
        "call's named extent claim",
        CALL_EXTENT_CLAIM,
        "numeric trap: domain in load at i64",
    ),
    (
        "callee's named result claim",
        NAMED_RESULT_CLAIM,
        "numeric trap: domain in shrink at i64",
    ),
    (
        "callee's literal result claim",
        LITERAL_RESULT_CLAIM,
        "numeric trap: domain in reshape at i64",
    ),
    (
        "guarded fail in a grad body",
        GUARDED_FAIL_UNDER_GRAD,
        "guard tripped",
    ),
];

/// Every kind the trap seed gained, as a discarded `let` of an entered
/// declaration (its trap) beside its total twin (a value the check accepts).
const DEAD_LETS: [(&str, &str, &str, &str); 10] = [
    (
        "integer sum (dead_sum)",
        "def f(x: tensor[1, f32]) -> tensor[f32] = {
  dead = sum(to_tensor([2000000000i32, 2000000000i32]), 0i32)
  sum(x, 0i32)
}
out = f(to_tensor([1.0f32]))
",
        "def f(x: tensor[1, f32]) -> tensor[f32] = {
  dead = sum(to_tensor([2i32, 2i32]), 0i32)
  sum(x, 0i32)
}
out = f(to_tensor([1.0f32]))
",
        SUM_OVERFLOW,
    ),
    (
        "integer product",
        "def f(x: tensor[4, f32]) -> tensor[f32] = {
  dead = prod_reduce(to_tensor([100000i32, 100000i32]), 0i32)
  sum(x, 0i32)
}
out = f(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
",
        "def f(x: tensor[4, f32]) -> tensor[f32] = {
  dead = prod_reduce(to_tensor([2i32, 3i32]), 0i32)
  sum(x, 0i32)
}
out = f(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
",
        "numeric trap: overflow in prod_reduce at i32",
    ),
    (
        "empty max_reduce",
        "def f(x: tensor[4, f32]) -> tensor[f32] = {
  e = insert(scalar_to_tensor(2.0f32), 0i32, sub(shape(&x, 0i32), 4i64))
  dead = max_reduce(e, 0i32)
  sum(x, 0i32)
}
out = f(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
",
        "def f(x: tensor[4, f32]) -> tensor[f32] = {
  e = insert(scalar_to_tensor(2.0f32), 0i32, sub(shape(&x, 0i32), 3i64))
  dead = max_reduce(e, 0i32)
  sum(x, 0i32)
}
out = f(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
",
        "numeric trap: domain in max_reduce at f32",
    ),
    (
        "empty argmax_reduce",
        "def f(x: tensor[4, f32]) -> tensor[f32] = {
  e = insert(scalar_to_tensor(2.0f32), 0i32, sub(shape(&x, 0i32), 4i64))
  dead = argmax_reduce(e, 0i32)
  sum(x, 0i32)
}
out = f(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
",
        "def f(x: tensor[4, f32]) -> tensor[f32] = {
  e = insert(scalar_to_tensor(2.0f32), 0i32, sub(shape(&x, 0i32), 3i64))
  dead = argmax_reduce(e, 0i32)
  sum(x, 0i32)
}
out = f(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
",
        "numeric trap: domain in argmax_reduce at i64",
    ),
    (
        "shrink past the end",
        "def f(x: tensor[4, f32]) -> tensor[f32] = {
  dead = shrink(&x, [[1i64, add(shape(&x, 0i32), 3i64)]])
  sum(x, 0i32)
}
out = f(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
",
        "def f(x: tensor[4, f32]) -> tensor[f32] = {
  dead = shrink(&x, [[1i64, add(shape(&x, 0i32), 0i64)]])
  sum(x, 0i32)
}
out = f(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
",
        "numeric trap: domain in shrink at i64",
    ),
    (
        "stride of zero",
        "def f(x: tensor[4, f32]) -> tensor[f32] = {
  dead = stride(&x, sub(shape(&x, 0i32), 4i64))
  sum(x, 0i32)
}
out = f(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
",
        "def f(x: tensor[4, f32]) -> tensor[f32] = {
  dead = stride(&x, sub(shape(&x, 0i32), 3i64))
  sum(x, 0i32)
}
out = f(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
",
        "numeric trap: domain in stride at i64",
    ),
    (
        "negative pad",
        "def f(x: tensor[4, f32]) -> tensor[f32] = {
  dead = pad(&x, [[sub(shape(&x, 0i32), 5i64), 0i64]], 0.0f32)
  sum(x, 0i32)
}
out = f(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
",
        "def f(x: tensor[4, f32]) -> tensor[f32] = {
  dead = pad(&x, [[sub(shape(&x, 0i32), 3i64), 0i64]], 0.0f32)
  sum(x, 0i32)
}
out = f(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
",
        "numeric trap: domain in pad at i64",
    ),
    (
        "call's named extent claim",
        "def g[n](a: tensor[n, f32], b: tensor[n, f32]) -> tensor[f32] = sum(a, 0i32)
def f(x: tensor[4, f32]) -> tensor[f32] = {
  dead = g(shrink(&x, [[0i64, sub(shape(&x, 0i32), 1i64)]]), copy(x))
  sum(x, 0i32)
}
out = f(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
",
        "def g[n](a: tensor[n, f32], b: tensor[n, f32]) -> tensor[f32] = sum(a, 0i32)
def f(x: tensor[4, f32]) -> tensor[f32] = {
  dead = g(shrink(&x, [[0i64, sub(shape(&x, 0i32), 0i64)]]), copy(x))
  sum(x, 0i32)
}
out = f(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
",
        "numeric trap: domain in load at i64",
    ),
    (
        "callee's named result claim",
        "def g[n](x: tensor[n, f32]) -> tensor[n, f32] = shrink(x, [[1i64, shape(x, 0i32)]])
def f(x: tensor[4, f32]) -> tensor[f32] = {
  dead = g(copy(x))
  sum(x, 0i32)
}
out = f(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
",
        "def g[n](x: tensor[n, f32]) -> tensor[n, f32] = shrink(x, [[0i64, shape(x, 0i32)]])
def f(x: tensor[4, f32]) -> tensor[f32] = {
  dead = g(copy(x))
  sum(x, 0i32)
}
out = f(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
",
        "numeric trap: domain in shrink at i64",
    ),
    (
        "callee's literal result claim",
        "def g[n](y: tensor[n, f32]) -> tensor[3, 2, f32] = reshape(y, [floor_div(shape(y, 0i32), 2i64), 2i64])
def f(x: tensor[4, f32]) -> tensor[f32] = {
  dead = g(copy(x))
  sum(x, 0i32)
}
out = f(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
",
        "def g[n](y: tensor[n, f32]) -> tensor[2, 2, f32] = reshape(y, [floor_div(shape(y, 0i32), 2i64), 2i64])
def f(x: tensor[4, f32]) -> tensor[f32] = {
  dead = g(copy(x))
  sum(x, 0i32)
}
out = f(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
",
        "numeric trap: domain in reshape at i64",
    ),
];

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
        .replace("gt(s, 50.0f32)", "gt(s, -5.0f32)")
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

/// Decisions section 11 for the kinds beyond operand values: untaken, each
/// arm checks nothing and the file evaluates.
///
/// Evidentiary status (224414e1f's binary on the same sources): REGRESSION
/// TEST for the consumed integer sum, both empty reductions, `shrink`,
/// `stride`, the call's extent claim and the callee's named result claim
/// (each traps there); DISPOSITION LOCK for the discarded integer sum,
/// `pad`, the literal result claim and the guarded abort (each evaluates
/// there).
#[test]
fn an_untaken_arms_reduction_movement_claim_or_abort_checks_nothing_in_eval_file() {
    let directory = tempfile::tempdir().unwrap();
    let mut failures = Vec::new();
    for (index, (shape, source, _)) in GATED_SHAPES.into_iter().enumerate() {
        let output = eval_file(directory.path(), &format!("gated_untaken_{index}"), source);
        if !output.status.success() {
            failures.push(format!("{shape}: {}", text(&output).trim()));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// The positive control: taken, each kind traps with its trap (`pad`'s is
/// the evaluator's untyped bound report).
///
/// Evidentiary status: REGRESSION TEST for the discarded integer sum (its
/// taken arm evaluated at 224414e1f: it was not a seed); DISPOSITION LOCK
/// for the rest.
#[test]
fn a_taken_arms_reduction_movement_claim_or_abort_traps_in_eval_file() {
    let directory = tempfile::tempdir().unwrap();
    let mut failures = Vec::new();
    for (index, (shape, source, trap)) in GATED_SHAPES.into_iter().enumerate() {
        let output = eval_file(
            directory.path(),
            &format!("gated_taken_{index}"),
            &taken(source),
        );
        if output.status.success() || !text(&output).contains(trap) {
            failures.push(format!("{shape}: {}", text(&output).trim()));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// Decisions sections 6.2 and 11: a discarded `let` whose initializer can
/// trap runs its check, for every kind the trap seed gained, and its total
/// twin evaluates. The integer sum row is `dead_sum`, the spec/03 section
/// 4.4 oracle.
///
/// Evidentiary status (224414e1f's binary on the same sources): REGRESSION
/// TEST for the integer sum and product, both empty reductions, `shrink`,
/// `stride`, `pad` and the callee's named result claim (each evaluates
/// there); DISPOSITION LOCK for the call's extent claim and the callee's
/// literal result claim (each traps there), and for every total twin.
#[test]
fn a_dead_let_of_each_newly_seeded_kind_traps_in_eval_file() {
    let directory = tempfile::tempdir().unwrap();
    let mut failures = Vec::new();
    for (index, (kind, dead, total, trap)) in DEAD_LETS.into_iter().enumerate() {
        let output = eval_file(directory.path(), &format!("dead_{index}"), dead);
        if output.status.success() || !text(&output).contains(trap) {
            failures.push(format!("dead {kind}: {}", text(&output).trim()));
        }
        let output = eval_file(directory.path(), &format!("total_{index}"), total);
        if !output.status.success() {
            failures.push(format!("total {kind}: {}", text(&output).trim()));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

// spec/06 §2.10.1: the untaken arm contributes nothing to a gradient, even
// where the arm's values (computed from substituted operands, or genuinely)
// are not finite. `chelis-compiler-api`'s `untaken_arm_gradients` runs the
// same shapes in the DAG evaluator and C, and documents each.
const GRAD_DIV_SUBSTITUTED: &str = "def loss(x: tensor[1, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  r = if lt(5.0f32, s) then mul(&x, log(div(&x, to_tensor([1.0f32])))) else copy(x)
  sum(r, 0i32)
}
def main() -> tensor[1, f32] = grad(loss)(to_tensor([1.0f32]))
";

const GRAD_DRAW_SUBSTITUTED: &str = "def loss(x: tensor[4, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(copy(x), 0i32))
  r = if lt(100.0f32, s) then mul(copy(x), log(uniform_like(key_from_seed(1i64), copy(x), 1.0f32, 2.0f32))) else copy(x)
  sum(r, 0i32)
}
def main() -> tensor[4, f32] = grad(loss)(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
";

const GRAD_GENUINE_LOG: &str = "def loss(x: tensor[1, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  r = if lt(5.0f32, s) then log(sub(&x, to_tensor([1.0f32]))) else copy(x)
  sum(r, 0i32)
}
def main() -> tensor[1, f32] = grad(loss)(to_tensor([1.0f32]))
";

const GRAD_GENUINE_PRODUCT: &str = "def loss(x: tensor[1, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  r = if lt(5.0f32, s) then mul(&x, log(sub(&x, to_tensor([1.0f32])))) else copy(x)
  sum(r, 0i32)
}
def main() -> tensor[1, f32] = grad(loss)(to_tensor([1.0f32]))
";

const GRAD_VMAP_GRAD_PRODUCT: &str = "def loss(x: tensor[1, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  r = if lt(5.0f32, s) then mul(&x, log(sub(&x, to_tensor([1.0f32])))) else copy(x)
  sum(r, 0i32)
}
def main() -> tensor[2, 1, f32] = vmap(grad(loss))(to_tensor([[1.0f32], [10.0f32]]))
";

const GRAD_VMAP_GRAD_DIV_SUBSTITUTED: &str = "def loss(x: tensor[1, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  r = if lt(5.0f32, s) then mul(&x, log(div(&x, to_tensor([1.0f32])))) else copy(x)
  sum(r, 0i32)
}
def main() -> tensor[2, 1, f32] = vmap(grad(loss))(to_tensor([[1.0f32], [10.0f32]]))
";

const GRAD_GRAD_OF_VMAPPED_ARM: &str = "def h(x: tensor[f32]) -> tensor[f32] = if lt(5.0f32, tensor_to_scalar(copy(x))) then mul(&x, log(sub(&x, scalar_to_tensor(1.0f32)))) else copy(x)
def loss(xs: tensor[2, f32]) -> tensor[f32] = sum(vmap(h)(xs), 0i32)
def main() -> tensor[2, f32] = grad(loss)(to_tensor([1.0f32, 10.0f32]))
";

const GRAD_GRAD_OF_VMAPPED_CAPTURE: &str = "def loss(xs: tensor[2, f32], w: tensor[f32]) -> tensor[f32] = {
  ys = vmap(fn (x: tensor[f32]) -> if lt(5.0f32, tensor_to_scalar(copy(x))) then mul(log(sub(&x, scalar_to_tensor(1.0f32))), copy(w)) else mul(x, copy(w)))(xs)
  sum(ys, 0i32)
}
def main() -> tensor[f32] = grad(loss, wrt=w)(to_tensor([1.0f32, 10.0f32]), scalar_to_tensor(2.0f32))
";

const GRAD_GRAD_OF_NESTED_VMAPPED_ARMS: &str = "def h(x: tensor[f32]) -> tensor[f32] = if lt(5.0f32, tensor_to_scalar(copy(x))) then mul(&x, log(sub(&x, scalar_to_tensor(1.0f32)))) else copy(x)
def row(xs: tensor[2, f32]) -> tensor[2, f32] = if lt(0.0f32, tensor_to_scalar(sum(copy(xs), 0i32))) then vmap(h)(xs) else mul(xs, to_tensor([3.0f32, 3.0f32]))
def loss(xss: tensor[2, 2, f32]) -> tensor[f32] = sum(sum(vmap(row)(xss), 0i32), 0i32)
def main() -> tensor[2, 2, f32] = grad(loss)(to_tensor([[1.0f32, 10.0f32], [-1.0f32, -2.0f32]]))
";

const GRAD_TAKEN_DIV: &str = "def loss(x: tensor[1, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  r = if lt(s, 5.0f32) then mul(&x, log(div(&x, to_tensor([1.0f32])))) else copy(x)
  sum(r, 0i32)
}
def main() -> tensor[1, f32] = grad(loss)(to_tensor([1.0f32]))
";

const GRAD_TAKEN_GENUINE_LOG: &str = "def loss(x: tensor[1, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  r = if lt(s, 5.0f32) then log(sub(&x, to_tensor([1.0f32]))) else copy(x)
  sum(r, 0i32)
}
def main() -> tensor[1, f32] = grad(loss)(to_tensor([1.0f32]))
";

const GRAD_TAKEN_GENUINE_PRODUCT: &str = "def loss(x: tensor[1, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  r = if lt(s, 5.0f32) then mul(&x, log(sub(&x, to_tensor([1.0f32])))) else copy(x)
  sum(r, 0i32)
}
def main() -> tensor[1, f32] = grad(loss)(to_tensor([1.0f32]))
";

const GRAD_VMAP_GRAD_TAKEN_GENUINE_LOG: &str = "def loss(x: tensor[1, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  r = if lt(s, 5.0f32) then log(sub(&x, to_tensor([1.0f32]))) else copy(x)
  sum(r, 0i32)
}
def main() -> tensor[2, 1, f32] = vmap(grad(loss))(to_tensor([[1.0f32], [10.0f32]]))
";

/// Each untaken row returns the else arm's gradient alone.
const UNTAKEN_GRADIENTS: [(&str, &str, &str); 9] = [
    (
        "div substituted",
        GRAD_DIV_SUBSTITUTED,
        "main = tensor(shape=[1], data=[1.0])",
    ),
    (
        "draw substituted",
        GRAD_DRAW_SUBSTITUTED,
        "main = tensor(shape=[4], data=[1.0, 1.0, 1.0, 1.0])",
    ),
    (
        "genuine log",
        GRAD_GENUINE_LOG,
        "main = tensor(shape=[1], data=[1.0])",
    ),
    (
        "genuine product",
        GRAD_GENUINE_PRODUCT,
        "main = tensor(shape=[1], data=[1.0])",
    ),
    (
        "vmap(grad) product",
        GRAD_VMAP_GRAD_PRODUCT,
        "main = tensor(shape=[2, 1], data=[1.0, 3.3083358])",
    ),
    (
        "vmap(grad) div substituted",
        GRAD_VMAP_GRAD_DIV_SUBSTITUTED,
        "main = tensor(shape=[2, 1], data=[1.0, 3.3025851])",
    ),
    (
        "grad of a vmapped arm",
        GRAD_GRAD_OF_VMAPPED_ARM,
        "main = tensor(shape=[2], data=[1.0, 3.3083358])",
    ),
    (
        "grad of a vmapped capture",
        GRAD_GRAD_OF_VMAPPED_CAPTURE,
        "main = 3.1972246",
    ),
    (
        "grad of nested vmapped arms",
        GRAD_GRAD_OF_NESTED_VMAPPED_ARMS,
        "main = tensor(shape=[2, 2], data=[1.0, 3.3083358, 3.0, 3.0])",
    ),
];

/// Each taken row keeps its true gradient, a non-finite one included.
const TAKEN_GRADIENTS: [(&str, &str, &str); 4] = [
    (
        "taken div",
        GRAD_TAKEN_DIV,
        "main = tensor(shape=[1], data=[1.0])",
    ),
    (
        "taken genuine log",
        GRAD_TAKEN_GENUINE_LOG,
        "main = tensor(shape=[1], data=[inf])",
    ),
    (
        "taken genuine product",
        GRAD_TAKEN_GENUINE_PRODUCT,
        "main = tensor(shape=[1], data=[NaN])",
    ),
    (
        "vmap(grad) taken genuine log",
        GRAD_VMAP_GRAD_TAKEN_GENUINE_LOG,
        "main = tensor(shape=[2, 1], data=[inf, 1.0])",
    ),
];

/// Every row's `eval --file` output is exactly its expected line.
fn check_gradients(prefix: &str, rows: &[(&str, &str, &str)]) {
    let directory = tempfile::tempdir().unwrap();
    let mut failures = Vec::new();
    for (index, (row, source, expected)) in rows.iter().enumerate() {
        let output = eval_file(directory.path(), &format!("{prefix}_{index}"), source);
        if !output.status.success() || String::from_utf8_lossy(&output.stdout).trim() != *expected {
            failures.push(format!("{row}: {}", text(&output).trim()));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// Evidentiary status: REGRESSION TEST for the substituted rows (NaN at
/// b47fdd7d3) and the genuine rows (NaN on main 7807ca4ff, chelis#2640).
#[test]
fn an_untaken_arm_contributes_nothing_to_a_gradient_in_eval_file() {
    check_gradients("gradient_untaken", &UNTAKEN_GRADIENTS);
}

/// Evidentiary status: DISPOSITION LOCK (every row holds at b47fdd7d3).
#[test]
fn a_taken_arm_keeps_its_gradient_in_eval_file() {
    check_gradients("gradient_taken", &TAKEN_GRADIENTS);
}

/// An arm whose extent rests on a claim, with the condition spelling that
/// leaves it untaken and the one that takes it with the claim false, and the
/// claim's typed trap. `{x}` is [`x32`]. The rows are the round-2b witnesses
/// of #2586: (a) a callee's result claim, (b) a guarded broadcast's unit
/// claim, (c) a local ascription, and a parameter's unit refinement.
const CLAIMED_ARMS: [(&str, &str, (&str, &str), &str); 7] = [
    (
        "(a) callee's result claim",
        "def drop_last(y: tensor[*, f32]) -> tensor[3, f32] = shrink(y, [[0i64, sub(shape(&y, 0i32), 1i64)]])
def selected(x: tensor[32, f32]) -> tensor[f32] = if eq(shape(&x, 0i32), 4i64) then sum(add(drop_last(x), to_tensor([1.0f32, 2.0f32, 3.0f32])), 0i32) else sum(x, 0i32)
def main() -> tensor[f32] = selected({x})
",
        ("eq(shape(&x, 0i32), 4i64)", "eq(shape(&x, 0i32), 32i64)"),
        "extent `3`: claimed = 3, shrink axis 0 = 31",
    ),
    (
        "(a) callee's result claim under a data condition",
        "def g(y: tensor[*, f32]) -> tensor[3, f32] = shrink(y, [[0i64, sub(shape(&y, 0i32), 1i64)]])
def selected(x: tensor[32, f32]) -> tensor[32, f32] = {
  s = tensor_to_scalar(sum(copy(x), 0i32))
  if lt(s, 0.0f32) then add(x, insert(sum(add(g(copy(x)), to_tensor([1.0f32, 2.0f32, 3.0f32])), 0i32), 0i32, 32i64)) else x
}
def main() -> tensor[32, f32] = selected({x})
",
        ("lt(s, 0.0f32)", "lt(0.0f32, s)"),
        "extent `3`: claimed = 3, shrink axis 0 = 31",
    ),
    (
        "(b) guarded broadcast of a local",
        "def selected(x: tensor[32, f32]) -> tensor[32, f32] = {
  t = shrink(copy(x), [[0i64, sub(shape(&x, 0i32), 29i64)]])
  if eq(shape(&t, 0i32), 1i64) then add(x, expand(t, 0i32, 32i64)) else x
}
def main() -> tensor[32, f32] = selected({x})
",
        ("eq(shape(&t, 0i32), 1i64)", "eq(shape(&t, 0i32), 3i64)"),
        "numeric trap: domain in expand at i64",
    ),
    (
        "(b) guarded broadcast of a local sized by a scalar",
        "def selected(x: tensor[32, f32], i: tensor[1, i64]) -> tensor[32, f32] = {
  m = tensor_to_scalar(sum(copy(i), 0i32))
  t = shrink(copy(x), [[0i64, m]])
  if eq(tensor_to_scalar(sum(i, 0i32)), 1i64) then add(x, expand(t, 0i32, 32i64)) else x
}
def main() -> tensor[32, f32] = selected({x}, to_tensor([3i64]))
",
        (
            "eq(tensor_to_scalar(sum(i, 0i32)), 1i64)",
            "eq(tensor_to_scalar(sum(i, 0i32)), 3i64)",
        ),
        "numeric trap: domain in expand at i64",
    ),
    (
        "(c) local ascription under a data condition",
        "def selected(x: tensor[32, f32]) -> tensor[32, f32] = {
  s = tensor_to_scalar(sum(copy(x), 0i32))
  if lt(s, 0.0f32) then {
    y: tensor[3, f32] = shrink(copy(x), [[0i64, sub(shape(&x, 0i32), 27i64)]])
    add(x, insert(sum(add(y, to_tensor([1.0f32, 2.0f32, 3.0f32])), 0i32), 0i32, 32i64))
  } else x
}
def main() -> tensor[32, f32] = selected({x})
",
        ("lt(s, 0.0f32)", "lt(0.0f32, s)"),
        "extent `3`: claimed = 3, shrink axis 0 = 5",
    ),
    (
        "(c) local ascription",
        "def selected(x: tensor[32, f32]) -> tensor[f32] = if eq(shape(&x, 0i32), 4i64) then {
  y: tensor[3, f32] = shrink(copy(x), [[0i64, sub(shape(&x, 0i32), 1i64)]])
  sum(add(y, to_tensor([1.0f32, 2.0f32, 3.0f32])), 0i32)
} else sum(x, 0i32)
def main() -> tensor[f32] = selected({x})
",
        ("eq(shape(&x, 0i32), 4i64)", "eq(shape(&x, 0i32), 32i64)"),
        "extent `3`: claimed = 3, shrink axis 0 = 31",
    ),
    (
        "parameter's unit refinement",
        "def selected[n](x: tensor[32, f32], t: tensor[n, f32]) -> tensor[32, f32] = if eq(shape(&t, 0i32), 1i64) then add(x, expand(t, 0i32, 32i64)) else x
def main() -> tensor[32, f32] = selected({x}, to_tensor([1.0f32, 2.0f32, 3.0f32]))
",
        ("eq(shape(&t, 0i32), 1i64)", "eq(shape(&t, 0i32), 3i64)"),
        "extent `1`: claimed = 1, t axis 0 = 3",
    ),
];

/// `[-3, -1, ..., 59]`: 32 elements summing to 896.
fn x32() -> String {
    let elements = (0..32)
        .map(|index| format!("{}.0f32", 2 * index - 3))
        .collect::<Vec<_>>();
    format!("to_tensor([{}])", elements.join(", "))
}

/// The whole-program C lane: `chelis build --target c`, linked and run.
/// Its stdout on success, or its build or run output on failure.
fn c_file(directory: &Path, stem: &str, source: &str) -> Result<String, String> {
    let path = directory.join(format!("{stem}.ch"));
    std::fs::write(&path, source).unwrap();
    let out_dir = directory.join(format!("{stem}-out"));
    let built = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
        ])
        .arg(&out_dir)
        .output()
        .unwrap();
    if !built.status.success() {
        return Err(format!("build: {}", text(&built)));
    }
    let linked = common::link_generated(&out_dir, &format!("{stem}.c"), stem);
    assert!(linked.success(), "{stem}: link failed: {linked}");
    let run = std::process::Command::new(out_dir.join(stem))
        .output()
        .unwrap();
    if run.status.success() {
        Ok(String::from_utf8_lossy(&run.stdout).into_owned())
    } else {
        Err(text(&run))
    }
}

/// The host interpreter's lane, style gate off: `chelis eval --file`.
fn h_file(directory: &Path, stem: &str, source: &str) -> Result<String, String> {
    let path = format!("{stem}.ch");
    std::fs::write(directory.join(&path), source).unwrap();
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .current_dir(directory)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", &path])
        .output()
        .unwrap();
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    } else {
        Err(text(&output))
    }
}

/// #2586 round 2b: an untaken arm whose extent rests on a claim checks
/// nothing, so the host interpreter and the whole-program C return the other
/// branch's value, the same in both; taken with the claim false, both trap
/// with the claim's typed extent message. The DAG-evaluator and selected-C
/// rows are in `chelis-backend-c`'s `exec_compile`.
///
/// Evidentiary status: REGRESSION TEST for every untaken row at 096daea8c
/// (C traps (a) and the parameter row, aborts (b) untyped, and every lane
/// fails (c)); DISPOSITION LOCK for the taken rows.
#[test]
fn an_untaken_claimed_arm_checks_nothing_and_a_taken_one_traps_in_eval_file_and_c() {
    let directory = tempfile::tempdir().unwrap();
    let mut failures = Vec::new();
    for (index, (row, source, (untaken, taken), trap)) in CLAIMED_ARMS.into_iter().enumerate() {
        let source = source.replace("{x}", &x32());
        assert!(source.contains(untaken), "{row}");
        let h = h_file(
            directory.path(),
            &format!("claimed_untaken_{index}"),
            &source,
        );
        let c = c_file(
            directory.path(),
            &format!("claimed_untaken_{index}"),
            &source,
        );
        match (&h, &c) {
            (Ok(h), Ok(c)) if h == c => {}
            _ => failures.push(format!("{row}, untaken: H {h:?}, C {c:?}")),
        }
        let source = source.replace(untaken, taken);
        let h = h_file(directory.path(), &format!("claimed_taken_{index}"), &source);
        let c = c_file(directory.path(), &format!("claimed_taken_{index}"), &source);
        match (&h, &c) {
            (Err(h), Err(c)) if h.contains(trap) && c.contains(trap) => {}
            _ => failures.push(format!("{row}, taken: H {h:?}, C {c:?}")),
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// [05-OP-53] in the host interpreter's `where` builtin: the condition's
/// shape equals the shape of every branch it selects, and a branch it selects
/// nowhere is not shape-checked. A uniform condition returns its branch when
/// that branch is shaped like the condition, whatever the other branch's
/// extent, and fails with the typed shape error when it is not; a mixed
/// condition over branches of different extents fails with it too (decisions
/// section 25). `b` is `a` without its last element.
///
/// Evidentiary status: REGRESSION TEST for the two refused uniform rows (at
/// 1a026f823 each returns its selected branch); DISPOSITION LOCK for the
/// accepted uniform rows and the mixed row.
#[test]
fn the_where_builtin_checks_only_the_branches_its_condition_selects_in_eval_file() {
    let source = "def pick(c: tensor[*, bool], a: tensor[*, f32], b: tensor[*, f32]) -> tensor[*, f32] = where(&c, &a, &b)
def main() -> tensor[*, f32] = {
  a = to_tensor([1.0f32, 2.0f32, 3.0f32])
  b = shrink(copy(a), [[0i64, sub(shape(&a, 0i32), 1i64)]])
  pick(to_tensor({c}), a, b)
}
";
    // spec/04-type-system.md section 4.7: a `Domain` trap in `where`.
    let shape_error = "numeric trap: domain in where at i64";
    let directory = tempfile::tempdir().unwrap();
    let mut failures = Vec::new();
    for (index, (condition, expected)) in [
        (
            "[true, true, true]",
            Ok("main = tensor(shape=[3], data=[1.0, 2.0, 3.0])"),
        ),
        (
            "[false, false]",
            Ok("main = tensor(shape=[2], data=[1.0, 2.0])"),
        ),
        ("[true, true]", Err(shape_error)),
        ("[false, false, false]", Err(shape_error)),
        ("[true, false, true]", Err(shape_error)),
    ]
    .into_iter()
    .enumerate()
    {
        let outcome = h_file(
            directory.path(),
            &format!("where_{index}"),
            &source.replace("{c}", condition),
        );
        match (&outcome, expected) {
            (Ok(stdout), Ok(line)) if stdout.trim() == line => {}
            (Err(stderr), Err(error)) if stderr.contains(error) => {}
            _ => failures.push(format!("{condition}: {outcome:?}")),
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// [05-OP-53]: a uniform condition shaped unlike the branch it selects, by
/// extent, by axis order, or with the unselected branch shaped like the
/// selected one, is refused. The host interpreter fails with the typed shape
/// error; the whole-program C checks the operands' agreement before `where`
/// reads them and fails when it runs, since the graph proves neither
/// agreement nor a contradiction between the wildcard extents
/// (runtime_extents.md C2.3, chelis#2642).
///
/// Evidentiary status: REGRESSION TEST for every H row (at 1a026f823 each
/// returns its selected branch); DISPOSITION LOCK for the C rows.
#[test]
fn a_condition_shaped_unlike_its_selected_branch_is_refused_in_eval_file_and_c() {
    let pick = |rank: &str| {
        format!(
            "def pick(c: tensor[{rank}bool], a: tensor[{rank}f32], b: tensor[{rank}f32]) -> tensor[{rank}f32] = where(c, a, b)\n"
        )
    };
    let rows = [
        (
            "an all-true condition longer than its branch",
            format!(
                "{}def main() -> tensor[*, f32] = pick(to_tensor([true, true, true, true, true]), to_tensor([1.0f32, 2.0f32, 3.0f32]), to_tensor([4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32]))\n",
                pick("*, ")
            ),
        ),
        (
            "an all-true condition transposed against its branch",
            format!(
                "{}def main() -> tensor[*, *, f32] = pick(to_tensor([[true, true, true], [true, true, true]]), to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32], [5.0f32, 6.0f32]]), to_tensor([[7.0f32, 8.0f32], [9.0f32, 1.0f32], [2.0f32, 3.0f32]]))\n",
                pick("*, *, ")
            ),
        ),
        (
            "an all-false condition longer than both branches",
            format!(
                "{}def main() -> tensor[*, f32] = pick(to_tensor([false, false, false, false]), to_tensor([1.0f32]), to_tensor([9.0f32]))\n",
                pick("*, ")
            ),
        ),
    ];
    let directory = tempfile::tempdir().unwrap();
    let mut failures = Vec::new();
    for (index, (row, source)) in rows.iter().enumerate() {
        let stem = format!("where_selected_{index}");
        let h = h_file(directory.path(), &stem, source);
        let h_refused = h
            .as_ref()
            .is_err_and(|stderr| stderr.contains("where operands disagree"));
        if !h_refused {
            failures.push(format!("{row}, H: {h:?}"));
        }
        let c = c_file(directory.path(), &stem, source);
        let c_refused = c.as_ref().is_err_and(|stderr| {
            !stderr.starts_with("build: ") && stderr.contains("operands disagree at axis")
        });
        if !c_refused {
            failures.push(format!("{row}, C: {c:?}"));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// spec/10 §3.2 and [05-OP-53]: a `where` under a false activation checks
/// nothing, so an untaken arm's inner join whose condition selects the
/// claim-sized zeros of `id0` (extent 0, against a condition of extent 3)
/// neither traps nor is read, and both lanes return the taken `else` arm's
/// sum, 6. The twin nests the same condition, so the inner join selects
/// the same branch as the outer.
///
/// Evidentiary status: REGRESSION TEST for the nested row in H (at
/// 0b20dcad5 it fails with the evaluator's `where` shape error);
/// DISPOSITION LOCK for its C lane and for the twin.
#[test]
fn an_untaken_arms_inner_where_checks_nothing_in_eval_file_and_c() {
    let source = "def id0(v: tensor[*, f32]) -> tensor[0, f32] = add(copy(v), v)
def selected(x: tensor[32, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(copy(x), 0i32))
  y = shrink(copy(x), [[1i64, sub(shape(&x, 0i32), 28i64)]])
  r = if lt(s, 0.0f32) then (if {inner} then id0(copy(y)) else neg(copy(y))) else add(copy(y), y)
  sum(r, 0i32)
}
def main() -> tensor[f32] = selected(to_tensor([-3.0f32, -1.0f32, 1.0f32, 3.0f32, 5.0f32, 7.0f32, 9.0f32, 11.0f32, 13.0f32, 15.0f32, 17.0f32, 19.0f32, 21.0f32, 23.0f32, 25.0f32, 27.0f32, 29.0f32, 31.0f32, 33.0f32, 35.0f32, 37.0f32, 39.0f32, 41.0f32, 43.0f32, 45.0f32, 47.0f32, 49.0f32, 51.0f32, 53.0f32, 55.0f32, 57.0f32, 59.0f32]))
";
    let directory = tempfile::tempdir().unwrap();
    let mut failures = Vec::new();
    for (index, inner) in ["lt(0.0f32, s)", "lt(s, 0.0f32)"].into_iter().enumerate() {
        let program = source.replace("{inner}", inner);
        let stem = format!("inner_where_{index}");
        let h = h_file(directory.path(), &stem, &program);
        let c = c_file(directory.path(), &stem, &program);
        let expected = |lane: &Result<String, String>| {
            lane.as_ref()
                .is_ok_and(|stdout| stdout.trim() == "main = 6.0")
        };
        if !expected(&h) || !expected(&c) {
            failures.push(format!("{inner}: H {h:?}, C {c:?}"));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
