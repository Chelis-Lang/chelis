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
        "must be a non-negative integer",
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
        "must be a non-negative integer",
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
