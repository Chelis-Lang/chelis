//! Evaluation bounds nested function applications at 10,000 levels
//! (spec/08-backends.md section 1.1, chelis#3842).
//!
//! Each application of a declared function or of a function value counts one
//! level while it is in progress, a list operation's callback included, and
//! the application that would begin level 10,001 reports
//! `evaluation recursion budget exhausted: more than 10000 nested function
//! applications`. The bound is a count, so it is the same in every build and
//! on every thread that evaluates: a recursive call nested inside two
//! expressions still reaches it before the native stack check, and a
//! `chelis test` worker thread reaches it exactly as `chelis eval` does.
//!
//! Evidentiary status: REGRESSION TEST, proven failing first. With the count
//! check removed, the 10,001-level recursions evaluate and both tests fail.
//! On the base, without the evaluation segment, an unoptimized build already
//! fails the 10,000-level recursion with the stack message.

#![allow(deprecated)] // `prepare_eval` is the no-library path `chelis test` still takes

use std::collections::BTreeMap;

use chelis_compiler_api::compiler::{CompilerError, eval, prepare_eval};
use chelis_compiler_api::schema::{EvalRequest, EvalResult, SourceKind};

const BUDGET_MESSAGE: &str =
    "evaluation recursion budget exhausted: more than 10000 nested function applications";
const LEVELS: usize = 10_000;

/// `count_down(k)` nests `k + 1` applications. `call` is the recursive call
/// as the body spells it, written over `count_down(sub(k, 1i64))`.
fn source(levels: usize, call: &str) -> String {
    format!(
        "def count_down(k: i64) -> i64 = if eq(k, 0i64) then 0i64 else add(1i64, {call})\n\
         out = {{\n  _ = print(\"before\")\n  r = count_down({k}i64)\n  _ = print(\"after\")\n  r\n}}\n",
        k = levels - 1
    )
}

/// Each round of `through_fold` applies itself and the callback that `fold`
/// applies, so `through_fold(500)` nests 1,001 applications. `descend(k)`
/// nests `k + 1` above it, so `levels` in total.
fn callback_source(levels: usize) -> String {
    format!(
        "def through_fold(k: i64) -> i64 = if eq(k, 0i64) then 0i64 else \
         fold(fn (acc: i64, x: i64) -> add(acc, add(1i64, through_fold(sub(k, 1i64)))), 0i64, range(0i64, 1i64))\n\
         def descend(k: i64) -> i64 = if eq(k, 0i64) then through_fold(500i64) else add(1i64, descend(sub(k, 1i64)))\n\
         out = {{\n  _ = print(\"before\")\n  r = descend({k}i64)\n  _ = print(\"after\")\n  r\n}}\n",
        k = levels - 1_002
    )
}

const DIRECT: &str = "count_down(sub(k, 1i64))";

/// The recursive call nested `depth` additions deep, so every level holds
/// the stack of those enclosing expressions while it recurses.
fn nested_call(depth: usize) -> String {
    let mut call = DIRECT.to_string();
    for _ in 0..depth {
        call = format!("add(0i64, {call})");
    }
    call
}

fn request(source: String) -> EvalRequest {
    EvalRequest {
        source_kind: SourceKind::Surf,
        source,
        bindings: BTreeMap::new(),
    }
}

fn assert_evaluates(case: &str, result: Result<EvalResult, CompilerError>, expected: usize) {
    let result = result.unwrap_or_else(|error| panic!("{case}: evaluates: {:?}", error.errors));
    assert_eq!(result.transcript, ["before", "after"], "{case}");
    assert_eq!(
        result.roots[0].display.as_deref(),
        Some(expected.to_string().as_str()),
        "{case}"
    );
}

fn assert_exhausts_budget(case: &str, result: Result<EvalResult, CompilerError>) {
    let error = result
        .err()
        .unwrap_or_else(|| panic!("{case}: evaluated past the recursion budget"));
    assert_eq!(
        error.transcript,
        ["before"],
        "{case}: the transcript before the failure is kept, and no later effect runs"
    );
    let messages: Vec<&str> = error
        .errors
        .iter()
        .filter(|diagnostic| diagnostic.kind().as_str() == "eval_error")
        .map(|diagnostic| diagnostic.message.as_str())
        .collect();
    assert_eq!(messages, [BUDGET_MESSAGE], "{case}: {:?}", error.errors);
}

#[test]
fn nested_applications_stop_at_the_recursion_budget() {
    let direct = DIRECT;
    assert_evaluates(
        "10,000 levels",
        eval(request(source(LEVELS, direct))),
        LEVELS - 1,
    );
    assert_exhausts_budget("10,001 levels", eval(request(source(LEVELS + 1, direct))));

    // A callback a list operation applies is a level.
    assert_evaluates(
        "callbacks, 10,000 levels",
        eval(request(callback_source(LEVELS))),
        LEVELS - 1_002 + 500,
    );
    assert_exhausts_budget(
        "callbacks, 10,001 levels",
        eval(request(callback_source(LEVELS + 1))),
    );

    // A recursive call nested two expressions deep reaches the budget, not
    // the native stack check.
    let nested = nested_call(2);
    assert_evaluates(
        "nested body, 10,000 levels",
        eval(request(source(LEVELS, &nested))),
        LEVELS - 1,
    );
    assert_exhausts_budget(
        "nested body, 10,001 levels",
        eval(request(source(LEVELS + 1, &nested))),
    );
}

#[test]
fn a_test_worker_thread_reaches_the_same_budget() {
    let direct = DIRECT;
    for (levels, within) in [(LEVELS, true), (LEVELS + 1, false)] {
        let prepared = prepare_eval(request(source(levels, direct))).expect("prepare");
        let outcome = std::thread::Builder::new()
            .name("issue-3842-worker".to_string())
            .stack_size(1024 * 1024)
            .spawn(move || prepared.eval_root(BTreeMap::new(), "out"))
            .expect("worker thread starts")
            .join()
            .expect("worker thread returns");
        let case = format!("worker, {levels} levels");
        if within {
            assert_evaluates(&case, outcome, levels - 1);
        } else {
            assert_exhausts_budget(&case, outcome);
        }
    }
}

/// Definitions applied at the bottom of [`charged_source`]'s recursion. The
/// `h` chain runs as one kernel; `leaf` runs as one routed named-axis call,
/// with `tot` inlined; `grad(total)` and `grad(gated)` run as one transform;
/// `staged_kernel` and `staged_source` run as staged plans, a `reshape` whose
/// target an interpreted source stage computes.
const UNITS: &str = "\
def h1(x: tensor[2, f32]) -> tensor[2, f32] = add(x, x)
def h2(x: tensor[2, f32]) -> tensor[2, f32] = h1(h1(x))
def h3(x: tensor[2, f32]) -> tensor[2, f32] = h2(x)
def h4(x: tensor[2, f32]) -> tensor[2, f32] = h3(x)
def total(x: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(h2(x), 0i32))
def gated(x: tensor[2, f32]) -> f32 = if gt(tensor_to_scalar(sum(x, 0i32)), 100.0f32) then tensor_to_scalar(sum(h4(x), 0i32)) else tensor_to_scalar(sum(x, 0i32))
def tot[pre, post](x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)
def leaf(x: tensor[seq, f32]) -> f32 = tensor_to_scalar(tot(x))
def via(g: tensor[seq, f32] -> f32, x: tensor[seq, f32]) -> f32 = g(x)
def t1[n](x: tensor[n, f32]) -> tensor[n, f32] = add(x, x)
def t2[n](x: tensor[n, f32]) -> tensor[n, f32] = t1(t1(x))
def k1(v: i64) -> i64 = add(v, 0i64)
def staged_kernel[m, n](source: tensor[m, f32], x: tensor[n, f32]) -> tensor[*, f32] = reshape(t2(x), [numel(source)])
def staged_source[m, n](source: tensor[m, f32], x: tensor[n, f32]) -> tensor[*, f32] = reshape(x, [k1(numel(source))])
def cu(x: tensor[2, f32], k: i64) -> tensor[2, f32] = if gte(k, cast(400, i64)) then x else cu(add(x, x), add(k, cast(1, i64)))
def loss(x: tensor[2, f32]) -> tensor[f32] = sum(cu(x, cast(0, i64)), 0i32)
def cus(x: tensor[f32], k: i64) -> tensor[f32] = if gte(k, cast(300, i64)) then x else cus(add(x, x), add(k, cast(1, i64)))
def one(x: tensor[f32]) -> tensor[f32] = cus(x, cast(0, i64))
def kfold(x: tensor[2, f32]) -> tensor[2, f32] = fold(fn (acc: tensor[2, f32], v: i64) -> h2(acc), x, [1i64, 2i64])
";

/// `padder` nests `levels - charge` applications and then evaluates `unit`,
/// which is charged `charge` more.
fn charged_source(levels: usize, unit: &str, charge: usize) -> String {
    format!(
        "{UNITS}def padder(n: i64) -> i64 = if eq(n, 0i64) then {unit} else padder(sub(n, 1i64))\n\
         out = {{\n  _ = print(\"before\")\n  r = padder({k}i64)\n  _ = print(\"after\")\n  r\n}}\n",
        k = levels - charge - 1
    )
}

/// An application evaluated as a single unit is charged the deepest nesting
/// of applications its evaluation as a unit performs, every function a
/// kernel, a routed call or a transform inlines included, so it meets the
/// bound exactly where the same applications interpreted one at a time
/// would.
#[test]
fn an_application_evaluated_as_a_unit_is_charged_its_deepest_chain() {
    let tensor = "to_tensor([1.0f32, 2.0f32])";
    let cases = [
        // A kernel, by the depth of the functions it inlines.
        ("kernel h1", format!("numel(h1({tensor}))"), 1, 2),
        ("kernel h2", format!("numel(h2({tensor}))"), 2, 2),
        ("kernel h3", format!("numel(h3({tensor}))"), 3, 2),
        ("kernel h4", format!("numel(h4({tensor}))"), 4, 2),
        // A routed named-axis call: `leaf` and the `tot` it inlines, and
        // through `via`'s function parameter one more.
        ("routed leaf", format!("cast(leaf({tensor}), i64)"), 2, 3),
        (
            "routed leaf via g",
            format!("cast(via(leaf, {tensor}), i64)"),
            3,
            3,
        ),
        // A transform: `grad(total)`, `total`, and the `h2` chain.
        ("transform", format!("numel(grad(total)({tensor}))"), 4, 2),
        // A staged plan: the `t2` chain runs inside a kernel stage and is
        // charged; `k1` runs in an interpreted source stage and counts
        // itself, once.
        (
            "staged kernel stage",
            format!("numel(staged_kernel({tensor}, {tensor}))"),
            3,
            2,
        ),
        (
            "staged source stage",
            format!("numel(staged_source({tensor}, {tensor}))"),
            2,
            2,
        ),
        // A recursion a transform's lowering unrolls: `grad(loss)`, `loss`,
        // and 401 levels of `cu`; `vmap(one)`, `one`, and 301 of `cus`.
        (
            "grad unrolled",
            format!("numel(grad(loss)({tensor}))"),
            403,
            2,
        ),
        (
            "vmap unrolled",
            format!("numel(vmap(one)({tensor}))"),
            303,
            2,
        ),
        // A captured local function: `grad`, the inline `fn`, and `h3`'s
        // chain through `f`.
        (
            "captured local",
            format!(
                "numel({{\n  f = h3\n  grad(fn (x: tensor[2, f32]) -> sum(f(x), 0i32))({tensor})\n}})"
            ),
            5,
            2,
        ),
        // A callback a `fold` over a literal list applies: `kfold`, the
        // callback, and the `h2` chain.
        (
            "literal-list fold callback",
            format!("numel(kfold({tensor}))"),
            4,
            2,
        ),
    ];
    for (case, unit, charge, expected) in cases {
        assert_evaluates(
            &format!("{case}, 10,000 levels"),
            eval(request(charged_source(LEVELS, &unit, charge))),
            expected,
        );
        assert_exhausts_budget(
            &format!("{case}, 10,001 levels"),
            eval(request(charged_source(LEVELS + 1, &unit, charge))),
        );
    }
}

/// A unit's evaluation can perform both arms of a conditional, so a unit
/// whose deeper arm is not taken can stop the evaluation early: `gated`
/// takes its shallow arm, two applications interpreted one at a time, but
/// its transform performs the deeper arm too and is charged six. The
/// boundary is the same on every run and in every build.
#[test]
fn a_unit_is_charged_its_deeper_arm_even_when_it_is_not_taken() {
    let unit = "numel(grad(gated)(to_tensor([1.0f32, 2.0f32])))";
    assert_evaluates(
        "untaken arm, 10,000 charged levels",
        eval(request(charged_source(LEVELS, unit, 6))),
        2,
    );
    assert_exhausts_budget(
        "untaken arm, 10,001 charged levels",
        eval(request(charged_source(LEVELS + 1, unit, 6))),
    );
}
