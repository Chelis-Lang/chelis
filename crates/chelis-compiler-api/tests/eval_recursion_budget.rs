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

/// A function chain `c0 .. c{depth}` over `ty`, each applying the one before:
/// interpreted, applying `c{depth}` nests `depth + 1` applications.
fn chain(prefix: &str, generic: &str, ty: &str, depth: usize) -> String {
    let mut chain = format!("def {prefix}_c0{generic}(x: {ty}) -> {ty} = add(x, x)\n");
    for index in 1..=depth {
        chain.push_str(&format!(
            "def {prefix}_c{index}{generic}(x: {ty}) -> {ty} = {prefix}_c{}(add(x, x))\n",
            index - 1
        ));
    }
    chain
}

/// The work inside a unit: the chain applied directly, or through a `fold`
/// or `map` callback, or through a nested `grad`.
#[derive(Clone, Copy, Debug)]
enum Inner {
    Chain,
    Fold,
    Map,
    Transform,
}

/// An application the evaluator runs as one compiled unit.
#[derive(Clone, Copy, Debug)]
enum Unit {
    Kernel,
    Staged,
    Grad,
    Vmap,
    NamedAxis,
    LocalAscription,
}

/// Where the unit is applied: directly at the bottom of the interpreted
/// recursion, or inside an interpreted `fold` callback there.
#[derive(Clone, Copy, Debug)]
enum Position {
    Direct,
    Callback,
}

/// A program whose interpreted applications, counted from its source, total
/// `levels`, with `unit` applied at the bottom of an interpreted recursion and
/// nesting `depth + 1` further applications inside its compiled body.
fn unit_source(
    unit: Unit,
    inner: Inner,
    position: Position,
    depth: usize,
    levels: usize,
) -> String {
    let tensor = "to_tensor([1.0f32, 2.0f32])";
    let (prefix, generic, ty) = match unit {
        Unit::Kernel | Unit::Grad => ("two", "", "tensor[2, f32]"),
        Unit::Staged => ("gen", "[n]", "tensor[n, f32]"),
        Unit::Vmap => ("scalar", "", "tensor[f32]"),
        Unit::NamedAxis => ("seq", "", "tensor[seq, f32]"),
        Unit::LocalAscription => ("star", "", "tensor[*, f32]"),
    };
    let mut program = chain(prefix, generic, ty, depth);
    program.push_str(
        "def tot[pre, post](x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\n",
    );
    let top = format!("{prefix}_c{depth}");
    let total = match unit {
        Unit::Vmap => format!("def inner_total(x: {ty}) -> tensor[f32] = {top}(x)\n"),
        _ => format!("def inner_total{generic}(x: {ty}) -> tensor[f32] = sum({top}(x), 0i32)\n"),
    };
    program.push_str(&total);
    let body = match inner {
        Inner::Chain => format!("{top}(x)"),
        Inner::Fold => format!("fold(fn (acc: {ty}, v: i64) -> {top}(acc), x, [1i64])"),
        Inner::Map => format!("index(map(fn (v: i64) -> {top}(x), [1i64]), 0i64)"),
        Inner::Transform => "grad(inner_total)(x)".to_owned(),
    };
    let (definition, call) = match unit {
        Unit::Kernel => (
            format!("def unit(x: tensor[2, f32]) -> tensor[2, f32] = {body}\n"),
            format!("numel(unit({tensor}))"),
        ),
        Unit::Staged => (
            format!(
                "def unit[m, n](source: tensor[m, f32], x: tensor[n, f32]) -> tensor[*, f32] = \
                 reshape({body}, [numel(source)])\n"
            ),
            format!("numel(unit({tensor}, {tensor}))"),
        ),
        Unit::Grad => (
            format!("def unit(x: tensor[2, f32]) -> tensor[f32] = sum({body}, 0i32)\n"),
            format!("numel(grad(unit)({tensor}))"),
        ),
        Unit::Vmap => (
            format!("def unit(x: tensor[f32]) -> tensor[f32] = {body}\n"),
            format!("numel(vmap(unit)({tensor}))"),
        ),
        Unit::NamedAxis => (
            format!("def unit(x: tensor[seq, f32]) -> f32 = tensor_to_scalar(tot({body}))\n"),
            format!("if gt(unit({tensor}), 0.0f32) then 1i64 else 0i64"),
        ),
        // `unit` prints, so it is interpreted; its ascribed binding runs as
        // one DAG, which is no application.
        Unit::LocalAscription => (
            format!(
                "def unit(x: tensor[*, f32]) -> tensor[f32] = {{\n  _ = print(\"in\")\n  \
                 y: tensor[2, f32] = {body}\n  sum(y, 0i32)\n}}\n"
            ),
            format!("numel(unit({tensor}))"),
        ),
    };
    program.push_str(&definition);
    // Above `padder`: the unit, one level, and in `Callback` position the
    // interpreted callback that applies it.
    let (above, leaf) = match position {
        Position::Direct => (1, call),
        Position::Callback => (
            2,
            format!("fold(fn (acc: i64, v: i64) -> add(acc, {call}), 0i64, [1i64])"),
        ),
    };
    // `padder(k)` nests `k + 1` applications.
    let k = levels - above - 1;
    program.push_str(&format!(
        "def padder(n: i64) -> i64 = if eq(n, 0i64) then {leaf} else padder(sub(n, 1i64))\n\
         out = padder({k}i64)\n"
    ));
    program
}

/// An application the evaluator runs as one compiled unit counts one level,
/// however many applications its compiled body nests: the trip point is the
/// count of interpreted applications in the source, whatever the depth
/// inside the unit (spec/08-backends.md section 1.1). Every unit kind with
/// every kind of inner work that compiles into it, in both positions, at two
/// inner depths: 10,000 counted levels evaluate and 10,001 report the budget.
///
/// Evidentiary status: REGRESSION TEST. A unit that also counted the
/// applications its body nests would trip at 10,000 for the deeper inner
/// chain; a unit that counted nothing would evaluate at 10,001.
#[test]
fn a_compiled_unit_counts_one_level_whatever_it_nests() {
    use Inner::{Chain, Fold, Map, Transform};
    // The inner work each unit compiles. A kernel, a staged plan or a
    // named-axis call that holds a list callback, or an ascription whose
    // initializer does, is not one unit: it is interpreted application by
    // application, which the other tests count.
    let cases = [
        (Unit::Kernel, [Chain, Transform].as_slice()),
        (Unit::Staged, [Chain, Transform].as_slice()),
        (Unit::Grad, [Chain, Fold, Map, Transform].as_slice()),
        (Unit::Vmap, [Chain, Fold, Map, Transform].as_slice()),
        (Unit::NamedAxis, [Chain, Transform].as_slice()),
        (Unit::LocalAscription, [Chain, Transform].as_slice()),
    ];
    for (unit, inners) in cases {
        for &inner in inners {
            for position in [Position::Direct, Position::Callback] {
                for depth in [0, 24] {
                    let case = format!("{unit:?} {inner:?} {position:?} inner depth {depth}");
                    let within = eval(request(unit_source(unit, inner, position, depth, LEVELS)));
                    let past = eval(request(unit_source(
                        unit,
                        inner,
                        position,
                        depth,
                        LEVELS + 1,
                    )));
                    assert!(
                        within.is_ok(),
                        "{case}, 10,000 levels: {:?}",
                        within.err().map(|error| error.errors)
                    );
                    let error = past.expect_err(&format!("{case}, 10,001 levels: evaluated"));
                    assert!(
                        error
                            .errors
                            .iter()
                            .any(|diagnostic| diagnostic.message == BUDGET_MESSAGE),
                        "{case}, 10,001 levels: {:?}",
                        error.errors
                    );
                }
            }
        }
    }
}
