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
