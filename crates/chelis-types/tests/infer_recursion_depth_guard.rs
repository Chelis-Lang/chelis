//! WI-1: the type checker's own deep-recursion is stack-budget-bounded.
//!
//! Every self-recursive walker in the `chelis_types::infer` check pipeline
//! (`infer_expr`/`infer_app` and ~30 sibling walkers) mutually recurses one
//! native frame per AST level over deeply-nested `app` trees -- a reef-linked
//! program such as the Shoals pricer desugars into very deep curried
//! applications. The recursion is finite but `chelis check` runs the checker
//! on a bounded stack, so deep input overflowed and aborted the process. A
//! shared `stacker::remaining_stack()` budget guard at each walker converts
//! that abort into a clean, located typed diagnostic, and a thread-local
//! exhaustion flag drained at every public check entry makes the bail ALWAYS
//! surface as a hard check failure -- never a silent green or partial result
//! (covered-or-rejected).
//!
//! Scope: these tests cover the *checker's own* inference recursion. The
//! general "never SIGSEGV on arbitrary deep input" property is NOT proven
//! here -- the foundational `chelis_deep::Expr` derived `Clone`/`Drop` still
//! overflow on deeper synthetic input, which is the deferred `maybe_grow`
//! follow-up's job (see the investigation note).
//!
//! These tests build the deep `app` tree directly in Deep (bypassing the Surf
//! parser/desugarer) so the recursion depth is exactly the chain length and
//! the test does not depend on desugaring expansion factors. They run the
//! checker on an explicitly-sized worker thread so the guard fires
//! deterministically regardless of the test harness default stack size (which
//! can be as small as 2 MiB).
//!
//! See docs/investigations/wi1_infer_recursion_depth.md and the
//! `STACK_RED_ZONE_BYTES` doc-comment in crates/chelis-types/src/infer.rs.

use chelis_deep::ast::{Atom, Expr, List, MetaMap};
use chelis_deep::span::Span;
use chelis_types::check_ir_program;

fn sym(s: &str) -> Expr {
    Expr::Atom(Atom::Symbol(s.to_string()), Span::new(0, 0))
}

fn empty_meta() -> Expr {
    Expr::Map(MetaMap::default(), Span::new(0, 0))
}

/// `(var name)` -- a 3-tuple `(tag {} children...)` Deep node.
fn var(name: &str) -> Expr {
    Expr::List(
        List {
            elements: vec![sym("var"), empty_meta(), sym(name)],
        },
        Span::new(0, 0),
    )
}

/// `(app func arg)`.
fn app(func: Expr, arg: Expr) -> Expr {
    Expr::List(
        List {
            elements: vec![sym("app"), empty_meta(), func, arg],
        },
        Span::new(0, 0),
    )
}

/// A left-nested curried application of `depth` distinct names:
/// `(app (app (app f0 f1) f2) ... fN)`. Each `app`'s function position
/// (`children[0]`) is itself an `app`, so checking the outer node drives
/// `infer_expr -> infer_app -> infer_expr` recursion exactly `depth` deep.
fn deep_app_chain(depth: usize) -> Expr {
    let mut e = var("f0");
    for i in 1..=depth {
        e = app(e, var(&format!("f{i}")));
    }
    e
}

/// Wrap an expression as the body of a top-level `def main` so the FULL check
/// pipeline (signature inference, cycle detection, precision/invariant
/// validation, annotation) runs over it, not just `infer_expr`.
fn program_with_body(body: Expr) -> Vec<Expr> {
    let params = Expr::List(
        List {
            elements: vec![sym("params"), empty_meta()],
        },
        Span::new(0, 0),
    );
    let func = Expr::List(
        List {
            elements: vec![sym("fn"), empty_meta(), params, body],
        },
        Span::new(0, 0),
    );
    let def = Expr::List(
        List {
            elements: vec![sym("def"), empty_meta(), sym("main"), func],
        },
        Span::new(0, 0),
    );
    vec![def]
}

/// Run `check_ir_program` on a worker thread with an explicit stack size, so
/// the depth guard (not a native stack overflow) is what bounds the
/// recursion. The thread join would surface a panic; a SIGSEGV would abort
/// the whole test process, so reaching the assertions at all proves the guard
/// prevented the overflow. 8 MiB matches the default `chelis check` main-thread
/// stack on Linux; the `infer_expr` budget guard fires well before a depth-4000
/// chain exhausts it.
fn check_deep_on_bounded_stack(program: Vec<Expr>, stack_mib: usize) -> Vec<String> {
    std::thread::Builder::new()
        .name("infer-depth-guard-test".to_string())
        .stack_size(stack_mib * 1024 * 1024)
        .spawn(move || match check_ir_program(&program) {
            Ok(_) => Vec::new(),
            Err(result) => result
                .errors
                .iter()
                .map(|e| e.message.clone())
                .collect::<Vec<_>>(),
        })
        .expect("spawn checker worker thread")
        .join()
        .expect("checker worker thread aborted (stack overflow?) instead of returning")
}

/// POSITIVE: a chain far deeper than the stack budget allows yields the typed
/// located depth diagnostic through the FULL pipeline, NOT a SIGSEGV. Depth
/// 4000 reliably exhausts the budget on an 8 MiB stack while staying well
/// under the depth that would overflow the same stack outright, so the guard
/// (not a crash) is what stops it.
#[test]
fn deep_app_chain_yields_stack_budget_diagnostic_not_sigsegv() {
    let messages = check_deep_on_bounded_stack(program_with_body(deep_app_chain(4000)), 8);
    assert!(
        messages
            .iter()
            .any(|m| m.contains("stack budget exhausted")),
        "expected a located 'stack budget exhausted' diagnostic for a 4000-deep \
         app chain through check_ir_program; got: {messages:?}",
    );
}

/// POSITIVE (located): the diagnostic names WHERE the depth is -- the walker
/// site -- so a valid-but-deep program tells the user where to look, rather
/// than a bare "too deep".
#[test]
fn stack_budget_diagnostic_names_the_walker_site() {
    let messages = check_deep_on_bounded_stack(program_with_body(deep_app_chain(4000)), 8);
    assert!(
        messages
            .iter()
            .any(|m| m.contains("stack budget exhausted") && m.contains("in `")),
        "the depth diagnostic must name the walker site (`in \\`...\\``); got: {messages:?}",
    );
}

/// POSITIVE (covered-or-rejected, never silent): a chain past the budget must
/// make the whole check FAIL -- `check_ir_program` returns `Err` with a
/// non-empty error list, never `Ok` and never an empty/partial result. This is
/// the soundness property: a stack bail can never be swallowed into a green
/// check.
#[test]
fn deep_app_chain_is_rejected_never_silently_passes() {
    let program = program_with_body(deep_app_chain(4000));
    let messages = check_deep_on_bounded_stack(program, 8);
    assert!(
        !messages.is_empty(),
        "a chain past the stack budget must reject (Err with errors), never \
         check clean / return Ok",
    );
}

/// NEGATIVE (no false positive, no behavior change for normal input): a
/// moderate-depth chain, comfortably within the budget, never trips the guard.
/// The unbound names still produce ordinary `unbound variable` errors, but NOT
/// the stack-depth diagnostic.
#[test]
fn moderate_depth_chain_does_not_trip_stack_guard() {
    // Depth 20 is far below any stack budget bail and below even the smallest
    // (2 MiB) native overflow depth, so it is safe on any stack.
    let messages = check_deep_on_bounded_stack(program_with_body(deep_app_chain(20)), 8);
    assert!(
        !messages
            .iter()
            .any(|m| m.contains("stack budget exhausted")),
        "a 20-deep chain is well within the stack budget and must not trip the \
         guard; got: {messages:?}",
    );
}

/// DEFERRED CAPABILITY (`#[ignore]`d): the real pricer -- or a synthetic chain
/// at the pricer's true nesting depth -- should `check` cleanly end-to-end. It
/// cannot today: a chain deep enough to model the pricer trips the stack-budget
/// guard (and, beyond `chelis_types`, the deep AST's own derived clone/drop in
/// `chelis_deep` would overflow -- see the follow-up note). This is `#[ignore]`d
/// so it anchors the deferred capability; it flips green when the
/// stack-growing (`stacker::maybe_grow`) follow-up lands.
///
/// Manual gate: `cargo nextest run -p chelis-types -- --ignored`.
/// Expected once the follow-up lands: zero `stack budget exhausted` errors for
/// a pricer-depth chain.
#[test]
#[ignore = "deferred: full-depth checking needs the WI-1 stacker::maybe_grow \
            follow-up (and the chelis_deep deep-clone/drop surface); today the \
            guard fires. See docs/investigations/wi1_infer_recursion_depth.md"]
fn pricer_depth_chain_checks_without_depth_error_once_followup_lands() {
    // 4000 stands in for the pricer's true (deep) nesting; the real pricer
    // fixture is reachable through the CLI, not this crate's unit surface.
    let messages = check_deep_on_bounded_stack(program_with_body(deep_app_chain(4000)), 8);
    assert!(
        !messages
            .iter()
            .any(|m| m.contains("stack budget exhausted")),
        "deferred: once depth handling grows the stack, a pricer-depth chain \
         must check without a stack-depth error",
    );
}
