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
//! Scope: these tests cover the per-site guard (the safety net) AND the WI-1
//! stack-growing follow-up. The follow-up runs the whole check pipeline on a
//! freshly-grown stack segment (`with_grown_stack` -> `stacker::grow`), so a
//! legitimately deep-but-finite program checks end-to-end -- covering not just
//! the checker's own inference recursion but the validate / annotate passes and
//! the foundational `chelis_deep::Expr` derived `Clone`/`Drop`, all of which run
//! inside that one grown segment. The safety-net tests shrink the segment (via
//! the test-only `set_grow_segment_bytes_for_test`) to confirm the per-site
//! guard still converts an overflow of even a grown segment into a located
//! diagnostic rather than a SIGSEGV (see the investigation note).
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

/// Run `check_ir_program` on a worker thread sized at `stack_mib`. The WI-1
/// follow-up grows the stack at the check entry, so the per-site guard fires
/// only when the recursion exhausts the GROWN segment, not the worker thread.
/// `grow_segment_bytes` overrides that grown-segment size (via the test-only
/// `set_grow_segment_bytes_for_test`, installed on this worker thread before
/// the check), so the safety-net tests can shrink it and drive a chain deep
/// enough to overflow it -- `None` keeps the production 512 MiB segment, used
/// by the capability test that must check a deep program cleanly.
///
/// The thread join would surface a panic; a SIGSEGV would abort the whole test
/// process, so reaching the assertions at all proves the guard prevented the
/// overflow.
fn check_deep_on_bounded_stack(
    program: Vec<Expr>,
    stack_mib: usize,
    grow_segment_bytes: Option<usize>,
) -> Vec<String> {
    std::thread::Builder::new()
        .name("infer-depth-guard-test".to_string())
        .stack_size(stack_mib * 1024 * 1024)
        .spawn(move || {
            if let Some(bytes) = grow_segment_bytes {
                chelis_types::set_grow_segment_bytes_for_test(bytes);
            }
            match check_ir_program(&program) {
                Ok(_) => Vec::new(),
                Err(result) => result
                    .errors
                    .iter()
                    .map(|e| e.message.clone())
                    .collect::<Vec<_>>(),
            }
        })
        .expect("spawn checker worker thread")
        .join()
        .expect("checker worker thread aborted (stack overflow?) instead of returning")
}

/// A grown-segment size small enough that a depth-4000 `app` chain exhausts the
/// guard's byte budget (so the per-site guard fires with a located diagnostic),
/// but large enough that the guard's 128 KiB red zone catches the recursion
/// well before a raw native overflow. 8 MiB matches the default `chelis check`
/// main-thread stack on Linux, the profile the original guard tests pinned: a
/// depth-4000 chain reliably trips the budget on 8 MiB while staying under the
/// depth (~1550 release / far more debug headroom via the red zone) that would
/// overflow it outright, so the guard -- not a crash -- is what stops it. A much
/// smaller segment (e.g. 2 MiB) leaves too little margin between the red zone
/// and the segment end for a single deep `infer_expr`/`infer_app` step plus the
/// validate passes, and can SIGSEGV before the guard fires.
const SAFETY_NET_SEGMENT_BYTES: usize = 8 * 1024 * 1024;

/// POSITIVE (safety net): a chain far deeper than a (shrunk) grown segment
/// allows yields the typed located depth diagnostic through the FULL pipeline,
/// NOT a SIGSEGV. With the grown segment shrunk to 2 MiB, depth 4000 reliably
/// exhausts the budget while staying well under the depth that would overflow
/// the same segment outright, so the per-site guard (not a crash) is what stops
/// it. This proves the guard still backs up the grow for input deeper than a
/// segment can hold.
#[test]
fn deep_app_chain_yields_stack_budget_diagnostic_not_sigsegv() {
    let messages = check_deep_on_bounded_stack(
        program_with_body(deep_app_chain(4000)),
        8,
        Some(SAFETY_NET_SEGMENT_BYTES),
    );
    assert!(
        messages
            .iter()
            .any(|m| m.contains("stack budget exhausted")),
        "expected a located 'stack budget exhausted' diagnostic for a 4000-deep \
         app chain that overflows a shrunk grown segment; got: {messages:?}",
    );
}

/// POSITIVE (located): the diagnostic names WHERE the depth is -- the walker
/// site -- so a valid-but-deep program tells the user where to look, rather
/// than a bare "too deep".
#[test]
fn stack_budget_diagnostic_names_the_walker_site() {
    let messages = check_deep_on_bounded_stack(
        program_with_body(deep_app_chain(4000)),
        8,
        Some(SAFETY_NET_SEGMENT_BYTES),
    );
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
    let messages = check_deep_on_bounded_stack(program, 8, Some(SAFETY_NET_SEGMENT_BYTES));
    assert!(
        !messages.is_empty(),
        "a chain past the stack budget must reject (Err with errors), never \
         check clean / return Ok",
    );
}

/// NEGATIVE (no false positive, no behavior change for normal input): a
/// moderate-depth chain, comfortably within the budget, never trips the guard.
/// The unbound names still produce ordinary `unbound variable` errors, but NOT
/// the stack-depth diagnostic. Run on the production grown segment (no
/// override) so this also confirms the grow path itself never spuriously trips
/// the guard on shallow input.
#[test]
fn moderate_depth_chain_does_not_trip_stack_guard() {
    // Depth 20 is far below any stack budget bail and below even the smallest
    // (2 MiB) native overflow depth, so it is safe on any stack.
    let messages = check_deep_on_bounded_stack(program_with_body(deep_app_chain(20)), 8, None);
    assert!(
        !messages
            .iter()
            .any(|m| m.contains("stack budget exhausted")),
        "a 20-deep chain is well within the stack budget and must not trip the \
         guard; got: {messages:?}",
    );
}

/// FOLLOW-UP LANDED (WI-1 stack-growing): the real pricer -- or a synthetic
/// chain at the pricer's true nesting depth -- now `check`s cleanly end-to-end.
/// A chain deep enough to model the pricer (4000) used to trip the stack-budget
/// guard on an 8 MiB stack; the public check entries now run the WHOLE pipeline
/// on a freshly-grown stack segment (`with_grown_stack` -> `stacker::grow` in
/// `chelis-types/src/infer.rs`), so inference AND every validate / annotate
/// pass (plus the deep `Expr` clones and the final drop) run with room and the
/// per-site guard never fires for a legitimately deep-but-finite program. The
/// per-site guard remains the safety net for input deeper than the grown
/// segment (covered-or-rejected; exercised by
/// `deep_app_chain_yields_stack_budget_diagnostic_not_sigsegv` with a shrunk
/// segment). The depth is finite and reef-linking-induced, not an infinite
/// recursion -- the reef linker concatenates each module's decls under mangled
/// internal names and references each export once rather than re-inlining
/// bodies, so it adds breadth, not unbounded depth, and `infer_app` recurses
/// only on strictly-smaller subtrees of a finite AST (see
/// docs/investigations/wi1_infer_recursion_depth.md).
///
/// This is the negative of `deep_app_chain_yields_stack_budget_diagnostic_not_sigsegv`:
/// that test SHRINKS the grown segment so depth 4000 overflows it and asserts
/// the guard converts the overflow into a located diagnostic; this one runs the
/// PRODUCTION grown segment and asserts the same depth instead checks WITHOUT a
/// stack-depth error. Both prove there is no SIGSEGV; they differ only in the
/// grown-segment size, which is the whole point of the follow-up.
#[test]
fn pricer_depth_chain_checks_without_depth_error_once_followup_lands() {
    // 4000 stands in for the pricer's true (deep) nesting; the real pricer
    // fixture is reachable through the CLI, not this crate's unit surface.
    // `None` -> the production grown segment.
    let messages = check_deep_on_bounded_stack(program_with_body(deep_app_chain(4000)), 8, None);
    assert!(
        !messages
            .iter()
            .any(|m| m.contains("stack budget exhausted")),
        "the WI-1 stack-growing follow-up must let a pricer-depth chain \
         check without a stack-depth error; got: {messages:?}",
    );
}
