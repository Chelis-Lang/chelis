//! WI-1 (funnel-hole closure): the library-compile check entries
//! `build_compiled_library_context` and `build_compiled_library_context_with_base`
//! must also convert a deep-recursion stack bail into a hard, located
//! rejection -- never a silent green or a partial `CheckedProgram`.
//!
//! Those two public entries run the same guarded recursive walkers as the
//! sibling `check_ir_with_signature_context` / `check_typed_program` /
//! `infer_program` entries (inference, the tensor-precision / polymorphic-op
//! validators, and the annotation pass), but originally never constructed a
//! `StackExhaustionScope` nor drained the thread-local exhaustion flag. A
//! stack-budget bail in those passes set the flag, the empty-errors gate never
//! saw it, and the function returned `Ok` with a partially-analyzed library
//! context (the next check entry then cleared the stale flag). These tests
//! drive a chain far deeper than the stack budget through those entries and
//! assert they reject with the located stack-budget diagnostic.
//!
//! They mirror `infer_recursion_depth_guard.rs`: the deep `app` tree is built
//! directly in Deep (bypassing the Surf parser/desugarer) so the recursion
//! depth is exactly the chain length, and the entries run on an
//! explicitly-sized worker thread so the guard fires deterministically
//! regardless of the harness default stack size.
//!
//! See docs/investigations/wi1_infer_recursion_depth.md and the
//! `STACK_RED_ZONE_BYTES` doc-comment in crates/chelis-types/src/infer.rs.

use chelis_deep::DeepTag;
use chelis_deep::ast::{Atom, Expr, Metadata};
use chelis_deep::span::Span;
use chelis_types::{
    TypeEnv, build_compiled_library_context, build_compiled_library_context_with_base,
};

fn sym(s: &str) -> Expr {
    Expr::Atom(Atom::Name(s.to_string()), Span::new(0, 0))
}

fn node(tag: DeepTag, children: Vec<Expr>) -> Expr {
    Expr::node(tag, Metadata::default(), children, Span::new(0, 0))
}

/// `(var name)`.
fn var(name: &str) -> Expr {
    node(DeepTag::Var, vec![sym(name)])
}

/// `(app func arg)`.
fn app(func: Expr, arg: Expr) -> Expr {
    node(DeepTag::App, vec![func, arg])
}

/// A left-nested curried application of `depth` distinct names:
/// `(app (app (app f0 f1) f2) ... fN)`. Each `app`'s function position is
/// itself an `app`, so analyzing the outer node drives the walker recursion
/// exactly `depth` deep.
fn deep_app_chain(depth: usize) -> Expr {
    let mut e = var("f0");
    for i in 1..=depth {
        e = app(e, var(&format!("f{i}")));
    }
    e
}

/// Wrap an expression as the body of a top-level `def lib_main` so the full
/// library-compile pipeline (inference, precision/invariant validation,
/// annotation) runs over it.
fn library_with_body(body: Expr) -> Vec<Expr> {
    let params = node(DeepTag::Params, Vec::new());
    let func = node(DeepTag::Fn, vec![params, body]);
    let def = node(DeepTag::Def, vec![sym("lib_main"), func]);
    vec![def]
}

/// The grown-segment size the library entries check on. Both entries run on
/// a freshly grown segment (`run_on_grown_stack`), and the production 512 MiB
/// segment checks a 4000-deep chain to completion without reaching the
/// guard. Shrinking it to 8 MiB, the size `infer_recursion_depth_guard.rs`
/// pins and the default `chelis check` main-thread stack on Linux, makes the
/// 4000-deep chain exhaust the byte budget so the per-site guard is what
/// stops it.
const SAFETY_NET_SEGMENT_BYTES: usize = 8 * 1024 * 1024;

/// Run `check` on a worker thread with an explicit stack size and the grown
/// segment pinned at `SAFETY_NET_SEGMENT_BYTES`. Returns the error messages
/// on `Err`, or an empty vector on `Ok` -- so a test can distinguish
/// "rejected with diagnostics" from "silently passed". Reaching the join at
/// all proves the guard (not a SIGSEGV) bounded the recursion; an abort would
/// take down the whole test process.
fn on_bounded_stack(
    thread_name: &str,
    library: Vec<Expr>,
    stack_mib: usize,
    check: fn(&[Expr]) -> Vec<String>,
) -> Vec<String> {
    std::thread::Builder::new()
        .name(thread_name.to_string())
        .stack_size(stack_mib * 1024 * 1024)
        .spawn(move || {
            chelis_types::set_grow_segment_bytes_for_test(SAFETY_NET_SEGMENT_BYTES);
            let messages = check(&library);

            // The entries borrow their input, so the worker drops the deep
            // fixture after the guarded check returns. Dropping a nested
            // `Expr` is itself recursive; run it on a production-sized
            // segment so teardown cannot overflow the bounded worker.
            chelis_types::reset_grow_segment_bytes_for_test();
            chelis_types::run_on_grown_stack(|| drop(library));
            messages
        })
        .expect("spawn library-context worker thread")
        .join()
        .expect("worker thread aborted (stack overflow?) instead of returning")
}

/// Run `build_compiled_library_context` on the bounded worker.
fn library_context_on_bounded_stack(library: Vec<Expr>, stack_mib: usize) -> Vec<String> {
    on_bounded_stack(
        "infer-library-depth-guard-test",
        library,
        stack_mib,
        |library| match build_compiled_library_context(library) {
            Ok(_) => Vec::new(),
            Err(result) => result.errors.iter().map(|e| e.message.clone()).collect(),
        },
    )
}

/// Same, for the layered `_with_base` entry, stacked on the empty base.
fn library_context_with_base_on_bounded_stack(library: Vec<Expr>, stack_mib: usize) -> Vec<String> {
    on_bounded_stack(
        "infer-library-base-depth-guard-test",
        library,
        stack_mib,
        |library| {
            let base = TypeEnv::empty();
            match build_compiled_library_context_with_base(&base, library) {
                Ok(_) => Vec::new(),
                Err(result) => result.errors.iter().map(|e| e.message.clone()).collect(),
            }
        },
    )
}

/// POSITIVE (covered-or-rejected, never silent): a chain past the stack budget
/// through `build_compiled_library_context` must return `Err` carrying the
/// located stack-budget diagnostic, never `Ok` with a partial library context.
/// This is the funnel-hole closure: the bail can never be swallowed into a
/// green library compile.
#[test]
fn deep_library_chain_rejects_with_stack_diagnostic_not_ok() {
    let library = library_with_body(deep_app_chain(4000));
    let messages = library_context_on_bounded_stack(library, 8);
    assert!(
        !messages.is_empty(),
        "a library chain past the stack budget must reject (Err with errors), \
         never return Ok with a partial CheckedProgram",
    );
    assert!(
        messages
            .iter()
            .any(|m| m.contains("stack budget exhausted")),
        "expected a located 'stack budget exhausted' diagnostic from \
         build_compiled_library_context for a 4000-deep chain; got: {messages:?}",
    );
}

/// POSITIVE (located): the library-path diagnostic also names the walker site.
#[test]
fn deep_library_chain_diagnostic_names_the_walker_site() {
    let library = library_with_body(deep_app_chain(4000));
    let messages = library_context_on_bounded_stack(library, 8);
    assert!(
        messages
            .iter()
            .any(|m| m.contains("stack budget exhausted") && m.contains("in `")),
        "the library depth diagnostic must name the walker site (`in \\`...\\``); \
         got: {messages:?}",
    );
}

/// POSITIVE: the layered `_with_base` entry closes the same funnel hole.
#[test]
fn deep_library_chain_with_base_rejects_with_stack_diagnostic_not_ok() {
    let library = library_with_body(deep_app_chain(4000));
    let messages = library_context_with_base_on_bounded_stack(library, 8);
    assert!(
        !messages.is_empty(),
        "a layered library chain past the stack budget must reject, never \
         return Ok with a partial context",
    );
    assert!(
        messages
            .iter()
            .any(|m| m.contains("stack budget exhausted")),
        "expected a located 'stack budget exhausted' diagnostic from \
         build_compiled_library_context_with_base; got: {messages:?}",
    );
}

/// NEGATIVE (no false positive): a moderate-depth library, comfortably within
/// the budget, never trips the guard. The unbound names still produce ordinary
/// `unbound variable` errors, but NOT the stack-depth diagnostic.
#[test]
fn moderate_depth_library_chain_does_not_trip_stack_guard() {
    let library = library_with_body(deep_app_chain(20));
    let messages = library_context_on_bounded_stack(library, 8);
    assert!(
        !messages
            .iter()
            .any(|m| m.contains("stack budget exhausted")),
        "a 20-deep library chain is well within the stack budget and must not \
         trip the guard; got: {messages:?}",
    );
}
