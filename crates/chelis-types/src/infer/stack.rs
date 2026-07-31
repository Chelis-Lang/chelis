//! Native-stack budget and recursion depth protection for the recursive checker.
//!
//! Extracted verbatim from the former `infer.rs` monolith by the openspec
//! `modularize-type-inference` change. Logic, statement order, and
//! diagnostic order are unchanged.

use super::*;

/// Stack red-zone (in bytes) the checker keeps in reserve. Every
/// recursive AST walker in the `infer_program` pipeline checks, before
/// recursing, that at least this much native stack remains; if less
/// remains it bails with a typed diagnostic instead of recursing into a
/// native stack overflow.
///
/// Why a stack-budget guard at all: `infer_expr`/`infer_app` (and the
/// sibling walkers below) mutually recurse one native frame per AST level
/// over deeply-nested `app` trees -- a reef-linked program such as the
/// Shoals pricer desugars into very deep curried-application chains. The
/// recursion is *finite* (a 512 MiB stack completes the real pricer) but
/// `chelis check` runs the checker on the process main thread (8 MiB
/// default on Linux), which overflows partway through. A native stack
/// overflow `abort()`s the process (Rust's overflow handler is not a
/// panic), so it cannot be turned into a diagnostic after the fact -- the
/// only correctness-preserving option is to refuse to recurse *before* the
/// frame that would overflow. (A type checker should not SIGSEGV on input;
/// this guard bounds the checker's *own* inference recursion. It is not a
/// general guarantee: the foundational `chelis_deep::Expr` derived
/// `Clone`/`Drop` and `chelis_deep::validate::validate_expr` still overflow
/// on deeper input, which a per-site guard here cannot reach -- see the
/// WI-1 follow-up note.)
///
/// Why a byte budget and not a depth constant: the depth at which a given
/// chain overflows is build- and stack-profile dependent -- the same
/// `var`-only chain overflows around depth ~27 (debug / 2 MiB) through
/// ~6090 (release / 32 MiB), a >200x spread (measured; see
/// docs/investigations/wi1_infer_recursion_depth.md). A static depth
/// constant is a tuning treadmill: it drifts with per-frame size and
/// assumes a fixed thread stack. `stacker::remaining_stack()` measures the
/// actual resource, so the guard stays correct under any build profile and
/// any thread stack size without tuning -- this is the same mechanism
/// rustc uses for its own recursive passes.
///
/// 128 KiB is a generous red zone: the heaviest single `infer_expr` frame
/// measured is well under that, so reserving 128 KiB guarantees the
/// current frame plus a few more can unwind and allocate the diagnostic
/// without touching the guard page, while costing negligibly little of an
/// 8 MiB stack. (The rustc 1.97 SIGABRT in the small-stack guard tests was
/// NOT a red-zone shortfall: the tests dropped their 4000-deep chain's
/// unguarded drop glue on the 1 MiB worker itself; measured empirically,
/// the guarded walkers still bail comfortably inside 128 KiB.)
pub(super) const STACK_RED_ZONE_BYTES: usize = 128 * 1024;

/// Fallback recursion-depth cap used *only* when
/// `stacker::remaining_stack()` returns `None` (the platform cannot report
/// remaining stack). On such a platform the byte budget is unavailable, so
/// the guard degrades to a conservative static depth cap -- never worse
/// than a pure static-depth guard would have been. Chosen below the
/// smallest measured overflow (debug / 2 MiB overflows around depth ~27,
/// so even a 2 MiB stack survives this cap with margin) and far above any
/// legitimate source nesting (hand-written and decompiler-emitted Surf
/// nests in the low hundreds at most -- but legitimate-deep programs on a
/// `None` platform are the price of having no byte budget there; the
/// primary, non-fallback path imposes no such depth ceiling). The
/// `RecursionDepthGuard` RAII counter exists to feed this fallback.
pub(super) const FALLBACK_MAX_DEPTH: usize = 20;

/// Size of the fresh stack segment `with_grown_stack` allocates for the whole
/// check pipeline. 512 MiB comfortably clears the depth the real reef-linked
/// Shoals pricer reaches (its deepest desugared `app` body nests in the low
/// thousands -- a finite source property: the reef linker concatenates each
/// module's decls under mangled internal names and references every export
/// once rather than re-inlining bodies, so linking adds breadth, not unbounded
/// depth; see docs/investigations/wi1_infer_recursion_depth.md). The
/// investigation measured a 512 MiB thread completing the real pricer, and
/// every recursive pass over the tree (inference, the validate / annotate
/// passes, plus the `deep::Expr` clones and the final drop) runs inside this
/// one segment, so the cliff is lifted uniformly rather than moved to the next
/// pass. `stacker::grow` reserves the segment via `mmap`; on Linux the pages
/// are demand-zeroed, so a shallow check that never descends deep only commits
/// the few pages it actually touches -- the 512 MiB is reserved address space,
/// not resident memory.
pub(super) const GROW_SEGMENT_BYTES: usize = 512 * 1024 * 1024;

thread_local! {
    /// Test-only per-thread override (in bytes) for the `with_grown_stack`
    /// segment size. Reserved for the recursion-depth-guard test corpus, which
    /// must still exercise the per-site `stack_guard!` SAFETY NET: with the
    /// production 512 MiB segment no realistic test depth exhausts the stack,
    /// so the safety-net tests shrink the segment (via `set_grow_segment_bytes_for_test`)
    /// to a few MiB and drive a chain deep enough to overflow it, proving the
    /// guard still converts the overflow into a located diagnostic rather than
    /// a SIGSEGV.
    ///
    /// A thread-local (not a process-global env var) is used deliberately: the
    /// override and the `grow` it governs run on the SAME check thread, so a
    /// concurrent check on another thread -- another test in the same process
    /// under `cargo test`, or a real parallel compile -- can never observe a
    /// shrunk segment. `None` means "use the production `GROW_SEGMENT_BYTES`".
    static GROW_SEGMENT_OVERRIDE: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Install a test-only per-thread `with_grown_stack` segment size. Call from a
/// test worker thread BEFORE invoking the checker on that thread. Not for
/// production use; a too-small value makes a legitimately-deep program bail
/// where it would otherwise check.
#[doc(hidden)]
pub fn set_grow_segment_bytes_for_test(bytes: usize) {
    GROW_SEGMENT_OVERRIDE.with(|cell| cell.set(Some(bytes)));
}

pub(super) fn grow_segment_bytes() -> usize {
    GROW_SEGMENT_OVERRIDE
        .with(Cell::get)
        .filter(|&n| n > 0)
        .unwrap_or(GROW_SEGMENT_BYTES)
}

/// Run the whole check pipeline `f` on a freshly-allocated stack segment
/// (`GROW_SEGMENT_BYTES`, or a `CHELIS_GROW_SEGMENT_BYTES` test override).
/// Wrap every PUBLIC check entry in this: a single grow at the boundary covers
/// EVERY recursive pass that descends the `deep::Expr` tree (inference, the
/// validate / annotate walkers, the deep `Expr` clones the pipeline makes, and
/// the implicit drop of the result), because they all run inside this one
/// closure on the grown segment. This is the WI-1 follow-up that lets a
/// legitimately deep-but-finite program (a reef-linked pricer) check
/// end-to-end instead of tripping a per-site `stack_guard!` partway through one
/// of those passes.
///
/// We use `stacker::grow` (always allocate a fresh segment) rather than
/// `stacker::maybe_grow` (allocate only when near exhaustion) on purpose: a
/// check typically starts on a fresh, near-empty thread stack, so `maybe_grow`
/// would see ample headroom and never grow -- the stack only nears exhaustion
/// many frames INTO the descent, by which point we are deep inside the
/// recursive walkers, far from this boundary, with no further grow site. One
/// of those walkers' per-site `stack_guard!`s would then fire first. Allocating
/// the big segment up front guarantees the headroom is in place before the
/// descent begins. The per-site guards remain the safety net for input deeper
/// than even this segment can hold (covered-or-rejected).
pub(super) fn with_grown_stack<R>(f: impl FnOnce() -> R) -> R {
    stacker::grow(grow_segment_bytes(), f)
}

/// Public wrapper over [`with_grown_stack`] for callers OUTSIDE this crate
/// (the CLI `chelis check` entry) that drive deep `chelis_deep::Expr` work the
/// chelis-types check entries do not themselves wrap: the reef/deep loader, the
/// `prepared.decls.clone()` of the linked program, the desugarer, the fitness
/// structure walk, `chelis_deep::validate`, and the final drop of the deep
/// tree. Wrapping the whole CLI check operation in this runs ALL of that on one
/// grown segment, so a deeply-nested but finite reef-linked program (the Shoals
/// pricer) cannot SIGSEGV in a derived `Clone`/`Drop`/validator outside the
/// type checker. The per-entry grows inside this crate still apply to their own
/// recursion when called directly (eval, build, lsp), so nesting this around
/// them is at worst a redundant -- not incorrect -- second grow.
pub fn run_on_grown_stack<R>(f: impl FnOnce() -> R) -> R {
    with_grown_stack(f)
}

thread_local! {
    /// Current pipeline native-recursion depth on this thread, maintained
    /// by `RecursionDepthGuard`. Only consulted on platforms where
    /// `stacker::remaining_stack()` returns `None`; on the normal path the
    /// byte budget governs and this counter is merely incremented and
    /// decremented. Thread-local so concurrent checks on different threads
    /// (e.g. nextest workers) do not share or corrupt the counter.
    static RECURSION_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// RAII recursion-depth counter. Increments `RECURSION_DEPTH` on
/// construction and decrements it on drop, so the count tracks live native
/// recursion depth even across the early-return paths inside the guarded
/// walkers. Used by `stack_guard_tripped` as the `None`-platform
/// fallback. Construct one at the top of every guarded recursive walker.
pub(super) struct RecursionDepthGuard;

impl RecursionDepthGuard {
    pub(super) fn enter() -> Self {
        RECURSION_DEPTH.with(|cell| cell.set(cell.get() + 1));
        RecursionDepthGuard
    }

    pub(super) fn depth() -> usize {
        RECURSION_DEPTH.with(Cell::get)
    }
}

impl Drop for RecursionDepthGuard {
    fn drop(&mut self) {
        RECURSION_DEPTH.with(|cell| cell.set(cell.get().saturating_sub(1)));
    }
}

thread_local! {
    /// First stack-exhaustion bail recorded during the current check unit,
    /// as `(walker_site, optional_span_id)`. Set (first-write-wins) by
    /// `stack_guard_tripped` whenever any guarded recursive walker bails on
    /// low stack; reset to `None` when the OUTERMOST `StackExhaustionScope`
    /// is entered and drained into the check's error vector before its
    /// empty-errors gate. This is the soundness funnel: a stack bail that
    /// merely stopped a no-error walker recursing (so it could not push its
    /// own diagnostic) must STILL turn the whole check into a hard, located
    /// failure rather than a silent green or partial result
    /// (covered-or-rejected: exhaustion -> rejected, never swallowed).
    /// Thread-local so concurrent checks on different threads do not
    /// cross-contaminate.
    pub(super) static STACK_EXHAUSTED: RefCell<Option<(String, Option<String>)>> = const { RefCell::new(None) };

    /// Re-entrancy depth of `StackExhaustionScope` on this thread. The flag
    /// is reset only when this transitions 0 -> 1 (the outermost scope), so
    /// an inner public entry (e.g. `check_typed_program` -> `infer_program`)
    /// does not wipe a bail the outer pass already recorded.
    pub(super) static STACK_SCOPE_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Marks a check unit for stack-exhaustion tracking. Construct one at the
/// top of every public check entry. Entering the OUTERMOST scope resets the
/// flag (so a prior exhausted run on a pooled thread cannot leak in);
/// nested scopes are no-ops for the reset. Call `drain_into` after the
/// pipeline to surface any recorded bail as a hard located error.
pub(super) struct StackExhaustionScope;

impl StackExhaustionScope {
    pub(super) fn enter() -> Self {
        STACK_SCOPE_DEPTH.with(|d| {
            let depth = d.get();
            if depth == 0 {
                STACK_EXHAUSTED.with(|cell| *cell.borrow_mut() = None);
            }
            d.set(depth + 1);
        });
        StackExhaustionScope
    }

    /// If a stack bail was recorded during this scope, push the located
    /// failure diagnostic into `errors`. Idempotent: takes the flag, so a
    /// second call is a no-op. Call after the whole pipeline has run and
    /// before the empty-errors gate, so exhaustion always surfaces as a hard
    /// check failure.
    pub(super) fn drain_into(&self, errors: &mut impl DiagnosticOutput) {
        if let Some((site, span_id)) = STACK_EXHAUSTED.with(|cell| cell.borrow_mut().take()) {
            errors.push(stack_depth_error(&site, span_id.as_deref()));
        }
    }
}

impl Drop for StackExhaustionScope {
    fn drop(&mut self) {
        STACK_SCOPE_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

/// True when the current native stack is too close to exhaustion to safely
/// recurse one more pipeline frame; when it returns `true` it has also
/// recorded the bail (first-write-wins) into `STACK_EXHAUSTED` so the check
/// entry boundary turns it into a hard located failure even for walkers
/// that carry no error vector and can only `return`.
///
/// Primary path: `true` when `stacker::remaining_stack()` reports fewer than
/// `STACK_RED_ZONE_BYTES` remaining. Fallback path (`remaining_stack()` is
/// `None`, i.e. the platform cannot report remaining stack): `true` when the
/// `RecursionDepthGuard` depth exceeds `FALLBACK_MAX_DEPTH`, so a `None`
/// platform is never worse off than a pure static-depth guard. Callers must
/// already hold a live `RecursionDepthGuard` so the fallback depth reflects
/// this call.
///
/// `span_id` is a closure, not an already-resolved `Option<&str>`, so the
/// span lookup -- a linear scan of the node's metadata map -- runs ONLY on
/// the rare exhausted branch and never on the per-node type-checker hot path.
pub(super) fn stack_guard_tripped<'a>(
    site: &str,
    span_id: impl FnOnce() -> Option<&'a str>,
) -> bool {
    let exceeded = match stacker::remaining_stack() {
        Some(remaining) => remaining < STACK_RED_ZONE_BYTES,
        None => RecursionDepthGuard::depth() > FALLBACK_MAX_DEPTH,
    };
    if exceeded {
        let span_id = span_id();
        STACK_EXHAUSTED.with(|cell| {
            let mut slot = cell.borrow_mut();
            if slot.is_none() {
                *slot = Some((site.to_string(), span_id.map(str::to_string)));
            }
        });
    }
    exceeded
}

/// Stack-recursion guard for one recursive AST walker. Place at the top of
/// every self-recursive walker that descends a `deep::Expr` tree:
///
/// ```ignore
/// stack_guard!("walker_name", expr, /* bail value */ false);
/// ```
///
/// It constructs a live `RecursionDepthGuard` for the current frame (so the
/// `None`-platform depth fallback is accurate), then, if the stack budget is
/// exhausted, records the located bail in `STACK_EXHAUSTED` (so the check
/// entry boundary fails hard -- never a silent partial result) and returns
/// the supplied neutral bail value, stopping the recursion before the frame
/// that would overflow. The bail value is whatever "stop / nothing found"
/// means for the walker's return type (`false`, `None`, `()` via no third
/// argument, an identity `expr.clone()`, etc.). `$expr` must be a
/// `&deep::Expr` (or anything with a `span_id()` method) so the diagnostic
/// can name where the depth is.
macro_rules! stack_guard {
    ($site:expr, $expr:expr, $bail:expr) => {
        let _stack_depth_guard = RecursionDepthGuard::enter();
        if stack_guard_tripped($site, || $expr.span_id()) {
            return $bail;
        }
    };
    ($site:expr, $expr:expr) => {
        let _stack_depth_guard = RecursionDepthGuard::enter();
        if stack_guard_tripped($site, || $expr.span_id()) {
            return;
        }
    };
}

/// Build the typed diagnostic emitted when the check fails because a
/// recursive walker bailed on a nearly-exhausted native stack. The message
/// names the `site` (the walker that bailed first) and, when available, the
/// external-source `span_id` of the construct it bailed at, so the user is
/// told *where* the depth is rather than just that the program is "too deep"
/// -- a valid-but-deeply-nested program can legitimately reach this while
/// inference remains recursive.
pub(super) fn stack_depth_error(site: &str, span_id: Option<&str>) -> CheckError {
    let where_clause = match span_id {
        Some(id) if !id.is_empty() => format!(" near source span `{id}`"),
        _ => String::new(),
    };
    CheckError::new(
        CheckErrorKind::Other,
        format!(
            "type-checker stack budget exhausted while analyzing deeply-nested \
             input in `{site}`{where_clause}: the program nests deeper than the \
             checker can analyze on the available native stack. The recursion is \
             finite (this is a checker limitation, not necessarily a type error); \
             see WI-1 / spec/design/verification_stack_master_plan.md \u{00A7}4.1."
        ),
        vec![
            "Deeply-nested application chains (often from reef-linking many \
             exports into one program) can exhaust the recursive checker's stack \
             budget; reduce the nesting or split the program. An iterative / \
             stack-growing inference follow-up (stacker::maybe_grow) is tracked to \
             lift this for legitimately-deep programs."
                .to_string(),
        ],
    )
}
