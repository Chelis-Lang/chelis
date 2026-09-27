//! Native-stack budget and recursion depth protection for the recursive checker.
//!
//! Extracted from the former `infer.rs` monolith by the openspec
//! `modularize-type-inference` change. Logic, statement order, and
//! diagnostic order are unchanged.
//!
//! All state is explicit. The native probes (remaining stack, fallback depth,
//! stack growth) are primitives on [`NativeServices`]; the first recorded bail
//! and the check-unit scope depth live in a [`StackBailRecord`] owned by the
//! invocation's `DiagnosticSink`. Every guarded recursive walker is a thin
//! wrapper that enters one frame through [`StackBudget::frame_exhausted`],
//! runs its body, and exits the frame explicitly on every return path.

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
pub(crate) const STACK_RED_ZONE_BYTES: usize = 128 * 1024;

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
/// explicit depth kept by `NativeServices::enter_frame`/`exit_frame` exists
/// to feed this fallback.
pub(crate) const FALLBACK_MAX_DEPTH: usize = 20;

/// Size of the fresh stack segment `NativeServices::with_grown_stack` allocates for the whole
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
pub(crate) const GROW_SEGMENT_BYTES: usize = 512 * 1024 * 1024;

/// First stack-exhaustion bail recorded during the current check unit, plus
/// the check-unit scope depth. Owned by the invocation's `DiagnosticSink`.
///
/// The bail is `(walker_site, optional_span_id)`, set first-write-wins by
/// [`StackBudget::frame_exhausted`] whenever any guarded recursive walker
/// bails on low stack; reset when the OUTERMOST check-unit scope is entered
/// and drained into the check's diagnostics before its empty-errors gate.
/// This is the soundness funnel: a stack bail that merely stopped a no-error
/// walker recursing (so it could not push its own diagnostic) must STILL turn
/// the whole check into a hard, located failure rather than a silent green or
/// partial result (covered-or-rejected: exhaustion -> rejected, never
/// swallowed).
#[derive(Debug, Default)]
pub(crate) struct StackBailRecord {
    first: Option<(String, Option<String>)>,
    /// Re-entrancy depth of check-unit scopes. The bail is reset only when
    /// this transitions 0 -> 1 (the outermost scope), so an inner check unit
    /// does not wipe a bail the outer pass already recorded.
    scope_depth: usize,
}

impl StackBailRecord {
    /// Enter a check unit. Entering the OUTERMOST scope resets the bail and
    /// the services' guarded-frame depth (a previous check that unwound out
    /// of guarded walkers left its frames unexited); nested scopes are no-ops
    /// for the reset. Pair with [`Self::exit_scope`] on every return path.
    pub(crate) fn enter_scope(&mut self, native: &mut NativeServices) {
        if self.scope_depth == 0 {
            self.first = None;
            native.reset_frames();
        }
        self.scope_depth += 1;
    }

    /// Leave a check unit entered with [`Self::enter_scope`].
    pub(crate) fn exit_scope(&mut self) {
        self.scope_depth = self.scope_depth.saturating_sub(1);
    }

    /// Take the recorded bail, if any, as the located failure diagnostic.
    /// Idempotent: a second call returns `None`.
    pub(crate) fn take_error(&mut self) -> Option<CheckError> {
        self.first
            .take()
            .map(|(site, span_id)| stack_depth_error(&site, span_id.as_deref()))
    }

    /// The recorded bail, for tests that observe the funnel directly.
    #[cfg(test)]
    pub(crate) fn recorded(&self) -> Option<&(String, Option<String>)> {
        self.first.as_ref()
    }

    fn record(&mut self, site: &str, span_id: Option<&str>) {
        if self.first.is_none() {
            self.first = Some((site.to_string(), span_id.map(str::to_string)));
        }
    }
}

/// The explicit state one guarded recursive walker needs: the invocation's
/// native services (stack probe and fallback depth) and the sink-owned bail
/// record. Borrowed from a `DiagnosticSink` via `DiagnosticSink::stack`.
pub(crate) struct StackBudget<'a> {
    native: &'a mut NativeServices,
    record: &'a mut StackBailRecord,
}

impl<'a> StackBudget<'a> {
    pub(crate) fn new(native: &'a mut NativeServices, record: &'a mut StackBailRecord) -> Self {
        Self { native, record }
    }

    /// A budget over the same native services and bail record, borrowed from
    /// this one, that a callback can own instead of capturing `&mut self`.
    pub(crate) fn reborrow(&mut self) -> StackBudget<'_> {
        StackBudget {
            native: self.native,
            record: self.record,
        }
    }

    /// Enter one guarded frame and probe the stack budget.
    ///
    /// Returns `false` when the walker may recurse; it must then call
    /// [`Self::exit_frame`] on every return path after its body. Returns
    /// `true` (with the frame already exited) when the native stack is too
    /// close to exhaustion: the bail has been recorded first-write-wins, and
    /// the walker must return its neutral bail value without recursing.
    ///
    /// `span_id` is a closure, not an already-resolved `Option<&str>`, so the
    /// span lookup -- a linear scan of the node's metadata map -- runs ONLY on
    /// the rare exhausted branch and never on the per-node hot path.
    pub(crate) fn frame_exhausted<'e>(
        &mut self,
        site: &str,
        span_id: impl FnOnce() -> Option<&'e str>,
    ) -> bool {
        self.native.enter_frame();
        if self.native.stack_exhausted() {
            self.record.record(site, span_id());
            self.native.exit_frame();
            return true;
        }
        false
    }

    /// Leave a frame entered by a non-exhausted [`Self::frame_exhausted`].
    pub(crate) fn exit_frame(&mut self) {
        self.native.exit_frame();
    }

    /// The invocation's native services.
    pub(crate) fn native(&mut self) -> &mut NativeServices {
        self.native
    }
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
