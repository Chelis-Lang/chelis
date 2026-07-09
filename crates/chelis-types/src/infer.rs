//! Type inference engine for the Chelis type checker.
//!
//! Walks Deep AST nodes and assigns types using Hindley-Milner inference.

use std::collections::{BTreeMap, HashMap, HashSet};

use chelis_deep::Span;
use chelis_deep::ast as deep;

use crate::adt::{AdtRegistry, CallShape};
use crate::builtins;
use crate::context::{TypeEnv, TypeEnvInner};
use crate::env::Env;
use crate::errors::*;
use crate::linearity::LinearityInfo;
use crate::types::*;
use crate::unify::*;

use std::cell::{Cell, RefCell};

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
const STACK_RED_ZONE_BYTES: usize = 128 * 1024;

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
const FALLBACK_MAX_DEPTH: usize = 20;

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
const GROW_SEGMENT_BYTES: usize = 512 * 1024 * 1024;

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

fn grow_segment_bytes() -> usize {
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
fn with_grown_stack<R>(f: impl FnOnce() -> R) -> R {
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
struct RecursionDepthGuard;

impl RecursionDepthGuard {
    fn enter() -> Self {
        RECURSION_DEPTH.with(|cell| cell.set(cell.get() + 1));
        RecursionDepthGuard
    }

    fn depth() -> usize {
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
    static STACK_EXHAUSTED: RefCell<Option<(String, Option<String>)>> = const { RefCell::new(None) };

    /// Re-entrancy depth of `StackExhaustionScope` on this thread. The flag
    /// is reset only when this transitions 0 -> 1 (the outermost scope), so
    /// an inner public entry (e.g. `check_typed_program` -> `infer_program`)
    /// does not wipe a bail the outer pass already recorded.
    static STACK_SCOPE_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Marks a check unit for stack-exhaustion tracking. Construct one at the
/// top of every public check entry. Entering the OUTERMOST scope resets the
/// flag (so a prior exhausted run on a pooled thread cannot leak in);
/// nested scopes are no-ops for the reset. Call `drain_into` after the
/// pipeline to surface any recorded bail as a hard located error.
struct StackExhaustionScope;

impl StackExhaustionScope {
    fn enter() -> Self {
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
    fn drain_into(&self, errors: &mut Vec<CheckError>) {
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
fn stack_guard_tripped<'a>(site: &str, span_id: impl FnOnce() -> Option<&'a str>) -> bool {
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
fn stack_depth_error(site: &str, span_id: Option<&str>) -> CheckError {
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

thread_local! {
    /// Per-annotation-pass map from a def name to its declared
    /// parameter type *expressions*, taken verbatim from the matching
    /// `(defsig name (t-fn arg-exprs... ret))` node.
    ///
    /// `annotate_fn_children` consults this when stamping a def's
    /// `(params ...)` node so a parameter whose type comes from a
    /// separate `sig` declaration gets the declared type -- preserving
    /// `&` borrow wrappers -- written where IR lowering reads it.
    /// Defs with no `defsig` are absent from the map and keep bare
    /// params, leaving read-only/borrow inference to
    /// `infer_signature_metadata`.
    ///
    /// Populated for the duration of `annotate_ir_program` /
    /// `annotate_ir_program_with_context` and cleared afterwards.
    static DECLARED_SIG_PARAM_TYPES: RefCell<HashMap<String, Vec<deep::Expr>>> =
        RefCell::new(HashMap::new());
}

/// Scan `exprs` for `(defsig name (t-fn ...))` nodes and install a
/// name -> declared-param-type-exprs map into `DECLARED_SIG_PARAM_TYPES`
/// for the duration of the returned guard. Restores the previous map
/// (typically empty) on drop so nested / re-entrant annotation passes
/// do not leak state.
fn install_declared_sig_param_types(exprs: &[deep::Expr]) -> DeclaredSigGuard {
    let mut map: HashMap<String, Vec<deep::Expr>> = HashMap::new();
    for expr in exprs {
        collect_defsig_param_types(expr, &mut map);
    }
    let previous =
        DECLARED_SIG_PARAM_TYPES.with(|cell| std::mem::replace(&mut *cell.borrow_mut(), map));
    DeclaredSigGuard { previous }
}

struct DeclaredSigGuard {
    previous: HashMap<String, Vec<deep::Expr>>,
}

impl Drop for DeclaredSigGuard {
    fn drop(&mut self) {
        let restored = std::mem::take(&mut self.previous);
        DECLARED_SIG_PARAM_TYPES.with(|cell| *cell.borrow_mut() = restored);
    }
}

/// Recursively collect `(defsig name (t-fn arg-exprs... ret))` entries,
/// descending through `(module ...)` wrappers. Only the leading
/// argument type expressions are stored (the trailing return type is
/// dropped). A re-declared name keeps the first sig seen.
fn collect_defsig_param_types(expr: &deep::Expr, map: &mut HashMap<String, Vec<deep::Expr>>) {
    // Bail before unbounded recursion exhausts the native stack on a
    // deeply-nested input. No error vector here; `stack_guard_tripped`
    // records the bail so the check entry boundary fails hard with a located
    // diagnostic. See `STACK_RED_ZONE_BYTES`. (In practice this walker only
    // descends `module` wrappers, which do not nest deeply, but the guard
    // keeps the "every recursive walker is bounded" invariant uniform and
    // cheap.)
    stack_guard!("collect_defsig_param_types", expr);
    let deep::Expr::List(list, _) = expr else {
        return;
    };
    match get_tag(list) {
        Some("module") => {
            for child in children(list) {
                collect_defsig_param_types(child, map);
            }
        }
        Some("defsig") => {
            let kids = children(list);
            let Some(name) = kids.first().and_then(symbol_name) else {
                return;
            };
            let Some(deep::Expr::List(fn_list, _)) = kids.get(1) else {
                return;
            };
            if get_tag(fn_list) != Some("t-fn") {
                return;
            }
            let fn_kids = children(fn_list);
            if fn_kids.len() < 2 {
                return;
            }
            // All but the trailing return type are parameter types.
            let param_type_exprs: Vec<deep::Expr> = fn_kids[..fn_kids.len() - 1].to_vec();
            map.entry(name.to_string()).or_insert(param_type_exprs);
        }
        _ => {}
    }
}

/// Result of running type inference on a program.
#[derive(Debug)]
pub struct InferResult {
    pub errors: Vec<CheckError>,
    pub typed_nodes: usize,
    pub total_nodes: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckedProgram {
    annotated_exprs: Vec<deep::Expr>,
    type_env: HashMap<String, deep::Expr>,
    linearity: LinearityInfo,
    signature_inference: SignatureInferenceMetadata,
}

impl CheckedProgram {
    pub fn from_parts(
        annotated_exprs: Vec<deep::Expr>,
        type_env: HashMap<String, deep::Expr>,
    ) -> Self {
        let signature_inference = infer_signature_metadata(&annotated_exprs, &type_env);
        Self {
            annotated_exprs,
            type_env,
            linearity: LinearityInfo::default(),
            signature_inference,
        }
    }

    pub fn from_parts_with_signature_context(
        annotated_exprs: Vec<deep::Expr>,
        type_env: HashMap<String, deep::Expr>,
        signature_context: &SignatureInferenceMetadata,
    ) -> Self {
        let signature_inference =
            infer_signature_metadata_with_context(&annotated_exprs, &type_env, signature_context);
        Self {
            annotated_exprs,
            type_env,
            linearity: LinearityInfo::default(),
            signature_inference,
        }
    }

    pub fn exprs(&self) -> &[deep::Expr] {
        &self.annotated_exprs
    }

    pub fn annotated_exprs(&self) -> &[deep::Expr] {
        &self.annotated_exprs
    }

    pub fn type_env(&self) -> &HashMap<String, deep::Expr> {
        &self.type_env
    }

    pub fn linearity(&self) -> &LinearityInfo {
        &self.linearity
    }

    pub fn signature_inference(&self) -> &SignatureInferenceMetadata {
        &self.signature_inference
    }

    pub fn with_linearity(mut self, linearity: LinearityInfo) -> Self {
        self.linearity = linearity;
        self
    }

    /// Compose a `library` checked program with a `new_code` checked
    /// program into a single whole-program `CheckedProgram`, equivalent
    /// to what `check_ir_program(library_exprs ++ new_code_exprs)` plus
    /// effects + linearity would produce — provided `new_code` was
    /// produced by the `_with_context` variants stacked on `library`.
    ///
    /// This is the seam the cross-process chelis-std typecheck cache's
    /// `chelis build` path uses: `library` is the cached chelis-std
    /// sub-context's `library_checked` and `new_code` is the
    /// `_with_context`-checked non-chelis-std decls + entry. The result
    /// is the one monolithic `CheckedProgram` the `build` lowering
    /// pipeline consumes, without re-inferring chelis-std.
    ///
    /// Composition rule (mirrors the monolithic `library ++ new` shape):
    /// - `annotated_exprs`: `library` exprs followed by `new_code` exprs,
    ///   in that order. Monolithic `check_ir_program` annotates in
    ///   source order, and the linked program places library decls
    ///   before the entry, so this ordering matches.
    /// - `type_env`: union, `new_code` winning on shadow. `new_code`'s
    ///   `type_env` is already unioned with the library's by the
    ///   `_with_context` builder, so this just back-fills any
    ///   library-only entries.
    /// - `linearity`: the two `reusable_inputs_by_offset` maps merged.
    /// - `signature_inference`: the two `functions` maps merged,
    ///   `new_code` winning on a name clash.
    ///
    /// The monolithic-vs-layered acceptance oracle is what proves this
    /// composition is byte-identical to the monolithic path; a
    /// divergence is a compiler-correctness bug, not a tuning knob.
    pub fn compose(library: &CheckedProgram, new_code: &CheckedProgram) -> Self {
        let mut annotated_exprs =
            Vec::with_capacity(library.annotated_exprs.len() + new_code.annotated_exprs.len());
        annotated_exprs.extend(library.annotated_exprs.iter().cloned());
        annotated_exprs.extend(new_code.annotated_exprs.iter().cloned());

        let mut type_env = new_code.type_env.clone();
        for (name, ty) in &library.type_env {
            type_env.entry(name.clone()).or_insert_with(|| ty.clone());
        }

        let linearity = library.linearity.merged_with(&new_code.linearity);

        let mut signature_inference = library.signature_inference.clone();
        for (name, sig) in &new_code.signature_inference.functions {
            signature_inference
                .functions
                .insert(name.clone(), sig.clone());
        }

        Self {
            annotated_exprs,
            type_env,
            linearity,
            signature_inference,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SignatureInferenceMetadata {
    pub functions: BTreeMap<String, FunctionSignatureInference>,
}

impl SignatureInferenceMetadata {
    pub fn is_empty(&self) -> bool {
        self.functions.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FunctionSignatureInference {
    pub name: String,
    pub recursive_cycle: bool,
    pub checked_signature: Type,
    pub display_signature: Type,
    pub params: Vec<ParamSignatureInference>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ParamSignatureInference {
    pub index: usize,
    pub name: String,
    pub written: bool,
    pub inferred_read_only: bool,
    pub checked_type: Type,
    pub display_type: Type,
}

/// Run type inference on a list of top-level Deep expressions.
pub fn infer_program(exprs: &[deep::Expr]) -> InferResult {
    // WI-1 follow-up: run the whole pipeline on a grown stack so a deeply
    // nested but finite program checks end-to-end instead of tripping a
    // per-site `stack_guard!` partway through one of the recursive passes.
    with_grown_stack(|| infer_program_inner(exprs))
}

fn infer_program_inner(exprs: &[deep::Expr]) -> InferResult {
    // Reset the stack-exhaustion flag for this check unit; `drain` below
    // turns any walker stack bail into a hard located error so deep input
    // can never produce a silent green / partial result.
    let stack_scope = StackExhaustionScope::enter();
    let (mut env, mut vg) = builtins::builtin_env();
    let mut subst = Subst::new();
    let mut adt_reg = AdtRegistry::new();
    builtins::register_prelude_adts(&mut env, &mut vg, &mut adt_reg);
    let mut errors = Vec::new();
    let mut typed_nodes = 0;
    let mut total_nodes = 0;

    // RFC v4b (RT-1 F2): reject a named module opened by more than one
    // wrapper in this check unit (module-identity forgery).
    detect_module_reopens(exprs, &mut errors);
    // RFC v5 (RT-1 F2 bypass): reject the reef linker's reserved
    // internal-name format in programs not produced by the linker.
    detect_forged_linker_names(exprs, &mut errors);

    // First pass: collect deftype and defsig declarations. Descend through
    // `(module {} name ...)` wrappers so declarations in every idiomatic
    // Surf source (every .ch starts with `module X`) get collected.
    let items = top_level_decl_items_with_modules(exprs);
    collect_all_declarations(
        &items,
        &mut env,
        &mut vg,
        &mut subst,
        &mut adt_reg,
        &mut errors,
    );

    // Checker-enforced opacity (RFC D-CHECK): install the per-run
    // context so the inference hooks see module identity, exports,
    // and producer text. Dropped at the end of this function.
    let opacity_meta = build_opacity_meta(&items, &adt_reg, &mut vg);
    let _opacity_guard = crate::opacity::install_opacity_context(
        crate::opacity::OpacityContextData::from_meta(opacity_meta),
    );

    // Second pass: infer def bodies. Same module-descent rationale as the
    // declaration pass — without it, the entire HM checker is a no-op on
    // module-wrapped programs.
    let user_def_names = collect_user_def_names(&top_level_decl_items(exprs));
    for (module, expr) in &items {
        let decl_name = top_level_decl_name(expr);
        crate::opacity::set_current_item(
            crate::opacity::module_key_for_item(module.as_deref(), decl_name),
            decl_name.map(str::to_string),
        );
        infer_top_level(
            expr,
            &mut env,
            &mut vg,
            &mut subst,
            &adt_reg,
            &mut errors,
            &mut typed_nodes,
            &mut total_nodes,
            &user_def_names,
        );
        // Issue #256 round 2: re-check each deferred borrow against the
        // now-complete substitution (see `validate_deferred_borrow_vars`).
        validate_deferred_borrow_vars(&subst, &adt_reg, &mut errors);
        // D-CHECK: drain the per-def deferred-access ledger (see
        // `validate_deferred_opaque_uses`).
        validate_deferred_opaque_uses(&subst, &adt_reg, &mut errors);
    }
    crate::opacity::set_current_item(None, None);

    // Third pass: reject tensor types whose element precision isn't supported
    // by the Phase 0f backend (f16/bf16/f8e4m3). These would silently get
    // downcast to f32 by the current build targets, violating the "no implicit
    // precision promotion" rule. f64 is supported as of v0.2.3.
    validate_tensor_precisions_in_program(exprs, &mut errors);
    crate::invariants::validate_type_invariants_in_program(exprs, &mut errors);

    // WS-A8 cross-row enforcement: reject `matmul`/transcendental ops that
    // are reached through a polymorphic-precision sig instantiated at a
    // dtype the spec rules forbid (§5.7.2 / §5.4).
    let local_ir_env = build_ir_type_env(exprs);
    validate_polymorphic_op_constraints(exprs, &local_ir_env, &mut errors);

    // If any walker bailed on a nearly-exhausted stack during this run,
    // surface it as a hard located failure (covered-or-rejected).
    stack_scope.drain_into(&mut errors);

    InferResult {
        errors,
        typed_nodes,
        total_nodes,
    }
}

pub fn check_ir_program(exprs: &[deep::Expr]) -> Result<CheckedProgram, InferResult> {
    // Compose: empty outer scope, then check exprs as new code against it.
    // This keeps a single source of truth for the IR check pipeline.
    check_ir_with_context(&TypeEnv::empty(), exprs)
}

/// Build a stacked outer-scope context from a library decl list. The
/// library is run through the full IR pipeline; if any errors are
/// found they are returned to the caller (the context cannot be built
/// from an unchecked library).
///
/// Once built, the returned [`TypeEnv`] can be re-used to type-check
/// many separate "new code" snippets via
/// [`check_ir_with_context`]. The library state is `Arc`-shared and
/// never mutated, so concurrent reads are cheap.
pub fn build_type_env_from_library(library_exprs: &[deep::Expr]) -> Result<TypeEnv, InferResult> {
    // WI-1 follow-up: grow the stack for the whole library pipeline.
    with_grown_stack(|| build_type_env_from_library_inner(library_exprs))
}

fn build_type_env_from_library_inner(library_exprs: &[deep::Expr]) -> Result<TypeEnv, InferResult> {
    // Reset the stack-exhaustion flag for this check unit; drained below
    // before the empty-errors gate (covered-or-rejected on deep input).
    let stack_scope = StackExhaustionScope::enter();
    let detail_profile = std::env::var_os("CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL")
        .map(|v| v == "1")
        .unwrap_or(false);
    let mut sub_t = std::time::Instant::now();
    let log_sub = |label: &str, t: &mut std::time::Instant| {
        if detail_profile {
            eprintln!(
                "build_type_env_sub: {:>8.4}s {}",
                t.elapsed().as_secs_f64(),
                label
            );
            *t = std::time::Instant::now();
        }
    };
    // Start from the empty (builtins + prelude ADTs) state.
    let empty = TypeEnv::empty();
    let mut state = empty.inner().clone();

    // Library IR declared-type lookup.
    let library_ir = build_ir_type_env(library_exprs);
    log_sub("build_ir_type_env_initial", &mut sub_t);

    let mut result = infer_ir_program_with_state(
        library_exprs,
        &library_ir,
        &mut state,
        /* combined_ir_for_validate = */ &library_ir,
        /* run_validate_passes_on = */ None,
    );
    log_sub("infer_ir_program_with_state", &mut sub_t);
    validate_ir_program(library_exprs, &library_ir, &mut result.errors);
    log_sub("validate_ir_program", &mut sub_t);
    validate_tensor_precisions_in_program(library_exprs, &mut result.errors);
    crate::invariants::validate_type_invariants_in_program(library_exprs, &mut result.errors);
    log_sub("validate_tensor_precisions", &mut sub_t);
    validate_polymorphic_op_constraints(library_exprs, &library_ir, &mut result.errors);
    log_sub("validate_polymorphic_op_constraints", &mut sub_t);
    suppress_unbound_for_cycle_members(library_exprs, &mut result.errors);
    log_sub("suppress_unbound_for_cycle", &mut sub_t);
    // Surface a stack-exhaustion bail from the passes above as a hard
    // located error before the gate (and before the errors drain below).
    stack_scope.drain_into(&mut result.errors);
    if !result.errors.is_empty() {
        return Err(result);
    }

    // Capture library def names — needed by new-code cycle / unbound
    // suppression to distinguish library refs from new-code refs.
    let mut library_def_names = std::collections::HashSet::new();
    for expr in top_level_decl_items(library_exprs) {
        if let deep::Expr::List(list, _) = expr
            && get_tag(list) == Some("def")
            && let Some(name) = children(list).first().and_then(symbol_name)
        {
            library_def_names.insert(name.to_string());
        }
    }
    log_sub("collect_library_def_names", &mut sub_t);

    // Drain accumulated errors back into the state's storage; they were
    // empty above so this is a no-op, but the call site is symmetric
    // with check_ir_with_context.
    let _ = result.errors.drain(..);

    // Build a richer `ir_types` by annotating library exprs against
    // the now-populated state and re-extracting type metadata. The raw
    // `library_ir` (built from un-annotated source) only catches defs
    // with explicit type annotations; for downstream callers that read
    // `CheckedProgram::type_env()` to resolve cross-context name refs
    // (Phase D effects, Phase E linearity, Phase F lower) we need every
    // library def's inferred function type, not just the explicitly-typed
    // ones. Mirrors the monolithic `check_ir_program` flow which
    // calls `annotate_ir_program` then `build_ir_type_env` on
    // the annotated result.
    //
    // Install the declared-`defsig` parameter type map so library defs
    // with separate `sig` declarations get borrow-correct
    // `(params ...)` stamps, consistent with every other annotation
    // entry point.
    let _declared_sig_guard = install_declared_sig_param_types(library_exprs);
    let library_annotated: Vec<deep::Expr> = library_exprs
        .iter()
        .map(|e| {
            annotate_expr_with_scope(e, &state.env, &state.var_gen, &state.subst, &state.adt_reg)
        })
        .collect();
    log_sub("annotate_library_exprs_outer_loop", &mut sub_t);
    let library_ir_annotated = build_ir_type_env(&library_annotated);
    log_sub("build_ir_type_env_from_annotated", &mut sub_t);

    // The annotation loop above recurses (annotate_expr_with_scope); if it
    // bailed on low stack, reject rather than return a partially-annotated
    // library context.
    let mut post_annotate_errors = Vec::new();
    stack_scope.drain_into(&mut post_annotate_errors);
    if !post_annotate_errors.is_empty() {
        return Err(InferResult {
            errors: post_annotate_errors,
            typed_nodes: 0,
            total_nodes: 0,
        });
    }

    Ok(TypeEnv::from_inner(TypeEnvInner {
        env: state.env,
        var_gen: state.var_gen,
        subst: state.subst,
        adt_reg: state.adt_reg,
        ir_types: library_ir_annotated,
        library_def_names,
        opacity: state.opacity,
    }))
}

/// Combined library-build helper: run the IR pipeline ONCE over the
/// library and return both the [`TypeEnv`] (for downstream `_with_context`
/// calls) and a [`CheckedProgram`] equivalent to what
/// `check_ir_with_context(&TypeEnv::empty(), library_exprs)` would
/// return.
///
/// This avoids the duplicated work that occurs when callers run
/// [`build_type_env_from_library`] followed by
/// `check_ir_with_context(empty, library)` — both paths separately
/// run a full HM inference + annotation pass over the same library
/// exprs. Per `docs/archive/perf/perf_baseline_investigation.md`, the unified path
/// saves ~16s of duplicated inference + annotation on Coral.
///
/// Behavior contract:
/// - The returned `TypeEnv` is identical (modulo non-determinism in
///   `HashMap` iteration) to `build_type_env_from_library(library_exprs)`.
/// - The returned `CheckedProgram` has the same `annotated_exprs()` and
///   `type_env()` shapes that
///   `check_ir_with_context(&TypeEnv::empty(), library_exprs)`
///   produces — namely, annotated library exprs in source order plus a
///   `ir_types` map keyed on every library def.
/// - On any error the same `Err(InferResult)` is returned that the
///   sequential calls would have returned.
///
/// Internal sequencing:
/// 1. Build the per-decl `IrTypeEnv` from un-annotated source.
/// 2. Run `infer_ir_program_with_state` once, populating `state`.
/// 3. Run all validators (`validate_ir_program`,
///    `validate_tensor_precisions_in_program`,
///    `suppress_unbound_for_cycle_members`).
/// 4. Annotate the library exprs once using the populated `state.env`.
/// 5. Build `library_ir_annotated` from the annotated exprs.
/// 6. Compose the `TypeEnv` from `state` + `library_ir_annotated`.
/// 7. Compose the `CheckedProgram` from the annotated exprs +
///    `library_ir_annotated`.
pub fn build_compiled_library_context(
    library_exprs: &[deep::Expr],
) -> Result<(TypeEnv, CheckedProgram), InferResult> {
    // Reset the stack-exhaustion flag for this check unit; drained into the
    // error vector below before the empty-errors gate (and again after the
    // annotation pass) so a deep-input stack bail on the library-compile path
    // always fails the check rather than returning a silent green / partial
    // CheckedProgram. Mirrors `check_ir_with_signature_context`.
    let stack_scope = StackExhaustionScope::enter();
    // Mirror `build_type_env_from_library` up through the validators so the
    // type_env half stays bit-compatible with the existing public API.
    let empty = TypeEnv::empty();
    let mut state = empty.inner().clone();

    let library_ir = build_ir_type_env(library_exprs);

    let mut result = infer_ir_program_with_state(
        library_exprs,
        &library_ir,
        &mut state,
        /* combined_ir_for_validate = */ &library_ir,
        /* run_validate_passes_on = */ None,
    );
    validate_ir_program(library_exprs, &library_ir, &mut result.errors);
    validate_tensor_precisions_in_program(library_exprs, &mut result.errors);
    crate::invariants::validate_type_invariants_in_program(library_exprs, &mut result.errors);
    validate_polymorphic_op_constraints(library_exprs, &library_ir, &mut result.errors);
    suppress_unbound_for_cycle_members(library_exprs, &mut result.errors);
    // Surface any stack-exhaustion bail from the passes above as a hard
    // located error (covered-or-rejected) before the empty-errors gate.
    stack_scope.drain_into(&mut result.errors);
    if !result.errors.is_empty() {
        return Err(result);
    }

    // Capture library def names before consuming `state` into `TypeEnv`.
    let mut library_def_names = std::collections::HashSet::new();
    for expr in top_level_decl_items(library_exprs) {
        if let deep::Expr::List(list, _) = expr
            && get_tag(list) == Some("def")
            && let Some(name) = children(list).first().and_then(symbol_name)
        {
            library_def_names.insert(name.to_string());
        }
    }

    // Drain accumulated errors back into the state's storage; they were
    // empty above so this is a no-op, but the call site is symmetric
    // with check_ir_with_context.
    let _ = result.errors.drain(..);

    // SINGLE annotation pass — feeds both the TypeEnv's
    // `ir_types` AND the returned CheckedProgram's `annotated_exprs`.
    // Previously `build_type_env_from_library` did one annotation here
    // (~13.8s on Coral) and `check_ir_with_context(empty, library)`
    // did a separate, redundant inference+annotation pass (~16.8s).
    //
    // Install the declared-`defsig` parameter type map so the `def`
    // arm of `annotate_expr_with_scope` stamps borrow-correct types
    // onto each library def's `(params ...)` node -- the library
    // compile path is exactly where chelis-std's separate-`sig` defs
    // (`School.Loss.CrossEntropy.loss` etc.) are annotated.
    let _declared_sig_guard = install_declared_sig_param_types(library_exprs);
    let library_annotated: Vec<deep::Expr> = library_exprs
        .iter()
        .map(|e| {
            annotate_expr_with_scope(e, &state.env, &state.var_gen, &state.subst, &state.adt_reg)
        })
        .collect();
    // Annotation also recurses (annotate_expr_with_scope); if it bailed on
    // low stack, reject rather than return a partially-annotated program.
    stack_scope.drain_into(&mut result.errors);
    if !result.errors.is_empty() {
        return Err(result);
    }
    let library_ir_annotated = build_ir_type_env(&library_annotated);

    let type_env = TypeEnv::from_inner(TypeEnvInner {
        env: state.env,
        var_gen: state.var_gen,
        subst: state.subst,
        adt_reg: state.adt_reg,
        ir_types: library_ir_annotated.clone(),
        library_def_names,
        opacity: state.opacity,
    });

    // Build the CheckedProgram with the same `annotated_type_env` shape
    // that `check_ir_with_context(empty, library)` produces. With an
    // empty outer scope, `context.inner().ir_types` is empty, so the
    // union step is a no-op and `annotated_type_env ==
    // library_ir_annotated`.
    let checked = CheckedProgram::from_parts(library_annotated, library_ir_annotated);

    Ok((type_env, checked))
}

/// Layered sibling of [`build_compiled_library_context`]: build a library
/// context for `library_exprs` *stacked on top of* an existing `base`
/// context instead of on the empty (builtins + prelude) state.
///
/// This is the seam the cross-process chelis-std typecheck cache uses for
/// its Layer 2 build: `base` is the cached chelis-std sub-context's
/// `TypeEnv`, and `library_exprs` is the non-chelis-std library decls
/// (the user package's own modules + path-deps). The chelis-std library
/// is checked once, cached, and never re-walked here; only the
/// `library_exprs` passed in are inferred + annotated.
///
/// Behavior contract:
/// - `library_exprs` are checked against `base` exactly as
///   [`check_ir_with_context`] would check new code against `base` — base
///   bindings are visible, base ADT constructor sets remain visible to
///   `match` exhaustivity, and `library_exprs` bindings shadow but do not
///   consume base bindings.
/// - The returned `TypeEnv` carries the **union** of base + `library_exprs`
///   declared types and def-name sets, so a subsequent
///   `check_ir_with_context` against it resolves `(var ...)` references
///   into both the base (chelis-std) and the `library_exprs` (package)
///   layers. `library_exprs` types win on shadow.
/// - The returned `CheckedProgram` carries the `library_exprs` annotated
///   bodies (NOT the base bodies — base bodies live in the base context's
///   own `CheckedProgram`). Downstream effects / linearity / lowering must
///   compose this against the base context's `CheckedProgram` /
///   `LoweredLibrary` via the `_with_context` variants, exactly as the
///   monolithic-vs-layered split requires.
/// - On any error the same `Err(InferResult)` is returned that
///   `check_ir_with_context(base, library_exprs)` would return.
pub fn build_compiled_library_context_with_base(
    base: &TypeEnv,
    library_exprs: &[deep::Expr],
) -> Result<(TypeEnv, CheckedProgram), InferResult> {
    // Reset the stack-exhaustion flag for this check unit; drained into the
    // error vector below before the empty-errors gate (and again after the
    // annotation pass) so a deep-input stack bail on the layered
    // library-compile path always fails the check rather than returning a
    // silent green / partial CheckedProgram. Mirrors
    // `check_ir_with_signature_context`.
    let stack_scope = StackExhaustionScope::enter();
    // Seed from the base context's snapshot rather than the empty state.
    let mut state = base.inner().clone();

    // `library_exprs` declared types (IR), layered on top of the base's.
    let new_ir = build_ir_type_env(library_exprs);
    let combined_ir: HashMap<String, deep::Expr> = state
        .ir_types
        .iter()
        .chain(new_ir.iter())
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    // Base is already validated; run inference + validators on
    // `library_exprs` only. The IR env passed to inference is the
    // `library_exprs`' own declared types (base schemes are already in
    // `state.env`); the combined IR env is supplied to the validators so
    // `(var basefoo)` references resolve to the base's declared type.
    let mut result = infer_ir_program_with_state(
        library_exprs,
        &new_ir,
        &mut state,
        &combined_ir,
        /* run_validate_passes_on = */ None,
    );
    validate_ir_program(library_exprs, &combined_ir, &mut result.errors);
    validate_tensor_precisions_in_program(library_exprs, &mut result.errors);
    crate::invariants::validate_type_invariants_in_program(library_exprs, &mut result.errors);
    validate_polymorphic_op_constraints(library_exprs, &combined_ir, &mut result.errors);
    suppress_unbound_for_cycle_members_against_context(
        library_exprs,
        &base.inner().library_def_names,
        &mut result.errors,
    );
    // Surface any stack-exhaustion bail from the passes above as a hard
    // located error (covered-or-rejected) before the empty-errors gate.
    stack_scope.drain_into(&mut result.errors);
    if !result.errors.is_empty() {
        return Err(result);
    }

    // Capture `library_exprs` def names, unioned with the base's, so a
    // subsequent `_with_context` check against the returned TypeEnv
    // distinguishes library refs (base + this layer) from new-code refs.
    let mut library_def_names = base.inner().library_def_names.clone();
    for expr in top_level_decl_items(library_exprs) {
        if let deep::Expr::List(list, _) = expr
            && get_tag(list) == Some("def")
            && let Some(name) = children(list).first().and_then(symbol_name)
        {
            library_def_names.insert(name.to_string());
        }
    }

    let _ = result.errors.drain(..);

    // Annotate ONLY the `library_exprs`, starting from the populated
    // `state` so base names resolve during annotation. Install the
    // declared-`defsig` parameter type map for this layer's exprs so
    // separate-`sig` defs get borrow-correct `(params ...)` stamps,
    // matching `build_compiled_library_context`.
    let _declared_sig_guard = install_declared_sig_param_types(library_exprs);
    let library_annotated: Vec<deep::Expr> = library_exprs
        .iter()
        .map(|e| {
            annotate_expr_with_scope(e, &state.env, &state.var_gen, &state.subst, &state.adt_reg)
        })
        .collect();
    // Annotation also recurses (annotate_expr_with_scope); if it bailed on
    // low stack, reject rather than return a partially-annotated program.
    stack_scope.drain_into(&mut result.errors);
    if !result.errors.is_empty() {
        return Err(result);
    }
    let new_ir_annotated = build_ir_type_env(&library_annotated);

    // The returned TypeEnv's `ir_types` is the union: base declared types
    // plus this layer's, this layer winning on shadow.
    let mut combined_ir_annotated = new_ir_annotated.clone();
    for (name, ty) in &base.inner().ir_types {
        combined_ir_annotated
            .entry(name.clone())
            .or_insert_with(|| ty.clone());
    }

    let type_env = TypeEnv::from_inner(TypeEnvInner {
        env: state.env,
        var_gen: state.var_gen,
        subst: state.subst,
        adt_reg: state.adt_reg,
        ir_types: combined_ir_annotated,
        library_def_names,
        opacity: state.opacity,
    });

    // The CheckedProgram carries this layer's annotated bodies plus a
    // unioned `type_env` so downstream `_with_context` passes resolve
    // both base and this-layer `(var ...)` references. This mirrors the
    // `check_ir_with_context` returned-CheckedProgram contract.
    let mut checked_type_env = new_ir_annotated;
    for (name, ty) in &base.inner().ir_types {
        checked_type_env
            .entry(name.clone())
            .or_insert_with(|| ty.clone());
    }
    let checked = CheckedProgram::from_parts(library_annotated, checked_type_env);

    Ok((type_env, checked))
}

/// Type-check `new_exprs` against an outer-scope `context`. New-code
/// bindings shadow but do not consume library bindings; library ADT
/// constructor sets remain visible to new-code `match` exhaustivity
/// checks. The returned [`CheckedProgram`] contains ONLY the new-code's
/// checked decls; library decls are not duplicated.
///
/// The `context` is `Arc`-shared and never mutated — repeated calls
/// against the same context see the same outer scope.
///
/// ## Downstream-caller contract
///
/// The returned `CheckedProgram` is asymmetric on purpose:
/// - `type_env()` is **unioned** — it carries library + new-code declared
///   types so callers like `chelis_ir::lower_program` and
///   `chelis_effects::check_program` can resolve `(var libname)` references
///   from new-code bodies. New-code types win on shadow.
/// - `annotated_exprs()` is **new-code only** — library bodies are NOT
///   present. Phase D / E / F (effects, linearity, lowering) callers MUST
///   use the corresponding `_with_context` variants, not the monolithic
///   `check_program` / `check_linearity` / `lower_program`. The monolithic
///   APIs need to walk library bodies and will silently mis-handle
///   library-effect propagation, library tensor consumption, and library
///   IR roots if fed only the new-code annotated decls.
pub fn check_ir_with_context(
    context: &TypeEnv,
    new_exprs: &[deep::Expr],
) -> Result<CheckedProgram, InferResult> {
    check_ir_with_signature_context(context, &SignatureInferenceMetadata::default(), new_exprs)
}

pub fn check_ir_with_signature_context(
    context: &TypeEnv,
    signature_context: &SignatureInferenceMetadata,
    new_exprs: &[deep::Expr],
) -> Result<CheckedProgram, InferResult> {
    // WI-1 follow-up: grow the stack for the whole check pipeline. This is the
    // funnel for `check_ir_program` and `check_ir_with_context`, so wrapping
    // here grows the stack for all three.
    with_grown_stack(|| {
        check_ir_with_signature_context_inner(context, signature_context, new_exprs)
    })
}

fn check_ir_with_signature_context_inner(
    context: &TypeEnv,
    signature_context: &SignatureInferenceMetadata,
    new_exprs: &[deep::Expr],
) -> Result<CheckedProgram, InferResult> {
    // Reset the stack-exhaustion flag for this check unit; drained into the
    // error vector below before the empty-errors gate so a deep-input stack
    // bail always fails the check (never a silent green / partial result).
    let stack_scope = StackExhaustionScope::enter();
    let detail_profile = std::env::var_os("CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL")
        .map(|v| v == "1")
        .unwrap_or(false);
    let mut sub_t = std::time::Instant::now();
    let log_sub = |label: &str, t: &mut std::time::Instant| {
        if detail_profile {
            eprintln!(
                "check_ir_sub: {:>8.4}s {}",
                t.elapsed().as_secs_f64(),
                label
            );
            *t = std::time::Instant::now();
        }
    };
    let mut state = context.inner().clone();

    // New-code declared types (IR) layered on top of library's.
    let new_ir = build_ir_type_env(new_exprs);
    let combined_ir: HashMap<String, deep::Expr> = state
        .ir_types
        .iter()
        .chain(new_ir.iter())
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    log_sub("build_ir_and_combine", &mut sub_t);

    // Library is already validated; only run validate / inference on
    // new exprs. The IR env passed to inference is the new-code's
    // own declared types (library schemes are already in state.env).
    let mut result = infer_ir_program_with_state(
        new_exprs,
        &new_ir,
        &mut state,
        &combined_ir,
        /* run_validate_passes_on = */ None,
    );
    log_sub("infer_ir_program_with_state", &mut sub_t);
    // Run cycle / shape / precision validators on new_exprs only. The
    // combined IR env is supplied so `(var libfoo)` references
    // resolve to the library's declared type during shape validation.
    validate_ir_program(new_exprs, &combined_ir, &mut result.errors);
    log_sub("validate_ir_program", &mut sub_t);
    validate_tensor_precisions_in_program(new_exprs, &mut result.errors);
    crate::invariants::validate_type_invariants_in_program(new_exprs, &mut result.errors);
    log_sub("validate_tensor_precisions", &mut sub_t);
    validate_polymorphic_op_constraints(new_exprs, &combined_ir, &mut result.errors);
    log_sub("validate_polymorphic_op_constraints", &mut sub_t);
    suppress_unbound_for_cycle_members_against_context(
        new_exprs,
        &context.inner().library_def_names,
        &mut result.errors,
    );
    log_sub("suppress_unbound_for_cycle", &mut sub_t);
    // Surface any stack-exhaustion bail from the passes above as a hard
    // located error (covered-or-rejected) before the empty-errors gate.
    stack_scope.drain_into(&mut result.errors);
    if !result.errors.is_empty() {
        return Err(result);
    }

    // Annotate ONLY the new-code exprs, starting from the library
    // snapshot state so library names resolve during annotation.
    let annotated_exprs = annotate_ir_program_with_context(context, new_exprs);
    log_sub("annotate_ir_program_with_context", &mut sub_t);
    // Annotation also recurses (annotate_expr_with_scope); if it bailed on
    // low stack, reject rather than return a partially-annotated program.
    stack_scope.drain_into(&mut result.errors);
    if !result.errors.is_empty() {
        return Err(result);
    }
    // Surface library declared types in the returned type_env so downstream
    // passes (lower, effects, linearity) can resolve `(var libname)` calls
    // from new-code without a separate library lookup. New-code types take
    // precedence on shadow.
    let mut annotated_type_env = build_ir_type_env(&annotated_exprs);
    for (name, ty) in &context.inner().ir_types {
        annotated_type_env
            .entry(name.clone())
            .or_insert_with(|| ty.clone());
    }
    log_sub("annotated_type_env_build", &mut sub_t);
    Ok(CheckedProgram::from_parts_with_signature_context(
        annotated_exprs,
        annotated_type_env,
        signature_context,
    ))
}

pub fn check_typed_program(exprs: &[deep::Expr]) -> Result<CheckedProgram, InferResult> {
    // WI-1 follow-up: grow the stack so both the inference call below and the
    // annotation pass run with headroom on deep input.
    with_grown_stack(|| check_typed_program_inner(exprs))
}

fn check_typed_program_inner(exprs: &[deep::Expr]) -> Result<CheckedProgram, InferResult> {
    // Outermost scope covers both inference (which has its own inner scope)
    // and the annotation pass below, so a bail in either surfaces as a hard
    // located failure rather than a partially-annotated `Ok`.
    let stack_scope = StackExhaustionScope::enter();
    let mut result = infer_program(exprs);
    if result.errors.is_empty() {
        let annotated_exprs = annotate_ir_program(exprs);
        let annotated_type_env = build_ir_type_env(&annotated_exprs);
        // Annotation recurses; reject if it bailed on low stack.
        stack_scope.drain_into(&mut result.errors);
        if !result.errors.is_empty() {
            return Err(result);
        }
        Ok(CheckedProgram::from_parts(
            annotated_exprs,
            annotated_type_env,
        ))
    } else {
        Err(result)
    }
}

pub fn infer_ir_program(exprs: &[deep::Expr]) -> InferResult {
    // WI-1 follow-up: grow the stack for the whole pipeline.
    with_grown_stack(|| infer_ir_program_inner(exprs))
}

fn infer_ir_program_inner(exprs: &[deep::Expr]) -> InferResult {
    let stack_scope = StackExhaustionScope::enter();
    let type_env = build_ir_type_env(exprs);
    let mut result = infer_ir_program_with_env(exprs, &type_env);
    validate_ir_program(exprs, &type_env, &mut result.errors);
    validate_tensor_precisions_in_program(exprs, &mut result.errors);
    crate::invariants::validate_type_invariants_in_program(exprs, &mut result.errors);
    validate_polymorphic_op_constraints(exprs, &type_env, &mut result.errors);
    suppress_unbound_for_cycle_members(exprs, &mut result.errors);
    // Surface any walker stack bail as a hard located error.
    stack_scope.drain_into(&mut result.errors);
    result
}

/// When a binding cycle is detected, the inference pass that processed
/// the cycle in textual order often reports `UnboundVariable` for the
/// later cycle members (the lookup landed before the subsequent def was
/// elaborated). Those errors are spurious noise — the names ARE defined,
/// they're just circularly. Drop any `UnboundVariable` whose name matches
/// a top-level def.
fn suppress_unbound_for_cycle_members(exprs: &[deep::Expr], errors: &mut Vec<CheckError>) {
    let mut def_names: HashSet<String> = HashSet::new();
    for expr in top_level_decl_items(exprs) {
        if let deep::Expr::List(list, _) = expr
            && get_tag(list) == Some("def")
            && let Some(name) = children(list).first().and_then(symbol_name)
        {
            def_names.insert(name.to_string());
        }
    }
    errors.retain(|err| {
        if !matches!(err.kind, CheckErrorKind::UnboundVariable) {
            return true;
        }
        let name_start = match err.message.find("unbound variable: ") {
            Some(start) => start + "unbound variable: ".len(),
            None => return true,
        };
        let name = err.message[name_start..]
            .split_whitespace()
            .next()
            .unwrap_or("");
        !def_names.contains(name)
    });
}

/// Stacked-context variant of `suppress_unbound_for_cycle_members`.
///
/// Drops `UnboundVariable` errors whose name matches either a new-code
/// def OR a library def — the latter is needed because a library def
/// referenced from the new code might temporarily look unbound during
/// inference if the inferred error path runs before the env scheme
/// lookup, but the name IS in the library context.
fn suppress_unbound_for_cycle_members_against_context(
    new_exprs: &[deep::Expr],
    library_def_names: &HashSet<String>,
    errors: &mut Vec<CheckError>,
) {
    let mut def_names: HashSet<String> = library_def_names.clone();
    for expr in top_level_decl_items(new_exprs) {
        if let deep::Expr::List(list, _) = expr
            && get_tag(list) == Some("def")
            && let Some(name) = children(list).first().and_then(symbol_name)
        {
            def_names.insert(name.to_string());
        }
    }
    errors.retain(|err| {
        if !matches!(err.kind, CheckErrorKind::UnboundVariable) {
            return true;
        }
        let name_start = match err.message.find("unbound variable: ") {
            Some(start) => start + "unbound variable: ".len(),
            None => return true,
        };
        let name = err.message[name_start..]
            .split_whitespace()
            .next()
            .unwrap_or("");
        !def_names.contains(name)
    });
}

fn infer_ir_program_with_env(exprs: &[deep::Expr], type_env: &IrTypeEnv) -> InferResult {
    // Backwards-compat wrapper. Callers (like `infer_ir_program` and
    // `check_typed_program` callers) run `validate_ir_program`
    // separately, so we pass `None` here to skip the embedded validate.
    let empty_inner = crate::context::TypeEnv::empty();
    let mut state = empty_inner.inner().clone();
    infer_ir_program_with_state(
        exprs, type_env, &mut state, type_env, /* run_validate_passes_on = */ None,
    )
}

/// Run the inference / IR binding / shape-validation passes against
/// `state`, mutating it as it goes. Library state should be supplied by
/// pre-cloning a snapshot; pass `&[]`-derived state for the monolithic
/// path. `new_ir_types` are bound into `state.env` here; the
/// `combined_ir` is what `validate_ir_program` consults so
/// new-code shape validation can look up declared types of library
/// references.
fn infer_ir_program_with_state(
    exprs: &[deep::Expr],
    new_ir_types: &IrTypeEnv,
    state: &mut TypeEnvInner,
    combined_ir: &IrTypeEnv,
    run_validate_passes_on: Option<&[deep::Expr]>,
) -> InferResult {
    let mut errors = Vec::new();
    let mut typed_nodes = 0;
    let mut total_nodes = 0;

    // RFC v4b (RT-1 F2): reject a named module opened by more than one
    // wrapper in this check unit (module-identity forgery). Reef-linked
    // decls carry no wrappers, so this only fires on hand-written `.dp`.
    detect_module_reopens(exprs, &mut errors);
    // RFC v5 (RT-1 F2 bypass): reject the reef linker's reserved
    // internal-name format in programs not produced by the linker.
    detect_forged_linker_names(exprs, &mut errors);

    // Descend through `(module {} name ...)` wrappers: every idiomatic
    // Surf source wraps its declarations in `module X`, and without
    // flattening none of the walkers below see any def/defsig/deftype.
    let items = top_level_decl_items_with_modules(exprs);
    collect_all_declarations(
        &items,
        &mut state.env,
        &mut state.var_gen,
        &mut state.subst,
        &mut state.adt_reg,
        &mut errors,
    );

    // Checker-enforced opacity (RFC D-CHECK): accumulate this phase's
    // program-shape metadata into the persistent state (so the
    // stacked library/new-code paths keep library exports visible)
    // and install the per-run context for the inference hooks.
    let phase_meta = build_opacity_meta(&items, &state.adt_reg, &mut state.var_gen);
    state.opacity.merge_from(&phase_meta);
    let _opacity_guard = crate::opacity::install_opacity_context(
        crate::opacity::OpacityContextData::from_meta(state.opacity.clone()),
    );

    for (name, ty_expr) in new_ir_types {
        let ty = deep_type_to_resolved_type(
            ty_expr,
            &mut state.var_gen,
            &state.adt_reg,
            &mut HashMap::new(),
        );
        let scheme = state.env.generalize(&ty, &state.subst);
        state.env.bind(name.clone(), scheme);
    }

    // Per-decl profile: when CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL=1, emit
    // one stderr line per top-level decl with its name and inference time.
    // Aggregated by name in caller scripts to attribute cost per module.
    let detail_profile = std::env::var_os("CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL")
        .map(|v| v == "1")
        .unwrap_or(false);
    let user_def_names = collect_user_def_names(&top_level_decl_items(exprs));
    for (module, expr) in &items {
        let t0 = if detail_profile {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let decl_name = top_level_decl_name(expr);
        crate::opacity::set_current_item(
            crate::opacity::module_key_for_item(module.as_deref(), decl_name),
            decl_name.map(str::to_string),
        );
        infer_top_level(
            expr,
            &mut state.env,
            &mut state.var_gen,
            &mut state.subst,
            &state.adt_reg,
            &mut errors,
            &mut typed_nodes,
            &mut total_nodes,
            &user_def_names,
        );
        if let Some(t0) = t0 {
            let elapsed = t0.elapsed();
            let name = top_level_decl_name(expr).unwrap_or("<anon>");
            eprintln!("infer_ir_decl: {:>8.4}s {}", elapsed.as_secs_f64(), name);
        }
        // Issue #256 round 2: drain the deferred-borrow ledger for this
        // def and re-check each recorded variable against the now-complete
        // substitution. Draining per-def keeps error attribution local and
        // prevents one def's deferrals from leaking into the next.
        validate_deferred_borrow_vars(&state.subst, &state.adt_reg, &mut errors);
        // D-CHECK: drain the per-def deferred-access ledger (see
        // `validate_deferred_opaque_uses`).
        validate_deferred_opaque_uses(&state.subst, &state.adt_reg, &mut errors);
    }
    crate::opacity::set_current_item(None, None);

    for warning in chelis_deep::validate::validate(exprs) {
        errors.push(
            CheckError::new(
                match warning.kind {
                    chelis_deep::validate::WarningKind::Arity => CheckErrorKind::ArityMismatch,
                    _ => CheckErrorKind::Other,
                },
                warning.message,
                vec!["Use canonical Deep 3-tuple forms from spec/03".to_string()],
            )
            .at_offset(warning.offset),
        );
    }

    if let Some(target_exprs) = run_validate_passes_on {
        validate_ir_program(target_exprs, combined_ir, &mut errors);
    }

    InferResult {
        errors,
        typed_nodes,
        total_nodes,
    }
}

type IrTypeEnv = HashMap<String, deep::Expr>;

fn build_ir_type_env(exprs: &[deep::Expr]) -> IrTypeEnv {
    let mut env = HashMap::new();
    for expr in top_level_decl_items(exprs) {
        collect_ir_types(expr, &mut env);
    }
    env
}

fn collect_ir_types(expr: &deep::Expr, env: &mut IrTypeEnv) {
    let deep::Expr::List(list, _) = expr else {
        return;
    };
    if get_tag(list) != Some("def") {
        return;
    }
    let kids = children(list);
    if kids.len() < 2 {
        return;
    }
    if let Some(name) = symbol_name(&kids[0])
        && let Some(ty) = expr_type_expr(&kids[1], env)
    {
        env.insert(name.to_string(), ty);
    }
}

fn validate_ir_program(exprs: &[deep::Expr], type_env: &IrTypeEnv, errors: &mut Vec<CheckError>) {
    detect_top_level_binding_cycles(exprs, errors);
    detect_trivial_non_terminating_fns(exprs, errors);
    let mut static_env = HashMap::new();
    // Names of let-bindings whose RHS validation already emitted a
    // diagnostic (so their derived output type is unknown). Downstream
    // shape-sensitive calls that consume such a name emit a redundant
    // cascade diagnostic; suppress it. See RT-205 round-2 F3.
    let mut failed_let_names: HashSet<String> = HashSet::new();
    for expr in top_level_decl_items(exprs) {
        validate_ir_expr(
            expr,
            type_env,
            &mut static_env,
            &mut failed_let_names,
            errors,
        );
    }
}

/// Detect fn defs whose body is a direct self-call with no conditional
/// guard — e.g. `def a(x) = a(x)`. These are guaranteed non-terminating
/// when called and, because the DAG lowerer can't represent recursion,
/// get silently elided to an identity in the generated C (source/object
/// divergence). Flag at check time so the user sees a clear error
/// instead of shipping a program that means something else than written.
///
/// This only catches the most trivial shape — a body that is literally
/// `(app (var name) ...)` with the def's own name as the callee. Real
/// recursive fns with a base case inside `if`/`match` (e.g. `fact n = if
/// n <= 1 then 1 else mul(n, fact(n-1))`) are NOT flagged.
fn detect_trivial_non_terminating_fns(exprs: &[deep::Expr], errors: &mut Vec<CheckError>) {
    // Collect each def's "terminal callees" — the top-level fn names
    // reached at every tail position of the body. `Some(set)` means
    // every tail is a call; the set is who's called. `None` means the
    // body has at least one non-call tail (a base case exists).
    let mut terminal_callees: HashMap<String, Option<HashSet<String>>> = HashMap::new();
    let mut def_order: Vec<String> = Vec::new();
    for expr in top_level_decl_items(exprs) {
        let deep::Expr::List(list, _) = expr else {
            continue;
        };
        if get_tag(list) != Some("def") {
            continue;
        }
        let kids = children(list);
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        let Some(body) = kids.get(1) else { continue };
        // Check params for a name that shadows the def — a body that
        // terminal-calls a shadowed name is NOT self-recursion.
        let mut shadows: HashSet<String> = HashSet::new();
        if let deep::Expr::List(fn_list, _) = body
            && get_tag(fn_list) == Some("fn")
            && let Some(deep::Expr::List(params, _)) = children(fn_list).first()
            && get_tag(params) == Some("params")
        {
            for param in children(params) {
                if let Some(pname) = param_name_for_refs(param) {
                    shadows.insert(pname);
                }
            }
        }
        let fn_body = match body {
            deep::Expr::List(list, _) if get_tag(list) == Some("fn") => children(list).get(1),
            _ => None,
        };
        let entry = if let Some(fn_body) = fn_body {
            let mut callees: HashSet<String> = HashSet::new();
            if collect_terminal_callees(fn_body, &shadows, &mut callees) {
                Some(callees)
            } else {
                None
            }
        } else {
            None
        };
        def_order.push(name.to_string());
        terminal_callees.insert(name.to_string(), entry);
    }

    // Greatest-fixed-point: start with ALL defs whose every tail is a
    // call (no base case) and iteratively remove any def that calls out
    // to a base-case def (outside the candidate set). What survives is
    // a closed recursion group with no base case anywhere.
    let mut non_terminating: HashSet<String> = terminal_callees
        .iter()
        .filter_map(|(name, callees)| {
            callees.as_ref().and_then(|set| {
                if set.is_empty() {
                    None
                } else {
                    Some(name.clone())
                }
            })
        })
        .collect();
    loop {
        let mut changed = false;
        let snapshot: Vec<String> = non_terminating.iter().cloned().collect();
        for name in &snapshot {
            let Some(Some(callees)) = terminal_callees.get(name) else {
                non_terminating.remove(name);
                changed = true;
                continue;
            };
            // Every callee must either be `name` itself OR remain in the
            // non_terminating candidate set. If any callee has a known
            // base case (isn't in non_terminating), this def has an
            // escape route and isn't trivially non-terminating.
            let ok = callees
                .iter()
                .all(|c| c == name || non_terminating.contains(c));
            if !ok {
                non_terminating.remove(name);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    for name in &def_order {
        if non_terminating.contains(name) {
            errors.push(CheckError::new(
                CheckErrorKind::CycleDetected,
                format!(
                    "def `{name}` is trivially non-terminating. Every tail position \
                     calls back into the same recursion group `{name}` with no base case; \
                     add an `if`/`match` exit that returns without recursing"
                ),
                vec![
                    "Trivial (self- or mutual-) recursion without a base case isn't \
                     representable in the Phase 0 DAG lowering and would compile to \
                     an infinite loop or silent identity."
                        .to_string(),
                ],
            ));
        }
    }
}

/// Walk `expr` and, for every terminal (tail) position, record the name
/// called (if the tail is `(app (var Y) ...)`). Returns `true` if EVERY
/// terminal is a call (no base-case leaf); `false` if any terminal is a
/// non-call (literal, var-read, tuple, etc.) — a base case exists.
fn collect_terminal_callees(
    expr: &deep::Expr,
    shadowed: &HashSet<String>,
    out: &mut HashSet<String>,
) -> bool {
    stack_guard!("collect_terminal_callees", expr, false);
    match expr {
        deep::Expr::MetaExpr(meta, _) => collect_terminal_callees(&meta.expr, shadowed, out),
        deep::Expr::List(list, _) => match get_tag(list) {
            Some("app") => {
                let kids = children(list);
                let Some(callee) = kids.first() else {
                    return false;
                };
                let deep::Expr::List(callee_list, _) = callee else {
                    return false;
                };
                if get_tag(callee_list) != Some("var") {
                    return false;
                }
                let Some(cname) = children(callee_list).first().and_then(symbol_name) else {
                    return false;
                };
                if shadowed.contains(cname) {
                    return false;
                }
                out.insert(cname.to_string());
                true
            }
            Some("let") => {
                let kids = children(list);
                kids.get(1)
                    .map(|body| collect_terminal_callees(body, shadowed, out))
                    .unwrap_or(false)
            }
            Some("if") => {
                let kids = children(list);
                if kids.len() < 3 {
                    return false;
                }
                let then_ok = collect_terminal_callees(&kids[1], shadowed, out);
                let else_ok = collect_terminal_callees(&kids[2], shadowed, out);
                then_ok && else_ok
            }
            Some("match") => {
                let kids = children(list);
                if kids.len() < 2 {
                    return false;
                }
                kids.iter().skip(1).all(|arm| {
                    if let deep::Expr::List(arm_list, _) = arm
                        && get_tag(arm_list) == Some("arm")
                    {
                        children(arm_list)
                            .get(2)
                            .map(|body| collect_terminal_callees(body, shadowed, out))
                            .unwrap_or(false)
                    } else {
                        false
                    }
                })
            }
            _ => false,
        },
        _ => false,
    }
}

#[allow(dead_code)]
fn fn_body_is_direct_self_call(def_body: &deep::Expr, def_name: &str) -> bool {
    let fn_list = match def_body {
        deep::Expr::List(list, _) if get_tag(list) == Some("fn") => list,
        _ => return false,
    };
    // If any fn param shadows the def name, the callee reference inside
    // the body refers to the param (a callable HOF argument), not the def
    // itself. This is a legitimate HOF call, not recursion.
    if let Some(params_list) = children(fn_list).first()
        && let deep::Expr::List(params, _) = params_list
        && get_tag(params) == Some("params")
    {
        for param in children(params) {
            if param_name_for_refs(param).as_deref() == Some(def_name) {
                return false;
            }
        }
    }
    let Some(body) = children(fn_list).get(1) else {
        return false;
    };
    every_terminal_is_self_call(body, def_name)
}

/// True when every terminal (tail) position of `expr` is a direct call
/// to `def_name`. Walks through `let` bodies, both arms of `if`, and
/// every `match` arm body. Any non-self-call terminal (a literal, a
/// different fn call, a non-self var) makes this false — that terminal
/// is a potential base case and the recursion isn't trivial.
#[allow(dead_code)]
fn every_terminal_is_self_call(expr: &deep::Expr, def_name: &str) -> bool {
    stack_guard!("every_terminal_is_self_call", expr, false);
    match expr {
        deep::Expr::MetaExpr(meta, _) => every_terminal_is_self_call(&meta.expr, def_name),
        deep::Expr::List(list, _) => match get_tag(list) {
            Some("app") => {
                let kids = children(list);
                let Some(callee) = kids.first() else {
                    return false;
                };
                let deep::Expr::List(callee_list, _) = callee else {
                    return false;
                };
                if get_tag(callee_list) != Some("var") {
                    return false;
                }
                children(callee_list).first().and_then(symbol_name) == Some(def_name)
            }
            Some("let") => {
                let kids = children(list);
                kids.get(1)
                    .map(|body| every_terminal_is_self_call(body, def_name))
                    .unwrap_or(false)
            }
            Some("if") => {
                let kids = children(list);
                if kids.len() < 3 {
                    return false;
                }
                every_terminal_is_self_call(&kids[1], def_name)
                    && every_terminal_is_self_call(&kids[2], def_name)
            }
            Some("match") => {
                let kids = children(list);
                if kids.len() < 2 {
                    return false;
                }
                kids.iter().skip(1).all(|arm| {
                    if let deep::Expr::List(arm_list, _) = arm
                        && get_tag(arm_list) == Some("arm")
                    {
                        children(arm_list)
                            .get(2)
                            .map(|body| every_terminal_is_self_call(body, def_name))
                            .unwrap_or(false)
                    } else {
                        false
                    }
                })
            }
            _ => false,
        },
        _ => false,
    }
}

/// Check whether a def body is literally a self-reference — the Nautilus
/// external-input pattern `x = x` or `x = (x : T)`. Only this structural
/// shape earns the self-loop carve-out; self-references reached via a fn
/// application, a tuple, an if, etc. are real cycles and must be reported.
fn body_is_literal_self_ref(body: &deep::Expr, name: &str) -> bool {
    let mut current = body;
    loop {
        match current {
            deep::Expr::MetaExpr(meta, _) => current = &meta.expr,
            deep::Expr::List(list, _) => match get_tag(list) {
                Some("var") => {
                    return children(list).first().and_then(symbol_name) == Some(name);
                }
                // Type ascription desugars into a `(cast ... )`-like node
                // in Deep: `(x : T)` keeps `x` as the first child. When
                // the underlying is a var with the self name, treat it as
                // the Nautilus pattern.
                Some("ascribe") | Some(":") => match children(list).first() {
                    Some(inner) => current = inner,
                    None => return false,
                },
                _ => return false,
            },
            _ => return false,
        }
    }
}

/// Yield each top-level declaration, flattening through a `(module {} name ...)`
/// wrapper if present. Deep sources produced by Surf `module X` desugaring
/// have every def/defsig/deftype inside this wrapper; without flattening,
/// top-level walkers see a single `(module ...)` and miss everything inside.
/// Extract the name of a top-level decl (`def`, `defsig`, `deftype`,
/// `typealias`) for profile instrumentation. Returns `None` for shapes
/// that don't have a leading symbol.
fn top_level_decl_name(expr: &deep::Expr) -> Option<&str> {
    let deep::Expr::List(list, _) = expr else {
        return None;
    };
    let tag = get_tag(list)?;
    if !matches!(tag, "def" | "defsig" | "deftype" | "typealias") {
        return None;
    }
    children(list).first().and_then(symbol_name)
}

fn top_level_decl_items(exprs: &[deep::Expr]) -> Vec<&deep::Expr> {
    fn push<'a>(expr: &'a deep::Expr, out: &mut Vec<&'a deep::Expr>) {
        if let deep::Expr::List(list, _) = expr
            && get_tag(list) == Some("module")
        {
            // `(module {} name children...)` — skip tag, meta, name.
            for child in list.elements.iter().skip(3) {
                push(child, out);
            }
            return;
        }
        out.push(expr);
    }
    let mut out = Vec::new();
    for expr in exprs {
        push(expr, &mut out);
    }
    out
}

/// Like [`top_level_decl_items`] but pairs each flattened item with
/// its enclosing lexical module key: nested `(module ...)` names
/// joined with `.`, `None` for items outside any wrapper. This is the
/// module-identity source for checker-enforced opacity (RFC D-CHECK);
/// reef package-linked items carry no wrapper and key through their
/// internal-name stem instead (see `opacity::module_key_for_item`).
fn top_level_decl_items_with_modules(exprs: &[deep::Expr]) -> Vec<(Option<String>, &deep::Expr)> {
    fn push<'a>(
        expr: &'a deep::Expr,
        prefix: Option<&str>,
        out: &mut Vec<(Option<String>, &'a deep::Expr)>,
    ) {
        if let deep::Expr::List(list, _) = expr
            && get_tag(list) == Some("module")
        {
            // `(module {} name children...)` — skip tag, meta, name.
            let name = list.elements.get(2).and_then(symbol_name);
            let key = match (prefix, name) {
                (Some(p), Some(n)) => Some(format!("{p}.{n}")),
                (None, Some(n)) => Some(n.to_string()),
                (p, None) => p.map(str::to_string),
            };
            for child in list.elements.iter().skip(3) {
                push(child, key.as_deref(), out);
            }
            return;
        }
        out.push((prefix.map(str::to_string), expr));
    }
    let mut out = Vec::new();
    for expr in exprs {
        push(expr, None, &mut out);
    }
    out
}

/// RFC v4b (RT-1 F2): a named module may be opened by at most one
/// `(module ...)` wrapper per check unit. Module identity is otherwise
/// a forgeable string -- a second wrapper of an opaque type's defining
/// module would construct and inspect the type as if it were inside.
/// Walks every wrapper (including nested ones, keyed by their full
/// `.`-joined path) and emits ONE `DuplicateModule` error per
/// re-opened name. Surf emits one module per file and reef strips
/// wrappers before inference, so this only fires on hand-written `.dp`
/// (the forge surface).
fn detect_module_reopens(exprs: &[deep::Expr], errors: &mut Vec<CheckError>) {
    fn walk(
        expr: &deep::Expr,
        prefix: Option<&str>,
        seen: &mut HashSet<String>,
        reported: &mut HashSet<String>,
        errors: &mut Vec<CheckError>,
    ) {
        let deep::Expr::List(list, _) = expr else {
            return;
        };
        if get_tag(list) != Some("module") {
            return;
        }
        let name = list.elements.get(2).and_then(symbol_name);
        let key = match (prefix, name) {
            (Some(p), Some(n)) => Some(format!("{p}.{n}")),
            (None, Some(n)) => Some(n.to_string()),
            (p, None) => p.map(str::to_string),
        };
        if let Some(key) = &key
            && !seen.insert(key.clone())
            && reported.insert(key.clone())
        {
            errors.push(CheckError::new(
                CheckErrorKind::DuplicateModule,
                format!(
                    "module `{key}` is opened by more than one module wrapper in this \
                     check unit; a named module may be opened at most once"
                ),
                vec![format!(
                    "merge the `{key}` wrappers into one, or rename one of them"
                )],
            ));
        }
        for child in list.elements.iter().skip(3) {
            walk(child, key.as_deref(), seen, reported, errors);
        }
    }
    let mut seen = HashSet::new();
    let mut reported = HashSet::new();
    for expr in exprs {
        walk(expr, None, &mut seen, &mut reported, errors);
    }

    // RFC v5 belt-and-suspenders (RT-1 F2 bypass): a stem-derived
    // module identity (from a top-level mangled `deftype`/`def` name)
    // that collides with a lexical wrapper key in the same check unit
    // is also a `DuplicateModule` error. Genuine linker output has NO
    // lexical wrappers, so this never fires on it; it defends the
    // stem-plus-wrapper forge shapes even if the name-format check is
    // somehow bypassed. Runs unconditionally (structural).
    for (lexical, expr) in top_level_decl_items_with_modules(exprs) {
        // Only flat (non-wrapped) mangled declarations introduce a
        // stem-derived module identity; a name inside a lexical
        // wrapper keys to the wrapper, not its stem.
        if lexical.is_some() {
            continue;
        }
        let deep::Expr::List(list, _) = expr else {
            continue;
        };
        if !matches!(get_tag(list), Some("deftype" | "def")) {
            continue;
        }
        let Some(name) = children(list).first().and_then(symbol_name) else {
            continue;
        };
        let Some(stem_key) = crate::opacity::reef_module_stem(name) else {
            continue;
        };
        if seen.contains(&stem_key) && reported.insert(stem_key.clone()) {
            errors.push(CheckError::new(
                CheckErrorKind::DuplicateModule,
                format!(
                    "module `{stem_key}` is opened by both a lexical wrapper and a \
                     reef-stem mangled name in this check unit; a named module may be \
                     opened at most once"
                ),
                vec![format!(
                    "rename the mangled declaration; the `{stem_key}` lexical module \
                     already exists"
                )],
            ));
        }
    }
}

/// RFC v5 (RT-1 F2 bypass): the reef package-linker's internal-name
/// format (`Pkg__<pkg>__<Module>__<Name>` / lowercase twin) is the
/// linker's PRIVATE output. A program NOT produced by the linker
/// (raw `.ch` or raw `.dp`) that uses it forges module identity
/// through the reef-stem channel, so any top-level declaration whose
/// binding name matches the format is a declaration error. Skipped
/// entirely when the linked-program provenance flag is set (the
/// linker's own output is accepted). The linker also re-mangles every
/// user source name, so user code inside a real package cannot smuggle
/// a clean mangled name into linked output.
fn detect_forged_linker_names(exprs: &[deep::Expr], errors: &mut Vec<CheckError>) {
    if crate::opacity::linked_program() {
        return;
    }
    for expr in top_level_decl_items(exprs) {
        let deep::Expr::List(list, _) = expr else {
            continue;
        };
        if !matches!(
            get_tag(list),
            Some("deftype" | "def" | "defsig" | "typealias" | "defmacro")
        ) {
            continue;
        }
        if let Some(name) = children(list).first().and_then(symbol_name)
            && crate::opacity::is_linker_format_name(name)
        {
            errors.push(crate::opacity::forged_linker_name_error(name));
        }
    }
}

fn infer_signature_metadata(
    exprs: &[deep::Expr],
    type_env: &HashMap<String, deep::Expr>,
) -> SignatureInferenceMetadata {
    infer_signature_metadata_with_context(exprs, type_env, &SignatureInferenceMetadata::default())
}

fn infer_signature_metadata_with_context(
    exprs: &[deep::Expr],
    type_env: &HashMap<String, deep::Expr>,
    signature_context: &SignatureInferenceMetadata,
) -> SignatureInferenceMetadata {
    let defsig_names = collect_defsig_names(exprs);
    let recursive_members = recursive_call_cycle_members(exprs);
    let mut functions = BTreeMap::new();
    let ordered_defs = signature_inference_def_order(exprs);
    let passes = ordered_defs.len().max(1);
    let imported_signatures = signature_context
        .functions
        .iter()
        .map(|(name, inference)| (name.clone(), inference.display_signature.clone()))
        .collect::<HashMap<_, _>>();

    for _ in 0..passes {
        functions.clear();
        let mut available_signatures = imported_signatures.clone();
        for expr in &ordered_defs {
            let deep::Expr::List(list, _) = expr else {
                continue;
            };
            if get_tag(list) != Some("def") {
                continue;
            }
            let kids = children(list);
            let Some(name) = kids.first().and_then(symbol_name) else {
                continue;
            };
            let Some(fn_list) = kids.get(1).and_then(as_tagged_list_expr("fn")) else {
                continue;
            };
            let Some(checked_signature) = type_env.get(name).and_then(type_from_deep_expr) else {
                continue;
            };
            let Type::Fn(checked_args, checked_ret) = checked_signature.clone() else {
                continue;
            };
            let fn_kids = children(fn_list);
            let Some(params_expr) = fn_kids.first() else {
                continue;
            };
            let Some(body) = fn_kids.get(1) else {
                continue;
            };
            let param_infos = param_source_infos(params_expr);
            let recursive_cycle = recursive_members.contains(name);
            let all_written_by_defsig =
                defsig_names.contains(name) && param_infos.iter().all(|(_, written)| !*written);
            let mut display_args = checked_args.clone();
            let mut params = Vec::new();

            for (index, (pname, param_written)) in param_infos.iter().enumerate() {
                let Some(checked_type) = checked_args.get(index).cloned() else {
                    continue;
                };
                let written = all_written_by_defsig || *param_written;
                let can_infer = !recursive_cycle
                    && !written
                    && type_contains_tensor(&checked_type)
                    && !matches!(checked_type, Type::Ref(_));
                let inferred_read_only = can_infer
                    && !param_has_consuming_use(body, pname, &available_signatures, type_env);
                let display_type = if inferred_read_only {
                    Type::Ref(Box::new(checked_type.clone()))
                } else {
                    checked_type.clone()
                };
                if let Some(slot) = display_args.get_mut(index) {
                    *slot = display_type.clone();
                }
                params.push(ParamSignatureInference {
                    index,
                    name: pname.clone(),
                    written,
                    inferred_read_only,
                    checked_type,
                    display_type,
                });
            }

            let display_signature = Type::Fn(display_args, checked_ret);
            available_signatures.insert(name.to_string(), display_signature.clone());
            functions.insert(
                name.to_string(),
                FunctionSignatureInference {
                    name: name.to_string(),
                    recursive_cycle,
                    checked_signature,
                    display_signature,
                    params,
                },
            );
        }
    }

    SignatureInferenceMetadata { functions }
}

fn signature_inference_def_order(exprs: &[deep::Expr]) -> Vec<&deep::Expr> {
    let def_items = top_level_decl_items(exprs)
        .into_iter()
        .filter_map(|expr| {
            let deep::Expr::List(list, _) = expr else {
                return None;
            };
            if get_tag(list) != Some("def") {
                return None;
            }
            let name = children(list).first().and_then(symbol_name)?;
            children(list).get(1).and_then(as_tagged_list_expr("fn"))?;
            Some((name.to_string(), expr))
        })
        .collect::<Vec<_>>();
    let def_names = def_items
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<HashSet<_>>();
    let def_by_name = def_items
        .iter()
        .map(|(name, expr)| (name.clone(), *expr))
        .collect::<HashMap<_, _>>();
    let mut graph = HashMap::<String, HashSet<String>>::new();
    for (name, expr) in &def_items {
        let deep::Expr::List(list, _) = expr else {
            continue;
        };
        let Some(fn_list) = children(list).get(1).and_then(as_tagged_list_expr("fn")) else {
            continue;
        };
        let fn_kids = children(fn_list);
        let (Some(params), Some(body)) = (fn_kids.first(), fn_kids.get(1)) else {
            continue;
        };
        let mut bound = vec![
            param_source_infos(params)
                .into_iter()
                .map(|(n, _)| n)
                .collect(),
        ];
        let mut calls = HashSet::new();
        collect_top_level_calls(body, &def_names, &mut bound, &mut calls);
        graph.insert(name.clone(), calls);
    }

    fn visit<'a>(
        name: &str,
        graph: &HashMap<String, HashSet<String>>,
        def_by_name: &HashMap<String, &'a deep::Expr>,
        visiting: &mut HashSet<String>,
        visited: &mut HashSet<String>,
        out: &mut Vec<&'a deep::Expr>,
    ) {
        if visited.contains(name) || !visiting.insert(name.to_string()) {
            return;
        }
        if let Some(callees) = graph.get(name) {
            let mut callees = callees.iter().collect::<Vec<_>>();
            callees.sort();
            for callee in callees {
                visit(callee, graph, def_by_name, visiting, visited, out);
            }
        }
        visiting.remove(name);
        visited.insert(name.to_string());
        if let Some(expr) = def_by_name.get(name) {
            out.push(*expr);
        }
    }

    let mut out = Vec::new();
    let mut visiting = HashSet::new();
    let mut visited = HashSet::new();
    for (name, _) in &def_items {
        visit(
            name,
            &graph,
            &def_by_name,
            &mut visiting,
            &mut visited,
            &mut out,
        );
    }
    out
}

fn collect_defsig_names(exprs: &[deep::Expr]) -> HashSet<String> {
    let mut names = HashSet::new();
    for expr in top_level_decl_items(exprs) {
        if let deep::Expr::List(list, _) = expr
            && get_tag(list) == Some("defsig")
            && let Some(name) = children(list).first().and_then(symbol_name)
        {
            names.insert(name.to_string());
        }
    }
    names
}

fn recursive_call_cycle_members(exprs: &[deep::Expr]) -> HashSet<String> {
    let mut def_names = HashSet::new();
    let mut graph: HashMap<String, HashSet<String>> = HashMap::new();

    for expr in top_level_decl_items(exprs) {
        if let deep::Expr::List(list, _) = expr
            && get_tag(list) == Some("def")
            && let Some(name) = children(list).first().and_then(symbol_name)
            && children(list)
                .get(1)
                .and_then(as_tagged_list_expr("fn"))
                .is_some()
        {
            def_names.insert(name.to_string());
        }
    }

    for expr in top_level_decl_items(exprs) {
        let deep::Expr::List(list, _) = expr else {
            continue;
        };
        if get_tag(list) != Some("def") {
            continue;
        }
        let kids = children(list);
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        let Some(fn_list) = kids.get(1).and_then(as_tagged_list_expr("fn")) else {
            continue;
        };
        let fn_kids = children(fn_list);
        let Some(params) = fn_kids.first() else {
            continue;
        };
        let Some(body) = fn_kids.get(1) else {
            continue;
        };
        let mut bound = vec![
            param_source_infos(params)
                .into_iter()
                .map(|(n, _)| n)
                .collect(),
        ];
        let mut calls = HashSet::new();
        collect_top_level_calls(body, &def_names, &mut bound, &mut calls);
        graph.insert(name.to_string(), calls);
    }

    let mut recursive = HashSet::new();
    for name in &def_names {
        let mut visited = HashSet::new();
        if reaches_name(name, name, &graph, &mut visited) {
            recursive.insert(name.clone());
        }
    }
    recursive
}

fn reaches_name(
    start: &str,
    current: &str,
    graph: &HashMap<String, HashSet<String>>,
    visited: &mut HashSet<String>,
) -> bool {
    let Some(nexts) = graph.get(current) else {
        return false;
    };
    for next in nexts {
        if next == start {
            return true;
        }
        if visited.insert(next.clone()) && reaches_name(start, next, graph, visited) {
            return true;
        }
    }
    false
}

fn collect_top_level_calls(
    expr: &deep::Expr,
    def_names: &HashSet<String>,
    bound: &mut Vec<HashSet<String>>,
    calls: &mut HashSet<String>,
) {
    // Bail before this walker's own unbounded recursion exhausts the native
    // stack on a deeply-nested `app` body. This pass accumulates into
    // `calls`/`bound` and carries no error vector, so it cannot push a
    // diagnostic itself; the guard records the bail in `STACK_EXHAUSTED` so
    // the check entry boundary still turns it into a hard located failure
    // (never a silent partial collection). See `STACK_RED_ZONE_BYTES`.
    stack_guard!("collect_top_level_calls", expr);
    match expr {
        deep::Expr::Atom(_, _) => {}
        deep::Expr::Map(map, _) => {
            for (_, value) in &map.entries {
                collect_top_level_calls(value, def_names, bound, calls);
            }
        }
        deep::Expr::MetaExpr(meta, _) => {
            collect_top_level_calls(&meta.expr, def_names, bound, calls)
        }
        deep::Expr::List(list, _) => match get_tag(list) {
            Some("app") => {
                let kids = children(list);
                if let Some(callee) = kids.first().and_then(var_name_expr)
                    && def_names.contains(callee)
                    && !is_bound_name(callee, bound)
                {
                    calls.insert(callee.to_string());
                }
                for child in kids {
                    collect_top_level_calls(child, def_names, bound, calls);
                }
            }
            Some("fn") => {
                let kids = children(list);
                if kids.len() >= 2 {
                    bound.push(
                        param_source_infos(&kids[0])
                            .into_iter()
                            .map(|(n, _)| n)
                            .collect(),
                    );
                    collect_top_level_calls(&kids[1], def_names, bound, calls);
                    bound.pop();
                }
            }
            Some("let") => {
                let kids = children(list);
                if kids.len() < 2 {
                    return;
                }
                let mut let_names = HashSet::new();
                if let Some(bind_list) = kids.first().and_then(as_tagged_list_expr("bind")) {
                    let bind_kids = children(bind_list);
                    let mut index = 0;
                    while index + 1 < bind_kids.len() {
                        collect_top_level_calls(&bind_kids[index + 1], def_names, bound, calls);
                        if let Some(name) = symbol_name(&bind_kids[index]) {
                            let_names.insert(name.to_string());
                        }
                        index += 2;
                    }
                }
                bound.push(let_names);
                collect_top_level_calls(&kids[1], def_names, bound, calls);
                bound.pop();
            }
            Some("match") => {
                let kids = children(list);
                if let Some(scrutinee) = kids.first() {
                    collect_top_level_calls(scrutinee, def_names, bound, calls);
                }
                for arm in kids.iter().skip(1) {
                    let Some(arm_list) = as_tagged_list_expr("arm")(arm) else {
                        continue;
                    };
                    let arm_kids = children(arm_list);
                    if arm_kids.len() < 3 {
                        continue;
                    }
                    bound.push(pattern_names_for_signature(&arm_kids[0]));
                    collect_top_level_calls(&arm_kids[1], def_names, bound, calls);
                    collect_top_level_calls(&arm_kids[2], def_names, bound, calls);
                    bound.pop();
                }
            }
            _ => {
                for child in children(list) {
                    collect_top_level_calls(child, def_names, bound, calls);
                }
            }
        },
    }
}

pub(crate) fn param_has_consuming_use(
    expr: &deep::Expr,
    param: &str,
    available_signatures: &HashMap<String, Type>,
    type_env: &HashMap<String, deep::Expr>,
) -> bool {
    let mut bound = Vec::new();
    param_has_consuming_use_inner(expr, param, &mut bound, available_signatures, type_env)
}

fn param_has_consuming_use_inner(
    expr: &deep::Expr,
    param: &str,
    bound: &mut Vec<HashSet<String>>,
    available_signatures: &HashMap<String, Type>,
    type_env: &HashMap<String, deep::Expr>,
) -> bool {
    stack_guard!("param_has_consuming_use_inner", expr, false);
    match expr {
        deep::Expr::Atom(_, _) => false,
        deep::Expr::Map(map, _) => map.entries.iter().any(|(_, value)| {
            param_has_consuming_use_inner(value, param, bound, available_signatures, type_env)
        }),
        deep::Expr::MetaExpr(meta, _) => {
            param_has_consuming_use_inner(&meta.expr, param, bound, available_signatures, type_env)
        }
        deep::Expr::List(list, _) => match get_tag(list) {
            Some("var") => var_name_list(list) == Some(param) && !is_bound_name(param, bound),
            Some("borrow") | Some("copy") => children(list).first().is_some_and(|child| {
                param_nested_consuming_use(child, param, bound, available_signatures, type_env)
            }),
            Some("drop") | Some("realize") => children(list)
                .first()
                .is_some_and(|child| expr_mentions_unshadowed_name(child, param, bound)),
            Some("app") => app_consumes_param(list, param, bound, available_signatures, type_env),
            Some("pipe") => pipe_consumes_param(list, param, bound, available_signatures, type_env),
            Some("fn") => {
                let kids = children(list);
                if kids.len() < 2 {
                    return false;
                }
                if expr_mentions_unshadowed_name(&kids[1], param, bound) {
                    return true;
                }
                false
            }
            Some("let") => {
                let kids = children(list);
                if kids.len() < 2 {
                    return false;
                }
                let mut let_names = HashSet::new();
                if let Some(bind_list) = kids.first().and_then(as_tagged_list_expr("bind")) {
                    let bind_kids = children(bind_list);
                    let mut index = 0;
                    while index + 1 < bind_kids.len() {
                        if param_has_consuming_use_inner(
                            &bind_kids[index + 1],
                            param,
                            bound,
                            available_signatures,
                            type_env,
                        ) {
                            return true;
                        }
                        if let Some(name) = symbol_name(&bind_kids[index]) {
                            let_names.insert(name.to_string());
                        }
                        index += 2;
                    }
                }
                bound.push(let_names);
                let result = param_has_consuming_use_inner(
                    &kids[1],
                    param,
                    bound,
                    available_signatures,
                    type_env,
                );
                bound.pop();
                result
            }
            Some("match") => {
                let kids = children(list);
                if kids
                    .first()
                    .is_some_and(|scrutinee| expr_mentions_unshadowed_name(scrutinee, param, bound))
                {
                    return true;
                }
                for arm in kids.iter().skip(1) {
                    let Some(arm_list) = as_tagged_list_expr("arm")(arm) else {
                        continue;
                    };
                    let arm_kids = children(arm_list);
                    if arm_kids.len() < 3 {
                        continue;
                    }
                    bound.push(pattern_names_for_signature(&arm_kids[0]));
                    let consumes = param_has_consuming_use_inner(
                        &arm_kids[1],
                        param,
                        bound,
                        available_signatures,
                        type_env,
                    ) || param_has_consuming_use_inner(
                        &arm_kids[2],
                        param,
                        bound,
                        available_signatures,
                        type_env,
                    );
                    bound.pop();
                    if consumes {
                        return true;
                    }
                }
                false
            }
            _ => children(list).iter().any(|child| {
                param_has_consuming_use_inner(child, param, bound, available_signatures, type_env)
            }),
        },
    }
}

fn param_nested_consuming_use(
    expr: &deep::Expr,
    param: &str,
    bound: &mut Vec<HashSet<String>>,
    available_signatures: &HashMap<String, Type>,
    type_env: &HashMap<String, deep::Expr>,
) -> bool {
    if is_direct_unshadowed_var(expr, param, bound) {
        return false;
    }
    param_has_consuming_use_inner(expr, param, bound, available_signatures, type_env)
}

fn app_consumes_param(
    list: &deep::List,
    param: &str,
    bound: &mut Vec<HashSet<String>>,
    available_signatures: &HashMap<String, Type>,
    type_env: &HashMap<String, deep::Expr>,
) -> bool {
    let kids = children(list);
    let callee = kids.first().and_then(var_name_expr);
    if let Some(func) = kids.first()
        && !matches!(callee, Some(name) if name != param)
        && param_has_consuming_use_inner(func, param, bound, available_signatures, type_env)
    {
        return true;
    }
    for (index, arg) in kids.iter().skip(1).enumerate() {
        if borrow_inner_for_signature(arg)
            .is_some_and(|inner| is_direct_unshadowed_var(inner, param, bound))
        {
            continue;
        }
        if is_direct_unshadowed_var(arg, param, bound) {
            if callee_arg_is_borrowed(callee, index, available_signatures, type_env) {
                continue;
            }
            return true;
        }
        if param_has_consuming_use_inner(arg, param, bound, available_signatures, type_env) {
            return true;
        }
    }
    false
}

fn pipe_consumes_param(
    list: &deep::List,
    param: &str,
    bound: &mut Vec<HashSet<String>>,
    available_signatures: &HashMap<String, Type>,
    type_env: &HashMap<String, deep::Expr>,
) -> bool {
    let kids = children(list);
    if kids.is_empty() {
        return false;
    }
    let mut current = &kids[0];
    for stage in &kids[1..] {
        // Issue #229 (sibling sweep of chelis#226): peer through any
        // synthesized `__chelis_pipe` lambda the desugarer emits for
        // explicit-arg pipe stages so the borrow-arg classifier sees
        // the inner callee and the piped value's actual arg position
        // — not the lambda's type. Without this peering, every
        // non-bare-var pipe stage is mis-classified as a consuming
        // use, the wrapping function never gets auto-borrow inferred,
        // and downstream calls spuriously consume their argument.
        let (_callee_expr, callee_builtin, piped_arg_index) =
            crate::pipe_stage::resolve_pipe_stage_callee(stage);
        if is_direct_unshadowed_var(current, param, bound) {
            if !callee_arg_is_borrowed(
                callee_builtin,
                piped_arg_index,
                available_signatures,
                type_env,
            ) {
                return true;
            }
        } else if param_has_consuming_use_inner(
            current,
            param,
            bound,
            available_signatures,
            type_env,
        ) {
            return true;
        }
        current = stage;
    }
    false
}

fn callee_arg_is_borrowed(
    callee: Option<&str>,
    index: usize,
    available_signatures: &HashMap<String, Type>,
    type_env: &HashMap<String, deep::Expr>,
) -> bool {
    let Some(callee) = callee else {
        return false;
    };
    if let Some(Type::Fn(args, _)) = available_signatures.get(callee)
        && args.get(index).is_some_and(|ty| matches!(ty, Type::Ref(_)))
    {
        return true;
    }
    if let Some(Type::Fn(args, _)) = type_env.get(callee).and_then(type_from_deep_expr)
        && args.get(index).is_some_and(|ty| matches!(ty, Type::Ref(_)))
    {
        return true;
    }
    builtin_arg_is_ref(callee, index)
}

fn builtin_arg_is_ref(name: &str, index: usize) -> bool {
    let (env, _) = builtins::builtin_env();
    if let Some(Type::Fn(args, _)) = env.lookup(name).map(|scheme| &scheme.body) {
        return args.get(index).is_some_and(|ty| matches!(ty, Type::Ref(_)));
    }
    false
}

fn expr_mentions_unshadowed_name(
    expr: &deep::Expr,
    name: &str,
    bound: &mut Vec<HashSet<String>>,
) -> bool {
    stack_guard!("expr_mentions_unshadowed_name", expr, false);
    match expr {
        deep::Expr::Atom(_, _) => false,
        deep::Expr::Map(map, _) => map
            .entries
            .iter()
            .any(|(_, value)| expr_mentions_unshadowed_name(value, name, bound)),
        deep::Expr::MetaExpr(meta, _) => expr_mentions_unshadowed_name(&meta.expr, name, bound),
        deep::Expr::List(list, _) => match get_tag(list) {
            Some("var") => var_name_list(list) == Some(name) && !is_bound_name(name, bound),
            Some("fn") => {
                let kids = children(list);
                if kids.len() < 2 {
                    return false;
                }
                bound.push(
                    param_source_infos(&kids[0])
                        .into_iter()
                        .map(|(n, _)| n)
                        .collect(),
                );
                let result = expr_mentions_unshadowed_name(&kids[1], name, bound);
                bound.pop();
                result
            }
            _ => children(list)
                .iter()
                .any(|child| expr_mentions_unshadowed_name(child, name, bound)),
        },
    }
}

fn type_from_deep_expr(expr: &deep::Expr) -> Option<Type> {
    let mut vg = VarGen::default();
    let mut tvar_map = HashMap::new();
    let ty = deep_type_to_type(expr, &mut vg, &mut tvar_map);
    (!matches!(ty, Type::Error)).then_some(ty)
}

fn type_contains_tensor(ty: &Type) -> bool {
    match ty {
        Type::Tensor(_, _) => true,
        Type::Ref(inner) => type_contains_tensor(inner),
        Type::Adt(_, args) | Type::Tuple(args) => args.iter().any(type_contains_tensor),
        Type::Fn(_, _) | Type::Prim(_) | Type::Var(_) | Type::Unit | Type::Error => false,
    }
}

/// Issue #256 round 3: does `ty` carry a tensor, consulting `carriers`
/// for the by-name ADT carry decision? This is the `Type`-level mirror
/// of linearity's `type_expr_contains_tensor`: an ADT carries iff its
/// name is in the precomputed carrier set (its definition has a
/// tensor-carrying field) OR one of its type arguments carries (e.g.
/// `Wrapper[tensor[..]]`). Bare `type_contains_tensor` cannot make the
/// by-name decision — it only sees the `Type::Adt` shell, not the
/// variant fields — which is exactly why the deferred-borrow gate must
/// be handed the carrier set rather than trust an args-only check.
fn type_carries_tensor_with_carriers(ty: &Type, carriers: &HashSet<String>) -> bool {
    match ty {
        Type::Tensor(_, _) => true,
        Type::Ref(inner) => type_carries_tensor_with_carriers(inner, carriers),
        Type::Tuple(args) => args
            .iter()
            .any(|a| type_carries_tensor_with_carriers(a, carriers)),
        Type::Adt(name, args) => {
            carriers.contains(name)
                || args
                    .iter()
                    .any(|a| type_carries_tensor_with_carriers(a, carriers))
        }
        Type::Fn(_, _) | Type::Prim(_) | Type::Var(_) | Type::Unit | Type::Error => false,
    }
}

/// Issue #256 round 3: compute the set of tensor-carrying ADT names from
/// the registry. This is the registry-backed mirror of linearity's
/// `compute_tensor_carrying_adts` (which works off stamped Deep exprs):
/// fixed-point iteration where an ADT joins the carrier set once any of
/// its variant fields carries a tensor against the in-progress set, so a
/// chain `A { f: B }, B { g: tensor }` resolves transitively. Bounded by
/// the ADT count. The two classifiers must agree: the gate uses this set
/// to reject a deferred borrow that resolved to a non-carrying ADT, and
/// linearity uses its own set to reject the concrete (non-deferred) form.
fn adt_carrier_set(adt_reg: &AdtRegistry) -> HashSet<String> {
    let mut carriers: HashSet<String> = HashSet::new();
    loop {
        let mut grew = false;
        for (name, def) in &adt_reg.defs {
            if carriers.contains(name) {
                continue;
            }
            let carries = def.variants.iter().any(|variant| {
                variant
                    .fields
                    .iter()
                    .any(|(_, field_ty)| type_carries_tensor_with_carriers(field_ty, &carriers))
            });
            if carries {
                carriers.insert(name.clone());
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    carriers
}

/// Issue #256 round 2 soundness gate. The `borrow` inference arm accepts a
/// borrow whose inner type is still an unresolved `Type::Var`, recording
/// the variable in the substitution's deferred-borrow ledger. That
/// deferral is sound only when the variable is *eventually* pinned to a
/// tensor or tensor-carrying type by a later unification (the surrounding
/// `&tensor[..]` / `&Carrier[..]` parameter). This pass drains the ledger
/// after a def body's inference completes and re-checks each recorded
/// variable against the now-complete substitution:
///
///   - `Tensor` / `Ref(Tensor)`: pinned to a tensor — sound, accept.
///   - `Adt` / `Tuple` / `Ref(Adt|Tuple)`: an aggregate that *may* carry a
///     tensor. Round 3 (#256 soundness): classify it here against the
///     registry-backed carrier set rather than blanket-accepting and
///     deferring to linearity. Deferring was unsound — round 1 loosened
///     linearity's `expr_is_owned_or_borrow_linear` to accept a stale
///     `(t-var ..)` stamp (so a tensor that resolved late is not
///     rejected), and a deferred borrow that resolves to a *non*-carrying
///     ADT/tuple keeps that same `(t-var ..)` stamp at the linearity
///     layer. Both gates would then wave it through. So the gate, which
///     already holds the final `Type`, must make the carry decision: a
///     tensor-carrying aggregate is accepted, a non-carrying one rejected.
///   - still `Var`: never pinned. A fully-polymorphic consumer (e.g.
///     `consume_any[a](t: a)`) unifies the parameter to `&a` without ever
///     forcing a tensor, so a genuinely-non-tensor value would slip past
///     every other gate. Reject.
///   - `Prim` / `Unit` / `Fn`: pinned to a concretely-non-tensor scalar
///     only after the borrow arm ran (so the arm's own `_ => TypeMismatch`
///     could not fire). Reject.
fn validate_deferred_borrow_vars(
    subst: &Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
) {
    let deferred = subst.take_deferred_borrow_vars();
    if deferred.is_empty() {
        return;
    }
    // Computed lazily: only programs that actually deferred a borrow pay
    // the fixed-point pass, and only once per drain.
    let carriers = adt_carrier_set(adt_reg);
    for tv in deferred {
        let resolved = subst.apply(&Type::Var(tv));
        // Peel every `Ref` layer: the recorded variable is the borrow
        // inner, but a later unification may have wrapped it in one or
        // more `&` layers (e.g. the parameter type was itself `&T`).
        let mut peeled = &resolved;
        while let Type::Ref(inner) = peeled {
            peeled = inner.as_ref();
        }
        let sound = match peeled {
            // Pinned to a tensor: always sound.
            Type::Tensor(_, _) => true,
            // Pinned to an aggregate: sound iff it actually carries a
            // tensor against the registry carrier set (round 3). A
            // non-carrying record/tuple resolved through the deferred
            // path is rejected here — linearity's loosened classifier
            // can no longer be relied on to catch it.
            Type::Adt(_, _) | Type::Tuple(_) => {
                type_carries_tensor_with_carriers(peeled, &carriers)
            }
            // Don't double-report an inner that already failed inference.
            Type::Error => true,
            // Never pinned, or pinned to a concretely-non-tensor value.
            Type::Var(_) | Type::Prim(_) | Type::Unit | Type::Fn(_, _) => false,
            // `Ref` is fully peeled above; treat as sound to avoid a
            // spurious reject if a future shape reaches here.
            Type::Ref(_) => true,
        };
        if !sound {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("borrow requires tensor or tensor-carrying input, got {peeled}"),
                vec!["Use `&x` only with tensor values".to_string()],
            ));
        }
    }
}

/// RFC D-CHECK: drain the deferred-access ledger after a def body's
/// inference completes and re-check each recorded target variable
/// against the final substitution: a target pinned to an
/// out-of-module opaque ADT (e.g. an unannotated lambda parameter
/// pinned by a later call) is rejected with the same action text as
/// the typed path. Draining per def keeps attribution exact and
/// prevents one def's deferrals from leaking into the next,
/// mirroring `validate_deferred_borrow_vars`.
fn validate_deferred_opaque_uses(
    subst: &Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
) {
    let mut any_opaque_in_scope: Option<bool> = None;
    for (tv, use_kind) in subst.take_deferred_opaque_uses() {
        let resolved = subst.apply(&Type::Var(tv));
        let mut peeled = &resolved;
        while let Type::Ref(inner) = peeled {
            peeled = inner.as_ref();
        }
        let action = match use_kind {
            crate::unify::DeferredOpaqueUse::Access => crate::opacity::OpaqueAction::FieldAccess,
            crate::unify::DeferredOpaqueUse::RecordUpdate => {
                crate::opacity::OpaqueAction::RecordUpdate
            }
        };
        match peeled {
            Type::Adt(adt_name, _) => {
                crate::opacity::check_opaque_use(action, adt_name, adt_reg, errors);
            }
            // Never pinned: let-generalization makes an unannotated
            // accessor polymorphic, so callers instantiate FRESH
            // variables and the recorded one stays unbound -- a
            // laundering channel for opaque values. Mirror the
            // deferred-borrow ledger's never-pinned rejection,
            // fail-closed, scoped to check units that declare any
            // opaque type so opaque-free programs keep the lenient
            // status quo.
            Type::Var(_) => {
                let opaque_in_scope = *any_opaque_in_scope
                    .get_or_insert_with(|| adt_reg.defs.values().any(|def| def.opaque));
                if opaque_in_scope
                    && let Some(error) = crate::opacity::with_context(|ctx| {
                        crate::opacity::unresolved_target_error(ctx, action)
                    })
                {
                    errors.push(error);
                }
            }
            _ => {}
        }
    }
}

fn param_source_infos(expr: &deep::Expr) -> Vec<(String, bool)> {
    let Some(list) = as_tagged_list_expr("params")(expr) else {
        return Vec::new();
    };
    children(list)
        .iter()
        .filter_map(|param| match param {
            deep::Expr::Atom(deep::Atom::Symbol(name), _) => Some((name.clone(), false)),
            deep::Expr::MetaExpr(meta, _) => {
                let deep::Expr::Atom(deep::Atom::Symbol(name), _) = meta.expr.as_ref() else {
                    return None;
                };
                Some((
                    name.clone(),
                    meta.entries.iter().any(|(key, _)| key == "type"),
                ))
            }
            deep::Expr::List(param_list, _) => {
                let name = param_list.elements.first().and_then(symbol_name)?;
                let written = get_meta(param_list)
                    .is_some_and(|meta| meta.entries.iter().any(|(key, _)| key == "type"));
                Some((name.to_string(), written))
            }
            _ => None,
        })
        .collect()
}

fn pattern_names_for_signature(expr: &deep::Expr) -> HashSet<String> {
    let mut names = HashSet::new();
    collect_pattern_names_for_signature(expr, &mut names);
    names
}

fn collect_pattern_names_for_signature(expr: &deep::Expr, names: &mut HashSet<String>) {
    // Bail before unbounded recursion exhausts the native stack on a
    // deeply-nested pattern. No error vector here; the guard records the bail
    // so the check entry boundary fails hard with a located diagnostic. See
    // `STACK_RED_ZONE_BYTES`.
    stack_guard!("collect_pattern_names_for_signature", expr);
    let deep::Expr::List(list, _) = expr else {
        return;
    };
    match get_tag(list) {
        Some("pat-var") => {
            if let Some(name) = children(list).first().and_then(symbol_name) {
                names.insert(name.to_string());
            }
        }
        Some("pat-as") => {
            let kids = children(list);
            if let Some(name) = kids.first().and_then(symbol_name) {
                names.insert(name.to_string());
            }
            if let Some(inner) = kids.get(1) {
                collect_pattern_names_for_signature(inner, names);
            }
        }
        _ => {
            for child in children(list) {
                collect_pattern_names_for_signature(child, names);
            }
        }
    }
}

fn as_tagged_list_expr(tag: &'static str) -> impl Fn(&deep::Expr) -> Option<&deep::List> {
    move |expr| match expr {
        deep::Expr::List(list, _) if get_tag(list) == Some(tag) => Some(list),
        _ => None,
    }
}

fn var_name_expr(expr: &deep::Expr) -> Option<&str> {
    let deep::Expr::List(list, _) = expr else {
        return None;
    };
    var_name_list(list)
}

fn var_name_list(list: &deep::List) -> Option<&str> {
    if get_tag(list) != Some("var") {
        return None;
    }
    children(list).first().and_then(symbol_name)
}

fn borrow_inner_for_signature(expr: &deep::Expr) -> Option<&deep::Expr> {
    let deep::Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("borrow") {
        return None;
    }
    children(list).first()
}

fn is_direct_unshadowed_var(expr: &deep::Expr, name: &str, bound: &[HashSet<String>]) -> bool {
    var_name_expr(expr) == Some(name) && !is_bound_name(name, bound)
}

fn is_bound_name(name: &str, bound: &[HashSet<String>]) -> bool {
    bound.iter().rev().any(|scope| scope.contains(name))
}

/// Detect cycles among top-level `def` bindings.
///
/// The Nautilus external-input pattern `x = (x : tensor[...])` is permitted —
/// a self-loop (the def body references the same name it is binding, with no
/// intermediate hops) is treated as a declaration of an external input, not as
/// a cycle. Any cycle of length >= 2 (e.g. `a -> b -> a`, `a -> b -> c -> a`)
/// is a real binding cycle and is reported as a `CycleDetected` error.
fn detect_top_level_binding_cycles(exprs: &[deep::Expr], errors: &mut Vec<CheckError>) {
    let mut def_names: Vec<String> = Vec::new();
    let mut def_name_set: HashSet<String> = HashSet::new();
    let mut def_bodies: HashMap<String, &deep::Expr> = HashMap::new();
    // Descend through `(module {} name ...)` wrappers so this check works
    // on idiomatic Surf sources (every `.ch` file starts with `module X`,
    // which desugars to a single top-level `module` list wrapping every
    // declaration). Without this, the cycle check is a no-op in practice.
    for expr in top_level_decl_items(exprs) {
        if let deep::Expr::List(list, _) = expr
            && get_tag(list) == Some("def")
        {
            let kids = children(list);
            let Some(name) = kids.first().and_then(symbol_name) else {
                continue;
            };
            let Some(body) = kids.get(1) else { continue };
            if !def_name_set.contains(name) {
                def_name_set.insert(name.to_string());
                def_names.push(name.to_string());
            }
            def_bodies.insert(name.to_string(), body);
        }
    }

    // Phase 1: per-def, collect both the vars referenced EAGERLY (outside fn
    // bodies) and the top-level fns APPLIED eagerly. Lazy refs inside fn
    // bodies are captured separately so we can chain them in on demand.
    let mut direct_refs: HashMap<String, HashSet<String>> = HashMap::new();
    let mut applied_fns: HashMap<String, HashSet<String>> = HashMap::new();
    let mut fn_body_refs: HashMap<String, (HashSet<String>, HashSet<String>)> = HashMap::new();
    for name in &def_names {
        let Some(body) = def_bodies.get(name) else {
            continue;
        };
        let mut refs: HashSet<String> = HashSet::new();
        let mut applied: HashSet<String> = HashSet::new();
        let mut bound: HashSet<String> = HashSet::new();
        collect_eager_refs(body, &mut bound, &mut refs, &mut applied);
        direct_refs.insert(name.clone(), refs);
        applied_fns.insert(name.clone(), applied);
        // If this def's body IS itself a fn, also collect what its body
        // references so callers of this def can chain.
        if let deep::Expr::List(list, _) = body
            && get_tag(list) == Some("fn")
            && let Some(fn_body) = children(list).get(1)
        {
            let mut inner_refs: HashSet<String> = HashSet::new();
            let mut inner_applied: HashSet<String> = HashSet::new();
            let mut inner_bound: HashSet<String> = HashSet::new();
            // Bind the fn's own params so they aren't flagged as refs.
            if let Some(params_list) = children(list).first()
                && let deep::Expr::List(params, _) = params_list
                && get_tag(params) == Some("params")
            {
                for param in children(params) {
                    if let Some(pname) = param_name_for_refs(param) {
                        inner_bound.insert(pname);
                    }
                }
            }
            collect_eager_refs(
                fn_body,
                &mut inner_bound,
                &mut inner_refs,
                &mut inner_applied,
            );
            fn_body_refs.insert(name.clone(), (inner_refs, inner_applied));
        }
    }

    // Phase 2: build two edge sets per def.
    //   - value_edges[d]: names read AS VALUES in d's body (i.e. `(var x)`
    //     where x is not a fn callee). Reading a value requires that value
    //     to already be bound — a cycle here is a real binding cycle.
    //   - call_edges[d]: names d CALLS (`(app (var f) ...)`). Calling a fn
    //     pushes its body into eager evaluation but does NOT require f's
    //     value — f is a callable, not a scalar. Recursive calls with base
    //     cases terminate and don't close a cycle.
    //
    // Cycle condition: DFS from each VALUE def X, traversing both edge
    // kinds transitively. Track the stack of VALUE defs we're currently
    // evaluating. If a value-edge lands on a stack member, that's a real
    // binding cycle. Fn names aren't pushed onto the stack — they are
    // intermediates in the path.
    let mut value_edges: HashMap<String, Vec<String>> = HashMap::new();
    let mut call_edges: HashMap<String, Vec<String>> = HashMap::new();
    for name in &def_names {
        let body = def_bodies.get(name).copied();
        let is_nautilus_literal_self = body.is_some_and(|b| body_is_literal_self_ref(b, name));
        let body_is_fn = matches!(
            body,
            Some(deep::Expr::List(list, _)) if get_tag(list) == Some("fn")
        );

        let (raw_refs, raw_applied) = if body_is_fn {
            let empty_refs: HashSet<String> = HashSet::new();
            let empty_applied: HashSet<String> = HashSet::new();
            fn_body_refs
                .get(name)
                .map(|(r, a)| (r.clone(), a.clone()))
                .unwrap_or((empty_refs, empty_applied))
        } else {
            (
                direct_refs.get(name).cloned().unwrap_or_default(),
                applied_fns.get(name).cloned().unwrap_or_default(),
            )
        };

        let mut value_out: Vec<String> = raw_refs
            .into_iter()
            .filter(|r| {
                if r == name && is_nautilus_literal_self {
                    return false;
                }
                def_name_set.contains(r)
            })
            .collect();
        let mut call_out: Vec<String> = raw_applied
            .into_iter()
            .filter(|r| def_name_set.contains(r))
            .collect();
        value_out.sort();
        call_out.sort();
        value_edges.insert(name.clone(), value_out);
        call_edges.insert(name.clone(), call_out);
    }

    let mut reported: HashSet<Vec<String>> = HashSet::new();

    // DFS from each value def. Track:
    //   - `value_stack`: the value defs we're "currently evaluating". A
    //     value_edge landing on a member of this stack is a cycle.
    //   - `visited`: nodes we've already fully explored from some starting
    //     value def. Avoids re-walking fn bodies we've cleared.
    //   - `path`: the traversal path for error reporting (includes both
    //     values and fns as intermediates).
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Color {
        White,
        Gray,
        Black,
    }

    #[allow(clippy::too_many_arguments)]
    fn dfs(
        node: &str,
        is_value: &dyn Fn(&str) -> bool,
        value_edges: &HashMap<String, Vec<String>>,
        call_edges: &HashMap<String, Vec<String>>,
        color: &mut HashMap<String, Color>,
        value_stack: &mut Vec<String>,
        path: &mut Vec<String>,
        reported: &mut HashSet<Vec<String>>,
        errors: &mut Vec<CheckError>,
    ) {
        color.insert(node.to_string(), Color::Gray);
        let this_is_value = is_value(node);
        if this_is_value {
            value_stack.push(node.to_string());
        }
        path.push(node.to_string());

        if let Some(neighbors) = value_edges.get(node) {
            for next in neighbors {
                if let Some(start) = value_stack.iter().position(|s| s == next) {
                    // Value-edge landing on a value currently being
                    // evaluated → real binding cycle.
                    let value_seg_start = path.iter().position(|s| s == &value_stack[start]);
                    let cycle: Vec<String> = if let Some(s) = value_seg_start {
                        path[s..].to_vec()
                    } else {
                        value_stack[start..].to_vec()
                    };
                    let mut canon = cycle.clone();
                    if let Some((min_idx, _)) = canon.iter().enumerate().min_by(|a, b| a.1.cmp(b.1))
                    {
                        canon.rotate_left(min_idx);
                    }
                    if reported.insert(canon.clone()) {
                        let mut pathstr = canon.clone();
                        pathstr.push(canon[0].clone());
                        let message = format!("binding cycle: {}", pathstr.join(" -> "));
                        errors.push(CheckError::new(
                            CheckErrorKind::CycleDetected,
                            message,
                            vec![
                                "Break the cycle by removing one of the \
                                 self-referential definitions or replacing it \
                                 with a concrete value."
                                    .to_string(),
                            ],
                        ));
                    }
                } else if color.get(next).copied().unwrap_or(Color::White) == Color::White {
                    dfs(
                        next,
                        is_value,
                        value_edges,
                        call_edges,
                        color,
                        value_stack,
                        path,
                        reported,
                        errors,
                    );
                }
            }
        }

        if let Some(neighbors) = call_edges.get(node) {
            for next in neighbors {
                // Fn calls don't require the callee's VALUE — they just
                // push the callee's body into eager evaluation. Gray nodes
                // are mid-exploration (recursive reentry) — skip to avoid
                // infinite DFS.
                if color.get(next).copied().unwrap_or(Color::White) == Color::White {
                    dfs(
                        next,
                        is_value,
                        value_edges,
                        call_edges,
                        color,
                        value_stack,
                        path,
                        reported,
                        errors,
                    );
                }
            }
        }

        path.pop();
        if this_is_value {
            value_stack.pop();
        }
        color.insert(node.to_string(), Color::Black);
    }

    let is_value = |name: &str| -> bool {
        def_bodies
            .get(name)
            .map(|body| !matches!(body, deep::Expr::List(list, _) if get_tag(list) == Some("fn")))
            .unwrap_or(false)
    };
    let mut color: HashMap<String, Color> = def_names
        .iter()
        .map(|n| (n.clone(), Color::White))
        .collect();
    let mut value_stack: Vec<String> = Vec::new();
    let mut path: Vec<String> = Vec::new();
    for name in &def_names {
        if !is_value(name) {
            continue;
        }
        if color.get(name).copied() == Some(Color::White) {
            dfs(
                name,
                &is_value,
                &value_edges,
                &call_edges,
                &mut color,
                &mut value_stack,
                &mut path,
                &mut reported,
                errors,
            );
        }
    }
}

fn param_name_for_refs(param: &deep::Expr) -> Option<String> {
    stack_guard!("param_name_for_refs", param, None);
    match param {
        deep::Expr::Atom(deep::Atom::Symbol(name), _) => Some(name.clone()),
        deep::Expr::MetaExpr(meta, _) => param_name_for_refs(&meta.expr),
        // A Deep param is `(name {type: ...})` — a List with the name as
        // the FIRST element and the meta map as the second. `children()`
        // skips first two (tag + meta) and returns nothing for a 2-elem
        // list, so read elements[0] directly.
        deep::Expr::List(list, _) => list
            .elements
            .first()
            .and_then(symbol_name)
            .map(str::to_string),
        _ => None,
    }
}

/// Collect both eager references and top-level fn applications.
///
/// `refs` gets free `(var name)` references that fire at definition time
/// (i.e., NOT inside an enclosing `fn` body).
///
/// `applied` gets names of fns called as `(app (var F) ...)` at definition
/// time (again, not inside a nested fn body). Callers use `applied` to
/// chain in the called fn's own eager refs for cycle detection — this is
/// what catches top-level value cycles that route through a fn call:
///
/// ```text
/// a = f()
/// b = g()
/// def f() = b
/// def g() = a
/// ```
///
/// Plain `collect_free_var_refs` (kept below for backward compatibility)
/// ignores fn bodies entirely, which correctly permits mutual recursion
/// between fn defs never called eagerly — but misses the cycle above.
fn collect_eager_refs(
    expr: &deep::Expr,
    bound: &mut HashSet<String>,
    refs: &mut HashSet<String>,
    applied: &mut HashSet<String>,
) {
    stack_guard!("collect_eager_refs", expr);
    match expr {
        deep::Expr::List(list, _) => match get_tag(list) {
            Some("var") => {
                if let Some(name) = children(list).first().and_then(symbol_name)
                    && !bound.contains(name)
                {
                    refs.insert(name.to_string());
                }
            }
            Some("fn") => {
                // Skip fn body — only its application at this site (if any)
                // is eager; the body itself is deferred.
            }
            Some("app") => {
                let kids = children(list);
                if let Some(callee) = kids.first()
                    && let deep::Expr::List(clist, _) = callee
                    && get_tag(clist) == Some("var")
                    && let Some(fname) = children(clist).first().and_then(symbol_name)
                    && !bound.contains(fname)
                {
                    // Callee is in `applied` only — NOT in `refs`. For cycle
                    // detection, reading `g` as a value is different from
                    // calling `g()`: the former requires g's value now, the
                    // latter just pushes g's body into eager evaluation and
                    // may terminate at a base case.
                    applied.insert(fname.to_string());
                } else if let Some(callee) = kids.first() {
                    collect_eager_refs(callee, bound, refs, applied);
                }
                for arg in kids.iter().skip(1) {
                    collect_eager_refs(arg, bound, refs, applied);
                }
            }
            Some("let") => {
                let kids = children(list);
                let mut added: Vec<String> = Vec::new();
                if let Some(deep::Expr::List(bind_list, _)) = kids.first()
                    && get_tag(bind_list) == Some("bind")
                {
                    let bind_kids = children(bind_list);
                    let mut i = 0;
                    while i + 1 < bind_kids.len() {
                        collect_eager_refs(&bind_kids[i + 1], bound, refs, applied);
                        if let Some(name) = symbol_name(&bind_kids[i])
                            && bound.insert(name.to_string())
                        {
                            added.push(name.to_string());
                        }
                        i += 2;
                    }
                }
                if let Some(body) = kids.get(1) {
                    collect_eager_refs(body, bound, refs, applied);
                }
                for name in added {
                    bound.remove(&name);
                }
            }
            _ => {
                for elem in &list.elements {
                    collect_eager_refs(elem, bound, refs, applied);
                }
            }
        },
        deep::Expr::Map(map, _) => {
            for (_, v) in &map.entries {
                collect_eager_refs(v, bound, refs, applied);
            }
        }
        deep::Expr::MetaExpr(meta, _) => {
            for (_, v) in &meta.entries {
                collect_eager_refs(v, bound, refs, applied);
            }
            collect_eager_refs(&meta.expr, bound, refs, applied);
        }
        deep::Expr::Atom(_, _) => {}
    }
}

/// Walk the program's Deep AST and reject any `(t-tensor ... (t-prim {} P))`
/// whose precision P is not supported by the Phase 0f tensor backend
/// (currently: f16, bf16, f64, f8e4m3, string).
///
/// This runs after HM inference so it catches user-written tensor type
/// ascriptions, defsig tensor types, parameter type annotations, literal
/// type metadata, and any cast target that produces a tensor with an
/// unsupported element precision.
fn validate_tensor_precisions_in_program(exprs: &[deep::Expr], errors: &mut Vec<CheckError>) {
    let mut seen: HashSet<(String, String)> = HashSet::new();
    // Descend through `(module {} name ...)` wrappers so per-def dedup
    // keeps each def's tensor types in their own key space (otherwise
    // every def lives under def_context="" and errors collapse).
    for expr in top_level_decl_items(exprs) {
        let def_name = match expr {
            deep::Expr::List(list, _)
                if matches!(
                    get_tag(list),
                    Some("def") | Some("defsig") | Some("deftype") | Some("typealias")
                ) =>
            {
                children(list)
                    .first()
                    .and_then(symbol_name)
                    .unwrap_or("")
                    .to_string()
            }
            _ => String::new(),
        };
        walk_for_tensor_precision(expr, errors, &mut seen, &def_name);
    }
}

fn walk_for_tensor_precision(
    expr: &deep::Expr,
    errors: &mut Vec<CheckError>,
    seen: &mut HashSet<(String, String)>,
    def_context: &str,
) {
    // Bail before this walker's own unbounded recursion exhausts the
    // native stack (gdb confirmed this is a real SIGSEGV site on deep `app`
    // trees, distinct from `infer_expr`). The bail records into
    // `STACK_EXHAUSTED`; the check entry boundary turns that into a single
    // located failure, so we just stop recursing here. See
    // `STACK_RED_ZONE_BYTES`.
    stack_guard!(
        "validate_tensor_precisions (walk_for_tensor_precision)",
        expr
    );
    match expr {
        deep::Expr::List(list, _span) => {
            // Check t-tensor nodes at this level.
            if get_tag(list) == Some("t-tensor") {
                let kids = children(list);
                if let Some(last) = kids.last()
                    && let deep::Expr::List(prec_list, _) = last
                    && get_tag(prec_list) == Some("t-prim")
                    && let Some(name) = children(prec_list).first().and_then(symbol_name)
                {
                    let active_set = "f32, f64, bf16, f16, bool, int8, int16, int32, int64";
                    // A1 (WS-A0 RT-1 fixup): unsigned dtype names per
                    // spec/04-type-system.md §1.1.2. Mirror the f8e4m3
                    // §1.1.1 rejection contract — these names never
                    // resolve through `Prim::parse_name`, so without
                    // this guard `tensor[..., u8]` would silently fall
                    // through with no diagnostic.
                    if is_unsigned_dtype_name(name)
                        && seen.insert((def_context.to_string(), name.to_string()))
                    {
                        if let Some(diag) =
                            unsigned_family_diagnostic(name, /* tensor = */ true)
                        {
                            errors.push(diag);
                        }
                    } else if let Some(prim) = Prim::parse_name(name)
                        && !prim.is_valid_tensor_precision()
                        && seen.insert((def_context.to_string(), name.to_string()))
                    {
                        if matches!(prim, Prim::F8e4m3) {
                            errors.push(CheckError::new(
                                CheckErrorKind::UnsupportedTensorPrecision,
                                format!(
                                    "tensor element precision `f8e4m3` is deferred per \
                                     spec/04-type-system.md §1.1.1 and is not part of the active \
                                     numeric primitive set ({active_set})",
                                ),
                                vec![format!(
                                    "f8e4m3 has no active backend in this cycle; pick one of \
                                     {active_set} or see spec/04-type-system.md §1.1.1 for the \
                                     deferral rationale",
                                )],
                            ));
                        } else {
                            errors.push(CheckError::new(
                                CheckErrorKind::UnsupportedTensorPrecision,
                                format!(
                                    "tensor element precision `{name}` is not supported by the \
                                     current backend set (supported: {active_set})",
                                ),
                                vec![format!(
                                    "Use tensor[..., f32] and cast host scalars explicitly, or \
                                     keep `{name}` as a host scalar",
                                )],
                            ));
                        }
                    } else if Prim::parse_name(name).is_none()
                        && !is_unsigned_dtype_name(name)
                        && seen.insert((def_context.to_string(), name.to_string()))
                    {
                        // WS-A5 RT-3a F3: an identifier in a `t-prim`
                        // precision slot that is neither a known active
                        // primitive nor a §1.1.2 unsigned alias is an
                        // unbound name. Inside a sig the desugarer emits
                        // such an identifier as `t-var`, so reaching this
                        // arm with `t-prim` proves the name appears in a
                        // value-position annotation (let binding, def
                        // param without a surrounding sig that quantified
                        // it) where the closed primitive set must apply.
                        // Without this guard the name silently collapses
                        // to `Type::Error` via `deep_type_to_type_inner`'s
                        // `Prim::parse_name` fall-through and the
                        // permissive unify rule absorbs the mismatch.
                        errors.push(CheckError::new(
                            CheckErrorKind::UnsupportedTensorPrecision,
                            format!(
                                "tensor element precision `{name}` is not a recognized \
                                 primitive (active set: {active_set}); inside a sig an \
                                 unbound lowercase name introduces a precision tvar per \
                                 spec/04-type-system.md §5.8, but in this position the \
                                 closed primitive set applies",
                            ),
                            vec![format!(
                                "Use one of {active_set}, or move the annotation into a \
                                 `sig` declaration that quantifies `{name}` as a precision \
                                 type variable",
                            )],
                        ));
                    }
                }
            }

            // `cast` is inferred by infer_cast which already emits a clearer
            // site-local error for bad precisions. Skip the walker's recursion
            // inside a cast so we don't duplicate the diagnostic.
            if get_tag(list) == Some("cast") {
                return;
            }

            // Recurse into metadata map (element[1]), which may carry
            // `type:` ascriptions that also contain t-tensor types.
            if list.elements.len() >= 2
                && let deep::Expr::Map(map, _) = &list.elements[1]
            {
                for (_, v) in &map.entries {
                    walk_for_tensor_precision(v, errors, seen, def_context);
                }
            }

            // Recurse into children (elements after index 1).
            for child in children(list) {
                walk_for_tensor_precision(child, errors, seen, def_context);
            }
        }
        deep::Expr::Map(map, _) => {
            for (_, v) in &map.entries {
                walk_for_tensor_precision(v, errors, seen, def_context);
            }
        }
        deep::Expr::MetaExpr(meta, _) => {
            for (_, v) in &meta.entries {
                walk_for_tensor_precision(v, errors, seen, def_context);
            }
            walk_for_tensor_precision(&meta.expr, errors, seen, def_context);
        }
        deep::Expr::Atom(_, _) => {}
    }
}

/// WS-A8 cross-row enforcement: spec/04-type-system.md §5.4 (transcendentals
/// on float-only) and §5.7.2 (matmul not admitted on integer operands)
/// fire correctly at direct primitive call sites
/// (see `check_matmul_signature` and the `TENSOR_OPS` post-check in
/// `infer_app`), but were silent when the same restricted op was reached
/// through a polymorphic-precision sig instantiation.
///
/// Example: a stdlib `linear.forward` body uses `matmul(x, w)`. Its sig
/// is `&tensor[a, b, p] -> &tensor[b, c, p] -> tensor[a, c, p]`. Inside
/// the body, `matmul`'s operand precisions are both `Var(p)`; the §5.7.2
/// check (`lhs_prec.is_integer()`) returns `false` for a `Var`. At a
/// concrete call site `linear.forward(x: int32, w: int32)`, the call
/// site instantiates `p` to `int32` via unification — but the body's
/// already-checked `matmul(x, w)` doesn't get re-checked. The integer
/// rejection silently slipped through.
///
/// This pass closes the gap. It walks every `(app (var name) ...)` call
/// site post-inference and, when the callee is a top-level user-def
/// with a polymorphic-precision sig, builds a precision-tvar
/// substitution from the call site's arg types vs the callee's sig
/// parameter types, then re-checks the callee's body for restricted
/// ops with the substituted operand precisions.
fn validate_polymorphic_op_constraints(
    exprs: &[deep::Expr],
    type_env: &IrTypeEnv,
    errors: &mut Vec<CheckError>,
) {
    let defs_with_bodies = collect_def_bodies(exprs);
    let defsigs = collect_defsig_exprs(exprs);
    // Merge defsig sigs into the type-env view so polymorphic
    // signatures (which haven't been annotated onto def bodies yet) are
    // visible to the call-site lookup.
    let mut combined_env: IrTypeEnv = type_env.clone();
    for (name, sig_expr) in &defsigs {
        combined_env.entry(name.clone()).or_insert(sig_expr.clone());
    }
    // For each top-level def we walk into, build a local scope mapping
    // body-param names to their declared types (extracted from the
    // def's sig in `combined_env`). This lets us resolve `(var x)`
    // references inside the body without requiring the bodies to have
    // been annotated. Bodies of polymorphic defs intentionally have
    // their inner exprs untouched by the annotator at this stage.
    for expr in top_level_decl_items(exprs) {
        let scope = build_def_param_scope(expr, &combined_env);
        walk_for_poly_op_constraint_violations(
            expr,
            &defs_with_bodies,
            &combined_env,
            &scope,
            errors,
        );
    }
}

/// Build a name → declared-type-expr map for a def's body params, by
/// pairing each param's name with the corresponding sig parameter slot.
fn build_def_param_scope(expr: &deep::Expr, sigs: &IrTypeEnv) -> HashMap<String, deep::Expr> {
    let mut scope = HashMap::new();
    let deep::Expr::List(list, _) = expr else {
        return scope;
    };
    if get_tag(list) != Some("def") {
        return scope;
    }
    let kids = children(list);
    let Some(name) = kids.first().and_then(symbol_name) else {
        return scope;
    };
    let Some(body_expr) = kids.get(1) else {
        return scope;
    };
    let Some((param_names, _)) = extract_fn_params_and_body(body_expr) else {
        return scope;
    };
    // Prefer the def's surrounding sig if any; fall back to inline
    // param-type annotations on the params themselves (def shape:
    // `def f(x: tensor[3, f32]) = ...`).
    if let Some(sig) = sigs.get(name)
        && let Some((sig_params, _)) = parse_t_fn_parts(sig)
    {
        for (pname, sig_param) in param_names.iter().zip(sig_params.iter()) {
            scope.insert(pname.clone(), sig_param.clone());
        }
        return scope;
    }
    // Fall back: inline param-type annotations.
    let deep::Expr::List(fn_list, _) = body_expr else {
        return scope;
    };
    let Some(deep::Expr::List(params_list, _)) = children(fn_list).first() else {
        return scope;
    };
    if get_tag(params_list) != Some("params") {
        return scope;
    }
    for param in children(params_list) {
        if let Some((pname, Some(ty_expr))) = param_name_and_inline_type(param) {
            scope.insert(pname, ty_expr);
        }
    }
    scope
}

/// Extract `(name {type: T} ...)` shape's name+type from a single param
/// expression. Returns `None` for plain `(name {})` shape (no inline
/// type) — the surrounding sig fills those in.
fn param_name_and_inline_type(param: &deep::Expr) -> Option<(String, Option<deep::Expr>)> {
    match param {
        deep::Expr::Atom(deep::Atom::Symbol(name), _) => Some((name.clone(), None)),
        deep::Expr::List(list, _) => {
            let Some(deep::Expr::Atom(deep::Atom::Symbol(name), _)) = list.elements.first() else {
                return None;
            };
            let ty = match list.elements.get(1) {
                Some(deep::Expr::Map(meta, _)) => meta
                    .entries
                    .iter()
                    .find(|(k, _)| k == "type")
                    .map(|(_, v)| v.clone()),
                _ => None,
            };
            Some((name.clone(), ty))
        }
        deep::Expr::MetaExpr(meta, _) => {
            let deep::Expr::Atom(deep::Atom::Symbol(name), _) = meta.expr.as_ref() else {
                return None;
            };
            let ty = meta
                .entries
                .iter()
                .find(|(k, _)| k == "type")
                .map(|(_, v)| v.clone());
            Some((name.clone(), ty))
        }
        _ => None,
    }
}

/// Collect every `(defsig {} name sig_expr)` at the top level into a
/// name → sig-expr map. WS-A8 needs this so polymorphic-sig info reaches
/// the cross-row enforcement pass even when the def's body annotation
/// has not yet been populated by the annotator.
fn collect_defsig_exprs(exprs: &[deep::Expr]) -> HashMap<String, deep::Expr> {
    let mut out = HashMap::new();
    for expr in top_level_decl_items(exprs) {
        let deep::Expr::List(list, _) = expr else {
            continue;
        };
        if get_tag(list) != Some("defsig") {
            continue;
        }
        let kids = children(list);
        if let Some(name) = kids.first().and_then(symbol_name)
            && let Some(sig) = kids.get(1)
        {
            out.insert(name.to_string(), sig.clone());
        }
    }
    out
}

/// Map from def name to (params: Vec<param-name>, body-expr).
type DefBodyMap = HashMap<String, (Vec<String>, deep::Expr)>;

fn collect_def_bodies(exprs: &[deep::Expr]) -> DefBodyMap {
    let mut out = HashMap::new();
    for expr in top_level_decl_items(exprs) {
        let deep::Expr::List(list, _) = expr else {
            continue;
        };
        if get_tag(list) != Some("def") {
            continue;
        }
        let kids = children(list);
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        let Some(body) = kids.get(1) else {
            continue;
        };
        let Some((params, body_expr)) = extract_fn_params_and_body(body) else {
            continue;
        };
        out.insert(name.to_string(), (params, body_expr));
    }
    out
}

fn extract_fn_params_and_body(expr: &deep::Expr) -> Option<(Vec<String>, deep::Expr)> {
    let deep::Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("fn") {
        return None;
    }
    let kids = children(list);
    let params_expr = kids.first()?;
    let body_expr = kids.get(1)?;
    let deep::Expr::List(params_list, _) = params_expr else {
        return None;
    };
    if get_tag(params_list) != Some("params") {
        return None;
    }
    let mut names = Vec::new();
    for param in children(params_list) {
        if let Some(name) = param_name_for_refs(param) {
            names.push(name);
        }
    }
    Some((names, body_expr.clone()))
}

fn walk_for_poly_op_constraint_violations(
    expr: &deep::Expr,
    defs: &DefBodyMap,
    type_env: &IrTypeEnv,
    scope: &HashMap<String, deep::Expr>,
    errors: &mut Vec<CheckError>,
) {
    stack_guard!("walk_for_poly_op_constraint_violations", expr);
    match expr {
        deep::Expr::List(list, _span) => {
            // Check if this is `(app (var name) arg1 arg2 ...)` calling
            // a top-level user-def with a polymorphic precision sig.
            if get_tag(list) == Some("app") {
                check_app_for_poly_op_constraint(list, defs, type_env, scope, errors);
            }
            for child in &list.elements {
                walk_for_poly_op_constraint_violations(child, defs, type_env, scope, errors);
            }
        }
        deep::Expr::Map(map, _) => {
            for (_, v) in &map.entries {
                walk_for_poly_op_constraint_violations(v, defs, type_env, scope, errors);
            }
        }
        deep::Expr::MetaExpr(meta, _) => {
            walk_for_poly_op_constraint_violations(&meta.expr, defs, type_env, scope, errors);
            for (_, v) in &meta.entries {
                walk_for_poly_op_constraint_violations(v, defs, type_env, scope, errors);
            }
        }
        deep::Expr::Atom(_, _) => {}
    }
}

/// At an `(app (var callee_name) arg1 arg2 ...)` call site, if `callee_name`
/// is a top-level user-def with a polymorphic-precision sig, compute the
/// call-site precision substitution and re-check the body for restricted
/// ops with substituted precisions.
fn check_app_for_poly_op_constraint(
    list: &deep::List,
    defs: &DefBodyMap,
    type_env: &IrTypeEnv,
    scope: &HashMap<String, deep::Expr>,
    errors: &mut Vec<CheckError>,
) {
    let kids = children(list);
    let Some(callee_expr) = kids.first() else {
        return;
    };
    let deep::Expr::List(callee_list, _) = callee_expr else {
        return;
    };
    if get_tag(callee_list) != Some("var") {
        return;
    }
    let Some(callee_name) = children(callee_list).first().and_then(symbol_name) else {
        return;
    };
    // Look up the callee's declared signature; only proceed if it carries
    // a precision tvar (otherwise there's nothing to monomorphize).
    let Some(sig_expr) = lookup_sig_in_type_env(type_env, callee_name) else {
        return;
    };
    let Some((sig_params, _sig_ret)) = parse_t_fn_parts(sig_expr) else {
        return;
    };
    if !sig_params.iter().any(type_expr_has_tensor_prec_var) {
        return;
    }
    // Look up the callee's body so we can scan it.
    let Some((body_params, body)) = defs.get(callee_name) else {
        return;
    };
    // Build the call-site precision substitution: for each sig parameter
    // whose precision slot is `(t-var {} q)`, resolve the corresponding
    // call-site argument's concrete precision. Try the arg's `type:`
    // annotation first; fall back to the enclosing-def `scope` when the
    // arg is `(var x)` for an unannotated body-level reference.
    let mut subst: HashMap<String, String> = HashMap::new();
    for (sig_param, arg_expr) in sig_params.iter().zip(kids.iter().skip(1)) {
        let Some(prec_var_name) = precision_var_name_in_type_expr(sig_param) else {
            continue;
        };
        let arg_ty =
            annotated_type_of_expr(arg_expr).or_else(|| resolve_var_type_in_scope(arg_expr, scope));
        let Some(arg_ty) = arg_ty else {
            continue;
        };
        let Some(prim_name) = precision_prim_name_in_type_expr(&arg_ty) else {
            continue;
        };
        // Last-write-wins is fine: if the same `q` appears in multiple
        // params, the call site's unification already enforced consistency
        // (otherwise `chelis check` would have surfaced a precision
        // mismatch earlier in the pipeline).
        subst.insert(prec_var_name, prim_name);
    }
    if subst.is_empty() {
        return;
    }
    // Map the body's parameter names to the sig's parameter precision-var
    // names, so we can resolve `(var x)` inside the body to a precision
    // variable. The body's params and the sig's params line up by
    // position.
    let mut param_to_prec: HashMap<String, String> = HashMap::new();
    for (body_param_name, sig_param) in body_params.iter().zip(sig_params.iter()) {
        if let Some(prec_var_name) = precision_var_name_in_type_expr(sig_param) {
            param_to_prec.insert(body_param_name.clone(), prec_var_name);
        }
    }
    // Walk the body looking for restricted ops applied to body parameters
    // whose precision tvar (after substitution) violates §5.4 / §5.7.2.
    walk_body_for_restricted_ops(body, &param_to_prec, &subst, callee_name, list, errors);
}

/// Top-level def signature lookup: scan `type_env` for the callee's
/// declared `(t-fn ...)` signature.
fn lookup_sig_in_type_env<'a>(type_env: &'a IrTypeEnv, name: &str) -> Option<&'a deep::Expr> {
    type_env.get(name).or_else(|| {
        // Fall back to terminal-name match (mirrors `lookup_declared_type_expr`).
        let mut matches = type_env.iter().filter_map(|(key, value)| {
            let key_terminal = key
                .rsplit_once("__")
                .map(|(_, t)| t)
                .unwrap_or(key.as_str());
            let key_terminal = key_terminal
                .rsplit_once('.')
                .map(|(_, t)| t)
                .unwrap_or(key_terminal);
            (key_terminal == name).then_some(value)
        });
        let first = matches.next()?;
        matches.next().is_none().then_some(first)
    })
}

fn parse_t_fn_parts(expr: &deep::Expr) -> Option<(Vec<deep::Expr>, deep::Expr)> {
    let deep::Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("t-fn") {
        return None;
    }
    let kids = children(list);
    let (ret, args) = kids.split_last()?;
    Some((args.iter().map(|e| (*e).clone()).collect(), (*ret).clone()))
}

/// Strip a leading `(t-ref {} ...)` wrapper for precision-var probing;
/// the borrow doesn't affect the precision slot.
fn strip_t_ref(expr: &deep::Expr) -> &deep::Expr {
    if let deep::Expr::List(list, _) = expr
        && get_tag(list) == Some("t-ref")
        && let Some(inner) = list.elements.get(2)
    {
        return inner;
    }
    expr
}

fn type_expr_has_tensor_prec_var(expr: &deep::Expr) -> bool {
    let stripped = strip_t_ref(expr);
    let deep::Expr::List(list, _) = stripped else {
        return false;
    };
    match get_tag(list) {
        Some("t-tensor") => precision_var_name_in_type_expr(stripped).is_some(),
        Some("t-fn") | Some("t-tuple") | Some("t-adt") => {
            children(list).iter().any(type_expr_has_tensor_prec_var)
        }
        _ => false,
    }
}

/// Pull the precision-var name out of a tensor-type expression's
/// last child (the precision slot). Returns `Some(name)` when the
/// slot is `(t-var {} name)`; `None` when concrete or non-tensor.
fn precision_var_name_in_type_expr(expr: &deep::Expr) -> Option<String> {
    let stripped = strip_t_ref(expr);
    let deep::Expr::List(list, _) = stripped else {
        return None;
    };
    if get_tag(list) != Some("t-tensor") {
        return None;
    }
    let last = list.elements.last()?;
    let deep::Expr::List(prec_list, _) = last else {
        return None;
    };
    if get_tag(prec_list) != Some("t-var") {
        return None;
    }
    children(prec_list)
        .first()
        .and_then(symbol_name)
        .map(String::from)
}

/// Pull the concrete `(t-prim {} name)` from a tensor-type expression's
/// precision slot. Returns `None` when the slot is a tvar.
fn precision_prim_name_in_type_expr(expr: &deep::Expr) -> Option<String> {
    let stripped = strip_t_ref(expr);
    let deep::Expr::List(list, _) = stripped else {
        return None;
    };
    if get_tag(list) != Some("t-tensor") {
        return None;
    }
    let last = list.elements.last()?;
    let deep::Expr::List(prec_list, _) = last else {
        return None;
    };
    if get_tag(prec_list) != Some("t-prim") {
        return None;
    }
    children(prec_list)
        .first()
        .and_then(symbol_name)
        .map(String::from)
}

/// Look up the type of `(var name)` in the enclosing-def `scope` map
/// (built from the def's sig + inline param annotations). Returns
/// `None` when the expression is not a var or the name is not in
/// scope.
fn resolve_var_type_in_scope(
    expr: &deep::Expr,
    scope: &HashMap<String, deep::Expr>,
) -> Option<deep::Expr> {
    let deep::Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("var") {
        return None;
    }
    let name = children(list).first().and_then(symbol_name)?;
    scope.get(name).cloned()
}

/// Pull the `type:` meta entry off a Deep expression. Returns the inner
/// type expression when present.
fn annotated_type_of_expr(expr: &deep::Expr) -> Option<deep::Expr> {
    let deep::Expr::List(list, _) = expr else {
        return None;
    };
    let meta = match list.elements.get(1) {
        Some(deep::Expr::Map(meta, _)) => meta,
        _ => return None,
    };
    meta.entries
        .iter()
        .find(|(k, _)| k == "type")
        .map(|(_, v)| v.clone())
}

/// Walk a polymorphic def's body looking for `(app (var op) arg1 arg2 ...)`
/// where `op` is one of the §5.4 (transcendental, float-only) or §5.7.2
/// (matmul, integer-rejected) restricted ops, and `arg1`/`arg2` are
/// `(var name)` references to body parameters. Validate the substituted
/// precision against the spec rule.
fn walk_body_for_restricted_ops(
    expr: &deep::Expr,
    param_to_prec: &HashMap<String, String>,
    subst: &HashMap<String, String>,
    callee_name: &str,
    call_site_list: &deep::List,
    errors: &mut Vec<CheckError>,
) {
    stack_guard!("walk_body_for_restricted_ops", expr);
    if let deep::Expr::List(list, _) = expr
        && get_tag(list) == Some("app")
    {
        let kids = children(list);
        if let Some(deep::Expr::List(callee, _)) = kids.first()
            && get_tag(callee) == Some("var")
            && let Some(op_name) = children(callee).first().and_then(symbol_name)
        {
            check_restricted_op_in_body(
                op_name,
                &kids[1..],
                param_to_prec,
                subst,
                callee_name,
                call_site_list,
                errors,
            );
        }
        for child in &list.elements {
            walk_body_for_restricted_ops(
                child,
                param_to_prec,
                subst,
                callee_name,
                call_site_list,
                errors,
            );
        }
    } else if let deep::Expr::List(list, _) = expr {
        for child in &list.elements {
            walk_body_for_restricted_ops(
                child,
                param_to_prec,
                subst,
                callee_name,
                call_site_list,
                errors,
            );
        }
    } else if let deep::Expr::MetaExpr(meta, _) = expr {
        walk_body_for_restricted_ops(
            &meta.expr,
            param_to_prec,
            subst,
            callee_name,
            call_site_list,
            errors,
        );
    }
}

/// `op_name`: the name of the inner operation (e.g. `matmul`, `exp`).
/// `op_args`: the arg expressions of the `(app (var op_name) ...)` form.
const TRANSCENDENTAL_FLOAT_ONLY_OPS: &[&str] = &[
    "exp",
    "log",
    "sin",
    "cos",
    "sqrt",
    "tan",
    "atan",
    "softmax",
    "sigmoid",
    "tanh",
    "silu",
    "gelu",
    "layer_norm",
    "normalize",
    // `recip` is float-only per spec/05-risc-primitives.md §2.2: an
    // integer reciprocal has no meaningful IEEE-754 interpretation
    // (would always be 0 for |x| > 1 and undefined for x = 0).
    // `div` is NOT here — it is float-only too (chelis#178) but carries
    // a §2.1 citation pointing at `floor_div` / `trunc_div`, so it is
    // handled by `FLOAT_ONLY_DIV_OPS` with a tailored diagnostic.
    "recip",
];

/// Float-only ops whose integer-operand rejection cites
/// spec/05-risc-primitives.md §2.1 and points at the integer-division
/// replacements (chelis#178). Kept separate from
/// `TRANSCENDENTAL_FLOAT_ONLY_OPS` so the diagnostic names the migration
/// ops rather than the generic transcendental §5.4 rule.
const FLOAT_ONLY_DIV_OPS: &[&str] = &["div"];

/// Integer-only ops whose float-operand rejection cites
/// spec/05-risc-primitives.md §2.1 (chelis#178). `trunc_div` is the
/// C/Rust truncating quotient and is not defined on float operands.
const INTEGER_ONLY_DIV_OPS: &[&str] = &["trunc_div"];

const INTEGER_REJECTED_OPS: &[&str] = &["matmul"];

fn check_restricted_op_in_body(
    op_name: &str,
    op_args: &[deep::Expr],
    param_to_prec: &HashMap<String, String>,
    subst: &HashMap<String, String>,
    callee_name: &str,
    call_site_list: &deep::List,
    errors: &mut Vec<CheckError>,
) {
    if !INTEGER_REJECTED_OPS.contains(&op_name)
        && !TRANSCENDENTAL_FLOAT_ONLY_OPS.contains(&op_name)
        && !FLOAT_ONLY_DIV_OPS.contains(&op_name)
        && !INTEGER_ONLY_DIV_OPS.contains(&op_name)
    {
        return;
    }
    // Resolve each arg's precision through the param-to-prec mapping and
    // the call-site substitution. The arg may be either a direct
    // `(var x)` reference to a body param OR a deeper expression — for
    // the latter we look at its annotated type's precision slot.
    for arg in op_args {
        let resolved_prim = resolve_arg_precision_through_subst(arg, param_to_prec, subst);
        let Some(prim_name) = resolved_prim else {
            continue;
        };
        let Some(prim) = Prim::parse_name(&prim_name) else {
            continue;
        };
        let _ = call_site_list; // span hint reserved for future plumbing
        if INTEGER_REJECTED_OPS.contains(&op_name) && prim.is_integer() {
            errors.push(CheckError::new(
                CheckErrorKind::PrecisionMismatch,
                format!(
                    "{op_name} on integer operand precision `{prim_name}` is not \
                     admitted in this cycle per spec/04-type-system.md \u{00a7}5.7.2: \
                     integer matmul not admitted (the spec deliberately defers the \
                     integer-matmul accumulator rule). Reached through the polymorphic \
                     sig for `{callee_name}` instantiated at integer precision; the \
                     restriction fires on every integer instantiation, including via \
                     stdlib wrappers."
                ),
                vec![format!(
                    "spec/04-type-system.md \u{00a7}5.7.2: there is no current backend \
                     that supports integer BLAS. Use reduce_sum over an explicit \
                     expand+mul lowering for integer inner products, or float \
                     instantiations of `{callee_name}`."
                )],
            ));
            return;
        }
        if TRANSCENDENTAL_FLOAT_ONLY_OPS.contains(&op_name) && !prim.is_float() {
            errors.push(CheckError::new(
                CheckErrorKind::PrecisionMismatch,
                format!(
                    "{op_name} on operand precision `{prim_name}` is not admitted per \
                     spec/04-type-system.md \u{00a7}5.4: transcendental operations are \
                     restricted to f32, f64, bf16, f16 (not integer). Reached through \
                     the polymorphic sig for `{callee_name}` instantiated at \
                     `{prim_name}`; the restriction fires on every non-float \
                     instantiation, including via stdlib wrappers."
                ),
                vec![format!(
                    "spec/04-type-system.md \u{00a7}5.4: cast to a float precision \
                     before applying `{op_name}`, or pick a float instantiation of \
                     `{callee_name}`."
                )],
            ));
            return;
        }
        if FLOAT_ONLY_DIV_OPS.contains(&op_name) && prim.is_integer() {
            // chelis#178: integer `div` reached through a polymorphic
            // wrapper instantiated at an integer dtype.
            errors.push(CheckError::new(
                CheckErrorKind::PrecisionMismatch,
                format!(
                    "div on integer operand precision `{prim_name}` is not admitted per \
                     spec/05-risc-primitives.md \u{00a7}2.1: `div` is float-only \
                     (IEEE-754). Use `floor_div` (round toward -inf) or `trunc_div` \
                     (round toward zero) for integers. Reached through the polymorphic \
                     sig for `{callee_name}` instantiated at `{prim_name}`; the \
                     restriction fires on every integer instantiation, including via \
                     stdlib wrappers."
                ),
                vec![format!(
                    "spec/05-risc-primitives.md \u{00a7}2.1: integer division uses \
                     `floor_div` or `trunc_div`; pick a float instantiation of \
                     `{callee_name}` for `div`."
                )],
            ));
            return;
        }
        if INTEGER_ONLY_DIV_OPS.contains(&op_name) && prim.is_float() {
            // chelis#178: `trunc_div` reached through a polymorphic
            // wrapper instantiated at a float dtype.
            errors.push(CheckError::new(
                CheckErrorKind::PrecisionMismatch,
                format!(
                    "trunc_div on float operand precision `{prim_name}` is not admitted \
                     per spec/05-risc-primitives.md \u{00a7}2.1: `trunc_div` is \
                     integer-only. Use `div` for IEEE-754 float division, or `floor_div` \
                     for a floored float quotient. Reached through the polymorphic sig \
                     for `{callee_name}` instantiated at `{prim_name}`."
                ),
                vec![format!(
                    "spec/05-risc-primitives.md \u{00a7}2.1: `trunc_div` requires integer \
                     operands; pick an integer instantiation of `{callee_name}`."
                )],
            ));
            return;
        }
    }
}

/// Resolve a single arg's precision-tvar binding for restricted-op
/// checking. If the arg is `(var name)` and `name` is in the body's
/// param-to-prec map, look up the call-site substitution.  If the arg
/// is itself an `app` with an annotated tensor type whose precision
/// slot is concrete, use that. Returns `Some(prim_name)` if resolvable.
fn resolve_arg_precision_through_subst(
    arg: &deep::Expr,
    param_to_prec: &HashMap<String, String>,
    subst: &HashMap<String, String>,
) -> Option<String> {
    if let deep::Expr::List(list, _) = arg
        && get_tag(list) == Some("var")
        && let Some(name) = children(list).first().and_then(symbol_name)
        && let Some(prec_var) = param_to_prec.get(name)
        && let Some(prim) = subst.get(prec_var)
    {
        return Some(prim.clone());
    }
    // Fall back to the arg's annotated type's concrete precision (when
    // the body did its own arithmetic, e.g. `wx = matmul(x, w);
    // softmax(wx)`).
    if let Some(ty) = annotated_type_of_expr(arg)
        && let Some(prim) = precision_prim_name_in_type_expr(&ty)
    {
        return Some(prim);
    }
    None
}

fn validate_ir_expr(
    expr: &deep::Expr,
    type_env: &IrTypeEnv,
    static_env: &mut HashMap<String, StaticValue>,
    failed_let_names: &mut HashSet<String>,
    errors: &mut Vec<CheckError>,
) -> StaticValue {
    stack_guard!("validate_ir_expr", expr, StaticValue::Unknown);
    match expr {
        deep::Expr::List(list, _) => {
            if get_tag(list) == Some("module") {
                for elem in list.elements.iter().skip(3) {
                    validate_ir_expr(elem, type_env, static_env, failed_let_names, errors);
                }
                return StaticValue::Unknown;
            }
            if get_tag(list) == Some("def") {
                let kids = children(list);
                let Some(name) = kids.first().and_then(symbol_name) else {
                    return StaticValue::Unknown;
                };
                let Some(value_expr) = kids.get(1) else {
                    return StaticValue::Unknown;
                };
                let value =
                    validate_ir_expr(value_expr, type_env, static_env, failed_let_names, errors);
                static_env.insert(name.to_string(), value);
                return StaticValue::Unknown;
            }
            if get_tag(list) == Some("fn") {
                let scoped_env = extend_ir_env_with_fn_params(list, type_env);
                let mut scoped_static_env = static_env.clone();
                bind_fn_params_unknown(list, &mut scoped_static_env);
                for elem in &list.elements {
                    validate_ir_expr(
                        elem,
                        &scoped_env,
                        &mut scoped_static_env,
                        failed_let_names,
                        errors,
                    );
                }
                return StaticValue::Unknown;
            }
            if get_tag(list) == Some("let") {
                let kids = children(list);
                let mut scoped_static_env = static_env.clone();
                // Clone the type env on let-scope entry so each binding's
                // derivable IR-shape-sensitive type (e.g. conv2d's output
                // dims) can extend the env visible to the let body. Without
                // this the validator cannot resolve `(var y)` for a let-
                // bound `y = conv2d(...)` and silently rejects the next
                // shape-sensitive call that consumes `y` (RT-205 F5).
                let mut scoped_type_env = type_env.clone();
                if let Some(deep::Expr::List(bind_list, _)) = kids.first()
                    && get_tag(bind_list) == Some("bind")
                {
                    let bind_children = children(bind_list);
                    let mut index = 0;
                    while index + 1 < bind_children.len() {
                        if let Some(name) = symbol_name(&bind_children[index]) {
                            let value_expr = &bind_children[index + 1];
                            // Recurse into the RHS so its own validation
                            // can push diagnostics and suppress downstream
                            // cascade errors via `failed_let_names` (set
                            // below when the RHS is a recognized
                            // shape-sensitive form whose output type is
                            // non-derivable, RT-205 round-4 / issue #212).
                            let value = validate_ir_expr(
                                value_expr,
                                &scoped_type_env,
                                &mut scoped_static_env,
                                failed_let_names,
                                errors,
                            );
                            scoped_static_env.insert(name.to_string(), value);
                            // If the RHS is a shape-sensitive IR builtin
                            // whose output type is derivable from its args,
                            // register the derived type so downstream uses
                            // of `name` resolve correctly.
                            let derived =
                                derive_ir_builtin_output_type(value_expr, &scoped_type_env);
                            match derived {
                                Some(ty) => {
                                    scoped_type_env.insert(name.to_string(), ty);
                                }
                                None => {
                                    // Mark as failed-derivation when the
                                    // RHS is structurally a recognized
                                    // shape-sensitive form (a known
                                    // shape-sensitive builtin or a
                                    // unary/binary passthrough wrapper
                                    // around one, recursively) but its
                                    // output type could not be derived.
                                    // This catches `y = conv2d(bad)`
                                    // and the R3 F-A passthrough cases
                                    // like `y = relu(conv2d(bad))`.
                                    //
                                    // RT-205 round-4 / issue #212: the
                                    // previous guard checked
                                    // `errors.len() > errs_before` to
                                    // detect an errored RHS, which fails
                                    // for chains of length 3+ because
                                    // cascade suppression already
                                    // silences the level-2 RHS's
                                    // diagnostic, so the level-2 name is
                                    // never marked and the level-3 RHS
                                    // re-emits a phantom error. The
                                    // structural check
                                    // `let_rhs_is_recognized_shape_sensitive`
                                    // does not depend on diagnostic
                                    // count and propagates the failed
                                    // marker unboundedly down the chain.
                                    //
                                    // The recognition is intentionally
                                    // narrow: a clean RHS that is not
                                    // a recognized shape-sensitive form
                                    // (e.g. a user-defined fn call) still
                                    // does NOT cause suppression
                                    // downstream, so legitimate
                                    // "really wrong arg" cases still
                                    // surface their own diagnostic.
                                    if let deep::Expr::List(_, _) = value_expr
                                        && let_rhs_is_recognized_shape_sensitive(value_expr)
                                    {
                                        failed_let_names.insert(name.to_string());
                                    }
                                }
                            }
                        }
                        index += 2;
                    }
                }
                if let Some(body) = kids.get(1) {
                    return validate_ir_expr(
                        body,
                        &scoped_type_env,
                        &mut scoped_static_env,
                        failed_let_names,
                        errors,
                    );
                }
                return StaticValue::Unknown;
            }
            if let Some(tag) = get_tag(list) {
                // `par` (sequential v1, spec/03-deep-syntax.md §2.3) and `jit`
                // (compilation trigger, §2.7) are spec-blessed pass-through
                // forms at Phase 0 evaluation. The validator used to reject
                // both; the rejection is removed because lowering handles them
                // (see `lower_par` and the `jit` lowering arm).
                if tag == "app"
                    && let Some(func_name) = ir_builtin_name(list)
                    && is_ir_shape_sensitive_builtin(func_name)
                {
                    validate_ir_builtin_symbolic_requirements(
                        list,
                        func_name,
                        type_env,
                        failed_let_names,
                        errors,
                    );
                }
            }

            if get_tag(list) == Some("var")
                && let Some(name) = children(list).first().and_then(symbol_name)
            {
                return if name == "Nil" {
                    StaticValue::List(Vec::new())
                } else {
                    static_env
                        .get(name)
                        .cloned()
                        .unwrap_or(StaticValue::Unknown)
                };
            }
            if get_tag(list) == Some("lit") {
                return literal_static_value(expr);
            }
            if get_tag(list) == Some("cast") {
                let kids = children(list);
                return kids
                    .first()
                    .map(|inner| {
                        validate_ir_expr(inner, type_env, static_env, failed_let_names, errors)
                    })
                    .unwrap_or(StaticValue::Unknown);
            }
            if get_tag(list) == Some("app") {
                let kids = children(list);
                let func_name = kids.first().and_then(app_builtin_name);
                let arg_values = kids
                    .iter()
                    .skip(1)
                    .map(|arg| {
                        validate_ir_expr(arg, type_env, static_env, failed_let_names, errors)
                    })
                    .collect::<Vec<_>>();
                if func_name == Some("Cons") && arg_values.len() == 2 {
                    if let StaticValue::List(mut tail) = arg_values[1].clone() {
                        tail.insert(0, arg_values[0].clone());
                        return StaticValue::List(tail);
                    }
                    return StaticValue::Unknown;
                }
                if let Some(name) = func_name {
                    return validate_static_builtin_application(name, &arg_values, expr, errors);
                }
                return StaticValue::Unknown;
            }

            for elem in &list.elements {
                validate_ir_expr(elem, type_env, static_env, failed_let_names, errors);
            }
            StaticValue::Unknown
        }
        deep::Expr::Map(map, _) => {
            for (_, value) in &map.entries {
                validate_ir_expr(value, type_env, static_env, failed_let_names, errors);
            }
            StaticValue::Unknown
        }
        deep::Expr::MetaExpr(meta, _) => {
            for (_, value) in &meta.entries {
                validate_ir_expr(value, type_env, static_env, failed_let_names, errors);
            }
            validate_ir_expr(&meta.expr, type_env, static_env, failed_let_names, errors)
        }
        deep::Expr::Atom(_, _) => literal_static_value(expr),
    }
}

#[derive(Debug, Clone, PartialEq)]
enum StaticValue {
    Unknown,
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    List(Vec<StaticValue>),
    Tensor(StaticTensor),
}

#[derive(Debug, Clone, PartialEq)]
struct StaticTensor {
    shape: Vec<usize>,
    int_values: Option<Vec<i64>>,
}

fn bind_fn_params_unknown(fn_list: &deep::List, env: &mut HashMap<String, StaticValue>) {
    let Some(params_expr) = children(fn_list).first() else {
        return;
    };
    let deep::Expr::List(params_list, _) = params_expr else {
        return;
    };
    if get_tag(params_list) != Some("params") {
        return;
    }
    for param in children(params_list) {
        match param {
            deep::Expr::Atom(deep::Atom::Symbol(name), _) => {
                env.insert(name.clone(), StaticValue::Unknown);
            }
            deep::Expr::List(param_list, _) => {
                if let Some(name) = param_list.elements.first().and_then(symbol_name) {
                    env.insert(name.to_string(), StaticValue::Unknown);
                }
            }
            _ => {}
        }
    }
}

fn literal_static_value(expr: &deep::Expr) -> StaticValue {
    match expr {
        deep::Expr::Atom(deep::Atom::Int(value), _) => StaticValue::Int(*value),
        deep::Expr::Atom(deep::Atom::Float(value), _) => StaticValue::Float(*value),
        deep::Expr::Atom(deep::Atom::Bool(value), _) => StaticValue::Bool(*value),
        deep::Expr::Atom(deep::Atom::Str(value), _) => StaticValue::String(value.clone()),
        deep::Expr::List(list, _) if get_tag(list) == Some("lit") => children(list)
            .first()
            .map(literal_static_value)
            .unwrap_or(StaticValue::Unknown),
        _ => StaticValue::Unknown,
    }
}

fn app_builtin_name(expr: &deep::Expr) -> Option<&str> {
    let deep::Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("var") {
        return None;
    }
    children(list).first().and_then(symbol_name)
}

fn validate_static_builtin_application(
    name: &str,
    args: &[StaticValue],
    expr: &deep::Expr,
    errors: &mut Vec<CheckError>,
) -> StaticValue {
    match name {
        "pad_sequences" => static_pad_sequences(args),
        "to_tensor" => static_to_tensor(args),
        "concat" => static_concat(args, expr, errors),
        "split" => static_split(args, expr, errors),
        "gather" => static_gather(args, expr, errors),
        "scatter" => static_scatter(args, expr, errors),
        "scatter_replace" => static_scatter_replace(args, expr, errors),
        "clamp" => static_clamp(args, expr, errors),
        "einsum" => static_einsum(args, expr, errors),
        _ => StaticValue::Unknown,
    }
}

fn static_pad_sequences(args: &[StaticValue]) -> StaticValue {
    let Some(StaticValue::List(rows)) = args.first() else {
        return StaticValue::Unknown;
    };
    let mut width = 0usize;
    for row in rows {
        let StaticValue::List(items) = row else {
            return StaticValue::Unknown;
        };
        width = width.max(items.len());
    }
    StaticValue::Tensor(StaticTensor {
        shape: vec![rows.len(), width],
        int_values: None,
    })
}

fn static_to_tensor(args: &[StaticValue]) -> StaticValue {
    let Some(StaticValue::List(items)) = args.first() else {
        return StaticValue::Unknown;
    };
    let mut ints = Vec::with_capacity(items.len());
    for item in items {
        match item {
            StaticValue::Int(value) => ints.push(*value),
            StaticValue::Float(_) | StaticValue::Bool(_) => {
                return StaticValue::Tensor(StaticTensor {
                    shape: vec![items.len()],
                    int_values: None,
                });
            }
            _ => return StaticValue::Unknown,
        }
    }
    StaticValue::Tensor(StaticTensor {
        shape: vec![items.len()],
        int_values: Some(ints),
    })
}

fn static_concat(
    args: &[StaticValue],
    expr: &deep::Expr,
    errors: &mut Vec<CheckError>,
) -> StaticValue {
    let (Some(StaticValue::List(parts)), Some(StaticValue::Int(axis))) =
        (args.first(), args.get(1))
    else {
        return StaticValue::Unknown;
    };
    if *axis < 0 {
        return StaticValue::Unknown;
    }
    let tensors = parts
        .iter()
        .map(|value| match value {
            StaticValue::Tensor(tensor) => Some(tensor.clone()),
            _ => None,
        })
        .collect::<Option<Vec<_>>>();
    let Some(tensors) = tensors else {
        return StaticValue::Unknown;
    };
    let Some(first) = tensors.first() else {
        return StaticValue::Unknown;
    };
    let axis = *axis as usize;
    if axis >= first.shape.len() {
        return StaticValue::Unknown;
    }
    let mut shape = first.shape.clone();
    let mut axis_total = shape[axis];
    for tensor in tensors.iter().skip(1) {
        if tensor.shape.len() != shape.len() {
            push_static_runtime_error(
                expr,
                errors,
                "concat expects matching tensor rank".to_string(),
            );
            return StaticValue::Unknown;
        }
        for (dim, expected_extent) in shape.iter().enumerate() {
            if dim != axis && tensor.shape[dim] != *expected_extent {
                push_static_runtime_error(
                    expr,
                    errors,
                    "concat expects matching non-concatenated axes".to_string(),
                );
                return StaticValue::Unknown;
            }
        }
        axis_total += tensor.shape[axis];
    }
    shape[axis] = axis_total;
    StaticValue::Tensor(StaticTensor {
        shape,
        int_values: None,
    })
}

fn static_split(
    args: &[StaticValue],
    expr: &deep::Expr,
    errors: &mut Vec<CheckError>,
) -> StaticValue {
    let (
        Some(StaticValue::Tensor(tensor)),
        Some(StaticValue::Int(axis)),
        Some(StaticValue::List(sizes)),
    ) = (args.first(), args.get(1), args.get(2))
    else {
        return StaticValue::Unknown;
    };
    if *axis < 0 {
        return StaticValue::Unknown;
    }
    let axis = *axis as usize;
    if axis >= tensor.shape.len() {
        return StaticValue::Unknown;
    }
    let Some(size_values) = sizes
        .iter()
        .map(|value| match value {
            StaticValue::Int(size) if *size >= 0 => Some(*size as usize),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
    else {
        return StaticValue::Unknown;
    };
    if size_values.iter().sum::<usize>() != tensor.shape[axis] {
        push_static_runtime_error(
            expr,
            errors,
            "split sizes must sum to the selected axis extent".to_string(),
        );
    }
    StaticValue::Unknown
}

fn static_gather(
    args: &[StaticValue],
    expr: &deep::Expr,
    errors: &mut Vec<CheckError>,
) -> StaticValue {
    let (
        Some(StaticValue::Tensor(tensor)),
        Some(StaticValue::Tensor(indices)),
        Some(StaticValue::Int(axis)),
    ) = (args.first(), args.get(1), args.get(2))
    else {
        return StaticValue::Unknown;
    };
    let Some(axis) = normalize_static_axis(tensor.shape.len(), *axis) else {
        return StaticValue::Unknown;
    };
    if let Some(index_values) = &indices.int_values {
        for value in index_values {
            if *value < 0 || *value >= tensor.shape[axis] as i64 {
                push_static_runtime_error(
                    expr,
                    errors,
                    format!("gather index {value} out of bounds"),
                );
                return StaticValue::Unknown;
            }
        }
    }
    StaticValue::Tensor(StaticTensor {
        shape: gather_result_shape(&tensor.shape, &indices.shape, axis),
        int_values: None,
    })
}

fn static_scatter(
    args: &[StaticValue],
    expr: &deep::Expr,
    errors: &mut Vec<CheckError>,
) -> StaticValue {
    let (
        Some(StaticValue::Tensor(base)),
        Some(StaticValue::Tensor(indices)),
        Some(StaticValue::Tensor(updates)),
        Some(StaticValue::Int(axis)),
        Some(StaticValue::String(mode)),
    ) = (
        args.first(),
        args.get(1),
        args.get(2),
        args.get(3),
        args.get(4),
    )
    else {
        return StaticValue::Unknown;
    };
    let Some(axis) = normalize_static_axis(base.shape.len(), *axis) else {
        return StaticValue::Unknown;
    };
    let expected_updates = gather_result_shape(&base.shape, &indices.shape, axis);
    if updates.shape != expected_updates {
        push_static_runtime_error(
            expr,
            errors,
            "scatter updates must match gathered tensor shape and precision".to_string(),
        );
        return StaticValue::Unknown;
    }
    if let Some(index_values) = &indices.int_values {
        let mut seen = HashSet::new();
        for linear in 0..updates_shape_numel(&updates.shape) {
            let update_index = unravel_index(linear, &updates.shape);
            let gather_index = update_index[axis..axis + indices.shape.len()].to_vec();
            let gather_linear = ravel_index(&gather_index, &indices.shape);
            let gathered = index_values[gather_linear];
            if gathered < 0 || gathered >= base.shape[axis] as i64 {
                push_static_runtime_error(
                    expr,
                    errors,
                    format!("scatter index {gathered} out of bounds"),
                );
                return StaticValue::Unknown;
            }
            if mode == "replace" {
                let mut out_index = Vec::with_capacity(base.shape.len());
                out_index.extend_from_slice(&update_index[..axis]);
                out_index.push(gathered as usize);
                out_index.extend_from_slice(&update_index[axis + indices.shape.len()..]);
                let out_linear = ravel_index(&out_index, &base.shape);
                if !seen.insert(out_linear) {
                    push_static_runtime_error(
                        expr,
                        errors,
                        format!("scatter replace mode rejects duplicate target index {out_linear}"),
                    );
                    return StaticValue::Unknown;
                }
            }
        }
    }
    StaticValue::Tensor(base.clone())
}

/// Static check for the tensor-lane `scatter_replace(base, indices,
/// updates, axis)` builtin. Mirrors `static_scatter` with `mode ==
/// "replace"` semantics: validates the updates-shape contract,
/// rejects out-of-bounds indices when statically knowable, and
/// rejects duplicate target indices that would resolve via
/// non-deterministic per-axis collisions. Differs from
/// `static_scatter` in that there is no `mode` argument.
fn static_scatter_replace(
    args: &[StaticValue],
    expr: &deep::Expr,
    errors: &mut Vec<CheckError>,
) -> StaticValue {
    let (
        Some(StaticValue::Tensor(base)),
        Some(StaticValue::Tensor(indices)),
        Some(StaticValue::Tensor(updates)),
        Some(StaticValue::Int(axis)),
    ) = (args.first(), args.get(1), args.get(2), args.get(3))
    else {
        return StaticValue::Unknown;
    };
    let Some(axis) = normalize_static_axis(base.shape.len(), *axis) else {
        return StaticValue::Unknown;
    };
    let expected_updates = gather_result_shape(&base.shape, &indices.shape, axis);
    if updates.shape != expected_updates {
        push_static_runtime_error(
            expr,
            errors,
            "scatter_replace updates must match gathered tensor shape and precision".to_string(),
        );
        return StaticValue::Unknown;
    }
    if let Some(index_values) = &indices.int_values {
        let mut seen = HashSet::new();
        for linear in 0..updates_shape_numel(&updates.shape) {
            let update_index = unravel_index(linear, &updates.shape);
            let gather_index = update_index[axis..axis + indices.shape.len()].to_vec();
            let gather_linear = ravel_index(&gather_index, &indices.shape);
            let gathered = index_values[gather_linear];
            if gathered < 0 || gathered >= base.shape[axis] as i64 {
                push_static_runtime_error(
                    expr,
                    errors,
                    format!("scatter_replace index {gathered} out of bounds"),
                );
                return StaticValue::Unknown;
            }
            let mut out_index = Vec::with_capacity(base.shape.len());
            out_index.extend_from_slice(&update_index[..axis]);
            out_index.push(gathered as usize);
            out_index.extend_from_slice(&update_index[axis + indices.shape.len()..]);
            let out_linear = ravel_index(&out_index, &base.shape);
            if !seen.insert(out_linear) {
                push_static_runtime_error(
                    expr,
                    errors,
                    format!(
                        "scatter_replace rejects statically-known duplicate target index {out_linear}; \
                         use scatter_add (or scatter with mode=\"add\") for commutative accumulation"
                    ),
                );
                return StaticValue::Unknown;
            }
        }
    }
    StaticValue::Tensor(base.clone())
}

fn static_clamp(
    args: &[StaticValue],
    expr: &deep::Expr,
    errors: &mut Vec<CheckError>,
) -> StaticValue {
    let (
        Some(StaticValue::Tensor(input)),
        Some(StaticValue::Tensor(low)),
        Some(StaticValue::Tensor(high)),
    ) = (args.first(), args.get(1), args.get(2))
    else {
        return StaticValue::Unknown;
    };
    let low_ok = low.shape.is_empty() || low.shape == input.shape;
    let high_ok = high.shape.is_empty() || high.shape == input.shape;
    if !low_ok || !high_ok {
        push_static_runtime_error(
            expr,
            errors,
            "clamp expects scalar bounds or matching-shape tensor bounds".to_string(),
        );
        return StaticValue::Unknown;
    }
    StaticValue::Tensor(input.clone())
}

fn static_einsum(
    args: &[StaticValue],
    expr: &deep::Expr,
    errors: &mut Vec<CheckError>,
) -> StaticValue {
    let (
        Some(StaticValue::String(equation)),
        Some(StaticValue::Tensor(lhs)),
        Some(StaticValue::Tensor(rhs)),
    ) = (args.first(), args.get(1), args.get(2))
    else {
        return StaticValue::Unknown;
    };
    let Some((inputs, output)) = equation.split_once("->") else {
        return StaticValue::Unknown;
    };
    let mut input_groups = inputs.split(',');
    let lhs_labels = input_groups
        .next()
        .unwrap_or_default()
        .chars()
        .collect::<Vec<_>>();
    let rhs_labels = input_groups
        .next()
        .unwrap_or_default()
        .chars()
        .collect::<Vec<_>>();
    if input_groups.next().is_some()
        || lhs_labels.len() != lhs.shape.len()
        || rhs_labels.len() != rhs.shape.len()
    {
        return StaticValue::Unknown;
    }
    let mut extents = HashMap::<char, usize>::new();
    for (label, extent) in lhs_labels.iter().zip(&lhs.shape) {
        if let Some(existing) = extents.insert(*label, *extent)
            && existing != *extent
        {
            push_static_runtime_error(
                expr,
                errors,
                format!("einsum label `{label}` has inconsistent extents"),
            );
            return StaticValue::Unknown;
        }
    }
    for (label, extent) in rhs_labels.iter().zip(&rhs.shape) {
        if let Some(existing) = extents.insert(*label, *extent)
            && existing != *extent
        {
            push_static_runtime_error(
                expr,
                errors,
                format!("einsum label `{label}` has inconsistent extents"),
            );
            return StaticValue::Unknown;
        }
    }
    let mut out_shape = Vec::new();
    for label in output.chars() {
        let Some(extent) = extents.get(&label).copied() else {
            return StaticValue::Unknown;
        };
        out_shape.push(extent);
    }
    StaticValue::Tensor(StaticTensor {
        shape: out_shape,
        int_values: None,
    })
}

fn normalize_static_axis(rank: usize, axis: i64) -> Option<usize> {
    let rank = rank as i64;
    let axis = if axis < 0 { rank + axis } else { axis };
    (0..rank).contains(&axis).then_some(axis as usize)
}

/// Resolve one member of a two-axis builtin (`trace`, `diagonal`)
/// against the operand `tensor_ty`. Like [`resolve_builtin_axis`], a
/// negative literal indexes from the end; a still-out-of-range axis
/// pushes a diagnostic and returns `Err(())`. When the axis argument
/// is absent (not a literal) the historical positional `default` is
/// used so the prior `unwrap_or(0)` / `unwrap_or(1)` behavior is
/// preserved for the no-arg case.
fn resolve_axis_pair_member(
    op: &str,
    axis_expr: Option<&deep::Expr>,
    tensor_ty: &Type,
    default: usize,
    list: &deep::List,
    errors: &mut Vec<CheckError>,
) -> Result<usize, ()> {
    // Issue #216: use the cast-aware extractor so `cast(N, int32)`-wrapped
    // axis literals trip the infer-time bounds check instead of slipping
    // through to host-runtime defense-in-depth.
    match axis_expr.and_then(extract_int_for_dim) {
        Some(raw) => match tensor_ty {
            Type::Tensor(dims, _) => match normalize_static_axis(dims.len(), raw) {
                Some(axis) => Ok(axis),
                None => {
                    errors.push(CheckError::new(
                        CheckErrorKind::TypeMismatch,
                        with_macro_provenance(
                            &deep::Expr::List(list.clone(), zero_span()),
                            format!("{op} axis {raw} out of bounds for rank {}", dims.len()),
                        ),
                        vec![],
                    ));
                    Err(())
                }
            },
            // Non-tensor operand: keep a non-negative literal verbatim
            // and let the downstream shape checker surface the real
            // mismatch.
            _ => Ok(if raw >= 0 { raw as usize } else { default }),
        },
        None => Ok(default),
    }
}

/// Resolve the axis argument of an axis-taking builtin against the
/// operand `tensor_ty`, normalizing a negative literal to index from
/// the end (`-1` is the last axis). On an axis still out of range
/// after normalization, push an out-of-bounds diagnostic and return
/// `None` so the caller bails to `Type::Error`.
///
/// When the operand is not a concrete tensor or the axis is not a
/// literal, falls back to the prior lenient behavior: a non-negative
/// literal is taken verbatim, anything else defaults to `0` and the
/// downstream shape checker surfaces any real mismatch. The negative
/// normalization itself only needs the operand rank, which a
/// `Type::Tensor` always carries.
fn resolve_builtin_axis(
    op: &str,
    axis_expr: Option<&deep::Expr>,
    tensor_ty: &Type,
    list: &deep::List,
    errors: &mut Vec<CheckError>,
) -> Option<usize> {
    // Issue #216: cast-aware extractor; see `resolve_axis_pair_member`.
    let raw_axis = axis_expr.and_then(extract_int_for_dim);
    match (tensor_ty, raw_axis) {
        (Type::Tensor(dims, _), Some(raw)) => match normalize_static_axis(dims.len(), raw) {
            Some(axis) => Some(axis),
            None => {
                errors.push(CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!("{op} axis {raw} out of bounds for rank {}", dims.len()),
                    ),
                    vec![],
                ));
                None
            }
        },
        _ => Some(
            raw_axis
                .filter(|raw| *raw >= 0)
                .map(|raw| raw as usize)
                .unwrap_or(0),
        ),
    }
}

fn gather_result_shape(base: &[usize], indices: &[usize], axis: usize) -> Vec<usize> {
    let mut shape = Vec::with_capacity(base.len().saturating_sub(1) + indices.len());
    shape.extend_from_slice(&base[..axis]);
    shape.extend_from_slice(indices);
    shape.extend_from_slice(&base[axis + 1..]);
    shape
}

fn updates_shape_numel(shape: &[usize]) -> usize {
    shape.iter().copied().product::<usize>().max(1)
}

fn unravel_index(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0; shape.len()];
    for dim in (0..shape.len()).rev() {
        out[dim] = linear % shape[dim];
        linear /= shape[dim];
    }
    out
}

fn ravel_index(indices: &[usize], shape: &[usize]) -> usize {
    let mut flat = 0usize;
    let mut stride = 1usize;
    for (index, extent) in indices.iter().zip(shape.iter()).rev() {
        flat += index * stride;
        stride *= extent;
    }
    flat
}

fn push_static_runtime_error(expr: &deep::Expr, errors: &mut Vec<CheckError>, message: String) {
    errors.push(CheckError::new(
        CheckErrorKind::Other,
        with_macro_provenance(expr, message),
        vec![],
    ));
}

fn annotate_ir_program(exprs: &[deep::Expr]) -> Vec<deep::Expr> {
    // Preserve historical behavior: build a fresh annotation state from
    // `builtin_env()` + an EMPTY ADT registry (no prelude registration).
    // This is asymmetric with `infer_ir_program_with_env` (which DOES
    // register prelude ADTs), but downstream tooling — the host pipeline's
    // `expr_type` reader, eval-result root expansion via
    // `extend_root_names_from_value` — depends on this asymmetric type
    // metadata shape. Phase C does NOT touch this; the
    // `_with_context` variant accepts a non-empty context and uses
    // library state as-is.
    let (mut env, mut vg) = builtins::builtin_env();
    let mut subst = Subst::new();
    let mut adt_reg = AdtRegistry::new();
    builtins::register_prelude_adts(&mut env, &mut vg, &mut adt_reg);
    let mut declaration_errors = Vec::new();

    // Declared `defsig` parameter type expressions, visible to the
    // `def` arm of `annotate_expr_with_scope` for the duration of this
    // pass. Restored on drop.
    let _declared_sig_guard = install_declared_sig_param_types(exprs);
    let user_def_names = collect_user_def_names(&top_level_decl_items(exprs));

    // Descend through `(module {} name ...)` wrappers when collecting
    // declarations, matching `infer_program`. Without this, module-
    // wrapped user ADTs and defsigs never reach `env` / `adt_reg`
    // during annotation, so `pat-record`'s constructor lookup (#181)
    // and every other annotation-time env query for a user-declared
    // name silently misses. See `infer_program` for the parallel
    // iteration.
    collect_all_declarations(
        &top_level_decl_items_with_modules(exprs),
        &mut env,
        &mut vg,
        &mut subst,
        &mut adt_reg,
        &mut declaration_errors,
    );

    let detail_profile = std::env::var_os("CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL")
        .map(|v| v == "1")
        .unwrap_or(false);
    let mut annotated = Vec::with_capacity(exprs.len());
    for expr in exprs {
        let t0 = if detail_profile {
            Some(std::time::Instant::now())
        } else {
            None
        };
        // Run `infer_top_level` on the unit being annotated; for a
        // `(module {} ...)` wrapper, this means inferring every inner
        // decl before annotating the wrapper, so that annotation's
        // recursive walk sees fully-inferred bindings for each inner
        // decl. For a bare top-level decl, this preserves the original
        // per-expr alternation (infer THIS decl, then annotate THIS
        // decl) that signature-inference forward-reference assertions
        // depend on. (closes #181)
        let mut step_errors = Vec::new();
        let mut typed_nodes = 0;
        let mut total_nodes = 0;
        for decl in top_level_decl_items(std::slice::from_ref(expr)) {
            infer_top_level(
                decl,
                &mut env,
                &mut vg,
                &mut subst,
                &adt_reg,
                &mut step_errors,
                &mut typed_nodes,
                &mut total_nodes,
                &user_def_names,
            );
        }

        let annotated_expr = annotate_expr_with_scope(expr, &env, &vg, &subst, &adt_reg);
        if let Some(t0) = t0 {
            let elapsed = t0.elapsed();
            let label = top_level_decl_name(expr)
                .map(|s| s.to_string())
                .unwrap_or_else(|| "<anon>".to_string());
            eprintln!(
                "annotate_ir_decl: {:>8.4}s {}",
                elapsed.as_secs_f64(),
                label
            );
        }
        annotated.push(annotated_expr);
    }

    annotated
}

/// Annotate `exprs` with inferred types, using `context` as the outer
/// scope. Library schemes are visible during inference; only the
/// new-code exprs are returned in the annotated result.
///
/// When the context is empty (`library_def_count() == 0`) this falls
/// back to [`annotate_ir_program`] to preserve the legacy
/// "no-prelude" annotation shape that downstream tooling depends on.
/// Once a non-empty library context is supplied, the library's prelude
/// ADTs and decls are visible during annotation.
fn annotate_ir_program_with_context(context: &TypeEnv, exprs: &[deep::Expr]) -> Vec<deep::Expr> {
    if context.library_def_count() == 0 {
        return annotate_ir_program(exprs);
    }
    let mut state = context.inner().clone();
    let mut declaration_errors = Vec::new();

    // Declared `defsig` parameter type expressions for the new-code
    // exprs being annotated here. Restored on drop.
    let _declared_sig_guard = install_declared_sig_param_types(exprs);
    let user_def_names = collect_user_def_names(&top_level_decl_items(exprs));

    // Descend through `(module {} name ...)` wrappers when collecting
    // declarations; mirrors the `infer_program` shape and the parallel
    // fix in `annotate_ir_program`. (closes #181)
    collect_all_declarations(
        &top_level_decl_items_with_modules(exprs),
        &mut state.env,
        &mut state.var_gen,
        &mut state.subst,
        &mut state.adt_reg,
        &mut declaration_errors,
    );

    let detail_profile = std::env::var_os("CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL")
        .map(|v| v == "1")
        .unwrap_or(false);
    let mut annotated = Vec::with_capacity(exprs.len());
    for expr in exprs {
        let t0 = if detail_profile {
            Some(std::time::Instant::now())
        } else {
            None
        };
        // See the parallel comment in `annotate_ir_program`: infer
        // every inner decl of a module wrapper before annotating, so
        // user-declared ADT constructors and defsigs are visible at
        // annotation time; preserve per-expr alternation otherwise.
        let mut step_errors = Vec::new();
        let mut typed_nodes = 0;
        let mut total_nodes = 0;
        for decl in top_level_decl_items(std::slice::from_ref(expr)) {
            infer_top_level(
                decl,
                &mut state.env,
                &mut state.var_gen,
                &mut state.subst,
                &state.adt_reg,
                &mut step_errors,
                &mut typed_nodes,
                &mut total_nodes,
                &user_def_names,
            );
        }

        let annotated_expr = annotate_expr_with_scope(
            expr,
            &state.env,
            &state.var_gen,
            &state.subst,
            &state.adt_reg,
        );
        if let Some(t0) = t0 {
            let elapsed = t0.elapsed();
            // exprs here is at the module-wrapper level; pull a label.
            let label = top_level_decl_name(expr)
                .map(|s| s.to_string())
                .or_else(|| module_name(expr).map(|m| format!("module:{m}")))
                .unwrap_or_else(|| "<anon>".to_string());
            eprintln!(
                "annotate_ir_decl: {:>8.4}s {}",
                elapsed.as_secs_f64(),
                label
            );
        }
        annotated.push(annotated_expr);
    }

    annotated
}

/// Extract the module name from a `(module {} name ...)` expr for
/// profile labeling. Returns `None` if `expr` is not a module.
fn module_name(expr: &deep::Expr) -> Option<&str> {
    let deep::Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("module") {
        return None;
    }
    children(list).first().and_then(symbol_name)
}

fn annotate_expr_with_scope(
    expr: &deep::Expr,
    env: &Env,
    vg: &VarGen,
    subst: &Subst,
    adt_reg: &AdtRegistry,
) -> deep::Expr {
    // Bail value is the identity (unannotated) expr: annotation is a
    // best-effort pass and the entry boundary fails the check anyway.
    stack_guard!("annotate_expr_with_scope", expr, expr.clone());
    match expr {
        deep::Expr::Atom(_, _) => expr.clone(),
        deep::Expr::Map(map, span) => deep::Expr::Map(
            deep::MetaMap {
                entries: map
                    .entries
                    .iter()
                    .map(|(key, value)| {
                        (
                            key.clone(),
                            annotate_expr_with_scope(value, env, vg, subst, adt_reg),
                        )
                    })
                    .collect(),
            },
            *span,
        ),
        deep::Expr::MetaExpr(meta, span) => deep::Expr::MetaExpr(
            deep::MetaExpr {
                expr: Box::new(annotate_expr_with_scope(
                    &meta.expr, env, vg, subst, adt_reg,
                )),
                entries: meta
                    .entries
                    .iter()
                    .map(|(key, value)| {
                        (
                            key.clone(),
                            annotate_expr_with_scope(value, env, vg, subst, adt_reg),
                        )
                    })
                    .collect(),
            },
            *span,
        ),
        deep::Expr::List(list, span) => {
            if !matches!(list.elements.get(1), Some(deep::Expr::Map(_, _))) {
                return deep::Expr::List(
                    deep::List {
                        elements: list
                            .elements
                            .iter()
                            .map(|element| {
                                annotate_expr_with_scope(element, env, vg, subst, adt_reg)
                            })
                            .collect(),
                    },
                    *span,
                );
            }

            let tag = get_tag(list);
            let (annotated_children, fn_ty_override) = match tag {
                Some("fn") => {
                    let (kids, fn_ty) = annotate_fn_children(list, env, vg, subst, adt_reg, None);
                    (kids, Some(fn_ty))
                }
                // `(def name (fn ...))`: when the def has a separate
                // `defsig`, its declared parameter type expressions
                // (captured verbatim in `DECLARED_SIG_PARAM_TYPES`)
                // preserve `&` borrow wrappers that neither the bare
                // `fn` literal nor the inferred function type carry.
                // Stamp those onto the def's `(params ...)` node so a
                // standalone-lowered borrowed param keeps its borrow
                // and IR lowering sees a non-rank-0 type. Defs without
                // a `defsig` are absent from the map and fall through
                // to the plain recursion (bare params, owned by
                // `infer_signature_metadata`).
                Some("def") => {
                    let kids = children(list);
                    let declared_param_types =
                        kids.first().and_then(symbol_name).and_then(|name| {
                            DECLARED_SIG_PARAM_TYPES.with(|cell| cell.borrow().get(name).cloned())
                        });
                    let annotated: Vec<deep::Expr> = kids
                        .iter()
                        .map(|child| {
                            if let (Some(declared), deep::Expr::List(fn_list, fn_span)) =
                                (declared_param_types.as_ref(), child)
                                && get_tag(fn_list) == Some("fn")
                            {
                                let (fn_kids, fn_ty) = annotate_fn_children(
                                    fn_list,
                                    env,
                                    vg,
                                    subst,
                                    adt_reg,
                                    Some(declared),
                                );
                                let mut elements = vec![
                                    fn_list.elements[0].clone(),
                                    annotated_meta_map_with_override(
                                        fn_list,
                                        child,
                                        env,
                                        vg,
                                        subst,
                                        adt_reg,
                                        Some(fn_ty),
                                    ),
                                ];
                                elements.extend(fn_kids);
                                deep::Expr::List(deep::List { elements }, *fn_span)
                            } else {
                                annotate_expr_with_scope(child, env, vg, subst, adt_reg)
                            }
                        })
                        .collect();
                    (annotated, None)
                }
                Some("let") => (annotate_let_children(list, env, vg, subst, adt_reg), None),
                Some("match") => (annotate_match_children(list, env, vg, subst, adt_reg), None),
                _ => (
                    children(list)
                        .iter()
                        .map(|child| annotate_expr_with_scope(child, env, vg, subst, adt_reg))
                        .collect(),
                    None,
                ),
            };

            let mut elements = vec![
                list.elements[0].clone(),
                annotated_meta_map_with_override(
                    list,
                    expr,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    fn_ty_override,
                ),
            ];
            elements.extend(annotated_children);
            deep::Expr::List(deep::List { elements }, *span)
        }
    }
}

/// True for the `(t-var _)` placeholder that `desugar_fun_def` emits
/// in a synthesized sig for a parameter that had no declared type.
fn is_wildcard_tvar_expr(expr: &deep::Expr) -> bool {
    let deep::Expr::List(list, _) = expr else {
        return false;
    };
    get_tag(list) == Some("t-var") && children(list).first().and_then(symbol_name) == Some("_")
}

/// Rebuild a `(params ...)` node so every previously-bare parameter
/// symbol carries a `{type: ...}` metadata entry, using the declared
/// signature's parameter type *expressions* (`declared_param_type_exprs`).
///
/// The desugarer only attaches type metadata to parameters with an
/// inline annotation (`def f(x: T)`); parameters whose types come from
/// a separate `sig` declaration desugar to bare symbols. IR lowering's
/// `lower_fn` reads param types straight off this node, so without this
/// step a standalone-lowered def's params fall back to a rank-0
/// `default_type()` and any shape-sensitive op on them panics in
/// `tier2`.
///
/// The declared type *expressions* are copied verbatim, so `(t-ref ...)`
/// borrow wrappers survive intact (a `Type` round-trip via the inferred
/// function type drops them, which would make the linearity checker
/// treat a borrowed param as owned).
///
/// Parameters that already carry a type annotation are left untouched.
fn annotate_params_node(
    params_expr: &deep::Expr,
    declared_param_type_exprs: &[deep::Expr],
) -> deep::Expr {
    let deep::Expr::List(list, span) = params_expr else {
        return params_expr.clone();
    };
    if get_tag(list) != Some("params") {
        return params_expr.clone();
    }
    let mut elements = vec![list.elements[0].clone(), list.elements[1].clone()];
    for (index, param) in children(list).iter().enumerate() {
        match param {
            deep::Expr::Atom(deep::Atom::Symbol(name), atom_span) => {
                // A synthesized sig from `desugar_fun_def` uses
                // `(t-var _)` as the placeholder for a parameter with
                // no declared type. That is not a real declared type:
                // stamping it would pre-empt `infer_signature_metadata`'s
                // read-only/borrow inference, so the param is left bare.
                let declared = declared_param_type_exprs
                    .get(index)
                    .filter(|expr| !is_wildcard_tvar_expr(expr));
                match declared {
                    Some(type_expr) => {
                        elements.push(deep::Expr::List(
                            deep::List {
                                elements: vec![
                                    deep::Expr::Atom(deep::Atom::Symbol(name.clone()), *atom_span),
                                    deep::Expr::Map(
                                        deep::MetaMap {
                                            entries: vec![("type".to_string(), type_expr.clone())],
                                        },
                                        *atom_span,
                                    ),
                                ],
                            },
                            *atom_span,
                        ));
                    }
                    None => elements.push(param.clone()),
                }
            }
            // Already-typed params (MetaExpr / List forms) are left as-is.
            _ => elements.push(param.clone()),
        }
    }
    deep::Expr::List(deep::List { elements }, *span)
}

/// Annotate the children of a `(fn ...)` node.
///
/// `declared_param_type_exprs`, when present, is the parameter type
/// expression list from the enclosing def's declared `sig`. It is used
/// to stamp the `(params ...)` node, preserving `(t-ref ...)` borrow
/// wrappers verbatim. `fn` literals with no declared signature pass
/// `None` and keep bare params.
fn annotate_fn_children(
    list: &deep::List,
    env: &Env,
    vg: &VarGen,
    subst: &Subst,
    adt_reg: &AdtRegistry,
    declared_param_type_exprs: Option<&[deep::Expr]>,
) -> (Vec<deep::Expr>, Type) {
    let kids = children(list);
    if kids.is_empty() {
        return (vec![], Type::Error);
    }

    let fn_ty = infer_expr_in_scope(
        &deep::Expr::List(list.clone(), span_of_list(list)),
        env,
        vg,
        subst,
        adt_reg,
    );
    let resolved_fn_ty = subst.apply(&fn_ty);
    let param_types = match &resolved_fn_ty {
        Type::Fn(args, _) => args.clone(),
        _ => Vec::new(),
    };

    let mut param_vg = vg.clone();
    let raw_params = extract_params(&kids[0], &mut param_vg, adt_reg);
    // issue #319: when the def carries a separate `sig`, the `fn`
    // literal's params are bare symbols, so the inference above seeds
    // each with an unconstrained fresh tvar — the body's shape-sensitive
    // ops (`matmul`, `permute`) then annotate as bare type variables
    // rather than resolved tensor types. Recovering the declared param
    // types from `declared_param_type_exprs` and binding them into
    // `fn_env` lets the recursive body annotation resolve those ops to
    // their true `tensor[...]` shapes. Without this, IR lowering reads a
    // rank-0 `default_type()` off a `(t-var ...)` body-node annotation
    // and `tier2::lower_matmul` panics with `expects rank >= 2`.
    //
    // A SHARED `tvar_map`/`dvar_map` is used across every declared param
    // so a dim/precision variable that recurs across parameters (e.g.
    // `tensor[s, d, p]` for `q`, `k`, and `v`) maps to the SAME `DimVar`
    // / `TypeVar` — preserving the inter-parameter shape relationships
    // (`q: [s, d]`, `kt: [d, s]` ⇒ `matmul(q, kt): [s, s]`) that the
    // matmul typing rule depends on. `param_vg` (the cloned `VarGen`)
    // feeds fresh-var allocation so it does not perturb the caller's.
    //
    // Why clone rather than thread the caller's `vg` and advance its
    // counter (which would be collision-free and shrink the argument
    // below to a sentence): this is a post-inference ANNOTATION pass and
    // `vg: &VarGen` is borrowed SHARED here — advancing the caller's
    // counter is not even available without widening the whole annotation
    // call-chain to `&mut VarGen`, a far larger change for a pass whose
    // vars never escape. The throwaway clone is the correct local choice;
    // the confinement argument below is why the resulting ID overlap is
    // harmless.
    //
    // Var-ID overlap is harmless. These freshly-minted `TypeVar`s are
    // used ONLY to seed `fn_env` for the body-ANNOTATION pass below; the
    // annotation re-infers each body node's `type:` via
    // `infer_expr_in_scope`, which runs in its OWN throwaway `Subst`
    // (`infer_expr_in_scope` creates `Subst::new()`). Nothing from
    // `param_vg` flows back into the caller's `vg`/`subst` or the
    // program's global type state, so a `TypeVar(N)` minted here that
    // happens to collide numerically with a `TypeVar(N)` elsewhere never
    // unifies the two: the collision is confined to this one node's
    // annotation scope. (Re-using the already-resolved declared `Fn` type
    // — as the WS-A7 `infer_def_body_with_sig` inference path does — would
    // also work, but is not reachable from this post-inference annotation
    // pass, which has no access to that resolved type or the error
    // vector.)
    let declared_param_types: Vec<Option<Type>> = match declared_param_type_exprs {
        Some(declared) => {
            let mut tvar_map: HashMap<String, TypeVar> = HashMap::new();
            let mut dvar_map: HashMap<String, DimVar> = HashMap::new();
            // Tier-2 rank polymorphism (#286): a `..r` rank var recurring
            // across params must map to the SAME `RankVar`, so the rvar map
            // is shared across the declared params exactly like tvar/dvar.
            let mut rvar_map = HashMap::new();
            declared
                .iter()
                .map(|expr| {
                    if is_wildcard_tvar_expr(expr) {
                        None
                    } else {
                        match deep_type_to_type_inner(
                            expr,
                            &mut param_vg,
                            &mut tvar_map,
                            &mut dvar_map,
                            &mut rvar_map,
                        ) {
                            Type::Error => None,
                            ty => Some(ty),
                        }
                    }
                })
                .collect()
        }
        None => Vec::new(),
    };
    let mut fn_env = env.clone();
    for (index, (name, maybe_ty)) in raw_params.iter().enumerate() {
        let ty = maybe_ty
            .clone()
            .or_else(|| declared_param_types.get(index).cloned().flatten())
            .or_else(|| param_types.get(index).cloned())
            .unwrap_or(Type::Error);
        fn_env.bind(name.clone(), Scheme::mono(ty));
        // chelis#397/#469: a fresh parameter has no size provenance; clear any
        // entry inherited from an outer name it shadows (BLOCKER C).
        fn_env.clear_size_provenance(name);
    }

    // Only stamp parameter types when they come from the def's
    // declared `sig`. A bare `fn` literal with no declared signature
    // keeps bare params: the inferred function type loses `&` borrow
    // wrappers, and stamping it would also pre-empt the
    // read-only/borrow inference that `infer_signature_metadata`
    // performs on parameters left bare.
    let annotated_params = match declared_param_type_exprs {
        Some(declared) => annotate_params_node(&kids[0], declared),
        None => kids[0].clone(),
    };
    let mut result = vec![annotate_expr_with_scope(
        &annotated_params,
        env,
        vg,
        subst,
        adt_reg,
    )];
    if let Some(body) = kids.get(1) {
        result.push(annotate_expr_with_scope(body, &fn_env, vg, subst, adt_reg));
    }
    (result, resolved_fn_ty)
}

fn annotate_let_children(
    list: &deep::List,
    env: &Env,
    vg: &VarGen,
    subst: &Subst,
    adt_reg: &AdtRegistry,
) -> Vec<deep::Expr> {
    let kids = children(list);
    if kids.len() < 2 {
        return kids.to_vec();
    }

    let mut let_env = env.clone();
    let annotated_bind = if let deep::Expr::List(bind_list, bind_span) = &kids[0] {
        let bind_kids = children(bind_list);
        let mut bind_elements = vec![bind_list.elements[0].clone(), bind_list.elements[1].clone()];
        let mut i = 0;
        while i + 1 < bind_kids.len() {
            bind_elements.push(bind_kids[i].clone());
            let value = annotate_expr_with_scope(&bind_kids[i + 1], &let_env, vg, subst, adt_reg);
            let value_ty = infer_expr_in_scope(&bind_kids[i + 1], &let_env, vg, subst, adt_reg);
            if let Some(name) = symbol_name(&bind_kids[i]) {
                let_env.bind(name.to_string(), let_env.generalize(&value_ty, subst));
            }
            bind_elements.push(value);
            i += 2;
        }
        deep::Expr::List(
            deep::List {
                elements: bind_elements,
            },
            *bind_span,
        )
    } else {
        annotate_expr_with_scope(&kids[0], env, vg, subst, adt_reg)
    };

    vec![
        annotated_bind,
        annotate_expr_with_scope(&kids[1], &let_env, vg, subst, adt_reg),
    ]
}

fn annotate_match_children(
    list: &deep::List,
    env: &Env,
    vg: &VarGen,
    subst: &Subst,
    adt_reg: &AdtRegistry,
) -> Vec<deep::Expr> {
    let kids = children(list);
    if kids.is_empty() {
        return vec![];
    }

    let scrutinee = annotate_expr_with_scope(&kids[0], env, vg, subst, adt_reg);
    let scrutinee_ty = infer_expr_in_scope(&kids[0], env, vg, subst, adt_reg);
    let mut result = vec![scrutinee];

    for arm in &kids[1..] {
        if let deep::Expr::List(arm_list, arm_span) = arm
            && get_tag(arm_list) == Some("arm")
        {
            let arm_kids = children(arm_list);
            let mut arm_env = env.clone();
            // `pattern_vg` and `pattern_subst` are clones rather than
            // shared refs with the outer state. The clone is safe
            // because the primary inference pass (`infer_program` →
            // `infer_top_level` → `infer_match`) has already executed
            // `pattern_bindings` against the unshared outer `subst`,
            // populating it with the same type-parameter unifications
            // we're about to (re-)derive here. So the body annotation
            // below using the outer `subst` sees the same mappings the
            // pattern-binding stamper would have written to
            // `pattern_subst`. If a future caller invokes
            // `annotate_ir_program` against a `Subst` that hasn't been
            // pre-populated by `infer_program`, this invariant breaks
            // and body-annotation type variables go stale; that's a bug
            // in the caller, not here.
            let mut pattern_vg = vg.clone();
            let mut pattern_subst = subst.clone();
            let mut pattern_errors = Vec::new();
            let mut covered = Vec::new();
            let mut wildcard = false;
            if let Some(pattern) = arm_kids.first() {
                pattern_bindings(
                    pattern,
                    &scrutinee_ty,
                    &mut arm_env,
                    &mut pattern_vg,
                    &mut pattern_subst,
                    adt_reg,
                    &mut pattern_errors,
                    &mut covered,
                    &mut wildcard,
                );
            }

            let mut elements = vec![arm_list.elements[0].clone(), arm_list.elements[1].clone()];
            if let Some(pattern) = arm_kids.first() {
                // Stamp pattern-binding types onto `pat-var`/`pat-as`
                // nodes so the linearity checker (which consumes the
                // annotated Deep) can declare scope entries with the
                // resolved binding type rather than `None`. Without
                // this, a destructured tensor field's `&x` borrow
                // fails the linearity check because `expr_type` can't
                // resolve the binding's type. (closes #181)
                let annotated_pattern = annotate_expr_with_scope(pattern, env, vg, subst, adt_reg);
                let annotated_pattern =
                    stamp_pattern_binding_types(&annotated_pattern, &arm_env, &pattern_subst);
                elements.push(annotated_pattern);
            }
            if let Some(guard) = arm_kids.get(1) {
                elements.push(annotate_expr_with_scope(
                    guard, &arm_env, vg, subst, adt_reg,
                ));
            }
            if let Some(body) = arm_kids.get(2) {
                elements.push(annotate_expr_with_scope(body, &arm_env, vg, subst, adt_reg));
            }
            result.push(deep::Expr::List(deep::List { elements }, *arm_span));
            continue;
        }
        result.push(annotate_expr_with_scope(arm, env, vg, subst, adt_reg));
    }

    result
}

/// Walk a pattern AST and stamp the resolved binding type onto each
/// `pat-var` (and `pat-as`) node's metadata map under the `type` key.
///
/// The binding type is looked up by name in `arm_env`, which was just
/// populated by `pattern_bindings` against `pattern_subst`. We re-apply
/// `pattern_subst` here so any post-unify substitutions (e.g. the ADT
/// type-parameter pinning that happens when `pat-record` unifies the
/// constructor's return ADT against the scrutinee) flow into the
/// stamped metadata.
///
/// The downstream consumer is `linearity::check_match`, which reads
/// each pattern var's stamped `:type` to populate the arm's
/// `LinearScope`. Without this, destructured field bindings stay
/// untyped at linearity time and `&field` fails `expr_is_owned_or_borrow_linear`.
/// (closes #181)
fn stamp_pattern_binding_types(
    pat: &deep::Expr,
    arm_env: &Env,
    pattern_subst: &Subst,
) -> deep::Expr {
    stack_guard!("stamp_pattern_binding_types", pat, pat.clone());
    match pat {
        deep::Expr::Atom(_, _) | deep::Expr::Map(_, _) => pat.clone(),
        deep::Expr::MetaExpr(meta, span) => deep::Expr::MetaExpr(
            deep::MetaExpr {
                expr: Box::new(stamp_pattern_binding_types(
                    &meta.expr,
                    arm_env,
                    pattern_subst,
                )),
                entries: meta.entries.clone(),
            },
            *span,
        ),
        deep::Expr::List(list, span) => {
            let tag = get_tag(list);
            let kids = children(list);
            let needs_type_stamp = matches!(tag, Some("pat-var") | Some("pat-as"));

            // The binding's name lives at the first child for both
            // `pat-var` and `pat-as`. Other pattern tags carry no
            // direct binding here (their sub-patterns recurse).
            //
            // The `Type::Error` filter is intentional: when a pattern
            // earlier in the same arm raised an error (e.g., unknown
            // record field), `pattern_bindings` stores `Type::Error`
            // for the bind name. Stamping that onto the metadata would
            // round-trip through `type_to_deep_expr` as
            // `(t-var {} _)` (see line ~4962) and the linearity check
            // would read it as an opaque type variable, possibly
            // surfacing a cascading "borrow requires tensor or
            // tensor-carrying input, got ?N" on top of the original
            // unknown-field error. Suppressing the stamp here lets the
            // linearity check fall through to its `None`-typed path,
            // which already produces a cleaner "borrowed arguments
            // must be tensor or tensor-carrying values" diagnostic
            // without inventing a fictional type for the binding.
            let resolved_ty = if needs_type_stamp {
                kids.first()
                    .and_then(symbol_name)
                    .and_then(|name| arm_env.lookup(name))
                    .map(|scheme| pattern_subst.apply(&scheme.body))
                    .filter(|ty| !matches!(ty, Type::Error))
            } else {
                None
            };

            let meta_expr =
                match list.elements.get(1) {
                    Some(deep::Expr::Map(meta, meta_span)) => {
                        if let Some(ty) = resolved_ty.as_ref() {
                            let mut entries = meta.entries.clone();
                            let ty_expr = type_to_deep_expr(ty);
                            if let Some((_, existing)) =
                                entries.iter_mut().find(|(key, _)| key == "type")
                            {
                                *existing = ty_expr;
                            } else {
                                entries.push(("type".to_string(), ty_expr));
                            }
                            deep::Expr::Map(deep::MetaMap { entries }, *meta_span)
                        } else {
                            list.elements[1].clone()
                        }
                    }
                    _ => list.elements.get(1).cloned().unwrap_or_else(|| {
                        deep::Expr::Map(deep::MetaMap { entries: vec![] }, *span)
                    }),
                };

            let mut elements = vec![list.elements[0].clone(), meta_expr];
            for child in kids {
                elements.push(stamp_pattern_binding_types(child, arm_env, pattern_subst));
            }
            deep::Expr::List(deep::List { elements }, *span)
        }
    }
}

fn annotated_meta_map_with_override(
    list: &deep::List,
    expr: &deep::Expr,
    env: &Env,
    vg: &VarGen,
    subst: &Subst,
    adt_reg: &AdtRegistry,
    precomputed_ty: Option<Type>,
) -> deep::Expr {
    let meta_span = match list.elements.get(1) {
        Some(deep::Expr::Map(_, span)) => *span,
        _ => span_of_expr(expr),
    };
    let mut entries = get_meta(list)
        .map(|meta| meta.entries.clone())
        .unwrap_or_default();

    let ty_for_meta = if let Some(tag) = get_tag(list) {
        match (tag, precomputed_ty) {
            ("fn", Some(ty)) => Some(ty),
            (t, _) if should_attach_type_metadata(t) => {
                Some(infer_expr_in_scope(expr, env, vg, subst, adt_reg))
            }
            _ => None,
        }
    } else {
        None
    };

    if let Some(ty) = ty_for_meta
        && !matches!(ty, Type::Error)
    {
        let ty_expr = type_to_deep_expr(&ty);
        if let Some((_, existing)) = entries.iter_mut().find(|(key, _)| key == "type") {
            *existing = ty_expr;
        } else {
            entries.push(("type".to_string(), ty_expr));
        }
    }

    deep::Expr::Map(deep::MetaMap { entries }, meta_span)
}

fn infer_expr_in_scope(
    expr: &deep::Expr,
    env: &Env,
    vg: &VarGen,
    _subst: &Subst,
    adt_reg: &AdtRegistry,
) -> Type {
    let mut env = env.clone();
    let mut vg = vg.clone();
    let mut subst = Subst::new();
    let mut errors = Vec::new();
    let mut typed_nodes = 0;
    let mut total_nodes = 0;
    let ty = infer_expr(
        expr,
        &mut env,
        &mut vg,
        &mut subst,
        adt_reg,
        &mut errors,
        &mut typed_nodes,
        &mut total_nodes,
    );
    subst.apply(&ty)
}

fn should_attach_type_metadata(tag: &str) -> bool {
    !matches!(
        tag,
        "module"
            | "import"
            | "import-all"
            | "export"
            | "let"
            | "fn"
            | "var"
            | "tuple"
            | "tuple-get"
            | "defsig"
            | "deftype"
            | "typealias"
            | "variant"
            | "field"
            | "defdim"
            | "params"
            | "bind"
            | "kv"
            | "arm"
            | "effects"
            | "resource"
            | "pat-var"
            | "pat-lit"
            | "pat-ctor"
            | "pat-tuple"
            | "pat-record"
            | "pat-wild"
            | "pat-as"
            | "t-prim"
            | "t-fn"
            | "t-tensor"
            | "t-adt"
            | "t-var"
            | "t-ref"
            | "t-unit"
            | "t-tuple"
            | "d-name"
            | "d-var"
            | "d-lit"
    )
}

fn type_to_deep_expr(ty: &Type) -> deep::Expr {
    match ty {
        Type::Prim(prim) => node_expr("t-prim", vec![symbol_expr(prim.name())]),
        Type::Fn(args, ret) => {
            let mut children: Vec<deep::Expr> = args.iter().map(type_to_deep_expr).collect();
            children.push(type_to_deep_expr(ret));
            node_expr("t-fn", children)
        }
        Type::Ref(inner) => node_expr("t-ref", vec![type_to_deep_expr(inner)]),
        Type::Tensor(dims, prec) => {
            let mut children: Vec<deep::Expr> = dims.iter().map(dim_to_deep_expr).collect();
            children.push(match prec {
                TensorPrec::Concrete(p) => type_to_deep_expr(&Type::Prim(*p)),
                TensorPrec::Var(v) => node_expr("t-var", vec![symbol_expr(&format!("t{}", v.0))]),
            });
            node_expr("t-tensor", children)
        }
        Type::Adt(name, args) => {
            let mut children = vec![symbol_expr(name)];
            children.extend(args.iter().map(type_to_deep_expr));
            node_expr("t-adt", children)
        }
        Type::Var(var) => node_expr("t-var", vec![symbol_expr(&format!("t{}", var.0))]),
        Type::Tuple(types) => node_expr("t-tuple", types.iter().map(type_to_deep_expr).collect()),
        Type::Unit => node_expr("t-unit", vec![]),
        Type::Error => node_expr("t-var", vec![symbol_expr("_")]),
    }
}

fn dim_to_deep_expr(dim: &Dim) -> deep::Expr {
    match dim {
        Dim::Name(name) => node_expr("d-name", vec![symbol_expr(name)]),
        Dim::Var(var) => node_expr("d-var", vec![symbol_expr(&format!("d{}", var.0))]),
        Dim::Lit(value) => node_expr(
            "d-lit",
            vec![deep::Expr::Atom(deep::Atom::Int(*value), zero_span())],
        ),
        Dim::Wildcard => node_expr("d-name", vec![symbol_expr("*")]),
        Dim::Rank(rank) => node_expr("d-rank", vec![symbol_expr(&format!("r{}", rank.0))]),
    }
}

fn node_expr(tag: &str, children: Vec<deep::Expr>) -> deep::Expr {
    let mut elements = vec![
        symbol_expr(tag),
        deep::Expr::Map(deep::MetaMap::default(), zero_span()),
    ];
    elements.extend(children);
    deep::Expr::List(deep::List { elements }, zero_span())
}

fn symbol_expr(name: &str) -> deep::Expr {
    deep::Expr::Atom(deep::Atom::Symbol(name.to_string()), zero_span())
}

fn zero_span() -> Span {
    Span::new(0, 0)
}

fn span_of_expr(expr: &deep::Expr) -> Span {
    match expr {
        deep::Expr::Atom(_, span)
        | deep::Expr::List(_, span)
        | deep::Expr::Map(_, span)
        | deep::Expr::MetaExpr(_, span) => *span,
    }
}

fn span_of_list(list: &deep::List) -> Span {
    list.elements
        .first()
        .map(span_of_expr)
        .unwrap_or_else(zero_span)
}

fn ir_builtin_name(list: &deep::List) -> Option<&str> {
    let func_expr = list.elements.get(2)?;
    let func_list = match func_expr {
        deep::Expr::List(list, _) => list,
        _ => return None,
    };
    match (func_list.elements.first(), func_list.elements.get(2)) {
        (
            Some(deep::Expr::Atom(deep::Atom::Symbol(tag), _)),
            Some(deep::Expr::Atom(deep::Atom::Symbol(name), _)),
        ) if tag == "var" => Some(name.as_str()),
        _ => None,
    }
}

fn is_ir_shape_sensitive_builtin(name: &str) -> bool {
    matches!(
        name,
        "matmul"
            | "softmax"
            | "mean"
            | "layer_norm"
            | "conv2d"
            | "sum"
            | "max_reduce"
            | "min_reduce"
            | "prod_reduce"
            | "argmax_reduce"
            | "argmin_reduce"
            | "reshape"
            | "permute"
            | "expand"
            | "pad"
            | "shrink"
            | "stride"
    )
}

/// Is `name` a unary shape-passthrough op for the purposes of let-RHS
/// recognition? Must match the unary arm of
/// `derive_ir_builtin_output_type` so the failed-marker insertion in
/// the let arm covers the same surface as the type-derivation
/// passthrough recognition (issue #212 / RT-205 round-4).
fn is_ir_unary_shape_passthrough_builtin(name: &str) -> bool {
    matches!(
        name,
        "relu"
            | "tanh"
            | "sigmoid"
            | "gelu"
            | "silu"
            | "exp"
            | "log"
            | "neg"
            | "recip"
            | "sqrt"
            | "abs"
            | "sin"
            | "cos"
            | "tan"
            | "atan"
            | "floor"
            | "ceil"
            | "round"
            | "not"
            | "softmax"
    )
}

/// Is `name` a binary shape-passthrough op? Must match the binary arm
/// of `derive_ir_builtin_output_type` for the same reason as
/// `is_ir_unary_shape_passthrough_builtin` (issue #212 / RT-205
/// round-4).
fn is_ir_binary_shape_passthrough_builtin(name: &str) -> bool {
    matches!(
        name,
        "add"
            | "sub"
            | "mul"
            | "div"
            | "max_elem"
            | "min_elem"
            | "cmplt"
            | "lt"
            | "gt"
            | "gte"
            | "lte"
            | "eq"
            | "neq"
            | "and"
            | "or"
    )
}

/// Recognise a let-RHS expression as a "shape-sensitive form" for the
/// purposes of cascade-suppression marker insertion: either a direct
/// recognised shape-sensitive IR builtin, or a unary/binary shape-
/// passthrough wrapper around one (recursively). Peeks through
/// borrow wrappers like the rest of the validator.
///
/// Returns true when, structurally, this RHS shape COULD have a
/// derivable output type via `derive_ir_builtin_output_type`; the
/// caller pairs this with `derived.is_none()` to detect the "should
/// have derived but didn't" failure mode (issue #212 / RT-205
/// round-4). The decoupled structural check means we no longer
/// depend on whether the RHS validation pushed a diagnostic at this
/// level: cascade-suppressed intermediate let-binders are still
/// marked failed so the suppression propagates unboundedly down the
/// chain.
fn let_rhs_is_recognized_shape_sensitive(expr: &deep::Expr) -> bool {
    stack_guard!("let_rhs_is_recognized_shape_sensitive", expr, false);
    let inner = peel_borrow(expr);
    let deep::Expr::List(list, _) = inner else {
        return false;
    };
    if get_tag(list) != Some("app") {
        return false;
    }
    let Some(func_name) = ir_builtin_name(list) else {
        return false;
    };
    if is_ir_shape_sensitive_builtin(func_name) {
        return true;
    }
    if is_ir_unary_shape_passthrough_builtin(func_name)
        && let Some(arg) = list.elements.get(3)
    {
        return let_rhs_is_recognized_shape_sensitive(arg);
    }
    if is_ir_binary_shape_passthrough_builtin(func_name) {
        // Either operand being a recognised shape-sensitive form is
        // sufficient: the passthrough derivation uses the first
        // resolvable operand's type and falls through to the second,
        // so a failed inner shape-sensitive call on either side
        // means the whole RHS is structurally broken.
        if let Some(lhs) = list.elements.get(3)
            && let_rhs_is_recognized_shape_sensitive(lhs)
        {
            return true;
        }
        if let Some(rhs) = list.elements.get(4)
            && let_rhs_is_recognized_shape_sensitive(rhs)
        {
            return true;
        }
    }
    false
}

fn expr_type_expr(expr: &deep::Expr, type_env: &IrTypeEnv) -> Option<deep::Expr> {
    stack_guard!("expr_type_expr", expr, None);
    match expr {
        deep::Expr::List(list, _) => {
            if let Some(meta) = get_meta(list)
                && let Some((_, ty)) = meta.entries.iter().find(|(k, _)| k == "type")
            {
                return Some(ty.clone());
            }
            if get_tag(list) == Some("var")
                && let Some(name) = children(list).first().and_then(symbol_name)
            {
                return type_env.get(name).cloned();
            }
            None
        }
        deep::Expr::MetaExpr(meta, _) => expr_type_expr(&meta.expr, type_env),
        _ => None,
    }
}

fn extend_ir_env_with_fn_params(fn_list: &deep::List, type_env: &IrTypeEnv) -> IrTypeEnv {
    let mut scoped = type_env.clone();
    let Some(params_expr) = children(fn_list).first() else {
        return scoped;
    };
    let deep::Expr::List(params_list, _) = params_expr else {
        return scoped;
    };
    if get_tag(params_list) != Some("params") {
        return scoped;
    }
    for param in children(params_list) {
        let deep::Expr::List(param_list, _) = param else {
            continue;
        };
        let Some(name) = param_list.elements.first().and_then(symbol_name) else {
            continue;
        };
        let Some(meta) = get_meta(param_list) else {
            continue;
        };
        let Some((_, ty)) = meta.entries.iter().find(|(k, _)| k == "type") else {
            continue;
        };
        scoped.insert(name.to_string(), ty.clone());
    }
    scoped
}

fn expr_tensor_type_is_concrete(expr: &deep::Expr, type_env: &IrTypeEnv) -> bool {
    // Peel `(borrow {} ...)` so the idiomatic Surf borrow form does
    // not silently bypass the dim-concreteness check.
    arg_tensor_type_expr(expr, type_env)
        .map(|ty| type_expr_is_ir_concrete(&ty))
        .unwrap_or(false)
}

/// Check whether a conv2d input tensor argument is concrete in every
/// dimension EXCEPT axis 0 (batch). Per spec/05-risc-primitives.md
/// §4.5 the canonical signature is `tensor[batch, in_c, h, w, p]`
/// and `batch` is named, so symbolic-batch programs are first-class
/// (RT-205 round-3 F-C). The spatial dims and `in_c` must remain
/// concrete because they appear in the im2col/matmul lowering.
///
/// Returns true when the type resolves to a rank-4 tensor whose
/// axes 1, 2, 3 are all `Dim::Lit`. Axis 0 may be `Dim::Lit` or
/// `Dim::NonConcrete`. Returns false on unresolvable type or any
/// non-concrete axis other than 0.
fn conv2d_input_dims_concrete_modulo_batch(
    expr: Option<&deep::Expr>,
    type_env: &IrTypeEnv,
) -> bool {
    let Some(expr) = expr else {
        return false;
    };
    let Some(ty) = arg_tensor_type_expr(expr, type_env) else {
        return false;
    };
    let Some(dims) = tensor_dims_from_type_expr(&ty) else {
        // Not a tensor; fall back to scalar-prim check.
        return type_expr_is_ir_concrete(&ty);
    };
    if dims.len() != 4 {
        // Rank mismatch is reported separately; return true so the
        // rank-4 guard later in the validator can fire instead of
        // suppressing it with a metadata error.
        return true;
    }
    // axes 1, 2, 3 must be concrete; axis 0 (batch) may be symbolic.
    dims[1..].iter().all(|d| matches!(d, DeepDimKind::Lit(_)))
}

fn validate_ir_builtin_symbolic_requirements(
    list: &deep::List,
    func_name: &str,
    type_env: &IrTypeEnv,
    failed_let_names: &HashSet<String>,
    errors: &mut Vec<CheckError>,
) {
    match func_name {
        "conv2d" => validate_conv2d_symbolic_requirements(list, type_env, failed_let_names, errors),
        "mean" if ir_builtin_axis_dim(list, type_env, 0, 1) == Some(DeepDimKind::NonConcrete) => {
            errors.push(validator_error(
                CheckErrorKind::DimensionMismatch,
                list,
                "IR builtin `mean` requires a concrete reduced axis extent".to_string(),
                vec!["Use a concrete d-lit dimension on the reduced axis".to_string()],
            ));
        }
        "layer_norm" => {
            let x_dims = list
                .elements
                .get(3)
                .and_then(|expr| arg_tensor_type_expr(expr, type_env))
                .and_then(|ty| tensor_dims_from_type_expr(&ty));
            if matches!(
                x_dims.as_ref().and_then(|dims| dims.last()),
                Some(DeepDimKind::NonConcrete)
            ) {
                errors.push(validator_error(
                    CheckErrorKind::DimensionMismatch,
                    list,
                    "IR builtin `layer_norm` requires a concrete normalized axis extent"
                        .to_string(),
                    vec!["Use a concrete d-lit dimension for the final axis".to_string()],
                ));
            }
        }
        _ => {}
    }
}

/// Return `true` if any of `list`'s tensor arguments (positional 3, 4)
/// is a `(var <name>)` whose `name` is in `failed_let_names`. Used by
/// `validate_conv2d_symbolic_requirements` to suppress the cascade
/// diagnostic when a let-bound name's own derivation already emitted
/// the owning diagnostic (RT-205 round-2 F3).
fn conv2d_input_is_failed_let_name(list: &deep::List, failed_let_names: &HashSet<String>) -> bool {
    if failed_let_names.is_empty() {
        return false;
    }
    for arg in list.elements.iter().skip(3).take(2) {
        let inner = peel_borrow(arg);
        if let deep::Expr::List(arg_list, _) = inner
            && get_tag(arg_list) == Some("var")
            && let Some(name) = children(arg_list).first().and_then(symbol_name)
            && failed_let_names.contains(name)
        {
            return true;
        }
    }
    false
}

/// Build a `CheckError` for a validator-arm diagnostic that
/// references a specific call site. Appends the call site's `:span`
/// metadata identifier (if present) to the message so JSON consumers
/// can locate the offending expression in the source.
///
/// All shape-sensitive validator errors flow through this helper so
/// they uniformly get DimensionMismatch-grade severity and span
/// suffixes, matching the inference-layer DimensionMismatch surface
/// that JSON tooling already understands (RT-205 F6).
fn validator_error(
    kind: CheckErrorKind,
    call_site: &deep::List,
    message: String,
    suggestions: Vec<String>,
) -> CheckError {
    let suffixed = match validator_span_suffix(call_site) {
        Some(span) => format!("{message} {span}"),
        None => message,
    };
    CheckError::new(kind, suffixed, suggestions)
}

/// Render the call site's source span as a parenthesized suffix
/// (e.g. ` (at surf:144..165)`). Returns `None` when the call site
/// carries no `:span` metadata so the unmodified message is used.
fn validator_span_suffix(call_site: &deep::List) -> Option<String> {
    let meta = get_meta(call_site)?;
    for (key, value) in &meta.entries {
        if key == "span"
            && let deep::Expr::Atom(deep::Atom::Str(s), _) = value
        {
            return Some(format!("(at {s})"));
        }
    }
    None
}

/// Extract the `:span` metadata string from a list node, if present.
/// Used to propagate external span identifiers into check diagnostics.
fn list_span_id(list: &deep::List) -> Option<&str> {
    let meta = get_meta(list)?;
    for (key, value) in &meta.entries {
        if key == "span"
            && let deep::Expr::Atom(deep::Atom::Str(s), _) = value
        {
            return Some(s.as_str());
        }
    }
    None
}

/// Parse the start byte offset from a span identifier string.
/// Handles the `"surf:<start>..<end>"` format emitted by the desugar step
/// and bare `"<start>..<end>"` ranges.
fn parse_span_offset(span_id: &str) -> Option<usize> {
    // Format: "surf:10..25" or "10..25" or "octant:line:7" (opaque)
    let numeric_part = span_id
        .rfind(':')
        .map(|i| &span_id[i + 1..])
        .unwrap_or(span_id);
    // Try to parse "start..end"
    numeric_part
        .split_once("..")
        .and_then(|(start, _)| start.parse::<usize>().ok())
}

/// Validate the symbolic requirements of an IR-level `conv2d` call.
///
/// The previous implementation read `:type` from the app node's
/// metadata via `app_result_type_is_concrete` to decide whether the
/// output dims were concrete. Surf-desugared apps only carry `:span`
/// metadata; the annotation pass that would stamp inferred app types
/// back into Deep runs after `validate_ir_program`, so that check was
/// structurally always-false for any Surf source (see issue #186).
///
/// The replacement derives output concreteness from the arguments
/// (input tensor dims, kernel tensor dims, stride/padding literal
/// values), all of which are knowable at validation time. After
/// extracting the args this function evaluates the output spatial-
/// dim formula `floor((in + 2 * padding - kernel) / stride) + 1`
/// per axis (spec/05-risc-primitives.md §471-483) and rejects calls
/// whose evaluated output dim is non-positive. Also enforces rank-4
/// input/kernel and positive-stride / non-negative-padding.
fn validate_conv2d_symbolic_requirements(
    list: &deep::List,
    type_env: &IrTypeEnv,
    failed_let_names: &HashSet<String>,
    errors: &mut Vec<CheckError>,
) {
    // RT-205 round-2 F3: if either tensor arg is a `(var <name>)`
    // whose `name` is in the failed-derivation set, the owning
    // diagnostic was already emitted for the let-binding's own RHS.
    // Suppress the cascade so the user sees one error per root cause,
    // not one per consumer.
    if conv2d_input_is_failed_let_name(list, failed_let_names) {
        return;
    }
    // Args at elements[3]..[6] for the canonical 4-arg call shape:
    // (app {} (var conv2d) input kernel stride padding).
    //
    // RT-205 round-3 F-C: input axis 0 (batch) is allowed to be
    // NonConcrete per spec/05 §4.5, since it does not enter the
    // spatial-dim formula and conv2d's IR lowering can carry a
    // symbolic batch through. All OTHER input axes (in_c, h, w) and
    // all kernel axes must remain concrete -- they appear in the
    // im2col/matmul lowering and must be statically knowable.
    if !conv2d_input_dims_concrete_modulo_batch(list.elements.get(3).map(peel_borrow), type_env)
        || !expr_tensor_type_is_concrete(
            list.elements.get(4).expect("arity already implicit"),
            type_env,
        )
    {
        errors.push(validator_error(
            CheckErrorKind::DimensionMismatch,
            list,
            "IR builtin `conv2d` requires concrete tensor argument metadata".to_string(),
            vec![
                "Use concrete d-lit dimensions for IR lowering (axis 0 / batch may be symbolic)"
                    .to_string(),
            ],
        ));
        return;
    }
    // Extract and range-check stride/padding. The IR lowering relies
    // on these being statically-knowable positive (stride) or
    // non-negative (padding) integers; the output spatial dim formula
    // `floor((in + 2p - k) / s) + 1` (spec/05-risc-primitives.md
    // §471-483) divides by stride, so `stride <= 0` is undefined and
    // a negative padding shrinks the effective input below zero.
    // Without these guards the validator silently accepts the
    // ill-formed call and the back-end ICEs at codegen time
    // (issue #186 RT findings F1, F2, F3).
    let stride = match extract_typed_scalar_literal(list, 5, "stride", errors) {
        Some(v) => v,
        None => return,
    };
    if stride <= 0 {
        errors.push(validator_error(
            CheckErrorKind::DimensionMismatch,
            list,
            format!("IR builtin `conv2d` requires a positive stride, got {stride}"),
            vec!["Stride must be >= 1; the output dim formula divides by stride".to_string()],
        ));
        return;
    }
    let padding = match extract_typed_scalar_literal(list, 6, "padding", errors) {
        Some(v) => v,
        None => return,
    };
    if padding < 0 {
        errors.push(validator_error(
            CheckErrorKind::DimensionMismatch,
            list,
            format!("IR builtin `conv2d` requires non-negative padding, got {padding}"),
            vec!["Padding must be >= 0".to_string()],
        ));
        return;
    }
    // Resolve input + kernel tensor dims so we can evaluate the
    // output spatial-dim formula. expr_tensor_type_is_concrete above
    // already established concreteness; the lookups below should both
    // succeed, but bail gracefully on the unexpected case rather than
    // unwrap-panicking.
    let Some(input_dims) = list
        .elements
        .get(3)
        .and_then(|expr| arg_tensor_type_expr(expr, type_env))
        .and_then(|ty| tensor_dims_from_type_expr(&ty))
    else {
        return;
    };
    let Some(kernel_dims) = list
        .elements
        .get(4)
        .and_then(|expr| arg_tensor_type_expr(expr, type_env))
        .and_then(|ty| tensor_dims_from_type_expr(&ty))
    else {
        return;
    };
    // Rank guard: the canonical conv2d shape is [N, C, H, W] x [F, C, kH, kW].
    // The HM signature check (check_conv2d_signature, infer.rs:9999+) also
    // catches rank errors and may have already emitted its diagnostic via
    // `check_conv2d_signature`. Dedupe so the user sees ONE rank error per
    // role (input/kernel), not two (RT-205 round-2 F4).
    if input_dims.len() != 4 {
        let rank = input_dims.len();
        let hm_emitted = errors.iter().any(|e| {
            e.message.contains(&format!(
                "conv2d expects rank-4 input tensor, got rank {rank}"
            ))
        });
        if !hm_emitted {
            errors.push(validator_error(
                CheckErrorKind::DimensionMismatch,
                list,
                format!("IR builtin `conv2d` requires a rank-4 input tensor, got rank {rank}"),
                vec!["Pass a [N, C, H, W] tensor as the first argument".to_string()],
            ));
        }
        return;
    }
    if kernel_dims.len() != 4 {
        let rank = kernel_dims.len();
        let hm_emitted = errors.iter().any(|e| {
            e.message.contains(&format!(
                "conv2d expects rank-4 kernel tensor, got rank {rank}"
            ))
        });
        if !hm_emitted {
            errors.push(validator_error(
                CheckErrorKind::DimensionMismatch,
                list,
                format!("IR builtin `conv2d` requires a rank-4 kernel tensor, got rank {rank}"),
                vec!["Pass a [F, C, kH, kW] tensor as the second argument".to_string()],
            ));
        }
        return;
    }
    // Output spatial-dim formula per spec/05-risc-primitives.md §471-483:
    //   out = floor((in + 2 * padding - kernel) / stride) + 1
    // for both H (axis 2) and W (axis 3). If either evaluates to <= 0
    // the call is ill-formed; without this guard the back-end emits a
    // less-actionable error after codegen begins.
    let in_h = match input_dims[2] {
        DeepDimKind::Lit(v) => v,
        DeepDimKind::NonConcrete => return,
    };
    let in_w = match input_dims[3] {
        DeepDimKind::Lit(v) => v,
        DeepDimKind::NonConcrete => return,
    };
    let k_h = match kernel_dims[2] {
        DeepDimKind::Lit(v) => v,
        DeepDimKind::NonConcrete => return,
    };
    let k_w = match kernel_dims[3] {
        DeepDimKind::Lit(v) => v,
        DeepDimKind::NonConcrete => return,
    };
    for (axis, name, in_extent, k_extent) in [("H", "height", in_h, k_h), ("W", "width", in_w, k_w)]
        .iter()
        .map(|(short, long, inp, kr)| (*short, *long, *inp, *kr))
    {
        let Some(val) = conv2d_output_extent(in_extent, k_extent, stride, padding) else {
            errors.push(validator_error(
                CheckErrorKind::DimensionMismatch,
                list,
                format!(
                    "IR builtin `conv2d` output {name} (axis {axis}) cannot be computed: input={in_extent}, kernel={k_extent}, stride={stride}, padding={padding} overflows i64 in the canonical formula"
                ),
                vec![
                    "Use input/kernel/stride/padding values whose intermediate `input + 2 * padding - kernel` and final `+ 1` fit in a signed 64-bit integer".to_string(),
                ],
            ));
            return;
        };
        if val <= 0 {
            // Reaching this branch implies `conv2d_output_extent`
            // returned `Some(val)`, which in turn means
            // `padding.checked_mul(2)` and
            // `in_extent.checked_add(2 * padding)` both succeeded
            // upstream. Plain arithmetic is safe here; the
            // saturating-mul + checked-add fallback that earlier
            // code carried is unreachable. (RT-205 round-3 F-D.)
            let padded_hint = in_extent + 2 * padding;
            errors.push(validator_error(
                CheckErrorKind::DimensionMismatch,
                list,
                format!(
                    "IR builtin `conv2d` output {name} (axis {axis}) evaluates to {val} for input={in_extent}, kernel={k_extent}, stride={stride}, padding={padding}; output dims must be positive"
                ),
                vec![format!(
                    "Increase padding, decrease stride, or shrink the kernel so the padded input ({padded_hint}) is at least the kernel size ({k_extent})"
                )],
            ));
            return;
        }
    }
}

/// Compute the output spatial extent of a conv2d axis using the
/// canonical formula `floor((in + 2 * padding - kernel) / stride) + 1`
/// (spec/05-risc-primitives.md §471-483). Returns a signed value so
/// the validator can detect ill-formed configurations (output <= 0)
/// before they reach the back-end.
///
/// Uses `div_euclid` for floor division so a negative numerator (the
/// kernel does not fit the padded input) produces an informative
/// negative output value rather than truncating toward zero.
/// `stride` is required to be positive by the caller, which is what
/// makes `div_euclid` equivalent to mathematical floor here.
///
/// Returns `None` on integer overflow in any intermediate (RT-205
/// round-2 F1). Callers must treat `None` as "input parameters
/// outside the representable range" and emit a diagnostic; previously
/// a huge `padding` like `i64::MAX/2` triggered `attempt to multiply
/// with overflow` and panicked `chelis check`.
fn conv2d_output_extent(input: i64, kernel: i64, stride: i64, padding: i64) -> Option<i64> {
    let two_p = padding.checked_mul(2)?;
    let padded = input.checked_add(two_p)?;
    let numerator = padded.checked_sub(kernel)?;
    numerator.checked_div_euclid(stride)?.checked_add(1)
}

/// If `expr` is a recognizable shape-sensitive IR builtin call whose
/// output tensor type can be derived from its argument types and
/// literal scalar args, return that type as a Deep `(t-tensor ...)`
/// expression. Used to extend the validator's per-let-scope type env
/// so downstream uses of a let-bound name resolve to a concrete
/// tensor type (RT-205 F5).
///
/// In addition to `conv2d` direct calls, this also handles
/// shape-PRESERVING unary and binary point-wise ops (relu, tanh,
/// add, mul, etc.) so the canonical CNN layer pattern
/// `y = relu(conv2d(...))` chains correctly into a downstream
/// `conv2d(&y, ...)` (RT-205 round-2 F2). Reductions and movement
/// ops are intentionally NOT handled here; they would need a
/// separate per-op derivation because they change rank or shape.
///
/// Returns `None` when the call shape is unrecognized, the args are
/// non-concrete, or the derived output would be ill-formed (in which
/// case the validator's own arm will report the diagnostic).
fn derive_ir_builtin_output_type(expr: &deep::Expr, type_env: &IrTypeEnv) -> Option<deep::Expr> {
    let deep::Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("app") {
        return None;
    }
    let func_name = ir_builtin_name(list)?;
    match func_name {
        "conv2d" => derive_conv2d_output_type(list, type_env),
        // Shape-preserving unary point-wise: output type == input type.
        // Each entry below is cross-verified against the lowerer's
        // accepted name set in `crates/chelis-ir/src/lower.rs` (the
        // canonical IR vocabulary) and spec/05-risc-primitives.md
        // §2.2 / §3.3 (RT-205 round-3 F-B audit).
        //
        // Reductions (sum, mean, max_reduce, argmax_reduce,
        // prod_reduce, min_reduce, argmin_reduce) and movement ops
        // (reshape, permute, gather, pad, shrink, stride, expand) are
        // EXCLUDED: they change rank or shape and need per-op
        // derivation.
        //
        // softmax takes a (tensor, axis) tuple but its output shape
        // equals the input tensor's shape, so it fits the unary
        // passthrough path (positional [3] is the tensor).
        "relu" | "tanh" | "sigmoid" | "gelu" | "silu" | "exp" | "log" | "neg" | "recip"
        | "sqrt" | "abs" | "sin" | "cos" | "tan" | "atan" | "floor" | "ceil" | "round" | "not"
        | "softmax" => derive_unary_shape_passthrough(list, type_env),
        // Shape-preserving binary point-wise: output type == first
        // operand's type. Broadcasting cases are caught by HM
        // elsewhere; here we fall through to None if the first
        // operand's type is not derivable and try the second.
        //
        // RT-205 round-3 F-B: `maximum` and `minimum` were the wrong
        // names. The canonical IR names per spec/05 §2.1 and §3.4 are
        // `max_elem` (Tier 1) and `min_elem` (Tier 2). The lowerer
        // accepts `max_elem`/`min_elem` (lower.rs:1329-1330);
        // `maximum`/`minimum` do not appear anywhere in the IR
        // vocabulary, so the old allowlist never matched.
        //
        // `lt` is an alias for `cmplt` accepted at lowerer.rs:3913
        // (kept). `gte`, `lte`, `neq` are Tier 2 comparison ops
        // (spec/05 §3.2) accepted by the lowerer (lower.rs:1349-1352)
        // and added here so passthrough recognizes them. `and`, `or`
        // are bool binaries (lower.rs:1353-1354).
        "add" | "sub" | "mul" | "div" | "max_elem" | "min_elem" | "cmplt" | "lt" | "gt" | "gte"
        | "lte" | "eq" | "neq" | "and" | "or" => derive_binary_shape_passthrough(list, type_env),
        _ => None,
    }
}

/// Derive the output tensor type of a shape-preserving unary
/// point-wise call: it equals the type of the single argument.
/// Recurses through nested apps so e.g. `relu(conv2d(...))`
/// resolves to conv2d's derived output type, peeking through any
/// borrow wrapper as usual (RT-205 round-2 F2).
fn derive_unary_shape_passthrough(list: &deep::List, type_env: &IrTypeEnv) -> Option<deep::Expr> {
    let arg = list.elements.get(3)?;
    resolve_let_value_tensor_type(arg, type_env)
}

/// Derive the output tensor type of a shape-preserving binary
/// point-wise call: it equals the type of whichever operand is
/// concretely resolvable (typically the first). Broadcasting and
/// dtype-promotion cases are caught by HM elsewhere; this helper
/// only needs to surface a shape that the next validator arm can
/// inspect (RT-205 round-2 F2).
fn derive_binary_shape_passthrough(list: &deep::List, type_env: &IrTypeEnv) -> Option<deep::Expr> {
    let lhs = list.elements.get(3)?;
    if let Some(ty) = resolve_let_value_tensor_type(lhs, type_env) {
        return Some(ty);
    }
    let rhs = list.elements.get(4)?;
    resolve_let_value_tensor_type(rhs, type_env)
}

/// Resolve the tensor type expression of a let-binding RHS or any
/// nested sub-expression: try the borrow-aware var/lit lookup first,
/// and if that fails recurse into the sub-expression as another
/// recognized shape-sensitive call. Used by the unary and binary
/// passthrough helpers (RT-205 round-2 F2).
fn resolve_let_value_tensor_type(expr: &deep::Expr, type_env: &IrTypeEnv) -> Option<deep::Expr> {
    if let Some(ty) = arg_tensor_type_expr(expr, type_env) {
        return Some(ty);
    }
    // Peek through borrow before recursing in case a wrapper op
    // appears under an `&` borrow (uncommon but cheap).
    let inner = peel_borrow(expr);
    derive_ir_builtin_output_type(inner, type_env)
}

/// Derive a conv2d call's output tensor type (rank-4 `[N, F, outH, outW]`
/// with the input's precision) from its argument types and literal
/// stride/padding values. Returns `None` if any non-batch input dim
/// or any kernel dim is non-concrete, stride/padding are not int
/// literals, ranks are wrong, or the output dims would be non-positive.
///
/// RT-205 round-3 F-C: input axis 0 (batch) is allowed to be
/// `Dim::NonConcrete` per spec/05 §4.5. When the input batch is
/// symbolic, the synthesized output type preserves the input
/// tensor's raw batch-dim expression (e.g. `(d-name {} batch)`)
/// rather than forcing a `d-lit`. This lets downstream chained
/// conv2d calls resolve `&y` to the symbolic-batch type.
fn derive_conv2d_output_type(list: &deep::List, type_env: &IrTypeEnv) -> Option<deep::Expr> {
    let input_ty = list
        .elements
        .get(3)
        .and_then(|e| arg_tensor_type_expr(e, type_env))?;
    let kernel_ty = list
        .elements
        .get(4)
        .and_then(|e| arg_tensor_type_expr(e, type_env))?;
    let input_dims = tensor_dims_from_type_expr(&input_ty)?;
    let kernel_dims = tensor_dims_from_type_expr(&kernel_ty)?;
    if input_dims.len() != 4 || kernel_dims.len() != 4 {
        return None;
    }
    // Issue #216: cast-aware so `conv2d(x, k, cast(1, int32), cast(0, int32))`
    // surfaces the same derived output type as the bare-literal form.
    let stride = extract_int_for_dim(list.elements.get(5)?)?;
    let padding = extract_int_for_dim(list.elements.get(6)?)?;
    if stride <= 0 || padding < 0 {
        return None;
    }
    // Capture the input tensor's raw batch-dim Expr (axis 0) so a
    // symbolic batch can pass through verbatim into the synthesized
    // output type. axes 1-3 must be concrete literals (RT-205 r3 F-C).
    let input_dim_exprs = tensor_dim_exprs_from_type_expr(&input_ty)?;
    if input_dim_exprs.len() != 4 {
        return None;
    }
    let batch_dim_expr = input_dim_exprs[0].clone();
    let f = match kernel_dims[0] {
        DeepDimKind::Lit(v) => v,
        DeepDimKind::NonConcrete => return None,
    };
    let in_h = match input_dims[2] {
        DeepDimKind::Lit(v) => v,
        DeepDimKind::NonConcrete => return None,
    };
    let in_w = match input_dims[3] {
        DeepDimKind::Lit(v) => v,
        DeepDimKind::NonConcrete => return None,
    };
    let k_h = match kernel_dims[2] {
        DeepDimKind::Lit(v) => v,
        DeepDimKind::NonConcrete => return None,
    };
    let k_w = match kernel_dims[3] {
        DeepDimKind::Lit(v) => v,
        DeepDimKind::NonConcrete => return None,
    };
    // `conv2d_output_extent` returns None on integer overflow (RT-205
    // round-2 F1); in that case there's no valid output tensor type
    // to register, so the caller falls back to no extension and the
    // validator's own arm will emit the overflow diagnostic.
    let out_h = conv2d_output_extent(in_h, k_h, stride, padding)?;
    let out_w = conv2d_output_extent(in_w, k_w, stride, padding)?;
    if out_h <= 0 || out_w <= 0 {
        return None;
    }
    // Build `(t-tensor {} <batch-expr> (d-lit {} f) (d-lit {} out_h)
    // (d-lit {} out_w) <precision-expr>)` from the input's precision
    // and the captured batch-dim expression (which may be a symbolic
    // `(d-name {} ...)` per RT-205 r3 F-C).
    let prec_expr = tensor_precision_expr(&input_ty)?;
    Some(build_tensor_type_expr_with_batch(
        batch_dim_expr,
        &[f, out_h, out_w],
        prec_expr,
    ))
}

/// Return the raw Deep `Expr` for each dimension in a `(t-tensor {} dim1
/// dim2 ... prec)`. Unlike `tensor_dims_from_type_expr`, which returns a
/// `DeepDimKind` flattening, this preserves the original
/// `(d-name {} batch)` / `(d-var {} ...)` / `(d-lit {} N)` sub-expression
/// so the caller can carry it forward verbatim when synthesizing a
/// derived tensor type (RT-205 round-3 F-C, symbolic batch propagation).
fn tensor_dim_exprs_from_type_expr(expr: &deep::Expr) -> Option<Vec<deep::Expr>> {
    let list = match expr {
        deep::Expr::List(list, _) => list,
        _ => return None,
    };
    if get_tag(list) == Some("t-ref") {
        return children(list)
            .first()
            .and_then(tensor_dim_exprs_from_type_expr);
    }
    if get_tag(list) != Some("t-tensor") {
        return None;
    }
    let kids = children(list);
    if kids.is_empty() {
        return None;
    }
    Some(kids[..kids.len().saturating_sub(1)].to_vec())
}

/// Extract the precision sub-expression (last child) of a
/// `(t-tensor {} dim1 dim2 ... precision)` expression. Returns the
/// raw Deep `Expr` so it can be re-used unchanged when synthesizing
/// a derived tensor type.
fn tensor_precision_expr(ty: &deep::Expr) -> Option<deep::Expr> {
    let deep::Expr::List(list, _) = ty else {
        return None;
    };
    if get_tag(list) == Some("t-ref") {
        return children(list).first().and_then(tensor_precision_expr);
    }
    if get_tag(list) != Some("t-tensor") {
        return None;
    }
    let kids = children(list);
    kids.last().cloned()
}

/// Build a synthetic `(t-tensor {} <batch-dim-expr> (d-lit {} d1)
/// (d-lit {} d2) ... prec)`, placing a verbatim Deep expression at
/// axis 0 (the batch dim) and integer literals for the remaining
/// axes. Used to preserve symbolic batch (`(d-name {} batch)`) when
/// deriving a chained conv2d's output type (RT-205 round-3 F-C).
/// Spans are zeroed because the derived type is synthetic; downstream
/// lookups care only about the structural shape.
fn build_tensor_type_expr_with_batch(
    batch_dim: deep::Expr,
    other_dims: &[i64],
    prec: deep::Expr,
) -> deep::Expr {
    let zero = zero_span();
    let empty_meta = || deep::MetaMap { entries: vec![] };
    let make_d_lit = |v: i64| {
        deep::Expr::List(
            deep::List {
                elements: vec![
                    deep::Expr::Atom(deep::Atom::Symbol("d-lit".to_string()), zero),
                    deep::Expr::Map(empty_meta(), zero),
                    deep::Expr::Atom(deep::Atom::Int(v), zero),
                ],
            },
            zero,
        )
    };
    let mut elements = vec![
        deep::Expr::Atom(deep::Atom::Symbol("t-tensor".to_string()), zero),
        deep::Expr::Map(empty_meta(), zero),
    ];
    elements.push(batch_dim);
    for &d in other_dims {
        elements.push(make_d_lit(d));
    }
    elements.push(prec);
    deep::Expr::List(deep::List { elements }, zero)
}

/// Look up positional arg `idx` of a `conv2d` call, attempt to
/// extract it as an integer literal, and emit a clear diagnostic if
/// the arg is missing or non-literal.
///
/// `label` names the role (`"stride"` / `"padding"`) for the error
/// message. Returns `Some(value)` on success and `None` when an error
/// was pushed (the caller should bail to avoid piling on cascading
/// diagnostics).
fn extract_typed_scalar_literal(
    list: &deep::List,
    idx: usize,
    label: &str,
    errors: &mut Vec<CheckError>,
) -> Option<i64> {
    let Some(arg) = list.elements.get(idx) else {
        // Arity mismatch is caught elsewhere; bail without piling on.
        return None;
    };
    // Issue #216: cast-aware so a cast-wrapped literal (e.g.
    // `conv2d(x, k, cast(0, int32), 0)`) lands the precise
    // positive-stride / non-negative-padding diagnostic instead of the
    // misleading "requires a literal integer stride" message that
    // pre-fix appeared whenever the literal was wrapped.
    match extract_int_for_dim(arg) {
        Some(v) => Some(v),
        None => {
            errors.push(validator_error(
                CheckErrorKind::DimensionMismatch,
                list,
                format!("IR builtin `conv2d` requires a literal integer {label}"),
                vec![format!(
                    "Pass `{label}` as a constant int literal, not a variable or expression"
                )],
            ));
            None
        }
    }
}

fn ir_builtin_axis_dim(
    list: &deep::List,
    type_env: &IrTypeEnv,
    tensor_arg_index: usize,
    axis_arg_index: usize,
) -> Option<DeepDimKind> {
    let tensor_dims = list
        .elements
        .get(3 + tensor_arg_index)
        .and_then(|expr| arg_tensor_type_expr(expr, type_env))
        .and_then(|ty| tensor_dims_from_type_expr(&ty))?;
    // Negative axes index from the end; normalize against the operand
    // rank so this concrete-extent check inspects the same axis the op
    // actually reduces.
    // Issue #216: cast-aware so a `cast(N, int32)`-wrapped axis arg
    // still resolves through to the operand's concrete dim.
    let raw_axis = list
        .elements
        .get(3 + axis_arg_index)
        .and_then(extract_int_for_dim)?;
    let axis = normalize_static_axis(tensor_dims.len(), raw_axis)?;
    tensor_dims.get(axis).copied()
}

/// Resolve the tensor type expression of a callsite argument, peeking
/// through a `(borrow {} <inner>)` wrapper if present.
///
/// Surf source idiomatically passes tensors to shape-sensitive IR
/// builtins via borrows (e.g. the `School.Nn.Conv.conv2d_small` sig
/// requires `&tensor[...]`). The validator's lookup helpers need to
/// see through that wrapper to find the underlying tensor type in the
/// IR type environment; otherwise the dim-concreteness checks in the
/// `conv2d`, `mean`, and `layer_norm` arms silently no-op on borrowed
/// inputs (see issue #186).
fn arg_tensor_type_expr(expr: &deep::Expr, type_env: &IrTypeEnv) -> Option<deep::Expr> {
    let inner = peel_borrow(expr);
    expr_type_expr(inner, type_env)
}

fn peel_borrow(expr: &deep::Expr) -> &deep::Expr {
    // Recurses only through nested `borrow` wrappers (shallow in practice),
    // but guarded for uniformity; bail value is the identity input.
    stack_guard!("peel_borrow", expr, expr);
    if let deep::Expr::List(list, _) = expr
        && get_tag(list) == Some("borrow")
        && let Some(child) = children(list).first()
    {
        return peel_borrow(child);
    }
    expr
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeepDimKind {
    Lit(i64),
    NonConcrete,
}

fn tensor_dims_from_type_expr(expr: &deep::Expr) -> Option<Vec<DeepDimKind>> {
    let list = match expr {
        deep::Expr::List(list, _) => list,
        _ => return None,
    };
    if get_tag(list) == Some("t-ref") {
        return children(list).first().and_then(tensor_dims_from_type_expr);
    }
    if get_tag(list) != Some("t-tensor") {
        return None;
    }
    let kids = children(list);
    if kids.is_empty() {
        return None;
    }
    let mut dims = Vec::new();
    for kid in &kids[..kids.len().saturating_sub(1)] {
        dims.push(match kid {
            deep::Expr::List(dim_list, _) if get_tag(dim_list) == Some("d-lit") => {
                match children(dim_list).first() {
                    Some(deep::Expr::Atom(deep::Atom::Int(n), _)) => DeepDimKind::Lit(*n),
                    _ => DeepDimKind::NonConcrete,
                }
            }
            _ => DeepDimKind::NonConcrete,
        });
    }
    Some(dims)
}

fn type_expr_is_ir_concrete(expr: &deep::Expr) -> bool {
    match expr {
        deep::Expr::List(list, _) if get_tag(list) == Some("t-prim") => true,
        _ => tensor_dims_from_type_expr(expr)
            .map(|dims| dims.iter().all(|d| matches!(d, DeepDimKind::Lit(_))))
            .unwrap_or(false),
    }
}

// ── Helpers ──────────────────────────────────────────────────────

fn get_tag(list: &deep::List) -> Option<&str> {
    if let Some(deep::Expr::Atom(deep::Atom::Symbol(tag), _)) = list.elements.first() {
        Some(tag.as_str())
    } else {
        None
    }
}

fn children(list: &deep::List) -> &[deep::Expr] {
    if list.elements.len() > 2 {
        &list.elements[2..]
    } else {
        &[]
    }
}

fn is_numeric_literal_expr(expr: &deep::Expr) -> bool {
    match expr {
        deep::Expr::Atom(deep::Atom::Float(_) | deep::Atom::Int(_), _) => true,
        deep::Expr::List(list, _) if get_tag(list) == Some("lit") => matches!(
            list.elements.get(2),
            Some(deep::Expr::Atom(
                deep::Atom::Float(_) | deep::Atom::Int(_),
                _
            ))
        ),
        _ => false,
    }
}

/// Get metadata map from element[1] of a list.
fn get_meta(list: &deep::List) -> Option<&deep::MetaMap> {
    if list.elements.len() > 1
        && let deep::Expr::Map(meta, _) = &list.elements[1]
    {
        return Some(meta);
    }
    None
}

/// True when a `deftype` node carries `opaque: true` metadata
/// (RFC D-META; the key is unprefixed language semantics).
fn deftype_opaque_meta(list: &deep::List) -> bool {
    get_meta(list).is_some_and(|meta| {
        meta.entries.iter().any(|(key, value)| {
            key == "opaque" && matches!(value, deep::Expr::Atom(deep::Atom::Bool(true), _))
        })
    })
}

/// Extract a symbol name from an Expr.
fn symbol_name(expr: &deep::Expr) -> Option<&str> {
    match expr {
        deep::Expr::Atom(deep::Atom::Symbol(s), _) => Some(s.as_str()),
        _ => None,
    }
}

fn with_macro_provenance(expr: &deep::Expr, message: String) -> String {
    let Some(source) = macro_source(expr) else {
        return message;
    };
    format!("{message} (in expansion of {source})")
}

// chelis#317 constructor-scope invariant (read before touching the helpers
// below). The reef name resolver guarantees that every constructor reference
// which is genuinely *in scope* — declared in the current module, imported by
// name, or module-qualified — reaches the type checker rewritten to its exact
// reef-mangled name (`Pkg__Mod__Ctor`), and the matching `deftype` registers
// that exact name in both the type env and the ADT registry. A bare,
// un-mangled constructor name therefore arrives at type-check ONLY when the
// importing module never brought it into scope (a type-only import, or no
// import at all). The `lookup_terminal_unique` / `lookup_variant_terminal_unique`
// fuzzy fallbacks are diagnostic-only: they exist so a partially-mangled or
// out-of-scope name can be *named* in an error, never to bind a reference to a
// scope. The guards below depend on this: an in-scope constructor is always
// exact-bound, so rejecting a name that resolves only through the fuzzy
// fallback (or not at all) cannot reject an in-scope constructor. A future
// half-mangled producer (mangled `deftype`, bare reference) would violate the
// invariant and be wrongly rejected here — which is the intended failure mode:
// a half-mangled program is a structural defect, not a valid reference. The
// `ir_resolves_consistently_mangled_constructor_names` /
// `ir_rejects_out_of_scope_terminal_constructor_name` unit tests pin both
// directions.

/// The terminal (last) segment of a possibly module-qualified or
/// reef-mangled name. Mirrors `env::terminal_name` / `adt::terminal_name`:
/// `Pkg__Demo__Adt__Alpha` and `Demo.Adt.Alpha` both have terminal `Alpha`.
fn terminal_segment(name: &str) -> &str {
    name.rsplit_once("__")
        .map(|(_, tail)| tail)
        .or_else(|| name.rsplit_once('.').map(|(_, tail)| tail))
        .unwrap_or(name)
}

/// Whether `name` is an ADT constructor reference by Chelis nomenclature:
/// its terminal segment starts with an uppercase ASCII letter (§3.1, the
/// same rule the surf parser uses to classify a bare uppercase identifier
/// as `Expr::Constructor` / `Pattern::Constructor`). Type names are also
/// PascalCase, but they never reach value-position `var`/`pat-ctor`
/// resolution, so an uppercase terminal in those positions is a
/// constructor.
fn is_constructor_name(name: &str) -> bool {
    terminal_segment(name)
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_uppercase())
}

/// A constructor reference is *in scope* only when its exact name is bound
/// in `env` — either a bare builtin constructor (`Some`/`None`/`Cons`/`Nil`,
/// registered bare by `register_prelude_adts`) or a reef-mangled in-scope
/// constructor (chelis#157/#316 rewrite the reference to its mangled name
/// when the importing module declares it locally or imports it by name).
///
/// Returns `true` when `name` looks like a constructor (PascalCase terminal)
/// but is *not* bound exactly and is *only* reachable through the registry's
/// fuzzy terminal-segment fallback (`lookup_terminal_unique`). That fallback
/// is exactly the silent cross-module mis-resolution chelis#317 reports: a
/// type-only import leaves the bare constructor un-rewritten, and the fuzzy
/// match binds it to another module's mangled tag, deferring the failure to
/// a runtime non-exhaustive match. Such a reference must be rejected at
/// `check` as an unknown constructor instead.
fn constructor_out_of_scope(name: &str, env: &Env) -> bool {
    is_constructor_name(name)
        && env.lookup(name).is_none()
        && env.lookup_terminal_unique(name).is_some()
}

/// Pattern-position counterpart of [`constructor_out_of_scope`]. A constructor
/// **pattern** head (`| Alpha =>`, `| Alpha { .. } =>`) is in scope only when it
/// resolves through an *exact* binding — either the type env (`env.lookup`, for
/// builtins and reef-mangled in-scope constructors) or the ADT registry
/// (`adt_reg.lookup_variant`, the exact mangled variant key). The terminal-unique
/// fallbacks (`env.lookup_terminal_unique` / `lookup_variant_terminal_unique`)
/// are diagnostic-only fuzzy matches, never an in-scope binding.
///
/// Returns `true` when `name` is a PascalCase constructor that resolves through
/// *neither* exact path. This rejects two out-of-scope cases the bare
/// [`constructor_out_of_scope`] env check misses for patterns (chelis#317):
///
///   1. unique fuzzy — exactly one foreign same-terminal variant exists, so a
///      bare `| Alpha =>` would fuzzy-bind to it; and
///   2. non-unique / unresolvable — two foreign modules export a same-terminal
///      `Dup`, so `lookup_*_terminal_unique` returns `None` and the arm would
///      otherwise push the bare name into `covered_variants` with no scheme and
///      no diagnostic. Normally that surfaces as `NonExhaustiveMatch`, but a `_`
///      wildcard arm (`has_wildcard`) suppresses exhaustiveness and the bogus
///      out-of-scope arm is silently accepted. Rejecting here closes that hole.
///
/// Soundness depends on the reef rewriter guaranteeing every genuinely in-scope
/// constructor reaches type-check exact-bound under its mangled name (see the
/// module-level note on the terminal-unique fallback). A future half-mangled
/// producer (mangled deftype, bare reference) would be wrongly rejected here —
/// which is the intended failure mode: a half-mangled program is a structural
/// defect, not a valid in-scope reference.
fn constructor_pattern_out_of_scope(name: &str, env: &Env, adt_reg: &AdtRegistry) -> bool {
    is_constructor_name(name)
        && env.lookup(name).is_none()
        && adt_reg.lookup_variant(name).is_none()
}

fn check_error_kind_from_type_error_kind(kind: &TypeErrorKind) -> CheckErrorKind {
    match kind {
        TypeErrorKind::TypeMismatch => CheckErrorKind::TypeMismatch,
        TypeErrorKind::PrecisionMismatch => CheckErrorKind::PrecisionMismatch,
        TypeErrorKind::DimensionMismatch => CheckErrorKind::DimensionMismatch,
        TypeErrorKind::ArityMismatch => CheckErrorKind::ArityMismatch,
        TypeErrorKind::OccursCheck => CheckErrorKind::OccursCheck,
        TypeErrorKind::NotAFunction => CheckErrorKind::NotAFunction,
    }
}

fn collection_helper_type_error(
    expr: &deep::Expr,
    helper: &str,
    contract: &str,
    te: TypeError,
) -> CheckError {
    let kind = check_error_kind_from_type_error_kind(&te.kind);
    let suggestions = match kind {
        CheckErrorKind::PrecisionMismatch => vec!["Insert explicit cast".to_string()],
        _ => vec![],
    };
    CheckError::new(
        kind,
        with_macro_provenance(expr, format!("{helper} {contract}; {}", te.message)),
        suggestions,
    )
}

fn extract_string_literal(expr: &deep::Expr) -> Option<String> {
    match expr {
        deep::Expr::Atom(deep::Atom::Str(value), _) => Some(value.clone()),
        deep::Expr::List(list, _) if get_tag(list) == Some("lit") => {
            children(list).first().and_then(|child| match child {
                deep::Expr::Atom(deep::Atom::Str(value), _) => Some(value.clone()),
                _ => None,
            })
        }
        _ => None,
    }
}

/// Narrow `Dim::Wildcard` slots in `ty` against the matching positions in
/// `template`, replacing each Wildcard with the template's concrete dim
/// where one is available. Used after a defsig unify to ensure the scheme
/// registered for callers reflects the declared concrete shape rather than
/// the body's permissive wildcards (#39).
///
/// This is structural and conservative: it only walks shapes that match
/// (same rank for tensors, same arity for fn/tuple/adt), and only narrows
/// Wildcard → a safe template dim. If shapes don't line up, the input is
/// returned unchanged so genuine type errors flagged by `unify` aren't
/// masked.
///
/// Which template dims are "safe" to substitute:
///
/// - `Dim::Lit(_)` — always safe. Concrete literals are self-contained.
///
/// - `Dim::Var(v)` — safe ONLY when `v` is also bound by a *parameter*
///   tensor-dim position of the declared signature (`param_dvars`). This
///   is the `const_col[n](spots: tensor[n, f32], ..) -> tensor[n, 1]`
///   family: the body's `to_tensor(map(..))` return is `tensor[*, 1]`,
///   but the declared return dim `n` is the same dim var as the `spots`
///   parameter's axis-0, so the caller binds `n` from its actual argument.
///   Narrowing `*` → `Var(n)` reconnects the return to the input dim,
///   which is what downstream lowering needs so a `vmap` lane kernel reads
///   a Load-declared batch dim instead of an undeclared `_anon_dim`
///   (chelis#405 / WS-3 build ICE).
///
///   A *return-only* dim var (one that appears in the declared return but
///   in NO parameter tensor position — e.g. `arange[n](start: int32,
///   stop: int32) -> tensor[n, int32]`, where the length comes from a
///   value parameter) is NOT in `param_dvars` and is deliberately left as
///   `Wildcard`. Baking such an unbound var into the generalized scheme is
///   the red-team RT-39+44 soundness regression (commit 8067c9ce): it
///   leaks a free dim var into callers and breaks the chelis-std self-test
///   corpus. The `param_dvars` gate is exactly the line between "the
///   caller supplies this dim" (safe) and "this dim is output-inferred /
///   value-parameter-derived" (unsafe).
///
/// - `Dim::Name`, `Dim::Rank`, existing `Var`/`Lit` in `ty` — preserved.
fn narrow_wildcards_with(ty: &Type, template: &Type, param_dvars: &HashSet<DimVar>) -> Type {
    match (ty, template) {
        (Type::Tensor(dims, prec), Type::Tensor(tmpl_dims, _)) if dims.len() == tmpl_dims.len() => {
            let new_dims = dims
                .iter()
                .zip(tmpl_dims.iter())
                .map(|(d, t)| match (d, t) {
                    (Dim::Wildcard, Dim::Lit(_)) => t.clone(),
                    (Dim::Wildcard, Dim::Var(v)) if param_dvars.contains(v) => t.clone(),
                    _ => d.clone(),
                })
                .collect();
            Type::Tensor(new_dims, prec.clone())
        }
        (Type::Fn(args, ret), Type::Fn(t_args, t_ret)) if args.len() == t_args.len() => {
            let new_args = args
                .iter()
                .zip(t_args.iter())
                .map(|(a, t)| narrow_wildcards_with(a, t, param_dvars))
                .collect();
            let new_ret = Box::new(narrow_wildcards_with(ret, t_ret, param_dvars));
            Type::Fn(new_args, new_ret)
        }
        (Type::Tuple(ts), Type::Tuple(t_ts)) if ts.len() == t_ts.len() => {
            let new_ts = ts
                .iter()
                .zip(t_ts.iter())
                .map(|(t, tt)| narrow_wildcards_with(t, tt, param_dvars))
                .collect();
            Type::Tuple(new_ts)
        }
        (Type::Adt(n, args), Type::Adt(_, t_args)) if args.len() == t_args.len() => {
            let new_args = args
                .iter()
                .zip(t_args.iter())
                .map(|(a, t)| narrow_wildcards_with(a, t, param_dvars))
                .collect();
            Type::Adt(n.clone(), new_args)
        }
        _ => ty.clone(),
    }
}

/// Collect the set of dim vars that occur in a *parameter* (non-return)
/// tensor-dim position of a resolved declared `Fn` signature. These are
/// the dims a caller binds from its actual arguments; the
/// `narrow_wildcards_with` gate uses this set to decide when a body
/// wildcard may be safely narrowed to a declared `Dim::Var`. A non-`Fn`
/// type (or one whose params carry no tensor dim vars) yields the empty
/// set, so narrowing falls back to the literal-only behavior.
fn param_bound_dvars(decl_ty: &Type) -> HashSet<DimVar> {
    let mut out = HashSet::new();
    if let Type::Fn(params, _) = decl_ty {
        for param in params {
            for dv in crate::env::free_dvars(param) {
                out.insert(dv);
            }
        }
    }
    out
}

fn tensor_concat_result_type(element_ty: &Type) -> Result<Type, String> {
    let Type::Tensor(dims, precision) = element_ty else {
        return Err(format!(
            "concat expects List[tensor[...]] for tensor concatenation, got {element_ty}"
        ));
    };
    if dims.is_empty() {
        return Err("concat expects tensor inputs with at least one axis".to_string());
    }
    let mut out_dims = dims.clone();
    let last_axis = out_dims.len() - 1;
    out_dims[last_axis] = Dim::Wildcard;
    Ok(Type::Tensor(out_dims, precision.clone()))
}

fn infer_gather_result_type(
    tensor_ty: &Type,
    indices_ty: &Type,
    axis: usize,
) -> Result<Type, String> {
    let Type::Tensor(tensor_dims, tensor_precision) = tensor_ty else {
        return Err(format!("gather expects tensor input, got {tensor_ty}"));
    };
    let Type::Tensor(index_dims, index_precision) = indices_ty else {
        return Err(format!(
            "gather expects integer tensor indices, got {indices_ty}"
        ));
    };
    if !index_precision.is_integer() {
        return Err(format!(
            "gather expects integer tensor indices, got tensor[..., {}]",
            index_precision.name()
        ));
    }
    if axis >= tensor_dims.len() {
        return Err(format!(
            "gather axis {axis} out of bounds for rank {}",
            tensor_dims.len()
        ));
    }
    let mut out_dims = tensor_dims[..axis].to_vec();
    out_dims.extend(index_dims.clone());
    out_dims.extend_from_slice(&tensor_dims[axis + 1..]);
    Ok(Type::Tensor(out_dims, tensor_precision.clone()))
}

fn infer_trace_result_type(tensor_ty: &Type, axis1: usize, axis2: usize) -> Result<Type, String> {
    let Type::Tensor(dims, precision) = tensor_ty else {
        return Err(format!("trace expects tensor input, got {tensor_ty}"));
    };
    if axis1 >= dims.len() || axis2 >= dims.len() || axis1 == axis2 {
        return Err(format!(
            "trace expects distinct in-bounds axes, got {axis1} and {axis2} for rank {}",
            dims.len()
        ));
    }
    let out_dims = dims
        .iter()
        .enumerate()
        .filter_map(|(index, dim)| ((index != axis1) && (index != axis2)).then_some(dim.clone()))
        .collect();
    Ok(Type::Tensor(out_dims, precision.clone()))
}

fn infer_diagonal_result_type(
    tensor_ty: &Type,
    axis1: usize,
    axis2: usize,
) -> Result<Type, String> {
    let Type::Tensor(dims, precision) = tensor_ty else {
        return Err(format!("diagonal expects tensor input, got {tensor_ty}"));
    };
    if axis1 >= dims.len() || axis2 >= dims.len() || axis1 == axis2 {
        return Err(format!(
            "diagonal expects distinct in-bounds axes, got {axis1} and {axis2} for rank {}",
            dims.len()
        ));
    }
    let diag_dim = match (&dims[axis1], &dims[axis2]) {
        (Dim::Lit(lhs), Dim::Lit(rhs)) if lhs == rhs => Dim::Lit(*lhs),
        _ => Dim::Wildcard,
    };
    let mut out_dims = Vec::with_capacity(dims.len() - 1);
    for (index, dim) in dims.iter().enumerate() {
        if index == axis1 {
            out_dims.push(diag_dim.clone());
        } else if index != axis2 {
            out_dims.push(dim.clone());
        }
    }
    Ok(Type::Tensor(out_dims, precision.clone()))
}

fn macro_source(expr: &deep::Expr) -> Option<String> {
    let deep::Expr::List(list, _) = expr else {
        return None;
    };
    let meta = get_meta(list)?;
    let source = meta
        .entries
        .iter()
        .find(|(key, _)| key == "source")
        .map(|(_, value)| value)?;
    let rendered = chelis_deep::printer::print_canonical(std::slice::from_ref(source));
    Some(rendered.replace('\n', " ").trim().to_string())
}

fn resolve_type_aliases(ty: &Type, adt_reg: &AdtRegistry) -> Type {
    let mut seen = HashSet::new();
    resolve_type_aliases_inner(ty, adt_reg, &mut seen)
}

fn resolve_type_aliases_inner(
    ty: &Type,
    adt_reg: &AdtRegistry,
    seen: &mut HashSet<String>,
) -> Type {
    match ty {
        Type::Adt(name, args) => {
            let resolved_args: Vec<Type> = args
                .iter()
                .map(|arg| resolve_type_aliases_inner(arg, adt_reg, seen))
                .collect();

            if seen.contains(name) {
                return Type::Adt(name.clone(), resolved_args);
            }

            if let Some(expanded) = adt_reg.instantiate_alias(name, &resolved_args) {
                seen.insert(name.clone());
                let resolved = resolve_type_aliases_inner(&expanded, adt_reg, seen);
                seen.remove(name);
                resolved
            } else {
                Type::Adt(name.clone(), resolved_args)
            }
        }
        Type::Fn(args, ret) => Type::Fn(
            args.iter()
                .map(|a| resolve_type_aliases_inner(a, adt_reg, seen))
                .collect(),
            Box::new(resolve_type_aliases_inner(ret, adt_reg, seen)),
        ),
        Type::Tuple(ts) => Type::Tuple(
            ts.iter()
                .map(|t| resolve_type_aliases_inner(t, adt_reg, seen))
                .collect(),
        ),
        _ => ty.clone(),
    }
}

fn deep_type_to_resolved_type(
    expr: &deep::Expr,
    vg: &mut VarGen,
    adt_reg: &AdtRegistry,
    tvar_map: &mut HashMap<String, TypeVar>,
) -> Type {
    let ty = deep_type_to_type(expr, vg, tvar_map);
    resolve_type_aliases(&ty, adt_reg)
}

// ── Declaration collection (first pass) ──────────────────────────

/// Which declaration kinds a `collect_declarations` sub-pass should process.
///
/// `deftype` constructor schemes expand transparent type aliases in their
/// field types at registration (see `AdtRegistry::expand_aliases`), so every
/// `typealias` must be in the registry first. Running `Aliases` over all
/// top-level items before `Rest` guarantees that even for a forward reference
/// — an alias declared textually after the `deftype` that uses it, as in
/// `Hull.Ast` where `type EffectRow = List[Effect]` follows `type Type = ...
/// | TArrow(Type, Type, EffectRow) | ...`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeclPhase {
    /// Process only `typealias` declarations.
    Aliases,
    /// Process everything except `typealias` (`deftype`, `defsig`, ...).
    Rest,
}

/// Run the two-phase declaration collection over `items` (already flattened
/// past `module` wrappers, each paired with its lexical module key):
/// register all type aliases, then everything else.
fn collect_all_declarations(
    items: &[(Option<String>, &deep::Expr)],
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &mut AdtRegistry,
    errors: &mut Vec<CheckError>,
) {
    // chelis#258 (main): duplicate-def / builtin-shadowing rejection runs
    // over the bare item list. Our `items` is paired with module keys, so
    // project to the `&deep::Expr` slice the reporters expect.
    let bare_items: Vec<&deep::Expr> = items.iter().map(|(_, expr)| *expr).collect();
    report_duplicate_defs(&bare_items, errors);
    report_duplicate_defsigs(&bare_items, errors);
    report_builtin_shadowing(&bare_items, errors);
    for (module, expr) in items {
        collect_declarations(
            expr,
            module.as_deref(),
            env,
            vg,
            subst,
            adt_reg,
            errors,
            DeclPhase::Aliases,
        );
    }
    for (module, expr) in items {
        collect_declarations(
            expr,
            module.as_deref(),
            env,
            vg,
            subst,
            adt_reg,
            errors,
            DeclPhase::Rest,
        );
    }
}

/// Build the program-shape opacity metadata (RFC D-CHECK) from the
/// flattened `(module key, item)` list: per-module export sets, the
/// top-level binding -> module map, and the formatted "exported
/// producers with signatures" entries per opaque type used by the
/// violation error contract. Runs after declaration collection so the
/// registry already carries every `deftype`'s `opaque` flag and
/// defining module.
fn build_opacity_meta(
    items: &[(Option<String>, &deep::Expr)],
    adt_reg: &AdtRegistry,
    vg: &mut VarGen,
) -> crate::opacity::OpacityModuleMeta {
    let mut meta = crate::opacity::OpacityModuleMeta::default();
    // Declared signature types (from `defsig` nodes) for producer
    // display; keyed by binding name like `meta.bindings`.
    let mut declared_sigs: HashMap<String, Type> = HashMap::new();
    // Pass 1: export sets. Lexical `(export ...)` nodes attribute to
    // their module wrapper; package-linked exports arrive as
    // top-level nodes whose names carry the reef internal-name stem
    // (the reef rewrite emits them with internal names), so each
    // exported name self-attributes through its stem.
    for (module, item) in items {
        let deep::Expr::List(list, _) = item else {
            continue;
        };
        if get_tag(list) != Some("export") {
            continue;
        }
        for child in children(list) {
            let Some(name) = symbol_name(child) else {
                continue;
            };
            let target = match module {
                Some(module) => Some(module.clone()),
                None => crate::opacity::reef_module_stem(name),
            };
            if let Some(target) = target {
                meta.exports
                    .entry(target)
                    .or_default()
                    .insert(name.to_string());
            }
        }
    }
    // Pass 2: binding -> module attribution and declared sigs.
    // Stem-attributed (package-linked) bindings are recorded only
    // when their module's export set is known: without it, the sixth
    // rejection's no-export-decl-means-sealed rule would reject
    // legitimately exported producers in pipelines that strip Export
    // decls (fail-open for unattributable names by design).
    for (module, item) in items {
        let deep::Expr::List(list, _) = item else {
            continue;
        };
        if !matches!(get_tag(list), Some("def") | Some("defsig")) {
            continue;
        }
        let kids = children(list);
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        let target = match module {
            Some(module) => Some(module.clone()),
            None => crate::opacity::reef_module_stem(name)
                .filter(|stem| meta.exports.contains_key(stem)),
        };
        let Some(target) = target else {
            continue;
        };
        meta.bindings.insert(name.to_string(), target);
        if get_tag(list) == Some("defsig")
            && let Some(ty_expr) = kids.get(1)
        {
            let ty = deep_type_to_resolved_type(ty_expr, vg, adt_reg, &mut HashMap::new());
            declared_sigs.insert(name.to_string(), ty);
        }
    }
    // Producer enumeration per opaque type: exported bindings of the
    // defining module whose declared RESULT type mentions the type
    // (containment chased through named type definitions).
    for (adt_name, def) in &adt_reg.defs {
        if !def.opaque {
            continue;
        }
        let Some(module) = &def.defining_module else {
            continue;
        };
        let Some(export_set) = meta.exports.get(module) else {
            continue;
        };
        let mut entries: std::collections::BTreeSet<String> = Default::default();
        for name in export_set {
            if meta.bindings.get(name) != Some(module) {
                continue;
            }
            let Some(sig) = declared_sigs.get(name) else {
                continue;
            };
            let result = match sig {
                Type::Fn(_, ret) => ret.as_ref(),
                other => other,
            };
            if crate::opacity::type_mentions_adt(result, adt_name, adt_reg) {
                // RT-1 F3: store the producer entry de-mangled so the
                // reef surface renders `probability: (f32) -> Probability`
                // rather than the internal `pkg__...` names.
                entries.insert(format!(
                    "{}: {}",
                    crate::opacity::demangle_ident(name),
                    crate::opacity::demangle_type(sig)
                ));
            }
        }
        if !entries.is_empty() {
            meta.producer_entries
                .entry(adt_name.clone())
                .or_default()
                .extend(entries);
        }
    }
    meta
}

/// Reject two same-name `def` declarations in one program (chelis#258).
///
/// A def's value binding is silent last-write-wins (`env.bind` →
/// `HashMap::insert`, like the `defsig` arm of `collect_declarations`), and
/// Chelis does not dispatch same-name `def`s by argument arity or tensor
/// rank. So two `def f`s whose sigs differ only in rank leave just one arm
/// reachable: callers of the other rank fire a confusing `DimensionMismatch`
/// at the call site instead of a clear error at the redundant definition.
/// This mirrors the duplicate-`deftype` / duplicate-`typealias` rejection
/// already in `collect_declarations`, moving the diagnostic to the
/// definition site.
///
/// Scoped to `def` (not `defsig`): a `defsig` legitimately co-occurs with a
/// synthesized signature for the same name (an inline-annotated `def`
/// desugars to both a `defsig` and a `def`), so a same-name `defsig` is not
/// on its own a duplicate definition. `items` is already flattened past
/// `module` wrappers, and the prelude lives in the builtin env rather than as
/// `def` nodes here, so only genuine in-program user redefinitions match.
fn report_duplicate_defs(items: &[&deep::Expr], errors: &mut Vec<CheckError>) {
    let mut seen: HashSet<&str> = HashSet::new();
    for expr in items {
        let deep::Expr::List(list, _) = expr else {
            continue;
        };
        if get_tag(list) != Some("def") {
            continue;
        }
        let Some(name) = children(list).first().and_then(symbol_name) else {
            continue;
        };
        if !seen.insert(name) {
            errors.push(CheckError::new(
                CheckErrorKind::DuplicateDefinition,
                format!("duplicate definition: `{name}` is defined more than once"),
                vec![format!(
                    "rename one of the `{name}` definitions: Chelis does not dispatch same-name `def`s by argument type or rank"
                )],
            ));
        }
    }
}

/// Reject two same-name `defsig` declarations in one program.
///
/// Chelis does not dispatch user functions by arity, type, or rank; the valid
/// same-name declaration pair is exactly one `defsig` plus one `def`. Multiple
/// `defsig`s for a name otherwise feed several last-write-wins maps
/// (`collect_declarations`, declared-param-type collection, signature metadata)
/// and make the enforced signature order-dependent.
fn report_duplicate_defsigs(items: &[&deep::Expr], errors: &mut Vec<CheckError>) {
    let mut seen: HashSet<&str> = HashSet::new();
    for expr in items {
        let deep::Expr::List(list, _) = expr else {
            continue;
        };
        if get_tag(list) != Some("defsig") {
            continue;
        }
        let Some(name) = children(list).first().and_then(symbol_name) else {
            continue;
        };
        if !seen.insert(name) {
            errors.push(CheckError::new(
                CheckErrorKind::DuplicateDefinition,
                format!("duplicate signature: `{name}` has more than one `defsig`"),
                vec![format!(
                    "keep a single `defsig` for `{name}`: Chelis does not dispatch same-name functions by argument type, arity, or rank"
                )],
            ));
        }
    }
}

/// Reject a top-level `def` or `defsig` whose name appears in the closed
/// builtin vocabulary (chelis#353, spec/04-type-system.md §8.6).
///
/// Call sites are dispatched builtin-first by name in both the host
/// evaluator (`runtime/host_ops.rs::builtin_name`) and IR lowering
/// (`lower.rs::builtin_name`); both import `BUILTIN_NAMES`, the same
/// table consulted here, so the rejected set and the dispatched set
/// cannot drift. A user definition with a builtin name is therefore
/// unreachable by name: pre-fix, `def sum` checked clean, hit the
/// builtin's arity error under eval, and segfaulted on the C backend —
/// three lanes, three different answers. Rejecting the declaration here,
/// in the collection chokepoint every checker entry point shares, makes
/// all lanes agree on the same diagnostic.
///
/// Deliberately narrow scope:
/// - Reef package modules never reach this check with bare names: reef
///   rewrites package decls to internal `pkg__...` names (and rewrites
///   their call sites with them) before the checker runs, so a package
///   `def sum` is allowed and genuinely dispatches to the user def (the
///   stdlib's `Std.Decimal.normalize` / `Std.Test.fail` rely on this).
/// - Function parameters and block-locals may reuse builtin names: they
///   bind values, not call-site dispatch, and shadow harmlessly on every
///   lane.
///
/// An inline-annotated `def` desugars to a `defsig` AND a `def` with the
/// same name; report once per name, as the `def` (what the user wrote).
fn report_builtin_shadowing(items: &[&deep::Expr], errors: &mut Vec<CheckError>) {
    let decl_name = |expr: &deep::Expr, tag: &str| -> Option<String> {
        let deep::Expr::List(list, _) = expr else {
            return None;
        };
        if get_tag(list) != Some(tag) {
            return None;
        }
        children(list)
            .first()
            .and_then(symbol_name)
            .filter(|name| builtins::BUILTIN_NAMES.contains(name))
            .map(str::to_string)
    };

    let def_names: HashSet<String> = items
        .iter()
        .filter_map(|expr| decl_name(expr, "def"))
        .collect();

    let mut reported: HashSet<String> = HashSet::new();
    for expr in items {
        let Some(name) = decl_name(expr, "def").or_else(|| decl_name(expr, "defsig")) else {
            continue;
        };
        if !reported.insert(name.clone()) {
            continue;
        }
        let decl_kw = if def_names.contains(&name) {
            "def"
        } else {
            "sig"
        };
        errors.push(CheckError::new(
            CheckErrorKind::BuiltinShadowing,
            format!(
                "`{decl_kw} {name}` shadows the builtin function `{name}`: user `def`/`sig` \
                 declarations may not reuse builtin names (spec/04-type-system.md \u{00a7}8.6). \
                 Calls to `{name}` always dispatch to the builtin under eval and lowering, so \
                 the shadowing declaration can never be reached by name."
            ),
            vec![format!(
                "rename `{name}` (e.g. `{name}2` or `my_{name}`); inside a reef package \
                 module the name is allowed because package declarations are \
                 internal-name-rewritten before checking"
            )],
        ));
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_declarations(
    expr: &deep::Expr,
    lexical_module: Option<&str>,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &mut AdtRegistry,
    errors: &mut Vec<CheckError>,
    phase: DeclPhase,
) {
    let list = match expr {
        deep::Expr::List(list, _) => list,
        _ => return,
    };

    let tag = match get_tag(list) {
        Some(t) => t,
        None => return,
    };

    // Aliases register first so `deftype` field-type alias expansion sees a
    // fully-populated alias table; every other decl kind runs in the second
    // sub-pass.
    let in_phase = match phase {
        DeclPhase::Aliases => tag == "typealias",
        DeclPhase::Rest => tag != "typealias",
    };
    if !in_phase {
        return;
    }

    let kids = children(list);

    match tag {
        "deftype" => {
            // Reject same-namespace collisions (another `deftype`, a
            // `typealias`, or a prelude ADT registered earlier in this
            // program). Without this check `AdtRegistry::defs` is
            // silently last-write-wins, which propagates wrong
            // constructor types and (per `compute_tensor_carrying_adts`
            // in linearity.rs) order-dependent borrow semantics.
            if let Some(name) = kids.first().and_then(symbol_name)
                && let Some(prior_kind) = adt_reg.existing_kind(name)
            {
                errors.push(CheckError::new(
                    CheckErrorKind::DuplicateDefinition,
                    format!(
                        "duplicate type definition: `{name}` was already declared as a {prior_kind}"
                    ),
                    vec![format!("rename one of the `{name}` declarations")],
                ));
                return;
            }
            // RFC D-CHECK: record opacity + module identity on the
            // registered AdtDef. The module key is the lexical
            // wrapper when present, else the reef internal-name stem
            // of the deftype's own (rewritten) name. `@opaque`
            // requires a named module (RT-0 M6): a top-level opaque
            // declaration has no module identity, which would make
            // the enforcement boundary collide across combined
            // sources.
            let opaque = deftype_opaque_meta(list);
            let defining_module = crate::opacity::module_key_for_item(
                lexical_module,
                kids.first().and_then(symbol_name),
            );
            if opaque
                && defining_module.is_none()
                && let Some(name) = kids.first().and_then(symbol_name)
            {
                errors.push(crate::opacity::unmoduled_opaque_error(name));
            }
            let ctors = adt_reg.register_deftype(kids, vg, opaque, defining_module);
            for (name, scheme) in ctors {
                env.bind(name, scheme);
            }
        }
        "defsig" => {
            // (defsig {} name type_expr)
            if kids.len() >= 2
                && let Some(name) = symbol_name(&kids[0])
            {
                let ty = deep_type_to_resolved_type(&kids[1], vg, adt_reg, &mut HashMap::new());
                let scheme = env.generalize(&ty, subst);
                env.bind(name.to_string(), scheme);
            }
        }
        "typealias" => {
            // (typealias {} Name (params...) type_expr)
            if kids.len() >= 3
                && let Some(name) = symbol_name(&kids[0])
            {
                if let Some(prior_kind) = adt_reg.existing_kind(name) {
                    errors.push(CheckError::new(
                        CheckErrorKind::DuplicateDefinition,
                        format!(
                            "duplicate type definition: `{name}` was already declared as a {prior_kind}"
                        ),
                        vec![format!("rename one of the `{name}` declarations")],
                    ));
                    return;
                }
                let params = match &kids[1] {
                    deep::Expr::List(list, _) => list
                        .elements
                        .iter()
                        .filter_map(symbol_name)
                        .map(str::to_string)
                        .collect::<Vec<_>>(),
                    _ => Vec::new(),
                };

                let mut tvar_map = HashMap::new();
                let mut param_vars = Vec::with_capacity(params.len());
                for param in &params {
                    let tv = vg.fresh_tvar();
                    tvar_map.insert(param.clone(), tv);
                    param_vars.push(tv);
                }

                let aliased_ty = deep_type_to_type(&kids[2], vg, &mut tvar_map);
                adt_reg.register_alias(name.to_string(), params, param_vars, aliased_ty);
            }
        }
        _ => {}
    }
}

// ── Tier-2 rank-polymorphism Body Discipline ─────────────────────

/// True if any tensor inside `ty` carries a `Dim::Rank` (a rank variable).
fn type_contains_rank(ty: &Type) -> bool {
    match ty {
        Type::Tensor(dims, _) => dims.iter().any(|d| matches!(d, Dim::Rank(_))),
        Type::Fn(args, ret) => args.iter().any(type_contains_rank) || type_contains_rank(ret),
        Type::Ref(inner) => type_contains_rank(inner),
        Type::Adt(_, args) => args.iter().any(type_contains_rank),
        Type::Tuple(ts) => ts.iter().any(type_contains_rank),
        Type::Prim(_) | Type::Var(_) | Type::Unit | Type::Error => false,
    }
}

/// Extract the callee name from an `app`'s first child when it is `(var {} name)`.
fn app_var_name(callee: &deep::Expr) -> Option<&str> {
    let deep::Expr::List(list, _) = callee else {
        return None;
    };
    if get_tag(list) != Some("var") {
        return None;
    }
    children(list).first().and_then(symbol_name)
}

/// Names of every top-level `def` in the program (after module flattening),
/// so the Body-Discipline check can reject a call that resolves to a user
/// function shadowing an Identity builtin name (chelis#258 §4.2).
fn collect_user_def_names(items: &[&deep::Expr]) -> HashSet<String> {
    let mut out = HashSet::new();
    for expr in items {
        if let deep::Expr::List(list, _) = expr
            && get_tag(list) == Some("def")
            && let Some(name) = children(list).first().and_then(symbol_name)
        {
            out.insert(name.to_string());
        }
    }
    out
}

/// Walk a rank-polymorphic def's body and reject any call whose output shape is
/// not *name-trackable* at symbolic rank. Admitted: shape-identity (elementwise)
/// builtins and named-axis reductions (the procedural arm verifies those drop a
/// named axis and carry the rest through). Rejected: positional shape-rewriting
/// builtins (`permute`/`reshape`/`matmul`/…), and any user/non-builtin/computed
/// callee not proven rank-safe — against a spread `..r` there are no named axes
/// left to catch an untracked transposition/reshape, so admitting one would
/// silently break §4.2 transposition safety (spec/design/rank_polymorphism.md
/// §Soundness Boundary).
fn check_rank_body_discipline(
    def_name: &str,
    expr: &deep::Expr,
    user_def_names: &HashSet<String>,
    errors: &mut Vec<CheckError>,
) {
    stack_guard!("check_rank_body_discipline", expr);
    let deep::Expr::List(list, _) = expr else {
        return;
    };
    match get_tag(list) {
        // Function-taking transforms apply a *referenced* user function across
        // the opaque rank. That callee is not inlined here, so its body can
        // transpose/reshape undetected — reject outright (spec §4.2).
        // `jit`/`realize`/`cast`/`copy` wrap an *inline* expression that the
        // recursion below still checks, so they are not rejected here.
        Some(t @ ("grad" | "vmap")) => {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!(
                    "rank-polymorphic def `{def_name}` may not use `{t}` in its body: it applies \
                     a function across the opaque rank `..r`, whose body cannot be proven \
                     shape-identity (spec/04-type-system.md \u{00a7}4.2)."
                ),
                vec![],
            ));
        }
        Some("app") => match children(list).first().and_then(app_var_name) {
            // A user-defined `def` of this name — possibly SHADOWING an
            // Identity builtin (`def relu(x) = permute(x,1,0)`). The call
            // resolves to the user def, whose body is not proven rank-safe, so
            // it must be rejected before the builtin-name classification below.
            Some(name) if user_def_names.contains(name) => {
                errors.push(CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!(
                        "rank-polymorphic def `{def_name}` may not call user-defined `{name}`: \
                         only shape-identity builtins are proven rank-safe in a `..r` body, and a \
                         user `def` (even one shadowing a builtin name) is not (spec/04-type-system.md \
                         \u{00a7}4.2)."
                    ),
                    vec![],
                ));
            }
            // Identity (elementwise) or NameTracked (named-axis reduction /
            // named-axis expand) builtin — admissible. For a NameTracked op
            // the procedural inference arm (`check_reduction_signature` /
            // `check_expand_signature`) is the real gate: it verifies the
            // addressed axis is name-anchored against the operand and
            // computes a symbolic output that carries the surviving named axes
            // through, rejecting a positional index at symbolic rank or a
            // non-existent/ambiguous/duplicate axis name. So no untracked
            // transposition can slip past.
            Some(name)
                if builtins::BUILTIN_NAMES.contains(&name)
                    && matches!(
                        builtins::shape_class(name),
                        builtins::ShapeClass::Identity | builtins::ShapeClass::NameTracked
                    ) => {}
            // A named builtin that rewrites shape positionally (not name-tracked).
            Some(name) if builtins::BUILTIN_NAMES.contains(&name) => {
                errors.push(CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!(
                        "rank-polymorphic def `{def_name}` may not call shape-rewriting builtin \
                         `{name}`: it is not name-trackable at symbolic rank, so against a spread \
                         `..r` there are no named axes left to catch a transposition or reshape \
                         (spec/04-type-system.md \u{00a7}4.2). A `..r` body may call shape-identity \
                         (elementwise) operations, named-axis reductions, and named-axis expand \
                         only."
                    ),
                    vec![format!(
                        "remove the `{name}` call from the rank-polymorphic body, or use \
                         concrete-rank `def`s instead of a `..r` signature"
                    )],
                ));
            }
            // A named user-defined function — not proven rank-safe.
            Some(name) => {
                errors.push(CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!(
                        "rank-polymorphic def `{def_name}` may not call `{name}`: only \
                         shape-identity builtins are proven rank-safe in a `..r` body \
                         (spec/04-type-system.md \u{00a7}4.2). Calling a user-defined function \
                         from a rank-polymorphic body is not supported."
                    ),
                    vec![],
                ));
            }
            // A computed callee (a transform result like `grad(f)(x)`, a
            // first-class function value, or an applied lambda's non-inline
            // form): cannot be proven rank-safe.
            None => {
                errors.push(CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!(
                        "rank-polymorphic def `{def_name}` may not apply a computed or \
                         non-builtin callee in a `..r` body: only shape-identity builtins are \
                         proven rank-safe (spec/04-type-system.md \u{00a7}4.2)."
                    ),
                    vec![],
                ));
            }
        },
        _ => {}
    }
    // Recurse so nested calls (in let/if/match/lambda bodies, args) are checked.
    for child in &list.elements {
        check_rank_body_discipline(def_name, child, user_def_names, errors);
    }
}

// ── Top-level inference (second pass) ────────────────────────────

#[allow(clippy::too_many_arguments)]
fn infer_top_level(
    expr: &deep::Expr,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
    user_def_names: &HashSet<String>,
) {
    let list = match expr {
        deep::Expr::List(list, _) => list,
        _ => return,
    };

    let tag = match get_tag(list) {
        Some(t) => t,
        None => return,
    };

    // Skip deftype/defsig/typealias (already processed in first pass)
    if tag == "deftype" || tag == "defsig" || tag == "typealias" {
        return;
    }

    let kids = children(list);

    if tag == "def" && kids.len() >= 2 {
        let name = match symbol_name(&kids[0]) {
            Some(n) => n.to_string(),
            None => return,
        };

        // Save declared type from defsig BEFORE inferring (it may get overwritten)
        let declared_ty = env.lookup(&name).map(|s| {
            let s = s.clone();
            env.instantiate(&s, vg)
        });

        let errors_before_body = errors.len();
        // WS-A7: when the body is a bare-arg `(fn (params) body)` and the
        // declared signature gives concrete param types, seed the body's
        // params with the declared types BEFORE inferring the body. Without
        // this seeding, bare params get fresh, unconstrained type variables.
        // For callee schemes that share a tvar between an `&T` param and a
        // non-borrow return position (e.g. `add: (&t, &t) -> t`), the
        // call-site `auto_borrow_call_arg_types` wraps the actual param
        // tvar in `Ref(...)` and unifies it with the formal `Ref(α)`,
        // collapsing the param-side and return-side of the callee into
        // the same equivalence class. The post-body sig-unify then drives
        // the return position to `Ref(t)` instead of the declared `t`,
        // surfacing as `def 'tadd' body doesn't match declared signature:
        // body has type `(&t, &t) -> &t`, declared type is `(&t, &t) -> t``.
        // Annotated params do not hit this because their concrete type
        // (`&tensor[..]`) flows through the call site directly. Seeding
        // bare params with the declared type here makes the bare-arg path
        // behave the same as the annotated path. See
        // `crates/chelis-cli/tests/bareref_return_inference.rs`.
        let body_ty = if let Some(decl_ty) = &declared_ty {
            infer_def_body_with_sig(
                &kids[1],
                decl_ty,
                env,
                vg,
                subst,
                adt_reg,
                errors,
                typed_nodes,
                total_nodes,
            )
        } else {
            infer_expr(
                &kids[1],
                env,
                vg,
                subst,
                adt_reg,
                errors,
                typed_nodes,
                total_nodes,
            )
        };
        // Did the body's inference report any UnboundVariable diagnostic?
        // We use this to discriminate WS-A5 RT-3a F1's masked-by-Error
        // case (where Error is a downstream consequence of a reportable
        // cause that the user can act on) from cascades where Error
        // emerges from an internal type-checker limitation that has no
        // matching upstream diagnostic. Without this discriminator the
        // F1 detector double-reports on legitimate code that exercises
        // type-checker gaps (record construction, region effects) which
        // the permissive unify rule was implicitly tolerating.
        let body_has_unbound_diagnostic = errors[errors_before_body..]
            .iter()
            .any(|e| matches!(e.kind, CheckErrorKind::UnboundVariable));

        // Enforce defsig: body must match declared signature.
        //
        // After unify succeeds, the body's inferred type may still carry
        // `Dim::Wildcard` slots (e.g. from `pad_sequences_to`, `concat`,
        // `to_tensor`) because `unify_dim` treats Wildcard as a permissive
        // matches-anything sentinel. Generalizing the raw body would expose
        // those wildcards to callers, who would then silently accept any
        // concrete dim. Narrow the body's Wildcards against the declared
        // template so the scheme registered for callers reflects the
        // declared concrete shape (#39).
        //
        // Implicit-copy fan-out v3 Shape A: if the initial unify fails
        // and the body's tail position resolves to a `(var x)`
        // reference (after walking `let`/`if`/`match` wrappers via
        // `descend_to_tail_var`) whose declared return is owned `T`
        // while the body's inferred type returns `Ref(T)`, retry the
        // unify against the declared return relaxed into `Ref(T)`.
        // This mirrors `auto_borrow_call_arg_types`'s owned-to-borrow
        // coercion at argument positions: the caller already arranged
        // the borrow lifetime via the param itself, and the tail-var
        // descent confirms every reachable return path returns the
        // same parameter.  Heterogeneous returns and bodies whose tail
        // is an `app` or other non-var expression still fail with the
        // existing TypeMismatch.
        let scheme_body = if let Some(decl_ty) = declared_ty {
            let unify_result = unify(&body_ty, &decl_ty, subst);
            let resolved_body = subst.apply(&body_ty);
            let resolved_decl = subst.apply(&decl_ty);
            // Declared-dim rigidity check (TypeCheck-FreeDimVarUnification-F1
            // Path B). The declared signature's param positions introduce
            // the universally-quantified dim parameters; after the
            // post-body sig-unify above, two distinct declared dims must
            // not have collapsed into one another (and none may have been
            // pinned to a concrete literal). This runs here, not inside
            // `infer_def_body_with_sig`, because the collapse for an
            // annotated-param body happens in the sig-unify itself, not
            // during body inference. The body's tail returns the wrong
            // declared dim (`def g[n, m](x: tensor[n, f32],
            // y: tensor[m, f32]) -> tensor[n, f32] = y`), and the
            // structural relaxed-retry guard does not see it because the
            // initial unify already succeeded by collapsing `n` and `m`.
            let mut declared_dvars: Vec<DimVar> = Vec::new();
            if let Type::Fn(decl_params, _) = &decl_ty {
                for t in decl_params {
                    for dv in crate::env::free_dvars(t) {
                        if !declared_dvars.contains(&dv) {
                            declared_dvars.push(dv);
                        }
                    }
                }
            }
            check_declared_dvars_rigid(&declared_dvars, subst, errors);
            // chelis#273: the param-position guard above never sees a dim
            // parameter that occurs only in the return type, so a body
            // could silently pin a return-only rigid dim. Reject the
            // input-coupled pins/collapses while keeping the legitimate
            // output-inferred uses (hello_tensor-style) green.
            check_return_only_dvars_rigid(&decl_ty, &declared_dvars, subst, errors);
            // Tier-2 rank-polymorphism Body Discipline
            // (spec/design/rank_polymorphism.md §Soundness Boundary, spec §4.2):
            // a def whose signature mentions a rank variable `..r` may call only
            // shape-identity (elementwise) builtins. Against an opaque rank there
            // are no named axes left to catch a transposition/reshape, so any
            // shape-rewriting op (or an unproven user call) is rejected here.
            if type_contains_rank(&decl_ty)
                && let Some((_, body_expr)) = extract_fn_params_and_body(&kids[1])
            {
                check_rank_body_discipline(&name, &body_expr, user_def_names, errors);
            }
            // chelis#272 list-uniformity check. A list literal of
            // tensors with *mismatched concrete* element axes joins to a
            // `Wildcard` along the differing axis (the deliberate #218
            // bare-`concat` ergonomic). That wildcard is a defensible
            // "I don't know the shape" result for an unannotated bare
            // list, but it must NOT silently satisfy a declared element
            // type that names a rigid/named dimension — `List[tensor[k]]`
            // promises every element has the *same* length `k`. A body
            // like `def make[k](a: tensor[2], b: tensor[3])
            //   -> List[tensor[k]] = [a, b]` produces
            // `List<tensor[Wildcard]>`; the wildcard unifies permissively
            // with the rigid `k` and leaves it unbound, so neither the
            // pin-to-literal nor the distinct-collapse arm of
            // `check_declared_dvars_rigid` fires. Flag that mismatch here
            // by comparing the declared return's list-element dims
            // against the resolved body's. (The `[k, m]` variant is
            // already caught above: the tightened Cons-join now unifies
            // the two rigid dims, and `check_declared_dvars_rigid`
            // reports the collapse.)
            check_list_elem_rigid_dim_vs_wildcard(&decl_ty, &resolved_body, errors);
            // Implicit-copy fan-out v3 Shape A relaxed retry: if the
            // initial unify fails and the body's tail position resolves
            // to a `(var x)` reference whose declared return is owned
            // `T` while the body's inferred type returns `Ref(T)`,
            // retry the unify against the declared return relaxed
            // into `Ref(T)`.
            let initial_failed = unify_result.is_err();
            let recovered_by_relaxed_retry = if initial_failed {
                let relaxed_decl = shape_a_relaxed_return(&kids[1], &resolved_body, &resolved_decl);
                relaxed_decl
                    .as_ref()
                    .is_some_and(|relaxed| unify(&body_ty, relaxed, subst).is_ok())
            } else {
                false
            };
            // WS-A5 RT-3a F1: the permissive `(Error, _)` unify rule lets
            // a body whose return position collapses to `Type::Error`
            // (e.g. `def use_mix(x: tensor[3, int32]) -> tensor[3, f32]
            // = poly_id(nonexistent_function(x))`, where the outer call
            // early-exits at `Type::Error` so the body's `fn` type is
            // `Fn([tensor[3, int32]], Type::Error)`) silently satisfy a
            // concrete declared signature, masking the precision/shape
            // mismatch the user would otherwise see. Surface the masked
            // mismatch here when the body collapses to `Type::Error` at
            // the top level OR at the return position of a function
            // type, and the corresponding declared slot is concrete.
            // We deliberately do NOT recurse through tuples/ADTs/Fn
            // arg positions: those would re-fire on cascade patterns
            // (e.g. tuple destructuring on an unannotated parameter
            // produces `Tuple([Error, Error, ...])` from a localized
            // inference gap that has already been surfaced upstream
            // and should not double-report the def-level diagnostic).
            let body_return_collapsed = match (&resolved_body, &resolved_decl) {
                (Type::Error, decl) => !matches!(decl, Type::Error),
                (Type::Fn(_, ret), Type::Fn(_, decl_ret))
                    if matches!(**ret, Type::Error) && !matches!(**decl_ret, Type::Error) =>
                {
                    true
                }
                _ => false,
            };
            let masked_by_error =
                unify_result.is_ok() && body_return_collapsed && body_has_unbound_diagnostic;
            let unrecovered_initial_failure = initial_failed && !recovered_by_relaxed_retry;
            if unrecovered_initial_failure || masked_by_error {
                // RT-2 fixup B1: when the mismatch is a tensor
                // precision mismatch (notably a `reduce_sum` body
                // whose result precision differs from the declared
                // one), include a §5.7.1 cite directly in the message
                // so the user sees the result-precision table rule
                // rather than a generic "doesn't match declared
                // signature".
                let extra = match (&resolved_body, &resolved_decl) {
                    (Type::Tensor(_, body_prec), Type::Tensor(_, decl_prec))
                        if body_prec != decl_prec =>
                    {
                        format!(
                            " (precision `{}` vs declared `{}`; if the body is a \
                             `reduce_sum`, see spec/04-type-system.md §5.7.1: \
                             narrow integer operands widen to int32 to prevent \
                             silent overflow; use `tensor[{}]` or omit the result \
                             type)",
                            body_prec.name(),
                            decl_prec.name(),
                            body_prec.name(),
                        )
                    }
                    _ => String::new(),
                };
                errors.push(CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!(
                        "def '{}' body doesn't match declared signature: \
                         body has type `{}`, declared type is `{}`{}",
                        name, resolved_body, resolved_decl, extra
                    ),
                    vec![],
                ));
            }
            // #39 wildcard narrowing. The narrow may now substitute a
            // declared `Dim::Var` (not just `Dim::Lit`) into a body
            // wildcard, but ONLY for dim vars bound by a parameter tensor
            // position (`param_bound_dvars`). That keeps the
            // `const_col[n](spots: tensor[n, ..]) -> tensor[n, 1]` family's
            // return dim tied to its input (chelis#405 / WS-3 build ICE)
            // while leaving return-only / value-parameter dims as
            // wildcards (the RT-39+44 soundness boundary, commit 8067c9ce).
            let param_dvars = param_bound_dvars(&resolved_decl);
            narrow_wildcards_with(&resolved_body, &resolved_decl, &param_dvars)
        } else {
            body_ty
        };

        let scheme = env.generalize(&scheme_body, subst);
        // chelis#397/#469: record the size provenance of a top-level value
        // binding (e.g. `zero_count = sub(cast(0, int32), cast(0, int32))`)
        // BEFORE binding it, so a later `expand(b, 0, zero_count)` recovers
        // whether it is a materializable extent (static / shape-sourced) or a
        // sourceless runtime scalar. Classified against the pre-binding scope.
        // The `Sourceless`/`Unknown` arm CLEARS any stale provenance so a
        // re-bind to a sourceless RHS does not inherit an earlier entry.
        match classify_expand_size(&kids[1], env) {
            SizeClass::Static => {
                env.mark_size_provenance(&name, crate::env::SizeProvenance::Static);
            }
            SizeClass::ShapeSourced => {
                env.mark_size_provenance(&name, crate::env::SizeProvenance::ShapeSourced);
            }
            SizeClass::Sourceless | SizeClass::Unknown => env.clear_size_provenance(&name),
        }
        env.bind(name, scheme);
    } else {
        // Any other top-level expression
        let _ = infer_expr(
            expr,
            env,
            vg,
            subst,
            adt_reg,
            errors,
            typed_nodes,
            total_nodes,
        );
    }
}

// ── Core inference ───────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn infer_expr(
    expr: &deep::Expr,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    // Bail before a deeply-nested `app` tree exhausts the native stack and
    // aborts the process (this is the gdb-pinned real-pricer crash site):
    // when the budget is gone we record the located bail in `STACK_EXHAUSTED`
    // (so the check entry boundary fails hard) and return `Type::Error`,
    // letting the stack unwind normally (no panic, no `catch_unwind`). See
    // `STACK_RED_ZONE_BYTES`. The WI-1 follow-up grows the native stack ONCE at
    // each public check entry (`with_grown_stack`), so on a legitimately deep
    // but finite program every recursive pass -- this one and the validate /
    // annotate / clone / drop passes -- runs inside the grown segment and this
    // guard does not fire; it remains the safety net for input deeper than the
    // grown segment can hold (covered-or-rejected).
    stack_guard!("infer_expr", expr, Type::Error);

    *total_nodes += 1;

    let result = match expr {
        deep::Expr::Atom(atom, _) => infer_atom(atom),
        deep::Expr::List(list, _) => {
            let tag = get_tag(list);
            match tag {
                Some("var") => infer_var(list, env, vg, subst, adt_reg, errors),
                Some("lit") => infer_lit(list, vg, adt_reg, errors),
                Some("app") => infer_app(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("fn") => infer_fn(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("let") => infer_let(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("if") => infer_if(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("match") => infer_match(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("pipe") => infer_pipe(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("tuple") => infer_tuple(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("tuple-get") => infer_tuple_get(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("record") => infer_record(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("access") => infer_access(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("record-update") => infer_record_update(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("cast") => infer_cast(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("grad") => infer_grad(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("vmap") => infer_vmap(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("def") => infer_def(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("defsig") => {
                    // Already handled in first pass
                    Type::Unit
                }
                Some("deftype") | Some("typealias") => {
                    // Already handled in first pass
                    Type::Unit
                }
                Some("par") => {
                    // par: evaluate all children, return type of last (v1: sequential)
                    let kids = children(list);
                    let mut last_ty = Type::Unit;
                    for kid in kids {
                        last_ty = infer_expr(
                            kid,
                            env,
                            vg,
                            subst,
                            adt_reg,
                            errors,
                            typed_nodes,
                            total_nodes,
                        );
                    }
                    last_ty
                }
                Some("jit") => {
                    // jit: compilation trigger; semantically a no-op at eval
                    // (spec/03-deep-syntax.md §2.7). Type is the type of the
                    // wrapped expression.
                    let kids = children(list);
                    if let Some(inner) = kids.first() {
                        infer_expr(
                            inner,
                            env,
                            vg,
                            subst,
                            adt_reg,
                            errors,
                            typed_nodes,
                            total_nodes,
                        )
                    } else {
                        Type::Error
                    }
                }
                Some("realize") => {
                    let kids = children(list);
                    if let Some(inner) = kids.first() {
                        infer_expr(
                            inner,
                            env,
                            vg,
                            subst,
                            adt_reg,
                            errors,
                            typed_nodes,
                            total_nodes,
                        )
                    } else {
                        Type::Error
                    }
                }
                Some("copy") => {
                    let kids = children(list);
                    if let Some(inner) = kids.first() {
                        let inner_ty = infer_expr(
                            inner,
                            env,
                            vg,
                            subst,
                            adt_reg,
                            errors,
                            typed_nodes,
                            total_nodes,
                        );
                        let resolved = subst.apply(&inner_ty);
                        match resolved {
                            Type::Tensor(_, _) | Type::Error => resolved,
                            Type::Ref(inner) if matches!(inner.as_ref(), Type::Tensor(_, _)) => {
                                *inner
                            }
                            _ => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    format!("copy requires tensor input, got {resolved}"),
                                    vec!["Wrap only tensor values in copy".to_string()],
                                ));
                                Type::Error
                            }
                        }
                    } else {
                        Type::Error
                    }
                }
                Some("borrow") => {
                    let kids = children(list);
                    if let Some(inner) = kids.first() {
                        let inner_ty = infer_expr(
                            inner,
                            env,
                            vg,
                            subst,
                            adt_reg,
                            errors,
                            typed_nodes,
                            total_nodes,
                        );
                        let resolved = subst.apply(&inner_ty);
                        match resolved {
                            Type::Ref(_) => resolved,
                            Type::Tensor(_, _) | Type::Adt(_, _) | Type::Tuple(_) | Type::Error => {
                                Type::Ref(Box::new(resolved))
                            }
                            // Issue #256: when the borrow inner is still an
                            // unresolved type variable (e.g. the output of a
                            // polymorphic-return call whose dim variables
                            // have not yet been pinned at this point in
                            // left-to-right inference), defer the
                            // tensor-or-carrier classification to subsequent
                            // unification. Wrapping as `Type::Ref(Type::Var)`
                            // lets the surrounding flow's expected argument
                            // type (e.g. a sig parameter `&tensor[..]`) pin
                            // the variable through unification. If the
                            // variable never gets pinned to a tensor or
                            // tensor-carrying type, the later unification
                            // failure surfaces the same diagnostic via the
                            // mismatched call site -- there is no silent
                            // accept. The linearity checker's
                            // `expr_is_owned_or_borrow_linear` still rejects
                            // a stamped `(t-var ...)` if no pinning happens.
                            //
                            // Soundness ledger (issue #256 round 2): record
                            // the inner type variable so the inference driver
                            // can re-check it against the *final*
                            // substitution after the def body completes. The
                            // deferral is sound only when the variable is
                            // eventually pinned to a tensor or tensor carrier;
                            // a fully-polymorphic consumer (e.g.
                            // `consume_any[a](t: a)`) never pins it, and a
                            // genuinely-non-tensor value would otherwise slip
                            // past every gate. See
                            // `validate_deferred_borrow_vars`.
                            Type::Var(tv) => {
                                subst.record_deferred_borrow_var(tv);
                                Type::Ref(Box::new(Type::Var(tv)))
                            }
                            _ => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    format!("borrow requires tensor or tensor-carrying input, got {resolved}"),
                                    vec!["Use `&x` only with tensor values".to_string()],
                                ));
                                Type::Error
                            }
                        }
                    } else {
                        Type::Error
                    }
                }
                _ => {
                    // Unknown tag -- try to infer children
                    Type::Error
                }
            }
        }
        deep::Expr::Map(_, _) => Type::Unit,
        deep::Expr::MetaExpr(meta, _) => infer_expr(
            &meta.expr,
            env,
            vg,
            subst,
            adt_reg,
            errors,
            typed_nodes,
            total_nodes,
        ),
    };

    if !matches!(result, Type::Error) {
        *typed_nodes += 1;
    }

    result
}

fn infer_atom(atom: &deep::Atom) -> Type {
    match atom {
        // D1 (WS-A0 RT-1 fixup): per spec/04-type-system.md §5.3 the
        // lexer parses unsuffixed integer tokens at i64 so that
        // out-of-range literals can be diagnosed before the int32
        // narrowing. The bare-atom path is the value-only fallback for
        // Deep code that bypasses the desugarer's `(lit {type: int32}
        // N)` wrapping; the same range check is enforced more visibly
        // at `infer_lit` where the type metadata is in scope.
        // Out-of-range here would silently wrap to a negative i32 if
        // we let it default unchecked — exactly what §5.3 forbids.
        // We can't push errors from this signature; the lit-form path
        // in `infer_lit` is the user-facing diagnostic site, and
        // bare-atom Deep code never round-trips through the surf
        // surface where the diagnostic is mandatory. Pin the decision
        // here so a future refactor doesn't mistakenly read this as
        // dead code.
        deep::Atom::Int(_) => Type::Prim(Prim::Int32),
        deep::Atom::Float(_) => Type::Prim(Prim::F32),
        deep::Atom::Bool(_) => Type::Prim(Prim::Bool),
        deep::Atom::Str(_) => Type::Prim(Prim::String),
        deep::Atom::Symbol(_) | deep::Atom::Keyword(_) => Type::Error,
    }
}

fn infer_var(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
) -> Type {
    let kids = children(list);
    if let Some(name) = kids.first().and_then(|e| symbol_name(e)) {
        // chelis#317: a nullary constructor at a construction site (a bare
        // `Alpha`, desugared to `(var Alpha)`) or an applied constructor
        // head (`Foo(x)` → `(app (var Foo) ...)`) that is out of scope must
        // be an `unknown constructor` error at `check`, not a silent bind
        // to a foreign module's same-terminal tag via the registry's fuzzy
        // fallback. Check exact scope first; the fuzzy `lookup_terminal_unique`
        // is the mis-resolution path the issue reports.
        if constructor_out_of_scope(name, env) {
            let mut err = CheckError::new(
                CheckErrorKind::UnknownConstructor,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!("unknown constructor: {name}"),
                ),
                vec![format!(
                    "Constructor '{name}' is not in scope. Declare it locally or add it \
                     to an import (e.g. `import Mod ({name})`)"
                )],
            );
            if let Some(sid) = list_span_id(list) {
                if let Some(off) = parse_span_offset(sid) {
                    err.span_offset = Some(off);
                }
                err.span_id = Some(sid.to_string());
            }
            errors.push(err);
            return Type::Error;
        }
        if let Some(scheme) = env
            .lookup(name)
            .or_else(|| env.lookup_terminal_unique(name))
        {
            let scheme = scheme.clone();
            let ty = env.instantiate(&scheme, vg);
            let resolved = subst.apply(&ty);
            // RFC D-CHECK: a bare reference to an out-of-module
            // opaque constructor is hidden, and an out-of-module
            // reference to an unexported binding whose signature
            // mentions an opaque type is the sixth rejection. Both
            // return the true type so no error cascades.
            if !crate::opacity::check_ctor_reference(name, adt_reg, errors) {
                crate::opacity::check_unexported_reference(name, &resolved, adt_reg, errors);
            }
            resolved
        } else {
            let mut err = CheckError::new(
                CheckErrorKind::UnboundVariable,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!("unbound variable: {name}"),
                ),
                vec![format!("Check spelling of '{}'", name)],
            );
            if let Some(sid) = list_span_id(list) {
                if let Some(off) = parse_span_offset(sid) {
                    err.span_offset = Some(off);
                }
                err.span_id = Some(sid.to_string());
            }
            errors.push(err);
            Type::Error
        }
    } else {
        Type::Error
    }
}

fn infer_lit(
    list: &deep::List,
    vg: &mut VarGen,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
) -> Type {
    let meta = get_meta(list);
    let kids = children(list);

    // D1 (WS-A0 RT-1 fixup): per spec/04-type-system.md §5.3 last
    // paragraph, the lexer parses unsuffixed integer literals at i64
    // so that out-of-range literals can be diagnosed before the
    // int32 narrowing. The desugarer attaches `type: int32` ahead of
    // type-check (because §5.3 declares int32 as the default), so
    // here we check whether the underlying i64 value actually fits in
    // i32. If it doesn't, emit the §5.3 diagnostic before defaulting
    // — silently wrapping to a negative i32 is the bug §5.3 was
    // written to prevent.
    //
    // RT-2 fixup B2/B3: extend the same range check to int8 and
    // int16 contextual positions (spec §5.6 / §P10b). When the
    // contextual tensor-literal rule (chelis-surf desugar) emits a
    // `(lit {type: (t-prim {} int8)} N)` for an `xs: tensor[N, int8]
    // = [..., 200]` source, the underlying i64 value (200) overflows
    // int8 (range [-128, 127]) and silently wraps to -56 if not
    // diagnosed here. Mirror the i32 check for the i8 and i16 rows.
    let value_atom = kids.first();
    let meta_prim_name = meta.and_then(|m| {
        m.entries.iter().find_map(|(k, v)| {
            if k == "type"
                && let deep::Expr::List(inner, _) = v
                && get_tag(inner) == Some("t-prim")
            {
                children(inner).first().and_then(symbol_name)
            } else {
                None
            }
        })
    });
    if let Some(prim_name) = meta_prim_name
        && let Some(deep::Expr::Atom(deep::Atom::Int(n), _)) = value_atom
    {
        // Per-prim range check. Only apply to integer prims; the float
        // contextual cases admit the i64 directly (Surf's float lexer
        // produces a Float atom; an Int atom in a float context is
        // either an error caught elsewhere or a Cons-mismatch).
        let range_check = match prim_name {
            "int8" => Some(("int8", i8::MIN as i64, i8::MAX as i64)),
            "int16" => Some(("int16", i16::MIN as i64, i16::MAX as i64)),
            "int32" => Some(("int32", i32::MIN as i64, i32::MAX as i64)),
            // int64 cannot overflow an i64 atom; bool/string don't
            // accept Int atoms.
            _ => None,
        };
        if let Some((dtype, lo, hi)) = range_check
            && (*n < lo || *n > hi)
        {
            // The int32 default path keeps the WS-A0 D1 message
            // shape (i64 suffix + cast(_, int64) hint) so existing
            // diagnostics-pinning tests stay green; the int8/int16
            // contextual paths cite §5.6 + §5.3 because the
            // narrowing came from contextual inference, not the
            // default. The cast hint spells the prec type name
            // `int64` (§1.1) — `i64` is only the literal-suffix
            // spelling (§5.5) and is not a valid `cast` target, so
            // recommending `cast({n}, i64)` would send the user to a
            // form that re-fires this same diagnostic (issue #308
            // review fix).
            if dtype == "int32" {
                errors.push(CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!(
                        "literal {n} out of range for default int32; use the `i64` \
                         suffix (`{n}i64`) or an explicit cast({n}, int64) \
                         (spec/04-type-system.md §5.3, §5.5)"
                    ),
                    vec![format!(
                        "spec/04-type-system.md §5.3: integer literals default to int32; \
                         the lexer parses at i64 so out-of-range tokens can be diagnosed \
                         before the narrowing rather than wrapping silently"
                    )],
                ));
            } else {
                errors.push(CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!(
                        "literal {n} out of range for context-inferred {dtype} \
                         [{lo}, {hi}]; use a wider integer type or an explicit \
                         cast (spec/04-type-system.md §5.6, §5.3)"
                    ),
                    vec![format!(
                        "spec/04-type-system.md §5.6 + §5.3: contextual tensor-literal \
                         element-type inference (§P10b) narrows unsuffixed integer \
                         literals to the declared element type ({dtype}). The lexer \
                         parses at i64 so values outside the {dtype} range \
                         [{lo}, {hi}] are diagnosed before the narrowing rather than \
                         wrapping silently"
                    )],
                ));
            }
            return Type::Error;
        }
    }

    // Check metadata for type annotation
    if let Some(meta) = meta {
        for (key, val) in &meta.entries {
            if key == "type" {
                let resolved = deep_type_to_resolved_type(val, vg, adt_reg, &mut HashMap::new());
                // RFC D-CHECK lit-forge gate: `{type: (t-adt ...)}`
                // metadata on a literal outside the defining module
                // forges an opaque value. Reachable from BOTH
                // surfaces: Surf expression ascription
                // (`0.5 : Probability`) and block-binding ascription
                // desugar to exactly this metadata (RT-0), so the
                // gate is not scoped to `.dp` ingestion.
                // `deep_type_to_resolved_type` expands transparent
                // aliases, so `0.5 : P2` cannot launder the gate.
                if let Type::Adt(adt_name, _) = &resolved {
                    crate::opacity::check_opaque_use(
                        crate::opacity::OpaqueAction::LitForge,
                        adt_name,
                        adt_reg,
                        errors,
                    );
                }
                return resolved;
            }
        }
    }

    // Fall back to value-based defaults
    if let Some(val) = kids.first() {
        match val {
            deep::Expr::Atom(deep::Atom::Int(n), _) => {
                // D1 (WS-A0 RT-1 fixup): same check as the metadata
                // path above but for Deep producers that omit the
                // explicit `type: int32` ascription on a `(lit {} N)`
                // form. Without this guard the bare-form path would
                // silently default to int32 and wrap.
                if i32::try_from(*n).is_err() {
                    errors.push(CheckError::new(
                        CheckErrorKind::TypeMismatch,
                        format!(
                            "literal {n} out of range for default int32; use the \
                             `{n}i64` literal suffix or an explicit cast({n}, int64) \
                             (spec/04-type-system.md §5.3, §5.5)"
                        ),
                        vec![format!(
                            "spec/04-type-system.md §5.3: integer literals default \
                             to int32; the lexer parses at i64 so out-of-range \
                             tokens can be diagnosed before the narrowing rather \
                             than wrapping silently"
                        )],
                    ));
                    Type::Error
                } else {
                    Type::Prim(Prim::Int32)
                }
            }
            deep::Expr::Atom(deep::Atom::Float(_), _) => Type::Prim(Prim::F32),
            deep::Expr::Atom(deep::Atom::Bool(_), _) => Type::Prim(Prim::Bool),
            deep::Expr::Atom(deep::Atom::Str(_), _) => Type::Prim(Prim::String),
            _ => Type::Error,
        }
    } else {
        Type::Error
    }
}

#[allow(clippy::too_many_arguments)]
fn infer_app(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.is_empty() {
        return Type::Error;
    }

    // Check if func is a comparison op (for special return type handling)
    let func_name = if let deep::Expr::List(flist, _) = &kids[0] {
        if get_tag(flist) == Some("var") {
            children(flist)
                .first()
                .and_then(|e| symbol_name(e))
                .map(|s| s.to_string())
        } else {
            None
        }
    } else {
        None
    };

    if matches!(func_name.as_deref(), Some("permute")) {
        return infer_permute_app(
            list,
            env,
            vg,
            subst,
            adt_reg,
            errors,
            typed_nodes,
            total_nodes,
        );
    }

    if matches!(func_name.as_deref(), Some("reshape")) {
        return infer_reshape_app(
            list,
            env,
            vg,
            subst,
            adt_reg,
            errors,
            typed_nodes,
            total_nodes,
        );
    }

    if matches!(func_name.as_deref(), Some("shrink")) {
        return infer_shrink_app(
            list,
            env,
            vg,
            subst,
            adt_reg,
            errors,
            typed_nodes,
            total_nodes,
        );
    }

    if matches!(func_name.as_deref(), Some("pad")) {
        return infer_pad_app(
            list,
            env,
            vg,
            subst,
            adt_reg,
            errors,
            typed_nodes,
            total_nodes,
        );
    }

    if matches!(func_name.as_deref(), Some("stride")) {
        return infer_stride_app(
            list,
            env,
            vg,
            subst,
            adt_reg,
            errors,
            typed_nodes,
            total_nodes,
        );
    }

    // chelis#339: the anchored named-axis expand form `expand(x, new, size,
    // anchor)` carries four arguments, but the builtin HM scheme is arity-3
    // (`(&tensor, int32, int32) -> out`), so it would hit the generic arity
    // check before the procedural arm. Dispatch it here (the
    // `infer_permute_app` pattern); 2-/3-arg expand keeps the generic path,
    // which reaches `check_expand_signature` with the scheme intact.
    if matches!(func_name.as_deref(), Some("expand")) && kids.len() >= 5 {
        return infer_expand_app(
            list,
            env,
            vg,
            subst,
            adt_reg,
            errors,
            typed_nodes,
            total_nodes,
        );
    }

    // chelis#339 Part 2: variadic named-axis reduction `sum(x, seq, head)`.
    // The reduction HM schemes are arity-2 (`(input, axis)`), so a 3+-arg
    // call would hit the generic arity check before the reduction arm;
    // dispatch it here. `check_reduction_signature`'s named loop handles N
    // axes (rejecting positional integers, unknown names, and duplicates).
    // The index-returning reductions are routed too, so they get a targeted
    // no-variadic-form rejection instead of a generic arity error.
    if matches!(
        func_name.as_deref(),
        Some(
            "sum"
                | "mean"
                | "max_reduce"
                | "min_reduce"
                | "prod_reduce"
                | "argmax_reduce"
                | "argmin_reduce"
        )
    ) && kids.len() >= 4
    {
        return infer_reduction_app(
            list,
            func_name.as_deref().unwrap(),
            env,
            vg,
            subst,
            adt_reg,
            errors,
            typed_nodes,
            total_nodes,
        );
    }

    if matches!(
        func_name.as_deref(),
        Some(
            "reduce_window_max" | "reduce_window_min" | "reduce_window_sum" | "reduce_window_mean"
        )
    ) {
        return infer_reduce_window_app(
            list,
            func_name.as_deref().unwrap(),
            env,
            vg,
            subst,
            adt_reg,
            errors,
            typed_nodes,
            total_nodes,
        );
    }

    let ctor_lookup_name = func_name.as_ref().and_then(|fname| {
        adt_reg
            .lookup_variant(fname)
            .map(|_| fname.clone())
            .or_else(|| {
                adt_reg
                    .lookup_variant_terminal_unique(fname)
                    .map(|(_, variant)| variant.name.clone())
            })
    });

    // The call site uses positional `(app)` syntax here (named-field
    // record construction lowers through a different builder, not
    // through `infer_app`). When two ADTs in the dep graph define
    // same-named constructors with different shapes (chelis#148: e.g.
    // Coral.Frame.Column.IntCol is positional, School.Data.Dataset.IntCol
    // is record), prefer the positional variant for this call site so
    // the call dispatches to the matching ADT instead of erroring on
    // the colliding record variant. Only emit the "must use named
    // fields" error when EVERY same-named variant in scope is record-
    // shaped, which is the original single-package case the error was
    // written for.
    // RFC D-CHECK: positional application of an out-of-module opaque
    // constructor is rejected (one violation per call site; the
    // callee `var`'s constructor-reference check is suppressed below
    // so the application does not double-report). Inference continues
    // so the call still yields its true type.
    if let Some(ref fname) = ctor_lookup_name
        && let Some((adt_name, _)) = adt_reg
            .lookup_variant_preferring_shape(fname, CallShape::Positional)
            .or_else(|| adt_reg.lookup_variant_terminal_unique(fname))
    {
        let adt_name = adt_name.to_string();
        crate::opacity::check_opaque_use(
            crate::opacity::OpaqueAction::CtorApplication,
            &adt_name,
            adt_reg,
            errors,
        );
    }

    // chelis#317: do not emit the record-shape diagnostic for an applied
    // constructor whose name is out of scope (a type-only import that calls
    // `Alpha(...)`). The shape check resolves through the same fuzzy
    // terminal fallback that mis-binds out-of-scope names, so firing it here
    // would mask the real defect with a confusing "must use named fields"
    // message. Let the head's `infer_var` report `unknown constructor`
    // instead.
    let ctor_call_out_of_scope = func_name
        .as_deref()
        .is_some_and(|fname| constructor_out_of_scope(fname, env));
    if !ctor_call_out_of_scope
        && let Some(ref fname) = ctor_lookup_name
        && let Some((_adt_name, variant)) = adt_reg
            .lookup_variant_preferring_shape(fname, CallShape::Positional)
            .or_else(|| adt_reg.lookup_variant_terminal_unique(fname))
        && !variant.fields.is_empty()
        && variant
            .fields
            .iter()
            .all(|(field_name, _)| field_name.is_some())
    {
        errors.push(CheckError::new(
            CheckErrorKind::TypeMismatch,
            format!("{fname} is a record constructor and must use named fields: {fname} {{ ... }}"),
            vec![],
        ));
        return Type::Error;
    }

    let func_ty = {
        let _ctor_guard = ctor_lookup_name
            .as_ref()
            .map(|_| crate::opacity::suppress_ctor_reference_check());
        infer_expr(
            &kids[0],
            env,
            vg,
            subst,
            adt_reg,
            errors,
            typed_nodes,
            total_nodes,
        )
    };
    // A reduction's axis argument may name a *dimension* of the operand
    // (`sum(x, seq)`, Tier-3 named-axis reduction, spec §4.5.3), not a bound
    // *value*. Like `expand`'s symbolic size arg below, such a name is typed as
    // an axis (`int32`) rather than inferred as a value — otherwise the
    // name-resolution pass would report a spurious `unbound variable`. The
    // actual name is read back from the arg expr in `check_reduction_signature`.
    let is_named_reduction = matches!(
        func_name.as_deref(),
        Some(
            "sum"
                | "mean"
                | "max_reduce"
                | "min_reduce"
                | "prod_reduce"
                | "argmax_reduce"
                | "argmin_reduce"
        )
    );
    let arg_tys: Vec<Type> = kids[1..]
        .iter()
        .enumerate()
        .map(|(index, arg)| {
            // `expand` axis slots that may carry a dim NAME instead of a bound
            // value: the size (index 2, §4.7.2 form 2) and — chelis#339
            // named-axis expand — the inserted-axis name (index 1). The
            // inserted-axis slot is scope-discriminated: a name bound in the
            // value environment is a *runtime value* (the issue #259 class,
            // `expand(&x, ax, 4)` with `ax: int32`), not a dim name, and must
            // keep flowing through ordinary inference into the
            // compile-time-constant rejection. The 4-arg anchored form routes
            // through `infer_expand_app` instead and never reaches this loop.
            let is_expand = matches!(func_name.as_deref(), Some("expand"));
            let is_expand_size = is_expand && index == 2;
            let is_expand_inserted_name = is_expand
                && index == 1
                && symbolic_dim_ref_name(arg).is_some_and(|name| env.lookup(name).is_none());
            let is_reduction_axis = is_named_reduction && index >= 1;
            if (is_expand_size || is_expand_inserted_name || is_reduction_axis)
                && symbolic_dim_ref_name(arg).is_some()
            {
                Type::Prim(Prim::Int32)
            } else {
                infer_expr(
                    arg,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                )
            }
        })
        .collect();

    if matches!(func_name.as_deref(), Some("drop")) && arg_tys.len() == 1 {
        return Type::Unit;
    }

    // If func or any arg is Error, propagate.
    //
    // chelis#530: the `expand` SIZE slot (kids[3], index 2 in `kids[1..]`)
    // is exempt. An inline size that is a tuple projection (`t.0`), an
    // inline `match`/`if`, or a `cast`/integer-arithmetic expression
    // *containing* one infers to `Type::Error`; letting that short-circuit
    // here returned `Type::Error` WITHOUT a diagnostic and WITHOUT ever
    // reaching the per-builtin expand Form-3 gate (positional arm) or the
    // named-axis literal check — silently accepting a sourceless size that
    // the C backend then hardcodes to extent 1 (eval `[3, 2]` vs C `[1, 2]`).
    // Both expand size gates read the size from the raw AST, not from
    // `arg_tys[2]`, and `unify` treats `Type::Error` as permissive
    // (`unify.rs`: `(Error, _) | (_, Error) => Ok(())`), so deferring to
    // them is sound and yields the correct per-form located diagnostic. A
    // genuinely sourced size never infers to `Error`, so this exemption only
    // ever reaches the rejection paths.
    let is_expand_call = matches!(func_name.as_deref(), Some("expand"));
    let non_size_arg_is_error = arg_tys
        .iter()
        .enumerate()
        .any(|(index, t)| !(is_expand_call && index == 2) && matches!(t, Type::Error));
    if matches!(func_ty, Type::Error) || non_size_arg_is_error {
        return Type::Error;
    }

    // Issue Chelis-Lang/chelis#218 R3 HIGH-CONCAT: when `Cons` is
    // called with two concrete tensor-typed args (head: tensor,
    // tail: List<tensor>), skip the generic per-dim equality
    // unification and produce a per-axis join (Lit(n) when both
    // dims are Lit(n) and equal; Wildcard otherwise). The generic
    // `unify` recursion walks dims pairwise and rejects
    // `Lit(2) vs Lit(3)`, which breaks `concat([a, b], 0)` after
    // the to_tensor source fix made nested-list literals emit
    // concrete dims.
    //
    // Precondition guards:
    //   * exactly two args (Cons signature)
    //   * head is a concrete tensor type
    //   * tail is `List<tensor[...]>` with a concrete tensor element
    //   * head and tail-element have matching rank (no rank join;
    //     mismatched ranks remain structural errors)
    //   * head and tail-element have matching precision (no
    //     precision join; mismatched precisions would mask real
    //     type errors)
    //
    // When any precondition fails, fall through to the generic
    // unify path so other Cons shapes (e.g. `Cons(scalar, list)` or
    // `Cons(head, Nil)` where `Nil`'s tvar binds the element type)
    // keep their existing semantics.
    if matches!(func_name.as_deref(), Some("Cons")) && arg_tys.len() == 2 {
        let head_resolved = subst.apply(&arg_tys[0]);
        let tail_resolved = subst.apply(&arg_tys[1]);
        if let (Type::Tensor(head_dims, head_prec), Type::Adt(list_name, list_args)) =
            (&head_resolved, &tail_resolved)
            && list_name == "List"
            && list_args.len() == 1
            && let Type::Tensor(tail_dims, tail_prec) = subst.apply(&list_args[0])
        {
            if head_dims.len() != tail_dims.len() {
                // chelis#255: surface the rank-uniform rule and the
                // reshape/flatten remediation in the diagnostic itself,
                // so users (and agents reading JSON output) are not
                // left guessing why a `List[tensor[k, f32]]` rejected
                // a rank-mixed literal. The dim slot `k` is a
                // dimension variable, not a shape-vector variable;
                // see spec/04-type-system.md §4.5.1.
                errors.push(CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!(
                            "list element rank mismatch: {} dims vs {} dims; \
                             List[tensor[...]] requires rank-uniform elements \
                             (the dim slot is a dimension variable, not a \
                             shape-vector variable). Reshape or flatten \
                             elements to a common rank before listing \
                             (spec/04-type-system.md §4.5.1).",
                            head_dims.len(),
                            tail_dims.len(),
                        ),
                    ),
                    vec![],
                ));
                return Type::Error;
            }
            if let Err(te) = unify_tensor_prec(head_prec, &tail_prec, subst) {
                errors.push(te.into());
                return Type::Error;
            }
            // Per-axis join (chelis#218 concat ergonomics, tightened
            // by chelis#272). Resolve each dim through the current
            // substitution first so already-bound dim variables compare
            // as their concrete value.
            //
            //   * equal concrete literals or equal names  -> keep them;
            //   * genuinely-mismatched concrete literals
            //     (e.g. `Lit(2)` vs `Lit(3)`) or mismatched concrete
            //     names                                    -> widen to
            //     `Wildcard`. This is the deliberate #218 behavior that
            //     lets bare `concat([a, b], axis)` accept ragged
            //     concrete axes; and
            //   * any pair that involves a dimension *variable*
            //     (a declared rigid dim parameter such as `k`/`m`)
            //     -> `unify_dim` the two dims instead of widening.
            //
            // The last arm is the #272 fix: the old `_ => Wildcard`
            // erased named dim variables, so a list body that violated
            // the §4.4 rigid-distinct-dim guarantee
            // (`def make[k, m](a: tensor[k], b: tensor[m])
            //   -> List[tensor[k]] = [a, b]`) collapsed `k`/`m` to a
            // wildcard before `check_declared_dvars_rigid` ran. Unifying
            // them instead keeps the surviving evidence: distinct rigid
            // dims unify with each other (the guard then reports the
            // collapse) and a `(rigid, concrete)` pair pins the rigid
            // dim to a literal (the guard reports the pin). A
            // `unify_dim` failure here (which the permissive
            // Name/Lit/Wildcard arms make rare) surfaces as a structural
            // dimension mismatch rather than being silently widened.
            let mut joined_dims: Vec<Dim> = Vec::with_capacity(head_dims.len());
            for (h, t) in head_dims.iter().zip(tail_dims.iter()) {
                let hr = subst.apply_dim(h);
                let tr = subst.apply_dim(t);
                let joined = match (&hr, &tr) {
                    (Dim::Lit(a), Dim::Lit(b)) if a == b => Dim::Lit(*a),
                    (Dim::Name(n1), Dim::Name(n2)) if n1 == n2 => Dim::Name(n1.clone()),
                    // Mismatched concrete dims (literal/literal or
                    // name/name): the deliberate #218 ragged-axis
                    // widening. Neither side is a dim variable, so there
                    // is no rigid-dim promise to preserve here.
                    (Dim::Lit(_), Dim::Lit(_))
                    | (Dim::Name(_), Dim::Name(_))
                    | (Dim::Lit(_), Dim::Name(_))
                    | (Dim::Name(_), Dim::Lit(_)) => Dim::Wildcard,
                    // At least one side is a dim variable (or a
                    // wildcard). Unify so rigid dim parameters keep their
                    // identity and `check_declared_dvars_rigid` can fire.
                    _ => {
                        if let Err(te) = unify_dim(&hr, &tr, subst) {
                            errors.push(te.into());
                            return Type::Error;
                        }
                        subst.apply_dim(&hr)
                    }
                };
                joined_dims.push(joined);
            }
            let joined_prec = subst.apply_tensor_prec(head_prec);
            let elem = Type::Tensor(joined_dims, joined_prec);
            return Type::Adt("List".to_string(), vec![elem]);
        }
    }

    let ret_tv = vg.fresh_type();

    // Comparison-op tensor/scalar broadcast: when a comparison op
    // (`cmplt`, `eq`, `neq`, `lt`, `gt`, `lte`, `gte`) is called with one
    // tensor argument and one scalar argument of matching precision, the
    // scalar is broadcast across the tensor at eval time. The polymorphic
    // scheme `(α, α) → α` would otherwise reject the call because
    // `tensor[D, p]` does not unify with `Prim(p)`. Rewrite the scalar's
    // type to the tensor type for unification purposes only; the
    // semantic post-check below still validates each original arg type.
    //
    // Ordered comparisons (`lt`, `gt`, `lte`, `gte`, `cmplt`) require
    // matching numeric precision. `eq`/`neq` allow any matching precision
    // (including `bool` and `string`).
    let unify_arg_tys: Vec<Type> = if let Some(ref fname) = func_name
        && builtins::COMPARISON_OPS.contains(&fname.as_str())
        && arg_tys.len() == 2
    {
        let lhs_resolved = type_for_readonly_check(&arg_tys[0], subst);
        let rhs_resolved = type_for_readonly_check(&arg_tys[1], subst);
        let is_eq_family = matches!(fname.as_str(), "eq" | "neq");
        let precisions_compatible = |tensor_prec: &TensorPrec, scalar_prec: &Prim| -> bool {
            // Polymorphic-precision tensors (TensorPrec::Var) are not
            // eligible for the scalar-broadcast rewrite: the rewrite
            // requires a known precision so the rewritten arg type can
            // unify against the actual scalar argument. Leave them to
            // the standard unification path (which will surface a
            // precise PrecisionMismatch if needed).
            match tensor_prec {
                TensorPrec::Concrete(p) => p == scalar_prec && (is_eq_family || p.is_numeric()),
                TensorPrec::Var(_) => false,
            }
        };
        match (&lhs_resolved, &rhs_resolved) {
            (Type::Tensor(dims, tensor_prec), Type::Prim(scalar_prec))
                if precisions_compatible(tensor_prec, scalar_prec) =>
            {
                let tensor_ty = Type::Tensor(dims.clone(), tensor_prec.clone());
                vec![arg_tys[0].clone(), tensor_ty]
            }
            (Type::Prim(scalar_prec), Type::Tensor(dims, tensor_prec))
                if precisions_compatible(tensor_prec, scalar_prec) =>
            {
                let tensor_ty = Type::Tensor(dims.clone(), tensor_prec.clone());
                vec![tensor_ty, arg_tys[1].clone()]
            }
            _ => arg_tys.clone(),
        }
    } else {
        arg_tys.clone()
    };

    let unify_arg_tys = auto_borrow_call_arg_types(&func_ty, unify_arg_tys, subst);
    let expected_fn = Type::Fn(unify_arg_tys, Box::new(ret_tv.clone()));

    const TENSOR_OPS: &[&str] = &[
        "add",
        "mul",
        "sub",
        "div",
        "floor_div",
        "trunc_div",
        "neg",
        "recip",
        "exp",
        "log",
        "sin",
        "sqrt",
        "relu",
        "sigmoid",
        "tanh",
        "silu",
        "gelu",
        "matmul",
        "layer_norm",
        "max_elem",
        "min_elem",
        "normalize",
        "cmplt",
        "eq",
        "neq",
        "lt",
        "gt",
        "lte",
        "gte",
        "and",
        "or",
        "not",
    ];

    const LOGICAL_OPS: &[&str] = &["and", "or", "not"];
    const INT_BINOPS: &[&str] = &["mod", "bitand", "bitor", "bitxor"];
    const INT_SHIFT_OPS: &[&str] = &["shl", "shr"];

    match unify(&func_ty, &expected_fn, subst) {
        Ok(()) => {
            let mut result_ty = subst.apply(&ret_tv);

            // Post-check: shared builtins can operate on either tensors or host scalars.
            if let Some(ref fname) = func_name
                && TENSOR_OPS.contains(&fname.as_str())
            {
                for arg_ty in &arg_tys {
                    let resolved = type_for_readonly_check(arg_ty, subst);
                    let ok = match fname.as_str() {
                        "matmul" | "layer_norm" | "normalize" => {
                            matches!(resolved, Type::Tensor(_, _) | Type::Var(_) | Type::Error)
                        }
                        "add" | "mul" | "sub" | "max_elem" | "min_elem" | "neg" => {
                            matches!(resolved, Type::Tensor(_, _) | Type::Var(_) | Type::Error)
                                || matches!(resolved, Type::Prim(prec) if prec.is_numeric())
                        }
                        "floor_div" => {
                            // chelis#178: `floor_div` accepts both integer
                            // and float operands (round toward -inf for
                            // ints, `floor(a/b)` for floats), so any
                            // numeric precision is admissible.
                            matches!(resolved, Type::Tensor(_, _) | Type::Var(_) | Type::Error)
                                || matches!(resolved, Type::Prim(prec) if prec.is_numeric())
                        }
                        "div" => {
                            // chelis#178: `div` is float-only. Integer
                            // operands are a type error pointing at
                            // `floor_div` / `trunc_div` (see the rejection
                            // diagnostic below). An unresolved
                            // `TensorPrec::Var(_)` is accepted so a
                            // polymorphic body type-checks; the cross-row
                            // pass `validate_polymorphic_op_constraints`
                            // catches integer instantiations at the call
                            // site.
                            matches!(
                                resolved,
                                Type::Tensor(_, TensorPrec::Var(_)) | Type::Var(_) | Type::Error
                            ) || matches!(resolved, Type::Tensor(_, TensorPrec::Concrete(p)) if p.is_float())
                                || matches!(resolved, Type::Prim(prec) if prec.is_float())
                        }
                        "trunc_div" => {
                            // chelis#178: `trunc_div` is integer-only (the
                            // C/Rust truncating quotient). Float operands
                            // are a type error. Unresolved precision vars
                            // are accepted for polymorphic bodies.
                            matches!(
                                resolved,
                                Type::Tensor(_, TensorPrec::Var(_)) | Type::Var(_) | Type::Error
                            ) || matches!(resolved, Type::Tensor(_, TensorPrec::Concrete(p)) if p.is_integer())
                                || matches!(resolved, Type::Prim(prec) if prec.is_integer())
                        }
                        "exp" | "log" | "sin" | "sqrt" | "relu" | "sigmoid" | "tanh" | "silu"
                        | "gelu" | "recip" => {
                            // WS-A8 / RT-3 F3: spec/04-type-system.md §5.4
                            // restricts transcendental ops to float
                            // precisions (f32, f64, bf16, f16). The
                            // tensor form was previously admitted with
                            // any precision, slipping integer instantiations
                            // past the type checker; the scalar form
                            // already enforced this. Tensors carrying
                            // an unresolved `TensorPrec::Var(_)` are
                            // accepted here so a polymorphic body
                            // type-checks; the cross-row enforcement
                            // pass `validate_polymorphic_op_constraints`
                            // catches integer instantiations at the
                            // call site.
                            matches!(
                                resolved,
                                Type::Tensor(_, TensorPrec::Var(_)) | Type::Var(_) | Type::Error
                            ) || matches!(resolved, Type::Tensor(_, TensorPrec::Concrete(p)) if p.is_float())
                                || matches!(resolved, Type::Prim(prec) if prec.is_float())
                        }
                        "cmplt" | "lt" | "gt" | "lte" | "gte" => {
                            matches!(resolved, Type::Tensor(_, _) | Type::Var(_) | Type::Error)
                                || matches!(resolved, Type::Prim(prec) if prec.is_numeric())
                        }
                        "eq" | "neq" => {
                            matches!(resolved, Type::Tensor(_, _) | Type::Var(_) | Type::Error)
                                || matches!(resolved, Type::Prim(_))
                        }
                        "and" | "or" | "not" => {
                            matches!(
                                resolved,
                                Type::Tensor(_, TensorPrec::Concrete(Prim::Bool))
                                    | Type::Var(_)
                                    | Type::Error
                            ) || matches!(resolved, Type::Prim(Prim::Bool))
                        }
                        _ => matches!(resolved, Type::Tensor(_, _) | Type::Var(_) | Type::Error),
                    };
                    if !ok {
                        // WS-A8: surface a §5.4 citation when a
                        // transcendental rejects an integer tensor.
                        let is_transcendental = matches!(
                            fname.as_str(),
                            "exp"
                                | "log"
                                | "sin"
                                | "sqrt"
                                | "relu"
                                | "sigmoid"
                                | "tanh"
                                | "silu"
                                | "gelu"
                                | "recip"
                        );
                        let resolved_int_prec = match &resolved {
                            Type::Tensor(_, TensorPrec::Concrete(p)) if p.is_integer() => {
                                Some(p.name())
                            }
                            Type::Prim(p) if p.is_integer() => Some(p.name()),
                            _ => None,
                        };
                        let resolved_float_prec = match &resolved {
                            Type::Tensor(_, TensorPrec::Concrete(p)) if p.is_float() => {
                                Some(p.name())
                            }
                            Type::Prim(p) if p.is_float() => Some(p.name()),
                            _ => None,
                        };
                        let (kind, message, hints) = if is_transcendental
                            && let Type::Tensor(_, TensorPrec::Concrete(p)) = &resolved
                            && !p.is_float()
                        {
                            (
                                CheckErrorKind::PrecisionMismatch,
                                format!(
                                    "{fname} on operand precision `{}` is not admitted per \
                                     spec/04-type-system.md \u{00a7}5.4: transcendental \
                                     operations are restricted to f32, f64, bf16, f16 (not \
                                     integer)",
                                    p.name()
                                ),
                                vec![format!(
                                    "spec/04-type-system.md \u{00a7}5.4: cast to a float \
                                     precision before applying `{fname}`."
                                )],
                            )
                        } else if fname == "div"
                            && let Some(pname) = resolved_int_prec
                        {
                            // chelis#178: integer `div` is rejected; point
                            // the user at the integer-division ops.
                            (
                                CheckErrorKind::PrecisionMismatch,
                                format!(
                                    "div on integer operand precision `{pname}` is not admitted \
                                     per spec/05-risc-primitives.md \u{00a7}2.1: `div` is \
                                     float-only (IEEE-754). Use `floor_div` (round toward -inf) \
                                     or `trunc_div` (round toward zero) for integers."
                                ),
                                vec![
                                    "spec/05-risc-primitives.md \u{00a7}2.1: integer division \
                                     uses `floor_div` or `trunc_div`; `div` requires float \
                                     operands."
                                        .to_string(),
                                ],
                            )
                        } else if fname == "trunc_div"
                            && let Some(pname) = resolved_float_prec
                        {
                            // chelis#178: `trunc_div` is integer-only.
                            (
                                CheckErrorKind::PrecisionMismatch,
                                format!(
                                    "trunc_div on float operand precision `{pname}` is not \
                                     admitted per spec/05-risc-primitives.md \u{00a7}2.1: \
                                     `trunc_div` is integer-only. Use `div` for IEEE-754 float \
                                     division, or `floor_div` for a floored float quotient."
                                ),
                                vec![
                                    "spec/05-risc-primitives.md \u{00a7}2.1: `trunc_div` requires \
                                     integer operands."
                                        .to_string(),
                                ],
                            )
                        } else {
                            (
                                CheckErrorKind::TypeMismatch,
                                format!(
                                    "{fname} does not accept argument type {resolved} in this \
                                     context"
                                ),
                                vec![],
                            )
                        };
                        errors.push(CheckError::new(
                            kind,
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
                                message,
                            ),
                            hints,
                        ));
                        return Type::Error;
                    }
                }
            }

            if let Some(ref fname) = func_name
                && matches!(
                    fname.as_str(),
                    "softmax"
                        | "mean"
                        | "sum"
                        | "max_reduce"
                        | "min_reduce"
                        | "prod_reduce"
                        | "argmax_reduce"
                        | "argmin_reduce"
                )
            {
                if let Some(first_arg) = arg_tys.first() {
                    let resolved = type_for_readonly_check(first_arg, subst);
                    match &resolved {
                        Type::Tensor(_, _) | Type::Var(_) | Type::Error => {}
                        _ => {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!("{} expects tensor input, got {}", fname, resolved),
                                ),
                                vec![],
                            ));
                            return Type::Error;
                        }
                    }
                    // WS-A8 / RT-3 F3: softmax is a transcendental row
                    // op per spec/04-type-system.md \u{00a7}5.4 and must
                    // reject integer operand precisions. Polymorphic
                    // (`Var`) precisions are accepted here so a poly
                    // body type-checks; the cross-row enforcement pass
                    // catches integer call-site instantiations.
                    if fname == "softmax"
                        && let Type::Tensor(_, TensorPrec::Concrete(p)) = &resolved
                        && !p.is_float()
                    {
                        errors.push(CheckError::new(
                            CheckErrorKind::PrecisionMismatch,
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
                                format!(
                                    "softmax on operand precision `{}` is not admitted per \
                                     spec/04-type-system.md \u{00a7}5.4: transcendental \
                                     operations are restricted to f32, f64, bf16, f16 \
                                     (not integer)",
                                    p.name()
                                ),
                            ),
                            vec![format!(
                                "spec/04-type-system.md \u{00a7}5.4: cast to a float \
                                 precision before applying softmax."
                            )],
                        ));
                        return Type::Error;
                    }
                }

                if let Some(axis_arg) = arg_tys.get(1) {
                    let resolved = subst.apply(axis_arg);
                    match &resolved {
                        Type::Prim(Prim::Int32) | Type::Var(_) | Type::Error => {}
                        _ => {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!("{} expects int32 axis, got {}", fname, resolved),
                                ),
                                vec![],
                            ));
                            return Type::Error;
                        }
                    }
                }

                // softmax does not go through `check_reduction_signature`
                // (it is shape-preserving, not shape-reducing), so its
                // axis range is validated here. Negative axes index from
                // the end via `normalize_static_axis`, consistent with
                // the reductions and gather/scatter.
                // Issue #216: cast-aware so `softmax(x, cast(N, int32))`
                // surfaces the bounds-check diagnostic at infer.
                if fname == "softmax"
                    && let Some(first_arg) = arg_tys.first()
                    && let Type::Tensor(dims, _) = type_for_readonly_check(first_arg, subst)
                    && let Some(raw) = kids.get(2).and_then(extract_int_for_dim)
                    && normalize_static_axis(dims.len(), raw).is_none()
                {
                    errors.push(CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        with_macro_provenance(
                            &deep::Expr::List(list.clone(), zero_span()),
                            format!(
                                "softmax axis {raw} is out of bounds for rank {} tensor",
                                dims.len()
                            ),
                        ),
                        vec![],
                    ));
                    return Type::Error;
                }
            }

            if let Some(ref fname) = func_name
                && fname == "uniform_like"
            {
                if let Some(first_arg) = arg_tys.first() {
                    let resolved = type_for_readonly_check(first_arg, subst);
                    match &resolved {
                        Type::Tensor(_, prim) if prim.is_float() => {}
                        Type::Tensor(_, _) => {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!(
                                        "uniform_like expects a float tensor template, got {}",
                                        resolved
                                    ),
                                ),
                                vec![],
                            ));
                            return Type::Error;
                        }
                        Type::Var(_) | Type::Error => {}
                        _ => {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!(
                                        "uniform_like expects tensor template input, got {}",
                                        resolved
                                    ),
                                ),
                                vec![],
                            ));
                            return Type::Error;
                        }
                    }
                }

                for (index, arg_ty) in arg_tys.iter().enumerate().skip(1).take(2) {
                    let resolved = type_for_readonly_check(arg_ty, subst);
                    match &resolved {
                        Type::Prim(Prim::F32) | Type::Var(_) | Type::Error => {}
                        _ => {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!(
                                        "uniform_like expects f32 bounds for args 2-3, got {}",
                                        resolved
                                    ),
                                ),
                                vec![],
                            ));
                            return Type::Error;
                        }
                    }

                    if let Some(expr) = kids.get(index + 1)
                        && !is_numeric_literal_expr(expr)
                    {
                        errors.push(CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
                                "uniform_like currently requires literal low/high bounds"
                                    .to_string(),
                            ),
                            vec![],
                        ));
                        return Type::Error;
                    }
                }
            }

            if let Some(ref fname) = func_name
                && fname == "dropout"
            {
                if let Some(first_arg) = arg_tys.first() {
                    let resolved = type_for_readonly_check(first_arg, subst);
                    match &resolved {
                        Type::Tensor(_, _) | Type::Var(_) | Type::Error => {}
                        _ => {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!("dropout expects tensor input, got {}", resolved),
                                ),
                                vec![],
                            ));
                            return Type::Error;
                        }
                    }
                }

                if let Some(rate_arg) = arg_tys.get(1) {
                    let resolved = subst.apply(rate_arg);
                    match &resolved {
                        Type::Prim(Prim::F32) | Type::Var(_) | Type::Error => {}
                        _ => {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!("dropout expects f32 rate, got {}", resolved),
                                ),
                                vec![],
                            ));
                            return Type::Error;
                        }
                    }
                }
            }

            if let Some(ref fname) = func_name
                && fname == "conv2d"
            {
                for (index, arg_ty) in arg_tys.iter().enumerate() {
                    let resolved = type_for_readonly_check(arg_ty, subst);
                    if index < 2 {
                        match &resolved {
                            Type::Tensor(_, _) | Type::Var(_) | Type::Error => {}
                            _ => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "conv2d expects tensor inputs for args 1-2, got {}",
                                            resolved
                                        ),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    } else {
                        match &resolved {
                            Type::Prim(Prim::Int32) | Type::Var(_) | Type::Error => {}
                            _ => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "conv2d expects int32 stride/padding, got {}",
                                            resolved
                                        ),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                }
            }

            if let Some(ref fname) = func_name
                && INT_BINOPS.contains(&fname.as_str())
            {
                let lhs = arg_tys
                    .first()
                    .map(|ty| subst.apply(ty))
                    .unwrap_or(Type::Error);
                let rhs = arg_tys
                    .get(1)
                    .map(|ty| subst.apply(ty))
                    .unwrap_or(Type::Error);
                match (&lhs, &rhs) {
                    (Type::Prim(lhs_prec), Type::Prim(rhs_prec))
                        if lhs_prec.is_integer()
                            && rhs_prec.is_integer()
                            && lhs_prec == rhs_prec =>
                    {
                        return Type::Prim(*lhs_prec);
                    }
                    (Type::Var(_), Type::Prim(rhs_prec)) if rhs_prec.is_integer() => {
                        return lhs;
                    }
                    (Type::Prim(lhs_prec), Type::Var(_)) if lhs_prec.is_integer() => {
                        return lhs;
                    }
                    (Type::Var(_), Type::Var(_)) | (Type::Error, _) | (_, Type::Error) => {
                        return lhs;
                    }
                    _ => {
                        errors.push(CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
                                format!(
                                    "{} requires matching integer arguments, got {} and {}",
                                    fname, lhs, rhs
                                ),
                            ),
                            vec![],
                        ));
                        return Type::Error;
                    }
                }
            }

            if let Some(ref fname) = func_name
                && INT_SHIFT_OPS.contains(&fname.as_str())
            {
                let lhs = arg_tys
                    .first()
                    .map(|ty| subst.apply(ty))
                    .unwrap_or(Type::Error);
                let rhs = arg_tys
                    .get(1)
                    .map(|ty| subst.apply(ty))
                    .unwrap_or(Type::Error);
                let lhs_ok = matches!(&lhs, Type::Prim(prec) if prec.is_integer())
                    || matches!(&lhs, Type::Var(_) | Type::Error);
                let rhs_ok = matches!(&rhs, Type::Prim(prec) if prec.is_integer())
                    || matches!(&rhs, Type::Var(_) | Type::Error);
                if lhs_ok && rhs_ok {
                    return lhs;
                }
                errors.push(CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!(
                            "{} requires integer lhs and shift amount, got {} and {}",
                            fname, lhs, rhs
                        ),
                    ),
                    vec![],
                ));
                return Type::Error;
            }

            if let Some(ref fname) = func_name {
                match fname.as_str() {
                    "matmul" => {
                        result_ty = check_matmul_signature(&arg_tys, &result_ty, subst, errors);
                    }
                    "sum" | "max_reduce" | "min_reduce" | "prod_reduce" | "argmax_reduce"
                    | "argmin_reduce" | "mean" => {
                        result_ty = check_reduction_signature(
                            fname,
                            &kids[1..],
                            &arg_tys,
                            &result_ty,
                            subst,
                            errors,
                        );
                    }
                    "expand" => {
                        // chelis#339: the axis slot is a dim NAME (the
                        // named-axis insert form) only when it is not bound in
                        // the value environment — a bound `int32` var is the
                        // issue #259 runtime-value class instead.
                        let axis_is_dim_name = kids.get(2).is_some_and(|arg| {
                            symbolic_dim_ref_name(arg)
                                .is_some_and(|name| env.lookup(name).is_none())
                        });
                        // chelis#397/#469: classify the size by PROVENANCE
                        // (static / shape-sourced / sourceless), following
                        // `let`/`cast`/arithmetic to a tensor shape source.
                        // A truly sourceless runtime scalar is rejected at
                        // check so it never reaches the build/eval-only
                        // rejection (a check-clean program must build).
                        let size_class = kids
                            .get(3)
                            .map(|arg| classify_expand_size(arg, env))
                            .unwrap_or(SizeClass::Unknown);
                        result_ty = check_expand_signature(
                            &kids[1..],
                            &arg_tys,
                            &result_ty,
                            axis_is_dim_name,
                            size_class,
                            env,
                            subst,
                            errors,
                        );
                    }
                    "layer_norm" => {
                        result_ty =
                            check_layer_norm_signature(&arg_tys, &result_ty, vg, subst, errors);
                    }
                    "conv2d" => {
                        result_ty = check_conv2d_signature(
                            &kids[1..],
                            &arg_tys,
                            &result_ty,
                            vg,
                            subst,
                            errors,
                        );
                    }
                    _ => {}
                }
            }

            // Post-check: logical ops require tensor[D, bool] arguments
            if let Some(ref fname) = func_name
                && LOGICAL_OPS.contains(&fname.as_str())
            {
                for arg_ty in &arg_tys {
                    let resolved = type_for_readonly_check(arg_ty, subst);
                    match &resolved {
                        Type::Tensor(_, TensorPrec::Concrete(Prim::Bool))
                        | Type::Prim(Prim::Bool)
                        | Type::Var(_)
                        | Type::Error => {} // OK
                        Type::Tensor(_, prec) => {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!(
                                        "{} requires tensor[D, bool] arguments, got tensor[D, {}]",
                                        fname,
                                        prec.name()
                                    ),
                                ),
                                vec!["Logical ops only work on bool tensors".to_string()],
                            ));
                            return Type::Error;
                        }
                        other => {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!("{} requires bool arguments, got {}", fname, other),
                                ),
                                vec!["Logical ops only work on bool values".to_string()],
                            ));
                            return Type::Error;
                        }
                    }
                }
            }

            // Special case: comparison ops return tensor[D, bool] when any
            // argument is tensor-shaped. Comparison ops broadcast a scalar
            // arg against a tensor arg (see the rewrite block above), so the
            // result shape comes from whichever argument is the tensor —
            // not necessarily the first one (issue #5: `gt(1.5, xs)` was
            // returning `Prim(Bool)` instead of `tensor[D, bool]` because
            // this override only looked at `arg_tys[0]`).
            if let Some(ref fname) = func_name
                && builtins::COMPARISON_OPS.contains(&fname.as_str())
            {
                // Prefer any tensor-shaped arg as the dim source.
                let tensor_dims = arg_tys.iter().find_map(|t| match subst.apply(t) {
                    Type::Tensor(dims, _) => Some(dims),
                    _ => None,
                });
                if let Some(dims) = tensor_dims {
                    return Type::Tensor(dims, TensorPrec::Concrete(Prim::Bool));
                }
                // No tensor arg → scalar comparison, returns scalar bool.
                if let Some(first_arg) = arg_tys.first() {
                    let resolved_arg = type_for_readonly_check(first_arg, subst);
                    if matches!(resolved_arg, Type::Prim(_)) {
                        return Type::Prim(Prim::Bool);
                    }
                }
            }

            if let Some(ref fname) = func_name {
                match fname.as_str() {
                    "print" => {
                        return Type::Unit;
                    }
                    "fail" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Prim(Prim::String) | Type::Var(_) | Type::Error => {
                                    return Type::Var(vg.fresh_tvar());
                                }
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("fail expects string input, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "debug" => {
                        if let Some(first_arg) = arg_tys.first() {
                            return subst.apply(first_arg);
                        }
                    }
                    "string_len" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Prim(Prim::String) | Type::Var(_) | Type::Error => {
                                    return Type::Prim(Prim::Int64);
                                }
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("string_len expects string input, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "string_concat" => {
                        for arg_ty in &arg_tys {
                            match subst.apply(arg_ty) {
                                Type::Prim(Prim::String) | Type::Var(_) | Type::Error => {}
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!(
                                                "string_concat expects string arguments, got {other}"
                                            ),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                        return Type::Prim(Prim::String);
                    }
                    "string_slice" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Prim(Prim::String) | Type::Var(_) | Type::Error => {}
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!(
                                                "string_slice expects string input, got {other}"
                                            ),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                        for (index, arg_ty) in arg_tys.iter().enumerate().skip(1) {
                            match subst.apply(arg_ty) {
                                Type::Prim(precision) if precision.is_integer() => {}
                                Type::Var(_) | Type::Error => {}
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!(
                                                "string_slice expects integer index arguments; arg {} was {other}",
                                                index + 1
                                            ),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                        return Type::Prim(Prim::String);
                    }
                    "string_contains" | "string_starts_with" | "string_ends_with" => {
                        for arg_ty in &arg_tys {
                            match subst.apply(arg_ty) {
                                Type::Prim(Prim::String) | Type::Var(_) | Type::Error => {}
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!(
                                                "{} expects string arguments, got {other}",
                                                fname
                                            ),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                        return Type::Prim(Prim::Bool);
                    }
                    "string_trim" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Prim(Prim::String) | Type::Var(_) | Type::Error => {
                                    return Type::Prim(Prim::String);
                                }
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!(
                                                "string_trim expects string input, got {other}"
                                            ),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "to_string" => {
                        return Type::Prim(Prim::String);
                    }
                    "to_int" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Prim(Prim::String) | Type::Var(_) | Type::Error => {
                                    return Type::Adt(
                                        "Option".to_string(),
                                        vec![Type::Prim(Prim::Int64)],
                                    );
                                }
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("to_int expects string input, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "to_float" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Prim(Prim::String) | Type::Var(_) | Type::Error => {
                                    return Type::Adt(
                                        "Option".to_string(),
                                        vec![Type::Prim(Prim::F64)],
                                    );
                                }
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("to_float expects string input, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "rank" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match type_for_readonly_check(first_arg, subst) {
                                Type::Tensor(_, _) | Type::Var(_) | Type::Error => {
                                    return Type::Prim(Prim::Int32);
                                }
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("rank expects tensor input, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "shape" => {
                        let input_dims = if let Some(first_arg) = arg_tys.first() {
                            match type_for_readonly_check(first_arg, subst) {
                                Type::Tensor(dims, _) => Some(dims),
                                Type::Var(_) | Type::Error => None,
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("shape expects tensor input, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        } else {
                            None
                        };
                        // Issue #216: cast-aware so `shape(x, cast(N, int32))`
                        // (the idiomatic form from issue #206 for runtime-dim
                        // reshape) surfaces the same diagnostic as the bare-
                        // literal form.
                        if let Some(axis_expr) = kids.get(2)
                            && let Some(axis) = extract_int_for_dim(axis_expr)
                        {
                            if axis < 0 {
                                errors.push(CheckError::new(
                                    CheckErrorKind::DimensionMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!("shape requires non-negative axis, got {axis}"),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                            if let Some(dims) = input_dims.as_ref() {
                                let axis = axis as usize;
                                if axis >= dims.len() {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::DimensionMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!(
                                                "shape axis {axis} is out of bounds for rank {} tensor",
                                                dims.len()
                                            ),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                        if let Some(axis_arg) = arg_tys.get(1) {
                            match subst.apply(axis_arg) {
                                Type::Prim(Prim::Int32) | Type::Var(_) | Type::Error => {
                                    return Type::Prim(Prim::Int32);
                                }
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("shape expects int32 axis, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "numel" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match type_for_readonly_check(first_arg, subst) {
                                Type::Tensor(_, _) | Type::Var(_) | Type::Error => {
                                    return Type::Prim(Prim::Int64);
                                }
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("numel expects tensor input, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "tensor_to_scalar" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match type_for_readonly_check(first_arg, subst) {
                                Type::Tensor(dims, precision) => {
                                    if !dims.is_empty() {
                                        errors.push(CheckError::new(
                                            CheckErrorKind::TypeMismatch,
                                            with_macro_provenance(
                                                &deep::Expr::List(list.clone(), zero_span()),
                                                "tensor_to_scalar expects a rank-0 tensor"
                                                    .to_string(),
                                            ),
                                            vec![],
                                        ));
                                        return Type::Error;
                                    }
                                    // tensor_to_scalar requires a fully
                                    // resolved precision. Polymorphic precision
                                    // must be resolved by unification before
                                    // this op can name a host scalar type.
                                    return match precision {
                                        TensorPrec::Concrete(p) => Type::Prim(p),
                                        TensorPrec::Var(_) => result_ty,
                                    };
                                }
                                Type::Var(_) | Type::Error => return result_ty,
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!(
                                                "tensor_to_scalar expects tensor input, got {other}"
                                            ),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "scalar_to_tensor" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Prim(precision) if !matches!(precision, Prim::String) => {
                                    return Type::Tensor(vec![], TensorPrec::Concrete(precision));
                                }
                                Type::Var(_) | Type::Error => return result_ty,
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!(
                                                "scalar_to_tensor expects scalar numeric/bool input, got {other}"
                                            ),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "einsum" => {
                        if arg_tys.len() != 3 {
                            return Type::Error;
                        }
                        let Some(equation) = kids.get(1).and_then(extract_string_literal) else {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    "einsum expects a string equation as its first argument"
                                        .to_string(),
                                ),
                                vec![],
                            ));
                            return Type::Error;
                        };
                        if equation.contains("...") {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    "einsum ellipsis support is deferred in 3h".to_string(),
                                ),
                                vec![],
                            ));
                            return Type::Error;
                        }
                        return result_ty;
                    }
                    "gather" => {
                        if arg_tys.len() != 3 {
                            return Type::Error;
                        }
                        let tensor_ty = type_for_readonly_check(&arg_tys[0], subst);
                        let indices_ty = type_for_readonly_check(&arg_tys[1], subst);
                        let Some(axis) =
                            resolve_builtin_axis("gather", kids.get(3), &tensor_ty, list, errors)
                        else {
                            return Type::Error;
                        };
                        match infer_gather_result_type(&tensor_ty, &indices_ty, axis) {
                            Ok(ty) => return ty,
                            Err(message) => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        message,
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "where" => {
                        if arg_tys.len() != 3 {
                            return Type::Error;
                        }
                        let cond_ty = type_for_readonly_check(&arg_tys[0], subst);
                        let then_ty = type_for_readonly_check(&arg_tys[1], subst);
                        let else_ty = type_for_readonly_check(&arg_tys[2], subst);
                        match (&cond_ty, &then_ty, &else_ty) {
                            (
                                Type::Tensor(cond_dims, TensorPrec::Concrete(Prim::Bool)),
                                Type::Tensor(then_dims, then_prec),
                                Type::Tensor(else_dims, else_prec),
                            ) => {
                                if then_prec != else_prec
                                    || cond_dims != then_dims
                                    || then_dims != else_dims
                                {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!(
                                                "where expects cond/both branches to have matching tensor shapes and branch precision, got {cond_ty}, {then_ty}, and {else_ty}"
                                            ),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                                return Type::Tensor(then_dims.clone(), then_prec.clone());
                            }
                            (Type::Var(_), _, _)
                            | (_, Type::Var(_), _)
                            | (_, _, Type::Var(_))
                            | (Type::Error, _, _)
                            | (_, Type::Error, _)
                            | (_, _, Type::Error) => return result_ty,
                            _ => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "where expects a bool tensor condition and matching tensor branches, got {cond_ty}, {then_ty}, and {else_ty}"
                                        ),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "cumsum" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        let cumsum_operand = type_for_readonly_check(&arg_tys[0], subst);
                        let Some(_axis) = resolve_builtin_axis(
                            "cumsum",
                            kids.get(2),
                            &cumsum_operand,
                            list,
                            errors,
                        ) else {
                            return Type::Error;
                        };
                        match cumsum_operand {
                            Type::Tensor(dims, precision) => {
                                return Type::Tensor(dims, precision);
                            }
                            Type::Var(_) | Type::Error => return result_ty,
                            other => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!("cumsum expects tensor input, got {other}"),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "diagonal" => {
                        if arg_tys.len() != 3 {
                            return Type::Error;
                        }
                        let diagonal_operand = type_for_readonly_check(&arg_tys[0], subst);
                        let axis1 = match resolve_axis_pair_member(
                            "diagonal",
                            kids.get(2),
                            &diagonal_operand,
                            0,
                            list,
                            errors,
                        ) {
                            Ok(axis) => axis,
                            Err(()) => return Type::Error,
                        };
                        let axis2 = match resolve_axis_pair_member(
                            "diagonal",
                            kids.get(3),
                            &diagonal_operand,
                            1,
                            list,
                            errors,
                        ) {
                            Ok(axis) => axis,
                            Err(()) => return Type::Error,
                        };
                        match infer_diagonal_result_type(&diagonal_operand, axis1, axis2) {
                            Ok(ty) => return ty,
                            Err(message) => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        message,
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "trace" => {
                        if arg_tys.len() != 3 {
                            return Type::Error;
                        }
                        let trace_operand = type_for_readonly_check(&arg_tys[0], subst);
                        let axis1 = match resolve_axis_pair_member(
                            "trace",
                            kids.get(2),
                            &trace_operand,
                            0,
                            list,
                            errors,
                        ) {
                            Ok(axis) => axis,
                            Err(()) => return Type::Error,
                        };
                        let axis2 = match resolve_axis_pair_member(
                            "trace",
                            kids.get(3),
                            &trace_operand,
                            1,
                            list,
                            errors,
                        ) {
                            Ok(axis) => axis,
                            Err(()) => return Type::Error,
                        };
                        match infer_trace_result_type(&trace_operand, axis1, axis2) {
                            Ok(ty) => return ty,
                            Err(message) => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        message,
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "clamp" => {
                        if arg_tys.len() != 3 {
                            return Type::Error;
                        }
                        let input_ty = type_for_readonly_check(&arg_tys[0], subst);
                        let low_ty = type_for_readonly_check(&arg_tys[1], subst);
                        let high_ty = type_for_readonly_check(&arg_tys[2], subst);
                        match (&input_ty, &low_ty, &high_ty) {
                            (
                                Type::Tensor(input_dims, input_prec),
                                Type::Tensor(low_dims, low_prec),
                                Type::Tensor(high_dims, high_prec),
                            ) => {
                                let low_ok = low_dims.is_empty() || low_dims == input_dims;
                                let high_ok = high_dims.is_empty() || high_dims == input_dims;
                                if low_ok
                                    && high_ok
                                    && low_prec == input_prec
                                    && high_prec == input_prec
                                {
                                    return input_ty;
                                }
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "clamp expects tensor input plus scalar-tensor or matching-shape tensor bounds of the same precision, got {input_ty}, {low_ty}, and {high_ty}"
                                        ),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                            (Type::Var(_), _, _)
                            | (_, Type::Var(_), _)
                            | (_, _, Type::Var(_))
                            | (Type::Error, _, _)
                            | (_, Type::Error, _)
                            | (_, _, Type::Error) => return result_ty,
                            _ => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "clamp expects tensor input and tensor bounds, got {input_ty}, {low_ty}, and {high_ty}"
                                        ),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "sort" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        let sort_operand = type_for_readonly_check(&arg_tys[0], subst);
                        let Some(_axis) =
                            resolve_builtin_axis("sort", kids.get(2), &sort_operand, list, errors)
                        else {
                            return Type::Error;
                        };
                        match sort_operand {
                            Type::Tensor(dims, precision) => {
                                return Type::Tuple(vec![
                                    Type::Tensor(dims.clone(), precision),
                                    Type::Tensor(dims, TensorPrec::Concrete(Prim::Int64)),
                                ]);
                            }
                            Type::Var(_) | Type::Error => return result_ty,
                            other => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!("sort expects tensor input, got {other}"),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "scatter" => {
                        if arg_tys.len() != 5 {
                            return Type::Error;
                        }
                        let base_ty = subst.apply(&arg_tys[0]);
                        let indices_ty = subst.apply(&arg_tys[1]);
                        let updates_ty = subst.apply(&arg_tys[2]);
                        let Some(axis) =
                            resolve_builtin_axis("scatter", kids.get(4), &base_ty, list, errors)
                        else {
                            return Type::Error;
                        };
                        let mode = kids.get(5).and_then(extract_string_literal);
                        match mode.as_deref() {
                            Some("replace") | Some("add") => {}
                            _ => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        "scatter mode must be \"replace\" or \"add\"".to_string(),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                        match infer_gather_result_type(&base_ty, &indices_ty, axis) {
                            Ok(expected_updates) => {
                                if let Err(te) = unify(&expected_updates, &updates_ty, subst) {
                                    errors.push(te.into());
                                    return Type::Error;
                                }
                                return base_ty;
                            }
                            Err(message) => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        message,
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "scatter_replace" => {
                        // Tensor-lane replace-scatter (last-write-wins) — lowers
                        // to RiscOp::Scatter. Distinct from the host-lane
                        // `scatter(..., mode)` pentaop. AD policy: no_grad
                        // (rejected via AdError::NotSupported); see
                        // spec/05-risc-primitives.md §3.5.
                        if arg_tys.len() != 4 {
                            return Type::Error;
                        }
                        let base_ty = subst.apply(&arg_tys[0]);
                        let indices_ty = subst.apply(&arg_tys[1]);
                        let updates_ty = subst.apply(&arg_tys[2]);
                        let Some(axis) = resolve_builtin_axis(
                            "scatter_replace",
                            kids.get(4),
                            &base_ty,
                            list,
                            errors,
                        ) else {
                            return Type::Error;
                        };
                        match infer_gather_result_type(&base_ty, &indices_ty, axis) {
                            Ok(expected_updates) => {
                                if let Err(te) = unify(&expected_updates, &updates_ty, subst) {
                                    errors.push(te.into());
                                    return Type::Error;
                                }
                                return base_ty;
                            }
                            Err(message) => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        message,
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "len" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Adt(name, _) if name == "List" || name == "Dict" => {
                                    return Type::Prim(Prim::Int64);
                                }
                                Type::Var(_) | Type::Error => return Type::Prim(Prim::Int64),
                                Type::Ref(inner) if matches!(&*inner, Type::Adt(name, _) if name == "List" || name == "Dict") =>
                                {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!(
                                                "len auto-borrows its List/Dict argument, so an explicit `&` is not a \
                                                 supported surface form: write `len(xs)`, not `len(&xs)` (got &{inner})"
                                            ),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("len expects List or Dict input, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "index" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        let list_arg = subst.apply(&arg_tys[0]);
                        let index_arg = subst.apply(&arg_tys[1]);
                        if !matches!(index_arg, Type::Prim(prec) if prec.is_integer())
                            && !matches!(index_arg, Type::Var(_) | Type::Error)
                        {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!("index expects integer index, got {index_arg}"),
                                ),
                                vec![],
                            ));
                            return Type::Error;
                        }
                        match list_arg {
                            Type::Adt(name, mut args) if name == "List" && args.len() == 1 => {
                                return args.remove(0);
                            }
                            Type::Var(_) | Type::Error => return result_ty,
                            Type::Ref(inner) if matches!(&*inner, Type::Adt(name, _) if name == "List") =>
                            {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "index auto-borrows its List argument, so an explicit `&` is not a \
                                             supported surface form: write `index(xs, i)`, not `index(&xs, i)` (got &{inner})"
                                        ),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                            other => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!("index expects List input, got {other}"),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "append" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        let list_arg = subst.apply(&arg_tys[0]);
                        let value_arg = subst.apply(&arg_tys[1]);
                        match list_arg {
                            Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                                if let Err(te) = unify(&args[0], &value_arg, subst) {
                                    errors.push(te.into());
                                    return Type::Error;
                                }
                                return Type::Adt("List".to_string(), vec![subst.apply(&args[0])]);
                            }
                            Type::Var(_) | Type::Error => return result_ty,
                            other => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!("append expects List input, got {other}"),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "concat" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        let lhs = subst.apply(&arg_tys[0]);
                        let rhs = subst.apply(&arg_tys[1]);
                        match (lhs, rhs) {
                            (Type::Adt(lhs_name, lhs_args), Type::Prim(precision))
                                if lhs_name == "List"
                                    && lhs_args.len() == 1
                                    && precision.is_integer() =>
                            {
                                match tensor_concat_result_type(&lhs_args[0]) {
                                    Ok(ty) => return ty,
                                    Err(message) => {
                                        errors.push(CheckError::new(
                                            CheckErrorKind::TypeMismatch,
                                            with_macro_provenance(
                                                &deep::Expr::List(list.clone(), zero_span()),
                                                message,
                                            ),
                                            vec![],
                                        ));
                                        return Type::Error;
                                    }
                                }
                            }
                            (Type::Adt(lhs_name, lhs_args), Type::Adt(rhs_name, rhs_args))
                                if lhs_name == "List"
                                    && rhs_name == "List"
                                    && lhs_args.len() == 1
                                    && rhs_args.len() == 1 =>
                            {
                                if let Err(te) = unify(&lhs_args[0], &rhs_args[0], subst) {
                                    errors.push(te.into());
                                    return Type::Error;
                                }
                                return Type::Adt(
                                    "List".to_string(),
                                    vec![subst.apply(&lhs_args[0])],
                                );
                            }
                            (Type::Var(_), _)
                            | (_, Type::Var(_))
                            | (Type::Error, _)
                            | (_, Type::Error) => {
                                return result_ty;
                            }
                            (lhs, rhs) => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "concat expects matching List inputs, got {lhs} and {rhs}"
                                        ),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "split" => {
                        if arg_tys.len() != 3 {
                            return Type::Error;
                        }
                        let tensor_ty = type_for_readonly_check(&arg_tys[0], subst);
                        let axis_ty = subst.apply(&arg_tys[1]);
                        let sizes_ty = subst.apply(&arg_tys[2]);
                        if !matches!(axis_ty, Type::Prim(prec) if prec.is_integer())
                            && !matches!(axis_ty, Type::Var(_) | Type::Error)
                        {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!("split expects integer axis, got {axis_ty}"),
                                ),
                                vec![],
                            ));
                            return Type::Error;
                        }
                        match (tensor_ty, sizes_ty) {
                            (Type::Tensor(dims, precision), Type::Adt(name, args))
                                if name == "List" && args.len() == 1 =>
                            {
                                if !matches!(&args[0], Type::Prim(prec) if prec.is_integer())
                                    && !matches!(&args[0], Type::Var(_) | Type::Error)
                                {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            "split expects List[int] sizes".to_string(),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                                // Negative axes index from the end.
                                // Issue #216: cast-aware so a
                                // `cast(N, int32)`-wrapped split axis still
                                // surfaces the bounds diagnostic at infer.
                                let raw_axis = kids.get(2).and_then(extract_int_for_dim);
                                let axis = match raw_axis {
                                    Some(raw) => match normalize_static_axis(dims.len(), raw) {
                                        Some(axis) => axis,
                                        None => {
                                            errors.push(CheckError::new(
                                                CheckErrorKind::TypeMismatch,
                                                with_macro_provenance(
                                                    &deep::Expr::List(list.clone(), zero_span()),
                                                    format!(
                                                        "split axis {raw} out of bounds for rank {}",
                                                        dims.len()
                                                    ),
                                                ),
                                                vec![],
                                            ));
                                            return Type::Error;
                                        }
                                    },
                                    None => 0,
                                };
                                let mut piece_dims = dims.clone();
                                piece_dims[axis] = Dim::Wildcard;
                                return Type::Adt(
                                    "List".to_string(),
                                    vec![Type::Tensor(piece_dims, precision)],
                                );
                            }
                            (Type::Var(_), _)
                            | (_, Type::Var(_))
                            | (Type::Error, _)
                            | (_, Type::Error) => {
                                return result_ty;
                            }
                            (tensor_ty, sizes_ty) => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "split expects tensor input and List[int] sizes, got {tensor_ty} and {sizes_ty}"
                                        ),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "take" | "drop" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        let op_name = func_name.as_deref().unwrap_or("collection helper");
                        let list_arg = subst.apply(&arg_tys[0]);
                        let count_arg = subst.apply(&arg_tys[1]);
                        if !matches!(count_arg, Type::Prim(prec) if prec.is_integer())
                            && !matches!(count_arg, Type::Var(_) | Type::Error)
                        {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!("{op_name} expects integer count, got {count_arg}"),
                                ),
                                vec![],
                            ));
                            return Type::Error;
                        }
                        match list_arg {
                            Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                                return Type::Adt("List".to_string(), vec![args[0].clone()]);
                            }
                            Type::Var(_) | Type::Error => return result_ty,
                            other => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!("{op_name} expects List input, got {other}"),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "chunk" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        let list_arg = subst.apply(&arg_tys[0]);
                        let count_arg = subst.apply(&arg_tys[1]);
                        if !matches!(count_arg, Type::Prim(prec) if prec.is_integer())
                            && !matches!(count_arg, Type::Var(_) | Type::Error)
                        {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!("chunk expects integer size, got {count_arg}"),
                                ),
                                vec![],
                            ));
                            return Type::Error;
                        }
                        match list_arg {
                            Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                                return Type::Adt(
                                    "List".to_string(),
                                    vec![Type::Adt("List".to_string(), vec![args[0].clone()])],
                                );
                            }
                            Type::Var(_) | Type::Error => return result_ty,
                            other => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!("chunk expects List input, got {other}"),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "range" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        for arg_ty in &arg_tys {
                            match subst.apply(arg_ty) {
                                Type::Prim(prec) if prec.is_integer() => {}
                                Type::Var(_) | Type::Error => {}
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("range expects integer arguments, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                        return Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int64)]);
                    }
                    "map" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        let elem_ty = vg.fresh_type();
                        let out_ty = vg.fresh_type();
                        if let Err(te) = unify(
                            &subst.apply(&arg_tys[0]),
                            &Type::Fn(vec![elem_ty.clone()], Box::new(out_ty.clone())),
                            subst,
                        ) {
                            errors.push(te.into());
                            return Type::Error;
                        }
                        if let Err(te) = unify(
                            &subst.apply(&arg_tys[1]),
                            &Type::Adt("List".to_string(), vec![elem_ty]),
                            subst,
                        ) {
                            errors.push(te.into());
                            return Type::Error;
                        }
                        return Type::Adt("List".to_string(), vec![subst.apply(&out_ty)]);
                    }
                    "filter" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        let elem_ty = vg.fresh_type();
                        let list_expr = deep::Expr::List(list.clone(), zero_span());
                        if let Err(te) = unify(
                            &subst.apply(&arg_tys[0]),
                            &Type::Fn(vec![elem_ty.clone()], Box::new(Type::Prim(Prim::Bool))),
                            subst,
                        ) {
                            errors.push(collection_helper_type_error(
                                &list_expr,
                                "filter",
                                "expects a callback that returns bool",
                                te,
                            ));
                            return Type::Error;
                        }
                        if let Err(te) = unify(
                            &subst.apply(&arg_tys[1]),
                            &Type::Adt("List".to_string(), vec![elem_ty.clone()]),
                            subst,
                        ) {
                            errors.push(te.into());
                            return Type::Error;
                        }
                        return Type::Adt("List".to_string(), vec![subst.apply(&elem_ty)]);
                    }
                    "fold" => {
                        if arg_tys.len() != 3 {
                            return Type::Error;
                        }
                        let acc_ty = vg.fresh_type();
                        let elem_ty = vg.fresh_type();
                        let list_expr = deep::Expr::List(list.clone(), zero_span());
                        if let Err(te) = unify(
                            &subst.apply(&arg_tys[0]),
                            &Type::Fn(
                                vec![acc_ty.clone(), elem_ty.clone()],
                                Box::new(acc_ty.clone()),
                            ),
                            subst,
                        ) {
                            errors.push(collection_helper_type_error(
                                &list_expr,
                                "fold",
                                "expects a callback whose accumulator/result type matches the initial accumulator",
                                te,
                            ));
                            return Type::Error;
                        }
                        if let Err(te) = unify(&subst.apply(&arg_tys[1]), &acc_ty.clone(), subst) {
                            errors.push(collection_helper_type_error(
                                &list_expr,
                                "fold",
                                "expects a callback whose accumulator/result type matches the initial accumulator",
                                te,
                            ));
                            return Type::Error;
                        }
                        if let Err(te) = unify(
                            &subst.apply(&arg_tys[2]),
                            &Type::Adt("List".to_string(), vec![elem_ty]),
                            subst,
                        ) {
                            errors.push(te.into());
                            return Type::Error;
                        }
                        return subst.apply(&acc_ty);
                    }
                    "scan" => {
                        if arg_tys.len() != 3 {
                            return Type::Error;
                        }
                        let acc_ty = vg.fresh_type();
                        let elem_ty = vg.fresh_type();
                        let list_expr = deep::Expr::List(list.clone(), zero_span());
                        if let Err(te) = unify(
                            &subst.apply(&arg_tys[0]),
                            &Type::Fn(
                                vec![acc_ty.clone(), elem_ty.clone()],
                                Box::new(acc_ty.clone()),
                            ),
                            subst,
                        ) {
                            errors.push(collection_helper_type_error(
                                &list_expr,
                                "scan",
                                "expects a callback whose accumulator/result type matches the initial accumulator",
                                te,
                            ));
                            return Type::Error;
                        }
                        if let Err(te) = unify(&subst.apply(&arg_tys[1]), &acc_ty.clone(), subst) {
                            errors.push(collection_helper_type_error(
                                &list_expr,
                                "scan",
                                "expects a callback whose accumulator/result type matches the initial accumulator",
                                te,
                            ));
                            return Type::Error;
                        }
                        if let Err(te) = unify(
                            &subst.apply(&arg_tys[2]),
                            &Type::Adt("List".to_string(), vec![elem_ty]),
                            subst,
                        ) {
                            errors.push(te.into());
                            return Type::Error;
                        }
                        return Type::Adt("List".to_string(), vec![subst.apply(&acc_ty)]);
                    }
                    "tensor_scan" => {
                        // `tensor_scan(initial: T, fn: (T, int64) -> T, n: int64) -> tensor[n, T]`.
                        //
                        // Issue #257: host-runtime scan that produces a tensor
                        // directly, sidestepping the right-recursive list build
                        // that overflows the worker stack at ~10k elements.
                        // Element type `T` must resolve to a concrete scalar
                        // Prim before tensor lowering; the runtime arm enforces
                        // that at execution time. At type-check time we accept
                        // any Type::Prim and let unification do the rest.
                        if arg_tys.len() != 3 {
                            return Type::Error;
                        }
                        let elem_ty = vg.fresh_type();
                        let int64 = Type::Prim(Prim::Int64);
                        let list_expr = deep::Expr::List(list.clone(), zero_span());
                        // arg 0: initial accumulator of type T.
                        if let Err(te) = unify(&subst.apply(&arg_tys[0]), &elem_ty.clone(), subst) {
                            errors.push(collection_helper_type_error(
                                &list_expr,
                                "tensor_scan",
                                "expects an initial value whose type matches the callback element type",
                                te,
                            ));
                            return Type::Error;
                        }
                        // arg 1: callback `(T, int64) -> T`.
                        if let Err(te) = unify(
                            &subst.apply(&arg_tys[1]),
                            &Type::Fn(
                                vec![elem_ty.clone(), int64.clone()],
                                Box::new(elem_ty.clone()),
                            ),
                            subst,
                        ) {
                            errors.push(collection_helper_type_error(
                                &list_expr,
                                "tensor_scan",
                                "expects a callback (T, int64) -> T",
                                te,
                            ));
                            return Type::Error;
                        }
                        // arg 2: length `n: int64`.
                        if let Err(te) = unify(&subst.apply(&arg_tys[2]), &int64, subst) {
                            errors.push(collection_helper_type_error(
                                &list_expr,
                                "tensor_scan",
                                "expects a length `n: int64`",
                                te,
                            ));
                            return Type::Error;
                        }
                        // Element type must be a concrete scalar Prim once
                        // unified. If it's still a Var the call site is
                        // under-constrained; if it's a Tensor/Adt/Fn the call
                        // is invalid. We only allow primitive scalars so the
                        // host-runtime arm can determine precision.
                        let resolved_elem = subst.apply(&elem_ty);
                        let precision = match &resolved_elem {
                            Type::Prim(p) => TensorPrec::Concrete(*p),
                            Type::Var(tv) => {
                                // Defer: leave the precision as the same type
                                // variable as the element. `Subst::apply` will
                                // resolve it once outer inference pins T.
                                // Using F32 as a placeholder (the previous
                                // behavior) silently lies about the dtype
                                // when T is later pinned to int64 or bool.
                                TensorPrec::Var(*tv)
                            }
                            other => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &list_expr,
                                        format!(
                                            "tensor_scan element type must be a scalar primitive, got {other}"
                                        ),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        };
                        return Type::Tensor(vec![Dim::Wildcard], precision);
                    }
                    "partition" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        let elem_ty = vg.fresh_type();
                        let list_expr = deep::Expr::List(list.clone(), zero_span());
                        if let Err(te) = unify(
                            &subst.apply(&arg_tys[0]),
                            &Type::Fn(vec![elem_ty.clone()], Box::new(Type::Prim(Prim::Bool))),
                            subst,
                        ) {
                            errors.push(collection_helper_type_error(
                                &list_expr,
                                "partition",
                                "expects a callback that returns bool",
                                te,
                            ));
                            return Type::Error;
                        }
                        if let Err(te) = unify(
                            &subst.apply(&arg_tys[1]),
                            &Type::Adt("List".to_string(), vec![elem_ty.clone()]),
                            subst,
                        ) {
                            errors.push(te.into());
                            return Type::Error;
                        }
                        let out_list = Type::Adt("List".to_string(), vec![subst.apply(&elem_ty)]);
                        return Type::Tuple(vec![out_list.clone(), out_list]);
                    }
                    "flat_map" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        let elem_ty = vg.fresh_type();
                        let out_elem_ty = vg.fresh_type();
                        if let Err(te) = unify(
                            &subst.apply(&arg_tys[0]),
                            &Type::Fn(
                                vec![elem_ty.clone()],
                                Box::new(Type::Adt("List".to_string(), vec![out_elem_ty.clone()])),
                            ),
                            subst,
                        ) {
                            errors.push(te.into());
                            return Type::Error;
                        }
                        if let Err(te) = unify(
                            &subst.apply(&arg_tys[1]),
                            &Type::Adt("List".to_string(), vec![elem_ty]),
                            subst,
                        ) {
                            errors.push(te.into());
                            return Type::Error;
                        }
                        return Type::Adt("List".to_string(), vec![subst.apply(&out_elem_ty)]);
                    }
                    "flatten" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Adt(outer_name, outer_args)
                                    if outer_name == "List" && outer_args.len() == 1 =>
                                {
                                    match &outer_args[0] {
                                        Type::Adt(inner_name, inner_args)
                                            if inner_name == "List" && inner_args.len() == 1 =>
                                        {
                                            return Type::Adt(
                                                "List".to_string(),
                                                vec![inner_args[0].clone()],
                                            );
                                        }
                                        Type::Var(_) | Type::Error => return result_ty,
                                        other => {
                                            errors.push(CheckError::new(
                                                CheckErrorKind::TypeMismatch,
                                                with_macro_provenance(
                                                    &deep::Expr::List(
                                                        list.clone(),
                                                        zero_span(),
                                                    ),
                                                    format!(
                                                        "flatten expects List[List[T]] input, got List[{other}]"
                                                    ),
                                                ),
                                                vec![],
                                            ));
                                            return Type::Error;
                                        }
                                    }
                                }
                                Type::Var(_) | Type::Error => return result_ty,
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!(
                                                "flatten expects List[List[T]] input, got {other}"
                                            ),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "zip" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        let lhs = subst.apply(&arg_tys[0]);
                        let rhs = subst.apply(&arg_tys[1]);
                        match (lhs, rhs) {
                            (Type::Adt(lhs_name, lhs_args), Type::Adt(rhs_name, rhs_args))
                                if lhs_name == "List"
                                    && rhs_name == "List"
                                    && lhs_args.len() == 1
                                    && rhs_args.len() == 1 =>
                            {
                                return Type::Adt(
                                    "List".to_string(),
                                    vec![Type::Tuple(vec![
                                        lhs_args[0].clone(),
                                        rhs_args[0].clone(),
                                    ])],
                                );
                            }
                            (Type::Var(_), _)
                            | (_, Type::Var(_))
                            | (Type::Error, _)
                            | (_, Type::Error) => {
                                return result_ty;
                            }
                            (lhs, rhs) => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!("zip expects List inputs, got {lhs} and {rhs}"),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "enumerate" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                                    return Type::Adt(
                                        "List".to_string(),
                                        vec![Type::Tuple(vec![
                                            Type::Prim(Prim::Int64),
                                            args[0].clone(),
                                        ])],
                                    );
                                }
                                Type::Var(_) | Type::Error => return result_ty,
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("enumerate expects List input, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "dict_of" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                                    match &args[0] {
                                        Type::Tuple(items) if items.len() == 2 => {
                                            match &items[0] {
                                                Type::Prim(Prim::Int64)
                                                | Type::Prim(Prim::String) => {}
                                                Type::Var(_) | Type::Error => return result_ty,
                                                other => {
                                                    errors.push(CheckError::new(
                                                        CheckErrorKind::TypeMismatch,
                                                        with_macro_provenance(
                                                            &deep::Expr::List(list.clone(), zero_span()),
                                                            format!(
                                                                "dict_of keys must be int64 or string, got {other}"
                                                            ),
                                                        ),
                                                        vec![],
                                                    ));
                                                    return Type::Error;
                                                }
                                            }
                                            return Type::Adt(
                                                "Dict".to_string(),
                                                vec![items[0].clone(), items[1].clone()],
                                            );
                                        }
                                        Type::Var(_) | Type::Error => return result_ty,
                                        other => {
                                            errors.push(CheckError::new(
                                                CheckErrorKind::TypeMismatch,
                                                with_macro_provenance(
                                                    &deep::Expr::List(list.clone(), zero_span()),
                                                    format!(
                                                        "dict_of expects List[(K, V)] input, got List[{other}]"
                                                    ),
                                                ),
                                                vec![],
                                            ));
                                            return Type::Error;
                                        }
                                    }
                                }
                                Type::Var(_) | Type::Error => return result_ty,
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("dict_of expects List input, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "dict_get" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        match (subst.apply(&arg_tys[0]), subst.apply(&arg_tys[1])) {
                            (Type::Adt(name, args), key_ty)
                                if name == "Dict" && args.len() == 2 =>
                            {
                                if let Err(te) = unify(&args[0], &key_ty, subst) {
                                    errors.push(te.into());
                                    return Type::Error;
                                }
                                return Type::Adt(
                                    "Option".to_string(),
                                    vec![subst.apply(&args[1])],
                                );
                            }
                            (Type::Var(_), _)
                            | (_, Type::Var(_))
                            | (Type::Error, _)
                            | (_, Type::Error) => {
                                return result_ty;
                            }
                            (dict_ty, key_ty) => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!("dict_get expects Dict[K, V] and K, got {dict_ty} and {key_ty}"),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "dict_contains" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        match (subst.apply(&arg_tys[0]), subst.apply(&arg_tys[1])) {
                            (Type::Adt(name, args), key_ty)
                                if name == "Dict" && args.len() == 2 =>
                            {
                                if let Err(te) = unify(&args[0], &key_ty, subst) {
                                    errors.push(te.into());
                                    return Type::Error;
                                }
                                return Type::Prim(Prim::Bool);
                            }
                            (Type::Var(_), _)
                            | (_, Type::Var(_))
                            | (Type::Error, _)
                            | (_, Type::Error) => {
                                return result_ty;
                            }
                            (dict_ty, key_ty) => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "dict_contains expects Dict[K, V] and K, got {dict_ty} and {key_ty}"
                                        ),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "dict_remove" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        match (subst.apply(&arg_tys[0]), subst.apply(&arg_tys[1])) {
                            (Type::Adt(name, args), key_ty)
                                if name == "Dict" && args.len() == 2 =>
                            {
                                if let Err(te) = unify(&args[0], &key_ty, subst) {
                                    errors.push(te.into());
                                    return Type::Error;
                                }
                                return Type::Adt(
                                    "Dict".to_string(),
                                    vec![subst.apply(&args[0]), subst.apply(&args[1])],
                                );
                            }
                            (Type::Var(_), _)
                            | (_, Type::Var(_))
                            | (Type::Error, _)
                            | (_, Type::Error) => {
                                return result_ty;
                            }
                            (dict_ty, key_ty) => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "dict_remove expects Dict[K, V] and K, got {dict_ty} and {key_ty}"
                                        ),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "dict_insert" => {
                        if arg_tys.len() != 3 {
                            return Type::Error;
                        }
                        match (
                            subst.apply(&arg_tys[0]),
                            subst.apply(&arg_tys[1]),
                            subst.apply(&arg_tys[2]),
                        ) {
                            (Type::Adt(name, args), key_ty, value_ty)
                                if name == "Dict" && args.len() == 2 =>
                            {
                                if let Err(te) = unify(&args[0], &key_ty, subst) {
                                    errors.push(te.into());
                                    return Type::Error;
                                }
                                if let Err(te) = unify(&args[1], &value_ty, subst) {
                                    errors.push(te.into());
                                    return Type::Error;
                                }
                                return Type::Adt(
                                    "Dict".to_string(),
                                    vec![subst.apply(&args[0]), subst.apply(&args[1])],
                                );
                            }
                            (Type::Var(_), _, _)
                            | (_, Type::Var(_), _)
                            | (_, _, Type::Var(_))
                            | (Type::Error, _, _)
                            | (_, Type::Error, _)
                            | (_, _, Type::Error) => {
                                return result_ty;
                            }
                            (dict_ty, key_ty, value_ty) => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "dict_insert expects Dict[K, V], K, and V, got {dict_ty}, {key_ty}, and {value_ty}"
                                        ),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "dict_merge" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        match (subst.apply(&arg_tys[0]), subst.apply(&arg_tys[1])) {
                            (Type::Adt(lhs_name, lhs_args), Type::Adt(rhs_name, rhs_args))
                                if lhs_name == "Dict"
                                    && rhs_name == "Dict"
                                    && lhs_args.len() == 2
                                    && rhs_args.len() == 2 =>
                            {
                                if let Err(te) = unify(&lhs_args[0], &rhs_args[0], subst) {
                                    errors.push(te.into());
                                    return Type::Error;
                                }
                                if let Err(te) = unify(&lhs_args[1], &rhs_args[1], subst) {
                                    errors.push(te.into());
                                    return Type::Error;
                                }
                                return Type::Adt(
                                    "Dict".to_string(),
                                    vec![subst.apply(&lhs_args[0]), subst.apply(&lhs_args[1])],
                                );
                            }
                            (Type::Var(_), _)
                            | (_, Type::Var(_))
                            | (Type::Error, _)
                            | (_, Type::Error) => {
                                return result_ty;
                            }
                            (lhs_ty, rhs_ty) => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "dict_merge expects matching Dict inputs, got {lhs_ty} and {rhs_ty}"
                                        ),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "dict_keys" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Adt(name, args) if name == "Dict" && args.len() == 2 => {
                                    return Type::Adt("List".to_string(), vec![args[0].clone()]);
                                }
                                Type::Var(_) | Type::Error => return result_ty,
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("dict_keys expects Dict input, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "dict_values" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Adt(name, args) if name == "Dict" && args.len() == 2 => {
                                    return Type::Adt("List".to_string(), vec![args[1].clone()]);
                                }
                                Type::Var(_) | Type::Error => return result_ty,
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("dict_values expects Dict input, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "dict_entries" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Adt(name, args) if name == "Dict" && args.len() == 2 => {
                                    return Type::Adt(
                                        "List".to_string(),
                                        vec![Type::Tuple(vec![args[0].clone(), args[1].clone()])],
                                    );
                                }
                                Type::Var(_) | Type::Error => return result_ty,
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("dict_entries expects Dict input, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "to_tensor" => {
                        if let Some(first_arg) = arg_tys.first() {
                            // Bucket 4b: support arbitrarily-nested numeric/bool
                            // lists. Each enclosing `List` adds one outer
                            // dimension, and the innermost element type must
                            // be a numeric or bool primitive.
                            //
                            // Issue Chelis-Lang/chelis#218 (R2 HIGH-A from
                            // PR #211): when the argument is a statically-
                            // resolvable Cons-chain literal, emit concrete
                            // `Dim::Lit(n)` per axis instead of wildcards.
                            // The wildcard fallback only fires when the
                            // argument is variable-fed (e.g.
                            // `to_tensor(items)`), where the shape is
                            // genuinely unknown at type-check time. Emitting
                            // concrete dims at this single source point
                            // means every downstream consumer (reductions,
                            // elementwise activations, anything that reads
                            // the to_tensor app's `type:` metadata) sees a
                            // sound shape instead of `Dim::Wildcard`.
                            let resolved = subst.apply(first_arg);
                            if matches!(resolved, Type::Var(_) | Type::Error) {
                                return result_ty;
                            }
                            match peel_to_tensor_argument(&resolved) {
                                ToTensorPeel::Ok { rank, precision } => {
                                    if rank == 0 {
                                        // Defensive: a bare scalar should never
                                        // hit this branch (the typer requires
                                        // a `List` head), but guard anyway.
                                        errors.push(CheckError::new(
                                            CheckErrorKind::TypeMismatch,
                                            with_macro_provenance(
                                                &deep::Expr::List(list.clone(), zero_span()),
                                                format!(
                                                    "to_tensor expects List input, got {resolved}"
                                                ),
                                            ),
                                            vec![],
                                        ));
                                        return Type::Error;
                                    }
                                    // R2 HIGH-A: try the static-shape walker
                                    // on the actual argument expression
                                    // first. `kids[0]` is the `(var
                                    // to_tensor)` callee; `kids[1]` is the
                                    // argument expression. If the walker
                                    // can't resolve a uniform shape
                                    // (variable-fed argument, ragged
                                    // literal, or unrecognized leaf), fall
                                    // back to the legacy wildcard rank.
                                    let dims = kids
                                        .get(1)
                                        .and_then(|arg| static_to_tensor_shape(arg, rank))
                                        .unwrap_or_else(|| vec![Dim::Wildcard; rank]);
                                    return Type::Tensor(dims, TensorPrec::Concrete(precision));
                                }
                                ToTensorPeel::Pending => return result_ty,
                                ToTensorPeel::BadInner(inner) => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!(
                                                "to_tensor expects numeric or bool elements at the innermost level, got {inner}"
                                            ),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                                ToTensorPeel::NotList => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("to_tensor expects List input, got {resolved}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "to_list" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match type_for_readonly_check(first_arg, subst) {
                                Type::Tensor(dims, precision) => {
                                    if dims.len() != 1 {
                                        errors.push(CheckError::new(
                                            CheckErrorKind::TypeMismatch,
                                            with_macro_provenance(
                                                &deep::Expr::List(list.clone(), zero_span()),
                                                format!(
                                                    "to_list expects a rank-1 tensor, got rank {} tensor",
                                                    dims.len()
                                                ),
                                            ),
                                            vec![],
                                        ));
                                        return Type::Error;
                                    }
                                    // to_list requires a fully resolved
                                    // precision: a polymorphic precision must
                                    // be resolved before to_list can name a
                                    // concrete element type. Defer if the
                                    // precision is still a var.
                                    let precision = match precision {
                                        TensorPrec::Concrete(p) => p,
                                        TensorPrec::Var(_) => return result_ty,
                                    };
                                    if !precision.is_numeric() && !matches!(precision, Prim::Bool) {
                                        errors.push(CheckError::new(
                                            CheckErrorKind::TypeMismatch,
                                            with_macro_provenance(
                                                &deep::Expr::List(list.clone(), zero_span()),
                                                format!(
                                                    "to_list expects numeric or bool tensor input, got {precision:?}"
                                                ),
                                            ),
                                            vec![],
                                        ));
                                        return Type::Error;
                                    }
                                    return Type::Adt(
                                        "List".to_string(),
                                        vec![Type::Prim(precision)],
                                    );
                                }
                                Type::Var(_) | Type::Error => return result_ty,
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("to_list expects Tensor input, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "pad_sequences" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        let seqs_ty = subst.apply(&arg_tys[0]);
                        let pad_ty = subst.apply(&arg_tys[1]);
                        match seqs_ty {
                            Type::Adt(outer_name, outer_args)
                                if outer_name == "List" && outer_args.len() == 1 =>
                            {
                                match &outer_args[0] {
                                    Type::Adt(inner_name, inner_args)
                                        if inner_name == "List" && inner_args.len() == 1 =>
                                    {
                                        if let Err(te) = unify(&inner_args[0], &pad_ty, subst) {
                                            errors.push(te.into());
                                            return Type::Error;
                                        }
                                        match subst.apply(&inner_args[0]) {
                                            Type::Prim(precision) if precision.is_numeric() => {
                                                return Type::Tensor(
                                                    vec![Dim::Wildcard, Dim::Wildcard],
                                                    TensorPrec::Concrete(precision),
                                                );
                                            }
                                            Type::Var(_) | Type::Error => return result_ty,
                                            other => {
                                                errors.push(CheckError::new(
                                                    CheckErrorKind::TypeMismatch,
                                                    with_macro_provenance(
                                                        &deep::Expr::List(
                                                            list.clone(),
                                                            zero_span(),
                                                        ),
                                                        format!(
                                                            "pad_sequences expects numeric nested lists, got {other}"
                                                        ),
                                                    ),
                                                    vec![],
                                                ));
                                                return Type::Error;
                                            }
                                        }
                                    }
                                    other => {
                                        errors.push(CheckError::new(
                                            CheckErrorKind::TypeMismatch,
                                            with_macro_provenance(
                                                &deep::Expr::List(list.clone(), zero_span()),
                                                format!(
                                                    "pad_sequences expects List[List[T]], got List[{other}]"
                                                ),
                                            ),
                                            vec![],
                                        ));
                                        return Type::Error;
                                    }
                                }
                            }
                            Type::Var(_) | Type::Error => return result_ty,
                            other => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "pad_sequences expects List[List[T]] input, got {other}"
                                        ),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "pad_sequences_to" => {
                        if arg_tys.len() != 3 {
                            return Type::Error;
                        }
                        let seqs_ty = subst.apply(&arg_tys[0]);
                        let width_ty = subst.apply(&arg_tys[1]);
                        let pad_ty = subst.apply(&arg_tys[2]);
                        if let Err(te) = unify(&width_ty, &Type::Prim(Prim::Int64), subst) {
                            errors.push(te.into());
                            return Type::Error;
                        }
                        // The padded (axis-1) dimension equals the `width`
                        // argument. When `width` is a literal — including
                        // `cast(N, int64)`, the form every caller uses —
                        // propagate `Dim::Lit(N)` so the padded width is a
                        // concrete dim that participates in shape checking.
                        // A non-literal or non-positive width stays
                        // `Dim::Wildcard` (the runtime validates the value).
                        // `extract_int_for_dim` (not `extract_int_literal`)
                        // is the cast-aware extractor used for dim contexts.
                        let width_dim = children(list)
                            .get(2)
                            .and_then(extract_int_for_dim)
                            .filter(|width| *width > 0)
                            .map_or(Dim::Wildcard, Dim::Lit);
                        match seqs_ty {
                            Type::Adt(outer_name, outer_args)
                                if outer_name == "List" && outer_args.len() == 1 =>
                            {
                                match &outer_args[0] {
                                    Type::Adt(inner_name, inner_args)
                                        if inner_name == "List" && inner_args.len() == 1 =>
                                    {
                                        if let Err(te) = unify(&inner_args[0], &pad_ty, subst) {
                                            errors.push(te.into());
                                            return Type::Error;
                                        }
                                        match subst.apply(&inner_args[0]) {
                                            Type::Prim(precision) if precision.is_numeric() => {
                                                return Type::Tensor(
                                                    vec![Dim::Wildcard, width_dim],
                                                    TensorPrec::Concrete(precision),
                                                );
                                            }
                                            Type::Var(_) | Type::Error => return result_ty,
                                            other => {
                                                errors.push(CheckError::new(
                                                    CheckErrorKind::TypeMismatch,
                                                    with_macro_provenance(
                                                        &deep::Expr::List(
                                                            list.clone(),
                                                            zero_span(),
                                                        ),
                                                        format!(
                                                            "pad_sequences_to expects numeric nested lists, got {other}"
                                                        ),
                                                    ),
                                                    vec![],
                                                ));
                                                return Type::Error;
                                            }
                                        }
                                    }
                                    other => {
                                        errors.push(CheckError::new(
                                            CheckErrorKind::TypeMismatch,
                                            with_macro_provenance(
                                                &deep::Expr::List(list.clone(), zero_span()),
                                                format!(
                                                    "pad_sequences_to expects List[List[T]], got List[{other}]"
                                                ),
                                            ),
                                            vec![],
                                        ));
                                        return Type::Error;
                                    }
                                }
                            }
                            Type::Var(_) | Type::Error => return result_ty,
                            other => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "pad_sequences_to expects List[List[T]] input, got {other}"
                                        ),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "read_file" => return Type::Prim(Prim::String),
                    "write_file" => return Type::Unit,
                    "read_lines" => {
                        return Type::Adt("List".to_string(), vec![Type::Prim(Prim::String)]);
                    }
                    "read_bytes" => {
                        return Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int64)]);
                    }
                    "file_exists" => return Type::Prim(Prim::Bool),
                    "list_dir" => {
                        return Type::Adt("List".to_string(), vec![Type::Prim(Prim::String)]);
                    }
                    "mmap_file" => return Type::Adt("MappedFile".to_string(), Vec::new()),
                    "mmap_read" => {
                        return Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int64)]);
                    }
                    "mmap_len" => return Type::Prim(Prim::Int64),
                    // Hull Phase 0a: `process_run(cmd, args)` returns
                    // `(exit_code, stdout, stderr)`. Eval/test-only; the build
                    // backends reject it (see `reject_eval_only_builtins_host`).
                    "process_run" => {
                        return Type::Tuple(vec![
                            Type::Prim(Prim::Int64),
                            Type::Prim(Prim::String),
                            Type::Prim(Prim::String),
                        ]);
                    }
                    _ => {}
                }
            }

            result_ty
        }
        Err(te) => {
            let mut e: CheckError = te.into();
            if let Some(id) = list_span_id(list) {
                e.span_offset = parse_span_offset(id);
                e.span_id = Some(id.to_string());
            } else {
                let off = span_of_list(list).offset;
                if off > 0 {
                    e.span_offset = Some(off);
                }
            }
            errors.push(e);
            Type::Error
        }
    }
}

fn auto_borrow_call_arg_types(func_ty: &Type, arg_tys: Vec<Type>, subst: &Subst) -> Vec<Type> {
    let Type::Fn(params, _) = subst.apply(func_ty) else {
        return arg_tys;
    };
    arg_tys
        .into_iter()
        .enumerate()
        .map(
            |(index, actual)| match params.get(index).map(|param| subst.apply(param)) {
                Some(Type::Ref(_)) if !matches!(subst.apply(&actual), Type::Ref(_)) => {
                    Type::Ref(Box::new(actual))
                }
                _ => actual,
            },
        )
        .collect()
}

/// Implicit-copy fan-out v3 Shape A relaxation: when a def's body's
/// tail-position expression is a bare `(var x)` reference (possibly
/// wrapped in `let`, `if`, or `match` structures whose sibling branches
/// all return the same name) and the body's inferred return type is
/// `Ref(R)` while the declared return is owned `R`, return a relaxed
/// declared type `Fn(params, Ref(R))` so the def-body unify can succeed.
/// Returns `None` for any other body shape; the caller surfaces the
/// existing TypeMismatch in that case.
///
/// The PR #91 (W4-A) version of this helper accepted only a bare
/// `(fn (params...) (var x))` body.  0.7.9 broadens the gate to walk
/// `let`/`if`/`match` tail-position structures via
/// `descend_to_tail_var`, closing `Linearity-ShapeABroadReturn-F1`.
fn shape_a_relaxed_return(body_expr: &deep::Expr, body_ty: &Type, decl_ty: &Type) -> Option<Type> {
    // body is the def's body, which the desugarer wraps as
    // `(fn (params ...) body_inner)` whenever the def has params.  Walk
    // the inner expression's tail position to confirm it resolves to a
    // bare `(var name)` reference across every reachable sibling.
    let body_list = match body_expr {
        deep::Expr::List(list, _) => list,
        _ => return None,
    };
    if get_tag(body_list) != Some("fn") {
        return None;
    }
    let inner = children(body_list).get(1)?;
    descend_to_tail_var(inner)?;

    // The body's inferred type and the declared type both must be
    // `Fn(params, ret)` with matching params and a return-position
    // mismatch of exactly `Ref(R)` (body) vs `R` (decl).
    let (Type::Fn(body_params, body_ret), Type::Fn(decl_params, decl_ret)) = (body_ty, decl_ty)
    else {
        return None;
    };
    if body_params.len() != decl_params.len() {
        return None;
    }
    let Type::Ref(body_inner_ret) = body_ret.as_ref() else {
        return None;
    };
    // The body's unwrapped return must match the declared return
    // structurally before we allow the relaxation; otherwise the
    // relaxed unify would still fail and only the loop would change.
    if !types_structurally_equal(body_inner_ret.as_ref(), decl_ret.as_ref()) {
        return None;
    }
    Some(Type::Fn(
        decl_params.clone(),
        Box::new(Type::Ref(Box::new(decl_ret.as_ref().clone()))),
    ))
}

/// Descend through `let`, `if`, and `match` to a tail-position
/// `(var name)` reference.  Returns `Some(name)` when every sibling
/// branch resolves to the same bare-var name, `None` otherwise.
///
/// This is the broader-Shape-A coverage closure for
/// `Linearity-ShapeABroadReturn-F1`.  The rules:
///
/// - `(var x)` returns `Some("x")` (the leaf case from PR #91).
/// - `(let bind body)` recurses into `body` (the second child).
/// - `(if cond then_e else_e)` recurses into both branches; both must
///   resolve to the same name.
/// - `(match scrutinee arm ...)` recurses into every arm body (the
///   third child of each `(arm pattern guard body)` triple); all arms
///   must resolve to the same name.
/// - Otherwise returns `None`.
///
/// The descent is type-agnostic; the surrounding logic in
/// `shape_a_relaxed_return` already verifies that the body's inferred
/// return type is `Ref(R)` and the declared return is `R` structurally.
///
/// The "same name across siblings" requirement is intentional: the
/// existing relaxation is justified by the caller's borrow lifetime
/// already covering the parameter being returned.  Heterogeneous
/// bare-var returns would extend the relaxation beyond v3 scope and
/// need a richer coercion story.
fn descend_to_tail_var(expr: &deep::Expr) -> Option<&str> {
    stack_guard!("descend_to_tail_var", expr, None);
    let list = match expr {
        deep::Expr::List(list, _) => list,
        _ => return None,
    };
    match get_tag(list) {
        Some("var") => var_name_list(list),
        Some("let") => {
            let body = children(list).get(1)?;
            descend_to_tail_var(body)
        }
        Some("if") => {
            let kids = children(list);
            let then_e = kids.get(1)?;
            let else_e = kids.get(2)?;
            let then_name = descend_to_tail_var(then_e)?;
            let else_name = descend_to_tail_var(else_e)?;
            if then_name == else_name {
                Some(then_name)
            } else {
                None
            }
        }
        Some("match") => {
            let kids = children(list);
            // Skip the scrutinee (first child); every remaining child is
            // expected to be an `(arm pattern guard body)` triple.
            let arms = kids.get(1..)?;
            if arms.is_empty() {
                return None;
            }
            let mut name: Option<&str> = None;
            for arm in arms {
                let arm_list = match arm {
                    deep::Expr::List(list, _) => list,
                    _ => return None,
                };
                if get_tag(arm_list) != Some("arm") {
                    return None;
                }
                let arm_body = children(arm_list).get(2)?;
                let arm_name = descend_to_tail_var(arm_body)?;
                match name {
                    None => name = Some(arm_name),
                    Some(prev) if prev == arm_name => {}
                    Some(_) => return None,
                }
            }
            name
        }
        _ => None,
    }
}

/// Two tensor dimensions are *identical* for structural-equality
/// purposes iff they denote the same dimension: same concrete
/// `Dim::Name`, same `Dim::Lit`, or the same `Dim::Var`.  `Dim::Wildcard`
/// matches anything on either side (it is the permissive "unknown"
/// sentinel, consistent with `unify_dim`).
///
/// Two *distinct* symbolic dim variables (`n` vs `m`) are NOT identical
/// even though both contribute rank 1.  This is the soundness fix for
/// `TypeCheck-FreeDimVarUnification-F1` (SR-LEAK-A): the Shape A
/// relaxed-retry must not accept a body whose return dim diverges from
/// the declared return dim.
///
/// Name <-> Lit (issue Chelis-Lang/chelis#219, Option A): mirrors
/// `unify_dim`'s permissive Name <-> Lit arm. The Shape A relaxed-
/// retry's structural check must agree with `unify_dim` so the
/// retry path doesn't silently reject a callee-shape pairing that
/// the call-site unification would accept.
fn dims_identical(d1: &Dim, d2: &Dim) -> bool {
    match (d1, d2) {
        (Dim::Wildcard, _) | (_, Dim::Wildcard) => true,
        (Dim::Name(n1), Dim::Name(n2)) => n1 == n2,
        (Dim::Lit(l1), Dim::Lit(l2)) => l1 == l2,
        (Dim::Var(v1), Dim::Var(v2)) => v1 == v2,
        // Issue #219 Option A: Name and Lit count as identical for
        // the Shape A relaxed-retry's structural check.
        (Dim::Name(_), Dim::Lit(_)) | (Dim::Lit(_), Dim::Name(_)) => true,
        _ => false,
    }
}

/// Structural type equality used by `shape_a_relaxed_return` to guard
/// the relaxed retry: the relaxation is only safe when the body and
/// declared return differ exactly by a top-level `Ref` wrapper, so the
/// dim structure underneath must match by *identity*, not just by rank.
///
/// Tensor dims are compared with `dims_identical`: same concrete name,
/// same literal, or the same dim variable.  Distinct dim variables do
/// not match (`TypeCheck-FreeDimVarUnification-F1`).
fn types_structurally_equal(a: &Type, b: &Type) -> bool {
    match (a, b) {
        (Type::Unit, Type::Unit) => true,
        (Type::Prim(p1), Type::Prim(p2)) => p1 == p2,
        (Type::Ref(i1), Type::Ref(i2)) => types_structurally_equal(i1, i2),
        (Type::Tensor(d1, p1), Type::Tensor(d2, p2)) => {
            p1 == p2
                && d1.len() == d2.len()
                && d1.iter().zip(d2.iter()).all(|(x, y)| dims_identical(x, y))
        }
        (Type::Tuple(es1), Type::Tuple(es2)) => {
            es1.len() == es2.len()
                && es1
                    .iter()
                    .zip(es2.iter())
                    .all(|(e1, e2)| types_structurally_equal(e1, e2))
        }
        (Type::Adt(n1, a1), Type::Adt(n2, a2)) => {
            n1 == n2
                && a1.len() == a2.len()
                && a1
                    .iter()
                    .zip(a2.iter())
                    .all(|(e1, e2)| types_structurally_equal(e1, e2))
        }
        (Type::Fn(p1, r1), Type::Fn(p2, r2)) => {
            p1.len() == p2.len()
                && p1
                    .iter()
                    .zip(p2.iter())
                    .all(|(e1, e2)| types_structurally_equal(e1, e2))
                && types_structurally_equal(r1, r2)
        }
        (Type::Var(_), Type::Var(_)) => true,
        (Type::Error, _) | (_, Type::Error) => true,
        _ => false,
    }
}

fn type_for_readonly_check(ty: &Type, subst: &Subst) -> Type {
    match subst.apply(ty) {
        Type::Ref(inner) => subst.apply(&inner),
        other => other,
    }
}

/// chelis#339 Part 2: infer a variadic named-axis reduction
/// `sum(x, seq, head)` (spec/04-type-system.md §4.5.3). The reduction HM
/// schemes are arity-2, so the 3+-arg form bypasses the generic arity
/// check (the `infer_permute_app` pattern); the existing
/// `check_reduction_signature` named loop validates every axis and
/// computes the symbolic output. The variadic form is defined for the
/// value reductions only — `argmax_reduce`/`argmin_reduce` produce
/// indices along ONE axis, which a second reduction cannot compose, so
/// they are rejected here with a targeted diagnostic.
#[allow(clippy::too_many_arguments)]
fn infer_reduction_app(
    list: &deep::List,
    fname: &str,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    if fname == "argmax_reduce" || fname == "argmin_reduce" {
        errors.push(CheckError::new(
            CheckErrorKind::ArityMismatch,
            format!(
                "{fname} is an index-returning reduction and has no variadic \
                 named-axis form: an index along one axis is not composable with a \
                 second reduction (spec/04-type-system.md \u{00a7}4.5.3). Reduce one \
                 axis at a time."
            ),
            vec![],
        ));
        return Type::Error;
    }

    let kids = children(list);
    let _func_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let arg_tys: Vec<Type> = kids[1..]
        .iter()
        .enumerate()
        .map(|(index, arg)| {
            // Axis slots carry dimension names, typed as axes (`int32`)
            // rather than inferred as values — the named-reduction exemption
            // from the generic path.
            if index >= 1 && symbolic_dim_ref_name(arg).is_some() {
                Type::Prim(Prim::Int32)
            } else {
                infer_expr(
                    arg,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                )
            }
        })
        .collect();
    if arg_tys.iter().any(|ty| matches!(ty, Type::Error)) {
        return Type::Error;
    }

    let result_ty = Type::Var(vg.fresh_tvar());
    check_reduction_signature(fname, &kids[1..], &arg_tys, &result_ty, subst, errors)
}

/// chelis#339: infer the 4-arg anchored named-axis expand form
/// `expand(x, new, size, anchor)` (spec/04-type-system.md §4.5.3). The
/// builtin scheme is arity-3, so this form bypasses the generic HM arity
/// check (the `infer_permute_app` pattern). The `new` and `anchor` slots
/// carry dimension *names*, not bound values — like a named reduction
/// axis they are typed as `int32` axes rather than inferred, and
/// `check_expand_signature` reads the actual names back from the arg
/// exprs.
#[allow(clippy::too_many_arguments)]
fn infer_expand_app(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.len() != 5 {
        errors.push(CheckError::new(
            CheckErrorKind::ArityMismatch,
            format!(
                "expand expects (tensor, axis, size) or the named-axis form \
                 (tensor, name, size, anchor), got {} arguments",
                kids.len() - 1
            ),
            vec![],
        ));
        return Type::Error;
    }

    let _func_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let arg_tys: Vec<Type> = kids[1..]
        .iter()
        .enumerate()
        .map(|(index, arg)| {
            // The name/size/anchor slots may carry dim names; a name bound in
            // the value environment is a runtime value instead (issue #259
            // scope discrimination, as in the generic-path exemption).
            if index >= 1
                && symbolic_dim_ref_name(arg).is_some_and(|name| env.lookup(name).is_none())
            {
                Type::Prim(Prim::Int32)
            } else {
                infer_expr(
                    arg,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                )
            }
        })
        .collect();
    // chelis#530: the size slot (index 2 / `kids[3]`) is exempt from the
    // error-propagation short-circuit so an inline tuple-get / `match`/`if`
    // size that infers to `Type::Error` still reaches the named-axis literal
    // check below (`check_named_expand_signature` reads the size from the raw
    // AST), rather than being silently accepted. Mirrors the generic-path
    // exemption in `infer_app`.
    if arg_tys
        .iter()
        .enumerate()
        .any(|(index, ty)| index != 2 && matches!(ty, Type::Error))
    {
        return Type::Error;
    }
    // The size slot must still be an int32 (a literal, a symbolic dim, or a
    // runtime int32 expression — §4.7.2); a non-int size is a type error the
    // arity-3 scheme would otherwise have caught.
    let size_ty = subst.apply(&arg_tys[2]);
    match size_ty {
        Type::Prim(Prim::Int32) | Type::Var(_) | Type::Error => {}
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!("expand expects an int32 size, got {other}"),
                ),
                vec![],
            ));
            return Type::Error;
        }
    }

    let axis_is_dim_name = kids.get(2).is_some_and(|arg| {
        symbolic_dim_ref_name(arg).is_some_and(|name| env.lookup(name).is_none())
    });
    // chelis#397/#469: classify the size slot by PROVENANCE, following
    // `let`/`cast`/arithmetic to a tensor shape source. The positive-rank
    // path would otherwise stamp a sourceless runtime scalar as a `Dim::Name`,
    // type-check clean, and then die at build/eval with the §4.7.2 Form-3
    // sourceless-size rejection (chelis#469: "no tensor in scope carries it").
    // Rejecting it at CHECK keeps check↔build↔eval in sync (a check-clean
    // program must build); a literal/static/shape-sourced size is materializable
    // and accepted, uniformly across the bare-`var`, `cast`-wrapped, `let`-bound,
    // and arithmetic spellings.
    let size_class = kids
        .get(3)
        .map(|arg| classify_expand_size(arg, env))
        .unwrap_or(SizeClass::Unknown);
    let result_ty = Type::Var(vg.fresh_tvar());
    check_expand_signature(
        &kids[1..],
        &arg_tys,
        &result_ty,
        axis_is_dim_name,
        size_class,
        env,
        subst,
        errors,
    )
}

#[allow(clippy::too_many_arguments)]
fn infer_permute_app(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.len() < 2 {
        errors.push(CheckError::new(
            CheckErrorKind::ArityMismatch,
            "permute expects a tensor followed by one or more axis indices".to_string(),
            vec![],
        ));
        return Type::Error;
    }

    let _func_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let input_ty = infer_expr(
        &kids[1],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let axis_tys: Vec<Type> = kids[2..]
        .iter()
        .map(|arg| {
            infer_expr(
                arg,
                env,
                vg,
                subst,
                adt_reg,
                errors,
                typed_nodes,
                total_nodes,
            )
        })
        .collect();

    if matches!(input_ty, Type::Error) || axis_tys.iter().any(|ty| matches!(ty, Type::Error)) {
        return Type::Error;
    }

    for axis_ty in &axis_tys {
        let resolved = subst.apply(axis_ty);
        match resolved {
            Type::Prim(Prim::Int32) | Type::Var(_) | Type::Error => {}
            other => {
                errors.push(CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!("permute expects int32 axis indices, got {other}"),
                    ),
                    vec![],
                ));
                return Type::Error;
            }
        }
    }

    let input_ty = type_for_readonly_check(&input_ty, subst);
    let Type::Tensor(dims, prec) = input_ty else {
        if matches!(input_ty, Type::Var(_) | Type::Error) {
            return input_ty;
        }
        errors.push(CheckError::new(
            CheckErrorKind::TypeMismatch,
            format!("permute expects tensor input, got {input_ty}"),
            vec![],
        ));
        return Type::Error;
    };

    // Uses `extract_int_for_dim` so `cast(N, int32)`-wrapped literal
    // axes reach the OOB-axis check and the unique-axis check at infer
    // time instead of silently falling back to original-dim order (red
    // team round 3 sibling sweep within the spec section 2.4 movement
    // family).
    let Some(axes) = kids[2..]
        .iter()
        .map(extract_int_for_dim)
        .collect::<Option<Vec<_>>>()
    else {
        return Type::Tensor(dims, prec);
    };

    if axes.len() != dims.len() {
        errors.push(CheckError::new(
            CheckErrorKind::ArityMismatch,
            format!(
                "permute expects {} axis indices for rank {} tensor, got {}",
                dims.len(),
                dims.len(),
                axes.len()
            ),
            vec![],
        ));
        return Type::Error;
    }

    let mut seen = HashSet::new();
    let mut reordered = Vec::with_capacity(dims.len());
    for axis in axes {
        if axis < 0 || axis as usize >= dims.len() {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!(
                    "permute axis {axis} is out of bounds for rank {} tensor",
                    dims.len()
                ),
                vec![],
            ));
            return Type::Error;
        }
        let axis = axis as usize;
        if !seen.insert(axis) {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!("permute axis {axis} appears more than once"),
                vec![],
            ));
            return Type::Error;
        }
        reordered.push(dims[axis].clone());
    }

    Type::Tensor(reordered, prec)
}

#[allow(clippy::too_many_arguments)]
fn infer_reshape_app(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.len() < 2 || kids.len() > 3 {
        errors.push(CheckError::new(
            CheckErrorKind::ArityMismatch,
            "reshape expects a tensor and an optional shape list".to_string(),
            vec![],
        ));
        return Type::Error;
    }

    let _func_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let input_ty = infer_expr(
        &kids[1],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let input_var_name = symbolic_dim_ref_name(&kids[1]).map(|s| s.to_string());
    match type_for_readonly_check(&input_ty, subst) {
        Type::Prim(precision) => {
            if let Some(shape_expr) = kids.get(2) {
                let shape_ty = infer_expr(
                    shape_expr,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                );
                let expected_shape_ty =
                    Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int64)]);
                if let Err(te) = unify(&shape_ty, &expected_shape_ty, subst) {
                    errors.push(te.into());
                    return Type::Error;
                }
                let dims = reshape_output_dims(shape_expr, input_var_name.as_deref(), &[], subst);
                return Type::Tensor(dims, TensorPrec::Concrete(precision));
            }

            Type::Tensor(vec![Dim::Wildcard], TensorPrec::Concrete(precision))
        }
        Type::Tensor(input_dims, precision) => {
            if let Some(shape_expr) = kids.get(2) {
                let shape_ty = infer_expr(
                    shape_expr,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                );
                let expected_shape_ty =
                    Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int64)]);
                if let Err(te) = unify(&shape_ty, &expected_shape_ty, subst) {
                    errors.push(te.into());
                    return Type::Error;
                }
                let dims =
                    reshape_output_dims(shape_expr, input_var_name.as_deref(), &input_dims, subst);
                return Type::Tensor(dims, precision);
            }

            Type::Tensor(vec![Dim::Wildcard], precision)
        }
        Type::Var(_) | Type::Error => {
            if let Some(shape_expr) = kids.get(2) {
                let shape_ty = infer_expr(
                    shape_expr,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                );
                let expected_shape_ty =
                    Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int64)]);
                if let Err(te) = unify(&shape_ty, &expected_shape_ty, subst) {
                    errors.push(te.into());
                    return Type::Error;
                }
            }
            input_ty
        }
        _ => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                "reshape expects tensor input".to_string(),
                vec![],
            ));
            Type::Error
        }
    }
}

/// `shrink(&x, [[s0, e0], [s1, e1], ...]) -> tensor[e0-s0, e1-s1, ..., p]`
///
/// Per spec/05-risc-primitives.md §2.4, `shrink` slices a sub-tensor whose
/// rank matches the input and whose i-th axis dim is `end_i - start_i`.
/// The second argument is a list-of-pair-of-int32 with one entry per input
/// axis. Each pair is `[start, end]` with `0 <= start < end <= input_dim[i]`.
///
/// Closes issue Chelis-Lang/chelis#187 on the type-system side: before this
/// path was added, `shrink` was registered as `tensor_unop` (1-arg
/// `&tensor -> tensor`) so `shrink(&x, bounds)` failed with
/// `function arity mismatch: expected 1 args` even though the IR lowering
/// at `crates/chelis-ir/src/lower.rs:4635-4647` reads bounds from `args[1]`.
#[allow(clippy::too_many_arguments)]
fn infer_shrink_app(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.len() != 3 {
        errors.push(CheckError::new(
            CheckErrorKind::ArityMismatch,
            "shrink expects a tensor and a list of [start, end] bounds pairs, one pair per axis"
                .to_string(),
            vec![],
        ));
        return Type::Error;
    }

    let _func_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let input_ty = infer_expr(
        &kids[1],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let bounds_ty = infer_expr(
        &kids[2],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );

    if matches!(input_ty, Type::Error) || matches!(bounds_ty, Type::Error) {
        return Type::Error;
    }

    // The bounds argument must be a `List[List[Int32]]`.
    let int_list = Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int32)]);
    let expected_bounds_ty = Type::Adt("List".to_string(), vec![int_list]);
    if let Err(_te) = unify(&bounds_ty, &expected_bounds_ty, subst) {
        errors.push(CheckError::new(
            CheckErrorKind::TypeMismatch,
            with_macro_provenance(
                &deep::Expr::List(list.clone(), zero_span()),
                format!(
                    "shrink expects a list of [start, end] int32 bounds pairs, got {}",
                    subst.apply(&bounds_ty)
                ),
            ),
            vec![],
        ));
        return Type::Error;
    }

    let input_resolved = type_for_readonly_check(&input_ty, subst);
    let (dims, prec) = match input_resolved {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(&input_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!("shrink expects tensor input, got {other}"),
                ),
                vec![],
            ));
            return Type::Error;
        }
    };

    // Classify the (already desugared) Cons/Nil chain. The three-way
    // result distinguishes "concrete literals" (validate precisely)
    // from "structure looks fine but elements are non-literal" (defer
    // to runtime, output wildcards) from "structurally malformed"
    // (reject at infer with a clear axis-tagged message). See PR #214
    // red team round 1 finding R1-F1.
    let bounds: Vec<Option<(i64, i64)>> = match cons_chain_int_pairs(&kids[2]) {
        PairListShape::Literal(pairs) => pairs.into_iter().map(Some).collect(),
        // chelis#616: per-axis mixing — a literal pair keeps its precise
        // extent (and its infer-time validation); only a RUNTIME pair's own
        // axis becomes a wildcard. Collapsing every axis let unification
        // fill a runtime axis from a sibling literal axis, which the C
        // backend then baked as a wrong, unguarded allocation extent.
        PairListShape::Mixed(pairs) => pairs,
        PairListShape::Unknown => {
            return Type::Tensor(vec![Dim::Wildcard; dims.len()], prec);
        }
        PairListShape::Malformed { axis, reason } => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!("shrink axis {axis} pair {reason}"),
                ),
                vec![],
            ));
            return Type::Error;
        }
    };

    if bounds.len() != dims.len() {
        errors.push(CheckError::new(
            CheckErrorKind::ArityMismatch,
            with_macro_provenance(
                &deep::Expr::List(list.clone(), zero_span()),
                format!(
                    "shrink expects {} bounds pairs for rank {} tensor, got {}",
                    dims.len(),
                    dims.len(),
                    bounds.len()
                ),
            ),
            vec![],
        ));
        return Type::Error;
    }

    let mut out_dims = Vec::with_capacity(dims.len());
    for (axis, (pair, dim)) in bounds.iter().zip(dims.iter()).enumerate() {
        let Some((start, end)) = pair else {
            // Runtime bounds on this axis: extent known only at run time.
            out_dims.push(Dim::Wildcard);
            continue;
        };
        if *start < 0 || *end < 0 {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!("shrink axis {axis} bound [{start}, {end}] has negative endpoint"),
                ),
                vec![],
            ));
            return Type::Error;
        }
        if *start >= *end {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!(
                        "shrink axis {axis} bound [{start}, {end}] is empty or inverted (start >= end)"
                    ),
                ),
                vec![],
            ));
            return Type::Error;
        }
        if let Dim::Lit(input_dim) = dim
            && *end > *input_dim
        {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!(
                        "shrink axis {axis} bound [{start}, {end}] is out of range for input dim {input_dim}"
                    ),
                ),
                vec![],
            ));
            return Type::Error;
        }
        out_dims.push(Dim::Lit(end - start));
    }

    Type::Tensor(out_dims, prec)
}

/// `stride(&x, s0, s1, ...) -> tensor[ceil_div(d0, s0), ...]`
///
/// Per spec/05-risc-primitives.md §2.4, `stride` takes every `s_i`-th
/// element along axis i; the i-th output dim is `ceil(input_dim[i] /
/// s_i)`. The strides are passed as variadic int32 args, one per input
/// axis. Zero or negative strides are rejected.
///
/// Closes issue Chelis-Lang/chelis#187 on the type-system side -- before
/// this path was added, `stride` was registered as `tensor_unop` (arity
/// 1) so `stride(&x, 1, 2)` failed with `function arity mismatch:
/// expected 1 args`.
#[allow(clippy::too_many_arguments)]
fn infer_stride_app(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.len() < 3 {
        errors.push(CheckError::new(
            CheckErrorKind::ArityMismatch,
            "stride expects a tensor followed by one positive int32 stride per axis".to_string(),
            vec![],
        ));
        return Type::Error;
    }

    let _func_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let input_ty = infer_expr(
        &kids[1],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let stride_tys: Vec<Type> = kids[2..]
        .iter()
        .map(|arg| {
            infer_expr(
                arg,
                env,
                vg,
                subst,
                adt_reg,
                errors,
                typed_nodes,
                total_nodes,
            )
        })
        .collect();

    if matches!(input_ty, Type::Error) || stride_tys.iter().any(|ty| matches!(ty, Type::Error)) {
        return Type::Error;
    }

    for stride_ty in &stride_tys {
        let resolved = subst.apply(stride_ty);
        match resolved {
            Type::Prim(Prim::Int32) | Type::Var(_) | Type::Error => {}
            other => {
                errors.push(CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!("stride expects int32 strides, got {other}"),
                    ),
                    vec![],
                ));
                return Type::Error;
            }
        }
    }

    let input_resolved = type_for_readonly_check(&input_ty, subst);
    let (dims, prec) = match input_resolved {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(&input_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!("stride expects tensor input, got {other}"),
                ),
                vec![],
            ));
            return Type::Error;
        }
    };

    // Per-axis extraction (chelis#616): a literal (possibly cast-wrapped)
    // step keeps its precise infer-time validation and extent math; only a
    // RUNTIME step's own axis becomes a wildcard. Collapsing every axis to
    // a wildcard let unification fill a runtime axis's extent from a
    // sibling literal axis (see the shrink arm). Uses
    // `extract_int_for_dim` so `cast(N, int32)`-wrapped literal strides
    // reach the positive-stride check at infer time instead of falling
    // back to host runtime (red team round 3 finding R3-HIGH1).
    let strides: Vec<Option<i64>> = kids[2..].iter().map(extract_int_for_dim).collect();

    if strides.len() != dims.len() {
        errors.push(CheckError::new(
            CheckErrorKind::ArityMismatch,
            with_macro_provenance(
                &deep::Expr::List(list.clone(), zero_span()),
                format!(
                    "stride expects {} strides for rank {} tensor, got {}",
                    dims.len(),
                    dims.len(),
                    strides.len()
                ),
            ),
            vec![],
        ));
        return Type::Error;
    }

    let mut out_dims = Vec::with_capacity(dims.len());
    for (axis, (step, dim)) in strides.iter().zip(dims.iter()).enumerate() {
        let Some(step) = step else {
            // Runtime step on this axis: extent known only at run time.
            out_dims.push(Dim::Wildcard);
            continue;
        };
        if *step <= 0 {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!(
                        "stride axis {axis} step {step} must be positive (zero or negative strides are not allowed)"
                    ),
                ),
                vec![],
            ));
            return Type::Error;
        }
        let step_us = *step as usize;
        match dim {
            Dim::Lit(input_dim) => {
                let out = (*input_dim as usize).div_ceil(step_us);
                out_dims.push(Dim::Lit(out as i64));
            }
            other => out_dims.push(other.clone()),
        }
    }

    Type::Tensor(out_dims, prec)
}

/// `pad(&x, [[lo_0, hi_0], [lo_1, hi_1], ...], fill) -> tensor[d_0 + lo_0
/// + hi_0, ..., p]`
///
/// Per spec/05-risc-primitives.md §2.4, `pad` widens each axis by the
/// `(lo, hi)` padding amounts and fills the inserted region with `fill`.
/// Same structural antipattern as `shrink`: the `tensor_unop` registration
/// said 1-arg, but the IR lowering at `crates/chelis-ir/src/lower.rs:4616-4634`
/// reads padding from `args[1]` and fill from `args[2]`. Sibling sweep
/// finding for issue Chelis-Lang/chelis#187.
#[allow(clippy::too_many_arguments)]
fn infer_pad_app(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.len() != 4 {
        errors.push(CheckError::new(
            CheckErrorKind::ArityMismatch,
            "pad expects a tensor, a list of [lo, hi] padding pairs (one per axis), and a fill scalar".to_string(),
            vec![],
        ));
        return Type::Error;
    }

    let _func_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let input_ty = infer_expr(
        &kids[1],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let padding_ty = infer_expr(
        &kids[2],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let fill_ty = infer_expr(
        &kids[3],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );

    if matches!(input_ty, Type::Error) || matches!(padding_ty, Type::Error) {
        return Type::Error;
    }

    let int_list = Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int32)]);
    let expected_padding_ty = Type::Adt("List".to_string(), vec![int_list]);
    if let Err(_te) = unify(&padding_ty, &expected_padding_ty, subst) {
        errors.push(CheckError::new(
            CheckErrorKind::TypeMismatch,
            with_macro_provenance(
                &deep::Expr::List(list.clone(), zero_span()),
                format!(
                    "pad expects a list of [lo, hi] int32 padding pairs, got {}",
                    subst.apply(&padding_ty)
                ),
            ),
            vec![],
        ));
        return Type::Error;
    }

    let input_resolved = type_for_readonly_check(&input_ty, subst);
    let (dims, prec) = match input_resolved {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(&input_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!("pad expects tensor input, got {other}"),
                ),
                vec![],
            ));
            return Type::Error;
        }
    };

    // R1-F2: enforce the fill arg is a scalar of the input tensor
    // precision. The previous code dropped `fill_ty` on the floor, so a
    // list, tuple, bool, or wrong-precision scalar would slip through to
    // host-runtime. Per spec/05-risc-primitives.md §2.4, `pad`'s fill
    // value is a single scalar of the input precision.
    //
    // Use unification rather than a hard match so polymorphic-precision
    // tensors (precision still a `TensorPrec::Var`) generate the
    // constraint cleanly instead of being rejected. The expected scalar
    // type is `Type::Prim(p)` where `p` is the tensor's element
    // precision.
    let expected_fill_ty = match prec {
        TensorPrec::Concrete(p) => Type::Prim(p),
        TensorPrec::Var(_) => {
            // Precision is still polymorphic; introduce a fresh tvar and
            // let unification tie it to whatever the tensor lands on.
            Type::Var(vg.fresh_tvar())
        }
    };
    if let Err(_te) = unify(&fill_ty, &expected_fill_ty, subst) {
        errors.push(CheckError::new(
            CheckErrorKind::TypeMismatch,
            with_macro_provenance(
                &deep::Expr::List(list.clone(), zero_span()),
                format!(
                    "pad fill must be a scalar of the input tensor precision ({expected_fill_ty}), got {}",
                    subst.apply(&fill_ty)
                ),
            ),
            vec![],
        ));
        return Type::Error;
    }

    let padding: Vec<Option<(i64, i64)>> = match cons_chain_int_pairs(&kids[2]) {
        PairListShape::Literal(pairs) => pairs.into_iter().map(Some).collect(),
        // chelis#616: per-axis mixing — only a RUNTIME pair's own axis
        // wildcards (see the shrink arm for the mis-size this prevents).
        PairListShape::Mixed(pairs) => pairs,
        PairListShape::Unknown => {
            return Type::Tensor(vec![Dim::Wildcard; dims.len()], prec);
        }
        PairListShape::Malformed { axis, reason } => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!("pad axis {axis} pair {reason}"),
                ),
                vec![],
            ));
            return Type::Error;
        }
    };

    if padding.len() != dims.len() {
        errors.push(CheckError::new(
            CheckErrorKind::ArityMismatch,
            with_macro_provenance(
                &deep::Expr::List(list.clone(), zero_span()),
                format!(
                    "pad expects {} padding pairs for rank {} tensor, got {}",
                    dims.len(),
                    dims.len(),
                    padding.len()
                ),
            ),
            vec![],
        ));
        return Type::Error;
    }

    let mut out_dims = Vec::with_capacity(dims.len());
    for (axis, (pair, dim)) in padding.iter().zip(dims.iter()).enumerate() {
        let Some((lo, hi)) = pair else {
            // Runtime padding on this axis: extent known only at run time.
            out_dims.push(Dim::Wildcard);
            continue;
        };
        if *lo < 0 || *hi < 0 {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!("pad axis {axis} padding [{lo}, {hi}] has negative entry"),
                ),
                vec![],
            ));
            return Type::Error;
        }
        match dim {
            Dim::Lit(input_dim) => {
                out_dims.push(Dim::Lit(input_dim + lo + hi));
            }
            other => out_dims.push(other.clone()),
        }
    }

    Type::Tensor(out_dims, prec)
}

/// `reduce_window_*(&x, window_shape, strides)` infer.
///
/// Per `spec/05-risc-primitives.md` §2.3.1:
/// - `window_shape` and `strides` are `List[int32]` of equal length
///   `n >= 1`.
/// - The trailing `n` axes of the input are the windowed axes; leading
///   `rank - n` axes pass through.
/// - Each window/stride entry must be a positive int32 literal at
///   check time (non-literal arguments fall back to a wildcard output
///   shape so runtime checks can still apply).
/// - Output rank equals input rank. Trailing dim i is
///   `floor((input_dims[rank - n + i] - window_shape[i]) / strides[i]) + 1`.
///   A non-positive result is rejected as a `DimensionMismatch` per
///   §2.3.1.
#[allow(clippy::too_many_arguments)]
fn infer_reduce_window_app(
    list: &deep::List,
    name: &str,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.len() != 4 {
        errors.push(CheckError::new(
            CheckErrorKind::ArityMismatch,
            format!(
                "{name} expects 3 arguments (tensor, window_shape, strides), got {}",
                kids.len().saturating_sub(1)
            ),
            vec![],
        ));
        return Type::Error;
    }
    let _func_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let input_ty = infer_expr(
        &kids[1],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let window_ty = infer_expr(
        &kids[2],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let stride_ty = infer_expr(
        &kids[3],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );

    if matches!(input_ty, Type::Error)
        || matches!(window_ty, Type::Error)
        || matches!(stride_ty, Type::Error)
    {
        return Type::Error;
    }

    let int_list = Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int32)]);
    if let Err(_te) = unify(&window_ty, &int_list, subst) {
        errors.push(CheckError::new(
            CheckErrorKind::TypeMismatch,
            with_macro_provenance(
                &deep::Expr::List(list.clone(), zero_span()),
                format!(
                    "{name} expects window_shape to be List[int32], got {}",
                    subst.apply(&window_ty)
                ),
            ),
            vec![],
        ));
        return Type::Error;
    }
    if let Err(_te) = unify(&stride_ty, &int_list, subst) {
        errors.push(CheckError::new(
            CheckErrorKind::TypeMismatch,
            with_macro_provenance(
                &deep::Expr::List(list.clone(), zero_span()),
                format!(
                    "{name} expects strides to be List[int32], got {}",
                    subst.apply(&stride_ty)
                ),
            ),
            vec![],
        ));
        return Type::Error;
    }

    let input_resolved = type_for_readonly_check(&input_ty, subst);
    let (dims, prec) = match input_resolved {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(&input_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!("{name} expects tensor input, got {other}"),
                ),
                vec![],
            ));
            return Type::Error;
        }
    };

    // Extract literal window / stride entries. Non-literal arguments
    // are accepted at infer time (the type is still `List[int32]`) but
    // the output shape collapses to wildcards so the host runtime can
    // do the final shape check.
    let window_lit = cons_chain_int_list(&kids[2]);
    let strides_lit = cons_chain_int_list(&kids[3]);
    let (Some(window_shape), Some(strides)) = (window_lit, strides_lit) else {
        return Type::Tensor(vec![Dim::Wildcard; dims.len()], prec);
    };

    if window_shape.is_empty() || strides.is_empty() {
        errors.push(CheckError::new(
            CheckErrorKind::ArityMismatch,
            with_macro_provenance(
                &deep::Expr::List(list.clone(), zero_span()),
                format!("{name} requires a non-empty window_shape and strides"),
            ),
            vec![],
        ));
        return Type::Error;
    }
    if window_shape.len() != strides.len() {
        errors.push(CheckError::new(
            CheckErrorKind::ArityMismatch,
            with_macro_provenance(
                &deep::Expr::List(list.clone(), zero_span()),
                format!(
                    "{name} window_shape (len {}) and strides (len {}) must agree",
                    window_shape.len(),
                    strides.len()
                ),
            ),
            vec![],
        ));
        return Type::Error;
    }
    let n = window_shape.len();
    if dims.len() < n {
        errors.push(CheckError::new(
            CheckErrorKind::DimensionMismatch,
            with_macro_provenance(
                &deep::Expr::List(list.clone(), zero_span()),
                format!("{name} window arity {n} exceeds tensor rank {}", dims.len()),
            ),
            vec![],
        ));
        return Type::Error;
    }
    for (i, &w) in window_shape.iter().enumerate() {
        if w <= 0 {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!("{name} window_shape[{i}] = {w} must be >= 1"),
                ),
                vec![],
            ));
            return Type::Error;
        }
    }
    for (i, &s) in strides.iter().enumerate() {
        if s <= 0 {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!("{name} strides[{i}] = {s} must be >= 1"),
                ),
                vec![],
            ));
            return Type::Error;
        }
    }
    let leading = dims.len() - n;
    let mut out_dims = Vec::with_capacity(dims.len());
    out_dims.extend(dims[..leading].iter().map(|d| subst.apply_dim(d)));
    for i in 0..n {
        let resolved = subst.apply_dim(&dims[leading + i]);
        match &resolved {
            Dim::Lit(in_dim) => {
                let w = window_shape[i];
                let s = strides[i];
                if *in_dim < w {
                    errors.push(CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        with_macro_provenance(
                            &deep::Expr::List(list.clone(), zero_span()),
                            format!(
                                "{name} axis {} input dim {in_dim} < window_shape[{i}] = {w}",
                                leading + i
                            ),
                        ),
                        vec![],
                    ));
                    return Type::Error;
                }
                // `in_dim >= w` (checked above) and `s >= 1` guarantee
                // `out = floor((in_dim - w) / s) + 1 >= 1`, so the Valid
                // output extent is always positive here — the `in_dim < w`
                // guard above is what rejects the empty-window case.
                let out = (*in_dim - w) / s + 1;
                out_dims.push(Dim::Lit(out));
            }
            _ => out_dims.push(Dim::Wildcard),
        }
    }

    Type::Tensor(out_dims, prec)
}

/// Walk a `Cons(a, Cons(b, ..., Nil))` chain and return the literal
/// integer entries (cast-aware via `extract_int_for_dim`). Returns
/// `None` when any element is non-literal or when the structure does
/// not terminate cleanly in `Nil`.
///
/// Shares the cons-chain walk with `collect_cons_chain_for_shape`
/// (the structural recognizer) and only adds the per-element
/// integer-literal extraction on top.
fn cons_chain_int_list(expr: &deep::Expr) -> Option<Vec<i64>> {
    collect_cons_chain_for_shape(expr)?
        .iter()
        .map(|e| extract_int_for_dim(e))
        .collect()
}

/// Three-way result of inspecting a `[[s_0, e_0], [s_1, e_1], ...]` list
/// literal arg: well-formed concrete literals, structurally malformed
/// (wrong inner length, missing `Nil`, etc.), or "structure looks fine
/// but inner entries are non-literal" (e.g. variables) so the caller
/// should fall back to a wildcard output shape.
///
/// Red team round 1 on PR #214 found that `cons_chain_int_pairs`
/// returning a plain `Option` couldn't distinguish "user wrote a triple"
/// from "user wrote a variable" -- both became `None`, both fell through
/// to `Dim::Wildcard`, so malformed input silently slipped past
/// `chelis check` and only failed at host-runtime or IR-verifier time.
enum PairListShape {
    /// Top-level chain closed by `Nil`, every entry was a literal
    /// `Cons(start, Cons(end, Nil))` pair.
    Literal(Vec<(i64, i64)>),
    /// Top-level chain closed by `Nil` and every entry was structurally
    /// a `Cons(_, Cons(_, Nil))`, but at least one inner element was a
    /// non-literal (variable, call, etc.). Carries the PER-AXIS
    /// classification: `Some((start, end))` for a literal pair (which
    /// the caller must still validate and size precisely), `None` for a
    /// runtime pair (that axis alone becomes a wildcard; runtime
    /// validates it). chelis#616 red-team finding: collapsing EVERY
    /// axis to a wildcard here let downstream unification fill a
    /// runtime axis's extent from a sibling literal axis, and the C
    /// backend baked the wrong literal with no guard — a silent
    /// mis-size.
    Mixed(Vec<Option<(i64, i64)>>),
    /// At least one inner entry has the wrong structural shape (wrong
    /// number of elements, missing `Nil` close, etc.). The caller MUST
    /// emit an infer-time error naming the offending axis.
    Malformed { axis: usize, reason: String },
    /// The top-level chain is well-typed as `List[List[Int32]]` but
    /// isn't a literal Cons/Nil chain (e.g. it's a variable resolved by
    /// the type system). Caller falls back to wildcard output shape.
    Unknown,
}

/// Walk a `Cons(Cons(start_i, Cons(end_i, Nil)), ..., Nil)` chain — the
/// desugared form of a Surf `[[start_0, end_0], [start_1, end_1], ...]`
/// list-of-pair literal — and classify it via [`PairListShape`].
fn cons_chain_int_pairs(expr: &deep::Expr) -> PairListShape {
    let mut pairs: Vec<Option<(i64, i64)>> = Vec::new();
    let mut any_non_literal = false;
    let mut cursor = expr;
    let mut axis = 0usize;
    loop {
        let deep::Expr::List(outer, _) = cursor else {
            return PairListShape::Unknown;
        };
        match get_tag(outer) {
            Some("var") => {
                let name = match children(outer).first().and_then(symbol_name) {
                    Some(name) => name,
                    None => return PairListShape::Unknown,
                };
                if name == "Nil" {
                    if any_non_literal {
                        return PairListShape::Mixed(pairs);
                    }
                    return PairListShape::Literal(
                        pairs
                            .into_iter()
                            .map(|pair| pair.expect("all literal"))
                            .collect(),
                    );
                }
                return PairListShape::Unknown;
            }
            Some("app") => {
                let app_children = children(outer);
                let func = match app_children.first() {
                    Some(func) => func,
                    None => return PairListShape::Unknown,
                };
                if !is_builtin_var(func, "Cons") {
                    return PairListShape::Unknown;
                }
                let pair_expr = match app_children.get(1) {
                    Some(p) => p,
                    None => return PairListShape::Unknown,
                };
                let tail = match app_children.get(2) {
                    Some(t) => t,
                    None => return PairListShape::Unknown,
                };
                match cons_chain_two_ints(pair_expr, axis) {
                    InnerPairShape::Literal(pair) => pairs.push(Some(pair)),
                    InnerPairShape::NonLiteral => {
                        any_non_literal = true;
                        pairs.push(None);
                    }
                    InnerPairShape::Malformed { reason } => {
                        return PairListShape::Malformed { axis, reason };
                    }
                    InnerPairShape::Unknown => return PairListShape::Unknown,
                }
                cursor = tail;
                axis += 1;
            }
            _ => return PairListShape::Unknown,
        }
    }
}

/// Classification of a single inner pair expression. Distinguishes the
/// "wrong shape" case (must be reported at infer) from the "right shape,
/// non-literal element" case (defer to runtime).
enum InnerPairShape {
    Literal((i64, i64)),
    NonLiteral,
    Malformed { reason: String },
    Unknown,
}

/// Walk a `Cons(start, Cons(end, Nil))` chain and classify it. Counts
/// the actual number of elements in the inner list so the error message
/// can name the bad arity explicitly (e.g. "got 3-element list").
///
/// Bare `(var Nil)` at the top level is the desugared form of `[]` --
/// a zero-element list literal. That is just as malformed as a triple
/// or singleton (it has zero of the required two endpoints), so it
/// must surface as `Malformed { reason: "got 0-element list" }` rather
/// than `NonLiteral` (red team round 2 finding R2-M1). Other `var` tags
/// represent opaque `List[Int32]` references the type system already
/// constrained; those still defer to runtime via `NonLiteral`.
///
/// Inner head values are extracted via [`extract_int_for_dim`], which
/// peels `cast(N, int32)` / `cast(N, int64)` -- so cast-wrapped int
/// literals participate in the infer-time bounds check rather than
/// silently falling back to `NonLiteral` (red team round 2 finding
/// R2-L1; mirrors how reshape extracts dim literals).
fn cons_chain_two_ints(expr: &deep::Expr, _axis: usize) -> InnerPairShape {
    let deep::Expr::List(list, _) = expr else {
        return InnerPairShape::Unknown;
    };
    if get_tag(list) != Some("app") {
        // Inner element is not a Cons-chain. The `Nil` case (zero-element
        // list literal) is malformed; any other `var` is an opaque
        // `List[Int32]` reference whose contents the runtime will check.
        if matches!(get_tag(list), Some("var")) {
            let is_nil = children(list)
                .first()
                .and_then(symbol_name)
                .map(|name| name == "Nil")
                .unwrap_or(false);
            if is_nil {
                return InnerPairShape::Malformed {
                    reason: "expects a pair [start, end] of two int literals, got 0-element list"
                        .to_string(),
                };
            }
            return InnerPairShape::NonLiteral;
        }
        return InnerPairShape::Unknown;
    }
    // Count the elements in the inner list so we can give a precise
    // "got N-element list" diagnostic. Walk the chain element-by-element.
    let mut elements_seen = 0usize;
    let mut head_values: Vec<Option<i64>> = Vec::new();
    let mut inner_cursor: &deep::Expr = expr;
    loop {
        let deep::Expr::List(inner, _) = inner_cursor else {
            return InnerPairShape::Unknown;
        };
        match get_tag(inner) {
            Some("var") => {
                let name = match children(inner).first().and_then(symbol_name) {
                    Some(n) => n,
                    None => return InnerPairShape::Unknown,
                };
                if name != "Nil" {
                    return InnerPairShape::Unknown;
                }
                if elements_seen != 2 {
                    return InnerPairShape::Malformed {
                        reason: format!(
                            "expects a pair [start, end] of two int literals, got {}-element list",
                            elements_seen
                        ),
                    };
                }
                let start = match head_values[0] {
                    Some(v) => v,
                    None => return InnerPairShape::NonLiteral,
                };
                let end = match head_values[1] {
                    Some(v) => v,
                    None => return InnerPairShape::NonLiteral,
                };
                return InnerPairShape::Literal((start, end));
            }
            Some("app") => {
                let app_children = children(inner);
                let func = match app_children.first() {
                    Some(f) => f,
                    None => return InnerPairShape::Unknown,
                };
                if !is_builtin_var(func, "Cons") {
                    return InnerPairShape::Unknown;
                }
                let head_expr = match app_children.get(1) {
                    Some(h) => h,
                    None => return InnerPairShape::Unknown,
                };
                let tail = match app_children.get(2) {
                    Some(t) => t,
                    None => return InnerPairShape::Unknown,
                };
                head_values.push(extract_int_for_dim(head_expr));
                elements_seen += 1;
                inner_cursor = tail;
                // Guard against extra trailing elements: if we already
                // saw a [start, end] pair but the chain continues past
                // `Nil`, report malformed. The Nil arm above catches the
                // n==2 happy path before we get here on subsequent
                // iterations, so just keep walking and the count check
                // at Nil-time will catch it.
            }
            _ => return InnerPairShape::Unknown,
        }
    }
}

fn list_literal_len(expr: &deep::Expr) -> Option<usize> {
    let deep::Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("list") {
        return None;
    }
    Some(children(list).len())
}

/// Build the output dim list for `reshape(input, shape_list)`.
///
/// Walks `shape_expr` element by element. For each element, the first
/// recognizer that matches wins:
///
/// 1. concrete int literal (or `cast(N, int{32,64})`) → `Dim::Lit(N)`;
/// 2. `cast(shape(input, lit_axis), int64)` where the inner var matches
///    the reshape input by name and `lit_axis` is a valid axis of the
///    input → the input's dim at that axis (resolved through `subst`);
/// 3. fallback → `Dim::Wildcard`.
///
/// The shape list itself may surface as the explicit `(list ...)` tag
/// form or as a `Cons(head, ..., Nil)` chain after desugaring; both are
/// recognized. If neither form is matched, the rank is inferred from
/// `list_literal_len` (best-effort), and the whole output is filled
/// with `Wildcard`s -- the pre-fix behavior.
///
/// `input_var_name` is the name of the reshape input expression when it
/// is a bare `(var {} NAME)`, otherwise `None`. The symbolic-dim
/// recognizer requires this to match; with a non-var reshape input the
/// rule conservatively falls back to `Wildcard`.
///
/// Pre-fix the body of `infer_reshape_app` ran a literal-only
/// recognizer (the now-removed `list_literal_dims`) and fell back to
/// `vec![Wildcard; rank]` for anything else, including the common
/// runtime-batch pattern `cast(shape(x, axis), int64)`. That blind spot
/// is chelis#206; this helper closes it. The earlier `Dim::Lit`-only
/// behavior is also still covered (see the RT-A1W1 CRIT root cause for
/// chelis#35: `reshape(t, [2, 1, 3])` must yield
/// `tensor[Lit(2), Lit(1), Lit(3), p]`, not `tensor[Wildcard, ..., p]`).
fn reshape_output_dims(
    shape_expr: &deep::Expr,
    input_var_name: Option<&str>,
    input_dims: &[Dim],
    subst: &Subst,
) -> Vec<Dim> {
    let elements = match collect_shape_list_elements(shape_expr) {
        Some(elems) => elems,
        None => {
            let rank = list_literal_len(shape_expr).unwrap_or(1);
            return vec![Dim::Wildcard; rank];
        }
    };
    elements
        .into_iter()
        .map(|elem| reshape_output_dim(elem, input_var_name, input_dims, subst))
        .collect()
}

/// Recognize a single dim-list element from a reshape shape list.
fn reshape_output_dim(
    elem: &deep::Expr,
    input_var_name: Option<&str>,
    input_dims: &[Dim],
    subst: &Subst,
) -> Dim {
    if let Some(n) = extract_int_for_dim(elem) {
        return Dim::Lit(n);
    }
    if input_var_name.is_some()
        && let Some(axis) = extract_shape_axis_of(elem, input_var_name)
        && let Some(dim) = input_dims.get(axis)
    {
        // Resolve through current substitution so a recently-bound dim
        // var surfaces as its concrete name/lit.
        return subst.apply_dim(dim);
    }
    Dim::Wildcard
}

/// Collect the elements of a reshape shape-list argument as a flat
/// `Vec` of expressions, handling both the `(list ...)` tag form and
/// the desugared `Cons(head, ..., Nil)` chain. Returns `None` if the
/// shape arg is not a recognized list form (in which case the caller
/// falls back to all-wildcards with rank inferred from `list_literal_len`).
fn collect_shape_list_elements(expr: &deep::Expr) -> Option<Vec<&deep::Expr>> {
    if let deep::Expr::List(list, _) = expr
        && get_tag(list) == Some("list")
    {
        return Some(children(list).iter().collect());
    }
    let mut elems = Vec::new();
    let mut cursor = expr;
    loop {
        let deep::Expr::List(list, _) = cursor else {
            return None;
        };
        match get_tag(list)? {
            "var" => {
                let name = children(list).first().and_then(symbol_name)?;
                if name == "Nil" {
                    return Some(elems);
                }
                return None;
            }
            "app" => {
                let app_children = children(list);
                let func = app_children.first()?;
                if !is_builtin_var(func, "Cons") {
                    return None;
                }
                elems.push(app_children.get(1)?);
                cursor = app_children.get(2)?;
            }
            _ => return None,
        }
    }
}

/// If `expr` has the syntactic form
/// `cast(shape(<var named input_var_name>, <concrete int axis>), int64)`,
/// return the axis. Both `cast` and `shape` may surface either as the
/// dedicated tag (`(cast {} ...)`, ...) or as `(app {} (var {} cast)
/// ...)`. The axis expression matches `extract_int_for_dim` -- it
/// accepts `N`, `lit N`, and `cast(N, int{32,64})`.
///
/// Returns `None` when:
/// - `expr` doesn't match the expected outer cast-to-int64,
/// - the inner expression is not a `shape(...)` call,
/// - the shape's tensor arg is not a `var` matching `input_var_name`,
/// - the axis is not a concrete non-negative int.
fn extract_shape_axis_of(expr: &deep::Expr, input_var_name: Option<&str>) -> Option<usize> {
    let input = input_var_name?;
    let (inner, target_ty) = peel_cast(expr)?;
    if !is_target_ty(target_ty, Prim::Int64) {
        return None;
    }
    let shape_call = inner_to_list(inner)?;
    if !is_shape_app(shape_call) {
        return None;
    }
    let shape_args = children(shape_call);
    let func = shape_args.first()?;
    if !is_builtin_var(func, "shape") {
        return None;
    }
    let tensor_arg = shape_args.get(1)?;
    let arg_name = symbolic_dim_ref_name(tensor_arg)?;
    if arg_name != input {
        return None;
    }
    let axis_expr = shape_args.get(2)?;
    let axis = extract_int_for_dim(axis_expr)?;
    if axis < 0 {
        return None;
    }
    Some(axis as usize)
}

/// Strip one layer of `cast` (tag-form or `app`-form) and return
/// (inner_expr, target_type_expr).
fn peel_cast(expr: &deep::Expr) -> Option<(&deep::Expr, &deep::Expr)> {
    let deep::Expr::List(list, _) = expr else {
        return None;
    };
    match get_tag(list)? {
        "cast" => {
            let kids = children(list);
            Some((kids.first()?, kids.get(1)?))
        }
        "app" => {
            let kids = children(list);
            let func = kids.first()?;
            if !is_builtin_var(func, "cast") {
                return None;
            }
            Some((kids.get(1)?, kids.get(2)?))
        }
        _ => None,
    }
}

/// Treat `(t-prim {} <name>)` as the target type marker emitted by
/// `cast(..., int64)` etc. Returns true iff the marker matches `prim`.
fn is_target_ty(expr: &deep::Expr, prim: Prim) -> bool {
    let deep::Expr::List(list, _) = expr else {
        return false;
    };
    if get_tag(list) != Some("t-prim") {
        return false;
    }
    let Some(name_expr) = children(list).first() else {
        return false;
    };
    symbol_name(name_expr) == Some(prim.name())
}

/// Treat `expr` as an `(app {} ...)` list and return its `deep::List`,
/// or `None` if it isn't.
fn inner_to_list(expr: &deep::Expr) -> Option<&deep::List> {
    let deep::Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) == Some("app") {
        Some(list)
    } else {
        None
    }
}

/// True iff `app_list` is an `(app {} (var {} shape) ...)`.
fn is_shape_app(app_list: &deep::List) -> bool {
    children(app_list)
        .first()
        .map(|f| is_builtin_var(f, "shape"))
        .unwrap_or(false)
}

/// Extract an int literal from a Deep expr, looking through `cast(N, int64)`
/// and `cast(N, int32)` — both are common in Chelis dim lists since integer
/// literals default to int32 and require an explicit cast for int64 contexts.
/// `cast` may surface either as the `(cast {} ... ...)` tag or as an `app`
/// of the `cast` var, depending on how far desugaring has progressed.
///
/// Note: callers in the spec section 2.4 movement family (shrink, pad,
/// stride, permute, expand) typically unify the surrounding bounds /
/// strides argument against an `int32`-pinned expected type before
/// reaching this extractor, so a `cast(N, int64)` endpoint is rejected
/// at the outer unification step rather than slipping through to here
/// (red team round 3 HIGH-2 contract note).
///
/// The cast arm recurses through `extract_int_for_dim` so that
/// `cast(cast(N, int32), int32)` and other doubly-nested forms peel to
/// their literal at any depth (red team round 3 finding R3-MED2).
/// Termination is bounded: each recursive call strictly reduces the
/// expression depth (peels one wrapper layer).
///
/// Audit catalog of issue #216 sites that use this cast-aware extractor
/// (one row per infer-time int-literal extraction that gates a
/// user-facing validation check). Each row also notes any host-runtime
/// defense-in-depth so a regression here does not silently corrupt
/// runtime behavior, only the diagnostic layer.
///
/// | Domain               | Site (approx)                        | User-reachable cast? | Validation                | Host-runtime defense |
/// |----------------------|--------------------------------------|----------------------|---------------------------|----------------------|
/// | trace/diagonal axis  | `resolve_axis_pair_member` (~4110)   | yes                  | rank bounds + diagnostic  | yes (eval)           |
/// | builtin axis         | `resolve_builtin_axis` (~4154)       | yes                  | rank bounds + diagnostic  | yes (eval)           |
/// | conv2d output type   | `derive_conv2d_output_type` (~5720)  | yes (`stride=cast`)  | positivity + spatial dim  | yes (validator arm)  |
/// | conv2d output type   | `derive_conv2d_output_type` (~5721)  | yes (`padding=cast`) | non-neg + spatial dim     | yes (validator arm)  |
/// | conv2d validator     | `extract_typed_scalar_literal`(~5874)| yes                  | literal-int + then >0/>=0 | yes (codegen panic)  |
/// | conv2d axis-dim      | `ir_builtin_axis_dim` (~5907)        | yes                  | rank bounds via normalize | yes (eval)           |
/// | softmax axis         | softmax arm (~7630)                  | yes (`axis=cast`)    | rank bounds + diagnostic  | yes (eval)           |
/// | shape axis           | shape arm (~8238)                    | yes (issue #206)     | non-neg + rank bounds     | yes (eval)           |
/// | split axis           | split arm (~8959)                    | yes                  | rank bounds + diagnostic  | yes (eval)           |
/// | conv2d spatial out   | `compute_concrete_conv2d_spatial`(11722)| yes                | positivity + spatial dim  | yes (validator arm)  |
/// | conv2d spatial out   | `compute_concrete_conv2d_spatial`(11723)| yes                | non-neg + spatial dim     | yes (validator arm)  |
/// | reduction axis       | `check_reduce_signature` (~12076)    | yes (`axis=cast`)    | rank bounds + diagnostic  | yes (eval)           |
/// | grad wrt tuple       | `grad_wrt_indices` (~13793)          | NO (surf desugar)    | int-type + non-neg        | yes (AD pass)        |
/// | grad wrt single      | `grad_wrt_indices` (~13814)          | NO (surf desugar)    | int-type + non-neg        | yes (AD pass)        |
/// | vmap axis            | `infer_vmap` (~13866)                | NO (surf parser)     | non-neg + diagnostic      | yes (eval)           |
///
/// The three "NO" rows -- vmap axis, both grad wrt sites -- have no
/// idiomatic Surf cast-wrapping pattern because the Surf parser /
/// desugarer normalizes them to bare literal ints before reaching the
/// extractor. They are reachable only through direct Deep input
/// (decompiler output, custom tooling, macro expansion). The swap there
/// is defense-in-depth on Deep-direct paths; the post-fix tests use
/// `parse_deep` rather than the Surf parser.
fn extract_int_for_dim(expr: &deep::Expr) -> Option<i64> {
    stack_guard!("extract_int_for_dim", expr, None);
    if let Some(value) = extract_int_literal(expr) {
        return Some(value);
    }
    let deep::Expr::List(list, _) = expr else {
        return None;
    };
    match get_tag(list)? {
        "cast" => extract_int_for_dim(children(list).first()?),
        "app" => {
            let app_children = children(list);
            let func = app_children.first()?;
            if !is_builtin_var(func, "cast") {
                return None;
            }
            extract_int_for_dim(app_children.get(1)?)
        }
        _ => None,
    }
}

fn check_layer_norm_signature(
    arg_tys: &[Type],
    result_ty: &Type,
    _vg: &mut VarGen,
    subst: &mut Subst,
    errors: &mut Vec<CheckError>,
) -> Type {
    if arg_tys.len() != 3 {
        return Type::Error;
    }

    let x_ty = type_for_readonly_check(&arg_tys[0], subst);
    let gamma_ty = type_for_readonly_check(&arg_tys[1], subst);
    let beta_ty = type_for_readonly_check(&arg_tys[2], subst);

    let (x_dims, x_prec) = match x_ty {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("layer_norm expects tensor input, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };

    let (gamma_dims, gamma_prec) = match gamma_ty {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("layer_norm expects tensor gamma, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };

    let (beta_dims, beta_prec) = match beta_ty {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("layer_norm expects tensor beta, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };

    if x_dims.is_empty() {
        errors.push(CheckError::new(
            CheckErrorKind::DimensionMismatch,
            "layer_norm expects rank >= 1 input tensor".to_string(),
            vec![],
        ));
        return Type::Error;
    }
    if gamma_dims.len() != 1 {
        errors.push(CheckError::new(
            CheckErrorKind::DimensionMismatch,
            format!(
                "layer_norm expects rank-1 gamma, got rank {}",
                gamma_dims.len()
            ),
            vec![],
        ));
        return Type::Error;
    }
    if beta_dims.len() != 1 {
        errors.push(CheckError::new(
            CheckErrorKind::DimensionMismatch,
            format!(
                "layer_norm expects rank-1 beta, got rank {}",
                beta_dims.len()
            ),
            vec![],
        ));
        return Type::Error;
    }
    if x_prec != gamma_prec || x_prec != beta_prec {
        errors.push(CheckError::new(
            CheckErrorKind::PrecisionMismatch,
            format!(
                "layer_norm requires matching precisions, got {}, {}, {}",
                x_prec.name(),
                gamma_prec.name(),
                beta_prec.name()
            ),
            vec!["Insert explicit cast".to_string()],
        ));
        return Type::Error;
    }

    let hidden_dim = x_dims.last().cloned().expect("checked non-empty");
    if let Err(te) = unify_dim(&hidden_dim, &gamma_dims[0], subst) {
        errors.push(te.into());
        return Type::Error;
    }
    if let Err(te) = unify_dim(&hidden_dim, &beta_dims[0], subst) {
        errors.push(te.into());
        return Type::Error;
    }

    let canonical = Type::Tensor(
        x_dims.into_iter().map(|d| subst.apply_dim(&d)).collect(),
        x_prec,
    );
    if let Err(te) = unify(result_ty, &canonical, subst) {
        errors.push(te.into());
        return Type::Error;
    }
    subst.apply(&canonical)
}

/// If `arg_exprs[2]` and `arg_exprs[3]` are integer literals and the
/// input/kernel spatial dims (axes 2, 3) resolve to concrete
/// `Dim::Lit` values after substitution, return the computed output
/// spatial extents `(out_h, out_w)`. Returns `None` if any of the
/// inputs are non-literal or non-concrete; the caller falls back to
/// fresh dim-vars in that case.
///
/// `extract_int_literal` already handles the canonical
/// `(lit {type: ...} N)` Deep shape used for stride/padding literals.
fn compute_concrete_conv2d_spatial(
    arg_exprs: &[deep::Expr],
    input_dims: &[Dim],
    kernel_dims: &[Dim],
    subst: &Subst,
) -> Option<(i64, i64)> {
    // Issue #216: cast-aware so cast-wrapped stride/padding still
    // resolve the concrete spatial output dims at infer time.
    let stride = arg_exprs.get(2).and_then(extract_int_for_dim)?;
    let padding = arg_exprs.get(3).and_then(extract_int_for_dim)?;
    if stride <= 0 || padding < 0 {
        return None;
    }
    let in_h = match subst.apply_dim(input_dims.get(2)?) {
        Dim::Lit(v) => v,
        _ => return None,
    };
    let in_w = match subst.apply_dim(input_dims.get(3)?) {
        Dim::Lit(v) => v,
        _ => return None,
    };
    let k_h = match subst.apply_dim(kernel_dims.get(2)?) {
        Dim::Lit(v) => v,
        _ => return None,
    };
    let k_w = match subst.apply_dim(kernel_dims.get(3)?) {
        Dim::Lit(v) => v,
        _ => return None,
    };
    // `conv2d_output_extent` returns None on i64 overflow (RT-205
    // round-2 F1); fall back to fresh dim-vars in that case so the
    // validator's arm reports the overflow with a precise diagnostic
    // rather than us computing here with saturating math and
    // producing a confusing dim-lit-vs-dim-lit mismatch.
    let out_h = conv2d_output_extent(in_h, k_h, stride, padding)?;
    let out_w = conv2d_output_extent(in_w, k_w, stride, padding)?;
    if out_h <= 0 || out_w <= 0 {
        // Let the validator's arm emit the diagnostic; here we just
        // fall back to fresh dim-vars so the inference pass produces
        // a useful (declared-vs-fresh) mismatch instead of failing
        // here with a confusing dim-lit-vs-dim-lit unify error.
        return None;
    }
    Some((out_h, out_w))
}

fn check_conv2d_signature(
    arg_exprs: &[deep::Expr],
    arg_tys: &[Type],
    result_ty: &Type,
    vg: &mut VarGen,
    subst: &mut Subst,
    errors: &mut Vec<CheckError>,
) -> Type {
    if arg_tys.len() < 2 {
        return Type::Error;
    }

    let input_ty = type_for_readonly_check(&arg_tys[0], subst);
    let kernel_ty = type_for_readonly_check(&arg_tys[1], subst);

    let (input_dims, input_prec) = match input_ty {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("conv2d expects tensor input, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };
    let (kernel_dims, kernel_prec) = match kernel_ty {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("conv2d expects tensor kernel, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };

    if input_dims.len() != 4 {
        errors.push(CheckError::new(
            CheckErrorKind::DimensionMismatch,
            format!(
                "conv2d expects rank-4 input tensor, got rank {}",
                input_dims.len()
            ),
            vec![],
        ));
        return Type::Error;
    }
    if kernel_dims.len() != 4 {
        errors.push(CheckError::new(
            CheckErrorKind::DimensionMismatch,
            format!(
                "conv2d expects rank-4 kernel tensor, got rank {}",
                kernel_dims.len()
            ),
            vec![],
        ));
        return Type::Error;
    }
    if input_prec != kernel_prec {
        errors.push(CheckError::new(
            CheckErrorKind::PrecisionMismatch,
            format!(
                "conv2d requires matching input/kernel precision, got {} and {}",
                input_prec.name(),
                kernel_prec.name()
            ),
            vec!["Insert explicit cast".to_string()],
        ));
        return Type::Error;
    }
    if let Err(te) = unify_dim(&input_dims[1], &kernel_dims[1], subst) {
        errors.push(te.into());
        return Type::Error;
    }

    // RT-205 F8: when stride/padding are integer literals and the
    // input/kernel spatial dims are concrete Dim::Lit values, compute
    // the output spatial dims via the canonical formula
    // (`floor((in + 2 * padding - kernel) / stride) + 1`) and place
    // concrete `Dim::Lit` values into the output template. Without
    // this the placeholders are fresh dim-vars that unify with any
    // positive declared spatial dim, so an explicit but WRONG
    // declared output (e.g. `tensor[1, 8, 100, 100]` for the
    // canonical 8x8 input + 3x3 kernel case whose real output is
    // 6x6) silently type-checks.
    let computed_spatial =
        compute_concrete_conv2d_spatial(arg_exprs, &input_dims, &kernel_dims, subst);
    let (out_h_dim, out_w_dim) = match computed_spatial {
        Some((h, w)) => (Dim::Lit(h), Dim::Lit(w)),
        None => (Dim::Var(vg.fresh_dvar()), Dim::Var(vg.fresh_dvar())),
    };
    let output_template = Type::Tensor(
        vec![
            subst.apply_dim(&input_dims[0]),
            subst.apply_dim(&kernel_dims[0]),
            out_h_dim,
            out_w_dim,
        ],
        input_prec.clone(),
    );
    if let Err(te) = unify(result_ty, &output_template, subst) {
        errors.push(te.into());
        return Type::Error;
    }

    let resolved_output = subst.apply(&output_template);
    if let Type::Tensor(out_dims, out_prec) = &resolved_output {
        if out_dims.len() != 4 {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!("conv2d result must be rank 4, got rank {}", out_dims.len()),
                vec![],
            ));
            return Type::Error;
        }
        if *out_prec != input_prec {
            errors.push(CheckError::new(
                CheckErrorKind::PrecisionMismatch,
                format!(
                    "conv2d result precision must match input/kernel precision {}, got {}",
                    input_prec.name(),
                    out_prec.name()
                ),
                vec!["Insert explicit cast".to_string()],
            ));
            return Type::Error;
        }
        if let Err(te) = unify_dim(&out_dims[0], &input_dims[0], subst) {
            errors.push(te.into());
            return Type::Error;
        }
        if let Err(te) = unify_dim(&out_dims[1], &kernel_dims[0], subst) {
            errors.push(te.into());
            return Type::Error;
        }
    }

    subst.apply(&output_template)
}

fn check_matmul_signature(
    arg_tys: &[Type],
    result_ty: &Type,
    subst: &mut Subst,
    errors: &mut Vec<CheckError>,
) -> Type {
    if arg_tys.len() != 2 {
        return Type::Error;
    }

    let lhs = type_for_readonly_check(&arg_tys[0], subst);
    let rhs = type_for_readonly_check(&arg_tys[1], subst);

    let (lhs_dims, lhs_prec) = match lhs {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("matmul expects tensor lhs, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };
    let (rhs_dims, rhs_prec) = match rhs {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("matmul expects tensor rhs, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };

    if lhs_prec != rhs_prec {
        errors.push(CheckError::new(
            CheckErrorKind::PrecisionMismatch,
            format!(
                "matmul requires matching precisions, got {} and {}",
                lhs_prec.name(),
                rhs_prec.name()
            ),
            vec!["Insert explicit cast".to_string()],
        ));
        return Type::Error;
    }
    // RT-2 fixup B6: per spec/04-type-system.md §5.7.2, the active
    // matmul signature does not admit integer operand precisions
    // (int8, int16, int32, int64). Reject upfront at the call site
    // with a §5.7.2-citing diagnostic so users see the spec rule
    // here, not as a downstream IR-verify or codegen failure. The
    // verify-layer F1 guard remains as defense in depth.
    if lhs_prec.is_integer() {
        errors.push(CheckError::new(
            CheckErrorKind::PrecisionMismatch,
            format!(
                "matmul on integer operand precision `{}` is not admitted in this \
                 cycle per spec/04-type-system.md §5.7.2: integer matmul not admitted \
                 (the spec deliberately defers the integer-matmul accumulator rule; \
                 use reduce_sum over an explicit expand+mul lowering for integer \
                 inner products)",
                lhs_prec.name()
            ),
            vec![format!(
                "spec/04-type-system.md §5.7.2: there is no current backend that \
                 supports integer BLAS, and an integer-matmul surface raises \
                 questions (saturating vs wrapping accumulator, signed-vs-unsigned \
                 interaction with §1.1.2) that are out of scope here. Integer \
                 reduce_sum is supported per §5.7.1."
            )],
        ));
        return Type::Error;
    }
    if lhs_dims.len() < 2 || rhs_dims.len() < 2 {
        errors.push(CheckError::new(
            CheckErrorKind::DimensionMismatch,
            format!(
                "matmul expects tensors of rank >= 2, got rank {} and {}",
                lhs_dims.len(),
                rhs_dims.len()
            ),
            vec![],
        ));
        return Type::Error;
    }
    if let Err(te) = unify_dim(
        &lhs_dims[lhs_dims.len() - 1],
        &rhs_dims[rhs_dims.len() - 2],
        subst,
    ) {
        errors.push(te.into());
        return Type::Error;
    }

    let lhs_lead = &lhs_dims[..lhs_dims.len() - 2];
    let rhs_lead = &rhs_dims[..rhs_dims.len() - 2];
    let lead_len = lhs_lead.len().max(rhs_lead.len());
    let mut out_dims = Vec::with_capacity(lead_len + 2);
    for offset in 0..lead_len {
        let lhs_idx = lhs_lead.len().checked_sub(lead_len - offset);
        let rhs_idx = rhs_lead.len().checked_sub(lead_len - offset);
        let dim = match (
            lhs_idx.map(|idx| &lhs_lead[idx]),
            rhs_idx.map(|idx| &rhs_lead[idx]),
        ) {
            (Some(lhs_dim), Some(rhs_dim)) => {
                let lhs_applied = subst.apply_dim(lhs_dim);
                let rhs_applied = subst.apply_dim(rhs_dim);
                if lhs_applied == Dim::Lit(1) {
                    rhs_applied
                } else if rhs_applied == Dim::Lit(1) {
                    lhs_applied
                } else {
                    if let Err(te) = unify_dim(&lhs_applied, &rhs_applied, subst) {
                        errors.push(te.into());
                        return Type::Error;
                    }
                    subst.apply_dim(&lhs_applied)
                }
            }
            (Some(lhs_dim), None) => subst.apply_dim(lhs_dim),
            (None, Some(rhs_dim)) => subst.apply_dim(rhs_dim),
            (None, None) => unreachable!(),
        };
        out_dims.push(dim);
    }
    out_dims.push(subst.apply_dim(&lhs_dims[lhs_dims.len() - 2]));
    out_dims.push(subst.apply_dim(&rhs_dims[rhs_dims.len() - 1]));

    let canonical = Type::Tensor(out_dims, lhs_prec);
    if let Err(te) = unify(result_ty, &canonical, subst) {
        errors.push(te.into());
        return Type::Error;
    }
    subst.apply(&canonical)
}

fn check_reduction_signature(
    name: &str,
    arg_exprs: &[deep::Expr],
    arg_tys: &[Type],
    result_ty: &Type,
    subst: &mut Subst,
    errors: &mut Vec<CheckError>,
) -> Type {
    if arg_tys.len() < 2 {
        return Type::Error;
    }

    let input_ty = type_for_readonly_check(&arg_tys[0], subst);
    let (dims, prec) = match input_ty {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("{name} expects tensor input, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };

    // Resolve which axis (or axes) the reduction removes. Two modes:
    //
    //  * Positional (legacy): a single compile-time-constant integer axis on a
    //    *concrete-rank* operand (`sum(x, 0)` / `sum(x, cast(-1, int32))`).
    //    `normalize_static_axis` handles negative indexing and bounds (issue
    //    #216), consistent with gather/scatter and IR lowering's
    //    `normalize_axis`.
    //
    //  * Named (Tier-3, spec/04-type-system.md §4.5.3): one or more axes named by
    //    the dimension they remove (`sum(x, seq)` / `sum(x, seq, head)`). Named
    //    axes are the only valid form on a rank-spread operand — a positional
    //    index is meaningless at symbolic rank — and they preserve the
    //    surviving named axes in the output.
    let axis_exprs = &arg_exprs[1..];
    let has_spread = dims.iter().any(|d| matches!(d, Dim::Rank(_)));

    // The reduction HM schemes are arity-2, but the variadic named-axis form
    // (`sum(x, seq, head)`, chelis#339) reaches this arm through the
    // `infer_reduction_app` dispatcher with N axis exprs; the loop below
    // resolves each named axis and rejects positional integers, unknown
    // names, ambiguity, and duplicates. Composition
    // (`sum(sum(x, head), seq)`) remains equivalent and order-insensitive.
    let mut remove: Vec<usize> = Vec::new();
    if axis_exprs.len() == 1
        && !has_spread
        && let Some(raw) = extract_int_for_dim(&axis_exprs[0])
    {
        match normalize_static_axis(dims.len(), raw) {
            Some(axis) => remove.push(axis),
            None => {
                errors.push(CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!(
                        "{name} axis {raw} is out of bounds for rank {} tensor",
                        dims.len()
                    ),
                    vec![],
                ));
                return Type::Error;
            }
        }
    } else {
        for ax in axis_exprs {
            // A positional integer that reaches the named path: either the
            // operand is rank-spread (index meaningless at symbolic rank) or it
            // is mixed with other axes. Direct the user to name each axis.
            if extract_int_for_dim(ax).is_some() {
                errors.push(CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!(
                        "{name}: a positional integer axis is only valid as the single axis of a \
                         concrete-rank operand; on a rank-spread operand or for multiple axes, \
                         name each axis (e.g. `{name}(x, seq)` or `{name}(x, seq, head)`) so it \
                         is located by name (spec/04-type-system.md \u{00a7}4.5.3)"
                    ),
                    vec![],
                ));
                return Type::Error;
            }
            // Issue #259: a non-literal, non-name axis (a runtime `int32`
            // binding) cannot determine which dimension is removed; emit the
            // targeted compile-time-constant diagnostic rather than leaking an
            // unresolved output type downstream.
            let Some(axis_name) = symbolic_dim_ref_name(ax) else {
                errors.push(CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!(
                        "{name} axis must be a compile-time constant or a named axis of the \
                         operand, got {}",
                        describe_axis_arg(Some(ax)),
                    ),
                    vec![format!(
                        "Pass a literal axis (e.g. `{name}(x, 0)`) on a concrete-rank operand, or \
                         name the axis (e.g. `{name}(x, seq)`) to reduce by name."
                    )],
                ));
                return Type::Error;
            };
            let hits: Vec<usize> = dims
                .iter()
                .enumerate()
                .filter(|(_, d)| matches!(d, Dim::Name(n) if n == axis_name))
                .map(|(i, _)| i)
                .collect();
            match hits.as_slice() {
                [i] => {
                    // chelis#339: a duplicate axis name in the variadic list
                    // is a hard error, never a silent deduplication.
                    if remove.contains(i) {
                        errors.push(CheckError::new(
                            CheckErrorKind::DimensionMismatch,
                            format!(
                                "{name}: duplicate reduction axis `{axis_name}`; each named \
                                 axis may appear at most once in a variadic reduction \
                                 (spec/04-type-system.md \u{00a7}4.5.3)"
                            ),
                            vec![],
                        ));
                        return Type::Error;
                    }
                    remove.push(*i);
                }
                [] if has_spread => {
                    errors.push(CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        format!(
                            "{name}: rank-spread operand has no named `{axis_name}` axis to \
                             reduce (spec/04-type-system.md \u{00a7}4.5.3)"
                        ),
                        vec![],
                    ));
                    return Type::Error;
                }
                [] => {
                    // Concrete operand: `axis_name` is neither a literal nor a
                    // named axis of the operand. Two causes share this arm — a
                    // runtime `int32` binding (issue #259) and a mistyped/absent
                    // axis name — so the message stays neutral between them
                    // rather than asserting "runtime value".
                    errors.push(CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        format!(
                            "{name} axis `{axis_name}` is neither a compile-time constant nor a \
                             named axis of the operand: a reduction axis must be a literal or \
                             `cast(N, int32)` constant, or the name of an existing axis"
                        ),
                        vec![format!(
                            "Pass a literal axis (e.g. `{name}(x, 0)`) or `cast(N, int32)`, or \
                             name an existing axis of the operand (e.g. `{name}(x, seq)`)."
                        )],
                    ));
                    return Type::Error;
                }
                _ => {
                    errors.push(CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        format!(
                            "{name}: named axis `{axis_name}` is ambiguous; it appears more than \
                             once in the operand shape"
                        ),
                        vec![],
                    ));
                    return Type::Error;
                }
            }
        }
    }

    // Build the output by dropping the selected axes (descending so earlier
    // indices stay valid). Surviving named/spread dims keep identity and
    // order. Duplicates were rejected loudly above (chelis#339), so no
    // silent dedup happens here.
    let mut out_dims = dims;
    remove.sort_unstable();
    for &idx in remove.iter().rev() {
        out_dims.remove(idx);
    }

    // RT-2 fixup B1: per spec/04-type-system.md §5.7.1, the result
    // precision of `reduce_sum` follows the §5.7.1 table — int8/int16
    // operand → int32 result, int32/int64/f32/f64 → operand precision,
    // bf16/f16 → operand precision (the f32 accumulator is consumed
    // inside the op and downcast on output). For `max_reduce`,
    // `min_reduce`, `prod_reduce`, and `mean` the result precision is
    // the operand precision.
    //
    // Issue #230: `argmax_reduce` and `argmin_reduce` are index-returning
    // reductions — they produce element indices, not reduced operand
    // values. Their result precision is canonically `int64`, regardless
    // of the input dtype. The std-package signatures in
    // `packages/chelis-std/src/tensor/reduce.ch` pin this (`tensor[b,
    // int64]`); the type checker was returning the input precision and
    // diverging from std. (The host-runtime/backend still stores
    // integer-valued floats internally per the Phase 3j-pre Batch 1
    // caveat documented on `RiscOp::Argmax`; the int64 label is the
    // declarative output type.)
    //
    // WS-A5: the §5.7.1 widening rule is defined over a known operand
    // precision. If the operand precision is still polymorphic
    // (TensorPrec::Var), defer the decision until the precision is
    // resolved by unification — return the canonical-but-still-poly
    // result type and let the standard unify path proceed.
    let result_prec: TensorPrec = if name == "sum" {
        match &prec {
            TensorPrec::Concrete(p) => match p.default_reduce_sum_result_precision() {
                Ok(rp) => TensorPrec::Concrete(rp),
                Err(msg) => {
                    errors.push(CheckError::new(
                        CheckErrorKind::TypeMismatch,
                        format!("sum: {msg}"),
                        vec![],
                    ));
                    return Type::Error;
                }
            },
            TensorPrec::Var(_) => prec.clone(),
        }
    } else if name == "argmax_reduce" || name == "argmin_reduce" {
        TensorPrec::Concrete(Prim::Int64)
    } else {
        prec.clone()
    };
    // RT-2 fixup B1: emit a §5.7.1-citing diagnostic at the call site
    // before falling back to the generic unify error, so users binding
    // `sum(int8 tensor)` to `tensor[int8]` see the spec-row hint
    // instead of the opaque "doesn't match declared signature" trail.
    if name == "sum" && result_prec != prec {
        let resolved_result = subst.apply(result_ty);
        if let Type::Tensor(_, declared_prec) = resolved_result
            && declared_prec != result_prec
        {
            errors.push(CheckError::new(
                CheckErrorKind::PrecisionMismatch,
                format!(
                    "sum on operand precision `{}` produces result precision `{}` per \
                     spec/04-type-system.md §5.7.1 (the §5.7.1 result-precision table \
                     widens narrow integer operands to int32 to prevent silent overflow); \
                     declared result precision `{}` is incompatible. Use `tensor[{}]` or \
                     omit the result type to accept the spec default.",
                    prec.render(),
                    result_prec.render(),
                    declared_prec.render(),
                    result_prec.render(),
                ),
                vec![format!(
                    "spec/04-type-system.md §5.7.1: `reduce_sum` on `{}` operands \
                     produces a `{}` result by default to prevent silent overflow",
                    prec.render(),
                    result_prec.render(),
                )],
            ));
            return Type::Error;
        }
    }
    let canonical = Type::Tensor(out_dims, result_prec);
    if let Err(te) = unify(result_ty, &canonical, subst) {
        errors.push(te.into());
        return Type::Error;
    }
    subst.apply(&canonical)
}

#[allow(clippy::too_many_arguments)]
fn check_expand_signature(
    arg_exprs: &[deep::Expr],
    arg_tys: &[Type],
    result_ty: &Type,
    axis_is_dim_name: bool,
    size_class: SizeClass,
    env: &Env,
    subst: &mut Subst,
    errors: &mut Vec<CheckError>,
) -> Type {
    if arg_tys.len() != 3 && arg_tys.len() != 4 {
        return Type::Error;
    }

    let input_ty = type_for_readonly_check(&arg_tys[0], subst);
    let (input_dims, input_prec) = match input_ty {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("expand expects tensor input, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };

    let has_spread = input_dims.iter().any(|d| matches!(d, Dim::Rank(_)));

    // chelis#339 named-axis expand (spec/04-type-system.md §4.5.3): when the
    // axis argument is a dimension NAME rather than an integer, the call
    // inserts a new named axis — at the trailing end (3-arg form) or
    // immediately before an existing named anchor (4-arg form). This is the
    // only valid expand form on a rank-spread operand. `axis_is_dim_name` is
    // scope-discriminated by the caller: a bare var bound in the value
    // environment is a runtime value (issue #259), not a dim name, and falls
    // through to the compile-time-constant rejection below.
    if axis_is_dim_name
        && arg_exprs.get(1).and_then(extract_int_for_dim).is_none()
        && let Some(new_name) = arg_exprs.get(1).and_then(symbolic_dim_ref_name)
    {
        return check_named_expand_signature(
            new_name,
            arg_exprs,
            &input_dims,
            has_spread,
            input_prec,
            result_ty,
            subst,
            errors,
        );
    }

    // From here on the call is the positional concrete-rank form. A fourth
    // (anchor) argument is only meaningful in the named-axis form.
    if arg_exprs.len() == 4 {
        errors.push(CheckError::new(
            CheckErrorKind::DimensionMismatch,
            "expand takes a fourth (anchor) argument only in the named-axis form \
             `expand(x, new, size, anchor)`, where `new` names the inserted axis \
             (spec/04-type-system.md \u{00a7}4.5.3)"
                .to_string(),
            vec![],
        ));
        return Type::Error;
    }
    // A positional index is meaningless at symbolic rank: against a spread
    // there is no fixed position to insert at. Mirror the reduction arm —
    // name the inserted axis instead.
    if has_spread {
        errors.push(CheckError::new(
            CheckErrorKind::DimensionMismatch,
            "expand: a positional integer axis is only valid on a concrete-rank \
             operand; on a rank-spread operand, name the inserted axis (e.g. \
             `expand(x, one, 1)` for a trailing insert, or `expand(x, c, n, seq)` \
             to insert before the named `seq` anchor) so the insertion point stays \
             name-anchored (spec/04-type-system.md \u{00a7}4.5.3)"
                .to_string(),
            vec![],
        ));
        return Type::Error;
    }

    // Uses `extract_int_for_dim` so `cast(N, int32)`-wrapped literal
    // axis/size reach the non-negative-axis and positive-size checks at
    // infer time (red team round 3 sibling sweep within the spec
    // section 2.4 movement family).
    let axis = match arg_exprs.get(1).and_then(extract_int_for_dim) {
        Some(axis) if axis >= 0 => axis as usize,
        Some(axis) => {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!("expand requires non-negative axis, got {axis}"),
                vec![],
            ));
            return Type::Error;
        }
        // Issue #259: the input is a concrete tensor (past the
        // `Var | Error` guard above), so the output shape is determinable
        // once the insert axis is known. When the axis arg is not a
        // compile-time constant, `extract_int_for_dim` returns `None` and
        // we cannot place the inserted dimension. Pre-fix this arm returned
        // the still-unresolved `Type::Var(out)` from `tensor_expand_to_out`,
        // which leaked downstream and surfaced as a misleading
        // `borrow requires tensor or tensor-carrying input, got ?N`. Emit a
        // targeted diagnostic at the expand call site naming the root cause.
        None => {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!(
                    "expand axis must be a compile-time constant for the output \
                     shape to be inferable, got {}",
                    describe_axis_arg(arg_exprs.get(1)),
                ),
                vec![
                    "Pass a literal axis (e.g. `expand(x, 0, n)`) or a `cast(N, int32)` \
                     literal. The axis selects where the new dimension is inserted, so \
                     it must be known at compile time."
                        .to_string(),
                ],
            ));
            return Type::Error;
        }
    };
    let size = match arg_exprs.get(2).and_then(extract_int_for_dim) {
        Some(size) if size > 0 => Dim::Lit(size),
        Some(size) => {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!("expand requires positive size, got {size}"),
                vec![],
            ));
            return Type::Error;
        }
        // A non-literal runtime size. chelis#397/#469: discriminate by
        // PROVENANCE (computed by the caller as `size_class`), not by the
        // surface spelling. A size whose value provably folds to a constant
        // (`Static`) or derives from an in-scope tensor's `shape(t, axis)`
        // read / dimension name (`ShapeSourced`) is materializable; a truly
        // sourceless runtime scalar (`Sourceless` — a bare `int32`/`int64`
        // parameter, a `cast`/arithmetic over one, or a `let` bound to such)
        // has no backend representation and is rejected here so check, build,
        // and eval all agree (a check-clean program must build). The walk
        // unifies the four spellings the #397 red team found drifting:
        // bare-`var`, `cast(var, _)`, `let`-bound, and arithmetic.
        None => {
            if size_class == SizeClass::Sourceless {
                errors.push(sourceless_expand_size_error(arg_exprs.get(2)));
                return Type::Error;
            }
            // A bare `var` naming a genuine §4.7.2 Form-2 symbolic dim — a
            // declared dim parameter (not a value binding) or a dim carried
            // by an in-scope tensor — stamps the named dim into the output so
            // declared results refer to it by name. Every other materializable
            // spelling (`shape(...)` reads, static arithmetic, `cast`-wrapped,
            // and `let`-bound sizes — Form-3) defers the output dim slot to
            // the declared return-type / call-context via unification.
            match arg_exprs.get(2).and_then(symbolic_dim_ref_name) {
                Some(name) if env.lookup(name).is_none() || env.tensor_carries_dim(name) => {
                    Dim::Name(name.to_string())
                }
                _ => return subst.apply(result_ty),
            }
        }
    };

    let resolved_result = subst.apply(result_ty);
    let canonical = match resolved_result {
        Type::Tensor(out_dims, out_prec) => {
            if out_prec != input_prec {
                errors.push(CheckError::new(
                    CheckErrorKind::PrecisionMismatch,
                    format!(
                        "expand output precision {} does not match input precision {}",
                        out_prec.name(),
                        input_prec.name()
                    ),
                    vec![],
                ));
                return Type::Error;
            }
            if out_dims.len() == input_dims.len() + 1 {
                if axis > input_dims.len() {
                    errors.push(CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        format!(
                            "expand insert axis {axis} is out of bounds for rank {} tensor",
                            input_dims.len()
                        ),
                        vec![],
                    ));
                    return Type::Error;
                }
                let mut expected = input_dims.clone();
                expected.insert(axis, size.clone());
                Type::Tensor(expected, input_prec)
            } else if out_dims.len() == input_dims.len() {
                if axis >= input_dims.len() {
                    errors.push(CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        format!(
                            "expand axis {axis} is out of bounds for rank {} tensor",
                            input_dims.len()
                        ),
                        vec![],
                    ));
                    return Type::Error;
                }
                let mut expected = input_dims.clone();
                expected[axis] = size.clone();
                Type::Tensor(expected, input_prec)
            } else {
                errors.push(CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!(
                        "expand output rank {} must equal input rank {} or {}",
                        out_dims.len(),
                        input_dims.len(),
                        input_dims.len() + 1
                    ),
                    vec![],
                ));
                return Type::Error;
            }
        }
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("expand expects tensor output, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };

    if let Err(te) = unify(result_ty, &canonical, subst) {
        errors.push(te.into());
        return Type::Error;
    }
    subst.apply(&canonical)
}

/// The named-axis expand arm (chelis#339, spec/04-type-system.md §4.5.3):
/// insert a new axis named `new_name` into the operand's row form — at the
/// trailing end (3-arg form) or immediately before the named `anchor` axis
/// (4-arg form). Insertion is unitary: the position is an end of the row or
/// fixed by an anchor located uniquely in the operand; everything else is a
/// hard error, never a guessed placement. The inserted dim enters the
/// symbolic output as `Dim::Name`, so declared results refer to it by name
/// and call-site monomorphization carries it through.
#[allow(clippy::too_many_arguments)]
fn check_named_expand_signature(
    new_name: &str,
    arg_exprs: &[deep::Expr],
    input_dims: &[Dim],
    has_spread: bool,
    input_prec: TensorPrec,
    result_ty: &Type,
    subst: &mut Subst,
    errors: &mut Vec<CheckError>,
) -> Type {
    // The inserted name must not collide with an existing named axis: a
    // duplicate dim name would make every later by-name lookup (reduction,
    // anchor location) ambiguous.
    if input_dims
        .iter()
        .any(|d| matches!(d, Dim::Name(n) if n == new_name))
    {
        errors.push(CheckError::new(
            CheckErrorKind::DimensionMismatch,
            format!(
                "expand: inserted axis `{new_name}` already names an axis of the operand; \
                 a duplicate dim name would make later by-name axis lookups ambiguous \
                 (spec/04-type-system.md \u{00a7}4.5.3). Pick a fresh name for the inserted \
                 axis."
            ),
            vec![],
        ));
        return Type::Error;
    }

    // The named-insert size must be a positive compile-time literal (a bare
    // int or `cast(N, int32)`). A symbolic-dim or runtime int32 size cannot
    // be stamped onto the inserted named dim at lowering: the eval lane has
    // no extent to stage and the C backend would emit an undeclared dim
    // symbol (silent shape-0 output) — both verified failure modes, so the
    // checker rejects the form outright rather than letting a check-clean
    // program break downstream (chelis#339).
    let Some(size) = arg_exprs.get(2).and_then(extract_int_for_dim) else {
        errors.push(CheckError::new(
            CheckErrorKind::DimensionMismatch,
            format!(
                "expand: the named-axis insert form requires a compile-time literal size \
                 (a literal or `cast(N, int32)` constant), got {}; the inserted axis's \
                 extent must be stampable onto the new named dim at lowering \
                 (spec/04-type-system.md \u{00a7}4.5.3)",
                describe_axis_arg(arg_exprs.get(2)),
            ),
            vec![],
        ));
        return Type::Error;
    };
    if size <= 0 {
        errors.push(CheckError::new(
            CheckErrorKind::DimensionMismatch,
            format!("expand requires positive size, got {size}"),
            vec![],
        ));
        return Type::Error;
    }

    // Resolve the insertion point: before the unique named anchor (4-arg
    // form) or at the trailing end of the row (3-arg form).
    let insert_at = match arg_exprs.get(3) {
        Some(anchor_expr) => {
            let Some(anchor) = symbolic_dim_ref_name(anchor_expr) else {
                errors.push(CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!(
                        "expand anchor must name an existing axis of the operand, got {} \
                         (spec/04-type-system.md \u{00a7}4.5.3)",
                        describe_axis_arg(arg_exprs.get(3)),
                    ),
                    vec![],
                ));
                return Type::Error;
            };
            let hits: Vec<usize> = input_dims
                .iter()
                .enumerate()
                .filter(|(_, d)| matches!(d, Dim::Name(n) if n == anchor))
                .map(|(i, _)| i)
                .collect();
            match hits.as_slice() {
                [i] => *i,
                [] if has_spread => {
                    errors.push(CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        format!(
                            "expand: rank-spread operand has no named `{anchor}` axis to \
                             anchor the insertion; insertion strictly inside an opaque \
                             spread has no anchor and is rejected \
                             (spec/04-type-system.md \u{00a7}4.5.3)"
                        ),
                        vec![],
                    ));
                    return Type::Error;
                }
                [] => {
                    errors.push(CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        format!(
                            "expand anchor `{anchor}` is not a named axis of the operand: \
                             the named-axis form inserts at the trailing end or immediately \
                             before an existing named anchor (spec/04-type-system.md \
                             \u{00a7}4.5.3)"
                        ),
                        vec![],
                    ));
                    return Type::Error;
                }
                _ => {
                    errors.push(CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        format!(
                            "expand: anchor `{anchor}` is ambiguous; it appears more than \
                             once in the operand shape (spec/04-type-system.md \u{00a7}4.5.3)"
                        ),
                        vec![],
                    ));
                    return Type::Error;
                }
            }
        }
        None => input_dims.len(),
    };

    // Symbolic output: insert into the row form. Surviving dims (spreads
    // included) keep identity and order; the new axis is a named dim.
    let mut out_dims = input_dims.to_vec();
    out_dims.insert(insert_at, Dim::Name(new_name.to_string()));
    let canonical = Type::Tensor(out_dims, input_prec);
    if let Err(te) = unify(result_ty, &canonical, subst) {
        errors.push(te.into());
        return Type::Error;
    }
    subst.apply(&canonical)
}

/// Result of peeling nested `List<...>` wrappers from a `to_tensor`
/// argument. Bucket 4b: previously the typer only accepted a single
/// `List<numeric|bool>` and rejected `List<List<f32>>` outright; now
/// we walk down through arbitrarily many `List` heads, count the rank,
/// and require the innermost element to be a numeric or bool prim.
enum ToTensorPeel<'a> {
    /// Successfully peeled `rank` `List` layers down to a `Prim`.
    Ok { rank: usize, precision: Prim },
    /// Some inner type is still a `Var(_)` or `Error`; the typer should
    /// defer to the explicit result type rather than emit a diagnostic.
    Pending,
    /// Reached a non-`List`, non-prim leaf — the innermost element is
    /// not numeric or bool, so emit a typed diagnostic.
    BadInner(&'a Type),
    /// The argument is not a `List` at all.
    NotList,
}

fn peel_to_tensor_argument(ty: &Type) -> ToTensorPeel<'_> {
    let mut current = ty;
    let mut rank = 0;
    loop {
        match current {
            Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                rank += 1;
                current = &args[0];
            }
            Type::Prim(precision) if precision.is_numeric() || matches!(precision, Prim::Bool) => {
                return ToTensorPeel::Ok {
                    rank,
                    precision: *precision,
                };
            }
            Type::Var(_) | Type::Error => {
                return ToTensorPeel::Pending;
            }
            other => {
                if rank == 0 {
                    return ToTensorPeel::NotList;
                }
                return ToTensorPeel::BadInner(other);
            }
        }
    }
}

/// Extract an int literal from a Deep expr, recognizing the canonical
/// literal forms (`Atom::Int`, `(lit {type: ...} N)`) and the `neg` app
/// wrapper. Float-in-cast intentionally is not recognized: the spec
/// says integer literals default to `int32` and require explicit
/// notation for other widths, so a float wrapped in a cast to an int
/// dtype is a precision-narrowing operation that the runtime should
/// validate -- not a literal int (round 3 LOW-2 design note).
///
/// The `neg` arm recurses through `extract_int_for_dim` so that
/// `neg(cast(N, int32))` peels both wrappers and resolves to `-N` at
/// infer time (red team round 3 finding R3-MED1). Mutual recursion
/// with `extract_int_for_dim` is bounded: each call strictly reduces
/// the expression depth (peels one wrapper layer).
fn extract_int_literal(expr: &deep::Expr) -> Option<i64> {
    match expr {
        deep::Expr::Atom(deep::Atom::Int(n), _) => Some(*n),
        deep::Expr::List(list, _) if get_tag(list) == Some("lit") => {
            children(list).first().and_then(|child| match child {
                deep::Expr::Atom(deep::Atom::Int(n), _) => Some(*n),
                _ => None,
            })
        }
        deep::Expr::List(list, _) if get_tag(list) == Some("app") => {
            let app_children = children(list);
            match (app_children.first(), app_children.get(1)) {
                (Some(func), Some(arg)) if is_builtin_var(func, "neg") => {
                    extract_int_for_dim(arg).map(|value| -value)
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn is_builtin_var(expr: &deep::Expr, expected: &str) -> bool {
    let deep::Expr::List(list, _) = expr else {
        return false;
    };
    get_tag(list) == Some("var") && children(list).first().and_then(symbol_name) == Some(expected)
}

/// Extract the static shape of a `to_tensor` argument when the
/// argument is a nested Cons-chain literal whose every leaf is a
/// numeric/bool atom (or a recognized `cast` / `neg` wrapper).
///
/// Returns `Some(dims)` with one `Dim::Lit(n)` per axis (outermost
/// first) when the structure is statically resolvable and every
/// axis is uniformly shaped. Returns `None` when:
///   * the argument is not a Cons-chain (e.g. a variable like
///     `to_tensor(items)`),
///   * the chain is malformed or not closed by `Nil`,
///   * a leaf is not a recognizable numeric atom,
///   * sibling axes have different lengths (ragged literal).
///
/// `expected_rank` is the rank inferred from peeling List wrappers
/// in the argument's TYPE; it is used as a sanity check, not as a
/// hard requirement. A mismatch returns `None` so the caller falls
/// back to the legacy wildcard-rank path.
///
/// This is the issue Chelis-Lang/chelis#218 R2 HIGH-A source fix:
/// by emitting concrete dims here, every downstream consumer
/// (reductions, elementwise activations, anything that reads the
/// to_tensor app's `type:` metadata) sees a sound shape instead of
/// a `Dim::Wildcard`.
fn static_to_tensor_shape(arg: &deep::Expr, expected_rank: usize) -> Option<Vec<Dim>> {
    let dims = walk_static_cons_chain_shape(arg)?;
    if dims.len() != expected_rank {
        return None;
    }
    Some(dims.into_iter().map(|n| Dim::Lit(n as i64)).collect())
}

/// Recursive helper for `static_to_tensor_shape`. Returns
/// `Some(dims)` if `expr` is a Cons / Nil chain whose every leaf
/// reduces to a numeric atom (or recursively to another Cons chain
/// of uniform length). `dims` is the rank-N shape (outermost axis
/// first).
fn walk_static_cons_chain_shape(expr: &deep::Expr) -> Option<Vec<usize>> {
    stack_guard!("walk_static_cons_chain_shape", expr, None);
    let elements = collect_cons_chain_for_shape(expr)?;
    if elements.is_empty() {
        // Empty list at the outermost level has rank 1, size 0.
        // For empty inner lists we still want a concrete dim list,
        // but ragged-but-empty cases can't be uniformly typed; the
        // top-level case is sufficient here.
        return Some(vec![0]);
    }
    // If every element is a numeric leaf, this is a rank-1 axis.
    if elements
        .iter()
        .all(|element| extract_numeric_leaf_for_shape(element).is_some())
    {
        return Some(vec![elements.len()]);
    }
    // Otherwise every element should recurse to a same-shape
    // sub-vector. The outermost axis is `elements.len()`; the inner
    // axes must agree.
    let nested: Vec<Vec<usize>> = elements
        .iter()
        .map(|element| walk_static_cons_chain_shape(element))
        .collect::<Option<_>>()?;
    let inner_shape = nested.first()?.clone();
    if nested.iter().any(|shape| shape != &inner_shape) {
        return None;
    }
    let mut out = vec![nested.len()];
    out.extend(inner_shape);
    Some(out)
}

/// Collect a `Cons(head, Cons(head, ..., Nil))` chain into a vector
/// of head expressions. Returns `None` if the chain isn't closed by
/// `Nil` or contains a non-Cons app. Local helper for
/// `walk_static_cons_chain_shape`; mirrors `cons_chain_int_dims`'s
/// chain-walking shape but returns the heads themselves so the
/// caller can recurse.
fn collect_cons_chain_for_shape(expr: &deep::Expr) -> Option<Vec<&deep::Expr>> {
    let mut out = Vec::new();
    let mut cursor = expr;
    loop {
        let deep::Expr::List(list, _) = cursor else {
            return None;
        };
        match get_tag(list)? {
            "var" => {
                let name = children(list).first().and_then(symbol_name)?;
                if name == "Nil" {
                    return Some(out);
                }
                return None;
            }
            "app" => {
                let app_children = children(list);
                let func = app_children.first()?;
                if !is_builtin_var(func, "Cons") {
                    return None;
                }
                let head = app_children.get(1)?;
                let tail = app_children.get(2)?;
                out.push(head);
                cursor = tail;
            }
            _ => return None,
        }
    }
}

/// True iff `expr` is a numeric leaf (Int/Float/Bool atom, `lit` of
/// the same, `cast` of one, or `neg` of one). Mirrors the shape of
/// `extract_numeric_leaf` in `crates/chelis-ir/src/lower.rs` so the
/// type-check and IR-lowering passes agree on what counts as a
/// "static to_tensor leaf." We don't need the actual value here,
/// only the static-recognition predicate.
fn extract_numeric_leaf_for_shape(expr: &deep::Expr) -> Option<()> {
    stack_guard!("extract_numeric_leaf_for_shape", expr, None);
    match expr {
        deep::Expr::Atom(deep::Atom::Int(_), _) => Some(()),
        deep::Expr::Atom(deep::Atom::Float(_), _) => Some(()),
        deep::Expr::Atom(deep::Atom::Bool(_), _) => Some(()),
        deep::Expr::List(list, _) => match get_tag(list)? {
            "lit" => match list.elements.get(2)? {
                deep::Expr::Atom(deep::Atom::Int(_), _)
                | deep::Expr::Atom(deep::Atom::Float(_), _)
                | deep::Expr::Atom(deep::Atom::Bool(_), _) => Some(()),
                _ => None,
            },
            "cast" => extract_numeric_leaf_for_shape(list.elements.get(2)?),
            "app" => {
                // Issue #218 R1 HIGH-1 mirror: a surface negative
                // literal `-x` desugars to `(app (var neg) <inner>)`.
                // Recurse through the unary minus so the static
                // recognizer matches the IR lowering's analogous
                // recognizer in `crates/chelis-ir/src/lower.rs`.
                let callee = children(list).first()?;
                if !is_builtin_var(callee, "neg") {
                    return None;
                }
                let inner = children(list).get(1)?;
                extract_numeric_leaf_for_shape(inner)
            }
            _ => None,
        },
        _ => None,
    }
}

fn symbolic_dim_ref_name(expr: &deep::Expr) -> Option<&str> {
    let deep::Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("var") {
        return None;
    }
    children(list).first().and_then(symbol_name)
}

/// chelis#397/#469: the materializability class of a runtime `expand` size
/// argument, by PROVENANCE rather than surface spelling.
///
/// A runtime `expand` size has a backend representation only when its
/// extent is recoverable. The discriminator must be uniform across the
/// bare-`var`, `cast`-wrapped, `let`-bound, and arithmetic spellings (the
/// four that #397's red team found drifting): all reduce to one of these
/// classes via the same recursive walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SizeClass {
    /// Folds to a compile-time constant (literal/`cast(N,_)`/arithmetic
    /// over such values, or a `let` name marked `Static`). The host runtime
    /// and the evaluator can compute it; a materializable extent.
    Static,
    /// Provably derives from an in-scope tensor's `shape(t, axis)` read, or
    /// names an in-scope tensor dimension (§4.7.2 Form-2) — directly,
    /// through `cast`, through integer arithmetic, or transitively through a
    /// `let` name marked `ShapeSourced`. The backend reads the extent from
    /// the tensor's shape.
    ShapeSourced,
    /// A runtime value with no static value and no tensor source — a bare
    /// `int32`/`int64` parameter, a `cast`/arithmetic over one, or a `let`
    /// name bound to such. No backend representation; rejected at check
    /// (#469) so check↔build↔eval agree.
    Sourceless,
    /// Not a recognized int-valued size shape (e.g. the input tensor is
    /// still a type var, or the expr is something the walk does not model).
    /// The caller leaves the existing non-rejecting behavior in place.
    Unknown,
}

/// The §4.7.2 Form-3 sourceless-size diagnostic (chelis#469), emitted by
/// `check_expand_signature` when an `expand` size resolves to `Sourceless`.
/// Factored out so the diagnostic text has a single source of truth.
/// `size_expr` is the size sub-expression (used only to name a symbolic
/// dimension when the size is a bare `var`).
fn sourceless_expand_size_error(size_expr: Option<&deep::Expr>) -> CheckError {
    let described = size_expr
        .and_then(symbolic_dim_ref_name)
        .map(|name| format!("the symbolic dimension `{name}`"))
        .unwrap_or_else(|| "a runtime scalar".to_string());
    CheckError::new(
        CheckErrorKind::DimensionMismatch,
        format!(
            "`expand` size resolves to {described}, but no tensor in scope carries \
             it: a \u{00a7}4.7.2 Form-3 runtime size must be a literal/`cast(N, \
             int32)`, an in-scope tensor dimension, or a `shape(tensor, axis)` read \
             (followed through `let`, `cast`, and integer arithmetic). A bare \
             runtime scalar (e.g. an `int32`/`int64` parameter) has no shape source \
             the backend can emit, so the extent cannot be materialized. Tracked by \
             Chelis-Lang/chelis#469 (spec/04-type-system.md \u{00a7}4.7.2)"
        ),
        vec![
            "Source the extent from a tensor in scope: read it with \
             `shape(x, cast(axis, int32))` (the `bias_broadcast` form), bind that \
             read to a `let` and pass it, or use a literal/`cast(N, int32)` size."
                .to_string(),
        ],
    )
}

/// chelis#397/#469: classify a runtime `expand` size argument by
/// provenance. Walks `cast`, integer arithmetic (`add`/`sub`/`mul`/`div`),
/// bare `var` references (resolved against `env` — an in-scope tensor dim
/// is Form-2 `ShapeSourced`, a recorded `let` provenance is followed, a
/// bare value binding with neither is `Sourceless`), and `shape(t, axis)`
/// reads. The recursion mirrors the IR layer's
/// `extract_dim_expr_value`/`symbol_has_tensor_source`/`shape_dep` triad so
/// the check-time accept set matches what the backends can materialize.
fn classify_expand_size(expr: &deep::Expr, env: &Env) -> SizeClass {
    stack_guard!("classify_expand_size", expr, SizeClass::Unknown);
    // A statically-extractable literal/`cast(N,_)` size is always Static.
    if extract_int_for_dim(expr).is_some() {
        return SizeClass::Static;
    }
    // An inline `shape(t, axis)` read (possibly `cast`-wrapped) of an
    // in-scope tensor is the canonical Form-3 shape source (`bias_broadcast`).
    if let Some(operand) = shape_read_operand(expr) {
        return if shape_operand_is_in_scope_tensor(operand, env) {
            SizeClass::ShapeSourced
        } else {
            // `shape(<non-tensor>, ...)` cannot supply an extent.
            SizeClass::Sourceless
        };
    }
    match expr {
        deep::Expr::List(list, _) => {
            match get_tag(list) {
                // `cast(<inner>, ty)` — provenance is the inner expr's.
                Some("cast") => children(list)
                    .first()
                    .map_or(SizeClass::Unknown, |inner| classify_expand_size(inner, env)),
                Some("var") => match symbolic_dim_ref_name(expr) {
                    // A name carried by an in-scope tensor's shape is a
                    // Form-2 symbolic dim with a real source.
                    Some(name) if env.tensor_carries_dim(name) => SizeClass::ShapeSourced,
                    // A recorded `let` provenance (shape-sourced or static).
                    Some(name) => match env.size_provenance(name) {
                        Some(crate::env::SizeProvenance::ShapeSourced) => SizeClass::ShapeSourced,
                        Some(crate::env::SizeProvenance::Static) => SizeClass::Static,
                        // A bare value binding (a runtime scalar parameter)
                        // with no tensor source and no static provenance.
                        None if env.lookup(name).is_some() => SizeClass::Sourceless,
                        None => SizeClass::Unknown,
                    },
                    None => SizeClass::Unknown,
                },
                // Integer arithmetic: combine the operands' classes.
                Some("app") => classify_arith_app(list, env),
                // chelis#530: any other List-shaped size — a tuple
                // projection (`t.0`), an inline `match`/`if`, a record
                // `access`, etc. — has NO backend-materializable shape
                // source. It is `Sourceless`, NOT `Unknown`: returning
                // `Unknown` here let the inline `expand(b, 0, t.0)` form
                // (and its `cast`/arithmetic wrappers) reach the
                // non-rejecting `_ => subst.apply(result_ty)` accept arm of
                // `check_expand_signature` and silently miscompile in C to a
                // hardcoded extent-1 axis (eval `[3, 2]` vs C `[1, 2]`),
                // exactly the silent-miscompile class #469 exists to
                // prevent. The `shape(t, ..)`, `cast(..)`, literal,
                // bare-`var`, and arithmetic forms are all recognized BEFORE
                // this arm, so reaching here means the size is genuinely
                // sourceless at the check layer. Mirrors the `classify_arith_app`
                // non-arith-`app` fail-closed default below.
                _ => SizeClass::Sourceless,
            }
        }
        _ => SizeClass::Unknown,
    }
}

/// Combine the size classes of an integer-arithmetic application's
/// operands (chelis#397/#469). `Sourceless` is absorbing (a sum/product
/// touching a sourceless scalar is itself sourceless); a `ShapeSourced`
/// operand makes the whole expression `ShapeSourced` (the extent is
/// recoverable from that tensor); all-`Static` operands stay `Static`.
///
/// A non-arithmetic `app` — any other function call, e.g. `ident(a_dim)` or
/// a user `def` — produces a runtime value with NO shape source the backend
/// can read the extent from, exactly like a bare runtime scalar. It is
/// `Sourceless`, NOT `Unknown`: returning `Unknown` here let the inline
/// `expand(b, 0, ident(a_dim))` form (and its `cast`/arith wrappers) reach
/// the non-rejecting `_` arm of `check_expand_signature` and silently
/// miscompile in C to a hardcoded extent-1 axis (chelis#397 BLOCKER A — the
/// same silent-miscompile class #469 exists to prevent). The `shape(t, ..)`,
/// `cast(..)`, literal, and bare-`var` forms are all recognized BEFORE this
/// arm, so reaching here means the call is genuinely sourceless at the check
/// layer (a user `def` wrapping `shape` is opaque here and would be rejected
/// at lowering too — no `shape_dep`).
fn classify_arith_app(list: &deep::List, env: &Env) -> SizeClass {
    const INT_ARITH: &[&str] = &["add", "sub", "mul", "div", "mod", "neg"];
    let kids = children(list);
    let Some(callee) = kids.first() else {
        return SizeClass::Sourceless;
    };
    let is_int_arith = INT_ARITH.iter().any(|name| is_builtin_var(callee, name));
    if !is_int_arith {
        return SizeClass::Sourceless;
    }
    let operand_classes: Vec<SizeClass> = kids[1..]
        .iter()
        .map(|arg| classify_expand_size(arg, env))
        .collect();
    if operand_classes.contains(&SizeClass::Sourceless) {
        return SizeClass::Sourceless;
    }
    if operand_classes.contains(&SizeClass::Unknown) {
        return SizeClass::Unknown;
    }
    if operand_classes.contains(&SizeClass::ShapeSourced) {
        return SizeClass::ShapeSourced;
    }
    SizeClass::Static
}

/// Recognize a `shape(operand, axis)` application — possibly wrapped in one
/// or more `cast(..., int32)` layers — and return its `operand` expr
/// (chelis#397/#469). The check-layer analog of the IR layer's
/// `shape_app_operand_axis`. The axis is not validated here (the operand's
/// presence is what proves a tensor source); a runtime axis is fine.
fn shape_read_operand(expr: &deep::Expr) -> Option<&deep::Expr> {
    stack_guard!("shape_read_operand", expr, None);
    let deep::Expr::List(list, _) = expr else {
        return None;
    };
    // Strip outer `cast(..., ty)` wrappers (tag form and app form).
    if get_tag(list) == Some("cast") {
        return children(list).first().and_then(shape_read_operand);
    }
    let kids = children(list);
    let callee = kids.first()?;
    if get_tag(list) == Some("app") && is_builtin_var(callee, "cast") {
        return kids.get(1).and_then(shape_read_operand);
    }
    if get_tag(list) == Some("app") && is_builtin_var(callee, "shape") {
        // `(app {} (var shape) <operand> <axis>)`.
        return kids.get(1);
    }
    None
}

/// True when a `shape(...)` operand expression resolves to an in-scope
/// tensor (chelis#397/#469). The operand is a bare `var` (`shape(x, 0)`) or
/// a borrow of one (`shape(&x, 0)`); either way the named binding must have
/// a tensor type in `env`. A non-tensor operand cannot supply an extent.
fn shape_operand_is_in_scope_tensor(operand: &deep::Expr, env: &Env) -> bool {
    // Unwrap a `borrow(x)`/`&x` wrapper to the underlying var.
    let var_name = shape_operand_var_name(operand);
    match var_name {
        Some(name) => env.lookup(name).is_some_and(scheme_is_tensor_carrying),
        None => false,
    }
}

/// The underlying `var` name of a `shape(...)` operand, unwrapping a
/// `borrow`/`&` layer (chelis#397/#469).
fn shape_operand_var_name(operand: &deep::Expr) -> Option<&str> {
    stack_guard!("shape_operand_var_name", operand, None);
    if let Some(name) = symbolic_dim_ref_name(operand) {
        return Some(name);
    }
    let deep::Expr::List(list, _) = operand else {
        return None;
    };
    // `&x` desugars to the `(borrow {} (var x))` TAG form; `borrow(x)`
    // may also appear as the `(app {} (var borrow) (var x))` builtin form.
    if get_tag(list) == Some("borrow") {
        return children(list).first().and_then(shape_operand_var_name);
    }
    let kids = children(list);
    let callee = kids.first()?;
    if get_tag(list) == Some("app") && is_builtin_var(callee, "borrow") {
        return kids.get(1).and_then(shape_operand_var_name);
    }
    None
}

/// True when a scheme's body is (or contains, through `Ref`) a tensor type
/// (chelis#397/#469).
fn scheme_is_tensor_carrying(scheme: &Scheme) -> bool {
    fn is_tensor(ty: &Type) -> bool {
        match ty {
            Type::Tensor(..) => true,
            Type::Ref(inner) => is_tensor(inner),
            _ => false,
        }
    }
    is_tensor(&scheme.body)
}

/// Describe a non-literal axis argument for the issue #259 diagnostic.
///
/// When the axis is a `(var name)` (the common case: a function-parameter
/// `int32` such as `mean(&x, ax)`), name it so the user can see which
/// binding is the runtime value. Otherwise fall back to a generic
/// "non-constant expression" phrasing. Kept deliberately small: this only
/// feeds a user-facing message, not a control-flow decision.
fn describe_axis_arg(axis_expr: Option<&deep::Expr>) -> String {
    match axis_expr {
        Some(expr) => match symbolic_dim_ref_name(expr) {
            Some(name) => format!("a runtime value `{name}`"),
            None => "a non-constant expression".to_string(),
        },
        None => "a missing argument".to_string(),
    }
}

/// Post-body rigidity check for a def's declared dimension parameters.
///
/// `declared_dvars` are the dim variables introduced by the declared
/// parameter signatures, snapshotted before body inference. After the
/// body is inferred, each declared dim parameter is universally
/// quantified and must stay distinct: the body must type-check for
/// *all* instantiations of those dims.
///
/// Two failure modes are flagged here, both `DimensionMismatch`:
///
/// - `Dim::Var -> Dim::Lit`: the body forced a polymorphic dim
///   parameter to a concrete literal (Nautilus Bug 2). The signature's
///   polymorphism claim is self-contradictory.
/// - `Dim::Var -> Dim::Var` (or any shared resolution) collapse: two
///   *distinct* declared dim parameters resolved to the *same*
///   dimension after body inference. The body unified two
///   universally-quantified dim parameters that must stay distinct.
///   This is Path B of `TypeCheck-FreeDimVarUnification-F1` (SR-LEAK-A):
///   `def g[n, m](x: tensor[n, f32], y: tensor[m, f32]) ->
///   tensor[n, f32] = y` collapses `n` and `m` via free `unify_dim`
///   and never routes through the Shape A relaxed-retry guard.
///
/// A single declared dim parameter appearing in multiple param
/// positions (`def h[n](x: tensor[n], y: tensor[n])`) is one dvar and
/// never trips the collapse check.
fn check_declared_dvars_rigid(
    declared_dvars: &[DimVar],
    subst: &Subst,
    errors: &mut Vec<CheckError>,
) {
    // First resolved dim seen -> the declared dvar that produced it.
    // A second declared dvar resolving to the same dim is a collapse.
    let mut seen: HashMap<Dim, DimVar> = HashMap::new();
    for dv in declared_dvars {
        let resolved = subst.apply_dim(&Dim::Var(*dv));
        if let Dim::Lit(n) = resolved {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!(
                    "polymorphic dim variable forced to concrete Lit({n}) by function body: \
                     declared dim parameters must remain polymorphic"
                ),
                vec![
                    "Replace the polymorphic dim with the concrete literal in the signature, or \
                     ensure the body does not pin the dim to a specific size"
                        .to_string(),
                ],
            ));
            continue;
        }
        // A declared dim parameter that resolves to itself (still
        // unbound) is the legitimate polymorphic case; it cannot
        // collide with another declared dvar's distinct identity.
        if let Some(&prev) = seen.get(&resolved) {
            if prev != *dv {
                errors.push(CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!(
                        "distinct declared dim parameters d{} and d{} were unified by the \
                         function body: declared dim parameters are rigid and must remain \
                         distinct",
                        prev.0, dv.0
                    ),
                    vec![
                        "The body returns or constrains a value whose dimension differs from \
                         the declared one. Use the same dim parameter on both sides if they \
                         are meant to be equal, or fix the body so each declared dim stays \
                         independent"
                            .to_string(),
                    ],
                ));
            }
        } else {
            seen.insert(resolved, *dv);
        }
    }
}

/// chelis#273 return-position rigidity guard.
///
/// `check_declared_dvars_rigid` collects declared dim parameters from
/// the signature's *parameter* positions only, so a dim parameter
/// appearing **only in the return type** was never checked and the body
/// could silently pin it (`def f[k](a: tensor[2, f32]) ->
/// tensor[k, f32] = a` pinned `k := 2`).
///
/// A return-only dim parameter is not fully rigid, though: the body is
/// the only place the output dimension can come from (Chelis has no
/// explicit dim application; callers instantiate dims by unification
/// against *arguments*, which never mention a return-only dim). The
/// legitimate **output-inferred** uses must stay green
/// (spec/04-type-system.md §4.4.1):
///
/// - the body leaves the dim var unbound (a clean, generalizable dim,
///   e.g. variable-fed `to_tensor`), or
/// - the body resolves it to a **body-internal** concrete dim
///   (`examples/hello_tensor.ch`: `def main() -> tensor[n, f32]` whose
///   body builds a `tensor[3, f32]`); the registered scheme then
///   resolves to the produced dim.
///
/// What is rejected is **input coupling** — the body deriving the
/// promised-independent output dim from the caller-visible parameter
/// world:
///
/// - `Dim::Var -> Dim::Lit` pin where the literal equals the
///   post-unification resolution of a dimension occurring in a declared
///   parameter position, or
/// - collapse with a distinct *param-position* declared dim parameter
///   (either binding orientation).
///
/// Two return-only dim parameters collapsing with each other are
/// tolerated (both are output-inferred; no caller-visible coupling).
/// Known residual: coupling through a named symbolic dim
/// (`def f(x: tensor[batch, f32]) -> tensor[m, f32] = x` binds
/// `m := Name("batch")`) is not flagged — `Dim::Name` unifies
/// permissively by design (chelis#219) and no declared dim parameter
/// participates.
fn check_return_only_dvars_rigid(
    decl_ty: &Type,
    param_dvars: &[DimVar],
    subst: &Subst,
    errors: &mut Vec<CheckError>,
) {
    let Type::Fn(decl_params, decl_ret) = decl_ty else {
        return;
    };
    let ret_only: Vec<DimVar> = crate::env::free_dvars(decl_ret)
        .into_iter()
        .filter(|dv| !param_dvars.contains(dv))
        .collect();
    if ret_only.is_empty() {
        return;
    }
    // Every dimension occurring in a declared parameter position,
    // resolved through the post-body substitution.
    let mut param_dims: Vec<Dim> = Vec::new();
    for t in decl_params {
        crate::env::collect_dims(t, &mut param_dims);
    }
    let resolved_param_dims: Vec<Dim> = param_dims.iter().map(|d| subst.apply_dim(d)).collect();
    for dv in &ret_only {
        let resolved = subst.apply_dim(&Dim::Var(*dv));
        if let Dim::Lit(n) = resolved {
            if resolved_param_dims.contains(&Dim::Lit(n)) {
                errors.push(CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!(
                        "return-position dim parameter d{} was pinned to concrete \
                         Lit({n}) flowing from a declared parameter dimension: a dim \
                         parameter that appears only in the return type promises an \
                         output dimension the body must not derive from the inputs \
                         (spec/04-type-system.md \u{00a7}4.4.1)",
                        dv.0
                    ),
                    vec![
                        "Name the input dimension in the return type (reuse the \
                         parameter's dim parameter or literal) if the output \
                         genuinely tracks an input dimension, or fix the body so the \
                         output dimension does not depend on the input dims"
                            .to_string(),
                    ],
                ));
            }
        } else if let Some(pdv) = param_dvars
            .iter()
            .find(|pdv| subst.apply_dim(&Dim::Var(**pdv)) == resolved)
        {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!(
                    "return-position dim parameter d{} was unified with the distinct \
                     declared dim parameter d{} from a parameter position: declared \
                     dim parameters are rigid and must remain distinct \
                     (spec/04-type-system.md \u{00a7}4.4.1)",
                    dv.0, pdv.0
                ),
                vec![
                    "Use the same dim parameter in both positions if the return \
                     dimension is meant to equal the input's, or fix the body so the \
                     declared dims stay independent"
                        .to_string(),
                ],
            ));
        }
    }
}

/// chelis#272 list-uniformity guard.
///
/// A declared return type of the form `List[tensor[..., d, ...]]` whose
/// element dim `d` is a *rigid/named* dimension (`Dim::Var` for a
/// declared dim parameter, or `Dim::Name` for a named symbolic dim)
/// promises that every list element has the *same* length at that axis.
/// The #218 Cons-join widens a *mismatched-concrete* list-element axis
/// to `Dim::Wildcard`, and `unify_dim` lets that wildcard satisfy a
/// rigid `Var`/`Name` permissively *without binding it* — so neither the
/// pin-to-literal nor the distinct-collapse arm of
/// `check_declared_dvars_rigid` observes the violation.
///
/// This check closes that gap structurally: it walks the declared type
/// and the resolved body type in parallel and flags any list-element
/// tensor axis where the declaration names a rigid/named dim but the
/// body produced a `Wildcard`. It is deliberately scoped to *list
/// element* tensors (`List[tensor[...]]`), the surface where the #272
/// soundness gap lives; it does not touch bare `tensor[...]` returns
/// whose wildcard axes legitimately flow from `expand`/`reshape`/`shape`
/// (§4.7), where the declared return's named dim binds the result tvar
/// directly rather than being absorbed by a heterogeneous-list wildcard.
///
/// chelis#276: the descent through `List` wrappers is *recursive*, so a
/// wildcard tensor nested under `List[List[tensor[k, f32]]]` (or any
/// deeper nesting) is checked against the inner rigid `k` too. `List[L]`
/// is statically homogeneous in `L`, so the uniformity promise of an
/// inner `List[tensor[k]]` holds at every depth; a single-level check
/// left the same #272 soundness gap one `List` deeper.
fn check_list_elem_rigid_dim_vs_wildcard(
    decl_ty: &Type,
    body_ty: &Type,
    errors: &mut Vec<CheckError>,
) {
    match (decl_ty, body_ty) {
        // Descend through the function type to its return position.
        (Type::Fn(_, decl_ret), Type::Fn(_, body_ret)) => {
            check_list_elem_rigid_dim_vs_wildcard(decl_ret, body_ret, errors);
        }
        // `List[T]`: check the element type. The list element is where
        // the uniformity promise lives. When the element is a tensor we
        // compare its declared vs body axes here; when it is itself a
        // `List` (or any further nesting), we recurse so the inner rigid
        // dim is protected at arbitrary depth (chelis#276).
        (Type::Adt(dn, dargs), Type::Adt(bn, bargs))
            if dn == "List" && bn == "List" && dargs.len() == 1 && bargs.len() == 1 =>
        {
            match (&dargs[0], &bargs[0]) {
                (Type::Tensor(ddims, _), Type::Tensor(bdims, _)) if ddims.len() == bdims.len() => {
                    for (dd, bd) in ddims.iter().zip(bdims.iter()) {
                        let rigid = matches!(dd, Dim::Var(_) | Dim::Name(_));
                        if rigid && matches!(bd, Dim::Wildcard) {
                            let promised = match dd {
                                Dim::Var(v) => format!("dim parameter d{}", v.0),
                                Dim::Name(n) => format!("named dimension `{n}`"),
                                _ => unreachable!(),
                            };
                            errors.push(CheckError::new(
                                CheckErrorKind::DimensionMismatch,
                                format!(
                                    "list element dimension is unknown (wildcard) in the function \
                                     body but the declared element type promises a uniform \
                                     {promised}: a heterogeneous list literal cannot satisfy a \
                                     declared List[tensor[..]] whose element dimension names a \
                                     rigid/named axis"
                                ),
                                vec![
                                    "Every element of a `List[tensor[k, ..]]` must share the same \
                                     length `k`. Either give the elements a uniform dimension, or \
                                     declare the element axis as a concrete literal / wildcard \
                                     (`tensor[*, ..]`) if the lengths genuinely differ"
                                        .to_string(),
                                ],
                            ));
                        }
                    }
                }
                // Nested `List[...]`: recurse into the element so an inner
                // rigid dim under `List[List[tensor[k]]]` is still checked
                // (chelis#276).
                (decl_elem, body_elem) => {
                    check_list_elem_rigid_dim_vs_wildcard(decl_elem, body_elem, errors);
                }
            }
        }
        _ => {}
    }
}

#[allow(clippy::too_many_arguments)]
fn infer_fn(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.is_empty() {
        return Type::Error;
    }

    // kids[0] = (params {} x1 ... xn)
    // kids[1] = body
    let params = extract_params(&kids[0], vg, adt_reg);
    let mut param_types = Vec::new();
    let mut fn_env = env.clone();

    for (pname, ty_ann) in &params {
        let ty = ty_ann.clone().unwrap_or_else(|| vg.fresh_type());
        fn_env.bind(pname.clone(), Scheme::mono(ty.clone()));
        // chelis#397/#469: a parameter is a fresh runtime binding with no
        // size provenance. Clear any entry inherited (through the derived
        // `Clone` of `env`) from an outer name it shadows, so a sourceless
        // value parameter `d` shadowing an outer shape-sourced `d` (BLOCKER C)
        // is not wrongly treated as a materializable extent.
        fn_env.clear_size_provenance(pname);
        param_types.push(ty);
    }

    // Snapshot the dimension variables introduced by the declared parameter
    // signatures. The post-body `check_declared_dvars_rigid` call flags both
    // (a) a declared dim forced to a concrete literal (Nautilus Bug 2) and
    // (b) two distinct declared dims collapsed into one another by the body
    // (Path B of TypeCheck-FreeDimVarUnification-F1). We flag this post-body
    // so legitimate polymorphic uses (where each dvar stays unbound and
    // distinct) still type-check.
    let mut declared_dvars: Vec<DimVar> = Vec::new();
    for t in &param_types {
        for dv in crate::env::free_dvars(t) {
            if !declared_dvars.contains(&dv) {
                declared_dvars.push(dv);
            }
        }
    }

    let body = if kids.len() > 1 {
        &kids[1]
    } else {
        return Type::Error;
    };
    let body_ty = infer_expr(
        body,
        &mut fn_env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );

    check_declared_dvars_rigid(&declared_dvars, subst, errors);

    let resolved_params: Vec<Type> = param_types.iter().map(|t| subst.apply(t)).collect();
    let resolved_body = subst.apply(&body_ty);

    Type::Fn(resolved_params, Box::new(resolved_body))
}

/// WS-A7: infer a `def`'s body when a declared signature is available, seeding
/// any bare-arg parameters of an outer `(fn ...)` body with the declared
/// signature's param types. This eliminates the bare-arg + sig-with-borrows
/// return-type miscompile where call-site auto-borrow on an unconstrained
/// param tvar collapses the param-side and return-side of the callee scheme
/// (e.g. `add: (&t, &t) -> t`) into the same equivalence class.
///
/// Falls back to the standard `infer_expr` path when the body is not a
/// `(fn ...)` or the declared type is not a `Fn` of matching arity. Already-
/// annotated params are not overridden.
#[allow(clippy::too_many_arguments)]
fn infer_def_body_with_sig(
    body: &deep::Expr,
    decl_ty: &Type,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    // Match: body is `(fn (params ...) body-expr)` AND decl is `Fn(args, ret)`.
    let fn_list = match body {
        deep::Expr::List(list, _) if get_tag(list) == Some("fn") => list,
        _ => {
            return infer_expr(
                body,
                env,
                vg,
                subst,
                adt_reg,
                errors,
                typed_nodes,
                total_nodes,
            );
        }
    };
    let (decl_args, decl_ret) = match decl_ty {
        Type::Fn(args, ret) => (args, ret.as_ref()),
        _ => {
            return infer_expr(
                body,
                env,
                vg,
                subst,
                adt_reg,
                errors,
                typed_nodes,
                total_nodes,
            );
        }
    };

    let kids = children(fn_list);
    if kids.len() < 2 {
        return Type::Error;
    }
    let params = extract_params(&kids[0], vg, adt_reg);
    if params.len() != decl_args.len() {
        // Arity mismatch between params and sig: fall back so the post-body
        // unify produces a clear ArityMismatch diagnostic.
        return infer_expr(
            body,
            env,
            vg,
            subst,
            adt_reg,
            errors,
            typed_nodes,
            total_nodes,
        );
    }

    let mut param_types = Vec::with_capacity(params.len());
    let mut fn_env = env.clone();
    for ((pname, ty_ann), decl_arg) in params.iter().zip(decl_args.iter()) {
        // Annotated params keep their annotation; bare params get seeded with
        // the declared sig type. The post-body unify still validates each
        // path in the standard way.
        let ty = ty_ann.clone().unwrap_or_else(|| decl_arg.clone());
        fn_env.bind(pname.clone(), Scheme::mono(ty.clone()));
        // chelis#397/#469: a fresh parameter has no size provenance; clear any
        // entry inherited from an outer name it shadows (BLOCKER C).
        fn_env.clear_size_provenance(pname);
        param_types.push(ty);
    }

    // Note: the declared-dim rigidity check (both the Var->Lit pin and
    // the Var->Var collapse of TypeCheck-FreeDimVarUnification-F1) runs
    // at the caller's defsig site, *after* the post-body sig-unify. The
    // sig-unify is where two distinct declared dims actually collapse
    // for an annotated-param body like
    // `def g[n, m](x: tensor[n, f32], y: tensor[m, f32]) ->
    // tensor[n, f32] = y`, so checking here (pre-sig-unify) would miss
    // it. Running it only at the caller also avoids double-reporting.
    let body_expr = &kids[1];
    let body_ty = infer_expr(
        body_expr,
        &mut fn_env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );

    let resolved_params: Vec<Type> = param_types.iter().map(|t| subst.apply(t)).collect();
    let resolved_body = subst.apply(&body_ty);
    let _ = decl_ret; // referenced for documentation; sig-unify happens at the call site

    Type::Fn(resolved_params, Box::new(resolved_body))
}

/// Extract parameter names (and optional type annotations) from (params {} x1 ... xn).
/// Each param can be a bare symbol, a metadata-annotated symbol, or a legacy
/// `(name {type: T})` helper pair.
fn extract_params(
    expr: &deep::Expr,
    vg: &mut VarGen,
    adt_reg: &AdtRegistry,
) -> Vec<(String, Option<Type>)> {
    match expr {
        deep::Expr::List(list, _) => {
            let tag = get_tag(list);
            let elems = if tag == Some("params") {
                children(list)
            } else {
                &list.elements
            };
            elems
                .iter()
                .filter_map(|e| match e {
                    deep::Expr::Atom(deep::Atom::Symbol(s), _) => Some((s.to_string(), None)),
                    deep::Expr::MetaExpr(meta, _) => {
                        let deep::Expr::Atom(deep::Atom::Symbol(name), _) = meta.expr.as_ref()
                        else {
                            return None;
                        };
                        let ty_ann = meta.entries.iter().find_map(|(key, val)| {
                            (key == "type").then(|| {
                                deep_type_to_resolved_type(val, vg, adt_reg, &mut HashMap::new())
                            })
                        });
                        Some((name.to_string(), ty_ann))
                    }
                    deep::Expr::List(plist, _) => {
                        // Typed param: (name {type: T}) — elements[0] is the name symbol,
                        // elements[1] is the metadata map with type annotation
                        if let Some(deep::Expr::Atom(deep::Atom::Symbol(name), _)) =
                            plist.elements.first()
                        {
                            let mut ty_ann = None;
                            if let Some(deep::Expr::Map(meta, _)) = plist.elements.get(1) {
                                for (key, val) in &meta.entries {
                                    if key == "type" {
                                        ty_ann = Some(deep_type_to_resolved_type(
                                            val,
                                            vg,
                                            adt_reg,
                                            &mut HashMap::new(),
                                        ));
                                    }
                                }
                            }
                            Some((name.to_string(), ty_ann))
                        } else {
                            None
                        }
                    }
                    _ => None,
                })
                .collect()
        }
        _ => vec![],
    }
}

#[allow(clippy::too_many_arguments)]
fn infer_let(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.len() < 2 {
        return Type::Error;
    }

    // kids[0] = (bind {} x1 e1 x2 e2 ...)
    // kids[1] = body
    let mut let_env = env.clone();

    if let deep::Expr::List(bind_list, _) = &kids[0] {
        let bind_children = children(bind_list);
        // Process pairs: name, expr
        let mut i = 0;
        while i + 1 < bind_children.len() {
            if let Some(name) = symbol_name(&bind_children[i]) {
                let rhs_expr = &bind_children[i + 1];
                let expr_ty = infer_expr(
                    rhs_expr,
                    &mut let_env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                );

                // chelis#159: block-scoped `let name: T = expr` desugars
                // inject the declared type `T` as a `"type"` metadata
                // entry on the RHS node (crates/chelis-surf/src/desugar.rs:1374-1379
                // via `inject_type_metadata`). Pre-fix, infer_let did
                // not consult that metadata, so a generic-builtin RHS
                // like `to_tensor(...)` left its output type var free
                // and the ascription was silently dropped. Unify the
                // inferred RHS type against the declared type so the
                // ascription propagates into downstream sig calls.
                let final_ty = if let deep::Expr::List(rhs_list, _) = rhs_expr
                    && let Some(meta) = get_meta(rhs_list)
                    && let Some(declared_ty_expr) = meta
                        .entries
                        .iter()
                        .find(|(k, _)| k == "type")
                        .map(|(_, v)| v)
                {
                    let declared_ty = deep_type_to_resolved_type(
                        declared_ty_expr,
                        vg,
                        adt_reg,
                        &mut HashMap::new(),
                    );
                    if let Err(e) = unify(&expr_ty, &declared_ty, subst) {
                        errors.push(CheckError::new(
                            check_error_kind_from_type_error_kind(&e.kind),
                            format!(
                                "let-binding `{name}` ascription does not match RHS: {}",
                                e.message
                            ),
                            vec![format!(
                                "Declared type for `{name}` is {declared_ty}; \
                                 RHS inferred to {expr_ty}"
                            )],
                        ));
                    }
                    // On unify failure, bind `name` to the declared
                    // type rather than the inferred RHS type. This
                    // produces a cleaner error cascade: downstream uses
                    // of `name` see what the user said they meant, not
                    // what the (already-rejected) RHS inferred to, so
                    // a single ascription-mismatch diagnostic stands
                    // alone instead of fanning out into multiple
                    // downstream errors. The trade-off: pathological
                    // bodies where the user's ascription is *also*
                    // independently wrong against later code may have
                    // a second mismatch masked. The single-error
                    // cascade is the better default for chelis#159's
                    // user-facing diagnostic ergonomics.
                    declared_ty
                } else {
                    expr_ty
                };

                let scheme = let_env.generalize(&final_ty, subst);
                // chelis#397/#469: record the size provenance of this binding
                // BEFORE binding it (so `classify_expand_size` resolves it
                // against the binding's RHS, not its own name) so a later
                // `expand(b, 0, name)` can recover whether `name` is a
                // materializable extent (static / shape-sourced) or a
                // sourceless runtime scalar. Bound BEFORE `let_env.bind` so
                // the RHS is classified against the pre-binding scope, and
                // transitively through earlier bindings in the same block.
                // The `Sourceless`/`Unknown` arm CLEARS any stale provenance so
                // a re-bind to a sourceless RHS — `len = shape(x, 0); len = k`
                // (BLOCKER B) — does not inherit the earlier shape-sourced entry.
                match classify_expand_size(rhs_expr, &let_env) {
                    SizeClass::Static => {
                        let_env.mark_size_provenance(name, crate::env::SizeProvenance::Static);
                    }
                    SizeClass::ShapeSourced => {
                        let_env
                            .mark_size_provenance(name, crate::env::SizeProvenance::ShapeSourced);
                    }
                    SizeClass::Sourceless | SizeClass::Unknown => {
                        let_env.clear_size_provenance(name)
                    }
                }
                let_env.bind(name.to_string(), scheme);
            }
            i += 2;
        }
    }

    infer_expr(
        &kids[1],
        &mut let_env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    )
}

#[allow(clippy::too_many_arguments)]
fn infer_if(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.len() < 3 {
        return Type::Error;
    }

    let cond_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );

    // Condition should be bool (or tensor[D, bool])
    if let Err(_te) = unify(&cond_ty, &Type::Prim(Prim::Bool), subst) {
        errors.push(CheckError::new(
            CheckErrorKind::TypeMismatch,
            format!("if condition must be bool, got {}", subst.apply(&cond_ty)),
            vec![],
        ));
    }

    let then_ty = infer_expr(
        &kids[1],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let else_ty = infer_expr(
        &kids[2],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );

    match unify(&then_ty, &else_ty, subst) {
        Ok(()) => subst.apply(&then_ty),
        Err(te) => {
            let mut e: CheckError = te.into();
            if let Some(id) = list_span_id(list) {
                e.span_offset = parse_span_offset(id);
                e.span_id = Some(id.to_string());
            } else {
                let off = span_of_list(list).offset;
                if off > 0 {
                    e.span_offset = Some(off);
                }
            }
            errors.push(e);
            subst.apply(&then_ty)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn infer_match(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.is_empty() {
        return Type::Error;
    }

    let scrutinee_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );

    let mut result_ty: Option<Type> = None;
    let mut covered_variants: Vec<String> = Vec::new();
    let mut has_wildcard = false;

    for arm_expr in &kids[1..] {
        if let deep::Expr::List(arm_list, _) = arm_expr
            && get_tag(arm_list) == Some("arm")
        {
            let arm_kids = children(arm_list);
            // arm_kids[0] = pattern, arm_kids[1] = guard (usually ()), arm_kids[2] = body
            if arm_kids.len() >= 3 {
                let mut arm_env = env.clone();
                let pat = &arm_kids[0];
                // RFC D-CHECK exhaustiveness fix (RT-0 verified false
                // positives): a TOP-LEVEL irrefutable arm covers the
                // match -- a bare `pat-var`, or a `pat-as` whose
                // inner pattern is irrefutable. Nested `pat-var`
                // keeps not-covering so exhaustiveness is not
                // weakened on ordinary ADTs.
                if top_level_arm_is_irrefutable(pat) {
                    has_wildcard = true;
                }
                pattern_bindings(
                    pat,
                    &scrutinee_ty,
                    &mut arm_env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    &mut covered_variants,
                    &mut has_wildcard,
                );

                let body_ty = infer_expr(
                    &arm_kids[2],
                    &mut arm_env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                );

                match &result_ty {
                    None => result_ty = Some(body_ty),
                    Some(prev) => {
                        if let Err(te) = unify(prev, &body_ty, subst) {
                            errors.push(te.into());
                        }
                        result_ty = Some(subst.apply(prev));
                    }
                }
            }
        }
    }

    // Exhaustiveness check (wildcard covers everything)
    if !has_wildcard {
        let resolved_scrutinee = subst.apply(&scrutinee_ty);
        if let Type::Adt(ref adt_name, _) = resolved_scrutinee
            && let Some(all_variants) = adt_reg.variant_names(adt_name)
        {
            let missing: Vec<&String> = all_variants
                .iter()
                .filter(|v| !covered_variants.contains(v))
                .collect();
            if !missing.is_empty() {
                let names: Vec<&str> = missing.iter().map(|s| s.as_str()).collect();
                errors.push(CheckError::new(
                    CheckErrorKind::NonExhaustiveMatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!("non-exhaustive match: missing variants {:?}", names),
                    ),
                    vec![],
                ));
            }
        }
    }

    result_ty.unwrap_or(Type::Error)
}

/// True for arm patterns that match every value of the scrutinee:
/// `pat-var`, `pat-wild`, and `pat-as` wrapping an irrefutable inner
/// pattern (`q @ x`). Applies at the ARM level only.
fn top_level_arm_is_irrefutable(pat: &deep::Expr) -> bool {
    let deep::Expr::List(list, _) = pat else {
        return false;
    };
    match get_tag(list) {
        Some("pat-var") | Some("pat-wild") => true,
        Some("pat-as") => children(list)
            .get(1)
            .is_some_and(top_level_arm_is_irrefutable),
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
fn pattern_bindings(
    pat: &deep::Expr,
    scrutinee_ty: &Type,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    covered_variants: &mut Vec<String>,
    has_wildcard: &mut bool,
) {
    stack_guard!("pattern_bindings", pat);
    if let deep::Expr::List(list, _) = pat {
        let tag = get_tag(list).unwrap_or("");
        let kids = children(list);
        match tag {
            "pat-var" => {
                if let Some(name) = kids.first().and_then(|e| symbol_name(e)) {
                    let resolved = subst.apply(scrutinee_ty);
                    env.bind(name.to_string(), Scheme::mono(resolved));
                }
            }
            "pat-wild" => {
                // Wildcard covers everything
                *has_wildcard = true;
            }
            "pat-lit" => {
                // No bindings, but value should match scrutinee type
            }
            "pat-ctor" => {
                if let Some(ctor_name) = kids.first().and_then(|e| symbol_name(e)) {
                    // chelis#317: an out-of-scope constructor pattern (a type-
                    // only import that names `| Alpha =>` without importing
                    // `Alpha`) must be rejected at `check` here, the same way
                    // the construction site is in `infer_var`. Without this the
                    // fuzzy terminal fallback below binds the arm to a foreign
                    // module's tag and the mismatch surfaces only as a runtime
                    // non-exhaustive match. `constructor_pattern_out_of_scope`
                    // rejects both the unique-fuzzy case and the non-unique /
                    // unresolvable case; the latter would otherwise push a bare
                    // name with no scheme into `covered_variants` and be silently
                    // accepted under a `_` wildcard arm. Skip coverage/binding so
                    // the bogus arm cannot also mask the real `non-exhaustive`
                    // diagnostic.
                    if constructor_pattern_out_of_scope(ctor_name, env, adt_reg) {
                        errors.push(CheckError::new(
                            CheckErrorKind::UnknownConstructor,
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
                                format!("unknown constructor: {ctor_name}"),
                            ),
                            vec![format!(
                                "Constructor '{ctor_name}' is not in scope. Declare it \
                                 locally or add it to an import (e.g. \
                                 `import Mod ({ctor_name})`)"
                            )],
                        ));
                        return;
                    }
                    // Record the *resolved* variant name for exhaustiveness,
                    // not the bare pattern name. After reef's module-scoped
                    // constructor mangling (chelis#157), the registry keys
                    // variants by their package/module-qualified name, while
                    // a pattern may still be written with the bare terminal
                    // name (e.g. an unqualified `JsonNull` arm). Pushing the
                    // bare name would leave the mangled variant uncovered and
                    // fire a spurious `non-exhaustive match`. Resolve through
                    // the registry's terminal-unique lookup so coverage is
                    // compared on the same (mangled) key `variant_names`
                    // returns.
                    let covered_name = adt_reg
                        .lookup_variant(ctor_name)
                        .or_else(|| adt_reg.lookup_variant_terminal_unique(ctor_name))
                        .map(|(_, variant)| variant.name.clone())
                        .unwrap_or_else(|| ctor_name.to_string());
                    covered_variants.push(covered_name);

                    // RFC D-CHECK: constructor pattern match on an
                    // out-of-module opaque type is rejected; binding
                    // inference continues so no error cascades.
                    if let Some((adt_name, _)) = adt_reg
                        .lookup_variant(ctor_name)
                        .or_else(|| adt_reg.lookup_variant_terminal_unique(ctor_name))
                    {
                        let adt_name = adt_name.to_string();
                        crate::opacity::check_opaque_use(
                            crate::opacity::OpaqueAction::PatCtor,
                            &adt_name,
                            adt_reg,
                            errors,
                        );
                    }

                    // Look up constructor in env and decompose
                    if let Some(scheme) = env
                        .lookup(ctor_name)
                        .or_else(|| env.lookup_terminal_unique(ctor_name))
                    {
                        let scheme = scheme.clone();
                        let ctor_ty = env.instantiate(&scheme, vg);
                        // Unify the result of the constructor with scrutinee type
                        match &ctor_ty {
                            Type::Fn(arg_types, ret) => {
                                let _ = unify(ret, scrutinee_ty, subst);
                                // Bind sub-patterns to argument types
                                for (i, sub_pat) in kids[1..].iter().enumerate() {
                                    if i < arg_types.len() {
                                        let resolved = subst.apply(&arg_types[i]);
                                        pattern_bindings(
                                            sub_pat,
                                            &resolved,
                                            env,
                                            vg,
                                            subst,
                                            adt_reg,
                                            errors,
                                            covered_variants,
                                            has_wildcard,
                                        );
                                    }
                                }
                            }
                            _ => {
                                // Nullary constructor
                                let _ = unify(&ctor_ty, scrutinee_ty, subst);
                            }
                        }
                    }
                }
            }
            "pat-as" => {
                // (pat-as {} name inner_pat): bind name to scrutinee type, recurse into inner_pat
                if let Some(name) = kids.first().and_then(|e| symbol_name(e)) {
                    let resolved = subst.apply(scrutinee_ty);
                    env.bind(name.to_string(), Scheme::mono(resolved));
                }
                if kids.len() >= 2 {
                    pattern_bindings(
                        &kids[1],
                        scrutinee_ty,
                        env,
                        vg,
                        subst,
                        adt_reg,
                        errors,
                        covered_variants,
                        has_wildcard,
                    );
                }
            }
            "pat-record" => {
                // (pat-record {} TypeName (kv {} k1 p1) ...): validate against ADT registry
                // kids[0] = TypeName, kids[1..] = (kv {} key pat)
                if let Some(ctor_name) = kids.first().and_then(|e| symbol_name(e)) {
                    // chelis#317: same out-of-scope guard as the positional
                    // `pat-ctor` arm — a record-shaped match against a
                    // constructor that was never imported must be an `unknown
                    // constructor` error, not a fuzzy bind to a foreign tag.
                    // `constructor_pattern_out_of_scope` also rejects the
                    // non-unique / unresolvable case a `_` wildcard arm would
                    // otherwise silently accept.
                    if constructor_pattern_out_of_scope(ctor_name, env, adt_reg) {
                        errors.push(CheckError::new(
                            CheckErrorKind::UnknownConstructor,
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
                                format!("unknown constructor: {ctor_name}"),
                            ),
                            vec![format!(
                                "Constructor '{ctor_name}' is not in scope. Declare it \
                                 locally or add it to an import (e.g. \
                                 `import Mod ({ctor_name})`)"
                            )],
                        ));
                        return;
                    }
                    // Look up variant in ADT registry for the canonical
                    // field order and known field-name set used for
                    // validation diagnostics.
                    let variant_info = adt_reg
                        .lookup_variant(ctor_name)
                        .or_else(|| adt_reg.lookup_variant_terminal_unique(ctor_name));
                    // RFC D-CHECK: record pattern match on an
                    // out-of-module opaque type is rejected; binding
                    // inference continues so no error cascades.
                    if let Some((adt_name, _)) = variant_info {
                        let adt_name = adt_name.to_string();
                        crate::opacity::check_opaque_use(
                            crate::opacity::OpaqueAction::PatRecord,
                            &adt_name,
                            adt_reg,
                            errors,
                        );
                    }

                    // Record the resolved (mangled) variant name for
                    // exhaustiveness, mirroring `pat-ctor`. See that arm for
                    // why the bare pattern name is not used (chelis#157).
                    let covered_name = variant_info
                        .map(|(_, vi)| vi.name.clone())
                        .unwrap_or_else(|| ctor_name.to_string());
                    covered_variants.push(covered_name);
                    let declared_field_names: Vec<Option<String>> = variant_info
                        .map(|(_, vi)| vi.fields.iter().map(|(n, _)| n.clone()).collect())
                        .unwrap_or_default();
                    let known_field_set: std::collections::HashSet<&str> = declared_field_names
                        .iter()
                        .filter_map(|n| n.as_deref())
                        .collect();

                    // Mirror the `pat-ctor` (positional) path: instantiate
                    // the constructor scheme and unify its return type
                    // with the scrutinee so the ADT's type parameters get
                    // pinned to the scrutinee's concrete instantiation
                    // (e.g. `FooState[a] -> FooState[tensor[n, f32]]`).
                    // The instantiated function's arg types are the
                    // properly substituted per-field types. Without this
                    // step, the declared field types still reference the
                    // ADT's abstract `a`, leaving record-pattern bindings
                    // stuck as fresh type variables and breaking
                    // downstream linearity/borrow checks. (closes #181)
                    let instantiated_arg_types: Vec<Type> = if let Some(scheme) = env
                        .lookup(ctor_name)
                        .or_else(|| env.lookup_terminal_unique(ctor_name))
                    {
                        let scheme = scheme.clone();
                        let ctor_ty = env.instantiate(&scheme, vg);
                        match ctor_ty {
                            Type::Fn(arg_types, ret) => {
                                let _ = unify(&ret, scrutinee_ty, subst);
                                arg_types
                            }
                            // Nullary constructor: the scheme body is the
                            // ADT type itself, no Fn-wrapping. Still unify
                            // with the scrutinee so the ADT's type
                            // parameters are pinned to its concrete
                            // instantiation, mirroring `pat-ctor`'s
                            // positional path. There are no fields to
                            // bind for `Foo {}`, so the empty
                            // `instantiated_arg_types` is the right
                            // return value either way; the unify is the
                            // side-effect that matters.
                            other => {
                                let _ = unify(&other, scrutinee_ty, subst);
                                Vec::new()
                            }
                        }
                    } else {
                        Vec::new()
                    };

                    for kv_expr in kids.iter().skip(1) {
                        if let deep::Expr::List(kv_list, _) = kv_expr
                            && get_tag(kv_list) == Some("kv")
                        {
                            let kv_kids = children(kv_list);
                            if kv_kids.len() >= 2 {
                                let field_name = symbol_name(&kv_kids[0]);
                                // Look up declared field type — reject unknown fields
                                let field_ty = match field_name {
                                    Some(n) => {
                                        if known_field_set.contains(n) {
                                            // Prefer the instantiated arg type from
                                            // the constructor scheme so the
                                            // scrutinee's concrete type arguments
                                            // are reflected in the pattern binding.
                                            let pos = declared_field_names
                                                .iter()
                                                .position(|nm| nm.as_deref() == Some(n));
                                            match pos.and_then(|i| instantiated_arg_types.get(i)) {
                                                Some(ty) => subst.apply(ty),
                                                None => {
                                                    // Fallback: un-instantiated declared field
                                                    // type when the constructor scheme isn't
                                                    // in `env`. This branch SHOULD be
                                                    // unreachable in practice: every `deftype`
                                                    // registered in `adt_reg` via
                                                    // `collect_declarations` also binds its
                                                    // constructor scheme in `env` in the same
                                                    // call. If that invariant drifts (e.g., a
                                                    // future code path populates `adt_reg`
                                                    // without binding into `env`), the
                                                    // fallback would silently produce
                                                    // `Var(T_a)` from the un-instantiated
                                                    // VariantInfo — exactly the bug #181 fixed.
                                                    // The debug_assert below flags the drift
                                                    // in tests; the runtime fallback to
                                                    // `vi.fields[i]` preserves pre-fix
                                                    // behavior in release builds.
                                                    debug_assert!(
                                                        false,
                                                        "env/adt_reg sync invariant violated: \
                                                         field `{n}` of constructor `{ctor_name}` \
                                                         is known to `adt_reg` (variant_info found) \
                                                         but the constructor scheme is missing from \
                                                         `env`. See infer.rs pat-record fallback note."
                                                    );
                                                    variant_info
                                                        .and_then(|(_, vi)| {
                                                            vi.fields.iter().find_map(
                                                                |(name, ty)| {
                                                                    (name.as_deref() == Some(n))
                                                                        .then(|| ty.clone())
                                                                },
                                                            )
                                                        })
                                                        // Per the loop guard `known_field_set
                                                        // .contains(n)` and the fact that
                                                        // `known_field_set` is derived from
                                                        // `declared_field_names` whose
                                                        // `Some(_)` entries are exactly the
                                                        // named fields of `vi.fields`, the
                                                        // find_map above always returns Some
                                                        // here. The expect makes that explicit;
                                                        // if it ever fires, both data sources
                                                        // are themselves out of sync — a bug
                                                        // upstream of this site.
                                                        .expect(
                                                            "known_field_set is derived from \
                                                             vi.fields' named entries; mismatch \
                                                             indicates a corrupted AdtRegistry",
                                                        )
                                                }
                                            }
                                        } else if !known_field_set.is_empty() {
                                            // Unknown field name — error
                                            errors.push(CheckError::new(
                                                CheckErrorKind::TypeMismatch,
                                                format!(
                                                    "unknown record field '{}' in pattern for {}",
                                                    n, ctor_name
                                                ),
                                                vec![format!(
                                                    "known fields: {:?}",
                                                    declared_field_names
                                                        .iter()
                                                        .filter_map(|f| f.as_deref())
                                                        .collect::<Vec<_>>()
                                                )],
                                            ));
                                            Type::Error
                                        } else {
                                            vg.fresh_type() // no ADT info available
                                        }
                                    }
                                    None => vg.fresh_type(),
                                };
                                pattern_bindings(
                                    &kv_kids[1],
                                    &field_ty,
                                    env,
                                    vg,
                                    subst,
                                    adt_reg,
                                    errors,
                                    covered_variants,
                                    has_wildcard,
                                );
                            }
                        }
                    }
                }
            }
            "pat-tuple" => {
                // (pat-tuple {} sub0 sub1 ...): every child is itself a
                // sub-pattern. Recurse into each so a nested
                // `pat-record` / `pat-ctor` reaches the RFC D-CHECK
                // opacity gate (and `pat-var` bindings get the right
                // element type) exactly as a top-level destructure does.
                // Without this recursion the catch-all below silently
                // dropped tuple-nested patterns, bypassing
                // `check_opaque_use` for out-of-module opaque types wrapped
                // in a tuple scrutinee.
                let resolved = subst.apply(scrutinee_ty);
                // Pair each child sub-pattern with the matching tuple
                // element type when the resolved scrutinee is a tuple of
                // equal arity; otherwise hand each child a fresh type
                // variable. The opacity check inside the nested
                // `pat-record` / `pat-ctor` arms keys off the pattern's
                // constructor name, not the scrutinee type, so the gate
                // still fires under a fresh-var element type.
                let elem_tys: Option<&[Type]> = match &resolved {
                    Type::Tuple(ts) if ts.len() == kids.len() => Some(ts.as_slice()),
                    _ => None,
                };
                for (i, sub_pat) in kids.iter().enumerate() {
                    let elem_ty = match elem_tys {
                        Some(ts) => subst.apply(&ts[i]),
                        None => vg.fresh_type(),
                    };
                    pattern_bindings(
                        sub_pat,
                        &elem_ty,
                        env,
                        vg,
                        subst,
                        adt_reg,
                        errors,
                        covered_variants,
                        has_wildcard,
                    );
                }
            }
            _ => {}
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn infer_pipe(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.is_empty() {
        return Type::Error;
    }

    let mut current_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );

    for stage in &kids[1..] {
        // If the stage is the canonical bare-keyword / `cast(type)` pipe-stage
        // shape `(fn (params <single unannotated param>) body)` produced by
        // `crates/chelis-surf/src/parser.rs::parse_pipe_stage` and
        // `desugar_pipe_stage`, infer the lambda with its parameter bound to
        // the upstream pipe value's type. Without this pre-binding, per-builtin
        // inference gates inside the body (e.g. `infer_copy`, `infer_cast`)
        // see a fresh type variable for the parameter and reject before the
        // pipe loop's unification can bind it to `current_ty`. See
        // `docs/investigations/pipe_copy_typecheck_diagnosis.md` for the trace.
        let stage_ty = if let Some(param_name) = synthesized_unary_lambda_param(stage, adt_reg, vg)
        {
            infer_pipe_stage_lambda(
                stage,
                &param_name,
                current_ty.clone(),
                env,
                vg,
                subst,
                adt_reg,
                errors,
                typed_nodes,
                total_nodes,
            )
        } else {
            infer_expr(
                stage,
                env,
                vg,
                subst,
                adt_reg,
                errors,
                typed_nodes,
                total_nodes,
            )
        };
        let ret_tv = vg.fresh_type();
        let stage_arg_tys = auto_borrow_call_arg_types(&stage_ty, vec![current_ty.clone()], subst);
        let expected = Type::Fn(stage_arg_tys, Box::new(ret_tv.clone()));

        match unify(&stage_ty, &expected, subst) {
            Ok(()) => {
                current_ty = subst.apply(&ret_tv);
            }
            Err(te) => {
                let mut e: CheckError = te.into();
                if let Some(id) = stage.span_id() {
                    e.span_offset = parse_span_offset(id);
                    e.span_id = Some(id.to_string());
                } else {
                    let off = stage.span().offset;
                    if off > 0 {
                        e.span_offset = Some(off);
                    }
                }
                errors.push(e);
                return Type::Error;
            }
        }
    }

    current_ty
}

/// If `stage` is a `(fn (params x) body)` Deep node with exactly one
/// unannotated parameter -- the canonical shape produced by the Surf
/// parser's `parse_pipe_stage` and `desugar_pipe_stage` for bare
/// unary-builtin keyword stages (`x |> copy`, `x |> realize`) and the
/// one-arg `cast(type)` form (`x |> cast(f32)`) -- return the
/// parameter's name. Otherwise return `None`.
///
/// Multi-arg lambdas, lambdas with annotated parameters, and any other
/// pipe-stage form (named reference, partial application, etc.) fall
/// through unchanged.
fn synthesized_unary_lambda_param(
    stage: &deep::Expr,
    _adt_reg: &AdtRegistry,
    _vg: &mut VarGen,
) -> Option<String> {
    let deep::Expr::List(list, _) = stage else {
        return None;
    };
    if get_tag(list) != Some("fn") {
        return None;
    }
    let kids = children(list);
    let params_expr = kids.first()?;
    let deep::Expr::List(params_list, _) = params_expr else {
        return None;
    };
    if get_tag(params_list) != Some("params") {
        return None;
    }
    let param_kids = children(params_list);
    if param_kids.len() != 1 {
        return None;
    }
    // Single param must be a bare symbol; an annotated form would
    // surface as `MetaExpr` or a nested `List`, and the user-written
    // annotation takes precedence over the upstream pipe value's type.
    match &param_kids[0] {
        deep::Expr::Atom(deep::Atom::Symbol(name), _) => Some(name.to_string()),
        _ => None,
    }
}

/// Infer a synthesized unary pipe-stage lambda with its parameter
/// pre-bound to `param_ty`. Mirrors `infer_fn` but seeds the
/// parameter's scheme from `param_ty` instead of allocating a fresh
/// type variable, so per-builtin inference gates inside the body see
/// the upstream pipe value's type.
#[allow(clippy::too_many_arguments)]
fn infer_pipe_stage_lambda(
    stage: &deep::Expr,
    param_name: &str,
    param_ty: Type,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let deep::Expr::List(list, _) = stage else {
        return Type::Error;
    };
    let kids = children(list);
    let body = match kids.get(1) {
        Some(body) => body,
        None => return Type::Error,
    };

    let mut fn_env = env.clone();
    fn_env.bind(param_name.to_string(), Scheme::mono(param_ty.clone()));
    // chelis#397/#469: a fresh parameter has no size provenance; clear any
    // entry inherited from an outer name it shadows (BLOCKER C).
    fn_env.clear_size_provenance(param_name);

    let body_ty = infer_expr(
        body,
        &mut fn_env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );

    let resolved_param = subst.apply(&param_ty);
    let resolved_body = subst.apply(&body_ty);
    Type::Fn(vec![resolved_param], Box::new(resolved_body))
}

#[allow(clippy::too_many_arguments)]
fn infer_tuple(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    let elems: Vec<Type> = kids
        .iter()
        .map(|e| infer_expr(e, env, vg, subst, adt_reg, errors, typed_nodes, total_nodes))
        .collect();
    Type::Tuple(elems)
}

#[allow(clippy::too_many_arguments)]
fn infer_tuple_get(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.len() < 2 {
        return Type::Error;
    }

    let tuple_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let resolved = subst.apply(&tuple_ty);

    let index = match &kids[1] {
        deep::Expr::Atom(deep::Atom::Int(n), _) => *n as usize,
        _ => {
            return Type::Error;
        }
    };

    match resolved {
        Type::Tuple(ref elems) => {
            if index < elems.len() {
                elems[index].clone()
            } else {
                errors.push(CheckError::new(
                    CheckErrorKind::TupleIndexOutOfBounds,
                    format!(
                        "tuple index {} out of bounds for tuple of size {}",
                        index,
                        elems.len()
                    ),
                    vec![],
                ));
                Type::Error
            }
        }
        Type::Error => Type::Error,
        _ => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("expected tuple type, got {resolved}"),
                vec![],
            ));
            Type::Error
        }
    }
}

/// Resolve a record head name to its constructor. The head is a
/// variant name (`Probability { ... }`); when it names a transparent
/// type alias instead, resolve through the alias to the nominal ADT
/// and use that ADT's same-named variant (alias transparency,
/// spec/02; this is also what keeps alias laundering from bypassing
/// opacity, RFC D-CHECK). Returns the canonical constructor name.
fn resolve_record_head<'a>(
    head: &'a str,
    adt_reg: &'a AdtRegistry,
) -> Option<(&'a str, &'a crate::adt::VariantInfo, String)> {
    if let Some((adt_name, variant)) = adt_reg
        .lookup_variant_preferring_shape(head, CallShape::Record)
        .or_else(|| adt_reg.lookup_variant_terminal_unique(head))
    {
        return Some((adt_name, variant, variant.name.clone()));
    }
    // Alias head: `type P2 = Probability` makes `P2 { ... }` mean
    // `Probability { ... }`.
    let alias = adt_reg.resolve_alias(head)?;
    if let Type::Adt(target, _) = &alias.body {
        let (adt_name, variant) = adt_reg.lookup_variant(target)?;
        return Some((adt_name, variant, variant.name.clone()));
    }
    None
}

/// Infer `(record {} Ctor (kv {} field value)...)` — named-field
/// record construction (RFC D-CHECK prerequisite inference; closes
/// the latent bogus-field hole: unknown fields are now TypeMismatch
/// errors instead of silently untyped).
#[allow(clippy::too_many_arguments)]
fn infer_record(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    let Some(head) = kids.first().and_then(symbol_name) else {
        return Type::Error;
    };

    let Some((adt_name, variant, ctor_name)) = resolve_record_head(head, adt_reg) else {
        // Infer field values so nested errors still surface, then
        // reject the unknown constructor.
        for kv_expr in kids.iter().skip(1) {
            if let deep::Expr::List(kv_list, _) = kv_expr
                && get_tag(kv_list) == Some("kv")
                && let Some(value) = children(kv_list).get(1)
            {
                infer_expr(
                    value,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                );
            }
        }
        errors.push(CheckError::new(
            CheckErrorKind::TypeMismatch,
            format!("unknown record constructor `{head}`"),
            vec![format!("declare `type {head} = | {head} {{ ... }}`")],
        ));
        return Type::Error;
    };
    // chelis#317: a registry-known but OUT-OF-SCOPE record constructor (a
    // type-only import constructing e.g. `AdamState { ... }`, whose ADT is in
    // the registry but whose constructor is not in scope at the use site) is
    // unknown here. Opaque types are handled by the opacity check below
    // instead, so only a non-opaque out-of-scope head is rejected here; this
    // keeps the #317 record-constructor guard while leaving opacity rejection
    // (D-CHECK) for opaque heads.
    let head_is_opaque = adt_reg.lookup(adt_name).is_some_and(|d| d.opaque);
    if !head_is_opaque && constructor_out_of_scope(head, env) {
        for kv_expr in kids.iter().skip(1) {
            if let deep::Expr::List(kv_list, _) = kv_expr
                && get_tag(kv_list) == Some("kv")
                && let Some(value) = children(kv_list).get(1)
            {
                infer_expr(
                    value,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                );
            }
        }
        errors.push(CheckError::new(
            CheckErrorKind::UnknownConstructor,
            with_macro_provenance(
                &deep::Expr::List(list.clone(), zero_span()),
                format!("unknown constructor: {head}"),
            ),
            vec![format!(
                "Constructor '{head}' is not in scope. Declare it locally or \
                 add it to an import (e.g. `import Mod ({head})`)"
            )],
        ));
        return Type::Error;
    }
    // RFC D-CHECK: record construction of an out-of-module opaque
    // type is rejected; inference continues so the literal still
    // yields its true type (no cascades).
    crate::opacity::check_opaque_use(
        crate::opacity::OpaqueAction::RecordConstruction,
        adt_name,
        adt_reg,
        errors,
    );

    let declared_field_names: Vec<Option<String>> =
        variant.fields.iter().map(|(n, _)| n.clone()).collect();
    let known_field_set: HashSet<&str> = declared_field_names
        .iter()
        .filter_map(|n| n.as_deref())
        .collect();

    // Instantiate the resolved ADT's constructor directly from its
    // registry definition (the issue #181 pat-record intent, made
    // collision-proof): the name-keyed env holds ONE scheme per
    // constructor name, so same-named constructors from colliding
    // ADTs (chelis#148) would dispatch the field types to whichever
    // deftype registered last.
    let (instantiated_arg_types, instantiated_ret) = match adt_reg.lookup(adt_name) {
        Some(adt_def) => instantiate_variant_of(adt_def, variant, vg),
        None => (Vec::new(), Type::Error),
    };

    for kv_expr in kids.iter().skip(1) {
        let deep::Expr::List(kv_list, _) = kv_expr else {
            continue;
        };
        if get_tag(kv_list) != Some("kv") {
            continue;
        }
        let kv_kids = children(kv_list);
        let (Some(field_name), Some(value)) =
            (kv_kids.first().and_then(symbol_name), kv_kids.get(1))
        else {
            continue;
        };
        let value_ty = infer_expr(
            value,
            env,
            vg,
            subst,
            adt_reg,
            errors,
            typed_nodes,
            total_nodes,
        );
        if known_field_set.contains(field_name) {
            let pos = declared_field_names
                .iter()
                .position(|n| n.as_deref() == Some(field_name));
            if let Some(field_ty) = pos.and_then(|i| instantiated_arg_types.get(i))
                && let Err(te) = unify(&value_ty, field_ty, subst)
            {
                errors.push(te.into());
            }
        } else {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("unknown record field '{field_name}' in construction of {ctor_name}"),
                vec![format!(
                    "known fields: {:?}",
                    declared_field_names
                        .iter()
                        .filter_map(|f| f.as_deref())
                        .collect::<Vec<_>>()
                )],
            ));
        }
    }

    subst.apply(&instantiated_ret)
}

/// The single record-shaped variant of an ADT, when it has exactly
/// one variant and every field is named — the representation idiom
/// `access`/`record-update` resolve against.
fn single_record_variant<'a>(
    adt_reg: &'a AdtRegistry,
    adt_name: &str,
) -> Option<&'a crate::adt::VariantInfo> {
    let def = adt_reg.lookup(adt_name)?;
    if def.variants.len() != 1 {
        return None;
    }
    let variant = &def.variants[0];
    (!variant.fields.is_empty() && variant.fields.iter().all(|(n, _)| n.is_some()))
        .then_some(variant)
}

/// Instantiate `variant` of `adt_def` with fresh type variables:
/// returns the per-field types and the ADT result type with the
/// def's registration-time param vars renamed fresh. Bypasses the
/// name-keyed env so same-named constructors from colliding ADTs
/// (chelis#148) cannot cross-wire field types.
fn instantiate_variant_of(
    adt_def: &crate::adt::AdtDef,
    variant: &crate::adt::VariantInfo,
    vg: &mut VarGen,
) -> (Vec<Type>, Type) {
    let map: HashMap<TypeVar, Type> = adt_def
        .param_vars
        .iter()
        .map(|tv| (*tv, vg.fresh_type()))
        .collect();
    let args: Vec<Type> = variant
        .fields
        .iter()
        .map(|(_, t)| crate::adt::substitute_alias_type(t, &map))
        .collect();
    let ret = Type::Adt(
        adt_def.name.clone(),
        adt_def
            .param_vars
            .iter()
            .map(|tv| map.get(tv).cloned().expect("map covers param_vars"))
            .collect(),
    );
    (args, ret)
}

/// Instantiate the single record variant of the ADT named by
/// `target_ty` and unify the instantiated result with the target,
/// returning the per-field types aligned with `variant.fields` so
/// they reflect the target's concrete type arguments.
fn instantiated_field_types(
    adt_name: &str,
    variant: &crate::adt::VariantInfo,
    target_ty: &Type,
    adt_reg: &AdtRegistry,
    vg: &mut VarGen,
    subst: &mut Subst,
) -> Vec<Type> {
    let Some(adt_def) = adt_reg.lookup(adt_name) else {
        return Vec::new();
    };
    let (args, ret) = instantiate_variant_of(adt_def, variant, vg);
    let _ = unify(&ret, target_ty, subst);
    args
}

/// Infer `(access {} target field)` — record field access (RFC
/// D-CHECK prerequisite inference).
#[allow(clippy::too_many_arguments)]
fn infer_access(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.len() < 2 {
        return Type::Error;
    }
    let target_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let Some(field_name) = symbol_name(&kids[1]) else {
        return Type::Error;
    };
    // Peel borrow layers: an `&T` target reads through the borrow.
    let mut resolved = subst.apply(&target_ty);
    while let Type::Ref(inner) = resolved {
        resolved = *inner;
    }
    match resolved {
        Type::Adt(ref adt_name, _) => {
            // RFC D-CHECK: field access on an out-of-module opaque
            // type is rejected; inference continues so the access
            // still yields its true field type (no cascades).
            crate::opacity::check_opaque_use(
                crate::opacity::OpaqueAction::FieldAccess,
                adt_name,
                adt_reg,
                errors,
            );
            let Some(variant) = single_record_variant(adt_reg, adt_name) else {
                // Multi-variant or positional-field ADT: field access
                // is not defined for it; conservative status quo
                // (silently untyped) to keep the blast radius of the
                // new inference at the single-record idiom.
                return Type::Error;
            };
            let pos = variant
                .fields
                .iter()
                .position(|(n, _)| n.as_deref() == Some(field_name));
            let Some(pos) = pos else {
                errors.push(CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!("unknown record field '{field_name}' on {adt_name}"),
                    vec![format!(
                        "known fields: {:?}",
                        variant
                            .fields
                            .iter()
                            .filter_map(|(n, _)| n.as_deref())
                            .collect::<Vec<_>>()
                    )],
                ));
                return Type::Error;
            };
            let field_types =
                instantiated_field_types(adt_name, variant, &resolved, adt_reg, vg, subst);
            match field_types.get(pos) {
                Some(ty) => subst.apply(ty),
                None => Type::Error,
            }
        }
        Type::Var(tv) => {
            // Target not yet pinned (e.g. unannotated lambda param):
            // register in the deferred-access ledger so a later pin to
            // an out-of-module opaque ADT is still rejected at
            // def-level resolution (D-CHECK). The result type keeps
            // the conservative status quo.
            subst.record_deferred_opaque_use(tv, crate::unify::DeferredOpaqueUse::Access);
            Type::Error
        }
        // Conservative status quo for non-record targets: `access` on
        // tensors/prims/tuples stays silently untyped in W1 rather
        // than newly rejecting shapes the corpus may rely on.
        _ => Type::Error,
    }
}

/// Infer `(record-update {} target (kv {} field value)...)` — Deep
/// functional record update (RFC D-CHECK prerequisite inference).
#[allow(clippy::too_many_arguments)]
fn infer_record_update(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.is_empty() {
        return Type::Error;
    }
    let target_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    // Infer the update values regardless of target resolution so
    // nested errors surface exactly once.
    let mut kv_pairs: Vec<(&str, Type)> = Vec::new();
    for kv_expr in kids.iter().skip(1) {
        let deep::Expr::List(kv_list, _) = kv_expr else {
            continue;
        };
        if get_tag(kv_list) != Some("kv") {
            continue;
        }
        let kv_kids = children(kv_list);
        let (Some(field_name), Some(value)) =
            (kv_kids.first().and_then(symbol_name), kv_kids.get(1))
        else {
            continue;
        };
        let value_ty = infer_expr(
            value,
            env,
            vg,
            subst,
            adt_reg,
            errors,
            typed_nodes,
            total_nodes,
        );
        kv_pairs.push((field_name, value_ty));
    }
    let mut resolved = subst.apply(&target_ty);
    while let Type::Ref(inner) = resolved {
        resolved = *inner;
    }
    match resolved {
        Type::Adt(ref adt_name, _) => {
            // RFC D-CHECK: record update of an out-of-module opaque
            // type is rejected; inference continues and returns the
            // target's true type (no cascades).
            crate::opacity::check_opaque_use(
                crate::opacity::OpaqueAction::RecordUpdate,
                adt_name,
                adt_reg,
                errors,
            );
            let Some(variant) = single_record_variant(adt_reg, adt_name) else {
                return Type::Error;
            };
            let field_types =
                instantiated_field_types(adt_name, variant, &resolved, adt_reg, vg, subst);
            for (field_name, value_ty) in &kv_pairs {
                let pos = variant
                    .fields
                    .iter()
                    .position(|(n, _)| n.as_deref() == Some(*field_name));
                match pos {
                    Some(pos) => {
                        if let Some(field_ty) = field_types.get(pos)
                            && let Err(te) = unify(value_ty, field_ty, subst)
                        {
                            errors.push(te.into());
                        }
                    }
                    None => {
                        errors.push(CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            format!("unknown record field '{field_name}' on {adt_name}"),
                            vec![format!(
                                "known fields: {:?}",
                                variant
                                    .fields
                                    .iter()
                                    .filter_map(|(n, _)| n.as_deref())
                                    .collect::<Vec<_>>()
                            )],
                        ));
                    }
                }
            }
            subst.apply(&resolved)
        }
        Type::Var(tv) => {
            subst.record_deferred_opaque_use(tv, crate::unify::DeferredOpaqueUse::RecordUpdate);
            // The update returns the target's (still-unresolved) type.
            Type::Var(tv)
        }
        Type::Error => Type::Error,
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("record-update requires a record-typed target, got {other}"),
                vec![],
            ));
            Type::Error
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn infer_cast(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.len() < 2 {
        return Type::Error;
    }

    let expr_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let resolved = subst.apply(&expr_ty);

    // RFC D-CHECK cast gates: cast-into an out-of-module opaque type
    // (both Deep target shapes, `t-prim` and `t-adt`, with aliases
    // expanded) and cast-out of an out-of-module opaque value. Each
    // pushes one OpaqueTypeViolation and returns the TRUE type of the
    // expression so no error cascades; inside the defining module the
    // existing cast semantics (including `CastNonTensor` for ADT
    // sources) are unchanged.
    if let Some(target_adt) = cast_target_adt_name(&kids[1], adt_reg) {
        if crate::opacity::check_opaque_use(
            crate::opacity::OpaqueAction::CastInto,
            &target_adt,
            adt_reg,
            errors,
        ) {
            return Type::Adt(target_adt, Vec::new());
        }
    } else {
        let mut peeled = &resolved;
        while let Type::Ref(inner) = peeled {
            peeled = inner.as_ref();
        }
        if let Type::Adt(source_adt, _) = peeled {
            let source_adt = source_adt.clone();
            if crate::opacity::check_opaque_use(
                crate::opacity::OpaqueAction::CastOut,
                &source_adt,
                adt_reg,
                errors,
            ) {
                return match deep_type_to_resolved_type(&kids[1], vg, adt_reg, &mut HashMap::new())
                {
                    Type::Prim(p) => Type::Prim(p),
                    _ => Type::Error,
                };
            }
        }
    }

    // kids[1] = (t-prim {} new_precision)
    // A1 (WS-A0 RT-1 fixup): per spec/04-type-system.md §1.1.2, unsigned
    // integer types (u8/u16/u32/u64 and the uint8/uint16/uint32/uint64
    // alias family) are explicitly out of scope for this cycle. They
    // never resolve through `Prim::parse_name`, so without this guard
    // `cast(_, u8)` would silently fall through to `Type::Error` with
    // no diagnostic — exactly the silent-cast pattern §1.1.1 was added
    // to avoid for f8e4m3. Mirror the f8e4m3 rejection path here.
    if let Some(name) = cast_target_prim_name(&kids[1])
        && let Some(diag) = unsigned_family_diagnostic(name, /* tensor = */ false)
    {
        errors.push(diag);
        return Type::Error;
    }
    let new_prec = match deep_type_to_resolved_type(&kids[1], vg, adt_reg, &mut HashMap::new()) {
        Type::Prim(p) => p,
        _ => return Type::Error,
    };

    match resolved {
        Type::Tensor(dims, _) => {
            if !new_prec.is_valid_tensor_precision() {
                push_unsupported_precision_error(errors, new_prec, /* tensor = */ true);
                return Type::Error;
            }
            Type::Tensor(dims, TensorPrec::Concrete(new_prec))
        }
        Type::Prim(_) => {
            if !new_prec.is_valid_scalar_cast_target() {
                push_unsupported_precision_error(errors, new_prec, /* tensor = */ false);
                return Type::Error;
            }
            Type::Prim(new_prec)
        }
        Type::Error => Type::Error,
        _ => {
            errors.push(CheckError::new(
                CheckErrorKind::CastNonTensor,
                format!("cast requires tensor or prim type, got {resolved}"),
                vec![],
            ));
            Type::Error
        }
    }
}

/// The nominal ADT a cast target names, if any: `(t-adt {} Name)` or
/// a `(t-prim {} Name)` whose name is not a primitive but resolves in
/// the ADT registry, with transparent aliases expanded to the nominal
/// entry. Returns `None` for genuine primitive targets.
fn cast_target_adt_name(expr: &deep::Expr, adt_reg: &AdtRegistry) -> Option<String> {
    let deep::Expr::List(list, _) = expr else {
        return None;
    };
    let name = children(list).first().and_then(symbol_name)?;
    match get_tag(list) {
        Some("t-adt") => {}
        Some("t-prim") => {
            if Prim::parse_name(name).is_some() {
                return None;
            }
        }
        _ => return None,
    }
    if adt_reg.lookup(name).is_some() {
        return Some(name.to_string());
    }
    if let Some(alias) = adt_reg.resolve_alias(name)
        && let Type::Adt(target, _) = &alias.body
    {
        return Some(target.clone());
    }
    None
}

/// Extract the symbol-name from a `(t-prim {} <name>)` Deep node so a
/// rejection path can run before `Prim::parse_name` returns `None` and
/// erases the spelling. Returns `None` for any other shape.
fn cast_target_prim_name(expr: &deep::Expr) -> Option<&str> {
    let list = match expr {
        deep::Expr::List(l, _) => l,
        _ => return None,
    };
    if get_tag(list) != Some("t-prim") {
        return None;
    }
    children(list).first().and_then(symbol_name)
}

/// True if `name` is one of the unsigned integer dtype names that
/// `spec/04-type-system.md` §1.1.2 declares out of scope. Covers both
/// the short form (`u8`/`u16`/`u32`/`u64`) and the explicit `uint*`
/// alias family that LLMs and cross-language users tend to write.
fn is_unsigned_dtype_name(name: &str) -> bool {
    matches!(
        name,
        "u8" | "u16" | "u32" | "u64" | "uint8" | "uint16" | "uint32" | "uint64"
    )
}

/// Build a §1.1.2 diagnostic for an unsigned dtype name appearing as a
/// cast target or a tensor element type. Returns `None` for non-unsigned
/// names so call sites can short-circuit with `&&`.
fn unsigned_family_diagnostic(name: &str, tensor: bool) -> Option<CheckError> {
    if !is_unsigned_dtype_name(name) {
        return None;
    }
    let surface = if tensor { "tensor element" } else { "scalar" };
    let active_set = "f32, f64, bf16, f16, bool, int8, int16, int32, int64";
    Some(CheckError::new(
        CheckErrorKind::UnsupportedTensorPrecision,
        format!(
            "cannot use `{name}` as a {surface} dtype: unsigned integer types \
             are out of scope per spec/04-type-system.md §1.1.2 (active set: \
             {active_set})"
        ),
        vec![format!(
            "spec/04-type-system.md §1.1.2 documents the workaround: cast to \
             int32 or int64 and reason at the wider signed precision; or use \
             a tensor of int8 / int16 / int32 / int64 if the bit-width matters"
        )],
    ))
}

/// Emit the canonical "unsupported precision" diagnostic for either a
/// tensor element or a scalar cast target. The deferred `f8e4m3` dtype
/// (`spec/04-type-system.md` §1.1.1) gets a specific diagnostic citing the
/// owning spec section so producers can resolve the deferral state without
/// guessing.
fn push_unsupported_precision_error(errors: &mut Vec<CheckError>, new_prec: Prim, tensor: bool) {
    let surface = if tensor { "tensor element" } else { "scalar" };
    let active_set = "f32, f64, bf16, f16, bool, int8, int16, int32, int64";
    if matches!(new_prec, Prim::F8e4m3) {
        errors.push(CheckError::new(
            CheckErrorKind::UnsupportedTensorPrecision,
            format!(
                "cannot cast {surface} to `f8e4m3`: f8e4m3 is deferred per \
                 spec/04-type-system.md §1.1.1 and is not part of the active \
                 numeric primitive set ({active_set})"
            ),
            vec![format!(
                "f8e4m3 has no active backend in this cycle; cast to one of \
                 {active_set} instead, or follow spec/04-type-system.md §1.1.1 \
                 for the deferral rationale"
            )],
        ));
    } else {
        errors.push(CheckError::new(
            CheckErrorKind::UnsupportedTensorPrecision,
            format!(
                "cannot cast {surface} to unsupported precision `{}` \
                 (supported: {active_set})",
                new_prec.name()
            ),
            vec![format!("Use a supported {surface} precision")],
        ));
    }
}

#[allow(clippy::too_many_arguments)]
fn infer_grad(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.is_empty() {
        return Type::Error;
    }

    let f_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let resolved = subst.apply(&f_ty);

    match resolved {
        Type::Fn(args, ret) => {
            let ret = *ret;
            if !grad_output_supported(&ret) {
                errors.push(CheckError::new(
                    CheckErrorKind::Other,
                    format!("grad requires a scalar floating output, got {}", ret),
                    vec!["Reduce the function result to a scalar before applying grad".to_string()],
                ));
                return Type::Error;
            }

            match grad_result_type(list, &args, adt_reg, errors) {
                Some(grad_ret) => Type::Fn(args, Box::new(grad_ret)),
                None => Type::Error,
            }
        }
        Type::Error => Type::Error,
        _ => {
            // Can't determine function structure, return fresh var
            vg.fresh_type()
        }
    }
}

fn grad_output_supported(ty: &Type) -> bool {
    match ty {
        Type::Prim(prim) => prim.is_float(),
        Type::Tensor(dims, prim) => dims.is_empty() && prim.is_float(),
        _ => false,
    }
}

fn grad_result_type(
    list: &deep::List,
    args: &[Type],
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
) -> Option<Type> {
    let targets = if let Some(indices) = grad_wrt_indices(list, errors)? {
        let mut selected = Vec::with_capacity(indices.len());
        for index in indices {
            let Some(arg) = args.get(index) else {
                errors.push(CheckError::new(
                    CheckErrorKind::ArityMismatch,
                    format!(
                        "grad `wrt` index {} is out of bounds for function with {} parameters",
                        index,
                        args.len()
                    ),
                    vec![],
                ));
                return None;
            };
            let Some(grad_ty) = grad_argument_type(arg, adt_reg) else {
                errors.push(CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!("grad `wrt` index {index} is not differentiable"),
                    vec!["Select floating scalar or tensor parameters in `wrt`".to_string()],
                ));
                return None;
            };
            selected.push(grad_ty);
        }
        selected
    } else {
        args.iter()
            .filter_map(|arg| grad_argument_type(arg, adt_reg))
            .collect()
    };

    // chelis#520 D2: an ADT gradient target is supported alongside plain
    // tensor/scalar targets in a multi-argument call. `grad_argument_type`
    // has already mapped each selected parameter to its gradient type (an
    // all-float-field ADT maps to itself; a tensor/scalar to itself; a
    // non-differentiable payload was skipped or rejected). The result type
    // is the per-target tuple, whose ADT slot is the field-wise gradient
    // struct (the pytree contract). The eval-lane marshalling packs the
    // flat gradient roots back into this exact structure per argument.
    Some(match targets.as_slice() {
        [] => Type::Unit,
        [single] => single.clone(),
        _ => Type::Tuple(targets),
    })
}

fn grad_wrt_indices(list: &deep::List, errors: &mut Vec<CheckError>) -> Option<Option<Vec<usize>>> {
    let kids = children(list);
    let Some(wrt_expr) = kids.get(1) else {
        return Some(None);
    };

    // Issue #216: cast-aware so a Deep-direct grad node with cast-wrapped
    // wrt indices peels to the underlying int and trips the
    // non-negative-index check at infer time. Surf desugar resolves
    // parameter names to bare literal ints before reaching here, so the
    // swap is defense-in-depth for Deep-direct callers (decompiler,
    // macro output, custom tooling).
    match wrt_expr {
        deep::Expr::List(tuple, _) if get_tag(tuple) == Some("tuple") => {
            let mut indices = Vec::new();
            for item in children(tuple) {
                let Some(index) = extract_int_for_dim(item) else {
                    errors.push(CheckError::new(
                        CheckErrorKind::TypeMismatch,
                        "grad `wrt` tuple must contain integer parameter indices".to_string(),
                        vec![],
                    ));
                    return None;
                };
                if index < 0 {
                    errors.push(CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        format!("grad `wrt` index must be non-negative, got {index}"),
                        vec![],
                    ));
                    return None;
                }
                indices.push(index as usize);
            }
            Some(Some(indices))
        }
        other => {
            let Some(index) = extract_int_for_dim(other) else {
                errors.push(CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    "grad `wrt` must be an integer parameter index or tuple of indices".to_string(),
                    vec![],
                ));
                return None;
            };
            if index < 0 {
                errors.push(CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!("grad `wrt` index must be non-negative, got {index}"),
                    vec![],
                ));
                return None;
            }
            Some(Some(vec![index as usize]))
        }
    }
}

fn grad_argument_type(arg: &Type, adt_reg: &AdtRegistry) -> Option<Type> {
    match arg {
        Type::Prim(prim) if prim.is_float() => Some(Type::Prim(*prim)),
        // WS-A5: a polymorphic precision (TensorPrec::Var) is not yet
        // known to be float, so reject it here. Once monomorphization
        // resolves the precision, the rule re-fires on the concrete
        // instantiation. `is_float()` returns false for Var precisions.
        Type::Tensor(dims, prec) if prec.is_float() => {
            Some(Type::Tensor(dims.clone(), prec.clone()))
        }
        // chelis#520 D2 slice: an ADT whose every variant carries only
        // float tensors / float scalars gets a field-wise gradient of
        // the same constructor shape (spec/06-transformations.md
        // §2.10.1). Mixed or non-tensor payloads stay
        // non-differentiable, so the arg is skipped (no `wrt`) or
        // rejected (`wrt`-selected) exactly as before. Generic ADTs
        // fall out naturally: an uninstantiated param var is not a
        // float tensor.
        Type::Adt(name, args) => {
            let def = adt_reg.defs.get(name)?;
            let all_float_fields = def.variants.iter().all(|variant| {
                variant.fields.iter().all(|(_, field_ty)| match field_ty {
                    Type::Prim(prim) => prim.is_float(),
                    Type::Tensor(_, prec) => prec.is_float(),
                    _ => false,
                })
            });
            // A pure enum (no fields in any variant) carries no
            // continuous payload: there is nothing to differentiate,
            // and typing its gradient as the enum itself would claim a
            // gradient value the runtime cannot produce. Keep it
            // non-differentiable (unit payload), the pre-#520 typing.
            let has_any_field = def
                .variants
                .iter()
                .any(|variant| !variant.fields.is_empty());
            (all_float_fields && has_any_field).then(|| Type::Adt(name.clone(), args.clone()))
        }
        Type::Ref(inner) => grad_argument_type(inner, adt_reg),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn infer_vmap(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.is_empty() {
        return Type::Error;
    }

    // Issue #216: cast-aware so a Deep-direct vmap node with a cast-
    // wrapped axis literal peels to the underlying int and trips the
    // non-negative check. Surf parser restricts vmap's axis to bare
    // ints, so this is defense-in-depth for Deep-direct callers.
    let axis = kids.get(1).and_then(extract_int_for_dim).unwrap_or(0);
    if axis < 0 {
        errors.push(CheckError::new(
            CheckErrorKind::DimensionMismatch,
            format!("vmap axis must be non-negative, got {axis}"),
            vec!["Use `vmap(f)` or `vmap(f, axis=n)` with n >= 0".to_string()],
        ));
        return Type::Error;
    }
    let axis = axis as usize;

    let f_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let resolved = subst.apply(&f_ty);

    match resolved {
        Type::Fn(args, ret) => {
            let batch_dim = Dim::Var(vg.fresh_dvar());
            let args = args
                .iter()
                .map(|arg| vmap_transform_param_type(arg, axis, &batch_dim))
                .collect::<Result<Vec<_>, _>>();
            let ret = vmap_transform_result_type(&ret, axis, &batch_dim);

            match (args, ret) {
                (Ok(args), Ok(ret)) => Type::Fn(args, Box::new(ret)),
                (Err(message), _) | (_, Err(message)) => {
                    errors.push(CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        message,
                        vec![
                            "Choose an axis that is in bounds for every vmapped tensor".to_string(),
                        ],
                    ));
                    Type::Error
                }
            }
        }
        Type::Error => Type::Error,
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("vmap expects a function, got {other}"),
                vec!["Apply `vmap` to a named function or inline lambda".to_string()],
            ));
            Type::Error
        }
    }
}

fn vmap_transform_param_type(ty: &Type, axis: usize, batch_dim: &Dim) -> Result<Type, String> {
    match ty {
        Type::Ref(inner) => Ok(Type::Ref(Box::new(vmap_transform_param_type(
            inner, axis, batch_dim,
        )?))),
        Type::Tensor(dims, precision) => {
            if axis > dims.len() {
                return Err(format!(
                    "vmap axis {axis} is out of bounds for rank {} tensor",
                    dims.len()
                ));
            }
            let mut dims = dims.clone();
            dims.insert(axis, batch_dim.clone());
            Ok(Type::Tensor(dims, precision.clone()))
        }
        Type::Tuple(elements) => Ok(Type::Tuple(
            elements
                .iter()
                .map(|element| vmap_transform_param_type(element, axis, batch_dim))
                .collect::<Result<_, _>>()?,
        )),
        other => Ok(other.clone()),
    }
}

fn vmap_transform_result_type(ty: &Type, axis: usize, batch_dim: &Dim) -> Result<Type, String> {
    match ty {
        Type::Ref(inner) => Ok(Type::Ref(Box::new(vmap_transform_result_type(
            inner, axis, batch_dim,
        )?))),
        Type::Prim(precision) => Ok(Type::Tensor(
            vec![batch_dim.clone()],
            TensorPrec::Concrete(*precision),
        )),
        Type::Tensor(dims, precision) => {
            if axis > dims.len() {
                return Err(format!(
                    "vmap axis {axis} is out of bounds for rank {} tensor",
                    dims.len()
                ));
            }
            let mut dims = dims.clone();
            dims.insert(axis, batch_dim.clone());
            Ok(Type::Tensor(dims, precision.clone()))
        }
        Type::Tuple(elements) => Ok(Type::Tuple(
            elements
                .iter()
                .map(|element| vmap_transform_result_type(element, axis, batch_dim))
                .collect::<Result<_, _>>()?,
        )),
        other => Ok(other.clone()),
    }
}

#[allow(clippy::too_many_arguments)]
fn infer_def(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.len() < 2 {
        return Type::Error;
    }

    let name = match symbol_name(&kids[0]) {
        Some(n) => n.to_string(),
        None => return Type::Error,
    };

    let body_ty = infer_expr(
        &kids[1],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let scheme = env.generalize(&body_ty, subst);
    // chelis#397/#469: record the size provenance (see `infer_top_level` /
    // `infer_let`) so a later `expand` size built from this binding can be
    // checked for materializability. Classified against the pre-binding scope.
    match classify_expand_size(&kids[1], env) {
        SizeClass::Static => {
            env.mark_size_provenance(&name, crate::env::SizeProvenance::Static);
        }
        SizeClass::ShapeSourced => {
            env.mark_size_provenance(&name, crate::env::SizeProvenance::ShapeSourced);
        }
        SizeClass::Sourceless | SizeClass::Unknown => env.clear_size_provenance(&name),
    }
    env.bind(name, scheme);
    body_ty
}

// ── Type conversion from Deep AST ───────────────────────────────

/// Convert a Deep type expression to an internal Type.
/// `tvar_map` maps type variable names to TypeVars (created on demand).
/// `dvar_map` maps dimension variable names to DimVars (created on demand).
fn deep_type_to_type(
    expr: &deep::Expr,
    vg: &mut VarGen,
    tvar_map: &mut HashMap<String, TypeVar>,
) -> Type {
    let mut dvar_map = HashMap::new();
    let mut rvar_map = HashMap::new();
    deep_type_to_type_inner(expr, vg, tvar_map, &mut dvar_map, &mut rvar_map)
}

fn deep_type_to_type_inner(
    expr: &deep::Expr,
    vg: &mut VarGen,
    tvar_map: &mut HashMap<String, TypeVar>,
    dvar_map: &mut HashMap<String, DimVar>,
    rvar_map: &mut HashMap<String, RankVar>,
) -> Type {
    stack_guard!("deep_type_to_type_inner", expr, Type::Error);
    match expr {
        deep::Expr::List(list, _) => {
            let tag = get_tag(list).unwrap_or("");
            let kids = children(list);
            match tag {
                "t-prim" => {
                    if let Some(name) = kids.first().and_then(|e| symbol_name(e)) {
                        Prim::parse_name(name)
                            .map(Type::Prim)
                            .unwrap_or(Type::Error)
                    } else {
                        Type::Error
                    }
                }
                "t-var" => {
                    if let Some(name) = kids.first().and_then(|e| symbol_name(e)) {
                        if name == "_" {
                            vg.fresh_type()
                        } else {
                            let tv = *tvar_map
                                .entry(name.to_string())
                                .or_insert_with(|| vg.fresh_tvar());
                            Type::Var(tv)
                        }
                    } else {
                        Type::Error
                    }
                }
                "t-fn" => {
                    if kids.is_empty() {
                        return Type::Error;
                    }
                    let args: Vec<Type> = kids[..kids.len() - 1]
                        .iter()
                        .map(|c| deep_type_to_type_inner(c, vg, tvar_map, dvar_map, rvar_map))
                        .collect();
                    let ret = deep_type_to_type_inner(
                        &kids[kids.len() - 1],
                        vg,
                        tvar_map,
                        dvar_map,
                        rvar_map,
                    );
                    Type::Fn(args, Box::new(ret))
                }
                "t-ref" => {
                    if kids.len() != 1 {
                        return Type::Error;
                    }
                    Type::Ref(Box::new(deep_type_to_type_inner(
                        &kids[0], vg, tvar_map, dvar_map, rvar_map,
                    )))
                }
                "t-tensor" => {
                    if kids.is_empty() {
                        return Type::Error;
                    }
                    let prec_expr = &kids[kids.len() - 1];
                    // WS-A5 (spec/04-type-system.md §5.8): the precision
                    // slot may be a concrete primitive (`(t-prim {} f32)`)
                    // or a sig-quantified type variable (`(t-var {} p)`).
                    // Both shapes are well-formed; any other shape (e.g.,
                    // a `t-fn` or a `t-prim` with an unknown name) is an
                    // ill-formed tensor and is reduced to `Type::Error`.
                    let prec = match deep_type_to_type_inner(
                        prec_expr, vg, tvar_map, dvar_map, rvar_map,
                    ) {
                        Type::Prim(p) => TensorPrec::Concrete(p),
                        Type::Var(v) => TensorPrec::Var(v),
                        _ => return Type::Error,
                    };
                    let dims: Vec<Dim> = kids[..kids.len() - 1]
                        .iter()
                        .filter_map(|c| parse_dim(c, vg, dvar_map, rvar_map))
                        .collect();
                    Type::Tensor(dims, prec)
                }
                "t-adt" => {
                    if let Some(name) = kids.first().and_then(|e| symbol_name(e)) {
                        let args: Vec<Type> = kids[1..]
                            .iter()
                            .map(|c| deep_type_to_type_inner(c, vg, tvar_map, dvar_map, rvar_map))
                            .collect();
                        Type::Adt(name.to_string(), args)
                    } else {
                        Type::Error
                    }
                }
                "t-tuple" => {
                    let elems: Vec<Type> = kids
                        .iter()
                        .map(|c| deep_type_to_type_inner(c, vg, tvar_map, dvar_map, rvar_map))
                        .collect();
                    Type::Tuple(elems)
                }
                "t-unit" => Type::Unit,
                _ => Type::Error,
            }
        }
        _ => Type::Error,
    }
}

/// Parse a dimension expression from Deep AST, with support for dim variables
/// and rank variables. `rvar_map` memoizes `..r` names to a single `RankVar`
/// so the same rank variable shared across tensor positions ties together.
fn parse_dim(
    expr: &deep::Expr,
    vg: &mut VarGen,
    dvar_map: &mut HashMap<String, DimVar>,
    rvar_map: &mut HashMap<String, RankVar>,
) -> Option<Dim> {
    match expr {
        deep::Expr::List(list, _) => {
            let tag = get_tag(list).unwrap_or("");
            let kids = children(list);
            match tag {
                "d-name" => {
                    if let Some(name) = kids.first().and_then(|e| symbol_name(e)) {
                        if name == "*" {
                            Some(Dim::Wildcard)
                        } else {
                            Some(Dim::Name(name.to_string()))
                        }
                    } else {
                        None
                    }
                }
                "d-var" => {
                    if let Some(name) = kids.first().and_then(|e| symbol_name(e)) {
                        let dv = *dvar_map
                            .entry(name.to_string())
                            .or_insert_with(|| vg.fresh_dvar());
                        Some(Dim::Var(dv))
                    } else {
                        Some(vg.fresh_dim())
                    }
                }
                "d-lit" => {
                    if let Some(deep::Expr::Atom(deep::Atom::Int(n), _)) = kids.first() {
                        Some(Dim::Lit(*n))
                    } else {
                        None
                    }
                }
                "d-rank" => {
                    let name = kids
                        .first()
                        .and_then(|e| symbol_name(e))
                        .unwrap_or("_")
                        .to_string();
                    let rv = *rvar_map.entry(name).or_insert_with(|| vg.fresh_rvar());
                    Some(Dim::Rank(rv))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(src: &str) -> InferResult {
        let exprs = chelis_deep::parser::parse_str(src).unwrap();
        infer_program(&exprs)
    }

    fn checked_surf(src: &str) -> CheckedProgram {
        let decls = chelis_surf::parser::parse_str(src).expect("surf parse");
        let exprs = chelis_surf::desugar::desugar_program(&decls);
        check_ir_program(&exprs).expect("IR check")
    }

    /// Run the full Surf → desugar → infer pipeline and return the raw
    /// `InferResult` (errors included) so a test can assert clean or assert
    /// a specific failure mode end-to-end.
    fn infer_surf(src: &str) -> InferResult {
        let decls = chelis_surf::parser::parse_str(src).expect("surf parse");
        let exprs = chelis_surf::desugar::desugar_program(&decls);
        infer_program(&exprs)
    }

    fn missing_shape_sensitive_app(expr: &deep::Expr) -> Option<String> {
        match expr {
            deep::Expr::List(list, _) => {
                if get_tag(list) == Some("app")
                    && is_shape_sensitive_app(list)
                    && !list
                        .elements
                        .get(1)
                        .and_then(|expr| match expr {
                            deep::Expr::Map(meta, _) => Some(meta),
                            _ => None,
                        })
                        .is_some_and(|meta| meta.entries.iter().any(|(key, _)| key == "type"))
                {
                    return Some(
                        chelis_deep::printer::print_canonical(std::slice::from_ref(expr))
                            .replace('\n', " ")
                            .trim()
                            .to_string(),
                    );
                }
                for child in &list.elements {
                    if let Some(missing) = missing_shape_sensitive_app(child) {
                        return Some(missing);
                    }
                }
                None
            }
            deep::Expr::Map(map, _) => map
                .entries
                .iter()
                .find_map(|(_, value)| missing_shape_sensitive_app(value)),
            deep::Expr::MetaExpr(meta, _) => {
                missing_shape_sensitive_app(&meta.expr).or_else(|| {
                    meta.entries
                        .iter()
                        .find_map(|(_, value)| missing_shape_sensitive_app(value))
                })
            }
            deep::Expr::Atom(_, _) => None,
        }
    }

    fn is_shape_sensitive_app(list: &deep::List) -> bool {
        get_tag(list) == Some("app")
            && ir_builtin_name(list).is_some_and(super::is_ir_shape_sensitive_builtin)
    }

    fn check_ok(src: &str) {
        let result = check(src);
        assert!(
            result.errors.is_empty(),
            "expected no errors, got: {:?}",
            result.errors
        );
    }

    fn check_err(src: &str, expected_kind: CheckErrorKind) {
        let result = check(src);
        assert!(
            !result.errors.is_empty(),
            "expected error {expected_kind:?}, got none"
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| std::mem::discriminant(&e.kind) == std::mem::discriminant(&expected_kind)),
            "expected {expected_kind:?}, got: {:?}",
            result.errors
        );
    }

    // ── WI-1 stack-budget guard (per-walker coverage) ────────────

    /// Build a left-nested `app` chain of `depth` distinct names directly in
    /// Deep, for the recursion-depth guard tests.
    fn deep_app_chain_node(depth: usize) -> deep::Expr {
        let sym = |s: &str| deep::Expr::Atom(deep::Atom::Symbol(s.to_string()), Span::new(0, 0));
        let meta = || deep::Expr::Map(deep::MetaMap::default(), Span::new(0, 0));
        let var = |n: &str| {
            deep::Expr::List(
                deep::List {
                    elements: vec![sym("var"), meta(), sym(n)],
                },
                Span::new(0, 0),
            )
        };
        let mut e = var("f0");
        for i in 1..=depth {
            e = deep::Expr::List(
                deep::List {
                    elements: vec![sym("app"), meta(), e, var(&format!("f{i}"))],
                },
                Span::new(0, 0),
            );
        }
        e
    }

    /// The sibling walker `walk_for_tensor_precision` is independently
    /// guarded: run it (via `validate_tensor_precisions_in_program`) on a
    /// chain deep enough to overflow an unguarded recursion, on a bounded
    /// stack. gdb showed this walker -- not `infer_expr` -- is the SIGSEGV
    /// site for this pass on deep input; reaching the assertion at all proves
    /// the guard prevented the overflow, and the recorded `STACK_EXHAUSTED`
    /// flag proves the bail surfaces.
    #[test]
    fn walk_for_tensor_precision_sibling_is_guarded_not_sigsegv() {
        // Built and dropped on the main thread: the chain's unguarded drop
        // glue overflows a 1 MiB stack on its own under rustc 1.97 codegen
        // (see `stack_exhaustion_drains_into_a_located_error`).
        let program = std::sync::Arc::new(vec![deep_app_chain_node(4000)]);
        let worker_program = std::sync::Arc::clone(&program);
        let flagged = std::thread::Builder::new()
            .name("sibling-guard-test".to_string())
            // 1 MiB: small enough that a 4000-deep chain trips the byte
            // budget early, well inside the guard, with no risk of overflow.
            .stack_size(1024 * 1024)
            .spawn(move || {
                let _scope = StackExhaustionScope::enter();
                let mut errors = Vec::new();
                // Drive ONLY the tensor-precision pass (whose deep walker is
                // walk_for_tensor_precision), isolating it from infer_expr.
                validate_tensor_precisions_in_program(&worker_program, &mut errors);
                STACK_EXHAUSTED.with(|cell| cell.borrow().clone())
            })
            .expect("spawn sibling-guard worker")
            .join()
            .expect("sibling walker aborted (stack overflow?) instead of returning");
        drop(program);

        let (site, _span) = flagged.expect(
            "walk_for_tensor_precision must record a stack-exhaustion bail on a \
             4000-deep chain run on a 1 MiB stack",
        );
        assert!(
            site.contains("walk_for_tensor_precision"),
            "the bail must be attributed to the precision walker, got site: {site}",
        );
    }

    /// The funnel is sound: a walker that bails with NO error vector (here the
    /// tensor-precision pass is driven in isolation) still makes the flag set,
    /// and `StackExhaustionScope::drain_into` turns it into a hard located
    /// error -- never a silent empty result.
    #[test]
    fn stack_exhaustion_drains_into_a_located_error() {
        // The 4000-deep chain is BUILT and DROPPED on the test's main thread:
        // its derived drop glue recurses the full chain depth with no guard,
        // and under rustc 1.97 codegen those frames overflow the worker's
        // 1 MiB stack on their own (the pre-1.97 frames merely happened to
        // fit). The worker thread exists to exercise the GUARDED walker on a
        // small stack; the unguarded collateral must not share it.
        // (Production runs construction, walkers, and drops on the
        // `with_grown_stack` segment, so this is a test-harness concern.)
        let program = std::sync::Arc::new(vec![deep_app_chain_node(4000)]);
        let worker_program = std::sync::Arc::clone(&program);
        let errors = std::thread::Builder::new()
            .stack_size(1024 * 1024)
            .spawn(move || {
                let scope = StackExhaustionScope::enter();
                let mut errors = Vec::new();
                validate_tensor_precisions_in_program(&worker_program, &mut errors);
                // Before drain: the precision pass carries no error vector of
                // its own for the bail, so `errors` may be empty here ...
                scope.drain_into(&mut errors);
                // ... but after drain the exhaustion is a hard error.
                drop(worker_program);
                errors
            })
            .expect("spawn drain-test worker")
            .join()
            .expect("worker aborted instead of returning");
        drop(program);
        assert!(
            errors
                .iter()
                .any(|e| e.message.contains("stack budget exhausted")),
            "drain_into must surface a located stack-budget error; got: {:?}",
            errors.iter().map(|e| &e.message).collect::<Vec<_>>(),
        );
    }

    /// Negative: a moderate-depth chain does not trip the guard in the
    /// precision pass either (no false positive on the sibling site).
    #[test]
    fn moderate_chain_does_not_trip_sibling_guard() {
        let flagged = std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let program = vec![deep_app_chain_node(20)];
                let _scope = StackExhaustionScope::enter();
                let mut errors = Vec::new();
                validate_tensor_precisions_in_program(&program, &mut errors);
                STACK_EXHAUSTED.with(|cell| cell.borrow().is_some())
            })
            .expect("spawn worker")
            .join()
            .expect("worker aborted");
        assert!(
            !flagged,
            "a 20-deep chain must not trip the precision walker's stack guard",
        );
    }

    // ── Variable / literal tests ─────────────────────────────────

    #[test]
    fn lit_int32() {
        check_ok("(def {} x (lit {type: (t-prim {} int32)} 42))");
    }

    #[test]
    fn lit_f32() {
        check_ok("(def {} x (lit {type: (t-prim {} f32)} 3.0))");
    }

    #[test]
    fn lit_bool() {
        check_ok("(def {} x (lit {type: (t-prim {} bool)} true))");
    }

    #[test]
    fn lit_string() {
        check_ok(r#"(def {} x (lit {type: (t-prim {} string)} "hello"))"#);
    }

    #[test]
    fn scalar_add_is_allowed() {
        check_ok(
            "(def {} x
                (app {} (var {} add)
                    (lit {type: (t-prim {} int64)} 2)
                    (lit {type: (t-prim {} int64)} 3)))",
        );
    }

    #[test]
    fn integer_mod_and_bitwise_builtins_are_allowed() {
        check_ok(
            "(def {} bits
                (app {} (var {} bitxor)
                    (app {} (var {} bitand)
                        (lit {type: (t-prim {} int64)} 7)
                        (lit {type: (t-prim {} int64)} 3))
                    (app {} (var {} shl)
                        (lit {type: (t-prim {} int64)} 1)
                        (lit {type: (t-prim {} int64)} 2))))
             (def {} rem
                (app {} (var {} mod)
                    (lit {type: (t-prim {} int64)} 17)
                    (lit {type: (t-prim {} int64)} 5)))
             (def {} shrunk
                (app {} (var {} shr)
                    (lit {type: (t-prim {} int64)} 8)
                    (lit {type: (t-prim {} int64)} 1)))",
        );
    }

    #[test]
    fn string_len_builtin_is_allowed() {
        check_ok(
            r#"(def {} x
                (app {} (var {} string_len)
                    (lit {type: (t-prim {} string)} "hé")))"#,
        );
    }

    #[test]
    fn string_predicates_and_transforms_are_allowed() {
        check_ok(
            r#"(def {} ok
                (if {}
                    (app {} (var {} and)
                        (app {} (var {} string_contains)
                            (lit {type: (t-prim {} string)} "ckpt-7.safetensors")
                            (lit {type: (t-prim {} string)} "ckpt"))
                        (app {} (var {} string_ends_with)
                            (app {} (var {} string_slice)
                                (app {} (var {} string_trim)
                                    (lit {type: (t-prim {} string)} "  ckpt-7.safetensors  "))
                                (lit {type: (t-prim {} int64)} 7)
                                (lit {type: (t-prim {} int64)} 12))
                            (lit {type: (t-prim {} string)} ".safetensors")))
                    (lit {type: (t-prim {} bool)} true)
                    (lit {type: (t-prim {} bool)} false)))"#,
        );
    }

    #[test]
    fn to_int_builtin_uses_prelude_option_without_local_deftype() {
        check_ok(
            r#"(def {} parsed
                (match {}
                    (app {} (var {} to_int)
                        (lit {type: (t-prim {} string)} "42"))
                    (arm {} (pat-ctor {} Some (pat-var {} n)) () (var {} n))
                    (arm {} (pat-ctor {} None) () (lit {type: (t-prim {} int64)} 0))))"#,
        );
    }

    #[test]
    fn to_float_builtin_uses_prelude_option_without_local_deftype() {
        check_ok(
            r#"(def {} parsed
                (match {}
                    (app {} (var {} to_float)
                        (lit {type: (t-prim {} string)} "0.125"))
                    (arm {} (pat-ctor {} Some (pat-var {} x)) () (var {} x))
                    (arm {} (pat-ctor {} None) () (lit {type: (t-prim {} f64)} 1.0))))"#,
        );
    }

    #[test]
    fn to_int_builtin_rejects_non_string_input() {
        check_err(
            r#"(def {} parsed
                (app {} (var {} to_int)
                    (lit {type: (t-prim {} int32)} 7)))"#,
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn mod_rejects_float_input() {
        check_err(
            r#"(def {} bad
                (app {} (var {} mod)
                    (lit {type: (t-prim {} f64)} 7.0)
                    (lit {type: (t-prim {} f64)} 3.0)))"#,
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn bitand_rejects_mismatched_integer_widths() {
        check_err(
            r#"(def {} bad
                (app {} (var {} bitand)
                    (lit {type: (t-prim {} int32)} 7)
                    (lit {type: (t-prim {} int64)} 3)))"#,
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn shl_rejects_non_integer_shift_amount() {
        check_err(
            r#"(def {} bad
                (app {} (var {} shl)
                    (lit {type: (t-prim {} int64)} 1)
                    (lit {type: (t-prim {} f64)} 2.0)))"#,
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn string_slice_rejects_non_string_input() {
        check_err(
            r#"(def {} bad
                (app {} (var {} string_slice)
                    (lit {type: (t-prim {} int32)} 7)
                    (lit {type: (t-prim {} int64)} 0)
                    (lit {type: (t-prim {} int64)} 1)))"#,
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn rank_builtin_requires_tensor_input() {
        check_err(
            "(def {} x
                (app {} (var {} rank)
                    (lit {type: (t-prim {} int64)} 2)))",
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn shape_builtin_accepts_tensor_input() {
        check_ok(
            r#"(def {} x
                (lit {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))} 0))
               (def {} dim
                (app {} (var {} shape)
                    (var {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))} x)
                    (lit {type: (t-prim {} int32)} 1)))"#,
        );
    }

    #[test]
    fn shape_builtin_rejects_negative_axis_when_rank_is_known() {
        check_err(
            r#"(def {} x
                (lit {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))} 0))
               (def {} dim
                (app {} (var {} shape)
                    (var {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))} x)
                    (lit {type: (t-prim {} int32)} -1)))"#,
            CheckErrorKind::DimensionMismatch,
        );
    }

    #[test]
    fn lit_default_int() {
        check_ok("(def {} x (lit {} 42))");
    }

    #[test]
    fn lit_default_float() {
        check_ok("(def {} x (lit {} 3.14))");
    }

    #[test]
    fn unbound_variable() {
        check_err(
            "(def {} x (var {} unknown))",
            CheckErrorKind::UnboundVariable,
        );
    }

    #[test]
    fn var_lookup_defined() {
        check_ok(
            "(def {} x (lit {type: (t-prim {} int32)} 42))
             (def {} y (var {} x))",
        );
    }

    // ── Application tests ────────────────────────────────────────

    #[test]
    fn app_add_tensors() {
        check_ok(
            "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} b (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} c (app {} (var {} add) (var {} a) (var {} b)))",
        );
    }

    #[test]
    fn app_precision_mismatch() {
        check_err(
            "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} b (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} bf16))} 0))
             (def {} c (app {} (var {} add) (var {} a) (var {} b)))",
            CheckErrorKind::PrecisionMismatch,
        );
    }

    #[test]
    fn app_not_a_function() {
        check_err(
            "(def {} x (lit {type: (t-prim {} int32)} 42))
             (def {} y (app {} (var {} x) (lit {type: (t-prim {} int32)} 1)))",
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn app_dimension_mismatch() {
        check_err(
            "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} b (lit {type: (t-tensor {} (d-name {} seq) (t-prim {} f32))} 0))
             (def {} c (app {} (var {} add) (var {} a) (var {} b)))",
            CheckErrorKind::DimensionMismatch,
        );
    }

    // ── Lambda tests ─────────────────────────────────────────────

    #[test]
    fn fn_identity() {
        check_ok("(def {} id (fn {} (params {} x) (var {} x)))");
    }

    #[test]
    fn fn_two_params() {
        check_ok("(def {} f (fn {} (params {} x y) (var {} x)))");
    }

    #[test]
    fn fn_with_body_app() {
        check_ok("(def {} f (fn {} (params {} x y) (app {} (var {} add) (var {} x) (var {} y))))");
    }

    // ── Let tests ────────────────────────────────────────────────

    #[test]
    fn let_simple() {
        check_ok(
            "(def {} result
               (let {} (bind {} x (lit {type: (t-prim {} int32)} 42))
                 (var {} x)))",
        );
    }

    #[test]
    fn let_multiple_bindings() {
        check_ok(
            "(def {} result
               (let {} (bind {} x (lit {type: (t-prim {} int32)} 1) y (lit {type: (t-prim {} f32)} 2.0))
                 (var {} x)))",
        );
    }

    #[test]
    fn let_scoping() {
        // Variable defined in let should be usable in body
        check_ok(
            "(def {} result
               (let {} (bind {} x (lit {type: (t-prim {} int32)} 42))
                 (var {} x)))",
        );
    }

    // ── If tests ─────────────────────────────────────────────────

    #[test]
    fn if_correct() {
        check_ok(
            "(def {} x (lit {type: (t-prim {} bool)} true))
             (def {} a (lit {type: (t-prim {} int32)} 1))
             (def {} b (lit {type: (t-prim {} int32)} 2))
             (def {} c (if {} (var {} x) (var {} a) (var {} b)))",
        );
    }

    #[test]
    fn if_branch_mismatch() {
        check_err(
            "(def {} x (lit {type: (t-prim {} bool)} true))
             (def {} a (lit {type: (t-prim {} int32)} 1))
             (def {} b (lit {type: (t-prim {} f32)} 2.0))
             (def {} c (if {} (var {} x) (var {} a) (var {} b)))",
            CheckErrorKind::PrecisionMismatch,
        );
    }

    // ── Pipe tests ───────────────────────────────────────────────

    #[test]
    fn pipe_simple() {
        check_ok(
            "(defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))
             (def {} f (fn {} (params {} x) (var {} x)))
             (def {} result (pipe {} (lit {type: (t-prim {} f32)} 1.0) (var {} f)))",
        );
    }

    #[test]
    fn pipe_chain() {
        check_ok(
            "(defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))
             (def {} f (fn {} (params {} x) (var {} x)))
             (def {} result (pipe {} (lit {type: (t-prim {} f32)} 1.0) (var {} f) (var {} f)))",
        );
    }

    // ── Tuple tests ──────────────────────────────────────────────

    #[test]
    fn tuple_creation() {
        check_ok(
            "(def {} t (tuple {} (lit {type: (t-prim {} int32)} 1) (lit {type: (t-prim {} f32)} 2.0)))",
        );
    }

    #[test]
    fn tuple_get_valid() {
        check_ok(
            "(def {} t (tuple {} (lit {type: (t-prim {} int32)} 1) (lit {type: (t-prim {} f32)} 2.0)))
             (def {} x (tuple-get {} (var {} t) 0))",
        );
    }

    #[test]
    fn tuple_get_out_of_bounds() {
        check_err(
            "(def {} t (tuple {} (lit {type: (t-prim {} int32)} 1)))
             (def {} x (tuple-get {} (var {} t) 5))",
            CheckErrorKind::TupleIndexOutOfBounds,
        );
    }

    // ── Cast tests ───────────────────────────────────────────────

    #[test]
    fn cast_tensor() {
        // Cast to a precision the Phase 0f tensor backend supports.
        // Reduced-float targets (bf16/f16/f64/f8e4m3) are rejected — see
        // `cast_tensor_rejects_unsupported_precision` below.
        check_ok(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} y (cast {} (var {} x) (t-prim {} int32)))",
        );
    }

    #[test]
    fn cast_prim() {
        check_ok(
            "(def {} x (lit {type: (t-prim {} int32)} 42))
             (def {} y (cast {} (var {} x) (t-prim {} f32)))",
        );
    }

    #[test]
    fn cast_accepts_alias_precision() {
        check_ok(
            "(typealias {} Floaty () (t-prim {} f32))
             (def {} x (lit {type: (t-prim {} int32)} 42))
             (def {} y (cast {} (var {} x) (t-adt {} Floaty)))",
        );
    }

    // ── Unsupported tensor precision tests ──────────────────────

    #[test]
    fn tensor_ascription_accepts_f64() {
        // v0.2.3: f64 tensors are now a first-class precision. The checker
        // accepts tensor[..., f64]; the C backend emits `double` arrays.
        check_ok("(def {} y (lit {type: (t-tensor {} (d-lit {} 4) (t-prim {} f64))} 0))");
    }

    #[test]
    fn tensor_ascription_accepts_f16() {
        // WS-0 lock cc47e6d: f16 is in the active dtype set per
        // spec/04-type-system.md §1.1 and must be admitted as a tensor
        // element type at check time. Backend coverage is staged
        // separately (WS-A1/A2/A3).
        check_ok("(def {} y (lit {type: (t-tensor {} (d-lit {} 4) (t-prim {} f16))} 0))");
    }

    #[test]
    fn tensor_ascription_accepts_bf16() {
        // WS-0 lock cc47e6d: bf16 is in the active dtype set per
        // spec/04-type-system.md §1.1 and must be admitted as a tensor
        // element type at check time. Backend coverage is staged
        // separately (WS-A1/A2/A3).
        check_ok("(def {} y (lit {type: (t-tensor {} (d-lit {} 4) (t-prim {} bf16))} 0))");
    }

    #[test]
    fn tensor_ascription_rejects_f8e4m3() {
        // f8e4m3 is deferred per spec/04-type-system.md §1.1.1 and must
        // be rejected at check time.
        check_err(
            "(def {} y (lit {type: (t-tensor {} (d-lit {} 4) (t-prim {} f8e4m3))} 0))",
            CheckErrorKind::UnsupportedTensorPrecision,
        );
    }

    #[test]
    fn tensor_ascription_accepts_int64() {
        // Integer tensor precisions remain valid.
        check_ok("(def {} y (lit {type: (t-tensor {} (d-lit {} 4) (t-prim {} int64))} 0))");
    }

    #[test]
    fn tensor_ascription_accepts_bool() {
        check_ok("(def {} y (lit {type: (t-tensor {} (d-lit {} 4) (t-prim {} bool))} false))");
    }

    #[test]
    fn cast_tensor_accepts_f64() {
        // v0.2.3: tensor-f64 is a valid cast target. The checker accepts
        // cast(tensor_f32, f64); the backend emits float→double conversion.
        check_ok(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} y (cast {} (var {} x) (t-prim {} f64)))",
        );
    }

    #[test]
    fn cast_tensor_accepts_bf16() {
        // WS-0 lock cc47e6d: cast(x: tensor[..., f32], bf16) is permitted
        // because bf16 is in the active tensor element set per
        // spec/04-type-system.md §1.1.
        check_ok(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} y (cast {} (var {} x) (t-prim {} bf16)))",
        );
    }

    #[test]
    fn cast_tensor_rejects_f8e4m3() {
        // f8e4m3 is deferred per spec/04-type-system.md §1.1.1; cast
        // targets must be rejected with the deferred-dtype diagnostic.
        check_err(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} y (cast {} (var {} x) (t-prim {} f8e4m3)))",
            CheckErrorKind::UnsupportedTensorPrecision,
        );
    }

    #[test]
    fn cast_prim_to_f64_is_allowed() {
        // Host scalar f64 is still valid — only tensor-precision f64 is banned.
        check_ok(
            "(def {} x (lit {type: (t-prim {} int32)} 42))
             (def {} y (cast {} (var {} x) (t-prim {} f64)))",
        );
    }

    #[test]
    fn surf_source_accepts_f64_tensor_ascription() {
        // v0.2.3: exercise the full surf → desugar → check pipeline to confirm
        // the user-facing syntax `(expr : tensor[4, f64])` is accepted.
        let decls = chelis_surf::parser::parse_str(
            "y = (to_tensor([1.0, 2.0, 3.0, 4.0]) : tensor[4, f64])",
        )
        .expect("surf parse");
        let exprs = chelis_surf::desugar::desugar_program(&decls);
        let result = infer_program(&exprs);
        assert!(
            !result
                .errors
                .iter()
                .any(|e| matches!(e.kind, CheckErrorKind::UnsupportedTensorPrecision)),
            "expected no UnsupportedTensorPrecision for f64 tensor ascription, got: {:?}",
            result.errors.iter().map(|e| &e.kind).collect::<Vec<_>>()
        );
    }

    #[test]
    fn surf_source_accepts_cast_to_f64_tensor() {
        // v0.2.3: exercise the full pipeline for `cast(tensor, f64)`.
        let decls =
            chelis_surf::parser::parse_str("y = cast(to_tensor([1.5]), f64)").expect("surf parse");
        let exprs = chelis_surf::desugar::desugar_program(&decls);
        let result = infer_program(&exprs);
        assert!(
            !result
                .errors
                .iter()
                .any(|e| matches!(e.kind, CheckErrorKind::UnsupportedTensorPrecision)),
            "expected no UnsupportedTensorPrecision for cast(tensor, f64), got: {:?}",
            result.errors.iter().map(|e| &e.kind).collect::<Vec<_>>()
        );
    }

    #[test]
    fn copy_accepts_tensor() {
        check_ok(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} y (copy {} (var {} x)))",
        );
    }

    #[test]
    fn copy_rejects_scalar() {
        check_err(
            "(def {} x (lit {type: (t-prim {} int32)} 42))
             (def {} y (copy {} (var {} x)))",
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn copy_rejects_unconstrained_generic() {
        check_err(
            "(def {} id (fn {} (params {} x) (copy {} (var {} x))))",
            CheckErrorKind::TypeMismatch,
        );
    }

    // ── Grad tests ───────────────────────────────────────────────

    #[test]
    fn grad_function() {
        check_ok(
            "(defsig {} loss (t-fn {} (t-prim {} f32) (t-prim {} f32)))
             (def {} loss (fn {} (params {} x) (var {} x)))
             (def {} g (grad {} (var {} loss)))",
        );
    }

    #[test]
    fn grad_non_float_param_is_ignored_by_default() {
        let exprs = chelis_deep::parser::parse_str(
            "(defsig {} f (t-fn {} (t-prim {} bool) (t-prim {} f32)))
             (def {} f (fn {} (params {} x) (lit {type: (t-prim {} f32)} 1.0)))
             (def {} g (grad {} (var {} f)))",
        )
        .unwrap();
        let checked = check_ir_program(&exprs).expect("IR check");
        let ty = checked.type_env().get("g").expect("g type");
        assert_eq!(
            chelis_deep::printer::print_canonical_flat(std::slice::from_ref(ty))
                .trim()
                .to_string(),
            "(t-fn {} (t-prim {} bool) (t-unit {}))"
        );
    }

    #[test]
    fn grad_over_all_float_field_adt_types_as_same_adt() {
        // chelis#520 D2 slice: an ADT whose fields are all float tensors
        // gets a field-wise gradient of the same constructor shape, so
        // `grad(f : Box -> f32) : Box -> Box`.
        let exprs = chelis_deep::parser::parse_str(
            "(deftype {} Box
                (variant {} Box
                    (field {} t (t-tensor {} (d-lit {} 2) (t-prim {} f32)))))
             (defsig {} f (t-fn {} (t-adt {} Box) (t-prim {} f32)))
             (def {} f (fn {} (params {} p) (lit {type: (t-prim {} f32)} 1.0)))
             (def {} g (grad {} (var {} f)))",
        )
        .unwrap();
        let checked = check_ir_program(&exprs).expect("IR check");
        let ty = checked.type_env().get("g").expect("g type");
        let printed = chelis_deep::printer::print_canonical_flat(std::slice::from_ref(ty))
            .trim()
            .to_string();
        assert_eq!(printed, "(t-fn {} (t-adt {} Box) (t-adt {} Box))");
    }

    #[test]
    fn grad_over_mixed_field_adt_stays_non_differentiable() {
        // chelis#520 D2 negative parity: a mixed struct (int field) is not
        // differentiable, so the default (no `wrt`) gradient payload is
        // unit, exactly as before the slice.
        let exprs = chelis_deep::parser::parse_str(
            "(deftype {} Mixed
                (variant {} Mixed
                    (field {} t (t-tensor {} (d-lit {} 2) (t-prim {} f32)))
                    (field {} n (t-prim {} int32))))
             (defsig {} f (t-fn {} (t-adt {} Mixed) (t-prim {} f32)))
             (def {} f (fn {} (params {} p) (lit {type: (t-prim {} f32)} 1.0)))
             (def {} g (grad {} (var {} f)))",
        )
        .unwrap();
        let checked = check_ir_program(&exprs).expect("IR check");
        let ty = checked.type_env().get("g").expect("g type");
        let printed = chelis_deep::printer::print_canonical_flat(std::slice::from_ref(ty))
            .trim()
            .to_string();
        assert_eq!(printed, "(t-fn {} (t-adt {} Mixed) (t-unit {}))");
    }

    #[test]
    fn grad_over_adt_plus_tensor_multi_target_returns_tuple() {
        // chelis#520 D2: an ADT gradient target is supported ALONGSIDE a
        // plain tensor argument (the closing bar). The default (no `wrt`)
        // gradient payload is the per-target tuple whose ADT slot is the
        // field-wise gradient struct and whose tensor slot is the bare
        // tensor gradient.
        let exprs = chelis_deep::parser::parse_str(
            "(deftype {} Box
                (variant {} Box
                    (field {} t (t-tensor {} (d-lit {} 2) (t-prim {} f32)))))
             (defsig {} f (t-fn {}
                (t-adt {} Box)
                (t-tensor {} (d-lit {} 2) (t-prim {} f32))
                (t-prim {} f32)))
             (def {} f (fn {} (params {} p y) (lit {type: (t-prim {} f32)} 1.0)))
             (def {} g (grad {} (var {} f)))",
        )
        .unwrap();
        let checked = check_ir_program(&exprs).expect("IR check");
        let ty = checked.type_env().get("g").expect("g type");
        let printed = chelis_deep::printer::print_canonical_flat(std::slice::from_ref(ty))
            .trim()
            .to_string();
        assert_eq!(
            printed,
            "(t-fn {} (t-adt {} Box) (t-tensor {} (d-lit {} 2) (t-prim {} f32)) (t-tuple {} (t-adt {} Box) (t-tensor {} (d-lit {} 2) (t-prim {} f32))))"
        );
    }

    #[test]
    fn grad_wrt_adt_in_multi_arg_call_returns_struct() {
        // chelis#520 D2: a `wrt`-restricted ADT target inside a
        // multi-argument function is supported; narrowing to the ADT alone
        // yields the bare field-wise gradient struct (a single target, so
        // no enclosing tuple).
        let exprs = chelis_deep::parser::parse_str(
            "(deftype {} Box
                (variant {} Box
                    (field {} t (t-tensor {} (d-lit {} 2) (t-prim {} f32)))))
             (defsig {} f (t-fn {}
                (t-adt {} Box)
                (t-tensor {} (d-lit {} 2) (t-prim {} f32))
                (t-prim {} f32)))
             (def {} f (fn {} (params {} p y) (lit {type: (t-prim {} f32)} 1.0)))
             (def {} g (grad {} (var {} f) (lit {type: (t-prim {} int32)} 0)))",
        )
        .unwrap();
        let checked = check_ir_program(&exprs).expect("IR check");
        let ty = checked.type_env().get("g").expect("g type");
        let printed = chelis_deep::printer::print_canonical_flat(std::slice::from_ref(ty))
            .trim()
            .to_string();
        assert_eq!(
            printed,
            "(t-fn {} (t-adt {} Box) (t-tensor {} (d-lit {} 2) (t-prim {} f32)) (t-adt {} Box))"
        );
    }

    #[test]
    fn grad_over_pure_enum_stays_unit() {
        // chelis#520 D2: a pure enum (no fields in any variant) carries
        // no continuous payload, so it stays non-differentiable and the
        // default gradient payload is unit, the pre-#520 typing.
        let exprs = chelis_deep::parser::parse_str(
            "(deftype {} Mode
                (variant {} ModeA)
                (variant {} ModeB))
             (defsig {} f (t-fn {} (t-adt {} Mode) (t-prim {} f32)))
             (def {} f (fn {} (params {} m) (lit {type: (t-prim {} f32)} 1.0)))
             (def {} g (grad {} (var {} f)))",
        )
        .unwrap();
        let checked = check_ir_program(&exprs).expect("IR check");
        let ty = checked.type_env().get("g").expect("g type");
        let printed = chelis_deep::printer::print_canonical_flat(std::slice::from_ref(ty))
            .trim()
            .to_string();
        assert_eq!(printed, "(t-fn {} (t-adt {} Mode) (t-unit {}))");
    }

    #[test]
    fn grad_over_enum_plus_tensor_keeps_tensor_only_payload() {
        // chelis#520 D2 regression guard: a nullary-enum parameter next
        // to a tensor parameter must not trip the single-argument ADT
        // rejection; the enum is non-differentiable (skipped), so the
        // gradient payload is the tensor alone, the pre-#520 typing.
        let exprs = chelis_deep::parser::parse_str(
            "(deftype {} Mode
                (variant {} ModeA)
                (variant {} ModeB))
             (defsig {} f (t-fn {}
                (t-adt {} Mode)
                (t-tensor {} (d-lit {} 2) (t-prim {} f32))
                (t-prim {} f32)))
             (def {} f (fn {} (params {} m x) (lit {type: (t-prim {} f32)} 1.0)))
             (def {} g (grad {} (var {} f)))",
        )
        .unwrap();
        let checked = check_ir_program(&exprs).expect("IR check");
        let ty = checked.type_env().get("g").expect("g type");
        let printed = chelis_deep::printer::print_canonical_flat(std::slice::from_ref(ty))
            .trim()
            .to_string();
        assert_eq!(
            printed,
            "(t-fn {} (t-adt {} Mode) (t-tensor {} (d-lit {} 2) (t-prim {} f32)) (t-tensor {} (d-lit {} 2) (t-prim {} f32)))"
        );
    }

    #[test]
    fn grad_rejects_non_scalar_output() {
        check_err(
            "(defsig {} f (t-fn {}
                (t-prim {} f32)
                (t-tensor {} (d-lit {} 2) (t-prim {} f32))))
             (def {} f
                (fn {} (params {} x)
                  (lit {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))} 1.0)))
             (def {} g (grad {} (var {} f)))",
            CheckErrorKind::Other,
        );
    }

    #[test]
    fn grad_with_explicit_wrt_returns_selected_gradient_only() {
        let exprs = chelis_deep::parser::parse_str(
            "(defsig {} loss
                (t-fn {}
                    (t-tensor {} (d-name {} features) (t-prim {} f32))
                    (t-tensor {} (d-name {} hidden) (t-prim {} f32))
                    (t-prim {} f32)))
             (def {} loss
                (fn {} (params {} x w)
                    (lit {type: (t-prim {} f32)} 1.0)))
             (def {} dw
                (grad {} (var {} loss) (lit {type: (t-prim {} int32)} 1)))",
        )
        .unwrap();
        let checked = check_ir_program(&exprs).expect("IR check");
        let ty = checked.type_env().get("dw").expect("dw type");
        assert_eq!(
            chelis_deep::printer::print_canonical_flat(std::slice::from_ref(ty))
                .trim()
                .to_string(),
            "(t-fn {} (t-tensor {} (d-name {} features) (t-prim {} f32)) (t-tensor {} (d-name {} hidden) (t-prim {} f32)) (t-tensor {} (d-name {} hidden) (t-prim {} f32)))"
        );
    }

    #[test]
    fn grad_with_multiple_wrt_returns_flat_tuple() {
        let exprs = chelis_deep::parser::parse_str(
            "(defsig {} loss
                (t-fn {}
                    (t-tensor {} (d-name {} features) (t-prim {} f32))
                    (t-tensor {} (d-name {} hidden) (t-prim {} f32))
                    (t-prim {} f32)))
             (def {} loss
                (fn {} (params {} x w)
                    (lit {type: (t-prim {} f32)} 1.0)))
             (def {} grads
                (grad {} (var {} loss)
                    (tuple {}
                        (lit {type: (t-prim {} int32)} 0)
                        (lit {type: (t-prim {} int32)} 1))))",
        )
        .unwrap();
        let checked = check_ir_program(&exprs).expect("IR check");
        let ty = checked.type_env().get("grads").expect("grads type");
        assert_eq!(
            chelis_deep::printer::print_canonical_flat(std::slice::from_ref(ty))
                .trim()
                .to_string(),
            "(t-fn {} (t-tensor {} (d-name {} features) (t-prim {} f32)) (t-tensor {} (d-name {} hidden) (t-prim {} f32)) (t-tuple {} (t-tensor {} (d-name {} features) (t-prim {} f32)) (t-tensor {} (d-name {} hidden) (t-prim {} f32))))"
        );
    }

    #[test]
    fn grad_rejects_nondifferentiable_explicit_wrt_target() {
        check_err(
            "(defsig {} f (t-fn {} (t-prim {} bool) (t-prim {} f32)))
             (def {} f (fn {} (params {} x) (lit {type: (t-prim {} f32)} 1.0)))
             (def {} g (grad {} (var {} f) (lit {type: (t-prim {} int32)} 0)))",
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn vmap_function_inserts_axis_zero_batch_dim() {
        check_ok(
            "(defsig {} process
                (t-fn {}
                    (t-tensor {} (d-name {} features) (t-prim {} f32))
                    (t-tensor {} (d-name {} features) (t-prim {} f32))))
             (def {} process
                (fn {} (params {} x)
                    (var {} x)))
             (defsig {} batch_process
                (t-fn {}
                    (t-tensor {} (d-name {} batch) (d-name {} features) (t-prim {} f32))
                    (t-tensor {} (d-name {} batch) (d-name {} features) (t-prim {} f32))))
             (def {} batch_process
                (vmap {} (var {} process) (lit {type: (t-prim {} int32)} 0)))",
        );
    }

    #[test]
    fn vmap_non_function_is_rejected() {
        check_err(
            "(def {} x (lit {type: (t-prim {} f32)} 1.0))
             (def {} y (vmap {} (var {} x) (lit {type: (t-prim {} int32)} 0)))",
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn vmap_axis_out_of_bounds_is_rejected() {
        check_err(
            "(defsig {} process
                (t-fn {}
                    (t-tensor {} (d-name {} features) (t-prim {} f32))
                    (t-tensor {} (d-name {} features) (t-prim {} f32))))
             (def {} process
                (fn {} (params {} x)
                    (var {} x)))
             (def {} batch_process
                (vmap {} (var {} process) (lit {type: (t-prim {} int32)} 2)))",
            CheckErrorKind::DimensionMismatch,
        );
    }

    #[test]
    fn vmap_grad_single_tensor_param_type_checks() {
        // WS-A7: pre-fix, the bare-arg `x` in `def loss(x) = sum(x, 0)` was
        // never seeded with the declared sig type (`tensor[features, f32]`)
        // before body inference, so `sum`'s reduction-signature check
        // early-returned an unconstrained result tvar (the input was still a
        // free `Var(_)`). The post-body sig-unify could then pin that
        // unconstrained tvar to the declared `f32` even though `sum` on a
        // rank-1 tensor produces `tensor[, f32]` per the spec — masking the
        // missing `tensor_to_scalar` coercion. Seeding bare params with the
        // declared sig types now exposes the rank-0 result, so the fixture
        // wraps the reduction in `tensor_to_scalar` to match the declared
        // scalar return.
        check_ok(
            "(defsig {} loss
                (t-fn {}
                    (t-tensor {} (d-name {} features) (t-prim {} f32))
                    (t-prim {} f32)))
             (def {} loss
                (fn {} (params {} x)
                    (app {type: (t-prim {} f32)} (var {} tensor_to_scalar)
                        (app {type: (t-tensor {} (t-prim {} f32))} (var {} sum)
                            (var {type: (t-tensor {} (d-name {} features) (t-prim {} f32))} x)
                            (lit {type: (t-prim {} int32)} 0))))
             )
             (defsig {} per_example_grad
                (t-fn {}
                    (t-tensor {} (d-name {} batch) (d-name {} features) (t-prim {} f32))
                    (t-tensor {} (d-name {} batch) (d-name {} features) (t-prim {} f32))))
             (def {} per_example_grad
                (vmap {} (grad {} (var {} loss)) (lit {type: (t-prim {} int32)} 0)))",
        );
    }

    #[test]
    fn vmap_grad_multiple_params_type_checks_with_tuple_result() {
        // WS-A7: see vmap_grad_single_tensor_param_type_checks — the same
        // rank-0 vs `Prim(f32)` distinction applies; wrap `sum` in
        // `tensor_to_scalar` to match the declared scalar return.
        check_ok(
            "(defsig {} loss
                (t-fn {}
                    (t-tensor {} (d-name {} features) (t-prim {} f32))
                    (t-tensor {} (d-name {} features) (t-prim {} f32))
                    (t-prim {} f32)))
             (def {} loss
                (fn {} (params {} x y)
                    (app {type: (t-prim {} f32)} (var {} tensor_to_scalar)
                        (app {type: (t-tensor {} (t-prim {} f32))} (var {} sum)
                            (app {type: (t-tensor {} (d-name {} features) (t-prim {} f32))} (var {} add)
                                (var {type: (t-tensor {} (d-name {} features) (t-prim {} f32))} x)
                                (var {type: (t-tensor {} (d-name {} features) (t-prim {} f32))} y))
                            (lit {type: (t-prim {} int32)} 0))))
             )
             (def {} per_example_grad
                (vmap {} (grad {} (var {} loss)) (lit {type: (t-prim {} int32)} 0)))",
        );
    }

    #[test]
    fn grad_of_vmap_is_rejected_for_non_scalar_output() {
        check_err(
            "(defsig {} loss
                (t-fn {}
                    (t-tensor {} (d-name {} features) (t-prim {} f32))
                    (t-prim {} f32)))
             (def {} loss
                (fn {} (params {} x)
                    (lit {type: (t-prim {} f32)} 1.0)))
             (def {} g
                (grad {} (vmap {} (var {} loss) (lit {type: (t-prim {} int32)} 0))))",
            CheckErrorKind::Other,
        );
    }

    // ── ADT tests ────────────────────────────────────────────────

    #[test]
    fn adt_deftype_and_construct() {
        check_ok(
            "(deftype {} MyOpt (a) (variant {} MySome (t-var {} a)) (variant {} MyNone))
             (def {} x (app {} (var {} MySome) (lit {type: (t-prim {} int32)} 42)))",
        );
    }

    #[test]
    fn adt_nullary_constructor() {
        check_ok(
            "(deftype {} MyOpt (a) (variant {} MySome (t-var {} a)) (variant {} MyNone))
             (def {} x (var {} MyNone))",
        );
    }

    #[test]
    fn adt_no_type_params() {
        check_ok(
            "(deftype {} Color () (variant {} Red) (variant {} Green) (variant {} Blue))
             (def {} c (var {} Red))",
        );
    }

    // ── Duplicate type-definition rejection ──────────────────────
    // The carrier-set in `linearity::compute_tensor_carrying_adts`
    // keys on bare ADT names, so silent last-write-wins on duplicate
    // `deftype`s would produce order-dependent borrow semantics. The
    // type checker rejects collisions at declaration time
    // (`CheckErrorKind::DuplicateDefinition`); the tests below pin
    // both sides of that rule.

    #[test]
    fn duplicate_deftype_in_same_program_is_rejected() {
        check_err(
            "(deftype {} Dup () (variant {} A))
             (deftype {} Dup () (variant {} B))",
            CheckErrorKind::DuplicateDefinition,
        );
    }

    #[test]
    fn distinct_deftypes_with_overlapping_variant_names_are_accepted() {
        // Two ADTs may share a variant name; only ADT-name collisions
        // are duplicates. This pins that the rejection is scoped to
        // the type name, not to constructor names.
        check_ok(
            "(deftype {} Lhs () (variant {} A))
             (deftype {} Rhs () (variant {} B))
             (def {} x (var {} A))
             (def {} y (var {} B))",
        );
    }

    #[test]
    fn deftype_colliding_with_prelude_option_is_rejected() {
        // `Option[a]` is registered by `register_prelude_adts` before
        // `collect_declarations` runs. User code re-declaring it would
        // overwrite the prelude entry under `HashMap::insert`.
        check_err(
            "(deftype {} Option (a) (variant {} Some (t-var {} a)) (variant {} None))",
            CheckErrorKind::DuplicateDefinition,
        );
    }

    #[test]
    fn deftype_then_typealias_with_same_name_is_rejected() {
        // `deftype` and `typealias` share the same type-name namespace
        // (both live in `AdtRegistry`). A later `typealias Holder = ...`
        // would silently overwrite an earlier `deftype Holder`.
        check_err(
            "(deftype {} Holder () (variant {} V))
             (typealias {} Holder () (t-prim {} int32))",
            CheckErrorKind::DuplicateDefinition,
        );
    }

    #[test]
    fn typealias_then_deftype_with_same_name_is_rejected() {
        check_err(
            "(typealias {} Holder () (t-prim {} int32))
             (deftype {} Holder () (variant {} V))",
            CheckErrorKind::DuplicateDefinition,
        );
    }

    #[test]
    fn duplicate_typealias_is_rejected() {
        check_err(
            "(typealias {} Alias () (t-prim {} int32))
             (typealias {} Alias () (t-prim {} f32))",
            CheckErrorKind::DuplicateDefinition,
        );
    }

    // ── Duplicate def rejection (chelis#258) ─────────────────────
    // A def's value binding is silent last-write-wins, and Chelis does
    // not dispatch same-name defs by arg arity or tensor rank. Two
    // same-name defs (e.g. rank-distinct "overloads") therefore leave
    // one arm unreachable and surface a confusing DimensionMismatch at
    // the other arm's call sites. `report_duplicate_defs` rejects them
    // at the definition site instead. The tests below pin both sides.

    #[test]
    fn duplicate_def_in_same_program_is_rejected() {
        check_err(
            "(def {} f (fn {} (params {} x) (var {} x)))
             (def {} f (fn {} (params {} y) (var {} y)))",
            CheckErrorKind::DuplicateDefinition,
        );
    }

    #[test]
    fn duplicate_value_def_in_same_program_is_rejected() {
        // The rule keys on the `def` tag, so duplicate value defs collide too.
        check_err(
            "(def {} x (lit {type: (t-prim {} int32)} 1))
             (def {} x (lit {type: (t-prim {} int32)} 2))",
            CheckErrorKind::DuplicateDefinition,
        );
    }

    #[test]
    fn distinct_name_defs_are_accepted() {
        // Only same-name collisions are duplicates; distinct names are fine.
        check_ok(
            "(def {} f (fn {} (params {} x) (var {} x)))
             (def {} g (fn {} (params {} y) (var {} y)))",
        );
    }

    #[test]
    fn sig_plus_def_same_name_is_not_a_duplicate() {
        // A `defsig` + a `def` for one name is the ordinary annotated-def
        // shape (and an inline-annotated def desugars to exactly that pair),
        // so it must not be flagged. Only two `def`s for one name collide.
        check_ok(
            "(defsig {} f (t-fn {} (t-var {} a) (t-var {} a)))
             (def {} f (fn {} (params {} x) (var {} x)))",
        );
    }

    #[test]
    fn duplicate_defsig_conflicting_order_a_is_rejected() {
        check_err(
            "(defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))
             (defsig {} f (t-fn {} (t-prim {} bool) (t-prim {} f32)))
             (def {} f (fn {} (params {} x) (var {} x)))",
            CheckErrorKind::DuplicateDefinition,
        );
    }

    #[test]
    fn duplicate_defsig_conflicting_order_b_is_rejected() {
        check_err(
            "(defsig {} f (t-fn {} (t-prim {} bool) (t-prim {} f32)))
             (defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))
             (def {} f (fn {} (params {} x) (var {} x)))",
            CheckErrorKind::DuplicateDefinition,
        );
    }

    #[test]
    fn duplicate_defsig_identical_signature_is_rejected() {
        check_err(
            "(defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))
             (defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))
             (def {} f (fn {} (params {} x) (var {} x)))",
            CheckErrorKind::DuplicateDefinition,
        );
    }

    // ── Match tests ──────────────────────────────────────────────

    #[test]
    fn match_simple_adt() {
        check_ok(
            "(deftype {} MyOpt (a) (variant {} MySome (t-var {} a)) (variant {} MyNone))
             (def {} x (app {} (var {} MySome) (lit {type: (t-prim {} int32)} 42)))
             (def {} result
               (match {} (var {} x)
                 (arm {} (pat-ctor {} MySome (pat-var {} v)) () (var {} v))
                 (arm {} (pat-ctor {} MyNone) () (lit {type: (t-prim {} int32)} 0))))",
        );
    }

    #[test]
    fn match_non_exhaustive() {
        check_err(
            "(deftype {} MyOpt (a) (variant {} MySome (t-var {} a)) (variant {} MyNone))
             (def {} x (app {} (var {} MySome) (lit {type: (t-prim {} int32)} 42)))
             (def {} result
               (match {} (var {} x)
                 (arm {} (pat-ctor {} MySome (pat-var {} v)) () (var {} v))))",
            CheckErrorKind::NonExhaustiveMatch,
        );
    }

    // ── Defsig tests ─────────────────────────────────────────────

    #[test]
    fn defsig_fn_signature() {
        check_ok(
            "(defsig {} double (t-fn {} (t-prim {} f32) (t-prim {} f32)))
             (def {} double (fn {} (params {} x) (var {} x)))",
        );
    }

    #[test]
    fn defsig_tensor_signature() {
        check_ok(
            "(defsig {} normalize_fn
               (t-fn {} (t-tensor {} (d-name {} batch) (t-prim {} f32))
                        (t-tensor {} (d-name {} batch) (t-prim {} f32))))
             (def {} normalize_fn (fn {} (params {} x) (var {} x)))",
        );
    }

    #[test]
    fn typealias_zero_param_resolves_in_defsig() {
        check_ok(
            "(typealias {} Scalar () (t-prim {} f32))
             (defsig {} id (t-fn {} (t-adt {} Scalar) (t-adt {} Scalar)))
             (def {} id (fn {} (params {} x) (var {} x)))",
        );
    }

    #[test]
    fn typealias_parameterized_resolves_in_defsig() {
        check_ok(
            "(typealias {} Boxed (a) (t-tuple {} (t-var {} a)))
             (defsig {} wrap (t-fn {} (t-adt {} Boxed (t-prim {} f32)) (t-adt {} Boxed (t-prim {} f32))))
             (def {} wrap (fn {} (params {} x) (var {} x)))",
        );
    }

    #[test]
    fn typealias_resolves_in_typed_param_metadata() {
        check_ok(
            "(typealias {} Scalar () (t-prim {} f32))
             (def {} id (fn {} (params {} (x {type: (t-adt {} Scalar)})) (var {} x)))",
        );
    }

    #[test]
    fn typealias_resolves_in_literal_metadata() {
        check_ok(
            "(typealias {} Scalar () (t-prim {} f32))
             (def {} x (lit {type: (t-adt {} Scalar)} 1.0))",
        );
    }

    // ── Transparent-alias-in-constructor-field tests ──────────────
    //
    // Per spec/02-surf-syntax.md ("Aliases are transparent — expanded
    // during desugaring"), a `deftype` field declared with a transparent
    // alias must unify against the alias expansion. The Hull.Ast scenario
    // that motivated this is `type EffectRow = List[Effect]` used in
    // `type Type = ... | TArrow(Type, Type, EffectRow) | ...`; constructing
    // `TArrow(a, b, [])` must not report `EffectRow vs List`.

    /// Surf source -> desugar -> IR check. Returns the collected check
    /// errors (empty on success). Parse failures panic — the source is
    /// the test's own fixture, not user input under test.
    fn surf_check_errors(src: &str) -> Vec<CheckError> {
        let decls = chelis_surf::parser::parse_str(src).expect("surf parse");
        let exprs = chelis_surf::desugar::desugar_program(&decls);
        match check_ir_program(&exprs) {
            Ok(_) => Vec::new(),
            Err(result) => result.errors,
        }
    }

    fn assert_surf_ok(src: &str) {
        let errors = surf_check_errors(src);
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
    }

    #[test]
    fn alias_typed_ctor_field_constructs_with_list_value() {
        // EffectRow = List[Effect] declared AFTER the deftype that uses it
        // (forward reference). Constructing TArrow(a, b, e) where the third
        // field is the alias must type-check: the field expands to
        // List[Effect] and unifies with the EffectRow-typed argument.
        assert_surf_ok(
            "module M\n\
             export (mk)\n\
             type Effect =\n\
               | Pure\n\
               | Impure\n\
             type Ty =\n\
               | TBase\n\
               | TArrow(Ty, Ty, EffectRow)\n\
             type EffectRow = List[Effect]\n\
             def mk(a: Ty, b: Ty, e: EffectRow) -> Ty = TArrow(a, b, e)\n",
        );
    }

    #[test]
    fn alias_typed_ctor_field_constructs_with_empty_list_literal() {
        // Constructing TArrow(a, b, []) where the third field is the alias
        // must type-check: [] is List[Effect], the field expands to
        // List[Effect].
        assert_surf_ok(
            "module M\n\
             export (mk)\n\
             type Effect =\n\
               | Pure\n\
               | Impure\n\
             type Ty =\n\
               | TBase\n\
               | TArrow(Ty, Ty, EffectRow)\n\
             type EffectRow = List[Effect]\n\
             def mk(a: Ty, b: Ty) -> Ty = TArrow(a, b, [])\n",
        );
    }

    #[test]
    fn def_returning_option_tuple_with_alias_field_type_checks() {
        // The spec §3 pattern `Some((Ctor(...), []))` returning
        // Option[(Ty, EffectRow)].
        assert_surf_ok(
            "module M\n\
             export (mk)\n\
             type Effect =\n\
               | Pure\n\
               | Impure\n\
             type Ty =\n\
               | TBase\n\
               | TArrow(Ty, Ty, EffectRow)\n\
             type EffectRow = List[Effect]\n\
             def mk(a: Ty, b: Ty) -> Option[(Ty, EffectRow)] = Some((TArrow(a, b, []), []))\n",
        );
    }

    #[test]
    fn two_level_alias_in_ctor_field_resolves() {
        // Alias of an alias: Effects = EffectRow = List[Effect].
        assert_surf_ok(
            "module M\n\
             export (mk)\n\
             type Effect =\n\
               | Pure\n\
               | Impure\n\
             type Ty =\n\
               | TBase\n\
               | TArrow(Ty, Ty, Effects)\n\
             type EffectRow = List[Effect]\n\
             type Effects = EffectRow\n\
             def mk(a: Ty, b: Ty) -> Ty = TArrow(a, b, [])\n",
        );
    }

    #[test]
    fn ctx_list_of_tuple_alias_used_in_value_position() {
        // Ctx = List[(String, Ty)] -- a tuple-bearing alias used as a
        // constructor field; constructing with an empty list must work.
        assert_surf_ok(
            "module M\n\
             export (mk)\n\
             type Ty =\n\
               | TBase\n\
               | TArrow(Ty, Ty)\n\
             type Ctx = List[(String, Ty)]\n\
             type Judgement =\n\
               | Judge(Ctx, Ty)\n\
             def mk(t: Ty) -> Judgement = Judge([], t)\n",
        );
    }

    #[test]
    fn genuine_mismatch_against_expanded_alias_still_rejected() {
        // NEGATIVE PARITY: passing an Int where the expanded alias is
        // List[Effect] must STILL be a TypeMismatch. Alias transparency
        // must not weaken genuine error detection.
        let errors = surf_check_errors(
            "module M\n\
             export (mk)\n\
             type Effect =\n\
               | Pure\n\
               | Impure\n\
             type Ty =\n\
               | TBase\n\
               | TArrow(Ty, Ty, EffectRow)\n\
             type EffectRow = List[Effect]\n\
             def mk(a: Ty, b: Ty) -> Ty = TArrow(a, b, 5)\n",
        );
        assert!(
            errors
                .iter()
                .any(|e| matches!(e.kind, CheckErrorKind::TypeMismatch)),
            "expected a TypeMismatch for Int passed to a List[Effect] field, got: {errors:?}"
        );
    }

    #[test]
    fn alias_field_rejects_wrong_list_element_type() {
        // NEGATIVE PARITY: List[Ty] where the field expands to
        // List[Effect] must still mismatch on the element type.
        let errors = surf_check_errors(
            "module M\n\
             export (mk)\n\
             type Effect =\n\
               | Pure\n\
               | Impure\n\
             type Ty =\n\
               | TBase\n\
               | TArrow(Ty, Ty, EffectRow)\n\
             type EffectRow = List[Effect]\n\
             def mk(a: Ty, b: Ty, ts: List[Ty]) -> Ty = TArrow(a, b, ts)\n",
        );
        assert!(
            errors
                .iter()
                .any(|e| matches!(e.kind, CheckErrorKind::TypeMismatch)),
            "expected a TypeMismatch for List[Ty] passed to a List[Effect] field, got: {errors:?}"
        );
    }

    // ── Partial inference tests ──────────────────────────────────

    #[test]
    fn partial_inference_continues_after_error() {
        // First def has an error, second should still be processed
        let result = check(
            "(def {} x (var {} nonexistent))
             (def {} y (lit {type: (t-prim {} int32)} 42))",
        );
        assert!(!result.errors.is_empty(), "expected at least one error");
        // y should still have been typed
        assert!(result.typed_nodes > 0, "expected some typed nodes");
    }

    #[test]
    fn multiple_errors_collected() {
        let result = check(
            "(def {} x (var {} unknown1))
             (def {} y (var {} unknown2))",
        );
        assert!(
            result.errors.len() >= 2,
            "expected at least 2 errors, got {}",
            result.errors.len()
        );
    }

    // ── Builtin operation tests ──────────────────────────────────

    #[test]
    fn builtin_neg() {
        check_ok(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} neg) (var {} x)))",
        );
    }

    #[test]
    fn builtin_exp() {
        check_ok(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} exp) (var {} x)))",
        );
    }

    #[test]
    fn builtin_matmul() {
        check_ok(
            "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} b (lit {type: (t-tensor {} (d-name {} hidden) (d-name {} classes) (t-prim {} f32))} 0))
             (def {} c (app {type: (t-tensor {} (d-name {} batch) (d-name {} classes) (t-prim {} f32))}
                 (var {} matmul) (var {} a) (var {} b)))",
        );
    }

    #[test]
    fn builtin_batched_matmul_rank4() {
        check_ok(
            "(def {} q (lit {type: (t-tensor {} (d-name {} batch) (d-name {} head) (d-name {} seq) (d-name {} dim) (t-prim {} f32))} 0))
             (def {} k (lit {type: (t-tensor {} (d-name {} batch) (d-name {} head) (d-name {} dim) (d-name {} seq) (t-prim {} f32))} 0))
             (def {} scores (app {type: (t-tensor {} (d-name {} batch) (d-name {} head) (d-name {} seq) (d-name {} seq) (t-prim {} f32))}
                 (var {} matmul) (var {} q) (var {} k)))",
        );
    }

    #[test]
    fn builtin_batched_matmul_rejects_incompatible_leading_dim() {
        check_err(
            "(def {} q (lit {type: (t-tensor {} (d-name {} batch) (d-name {} head) (d-name {} seq) (d-name {} dim) (t-prim {} f32))} 0))
             (def {} k (lit {type: (t-tensor {} (d-name {} other_batch) (d-name {} head) (d-name {} dim) (d-name {} seq) (t-prim {} f32))} 0))
             (def {} scores (app {type: (t-tensor {} (d-name {} batch) (d-name {} head) (d-name {} seq) (d-name {} seq) (t-prim {} f32))}
                 (var {} matmul) (var {} q) (var {} k)))",
            CheckErrorKind::DimensionMismatch,
        );
    }

    #[test]
    fn builtin_layer_norm() {
        check_ok(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} gamma (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} beta (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} layer_norm) (var {} x) (var {} gamma) (var {} beta)))",
        );
    }

    #[test]
    fn builtin_layer_norm_rejects_rank2_gamma() {
        check_err(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} gamma (lit {type: (t-tensor {} (d-name {} hidden) (d-name {} extra) (t-prim {} f32))} 0))
             (def {} beta (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} layer_norm) (var {} x) (var {} gamma) (var {} beta)))",
            CheckErrorKind::DimensionMismatch,
        );
    }

    #[test]
    fn builtin_layer_norm_rejects_precision_mismatch() {
        check_err(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} gamma (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} bf16))} 0))
             (def {} beta (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} layer_norm) (var {} x) (var {} gamma) (var {} beta)))",
            CheckErrorKind::PrecisionMismatch,
        );
    }

    #[test]
    fn builtin_conv2d_accepts_int_stride_padding() {
        check_ok(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} in_c) (d-name {} h) (d-name {} w) (t-prim {} f32))} 0))
             (def {} k (lit {type: (t-tensor {} (d-name {} out_c) (d-name {} in_c) (d-lit {} 3) (d-lit {} 3) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} conv2d) (var {} x) (var {} k) (lit {type: (t-prim {} int32)} 1) (lit {type: (t-prim {} int32)} 1)))",
        );
    }

    #[test]
    fn builtin_conv2d_rejects_channel_mismatch() {
        check_err(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} in_a) (d-name {} h) (d-name {} w) (t-prim {} f32))} 0))
             (def {} k (lit {type: (t-tensor {} (d-name {} out_c) (d-name {} in_b) (d-lit {} 3) (d-lit {} 3) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} conv2d) (var {} x) (var {} k) (lit {type: (t-prim {} int32)} 1) (lit {type: (t-prim {} int32)} 1)))",
            CheckErrorKind::DimensionMismatch,
        );
    }

    #[test]
    fn builtin_conv2d_rejects_kernel_precision_mismatch() {
        check_err(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} in_c) (d-name {} h) (d-name {} w) (t-prim {} f32))} 0))
             (def {} k (lit {type: (t-tensor {} (d-name {} out_c) (d-name {} in_c) (d-lit {} 3) (d-lit {} 3) (t-prim {} bf16))} 0))
             (def {} y (app {} (var {} conv2d) (var {} x) (var {} k) (lit {type: (t-prim {} int32)} 1) (lit {type: (t-prim {} int32)} 1)))",
            CheckErrorKind::PrecisionMismatch,
        );
    }

    #[test]
    fn builtin_relu() {
        check_ok(
            "(def {} x (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} relu) (var {} x)))",
        );
    }

    // ── Tensor type tests ────────────────────────────────────────

    #[test]
    fn tensor_2d() {
        check_ok(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))} 0))",
        );
    }

    #[test]
    fn tensor_literal_dim() {
        check_ok("(def {} x (lit {type: (t-tensor {} (d-lit {} 512) (t-prim {} f32))} 0))");
    }

    // ── Edge cases ───────────────────────────────────────────────

    #[test]
    fn empty_program() {
        let result = check("");
        assert!(result.errors.is_empty());
        assert_eq!(result.typed_nodes, 0);
        assert_eq!(result.total_nodes, 0);
    }

    #[test]
    fn def_with_fn_body() {
        check_ok(
            "(def {} double (fn {} (params {} x) (app {} (var {} add) (var {} x) (var {} x))))",
        );
    }

    #[test]
    fn nested_let() {
        check_ok(
            "(def {} result
               (let {} (bind {} x (lit {type: (t-prim {} int32)} 1))
                 (let {} (bind {} y (var {} x))
                   (var {} y))))",
        );
    }

    #[test]
    fn if_with_tensor_branches() {
        check_ok(
            "(def {} cond (lit {type: (t-prim {} bool)} true))
             (def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} b (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} c (if {} (var {} cond) (var {} a) (var {} b)))",
        );
    }

    #[test]
    fn fn_applied_to_args() {
        check_ok(
            "(def {} f (fn {} (params {} x) (var {} x)))
             (def {} result (app {} (var {} f) (lit {type: (t-prim {} int32)} 42)))",
        );
    }

    #[test]
    fn adt_with_fields() {
        check_ok(
            "(deftype {} Pair ()
               (variant {} MkPair
                 (field {} fst (t-prim {} int32))
                 (field {} snd (t-prim {} f32))))
             (def {} p
               (record {}
                 (var {} MkPair)
                 (kv {} fst (lit {type: (t-prim {} int32)} 1))
                 (kv {} snd (lit {type: (t-prim {} f32)} 2.0))))",
        );
    }

    #[test]
    fn pipe_with_lambda() {
        check_ok(
            "(def {} result
               (pipe {} (lit {type: (t-prim {} f32)} 1.0)
                        (fn {} (params {} x) (var {} x))))",
        );
    }

    #[test]
    fn total_nodes_counted() {
        let result = check("(def {} x (lit {type: (t-prim {} int32)} 42))");
        assert!(result.total_nodes > 0, "expected some total nodes");
    }

    #[test]
    fn tuple_three_elems() {
        check_ok(
            "(def {} t (tuple {}
               (lit {type: (t-prim {} int32)} 1)
               (lit {type: (t-prim {} f32)} 2.0)
               (lit {type: (t-prim {} bool)} true)))",
        );
    }

    // ── Regression tests for bug fixes ──────────────────────────────

    // Fix 1: scalar arithmetic accepts matching numeric scalars but still rejects bad mixes
    #[test]
    fn fix1_tensor_op_rejects_non_tensor_args() {
        check_err(
            "(def {} r (app {} (var {} add) (lit {type: (t-prim {} int32)} 1) (lit {type: (t-prim {} bool)} true)))",
            CheckErrorKind::PrecisionMismatch,
        );
    }

    // Fix 2: unsound generalization — fn x -> { y = x; (y 1, y true) } should fail
    #[test]
    fn fix2_unsound_generalization_rejected() {
        // x is a monomorphic param, y = x so y is also monomorphic.
        // Applying y to both int32 and bool should fail.
        check_err(
            "(def {} test \
               (fn {} (params {} x) \
                 (let {} (bind {} y (var {} x)) \
                   (tuple {} \
                     (app {} (var {} y) (lit {type: (t-prim {} int32)} 1)) \
                     (app {} (var {} y) (lit {type: (t-prim {} bool)} true))))))",
            CheckErrorKind::PrecisionMismatch,
        );
    }

    // Fix 3: defsig not enforced — body must match declared signature
    #[test]
    fn fix3_defsig_enforced() {
        check_err(
            "(defsig {} f (t-fn {} (t-prim {} int32) (t-prim {} int32))) \
             (def {} f (lit {type: (t-prim {} bool)} true))",
            CheckErrorKind::TypeMismatch,
        );
    }

    // Fix 4: d-var names shared within a type — same d-var name maps to same DimVar
    #[test]
    fn fix4_dvar_names_shared() {
        // Declare a function requiring same dim 'a' in both args.
        // Call with tensor[batch,f32] and tensor[seq,f32] — should fail.
        check_err(
            "(defsig {} myfn \
               (t-fn {} \
                 (t-tensor {} (d-var {} a) (t-prim {} f32)) \
                 (t-tensor {} (d-var {} a) (t-prim {} f32)) \
                 (t-tensor {} (d-var {} a) (t-prim {} f32)))) \
             (def {} myfn (fn {} (params {} x y) (var {} x))) \
             (def {} result (app {} (var {} myfn) \
               (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0) \
               (lit {type: (t-tensor {} (d-name {} seq) (t-prim {} f32))} 0)))",
            CheckErrorKind::DimensionMismatch,
        );
    }

    // Fix 5: if condition must be bool
    #[test]
    fn fix5_if_condition_must_be_bool() {
        check_err(
            "(if {} (lit {type: (t-prim {} int32)} 0) \
                    (lit {type: (t-prim {} int32)} 1) \
                    (lit {type: (t-prim {} int32)} 2))",
            CheckErrorKind::TypeMismatch,
        );
    }

    // Fix 6a: wildcard satisfies exhaustiveness
    #[test]
    fn fix6a_wildcard_exhaustive() {
        check_ok(
            "(deftype {} MyOpt (a) (variant {} MySome (t-var {} a)) (variant {} MyNone)) \
             (def {} x (app {} (var {} MySome) (lit {type: (t-prim {} int32)} 42))) \
             (def {} result \
               (match {} (var {} x) \
                 (arm {} (pat-wild {}) () (lit {type: (t-prim {} int32)} 0))))",
        );
    }

    // Fix 6b: pat-as binds name
    #[test]
    fn fix6b_pat_as_binds_name() {
        check_ok(
            "(deftype {} MyOpt (a) (variant {} MySome (t-var {} a)) (variant {} MyNone)) \
             (def {} x (app {} (var {} MySome) (lit {type: (t-prim {} int32)} 42))) \
             (def {} result \
               (match {} (var {} x) \
                 (arm {} (pat-as {} whole (pat-wild {})) () (var {} whole))))",
        );
    }

    // Fix 7: fitness report includes unresolved names
    #[test]
    fn fix7_fitness_unresolved_names() {
        let exprs = chelis_deep::parser::parse_str("(def {} x (var {} unknown))").unwrap();
        let result = crate::infer::infer_program(&exprs);
        let report = crate::fitness::FitnessReport::from_infer_result(&result);
        assert!(
            report.unresolved_names.contains(&"unknown".to_string()),
            "expected 'unknown' in unresolved_names, got {:?}",
            report.unresolved_names
        );
        // Check severity is set
        assert!(report.errors.iter().all(|e| e.severity > 0.0));
    }

    // Fix 7b: suggestions populated for UnboundVariable
    #[test]
    fn fix7b_suggestions_for_unbound() {
        let result = check("(def {} x (var {} typo))");
        let unbound_err = result
            .errors
            .iter()
            .find(|e| matches!(e.kind, CheckErrorKind::UnboundVariable))
            .expect("expected UnboundVariable error");
        assert!(
            !unbound_err.suggestions.is_empty(),
            "expected suggestions for unbound variable"
        );
    }

    #[test]
    fn ir_literal_dimension_mismatch_surfaces_error() {
        let decls = chelis_surf::parser::parse_str(
            "def want_2x2(a: tensor[2, 2, f32]) -> f32 = trace(a, 0, 1)\n\
             def main(a: tensor[3, 3, f32]) -> f32 = want_2x2(a)\n",
        )
        .expect("surf parse");
        let exprs = chelis_surf::desugar::desugar_program(&decls);

        let result = infer_ir_program(&exprs);
        assert!(
            result
                .errors
                .iter()
                .any(|error| matches!(error.kind, CheckErrorKind::DimensionMismatch)),
            "expected ir inference to preserve literal dimension mismatches, got {:?}",
            result.errors
        );
    }

    #[test]
    fn ir_rejects_polymorphic_dims_pinned_by_body() {
        let decls = chelis_surf::parser::parse_str(
            "def want_2x2(a: tensor[2, 2, f32]) -> f32 = trace(a, 0, 1)\n\
             def bad_consumer[m, n](a: tensor[m, n, f32]) -> f32 = want_2x2(a)\n\
             def main() -> f32 = cast(0.0, f32)\n",
        )
        .expect("surf parse");
        let exprs = chelis_surf::desugar::desugar_program(&decls);

        let result = infer_ir_program(&exprs);
        assert!(
            result
                .errors
                .iter()
                .any(|error| matches!(error.kind, CheckErrorKind::DimensionMismatch)),
            "expected ir inference to reject polymorphic dims forced to literals by the body, got {:?}",
            result.errors
        );
    }

    #[test]
    fn ir_preserves_unresolved_name_errors() {
        let decls = chelis_surf::parser::parse_str(
            "def probe(x: f32) -> f32 = sub(x, frobnicate(x))\n\
             def main() -> f32 = probe(cast(1.0, f32))\n",
        )
        .expect("surf parse");
        let exprs = chelis_surf::desugar::desugar_program(&decls);

        let result = infer_ir_program(&exprs);
        assert!(
            result
                .errors
                .iter()
                .any(|error| matches!(error.kind, CheckErrorKind::UnboundVariable)),
            "expected ir inference to preserve unresolved-name errors, got {:?}",
            result.errors
        );

        let report = crate::fitness::check_ir_program(&exprs);
        assert!(
            report.unresolved_names.contains(&"frobnicate".to_string()),
            "expected ir fitness report to include unresolved frobnicate, got {:?}",
            report.unresolved_names
        );
    }

    // chelis#317: an in-scope constructor resolves through its *exact*
    // (consistently mangled) name. Before #317 this test fed a half-mangled
    // program — a bare `deftype KVCache` plus a `Pkg__..__KVCache` reference —
    // and relied on the registry's fuzzy terminal-segment fallback to bind the
    // two. That fuzzy bind is exactly the cross-module mis-resolution #317
    // removes, so the program the real reef pipeline produces (deftype AND
    // reference carry the same mangled name) is the one that must resolve. Both
    // the construction site (`None => KVCache([])`) and the type annotations
    // resolve against the same key without any terminal-match fuzziness.
    #[test]
    fn ir_resolves_consistently_mangled_constructor_names() {
        let decls = chelis_surf::parser::parse_str(
            "type Pkg__chelis__std__Std__Nn__Generate__KVCache[a] = \
                | Pkg__chelis__std__Std__Nn__Generate__KVCache(List[a])\n\
             def keep_cache[p](cache: \
                 Option[Pkg__chelis__std__Std__Nn__Generate__KVCache[p]]) -> \
                 Pkg__chelis__std__Std__Nn__Generate__KVCache[p] =\n\
               match cache with {\n\
                 | Some(value) => value\n\
                 | None => Pkg__chelis__std__Std__Nn__Generate__KVCache([])\n\
               }\n",
        )
        .expect("surf parse");
        let exprs = chelis_surf::desugar::desugar_program(&decls);

        let result = infer_ir_program(&exprs);
        assert!(
            !result.errors.iter().any(|error| matches!(
                error.kind,
                CheckErrorKind::UnboundVariable | CheckErrorKind::UnknownConstructor
            )),
            "expected the consistently mangled constructor to resolve clean, got {:?}",
            result.errors
        );
    }

    // chelis#317 negative parity: a bare constructor referenced against a
    // registry that only holds a *different* (mangled) same-terminal name is
    // out of scope and must be an `UnknownConstructor` error — not a silent
    // fuzzy bind to the foreign tag that defers to a runtime non-exhaustive
    // match. This is the half-mangled state the old
    // `ir_resolves_unique_terminal_constructor_names` test accepted.
    #[test]
    fn ir_rejects_out_of_scope_terminal_constructor_name() {
        let decls = chelis_surf::parser::parse_str(
            "type Pkg__chelis__std__Std__Nn__Generate__KVCache[a] = \
                | Pkg__chelis__std__Std__Nn__Generate__KVCache(List[a])\n\
             def make_cache[p]() -> \
                 Pkg__chelis__std__Std__Nn__Generate__KVCache[p] = KVCache([])\n",
        )
        .expect("surf parse");
        let exprs = chelis_surf::desugar::desugar_program(&decls);

        let result = infer_ir_program(&exprs);
        assert!(
            result.errors.iter().any(|error| matches!(
                error.kind,
                CheckErrorKind::UnknownConstructor
            ) && error.message.contains("KVCache")),
            "expected an UnknownConstructor for the out-of-scope bare `KVCache`, got {:?}",
            result.errors
        );
    }

    // Fix 8: typed params in Deep
    #[test]
    fn fix8_typed_params() {
        // fn with typed param x: f32 — using x should give f32
        check_ok("(def {} f (fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x)))");
    }

    // Fix 8b: typed param enforces type
    #[test]
    fn fix8b_typed_param_enforced() {
        // Param x is f32, so mixing it with a bool in arithmetic must fail.
        check_err(
            "(def {} f (fn {} (params {} (x {type: (t-prim {} f32)})) \
               (app {} (var {} add) (var {} x) (lit {type: (t-prim {} bool)} true))))",
            CheckErrorKind::PrecisionMismatch,
        );
    }

    // ── Round 3 regression tests ──────────────────────────────────

    #[test]
    fn fix9_logical_ops_reject_non_bool_tensors() {
        // and(tensor[batch, f32], tensor[batch, f32]) should fail — requires bool
        check_err(
            "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0)) \
             (def {} b (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0)) \
             (def {} r (app {} (var {} and) (var {} a) (var {} b)))",
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn fix9b_logical_ops_reject_non_tensor() {
        // and(int32, int32) should fail — requires tensor
        check_err(
            "(def {} r (app {} (var {} and) (lit {type: (t-prim {} int32)} 1) (lit {type: (t-prim {} int32)} 2)))",
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn fix9c_not_rejects_non_bool_tensor() {
        // not(tensor[batch, f32]) should fail — requires bool
        check_err(
            "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0)) \
             (def {} r (app {} (var {} not) (var {} a)))",
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn fix9d_logical_ops_accept_bool_tensors() {
        // and(tensor[batch, bool], tensor[batch, bool]) should pass
        check_ok(
            "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} bool))} 0)) \
             (def {} b (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} bool))} 0)) \
             (def {} r (app {} (var {} and) (var {} a) (var {} b)))",
        );
    }

    #[test]
    fn fix10_fitness_has_untyped_nodes() {
        let result = check(
            "(def {} good (lit {type: (t-prim {} int32)} 42)) \
                            (def {} bad (var {} nope))",
        );
        let report = crate::fitness::FitnessReport::from_infer_result(&result);
        assert!(report.untyped_nodes > 0, "expected untyped_nodes > 0");
        // Errors should have severity set
        assert!(
            report.errors.iter().all(|e| e.severity > 0.0),
            "all errors should have severity > 0"
        );
    }

    // ── Round 4 regression tests ──────────────────────────────────

    #[test]
    fn fix11_pat_record_rejects_unknown_field() {
        // Define Adam with field lr, then match on nonexistent field 'nope'
        check_err(
            "(deftype {} Optimizer () \
               (variant {} Adam (field {} lr (t-prim {} f32)))) \
             (def {} x (lit {type: (t-adt {} Optimizer)} 0)) \
             (def {} r \
               (match {} (var {} x) \
                 (arm {} (pat-record {} Adam (kv {} nope (pat-var {} v))) () (var {} v))))",
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn fix11b_pat_record_accepts_valid_field() {
        // Match on actual field lr — should pass
        check_ok(
            "(deftype {} Optimizer () \
               (variant {} Adam (field {} lr (t-prim {} f32)))) \
             (def {} x (lit {type: (t-adt {} Optimizer)} 0)) \
             (def {} r \
               (match {} (var {} x) \
                 (arm {} (pat-record {} Adam (kv {} lr (pat-var {} v))) () (var {} v))))",
        );
    }

    #[test]
    fn fix12_structure_score_measured() {
        // check_program runs tag validator — structure should be 1.0 for valid programs
        let exprs = chelis_deep::parser::parse_str("(def {} x (lit {type: (t-prim {} int32)} 42))")
            .unwrap();
        let report = crate::fitness::check_program(&exprs);
        assert!(
            (report.components.structure - 1.0).abs() < 0.01,
            "valid program structure should be ~1.0, got {}",
            report.components.structure
        );
    }

    #[test]
    fn checked_program_annotates_fn_bodies() {
        let exprs = chelis_deep::parser::parse_str(
            "(def {} f (fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x)))",
        )
        .unwrap();
        let checked = check_ir_program(&exprs).expect("checked program");
        let text = chelis_deep::printer::print_canonical(checked.annotated_exprs());
        assert!(
            text.contains("(fn {type: (t-fn {} (t-prim {} f32) (t-prim {} f32))}"),
            "expected typed fn metadata, got:\n{text}"
        );
    }

    #[test]
    fn checked_program_annotates_apps_and_updates_type_env() {
        let exprs = chelis_deep::parser::parse_str(
            "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} b (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} c (app {} (var {} add) (var {} a) (var {} b)))",
        )
        .unwrap();
        let checked = check_ir_program(&exprs).expect("checked program");
        let text = chelis_deep::printer::print_canonical(checked.annotated_exprs());
        assert!(
            text.contains("(app {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))}"),
            "expected typed app metadata, got:\n{text}"
        );
        assert!(checked.type_env().contains_key("c"));
    }

    #[test]
    fn ir_rejects_symbolic_normalized_axis_for_layer_norm() {
        let exprs = chelis_deep::parser::parse_str(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} gamma (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} beta (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} layer_norm) (var {} x) (var {} gamma) (var {} beta)))",
        )
        .unwrap();
        let err = check_ir_program(&exprs).expect_err("symbolic hidden axis should be rejected");
        assert!(
            err.errors
                .iter()
                .any(|error| error.message.contains("concrete normalized axis extent")),
            "expected layer_norm symbolic normalized-axis error, got: {:?}",
            err.errors
        );
    }

    #[test]
    fn checked_program_annotates_symbolic_expand_apps_from_surf() {
        let checked = checked_surf(
            r#"
def predict(
  x: tensor[batch, 64, f32],
  w: tensor[64, 1, f32],
  b: tensor[1, f32]
) -> tensor[batch, 1, f32] =
  add(matmul(x, w), expand(b, 0, batch))
"#,
        );
        let missing = checked
            .annotated_exprs()
            .iter()
            .find_map(missing_shape_sensitive_app);
        assert!(
            missing.is_none(),
            "expected all shape-sensitive apps to be annotated, missing: {:?}",
            missing
        );
    }

    #[test]
    fn surf_permute_with_axis_arguments_type_checks() {
        let checked = checked_surf(
            r#"
def transpose(x: tensor[seq, hidden, f32]) -> tensor[hidden, seq, f32] =
  permute(x, 1, 0)
"#,
        );
        let missing = checked
            .annotated_exprs()
            .iter()
            .find_map(missing_shape_sensitive_app);
        assert!(
            missing.is_none(),
            "expected typed shape-sensitive apps after permute, missing: {:?}",
            missing
        );
    }

    #[test]
    fn surf_list_builtins_type_check() {
        let checked = checked_surf(
            r#"
xs: List[f32] = [1.0, 2.0]
ys = append(xs, 3.0)
total = tensor_to_scalar(sum(to_tensor(ys), 0))
roundtrip = to_list(to_tensor(ys))
"#,
        );
        assert!(checked.annotated_exprs().len() >= 4);
    }

    #[test]
    fn surf_pad_sequences_type_checks() {
        let checked = checked_surf(
            r#"
tokens: List[List[int64]] = [[cast(1, int64), cast(2, int64)], [cast(3, int64)]]
padded = pad_sequences(tokens, cast(0, int64))
"#,
        );
        assert!(checked.annotated_exprs().len() >= 2);
    }

    #[test]
    fn surf_3g_io_and_exact_padding_builtins_type_check() {
        let checked = checked_surf(
            r#"
contents = read_file("dataset.txt")
lines = read_lines("dataset.txt")
bytes = read_bytes("dataset.txt")
exists = file_exists("dataset.txt")
names = list_dir(".")
mapped = mmap_file("dataset.txt")
mapped_len = mmap_len(mapped)
prefix = mmap_read(mapped, cast(0, int64), cast(4, int64))
padded = pad_sequences_to([[cast(1, int64)], [cast(2, int64), cast(3, int64)]], cast(4, int64), cast(0, int64))
"#,
        );
        assert!(checked.annotated_exprs().len() >= 9);
    }

    // #143: `pad_sequences_to`'s padded (axis-1) dimension equals its
    // literal `width` argument. The result type carries `Dim::Lit(width)`
    // for that axis (not `Dim::Wildcard`), so a declared return type with
    // the matching concrete width type-checks and a mismatched one is
    // rejected. The width arrives as `cast(N, int64)` in every caller.

    #[test]
    fn pad_sequences_to_literal_width_matches_declared_shape() {
        let decls = chelis_surf::parser::parse_str(
            "def f() -> tensor[1, 4, f32] = \
             pad_sequences_to([[cast(10.0, f32)]], cast(4, int64), cast(0.0, f32))\n",
        )
        .expect("surf parse");
        let exprs = chelis_surf::desugar::desugar_program(&decls);
        let result = infer_ir_program(&exprs);
        assert!(
            result.errors.is_empty(),
            "literal pad width 4 should match declared tensor[1, 4, f32], got {:?}",
            result.errors
        );
    }

    #[test]
    fn pad_sequences_to_wrong_literal_width_is_rejected() {
        let decls = chelis_surf::parser::parse_str(
            "def f() -> tensor[1, 5, f32] = \
             pad_sequences_to([[cast(10.0, f32)]], cast(4, int64), cast(0.0, f32))\n",
        )
        .expect("surf parse");
        let exprs = chelis_surf::desugar::desugar_program(&decls);
        let result = infer_ir_program(&exprs);
        assert!(
            !result.errors.is_empty(),
            "pad width 4 declared as tensor[1, 5, f32] should be a type error"
        );
    }

    #[test]
    fn pad_sequences_to_mismatched_width_through_shared_sig_dim_is_rejected() {
        // The two `pad_sequences_to` results flow into a function whose
        // sig declares the same dim variable `d` on both parameters.
        // Different literal widths (8 vs 5) must collide on `d`.
        let decls = chelis_surf::parser::parse_str(
            "sig demo_unify: &tensor[s, d, p] -> &tensor[s, d, p] -> tensor[s, d, p]\n\
             def demo_unify(a, b) = a\n\
             def test_mismatch() -> tensor[s, d, f32] = {\n\
               q = pad_sequences_to([[cast(0.0, f32)]], cast(8, int64), cast(0.0, f32))\n\
               k = pad_sequences_to([[cast(0.0, f32)]], cast(5, int64), cast(0.0, f32))\n\
               demo_unify(q, k)\n\
             }\n",
        )
        .expect("surf parse");
        let exprs = chelis_surf::desugar::desugar_program(&decls);
        let result = infer_ir_program(&exprs);
        assert!(
            result
                .errors
                .iter()
                .any(|error| matches!(error.kind, CheckErrorKind::DimensionMismatch)),
            "widths 8 and 5 sharing sig dim `d` should surface a DimensionMismatch, got {:?}",
            result.errors
        );
    }

    #[test]
    fn surf_dict_and_iteration_builtins_type_check() {
        let checked = checked_surf(
            r#"
keys: List[string] = ["alpha", "beta"]
ids: List[int64] = [cast(1, int64), cast(2, int64)]
pairs = zip(keys, ids)
indexed = enumerate(keys)
vocab: Dict[string, int64] = dict_of(pairs)
found = dict_contains(vocab, "alpha")
id = dict_get(vocab, "beta")
only_keys = dict_keys(vocab)
only_values = dict_values(vocab)
roundtrip = dict_entries(vocab)
"#,
        );
        assert!(checked.annotated_exprs().len() >= 9);
    }

    #[test]
    fn surf_3h_tensor_numeric_builtins_type_check() {
        let checked = checked_surf(
            r#"
def projection(
  x: tensor[batch, seq, hidden, f32],
  w: tensor[hidden, out_dim, f32],
  token_ids: tensor[batch, seq, int64],
  mask: tensor[batch, seq, out_dim, bool],
  table: tensor[vocab, out_dim, f32]
) -> tensor[batch, seq, out_dim, f32] = {
  logits = einsum("bsh,ho->bso", x, w)
  embed = gather(table, token_ids, 0)
  clipped = clamp(add(logits, embed), scalar_to_tensor(0.0), scalar_to_tensor(6.0))
  running = cumsum(clipped, 1)
  where(mask, running, clipped)
}
"#,
        );
        assert!(!checked.annotated_exprs().is_empty());
    }

    #[test]
    fn surf_3h_structural_tensor_builtins_type_check() {
        let checked = checked_surf(
            r#"
def pack_heads(
  q: tensor[batch, seq, 2, f32],
  k: tensor[batch, seq, 2, f32]
) -> tensor[batch, seq, *, f32] = {
  packed = concat([q, k], 2)
  pieces = split(packed, 2, [2, 2])
  concat(pieces, 2)
}
"#,
        );
        assert!(!checked.annotated_exprs().is_empty());
    }

    #[test]
    fn surf_3h_sort_and_trace_type_check() {
        let checked = checked_surf(
            r#"
def summarize(x: tensor[batch, hidden, hidden, f32]) -> (tensor[batch, hidden, f32], tensor[batch, hidden, int64], tensor[batch, f32]) = {
  diag = diagonal(x, 1, 2)
  sorted = sort(diag, 1)
  values = sorted.0
  indices = sorted.1
  total = trace(x, 1, 2)
  (values, indices, total)
}
"#,
        );
        assert!(!checked.annotated_exprs().is_empty());
    }

    #[test]
    fn surf_einsum_rejects_ellipsis_in_3h() {
        let result = check_ir_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
def bad(
  x: tensor[batch, seq, hidden, f32],
  w: tensor[hidden, out_dim, f32]
) -> tensor[batch, seq, out_dim, f32] =
  einsum("...h,ho->...o", x, w)
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("ellipsis should be rejected in 3h einsum");
        assert!(
            err.errors.iter().any(|error| {
                error.message.contains("einsum")
                    && (error.message.contains("ellipsis") || error.message.contains("..."))
            }),
            "expected einsum ellipsis rejection, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_where_rejects_non_bool_condition_tensor() {
        let result = check_ir_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
def bad(
  cond: tensor[batch, hidden, f32],
  x: tensor[batch, hidden, f32],
  y: tensor[batch, hidden, f32]
) -> tensor[batch, hidden, f32] =
  where(cond, x, y)
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("where should reject non-bool condition tensors");
        assert!(
            err.errors
                .iter()
                .any(|error| { error.message.contains("where") && error.message.contains("bool") }),
            "expected where bool mismatch, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_scatter_replace_rejects_unknown_mode() {
        let result = check_ir_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
def bad(
  base: tensor[seq, hidden, f32],
  ids: tensor[seq, int64],
  updates: tensor[seq, hidden, f32]
) -> tensor[seq, hidden, f32] =
  scatter(base, ids, updates, 0, "last")
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("scatter should reject unsupported mode");
        assert!(
            err.errors.iter().any(|error| {
                error.message.contains("scatter")
                    && error.message.contains("replace")
                    && error.message.contains("add")
            }),
            "expected scatter mode rejection, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_einsum_rejects_static_extent_mismatch() {
        let result = check_ir_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
a = pad_sequences([[1.0, 2.0], [3.0, 4.0]], 0.0)
b = pad_sequences([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], 0.0)
out = einsum("ij,jk->ik", a, b)
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("static einsum extent mismatch should be rejected");
        assert!(
            err.errors.iter().any(|error| {
                error.message.contains("einsum") && error.message.contains("inconsistent extents")
            }),
            "expected einsum extent mismatch, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_scatter_replace_rejects_static_duplicate_indices() {
        let result = check_ir_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
base = pad_sequences([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], 0.0)
ids: List[int64] = [cast(1, int64), cast(1, int64)]
idx = to_tensor(ids)
updates = pad_sequences([[5.0, 5.0], [6.0, 6.0]], 0.0)
out = scatter(base, idx, updates, 0, "replace")
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("static scatter duplicate indices should be rejected");
        assert!(
            err.errors.iter().any(|error| {
                error.message.contains("scatter")
                    && error.message.contains("duplicate target index")
            }),
            "expected scatter duplicate-index rejection, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_map_filter_fold_type_check() {
        let checked = checked_surf(
            r#"
def inc(x: int64) -> int64 = add(x, cast(1, int64))
xs: List[int64] = [cast(1, int64), cast(2, int64), cast(3, int64)]
mapped = map(inc, xs)
filtered = filter(fn (x: int64) -> eq(mod(x, cast(2, int64)), cast(0, int64)), mapped)
total = fold(fn (acc: int64, x: int64) -> add(acc, x), cast(0, int64), filtered)
"#,
        );
        assert!(checked.annotated_exprs().len() >= 5);
    }

    #[test]
    fn surf_append_rejects_wrong_element_type() {
        let result = check_ir_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[int64] = [cast(1, int64)]
bad = append(xs, "oops")
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("append should reject mismatched element type");
        assert!(
            err.errors
                .iter()
                .any(|error| error.message.contains("List") || error.message.contains("string")),
            "expected list element mismatch, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_filter_rejects_non_bool_callback() {
        let result = check_ir_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[int64] = [cast(1, int64)]
bad = filter(fn (x: int64) -> add(x, cast(1, int64)), xs)
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("filter should reject non-bool callback");
        assert!(
            err.errors
                .iter()
                .any(|error| error.message.contains("filter")
                    && error.message.contains("bool")
                    && error.message.contains("callback")),
            "expected bool callback mismatch, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_dict_entries_rejects_non_dict_input() {
        let result = check_ir_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[int64] = [cast(1, int64)]
bad = dict_entries(xs)
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("dict_entries should reject non-dict input");
        assert!(
            err.errors
                .iter()
                .any(|error| error.message.contains("dict_entries")
                    || error.message.contains("Dict")),
            "expected dict input mismatch, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_to_list_rejects_rank2_tensor() {
        let result = check_ir_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
def bad(x: tensor[2, 2, f32]) -> List[f32] = to_list(x)
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("to_list should reject rank-2 tensor input");
        assert!(
            err.errors.iter().any(|error| {
                error.message.contains("rank-1 tensor") || error.message.contains("to_list")
            }),
            "expected rank mismatch, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_fold_rejects_accumulator_mismatch() {
        let result = check_ir_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[int64] = [cast(1, int64)]
bad = fold(fn (acc: string, x: int64) -> string_concat(acc, to_string(x)), cast(0, int64), xs)
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("fold should reject mismatched accumulator type");
        assert!(
            err.errors.iter().any(|error| error.message.contains("fold")
                && error.message.contains("accumulator")
                && error.message.contains("string")
                && error.message.contains("int64")),
            "expected accumulator mismatch, got {:?}",
            err.errors
        );
    }

    fn surf_tuple_fold_tensor_slot_program() -> Vec<deep::Expr> {
        chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
def f[n](xs: tensor[n, f32]) -> (tensor[n, f32], int64) = {
  idxs = range(cast(0, int64), numel(copy(xs)))
  state0 = (to_tensor(map(fn (x: f32) -> cast(0.0, f32), to_list(copy(xs)))), cast(0, int64))
  step = fn (state, i) -> {
    acc = state.0
    total = state.1
    (acc, add(total, i))
  }
  fold(step, state0, idxs)
}

out = f(to_tensor([1.0, 2.0, 3.0]))
"#,
            )
            .expect("surf parse"),
        )
    }

    #[test]
    fn surf_polymorphic_tuple_fold_with_tensor_slot_infers_ir() {
        let result = infer_ir_program(&surf_tuple_fold_tensor_slot_program());
        assert!(
            result.errors.is_empty(),
            "ir inference should succeed without overflowing: {:?}",
            result.errors
        );
    }

    #[test]
    fn surf_polymorphic_tuple_fold_with_tensor_slot_annotates_ir() {
        let program = surf_tuple_fold_tensor_slot_program();
        let _ = annotate_ir_program(&program);
    }

    #[test]
    fn surf_polymorphic_tuple_fold_with_tensor_slot_type_checks() {
        let result = check_ir_program(&surf_tuple_fold_tensor_slot_program());
        result.expect("polymorphic tuple fold should type check without overflowing");
    }

    #[test]
    fn surf_collection_helper_builtins_type_check() {
        let result = check_ir_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[int64] = [cast(1, int64), cast(2, int64), cast(3, int64)]
prefix = take(xs, cast(2, int64))
suffix = drop(xs, cast(1, int64))
groups = chunk(xs, cast(2, int64))
scanned = scan(fn (acc: int64, x: int64) -> add(acc, x), cast(0, int64), xs)
buckets = partition(fn (x: int64) -> gt(x, cast(1, int64)), xs)
exploded = flat_map(fn (x: int64) -> [x, add(x, cast(10, int64))], xs)
flattened = flatten([[cast(1, int64)], [cast(2, int64), cast(3, int64)]])
base: Dict[string, int64] = dict_of([("alpha", cast(1, int64))])
extended = dict_insert(base, "beta", cast(2, int64))
merged = dict_merge(extended, dict_of([("beta", cast(20, int64)), ("gamma", cast(3, int64))]))
trimmed = dict_remove(merged, "gamma")
"#,
            )
            .expect("surf parse"),
        ));
        result.expect("collection helper builtins should type check");
    }

    #[test]
    fn surf_take_rejects_non_integer_count() {
        let result = check_ir_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[int64] = [cast(1, int64), cast(2, int64)]
bad = take(xs, "two")
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("take should reject non-integer count");
        assert!(
            err.errors
                .iter()
                .any(|error| error.message.contains("take") || error.message.contains("integer")),
            "expected integer count mismatch, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_dict_insert_rejects_value_type_mismatch() {
        let result = check_ir_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
base: Dict[string, int64] = dict_of([("alpha", cast(1, int64))])
bad = dict_insert(base, "beta", "two")
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("dict_insert should reject mismatched value type");
        assert!(
            err.errors
                .iter()
                .any(|error| error.message.contains("dict_insert")
                    || error.message.contains("int64")
                    || error.message.contains("string")),
            "expected dict value mismatch, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_dict_merge_rejects_mismatched_dict_value_types() {
        let result = check_ir_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
lhs: Dict[string, int64] = dict_of([("alpha", cast(1, int64))])
rhs: Dict[string, string] = dict_of([("beta", "two")])
bad = dict_merge(lhs, rhs)
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("dict_merge should reject mismatched dict value types");
        assert!(
            err.errors
                .iter()
                .any(|error| error.message.contains("dict_merge")
                    || error.message.contains("Dict")
                    || error.message.contains("int64")
                    || error.message.contains("string")),
            "expected dict merge mismatch, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_scan_rejects_accumulator_mismatch() {
        let result = check_ir_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[int64] = [cast(1, int64)]
bad = scan(fn (acc: string, x: int64) -> string_concat(acc, to_string(x)), cast(0, int64), xs)
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("scan should reject mismatched accumulator type");
        assert!(
            err.errors.iter().any(|error| error.message.contains("scan")
                && error.message.contains("accumulator")
                && error.message.contains("string")
                && error.message.contains("int64")),
            "expected scan accumulator mismatch, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_partition_rejects_non_bool_callback() {
        let result = check_ir_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[int64] = [cast(1, int64)]
bad = partition(fn (x: int64) -> add(x, cast(1, int64)), xs)
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("partition should reject non-bool callback");
        assert!(
            err.errors
                .iter()
                .any(|error| error.message.contains("partition")
                    && error.message.contains("bool")
                    && error.message.contains("callback")),
            "expected partition callback mismatch, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_flat_map_rejects_non_list_callback() {
        let result = check_ir_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[int64] = [cast(1, int64)]
bad = flat_map(fn (x: int64) -> add(x, cast(1, int64)), xs)
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("flat_map should reject non-list callback");
        assert!(
            err.errors
                .iter()
                .any(|error| error.message.contains("List") || error.message.contains("flat_map")),
            "expected flat_map callback mismatch, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_flatten_rejects_non_nested_list_input() {
        let result = check_ir_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[int64] = [cast(1, int64)]
bad = flatten(xs)
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("flatten should reject non-nested list input");
        assert!(
            err.errors
                .iter()
                .any(|error| error.message.contains("flatten") || error.message.contains("List")),
            "expected flatten input mismatch, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_dict_remove_rejects_mismatched_key_type() {
        let result = check_ir_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
base: Dict[string, int64] = dict_of([("alpha", cast(1, int64))])
bad = dict_remove(base, cast(7, int64))
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("dict_remove should reject mismatched key type");
        assert!(
            err.errors.iter().any(|error| {
                error.message.contains("dict_remove")
                    || (error.message.contains("string") && error.message.contains("int64"))
            }),
            "expected dict_remove key mismatch, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_chunk_rejects_non_integer_size() {
        let result = check_ir_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[int64] = [cast(1, int64)]
bad = chunk(xs, "two")
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("chunk should reject non-integer size");
        assert!(
            err.errors
                .iter()
                .any(|error| error.message.contains("chunk") || error.message.contains("integer")),
            "expected chunk size mismatch, got {:?}",
            err.errors
        );
    }

    // ------------------------------------------------------------------
    // Top-level binding cycle detection
    // ------------------------------------------------------------------

    fn ir_errors_from_surf(src: &str) -> Vec<CheckError> {
        let decls = chelis_surf::parser::parse_str(src).expect("surf parse");
        let exprs = chelis_surf::desugar::desugar_program(&decls);
        infer_ir_program(&exprs).errors
    }

    #[test]
    fn nautilus_self_reference_is_allowed() {
        // The single-hop identity `x = (x : tensor[...])` is a pinned Nautilus
        // external-input pattern and must NOT be flagged as a binding cycle.
        let errors = ir_errors_from_surf("x = (x : tensor[4, f32])\n");
        assert!(
            !errors
                .iter()
                .any(|e| matches!(e.kind, CheckErrorKind::CycleDetected)),
            "Nautilus self-reference should not be flagged; got {errors:?}"
        );
    }

    #[test]
    fn two_hop_binding_cycle_is_detected() {
        let errors = ir_errors_from_surf(
            "a = (b : tensor[4, f32])\n\
             b = (a : tensor[4, f32])\n",
        );
        let cycle_err = errors
            .iter()
            .find(|e| matches!(e.kind, CheckErrorKind::CycleDetected))
            .unwrap_or_else(|| {
                panic!("expected a CycleDetected error for a two-hop cycle; got {errors:?}")
            });
        assert!(
            cycle_err.message.contains("binding cycle"),
            "message should mention 'binding cycle'; got {:?}",
            cycle_err.message
        );
        assert!(
            cycle_err.message.contains("a -> b -> a"),
            "expected 'a -> b -> a' in message; got {:?}",
            cycle_err.message
        );
    }

    #[test]
    fn three_hop_binding_cycle_is_detected() {
        let errors = ir_errors_from_surf(
            "a = (b : tensor[4, f32])\n\
             b = (c : tensor[4, f32])\n\
             c = (a : tensor[4, f32])\n",
        );
        let cycle_err = errors
            .iter()
            .find(|e| matches!(e.kind, CheckErrorKind::CycleDetected))
            .unwrap_or_else(|| {
                panic!("expected a CycleDetected error for a three-hop cycle; got {errors:?}")
            });
        assert!(
            cycle_err.message.contains("a -> b -> c -> a"),
            "expected 'a -> b -> c -> a' in message; got {:?}",
            cycle_err.message
        );
    }

    #[test]
    fn unrelated_defs_do_not_trigger_cycle_false_positive() {
        // Sanity: multiple Nautilus self-references together should still pass.
        let errors = ir_errors_from_surf(
            "x = (x : tensor[4, f32])\n\
             y = (y : tensor[4, f32])\n",
        );
        assert!(
            !errors
                .iter()
                .any(|e| matches!(e.kind, CheckErrorKind::CycleDetected)),
            "multiple independent Nautilus inputs must not trigger a cycle error; got {errors:?}"
        );
    }

    // ── chelis#293: general type var through a function-typed parameter ──
    //
    // A def generic over a general type variable `P` declared in its
    // explicit `[..]` quantifier list, where `P` is threaded through a
    // function-typed parameter, must instantiate `P` to a fresh variable at
    // each call site and unify it against the concrete callback argument.
    // Before the fix, the Surf desugarer misclassified the uppercase `P` as
    // a rigid ADT `(t-adt {} P)`, so every call site failed with
    // `type mismatch: P vs tensor[..]`.

    #[test]
    fn issue_293_general_tvar_through_arrow_param_checks_clean() {
        // Positive: the reproducer must check clean — no TypeMismatch.
        let result = infer_surf(
            r#"
module Repro.GenericCallback
def apply_resid[n, P](x: tensor[n, f32], inner_p: P, f: tensor[n, f32] -> P -> tensor[n, f32]) -> tensor[n, f32] =
  add(x, f(x, inner_p))
def use_it(x: tensor[3, f32], w: tensor[3, f32]) -> tensor[3, f32] =
  apply_resid(x, w, fn (t, q) -> mul(t, q))
"#,
        );
        assert!(
            result.errors.is_empty(),
            "expected the general-tvar-through-arrow reproducer to check clean, got: {:?}",
            result.errors
        );
    }

    #[test]
    fn issue_293_apply_resid_checks_clean_in_isolation() {
        // Control: the generic def alone already checked clean before the
        // fix; it must keep checking clean.
        let result = infer_surf(
            r#"
module Repro.GenericCallback
def apply_resid[n, P](x: tensor[n, f32], inner_p: P, f: tensor[n, f32] -> P -> tensor[n, f32]) -> tensor[n, f32] =
  add(x, f(x, inner_p))
"#,
        );
        assert!(
            result.errors.is_empty(),
            "apply_resid must check clean in isolation, got: {:?}",
            result.errors
        );
    }

    #[test]
    fn issue_293_dim_var_callback_variant_checks_clean() {
        // Control (dim-var path): the same shape where the threaded
        // parameter is a *dim* var `m` rather than a general type var
        // already worked and must keep working.
        let result = infer_surf(
            r#"
module Repro.DimCallback
def apply_resid[n, m](x: tensor[n, f32], inner: tensor[m, f32], f: tensor[n, f32] -> tensor[m, f32] -> tensor[n, f32]) -> tensor[n, f32] =
  f(x, inner)
def use_it(x: tensor[3, f32], w: tensor[3, f32]) -> tensor[3, f32] =
  apply_resid(x, w, fn (t, q) -> add(t, q))
"#,
        );
        assert!(
            result.errors.is_empty(),
            "the dim-var callback control must check clean, got: {:?}",
            result.errors
        );
    }

    #[test]
    fn issue_293_monomorphic_callback_variant_checks_clean() {
        // Control (monomorphic path): a fully concrete callback already
        // worked and must keep working.
        let result = infer_surf(
            r#"
module Repro.MonoCallback
def apply_resid[n](x: tensor[n, f32], inner: tensor[n, f32], f: tensor[n, f32] -> tensor[n, f32] -> tensor[n, f32]) -> tensor[n, f32] =
  add(x, f(x, inner))
def use_it(x: tensor[3, f32], w: tensor[3, f32]) -> tensor[3, f32] =
  apply_resid(x, w, fn (t, q) -> mul(t, q))
"#,
        );
        assert!(
            result.errors.is_empty(),
            "the monomorphic callback control must check clean, got: {:?}",
            result.errors
        );
    }

    #[test]
    fn issue_293_incompatible_callback_still_rejected() {
        // Negative parity: the fix must NOT over-loosen unification. Here
        // the callback's second parameter `q` is multiplied with `t`
        // (a `tensor[n, f32]`), so `q` must be `tensor[n, f32]`. But the
        // `inner_p` argument supplied at the call site is a scalar `int32`,
        // which is bound to the same `P`. `P` cannot be both a tensor and a
        // scalar int32, so this must still produce a clear mismatch.
        let result = infer_surf(
            r#"
module Repro.BadCallback
def apply_resid[n, P](x: tensor[n, f32], inner_p: P, f: tensor[n, f32] -> P -> tensor[n, f32]) -> tensor[n, f32] =
  add(x, f(x, inner_p))
def use_it(x: tensor[3, f32]) -> tensor[3, f32] =
  apply_resid(x, 1, fn (t, q) -> mul(t, q))
"#,
        );
        assert!(
            result.errors.iter().any(|e| matches!(
                e.kind,
                CheckErrorKind::TypeMismatch
                    | CheckErrorKind::PrecisionMismatch
                    | CheckErrorKind::DimensionMismatch
            )),
            "an incompatible callback/argument combination must still be rejected with a \
             unification mismatch; unification must not have been over-loosened (chelis#293), \
             got: {:?}",
            result.errors
        );
    }

    // ── #39 / chelis#405 wildcard-narrowing param-bound-dvar gate ────

    /// `narrow_wildcards_with` narrows a body `Wildcard` to a declared
    /// `Dim::Var` ONLY when that var is bound by a parameter tensor
    /// position. This is the `const_col[n](spots: tensor[n, ..]) ->
    /// tensor[n, 1]` family: the return dim `n` is the same var as the
    /// `spots` parameter axis-0, so the caller binds it. The narrow
    /// reconnects the wildcard return to that input dim.
    #[test]
    fn narrow_substitutes_param_bound_dim_var_for_wildcard() {
        let n = DimVar(7);
        let decl = Type::Fn(
            vec![
                Type::Tensor(
                    vec![Dim::Var(n), Dim::Lit(1)],
                    TensorPrec::Concrete(Prim::F64),
                ),
                Type::Prim(Prim::F64),
            ],
            Box::new(Type::Tensor(
                vec![Dim::Var(n), Dim::Lit(1)],
                TensorPrec::Concrete(Prim::F64),
            )),
        );
        // The inferred body return is the shape-erased `tensor[*, 1]`.
        let body = Type::Fn(
            vec![
                Type::Tensor(
                    vec![Dim::Var(n), Dim::Lit(1)],
                    TensorPrec::Concrete(Prim::F64),
                ),
                Type::Prim(Prim::F64),
            ],
            Box::new(Type::Tensor(
                vec![Dim::Wildcard, Dim::Lit(1)],
                TensorPrec::Concrete(Prim::F64),
            )),
        );
        let param_dvars = param_bound_dvars(&decl);
        assert!(
            param_dvars.contains(&n),
            "n appears in a parameter tensor position, so it is param-bound"
        );
        let narrowed = narrow_wildcards_with(&body, &decl, &param_dvars);
        let Type::Fn(_, ret) = &narrowed else {
            panic!("expected Fn, got {narrowed:?}");
        };
        assert_eq!(
            **ret,
            Type::Tensor(
                vec![Dim::Var(n), Dim::Lit(1)],
                TensorPrec::Concrete(Prim::F64)
            ),
            "the return wildcard must narrow to the param-bound dim var n, not stay `*`"
        );
    }

    /// A *return-only* dim var (it appears in the declared return but in
    /// NO parameter tensor position, e.g. `arange[n](start: int32, stop:
    /// int32) -> tensor[n, int32]`) is NOT param-bound. The body wildcard
    /// must stay `Wildcard`: narrowing it to the unbound var would leak a
    /// free dim var into callers (the RT-39+44 soundness regression,
    /// commit 8067c9ce).
    #[test]
    fn narrow_keeps_wildcard_for_return_only_dim_var() {
        let n = DimVar(11);
        let decl = Type::Fn(
            vec![Type::Prim(Prim::Int32), Type::Prim(Prim::Int32)],
            Box::new(Type::Tensor(
                vec![Dim::Var(n)],
                TensorPrec::Concrete(Prim::Int32),
            )),
        );
        let body = Type::Fn(
            vec![Type::Prim(Prim::Int32), Type::Prim(Prim::Int32)],
            Box::new(Type::Tensor(
                vec![Dim::Wildcard],
                TensorPrec::Concrete(Prim::Int32),
            )),
        );
        let param_dvars = param_bound_dvars(&decl);
        assert!(
            !param_dvars.contains(&n),
            "n is return-only: it must not be in the param-bound set"
        );
        let narrowed = narrow_wildcards_with(&body, &decl, &param_dvars);
        let Type::Fn(_, ret) = &narrowed else {
            panic!("expected Fn, got {narrowed:?}");
        };
        assert_eq!(
            **ret,
            Type::Tensor(vec![Dim::Wildcard], TensorPrec::Concrete(Prim::Int32)),
            "a return-only dim var's body wildcard must stay `*` (no free-var leak)"
        );
    }

    /// `Wildcard` against a declared `Dim::Lit` is always narrowed
    /// regardless of the param-bound set -- concrete literals are
    /// self-contained (the original #39 behavior).
    #[test]
    fn narrow_substitutes_literal_for_wildcard_unconditionally() {
        let empty = HashSet::new();
        let body = Type::Tensor(
            vec![Dim::Wildcard, Dim::Wildcard],
            TensorPrec::Concrete(Prim::F32),
        );
        let decl = Type::Tensor(
            vec![Dim::Lit(4), Dim::Lit(1)],
            TensorPrec::Concrete(Prim::F32),
        );
        let narrowed = narrow_wildcards_with(&body, &decl, &empty);
        assert_eq!(
            narrowed,
            Type::Tensor(
                vec![Dim::Lit(4), Dim::Lit(1)],
                TensorPrec::Concrete(Prim::F32)
            )
        );
    }

    /// End-to-end regression lock (chelis#405 / WS-3): the
    /// `const_col`-style `shape -> to_tensor(map(range)) -> reshape` chain
    /// must publish a scheme whose return batch dim is the param-bound
    /// DECLARED dim, not the shape-erased `*`. Pre-fix the published scheme
    /// was `tensor[*, 1]`, so a CALLER that forwards the result observed
    /// `*` for its batch axis; that erased the link the C backend needs and
    /// ICEd the §345 `symbolic_occurrences` guard at the `vmap` lane
    /// kernel. The fix ties the return dim to the `spots` parameter dim
    /// var, so a caller forwarding `const_col`'s result sees a declared
    /// dim. (`type_env` records each def's body annotation; the call-site
    /// result type in the *caller's* body annotation is what reflects the
    /// narrowed published scheme, so the lock inspects the caller.)
    #[test]
    fn const_col_chain_propagates_declared_dim_to_callers() {
        let checked = checked_surf(
            r#"
def const_col[n](spots: tensor[n, f32], v: f64) -> tensor[n, 1, f64] = {
  nn = cast(shape(copy(spots), cast(0, int32)), int64)
  reshape(to_tensor(map(fn (i: int64) -> v, range(cast(0, int64), nn))), [nn, cast(1, int64)])
}
def caller[n](spots: tensor[n, f32]) -> tensor[n, 1, f64] = const_col(spots, cast(1.0, f64))
"#,
        );
        let caller_ty = checked
            .type_env()
            .get("caller")
            .expect("caller must be in the published type env");
        let rendered = chelis_deep::printer::print_canonical_flat(std::slice::from_ref(caller_ty));
        // The caller's body is `const_col(spots, ..)`; its annotated type
        // is the call-site result. A param-bound declared dim narrows the
        // wildcard, so the caller's batch axis is a `d-var`, never the
        // shape-erased `(d-name {} *)`.
        assert!(
            !rendered.contains("(d-name {} *)"),
            "const_col's result must not carry a wildcard `*` dim at the call \
             site; got {rendered}"
        );
        assert!(
            rendered.contains("d-var"),
            "the const_col call-site batch dim must be a declared dim var; \
             got {rendered}"
        );
    }
}
