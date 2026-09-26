//! Native lowering services and the trusted primitive boundary of lowering.
//!
//! Semantic lowering receives `&mut` [`LoweringRuntime`] and calls only the
//! methods below. Every method marked *Primitive* is one native effect on
//! explicit state and carries its intended Lean model; everything else here
//! is ordinary code over those primitives. No other lowering code touches
//! thread-local storage, atomics, the stack, the clock, the environment,
//! stderr, or unwinding.
//!
//! [`LoweringRuntime`] is a handle to the current thread's lowering state: two
//! suppression flags `u` (speculative unrepresentable-DAG attempt) and `o`
//! (quiet output inside a catch), plus a persistent summary-probe counter.
//! Nested native acquisitions borrow the same per-thread state; acquiring a
//! handle never resets or copies it. The process-global panic hook reads the
//! same flags when Rust raises. Scope exits clear their flag to `false`
//! rather than restoring an enclosing value, and a fatal re-raise explicitly
//! re-enables `o`.

use std::any::Any;
use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use crate::lower::LowerDiagnostic;

/// Why lowering rejected its input: the explicit error channel that
/// replaced panic-payload control flow.
///
/// The variants are exactly the two typed payloads a lowering rejection
/// used to unwind with. [`LoweringRuntime::catch_lowering`] maps them to the
/// same diagnostics it produced from those payloads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoweringFailure {
    /// A nonfatal rejection inside a speculative (suppressed) DAG attempt.
    Unrepresentable,
    /// A structured rejection, fatal or not.
    Diagnostic(LowerDiagnostic),
}

/// Result of a lowering step that may reject its input.
pub type LowerResult<T> = Result<T, LoweringFailure>;

/// `?` for an inner `Option` inside a `LowerResult<Option<_>>` function:
/// `None` returns `Ok(None)`, leaving the failure channel untouched.
macro_rules! try_some {
    ($option:expr) => {
        match $option {
            Some(value) => value,
            None => return Ok(None),
        }
    };
}
pub(crate) use try_some;

/// `?` for an inner `Result` inside a `LowerResult<Result<_, E>>` function:
/// an ordinary error returns `Ok(Err(E::from(error)))`, exactly like `?`
/// in the function's former signature, leaving lowering failures separate.
macro_rules! try_ok {
    ($result:expr) => {
        match $result {
            Ok(value) => value,
            Err(error) => return Ok(Err(::core::convert::From::from(error))),
        }
    };
}
pub(crate) use try_ok;

/// [`Iterator::find_map`] with a fallible step: the same visiting order, and
/// it stops at the first `Some` or at the first failure.
pub(crate) fn try_find_map<I: IntoIterator, U>(
    items: I,
    mut step: impl FnMut(I::Item) -> LowerResult<Option<U>>,
) -> LowerResult<Option<U>> {
    for item in items {
        if let Some(found) = step(item)? {
            return Ok(Some(found));
        }
    }
    Ok(None)
}

/// [`Iterator::any`] with a fallible predicate: stops at the first `true` or
/// at the first failure.
pub(crate) fn try_any<I: IntoIterator>(
    items: I,
    mut predicate: impl FnMut(I::Item) -> LowerResult<bool>,
) -> LowerResult<bool> {
    for item in items {
        if predicate(item)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// `collect::<Result<Vec<T>, E>>()` over steps that may also fail: visits in
/// order and stops at the first lowering failure or the first ordinary error.
pub(crate) fn try_collect_results<T, E>(
    items: impl IntoIterator<Item = LowerResult<Result<T, E>>>,
) -> LowerResult<Result<Vec<T>, E>> {
    let mut collected = Vec::new();
    for item in items {
        match item? {
            Ok(value) => collected.push(value),
            Err(error) => return Ok(Err(error)),
        }
    }
    Ok(Ok(collected))
}

/// `collect::<Option<Vec<T>>>()` over steps that may also fail: visits in
/// order and stops at the first lowering failure or the first `None`.
pub(crate) fn try_collect_options<T>(
    items: impl IntoIterator<Item = LowerResult<Option<T>>>,
) -> LowerResult<Option<Vec<T>>> {
    let mut collected = Vec::new();
    for item in items {
        match item? {
            Some(value) => collected.push(value),
            None => return Ok(None),
        }
    }
    Ok(Some(collected))
}

/// [`chelis_deep::ast::Metadata::any_expression`] with a fallible predicate:
/// the same visiting order, stopping at the first `true` or failure.
pub(crate) fn try_any_metadata_expression(
    metadata: &chelis_deep::ast::Metadata,
    mut predicate: impl FnMut(&chelis_deep::ast::Expr) -> LowerResult<bool>,
) -> LowerResult<bool> {
    match metadata.find_expression(|value| match predicate(value) {
        Ok(false) => None,
        Ok(true) => Some(Ok(())),
        Err(failure) => Some(Err(failure)),
    }) {
        None => Ok(false),
        Some(outcome) => outcome.map(|()| true),
    }
}

/// [`chelis_deep::ast::Metadata::visit_expressions`] with a fallible visitor:
/// the same visiting order, and nothing is visited after the first failure.
pub(crate) fn try_visit_metadata_expressions(
    metadata: &chelis_deep::ast::Metadata,
    mut visit: impl FnMut(&chelis_deep::ast::Expr) -> LowerResult<()>,
) -> LowerResult<()> {
    match metadata.find_expression(|value| visit(value).err()) {
        Some(failure) => Err(failure),
        None => Ok(()),
    }
}

impl LoweringFailure {
    /// The diagnostic a caught failure reports: exactly the one the former
    /// panic payload converted to. Ordinary code, not a native primitive.
    pub fn into_diagnostic(self) -> LowerDiagnostic {
        match self {
            Self::Unrepresentable => unrepresentable_diagnostic(),
            Self::Diagnostic(diagnostic) => diagnostic,
        }
    }

    /// Continue a failure as a native unwind past a public entry whose
    /// signature cannot return it. Only public boundary wrappers call it;
    /// semantic lowering propagates the failure instead.
    pub(crate) fn resume_at_boundary(self) -> ! {
        match self {
            Self::Unrepresentable => resume_native_unwind(UnrepresentableDag),
            Self::Diagnostic(diagnostic) => resume_native_unwind(diagnostic),
        }
    }
}

/// Primitive. Unwind with `payload` without running the panic hook: the raise
/// that produced it already did.
///
/// Model: failure of the enclosing native entry (outside the modeled domain).
fn resume_native_unwind<P: Any + Send>(payload: P) -> ! {
    std::panic::resume_unwind(Box::new(payload))
}

/// Primitive. Run `body` between `enter` and `exit` on explicit `state`.
///
/// `exit` runs after `body` returns, whatever it returns. If `body` unwinds
/// natively, `exit` runs before the unwind continues, so a production catch
/// that observes `state` afterwards sees it exited. The hook has already run
/// at the raise; frames inside `body` are finalized first, then `exit`, then
/// the enclosing frames, and resuming does not run the hook again.
///
/// Model: `let t := enter s; let r := body s; exit s t; r`.
pub(crate) fn scoped<S, T, R>(
    state: &mut S,
    enter: impl FnOnce(&mut S) -> T,
    exit: impl FnOnce(&mut S, T),
    body: impl FnOnce(&mut S) -> R,
) -> R {
    let token = enter(state);
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| body(&mut *state))) {
        Ok(result) => {
            exit(state, token);
            result
        }
        Err(payload) => {
            exit(state, token);
            std::panic::resume_unwind(payload)
        }
    }
}

/// Primitive. Run `f` on a fresh native stack segment of `stack_size` bytes
/// when fewer than `red_zone` bytes remain (`stacker::maybe_grow`).
///
/// Model: `f state`.
pub(crate) fn grow_stack_if_needed<S, R>(
    state: &mut S,
    red_zone: usize,
    stack_size: usize,
    f: impl FnOnce(&mut S) -> R,
) -> R {
    stacker::maybe_grow(red_zone, stack_size, || f(state))
}

/// Handle to the current native thread's lowering state.
///
/// Obtain one with [`with_native_lowering_runtime`]; semantic code only
/// borrows it. The handle names this thread's state, so it is neither `Send`
/// nor `Sync`, and checked programs and persisted lowering facts never
/// contain it. It has no fields a caller can read: every access goes through
/// a primitive, so the Lean model threads the flags as explicit state.
pub struct LoweringRuntime {
    thread_bound: PhantomData<*const ()>,
}

impl LoweringRuntime {
    /// Primitive. Whether a speculative DAG attempt is active.
    ///
    /// Model: read `u`.
    pub(crate) fn unrepresentable_panic_suppressed(&mut self) -> bool {
        NATIVE_LOWERING_STATE.with(|state| state.suppress_unrepresentable.get())
    }

    /// Primitive. Set the speculation flag.
    ///
    /// Model: `u := value`.
    fn set_unrepresentable_panic_suppressed(&mut self, value: bool) {
        NATIVE_LOWERING_STATE.with(|state| state.suppress_unrepresentable.set(value));
    }

    /// Primitive. Set the quiet-output flag.
    ///
    /// Model: `o := value`.
    fn set_output_suppressed(&mut self, value: bool) {
        NATIVE_LOWERING_STATE.with(|state| state.suppress_output.set(value));
    }

    /// Run a speculative DAG attempt without native panic output.
    ///
    /// A nonfatal lowering rejection becomes [`LoweringFailure::Unrepresentable`];
    /// fatal diagnostics stay structured. Return and unwind both clear the
    /// flag, even when an enclosing scope had set it.
    ///
    /// Model: `u := true; let r := f; u := false; r`.
    pub fn with_suppress_unrepresentable_panic<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        scoped(
            self,
            |runtime| runtime.set_unrepresentable_panic_suppressed(true),
            |runtime, ()| runtime.set_unrepresentable_panic_suppressed(false),
            f,
        )
    }

    /// Reject the current lowering step.
    ///
    /// A nonfatal rejection under speculation becomes
    /// [`LoweringFailure::Unrepresentable`]; fatal diagnostics stay structured.
    pub(crate) fn raise_lowering_diagnostic(
        &mut self,
        diagnostic: LowerDiagnostic,
    ) -> LoweringFailure {
        // Fatal AD rejection must not become a speculative host fallback.
        if !diagnostic.fatal && self.unrepresentable_panic_suppressed() {
            self.raise_unrepresentable()
        } else {
            LoweringFailure::Diagnostic(self.observe_raise(diagnostic))
        }
    }

    /// Reject a speculative attempt that has no diagnostic of its own.
    /// Callers reach this only while speculation is suppressed.
    pub(crate) fn raise_unrepresentable(&mut self) -> LoweringFailure {
        let UnrepresentableDag = self.observe_raise(UnrepresentableDag);
        LoweringFailure::Unrepresentable
    }

    /// Primitive. Run the native panic hook for an explicit raise.
    ///
    /// Raises `payload` from this module and catches it before any caller
    /// frame unwinds, so the hook observes the same thread, flags, payload
    /// and location as the former unwinding raise, before any finalizer.
    ///
    /// Model: append one hook observation `(u, o, kind of P)` to the external
    /// hook trace; return `payload`.
    fn observe_raise<P: Any + Send>(&mut self, payload: P) -> P {
        let raised = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            std::panic::panic_any(payload)
        }));
        match raised {
            Ok(never) => never,
            Err(caught) => match caught.downcast::<P>() {
                Ok(payload) => *payload,
                Err(_) => unreachable!("the caught payload is the one just raised"),
            },
        }
    }

    /// Re-enable quiet output after an inner catch has cleared its flag.
    pub(crate) fn raise_fatal_lowering_diagnostic(
        &mut self,
        diagnostic: LowerDiagnostic,
    ) -> LoweringFailure {
        self.set_output_suppressed(true);
        let fatal = if diagnostic.fatal {
            diagnostic
        } else {
            diagnostic.fatal()
        };
        self.raise_lowering_diagnostic(fatal)
    }

    /// Catch lowering failures and native lowering unwinds
    /// ([`Self::catch_lowering_with`] without state).
    pub fn catch_lowering<R>(
        &mut self,
        f: impl FnOnce(&mut LoweringRuntime) -> LowerResult<R> + std::panic::UnwindSafe,
    ) -> Result<R, LowerDiagnostic> {
        self.catch_lowering_with(&mut (), move |runtime, _| f(runtime))
    }

    /// Catch lowering failures and native lowering unwinds, with explicit
    /// `state` threaded through the catch.
    ///
    /// An explicit [`LoweringFailure`] maps to the diagnostic its former panic
    /// payload produced; an unexpected native panic maps as before.
    ///
    /// `f` receives the catch's runtime borrow and `state`; its other captures
    /// must still be unwind safe. After a caught unwind `state` is the
    /// caller's again, so callers pass state that every mutating scope exits
    /// on unwind (see [`scoped`]) or that an unwind leaves otherwise valid,
    /// such as native services whose next outermost entry resets them.
    ///
    /// Model: `f runtime state`, a [`LoweringFailure`] mapped to its
    /// diagnostic.
    pub fn catch_lowering_with<S, R>(
        &mut self,
        state: &mut S,
        f: impl FnOnce(&mut LoweringRuntime, &mut S) -> LowerResult<R> + std::panic::UnwindSafe,
    ) -> Result<R, LowerDiagnostic> {
        self.catch_native_unwind(
            state,
            move |runtime, state| f(runtime, state).map_err(LoweringFailure::into_diagnostic),
            unrepresentable_diagnostic,
            |message| {
                LowerDiagnostic::new(
                    message.unwrap_or_else(|| "internal lowering error".to_string()),
                    None,
                    None,
                )
            },
        )
    }

    /// Primitive. Run `f` on explicit `state`, quietly on the grown lowering
    /// stack.
    ///
    /// The grown stack covers lowering and its recursive consumers (AD, DCE,
    /// verification), not just the AST walk. An unexpected native unwind is
    /// converted in payload order: the unrepresentable marker by `marker`, a
    /// typed `D` payload by cloning it, and a `String`/`&str` message (or
    /// `None` for any other payload) by `message`. Quiet output stays enabled
    /// through that conversion and the payload's destruction, then clears.
    /// After a caught unwind `state` is the caller's again (see
    /// [`Self::catch_lowering_with`]).
    ///
    /// Model: `o := true; let r := f self state; o := false; r` (an
    /// unexpected native panic is outside the modeled domain).
    fn catch_native_unwind<S, T, D: Clone + 'static>(
        &mut self,
        state: &mut S,
        f: impl FnOnce(&mut Self, &mut S) -> Result<T, D> + std::panic::UnwindSafe,
        marker: impl FnOnce() -> D,
        message: impl FnOnce(Option<String>) -> D,
    ) -> Result<T, D> {
        struct ClearQuietOutput;
        impl Drop for ClearQuietOutput {
            fn drop(&mut self) {
                NATIVE_LOWERING_STATE.with(|state| state.suppress_output.set(false));
            }
        }
        install_chelis_panic_hook();
        self.set_output_suppressed(true);
        let _clear = ClearQuietOutput;
        let borrowed = std::panic::AssertUnwindSafe((self, state));
        match std::panic::catch_unwind(move || {
            // Move the whole wrapper; a field pattern would capture the borrows.
            let wrapped = borrowed;
            let (runtime, state) = wrapped.0;
            stacker::grow(LOWERING_GROW_SEGMENT_BYTES, move || f(runtime, state))
        }) {
            Ok(result) => result,
            Err(payload) => Err(if payload.is::<UnrepresentableDag>() {
                marker()
            } else if let Some(diagnostic) = payload.downcast_ref::<D>() {
                diagnostic.clone()
            } else {
                message(payload.downcast_ref::<String>().cloned().or_else(|| {
                    payload
                        .downcast_ref::<&str>()
                        .map(|value| (*value).to_string())
                }))
            }),
        }
    }

    /// Primitive. Allocate a host inference variable id from the
    /// process-global supply (relaxed `fetch_add`, wrapping at `u32::MAX`).
    /// The supply is never reset per invocation.
    ///
    /// Model: next element of the process-global allocation schedule.
    pub(crate) fn fresh_host_inference_var(&mut self) -> u32 {
        NEXT_HOST_INFERENCE_VAR.fetch_add(1, Ordering::Relaxed)
    }

    /// Primitive. Count one kernel-decision summary probe that missed its memo
    /// and ran a full callee lowering (chelis#1829). The count persists on
    /// this thread across invocations and saturates at `u64::MAX`.
    ///
    /// Model: `builds := min(builds + 1, u64::MAX)`.
    pub(crate) fn record_summary_probe_build(&mut self) {
        NATIVE_LOWERING_STATE.with(|state| {
            let counted = state.summary_probe_builds.get();
            state.summary_probe_builds.set(if counted == u64::MAX {
                counted
            } else {
                counted + 1
            });
        });
    }

    /// Primitive. This thread's summary-probe count.
    ///
    /// Model: read `builds`.
    pub(crate) fn summary_probe_builds(&mut self) -> u64 {
        NATIVE_LOWERING_STATE.with(|state| state.summary_probe_builds.get())
    }

    /// Primitive. Reset this thread's summary-probe count.
    ///
    /// Model: `builds := 0`.
    pub(crate) fn reset_summary_probe_builds(&mut self) {
        NATIVE_LOWERING_STATE.with(|state| state.summary_probe_builds.set(0));
    }

    /// Primitive. Allocate a tensor-helper preflight frame id from this
    /// thread's supply. The supply is never reset, so an id is never
    /// reissued on this thread: neither a sibling frame pushed after a frame
    /// pops nor a frame of a nested session can carry a live frame's id.
    ///
    /// Model: next element of this thread's frame-id allocation schedule.
    pub(crate) fn fresh_preflight_frame(&mut self) -> u64 {
        NATIVE_LOWERING_STATE.with(|state| {
            let id = state.next_preflight_frame.get();
            state.next_preflight_frame.set(id.wrapping_add(1));
            id
        })
    }

    /// Primitive. Start the detailed library-lowering profile. When
    /// `CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL=1` this reads the clock and
    /// returns a nonzero mark; otherwise it returns `0` and reads nothing.
    ///
    /// Model: an arbitrary `usize`, consumed only by the profile primitives.
    pub(crate) fn detail_profile_start(&mut self) -> usize {
        if std::env::var_os("CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL")
            .is_some_and(|value| value == "1")
        {
            self.profile_mark()
        } else {
            0
        }
    }

    /// Primitive. When `mark` is nonzero, report `"lower_sub: {seconds:>8.4}s
    /// {label}"` on stderr for the phase since `mark` and restart it.
    ///
    /// Model: no-op.
    pub(crate) fn detail_profile_phase(&mut self, mark: usize, label: &str) {
        if let Some(elapsed) = profile_restart(mark) {
            eprintln!("lower_sub: {:>8.4}s {}", elapsed, label);
        }
    }

    /// Primitive. When `mark` is nonzero, start one declaration's clock and
    /// return its nonzero mark; otherwise return `0`.
    ///
    /// Model: an arbitrary `usize`, consumed only by the profile primitives.
    pub(crate) fn detail_profile_declaration_start(&mut self, mark: usize) -> usize {
        if mark == 0 { 0 } else { self.profile_mark() }
    }

    /// Primitive. When `start` is nonzero, report one declaration on stderr as
    /// `"lower_decl: {seconds:>8.4}s nodes_added={:>5} dag_total={:>6} {name}"`
    /// with `nodes_added` measured from `last_dag_size`, which it advances.
    ///
    /// Model: no-op.
    pub(crate) fn detail_profile_declaration(
        &mut self,
        start: usize,
        last_dag_size: &mut usize,
        dag_total: usize,
        name: &str,
    ) {
        if let Some(elapsed) = profile_restart(start) {
            let added = dag_total.saturating_sub(*last_dag_size);
            *last_dag_size = dag_total;
            eprintln!(
                "lower_decl: {:>8.4}s nodes_added={:>5} dag_total={:>6} {}",
                elapsed, added, dag_total, name
            );
        }
    }

    fn profile_mark(&mut self) -> usize {
        PROFILE_MARKS.with(|marks| {
            let mut marks = marks.borrow_mut();
            marks.push(Instant::now());
            marks.len()
        })
    }

    /// Primitive. Write one internal diagnostic line to stderr.
    ///
    /// Model: no-op.
    pub(crate) fn report_internal(&mut self, line: &str) {
        eprintln!("{line}");
    }
}

/// Marker payload for a suppressed unrepresentable-DAG unwind.
pub(crate) struct UnrepresentableDag;

/// The per-thread state a [`LoweringRuntime`] handle names.
struct NativeLoweringState {
    /// `u`: a speculative DAG attempt is active.
    suppress_unrepresentable: Cell<bool>,
    /// `o`: panic output is quiet inside a lowering catch.
    suppress_output: Cell<bool>,
    /// Persistent kernel-decision summary-probe count (chelis#1829).
    summary_probe_builds: Cell<u64>,
    /// Next tensor-helper preflight frame id.
    next_preflight_frame: Cell<u64>,
}

thread_local! {
    static NATIVE_LOWERING_STATE: NativeLoweringState = const {
        NativeLoweringState {
            suppress_unrepresentable: Cell::new(false),
            suppress_output: Cell::new(false),
            summary_probe_builds: Cell::new(0),
            next_preflight_frame: Cell::new(0),
        }
    };
}

thread_local! {
    /// Profiling clocks named by nonzero marks; used only when detailed
    /// lowering profiling is enabled.
    static PROFILE_MARKS: std::cell::RefCell<Vec<Instant>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Seconds since `mark`'s clock, restarting it; `None` for mark `0`.
fn profile_restart(mark: usize) -> Option<f64> {
    let index = mark.checked_sub(1)?;
    PROFILE_MARKS.with(|marks| {
        let mut marks = marks.borrow_mut();
        let start = marks.get_mut(index)?;
        let elapsed = start.elapsed().as_secs_f64();
        *start = Instant::now();
        Some(elapsed)
    })
}

/// Process-global host inference variable supply.
static NEXT_HOST_INFERENCE_VAR: AtomicU32 = AtomicU32::new(0);

/// Borrow this thread's native lowering state without changing it.
///
/// Shell-only: native adapters acquire the handle outside semantic lowering
/// and pass it through their whole operation. Nested acquisitions name the
/// same state.
pub fn with_native_lowering_runtime<R>(f: impl FnOnce(&mut LoweringRuntime) -> R) -> R {
    f(&mut LoweringRuntime {
        thread_bound: PhantomData,
    })
}

// The existing boundary size also protects recursive consumers of bounded
// static unrolling; lowering's per-level stack growth alone is insufficient.
const LOWERING_GROW_SEGMENT_BYTES: usize = 512 * 1024 * 1024;

static CHELIS_PANIC_HOOK_INSTALLED: LazyLock<()> = LazyLock::new(|| {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let suppressed = NATIVE_LOWERING_STATE
            .with(|state| state.suppress_unrepresentable.get() || state.suppress_output.get());
        if !suppressed {
            previous(info);
        }
    }));
});

/// Install the native lowering hook once, retaining the previous hook.
///
/// Rust calls this hook before scope destruction. Reading native thread state
/// here is intentional: moving the read to a catch would change stderr and
/// previous-hook observations. Other threads use their own suppression flags.
pub fn install_chelis_panic_hook() {
    LazyLock::force(&CHELIS_PANIC_HOOK_INSTALLED);
}

fn unrepresentable_diagnostic() -> LowerDiagnostic {
    LowerDiagnostic::new(
        "program uses a form that is not supported by IR evaluation yet; use `chelis build --target c` instead",
        None,
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{AssertUnwindSafe, catch_unwind, panic_any};
    use std::process::Command;
    use std::sync::mpsc::{self, Receiver, Sender};

    const CHILD_ENV: &str = "CHELIS_LOWERING_RUNTIME_OBSERVATION_CHILD";
    const CHILD_DONE: &str = "native lowering runtime observations passed";

    #[derive(Debug, PartialEq, Eq)]
    enum Event {
        Hook(String),
        Drop(&'static str),
    }

    type Events = Sender<Event>;

    struct RecordDrop(&'static str, Events);

    impl Drop for RecordDrop {
        fn drop(&mut self) {
            self.1.send(Event::Drop(self.0)).unwrap();
        }
    }

    struct PayloadDrop(Events);

    impl Drop for PayloadDrop {
        fn drop(&mut self) {
            self.0.send(Event::Drop("payload")).unwrap();
            // Payload conversion is outside the caught unwind, but output
            // suppression must still cover destruction of the owned payload.
            assert!(catch_unwind(|| panic_any("payload-finalizer")).is_err());
        }
    }

    fn assert_events(events: &Receiver<Event>, expected: &[Event]) {
        let actual: Vec<_> = events.try_iter().collect();
        assert_eq!(actual, expected);
    }

    fn hook(message: &str) -> Event {
        Event::Hook(message.to_owned())
    }

    fn observe_native_runtime() {
        let (events, observed) = mpsc::channel();
        let hook_events = events.clone();
        std::panic::set_hook(Box::new(move |info| {
            let payload = info.payload();
            let label = if let Some(diagnostic) = payload.downcast_ref::<LowerDiagnostic>() {
                diagnostic.message.clone()
            } else if let Some(message) = payload.downcast_ref::<&str>() {
                (*message).to_owned()
            } else if let Some(message) = payload.downcast_ref::<String>() {
                message.clone()
            } else if payload.is::<UnrepresentableDag>() {
                "unrepresentable".to_owned()
            } else {
                "unknown".to_owned()
            };
            hook_events.send(Event::Hook(label)).unwrap();
        }));

        with_native_lowering_runtime(|runtime| {
            let diagnostic = LowerDiagnostic::new(
                "typed-rejection",
                Some(chelis_deep::Span { offset: 7, len: 13 }),
                Some("source-node".to_owned()),
            );
            let result: Result<(), _> = runtime.catch_lowering(|runtime| {
                let _drop = RecordDrop("typed", events.clone());
                Err(runtime.raise_lowering_diagnostic(diagnostic.clone()))
            });
            assert_eq!(result, Err(diagnostic.clone()));
            assert_events(&observed, &[Event::Drop("typed")]);

            // Both returning and unwinding inner catches CLEAR an outer flag.
            // A restore-previous guard would incorrectly hide the next hook.
            let result: Result<(), _> = runtime.catch_lowering(|runtime| {
                let _drop = RecordDrop("outer-return", events.clone());
                assert_eq!(runtime.catch_lowering(|_| Ok(17)), Ok(17));
                panic_any("after-inner-return");
            });
            assert_eq!(result.unwrap_err().message, "after-inner-return");
            assert_events(
                &observed,
                &[hook("after-inner-return"), Event::Drop("outer-return")],
            );

            let result: Result<(), _> = runtime.catch_lowering(|runtime| {
                let _drop = RecordDrop("outer-unwind", events.clone());
                let inner: Result<(), _> = runtime.catch_lowering(|_| {
                    let _drop = RecordDrop("inner-unwind", events.clone());
                    panic_any("quiet-inner");
                });
                assert_eq!(inner.unwrap_err().message, "quiet-inner");
                panic_any("after-inner-unwind");
            });
            assert_eq!(result.unwrap_err().message, "after-inner-unwind");
            assert_events(
                &observed,
                &[
                    Event::Drop("inner-unwind"),
                    hook("after-inner-unwind"),
                    Event::Drop("outer-unwind"),
                ],
            );

            let fatal = diagnostic.clone().fatal();
            let result: Result<(), _> = runtime.catch_lowering(|runtime| {
                runtime.with_suppress_unrepresentable_panic(|runtime| {
                    let inner: Result<(), _> = runtime.catch_lowering(|runtime| {
                        Err(runtime.raise_lowering_diagnostic(fatal.clone()))
                    });
                    let caught = inner.unwrap_err();
                    assert_eq!(caught, fatal);
                    Err(runtime.raise_fatal_lowering_diagnostic(caught))
                })
            });
            assert_eq!(result, Err(fatal));
            assert_events(&observed, &[]);

            // The same diagnostic becomes a marker only under speculation.
            let speculative = catch_unwind(AssertUnwindSafe(|| {
                runtime.with_suppress_unrepresentable_panic(|runtime| {
                    runtime
                        .raise_lowering_diagnostic(diagnostic.clone())
                        .resume_at_boundary();
                });
            }))
            .unwrap_err();
            assert!(speculative.is::<UnrepresentableDag>());
            assert_events(&observed, &[]);
            let ordinary = catch_unwind(AssertUnwindSafe(|| {
                runtime
                    .raise_lowering_diagnostic(diagnostic.clone())
                    .resume_at_boundary();
            }))
            .unwrap_err();
            assert_eq!(*ordinary.downcast::<LowerDiagnostic>().unwrap(), diagnostic);
            assert_events(&observed, &[hook("typed-rejection")]);

            // Nested acquisition shares the current native state rather than
            // allocating a fresh service or resetting its flags.
            runtime.with_suppress_unrepresentable_panic(|runtime| {
                let payload = catch_unwind(|| {
                    with_native_lowering_runtime(|nested| {
                        nested
                            .raise_lowering_diagnostic(diagnostic.clone())
                            .resume_at_boundary();
                    });
                })
                .unwrap_err();
                assert!(payload.is::<UnrepresentableDag>());
                runtime.with_suppress_unrepresentable_panic(|_| ());
                let payload = catch_unwind(AssertUnwindSafe(|| {
                    runtime
                        .raise_lowering_diagnostic(diagnostic.clone())
                        .resume_at_boundary();
                }))
                .unwrap_err();
                assert_eq!(*payload.downcast::<LowerDiagnostic>().unwrap(), diagnostic);
            });
            assert_events(&observed, &[hook("typed-rejection")]);

            runtime.with_suppress_unrepresentable_panic(|runtime| {
                let inner = catch_unwind(AssertUnwindSafe(|| {
                    runtime.with_suppress_unrepresentable_panic(|_| panic_any("speculative"));
                }));
                assert!(inner.is_err());
                let payload = catch_unwind(AssertUnwindSafe(|| {
                    runtime
                        .raise_lowering_diagnostic(diagnostic.clone())
                        .resume_at_boundary();
                }))
                .unwrap_err();
                assert!(payload.is::<LowerDiagnostic>());
            });
            assert_events(&observed, &[hook("typed-rejection")]);

            let unknown: Result<(), _> =
                runtime.catch_lowering(|_| panic_any(PayloadDrop(events.clone())));
            assert!(unknown.is_err());
            assert_events(&observed, &[Event::Drop("payload")]);
            assert!(catch_unwind(|| panic_any("after-payload")).is_err());
            assert_events(&observed, &[hook("after-payload")]);

            // A live quiet scope on this thread cannot suppress another
            // thread's hook. Its own explicit runtime still catches quietly.
            let quiet: Result<(), _> = runtime.catch_lowering(|_| {
                std::thread::spawn(|| {
                    assert!(catch_unwind(|| panic_any("other-thread")).is_err());
                    with_native_lowering_runtime(|other| {
                        let rejected: Result<(), _> =
                            other.catch_lowering(|_| panic_any("other-quiet"));
                        assert_eq!(rejected.unwrap_err().message, "other-quiet");
                    });
                })
                .join()
                .unwrap();
                panic_any("this-thread-quiet");
            });
            assert_eq!(quiet.unwrap_err().message, "this-thread-quiet");
            assert_events(&observed, &[hook("other-thread")]);
        });
    }

    #[test]
    fn native_hook_and_unwind_observations() {
        if std::env::var_os(CHILD_ENV).is_some() {
            observe_native_runtime();
            println!("{CHILD_DONE}");
            return;
        }

        // Hook installation is process-global and one-time. Isolate the
        // observation from every other test and native lowering invocation.
        let child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "lowering_runtime::tests::native_hook_and_unwind_observations",
                "--nocapture",
            ])
            .env(CHILD_ENV, "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&child.stdout);
        let stderr = String::from_utf8_lossy(&child.stderr);
        assert!(
            child.status.success() && stdout.lines().any(|line| line == CHILD_DONE),
            "native runtime probe failed: {}\n{stdout}\n{stderr}",
            child.status,
        );
    }
}
