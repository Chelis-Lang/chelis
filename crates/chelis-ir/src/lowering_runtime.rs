//! Native panic services borrowed by DAG and host lowering.
//!
//! Semantic callers receive [`LoweringRuntime`] explicitly. Only this boundary
//! selects the current thread's state and connects it to Rust's process-global
//! panic hook. The state is shared by nested native entries on that thread;
//! borrowing it does not start or reset a suppression scope.
//!
//! This remains a native effect boundary, not a pure error interpreter. Rust
//! invokes the hook before unwinding, and lowering's guards deliberately clear
//! their flags rather than restoring enclosing values.

use std::any::Any;
use std::cell::Cell;
use std::sync::LazyLock;

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

/// The current native thread's lowering panic state.
///
/// Obtain a borrowed service with [`with_native_lowering_runtime`]. The private
/// constructor prevents a detached state whose suppression flags the native
/// hook cannot observe. The service is thread-bound; checked programs and
/// persisted lowering facts do not contain it.
#[derive(Debug)]
pub struct LoweringRuntime {
    suppress_unrepresentable: Cell<bool>,
    suppress_output: Cell<bool>,
}

// Every flag value is valid across an unwind. Scope destruction clears its own
// flag, including after a nested catch; fatal re-raise explicitly re-enables
// output suppression. No partially initialized semantic product lives here.
impl std::panic::RefUnwindSafe for LoweringRuntime {}

impl LoweringRuntime {
    const fn new() -> Self {
        Self {
            suppress_unrepresentable: Cell::new(false),
            suppress_output: Cell::new(false),
        }
    }

    /// Run a speculative DAG attempt without native panic output.
    ///
    /// A nonfatal lowering rejection becomes an `UnrepresentableDag` payload;
    /// fatal diagnostics retain their structured payload. Return and unwind
    /// both clear the flag, even when an enclosing scope had set it.
    pub fn with_suppress_unrepresentable_panic<R>(&self, f: impl FnOnce() -> R) -> R {
        self.suppress_unrepresentable.set(true);
        let _guard = ClearOnDrop(&self.suppress_unrepresentable);
        f()
    }

    pub(crate) fn unrepresentable_panic_suppressed(&self) -> bool {
        self.suppress_unrepresentable.get()
    }

    /// Reject the current lowering step.
    ///
    /// A nonfatal rejection under speculation becomes
    /// [`LoweringFailure::Unrepresentable`]; fatal diagnostics stay structured.
    pub(crate) fn raise_lowering_diagnostic(&self, diagnostic: LowerDiagnostic) -> LoweringFailure {
        // Fatal AD rejection must not become a speculative host fallback.
        if !diagnostic.fatal && self.unrepresentable_panic_suppressed() {
            self.raise_unrepresentable()
        } else {
            LoweringFailure::Diagnostic(self.observe_raise(diagnostic))
        }
    }

    /// Reject a speculative attempt that has no diagnostic of its own.
    /// Callers reach this only while speculation is suppressed.
    pub(crate) fn raise_unrepresentable(&self) -> LoweringFailure {
        let UnrepresentableDag = self.observe_raise(UnrepresentableDag);
        LoweringFailure::Unrepresentable
    }

    /// Primitive. Run the native panic hook for an explicit raise.
    ///
    /// Raises `payload` from this module and catches it before any caller
    /// frame unwinds, so the hook observes the same thread, flags, payload
    /// and location as the former unwinding raise, before any finalizer.
    ///
    /// Model: identity; the hook's stderr effect is external.
    fn observe_raise<P: Any + Send>(&self, payload: P) -> P {
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
        &self,
        diagnostic: LowerDiagnostic,
    ) -> LoweringFailure {
        self.suppress_output.set(true);
        let fatal = if diagnostic.fatal {
            diagnostic
        } else {
            diagnostic.fatal()
        };
        self.raise_lowering_diagnostic(fatal)
    }

    /// Catch lowering failures and native lowering unwinds.
    ///
    /// An explicit [`LoweringFailure`] maps to the diagnostic its former panic
    /// payload produced; an unexpected native panic maps as before.
    pub fn catch_lowering<R>(
        &self,
        f: impl FnOnce() -> LowerResult<R> + std::panic::UnwindSafe,
    ) -> Result<R, LowerDiagnostic> {
        self.catch_native_unwind(
            move || f().map_err(LoweringFailure::into_diagnostic),
            panic_payload_to_lower_diagnostic,
        )
    }

    /// Primitive. Run `f` quietly on the grown lowering stack.
    ///
    /// The grown stack covers lowering and its recursive consumers (AD, DCE,
    /// verification), not just the AST walk. An unexpected native unwind is
    /// converted by `recover`; the guard stays live through that conversion
    /// and clears output suppression before returning.
    ///
    /// Model: `f()`; an unexpected native panic is outside the modeled domain.
    fn catch_native_unwind<T, E>(
        &self,
        f: impl FnOnce() -> Result<T, E> + std::panic::UnwindSafe,
        recover: impl FnOnce(&(dyn Any + Send)) -> E,
    ) -> Result<T, E> {
        install_chelis_panic_hook();
        self.suppress_output.set(true);
        let _guard = ClearOnDrop(&self.suppress_output);
        match std::panic::catch_unwind(|| stacker::grow(LOWERING_GROW_SEGMENT_BYTES, f)) {
            Ok(result) => result,
            Err(payload) => Err(recover(&*payload)),
        }
    }
}

/// Marker payload for a suppressed unrepresentable-DAG unwind.
pub(crate) struct UnrepresentableDag;

struct ClearOnDrop<'a>(&'a Cell<bool>);

impl Drop for ClearOnDrop<'_> {
    fn drop(&mut self) {
        self.0.set(false);
    }
}

thread_local! {
    static NATIVE_LOWERING_RUNTIME: LoweringRuntime = const { LoweringRuntime::new() };
}

/// Borrow this thread's native lowering service without changing its state.
///
/// Native adapters acquire it outside semantic lowering and pass the reference
/// through their whole operation. Nested acquisitions borrow the same state.
pub fn with_native_lowering_runtime<R>(f: impl FnOnce(&LoweringRuntime) -> R) -> R {
    NATIVE_LOWERING_RUNTIME.with(f)
}

// The existing boundary size also protects recursive consumers of bounded
// static unrolling; lowering's per-level stack growth alone is insufficient.
const LOWERING_GROW_SEGMENT_BYTES: usize = 512 * 1024 * 1024;

static CHELIS_PANIC_HOOK_INSTALLED: LazyLock<()> = LazyLock::new(|| {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let suppressed = NATIVE_LOWERING_RUNTIME.with(|runtime| {
            runtime.suppress_unrepresentable.get() || runtime.suppress_output.get()
        });
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

fn panic_payload_to_lower_diagnostic(payload: &(dyn Any + Send)) -> LowerDiagnostic {
    if payload.is::<UnrepresentableDag>() {
        return unrepresentable_diagnostic();
    }
    if let Some(diagnostic) = payload.downcast_ref::<LowerDiagnostic>() {
        return diagnostic.clone();
    }
    let message = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| {
            payload
                .downcast_ref::<&str>()
                .map(|value| (*value).to_string())
        })
        .unwrap_or_else(|| "internal lowering error".to_string());
    LowerDiagnostic::new(message, None, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, panic_any};
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
            let result: Result<(), _> = runtime.catch_lowering(|| {
                let _drop = RecordDrop("typed", events.clone());
                Err(runtime.raise_lowering_diagnostic(diagnostic.clone()))
            });
            assert_eq!(result, Err(diagnostic.clone()));
            assert_events(&observed, &[Event::Drop("typed")]);

            // Both returning and unwinding inner catches CLEAR an outer flag.
            // A restore-previous guard would incorrectly hide the next hook.
            let result: Result<(), _> = runtime.catch_lowering(|| {
                let _drop = RecordDrop("outer-return", events.clone());
                assert_eq!(runtime.catch_lowering(|| Ok(17)), Ok(17));
                panic_any("after-inner-return");
            });
            assert_eq!(result.unwrap_err().message, "after-inner-return");
            assert_events(
                &observed,
                &[hook("after-inner-return"), Event::Drop("outer-return")],
            );

            let result: Result<(), _> = runtime.catch_lowering(|| {
                let _drop = RecordDrop("outer-unwind", events.clone());
                let inner: Result<(), _> = runtime.catch_lowering(|| {
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
            let result: Result<(), _> = runtime.catch_lowering(|| {
                runtime.with_suppress_unrepresentable_panic(|| {
                    let inner: Result<(), _> = runtime
                        .catch_lowering(|| Err(runtime.raise_lowering_diagnostic(fatal.clone())));
                    let caught = inner.unwrap_err();
                    assert_eq!(caught, fatal);
                    Err(runtime.raise_fatal_lowering_diagnostic(caught))
                })
            });
            assert_eq!(result, Err(fatal));
            assert_events(&observed, &[]);

            // The same diagnostic becomes a marker only under speculation.
            let speculative = catch_unwind(|| {
                runtime.with_suppress_unrepresentable_panic(|| {
                    runtime
                        .raise_lowering_diagnostic(diagnostic.clone())
                        .resume_at_boundary();
                });
            })
            .unwrap_err();
            assert!(speculative.is::<UnrepresentableDag>());
            assert_events(&observed, &[]);
            let ordinary = catch_unwind(|| {
                runtime
                    .raise_lowering_diagnostic(diagnostic.clone())
                    .resume_at_boundary();
            })
            .unwrap_err();
            assert_eq!(*ordinary.downcast::<LowerDiagnostic>().unwrap(), diagnostic);
            assert_events(&observed, &[hook("typed-rejection")]);

            // Nested acquisition shares the current native state rather than
            // allocating a fresh service or resetting its flags.
            runtime.with_suppress_unrepresentable_panic(|| {
                let payload = catch_unwind(|| {
                    with_native_lowering_runtime(|nested| {
                        nested
                            .raise_lowering_diagnostic(diagnostic.clone())
                            .resume_at_boundary();
                    });
                })
                .unwrap_err();
                assert!(payload.is::<UnrepresentableDag>());
                runtime.with_suppress_unrepresentable_panic(|| ());
                let payload = catch_unwind(|| {
                    runtime
                        .raise_lowering_diagnostic(diagnostic.clone())
                        .resume_at_boundary();
                })
                .unwrap_err();
                assert_eq!(*payload.downcast::<LowerDiagnostic>().unwrap(), diagnostic);
            });
            assert_events(&observed, &[hook("typed-rejection")]);

            runtime.with_suppress_unrepresentable_panic(|| {
                let inner = catch_unwind(|| {
                    runtime.with_suppress_unrepresentable_panic(|| panic_any("speculative"));
                });
                assert!(inner.is_err());
                let payload = catch_unwind(|| {
                    runtime
                        .raise_lowering_diagnostic(diagnostic.clone())
                        .resume_at_boundary();
                })
                .unwrap_err();
                assert!(payload.is::<LowerDiagnostic>());
            });
            assert_events(&observed, &[hook("typed-rejection")]);

            let unknown: Result<(), _> =
                runtime.catch_lowering(|| panic_any(PayloadDrop(events.clone())));
            assert!(unknown.is_err());
            assert_events(&observed, &[Event::Drop("payload")]);
            assert!(catch_unwind(|| panic_any("after-payload")).is_err());
            assert_events(&observed, &[hook("after-payload")]);

            // A live quiet scope on this thread cannot suppress another
            // thread's hook. Its own explicit runtime still catches quietly.
            let quiet: Result<(), _> = runtime.catch_lowering(|| {
                std::thread::spawn(|| {
                    assert!(catch_unwind(|| panic_any("other-thread")).is_err());
                    with_native_lowering_runtime(|other| {
                        let rejected: Result<(), _> =
                            other.catch_lowering(|| panic_any("other-quiet"));
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
