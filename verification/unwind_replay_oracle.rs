//! Native oracle of the unwind replays (NATIVE_SHELL_SCOPE.md O8).
//!
//! The module `lowering_runtime` below repeats the native lowering runtime of
//! `crates/chelis-ir/src/lowering_runtime.rs` (LR:181-639) with two changes.
//! The fixture has no `stacker`, so `catch_native_unwind` runs its body on the
//! current stack. `LowerDiagnostic` keeps only its message, span and `fatal`.
//!
//! `main` runs the segments of `native_hook_and_unwind_observations`
//! (LR:703-846) in the same order on one thread, then three payload
//! conversions (LR:413-421). The previous hook records each raise that the
//! Chelis hook forwards. For each segment, `main` prints the recorded events
//! and the values that the statements return. The file
//! `verification/UnwindReplayChecks.lean` prints the same lines from the Lean
//! replays of `Runtime/Unwind.lean`. `check_models.py` compares the two outputs.
//!
//! A guard scope of the native test has no recorder. Thus the body of each
//! guard here declares `RecordDrop(fid)` as its first local. That local drops
//! immediately before the exit of the guard, at the position of the finalizer
//! event of the model.

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind, panic_any};
use std::sync::mpsc::{self, Receiver, Sender};

use lower::LowerDiagnostic;
use lowering_runtime::{UnrepresentableDag, with_native_lowering_runtime};

pub mod lower {
    /// The structured rejection, reduced to what the replays observe.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct LowerDiagnostic {
        pub message: String,
        pub span: Option<usize>,
        pub fatal: bool,
    }

    impl LowerDiagnostic {
        pub fn new(message: impl Into<String>, span: Option<usize>, span_id: Option<String>) -> Self {
            let _ = span_id;
            Self {
                message: message.into(),
                span,
                fatal: false,
            }
        }

        pub fn fatal(mut self) -> Self {
            self.fatal = true;
            self
        }
    }
}

pub mod lowering_runtime {
    use std::any::Any;
    use std::cell::Cell;
    use std::marker::PhantomData;
    use std::sync::LazyLock;

    use crate::lower::LowerDiagnostic;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum LoweringFailure {
        Unrepresentable,
        Diagnostic(LowerDiagnostic),
    }

    pub type LowerResult<T> = Result<T, LoweringFailure>;

    impl LoweringFailure {
        pub fn into_diagnostic(self) -> LowerDiagnostic {
            match self {
                Self::Unrepresentable => unrepresentable_diagnostic(),
                Self::Diagnostic(diagnostic) => diagnostic,
            }
        }

        pub fn resume_at_boundary(self) -> ! {
            match self {
                Self::Unrepresentable => resume_native_unwind(UnrepresentableDag),
                Self::Diagnostic(diagnostic) => resume_native_unwind(diagnostic),
            }
        }
    }

    fn resume_native_unwind<P: Any + Send>(payload: P) -> ! {
        std::panic::resume_unwind(Box::new(payload))
    }

    pub fn scoped<S, T, R>(
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

    pub struct LoweringRuntime {
        thread_bound: PhantomData<*const ()>,
    }

    impl LoweringRuntime {
        pub fn unrepresentable_panic_suppressed(&mut self) -> bool {
            NATIVE_LOWERING_STATE.with(|state| state.suppress_unrepresentable.get())
        }

        fn set_unrepresentable_panic_suppressed(&mut self, value: bool) {
            NATIVE_LOWERING_STATE.with(|state| state.suppress_unrepresentable.set(value));
        }

        fn set_output_suppressed(&mut self, value: bool) {
            NATIVE_LOWERING_STATE.with(|state| state.suppress_output.set(value));
        }

        pub fn with_suppress_unrepresentable_panic<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
            scoped(
                self,
                |runtime| runtime.set_unrepresentable_panic_suppressed(true),
                |runtime, ()| runtime.set_unrepresentable_panic_suppressed(false),
                f,
            )
        }

        pub fn raise_lowering_diagnostic(&mut self, diagnostic: LowerDiagnostic) -> LoweringFailure {
            if !diagnostic.fatal && self.unrepresentable_panic_suppressed() {
                self.raise_unrepresentable()
            } else {
                LoweringFailure::Diagnostic(self.observe_raise(diagnostic))
            }
        }

        pub fn raise_unrepresentable(&mut self) -> LoweringFailure {
            let UnrepresentableDag = self.observe_raise(UnrepresentableDag);
            LoweringFailure::Unrepresentable
        }

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

        pub fn raise_fatal_lowering_diagnostic(&mut self, diagnostic: LowerDiagnostic) -> LoweringFailure {
            self.set_output_suppressed(true);
            let fatal = if diagnostic.fatal {
                diagnostic
            } else {
                diagnostic.fatal()
            };
            self.raise_lowering_diagnostic(fatal)
        }

        pub fn catch_lowering<R>(
            &mut self,
            f: impl FnOnce(&mut LoweringRuntime) -> LowerResult<R> + std::panic::UnwindSafe,
        ) -> Result<R, LowerDiagnostic> {
            self.catch_lowering_with(&mut (), move |runtime, _| f(runtime))
        }

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
                let wrapped = borrowed;
                let (runtime, state) = wrapped.0;
                f(runtime, state)
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
    }

    pub struct UnrepresentableDag;

    struct NativeLoweringState {
        suppress_unrepresentable: Cell<bool>,
        suppress_output: Cell<bool>,
    }

    thread_local! {
        static NATIVE_LOWERING_STATE: NativeLoweringState = const {
            NativeLoweringState {
                suppress_unrepresentable: Cell::new(false),
                suppress_output: Cell::new(false),
            }
        };
    }

    pub fn with_native_lowering_runtime<R>(f: impl FnOnce(&mut LoweringRuntime) -> R) -> R {
        f(&mut LoweringRuntime {
            thread_bound: PhantomData,
        })
    }

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
}

type Events = Sender<String>;

/// A local with an observable destructor (`Prog.scope .noop fid`).
struct RecordDrop(u32, Events);

impl Drop for RecordDrop {
    fn drop(&mut self) {
        self.1.send(format!("drop:{}", self.0)).unwrap();
    }
}

/// A payload with an observable destructor (`Payload.other id true`).
struct Other(u32, Events);

impl Drop for Other {
    fn drop(&mut self) {
        self.1.send(format!("payload-drop:{}", self.0)).unwrap();
        // The native test also panics in this destructor. Quiet output covers it.
        assert!(catch_unwind(|| panic_any("payload-finalizer")).is_err());
    }
}

fn diagnostic_label(diagnostic: &LowerDiagnostic) -> String {
    format!("{}{}", diagnostic.message, if diagnostic.fatal { "!" } else { "" })
}

fn payload_label(payload: &(dyn Any + Send)) -> String {
    if let Some(diagnostic) = payload.downcast_ref::<LowerDiagnostic>() {
        format!("diag({})", diagnostic_label(diagnostic))
    } else if payload.is::<UnrepresentableDag>() {
        "unrep".to_owned()
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        format!("str({message})")
    } else if let Some(message) = payload.downcast_ref::<String>() {
        format!("str({message})")
    } else if let Some(other) = payload.downcast_ref::<Other>() {
        format!("other({})", other.0)
    } else {
        "unknown".to_owned()
    }
}

/// The value of a `catch_unwind` statement.
fn caught(result: std::thread::Result<()>) -> String {
    match result {
        Ok(()) => "ok(0)".to_owned(),
        Err(payload) => format!("unwind:{}", payload_label(&*payload)),
    }
}

/// The value of a `catch_lowering` statement.
fn lowered(result: Result<(), LowerDiagnostic>) -> String {
    match result {
        Ok(()) => "ok(0)".to_owned(),
        Err(diagnostic) => format!("err({})", diagnostic_label(&diagnostic)),
    }
}

fn segment(name: &str, observed: &Receiver<String>, returned: Vec<String>) {
    let events: Vec<String> = observed.try_iter().collect();
    println!("segment {name}: [{}] [{}]", events.join(", "), returned.join(", "));
}

fn main() {
    let (events, observed) = mpsc::channel();
    let hook_events = events.clone();
    std::panic::set_hook(Box::new(move |info| {
        hook_events.send(format!("hook:{}", payload_label(info.payload()))).unwrap();
    }));

    with_native_lowering_runtime(|runtime| {
        let diagnostic = LowerDiagnostic::new("typed-rejection", Some(7), Some("source-node".to_owned()));

        let result: Result<(), _> = runtime.catch_lowering(|runtime| {
            let _drop = RecordDrop(1, events.clone());
            Err(runtime.raise_lowering_diagnostic(diagnostic.clone()))
        });
        segment("typed", &observed, vec![lowered(result)]);

        let result: Result<(), _> = runtime.catch_lowering(|runtime| {
            let _drop = RecordDrop(2, events.clone());
            assert_eq!(runtime.catch_lowering(|_| Ok(17)), Ok(17));
            panic_any("after-inner-return");
        });
        segment("after-inner-return", &observed, vec![lowered(result)]);

        let result: Result<(), _> = runtime.catch_lowering(|runtime| {
            let _drop = RecordDrop(3, events.clone());
            let inner: Result<(), _> = runtime.catch_lowering(|_| {
                let _drop = RecordDrop(4, events.clone());
                panic_any("quiet-inner");
            });
            assert_eq!(inner.unwrap_err().message, "quiet-inner");
            panic_any("after-inner-unwind");
        });
        segment("after-inner-unwind", &observed, vec![lowered(result)]);

        let fatal = diagnostic.clone().fatal();
        let result: Result<(), _> = runtime.catch_lowering(|runtime| {
            runtime.with_suppress_unrepresentable_panic(|runtime| {
                let _drop = RecordDrop(5, events.clone());
                let inner: Result<(), _> = runtime.catch_lowering(|runtime| {
                    Err(runtime.raise_lowering_diagnostic(fatal.clone()))
                });
                let caught = inner.unwrap_err();
                assert_eq!(caught, fatal);
                Err(runtime.raise_fatal_lowering_diagnostic(caught))
            })
        });
        segment("fatal-reraise", &observed, vec![lowered(result)]);

        let speculative = catch_unwind(AssertUnwindSafe(|| {
            runtime.with_suppress_unrepresentable_panic(|runtime| {
                let _drop = RecordDrop(6, events.clone());
                runtime
                    .raise_lowering_diagnostic(diagnostic.clone())
                    .resume_at_boundary();
            });
        }));
        let ordinary = catch_unwind(AssertUnwindSafe(|| {
            runtime
                .raise_lowering_diagnostic(diagnostic.clone())
                .resume_at_boundary();
        }));
        segment("speculative-boundary", &observed, vec![caught(speculative), caught(ordinary)]);

        let mut returned = Vec::new();
        runtime.with_suppress_unrepresentable_panic(|runtime| {
            let _drop = RecordDrop(8, events.clone());
            returned.push(caught(catch_unwind(AssertUnwindSafe(|| {
                with_native_lowering_runtime(|nested| {
                    nested
                        .raise_lowering_diagnostic(diagnostic.clone())
                        .resume_at_boundary();
                });
            }))));
            runtime.with_suppress_unrepresentable_panic(|_| {
                let _drop = RecordDrop(7, events.clone());
            });
            returned.push(caught(catch_unwind(AssertUnwindSafe(|| {
                runtime
                    .raise_lowering_diagnostic(diagnostic.clone())
                    .resume_at_boundary();
            }))));
        });
        segment("nested-acquisition", &observed, returned);

        let mut returned = Vec::new();
        runtime.with_suppress_unrepresentable_panic(|runtime| {
            let _drop = RecordDrop(10, events.clone());
            returned.push(caught(catch_unwind(AssertUnwindSafe(|| {
                runtime.with_suppress_unrepresentable_panic(|_| {
                    let _drop = RecordDrop(9, events.clone());
                    panic_any("speculative")
                });
            }))));
            returned.push(caught(catch_unwind(AssertUnwindSafe(|| {
                runtime
                    .raise_lowering_diagnostic(diagnostic.clone())
                    .resume_at_boundary();
            }))));
        });
        segment("speculative-unwind", &observed, returned);

        let unknown: Result<(), _> =
            runtime.catch_lowering(|_| panic_any(Other(11, events.clone())));
        let after = caught(catch_unwind(|| panic_any("after-payload")));
        segment("payload-drop", &observed, vec![lowered(unknown), after]);

        // A live quiet scope on this thread cannot suppress the hook of another
        // thread. The other thread still catches quietly in its own catch.
        let mut returned = Vec::new();
        let mut slot = AssertUnwindSafe(&mut returned);
        let quiet: Result<(), _> = runtime.catch_lowering(move |_| {
            let other = std::thread::spawn(|| {
                let first = caught(catch_unwind(|| panic_any("other-thread")));
                let second = with_native_lowering_runtime(|other| {
                    lowered(other.catch_lowering(|_| panic_any("other-quiet")))
                });
                vec![first, second]
            })
            .join()
            .unwrap();
            slot.extend(other);
            panic_any("this-thread-quiet");
        });
        returned.push(lowered(quiet));
        segment("other-thread", &observed, returned);

        let converted: Result<(), _> = runtime.catch_lowering(|_| {
            panic_any(LowerDiagnostic::new("converted", Some(3), None).fatal())
        });
        segment("convert-diagnostic", &observed, vec![lowered(converted)]);

        let converted: Result<(), _> = runtime.catch_lowering(|_| panic_any(UnrepresentableDag));
        segment("convert-unrepresentable", &observed, vec![lowered(converted)]);

        let converted: Result<(), _> = runtime.catch_lowering(|_| panic_any(String::from("owned")));
        segment("convert-string", &observed, vec![lowered(converted)]);
    });
}
