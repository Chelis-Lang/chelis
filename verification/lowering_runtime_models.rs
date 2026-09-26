//! The native lowering-runtime primitives of `crates/chelis-ir/src/lowering_runtime.rs`
//! against their Lean models (`Aeneas/Std/ChelisNative.lean`).
//!
//! Compiled as crate `chelis_ir`, so every primitive has its production path and
//! Aeneas replaces it with its model. `lowering_runtime` repeats the primitive
//! bodies and their thread-local state verbatim, and the ordinary wrappers
//! (`with_suppress_unrepresentable_panic`, the raises, `catch_lowering`,
//! `catch_lowering_with`) with the same structure. Differences:
//! - `stacker` is not linkable into this single-file fixture, so the grown stack
//!   segments are omitted (a segment only changes the stack an unchanged closure
//!   runs on);
//! - `LowerDiagnostic` is reduced to a code and the `fatal` flag.
//! The native shell is hard-opaque through the `chelis_ir::lowering_runtime`
//! patterns of `verification/native_shell.json`, as in the production extraction.
//! The scenario functions call the primitives the way `lower.rs` and `host.rs` do.

pub mod lower {
    /// The structured rejection, reduced to what the primitives observe.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct LowerDiagnostic {
        pub code: u32,
        pub fatal: bool,
    }

    impl LowerDiagnostic {
        pub fn new(code: u32) -> Self {
            Self { code, fatal: false }
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
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Instant;

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

    pub fn grow_stack_if_needed<S, R>(
        state: &mut S,
        red_zone: usize,
        stack_size: usize,
        f: impl FnOnce(&mut S) -> R,
    ) -> R {
        let _ = (red_zone, stack_size);
        f(state)
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

        pub fn with_suppress_unrepresentable_panic<R>(
            &mut self,
            f: impl FnOnce(&mut Self) -> R,
        ) -> R {
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

        pub fn raise_fatal_lowering_diagnostic(
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
                |message| LowerDiagnostic::new(if message.is_some() { 2 } else { 3 }),
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

        pub fn fresh_host_inference_var(&mut self) -> u32 {
            NEXT_HOST_INFERENCE_VAR.fetch_add(1, Ordering::Relaxed)
        }

        pub fn record_summary_probe_build(&mut self) {
            NATIVE_LOWERING_STATE.with(|state| {
                let counted = state.summary_probe_builds.get();
                state.summary_probe_builds.set(if counted == u64::MAX {
                    counted
                } else {
                    counted + 1
                });
            });
        }

        pub fn summary_probe_builds(&mut self) -> u64 {
            NATIVE_LOWERING_STATE.with(|state| state.summary_probe_builds.get())
        }

        pub fn reset_summary_probe_builds(&mut self) {
            NATIVE_LOWERING_STATE.with(|state| state.summary_probe_builds.set(0));
        }

        pub fn fresh_preflight_frame(&mut self) -> u64 {
            NATIVE_LOWERING_STATE.with(|state| {
                let id = state.next_preflight_frame.get();
                state.next_preflight_frame.set(id.wrapping_add(1));
                id
            })
        }

        pub fn detail_profile_start(&mut self) -> usize {
            if std::env::var_os("CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL")
                .is_some_and(|value| value == "1")
            {
                self.profile_mark()
            } else {
                0
            }
        }

        pub fn detail_profile_phase(&mut self, mark: usize, label: &str) {
            if let Some(elapsed) = profile_restart(mark) {
                eprintln!("lower_sub: {:>8.4}s {}", elapsed, label);
            }
        }

        pub fn detail_profile_declaration_start(&mut self, mark: usize) -> usize {
            if mark == 0 { 0 } else { self.profile_mark() }
        }

        pub fn detail_profile_declaration(
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

        pub fn report_internal(&mut self, line: &str) {
            eprintln!("{line}");
        }
    }

    pub struct UnrepresentableDag;

    struct NativeLoweringState {
        suppress_unrepresentable: Cell<bool>,
        suppress_output: Cell<bool>,
        summary_probe_builds: Cell<u64>,
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
        static PROFILE_MARKS: std::cell::RefCell<Vec<Instant>> = const { std::cell::RefCell::new(Vec::new()) };
    }

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

    static NEXT_HOST_INFERENCE_VAR: AtomicU32 = AtomicU32::new(0);

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
        LowerDiagnostic::new(1)
    }

    /// Test access to this thread's native state (the fixture's observation
    /// points; production code has none).
    #[cfg(test)]
    pub(crate) mod observe {
        use super::NATIVE_LOWERING_STATE;

        pub fn flags() -> (bool, bool) {
            NATIVE_LOWERING_STATE
                .with(|state| (state.suppress_unrepresentable.get(), state.suppress_output.get()))
        }

        pub fn set_counters(builds: u64, next_frame: u64) {
            NATIVE_LOWERING_STATE.with(|state| {
                state.summary_probe_builds.set(builds);
                state.next_preflight_frame.set(next_frame);
            });
        }
    }
}

use lower::LowerDiagnostic;
use lowering_runtime::{
    LowerResult, LoweringFailure, LoweringRuntime, grow_stack_if_needed, scoped,
};

/// `u` outside, inside and after a speculative attempt.
pub fn speculation_flags(runtime: &mut LoweringRuntime) -> (bool, bool, bool) {
    let before = runtime.unrepresentable_panic_suppressed();
    let inside = runtime.with_suppress_unrepresentable_panic(|runtime| {
        runtime.unrepresentable_panic_suppressed()
    });
    let after = runtime.unrepresentable_panic_suppressed();
    (before, inside, after)
}

/// A nonfatal rejection inside a speculative attempt inside a catch.
pub fn speculative_rejection(runtime: &mut LoweringRuntime, code: u32) -> Result<u32, LowerDiagnostic> {
    runtime.catch_lowering(|runtime| {
        runtime.with_suppress_unrepresentable_panic(|runtime| {
            Err(runtime.raise_lowering_diagnostic(LowerDiagnostic::new(code)))
        })
    })
}

/// A rejection inside a catch, after an inner catch returned (which clears `o`).
pub fn rejection_after_inner_catch(
    runtime: &mut LoweringRuntime,
    code: u32,
) -> Result<u32, LowerDiagnostic> {
    runtime.catch_lowering(|runtime| {
        let inner = runtime.catch_lowering(|_| Ok(17u32));
        match inner {
            Ok(value) if value == code => Ok(value),
            _ => Err(runtime.raise_lowering_diagnostic(LowerDiagnostic::new(code))),
        }
    })
}

/// A fatal rejection caught under speculation and re-raised (as `host.rs` does).
pub fn fatal_reraise(runtime: &mut LoweringRuntime, code: u32) -> Result<u32, LowerDiagnostic> {
    runtime.catch_lowering(|runtime| {
        runtime.with_suppress_unrepresentable_panic(|runtime| {
            let inner: Result<u32, LowerDiagnostic> = runtime.catch_lowering(|runtime| {
                Err(runtime.raise_lowering_diagnostic(LowerDiagnostic::new(code).fatal()))
            });
            match inner {
                Ok(value) => Ok(value),
                Err(caught) => Err(runtime.raise_fatal_lowering_diagnostic(caught)),
            }
        })
    })
}

/// A rejection outside any catch or speculation.
pub fn uncaught_rejection(runtime: &mut LoweringRuntime, code: u32) -> LoweringFailure {
    runtime.raise_lowering_diagnostic(LowerDiagnostic::new(code))
}

/// A counter threaded through `catch_lowering_with`, as `lower.rs` threads `native`.
pub fn caught_with_state(
    runtime: &mut LoweringRuntime,
    count: &mut u32,
    reject: bool,
) -> Result<u32, LowerDiagnostic> {
    runtime.catch_lowering_with(count, |runtime, count| {
        *count += 1;
        if reject {
            Err(runtime.raise_unrepresentable())
        } else {
            Ok(*count)
        }
    })
}

/// Summary-probe builds: `n` records, the count, then the count after a reset.
pub fn summary_probes(runtime: &mut LoweringRuntime, n: u32) -> (u64, u64) {
    let mut recorded = 0;
    while recorded < n {
        runtime.record_summary_probe_build();
        recorded += 1;
    }
    let counted = runtime.summary_probe_builds();
    runtime.reset_summary_probe_builds();
    (counted, runtime.summary_probe_builds())
}

pub fn preflight_frames(runtime: &mut LoweringRuntime) -> (u64, u64, u64) {
    let first = runtime.fresh_preflight_frame();
    let second = runtime.fresh_preflight_frame();
    let third = runtime.fresh_preflight_frame();
    (first, second, third)
}

pub fn host_inference_vars(runtime: &mut LoweringRuntime) -> (u32, u32) {
    let first = runtime.fresh_host_inference_var();
    let second = runtime.fresh_host_inference_var();
    (first, second)
}

/// The detailed profile of `lower.rs`'s top-level loop: the start mark, each
/// declaration's mark, and the node count the declarations last advanced to.
pub fn profile_declarations(runtime: &mut LoweringRuntime, totals: &[usize]) -> (usize, usize, usize) {
    let profile = runtime.detail_profile_start();
    runtime.detail_profile_phase(profile, "collect_top_level_defs");
    let mut last_dag_size = 0;
    let mut last_start = 0;
    let mut index = 0;
    while index < totals.len() {
        let declaration_start = runtime.detail_profile_declaration_start(profile);
        runtime.detail_profile_declaration(declaration_start, &mut last_dag_size, totals[index], "decl");
        last_start = declaration_start;
        index += 1;
    }
    runtime.detail_profile_phase(profile, "lower_top_level_loop");
    runtime.report_internal("internal");
    (profile, last_start, last_dag_size)
}

/// A declaration reported under a mark this thread never issued.
pub fn profile_unissued_mark(runtime: &mut LoweringRuntime, mark: usize, total: usize) -> usize {
    let mut last_dag_size = 5;
    runtime.detail_profile_declaration(mark, &mut last_dag_size, total, "decl");
    last_dag_size
}

pub struct Scope {
    pub items: Vec<u32>,
    pub exits: u32,
}

/// `scoped` as `host.rs`'s `with_inlining`: enter pushes, exit pops.
pub fn scoped_push(scope: &mut Scope, item: u32) -> usize {
    scoped(
        scope,
        move |scope| {
            scope.items.push(item);
            scope.items.len()
        },
        |scope, entered| {
            scope.exits += entered as u32;
            scope.items.pop();
        },
        |scope| {
            scope.items.push(item + 1);
            scope.items.len()
        },
    )
}

/// `grow_stack_if_needed` as `lower.rs` calls it on its context.
pub fn grown_push(scope: &mut Scope, item: u32) -> LowerResult<usize> {
    let length = grow_stack_if_needed(scope, 64 * 1024, 4 * 1024 * 1024, |scope| {
        scope.items.push(item);
        Ok(scope.items.len())
    })?;
    Ok(length + 1)
}

/// `host.rs`'s `HostCx`: a shared and three mutable borrows.
pub struct HostCx<'a> {
    pub checked: &'a u32,
    pub scope: &'a mut Scope,
    pub runtime: &'a mut LoweringRuntime,
    pub native: &'a mut u32,
}

/// `scoped` on the borrowing context, as `host.rs`'s `with_inlining`: enter
/// pushes, exit pops. The body grows the stack on a tuple of the context's
/// mutable borrows, then catches on the context's runtime.
pub fn host_scopes(
    checked: &u32,
    scope: &mut Scope,
    runtime: &mut LoweringRuntime,
    native: &mut u32,
    reject: bool,
) -> Result<usize, LowerDiagnostic> {
    let mut cx = HostCx { checked, scope, runtime, native };
    scoped(
        &mut cx,
        |cx| {
            cx.scope.items.push(*cx.checked);
            cx.scope.items.len()
        },
        |cx, entered| {
            cx.scope.exits += entered as u32;
            cx.scope.items.pop();
        },
        |cx| {
            let grown = grow_stack_if_needed(
                &mut (&mut *cx.scope, &mut *cx.native),
                64 * 1024,
                4 * 1024 * 1024,
                |(scope, native)| {
                    **native += 1;
                    scope.items.push(**native);
                    scope.items.len()
                },
            );
            cx.runtime.catch_lowering(move |runtime| {
                if reject {
                    Err(runtime.raise_unrepresentable())
                } else {
                    Ok(grown)
                }
            })
        },
    )
}

/// Not modeled: a scope on a borrow of a borrow, whose innermost region gives
/// back through a staged backward function. Strict extraction rejects it.
pub fn reject_nested_borrow_state(x: &mut u32) -> u32 {
    let mut inner: &mut u32 = x;
    let mut outer: &mut &mut u32 = &mut inner;
    scoped(
        &mut outer,
        |o| {
            ***o += 1;
            7
        },
        |o, t| ***o += t,
        |o| {
            ***o *= 2;
            ***o
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lowering_runtime::{install_chelis_panic_hook, observe, with_native_lowering_runtime};
    use std::cell::RefCell;
    use std::sync::Once;

    thread_local! {
        /// `(u, o)` at every raise on this thread, as the process hook sees them.
        static RAISES: RefCell<Vec<(bool, bool)>> = const { RefCell::new(Vec::new()) };
    }

    /// Record the flags of every raise in front of the Chelis hook.
    fn record_raises() {
        static INSTALL: Once = Once::new();
        INSTALL.call_once(|| {
            install_chelis_panic_hook();
            let chelis = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                RAISES.with(|raises| raises.borrow_mut().push(observe::flags()));
                chelis(info);
            }));
        });
        RAISES.with(|raises| raises.borrow_mut().clear());
    }

    fn raises() -> Vec<(bool, bool)> {
        RAISES.with(|raises| raises.borrow_mut().drain(..).collect())
    }

    fn diagnostic(code: u32) -> LowerDiagnostic {
        LowerDiagnostic::new(code)
    }

    #[test]
    fn speculation_scopes_set_and_clear_u() {
        with_native_lowering_runtime(|runtime| {
            assert_eq!(speculation_flags(runtime), (false, true, false));
            // An enclosing attempt is cleared, not restored, by an inner scope.
            let nested = runtime.with_suppress_unrepresentable_panic(speculation_flags);
            assert_eq!(nested, (true, true, false));
            assert_eq!(observe::flags(), (false, false));
        });
    }

    #[test]
    fn raises_record_their_flags() {
        record_raises();
        with_native_lowering_runtime(|runtime| {
            assert_eq!(speculative_rejection(runtime, 9), Err(diagnostic(1)));
            assert_eq!(raises(), [(true, true)]);
            assert_eq!(rejection_after_inner_catch(runtime, 9), Err(diagnostic(9)));
            assert_eq!(raises(), [(false, false)]);
            assert_eq!(rejection_after_inner_catch(runtime, 17), Ok(17));
            assert_eq!(raises(), []);
            assert_eq!(fatal_reraise(runtime, 4), Err(diagnostic(4).fatal()));
            assert_eq!(raises(), [(true, true), (true, true)]);
            assert_eq!(
                uncaught_rejection(runtime, 5),
                LoweringFailure::Diagnostic(diagnostic(5))
            );
            assert_eq!(raises(), [(false, false)]);
            assert_eq!(observe::flags(), (false, false));
        });
    }

    #[test]
    fn catch_threads_state() {
        record_raises();
        with_native_lowering_runtime(|runtime| {
            let mut count = 3;
            assert_eq!(caught_with_state(runtime, &mut count, false), Ok(4));
            assert_eq!(caught_with_state(runtime, &mut count, true), Err(diagnostic(1)));
            assert_eq!(count, 5);
            assert_eq!(raises(), [(false, true)]);
        });
    }

    #[test]
    fn boundary_resume_skips_the_hook() {
        record_raises();
        with_native_lowering_runtime(|runtime| {
            let failure = uncaught_rejection(runtime, 6);
            assert_eq!(raises(), [(false, false)]);
            let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                failure.resume_at_boundary();
            }))
            .unwrap_err();
            assert_eq!(*payload.downcast::<LowerDiagnostic>().unwrap(), diagnostic(6));
            assert_eq!(raises(), []);
        });
    }

    #[test]
    fn summary_probes_saturate() {
        with_native_lowering_runtime(|runtime| {
            assert_eq!(summary_probes(runtime, 3), (3, 0));
            observe::set_counters(u64::MAX - 1, 0);
            assert_eq!(summary_probes(runtime, 3), (u64::MAX, 0));
        });
    }

    #[test]
    fn preflight_frames_wrap() {
        with_native_lowering_runtime(|runtime| {
            assert_eq!(preflight_frames(runtime), (0, 1, 2));
            assert_eq!(preflight_frames(runtime), (3, 4, 5));
            observe::set_counters(0, u64::MAX - 1);
            assert_eq!(preflight_frames(runtime), (u64::MAX - 1, u64::MAX, 0));
        });
    }

    #[test]
    fn host_inference_vars_are_consecutive() {
        // The only allocations of this process's supply.
        with_native_lowering_runtime(|runtime| {
            assert_eq!(host_inference_vars(runtime), (0, 1));
            assert_eq!(host_inference_vars(runtime), (2, 3));
        });
    }

    #[test]
    fn profile_marks_and_declarations() {
        with_native_lowering_runtime(|runtime| {
            // Disabled: no mark, no clock, and `last_dag_size` is untouched.
            unsafe { std::env::remove_var("CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL") };
            assert_eq!(profile_declarations(runtime, &[4, 9]), (0, 0, 0));
            assert_eq!(profile_unissued_mark(runtime, 1, 8), 5);
            // Enabled: marks are this thread's clock count.
            unsafe { std::env::set_var("CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL", "1") };
            assert_eq!(profile_declarations(runtime, &[4, 9]), (1, 3, 9));
            assert_eq!(profile_declarations(runtime, &[]), (4, 0, 0));
            unsafe { std::env::set_var("CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL", "0") };
            assert_eq!(profile_declarations(runtime, &[7]), (0, 0, 0));
            unsafe { std::env::remove_var("CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL") };
            // Issued marks name clocks; others do not.
            assert_eq!(profile_unissued_mark(runtime, 4, 8), 8);
            assert_eq!(profile_unissued_mark(runtime, 5, 8), 5);
            assert_eq!(profile_unissued_mark(runtime, 0, 8), 5);
        });
    }

    #[test]
    fn host_scopes_thread_the_context() {
        record_raises();
        with_native_lowering_runtime(|runtime| {
            let mut scope = Scope { items: vec![1], exits: 0 };
            let mut native = 4;
            assert_eq!(host_scopes(&7, &mut scope, runtime, &mut native, false), Ok(3));
            assert_eq!((scope.items.clone(), scope.exits, native), (vec![1, 7], 2, 5));
            assert_eq!(raises(), []);
            assert_eq!(host_scopes(&9, &mut scope, runtime, &mut native, true), Err(diagnostic(1)));
            assert_eq!((scope.items.clone(), scope.exits, native), (vec![1, 7, 9], 5, 6));
            assert_eq!(raises(), [(false, true)]);
            assert_eq!(observe::flags(), (false, false));
        });
    }

    #[test]
    fn nested_borrow_state_scope() {
        let mut x = 3;
        assert_eq!(reject_nested_borrow_state(&mut x), 8);
        assert_eq!(x, 15);
    }

    #[test]
    fn scoped_runs_enter_body_exit() {
        let mut scope = Scope { items: vec![1], exits: 0 };
        assert_eq!(scoped_push(&mut scope, 5), 3);
        assert_eq!((scope.items.clone(), scope.exits), (vec![1, 5], 2));
        assert_eq!(grown_push(&mut scope, 8), Ok(4));
        assert_eq!(scope.items, [1, 5, 8]);
    }
}
