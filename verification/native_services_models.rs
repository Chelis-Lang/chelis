//! The native checking services of `crates/chelis-types/src/native.rs` against their Lean
//! models (`Aeneas/Std/ChelisNative.lean`).
//!
//! Compiled as crate `chelis_types`, so every primitive has its production path and Aeneas
//! replaces it with its model. `native` repeats `NativeServices` and its primitives verbatim
//! (`check_models.py` compares the copy with production). The native test links the real
//! `stacker` crate of the workspace, as production does. Charon compiles the library without
//! it, through the `stacker` stand-in of `native`: only the hard-opaque primitives call it, so
//! the stand-in is never extracted. Differences: `sha256_concat` is omitted (its model has its
//! own oracle, `sha256_model`), and `cancel` and `infer` keep only the items `native` uses.
//! The native shell is hard-opaque through the `chelis_types::native` patterns of
//! `verification/native_shell.json`, as in the production extraction. The scenario functions
//! call the primitives the way `infer/stack.rs`, `session.rs` and the checker entries do.

pub mod cancel {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[derive(Debug, Clone, Default)]
    pub struct CancelToken {
        flag: Arc<AtomicBool>,
    }

    impl CancelToken {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn cancel(&self) {
            self.flag.store(true, Ordering::Relaxed);
        }

        pub fn is_cancelled(&self) -> bool {
            self.flag.load(Ordering::Relaxed)
        }
    }
}

pub mod infer {
    pub(crate) const STACK_RED_ZONE_BYTES: usize = 128 * 1024;
    pub(crate) const FALLBACK_MAX_DEPTH: usize = 20;
    pub(crate) const GROW_SEGMENT_BYTES: usize = 512 * 1024 * 1024;
}

pub mod native {
    use std::time::Instant;

    use crate::cancel::CancelToken;
    use crate::infer::{FALLBACK_MAX_DEPTH, GROW_SEGMENT_BYTES, STACK_RED_ZONE_BYTES};

    /// Native effects available to one checking invocation. Constructed by the
    /// embedding shell and threaded by `&mut` through every public checker entry.
    pub struct NativeServices {
        /// Shared cancellation flag of this invocation, if the shell armed one.
        cancel: Option<CancelToken>,
        /// Live guarded-walker recursion depth. Consulted only when the platform
        /// cannot report remaining stack.
        depth: usize,
        /// Stack segment size used by [`Self::with_grown_stack`].
        grow_segment_bytes: usize,
        /// Profiling phase marks, `Some` only when profiling is enabled: one per
        /// [`Self::profile_start`], each holding the end of its latest phase.
        /// Marks are never reused, so nested or re-entered profiled scopes keep
        /// independent clocks.
        profile_marks: Option<Vec<Instant>>,
    }

    impl NativeServices {
        /// Primitive. Services for one invocation, polling `cancel` when present.
        /// Profiling is off and the production stack segment is used. Shell-only:
        /// extracted code receives services and never constructs them.
        ///
        /// Model: an arbitrary native state (the correspondence theorems quantify
        /// over it).
        pub fn new(cancel: Option<CancelToken>) -> Self {
            Self {
                cancel,
                depth: 0,
                grow_segment_bytes: GROW_SEGMENT_BYTES,
                profile_marks: None,
            }
        }

        /// Primitive. Enable or disable per-phase profiling on stderr. The shell
        /// decides (historically from `CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL=1`).
        /// Shell-only.
        ///
        /// Model: identity.
        pub fn with_profiling(mut self, enabled: bool) -> Self {
            self.profile_marks = enabled.then(Vec::new);
            self
        }

        /// Primitive. Test-only stack segment size for [`Self::with_grown_stack`].
        /// The recursion-depth-guard corpus shrinks the segment to a few MiB so
        /// the per-site stack guards remain exercised; `0` keeps the production
        /// segment. Not for production use: a too-small value makes a
        /// legitimately deep program bail where it would otherwise check.
        /// Shell-only.
        ///
        /// Model: identity.
        #[doc(hidden)]
        pub fn with_grow_segment_bytes_for_test(mut self, bytes: usize) -> Self {
            self.grow_segment_bytes = if bytes > 0 { bytes } else { GROW_SEGMENT_BYTES };
            self
        }

        /// Primitive. Relaxed load of the shared cancellation flag; `false`
        /// without a token.
        ///
        /// Model: next element of a monotone boolean observation schedule (once
        /// observed `true`, every later observation is `true`).
        pub fn cancellation_observed(&mut self) -> bool {
            self.cancel.as_ref().is_some_and(CancelToken::is_cancelled)
        }

        /// Primitive. Whether the native stack is too close to exhaustion to
        /// recurse one more guarded frame: `remaining_stack() <
        /// STACK_RED_ZONE_BYTES`, or, when the platform cannot report remaining
        /// stack, live depth `> FALLBACK_MAX_DEPTH`.
        ///
        /// Model: next element of an arbitrary boolean observation schedule.
        pub(crate) fn stack_exhausted(&mut self) -> bool {
            match stacker::remaining_stack() {
                Some(remaining) => remaining < STACK_RED_ZONE_BYTES,
                None => self.depth > FALLBACK_MAX_DEPTH,
            }
        }

        /// Primitive. Record entry into one guarded recursive frame.
        ///
        /// Model: `depth := depth + 1`.
        pub(crate) fn enter_frame(&mut self) {
            self.depth += 1;
        }

        /// Primitive. Record exit from one guarded recursive frame.
        ///
        /// Model: `depth := depth - 1` (saturating).
        pub(crate) fn exit_frame(&mut self) {
            self.depth = self.depth.saturating_sub(1);
        }

        /// Primitive. Forget every guarded frame. Called when the OUTERMOST check
        /// unit is entered: a check that unwound out of guarded walkers never ran
        /// their frame exits, and its leftover depth must not make the next check
        /// on these services trip the fallback cap early.
        ///
        /// Model: `depth := 0`.
        pub(crate) fn reset_frames(&mut self) {
            self.depth = 0;
        }

        /// Primitive. Run `f` on a freshly allocated native stack segment
        /// (`stacker::grow`) of the production size or the test override.
        ///
        /// Always allocates rather than growing on demand: a check starts on a
        /// near-empty stack and only nears exhaustion deep inside the recursive
        /// walkers, far from any grow site, so the headroom must be in place
        /// before the descent begins. Every public checker entry runs its whole
        /// pipeline (inference, validation, annotation, deep clones and the final
        /// drop) inside one call; the per-site stack guards remain the safety net
        /// for input deeper than the segment. Nesting is at worst a redundant
        /// second segment.
        ///
        /// Model: `f self`.
        pub fn with_grown_stack<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
            stacker::grow(self.grow_segment_bytes, || f(self))
        }

        /// Primitive. Start a profiled scope and return its mark. When profiling
        /// is enabled this reads the clock; otherwise it returns `0` and reads
        /// nothing.
        ///
        /// Model: an arbitrary `usize`; no effect on checking.
        pub(crate) fn profile_start(&mut self) -> usize {
            match &mut self.profile_marks {
                Some(marks) => {
                    marks.push(Instant::now());
                    marks.len() - 1
                }
                None => 0,
            }
        }

        /// Primitive. When profiling is enabled, report on stderr the time since
        /// `mark`'s latest phase as `"{scope}: {seconds:>8.4}s {label}"` and start
        /// its next phase.
        ///
        /// Model: no-op.
        pub(crate) fn profile_phase(&mut self, mark: usize, scope: &str, label: &str) {
            if let Some(start) = self
                .profile_marks
                .as_mut()
                .and_then(|marks| marks.get_mut(mark))
            {
                eprintln!("{scope}: {:>8.4}s {label}", start.elapsed().as_secs_f64());
                *start = Instant::now();
            }
        }
    }

    /// Stand-in for the `stacker` crate in the library that Charon compiles. Only the
    /// hard-opaque primitives `stack_exhausted` and `with_grown_stack` call it.
    #[cfg(not(test))]
    mod stacker {
        pub fn remaining_stack() -> Option<usize> {
            None
        }

        pub fn grow<R>(_stack_size: usize, callback: impl FnOnce() -> R) -> R {
            callback()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::NativeServices;
        use crate::cancel::CancelToken;
        use crate::scenarios::{
            ascend, descend, enter_outermost_scope, frame_exhausted, grown_entry, polls, profiled,
        };

        /// A segment smaller than `STACK_RED_ZONE_BYTES`: every probe on it is exhausted.
        const SMALL_SEGMENT: usize = 64 * 1024;

        #[test]
        fn frames_on_ample_stack() {
            let mut native = NativeServices::new(None);
            assert!(!frame_exhausted(&mut native));
            assert_eq!(native.depth, 1);
            assert_eq!(descend(&mut native, 4), 0);
            assert_eq!(native.depth, 5);
            ascend(&mut native, 7);
            assert_eq!(native.depth, 0);
            assert_eq!(descend(&mut native, 3), 0);
            enter_outermost_scope(&mut native);
            assert_eq!(native.depth, 0);
        }

        #[test]
        fn frames_on_a_small_segment() {
            stacker::grow(SMALL_SEGMENT, || {
                let mut native = NativeServices::new(None);
                assert!(frame_exhausted(&mut native));
                assert_eq!(native.depth, 0);
                assert_eq!(descend(&mut native, 3), 3);
                assert_eq!(native.depth, 0);
            });
        }

        #[test]
        fn grown_entry_applies_the_closure() {
            let token = CancelToken::new();
            token.cancel();
            let mut native =
                NativeServices::new(Some(token)).with_grow_segment_bytes_for_test(SMALL_SEGMENT);
            assert_eq!(grown_entry(&mut native, 2), (2, true));
            assert_eq!(native.depth, 0);
            let mut native = NativeServices::new(None);
            assert_eq!(grown_entry(&mut native, 2), (0, false));
            assert_eq!(native.depth, 2);
        }

        #[test]
        fn cancellation_polls() {
            assert_eq!(polls(&mut NativeServices::new(None)), (false, false, false));
            let token = CancelToken::new();
            let mut native = NativeServices::new(Some(token.clone()));
            assert_eq!(polls(&mut native), (false, false, false));
            token.cancel();
            assert_eq!(polls(&mut native), (true, true, true));
        }

        #[test]
        fn profiling_marks() {
            assert_eq!(profiled(&mut NativeServices::new(None)), (0, 0));
            let mut native = NativeServices::new(None).with_profiling(true);
            assert_eq!(profiled(&mut native), (0, 1));
            assert_eq!(profiled(&mut native), (2, 3));
        }
    }
}

pub mod scenarios {
    use crate::native::NativeServices;

    /// `StackBudget::frame_exhausted` (`infer/stack.rs:185-197`) without the bail record:
    /// enter one guarded frame and probe; an exhausted probe exits the frame again.
    pub fn frame_exhausted(native: &mut NativeServices) -> bool {
        native.enter_frame();
        if native.stack_exhausted() {
            native.exit_frame();
            return true;
        }
        false
    }

    /// `frames` guarded walkers entered one inside the other and never exited (a walk that
    /// unwinds out of them). Returns the number of exhausted probes.
    pub fn descend(native: &mut NativeServices, frames: u32) -> u32 {
        let mut bails = 0;
        let mut index = 0;
        while index < frames {
            if frame_exhausted(native) {
                bails += 1;
            }
            index += 1;
        }
        bails
    }

    /// `frames` walker exits (`StackBudget::exit_frame`).
    pub fn ascend(native: &mut NativeServices, frames: u32) {
        let mut index = 0;
        while index < frames {
            native.exit_frame();
            index += 1;
        }
    }

    /// `StackBailRecord::enter_scope` at the outermost scope (`infer/stack.rs:118-125`).
    pub fn enter_outermost_scope(native: &mut NativeServices) {
        native.reset_frames();
    }

    /// A public checker entry: its whole pipeline runs on a grown stack segment.
    pub fn grown_entry(native: &mut NativeServices, frames: u32) -> (u32, bool) {
        native.with_grown_stack(|native| {
            let bails = descend(native, frames);
            (bails, native.cancellation_observed())
        })
    }

    /// Three successive polling sites.
    pub fn polls(native: &mut NativeServices) -> (bool, bool, bool) {
        let first = native.cancellation_observed();
        let second = native.cancellation_observed();
        let third = native.cancellation_observed();
        (first, second, third)
    }

    /// Two profiled scopes, the first with one phase.
    pub fn profiled(native: &mut NativeServices) -> (usize, usize) {
        let first = native.profile_start();
        native.profile_phase(first, "check", "phase");
        let second = native.profile_start();
        (first, second)
    }
}
