//! Native effects available to one checking invocation.
//!
//! The semantic checker is a pure function of its inputs except for a short,
//! explicit list of native observations: a cancellation flag another thread
//! may raise, the remaining native stack, a stack-growth request, and an
//! optional profiling clock. Each is a *primitive* below: a method whose body
//! is one platform call on this explicitly owned state. Semantic code receives
//! `&mut NativeServices` from the embedding shell and never reaches thread
//! locals, globals, the environment, the clock, or stderr directly.
//! [`sha256_concat`] is the one pure primitive: the checker's only use of the
//! `sha2` crate.
//!
//! The primitives are the trusted boundary of the functional model. Each is
//! documented with the Lean model it is intended to have. Their signatures
//! mention only `NativeServices` itself, the separately modeled
//! [`CancelToken`] (shell-only constructor argument), std types, and
//! closures, and `NativeServices` has no trait impls, so the Lean model can be
//! written before any crate is extracted.

use std::time::Instant;

use sha2::{Digest, Sha256};

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

/// Primitive. SHA-256 digest of the in-order concatenation of `parts`.
///
/// Model: `sha256 (parts.flatten)` (FIPS 180-4).
pub(crate) fn sha256_concat(parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part);
    }
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three states are distinct: no token, an armed-but-untripped
    /// token, and a tripped one. Collapsing the middle case into `true`
    /// would cancel every check that merely armed a timeout.
    #[test]
    fn cancellation_observed_distinguishes_absent_armed_and_tripped() {
        assert!(!NativeServices::new(None).cancellation_observed());
        let token = CancelToken::new();
        let mut native = NativeServices::new(Some(token.clone()));
        assert!(!native.cancellation_observed());
        token.cancel();
        assert!(native.cancellation_observed());
    }

    /// The flag is observed on every poll, never cached at construction: a
    /// token cancelled by another thread mid-invocation is seen at the next
    /// polling site.
    #[test]
    fn cancellation_from_another_thread_is_observed_at_the_next_poll() {
        let token = CancelToken::new();
        let mut native = NativeServices::new(Some(token.clone()));
        assert!(!native.cancellation_observed());
        std::thread::spawn(move || token.cancel())
            .join()
            .expect("join");
        assert!(native.cancellation_observed());
    }

    /// Services are per invocation: a cancelled invocation cannot leak its
    /// token into an unrelated one.
    #[test]
    fn invocations_do_not_share_tokens() {
        let token = CancelToken::new();
        token.cancel();
        let mut cancelled = NativeServices::new(Some(token));
        let mut fresh = NativeServices::new(None);
        assert!(cancelled.cancellation_observed());
        assert!(!fresh.cancellation_observed());
    }

    #[test]
    fn grown_stack_returns_the_closure_value_and_threads_state() {
        let token = CancelToken::new();
        let mut native = NativeServices::new(Some(token.clone()));
        token.cancel();
        let observed = native.with_grown_stack(NativeServices::cancellation_observed);
        assert!(observed);
    }

    /// A check that unwinds out of guarded walkers never runs their frame
    /// exits. Services reused after the caught unwind must start the next
    /// check like fresh services: on a platform that cannot report remaining
    /// stack, leftover depth would trip the `depth > FALLBACK_MAX_DEPTH`
    /// fallback before the new check has recursed at all.
    #[test]
    fn check_after_a_caught_unwind_starts_like_fresh_services() {
        let mut reused = NativeServices::new(None);
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::session::run_test_session(&mut reused, |sink| {
                sink.enter_stack_scope();
                for _ in 0..=FALLBACK_MAX_DEPTH {
                    assert!(!sink.frame_exhausted("unwinding_walker", || None));
                }
                panic!("guarded walker panicked mid-descent");
            })
        }));
        assert!(unwound.is_err());
        assert!(
            reused.depth > FALLBACK_MAX_DEPTH,
            "the unwind must leave its frames unexited for this test to mean anything"
        );

        let depth_at_check_entry = |native: &mut NativeServices| {
            crate::session::run_test_session(native, |sink| {
                sink.enter_stack_scope();
                let depth = sink.stack().native().depth;
                sink.exit_stack_scope();
                depth
            })
            .0
        };
        assert_eq!(
            depth_at_check_entry(&mut reused),
            depth_at_check_entry(&mut NativeServices::new(None))
        );
    }
}
