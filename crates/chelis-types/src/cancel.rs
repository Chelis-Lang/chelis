//! Cooperative cancellation for long-running evaluation (chelis#914).
//!
//! An evaluation started from the Python bindings runs inside
//! `py.allow_threads`, so a SIGINT sets Python's handler flag but
//! `KeyboardInterrupt` cannot be raised until control returns to the
//! interpreter loop — i.e. until the eval has already finished. To make a
//! slow eval abandonable, the eval lanes have to be able to notice a
//! cancellation request and unwind on their own.
//!
//! The token is installed per-thread with a restore-on-drop guard, the same
//! shape as [`crate::install_linked_program_guard`] in `opacity.rs`: the
//! evaluation entry points do not need a new parameter, and nested or
//! re-entrant evals cannot leak a token into an outer scope.
//!
//! chelis#930 extended the same token to the **front end**. Parse, desugar,
//! type-check and lowering have no node visit to poll at, so they poll at
//! phase boundaries and at top-level-declaration boundaries inside the passes
//! that scale with declaration count. Nothing new was introduced for that: the
//! token, the guard, and the error kind are the ones below.
//!
//! Cost discipline: consumers are expected to call [`current_cancel_token`] **once**
//! at context construction and cache the `Arc` for the duration of the
//! evaluation, so the hot path is one relaxed atomic load behind an `Option`
//! check rather than a TLS lookup per node visit. [`CancelToken::is_cancelled`]
//! uses `Ordering::Relaxed` deliberately: the flag is a single-writer,
//! many-reader hint whose only requirement is eventual visibility, and losing
//! a race by one node visit is immaterial when the alternative is minutes.
//! [`cancellation_requested`] is the uncached convenience form, for the
//! handful-of-calls-per-compile phase boundaries only.

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::errors::{CheckError, CheckErrorKind};

/// Stable sentinel for the error a cancelled evaluation returns.
///
/// The eval lanes propagate failures as `Result<_, String>`, so callers
/// distinguish "the user asked us to stop" from "the program is wrong" by
/// testing for this marker via [`is_cancellation`].
///
/// It is deliberately namespaced rather than plain prose. Detection is a
/// substring test (stage wrapping prepends context), so a bare phrase like
/// "evaluation cancelled" could in principle appear inside a genuine
/// evaluation error and be misreported to the user as a timeout. Nothing a
/// user program can produce contains this token.
///
/// Internal by construction: the CLI translates it to the `--timeout`
/// message, and the bindings never surface it (a cancelled call raises the
/// pending Python exception instead). Keep it stable regardless — both
/// callers match on it.
///
/// This string is the *lane-internal* carrier only. It is interpreted exactly
/// once, at the eval-stage boundary in `compiler-api`, which re-expresses
/// cancellation structurally as `Diagnostic::kind ==
/// chelis_compiler_api::EVAL_CANCELLED_KIND`. Embedders should test that kind
/// (or `CompilerError::is_cancellation`) rather than matching on text.
pub const EVAL_CANCELLED_MSG: &str = "chelis::eval::cancelled";

/// Shared cancellation flag handed to an evaluation.
///
/// Cheap to clone (one `Arc` bump). The requester keeps a handle and calls
/// [`CancelToken::cancel`]; the evaluating thread polls
/// [`CancelToken::is_cancelled`].
#[derive(Debug, Clone, Default)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
}

impl CancelToken {
    /// A fresh, un-cancelled token.
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation. Idempotent, and callable from any thread.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Relaxed);
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }
}

thread_local! {
    /// The token installed for the current thread, if any. `None` — the
    /// default — means compiler work on this thread is not cancellable, which
    /// is the behaviour every existing caller gets for free.
    static CANCEL_TOKEN: RefCell<Option<CancelToken>> = const { RefCell::new(None) };
}

/// Install `token` for the current thread for the duration of the returned
/// guard, restoring the previous value on drop.
///
/// Call this on the thread that will run the evaluation, before entering it.
/// The guard's restore-on-drop is what makes nesting safe: an inner eval that
/// installs its own token cannot strand it in the outer scope.
#[must_use = "cancellation is uninstalled when the guard drops"]
pub fn install_cancel_token(token: CancelToken) -> CancelTokenGuard {
    let previous = CANCEL_TOKEN.with(|cell| cell.replace(Some(token)));
    CancelTokenGuard { previous }
}

/// The token installed for the current thread, if any.
///
/// Intended to be called once per evaluation context and cached; see the
/// module docs on cost.
pub fn current_cancel_token() -> Option<CancelToken> {
    CANCEL_TOKEN.with(|cell| cell.borrow().clone())
}

/// Whether the current thread has an installed token that has been cancelled.
///
/// One TLS borrow plus one relaxed load, with no `Arc` clone. That budget is
/// right for the *coarse* polling sites added by chelis#930 — front-end phase
/// boundaries, where the check runs a handful of times per compile — and wrong
/// for anything iterating. A loop over declarations should hoist
/// [`current_cancel_token`] above the loop and poll
/// [`CancelToken::is_cancelled`] on the cached handle instead, the same
/// discipline `EvalContext` follows for node visits.
pub fn cancellation_requested() -> bool {
    CANCEL_TOKEN.with(|cell| {
        cell.borrow()
            .as_ref()
            .is_some_and(CancelToken::is_cancelled)
    })
}

/// The hard failure a front-end pass reports when it abandons its walk
/// because cancellation was requested (chelis#930).
///
/// A pass that stopped early has proved nothing about the program, so it must
/// fail rather than return a partial `Ok` — the same covered-or-rejected rule
/// the type checker's stack-exhaustion bail follows. [`CheckErrorKind::Other`]
/// is deliberate: cancellation is not a property of the source, so it does not
/// earn a diagnostic classification of its own, and callers holding structured
/// errors should test `CompilerError::is_cancellation` (which reads
/// `Diagnostic::kind`) rather than this variant.
///
/// The message is the [`EVAL_CANCELLED_MSG`] sentinel so the one caller that
/// only ever sees flattened error text — the CLI's `--timeout` boundary, where
/// the typed error is already gone — still classifies it correctly.
pub fn cancellation_check_error() -> CheckError {
    CheckError::new(
        CheckErrorKind::Other,
        EVAL_CANCELLED_MSG.to_string(),
        vec![],
    )
}

/// Restores the previously installed token when dropped.
pub struct CancelTokenGuard {
    previous: Option<CancelToken>,
}

impl Drop for CancelTokenGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        CANCEL_TOKEN.with(|cell| *cell.borrow_mut() = previous);
    }
}

/// Whether `message` is the cancellation sentinel rather than a genuine
/// evaluation error.
///
/// Uses `contains` rather than `starts_with` because the eval lanes wrap
/// lane-level failures with stage context before they reach the caller.
pub fn is_cancellation(message: &str) -> bool {
    message.contains(EVAL_CANCELLED_MSG)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_token_installed_by_default() {
        assert!(current_cancel_token().is_none());
    }

    #[test]
    fn install_exposes_token_and_guard_restores() {
        let token = CancelToken::new();
        {
            let _guard = install_cancel_token(token.clone());
            let seen = current_cancel_token().expect("token should be visible");
            assert!(!seen.is_cancelled());
            token.cancel();
            assert!(seen.is_cancelled(), "clones share one flag");
        }
        assert!(
            current_cancel_token().is_none(),
            "guard must restore the previous (absent) token"
        );
    }

    /// chelis#930's polling form. The three states have to be distinguished:
    /// no token at all, an installed-but-untripped token, and a tripped one.
    /// Collapsing the middle case into `true` would cancel every compile on a
    /// thread that merely *armed* a timeout.
    #[test]
    fn cancellation_requested_distinguishes_absent_armed_and_tripped() {
        assert!(
            !cancellation_requested(),
            "no token installed must never read as cancelled"
        );
        let token = CancelToken::new();
        let _guard = install_cancel_token(token.clone());
        assert!(
            !cancellation_requested(),
            "an armed but untripped token must not read as cancelled"
        );
        token.cancel();
        assert!(cancellation_requested());
    }

    #[test]
    fn cancellation_requested_is_restored_by_the_guard() {
        {
            let token = CancelToken::new();
            token.cancel();
            let _guard = install_cancel_token(token);
            assert!(cancellation_requested());
        }
        assert!(
            !cancellation_requested(),
            "a cancelled token must not leak past its guard into the next compile"
        );
    }

    /// The front-end passes report cancellation through the check-error
    /// channel so an abandoned walk is a hard failure, never a partial `Ok`.
    /// The CLI's `--timeout` boundary only ever sees flattened text, so the
    /// sentinel has to survive that flattening.
    #[test]
    fn cancellation_check_error_is_recognised_as_cancellation() {
        let error = cancellation_check_error();
        assert!(is_cancellation(&error.message));
        assert!(is_cancellation(&format!("check: {}", error.message)));
    }

    #[test]
    fn nested_installs_restore_the_outer_token() {
        let outer = CancelToken::new();
        let inner = CancelToken::new();
        let _outer_guard = install_cancel_token(outer.clone());
        {
            let _inner_guard = install_cancel_token(inner.clone());
            inner.cancel();
            assert!(current_cancel_token().expect("inner").is_cancelled());
        }
        // The outer token is back, and was never cancelled by the inner one.
        assert!(!current_cancel_token().expect("outer").is_cancelled());
        outer.cancel();
        assert!(current_cancel_token().expect("outer").is_cancelled());
    }

    #[test]
    fn tokens_are_thread_local_but_flags_are_shared() {
        let token = CancelToken::new();
        let _guard = install_cancel_token(token.clone());
        let observed = std::thread::spawn(current_cancel_token)
            .join()
            .expect("join");
        assert!(
            observed.is_none(),
            "a fresh thread inherits no token; it must be installed explicitly"
        );
        // ...but an explicitly shared clone does see the flag across threads.
        let shared = token.clone();
        token.cancel();
        assert!(
            std::thread::spawn(move || shared.is_cancelled())
                .join()
                .expect("join")
        );
    }

    #[test]
    fn sentinel_is_recognised_through_stage_wrapping() {
        assert!(is_cancellation(EVAL_CANCELLED_MSG));
        assert!(is_cancellation(&format!("eval: {EVAL_CANCELLED_MSG}")));
        assert!(!is_cancellation("unknown runtime name foo"));
    }

    /// The sentinel must not collide with prose a real program could emit.
    /// Detection is a substring test, so a bare phrase like "evaluation
    /// cancelled" in a genuine error would be misreported to the user as a
    /// timeout.
    #[test]
    fn sentinel_does_not_collide_with_ordinary_error_prose() {
        for message in [
            "evaluation cancelled",
            "the evaluation was cancelled by the user",
            "cancelled",
            "error: order cancelled before settlement",
            "eval error: cancellation policy violated",
        ] {
            assert!(
                !is_cancellation(message),
                "{message:?} must not be mistaken for the cancellation sentinel"
            );
        }
    }
}
