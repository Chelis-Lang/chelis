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
//! Cost discipline: consumers are expected to call [`current_cancel_token`] **once**
//! at context construction and cache the `Arc` for the duration of the
//! evaluation, so the hot path is one relaxed atomic load behind an `Option`
//! check rather than a TLS lookup per node visit. [`CancelToken::is_cancelled`]
//! uses `Ordering::Relaxed` deliberately: the flag is a single-writer,
//! many-reader hint whose only requirement is eventual visibility, and losing
//! a race by one node visit is immaterial when the alternative is minutes.

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Stable sentinel prefix for the error a cancelled evaluation returns.
///
/// The eval lanes propagate failures as `Result<_, String>`, so callers
/// distinguish "the user asked us to stop" from "the program is wrong" by
/// testing for this prefix. Keep it stable: the Python bindings and the CLI
/// `--timeout` path both match on it to pick their user-facing message.
pub const EVAL_CANCELLED_MSG: &str = "evaluation cancelled";

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
    /// default — means this thread's evaluations are not cancellable, which
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
}
