//! Cooperative cancellation for long-running evaluation (chelis#914).
//!
//! An evaluation started from the Python bindings runs inside
//! `py.allow_threads`, so a SIGINT sets Python's handler flag but
//! `KeyboardInterrupt` cannot be raised until control returns to the
//! interpreter loop — i.e. until the eval has already finished. To make a
//! slow eval abandonable, the eval lanes have to be able to notice a
//! cancellation request and unwind on their own.
//!
//! The token is handed to one invocation explicitly, inside
//! [`crate::NativeServices`]; there is no per-thread installation. Nested or
//! concurrent invocations each carry their own services, so a token cannot
//! leak into an unrelated scope. Unlike immutable [`crate::ProgramOrigin`],
//! cancellation can change while a check is running.
//!
//! chelis#930 extended the same token to the **front end**. Parse, desugar,
//! type-check and lowering have no node visit to poll at, so they poll at
//! phase boundaries and at top-level-declaration boundaries inside the passes
//! that scale with declaration count. Nothing new was introduced for that: the
//! token and the error kind are the ones below.
//!
//! Cost discipline: every polling site calls
//! [`crate::NativeServices::cancellation_observed`], which is one relaxed
//! atomic load behind an `Option` check. [`CancelToken::is_cancelled`] uses
//! `Ordering::Relaxed` deliberately: the flag is a single-writer, many-reader
//! hint whose only requirement is eventual visibility, and losing a race by one
//! node visit is immaterial when the alternative is minutes.

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

    /// Clones share one flag, including across threads: the requester's
    /// handle and the evaluating invocation's handle are the same `Arc`.
    #[test]
    fn clones_share_one_flag_across_threads() {
        let token = CancelToken::new();
        let seen = token.clone();
        assert!(!seen.is_cancelled());
        let shared = token.clone();
        std::thread::spawn(move || shared.cancel())
            .join()
            .expect("join");
        assert!(seen.is_cancelled(), "clones share one flag");
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
