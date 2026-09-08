//! Cooperative eval cancellation (chelis#914).
//!
//! ## Background
//!
//! A long-running evaluation started from the Python bindings could not be
//! interrupted: `run_json` parked the Python main thread inside
//! `py.allow_threads` for the whole job, so Python's SIGINT flag went
//! unobserved until the eval finished on its own. The fix is a cooperative
//! cancellation token that the eval lanes poll, installed per-thread with a
//! restore-on-drop guard (`chelis_types::install_cancel_token`).
//!
//! `eval_compiled` runs TWO lanes and both must honour the token:
//!
//! * the host-runtime lane — `EvalContext::eval_expr`, one check per node
//!   visit, which is what covers each element of a fold/map;
//! * the tensor-DAG lane — `chelis_ir::eval::eval_tensor_internal`, one check
//!   per DAG node.
//!
//! ## Coverage
//!
//! Positive (cancellation is observed):
//! * `cancelled_before_eval_returns_sentinel` — host lane, pre-set token.
//! * `cancelled_tensor_program_returns_sentinel` — tensor lane, pre-set token.
//! * `cancellation_interrupts_a_running_eval` — mid-flight cancellation from
//!   another thread returns promptly instead of running to completion.
//!
//! Negative parity (cancellation does NOT fire when it must not):
//! * `installed_but_uncancelled_token_evaluates_normally` — the token's mere
//!   presence changes nothing. This is the regression guard for the hot-path
//!   check being wrong in the `false` direction.
//! * `token_cancelled_after_completion_still_returns_its_result` — the
//!   explicit acceptance case from the issue: a late cancel must not
//!   retroactively poison a finished eval.
//! * `guard_drop_restores_uncancellable_evaluation` — a cancelled token from
//!   an earlier scope must not leak into a later eval on the same thread.
//!   Without the restore-on-drop guard this is exactly how one cancelled
//!   Python call would break every subsequent one on that thread.

use std::sync::mpsc;
use std::time::{Duration, Instant};

use chelis_compiler_api::compiler;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use chelis_compiler_api::{
    CancelToken, EVAL_CANCELLED_KIND, install_cancel_token, is_cancellation,
};

/// A host-lane program: plain scalar arithmetic through the interpreter.
const HOST_PROGRAM: &str = "def answer() -> i32 = 6 * 7\n\nresult = answer()\n";

/// A tensor-lane program: the work lands in the DAG evaluator.
const TENSOR_PROGRAM: &str = "result = sum(to_tensor([1.0, 2.0, 3.0]), 0)\n";

fn request(source: &str) -> EvalRequest {
    EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: Default::default(),
    }
}

fn eval_error(source: &str) -> String {
    match compiler::eval(request(source)) {
        Ok(result) => panic!("expected cancellation, evaluated to {result:?}"),
        Err(err) => format!("{err:?}"),
    }
}

#[test]
fn cancelled_before_eval_returns_sentinel() {
    let token = CancelToken::new();
    token.cancel();
    let _guard = install_cancel_token(token);

    let err = eval_error(HOST_PROGRAM);
    assert!(
        is_cancellation(&err),
        "host lane must report cancellation, got: {err}"
    );
}

#[test]
fn cancelled_tensor_program_returns_sentinel() {
    let token = CancelToken::new();
    token.cancel();
    let _guard = install_cancel_token(token);

    let err = eval_error(TENSOR_PROGRAM);
    assert!(
        is_cancellation(&err),
        "tensor lane must report cancellation, got: {err}"
    );
}

/// Embedders must be able to tell cancellation from failure WITHOUT parsing
/// message text. The string sentinel is a lane-internal carrier, interpreted
/// once at the eval-stage boundary and re-expressed as a diagnostic kind.
#[test]
fn cancellation_is_reported_structurally_not_just_in_the_message() {
    let token = CancelToken::new();
    token.cancel();
    let _guard = install_cancel_token(token);

    let err = compiler::eval(request(HOST_PROGRAM)).expect_err("expected cancellation");

    assert!(
        err.is_cancellation(),
        "CompilerError::is_cancellation must recognise it: {err:?}"
    );
    assert!(
        err.errors
            .iter()
            .any(|diagnostic| diagnostic.kind().as_str() == EVAL_CANCELLED_KIND),
        "expected a diagnostic with kind {EVAL_CANCELLED_KIND:?}, got {:?}",
        err.errors.iter().map(|d| d.kind()).collect::<Vec<_>>()
    );
}

/// The converse: a genuine evaluation failure must NOT be classified as
/// cancellation. Without this, the CLI would translate a real error into
/// `evaluation timed out`, reporting the wrong cause.
#[test]
fn a_real_evaluation_error_is_not_classified_as_cancellation() {
    let err = compiler::eval(request("result = no_such_function(1)\n"))
        .expect_err("expected an evaluation error");

    assert!(
        !err.is_cancellation(),
        "a real error must not be classified as cancellation: {err:?}"
    );
    assert!(
        err.errors
            .iter()
            .all(|diagnostic| diagnostic.kind().as_str() != EVAL_CANCELLED_KIND),
        "no diagnostic should carry the cancellation kind: {err:?}"
    );
}

#[test]
fn installed_but_uncancelled_token_evaluates_normally() {
    let token = CancelToken::new();
    let _guard = install_cancel_token(token.clone());

    let result = compiler::eval(request(HOST_PROGRAM))
        .expect("an uncancelled token must not affect evaluation");
    assert!(
        !result.roots.is_empty(),
        "expected the program's root to evaluate"
    );
    assert!(!token.is_cancelled(), "nothing should have cancelled it");

    // Same for the tensor lane.
    compiler::eval(request(TENSOR_PROGRAM)).expect("tensor lane must evaluate normally too");
}

#[test]
fn token_cancelled_after_completion_still_returns_its_result() {
    let token = CancelToken::new();
    let guard = install_cancel_token(token.clone());

    let result = compiler::eval(request(HOST_PROGRAM)).expect("eval completes before any cancel");
    let roots_before = result.roots.len();

    // The eval already returned; cancelling now must not rewrite history.
    token.cancel();
    assert_eq!(
        result.roots.len(),
        roots_before,
        "a late cancel must not disturb an already-returned result"
    );
    drop(guard);
}

#[test]
fn guard_drop_restores_uncancellable_evaluation() {
    {
        let token = CancelToken::new();
        token.cancel();
        let _guard = install_cancel_token(token);
        let err = eval_error(HOST_PROGRAM);
        assert!(is_cancellation(&err), "setup: expected cancellation");
    }

    // The guard dropped, so this thread is back to having no token. A
    // subsequent eval must succeed — the leak-check that makes repeated
    // calls on one thread safe.
    let result = compiler::eval(request(HOST_PROGRAM))
        .expect("after the guard drops, evaluation must be unaffected");
    assert!(!result.roots.is_empty());
}

/// Mid-flight cancellation: the token is flipped by another thread while the
/// eval is running, and the eval must abandon rather than run to completion.
///
/// The program is a NESTED FOLD, not a deep recursion. Two reasons:
///
/// * a recursive formulation deep enough to be slow overflows the interpreter
///   stack before it is slow enough to cancel (the host runtime recurses on
///   the Rust stack per call frame);
/// * nesting keeps every intermediate list small. Cancellation latency is
///   bounded by the longest single uninterruptible step, and allocating or
///   dropping one huge list IS such a step — so a flat `range(0, 40_000_000)`
///   would measure list teardown rather than the cancellation mechanism.
///
/// The assertion is deliberately loose on timing: this test asserts the
/// *mechanism*, not the 250 ms bindings latency budget, which is measured by
/// the bindings probe where SIGINT delivery is the thing under test.
#[test]
fn cancellation_interrupts_a_running_eval() {
    const SLOW_PROGRAM: &str = "def inner(seed: i64) -> i64 =\n  \
        fold(fn (acc, x) -> acc + x * seed, 0i64, range(0, 2000))\n\n\
        result = fold(fn (acc, k) -> acc + inner(k), 0i64, range(0, 2000))\n";

    let token = CancelToken::new();
    let flipper = token.clone();
    let (started_tx, started_rx) = mpsc::channel();

    let worker = std::thread::spawn(move || {
        let _guard = install_cancel_token(token);
        started_tx.send(()).expect("signal start");
        let outcome = compiler::eval(request(SLOW_PROGRAM));
        outcome.map_err(|err| format!("{err:?}"))
    });

    started_rx.recv().expect("worker started");
    // Give the eval time to get past compilation and into the interpreter,
    // so we are cancelling a *running* evaluation.
    std::thread::sleep(Duration::from_millis(300));

    let cancelled_at = Instant::now();
    flipper.cancel();
    let outcome = worker.join().expect("worker must not panic");
    let elapsed = cancelled_at.elapsed();

    match outcome {
        Err(err) => assert!(
            is_cancellation(&err),
            "expected the cancellation sentinel, got: {err}"
        ),
        Ok(result) => panic!(
            "eval ran to completion ({} roots) instead of cancelling",
            result.roots.len()
        ),
    }
    assert!(
        elapsed < Duration::from_secs(30),
        "cancellation should unwind within one node visit, took {elapsed:?}"
    );
}
