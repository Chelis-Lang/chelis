//! Cooperative FRONT-END cancellation (chelis#930).
//!
//! ## Background
//!
//! chelis#914 made *evaluation* cancellable: both eval lanes poll a
//! cancellation token at every node visit. The front end had no node visit to
//! poll at, so parse / desugar / check / lower ran to completion regardless of
//! the token and a signal arriving during compilation was not observed until
//! evaluation started. Measured on a 70 KB source with a ~19.3 s front end, an
//! interrupt at t+2 s raised at t+19.32 s — the *remaining compile time*,
//! which is the original chelis#914 defect relocated one phase earlier.
//!
//! ## What is checked here
//!
//! The mechanism, not the bindings-level SIGINT latency. The latter is the
//! acceptance oracle and lives in
//! `bindings/python/tests/manual_eval_interrupt.py`, driven by
//! `crates/chelis-python/tests/manual_eval_interrupt.rs`.
//!
//! Positive (cancellation is observed):
//! * `cancelled_before_compile_is_reported_at_the_first_phase`
//! * `cancellation_is_reported_structurally_not_just_in_the_message`
//! * `cancellation_interrupts_a_running_front_end` — the core claim: a
//!   cancelled compile abandons in a small fraction of the compile it would
//!   otherwise have finished.
//! * `cancellation_interrupts_the_front_end_of_a_trivial_evaluation` — the
//!   chelis#930 repro shape, with the eval deliberately trivial so nothing but
//!   the front end can be responsible for the latency.
//!
//! Negative parity (cancellation does NOT fire when it must not):
//! * `installed_but_uncancelled_token_compiles_normally` — presence of a token
//!   changes neither the result nor (materially) the time.
//! * `a_real_type_error_is_not_classified_as_cancellation`
//! * `guard_drop_restores_uncancellable_compilation`
//!
//! Attribution:
//! * `cancellation_supersedes_the_abandoned_pass_diagnostics` — an abandoned
//!   walk reports its truncated view (unbound tail names). Those are artefacts
//!   of the abandonment, not findings about the program, and must never be
//!   shown to a user who simply asked the compile to stop.

use std::sync::mpsc;
use std::time::{Duration, Instant};

use chelis_compiler_api::compiler;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use chelis_compiler_api::{
    CancelToken, EVAL_CANCELLED_KIND, install_cancel_token, is_cancellation,
};

/// Declaration count for the front-end-dominated program. chelis#930's own
/// repro used 1500 (~70 KB, ~19.3 s front end); this is sized down so the test
/// stays inside the inner-loop budget while keeping the same shape — the front
/// end is seconds and the evaluation is a single multiply-add. The full-size
/// repro is the bindings probe's job.
const DEFS: usize = 400;

/// A program whose cost is entirely front-end: many trivial declarations, one
/// trivial root. Nothing here can make *evaluation* slow, so any latency
/// measured against it is compile latency.
fn front_end_heavy_source(defs: usize) -> String {
    let mut source = String::with_capacity(defs * 48);
    for index in 0..defs {
        source.push_str(&format!(
            "def f{index}(x: i64) -> i64 = x * {index}i64 + 1i64\n"
        ));
    }
    source.push_str("result = f0(1i64)\n");
    source
}

fn request(source: &str) -> EvalRequest {
    EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: Default::default(),
    }
}

/// Run `source` on a worker thread with `token` installed, cancel it after
/// `cancel_after`, and report `(latency from cancel to unwind, error text)`.
fn cancel_mid_compile(source: String, cancel_after: Duration) -> (Duration, String) {
    let token = CancelToken::new();
    let flipper = token.clone();
    let (started_tx, started_rx) = mpsc::channel();

    let worker = std::thread::spawn(move || {
        let _guard = install_cancel_token(token);
        started_tx.send(()).expect("signal start");
        compiler::eval(request(&source)).map_err(|err| format!("{err:?}"))
    });

    started_rx.recv().expect("worker started");
    std::thread::sleep(cancel_after);

    let cancelled_at = Instant::now();
    flipper.cancel();
    let outcome = worker.join().expect("worker must not panic");
    let elapsed = cancelled_at.elapsed();

    match outcome {
        Err(err) => (elapsed, err),
        Ok(result) => panic!(
            "the compile ran to completion ({} roots) instead of cancelling",
            result.roots.len()
        ),
    }
}

/// Wall-clock for an uncancelled compile + eval of the same program. The
/// baseline every latency claim below is relative to.
fn uncancelled_duration(source: &str) -> Duration {
    let started = Instant::now();
    compiler::eval(request(source)).expect("the program must compile and evaluate");
    started.elapsed()
}

#[test]
fn cancelled_before_compile_is_reported_at_the_first_phase() {
    let token = CancelToken::new();
    token.cancel();
    let _guard = install_cancel_token(token);

    let err = compiler::eval(request("result = 1i64\n")).expect_err("expected cancellation");

    assert!(
        err.is_cancellation(),
        "an already-cancelled token must abandon before parsing: {err:?}"
    );
    assert_eq!(
        err.stage, "parse",
        "the reported stage should name the phase that was about to run"
    );
}

/// Embedders must classify on `Diagnostic::kind`, never on message text. Same
/// contract chelis#914 established for the eval lanes; the front end reuses it
/// rather than inventing a second signal.
#[test]
fn cancellation_is_reported_structurally_not_just_in_the_message() {
    let token = CancelToken::new();
    token.cancel();
    let _guard = install_cancel_token(token);

    let err = compiler::eval(request("result = 1i64\n")).expect_err("expected cancellation");

    assert!(
        err.errors
            .iter()
            .any(|diagnostic| diagnostic.kind == EVAL_CANCELLED_KIND),
        "expected a diagnostic with kind {EVAL_CANCELLED_KIND:?}, got {:?}",
        err.errors.iter().map(|d| &d.kind).collect::<Vec<_>>()
    );
    assert!(
        is_cancellation(&format!("{err:?}")),
        "the flattened text must still carry the sentinel for the CLI boundary"
    );
}

/// The core claim of chelis#930. Before this change the worker below would
/// have compiled to completion and the measured latency would have equalled
/// the *remaining* compile time.
#[test]
fn cancellation_interrupts_a_running_front_end() {
    let source = front_end_heavy_source(DEFS);
    let baseline = uncancelled_duration(&source);

    // Late enough to be unambiguously inside the type checker rather than in
    // parse, early enough that most of the compile is still ahead.
    let cancel_after = baseline / 4;
    let (latency, err) = cancel_mid_compile(source, cancel_after);

    println!(
        "issue#930: {DEFS} defs, uncancelled compile {baseline:?}, \
         cancelled at {cancel_after:?}, unwound in {latency:?}"
    );

    assert!(
        is_cancellation(&err),
        "expected the cancellation sentinel, got: {err}"
    );
    // The bound that matters is "a small fraction of the work abandoned", not
    // an absolute millisecond figure: this test runs on shared CI hardware
    // where an absolute budget measures the runner, not the mechanism. The
    // bindings probe owns the absolute number.
    assert!(
        latency * 4 < baseline,
        "cancellation should abandon promptly: unwound in {latency:?} \
         out of a {baseline:?} compile (cancelled at {cancel_after:?})"
    );
}

/// The chelis#930 repro shape stated explicitly: a front-end-dominated program
/// with a trivial evaluation. If the front end were still uncancellable, the
/// token could only be observed once evaluation started — and this program's
/// evaluation is a single multiply-add, so there would be nothing left to
/// interrupt and the latency would be the whole remaining compile.
#[test]
fn cancellation_interrupts_the_front_end_of_a_trivial_evaluation() {
    let source = front_end_heavy_source(DEFS);
    let baseline = uncancelled_duration(&source);

    let (latency, err) = cancel_mid_compile(source, Duration::from_millis(50));

    assert!(is_cancellation(&err), "expected cancellation, got: {err}");
    assert!(
        latency * 2 < baseline,
        "a compile cancelled 50 ms in must not wait out the rest of the \
         front end: unwound in {latency:?} of a {baseline:?} compile"
    );
}

/// Full-size chelis#930 repro (1500 declarations, ~70 KB), swept across the
/// compile so no single phase can hide behind an average. `#[ignore]` because
/// one uncancelled baseline plus eight cancelled runs is well outside the
/// inner-loop budget.
///
/// **Manual command:**
///
/// ```text
/// cargo nextest run -p chelis-compiler-api \
///   --test issue_930_frontend_cancellation -- --ignored --no-capture
/// ```
///
/// **Expected success condition:** every sweep point unwinds in a small
/// fraction of the remaining compile, and the printed table shows no point
/// where latency tracks "time left" (which is the pre-fix signature).
///
/// The two early sample points are deliberate. Declaration collection and the
/// inference schedule's component search run *before* body inference and both
/// scale with declaration count (measured 1.3 s and 0.9 s over 1500
/// declarations), so a sweep that starts a ninth of the way in would step over
/// them entirely — and they were the window that made a loaded machine fall
/// through to `--timeout`'s hard-exit backstop.
#[test]
#[ignore = "manual chelis#930 sweep: ~1500-declaration compile, run eleven times"]
fn frontend_cancellation_latency_sweep_at_repro_scale() {
    const REPRO_DEFS: usize = 1500;
    /// Sample points as (numerator, denominator) fractions of the baseline.
    const SAMPLES: [(u32, u32); 10] = [
        (1, 100),
        (1, 20),
        (1, 9),
        (2, 9),
        (3, 9),
        (4, 9),
        (5, 9),
        (6, 9),
        (7, 9),
        (8, 9),
    ];

    let source = front_end_heavy_source(REPRO_DEFS);
    let baseline = uncancelled_duration(&source);
    println!("issue#930 sweep: {REPRO_DEFS} defs, uncancelled compile {baseline:?}");

    let mut worst = Duration::ZERO;
    for (numerator, denominator) in SAMPLES {
        let cancel_after = baseline * numerator / denominator;
        let (latency, err) = cancel_mid_compile(source.clone(), cancel_after);
        assert!(is_cancellation(&err), "expected cancellation, got: {err}");
        let remaining = baseline.saturating_sub(cancel_after);
        println!("  cancel at {cancel_after:?} (of {baseline:?}) -> unwound in {latency:?}");
        assert!(
            latency * 4 < remaining,
            "latency {latency:?} is not a small fraction of the {remaining:?} \
             of compile still ahead: that is the pre-fix signature"
        );
        worst = worst.max(latency);
    }
    println!("issue#930 sweep: worst-case latency {worst:?}");
}

#[test]
fn installed_but_uncancelled_token_compiles_normally() {
    let source = front_end_heavy_source(DEFS);
    let token = CancelToken::new();
    let _guard = install_cancel_token(token.clone());

    let result =
        compiler::eval(request(&source)).expect("an uncancelled token must not affect compilation");
    assert!(
        !result.roots.is_empty(),
        "expected the program's root to evaluate"
    );
    assert!(!token.is_cancelled(), "nothing should have cancelled it");
}

/// The converse of the positive tests: a genuine front-end failure must keep
/// its own diagnosis. Without this the CLI would translate a plain type error
/// into `evaluation timed out`, reporting the wrong cause.
#[test]
fn a_real_type_error_is_not_classified_as_cancellation() {
    let err = compiler::eval(request("result = no_such_function(1i64)\n"))
        .expect_err("expected a front-end error");

    assert!(
        !err.is_cancellation(),
        "a real error must not be classified as cancellation: {err:?}"
    );
    assert!(
        err.errors
            .iter()
            .all(|diagnostic| diagnostic.kind != EVAL_CANCELLED_KIND),
        "no diagnostic should carry the cancellation kind: {err:?}"
    );
}

/// A cancelled compile must not report the abandoned pass's partial view.
/// This program has a genuine type error, but the compile was abandoned before
/// anything could be established about it, so "cancelled" is the only honest
/// answer — reporting the type error would assert a finding the compiler never
/// actually reached.
#[test]
fn cancellation_supersedes_the_abandoned_pass_diagnostics() {
    let token = CancelToken::new();
    token.cancel();
    let _guard = install_cancel_token(token);

    let err = compiler::eval(request("result = no_such_function(1i64)\n"))
        .expect_err("expected cancellation");

    assert!(
        err.is_cancellation(),
        "cancellation must supersede the abandoned pass's diagnostics: {err:?}"
    );
    assert!(
        !format!("{err:?}").contains("no_such_function"),
        "a cancelled compile must not report findings it never established: {err:?}"
    );
}

#[test]
fn guard_drop_restores_uncancellable_compilation() {
    {
        let token = CancelToken::new();
        token.cancel();
        let _guard = install_cancel_token(token);
        let err = compiler::eval(request("result = 1i64\n")).expect_err("setup: expect cancel");
        assert!(err.is_cancellation(), "setup: expected cancellation");
    }

    // The guard dropped, so this thread has no token again. Without
    // restore-on-drop, one cancelled call would break every later compile on
    // the same thread — the exact leak chelis#914's guard exists to prevent.
    let result = compiler::eval(request("result = 1i64\n"))
        .expect("after the guard drops, compilation must be unaffected");
    assert!(!result.roots.is_empty());
}
