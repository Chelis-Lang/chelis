//! Tier B process isolation.
//!
//! cvc5's `mk_term` aborts the PROCESS (not a catchable error) on a malformed
//! term, and the recursive lowering can also overflow the stack (SIGABRT) or
//! OOM. The in-process guards in [`crate::tier_b`] close every known cause,
//! but because cvc5 fails by process-abort, in-process guarding can never be
//! PROVEN exhaustive. Isolation makes it moot: a production host runs every
//! cvc5 solve in a short-lived CHILD process, so ANY way the solve can take a
//! process down -- a cvc5 C++ abort, a cvc5-internal assertion on a
//! well-formed formula, a stack overflow, an OOM kill, a panic -- becomes a
//! clean [`TierBResult::Error`]/`Unknown` in the parent (routed to Tier C)
//! rather than a bare process exit with empty stdout.
//!
//! Production-host isolation is OPT-IN: a host whose `main` calls
//! [`enable_isolation`] (and [`run_worker_if_requested`] first) spawns workers.
//! The `chelis-prove` unit-test build enables the same process boundary by
//! default and re-execs one dedicated worker-entry test. That keeps libtest's
//! parallel threads from entering cvc5/LibPoly/GMP concurrently while still
//! exercising the real solver and the same fail-closed parent mapping. The
//! end-to-end production path is covered by an integration test that runs the
//! real `chelis` binary.

use std::sync::atomic::{AtomicBool, Ordering};

/// Env var that marks a spawned process as a Tier B worker. The parent sets
/// it on the child; the child's `main` sees it (via
/// [`run_worker_if_requested`]) and runs one solve instead of its normal
/// logic.
#[cfg(feature = "smt")]
const WORKER_ENV: &str = "CHELIS_PROVE_WORKER";

#[cfg(all(feature = "smt", test))]
const TEST_WORKER_ENV: &str = "CHELIS_PROVE_TEST_WORKER";

#[cfg(all(feature = "smt", test))]
const TEST_WORKER_ENTRY: &str = "worker::imp::tests::issue_1333_worker_process_entry";

// Unit tests default to the shipped process-containment behavior. Production
// hosts remain explicitly opt-in because their main must dispatch the worker
// marker before normal argument parsing.
static ISOLATION_ENABLED: AtomicBool = AtomicBool::new(cfg!(test));

/// Enable Tier B subprocess isolation for this process. Call once at startup
/// in a production binary whose `main` ALSO calls [`run_worker_if_requested`]
/// first (so a spawned worker dispatches before reaching this). Idempotent.
pub fn enable_isolation() {
    ISOLATION_ENABLED.store(true, Ordering::SeqCst);
}

/// Whether [`enable_isolation`] has been called in this process.
#[cfg(feature = "smt")]
pub(crate) fn isolation_enabled() -> bool {
    ISOLATION_ENABLED.load(Ordering::SeqCst)
}

/// If this process was spawned as a Tier B worker, run one solve and EXIT;
/// otherwise return so the host proceeds normally. Call this as the FIRST
/// thing in `main`, before argument parsing and before [`enable_isolation`],
/// so a worker child never runs the host's normal logic. A no-op without the
/// `smt` feature (no cvc5 to isolate).
pub fn run_worker_if_requested() {
    #[cfg(feature = "smt")]
    {
        use std::io::IsTerminal;
        // A real worker is always spawned with a PIPED stdin (the parent feeds
        // the request that way). If the marker is set but stdin is a terminal,
        // it is almost certainly a user who exported the internal env var by
        // mistake -- do NOT hijack their command into the worker loop (which
        // would block reading the terminal forever). Proceed normally instead.
        if std::env::var_os(WORKER_ENV).is_some() && !std::io::stdin().is_terminal() {
            imp::run_worker_loop();
        }
    }
}

#[cfg(feature = "smt")]
pub(crate) use imp::solve_property_isolated;

#[cfg(feature = "smt")]
mod imp {
    use super::WORKER_ENV;
    #[cfg(test)]
    use super::{TEST_WORKER_ENTRY, TEST_WORKER_ENV};
    use crate::tier_b::{SmtProperty, TierBResult, solve_property_cvc5};
    use serde::{Deserialize, Serialize};

    #[cfg(test)]
    thread_local! {
        /// Per-test-thread crash injection. Keeping this thread-local prevents
        /// the negative control from poisoning unrelated parallel solver tests.
        static TEST_WORKER_CRASH: std::cell::Cell<Option<&'static str>> =
            const { std::cell::Cell::new(None) };
    }

    #[cfg(test)]
    fn with_test_worker_crash<T>(mode: &'static str, f: impl FnOnce() -> T) -> T {
        struct Reset(Option<&'static str>);

        impl Drop for Reset {
            fn drop(&mut self) {
                TEST_WORKER_CRASH.set(self.0);
            }
        }

        let reset = Reset(TEST_WORKER_CRASH.replace(Some(mode)));
        let result = f();
        drop(reset);
        result
    }

    /// The request sent (bincode) to a worker over stdin. bincode is used
    /// rather than JSON because a property may carry a non-finite `RealLit`
    /// (which JSON cannot represent) -- the worker's own guards reject it,
    /// but it must survive transport to get there.
    #[derive(Serialize, Deserialize)]
    struct WireRequest {
        property: SmtProperty,
        timeout_ms: u64,
    }

    /// The result returned (bincode) from a worker over stdout. The
    /// disproved model is carried as a JSON STRING rather than a
    /// `serde_json::Value`, because `Value`'s `Deserialize` uses
    /// `deserialize_any`, which bincode (a non-self-describing format) does
    /// not support.
    #[derive(Serialize, Deserialize)]
    enum WireResult {
        Proved,
        Disproved(String),
        Timeout,
        Unknown,
        Error(String),
    }

    impl WireResult {
        fn from_tier_b(r: TierBResult) -> Self {
            match r {
                TierBResult::Proved => WireResult::Proved,
                TierBResult::Disproved(v) => WireResult::Disproved(
                    serde_json::to_string(&v).unwrap_or_else(|_| "null".into()),
                ),
                TierBResult::Timeout => WireResult::Timeout,
                TierBResult::Unknown => WireResult::Unknown,
                TierBResult::Error(s) => WireResult::Error(s),
            }
        }

        fn into_tier_b(self) -> TierBResult {
            match self {
                WireResult::Proved => TierBResult::Proved,
                WireResult::Disproved(s) => TierBResult::Disproved(
                    serde_json::from_str(&s).unwrap_or(serde_json::Value::Null),
                ),
                WireResult::Timeout => TierBResult::Timeout,
                WireResult::Unknown => TierBResult::Unknown,
                WireResult::Error(s) => TierBResult::Error(s),
            }
        }
    }

    /// The worker body: read one [`WireRequest`] from stdin, solve it
    /// IN-PROCESS, write a [`WireResult`] to stdout, exit. Never returns. A
    /// non-zero exit / signal death here is exactly what the parent maps to a
    /// clean Tier C result, so every failure path simply exits non-zero.
    pub(super) fn run_worker_loop() -> ! {
        use std::io::{Read, Write};

        // Deterministic crash hook for the isolation self-test ONLY: a real
        // run never sets this env var. It lets an integration test prove the
        // PARENT recovers from a worker that aborts / panics / overflows,
        // without needing to find a real cvc5 abort.
        if let Some(mode) = std::env::var_os("CHELIS_PROVE_WORKER_CRASH") {
            match mode.to_str() {
                Some("abort") => std::process::abort(),
                Some("panic") => panic!("CHELIS_PROVE_WORKER_CRASH=panic (isolation self-test)"),
                Some("hang") => loop {
                    // Block forever; the parent's deadline watchdog must kill
                    // this and recover (isolation self-test).
                    std::thread::sleep(std::time::Duration::from_secs(3600));
                },
                Some("overflow") => {
                    // Deliberately unbounded recursion to overflow the stack
                    // (SIGABRT) -- the whole point of this self-test branch.
                    #[allow(unconditional_recursion)]
                    fn blow(n: u64) -> u64 {
                        std::hint::black_box(blow(n.wrapping_add(1))).wrapping_add(1)
                    }
                    std::hint::black_box(blow(0));
                }
                _ => {}
            }
        }

        let mut buf = Vec::new();
        if std::io::stdin().read_to_end(&mut buf).is_err() {
            std::process::exit(2);
        }
        let req: WireRequest = match bincode::deserialize(&buf) {
            Ok(r) => r,
            Err(_) => std::process::exit(3),
        };
        let result = solve_property_cvc5(&req.property, req.timeout_ms);
        let bytes = match bincode::serialize(&WireResult::from_tier_b(result)) {
            Ok(b) => b,
            Err(_) => std::process::exit(4),
        };
        // A re-executed libtest process writes harness status to stdout. Its
        // dedicated worker entry therefore returns the binary frame on stderr;
        // production workers retain the original stdout transport.
        #[cfg(test)]
        if std::env::var_os(TEST_WORKER_ENV).is_some() {
            let stderr = std::io::stderr();
            let mut lock = stderr.lock();
            if lock.write_all(&bytes).is_err() || lock.flush().is_err() {
                std::process::exit(5);
            }
            std::process::exit(0);
        }

        let stdout = std::io::stdout();
        let mut lock = stdout.lock();
        if lock.write_all(&bytes).is_err() || lock.flush().is_err() {
            std::process::exit(5);
        }
        std::process::exit(0);
    }

    /// Parent side: spawn `current_exe` as a worker, send the request, read
    /// the result. ANY failure (spawn error, abnormal exit / signal, hang,
    /// undecodable output) maps to a clean `TierBResult` routed to Tier C --
    /// the parent process is never taken down by the solve.
    pub(crate) fn solve_property_isolated(property: &SmtProperty, timeout_ms: u64) -> TierBResult {
        use std::io::{Read, Write};
        use std::process::{Command, Stdio};
        use std::time::{Duration, Instant};

        // Screen depth BEFORE `clone` / `bincode::serialize` below: both
        // recurse on `SmtExpr` depth, so a pathologically deep property would
        // overflow THIS process (the parent is not isolated from itself)
        // before a child is ever spawned. The check is iterative and cannot
        // itself overflow.
        if crate::tier_b::property_exceeds_smt_depth(property) {
            return TierBResult::Error(
                "prove isolation: property nests too deep to lower (routes to Tier C)".to_string(),
            );
        }

        let exe = match std::env::current_exe() {
            Ok(p) => p,
            Err(e) => {
                return TierBResult::Error(format!(
                    "prove isolation: cannot resolve current executable: {e} (routes to Tier C)"
                ));
            }
        };
        let req = WireRequest {
            property: property.clone(),
            timeout_ms,
        };
        let payload = match bincode::serialize(&req) {
            Ok(b) => b,
            Err(e) => {
                return TierBResult::Error(format!(
                    "prove isolation: request serialize failed: {e} (routes to Tier C)"
                ));
            }
        };

        let mut command = Command::new(&exe);
        command.env(WORKER_ENV, "1").stdin(Stdio::piped());

        #[cfg(test)]
        {
            // Re-enter only the worker test, never the full suite. `--nocapture`
            // lets its binary stderr frame reach the parent unchanged.
            command
                .arg("--exact")
                .arg(TEST_WORKER_ENTRY)
                .arg("--nocapture")
                .env(TEST_WORKER_ENV, "1")
                .stdout(Stdio::null())
                .stderr(Stdio::piped());
            TEST_WORKER_CRASH.with(|mode| {
                if let Some(mode) = mode.get() {
                    command.env("CHELIS_PROVE_WORKER_CRASH", mode);
                }
            });
        }

        #[cfg(not(test))]
        command.stdout(Stdio::piped()).stderr(Stdio::null());

        let mut child = match command.spawn() {
            Ok(c) => c,
            Err(e) => {
                return TierBResult::Error(format!(
                    "prove isolation: worker spawn failed: {e} (routes to Tier C)"
                ));
            }
        };

        // Feed the request and close stdin (drop) so the worker sees EOF.
        if let Some(mut stdin) = child.stdin.take() {
            let _ = stdin.write_all(&payload);
        }

        // Drain stdout on a DETACHED thread that hands the bytes back over a
        // channel, so the parent's wait for output is BOUNDED. `read_to_end`
        // only returns on stdout EOF, which needs every write-end of the pipe
        // closed; if a (hypothetical future) worker left a grandchild holding
        // stdout, an unconditional `join` here would block the parent FOREVER
        // even after the worker is killed. Bounding the receive guarantees the
        // parent can never hang regardless of what the worker does with its
        // stdout (RT-isolation F1).
        let (tx, rx) = std::sync::mpsc::channel::<Vec<u8>>();
        #[cfg(test)]
        let worker_output = child.stderr.take();
        #[cfg(not(test))]
        let worker_output = child.stdout.take();
        if let Some(mut out) = worker_output {
            std::thread::spawn(move || {
                let mut buf = Vec::new();
                let _ = out.read_to_end(&mut buf);
                let _ = tx.send(buf); // detached: a late send after we gave up is harmless.
            });
        }

        // Wait for the child, killing it past a generous deadline. Solver
        // phases share a request-wide cvc5 budget bounded by `timeout_ms`;
        // requests without an auxiliary phase retain that whole budget. The
        // unchanged grace period therefore covers orchestration overhead
        // without misclassifying valid multi-phase work as a hung child.
        let deadline = Instant::now() + Duration::from_millis(timeout_ms.saturating_add(10_000));
        let mut timed_out = false;
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break Some(status),
                Ok(None) => {
                    if Instant::now() >= deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        timed_out = true;
                        break None;
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
                Err(e) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return TierBResult::Error(format!(
                        "prove isolation: waiting on worker failed: {e} (routes to Tier C)"
                    ));
                }
            }
        };

        if timed_out {
            // A hung worker behaves like a solver timeout. Do NOT block on the
            // reader (its EOF may never come); the thread is detached.
            return TierBResult::Unknown;
        }

        // The worker has exited, so its stdout write-end is closed and the
        // reader reaches EOF promptly -- but bound the wait anyway so a stray
        // inherited write-end can never hang us.
        let output = rx
            .recv_timeout(Duration::from_millis(2_000))
            .unwrap_or_default();

        match bincode::deserialize::<WireResult>(&output) {
            Ok(wire) => wire.into_tier_b(),
            Err(_) => {
                let detail = match status {
                    Some(s) if s.success() => "worker produced no decodable result".to_string(),
                    Some(s) => format!("worker exited abnormally ({s})"),
                    None => "worker did not exit".to_string(),
                };
                TierBResult::Error(format!("prove isolation: {detail} (routes to Tier C)"))
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::solver::{CmpOp, SmtExpr, SmtSort};

        fn trivially_true_property() -> SmtProperty {
            SmtProperty {
                variables: vec![],
                preconditions: vec![],
                postcondition: SmtExpr::BoolLit(true),
            }
        }

        /// Re-exec entry used only by the unit-test binary's isolation path.
        /// A normal test run has no worker marker, so this returns immediately.
        #[test]
        fn issue_1333_worker_process_entry() {
            crate::worker::run_worker_if_requested();
        }

        /// Positive control: the unit-test build must select the same process
        /// containment boundary as the shipped CLI rather than calling cvc5
        /// in parallel libtest threads.
        #[test]
        fn issue_1333_test_lane_defaults_to_process_isolation() {
            assert!(
                crate::worker::isolation_enabled(),
                "the chelis-prove unit-test binary must isolate every cvc5 solve"
            );
        }

        /// Positive control against a vacuous containment fix: a healthy child
        /// still reaches cvc5 and returns its exact Tier-B verdict.
        #[test]
        fn issue_1333_isolated_test_worker_preserves_real_smt_verdict() {
            assert_eq!(
                crate::tier_b::solve_property(&trivially_true_property(), 5_000),
                TierBResult::Proved
            );
        }

        /// Negative parity: a solver child that dies by signal must become a
        /// non-empty, typed failure-channel result. It may never be reported as
        /// a proof or silently collapse to timeout/unknown.
        #[test]
        fn issue_1333_test_worker_abort_fails_closed_with_reason() {
            let result = with_test_worker_crash("abort", || {
                crate::tier_b::solve_property(&trivially_true_property(), 5_000)
            });
            match result {
                TierBResult::Error(reason) => {
                    assert!(
                        !reason.trim().is_empty(),
                        "failure reason must be non-empty"
                    );
                    assert!(
                        reason.contains("worker exited abnormally")
                            && reason.contains("routes to Tier C"),
                        "unexpected containment diagnostic: {reason}"
                    );
                }
                other => {
                    panic!("a signalled cvc5 worker must fail closed with Error, got {other:?}")
                }
            }
        }

        /// The IPC types round-trip through bincode -- including a property
        /// carrying a NON-FINITE literal, which JSON could not represent and
        /// which is exactly why the request is bincode-encoded.
        #[test]
        fn wire_request_roundtrips_through_bincode_including_non_finite() {
            let property = SmtProperty {
                variables: vec![("x".to_string(), SmtSort::Real)],
                preconditions: vec![],
                postcondition: SmtExpr::Cmp(
                    CmpOp::Ge,
                    Box::new(SmtExpr::Var("x".to_string())),
                    Box::new(SmtExpr::RealLit(f64::INFINITY)),
                ),
            };
            let req = WireRequest {
                property,
                timeout_ms: 5000,
            };
            let bytes = bincode::serialize(&req).expect("serialize");
            let back: WireRequest = bincode::deserialize(&bytes).expect("deserialize");
            assert_eq!(back.timeout_ms, 5000);
            assert_eq!(back.property.variables.len(), 1);
            match &back.property.postcondition {
                SmtExpr::Cmp(CmpOp::Ge, _, r) => match &**r {
                    SmtExpr::RealLit(v) => {
                        assert!(v.is_infinite(), "non-finite survived transport")
                    }
                    other => panic!("expected RealLit, got {other:?}"),
                },
                other => panic!("expected Cmp, got {other:?}"),
            }
        }

        /// A disproved model with its JSON-string carrier round-trips back to
        /// a `serde_json::Value`.
        #[test]
        fn wire_result_disproved_roundtrips_the_model() {
            let mut model = serde_json::Map::new();
            model.insert("x".to_string(), serde_json::Value::String("3".to_string()));
            let original = TierBResult::Disproved(serde_json::Value::Object(model));
            let wire = WireResult::from_tier_b(original.clone());
            let bytes = bincode::serialize(&wire).expect("serialize");
            let back: WireResult = bincode::deserialize(&bytes).expect("deserialize");
            assert_eq!(back.into_tier_b(), original);
        }
    }
}
