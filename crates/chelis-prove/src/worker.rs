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
//! Isolation is OPT-IN: only a host whose `main` calls [`enable_isolation`]
//! (and [`run_worker_if_requested`] first) spawns workers. Tests do not opt
//! in, so they solve in-process (no spawn, fast) and exercise the in-process
//! lowering directly. The end-to-end isolated path is covered by an
//! integration test that runs the real `chelis` binary.

use std::sync::atomic::{AtomicBool, Ordering};

/// Env var that marks a spawned process as a Tier B worker. The parent sets
/// it on the child; the child's `main` sees it (via
/// [`run_worker_if_requested`]) and runs one solve instead of its normal
/// logic.
#[cfg(feature = "smt")]
const WORKER_ENV: &str = "CHELIS_PROVE_WORKER";

static ISOLATION_ENABLED: AtomicBool = AtomicBool::new(false);

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
    use crate::tier_b::{SmtProperty, TierBResult, solve_property_cvc5};
    use serde::{Deserialize, Serialize};

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

        let mut child = match Command::new(&exe)
            .env(WORKER_ENV, "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
        {
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
        if let Some(mut out) = child.stdout.take() {
            std::thread::spawn(move || {
                let mut buf = Vec::new();
                let _ = out.read_to_end(&mut buf);
                let _ = tx.send(buf); // detached: a late send after we gave up is harmless.
            });
        }

        // Wait for the child, killing it past a generous deadline. cvc5
        // honors `tlimit-per` (= timeout_ms), so the kill only fires on a
        // genuine hang.
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
