//! Cooperative cancellation followed by a single-owner forced-exit backstop.
use chelis_compiler_api::{
    CancelToken, CancelTokenGuard, TranscriptCapture, TranscriptCaptureGuard,
};
use std::{io::Write, sync::mpsc, time::Duration};

const HARD_EXIT_GRACE: Duration = Duration::from_secs(5);

pub(crate) struct EvalTimeout {
    capture: TranscriptCapture,
    stop: mpsc::Sender<()>,
    _cancel: CancelTokenGuard,
    _capture: TranscriptCaptureGuard,
}

impl EvalTimeout {
    pub(crate) fn install(secs: u64, json: bool) -> Self {
        let capture = TranscriptCapture::new();
        let token = CancelToken::new();
        let (stop, stopped) = mpsc::channel();
        let worker_capture = capture.clone();
        let worker_token = token.clone();
        std::thread::spawn(move || {
            run_watchdog(secs, json, worker_token, worker_capture, |duration| {
                !matches!(
                    stopped.recv_timeout(duration),
                    Err(mpsc::RecvTimeoutError::Timeout)
                )
            });
        });
        Self {
            _capture: chelis_compiler_api::install_transcript_capture(capture.clone()),
            _cancel: chelis_compiler_api::install_cancel_token(token),
            capture,
            stop,
        }
    }

    /// Call only after normal result and diagnostic formatting has completed.
    pub(crate) fn claim_normal(&self) -> bool {
        self.capture.finish().is_some()
    }
}

impl Drop for EvalTimeout {
    fn drop(&mut self) {
        let _ = self.stop.send(());
    }
}

fn run_watchdog(
    secs: u64,
    json: bool,
    token: CancelToken,
    capture: TranscriptCapture,
    mut stopped: impl FnMut(Duration) -> bool,
) {
    if stopped(Duration::from_secs(secs)) {
        return;
    }
    token.cancel();
    if stopped(HARD_EXIT_GRACE) {
        return;
    }
    if let Some(lines) = capture.finish() {
        let message = format!(
            "evaluation timed out after {secs}s (--timeout); cancellation did not complete within {grace}s, forced exit",
            grace = HARD_EXIT_GRACE.as_secs()
        );
        // The transcript is complete before claiming; no rendering or I/O
        // occurs while the capture mutex is held.
        let output = crate::eval_output::EvalOutput::failure(lines, message.clone(), json);
        if let Err(error) = output.emit_effects() {
            let _ = writeln!(
                std::io::stderr(),
                "error: could not write completed evaluation output: {error}"
            );
        }
        // A closed diagnostic sink must not panic the terminal owner and
        // strand the main thread waiting for an exit that will never happen.
        let _ = writeln!(std::io::stderr(), "error: {message}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disarming_before_deadline_does_not_cancel() {
        let token = CancelToken::new();
        let capture = TranscriptCapture::new();
        run_watchdog(30, false, token.clone(), capture.clone(), |_| true);
        assert!(!token.is_cancelled());
        assert_eq!(capture.finish(), Some(vec![]));
    }

    #[test]
    fn normal_claim_wins_after_cancellation_without_a_forced_exit() {
        let token = CancelToken::new();
        let capture = TranscriptCapture::new();
        let mut deadline_seen = false;
        run_watchdog(30, false, token.clone(), capture.clone(), |_| {
            if !deadline_seen {
                deadline_seen = true;
                assert!(!token.is_cancelled());
            } else {
                assert!(token.is_cancelled());
                assert_eq!(capture.finish(), Some(vec![]));
            }
            false
        });
        assert!(token.is_cancelled());
        assert_eq!(capture.finish(), None);
    }

    #[test]
    fn forced_exit_retains_real_print_output_and_empty_negative() {
        for mode in ["text", "json", "empty-text", "empty-json"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "eval_timeout::tests::forced_exit_child",
                    "--nocapture",
                ])
                .env("CHELIS_TEST_FORCED_TRANSCRIPT", mode)
                .output()
                .unwrap();
            assert_eq!(output.status.code(), Some(1));
            let json = mode.ends_with("json");
            let empty = mode.starts_with("empty");
            let mut expected = String::new();
            if json && !empty {
                expected.push_str("before-timeout\n");
            }
            expected.push_str("error: evaluation timed out after 30s (--timeout); cancellation did not complete within 5s, forced exit\n");
            assert_eq!(output.stderr, expected.as_bytes());
            // The test harness writes its own preamble, but no test footer:
            // process::exit is real. The program's output is exact and once.
            let stdout = String::from_utf8(output.stdout).unwrap();
            assert_eq!(
                stdout.matches("before-timeout\n").count(),
                usize::from(!json && !empty)
            );
            if !json && !empty {
                assert!(stdout.ends_with("before-timeout\n"));
            }
        }
    }

    #[test]
    #[cfg(unix)]
    fn closed_stderr_does_not_prevent_forced_exit() {
        use std::os::{fd::OwnedFd, unix::net::UnixStream};
        use std::process::Stdio;
        for mode in ["text", "json", "empty-text", "empty-json"] {
            let (reader, writer) = UnixStream::pair().unwrap();
            drop(reader);
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "eval_timeout::tests::forced_exit_child",
                    "--nocapture",
                ])
                .env("CHELIS_TEST_FORCED_TRANSCRIPT", mode)
                .stdout(Stdio::null())
                .stderr(Stdio::from(OwnedFd::from(writer)))
                .status()
                .unwrap();
            assert_eq!(status.code(), Some(1), "closed stderr in {mode}");
        }
    }

    #[test]
    fn forced_exit_child() {
        let Ok(mode) = std::env::var("CHELIS_TEST_FORCED_TRANSCRIPT") else {
            return;
        };
        let capture = TranscriptCapture::new();
        let _guard = chelis_compiler_api::install_transcript_capture(capture.clone());
        if !mode.starts_with("empty") {
            let result = chelis_compiler_api::compiler::eval(chelis_compiler_api::schema::EvalRequest {
                source_kind: chelis_compiler_api::schema::SourceKind::Surf,
                source: "def run() -> i64 ! { IO } = {\n_ = print(\"before-timeout\")\n7i64\n}\nout = run()\n".into(),
                bindings: Default::default(),
            }).unwrap();
            assert_eq!(result.transcript, ["before-timeout"]);
        }
        // Deterministic deadline/grace events model a stalled evaluator or
        // formatter after its effect. No large allocation or timing bound.
        run_watchdog(
            30,
            mode.ends_with("json"),
            CancelToken::new(),
            capture,
            |_| false,
        );
        panic!("the forced-exit owner must terminate the process");
    }
}
