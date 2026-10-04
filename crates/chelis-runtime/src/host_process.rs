//! `process_run` (spec/05 §2.6, [05-OP-38], [05-HOST-2]), shared by every
//! execution lane.
//!
//! The evaluator spawns through its policy-checked system port and compiled
//! code through the `chelis_process_run` C export; both build the result
//! with [`decode_process_output`], so the exit-code and capture rules and
//! every failure message have one definition.

use std::process::Command;

/// One child process as the host reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawProcessOutput {
    /// `None` when a signal ended the process.
    pub exit_status: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

/// The `(exit_code, stdout, stderr)` result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessOutput {
    pub exit_code: i64,
    pub stdout: String,
    pub stderr: String,
}

/// Runs `program` with `args` passed straight to the OS as argv: no shell,
/// glob, variable, or backtick interpolation.
pub fn spawn_process(program: &str, args: &[String]) -> std::io::Result<RawProcessOutput> {
    let output = Command::new(program).args(args).output()?;
    Ok(RawProcessOutput {
        exit_status: output.status.code(),
        stdout: output.stdout,
        stderr: output.stderr,
    })
}

/// The failure text when the program cannot be started.
pub fn spawn_failure_message(program: &str, source: &std::io::Error) -> String {
    format!("process_run failed to spawn `{program}`: {source}")
}

/// The failure text for a capture that is not UTF-8.
pub fn invalid_capture_message(program: &str, stream: &str) -> String {
    format!("IO trap in process_run: program `{program}`, {stream}: output is not valid UTF-8")
}

/// Builds the result from one child process. A signal reports exit code
/// `-1`. Each capture converts strictly, stdout first: the first capture that
/// is not UTF-8 fails the call, and no replacement character substitutes.
pub fn decode_process_output(
    program: &str,
    output: RawProcessOutput,
) -> Result<ProcessOutput, String> {
    let RawProcessOutput {
        exit_status,
        stdout,
        stderr,
    } = output;
    let stdout =
        String::from_utf8(stdout).map_err(|_| invalid_capture_message(program, "stdout"))?;
    let stderr =
        String::from_utf8(stderr).map_err(|_| invalid_capture_message(program, "stderr"))?;
    Ok(ProcessOutput {
        exit_code: exit_status.map_or(-1_i64, i64::from),
        stdout,
        stderr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(exit_status: Option<i32>, stdout: &[u8], stderr: &[u8]) -> RawProcessOutput {
        RawProcessOutput {
            exit_status,
            stdout: stdout.to_vec(),
            stderr: stderr.to_vec(),
        }
    }

    #[test]
    fn valid_captures_keep_their_exact_bytes() {
        assert_eq!(
            decode_process_output("p", raw(Some(3), "é\n".as_bytes(), b"")),
            Ok(ProcessOutput {
                exit_code: 3,
                stdout: "é\n".to_string(),
                stderr: String::new(),
            })
        );
    }

    #[test]
    fn a_signal_reports_minus_one() {
        assert_eq!(
            decode_process_output("p", raw(None, b"", b"")).map(|out| out.exit_code),
            Ok(-1)
        );
    }

    #[test]
    fn stdout_is_checked_before_stderr() {
        assert_eq!(
            decode_process_output("p", raw(Some(0), b"\xff", b"\xff")),
            Err(invalid_capture_message("p", "stdout"))
        );
        assert_eq!(
            decode_process_output("p", raw(Some(0), b"ok", b"\xfe")),
            Err(
                "IO trap in process_run: program `p`, stderr: output is not valid UTF-8"
                    .to_string()
            )
        );
    }
}
