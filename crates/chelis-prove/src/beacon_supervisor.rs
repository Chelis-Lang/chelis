//! LU1: one fallible subprocess boundary for both Beacon proof routes.

use std::ffi::OsString;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use chelis_types::unsupported::{Stage, Unsupported, UnsupportedKind};

const POLL_INTERVAL: Duration = Duration::from_millis(5);
const REAP_BUDGET: Duration = Duration::from_secs(1);
const MAX_CAPTURE_BYTES: u64 = 8 * 1024 * 1024;

pub(crate) struct Invocation<'a> {
    pub binary: &'a Path,
    pub args: &'a [OsString],
    pub input: Input<'a>,
    pub timeout: Duration,
}

#[derive(Clone, Copy)]
pub(crate) enum Input<'a> {
    Stdin(&'a [u8]),
    RequestFile(&'a [u8]),
}

#[derive(Debug)]
pub(crate) enum Outcome {
    Completed {
        status: ExitStatus,
        stdout: Vec<u8>,
        stderr: Vec<u8>,
    },
    TimedOut,
    Degraded(Unsupported),
}

/// Every OS operation the supervisor performs passes through this seam so a
/// denied mechanism can be exercised without imposing a sandbox on the test
/// runner itself. Production uses `SystemOps` only.
pub(crate) trait ProcessOps {
    fn capture(&self) -> io::Result<tempfile::NamedTempFile>;
    fn reopen(&self, file: &tempfile::NamedTempFile) -> io::Result<File>;
    fn spawn(&self, command: &mut Command) -> io::Result<Child>;
    fn write(&self, stdin: &mut tempfile::NamedTempFile, bytes: &[u8]) -> io::Result<()>;
    fn poll(&self, child: &mut Child) -> io::Result<Option<ExitStatus>>;
    fn kill(&self, child: &mut Child) -> io::Result<()>;
    fn size(&self, path: &Path) -> io::Result<u64>;
    fn read(&self, path: &Path) -> io::Result<Vec<u8>>;
}

pub(crate) struct SystemOps;

impl ProcessOps for SystemOps {
    fn capture(&self) -> io::Result<tempfile::NamedTempFile> {
        tempfile::NamedTempFile::new()
    }

    fn reopen(&self, file: &tempfile::NamedTempFile) -> io::Result<File> {
        file.reopen()
    }

    fn spawn(&self, command: &mut Command) -> io::Result<Child> {
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;

            // This runs in the child after fork. The two syscalls do not
            // allocate or invoke Rust callbacks. Keep an existing lower
            // file-size limit, and cap regular-file output even if the parent
            // is delayed between polling observations.
            unsafe {
                command.pre_exec(|| {
                    let mut limit = libc::rlimit {
                        rlim_cur: 0,
                        rlim_max: 0,
                    };
                    if libc::getrlimit(libc::RLIMIT_FSIZE, &mut limit) != 0 {
                        return Err(io::Error::last_os_error());
                    }
                    limit.rlim_cur = limit.rlim_cur.min(MAX_CAPTURE_BYTES as libc::rlim_t);
                    if libc::setrlimit(libc::RLIMIT_FSIZE, &limit) != 0 {
                        return Err(io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
        }
        command.spawn()
    }

    fn write(&self, stdin: &mut tempfile::NamedTempFile, bytes: &[u8]) -> io::Result<()> {
        stdin.write_all(bytes)
    }

    fn poll(&self, child: &mut Child) -> io::Result<Option<ExitStatus>> {
        child.try_wait()
    }

    fn kill(&self, child: &mut Child) -> io::Result<()> {
        child.kill()
    }

    fn size(&self, path: &Path) -> io::Result<u64> {
        Ok(std::fs::metadata(path)?.len())
    }

    fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        let mut bytes = Vec::new();
        File::open(path)?
            .take(MAX_CAPTURE_BYTES + 1)
            .read_to_end(&mut bytes)?;
        Ok(bytes)
    }
}

pub(crate) fn run_with_ops(invocation: Invocation<'_>, ops: &impl ProcessOps) -> Outcome {
    let stdout = match ops.capture() {
        Ok(file) => file,
        Err(error) => return Outcome::Degraded(degraded("stdout capture", &error)),
    };
    let stderr = match ops.capture() {
        Ok(file) => file,
        Err(error) => return Outcome::Degraded(degraded("stderr capture", &error)),
    };
    let stdout_writer = match ops.reopen(&stdout) {
        Ok(file) => file,
        Err(error) => return Outcome::Degraded(degraded("stdout capture reopen", &error)),
    };
    let stderr_writer = match ops.reopen(&stderr) {
        Ok(file) => file,
        Err(error) => return Outcome::Degraded(degraded("stderr capture reopen", &error)),
    };

    // A file-backed stdin cannot fill a pipe while Beacon is not reading.
    // Stage it before spawning so a write failure leaves no child to reap.
    let mut input_capture = match ops.capture() {
        Ok(file) => file,
        Err(error) => return Outcome::Degraded(degraded("request capture", &error)),
    };
    let input_bytes = match invocation.input {
        Input::Stdin(bytes) | Input::RequestFile(bytes) => bytes,
    };
    if let Err(error) = ops.write(&mut input_capture, input_bytes) {
        return Outcome::Degraded(degraded("request write", &error));
    }
    let stdin_reader = match invocation.input {
        Input::Stdin(_) => match ops.reopen(&input_capture) {
            Ok(reader) => Some(reader),
            Err(error) => return Outcome::Degraded(degraded("stdin capture reopen", &error)),
        },
        Input::RequestFile(_) => None,
    };

    let mut command = Command::new(invocation.binary);
    command.args(invocation.args);
    if let Input::RequestFile(_) = invocation.input {
        command.arg(input_capture.path());
    }
    command.stdin(stdin_reader.map_or_else(Stdio::null, Stdio::from));
    command.stdout(Stdio::from(stdout_writer));
    command.stderr(Stdio::from(stderr_writer));
    let mut child = match ops.spawn(&mut command) {
        Ok(child) => child,
        Err(error) => return Outcome::Degraded(degraded("spawn", &error)),
    };

    let started = Instant::now();
    let status = loop {
        if let Some(fault) = capture_fault(&stdout, &stderr, ops) {
            let cleanup = terminate(&mut child, ops);
            return Outcome::Degraded(degraded_detail(
                &fault.description(),
                cleanup.err().as_deref(),
            ));
        }
        match ops.poll(&mut child) {
            Ok(Some(status)) => break status,
            Ok(None) if started.elapsed() >= invocation.timeout => {
                return match terminate(&mut child, ops) {
                    Ok(()) => Outcome::TimedOut,
                    Err(reason) => {
                        Outcome::Degraded(degraded_detail("timeout teardown", Some(&reason)))
                    }
                };
            }
            Ok(None) => thread::sleep(
                POLL_INTERVAL.min(invocation.timeout.saturating_sub(started.elapsed())),
            ),
            Err(error) => {
                let cleanup = terminate(&mut child, ops);
                return Outcome::Degraded(degraded_detail(
                    &format!("wait failed ({})", io_identity(&error)),
                    cleanup.err().as_deref(),
                ));
            }
        }
    };

    if let Some(fault) = capture_fault(&stdout, &stderr, ops) {
        return Outcome::Degraded(degraded_detail(&fault.description(), None));
    }

    let stdout_bytes = match ops.read(stdout.path()) {
        Ok(bytes) => bytes,
        Err(error) => return Outcome::Degraded(degraded("stdout read", &error)),
    };
    if stdout_bytes.len() as u64 >= MAX_CAPTURE_BYTES {
        return Outcome::Degraded(degraded_detail("stdout capture limit exceeded", None));
    }
    let stderr_bytes = match ops.read(stderr.path()) {
        Ok(bytes) => bytes,
        Err(error) => return Outcome::Degraded(degraded("stderr read", &error)),
    };
    if stderr_bytes.len() as u64 >= MAX_CAPTURE_BYTES {
        return Outcome::Degraded(degraded_detail("stderr capture limit exceeded", None));
    }
    Outcome::Completed {
        status,
        stdout: stdout_bytes,
        stderr: stderr_bytes,
    }
}

enum CaptureFault {
    Limit(&'static str),
    Metadata(&'static str, io::Error),
}

impl CaptureFault {
    fn description(&self) -> String {
        match self {
            Self::Limit(stream) => format!("{stream} capture limit exceeded"),
            Self::Metadata(stream, error) => {
                format!("{stream} capture metadata failed ({})", io_identity(error))
            }
        }
    }
}

fn capture_fault(
    stdout: &tempfile::NamedTempFile,
    stderr: &tempfile::NamedTempFile,
    ops: &impl ProcessOps,
) -> Option<CaptureFault> {
    for (stream, path) in [("stdout", stdout.path()), ("stderr", stderr.path())] {
        match ops.size(path) {
            Ok(size) if size >= MAX_CAPTURE_BYTES => return Some(CaptureFault::Limit(stream)),
            Ok(_) => {}
            Err(error) => return Some(CaptureFault::Metadata(stream, error)),
        }
    }
    None
}

/// Never call blocking `wait`: a sandbox may deny the wait or kill operation.
/// A failed teardown is a degradation, and the child's handle is dropped.
fn terminate(child: &mut Child, ops: &impl ProcessOps) -> Result<(), String> {
    if let Ok(Some(_)) = ops.poll(child) {
        return Ok(());
    }
    let kill_error = ops.kill(child).err();
    let started = Instant::now();
    loop {
        match ops.poll(child) {
            Ok(Some(_)) => {
                return match kill_error.as_ref() {
                    Some(error) => Err(format!("kill failed ({})", io_identity(error))),
                    None => Ok(()),
                };
            }
            Ok(None) if started.elapsed() < REAP_BUDGET => thread::sleep(POLL_INTERVAL),
            Ok(None) => break,
            Err(error) => {
                return Err(format!("reap failed ({})", io_identity(&error)));
            }
        }
    }
    Err(match kill_error {
        Some(error) => format!("kill failed ({}); child did not exit", io_identity(&error)),
        None => "child did not exit after kill".to_owned(),
    })
}

fn io_identity(error: &io::Error) -> String {
    format!("{:?}, os error {:?}", error.kind(), error.raw_os_error())
}

fn degraded(mechanism: &str, error: &io::Error) -> Unsupported {
    degraded_detail(
        &format!("{mechanism} failed ({})", io_identity(error)),
        None,
    )
}

fn degraded_detail(mechanism: &str, cleanup: Option<&str>) -> Unsupported {
    let mut what = format!("Beacon subprocess {mechanism}");
    if let Some(cleanup) = cleanup {
        what.push_str("; ");
        what.push_str(cleanup);
    }
    Unsupported::new(
        UnsupportedKind::Construct(what),
        "prove external Beacon engine",
        Stage::Runtime,
        chelis_types::unimplemented_rejection!(
            730,
            "Beacon subprocess supervision failed; retry in a supported execution environment"
        ),
    )
}

pub(crate) fn abnormal_exit(status: ExitStatus) -> Unsupported {
    degraded_detail(
        &format!(
            "exited abnormally ({})",
            status
                .code()
                .map_or_else(|| "signal".to_owned(), |code| code.to_string())
        ),
        None,
    )
}

pub(crate) fn timeout_unsupported() -> Unsupported {
    degraded_detail("timed out and was killed", None)
}

pub(crate) fn protocol_unsupported(reason: &str) -> Unsupported {
    degraded_detail(&format!("contract protocol failed ({reason})"), None)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::beacon_contract_prover::BeaconContractProver;
    use crate::beacon_shim::{BeaconShim, SubprocessOutcome, WireDagByteStore};

    #[derive(Clone, Copy, Debug)]
    enum Denied {
        FirstCapture,
        SecondCapture,
        FirstReopen,
        SecondReopen,
        ThirdCapture,
        ThirdReopen,
        Spawn,
        RequestWrite,
        Wait,
        Kill,
        FirstSize,
        SecondSize,
        FirstRead,
        SecondRead,
    }

    struct FaultOps {
        denied: Denied,
        captures: Cell<usize>,
        reopens: Cell<usize>,
        polls: Cell<usize>,
        sizes: Cell<usize>,
        reads: Cell<usize>,
    }

    impl FaultOps {
        fn new(denied: Denied) -> Self {
            Self {
                denied,
                captures: Cell::new(0),
                reopens: Cell::new(0),
                polls: Cell::new(0),
                sizes: Cell::new(0),
                reads: Cell::new(0),
            }
        }

        fn denied() -> io::Error {
            io::Error::from_raw_os_error(13)
        }
    }

    impl ProcessOps for FaultOps {
        fn capture(&self) -> io::Result<tempfile::NamedTempFile> {
            let call = self.captures.get() + 1;
            self.captures.set(call);
            if matches!(
                (self.denied, call),
                (Denied::FirstCapture, 1) | (Denied::SecondCapture, 2) | (Denied::ThirdCapture, 3)
            ) {
                Err(Self::denied())
            } else {
                SystemOps.capture()
            }
        }

        fn reopen(&self, file: &tempfile::NamedTempFile) -> io::Result<File> {
            let call = self.reopens.get() + 1;
            self.reopens.set(call);
            if matches!(
                (self.denied, call),
                (Denied::FirstReopen, 1) | (Denied::SecondReopen, 2) | (Denied::ThirdReopen, 3)
            ) {
                Err(Self::denied())
            } else {
                SystemOps.reopen(file)
            }
        }

        fn spawn(&self, _command: &mut Command) -> io::Result<Child> {
            if matches!(self.denied, Denied::Spawn) {
                return Err(Self::denied());
            }
            // A real child exercises cleanup and wait. Long sleep is used only
            // for faults that require the child to remain alive.
            if matches!(self.denied, Denied::Wait | Denied::Kill) {
                return Command::new("/bin/sleep")
                    .arg("5")
                    .stdin(Stdio::piped())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn();
            }
            Command::new(std::env::current_exe()?)
                .arg("--help")
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
        }

        fn write(&self, stdin: &mut tempfile::NamedTempFile, bytes: &[u8]) -> io::Result<()> {
            if matches!(self.denied, Denied::RequestWrite) {
                Err(Self::denied())
            } else {
                SystemOps.write(stdin, bytes)
            }
        }

        fn poll(&self, child: &mut Child) -> io::Result<Option<ExitStatus>> {
            let call = self.polls.get() + 1;
            self.polls.set(call);
            if matches!(self.denied, Denied::Wait) && call == 1 {
                Err(Self::denied())
            } else {
                SystemOps.poll(child)
            }
        }

        fn kill(&self, child: &mut Child) -> io::Result<()> {
            if matches!(self.denied, Denied::Kill) {
                // Report a denied operation while terminating the real test
                // child; the test must not leave a sleeper behind.
                let _ = SystemOps.kill(child);
                Err(Self::denied())
            } else {
                SystemOps.kill(child)
            }
        }

        fn size(&self, path: &Path) -> io::Result<u64> {
            let call = self.sizes.get() + 1;
            self.sizes.set(call);
            if matches!(
                (self.denied, call),
                (Denied::FirstSize, 1) | (Denied::SecondSize, 2)
            ) {
                Err(Self::denied())
            } else {
                SystemOps.size(path)
            }
        }

        fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
            let call = self.reads.get() + 1;
            self.reads.set(call);
            if matches!(
                (self.denied, call),
                (Denied::FirstRead, 1) | (Denied::SecondRead, 2)
            ) {
                Err(Self::denied())
            } else {
                SystemOps.read(path)
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn lu1_every_denied_mechanism_degrades_on_both_beacon_routes() {
        let cases = [
            Denied::FirstCapture,
            Denied::SecondCapture,
            Denied::FirstReopen,
            Denied::SecondReopen,
            Denied::ThirdCapture,
            Denied::ThirdReopen,
            Denied::Spawn,
            Denied::RequestWrite,
            Denied::Wait,
            Denied::Kill,
            Denied::FirstSize,
            Denied::SecondSize,
            Denied::FirstRead,
            Denied::SecondRead,
        ];
        let binary = std::env::current_exe().expect("test executable");
        let shim = BeaconShim::new(&binary, WireDagByteStore::new());
        for denied in cases {
            let contract = BeaconContractProver::new(&binary).with_timeout_ms(
                if matches!(denied, Denied::Kill) {
                    0
                } else {
                    10_000
                },
            );
            let contract_ops = FaultOps::new(denied);
            let contract_result =
                contract.prove_contract_with_ops("std.normal_cdf.reflection", &contract_ops);
            let contract_error = contract_result.expect_err("denied operation must degrade");
            assert!(
                contract_error.to_string().starts_with("unsupported: "),
                "{denied:?}: {contract_error}"
            );
            assert!(
                contract_error.to_string().contains("chelis#730"),
                "{denied:?}: {contract_error}"
            );

            let shim_ops = FaultOps::new(denied);
            let shim_timeout = if matches!(denied, Denied::Kill) {
                0
            } else {
                5_000
            };
            let shim_result = shim.run_beacon_with_ops(b"{}", shim_timeout, &shim_ops);
            let SubprocessOutcome::Failed { reason, .. } = shim_result else {
                panic!("{denied:?}: expected degraded shim result, got {shim_result:?}");
            };
            assert!(reason.starts_with("unsupported: "), "{denied:?}: {reason}");
            assert!(reason.contains("chelis#730"), "{denied:?}: {reason}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn lu1_request_file_preparation_denials_are_branded() {
        use crate::beacon_shim::RequestTransport;

        let binary = std::env::current_exe().expect("test executable");
        let shim = BeaconShim::new(&binary, WireDagByteStore::new())
            .with_transport(RequestTransport::TempFile);
        for denied in [Denied::ThirdCapture, Denied::RequestWrite] {
            let result = shim.run_beacon_with_ops(b"{}", 5_000, &FaultOps::new(denied));
            let SubprocessOutcome::Failed { reason, .. } = result else {
                panic!("{denied:?}: expected degraded shim result, got {result:?}");
            };
            assert!(reason.starts_with("unsupported: "), "{denied:?}: {reason}");
            assert!(reason.contains("chelis#730"), "{denied:?}: {reason}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn lu1_normal_completion_and_real_timeout_are_distinct() {
        let binary = std::env::current_exe().expect("test executable");
        let args = [std::ffi::OsString::from("--help")];
        let completed = run_with_ops(
            Invocation {
                binary: &binary,
                args: &args,
                input: Input::Stdin(b""),
                timeout: Duration::from_secs(5),
            },
            &SystemOps,
        );
        assert!(matches!(completed, Outcome::Completed { status, .. } if status.success()));

        let args = [std::ffi::OsString::from("5")];
        let unread_input = vec![b'x'; 1024 * 1024];
        let timed_out = run_with_ops(
            Invocation {
                binary: Path::new("/bin/sleep"),
                args: &args,
                input: Input::Stdin(&unread_input),
                timeout: Duration::from_millis(10),
            },
            &SystemOps,
        );
        assert!(matches!(timed_out, Outcome::TimedOut), "{timed_out:?}");
    }
}
