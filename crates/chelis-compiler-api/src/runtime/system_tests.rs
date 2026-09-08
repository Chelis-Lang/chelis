//! Tests for the evaluator system boundary (OpenSpec `add-eval-system-boundary`).
//!
//! These are the "red tests and fixtures" (tasks.md §1): fake-adapter tests
//! for all eight covered builtins, refusal tests that prove zero adapter
//! calls, and default-adapter parity tests (success values and the seven
//! exact error templates) against the real operating system.
//!
//! Source-guard fixtures live in `scripts/test_eval_system_guard.py`, not
//! here -- the guard is a Python source scan, not Rust.

use chelis_unord::UnordMap;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use super::system::{
    EvalSystem, EvalSystemBoundary, EvalSystemCapability, EvalSystemError, EvalSystemIoOperation,
    EvalSystemPolicy,
};
use super::system_adapter::DefaultEvalSystem;

// ── Fake adapter ────────────────────────────────────────────────────────

/// One operation call recorded by [`FakeEvalSystem`], with its exact path
/// or argument vector (task 1.2).
#[derive(Debug, Clone, PartialEq)]
enum RecordedCall {
    ReadFile(PathBuf),
    WriteFile(PathBuf, String),
    ReadLinesSource(PathBuf),
    ReadBytes(PathBuf),
    FileExists(PathBuf),
    ListDir(PathBuf),
    LoadMappedFileBytes(PathBuf),
    RunProcess(String, Vec<String>),
}

/// A deterministic fake [`EvalSystem`] adapter with no real filesystem or
/// process access. Every method records its call into the shared `calls`
/// log, then returns the pre-configured result for that operation. Calling
/// an operation whose result was never configured is a test-authoring bug
/// (not a legitimate "no value" case): it panics naming the operation, so a
/// refusal test that unexpectedly reaches the adapter fails loudly instead
/// of returning a fabricated default.
struct FakeEvalSystem {
    calls: Rc<RefCell<Vec<RecordedCall>>>,
    read_file: Option<Result<String, EvalSystemError>>,
    write_file: Option<Result<(), EvalSystemError>>,
    read_lines_source: Option<Result<String, EvalSystemError>>,
    read_bytes: Option<Result<Vec<u8>, EvalSystemError>>,
    file_exists: Option<Result<bool, EvalSystemError>>,
    list_dir: Option<Result<Vec<String>, EvalSystemError>>,
    load_mapped_file_bytes: Option<Result<Vec<u8>, EvalSystemError>>,
    run_process: Option<Result<super::system::EvalProcessOutput, EvalSystemError>>,
}

impl FakeEvalSystem {
    fn new(calls: Rc<RefCell<Vec<RecordedCall>>>) -> Self {
        Self {
            calls,
            read_file: None,
            write_file: None,
            read_lines_source: None,
            read_bytes: None,
            file_exists: None,
            list_dir: None,
            load_mapped_file_bytes: None,
            run_process: None,
        }
    }
}

impl EvalSystem for FakeEvalSystem {
    fn read_file(&mut self, path: &Path) -> Result<String, EvalSystemError> {
        self.calls
            .borrow_mut()
            .push(RecordedCall::ReadFile(path.to_path_buf()));
        self.read_file
            .take()
            .expect("test bug: read_file result not configured")
    }

    fn write_file(&mut self, path: &Path, contents: &str) -> Result<(), EvalSystemError> {
        self.calls.borrow_mut().push(RecordedCall::WriteFile(
            path.to_path_buf(),
            contents.to_string(),
        ));
        self.write_file
            .take()
            .expect("test bug: write_file result not configured")
    }

    fn read_lines_source(&mut self, path: &Path) -> Result<String, EvalSystemError> {
        self.calls
            .borrow_mut()
            .push(RecordedCall::ReadLinesSource(path.to_path_buf()));
        self.read_lines_source
            .take()
            .expect("test bug: read_lines_source result not configured")
    }

    fn read_bytes(&mut self, path: &Path) -> Result<Vec<u8>, EvalSystemError> {
        self.calls
            .borrow_mut()
            .push(RecordedCall::ReadBytes(path.to_path_buf()));
        self.read_bytes
            .take()
            .expect("test bug: read_bytes result not configured")
    }

    fn file_exists(&mut self, path: &Path) -> Result<bool, EvalSystemError> {
        self.calls
            .borrow_mut()
            .push(RecordedCall::FileExists(path.to_path_buf()));
        self.file_exists
            .take()
            .expect("test bug: file_exists result not configured")
    }

    fn list_dir(&mut self, path: &Path) -> Result<Vec<String>, EvalSystemError> {
        self.calls
            .borrow_mut()
            .push(RecordedCall::ListDir(path.to_path_buf()));
        self.list_dir
            .take()
            .expect("test bug: list_dir result not configured")
    }

    fn load_mapped_file_bytes(&mut self, path: &Path) -> Result<Vec<u8>, EvalSystemError> {
        self.calls
            .borrow_mut()
            .push(RecordedCall::LoadMappedFileBytes(path.to_path_buf()));
        self.load_mapped_file_bytes
            .take()
            .expect("test bug: load_mapped_file_bytes result not configured")
    }

    fn run_process(
        &mut self,
        program: &str,
        args: &[String],
    ) -> Result<super::system::EvalProcessOutput, EvalSystemError> {
        self.calls
            .borrow_mut()
            .push(RecordedCall::RunProcess(program.to_string(), args.to_vec()));
        self.run_process
            .take()
            .expect("test bug: run_process result not configured")
    }
}

fn checked_surf(source: &str) -> chelis_types::CheckedProgram {
    let declarations = chelis_surf::parser::parse_str(source).expect("surf parse");
    let expressions = chelis_surf::desugar::desugar_program(&declarations);
    chelis_types::check_ir_program(&expressions).expect("IR check")
}

#[test]
fn evaluator_dispatch_routes_all_eight_operations_exactly_once() {
    let calls = Rc::new(RefCell::new(Vec::new()));
    let mut fake = FakeEvalSystem::new(calls.clone());
    fake.read_file = Some(Ok("text".to_string()));
    fake.write_file = Some(Ok(()));
    fake.read_lines_source = Some(Ok("one\r\ntwo\n".to_string()));
    fake.read_bytes = Some(Ok(vec![0, 255]));
    fake.file_exists = Some(Ok(true));
    fake.list_dir = Some(Ok(vec!["b.txt".to_string(), "a.txt".to_string()]));
    fake.load_mapped_file_bytes = Some(Ok(vec![9, 8]));
    fake.run_process = Some(Ok(super::system::EvalProcessOutput {
        exit_status: None,
        stdout: vec![b'x', 0x80],
        stderr: b"err".to_vec(),
    }));
    let mut boundary =
        EvalSystemBoundary::with_adapter(EvalSystemPolicy::permissive(), Box::new(fake));
    let checked = checked_surf(
        r#"
text = read_file("/virtual/read.txt")
written = write_file("/virtual/write.txt", "payload")
lines = read_lines("/virtual/lines.txt")
bytes = read_bytes("/virtual/bytes.bin")
exists = file_exists("/virtual/present")
names = list_dir("/virtual/dir")
mapped = mmap_file("/virtual/mapped.bin")
process = process_run("virtual-program", ["--flag", "value"])
"#,
    );

    let outcome = super::evaluate_host_program_with_library_and_types_and_system(
        &checked,
        &[],
        &BTreeMap::new(),
        None,
        &UnordMap::new(),
        None,
        None,
        &mut boundary,
    )
    .expect("fake-backed evaluation succeeds");

    assert_eq!(
        calls.borrow().as_slice(),
        &[
            RecordedCall::ReadFile(PathBuf::from("/virtual/read.txt")),
            RecordedCall::WriteFile(PathBuf::from("/virtual/write.txt"), "payload".to_string(),),
            RecordedCall::ReadLinesSource(PathBuf::from("/virtual/lines.txt")),
            RecordedCall::ReadBytes(PathBuf::from("/virtual/bytes.bin")),
            RecordedCall::FileExists(PathBuf::from("/virtual/present")),
            RecordedCall::ListDir(PathBuf::from("/virtual/dir")),
            RecordedCall::LoadMappedFileBytes(PathBuf::from("/virtual/mapped.bin")),
            RecordedCall::RunProcess(
                "virtual-program".to_string(),
                vec!["--flag".to_string(), "value".to_string()],
            ),
        ]
    );
    let Some(super::RuntimeValue::List(lines)) = outcome.host_bindings.get("lines") else {
        panic!("expected transformed line list");
    };
    assert_eq!(lines.len(), 2);
    assert!(matches!(&lines[0], super::RuntimeValue::String(value) if value == "one"));
    assert!(matches!(&lines[1], super::RuntimeValue::String(value) if value == "two"));

    let Some(super::RuntimeValue::Tuple(process)) = outcome.host_bindings.get("process") else {
        panic!("expected transformed process tuple");
    };
    assert_eq!(process.len(), 3);
    assert_eq!(process[0].as_i64(), Some(-1));
    assert!(matches!(&process[1], super::RuntimeValue::String(value) if value == "x�"));
    assert!(matches!(&process[2], super::RuntimeValue::String(value) if value == "err"));
}

// ── Positive boundary-adapter tests: one op each ──────────────────────────

macro_rules! fake_test {
    ($name:ident, $set:ident = $configured:expr, $call:expr, $expect_recorded:expr, $expect_value:expr $(,)?) => {
        #[test]
        fn $name() {
            let calls = Rc::new(RefCell::new(Vec::new()));
            let mut fake = FakeEvalSystem::new(calls.clone());
            fake.$set = Some($configured);
            let mut boundary =
                EvalSystemBoundary::with_adapter(EvalSystemPolicy::permissive(), Box::new(fake));
            let result = $call(&mut boundary);
            assert_eq!(result, $expect_value);
            assert_eq!(
                calls.borrow().as_slice(),
                &[$expect_recorded],
                "exactly one operation must be recorded"
            );
        }
    };
}

fake_test!(
    fake_adapter_read_file_records_one_call,
    read_file = Ok("hello world".to_string()),
    |boundary: &mut EvalSystemBoundary| boundary
        .read_file(Path::new("/tmp/probe.txt"))
        .expect("configured Ok"),
    RecordedCall::ReadFile(PathBuf::from("/tmp/probe.txt")),
    "hello world".to_string(),
);

fake_test!(
    fake_adapter_write_file_records_one_call,
    write_file = Ok(()),
    |boundary: &mut EvalSystemBoundary| boundary
        .write_file(Path::new("/tmp/probe.txt"), "payload")
        .expect("configured Ok"),
    RecordedCall::WriteFile(PathBuf::from("/tmp/probe.txt"), "payload".to_string()),
    (),
);

fake_test!(
    fake_adapter_read_lines_source_records_one_call,
    read_lines_source = Ok("a\nb\n".to_string()),
    |boundary: &mut EvalSystemBoundary| boundary
        .read_lines_source(Path::new("/tmp/lines.txt"))
        .expect("configured Ok"),
    RecordedCall::ReadLinesSource(PathBuf::from("/tmp/lines.txt")),
    "a\nb\n".to_string(),
);

fake_test!(
    fake_adapter_read_bytes_records_one_call,
    read_bytes = Ok(vec![1, 2, 3]),
    |boundary: &mut EvalSystemBoundary| boundary
        .read_bytes(Path::new("/tmp/bytes.bin"))
        .expect("configured Ok"),
    RecordedCall::ReadBytes(PathBuf::from("/tmp/bytes.bin")),
    vec![1_u8, 2, 3],
);

fake_test!(
    fake_adapter_file_exists_records_one_call,
    file_exists = Ok(true),
    |boundary: &mut EvalSystemBoundary| boundary
        .file_exists(Path::new("/tmp/maybe.txt"))
        .expect("configured Ok"),
    RecordedCall::FileExists(PathBuf::from("/tmp/maybe.txt")),
    true,
);

fake_test!(
    fake_adapter_list_dir_records_one_call,
    list_dir = Ok(vec!["b.txt".to_string(), "a.txt".to_string()]),
    |boundary: &mut EvalSystemBoundary| boundary
        .list_dir(Path::new("/tmp/dir"))
        .expect("configured Ok"),
    RecordedCall::ListDir(PathBuf::from("/tmp/dir")),
    vec!["b.txt".to_string(), "a.txt".to_string()],
);

fake_test!(
    fake_adapter_load_mapped_file_bytes_records_one_call,
    load_mapped_file_bytes = Ok(vec![9, 8, 7]),
    |boundary: &mut EvalSystemBoundary| boundary
        .load_mapped_file_bytes(Path::new("/tmp/mapped.bin"))
        .expect("configured Ok"),
    RecordedCall::LoadMappedFileBytes(PathBuf::from("/tmp/mapped.bin")),
    vec![9_u8, 8, 7],
);

#[test]
fn fake_adapter_run_process_records_one_call_with_argv() {
    let calls = Rc::new(RefCell::new(Vec::new()));
    let mut fake = FakeEvalSystem::new(calls.clone());
    fake.run_process = Some(Ok(super::system::EvalProcessOutput {
        exit_status: Some(0),
        stdout: b"hi".to_vec(),
        stderr: Vec::new(),
    }));
    let mut boundary =
        EvalSystemBoundary::with_adapter(EvalSystemPolicy::permissive(), Box::new(fake));
    let argv = vec!["hi".to_string()];
    let output = boundary.run_process("echo", &argv).expect("configured Ok");
    assert_eq!(output.exit_status, Some(0));
    assert_eq!(output.stdout, b"hi");
    assert_eq!(
        calls.borrow().as_slice(),
        &[RecordedCall::RunProcess(
            "echo".to_string(),
            vec!["hi".to_string()]
        )]
    );
}

// ── Refusal tests (task 1.3): zero adapter calls ───────────────────────────

#[test]
fn deny_all_refuses_filesystem_operation_with_zero_adapter_calls() {
    let calls = Rc::new(RefCell::new(Vec::new()));
    // Deliberately unconfigured: reaching the adapter would panic, proving
    // the refusal short-circuits before any adapter method runs.
    let fake = FakeEvalSystem::new(calls.clone());
    let mut boundary =
        EvalSystemBoundary::with_adapter(EvalSystemPolicy::deny_all(), Box::new(fake));
    let err = boundary
        .read_file(Path::new("/tmp/should-not-be-read"))
        .expect_err("Filesystem capability must be refused");
    assert!(matches!(
        err,
        EvalSystemError::Refused {
            capability: EvalSystemCapability::Filesystem,
            ..
        }
    ));
    assert!(
        calls.borrow().is_empty(),
        "a refused operation must not reach the adapter"
    );
}

#[test]
fn deny_all_refuses_process_operation_with_zero_adapter_calls() {
    let calls = Rc::new(RefCell::new(Vec::new()));
    let fake = FakeEvalSystem::new(calls.clone());
    let mut boundary =
        EvalSystemBoundary::with_adapter(EvalSystemPolicy::deny_all(), Box::new(fake));
    let err = boundary
        .run_process("echo", &["hi".to_string()])
        .expect_err("Process capability must be refused");
    assert!(matches!(
        err,
        EvalSystemError::Refused {
            capability: EvalSystemCapability::Process,
            ..
        }
    ));
    assert!(
        calls.borrow().is_empty(),
        "a refused operation must not reach the adapter"
    );
}

#[test]
fn refused_operation_names_the_operation_and_capability() {
    // Spec requirement: "A refused operation SHALL return an error that
    // names the operation and capability."
    let calls = Rc::new(RefCell::new(Vec::new()));
    let fake = FakeEvalSystem::new(calls);
    let mut boundary =
        EvalSystemBoundary::with_adapter(EvalSystemPolicy::deny_all(), Box::new(fake));
    let err = boundary
        .read_file(Path::new("/tmp/x"))
        .expect_err("refused");
    let message = err.to_string();
    assert!(message.contains("read_file"), "got: {message}");
    assert!(message.contains("Filesystem"), "got: {message}");
}

// ── Default-adapter parity tests (task 1.5, 1.6): the real operating
//    system, matching the pre-boundary evaluator behavior exactly ─────────

#[test]
fn default_adapter_read_file_success_matches_source_bytes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("probe.txt");
    std::fs::write(&path, "hello parity").expect("seed file");
    let mut adapter = DefaultEvalSystem;
    let text = adapter.read_file(&path).expect("read succeeds");
    assert_eq!(text, "hello parity");
}

#[test]
fn default_adapter_write_file_success_creates_or_truncates() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("out.txt");
    std::fs::write(&path, "stale contents that must be truncated").expect("seed file");
    let mut adapter = DefaultEvalSystem;
    adapter.write_file(&path, "fresh").expect("write succeeds");
    assert_eq!(std::fs::read_to_string(&path).expect("read back"), "fresh");
}

#[test]
fn default_adapter_read_lines_source_returns_raw_text() {
    // The pure line-splitting transformation lives in `runtime/eval.rs`
    // (design D2), so the adapter itself hands back the raw source text.
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("lines.txt");
    std::fs::write(&path, "one\ntwo\r\nthree\n").expect("seed file");
    let mut adapter = DefaultEvalSystem;
    let text = adapter.read_lines_source(&path).expect("read succeeds");
    assert_eq!(text, "one\ntwo\r\nthree\n");
}

#[test]
fn default_adapter_read_bytes_matches_source_bytes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("bytes.bin");
    std::fs::write(&path, [0_u8, 1, 255, 42]).expect("seed file");
    let mut adapter = DefaultEvalSystem;
    let bytes = adapter.read_bytes(&path).expect("read succeeds");
    assert_eq!(bytes, vec![0_u8, 1, 255, 42]);
}

#[test]
fn default_adapter_file_exists_true_for_present_path() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("present.txt");
    std::fs::write(&path, "x").expect("seed file");
    let mut adapter = DefaultEvalSystem;
    assert!(adapter.file_exists(&path).expect("no error"));
}

#[test]
fn default_adapter_file_exists_false_for_absent_path() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("absent.txt");
    let mut adapter = DefaultEvalSystem;
    assert!(!adapter.file_exists(&path).expect("no error"));
}

#[test]
fn default_adapter_load_mapped_file_bytes_matches_source_bytes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("mapped.bin");
    std::fs::write(&path, [7_u8, 6, 5]).expect("seed file");
    let mut adapter = DefaultEvalSystem;
    let bytes = adapter
        .load_mapped_file_bytes(&path)
        .expect("read succeeds");
    assert_eq!(bytes, vec![7_u8, 6, 5]);
}

#[test]
fn default_adapter_list_dir_orders_by_host_name_bytes() {
    let dir = tempfile::tempdir().expect("tempdir");
    for name in ["zeta.txt", "alpha.txt", "mid.txt"] {
        std::fs::write(dir.path().join(name), "x").expect("seed file");
    }
    // [05-HOST-4]: the listing is ordered by the host's own name bytes,
    // independent of `std::fs::read_dir`'s arrival order. The adapter is the
    // only module that reads the directory, so the ordering is applied here
    // rather than in `runtime/eval.rs`.
    let mut adapter = DefaultEvalSystem;
    let actual = adapter.list_dir(dir.path()).expect("list_dir succeeds");
    assert_eq!(actual, vec!["alpha.txt", "mid.txt", "zeta.txt"]);
}

#[test]
fn default_adapter_list_dir_orders_before_the_lossy_conversion() {
    // [05-HOST-4] orders on the host bytes, before `to_string_lossy` can
    // collapse distinct names onto U+FFFD. Sorting after that conversion
    // would leave such names tie-broken by directory order.
    let dir = tempfile::tempdir().expect("tempdir");
    for name in ["b.txt", "a.txt"] {
        std::fs::write(dir.path().join(name), "x").expect("seed file");
    }
    let mut adapter = DefaultEvalSystem;
    let listed = adapter.list_dir(dir.path()).expect("list_dir succeeds");
    let mut sorted = listed.clone();
    sorted.sort();
    assert_eq!(listed, sorted, "the listing is already ordered");
}

#[test]
fn default_adapter_process_run_exit_code_and_output_parity() {
    let mut adapter = DefaultEvalSystem;
    let output = adapter
        .run_process("echo", &["hi".to_string()])
        .expect("echo spawns");
    assert_eq!(output.exit_status, Some(0));
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "hi");
}

// ── The seven exact system-error templates (task 1.6) ─────────────────────

#[test]
fn system_error_templates_preserve_the_complete_source_display() {
    let cases = [
        (
            EvalSystemIoOperation::ReadFile,
            "/virtual/read.txt",
            "read_file failed for `/virtual/read.txt`: sentinel source",
        ),
        (
            EvalSystemIoOperation::WriteFile,
            "/virtual/write.txt",
            "write_file failed for `/virtual/write.txt`: sentinel source",
        ),
        (
            EvalSystemIoOperation::ReadLines,
            "/virtual/lines.txt",
            "read_lines failed for `/virtual/lines.txt`: sentinel source",
        ),
        (
            EvalSystemIoOperation::ReadBytes,
            "/virtual/bytes.bin",
            "read_bytes failed for `/virtual/bytes.bin`: sentinel source",
        ),
        (
            EvalSystemIoOperation::ListDir,
            "/virtual/dir",
            "list_dir failed for `/virtual/dir`: sentinel source",
        ),
        (
            EvalSystemIoOperation::MmapFile,
            "/virtual/mapped.bin",
            "mmap_file failed for `/virtual/mapped.bin`: sentinel source",
        ),
        (
            EvalSystemIoOperation::ProcessRun,
            "virtual-program",
            "process_run failed to spawn `virtual-program`: sentinel source",
        ),
    ];

    for (operation, subject, expected) in cases {
        let error = EvalSystemError::System {
            operation,
            path_or_program: subject.to_string(),
            source: std::io::Error::other("sentinel source"),
        };
        assert_eq!(error.to_string(), expected);
    }
}
//
// `file_exists` is deliberately absent: the default adapter contract folds
// its access error into `Ok(false)` (design D4), so it never produces a
// System error / template.

#[test]
fn default_adapter_read_file_error_template_is_exact() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("missing.txt");
    let mut adapter = DefaultEvalSystem;
    let err = adapter.read_file(&path).expect_err("missing file errors");
    let expected_prefix = format!("read_file failed for `{}`: ", path.display());
    assert!(err.to_string().starts_with(&expected_prefix), "got: {err}");
}

#[test]
fn default_adapter_write_file_error_template_is_exact() {
    let dir = tempfile::tempdir().expect("tempdir");
    // A parent directory that does not exist makes the write fail.
    let path = dir.path().join("missing-parent").join("out.txt");
    let mut adapter = DefaultEvalSystem;
    let err = adapter
        .write_file(&path, "x")
        .expect_err("missing parent directory errors");
    let expected_prefix = format!("write_file failed for `{}`: ", path.display());
    assert!(err.to_string().starts_with(&expected_prefix), "got: {err}");
}

#[test]
fn default_adapter_read_lines_error_template_is_exact() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("missing.txt");
    let mut adapter = DefaultEvalSystem;
    let err = adapter
        .read_lines_source(&path)
        .expect_err("missing file errors");
    let expected_prefix = format!("read_lines failed for `{}`: ", path.display());
    assert!(err.to_string().starts_with(&expected_prefix), "got: {err}");
}

#[test]
fn default_adapter_read_bytes_error_template_is_exact() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("missing.bin");
    let mut adapter = DefaultEvalSystem;
    let err = adapter.read_bytes(&path).expect_err("missing file errors");
    let expected_prefix = format!("read_bytes failed for `{}`: ", path.display());
    assert!(err.to_string().starts_with(&expected_prefix), "got: {err}");
}

#[test]
fn default_adapter_list_dir_open_error_template_is_exact() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("missing-dir");
    let mut adapter = DefaultEvalSystem;
    let err = adapter
        .list_dir(&path)
        .expect_err("missing directory errors");
    let expected_prefix = format!("list_dir failed for `{}`: ", path.display());
    assert!(err.to_string().starts_with(&expected_prefix), "got: {err}");
}

#[test]
fn default_adapter_mmap_file_error_template_is_exact() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("missing.bin");
    let mut adapter = DefaultEvalSystem;
    let err = adapter
        .load_mapped_file_bytes(&path)
        .expect_err("missing file errors");
    let expected_prefix = format!("mmap_file failed for `{}`: ", path.display());
    assert!(err.to_string().starts_with(&expected_prefix), "got: {err}");
}

#[test]
fn default_adapter_process_run_spawn_error_template_is_exact() {
    let mut adapter = DefaultEvalSystem;
    let program = "chelis-eval-system-boundary-definitely-not-a-real-binary";
    let err = adapter
        .run_process(program, &[])
        .expect_err("nonexistent binary errors");
    let expected_prefix = format!("process_run failed to spawn `{program}`: ");
    assert!(err.to_string().starts_with(&expected_prefix), "got: {err}");
}

// ── list_dir directory-open vs. mid-iteration entry-error template parity
//    (task 1.7) ─────────────────────────────────────────────────────────
//
// A genuine per-entry `std::fs::ReadDir` iteration failure is a real-OS
// race (the directory changes between opening the handle and reading an
// entry) with no portable, deterministic reproduction. `default_adapter_
// list_dir_open_error_template_is_exact` above covers the directory-open
// half against the real OS. This test pins the CONTRACT half of task 1.7 --
// "both error paths SHALL use the same template" -- by constructing the
// exact `EvalSystemError::System` value the adapter's directory-open branch
// and its per-entry branch each build (same operation, same `path_or_program`
// shape, same `io::Error` shape) and asserting their `Display` output is
// template-identical.
#[test]
fn list_dir_open_and_entry_errors_render_the_same_template() {
    use std::io;

    let path = PathBuf::from("/tmp/probe-dir");
    let open_failure = EvalSystemError::System {
        operation: super::system::EvalSystemIoOperation::ListDir,
        path_or_program: path.display().to_string(),
        source: io::Error::new(io::ErrorKind::NotFound, "directory open failed"),
    };
    let entry_failure = EvalSystemError::System {
        operation: super::system::EvalSystemIoOperation::ListDir,
        path_or_program: path.display().to_string(),
        source: io::Error::other("entry read failed"),
    };
    let open_message = open_failure.to_string();
    let entry_message = entry_failure.to_string();
    let expected_prefix = format!("list_dir failed for `{}`: ", path.display());
    assert!(open_message.starts_with(&expected_prefix), "{open_message}");
    assert!(
        entry_message.starts_with(&expected_prefix),
        "{entry_message}"
    );
}
