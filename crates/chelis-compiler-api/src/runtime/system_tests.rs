use std::path::Path;

use chelis_unord::UnordMap;

use super::system::{
    EvalProcessOutput, EvalSystem, EvalSystemBoundary, EvalSystemError, EvalSystemOperation,
    EvalSystemPolicy,
};
use super::{HostEvaluationInputs, RuntimeFailureKind, RuntimeTensorValue};

/// A denied operation must never invoke even the first instruction of its
/// adapter: panic makes that invariant observable for every covered builtin.
struct UnreachableAdapter;

impl EvalSystem for UnreachableAdapter {
    fn read_file(&mut self, _: &Path) -> Result<String, EvalSystemError> {
        panic!("denied filesystem operation reached the adapter")
    }
    fn write_file(&mut self, _: &Path, _: &str) -> Result<(), EvalSystemError> {
        panic!("denied filesystem operation reached the adapter")
    }
    fn read_lines_source(&mut self, _: &Path) -> Result<String, EvalSystemError> {
        panic!("denied filesystem operation reached the adapter")
    }
    fn read_bytes(&mut self, _: &Path) -> Result<Vec<u8>, EvalSystemError> {
        panic!("denied filesystem operation reached the adapter")
    }
    fn file_exists(&mut self, _: &Path) -> Result<bool, EvalSystemError> {
        panic!("denied filesystem operation reached the adapter")
    }
    fn list_dir(&mut self, _: &Path) -> Result<Vec<String>, EvalSystemError> {
        panic!("denied filesystem operation reached the adapter")
    }
    fn load_mapped_file_bytes(&mut self, _: &Path) -> Result<Vec<u8>, EvalSystemError> {
        panic!("denied filesystem operation reached the adapter")
    }
    fn run_process(&mut self, _: &str, _: &[String]) -> Result<EvalProcessOutput, EvalSystemError> {
        panic!("denied process operation reached the adapter")
    }
}

#[test]
fn deny_all_refuses_each_operation_before_adapter_execution() {
    let mut system =
        EvalSystemBoundary::with_adapter(EvalSystemPolicy::DENY_ALL, Box::new(UnreachableAdapter));
    let path = Path::new("/no-fs-access");
    let failures = [
        system.read_file(path).unwrap_err(),
        system.write_file(path, "secret").unwrap_err(),
        system.read_lines_source(path).unwrap_err(),
        system.read_bytes(path).unwrap_err(),
        system.file_exists(path).unwrap_err(),
        system.list_dir(path).unwrap_err(),
        system.load_mapped_file_bytes(path).unwrap_err(),
        system
            .run_process("no-process-access", &[])
            .err()
            .expect("denied process"),
    ];
    for (error, operation) in failures.into_iter().zip([
        EvalSystemOperation::ReadFile,
        EvalSystemOperation::WriteFile,
        EvalSystemOperation::ReadLines,
        EvalSystemOperation::ReadBytes,
        EvalSystemOperation::FileExists,
        EvalSystemOperation::ListDir,
        EvalSystemOperation::MmapFile,
        EvalSystemOperation::ProcessRun,
    ]) {
        assert_eq!(
            error.to_string(),
            format!(
                "{operation} refused: {} capability is not permitted",
                operation.capability(),
            )
        );
    }
}

#[test]
fn mixed_capability_policies_allow_only_their_independent_host_effects() {
    let dir = tempfile::tempdir().expect("tempdir");
    let file = dir.path().join("payload.txt");
    std::fs::write(&file, "permitted data").expect("fixture");

    let mut filesystem_only = EvalSystemBoundary::with_policy(EvalSystemPolicy {
        filesystem: true,
        process: false,
    });
    assert_eq!(
        filesystem_only
            .read_file(&file)
            .expect("filesystem allowed"),
        "permitted data"
    );
    assert_eq!(
        filesystem_only
            .run_process("echo", &["not spawned".to_owned()])
            .err()
            .expect("process denied")
            .to_string(),
        "process_run refused: Process capability is not permitted"
    );

    let mut process_only = EvalSystemBoundary::with_policy(EvalSystemPolicy {
        filesystem: false,
        process: true,
    });
    assert_eq!(
        process_only.read_file(&file).unwrap_err().to_string(),
        "read_file refused: Filesystem capability is not permitted"
    );
    let output = process_only
        .run_process("echo", &["safe".to_owned(), "$HOME".to_owned()])
        .expect("direct-argv process allowed");
    assert_eq!(output.exit_status, Some(0));
    assert_eq!(output.stdout, b"safe $HOME\n");
    assert!(output.stderr.is_empty());
}

struct FailingReadAdapter;

impl EvalSystem for FailingReadAdapter {
    fn read_file(&mut self, path: &Path) -> Result<String, EvalSystemError> {
        Err(EvalSystemError::System {
            operation: EvalSystemOperation::ReadFile,
            path_or_program: path.display().to_string(),
            source: std::io::Error::other("injected storage fault"),
        })
    }
    fn write_file(&mut self, _: &Path, _: &str) -> Result<(), EvalSystemError> {
        unreachable!()
    }
    fn read_lines_source(&mut self, _: &Path) -> Result<String, EvalSystemError> {
        unreachable!()
    }
    fn read_bytes(&mut self, _: &Path) -> Result<Vec<u8>, EvalSystemError> {
        unreachable!()
    }
    fn file_exists(&mut self, _: &Path) -> Result<bool, EvalSystemError> {
        unreachable!()
    }
    fn list_dir(&mut self, _: &Path) -> Result<Vec<String>, EvalSystemError> {
        unreachable!()
    }
    fn load_mapped_file_bytes(&mut self, _: &Path) -> Result<Vec<u8>, EvalSystemError> {
        unreachable!()
    }
    fn run_process(&mut self, _: &str, _: &[String]) -> Result<EvalProcessOutput, EvalSystemError> {
        unreachable!()
    }
}

#[test]
fn injected_system_failure_preserves_prior_transcript_and_failure_kind() {
    let source = "before = print(\"prior output\")\nresult = read_file(\"/virtual/input\")\n";
    let decls = chelis_surf::parser::parse_str(source).expect("parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls).expect("desugar");
    let program = chelis_types::check_ir_program(&exprs).expect("checked fixture");
    let roots: UnordMap<String, RuntimeTensorValue> = UnordMap::new();
    let failure = super::evaluate_host_program_with_library_and_types_and_system(
        &program,
        None,
        None,
        HostEvaluationInputs {
            roots: &roots,
            bindings: None,
        },
        None,
        None,
        EvalSystemBoundary::with_adapter(EvalSystemPolicy::ALLOW_ALL, Box::new(FailingReadAdapter)),
    )
    .expect_err("injected filesystem fault must stop evaluation");
    assert_eq!(
        failure.message,
        "read_file failed for `/virtual/input`: injected storage fault"
    );
    assert_eq!(failure.transcript, ["prior output"]);
    assert_eq!(failure.kind, RuntimeFailureKind::Ordinary);
}
