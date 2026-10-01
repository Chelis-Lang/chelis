use std::path::Path;
use std::time::Duration;

use chelis_unord::UnordMap;

use super::system::{
    CLOCK_SECONDS_MAX, CLOCK_SECONDS_MIN, EvalClockReading, EvalClockTime, EvalProcessOutput,
    EvalSystem, EvalSystemBoundary, EvalSystemError, EvalSystemOperation, EvalSystemPolicy,
};
use super::{
    HostEvaluationInputs, RuntimeFailure, RuntimeFailureKind, RuntimeTensorValue, RuntimeValue,
};

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
    fn read_wall_clock(&mut self) -> std::io::Result<EvalClockReading> {
        panic!("denied clock operation reached the adapter")
    }
    fn read_monotonic_clock(&mut self) -> std::io::Result<EvalClockReading> {
        panic!("denied clock operation reached the adapter")
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
        clock: false,
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
        clock: false,
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
    fn read_wall_clock(&mut self) -> std::io::Result<EvalClockReading> {
        unreachable!()
    }
    fn read_monotonic_clock(&mut self) -> std::io::Result<EvalClockReading> {
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

/// [05-OP-75]: a clock adapter that serves one fixed reading per clock, or a
/// host clock error, and refuses every other host operation.
struct FixedClockAdapter {
    wall: Result<EvalClockReading, &'static str>,
    monotonic: Result<EvalClockReading, &'static str>,
}

impl FixedClockAdapter {
    fn both(reading: EvalClockReading) -> Self {
        Self {
            wall: Ok(reading),
            monotonic: Ok(reading),
        }
    }

    fn serve(reading: Result<EvalClockReading, &'static str>) -> std::io::Result<EvalClockReading> {
        reading.map_err(std::io::Error::other)
    }
}

impl EvalSystem for FixedClockAdapter {
    fn read_file(&mut self, _: &Path) -> Result<String, EvalSystemError> {
        unreachable!("the clock fixture performs no filesystem access")
    }
    fn write_file(&mut self, _: &Path, _: &str) -> Result<(), EvalSystemError> {
        unreachable!("the clock fixture performs no filesystem access")
    }
    fn read_lines_source(&mut self, _: &Path) -> Result<String, EvalSystemError> {
        unreachable!("the clock fixture performs no filesystem access")
    }
    fn read_bytes(&mut self, _: &Path) -> Result<Vec<u8>, EvalSystemError> {
        unreachable!("the clock fixture performs no filesystem access")
    }
    fn file_exists(&mut self, _: &Path) -> Result<bool, EvalSystemError> {
        unreachable!("the clock fixture performs no filesystem access")
    }
    fn list_dir(&mut self, _: &Path) -> Result<Vec<String>, EvalSystemError> {
        unreachable!("the clock fixture performs no filesystem access")
    }
    fn load_mapped_file_bytes(&mut self, _: &Path) -> Result<Vec<u8>, EvalSystemError> {
        unreachable!("the clock fixture performs no filesystem access")
    }
    fn run_process(&mut self, _: &str, _: &[String]) -> Result<EvalProcessOutput, EvalSystemError> {
        unreachable!("the clock fixture performs no process access")
    }
    fn read_wall_clock(&mut self) -> std::io::Result<EvalClockReading> {
        Self::serve(self.wall)
    }
    fn read_monotonic_clock(&mut self) -> std::io::Result<EvalClockReading> {
        Self::serve(self.monotonic)
    }
}

fn after(seconds: u64, nanoseconds: u32) -> EvalClockReading {
    EvalClockReading::AtOrAfterOrigin(Duration::new(seconds, nanoseconds))
}

fn before(seconds: u64, nanoseconds: u32) -> EvalClockReading {
    EvalClockReading::BeforeOrigin(Duration::new(seconds, nanoseconds))
}

fn time(seconds: i64, nanoseconds: i64) -> EvalClockTime {
    EvalClockTime {
        seconds,
        nanoseconds,
    }
}

/// Both clock reads through a policy-permitting boundary over `adapter`.
fn read_both(adapter: FixedClockAdapter) -> [Result<EvalClockTime, String>; 2] {
    let mut system =
        EvalSystemBoundary::with_adapter(EvalSystemPolicy::ALLOW_ALL, Box::new(adapter));
    [
        system.clock_wall_read().map_err(|error| error.to_string()),
        system
            .clock_monotonic_read()
            .map_err(|error| error.to_string()),
    ]
}

/// Evaluate `source` with `adapter` as the evaluator's only system port.
fn evaluate_with(
    source: &str,
    adapter: FixedClockAdapter,
) -> Result<UnordMap<String, RuntimeValue>, RuntimeFailure> {
    let decls = chelis_surf::parser::parse_str(source).expect("parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls).expect("desugar");
    let program = chelis_types::check_ir_program(&exprs).expect("checked fixture");
    let roots: UnordMap<String, RuntimeTensorValue> = UnordMap::new();
    super::evaluate_host_program_with_library_and_types_and_system(
        &program,
        None,
        None,
        HostEvaluationInputs {
            roots: &roots,
            bindings: None,
        },
        None,
        None,
        EvalSystemBoundary::with_adapter(EvalSystemPolicy::ALLOW_ALL, Box::new(adapter)),
    )
    .map(|outcome| outcome.host_bindings)
}

fn pair(value: &RuntimeValue) -> (i64, i64) {
    let RuntimeValue::Tuple(items) = value else {
        panic!("a clock read returns a tuple, got {value:?}");
    };
    let halves: Vec<Option<i64>> = items.iter().map(RuntimeValue::as_i64).collect();
    match halves.as_slice() {
        [Some(seconds), Some(nanoseconds)] => (*seconds, *nanoseconds),
        other => panic!("a clock read returns two i64 halves, got {other:?}"),
    }
}

/// [05-OP-75]: an injected reading reaches the program exactly, through the
/// evaluator's system port and not a separate path.
#[test]
fn injected_clock_readings_evaluate_exactly() {
    let bindings = evaluate_with(
        "wall = clock_wall_read()\nmono = clock_monotonic_read()\n",
        FixedClockAdapter {
            wall: Ok(after(1_700_000_000, 123_456_789)),
            monotonic: Ok(after(42, 7)),
        },
    )
    .expect("permitted clock reads evaluate");
    let root = |name: &str| pair(bindings.get(name).expect("clock root is bound"));
    assert_eq!(root("wall"), (1_700_000_000, 123_456_789));
    assert_eq!(root("mono"), (42, 7));
}

/// The failing twin: an injected host clock error stops evaluation with the
/// pinned `<operation>: io: <detail>` message.
#[test]
fn injected_clock_error_fails_with_the_io_message() {
    let failure = evaluate_with(
        "before = print(\"prior output\")\nwall = clock_wall_read()\n",
        FixedClockAdapter {
            wall: Err("injected clock fault"),
            monotonic: Ok(after(0, 0)),
        },
    )
    .expect_err("a host clock error must stop evaluation");
    assert_eq!(
        failure.message,
        "clock_wall_read: io: host clock error: injected clock fault"
    );
    assert_eq!(failure.transcript, ["prior output"]);
    assert_eq!(failure.kind, RuntimeFailureKind::Ordinary);

    let failure = evaluate_with(
        "mono = clock_monotonic_read()\n",
        FixedClockAdapter {
            wall: Ok(after(0, 0)),
            monotonic: Err("injected clock fault"),
        },
    )
    .expect_err("a host clock error must stop evaluation");
    assert_eq!(
        failure.message,
        "clock_monotonic_read: io: host clock error: injected clock fault"
    );
}

/// [05-OP-75] Euclidean form: a reading before the origin keeps a
/// nonnegative remainder, so -1.5 s is `(-2, 500000000)` and a whole
/// negative second keeps a zero remainder.
#[test]
fn readings_before_the_origin_are_euclidean() {
    for (reading, expected) in [
        (before(1, 500_000_000), time(-2, 500_000_000)),
        (before(2, 0), time(-2, 0)),
        (before(0, 1), time(-1, 999_999_999)),
        (before(0, 0), time(0, 0)),
        (after(0, 999_999_999), time(0, 999_999_999)),
    ] {
        for read in read_both(FixedClockAdapter::both(reading)) {
            assert_eq!(read, Ok(expected), "reading {reading:?}");
        }
    }
}

/// The admitted range is inclusive at both ends, the remainder included.
#[test]
fn readings_at_the_range_bounds_are_admitted() {
    let greatest = u64::try_from(CLOCK_SECONDS_MAX).expect("positive bound");
    let least = CLOCK_SECONDS_MIN.unsigned_abs();
    for (reading, expected) in [
        (
            after(greatest, 999_999_999),
            time(CLOCK_SECONDS_MAX, 999_999_999),
        ),
        (before(least, 0), time(CLOCK_SECONDS_MIN, 0)),
    ] {
        for read in read_both(FixedClockAdapter::both(reading)) {
            assert_eq!(read, Ok(expected), "reading {reading:?}");
        }
    }
}

/// The failing twin: one nanosecond past either bound fails `io`, naming
/// the exact reading, never clamping or wrapping it. A host distance beyond
/// i64 is reported exactly too.
#[test]
fn readings_outside_the_range_fail_with_the_io_message() {
    let greatest = u64::try_from(CLOCK_SECONDS_MAX).expect("positive bound");
    let least = CLOCK_SECONDS_MIN.unsigned_abs();
    for (reading, seconds, nanoseconds) in [
        (after(greatest + 1, 0), "253402214401", 0),
        (before(least, 1), "-377705030402", 999_999_999),
        (after(u64::MAX, 0), "18446744073709551615", 0),
        (before(u64::MAX, 1), "-18446744073709551616", 999_999_999),
    ] {
        let [wall, monotonic] = read_both(FixedClockAdapter::both(reading));
        for (operation, read) in [
            ("clock_wall_read", wall),
            ("clock_monotonic_read", monotonic),
        ] {
            assert_eq!(
                read,
                Err(format!(
                    "{operation}: io: host reading (seconds {seconds}, nanoseconds {nanoseconds}) \
                     is outside seconds -377705030401..253402214400"
                )),
                "reading {reading:?}"
            );
        }
    }
}

/// The policy refuses a clock read before its adapter runs, under the clock
/// grammar, and the clock capability is independent of the other two.
#[test]
fn clock_reads_obey_their_own_capability() {
    let mut denied =
        EvalSystemBoundary::with_adapter(EvalSystemPolicy::DENY_ALL, Box::new(UnreachableAdapter));
    assert_eq!(
        denied.clock_wall_read().unwrap_err().to_string(),
        "clock_wall_read: io: Clock capability is not permitted"
    );
    assert_eq!(
        denied.clock_monotonic_read().unwrap_err().to_string(),
        "clock_monotonic_read: io: Clock capability is not permitted"
    );

    let mut all_but_clock = EvalSystemBoundary::with_adapter(
        EvalSystemPolicy {
            filesystem: true,
            process: true,
            clock: false,
        },
        Box::new(UnreachableAdapter),
    );
    assert!(all_but_clock.clock_wall_read().is_err());
    assert!(all_but_clock.clock_monotonic_read().is_err());

    let mut clock_only = EvalSystemBoundary::with_adapter(
        EvalSystemPolicy {
            filesystem: false,
            process: false,
            clock: true,
        },
        Box::new(FixedClockAdapter::both(after(5, 6))),
    );
    assert_eq!(
        clock_only.clock_wall_read().map_err(|e| e.to_string()),
        Ok(time(5, 6))
    );
    assert_eq!(
        clock_only
            .read_file(Path::new("/no-fs-access"))
            .unwrap_err()
            .to_string(),
        "read_file refused: Filesystem capability is not permitted"
    );
}

/// The shipped adapter: a wall reading lies between two host reads taken
/// around it, and successive monotonic readings never decrease.
#[test]
fn default_adapter_reads_the_host_clocks() {
    let mut system = EvalSystemBoundary::permissive();
    let host = |instant: std::time::SystemTime| {
        let since = instant
            .duration_since(std::time::UNIX_EPOCH)
            .expect("the host clock is after 1970");
        (
            i64::try_from(since.as_secs()).expect("fits"),
            i64::from(since.subsec_nanos()),
        )
    };
    let lower = host(std::time::SystemTime::now());
    let wall = system.clock_wall_read().expect("wall clock");
    let upper = host(std::time::SystemTime::now());
    let wall = (wall.seconds, wall.nanoseconds);
    assert!(
        lower <= wall && wall <= upper,
        "{lower:?} <= {wall:?} <= {upper:?}"
    );

    let mut previous = system.clock_monotonic_read().expect("monotonic clock");
    for _ in 0..1000 {
        let next = system.clock_monotonic_read().expect("monotonic clock");
        assert!(
            (previous.seconds, previous.nanoseconds) <= (next.seconds, next.nanoseconds),
            "{previous:?} then {next:?}"
        );
        assert!((0..1_000_000_000).contains(&next.nanoseconds));
        previous = next;
    }
}
