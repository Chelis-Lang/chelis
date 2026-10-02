//! chelis#2863: `Std.Datetime.Clock` under a fixed clock injected through the
//! evaluator's system port. The program links the bundled standard library
//! as `chelis eval --file` does inside a package, so these tests run the
//! shipped module source against exact [05-OP-75] readings.

use std::collections::VecDeque;
use std::path::Path;
use std::time::Duration;

use chelis_unord::UnordMap;

use super::system::{
    EvalClockReading, EvalProcessOutput, EvalSystem, EvalSystemBoundary, EvalSystemError,
    EvalSystemPolicy,
};
use super::{
    HostEvaluationInputs, RuntimeFailure, RuntimeFailureKind, RuntimeTensorValue, RuntimeValue,
};
use crate::pipeline::{PipelineGoal, PipelineOutcome};

/// Serves scripted readings in order and panics on an unscripted read, so a
/// test also pins how many host reads a program performs.
struct ScriptedClockAdapter {
    wall: VecDeque<Result<EvalClockReading, &'static str>>,
    monotonic: VecDeque<Result<EvalClockReading, &'static str>>,
}

impl ScriptedClockAdapter {
    fn new(
        wall: impl IntoIterator<Item = Result<EvalClockReading, &'static str>>,
        monotonic: impl IntoIterator<Item = Result<EvalClockReading, &'static str>>,
    ) -> Self {
        Self {
            wall: wall.into_iter().collect(),
            monotonic: monotonic.into_iter().collect(),
        }
    }

    fn serve(
        queue: &mut VecDeque<Result<EvalClockReading, &'static str>>,
        clock: &str,
    ) -> std::io::Result<EvalClockReading> {
        queue
            .pop_front()
            .unwrap_or_else(|| {
                panic!("the program read the {clock} clock more often than scripted")
            })
            .map_err(std::io::Error::other)
    }
}

impl EvalSystem for ScriptedClockAdapter {
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
        Self::serve(&mut self.wall, "wall")
    }
    fn read_monotonic_clock(&mut self) -> std::io::Result<EvalClockReading> {
        Self::serve(&mut self.monotonic, "monotonic")
    }
}

fn after(seconds: u64, nanoseconds: u32) -> EvalClockReading {
    EvalClockReading::AtOrAfterOrigin(Duration::new(seconds, nanoseconds))
}

fn before(seconds: u64, nanoseconds: u32) -> EvalClockReading {
    EvalClockReading::BeforeOrigin(Duration::new(seconds, nanoseconds))
}

const IMPORTS: &str = "import Std.Datetime (Instant, Duration, instant_unix_second, instant_nanosecond, duration_second, duration_nanosecond)\nimport Std.Datetime.Clock (MonotonicInstant, clock_now, monotonic_now, monotonic_until)\n";

/// Reads one wall instant as `(unix second, nanosecond)`.
const WALL_READING: &str = "def wall_reading() -> (i64, i64) ! { IO } = {\n  now = clock_now()\n  (instant_unix_second(now), instant_nanosecond(now))\n}\n";

/// Reads two monotonic instants and measures them both ways.
const ELAPSED: &str = "def elapsed() -> ((i64, i64), (i64, i64)) ! { IO } = {\n  a = monotonic_now()\n  b = monotonic_now()\n  forward = monotonic_until(a, b)\n  backward = monotonic_until(b, a)\n  ((duration_second(forward), duration_nanosecond(forward)), (duration_second(backward), duration_nanosecond(backward)))\n}\n";

/// Evaluate `body` as the entry module of a package that depends on the
/// bundled standard library, with `adapter` behind `policy` as the
/// evaluator's only system port.
fn evaluate_std_entry(
    body: &str,
    policy: EvalSystemPolicy,
    adapter: ScriptedClockAdapter,
) -> Result<UnordMap<String, RuntimeValue>, RuntimeFailure> {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().join("clockapp");
    std::fs::create_dir_all(root.join("src")).expect("mkdir src");
    std::fs::write(
        root.join("reef.toml"),
        format!(
            "[package]\nname = \"clockapp\"\nversion = \"0.1.0\"\ncompiler = \"={}\"\nmodule_prefix = \"App\"\n\n[dependencies]\nchelis-std = {{ version = \"{}\" }}\n",
            crate::COMPILER_VERSION,
            chelis_std_bundle::BUNDLED_CHELIS_STD_VERSION
        ),
    )
    .expect("write reef.toml");
    let main = root.join("src/main.ch");
    std::fs::write(&main, format!("module App.Main\n{IMPORTS}{body}")).expect("write main.ch");
    let prepared =
        chelis_reef::prepare_program_for_file(&main, &chelis_std_bundle::EMBEDDED_RUNTIME)
            .expect("the fixture package resolves")
            .expect("the entry file is inside the fixture package");
    let _linked = chelis_types::install_linked_program_guard();
    let program = crate::pipeline::prepare_surf_decls(&prepared.decls, None)
        .expect("the linked program prepares");
    let checked = match crate::pipeline::run_prepared(program, PipelineGoal::FullCheck) {
        Ok(PipelineOutcome::Checked(checked)) => checked,
        Ok(_) => unreachable!("a full check returns a checked program"),
        Err(rejection) => panic!("the fixture program checks: {rejection}"),
    };
    let roots: UnordMap<String, RuntimeTensorValue> = UnordMap::new();
    super::evaluate_host_program_with_library_and_types_and_system(
        checked.program(),
        None,
        None,
        HostEvaluationInputs {
            roots: &roots,
            bindings: None,
        },
        None,
        None,
        EvalSystemBoundary::with_adapter(policy, Box::new(adapter)),
    )
    .map(|outcome| outcome.host_bindings)
}

/// The value bound to the entry's top-level `name`, whatever prefix the
/// package linker gives it.
fn bound<'a>(bindings: &'a UnordMap<String, RuntimeValue>, name: &str) -> &'a RuntimeValue {
    let suffix = format!("__{name}");
    let sorted = bindings.to_sorted();
    let matches: Vec<&RuntimeValue> = sorted
        .iter()
        .filter(|(key, _)| key.as_str() == name || key.ends_with(&suffix))
        .map(|(_, value)| *value)
        .collect();
    match matches.as_slice() {
        [value] => value,
        [] => {
            let keys: Vec<&str> = sorted.iter().map(|(key, _)| key.as_str()).collect();
            panic!("no binding for `{name}` among [{}]", keys.join(", "))
        }
        _ => panic!("`{name}` is bound {} times", matches.len()),
    }
}

fn pair(value: &RuntimeValue) -> (i64, i64) {
    let RuntimeValue::Tuple(items) = value else {
        panic!("expected a tuple");
    };
    let halves: Vec<Option<i64>> = items.iter().map(RuntimeValue::as_i64).collect();
    match halves.as_slice() {
        [Some(first), Some(second)] => (*first, *second),
        other => panic!("expected two i64 values, got {} values", other.len()),
    }
}

fn pair_of_pairs(value: &RuntimeValue) -> ((i64, i64), (i64, i64)) {
    let RuntimeValue::Tuple(items) = value else {
        panic!("expected a tuple");
    };
    match &items[..] {
        [first, second] => (pair(first), pair(second)),
        other => panic!("expected two pairs, got {} values", other.len()),
    }
}

/// `clock_now` is the exact `Instant` of the wall reading, Euclidean form and
/// both [05-OP-75] bounds included, one host read per call.
#[test]
fn clock_now_is_the_exact_instant_of_the_injected_reading() {
    let readings = [
        (
            after(1_700_000_000, 123_456_789),
            (1_700_000_000, 123_456_789),
        ),
        (before(1, 500_000_000), (-2, 500_000_000)),
        (
            after(253_402_214_400, 999_999_999),
            (253_402_214_400, 999_999_999),
        ),
        (before(377_705_030_401, 0), (-377_705_030_401, 0)),
    ];
    let mut body = String::from(WALL_READING);
    for index in 0..readings.len() {
        body.push_str(&format!("wall_{index} = wall_reading()\n"));
    }
    let bindings = evaluate_std_entry(
        &body,
        EvalSystemPolicy::ALLOW_ALL,
        ScriptedClockAdapter::new(readings.iter().map(|(reading, _)| Ok(*reading)), []),
    )
    .expect("permitted wall reads evaluate");
    for (index, (_, expected)) in readings.iter().enumerate() {
        assert_eq!(
            pair(bound(&bindings, &format!("wall_{index}"))),
            *expected,
            "reading {index}"
        );
    }
}

/// `monotonic_until(a, b)` is the exact Euclidean `Duration` from `a` to `b`,
/// and the reversed pair is its negation, across the whole admitted range.
#[test]
fn monotonic_until_is_the_exact_duration_between_injected_readings() {
    let cases = [
        (
            after(5, 900_000_000),
            after(7, 100_000_000),
            ((1, 200_000_000), (-2, 800_000_000)),
        ),
        (after(42, 7), after(42, 7), ((0, 0), (0, 0))),
        (
            before(377_705_030_401, 0),
            after(253_402_214_400, 999_999_999),
            ((631_107_244_801, 999_999_999), (-631_107_244_802, 1)),
        ),
    ];
    let mut body = String::from(ELAPSED);
    for index in 0..cases.len() {
        body.push_str(&format!("span_{index} = elapsed()\n"));
    }
    let bindings = evaluate_std_entry(
        &body,
        EvalSystemPolicy::ALLOW_ALL,
        ScriptedClockAdapter::new([], cases.iter().flat_map(|(a, b, _)| [Ok(*a), Ok(*b)])),
    )
    .expect("permitted monotonic reads evaluate");
    for (index, (_, _, expected)) in cases.iter().enumerate() {
        assert_eq!(
            pair_of_pairs(bound(&bindings, &format!("span_{index}"))),
            *expected,
            "case {index}"
        );
    }
}

/// The failing twin of both: a read the host cannot supply, a reading
/// outside the range, or a refused read fails under the builtin's name,
/// `clock_wall_read: io: ...` or `clock_monotonic_read: io: ...`, because the
/// library cannot relabel a builtin's failure.
#[test]
fn clock_failures_name_the_builtin() {
    let wall = "stamp = wall_reading()\n";
    let monotonic = "span = elapsed()\n";
    let cases: [(&str, &str, EvalSystemPolicy, ScriptedClockAdapter, &str); 5] = [
        (
            WALL_READING,
            wall,
            EvalSystemPolicy::ALLOW_ALL,
            ScriptedClockAdapter::new([Err("injected clock fault")], []),
            "clock_wall_read: io: host clock error: injected clock fault",
        ),
        (
            WALL_READING,
            wall,
            EvalSystemPolicy::ALLOW_ALL,
            ScriptedClockAdapter::new([Ok(after(253_402_214_401, 0))], []),
            "clock_wall_read: io: host reading (seconds 253402214401, nanoseconds 0) is outside \
             seconds -377705030401..253402214400",
        ),
        (
            WALL_READING,
            wall,
            EvalSystemPolicy::DENY_ALL,
            ScriptedClockAdapter::new([], []),
            "clock_wall_read: io: Clock capability is not permitted",
        ),
        (
            ELAPSED,
            monotonic,
            EvalSystemPolicy::ALLOW_ALL,
            ScriptedClockAdapter::new([], [Ok(after(1, 0)), Err("injected clock fault")]),
            "clock_monotonic_read: io: host clock error: injected clock fault",
        ),
        (
            ELAPSED,
            monotonic,
            EvalSystemPolicy::DENY_ALL,
            ScriptedClockAdapter::new([], []),
            "clock_monotonic_read: io: Clock capability is not permitted",
        ),
    ];
    for (definition, root, policy, adapter, expected) in cases {
        let failure = evaluate_std_entry(&format!("{definition}{root}"), policy, adapter)
            .expect_err("a failed clock read stops evaluation");
        assert_eq!(failure.message, expected);
        assert_eq!(failure.kind, RuntimeFailureKind::Ordinary, "{expected}");
    }
}
