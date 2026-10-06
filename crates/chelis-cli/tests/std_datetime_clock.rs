//! chelis#2863: the CLI surface of `Std.Datetime.Clock` under [05-OP-73].
//!
//! `clock_now` and `monotonic_now` are the only reads of the host in the
//! `Std.Datetime` family. Each carries `IO`, which `chelis check` holds every
//! caller to. `MonotonicInstant` is opaque, and `monotonic_until` is the only
//! datetime callable that takes one. Compiled C runs the underlying [05-OP-75]
//! reads through the same runtime definition as the evaluator (chelis#1297).
//! Exact readings from a fixed clock are pinned by the evaluator's
//! injected-clock tests in `chelis-compiler-api`; these run the real clocks.
//! Every accepted program has a rejected twin.

#[path = "common/mod.rs"]
mod common;

use assert_cmd::Command;
use common::{make_app, write_file};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

const IMPORTS: &str = "import Std.Datetime (Instant, Duration, instant_unix_second, instant_nanosecond, instant_until, duration_second, duration_nanosecond)\nimport Std.Datetime.Clock (MonotonicInstant, clock_now, monotonic_now, monotonic_until)\n";

fn run_chelis(app_pkg: &Path, reef_home: &Path, args: &[&str]) -> (bool, String, String) {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(app_pkg)
        .args(args)
        .output()
        .expect("chelis runs");
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

/// Writes `body` after the clock imports as the app's `src/main.ch` and runs
/// `chelis <command> src/main.ch <extra>` in the app package; `eval` reads
/// the file through `--file`.
fn run_main(dir_name: &str, command: &str, body: &str, extra: &[&str]) -> (bool, String, String) {
    let (_dir, reef_home, app_pkg) = make_app(dir_name);
    let path = app_pkg.join("src/main.ch");
    write_file(&path, &format!("module Demo.Main\n{IMPORTS}{body}"));
    let mut args = vec![command];
    if command == "eval" {
        args.push("--file");
    }
    args.push(path.to_str().unwrap());
    args.extend_from_slice(extra);
    run_chelis(&app_pkg, &reef_home, &args)
}

/// The i64 printed as `key = <value>` in `stdout`.
fn printed(stdout: &str, key: &str) -> i64 {
    let prefix = format!("{key} = ");
    let line = stdout
        .lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .unwrap_or_else(|| panic!("no `{prefix}` line in stdout={stdout}"));
    line.trim()
        .parse::<i64>()
        .unwrap_or_else(|error| panic!("`{prefix}{line}` is not an i64: {error}"))
}

fn host_now() -> (i64, i64) {
    let since = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("the host clock is after 1970");
    (
        i64::try_from(since.as_secs()).expect("fits"),
        i64::from(since.subsec_nanos()),
    )
}

/// `clock_now` reads the host wall clock: its instant lies between two host
/// reads taken around the command.
#[test]
fn clock_now_lies_between_host_reads() {
    let lower = host_now();
    let (success, stdout, stderr) = run_main(
        "clock-now-2863",
        "eval",
        "def reading() -> (i64, i64) ! { IO } = {\n  now = clock_now()\n  (instant_unix_second(now), instant_nanosecond(now))\n}\nwall = reading()\n",
        &[],
    );
    let upper = host_now();
    assert!(success, "eval failed: stdout={stdout} stderr={stderr}");
    let reading = (printed(&stdout, "wall.0"), printed(&stdout, "wall.1"));
    assert!(
        lower <= reading && reading <= upper,
        "{lower:?} <= {reading:?} <= {upper:?}"
    );
}

/// `monotonic_until(a, b)` of two successive readings is never negative, and
/// the reversed pair is its exact negation, so it is never positive.
#[test]
fn monotonic_until_orders_successive_readings() {
    let (success, stdout, stderr) = run_main(
        "monotonic-until-2863",
        "eval",
        "def elapsed() -> ((i64, i64), (i64, i64)) ! { IO } = {\n  a = monotonic_now()\n  b = monotonic_now()\n  forward = monotonic_until(a, b)\n  backward = monotonic_until(b, a)\n  ((duration_second(forward), duration_nanosecond(forward)), (duration_second(backward), duration_nanosecond(backward)))\n}\nspan = elapsed()\n",
        &[],
    );
    assert!(success, "eval failed: stdout={stdout} stderr={stderr}");
    let forward = (printed(&stdout, "span.0.0"), printed(&stdout, "span.0.1"));
    let backward = (printed(&stdout, "span.1.0"), printed(&stdout, "span.1.1"));
    assert!(forward >= (0, 0), "a later reading came first: {forward:?}");
    assert!(
        backward <= (0, 0),
        "the reversed pair is positive: {backward:?}"
    );
    // Euclidean negation: -(s, ns) is (-s, 0) for ns = 0 and (-s - 1, 10^9 - ns) otherwise.
    let negated = if forward.1 == 0 {
        (-forward.0, 0)
    } else {
        (-forward.0 - 1, 1_000_000_000 - forward.1)
    };
    assert_eq!(backward, negated, "forward {forward:?}");
}

/// `chelis check` accepts clock reads in functions that declare `IO`, and the
/// effect a callee infers from them.
#[test]
fn check_accepts_clock_reads_that_declare_io() {
    let (success, stdout, stderr) = run_main(
        "clock-io-accepted-2863",
        "check",
        "def stamp() -> Instant ! { IO } = clock_now()\ndef inferred() -> MonotonicInstant = monotonic_now()\ndef elapsed() -> Duration ! { IO } = monotonic_until(inferred(), monotonic_now())\n",
        &[],
    );
    assert!(
        success,
        "declared IO clock reads must check: stdout={stdout} stderr={stderr}"
    );
    let json: serde_json::Value =
        serde_json::from_str(&stdout).expect("check output should be json");
    assert_eq!(json["errors"].as_array().map(Vec::len), Some(0), "{json}");
}

/// The rejected twin: a function declared pure cannot call `clock_now` or
/// `monotonic_now`, directly or through a callee whose `IO` is inferred.
#[test]
fn check_rejects_clock_reads_in_pure_functions() {
    for (name, body) in [
        ("wall", "def stamp() -> Instant ! {} = clock_now()\n"),
        (
            "monotonic",
            "def mark() -> MonotonicInstant ! {} = monotonic_now()\n",
        ),
        (
            "callee",
            "def inferred() -> Instant = clock_now()\ndef caller() -> i64 ! {} = instant_unix_second(inferred())\n",
        ),
    ] {
        let (success, stdout, stderr) =
            run_main(&format!("clock-pure-{name}-2863"), "check", body, &[]);
        assert!(
            !success && stdout.contains("UnhandledEffect") && stdout.contains("IO"),
            "{name}: expected an UnhandledEffect naming IO, got stdout={stdout} stderr={stderr}"
        );
    }
}

/// [05-OP-73]: `MonotonicInstant` has no construction path and no field
/// access outside its module, and nothing converts it to an `Instant`. The
/// accepted twin is `check_accepts_clock_reads_that_declare_io`, whose values
/// come from `monotonic_now`.
#[test]
fn monotonic_instant_is_opaque_outside_its_module() {
    let (success, stdout, stderr) = run_main(
        "monotonic-opaque-2863",
        "check",
        "forged = MonotonicInstant { second: 0i64, nanosecond: 0i64 }\ndef peek() -> i64 ! { IO } = monotonic_now().second\ndef converted() -> Duration ! { IO } = instant_until(monotonic_now(), monotonic_now())\n",
        &[],
    );
    let rendered = format!("{stdout}{stderr}");
    assert!(
        !success,
        "forging a MonotonicInstant must not check:\n{rendered}"
    );
    assert!(
        rendered.contains(
            "record construction of opaque type `MonotonicInstant` outside its defining module"
        ),
        "construction was not rejected as opaque:\n{rendered}"
    );
    assert!(
        rendered.contains("field access") && rendered.contains("opaque type `MonotonicInstant`"),
        "field access was not rejected as opaque:\n{rendered}"
    );
    // Diagnostics name another module's type by its module path (chelis#3269),
    // so the two distinct types read apart.
    assert!(
        rendered.contains(
            r#""kind":"TypeMismatch","message":"type mismatch: Std.Datetime.Instant vs Std.Datetime.Clock.MonotonicInstant""#
        ),
        "a MonotonicInstant was accepted as an Instant:\n{rendered}"
    );
}

/// `chelis build --target c` runs a program that reads either clock through
/// the module: `clock_now` lies between host reads around the run, and
/// `monotonic_until` between successive readings is never negative.
#[test]
fn compiled_c_runs_clock_reads_through_the_module() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("out");
    let (success, stdout, stderr) = run_main(
        "clock-build-run-1297",
        "build",
        "stamp = instant_unix_second(clock_now())\n\
         span = duration_second(monotonic_until(monotonic_now(), monotonic_now()))\n",
        &["--target", "c", "--output", out.to_str().unwrap()],
    );
    assert!(
        success,
        "clock reads must build: stdout={stdout} stderr={stderr}"
    );
    let lower = host_now();
    let output = std::process::Command::new(out.join("main"))
        .output()
        .expect("the built executable runs");
    let upper = host_now();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{output:?}");
    let stamp = printed(&stdout, "stamp");
    assert!(
        lower.0 <= stamp && stamp <= upper.0,
        "{lower:?} <= {stamp} <= {upper:?}"
    );
    assert!(printed(&stdout, "span") >= 0, "{stdout}");
}

/// The twin: importing the module without reading a clock builds too, and
/// its executable reads no clock.
#[test]
fn build_accepts_a_program_that_reads_no_clock() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("out");
    let (success, stdout, stderr) = run_main(
        "clock-build-pure-2863",
        "build",
        "anchor = 1i64\n",
        &["--target", "c", "--output", out.to_str().unwrap()],
    );
    assert!(
        success,
        "a program that reads no clock must build: stdout={stdout} stderr={stderr}"
    );
    assert!(
        out.join("main.c").exists(),
        "a program that reads no clock must emit C"
    );
}
