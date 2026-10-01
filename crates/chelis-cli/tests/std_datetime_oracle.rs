//! chelis#2859 (Std.Datetime S1): `chelis eval` and compiled C against an
//! independent reference.
//!
//! The corpus and its oracle are Python, invoked here and not reimplemented in
//! Rust: `scripts/datetime_reference.py` computes every expected value from
//! the algorithms' definitions and `spec/design/std_datetime.md`, and
//! `scripts/datetime_differential.py` generates the Chelis programs, runs them
//! on both lanes, and compares each lane with the reference. This test
//! supplies the freshly built `chelis`, a published `chelis-std`, and the
//! strict reference toolchain the `lane-check` gate uses, then holds the
//! harness to its pass line.
//!
//! The default test is the CI profile: the range edges, seeded samples across
//! the whole range, every policy branch, every year's Easter, and the valid and
//! invalid text corpus on both lanes, plus every day from 1900-01-01 through
//! 2100-12-31 in compiled C. The ignored test is the manual gate in
//! `docs/manual_gates.md`: every day of the range in compiled C, and every day
//! from 1900 through 2100 on `chelis eval`.

use std::path::PathBuf;
use std::process::Command;

use chelis_backend_c::toolchain::{
    CodegenRequirements, strict_reference_toolchain, test_toolchain,
};

#[path = "common/mod.rs"]
mod common;

#[path = "../../../tests/support/managed_python.rs"]
mod managed_python;

const PASS_LINE: &str = "STD DATETIME ORACLE: PASS";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crate is two levels below the repo root")
        .to_path_buf()
}

fn toolchain_json() -> String {
    let requirements = CodegenRequirements::default();
    let toolchain = strict_reference_toolchain(test_toolchain(requirements).compiler, requirements);
    serde_json::json!({
        "compiler": toolchain.compiler,
        "compile_flags": toolchain.compile_flags,
        "link_flags": toolchain.link_flags,
    })
    .to_string()
}

fn run_harness(profile: &str, lanes: &str) -> String {
    let python =
        managed_python::managed_python(&repo_root()).unwrap_or_else(|error| panic!("{error}"));
    let chelis = assert_cmd::cargo_bin!("chelis").to_path_buf();
    let output = Command::new(python)
        .arg(repo_root().join("scripts/datetime_differential.py"))
        .arg("--chelis")
        .arg(&chelis)
        .arg("--reef-home")
        .arg(&common::SHARED_REEF.reef_home)
        .args([
            "--toolchain-json",
            &toolchain_json(),
            "--profile",
            profile,
            "--lanes",
            lanes,
        ])
        .output()
        .expect("spawn the datetime differential harness");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && stdout.lines().any(|line| line.starts_with(PASS_LINE)),
        "Std.Datetime differential failed ({}).\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status
    );
    stdout
}

#[test]
fn std_datetime_agrees_with_the_reference_on_eval_and_c() {
    let stdout = run_harness("ci", "eval,c");
    assert!(stdout.contains("on lanes eval+c"), "{stdout}");
}

#[test]
#[ignore = "manual gate (docs/manual_gates.md, Std.Datetime S1): every day of the range in compiled C and 1900-2100 on eval. Run `cargo nextest run -p chelis-cli --test std_datetime_oracle --run-ignored only`."]
fn std_datetime_every_day_of_the_range_in_compiled_c() {
    let stdout = run_harness("exhaustive", "eval,c");
    assert!(
        stdout.contains("on lanes eval+c; 7304484 days in day rows"),
        "{stdout}"
    );
}
