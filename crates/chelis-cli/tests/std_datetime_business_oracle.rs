//! chelis#2860 (Std.Datetime.Business): `chelis eval` and compiled C against
//! an independent reference.
//!
//! The corpus and its oracle are Python, invoked here and not reimplemented in
//! Rust: `scripts/datetime_business_reference.py` computes every expected value
//! by walking days under the definitions of [05-OP-73], and
//! `scripts/datetime_business_differential.py` generates the Chelis programs,
//! runs them on both lanes, and compares each lane with the reference. The
//! reference is itself checked against NumPy's business-day functions by
//! `scripts/test_datetime_business_reference.py`. This test supplies the
//! freshly built `chelis`, a published `chelis-std`, and the strict reference
//! toolchain the `lane-check` gate uses, then holds the harness to its pass
//! line.
//!
//! The corpus covers range-edge, month-edge, one-day and no-business-day
//! calendars and seeded random ones: every roll, every start with offsets up
//! to both horizon ends and both i64 extremes, counts in both directions, the
//! vectorized forms, both combinations, and one failure per failing path.
//! Compiled C observes every query; `chelis eval` observes a subset of each
//! calendar's queries.

use std::path::PathBuf;
use std::process::Command;

use chelis_backend_c::toolchain::{
    CodegenRequirements, strict_reference_toolchain, test_toolchain,
};

#[path = "common/mod.rs"]
mod common;

#[path = "../../../tests/support/managed_python.rs"]
mod managed_python;

const PASS_LINE: &str = "STD DATETIME BUSINESS ORACLE: PASS";

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

#[test]
fn std_datetime_business_agrees_with_the_reference_on_eval_and_c() {
    let python =
        managed_python::managed_python(&repo_root()).unwrap_or_else(|error| panic!("{error}"));
    let chelis = assert_cmd::cargo_bin!("chelis").to_path_buf();
    let output = Command::new(python)
        .arg(repo_root().join("scripts/datetime_business_differential.py"))
        .arg("--chelis")
        .arg(&chelis)
        .arg("--reef-home")
        .arg(&common::SHARED_REEF.reef_home)
        .args(["--toolchain-json", &toolchain_json(), "--lanes", "eval,c"])
        .output()
        .expect("spawn the business-day differential harness");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && stdout.lines().any(|line| line.starts_with(PASS_LINE)),
        "Std.Datetime.Business differential failed ({}).\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status
    );
    assert!(stdout.contains("on lanes eval+c"), "{stdout}");
}
