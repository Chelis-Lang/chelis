//! chelis#2778 (Std.Decimal): `chelis eval` and compiled C against an
//! independent reference.
//!
//! The corpus and its oracle are Python, invoked here and not reimplemented in
//! Rust: `scripts/decimal_reference.py` computes every expected value from the
//! [05-OP-76] contract on exact rationals, and `scripts/decimal_differential.py`
//! generates the Chelis programs, runs them on both lanes, and compares each
//! lane with the reference. This test supplies the freshly built `chelis`, a
//! published `chelis-std`, and the strict reference toolchain the `lane-check`
//! gate uses, then holds the harness to its pass line.
//!
//! The default test is the edge corpus: envelope and limb boundaries, carry
//! chains, removable zeros, i64 edges, ties in every rounding mode and sign,
//! subnormal and double-rounding f32 witnesses, the parser's accepted and
//! rejected spellings, seeded random values, and a sampled program for every
//! failure path, on both lanes. The ignored test is the manual gate in
//! `docs/manual_gates.md`: the same corpus with many more seeded random cases
//! and failure samples.

use std::path::PathBuf;
use std::process::Command;

use chelis_backend_c::toolchain::{
    CodegenRequirements, strict_reference_toolchain, test_toolchain,
};

#[path = "common/mod.rs"]
mod common;

#[path = "../../../tests/support/managed_python.rs"]
mod managed_python;

const PASS_LINE: &str = "STD DECIMAL ORACLE: PASS";

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

fn run_harness(extra: &[&str]) -> String {
    let python =
        managed_python::managed_python(&repo_root()).unwrap_or_else(|error| panic!("{error}"));
    let chelis = assert_cmd::cargo_bin!("chelis").to_path_buf();
    let output = Command::new(python)
        .arg(repo_root().join("scripts/decimal_differential.py"))
        .arg("--chelis")
        .arg(&chelis)
        .arg("--reef-home")
        .arg(&common::SHARED_REEF.reef_home)
        .args(["--toolchain-json", &toolchain_json(), "--lanes", "eval,c"])
        .args(extra)
        .output()
        .expect("spawn the decimal differential harness");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && stdout.lines().any(|line| line.starts_with(PASS_LINE)),
        "Std.Decimal differential failed ({}).\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status
    );
    stdout
}

#[test]
fn std_decimal_agrees_with_the_reference_on_eval_and_c() {
    let stdout = run_harness(&[]);
    assert!(
        stdout.contains("on lanes eval+c; default corpus)"),
        "{stdout}"
    );
}

#[test]
#[ignore = "manual gate (docs/manual_gates.md, Std.Decimal): the large seeded corpus on eval and compiled C. Run `cargo nextest run -p chelis-cli --test std_decimal_oracle --run-ignored only`."]
fn std_decimal_large_corpus_agrees_with_the_reference_on_eval_and_c() {
    let stdout = run_harness(&["--large"]);
    assert!(
        stdout.contains("on lanes eval+c; large corpus)"),
        "{stdout}"
    );
}
