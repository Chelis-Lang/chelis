//! chelis#2946 (Std.Decimal): the module's laws as `chelis prove` properties.
//!
//! `packages/chelis-std/properties/decimal.ch` states [05-OP-76]'s laws over
//! `string`, `i64` and `f64` binders, building each decimal inside the
//! property, with a `where` guard that keeps only the inputs a law speaks
//! about. This test stages the standard library, runs `chelis prove` at the
//! fuzz tier over that file, and requires every property to pass on fuzz
//! evidence with the requested number of accepted samples. It then negates
//! each property's law, a deliberately false variant, and requires `chelis
//! prove` to report the variant as failed with a counterexample, which shows
//! that the guard admits inputs and that the property can fail.
//!
//! `chelis prove` checks and lowers the linked library once per property,
//! then checks each sample's probe against that library, including guards and
//! shrinking trials. The canary is what pull-request CI runs: two samples of every
//! property and the false variants of the two properties whose refutation is
//! cheap. The complete test runs the default hundred samples of every property
//! and every false variant in the nightly workflow, not in pull-request CI
//! (`.config/ci-test-targets.toml`).

use assert_cmd::Command;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use tempfile::{TempDir, tempdir};

#[path = "common/mod.rs"]
mod common;

const PROPERTIES_REL: &str = "properties/decimal.ch";

/// The properties the file states, each with the text of its law.
const LAWS: &[(&str, &str)] = &[
    (
        "decimal_text_round_trips",
        "eq(decimal(decimal_to_string(x)), x)",
    ),
    (
        "decimal_add_commutes",
        "eq(decimal_add(x, y), decimal_add(y, x))",
    ),
    (
        "decimal_sub_inverts_decimal_add",
        "eq(decimal_sub(decimal_add(x, y), y), x)",
    ),
    (
        "decimal_to_f64_agrees_with_to_float",
        "eq(decimal_to_f64(x), expected)",
    ),
    (
        "decimal_round_is_idempotent",
        "settles(decimal_round(x, n, mode), n, mode)",
    ),
    (
        "decimal_from_i64_round_trips",
        "eq(decimal_to_i64(decimal_from_i64(v), any_mode(choice)), v)",
    ),
];

/// The properties whose false variant the canary refutes.
const CANARY_FALSE_VARIANTS: &[&str] = &[
    "decimal_round_is_idempotent",
    "decimal_from_i64_round_trips",
];

/// A copy of `packages/chelis-std`, so a run reads this checkout's sources and
/// writes nothing into it, with its own reef home.
struct Staged {
    _dir: TempDir,
    package: PathBuf,
    reef_home: PathBuf,
}

fn stage() -> Staged {
    let dir = tempdir().expect("tempdir");
    let package = dir.path().join("chelis-std");
    let reef_home = dir.path().join("reef-home");
    common::copy_dir_recursive(&common::package_std(), &package);
    Staged {
        _dir: dir,
        package,
        reef_home,
    }
}

/// Runs `chelis prove --tier fuzz-only --json` over the staged properties
/// file, restricted to the property `only` when given, and returns its exit
/// code and property records by name.
fn prove(staged: &Staged, samples: usize, only: Option<&str>) -> (i32, BTreeMap<String, Value>) {
    let mut command = Command::cargo_bin("chelis").expect("binary");
    command
        .env("CHELIS_REEF_HOME", &staged.reef_home)
        .current_dir(&staged.package)
        .args(["prove", "--tier", "fuzz-only", "--json", "--samples"])
        .arg(samples.to_string());
    if let Some(name) = only {
        command.args(["--only", name]);
    }
    let output = command
        .arg(PROPERTIES_REL)
        .output()
        .expect("run chelis prove");
    let stdout = String::from_utf8(output.stdout).expect("utf-8 stdout from chelis prove");
    let mut records = BTreeMap::new();
    let mut summaries = 0;
    for line in stdout.lines().filter(|line| !line.trim().is_empty()) {
        let record: Value = serde_json::from_str(line).unwrap_or_else(|error| {
            panic!("chelis prove printed a non-JSON line {line:?}: {error}")
        });
        match record["kind"].as_str() {
            Some("property") => {
                let name = record["name"].as_str().expect("property name").to_string();
                assert!(
                    records.insert(name.clone(), record).is_none(),
                    "property {name} reported twice"
                );
            }
            Some("summary") => summaries += 1,
            other => panic!("unexpected record kind {other:?} in {line}"),
        }
    }
    assert_eq!(
        summaries,
        1,
        "chelis prove must print one summary record.\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let names: Vec<&str> = records.keys().map(String::as_str).collect();
    let mut expected: Vec<&str> = match only {
        Some(name) => vec![name],
        None => LAWS.iter().map(|(name, _)| *name).collect(),
    };
    expected.sort_unstable();
    assert_eq!(names, expected, "the run reports exactly these properties");
    (output.status.code().expect("exit code"), records)
}

/// Every law passes on fuzz evidence with `samples` accepted samples.
fn check_laws(staged: &Staged, samples: usize) {
    let (code, records) = prove(staged, samples, None);
    for (name, record) in &records {
        assert_eq!(record["status"], "passed", "{name}: {record}");
        assert_eq!(record["proof_tier"], "fuzz", "{name}: {record}");
        assert_eq!(
            record["composite_verdict"], "fuzz_validated",
            "{name}: {record}"
        );
        assert_eq!(record["accepted_samples"], samples, "{name}: {record}");
    }
    assert_eq!(code, 0, "every law passes, so chelis prove exits 0");
}

/// Negates every law in the staged file, then requires each named property,
/// or every property when `names` is `None`, to be refuted.
fn check_false_variants(staged: &Staged, names: Option<&[&str]>) {
    let path = staged.package.join(PROPERTIES_REL);
    let mut text = fs::read_to_string(&path).expect("read the staged properties");
    for (name, law) in LAWS {
        assert_eq!(
            text.matches(law).count(),
            1,
            "{name}: its law must appear exactly once in {PROPERTIES_REL}"
        );
        text = text.replacen(law, &format!("not({law})"), 1);
    }
    fs::write(&path, text).expect("write the false variants");
    let runs: Vec<(i32, BTreeMap<String, Value>)> = match names {
        Some(names) => names
            .iter()
            .map(|name| prove(staged, 1, Some(name)))
            .collect(),
        None => vec![prove(staged, 1, None)],
    };
    for (code, records) in runs {
        for (name, record) in &records {
            assert_eq!(
                record["status"], "failed",
                "{name}: its false variant must be refuted: {record}"
            );
            assert!(
                record.get("counterexample").is_some(),
                "{name}: a refutation names its counterexample: {record}"
            );
        }
        assert_eq!(code, 1, "a refuted property makes chelis prove exit 1");
    }
}

#[test]
fn std_decimal_property_canary_passes_and_its_false_variants_fail() {
    let staged = stage();
    check_laws(&staged, 2);
    check_false_variants(&staged, Some(CANARY_FALSE_VARIANTS));
}

/// Runs in the nightly workflow, not in pull-request CI
/// (`.config/ci-test-targets.toml`).
#[test]
fn std_decimal_properties_pass_at_the_default_samples_and_their_false_variants_fail() {
    let staged = stage();
    check_laws(&staged, 100);
    check_false_variants(&staged, None);
}
