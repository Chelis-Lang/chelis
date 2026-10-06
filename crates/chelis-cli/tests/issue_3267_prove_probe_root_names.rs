//! Acceptance surface for chelis#3267: `chelis prove` names every root it
//! declares in the user's program (the property and precondition sample
//! probes, the opaque generator's constructor probe, and the constant probe)
//! fresh against that program. A module that defines one of the plain
//! spellings gets the verdict of the same module with the definition
//! renamed, in `fuzz-only` and in `auto`, and a false property beside such a
//! definition still fails with a counterexample.

use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

/// The verdict-bearing fields of every property record, in order, and the
/// exit code.
fn prove(source: &str, tier: &str) -> (i32, Vec<Value>, String) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("probe.ch");
    std::fs::write(&path, source).expect("write fixture");
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["prove"])
        .arg(&path)
        .args(["--tier", tier, "--samples", "16", "--seed", "1", "--json"])
        .output()
        .expect("run prove");
    let transcript = format!(
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let records = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|record| record["kind"] == "property")
        .map(|record| {
            serde_json::json!({
                "name": record["name"],
                "status": record["status"],
                "composite_verdict": record["composite_verdict"],
                "proof_tier": record["proof_tier"],
                "samples": record["samples"],
                "accepted_samples": record["accepted_samples"],
                "counterexample": record["counterexample"],
                "reason": record["reason"],
            })
        })
        .collect();
    (output.status.code().unwrap_or(-1), records, transcript)
}

/// `holds` passes and `too_strong` fails; `name` is defined beside them.
fn sampled_module(name: &str) -> String {
    format!(
        "module Probe.Fuzz
{name} = true
@property holds forall(x: f32) where x >= 0.0f32, x <= 1.0f32:
  (x <= 2.0f32)
@property too_strong forall(x: f32) where x >= 0.0f32, x <= 1.0f32:
  (x <= 0.5f32)
"
    )
}

/// The issue's opaque-generator witness: `w` holds over a band that only the
/// constructor tier can serve, and its invariant reads the constant `eps`;
/// `w_false` does not hold.
fn opaque_module(name: &str) -> String {
    format!(
        "module M
export (norm, prob_value)
@opaque
@invariant(p) ((p.value >= (0.5 - eps)) && (p.value <= (0.5 + eps)))
type T =
  | T {{ value: f32 }}
def eps() -> f32 = 0.0005
def norm(x: f32) -> T = T {{ value: 0.5 }}
def prob_value(p: T) -> f32 = p.value
{name} = true
@property w forall(p: T):
  (prob_value(p) <= 0.5005)
@property w_false forall(p: T):
  (prob_value(p) <= 0.4)
"
    )
}

fn record<'a>(records: &'a [Value], name: &str, transcript: &str) -> &'a Value {
    records
        .iter()
        .find(|record| record["name"] == name)
        .unwrap_or_else(|| panic!("no record for {name}\n{transcript}"))
}

/// The module defining `name` matches the renamed control record for record,
/// `holds` passes by fuzzing, and `refuted` fails with a counterexample.
fn assert_matches_control(
    module: fn(&str) -> String,
    name: &str,
    tier: &str,
    holds: &str,
    refuted: &str,
) {
    let (control_code, control, control_transcript) = prove(&module("renamed_probe"), tier);
    let (code, records, transcript) = prove(&module(name), tier);
    assert_eq!(
        (code, &records),
        (control_code, &control),
        "{name} under {tier}\n{transcript}\ncontrol: {control_transcript}"
    );
    let holds = record(&records, holds, &transcript);
    assert_eq!(holds["status"], "passed", "{holds}\n{transcript}");
    assert_eq!(holds["composite_verdict"], "fuzz_validated", "{holds}");
    let refuted = record(&records, refuted, &transcript);
    assert_eq!(refuted["status"], "failed", "{refuted}\n{transcript}");
    assert!(
        refuted["counterexample"].is_object(),
        "a false property keeps its counterexample: {refuted}"
    );
}

#[test]
fn sample_probe_definitions_keep_the_fuzz_verdict() {
    for name in ["__chelis_property_probe", "__chelis_property_pre"] {
        assert_matches_control(sampled_module, name, "fuzz-only", "holds", "too_strong");
    }
}

#[test]
fn generator_and_constant_probe_definitions_keep_the_verdict_in_both_tiers() {
    for tier in ["fuzz-only", "auto"] {
        for name in [
            "__chelis_gen_probe",
            "__chelis_const_probe",
            "__chelis_prop_probe",
        ] {
            assert_matches_control(opaque_module, name, tier, "w", "w_false");
        }
    }
}
