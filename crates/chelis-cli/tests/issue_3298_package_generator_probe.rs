//! A constructor probe for an opaque binder belongs to the type's module
//! whether `prove` reads a standalone file or a Reef-linked package.

mod common;

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

const SOURCE: &str = "module Demo.Main
export (norm, prob_value)
@opaque
@invariant(p) ((p.value >= 0.4995f32) && (p.value <= 0.5005f32))
type T =
  | T { value: f32 }
def norm(x: f32) -> T = T { value: 0.5f32 }
def prob_value(p: T) -> f32 = p.value
@property w forall(p: T):
  (prob_value(p) <= 0.5005f32)
@property w_false forall(p: T):
  (prob_value(p) <= 0.4f32)
";

fn prove(root: &Path, file: &str, tier: &str, reef_home: Option<&Path>) -> (i32, Vec<Value>) {
    let mut command = Command::cargo_bin("chelis").expect("chelis binary");
    command
        .current_dir(root)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "prove",
            file,
            "--tier",
            tier,
            "--samples",
            "16",
            "--seed",
            "1",
            "--json",
        ]);
    if let Some(reef_home) = reef_home {
        command.env("CHELIS_REEF_HOME", reef_home);
    }
    let output = command.output().expect("prove runs");
    let records = String::from_utf8(output.stdout)
        .expect("utf8 output")
        .lines()
        .map(|line| serde_json::from_str(line).expect("JSON record"))
        .collect();
    (output.status.code().expect("exit code"), records)
}

fn property<'a>(records: &'a [Value], name: &str) -> &'a Value {
    records
        .iter()
        .find(|record| record["kind"] == "property" && record["name"] == name)
        .unwrap_or_else(|| panic!("missing {name}: {records:#?}"))
}

fn assert_constructor_outcomes(records: &[Value]) {
    let good = property(records, "w");
    let bad = property(records, "w_false");
    assert_eq!(good["status"], "passed", "{good:#}");
    assert_eq!(good["composite_verdict"], "fuzz_validated", "{good:#}");
    assert_eq!(bad["status"], "failed", "{bad:#}");
    assert_eq!(bad["composite_verdict"], "failed", "{bad:#}");
    assert!(bad["counterexample"].is_object(), "{bad:#}");
    assert_eq!(good["samples"], 16, "{good:#}");
    assert_eq!(bad["samples"], 1, "{bad:#}");
}

#[test]
fn linked_package_matches_standalone_for_true_and_false_properties() {
    let (_dir, reef_home, app) = common::make_app("issue-3298-package");
    fs::write(app.join("src/main.ch"), SOURCE).expect("write package module");
    let bare = tempdir().expect("bare dir");
    fs::write(
        bare.path().join("bare.ch"),
        SOURCE.replace("Demo.Main", "Bare"),
    )
    .expect("write bare module");
    for tier in ["fuzz-only", "auto"] {
        let (package_code, package_records) = prove(&app, "src/main.ch", tier, Some(&reef_home));
        let (bare_code, bare_records) = prove(bare.path(), "bare.ch", tier, None);
        assert_eq!(package_code, bare_code, "{package_records:#?}");
        assert_constructor_outcomes(&package_records);
        assert_constructor_outcomes(&bare_records);
        for name in ["w", "w_false"] {
            let package = property(&package_records, name);
            let standalone = property(&bare_records, name);
            for field in ["status", "composite_verdict", "samples"] {
                assert_eq!(package[field], standalone[field], "{name} {field}");
            }
        }
    }
}

#[test]
fn generated_probe_name_is_fresh_in_a_linked_package() {
    let (_dir, _reef_home, app) = common::make_app("issue-3298-collision");
    // The linked generator stem ends in `chelis_gen_probe`, so this user
    // definition occupies the name the generator would otherwise choose.
    // Calling it from `w` also checks that the user definition survives.
    let source = SOURCE
        .replace(
            "@property w forall",
            "def chelis_gen_probe() -> bool = true\n@property w forall",
        )
        .replace(
            "(prob_value(p) <= 0.5005f32)",
            "((prob_value(p) <= 0.5005f32) && chelis_gen_probe())",
        );
    fs::write(app.join("src/main.ch"), source).expect("write package module");
    let (_code, records) = prove(&app, "src/main.ch", "fuzz-only", Some(&_reef_home));
    assert_constructor_outcomes(&records);
    assert!(
        property(&records, "w")["goal"]
            .as_str()
            .expect("goal")
            .contains("chelis_gen_probe("),
        "the true property must call the colliding user definition: {records:#?}"
    );
}
