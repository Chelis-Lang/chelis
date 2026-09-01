//! chelis#1125 / chelis#731: ordinary `.dp` file ingress applies the same
//! recursive type grammar as the serialized-type boundary.

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

const INVALID_NOMINAL_RANK: &str =
    "(defsig {} bad (t-fn {} (t-adt {} Rows (d-rank {} r)) (t-unit {})))\n";
const INVALID_METADATA_NOMINAL_RANK: &str =
    "(def {} bad (fn {} (params {} (x {type: (t-adt {} Rows (d-rank {} r))})) (var {} x)))\n";
const VALID_NOMINAL_DIMENSION: &str =
    "(defsig {} sized (t-fn {} (t-adt {} Rows (d-lit {} 3)) (t-unit {})))\n";
const VALID_TENSOR_RANK: &str =
    "(defsig {} ranked (t-fn {} (t-tensor {} (d-rank {} r) (t-prim {} f32)) (t-unit {})))\n";

fn write_deep(name: &str, source: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(name);
    write_file(&path, source);
    (dir, path)
}

fn chelis() -> Command {
    let mut command = Command::cargo_bin("chelis").expect("chelis binary");
    command.env("CHELIS_STYLE_GATE_DISABLE", "1");
    command
}

#[test]
fn surf_and_validate_reject_rank_spreads_in_nominal_arguments() {
    for (name, source) in [
        ("invalid_nominal_rank.dp", INVALID_NOMINAL_RANK),
        (
            "invalid_metadata_nominal_rank.dp",
            INVALID_METADATA_NOMINAL_RANK,
        ),
    ] {
        let (_dir, path) = write_deep(name, source);
        for args in [vec!["surf"], vec!["validate", "--deep"]] {
            chelis()
                .args(args)
                .arg(&path)
                .assert()
                .failure()
                .stderr(predicate::str::contains(
                    "expected NominalArgument type syntax, got `d-rank`",
                ));
        }
    }
}

#[test]
fn surf_and_validate_accept_nominal_dimension_arguments() {
    let (_dir, path) = write_deep("valid_nominal_dimension.dp", VALID_NOMINAL_DIMENSION);

    chelis()
        .args(["surf"])
        .arg(&path)
        .assert()
        .success()
        .stdout(predicate::str::contains("Rows[3]"));
    chelis()
        .args(["validate", "--deep"])
        .arg(&path)
        .assert()
        .success()
        .stdout(predicate::str::contains("validated deep:"));
}

#[test]
fn surf_and_validate_keep_tensor_rank_spreads_legal() {
    let (_dir, path) = write_deep("valid_tensor_rank.dp", VALID_TENSOR_RANK);

    chelis()
        .args(["surf"])
        .arg(&path)
        .assert()
        .success()
        .stdout(predicate::str::contains("tensor[..r, f32]"));
    chelis()
        .args(["validate", "--deep"])
        .arg(&path)
        .assert()
        .success()
        .stdout(predicate::str::contains("validated deep:"));
}
