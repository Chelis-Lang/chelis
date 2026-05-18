// Regression for chelis#148: when two ADTs in the dep graph export
// constructors with the same unqualified name but different shapes
// (one positional, one record), `chelis check` resolved bare
// constructor occurrences via a flat global ADT-registry lookup and
// the call site dispatched to the wrong variant. The downstream
// symptom in `Chelis-Lang/school` PR #16 was a flood of "IntCol is a
// record constructor and must use named fields" errors when coral's
// own source was type-checked against an env that also contained
// school's record-shaped `IntCol`.
//
// Fix in `crates/chelis-types/src/adt.rs`: add
// `lookup_variant_preferring_shape`, which returns the variant whose
// field-naming style matches the call shape (positional vs record).
// `infer_app` uses it before emitting the "must use named fields"
// error so the call dispatches to the matching ADT when both are in
// scope.
//
// The single-file reproducer here exercises the same fork: two ADTs
// declared in one module, each with an `IntCol` variant, one
// positional and one record. Pre-fix, calling `IntCol(xs, mask)` as
// a positional constructor errored because the lookup hit the record
// variant first. Post-fix, the positional variant is preferred and
// the call type-checks.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent");
    }
    fs::write(path, contents).expect("write file");
}

fn check_file(path: &Path) -> assert_cmd::assert::Assert {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", path.to_str().unwrap()])
        .assert()
}

fn fmt_inplace(path: &Path) {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", "--inplace", path.to_str().unwrap()])
        .assert()
        .success();
}

#[test]
fn positional_ctor_call_resolves_to_positional_variant_when_record_collides() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("ctor_collision.ch");
    write_file(
        &fixture,
        "module CtorCollision\n\
         type RecordIntCol = | IntCol { values: tensor[n, int64] }\n\
         type PositionalIntCol[n] = | IntCol(tensor[n, int64], tensor[n, bool])\n\
         def make_positional[n](xs: tensor[n, int64], mask: tensor[n, bool]) -> PositionalIntCol[n] = {\n\
           IntCol(xs, mask)\n\
         }\n",
    );
    fmt_inplace(&fixture);

    check_file(&fixture)
        .stdout(predicate::str::contains("\"score\": 1"))
        .stdout(predicate::str::contains("\"errors\": []"))
        .stdout(predicate::str::contains("must use named fields").not());
}

#[test]
fn record_only_ctor_still_errors_when_called_positionally() {
    // Regression-guard: when ONLY a record-shaped variant exists for
    // the name (no positional collision), positional calls must still
    // emit the existing "must use named fields" error. The fix only
    // changes behavior when both shapes are in scope.
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("record_only.ch");
    write_file(
        &fixture,
        "module RecordOnly\n\
         type WrapperRecord = | Wrapper { value: int64 }\n\
         def make() -> WrapperRecord = {\n\
           Wrapper(cast(7, int64))\n\
         }\n",
    );
    fmt_inplace(&fixture);

    check_file(&fixture).stdout(predicate::str::contains("must use named fields"));
}
