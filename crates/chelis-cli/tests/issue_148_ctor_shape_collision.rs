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
// The single-file reproducers here exercise the same fork: two ADTs
// declared in one module, each with an `IntCol` variant. Pre-fix,
// the call dispatched to whichever variant the registry's UnordMap
// iteration yielded first; post-fix, dispatch is shape-preferred and
// candidate ordering is sorted for determinism.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent");
    }
    fs::write(path, contents).expect("write file");
}

/// Run `chelis check <path>` and parse stdout as JSON. Parsing rather
/// than substring-matching keeps these assertions insensitive to
/// whitespace/key-order changes in the renderer. Issue #207 made
/// `chelis check` exit non-zero when the JSON `errors` array is
/// non-empty, and `record_only_ctor_still_errors_when_called_positionally`
/// deliberately expects errors, so this helper captures stdout
/// without asserting on the process exit code.
fn run_check(path: &Path) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    serde_json::from_slice(&output.stdout).expect("check output should be json")
}

fn error_messages(json: &Value) -> Vec<String> {
    json["errors"]
        .as_array()
        .expect("errors should be a json array")
        .iter()
        .map(|e| e["message"].as_str().unwrap_or("").to_string())
        .collect()
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
    // Headline case for the fix: two ADTs declare same-named `IntCol`,
    // one positional, one record. A positional call `IntCol(xs, mask)`
    // must dispatch to the positional variant — pre-fix this dispatched
    // to whichever the UnordMap iteration returned first and fired the
    // "must use named fields" error on the record half.
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("ctor_collision.ch");
    write_file(
        &fixture,
        "module CtorCollision\n\
         type RecordIntCol[n] = | IntCol { values: tensor[n, int64] }\n\
         type PositionalIntCol[n] = | IntCol(tensor[n, int64], tensor[n, bool])\n\
         def make_positional[n](xs: tensor[n, int64], mask: tensor[n, bool]) -> PositionalIntCol[n] = IntCol(xs, mask)\n",
    );
    fmt_inplace(&fixture);

    let json = run_check(&fixture);
    let msgs = error_messages(&json);
    assert_eq!(json["score"], 1.0, "perfect-score contract: {json}");
    assert!(
        msgs.is_empty(),
        "positional dispatch on mixed-shape collision must produce no errors; got {msgs:?}"
    );
}

#[test]
fn record_ctor_call_resolves_to_record_variant_when_positional_collides() {
    // Symmetric of the headline case: with the SAME two ADTs in scope,
    // a record-style call `IntCol { values: xs }` must dispatch to the
    // record variant. Locks in the symmetry of
    // `lookup_variant_preferring_shape` even though record construction
    // is parsed/lowered through a different builder than positional
    // `app` — if a future refactor re-routes record builds through
    // `infer_app`, the shape preference must still hold.
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("ctor_collision_record_call.ch");
    write_file(
        &fixture,
        "module CtorCollisionRecord\n\
         type RecordIntCol[n] = | IntCol { values: tensor[n, int64] }\n\
         type PositionalIntCol[n] = | IntCol(tensor[n, int64], tensor[n, bool])\n\
         def make_record[n](xs: tensor[n, int64]) -> RecordIntCol[n] = IntCol { values: xs }\n",
    );
    fmt_inplace(&fixture);

    let json = run_check(&fixture);
    let msgs = error_messages(&json);
    assert_eq!(json["score"], 1.0, "perfect-score contract: {json}");
    assert!(
        msgs.is_empty(),
        "record dispatch on mixed-shape collision must produce no errors; got {msgs:?}"
    );
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
         def make() -> WrapperRecord = Wrapper(cast(7, int64))\n",
    );
    fmt_inplace(&fixture);

    let json = run_check(&fixture);
    let msgs = error_messages(&json);
    assert!(
        msgs.iter().any(|m| m.contains("must use named fields")),
        "record-only positional call must produce the named-fields error; got {msgs:?}"
    );
}

#[test]
fn two_positional_same_name_emits_no_shape_error() {
    // Same-shape collision: two ADTs each define a positional `IntCol`
    // with identical fields. Both candidates match the call shape, so
    // `lookup_variant_preferring_shape` correctly never fires the
    // "must use named fields" error. This is the contract surface
    // this PR locks down for the two-positional case.
    //
    // What this PR does NOT fix: which of the two variants the call
    // *type-resolves to* for inference. That is governed by env
    // binding order (last-registered wins), not by this helper. When
    // the two variants have different field types, the call can
    // surface a type mismatch against whichever variant got picked.
    // The proper fix is module-scoped constructor resolution; tracked
    // as a follow-up to this PR (see chelis#157).
    //
    // The fixture uses identical field types in both variants so the
    // test is insensitive to which variant env-binding picks.
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("two_positional.ch");
    write_file(
        &fixture,
        "module TwoPositional\n\
         type AaaCol[n] = | IntCol(tensor[n, int64])\n\
         type BbbCol[n] = | IntCol(tensor[n, int64])\n\
         def make_aaa[n](xs: tensor[n, int64]) -> AaaCol[n] = IntCol(xs)\n",
    );
    fmt_inplace(&fixture);

    let json = run_check(&fixture);
    let msgs = error_messages(&json);
    assert!(
        !msgs.iter().any(|m| m.contains("must use named fields")),
        "two-positional same-shape collision must not fire the named-fields error; got {msgs:?}"
    );
}
