//! `chelis prove` type-checks the module and errors on a type-broken one in
//! EVERY build, including the default (non-smt) build.
//!
//! Before this, the whole-module type-check rode only on the smt-gated
//! obligation path, so in the default build a type-broken module silently
//! passed `chelis prove` (exit 0) while the smt build rejected it. The default
//! build cannot verify producer obligations without smt -- that is a separate
//! stderr warning -- but it CAN type-check (the checker needs no solver), so it
//! does: a type-broken module is an Error (exit 3) on the surf AND deep paths.
//!
//! These tests are NOT feature-gated: they run against whatever `chelis` binary
//! the test build produces. In the default build the up-front check fires; in
//! the smt build the obligation path fires; both must exit 3 on a type-broken
//! module and must NOT exit 3 on a well-typed one.

use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

/// Write `contents` to `m.<ext>` in a fresh tempdir and run `chelis prove
/// --json` on it. Returns (exit_code, ndjson_records).
fn prove(contents: &str, ext: &str) -> (i32, Vec<Value>) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("m.{ext}"));
    std::fs::write(&path, contents).expect("write fixture");
    let out = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("prove")
        .arg(&path)
        .arg("--json")
        .output()
        .expect("run chelis prove");
    let code = out.status.code().unwrap_or(-1);
    let records = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .collect();
    (code, records)
}

/// Lower a `.ch` to canonical Deep via `chelis deep` (desugar does not
/// type-check, so a type-broken module still produces a `.dp`).
fn to_deep(surf: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let ch = dir.path().join("m.ch");
    std::fs::write(&ch, surf).expect("write .ch");
    let out = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("deep")
        .arg(&ch)
        .output()
        .expect("run chelis deep");
    assert!(
        out.status.success(),
        "chelis deep should desugar a type-broken module: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("utf8 deep")
}

/// `def bad(x: int32) -> f32 = x` -- an int32 body where f32 is declared: a
/// hard type error, no opaque type or invariant involved.
const TYPE_BROKEN: &str = "module M\nexport (bad)\ndef bad(x: int32) -> f32 = x\n";

#[test]
fn type_broken_surf_module_errors_not_silent_pass() {
    let (code, records) = prove(TYPE_BROKEN, "ch");
    assert_eq!(
        code, 3,
        "a type-broken module must exit 3 (Error), not silently pass: {records:?}"
    );
    assert!(
        records
            .iter()
            .any(|r| r["kind"] == "error" && r["stage"] == "check"),
        "the type-check failure must be surfaced as a check-error record: {records:?}"
    );
}

#[test]
fn type_broken_deep_module_errors_not_silent_pass() {
    let deep = to_deep(TYPE_BROKEN);
    let (code, records) = prove(&deep, "dp");
    assert_eq!(
        code, 3,
        "a type-broken deep module must exit 3 (Error), not silently pass: {records:?}"
    );
    assert!(
        records
            .iter()
            .any(|r| r["kind"] == "error" && r["stage"] == "check"),
        "the type-check failure must be surfaced as a check-error record: {records:?}"
    );
}

#[test]
fn well_typed_module_is_not_blocked_by_the_type_check_gate() {
    // A trivially-true, non-opaque property over a well-typed module: the
    // up-front type-check must pass it through (no false-positive Error), and
    // the Tier-C fuzz path proves `x <= x` cleanly. The point is exit != 3:
    // the gate rejects only genuine type errors.
    let source = "module M\nexport (f)\ndef f(x: f32) -> f32 = x\n@property triv forall(x: f32):\n  x <= x\n";
    let (code, records) = prove(source, "ch");
    assert_ne!(
        code, 3,
        "a well-typed module must not be rejected by the type-check gate: {records:?}"
    );
    assert!(
        !records
            .iter()
            .any(|r| r["kind"] == "error" && r["stage"] == "check"),
        "a well-typed module must emit no check-error record: {records:?}"
    );
}
