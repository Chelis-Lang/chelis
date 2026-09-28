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

/// `def bad(x: i32) -> f32 = x` -- an i32 body where f32 is declared: a
/// hard type error, no opaque type or invariant involved.
const TYPE_BROKEN: &str = "module M\nexport (bad)\ndef bad(x: i32) -> f32 = x\n";

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

// ===========================================================================
// chelis#1125 PP7 slice E5c: the obligations-skipped warning must not depend
// on which admitted representation carries the module ([04-TOT-5]).
//
// Spec authority: spec/04-type-system.md §10 [04-TOT-5]. Design:
// spec/design/checker_totality.md §"PP7. Stamped-ingress reader parity",
// which calls this the class's worst shape: a reader dead twice over, on a
// release-blocking surface, silently. `count_invariant_opaque_deep` was
// `Expr::List`-only AND read its tag as `Atom::Name("deftype")`, which
// decode-once (§C4.2) forbids: the parser stamps every vocabulary tag, so
// element 0 of a decoded form is `Atom::Tag`, never `Atom::Name`. It
// therefore returned zero for EVERY input on both carriers, and the warning
// could not fire on the `.dp` path at all.
// ===========================================================================

/// A module whose `@opaque` type carries an `@invariant`: exactly one
/// invariant-carrying opaque type, and a producer whose guard matches it.
const OPAQUE_INVARIANT: &str = "module Stats.Prob\nexport (probability)\n@opaque\n\
   @invariant(p) p.value >= 0.0 && p.value <= 1.0\n\
   type Probability =\n  | Probability { value: f32 }\n\
   def probability(x: f32) -> Option[Probability] =\n  \
   if x >= 0.0 && x <= 1.0 then Some(Probability { value: x }) else None\n";

/// The same shape with NO `@invariant`: an opaque type alone owes no producer
/// obligation, so nothing is skipped and nothing is warned about.
const OPAQUE_WITHOUT_INVARIANT: &str = "module Stats.Prob\nexport (probability)\n@opaque\n\
   type Probability =\n  | Probability { value: f32 }\n\
   def probability(x: f32) -> Option[Probability] =\n  \
   if x >= 0.0 && x <= 1.0 then Some(Probability { value: x }) else None\n";

/// The `{kind:"warning", stage:"obligations"}` records of a prove run, as
/// `skipped` counts.
fn obligation_warnings(records: &[Value]) -> Vec<i64> {
    records
        .iter()
        .filter(|r| {
            r.get("kind").and_then(Value::as_str) == Some("warning")
                && r.get("stage").and_then(Value::as_str) == Some("obligations")
        })
        .filter_map(|r| r.get("skipped").and_then(Value::as_i64))
        .collect()
}

/// PP7's `count_invariant_opaque_deep` row. REGRESSION TEST (red before the
/// `crates/chelis-cli/src/prove/mod.rs` repair, green after): the same module
/// warns as `.ch` and does not warn as `.dp`, though the `.dp` declares
/// exactly one invariant-carrying opaque type.
///
/// The carrier-parity assertion holds in every build. The count assertion is
/// `smt`-off only, because the warning exists only there: with `smt` on, the
/// obligations are actually verified and neither surface warns, which is
/// still parity. The oracle command for this row is the default build.
///
/// One row, two defects: the repair must fix BOTH the `Expr::List`-only
/// carrier read and the `Atom::Name("deftype")` tag read to turn it green,
/// and this row cannot tell you which of the two you left behind.
#[test]
fn obligation_skip_warning_agrees_across_both_surfaces() {
    let (ch_code, ch_records) = prove(OPAQUE_INVARIANT, "ch");
    let deep = to_deep(OPAQUE_INVARIANT);
    let (dp_code, dp_records) = prove(&deep, "dp");
    assert_ne!(ch_code, 3, "the .ch module is well-typed: {ch_records:?}");
    assert_ne!(dp_code, 3, "the .dp module is well-typed: {dp_records:?}");
    assert_eq!(
        obligation_warnings(&dp_records),
        obligation_warnings(&ch_records),
        "the obligations-skipped warning must not depend on which admitted \
         representation carries the module (chelis#1125 [04-TOT-5]); the .dp \
         surface emitted none for a module declaring one invariant-carrying \
         opaque type.\n.ch records: {ch_records:?}\n.dp records: {dp_records:?}"
    );
    #[cfg(not(feature = "smt"))]
    assert_eq!(
        obligation_warnings(&ch_records),
        vec![1],
        "a non-smt build warns once, for the one invariant-carrying opaque \
         type: {ch_records:?}"
    );
}

/// The negative twin, DISPOSITION LOCK (green before and after): an opaque
/// type with NO invariant owes no obligation, so neither surface warns.
/// Without it, "always warn once" would satisfy the row above.
#[test]
fn opaque_type_without_an_invariant_warns_on_neither_surface() {
    let (_, ch_records) = prove(OPAQUE_WITHOUT_INVARIANT, "ch");
    let deep = to_deep(OPAQUE_WITHOUT_INVARIANT);
    let (_, dp_records) = prove(&deep, "dp");
    assert!(
        obligation_warnings(&ch_records).is_empty(),
        "an opaque type with no invariant owes no obligation, so the .ch \
         surface must not warn: {ch_records:?}"
    );
    assert!(
        obligation_warnings(&dp_records).is_empty(),
        "the .dp surface must not warn either: {dp_records:?}"
    );
}
