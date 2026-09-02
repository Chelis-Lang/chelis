//! chelis#886: `chelis check`'s error objects have one typed producer.
//!
//! The document was hand-assembled with `format!` and `{:?}` on the kind
//! enum. The producer is now `schema::Diagnostic`, so the emitted object and
//! the type that describes it cannot drift apart.
//!
//! The carrier is deliberately the SCHEMA type rather than the checker's own
//! `CheckError`. Serializing `CheckError` would make a `chelis-types` struct
//! a numeric wire root outside the §C6 census, which is rooted in
//! `chelis-compiler-api`; and it would spell `kind` from a Rust identifier
//! instead of the sealed `DiagnosticKind` vocabulary.
//!
//! The existing round-trip oracle (`dp_check_ingest.rs::check_dp_json_round_
//! trips_through_schema`) uses a WELL-TYPED fixture, so its `errors` array is
//! empty and `WireDiagnostic` is never exercised. These use ERRORING
//! documents, which is the case that matters.

use std::fs;

use assert_cmd::Command;
use tempfile::tempdir;

/// The check report's raw stdout bytes.
///
/// The parsing helper below is right for structural claims, but this PR
/// makes a BYTE claim: the carrier's declaration order is the wire order.
/// `serde_json::Value` equality erases object member order, so a test that
/// compares parsed values cannot witness that claim: moving `severity`
/// before `message` changes the bytes and leaves every such assertion green.
fn check_stdout_bytes(source: &str) -> Vec<u8> {
    let directory = tempdir().expect("tempdir");
    let path = directory.path().join("probe.ch");
    fs::write(&path, source).expect("write fixture");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check"])
        .arg(&path)
        .output()
        .expect("run chelis check")
        .stdout
}

fn check_json(source: &str) -> serde_json::Value {
    let directory = tempdir().expect("tempdir");
    let path = directory.path().join("probe.ch");
    fs::write(&path, source).expect("write fixture");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check"])
        .arg(&path)
        .output()
        .expect("run chelis check");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    serde_json::from_str(&stdout)
        .unwrap_or_else(|error| panic!("check must emit JSON; {error}\n{stdout}"))
}

/// Every spelling the wire may carry for a check diagnostic's `kind`.
///
/// End-to-end coverage: this asserts what a real `chelis check` process
/// emits. The exhaustive owner-level pin -- every kind, including the ones no
/// fixture here provokes -- lives beside the projection in
/// `chelis-compiler-api`'s `schema.rs`.
const CHECK_KIND_SPELLINGS: &[&str] = &[
    "TypeMismatch",
    "PrecisionMismatch",
    "DimensionMismatch",
    "ArityMismatch",
    "UnboundVariable",
    "UnknownConstructor",
    "NotAFunction",
    "NonExhaustiveMatch",
    "OccursCheck",
    "CastNonTensor",
    "TupleIndexOutOfBounds",
    "UseAfterConsume",
    "UnconsumedLinear",
    "InvalidBorrow",
    "CycleDetected",
    "UnsupportedTensorPrecision",
    "DuplicateDefinition",
    "DuplicateModule",
    "OpaqueTypeViolation",
    "ReservedLinkerName",
    "BuiltinShadowing",
    "UnknownForm",
    "MalformedForm",
    "Other",
    // effect diagnostics share the array
    "UnhandledEffect",
    "InvalidHandler",
    "BuildTargetMismatch",
    "TypeTotality",
];

#[test]
fn an_erroring_document_round_trips_through_the_consumer_schema() {
    let report = check_json("def f(x: f32) -> f32 = add(x, nope)\n");
    let errors = report["errors"].as_array().expect("errors array");
    assert!(!errors.is_empty(), "the fixture must produce a diagnostic");
    let raw = serde_json::to_string(&report).expect("re-serialize");
    let parsed: Result<chelis_compiler_api::schema::WireCheckResult, _> =
        serde_json::from_str(&raw);
    assert!(
        parsed.is_ok(),
        "an ERRORING document must deserialize into the consumer schema; {:?}",
        parsed.err()
    );
}

#[test]
fn every_emitted_kind_is_a_known_spelling() {
    // The wire-spelling pin. An unknown or renamed kind fails here rather
    // than reaching a consumer that keys on it.
    for source in [
        "def f(x: f32) -> f32 = add(x, nope)\n",
        "def g(x: f32) -> f32 = add(x, cast(1, int32))\n",
        "def k(x: int64) -> int64 = copy(x)\n",
    ] {
        let report = check_json(source);
        for error in report["errors"].as_array().expect("errors array") {
            let kind = error["kind"].as_str().expect("kind is a string");
            assert!(
                CHECK_KIND_SPELLINGS.contains(&kind),
                "unknown diagnostic kind on the wire: {kind:?}. \
                 A renamed variant changes a published surface; add the new \
                 spelling here deliberately, or keep the old one."
            );
        }
    }
}

#[test]
fn a_diagnostic_carries_the_documented_field_set() {
    let report = check_json("def f(x: f32) -> f32 = add(x, nope)\n");
    let error = &report["errors"].as_array().expect("errors")[0];
    for required in ["kind", "message", "severity"] {
        assert!(
            !error[required].is_null(),
            "`{required}` is always present; got {error}"
        );
    }
    // Absent-not-null: the optional fields are skipped, never emitted as
    // JSON null, so a consumer distinguishes "not determined" from "null".
    let object = error.as_object().expect("diagnostic is an object");
    for optional in ["expected", "got", "span", "span_id"] {
        if let Some(value) = object.get(optional) {
            assert!(!value.is_null(), "`{optional}` must be omitted, not null");
        }
    }
}

#[test]
fn a_clean_document_reports_no_errors_and_succeeds() {
    // Score, error count, and exit status must agree (chelis#207/#731).
    let directory = tempdir().expect("tempdir");
    let path = directory.path().join("clean.ch");
    fs::write(&path, "def f(x: f32) -> f32 = add(x, x)\n").expect("write");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check"])
        .arg(&path)
        .output()
        .expect("run");
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).expect("JSON");
    assert!(output.status.success(), "a clean check exits zero");
    assert_eq!(
        report["errors"].as_array().map(Vec::len),
        Some(0),
        "a clean check reports no errors"
    );
}

#[test]
fn repeated_processes_emit_identical_bytes() {
    // Stable ordering and identities across fresh processes (chelis#1341).
    //
    // Compared as BYTES, not as parsed values: member order is part of the
    // claim, and `Value` equality would erase exactly the property under
    // test.
    let source = "def f(x: f32) -> f32 = add(x, nope)\n";
    let first = check_stdout_bytes(source);
    let second = check_stdout_bytes(source);
    assert_eq!(
        String::from_utf8_lossy(&first),
        String::from_utf8_lossy(&second),
        "the document must be byte-stable across fresh processes"
    );
}

#[test]
fn the_error_object_member_order_is_pinned_to_its_bytes() {
    // The executable form of "declaration order is wire order".
    //
    // Two fresh processes agreeing proves determinism but not ORDER -- both
    // would agree on a reordered document. This pins the exact bytes, so a
    // field moved in the carrier fails here instead of silently changing a
    // published wire contract.
    //
    // These bytes are NOT the ones the hand-assembled `format!` produced.
    // That template emitted neither `suggestions` (it had no slot for the
    // field at all) nor a range (`"span_offset":30` was the whole location),
    // so the error object changes shape in FOUR ways: `suggestions` appears
    // when non-empty, `span_offset` becomes `span` carrying an offset AND a
    // length, `suggestions` sits between `severity` and `span`, and effect
    // diagnostics carry their hints too. All four follow from [04-FIT-15]
    // and [04-FIT-16]; none is incidental.
    //
    // chelis#1395 adds a FIFTH: `span` is a tagged carrier, so a measured
    // range is `{"span":"range",...}`. This pin caught that change -- it
    // failed on the tag before the expectation was updated, which is the
    // guarantee working.
    //
    // It does NOT cover the `point` half. No CLI fixture here produces a
    // coordinate without an identity, so the measured/unmeasured distinction
    // is asserted at the carrier level in `schema.rs`'s unit module, not in
    // these bytes. Saying otherwise would repeat the overclaim this pin was
    // already corrected for once.
    //
    // The fourth arrived unnoticed because this pin covered only a CHECK
    // diagnostic, and the effect projection was the one dropping a field.
    // A review found it; this pin did not. Both classes are covered below,
    // so the guarantee the comment makes is now one the test can keep.
    let stdout = check_stdout_bytes("def f(x: f32) -> f32 = add(x, nope)\n");
    let rendered = String::from_utf8(stdout).expect("stdout is UTF-8");
    let errors_line = rendered
        .lines()
        .find(|line| line.contains("\"errors\""))
        .unwrap_or_else(|| panic!("no errors line; got:\n{rendered}"));
    assert_eq!(
        errors_line,
        "  \"errors\": [{\"kind\":\"UnboundVariable\",\"message\":\"unbound variable: nope\",\
         \"severity\":0.6,\"suggestions\":[\"Check spelling of 'nope'\"],\
         \"span\":{\"span\":\"range\",\"offset\":30,\"len\":4},\
         \"span_id\":\"surf:30..34\"}]",
        "the error object's member order and spelling are the wire contract"
    );

    // The effect class, byte-for-byte. `pipeline_artifact_semantic_reports_stay_exact`
    // compares through `serde_json::json!`, which erases member order, so
    // until now no test pinned the bytes of an effect diagnostic at all.
    let effect = check_stdout_bytes(
        "def noisy(x: tensor[3, f32]) -> tensor[3, f32] ! {} = dropout(x, 0.5f32)\n",
    );
    let effect_rendered = String::from_utf8(effect).expect("stdout is UTF-8");
    let effect_line = effect_rendered
        .lines()
        .find(|line| line.contains("\"errors\""))
        .unwrap_or_else(|| panic!("no errors line; got:\n{effect_rendered}"));
    assert_eq!(
        effect_line,
        "  \"errors\": [{\"kind\":\"UnhandledEffect\",\"message\":\"Function `noisy` is \
         declared with effects `{}` but its body performs effects `{Random}` that were \
         not declared\",\"severity\":0.8,\"suggestions\":[\"Either add the missing \
         effect(s) to the signature of `noisy` (e.g. `! { Random }`) or refactor the \
         body so it does not perform them.\"]}]",
        "an effect diagnostic's member order and spelling are the wire contract too"
    );
}
