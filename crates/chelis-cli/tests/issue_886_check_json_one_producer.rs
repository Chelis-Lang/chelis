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
    "unsupported_feature",
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
        "def g(x: f32) -> f32 = add(x, cast(1, i32))\n",
        "def k(x: i64) -> i64 = copy(x)\n",
        "def p() -> f32 = par { 1.0; 2.0 }\n",
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
    // field at all) nor a structured location (`"span_offset":30` was the
    // whole of it), so the error object changes shape in FOUR ways:
    // `suggestions` appears when non-empty, `span_offset` becomes the tagged
    // `span` carrier, `suggestions` sits between `severity` and `span`, and
    // effect diagnostics carry their hints too. All four follow from
    // [04-FIT-15] and [04-FIT-16]; none is incidental.
    //
    // chelis#1395 adds a FIFTH: `span` is a tagged carrier. This pin caught
    // that change -- it failed on the tag before the expectation was updated,
    // which is the guarantee working.
    //
    // The tag here is `point`, and that is the contract, not an accident of
    // this fixture. `span_id` is `surf:30..34`, and an earlier revision read
    // a length back out of that string. It is an opaque identity, not a
    // measured extent: `spec/03-deep-syntax.md` §1.1.1 makes span IDs opaque
    // whatever they are spelled like, so recovering `len` from one invents an
    // extent [04-FIT-17] forbids. A check diagnostic therefore reports a
    // coordinate, and a `range` on this line would be the defect.
    //
    // The carrier's OTHER half -- a measured range from a producer that has
    // one -- is the Deep stamp path, pinned in `chelis-compiler-api`'s
    // `phase3_stamped_ingress`, not in these bytes.
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
         \"span\":{\"span\":\"point\",\"offset\":30},\
         \"span_id\":\"surf:30..34\"}]",
        "the error object's member order and spelling are the wire contract"
    );

    // The effect class, byte-for-byte. `pipeline_artifact_semantic_reports_stay_exact`
    // compares through `serde_json::json!`, which erases member order, so
    // until now no test pinned the bytes of an effect diagnostic at all.
    let effect = check_stdout_bytes(
        "def noisy(x: tensor[3, f32]) -> tensor[3, f32] ! {} = { _ = print(x)\n x }\n",
    );
    let effect_rendered = String::from_utf8(effect).expect("stdout is UTF-8");
    let effect_line = effect_rendered
        .lines()
        .find(|line| line.contains("\"errors\""))
        .unwrap_or_else(|| panic!("no errors line; got:\n{effect_rendered}"));
    assert_eq!(
        effect_line,
        "  \"errors\": [{\"kind\":\"UnhandledEffect\",\"message\":\"Function `noisy` is \
         declared with effects `{}` but its body performs effects `{IO}` that were \
         not declared\",\"severity\":0.8,\"suggestions\":[\"Either add the missing \
         effect(s) to the signature of `noisy` (e.g. `! { IO }`) or refactor the \
         body so it does not perform them.\"]}]",
        "an effect diagnostic's member order and spelling are the wire contract too"
    );
}

/// The document with every number replaced by `N`.
///
/// Node counts move whenever the checker's walk changes, so pinning them
/// would make an unrelated change fail here for the wrong reason. The
/// LAYOUT is the published contract: member order, the two-space indent,
/// and which values sit on one line. This keeps the layout claim exact and
/// drops only the arithmetic, which other tests own.
fn document_shape(source: &str) -> Vec<String> {
    let rendered = String::from_utf8(check_stdout_bytes(source)).expect("stdout is UTF-8");
    rendered
        .lines()
        .map(|line| {
            let mut shape = String::new();
            let mut in_number = false;
            for character in line.chars() {
                let numeric = character.is_ascii_digit() || character == '.' || character == '-';
                if numeric && !shape.ends_with('"') {
                    if !in_number {
                        shape.push('N');
                        in_number = true;
                    }
                } else {
                    in_number = false;
                    shape.push(character);
                }
            }
            shape
        })
        .collect()
}

/// Run `chelis check` on a file with the given extension and contents.
fn check_output(extension: &str, source: &str, extra: &[&str]) -> std::process::Output {
    let directory = tempdir().expect("tempdir");
    let path = directory.path().join(format!("probe.{extension}"));
    fs::write(&path, source).expect("write fixture");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check"])
        .args(extra)
        .arg(&path)
        .output()
        .expect("run chelis check")
}

#[test]
fn the_document_layout_is_the_published_contract() {
    // [04-FIT-11]. The document used to be a `format!` template, so this
    // layout was whatever the string literal said; it is now whatever the
    // report type serializes to. Both must be the same bytes, because the
    // document is a reward surface, the hull conformance input, and a
    // downstream shell surface -- adopting a typed producer is not licence
    // to restyle it.
    assert_eq!(
        document_shape("def f(x: f32) -> f32 = add(x, x)\n"),
        vec![
            "{",
            "  \"score\": N,",
            "  \"components\": {",
            "    \"parse\": N,",
            "    \"structure\": N,",
            "    \"names\": N,",
            "    \"types\": N",
            "  },",
            "  \"typed_nodes\": N,",
            "  \"untyped_nodes\": N,",
            "  \"total_nodes\": N,",
            "  \"unresolved_names\": [],",
            "  \"errors\": []",
            "}",
        ],
        "the clean document's layout"
    );
}

#[test]
fn an_integral_score_is_not_respelled_as_a_double() {
    // The single highest-risk byte in this change. `serde_json` writes an
    // integral `f64` as `1.0`; the template it replaces used `format!`, so
    // `1` is what shipped, and 17 CLI test files assert the substring
    // `"score": 1`. A stock formatter passes every structural assertion in
    // this file and fails here.
    let rendered =
        String::from_utf8(check_stdout_bytes("def f(x: f32) -> f32 = add(x, x)\n")).expect("UTF-8");
    assert!(rendered.contains("\"score\": 1,"), "{rendered}");
    assert!(!rendered.contains("\"score\": 1.0"), "{rendered}");
}

#[test]
fn a_failure_before_the_checker_emits_the_same_document() {
    // The failures that short-circuit INSIDE the producer reach the report
    // type, not a display string and not a second producer that happens to
    // agree with the first.
    //
    // This is NOT [04-FIT-12] satisfied, and must not be read as it. §6.4
    // keeps its "not fully implemented" caveat.
    //
    // The REASON changed after this comment was written, so do not trust an
    // older reading of it. It used to be that an unreadable or non-UTF-8
    // `.ch`, a style-gate rejection, and directory mode all bypassed the
    // report and emitted a display string; chelis#1679 routed every one of
    // those through it. What survives is one level up: `chelis check <dir>`
    // on a directory it cannot enumerate fails in `discover_check_files`
    // before any file is reached, so there is no per-file report to produce.
    // That is the envelope surface, which spec/04 does not specify, and
    // chelis#1678 owns both it and the caveat's eventual removal.
    //
    // `check_output` sets `CHELIS_STYLE_GATE_DISABLE=1`, which makes one of
    // the five fixtures below unrepresentative of default CLI behaviour: a
    // whitespace-only `.ch` is rejected by the style gate first and never
    // reaches the producer. It is kept because it shares its code site with
    // the truly-empty case, which IS representative; four of the five hold
    // with the gate on.
    //
    // The `.dp` cases matter on their own: the Deep arm has its own
    // short-circuit sites, and a fix applied only to the Surf arm would
    // leave them outside the contract. The Deep arm used to be the stricter
    // of the two -- it reported an unreadable `.dp` while the Surf arm
    // bypassed -- which is what chelis#1679 converged, so the two now agree.
    for (extension, source, label) in [
        (
            "ch",
            "def f(x: f32) -> f32 = add(x,\n",
            "surf parse failure",
        ),
        ("ch", "", "empty surf program"),
        ("ch", "   \n\n", "whitespace-only surf program"),
        ("dp", "(this is not deep)\n", "deep parse failure"),
        ("dp", "", "empty deep program"),
    ] {
        let output = check_output(extension, source, &[]);
        let stdout = String::from_utf8(output.stdout).expect("stdout is UTF-8");
        let parsed: chelis_compiler_api::schema::WireCheckResult = serde_json::from_str(&stdout)
            .unwrap_or_else(|error| {
                panic!("{label} must emit a full report document; {error}\n{stdout}")
            });
        assert_eq!(
            parsed.errors.len(),
            1,
            "{label} reports exactly one diagnostic; got {stdout}"
        );
        let kind = &parsed.errors[0].kind;
        assert!(
            CHECK_KIND_SPELLINGS.contains(&kind.as_str()),
            "{label} must use a governed kind spelling, got {kind:?}"
        );
        // chelis#207/#731: a non-empty errors array and the exit status
        // agree on every path, including this one.
        assert_eq!(
            output.status.code(),
            Some(2),
            "{label} carries errors, so it must exit 2"
        );
        assert_eq!(
            document_shape_of(&stdout),
            vec![
                "{",
                "  \"score\": N,",
                "  \"components\": {",
                "    \"parse\": N,",
                "    \"structure\": N,",
                "    \"names\": N,",
                "    \"types\": N",
                "  },",
                "  \"typed_nodes\": N,",
                "  \"untyped_nodes\": N,",
                "  \"total_nodes\": N,",
                "  \"unresolved_names\": [],",
                "  \"errors\": [{...}]",
                "}",
            ],
            "{label} emits the same document shape as the checker's own path"
        );
    }
}

/// `document_shape`, for stdout already in hand, with the `errors` array
/// elided so the assertion is about the DOCUMENT rather than the
/// diagnostics its own tests pin.
fn document_shape_of(stdout: &str) -> Vec<String> {
    stdout
        .lines()
        .map(|line| {
            if let Some(prefix) = line.strip_suffix("]")
                && prefix.starts_with("  \"errors\": [{")
            {
                return "  \"errors\": [{...}]".to_string();
            }
            let mut shape = String::new();
            let mut in_number = false;
            for character in line.chars() {
                let numeric = character.is_ascii_digit() || character == '.' || character == '-';
                if numeric && !shape.ends_with('"') {
                    if !in_number {
                        shape.push('N');
                        in_number = true;
                    }
                } else {
                    in_number = false;
                    shape.push(character);
                }
            }
            shape
        })
        .collect()
}

#[test]
fn inferred_signatures_is_a_member_of_the_report_not_a_spliced_fragment() {
    // [04-FIT-13]. The rows used to be rendered to a string and spliced
    // between two literal keys of the template, which is why `CheckResult`
    // had no field for them and "move the outer object to `CheckResult`"
    // could not work as originally filed.
    let source = "def add_one(x: f32) -> f32 = add(x, x)\n";

    let without = check_output("ch", source, &[]);
    let parsed: chelis_compiler_api::schema::WireCheckResult =
        serde_json::from_slice(&without.stdout).expect("JSON");
    assert!(
        parsed.inferred_signatures.is_none(),
        "absent by omission when the caller did not ask"
    );
    assert!(
        !String::from_utf8_lossy(&without.stdout).contains("inferred_signatures"),
        "absent means absent, not null"
    );

    let with = check_output("ch", source, &["--show-inferred"]);
    let stdout = String::from_utf8(with.stdout).expect("UTF-8");
    let parsed: chelis_compiler_api::schema::WireCheckResult =
        serde_json::from_str(&stdout).expect("JSON");
    // `Vec`, not a `Value` that has to be re-checked for arrayness: the
    // carrier makes the sequence a sequence.
    let rows = parsed.inferred_signatures.expect("present when requested");
    assert_eq!(rows.len(), 1, "one def, one row: {stdout}");
    assert_eq!(rows[0].function, "add_one");

    // Position and single-line rendering are part of the document, not of
    // the rows: the member sits between `unresolved_names` and `errors`,
    // where the template spliced it.
    let lines: Vec<&str> = stdout.lines().collect();
    let index = lines
        .iter()
        .position(|line| line.starts_with("  \"inferred_signatures\": ["))
        .unwrap_or_else(|| panic!("no inferred_signatures line in:\n{stdout}"));
    assert!(lines[index].ends_with("],"), "the rows stay on one line");
    assert!(lines[index - 1].starts_with("  \"unresolved_names\":"));
    assert!(lines[index + 1].starts_with("  \"errors\":"));
}

#[test]
fn a_rejected_program_still_reports_the_requested_member() {
    // "Requested but empty" and "not requested" are different documents. A
    // consumer that asked for the rows and got none has been told there are
    // none; a consumer that did not ask has been told nothing.
    let output = check_output(
        "ch",
        "def f(x: f32) -> f32 = add(x, nope)\n",
        &["--show-inferred"],
    );
    let parsed: chelis_compiler_api::schema::WireCheckResult =
        serde_json::from_slice(&output.stdout).expect("JSON");
    assert!(
        parsed
            .inferred_signatures
            .as_ref()
            .is_some_and(Vec::is_empty),
        "the member is present and empty, not absent"
    );
    assert!(!parsed.errors.is_empty(), "the fixture must be rejected");
}

#[test]
fn the_cli_holds_no_second_producer_of_the_document() {
    // [04-FIT-11]'s structural half. Every other test here reads the
    // document's BYTES, and bytes cannot distinguish one producer from two
    // that agree -- which is precisely the state this issue describes: the
    // template and `CheckResult` emitted the same shape for as long as
    // someone kept them in step by hand.
    //
    // So this reads the source instead. The document's keys may not appear
    // as string literals in the CLI at all: their only spelling is the
    // report type's field names, and their only renderer is
    // `CheckResult::to_report_json`.
    //
    // It is a SPELLING HEURISTIC, not a proof, and should not be read as
    // one. It catches the two forms a template is actually written in; a
    // red-team pass got a byte-identical second producer past it using
    // `concat!` on split fragments, and another using a `const` array of
    // the keys with `{:?}`. A source grep cannot be complete, and chasing
    // each new spelling would be filling gaps rather than fixing a class.
    // What it buys is that the removed defect cannot come back by the route
    // it left by; the byte pins above are what catch a producer that
    // reappears some other way, by disagreeing with them.
    //
    // If you are here because you added a report field: add it to
    // `CheckResult`. If you are here because you need a document the type
    // cannot express, that is a change to the type, not a second template.
    //
    // Scope, stated so it is a known limit rather than an assumed one: this
    // reads `chelis-cli`'s `main.rs` only, because that is where both
    // removed templates lived and where a regression would most plausibly
    // reappear. A producer written into another crate would not be caught
    // here; the byte pins above are what would catch it, by disagreeing.
    let source = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/main.rs"))
        .expect("read the CLI source");
    for key in [
        "score",
        "components",
        "typed_nodes",
        "untyped_nodes",
        "total_nodes",
        "unresolved_names",
        "inferred_signatures",
        // `errors` is deliberately NOT in this list, and the reason is not
        // an oversight: `cmd_check`'s DIRECTORY mode emits a different
        // document, the envelope `{"files":[...],"errors":[]}`, whose own
        // `errors` key collides with the report's. Adding it here fails on
        // that envelope, which is not a second producer of the report. The
        // seven keys above are unique to the report, which is what makes them
        // usable as a signature. (The collision is with the two literal
        // forms this checks; `\"errors\": [` with a space would not match
        // the envelope. Excluded anyway -- the key is not distinctive.)
    ] {
        // BOTH quote forms. A red-team pass evaded the first version of this
        // check by writing the replacement template as a raw string literal
        // (`r#"..."#`), where the document's keys need no backslash and the
        // escaped spelling never appears. Checking only the form the removed
        // code happened to use is checking for a typo, not for a producer.
        for literal in [format!("\\\"{key}\\\""), format!("\"{key}\":")] {
            assert!(
                !source.contains(&literal),
                "`{literal}` is spelled as a string literal in the CLI source. \
                 The check report has one producer (`CheckResult::to_report_json`); \
                 a template that spells its keys is the defect chelis#886 removed."
            );
        }
    }
}
