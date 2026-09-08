//! The `check_snippet` wire document, pinned to its exact bytes.
//!
//! chelis#886's typed-producer step replaced a `format!` template with
//! `serde_json` over `SnippetReport`/`SnippetDiagnostic`. That is the right
//! move, but it relocates the wire contract: member names and order are now
//! the STRUCT DECLARATION ORDER, which an ordinary refactor changes silently.
//! Before this file, reordering those fields or renaming `kind` left every
//! suite in the workspace green (chelis-e2e 159 passed, chelis-deep 296
//! passed) while `py/src/chelis_tools/skill_eval.py` -- the only consumer --
//! broke.
//!
//! The `fitness` case is the one worth spelling out: `skill_eval` reads it as
//! `report.get("fitness", 0.0)`, so a renamed or reordered-away field does not
//! raise. It silently returns the default, every eval verdict flips to fail,
//! and it reads as a model regression rather than a wire break.
//!
//! This mirrors `chelis-cli`'s `issue_886_check_json_one_producer.rs`, whose
//! `the_error_object_member_order_is_pinned_to_its_bytes` makes the same
//! argument for the `chelis check` document one slice earlier in this issue.
use assert_cmd::Command;

fn wire(lang: &str, source: &str) -> String {
    let out = Command::cargo_bin("check_snippet")
        .expect("check_snippet binary")
        .args(["--lang", lang])
        .write_stdin(source.to_string())
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    String::from_utf8(out).expect("stdout is UTF-8")
}

#[test]
fn a_clean_snippet_reports_the_exact_wire_bytes() {
    assert_eq!(
        wire("surf", "def f(x: f32) -> f32 = add(x, x)\n").trim_end(),
        r#"{"lang":"surf","parse_error":null,"fitness":1.0,"warnings":[],"errors":[]}"#,
        "member order and spelling are the wire contract, not an artifact of \
         SnippetReport's field order"
    );
}

#[test]
fn a_warning_reports_the_exact_wire_bytes() {
    // `kind` before `message`, and the kind is a governed spelling rather than
    // a `{:?}` Rust identifier -- both are the point of this slice.
    let text = wire(
        "deep",
        "(module {} probe.wire (def {} f (bogustag {} 1)))\n",
    );
    let text = text.trim_end();
    assert!(
        text.contains(r#"{"kind":"UnknownTag","message":"#),
        "a diagnostic's member order and governed kind spelling are the wire \
         contract: {text}"
    );
    assert!(
        text.starts_with(r#"{"lang":"deep","parse_error":null,"fitness":"#),
        "the report's leading members are pinned too: {text}"
    );
}

#[test]
fn fitness_is_a_json_number_not_an_integer_spelling() {
    // `format!("{}", 1.0f64)` printed `1`; serde_json prints `1.0`. The
    // consumer compares `>= 0.9`, so both parse -- but the bytes changed, and
    // an unpinned wire is how that goes unnoticed twice.
    let text = wire("surf", "def f(x: f32) -> f32 = add(x, x)\n");
    assert!(
        text.contains(r#""fitness":1.0"#),
        "fitness must serialize as a float: {text}"
    );
}
