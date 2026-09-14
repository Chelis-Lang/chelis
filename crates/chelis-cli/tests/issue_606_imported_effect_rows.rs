//! chelis#606: `chelis check --show-inferred` must report the IO
//! effect a function inherits from an IMPORTED stdlib wrapper.
//!
//! The issue's August update localizes the defect precisely: effect
//! ENFORCEMENT was already right across the package boundary -- a caller of
//! `Std.Io.Json.load_json` that declares `! {}` is rejected -- but the
//! REPORTED `inferred_signatures[].effect_row` was computed by a second,
//! context-free pass over the consuming package alone. A callee defined in
//! `chelis-std` was an unknown name to that pass, so it contributed
//! nothing, and the report published an empty row for a function that reads
//! the host filesystem. A downstream purity gate built on that row was
//! unsound.
//!
//! spec/04-type-system.md §7.1 states one inferred effect set per function
//! -- the union of the effects of the operations in its body, inherited
//! through calls -- with no package-relative qualification, so the reported
//! row and the enforced row must be the same row.
//!
//! These run against a real reef package that depends on the published
//! `chelis-std`, because the boundary IS the defect: an in-memory fixture
//! with both sides in one program never reproduced it.
//!
//! Focused acceptance command:
//!
//! ```text
//! cargo nextest run -p chelis-cli --test issue_606_imported_effect_rows
//! ```
//!
//! Expected success condition: every test passes.

use assert_cmd::Command;
use chelis_compiler_api::schema::{WireCheckResult, WireInferredSignature};

#[path = "common/mod.rs"]
mod common;

use common::{make_app, write_file};

/// The issue's reproducer, plus the controls it asks for: a direct IO
/// builtin, the imported wrapper, a same-package helper that reaches IO
/// only through that import, and two pure functions.
///
/// `load_json` is `load_json -> try_load_json -> try_parse_json(read_file(
/// path))`, so every caller of it reads the host filesystem.
const PROBE: &str = r#"module Demo.Main

import Std.Io.Json (Json, load_json)

def direct_io(path: string) -> string = read_file(path)

def via_wrapper(path: string) -> Json = load_json(path)

def via_local_helper(path: string) -> Json = via_wrapper(path)

def pure_local(x: int64) -> int64 = add(x, 1i64)
"#;

/// Run `chelis check --show-inferred` over `source` in a fresh reef
/// package that depends on `chelis-std`.
fn check_json(slug: &str, source: &str) -> (WireCheckResult, std::process::Output) {
    let (_dir, reef_home, app_pkg) = make_app(slug);
    let entry = app_pkg.join("src/main.ch");
    write_file(&entry, source);
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        // `chelis check` has one JSON producer, so `--show-inferred` is
        // the whole request; the issue's `--json` is not a flag.
        .args(["check", entry.to_str().unwrap(), "--show-inferred"])
        .output()
        .expect("run chelis check");
    let stdout = String::from_utf8(output.stdout.clone()).expect("UTF-8 report");
    let parsed: WireCheckResult = serde_json::from_str(&stdout).unwrap_or_else(|error| {
        panic!(
            "`chelis check --show-inferred` must emit one JSON report; {error}\n\
             stdout:\n{stdout}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stderr)
        )
    });
    if parsed.inferred_signatures.is_none() {
        eprintln!("check report without inferred signatures:\n{stdout}");
    }
    (parsed, output)
}

/// The reef linker rewrites a def to a package-internal name, so a row is
/// matched on its authored terminal segment rather than on a whole
/// mangled spelling this test would otherwise have to predict.
fn row<'a>(rows: &'a [WireInferredSignature], name: &str) -> &'a WireInferredSignature {
    let mut matches = rows
        .iter()
        .filter(|row| row.function == name || row.function.rsplit("__").next() == Some(name));
    let found = matches
        .next()
        .unwrap_or_else(|| panic!("no inferred row for `{name}` in {:?}", names(rows)));
    assert!(
        matches.next().is_none(),
        "`{name}` matched more than one row in {:?}",
        names(rows)
    );
    found
}

fn names(rows: &[WireInferredSignature]) -> Vec<&str> {
    rows.iter().map(|row| row.function.as_str()).collect()
}

fn signatures(slug: &str, source: &str) -> Vec<WireInferredSignature> {
    let (parsed, output) = check_json(slug, source);
    assert!(output.status.success(), "check failed: {}", String::from_utf8_lossy(&output.stdout));
    assert!(parsed.errors.is_empty(), "{:?}", parsed.errors);
    parsed.inferred_signatures.unwrap_or_else(|| {
        panic!(
            "`--show-inferred` must produce the member; stderr:\n{}",
            String::from_utf8_lossy(&output.stderr)
        )
    })
}

/// The filed witness. Before the fix `via_wrapper` reported `[]` while
/// `direct_io` reported `['IO']`, at the same score.
#[test]
fn imported_wrapper_reports_the_io_it_performs() {
    let rows = signatures("issue-606-imported-wrapper", PROBE);

    assert_eq!(
        row(&rows, "direct_io").effect_row_display,
        vec!["IO".to_string()],
        "control: a direct `read_file` has always reported IO"
    );
    assert_eq!(
        row(&rows, "via_wrapper").effect_row_display,
        vec!["IO".to_string()],
        "`load_json` reads the host filesystem, so its caller carries IO \
         even though `load_json` is defined in another package"
    );
    assert_eq!(
        row(&rows, "via_local_helper").effect_row_display,
        vec!["IO".to_string()],
        "the effect must survive a same-package hop after the import"
    );
}

/// Negative parity: the fix must not stain everything with IO. A pure
/// function keeps a present-and-empty row, which is what lets a consumer
/// tell "pure" from "unknown".
#[test]
fn pure_functions_keep_an_empty_row_across_the_boundary() {
    let rows = signatures("issue-606-pure-control", PROBE);
    let pure_local = row(&rows, "pure_local");
    assert!(
        pure_local.effect_row.is_empty() && pure_local.effect_row_display.is_empty(),
        "`pure_local` performs no effect, got {:?}",
        pure_local.effect_row_display
    );
}

/// The report and the enforcement verdict must describe the same program.
/// An explicit `! {}` over the imported wrapper is rejected -- that always
/// worked -- and the SAME run must now report the non-empty row the
/// rejection was raised from, instead of contradicting it with `[]`.
#[test]
fn explicit_empty_declaration_is_rejected_and_the_report_agrees() {
    let source = r#"module Demo.Main

import Std.Io.Json (Json, load_json)

def claims_pure(path: string) -> Json ! { } = load_json(path)
"#;
    let (parsed, output) = check_json("issue-606-explicit-empty", source);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        !output.status.success(),
        "`! {{}}` over an imported IO wrapper must be rejected; stderr:\n{stderr}"
    );
    assert!(parsed.errors.iter().any(|error| {
        error.message.contains("declared with effects") && error.message.contains("IO")
    }), "wrong rejection: {:?}", parsed.errors);
    let rows = parsed
        .inferred_signatures
        .expect("a rejected check still emits the requested member");
    assert_eq!(
        row(&rows, "claims_pure").effect_row_display,
        vec!["IO".to_string()],
        "the row reported must be the row the rejection was raised from"
    );
}

/// The same declaration honestly written is accepted, and reports IO. This
/// is the pair to the rejection above: without it, a fix that rejected
/// every `! {...}` declaration would still look green.
#[test]
fn honest_io_declaration_is_accepted_and_reports_io() {
    let source = r#"module Demo.Main

import Std.Io.Json (Json, load_json)

def declares_io(path: string) -> Json ! { IO } = load_json(path)

def pure_local(x: int64) -> int64 ! { } = add(x, 1i64)
"#;
    let (parsed, output) = check_json("issue-606-honest-io", source);
    assert!(
        output.status.success(),
        "an honest `! {{ IO }}` must be accepted; stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let rows = parsed
        .inferred_signatures
        .expect("`--show-inferred` must produce the member");
    assert_eq!(
        row(&rows, "declares_io").effect_row_display,
        vec!["IO".to_string()]
    );
    assert!(row(&rows, "pure_local").effect_row_display.is_empty());
}
