//! Integration tests for the style gate wired into `chelis build`,
//! `chelis check`, `chelis validate`, and `chelis eval --file`.
//!
//! Each test writes a synthetic source file to a tempdir, runs the
//! relevant CLI subcommand, and asserts pass/fail behavior. Real
//! committed fixtures stay untouched — the gate runs in isolation.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::tempdir;

/// `chelis check` rejects a source whose formatting does not match the
/// canonical re-print.
#[test]
fn check_fails_on_non_canonical_surf_source() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("noncanonical.ch");
    fs::write(&path, "def foo() -> i32 = 1   \n").unwrap();
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicates::str::contains("not canonically formatted"));
}

/// `chelis check` rejects a source whose formatting is canonical but
/// which violates a registered lint rule.
#[test]
fn check_fails_on_lint_violation() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("leading_underscore.ch");
    // `_internal` parses cleanly but trips `surf-value-snake-case`
    // (§3.2 forbids leading underscore).
    let src = "def _internal() -> i32 = 1\n";
    let decls = chelis_surf::parser::parse_str(src).expect("parses");
    let canonical = chelis_surf::format::format_program(&decls);
    fs::write(&path, &canonical).unwrap();
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicates::str::contains("surf-value-snake-case"));
}

/// Package-level lint walks all Surf files, builds the opaque-type catalog,
/// and rejects direct construction from another module.
#[test]
fn lint_check_fails_on_cross_file_opaque_domain_construction() {
    let dir = tempdir().unwrap();
    fs::write(
        dir.path().join("whale.ch"),
        "module Whale.Types\n@opaque\ntype Probability =\n  | Probability { value: f32 }\n",
    )
    .unwrap();
    fs::write(
        dir.path().join("agent.ch"),
        "module Agent.Strategy\ndef bad(x: f32) -> Probability = Probability { value: x }\n",
    )
    .unwrap();
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["lint", "--check", dir.path().to_str().unwrap()])
        .assert()
        .failure()
        .stdout(predicates::str::contains("opaque-domain-construction"));
}

/// Blocking lint rules added to the registry also participate in the
/// built-in style gate.
#[test]
fn check_fails_on_no_em_dash_public_string() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("message.ch");
    let dash = '\u{2014}';
    let src = format!("def message() -> string = \"one {dash} two\"\n");
    let decls = chelis_surf::parser::parse_str(&src).expect("parses");
    let canonical = chelis_surf::format::format_program(&decls);
    fs::write(&path, &canonical).unwrap();
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicates::str::contains("no-em-dash-in-public-strings"));
}

/// Relative bare filenames must still match violations returned by
/// the lint walker, which reports paths rooted at `./`.
#[test]
fn check_relative_bare_filename_fails_on_lint_violation() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("message.ch");
    let dash = '\u{2014}';
    let src = format!("def message() -> string = \"one {dash} two\"\n");
    let decls = chelis_surf::parser::parse_str(&src).expect("parses");
    let canonical = chelis_surf::format::format_program(&decls);
    fs::write(&path, &canonical).unwrap();
    Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(dir.path())
        .args(["check", "message.ch"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("no-em-dash-in-public-strings"));
}

/// `--allow-style-violations` lets a non-canonical build through with a
/// stderr warning.
#[test]
fn check_passes_with_allow_style_violations() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("noncanonical.ch");
    fs::write(&path, "def foo() -> i32 = 1   \n").unwrap();
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", path.to_str().unwrap(), "--allow-style-violations"])
        .assert()
        // The check command may still fail on type/effect grounds, but
        // the gate-bypass message must appear on stderr regardless.
        .stderr(predicates::str::contains("--allow-style-violations"));
}

/// Style checking may canonicalize a separate representation for comparison,
/// but semantic diagnostics must still refer to the bytes the author supplied.
#[test]
fn check_bypass_reports_offsets_in_the_authored_noncanonical_source() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("authored_offsets.ch");
    let source = concat!(
        "def broken(\n",
        "  x: i32,\n",
        ") -> i32 =\n",
        "  missing(x,)\n",
    );
    fs::write(&path, source).unwrap();

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", path.to_str().unwrap(), "--allow-style-violations"])
        .output()
        .expect("run check");
    assert!(!output.status.success(), "unbound name must fail checking");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("--allow-style-violations"),
        "the test must exercise the noncanonical-source bypass"
    );

    let report: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("check emits a JSON report");
    let diagnostic = report["errors"]
        .as_array()
        .expect("errors array")
        .iter()
        .find(|error| {
            error["message"]
                .as_str()
                .is_some_and(|message| message.contains("missing"))
        })
        .expect("unbound-name diagnostic");
    // chelis#886 moved the coordinate into the schema carrier's `span`.
    // The claim is unchanged: the offset addresses the AUTHORED source, not
    // the canonical formatter output the style gate would have produced.
    assert_eq!(
        diagnostic["span"]["offset"].as_u64(),
        Some(source.find("missing").expect("fixture contains missing") as u64),
        "the diagnostic offset must address the authored source, not canonical formatter output"
    );
}

/// `chelis build` rejects a non-canonical Surf source before it hits
/// the front-end pipeline.
#[test]
fn build_fails_on_non_canonical_surf_source() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("noncanonical.ch");
    fs::write(&path, "def foo() -> i32 = 1   \n").unwrap();
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["build", "--emit-c", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicates::str::contains("not canonically formatted"));
}

/// `chelis build --allow-style-violations` lets the gate-bypass through.
#[test]
fn build_bypass_emits_warning_on_stderr() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("noncanonical.ch");
    fs::write(&path, "def foo() -> i32 = 1   \n").unwrap();
    Command::cargo_bin("chelis")
        .expect("binary")
        // Build without `--output` writes artifacts into the cwd; keep
        // them in the tempdir instead of the test binary's crate dir.
        .current_dir(dir.path())
        .args([
            "build",
            "--emit-c",
            "noncanonical.ch",
            "--allow-style-violations",
        ])
        .assert()
        .stderr(predicates::str::contains("style gate bypassed"));
}

/// `chelis eval --file` runs the gate; `chelis eval EXPR` does not.
#[test]
fn eval_file_runs_gate_but_eval_expr_does_not() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("noncanonical.ch");
    fs::write(&path, "def foo() -> i32 = 1   \n").unwrap();
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicates::str::contains("not canonically formatted"));

    // `eval EXPR` is a one-liner with no on-disk source; the gate must
    // not block it. The expression itself is trivial; we just check
    // that the gate-failure message is *absent*.
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "1 + 1"])
        .assert()
        .stderr(predicates::str::contains("not canonically formatted").not());
}

/// `chelis validate --surf` also runs the style gate before validation.
#[test]
fn validate_surf_fails_on_non_canonical_source() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("noncanonical.ch");
    fs::write(&path, "def foo() -> i32 = 1   \n").unwrap();
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["validate", "--surf", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicates::str::contains("not canonically formatted"));
}

/// `chelis validate --deep` runs Deep lint rules through the same gate.
#[test]
fn validate_deep_fails_on_deep_lint_violation() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bad_symbol.dp");
    fs::write(&path, "(def {} my-func (lit {} 1))\n").unwrap();
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["validate", "--deep", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicates::str::contains("deep-user-symbol-charset"));
}

/// The style gate enforces opaque domain construction discipline for
/// generated Deep as well as Surf.
#[test]
fn validate_deep_fails_on_opaque_domain_construction() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("opaque_forge.dp");
    fs::write(
        &path,
        "(module {} whale.types\n  (deftype {opaque: true} Probability () (variant {} Probability (field {} value (t-prim {} f32)))))\n(module {} agent.strategy\n  (def {} forge\n    (record {} Probability (kv {} value (lit {type: (t-prim {} f32)} 2.0)))))\n",
    )
    .unwrap();
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["validate", "--deep", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicates::str::contains("opaque-domain-construction"));
}

/// Untyped Deep record updates fail closed when opaque domain types are
/// in scope outside their defining module.
#[test]
fn validate_deep_fails_on_untyped_opaque_record_update() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("opaque_untyped_update.dp");
    fs::write(
        &path,
        "(module {} whale.types\n  (deftype {opaque: true} Probability () (variant {} Probability (field {} value (t-prim {} f32)))))\n(module {} agent.strategy\n  (def {} update\n    (record-update {} (var {} p) (kv {} value (lit {type: (t-prim {} f32)} 2.0)))))\n",
    )
    .unwrap();
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["validate", "--deep", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicates::str::contains("opaque-domain-construction"));
}

/// The validate gate can be bypassed with the same emergency flag.
#[test]
fn validate_deep_bypass_emits_warning_on_stderr() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("bad_symbol.dp");
    fs::write(&path, "(def {} my-func (lit {} 1))\n").unwrap();
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "validate",
            "--deep",
            path.to_str().unwrap(),
            "--allow-style-violations",
        ])
        .assert()
        .stderr(predicates::str::contains("style gate bypassed"));
}

/// Deep lint directives are comments in source but are ignored for the
/// canonical-format comparison so they can suppress style-gate lint
/// diagnostics without creating a formatting failure.
#[test]
fn validate_deep_allows_lint_directive_without_format_failure() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("allowed_dash.dp");
    let dash = '\u{2014}';
    fs::write(
        &path,
        format!(
            "; chelis-lint: allow no-em-dash-in-public-strings\n(def {{}} message (lit {{}} \"one {dash} two\"))\n"
        ),
    )
    .unwrap();
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["validate", "--deep", path.to_str().unwrap()])
        .assert()
        .success()
        .stderr(predicates::str::contains("not canonically formatted").not());
}

/// Deep lint directive stripping must not normalize unrelated bytes;
/// missing final newlines remain a formatting failure.
#[test]
fn validate_deep_still_rejects_missing_final_newline_after_directive_stripping() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("no_newline.dp");
    fs::write(&path, "(def {} value (lit {} 1))").unwrap();
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["validate", "--deep", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicates::str::contains("not canonically formatted"));
}

/// Deep lint directive stripping must preserve line endings on retained
/// source lines, so CRLF input does not pass the canonical LF gate.
#[test]
fn validate_deep_still_rejects_crlf_after_directive_stripping() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("crlf.dp");
    fs::write(&path, "(def {} value (lit {} 1))\r\n").unwrap();
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["validate", "--deep", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicates::str::contains("not canonically formatted"));
}

/// A `cargo` on `PATH` that records each invocation's arguments and fails, as
/// `cargo locate-project` does outside a workspace.
#[cfg(unix)]
fn recording_cargo(dir: &std::path::Path) -> (std::ffi::OsString, std::path::PathBuf) {
    use std::os::unix::fs::PermissionsExt;
    let bin = dir.join("bin");
    fs::create_dir_all(&bin).unwrap();
    let log = dir.join("cargo-invocations.log");
    let shim = bin.join("cargo");
    fs::write(
        &shim,
        format!("#!/bin/sh\necho \"$*\" >> '{}'\nexit 1\n", log.display()),
    )
    .unwrap();
    fs::set_permissions(&shim, fs::Permissions::from_mode(0o755)).unwrap();
    let mut path = std::ffi::OsString::from(&bin);
    path.push(":");
    path.push(std::env::var_os("PATH").unwrap_or_default());
    (path, log)
}

#[cfg(unix)]
fn recorded_locate_project_spawns(log: &std::path::Path) -> usize {
    fs::read_to_string(log)
        .unwrap_or_default()
        .lines()
        .filter(|line| line.starts_with("locate-project"))
        .count()
}

/// chelis#1329: the workspace probe anchors exception globs, and no
/// exception names a rule a `.ch` or `.dp` file can violate, so the style
/// gate and `check`'s advisory pass never spawn `cargo locate-project` for
/// one. `chelis lint --check` over a directory still probes once per walk,
/// which shows the recording shim is the `cargo` the CLI resolves.
#[cfg(unix)]
#[test]
fn style_gated_commands_spawn_no_workspace_probe() {
    let dir = tempdir().unwrap();
    let (path_env, log) = recording_cargo(dir.path());
    let surf = dir.path().join("clean.ch");
    fs::write(&surf, "def main() -> i32 = 1\n").unwrap();
    let deep = dir.path().join("clean.dp");
    fs::write(&deep, "(def {} value (lit {} 1))\n").unwrap();
    let flagged = dir.path().join("flagged.ch");
    let decls = chelis_surf::parser::parse_str("def _internal() -> i32 = 1\n").expect("parses");
    fs::write(&flagged, chelis_surf::format::format_program(&decls)).unwrap();

    let runs: [(&[&str], &std::path::Path, bool); 5] = [
        (&["check"], &surf, true),
        (&["eval", "--file"], &surf, true),
        (&["validate", "--surf"], &surf, true),
        (&["validate", "--deep"], &deep, true),
        (&["check"], &flagged, false),
    ];
    for (args, file, succeeds) in runs {
        let assert = Command::cargo_bin("chelis")
            .expect("binary")
            .env("PATH", &path_env)
            .env_remove("CHELIS_STYLE_GATE_DISABLE")
            .args(args)
            .arg(file)
            .assert();
        if succeeds {
            assert.success();
        } else {
            assert
                .failure()
                .stderr(predicates::str::contains("surf-value-snake-case"));
        }
        assert_eq!(
            recorded_locate_project_spawns(&log),
            0,
            "`chelis {} {}` spawned `cargo locate-project`",
            args.join(" "),
            file.display()
        );
    }

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("PATH", &path_env)
        .args(["lint", "--check"])
        .arg(dir.path())
        .assert()
        .failure()
        .stdout(predicates::str::contains("surf-value-snake-case"));
    assert_eq!(recorded_locate_project_spawns(&log), 1);
}
