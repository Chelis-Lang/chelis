//! chelis#2425: `chelis prove` runs its front end on the same grown stack
//! segment as `chelis check`, and the Deep parser rejects input nested deeper
//! than its own segment with a located diagnostic. Neither command aborts on a
//! deeply nested program, and both accept or reject it identically.

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Output;
use tempfile::tempdir;

/// A `main` whose body is `depth` nested applications of the identity.
fn nested_application_program(dir: &Path, depth: usize) -> PathBuf {
    let body = format!(
        "{}(lit {{type: (t-prim {{}} i32)}} 1){}",
        "(app {} (var {} id) ".repeat(depth),
        ")".repeat(depth)
    );
    let source = format!(
        "(defsig {{}} id (t-fn {{}} (t-prim {{}} i32) (t-prim {{}} i32)))\n\
         (def {{}} id (fn {{}} (params {{}} x) (var {{}} x)))\n\
         (defsig {{}} main (t-fn {{}} (t-prim {{}} i32)))\n\
         (def {{}} main (fn {{}} (params {{}}) {body}))\n"
    );
    let path = dir.join(format!("app_{depth}.dp"));
    fs::write(&path, source).expect("write program");
    path
}

fn run(command: &str, path: &Path) -> Output {
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([command, path.to_str().expect("utf8 path")])
        .output()
        .expect("run chelis")
}

/// Exit code, asserting the process exited rather than dying on a signal
/// (a stack overflow aborts with SIGABRT or SIGBUS and has no code).
fn exit_code(output: &Output, what: &str) -> i32 {
    output.status.code().unwrap_or_else(|| {
        panic!(
            "{what} died on a signal instead of exiting: {:?}\nstderr: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )
    })
}

/// The first diagnostic message in a `chelis check` report.
fn check_message(output: &Output) -> Option<String> {
    let report: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("check report JSON");
    report["errors"][0]["message"].as_str().map(str::to_string)
}

/// Depths `check` accepts must prove; depths it rejects must fail `prove`
/// with the same diagnostic.
#[test]
fn prove_agrees_with_check_on_deep_application_chains() {
    let dir = tempdir().expect("tempdir");
    for depth in [1_000, 4_000, 16_000] {
        let path = nested_application_program(dir.path(), depth);
        let check = run("check", &path);
        let check_code = exit_code(&check, "check");
        let prove = run("prove", &path);
        let prove_code = exit_code(&prove, "prove");
        match check_message(&check) {
            None => {
                assert_eq!(check_code, 0, "depth {depth}: check");
                assert_eq!(
                    prove_code,
                    0,
                    "depth {depth}: prove must accept what check accepts: {}",
                    String::from_utf8_lossy(&prove.stderr)
                );
            }
            Some(message) => {
                assert_ne!(prove_code, 0, "depth {depth}: prove must reject");
                let stderr = String::from_utf8_lossy(&prove.stderr);
                let stdout = String::from_utf8_lossy(&prove.stdout);
                assert!(
                    stderr.contains(&message) || stdout.contains(&message),
                    "depth {depth}: prove must report check's diagnostic `{message}`:\n\
                     {stdout}\n{stderr}"
                );
            }
        }
    }
}

/// Input deeper than the parser's segment is a located parse diagnostic in
/// both commands, at the same byte, never an abort.
#[test]
fn extreme_nesting_is_a_located_parse_diagnostic_in_check_and_prove() {
    let dir = tempdir().expect("tempdir");
    let path = nested_application_program(dir.path(), 400_000);

    let check = run("check", &path);
    assert_eq!(exit_code(&check, "check"), 2);
    let message = check_message(&check).expect("check reports a diagnostic");
    assert!(
        message.contains("Deep input nests deeper than the parser supports at byte "),
        "{message}"
    );

    let prove = run("prove", &path);
    assert_eq!(exit_code(&prove, "prove"), 3);
    let stderr = String::from_utf8_lossy(&prove.stderr);
    assert!(
        stderr.contains(&message),
        "prove must report check's located diagnostic `{message}`: {stderr}"
    );
}
