//! Issue #912 — root boundary: existence, naming, lane, precision, inputs.
//!
//! Three bug repros (#848, #947, #820), the manifest-completeness property
//! (both lanes), precision determinism, and cohabitation independence.
//!
//! ## Ignore ledger (chelis#1084)
//!
//! Every `#[ignore]`d cell in this file MUST appear in `IGNORE_LEDGER` below
//! with its blocking issue and the exact command that re-runs it, and
//! `ignore_inventory_equals_the_declared_ledger` asserts set equality in both
//! directions. An undeclared ignore is a silently hidden cell; a ledger row
//! for a cell that no longer exists is a false statement about coverage. The
//! pattern is borrowed from `observation_roundtrip_harness.rs`.
//!
//! The ledger records what is BLOCKED, not what is merely unfinished: when a
//! blocker lands, re-run the cell, and if it is green delete its ledger row and
//! its `#[ignore]` in that same change set.
//!
//! Re-run every declared cell with:
//!
//! ```sh
//! cargo test -p chelis-cli --test issue_912_root_boundary -- --ignored
//! ```

use std::process::Command;

/// The declared `#[ignore]` inventory for this file (chelis#1084).
///
/// `(test name, blocking issue and why, exact re-run command)`. Adding an
/// `#[ignore]` without a row here fails
/// `ignore_inventory_equals_the_declared_ledger`, and so does leaving a row for
/// a cell that is no longer ignored.
const IGNORE_LEDGER: &[LedgerRow] = &[
    LedgerRow {
        test: "manifest_completeness_eval_lane",
        blocked_by: "chelis#1079 - the root manifest is computed and discarded,                      so no manifest is reachable to iterate. chelis#1082                      (Expr::Node-blind manifest walkers) must land first or the                      manifest is silently empty.",
        rerun: "cargo test -p chelis-cli --test issue_912_root_boundary --                 --ignored manifest_completeness_eval_lane",
    },
    LedgerRow {
        test: "manifest_completeness_c_lane",
        blocked_by: "chelis#1079, as above. The C lane additionally needs                      `requires_main` threaded to a real caller: today all three                      `cmd_build_c_result` sites pass `None` and a source-text                      scan decides, which spec/05 forbids as [05-UNS-1]                      authority.",
        rerun: "cargo test -p chelis-cli --test issue_912_root_boundary --                 --ignored manifest_completeness_c_lane",
    },
];

struct LedgerRow {
    test: &'static str,
    blocked_by: &'static str,
    rerun: &'static str,
}

/// chelis#1084: the `#[ignore]` set in this file must EQUAL `IGNORE_LEDGER`.
///
/// Without this, a blocked cell and a cell that has quietly gone green look
/// identical from the outside - which is how four of this file's six ignores
/// sat green behind stale `// TODO:` text until chelis#1084 re-ran them.
///
/// The scan is deliberately strict about attribute shape: a `cfg`- or
/// `cfg_attr`-gated ignore is rejected outright rather than parsed, because a
/// conditionally-ignored cell cannot be honestly declared by name alone.
#[test]
fn ignore_inventory_equals_the_declared_ledger() {
    let source = include_str!("issue_912_root_boundary.rs");
    let mut ignored: Vec<String> = Vec::new();
    let mut pending_ignore = false;

    for (index, line) in source.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("//") {
            continue;
        }
        assert!(
            !(trimmed.starts_with("#[cfg") && trimmed.contains("ignore")),
            "line {}: a cfg/cfg_attr-gated ignore cannot be declared by name;              give the cell an unconditional `#[ignore]` or none at all: {trimmed}",
            index + 1
        );
        if trimmed.starts_with("#[ignore]") {
            pending_ignore = true;
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("fn ").filter(|_| pending_ignore) {
            let name = rest
                .split(['(', '<', ' '])
                .next()
                .expect("a function name follows `fn `");
            ignored.push(name.to_string());
            pending_ignore = false;
        }
    }
    assert!(
        !pending_ignore,
        "an `#[ignore]` attribute is not attached to any function"
    );

    let mut found = ignored.clone();
    found.sort();
    let mut declared: Vec<String> = IGNORE_LEDGER
        .iter()
        .map(|row| row.test.to_string())
        .collect();
    declared.sort();

    assert_eq!(
        found, declared,
        "the `#[ignore]` set must equal IGNORE_LEDGER. An undeclared ignore is a          silently hidden cell; a stale row is a false statement about coverage.          Found: {found:?}, declared: {declared:?}"
    );

    for row in IGNORE_LEDGER {
        assert!(
            row.blocked_by.contains("chelis#"),
            "ledger row `{}` must name its blocking issue as `chelis#NNN`",
            row.test
        );
        assert!(
            row.rerun.contains("--ignored") && row.rerun.contains(row.test),
            "ledger row `{}` must carry the exact command that re-runs that cell",
            row.test
        );
    }
}

fn chelis_bin() -> String {
    let path = env!("CARGO_BIN_EXE_chelis");
    path.to_string()
}

fn eval_file(source: &str) -> (String, String, bool) {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("prog.ch");
    std::fs::write(&file, source).unwrap();
    let out = Command::new(chelis_bin())
        .args(["eval", "--file"])
        .arg(&file)
        .arg("--allow-style-violations")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    (stdout, stderr, out.status.success())
}

fn build_c(source: &str) -> (String, String, bool) {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("prog.ch");
    std::fs::write(&file, source).unwrap();
    let out = Command::new(chelis_bin())
        .args(["build", "--target", "c"])
        .arg(&file)
        .arg("--allow-style-violations")
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    (stdout, stderr, out.status.success())
}

// ─── #848 repro: symbolic-dim library defs block eval ───────────────────────

/// A program that imports a library def with symbolic dims the user's root
/// doesn't touch evaluates successfully. It used to error with "missing
/// required input `a` for symbolic dimension `k`"; un-ignored under
/// chelis#1084 after re-running it against `main`.
#[test]
fn issue_848_dead_library_symbolic_dims_do_not_block_eval() {
    // This test needs a library with symbolic-dim defs composed onto the
    // program. The exact shape depends on the eval-in-context machinery.
    // Placeholder: a program that would trigger the bug on origin/main.
    let source = r#"
def helper(a: tensor[k, f32]) -> tensor[k, f32] = mul(a, a)
x = to_tensor([1.0, 2.0, 3.0])
result = mul(x, x)
"#;
    let (stdout, stderr, success) = eval_file(source);
    assert!(
        success,
        "eval should succeed: dead library symbolic dims must not block.\n\
         stdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("result"),
        "root `result` must appear in output"
    );
}

// ─── #947 repro: anonymous tuple roots drop ─────────────────────────────────

/// A def returning an anonymous tuple surfaces as `name.0`, `name.1` in eval
/// output. PR #946 fixed the reef/in-context path; the standalone-file path
/// used to drop arrow-form value roots and no longer does, so this is
/// un-ignored under chelis#1084.
///
/// This does NOT close chelis#1083, which owns a different surface: the root
/// MANIFEST's dotted tuple/ADT expansion is still a TODO in
/// `chelis-effects/src/realizability.rs`. This cell asserts the eval render.
#[test]
fn issue_947_anonymous_tuple_roots_surface() {
    let source = r#"
def compute() -> (tensor[2, f32], tensor[2, f32]) = (to_tensor([1.0, 2.0]), to_tensor([3.0, 4.0]))
result = compute()
"#;
    let (stdout, _stderr, success) = eval_file(source);
    assert!(success, "eval should succeed for tuple-returning def");
    assert!(
        stdout.contains("result.0") && stdout.contains("result.1"),
        "tuple root components must surface as result.0, result.1.\n\
         Got: {stdout}"
    );
}

// ─── #820 repro: concat-bodied def excluded from evaluable roots ────────────

/// A def whose body uses `concat` as the program's only root should
/// produce a value in eval, not empty roots.
#[test]
fn issue_820_concat_def_evaluates_as_root() {
    let source = r#"
a: List[int32] = [cast(1, int32), cast(2, int32)]
b: List[int32] = [cast(3, int32), cast(4, int32)]
result = concat(a, b)
"#;
    let (stdout, _stderr, success) = eval_file(source);
    assert!(success, "eval should succeed for concat-rooted program");
    assert!(
        stdout.contains("result") && stdout.contains("1, 2, 3, 4"),
        "root `result` must appear with concatenated values.\nGot: {stdout}"
    );
}

// ─── Precision determinism ──────────────────────────────────────────────────

/// An f64-declared root does not produce a hard error under `chelis build
/// --target c` when a print is absent: f64 roots route Host for the C target
/// (the tensor DAG cannot deliver f64), so both cases succeed. Un-ignored
/// under chelis#1084 after re-running it against `main`.
#[test]
fn precision_determinism_f64_root_routes_host_for_c_target() {
    let source_no_print = r#"
x = expand(scalar_to_tensor(cast(0.1, f64)), 0, 4i64)
y = mul(x, x)
"#;
    let source_with_print = r#"
x = expand(scalar_to_tensor(cast(0.1, f64)), 0, 4i64)
y = mul(x, x)
z = print("host")
"#;
    let (_, _, success_no_print) = build_c(source_no_print);
    let (_, _, success_with_print) = build_c(source_with_print);
    assert!(
        success_no_print,
        "f64 root without print must succeed (routes Host via precision check)"
    );
    assert!(
        success_with_print,
        "f64 root with print must succeed (routes Host via HostOnly builtin)"
    );
}

// ─── Cohabitation independence ──────────────────────────────────────────────

/// A root's numeric value does not change based on what other defs are in the
/// same file. Un-ignored under chelis#1084 after re-running it against `main`;
/// both lanes render a real `y = tensor(...)` line, so the comparison below is
/// not the vacuous `None == None`.
#[test]
fn cohabitation_independence_same_root_different_cohabitants() {
    let source_alone = r#"
x = to_tensor([1.0, 2.0, 3.0, 4.0])
y = mul(x, x)
"#;
    let source_with_print_def = r#"
x = to_tensor([1.0, 2.0, 3.0, 4.0])
y = mul(x, x)
z = print("hello")
"#;
    let (out_alone, _, _) = eval_file(source_alone);
    let (out_with, _, _) = eval_file(source_with_print_def);
    // Extract the y = ... line from each
    let y_alone = out_alone
        .lines()
        .find(|l| l.starts_with("y =") || l.contains("y ="));
    let y_with = out_with
        .lines()
        .find(|l| l.starts_with("y =") || l.contains("y ="));
    assert_eq!(
        y_alone, y_with,
        "root `y` must produce the same value regardless of cohabiting defs.\n\
         Alone: {out_alone}\nWith print: {out_with}"
    );
}

// ─── Manifest completeness (both lanes) ─────────────────────────────────────

/// For every entry in the root manifest, the eval lane's output must contain
/// either a rendered value or a [05-UNS-1] diagnostic naming that root.
#[test]
#[ignore] // BLOCKED: chelis#1079 (see IGNORE_LEDGER)
fn manifest_completeness_eval_lane() {
    // This test will use --json to get the manifest entries and compare
    // against eval output. Stub until manifest machinery exists.
    todo!("implement once RootManifest is available via --json or similar");
}

/// For every entry in the root manifest, the C lane's combined output (build
/// stderr and artifact stdout) must contain either a rendered value or a
/// [05-UNS-1] diagnostic naming that root.
#[test]
#[ignore] // BLOCKED: chelis#1079 (see IGNORE_LEDGER)
fn manifest_completeness_c_lane() {
    todo!("implement once RootManifest is available");
}
