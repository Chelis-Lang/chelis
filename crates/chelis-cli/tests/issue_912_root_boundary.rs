//! Issue #912 — root boundary: existence, naming, lane, precision, inputs.
//!
//! Three bug repros (#848, #947, #820), the manifest-completeness property
//! (both lanes), precision determinism, and cohabitation independence.
//!
//! All tests are initially expected to fail (the manifest does not exist yet).
//! They turn green as the structural work lands.

use std::process::Command;

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
/// doesn't touch should evaluate successfully. Currently errors with
/// "missing required input `a` for symbolic dimension `k`".
#[test]
#[ignore] // TODO: enable once manifest + realizability land
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

/// A def returning an anonymous tuple should surface as `name.0`, `name.1`
/// in eval output. Currently produces empty roots for the arrow-form path
/// in standalone file eval. PR #946 fixed this for the reef/in-context path;
/// the standalone-file path still drops arrow-form value roots.
#[test]
#[ignore] // TODO: standalone-file arrow-form root surfacing
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

/// An f64-declared root must not produce a hard error under `chelis build
/// --target c` when a print is absent. Under the new design, f64 roots
/// route Host for the C target (tensor DAG can't deliver f64), so both
/// cases succeed.
#[test]
#[ignore] // TODO: enable once def-level precision check routes f64 → Host
fn precision_determinism_f64_root_routes_host_for_c_target() {
    let source_no_print = r#"
x = expand(scalar_to_tensor(cast(0.1, f64)), 0, 4)
y = mul(x, x)
"#;
    let source_with_print = r#"
x = expand(scalar_to_tensor(cast(0.1, f64)), 0, 4)
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

/// A root's numeric value must not change based on what other defs are in
/// the same file.
#[test]
#[ignore] // TODO: enable once per-def realizability is live
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
#[ignore] // TODO: enable once manifest exists and eval iterates it
fn manifest_completeness_eval_lane() {
    // This test will use --json to get the manifest entries and compare
    // against eval output. Stub until manifest machinery exists.
    todo!("implement once RootManifest is available via --json or similar");
}

/// For every entry in the root manifest, the C lane's combined output (build
/// stderr and artifact stdout) must contain either a rendered value or a
/// [05-UNS-1] diagnostic naming that root.
#[test]
#[ignore] // TODO: enable once manifest exists and C lane iterates it
fn manifest_completeness_c_lane() {
    todo!("implement once RootManifest is available");
}
