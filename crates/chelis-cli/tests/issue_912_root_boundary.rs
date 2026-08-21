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
        rerun: concat!(
            "cargo test -p chelis-cli --test issue_912_root_boundary",
            " -- --ignored manifest_completeness_eval_lane"
        ),
    },
    LedgerRow {
        test: "manifest_completeness_c_lane",
        blocked_by: "chelis#1079, as above. The C lane additionally needs                      `requires_main` threaded to a real caller: today all three                      `cmd_build_c_result` sites pass `None` and a source-text                      scan decides, which spec/05 forbids as [05-UNS-1]                      authority.",
        rerun: concat!(
            "cargo test -p chelis-cli --test issue_912_root_boundary",
            " -- --ignored manifest_completeness_c_lane"
        ),
    },
];

/// The one command shape a ledger row may carry, as a function of the cell name.
///
/// Derived rather than trusted: chelis#1084 review found that a `contains`
/// check on `--ignored` and the test name accepts `not-a-command --ignored
/// <test>`. A row that cannot be spelled any other way cannot drift.
fn canonical_rerun(test: &str) -> String {
    format!("cargo test -p chelis-cli --test issue_912_root_boundary -- --ignored {test}")
}

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
        assert_eq!(
            row.rerun,
            canonical_rerun(row.test),
            "ledger row `{}` must carry the exact command that re-runs that cell, \
             character for character",
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

/// A `chelis build --target c` run plus the C it actually emitted.
///
/// The emitted translation unit is the only observable artifact this target
/// produces - `build` has no `--json` or manifest flag - so a test that wants
/// to prove a root survived the build has to read `prog.c`. Asserting on exit
/// status alone cannot distinguish a retained root from a silently dropped one
/// (chelis#1084 review).
struct CBuild {
    success: bool,
    stderr: String,
    /// Contents of the emitted `prog.c`, empty if the build wrote none.
    emitted_c: String,
    /// Contents of the emitted `prog.h`, empty if the build wrote none.
    emitted_h: String,
}

fn build_c(source: &str) -> CBuild {
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
    let read = |name: &str| std::fs::read_to_string(dir.path().join(name)).unwrap_or_default();
    CBuild {
        success: out.status.success(),
        stderr: String::from_utf8_lossy(&out.stderr).to_string(),
        emitted_c: read("prog.c"),
        emitted_h: read("prog.h"),
    }
}

/// Build a throwaway Reef package whose `mylib` path dependency exports a
/// generic def over a symbolic dimension, with `entry_body` as the entry file.
///
/// chelis#848 is specifically about an *unused package import* dragging an
/// unrelated symbolic-input demand into eval, so the repro needs a real
/// package and a real import. Before the chelis#1084 review this cell used a
/// standalone file with a local dead `helper` def and no import at all, which
/// could have stayed green straight through a regression of the actual bug.
fn unused_import_package(entry_body: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("myapp");
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::create_dir_all(root.join("mylib/src")).unwrap();
    let version = env!("CARGO_PKG_VERSION");

    std::fs::write(
        root.join("reef.toml"),
        format!(
            "[package]\nname = \"myapp\"\nversion = \"0.1.0\"\n\
             compiler = \"={version}\"\nmodule_prefix = \"App\"\n\n\
             [dependencies]\nmylib = {{ path = \"./mylib\" }}\n"
        ),
    )
    .unwrap();
    std::fs::write(
        root.join("reef.lock"),
        format!(
            "[package]\nname = \"myapp\"\nversion = \"0.1.0\"\n\n\
             [[dependencies]]\nname = \"mylib\"\nversion = \"0.1.0\"\n\
             compiler = \"={version}\"\narchive_sha256 = \"\"\nshell_sha256 = \"\"\n\n\
             [dependencies.source]\nkind = \"path\"\npath = \"./mylib\"\n"
        ),
    )
    .unwrap();
    std::fs::write(
        root.join("mylib/reef.toml"),
        format!(
            "[package]\nname = \"mylib\"\nversion = \"0.1.0\"\n\
             compiler = \"={version}\"\nmodule_prefix = \"Mylib\"\n"
        ),
    )
    .unwrap();
    // The symbolic dimension `k` is the whole point: this is the def whose
    // input demand chelis#848 leaked into an unrelated root.
    std::fs::write(
        root.join("mylib/src/shapes.ch"),
        "module Mylib.Shapes\nexport (scale)\n\n\
         def scale(a: tensor[k, f32]) -> tensor[k, f32] = mul(a, a)\n",
    )
    .unwrap();

    let entry = root.join("src/probe.ch");
    std::fs::write(&entry, entry_body).unwrap();
    (dir, entry)
}

fn eval_in_package(entry: &std::path::Path) -> (String, String, bool) {
    let out = Command::new(chelis_bin())
        .args(["eval", "--file"])
        .arg(entry)
        .arg("--allow-style-violations")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    (stdout, stderr, out.status.success())
}

// ─── #848 repro: symbolic-dim library defs block eval ───────────────────────

/// Evaluating a root that does not touch an *imported* library def with
/// symbolic dims succeeds. On chelis#848 the unused import alone made eval
/// demand an unrelated input: `missing required input `a` for symbolic
/// dimension `k``. Un-ignored under chelis#1084; rebuilt onto a real package
/// import under that PR's review, because the earlier standalone-file shape
/// did not exercise the bug at all.
#[test]
fn issue_848_dead_library_symbolic_dims_do_not_block_eval() {
    let (_dir, entry) = unused_import_package(
        "module App.Probe\nimport Mylib.Shapes (scale)\n\nbench = cast(0.0, f32)\n",
    );
    let (stdout, stderr, success) = eval_in_package(&entry);
    assert!(
        success,
        "eval of an unused symbolic-dim import must succeed.\n\
         stdout: {stdout}\nstderr: {stderr}"
    );
    // The exact chelis#848 failure mode, named so a regression is unmistakable.
    assert!(
        !stderr.contains("missing required input"),
        "chelis#848 regression: the unused import demanded an input.\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("bench"),
        "the unrelated root `bench` must still surface.\nGot: {stdout}"
    );
}

/// Negative control for the cell above: the import it relies on must actually
/// resolve, or that cell would pass for the wrong reason.
///
/// Without this, a typo in the module name, a broken `reef.lock`, or a change
/// that stopped composing path dependencies would leave the #848 cell green
/// while testing nothing. A bogus import fails, so the sibling's success is
/// evidence that a real library was composed onto the program.
#[test]
fn issue_848_fixture_import_is_load_bearing() {
    let (_dir, entry) = unused_import_package(
        "module App.Probe\nimport Mylib.NotThere (scale)\n\nbench = cast(0.0, f32)\n",
    );
    let (stdout, stderr, success) = eval_in_package(&entry);
    assert!(
        !success,
        "an unresolvable import must fail, else the #848 cell proves nothing.\n\
         stdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stderr.contains("unresolved import"),
        "the failure must be the import itself.\nstderr: {stderr}"
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

/// An f64-declared root is retained by `chelis build --target c`, with and
/// without a print in the file.
///
/// Un-ignored under chelis#1084, then rewritten under that PR's review, which
/// found the cell asserted only two exit statuses - a build that silently
/// dropped the f64 root would have passed it.
///
/// The rewrite also corrected the cell's premise. It used to be named
/// `..._routes_host_for_c_target` and claimed f64 roots route Host "because
/// the tensor DAG cannot deliver f64". The emitted C says otherwise: the
/// no-print build hands `y` back as one of `prog()`'s two tensor outputs,
/// computed by a kernel over `double`. Retention is what this cell can honestly
/// assert; lane choice is not what happens.
#[test]
fn precision_determinism_f64_root_is_retained_by_c_build() {
    let source_no_print = r#"
x = expand(scalar_to_tensor(cast(0.1, f64)), 0, 4i64)
y = mul(x, x)
"#;
    let source_with_print = r#"
x = expand(scalar_to_tensor(cast(0.1, f64)), 0, 4i64)
y = mul(x, x)
z = print("host")
"#;
    let no_print = build_c(source_no_print);
    let with_print = build_c(source_with_print);

    assert!(
        no_print.success,
        "f64 root without print must build.\nstderr: {}",
        no_print.stderr
    );
    assert!(
        with_print.success,
        "f64 root with print must build.\nstderr: {}",
        with_print.stderr
    );

    // Without a print the program is emitted in library form, and BOTH roots
    // must come back as outputs. A dropped `y` shows up here as `1`.
    assert!(
        no_print.emitted_h.contains("void prog("),
        "no-print build must emit the library entry point.\nprog.h: {}",
        no_print.emitted_h
    );
    assert!(
        no_print
            .emitted_c
            .contains(r#"outputs are required\n", 2)"#),
        "both roots must be retained as outputs of `prog`; a dropped f64 root \
         leaves 1.\nprog.c: {}",
        no_print.emitted_c
    );

    // With a print the program is emitted as a host main, which renders roots
    // by name - so the root NAME is observable in this shape.
    assert!(
        with_print.emitted_c.contains(r#"printf("%s = ", "y")"#),
        "the host-main build must render root `y` by name.\nprog.c: {}",
        with_print.emitted_c
    );
}

// ─── Cohabitation independence ──────────────────────────────────────────────

/// A root's numeric value does not change based on what other defs are in the
/// same file.
///
/// Un-ignored under chelis#1084. The equality below compares two `Option`s, so
/// that PR's review reproduced a vacuous pass by making both inputs invalid:
/// neither lane rendered `y`, and `None == None` held. Both evals are now
/// required to succeed and to actually render the root before the values are
/// compared.
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
    let (out_alone, err_alone, ok_alone) = eval_file(source_alone);
    let (out_with, err_with, ok_with) = eval_file(source_with_print_def);
    assert!(
        ok_alone,
        "eval of the lone root failed.\nstderr: {err_alone}"
    );
    assert!(
        ok_with,
        "eval with a cohabiting def failed.\nstderr: {err_with}"
    );

    // Extract the y = ... line from each
    let y_alone = out_alone
        .lines()
        .find(|l| l.starts_with("y =") || l.contains("y ="));
    let y_with = out_with
        .lines()
        .find(|l| l.starts_with("y =") || l.contains("y ="));

    // Guard the comparison before making it: two absent roots are equal, and
    // that equality proves nothing about cohabitation.
    assert!(
        y_alone.is_some(),
        "root `y` did not render alone.\n{out_alone}"
    );
    assert!(
        y_with.is_some(),
        "root `y` did not render beside a print.\n{out_with}"
    );

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
