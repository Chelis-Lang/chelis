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
//! The ledger records what is BLOCKED, not what is merely unfinished. The
//! manifest-completeness cells are active; the empty ledger proves no cell is
//! still hidden behind the former #1079/#1082 blockers.
//!
//! Re-run every declared cell with:
//!
//! ```sh
//! cargo test -p chelis-cli --test issue_912_root_boundary -- --ignored
//! ```

use std::process::Command;

mod common;

/// The declared `#[ignore]` inventory for this file (chelis#1084).
///
/// `(test name, blocking issue and why, exact re-run command)`. Adding an
/// `#[ignore]` without a row here fails
/// `ignore_inventory_equals_the_declared_ledger`, and so does leaving a row for
/// a cell that is no longer ignored.
const IGNORE_LEDGER: &[LedgerRow] = &[];

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

fn eval_json_file(source: &str, target: Option<&str>) -> (serde_json::Value, String, bool) {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("prog.ch");
    std::fs::write(&file, source).unwrap();
    let mut command = Command::new(chelis_bin());
    command
        .args(["eval", "--file"])
        .arg(&file)
        .args(["--json", "--allow-style-violations"]);
    if let Some(target) = target {
        command.args(["--target", target]);
    }
    let out = command.output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let json = serde_json::from_str(&stdout).unwrap_or_else(|error| {
        panic!(
            "eval --json must emit one JSON document: {error}\nstdout: {stdout}\nstderr: {stderr}"
        )
    });
    (json, stderr, out.status.success())
}

/// A `chelis build --target c` run plus the C it actually emitted.
///
/// The emitted translation unit is the only observable artifact this target
/// produces - `build` has no `--json` or manifest flag - so a test that wants
/// to prove a root survived the build has to read `prog.c`. Asserting on exit
/// status alone cannot distinguish a retained root from a silently dropped one
/// (chelis#1084 review).
struct CBuild {
    dir: tempfile::TempDir,
    success: bool,
    stderr: String,
    /// Contents of the emitted `prog.c`, empty if the build wrote none.
    emitted_c: String,
}

fn build_c(source: &str) -> CBuild {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("prog.ch");
    std::fs::write(&file, source).unwrap();
    let out = Command::new(chelis_bin())
        .args(["build", "--emit-c", "--target", "c"])
        .arg(&file)
        .arg("--allow-style-violations")
        .current_dir(dir.path())
        .output()
        .unwrap();
    let read = |name: &str| std::fs::read_to_string(dir.path().join(name)).unwrap_or_default();
    let emitted_c = read("prog.c");
    CBuild {
        dir,
        success: out.status.success(),
        stderr: String::from_utf8_lossy(&out.stderr).to_string(),
        emitted_c,
    }
}

impl CBuild {
    fn link_and_run(&self) -> std::process::Output {
        let status = common::link_generated(self.dir.path(), "prog.c", "prog");
        assert!(
            status.success(),
            "generated C must link as an executable when the manifest requires main"
        );
        Command::new(self.dir.path().join("prog"))
            .output()
            .expect("run generated C artifact")
    }
}

struct HipBuild {
    _dir: tempfile::TempDir,
    success: bool,
    stderr: String,
    emitted_cpp: String,
}

fn build_hip(source: &str) -> HipBuild {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("prog.ch");
    std::fs::write(&file, source).unwrap();
    let out = Command::new(chelis_bin())
        .args(["build", "--emit-c", "--target", "hip"])
        .arg(&file)
        .arg("--allow-style-violations")
        .current_dir(dir.path())
        .output()
        .unwrap();
    let emitted_cpp = std::fs::read_to_string(dir.path().join("prog_hip.cpp")).unwrap_or_default();
    HipBuild {
        _dir: dir,
        success: out.status.success(),
        stderr: String::from_utf8_lossy(&out.stderr).to_string(),
        emitted_cpp,
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
         def scale[k](a: tensor[k, f32]) -> tensor[k, f32] = mul(a, a)\n",
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
/// This remains the standalone eval-render regression for chelis#947. The
/// separate chelis#1083 manifest-topology contract is covered below by the
/// completeness cells and in compiler-API selected-callable tests.
#[test]
fn issue_947_anonymous_tuple_roots_surface() {
    let source = r#"
def compute() -> (tensor[2, f32], tensor[2, f32]) = (to_tensor([1.0, 2.0], f32), to_tensor([3.0, 4.0], f32))
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
a: List[i32] = [cast(1, i32), cast(2, i32)]
b: List[i32] = [cast(3, i32), cast(4, i32)]
result = concat(a, b)
"#;
    let (stdout, stderr, success) = eval_file(source);
    assert!(
        success,
        "eval should succeed for concat-rooted program.\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("result") && stdout.contains("1, 2, 3, 4"),
        "root `result` must appear with concatenated values.\nGot: {stdout}"
    );
}

// ─── Precision determinism ──────────────────────────────────────────────────

/// An f64-declared root is routed Host for the C target and Tensor for eval,
/// and a root-bearing C build emits an executable main with or without print.
///
/// Un-ignored under chelis#1084, then rewritten under that PR's review, which
/// found the cell asserted only two exit statuses - a build that silently
/// dropped the f64 root would have passed it.
///
#[test]
fn precision_determinism_f64_root_is_retained_by_c_build() {
    let source_no_print = r#"
x = insert(scalar_to_tensor(cast(0.1, f64)), 0, 4i64)
y = mul(x, x)
"#;
    let source_with_print = r#"
x = insert(scalar_to_tensor(cast(0.1, f64)), 0, 4i64)
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

    let (c_manifest, c_stderr, c_eval_ok) = eval_json_file(source_no_print, Some("c"));
    assert!(c_eval_ok, "C-target eval must succeed: {c_stderr}");
    let (eval_manifest, eval_stderr, eval_ok) = eval_json_file(source_no_print, None);
    assert!(eval_ok, "default eval must succeed: {eval_stderr}");
    let lane_for = |json: &serde_json::Value, name: &str| {
        json["manifest"]["entries"]
            .as_array()
            .expect("manifest entries")
            .iter()
            .find(|entry| entry["name"] == name)
            .and_then(|entry| entry["lane"].as_str())
            .map(str::to_string)
            .unwrap_or_else(|| panic!("manifest must contain {name}: {json}"))
    };
    assert_eq!(lane_for(&c_manifest, "y"), "Host");
    assert_eq!(lane_for(&eval_manifest, "y"), "Tensor");

    // A non-empty manifest requires a real executable entry point. This is
    // selected from the manifest before emission, never by scanning C text.
    assert!(
        no_print.emitted_c.contains("int main(void)"),
        "root-bearing build must emit main.\nprog.c: {}",
        no_print.emitted_c
    );
    assert!(
        no_print.emitted_c.contains(r#"printf("%s = ", "y")"#),
        "the no-print build must render root `y` by name.\nprog.c: {}",
        no_print.emitted_c
    );

    // With a print the program is emitted as a host main, which renders roots
    // by name - so the root NAME is observable in this shape.
    assert!(
        with_print.emitted_c.contains(r#"printf("%s = ", "y")"#),
        "the host-main build must render root `y` by name.\nprog.c: {}",
        with_print.emitted_c
    );

    // Mutation control for the Tensor-only C path: the f64 cases above route
    // Host, whose backend also needs a main for unrelated reasons. A false
    // `RootManifest::requires_main()` must therefore still be killed here,
    // where no host-backend condition can accidentally supply the driver.
    let tensor_only = build_c(
        "x = insert(scalar_to_tensor(cast(0.1, f32)), 0, 4i64)\n\
         y = mul(x, x)\n",
    );
    assert!(
        tensor_only.success,
        "Tensor-only f32 root must build.\nstderr: {}",
        tensor_only.stderr
    );
    assert!(
        tensor_only.emitted_c.contains("int main(void)"),
        "a Tensor-only non-empty C manifest must emit main.\nprog.c: {}",
        tensor_only.emitted_c
    );
    assert!(
        tensor_only.emitted_c.contains(r#"printf("y = ")"#)
            || tensor_only.emitted_c.contains(r#"printf("%s = ", "y")"#),
        "the Tensor-only C driver must render `y` by manifest name.\nprog.c: {}",
        tensor_only.emitted_c
    );
    let tensor_output = tensor_only.link_and_run();
    assert!(
        tensor_output.status.success(),
        "the Tensor-only manifest driver must run: {}",
        String::from_utf8_lossy(&tensor_output.stderr)
    );
    let tensor_stdout = String::from_utf8_lossy(&tensor_output.stdout);
    assert!(
        tensor_stdout.contains("x = tensor(") && tensor_stdout.contains("y = tensor("),
        "the Tensor-only manifest driver must realize both owed roots: {tensor_stdout}"
    );
}

/// A pure nullary definition is an owed Host-lane root for C even when the
/// program has no top-level value bindings. The manifest, not the legacy
/// `host_program.globals` shape, therefore has to create the executable
/// observation boundary.
#[test]
fn pure_nullary_definition_is_an_executable_c_root() {
    let source = "def answer() -> i32 = cast(42, i32)\n";
    let (manifest, stderr, eval_ok) = eval_json_file(source, Some("c"));
    assert!(eval_ok, "C-target manifest probe failed: {stderr}");
    let answer = manifest["manifest"]["entries"]
        .as_array()
        .expect("manifest entries")
        .iter()
        .find(|entry| entry["name"] == "answer")
        .expect("answer manifest entry");
    assert_eq!(answer["lane"], "Host");
    assert_eq!(manifest["manifest"]["requires_main"], true);

    let build = build_c(source);
    assert!(build.success, "C build failed: {}", build.stderr);
    assert!(
        build.emitted_c.contains("int main(void)"),
        "a pure nullary owed root must emit main.\nprog.c: {}",
        build.emitted_c
    );
    assert!(
        build.emitted_c.contains(r#"printf("%s = ", "answer")"#),
        "the generated main must render the manifest root by name.\nprog.c: {}",
        build.emitted_c
    );
    let output = build.link_and_run();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "generated C failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(stdout.trim(), "answer = 42");

    let parameterized = build_c("def identity(x: i32) -> i32 = x\n");
    assert!(
        parameterized.success,
        "an unselected parameterized declaration must still build as an object: {}",
        parameterized.stderr
    );
    assert!(
        !parameterized.emitted_c.contains("int main(void)"),
        "a parameterized declaration is not an automatically owed root.\nprog.c: {}",
        parameterized.emitted_c
    );
}

/// Unwrapping a pure nullary declaration for automatic observation must feed
/// its concrete result body to the shared tuple/ADT topology expander. The
/// function wrapper itself carries no statically selected ADT constructor.
#[test]
fn pure_nullary_tuple_and_record_roots_expand_and_execute() {
    let source = "type Pair = | Pair { left: i32, right: i32 }\n\
                  def tupled() -> (i32, i32) = (cast(1, i32), cast(2, i32))\n\
                  def answer() -> Pair = Pair { left: cast(3, i32), right: cast(4, i32) }\n";
    let (manifest, stderr, eval_ok) = eval_json_file(source, Some("c"));
    assert!(eval_ok, "nullary product manifest probe failed: {stderr}");
    let manifest_names = manifest["manifest"]["entries"]
        .as_array()
        .expect("manifest entries")
        .iter()
        .map(|entry| entry["name"].as_str().expect("manifest entry name"))
        .collect::<Vec<_>>();
    let root_names = manifest["roots"]
        .as_array()
        .expect("realized roots")
        .iter()
        .map(|root| root["name"].as_str().expect("realized root name"))
        .collect::<Vec<_>>();
    let expected = ["tupled.0", "tupled.1", "answer.left", "answer.right"];
    assert_eq!(manifest_names, expected);
    assert_eq!(root_names, expected);

    let build = build_c(source);
    assert!(build.success, "C build failed: {}", build.stderr);
    let output = build.link_and_run();
    assert!(
        output.status.success(),
        "generated C failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "tupled.0 = 1\ntupled.1 = 2\nanswer.left = 3\nanswer.right = 4\n"
    );
}

/// [05-OBS-11]: realization follows manifest order. Synthetic Host bindings
/// for pure nullary roots must not introduce their own alphabetical order.
#[test]
fn pure_nullary_host_roots_preserve_manifest_order() {
    let source = "def zed() -> i32 = cast(1, i32)\n\
                  def alpha() -> i32 = cast(2, i32)\n";
    let (manifest, stderr, eval_ok) = eval_json_file(source, Some("c"));
    assert!(eval_ok, "C-target manifest probe failed: {stderr}");
    let names = manifest["manifest"]["entries"]
        .as_array()
        .expect("manifest entries")
        .iter()
        .filter_map(|entry| entry["name"].as_str())
        .collect::<Vec<_>>();
    assert_eq!(names, ["zed", "alpha"], "manifest source order");

    let build = build_c(source);
    assert!(build.success, "C build failed: {}", build.stderr);
    let output = build.link_and_run();
    assert!(
        output.status.success(),
        "generated C failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "zed = 1\nalpha = 2\n",
        "compiled realization must preserve manifest order"
    );
}

/// Existing value bindings and compiler-owned nullary observations share one
/// manifest order; appending either class after the other is non-conforming.
#[test]
fn nullary_and_value_host_roots_preserve_manifest_order() {
    let source = "def zed() -> i32 = cast(1, i32)\n\
                  alpha = cast(2, i32)\n";
    let (manifest, stderr, eval_ok) = eval_json_file(source, Some("c"));
    assert!(eval_ok, "C-target manifest probe failed: {stderr}");
    let names = manifest["manifest"]["entries"]
        .as_array()
        .expect("manifest entries")
        .iter()
        .filter_map(|entry| entry["name"].as_str())
        .collect::<Vec<_>>();
    assert_eq!(names, ["zed", "alpha"], "manifest source order");

    let build = build_c(source);
    assert!(build.success, "C build failed: {}", build.stderr);
    let output = build.link_and_run();
    assert!(
        output.status.success(),
        "generated C failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "zed = 1\nalpha = 2\n",
        "compiled realization must preserve mixed-root manifest order"
    );
}

/// Reordering represented roots must not move an unmanifested lowering
/// dependency after the Host binding that consumes it.
#[test]
fn host_manifest_order_preserves_lowering_dependencies() {
    let source = "rows: List[List[i64]] = [[cast(1, i64)], [cast(2, i64)]]\n\
                  padded = pad_sequences(rows, cast(0, i64))\n\
                  report = to_string(shape(padded, 0))\n\
                  shown = print(report)\n";
    let build = build_c(source);
    assert!(
        build.success,
        "manifest ordering must retain Host dependency order: {}",
        build.stderr
    );
    let output = build.link_and_run();
    assert!(
        output.status.success(),
        "generated C failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("report = 2"),
        "the dependent root must realize from the earlier padded tensor"
    );
}

#[test]
fn hip_tensor_root_uses_the_manifest_to_emit_a_gpu_executable() {
    let source = r#"
x = insert(scalar_to_tensor(cast(0.1, f64)), 0, 4i64)
y = mul(x, x)
"#;
    let (manifest, stderr, eval_ok) = eval_json_file(source, Some("hip"));
    assert!(eval_ok, "HIP-target manifest probe failed: {stderr}");
    let y = manifest["manifest"]["entries"]
        .as_array()
        .expect("manifest entries")
        .iter()
        .find(|entry| entry["name"] == "y")
        .expect("y manifest entry");
    assert_eq!(y["lane"], "Tensor");
    assert!(manifest["manifest"]["requires_main"] == true);

    let build = build_hip(source);
    assert!(build.success, "HIP build failed: {}", build.stderr);
    assert!(
        build
            .emitted_cpp
            .contains("#include \"chelis_hip_runtime.h\""),
        "a Tensor-lane HIP root must not be silently rerouted through CPU host C"
    );
    assert!(
        build.emitted_cpp.contains("int main(void)"),
        "a non-empty HIP manifest must emit an executable entry point"
    );
    assert!(
        build.emitted_cpp.contains(r#"printf("y = ")"#),
        "the HIP driver must render the owed root by manifest name"
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
x = to_tensor([1.0, 2.0, 3.0, 4.0], f32)
y = mul(x, x)
"#;
    let source_with_print_def = r#"
x = to_tensor([1.0, 2.0, 3.0, 4.0], f32)
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
fn manifest_completeness_eval_lane() {
    let source = r#"
type Pair = | Pair { left: i32, right: i32 }
scalar_root = cast(7, i32)
tensor_root = insert(scalar_to_tensor(cast(0.25, f32)), 0, 2i64)
tuple_root = (cast(1, i32), (cast(2, i32), cast(3, i32)))
record_root = Pair { left: cast(4, i32), right: cast(5, i32) }
"#;
    let (json, stderr, success) = eval_json_file(source, None);
    assert!(success, "eval completeness probe failed: {stderr}");
    let manifest_names = json["manifest"]["entries"]
        .as_array()
        .expect("manifest entries")
        .iter()
        .map(|entry| entry["name"].as_str().expect("manifest entry name"))
        .collect::<Vec<_>>();
    let root_names = json["roots"]
        .as_array()
        .expect("evaluated roots")
        .iter()
        .map(|root| root["name"].as_str().expect("evaluated root name"))
        .collect::<Vec<_>>();
    assert_eq!(
        root_names, manifest_names,
        "every owed root must be realized in manifest order"
    );
    assert!(manifest_names.contains(&"tuple_root.1.0"));
    assert!(manifest_names.contains(&"record_root.left"));
    let lanes = json["manifest"]["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["lane"].as_str().unwrap())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(lanes, ["Host", "Tensor"].into_iter().collect());
}

/// A dynamic tensor dimension is a runtime property of an already concrete
/// value root, not an uninstantiated generic declaration. It therefore stays
/// in both the target-aware manifest and the realized root sequence.
#[test]
fn manifest_completeness_keeps_dynamic_shape_value_roots() {
    let source = "rows: List[List[i64]] = [[cast(1, i64)], [cast(2, i64)]]\n\
                  padded = pad_sequences(rows, cast(0, i64))\n";
    let (json, stderr, success) = eval_json_file(source, Some("c"));
    assert!(success, "dynamic-shape manifest probe failed: {stderr}");
    let manifest_names = json["manifest"]["entries"]
        .as_array()
        .expect("manifest entries")
        .iter()
        .map(|entry| entry["name"].as_str().expect("manifest entry name"))
        .collect::<Vec<_>>();
    let root_names = json["roots"]
        .as_array()
        .expect("evaluated roots")
        .iter()
        .map(|root| root["name"].as_str().expect("evaluated root name"))
        .collect::<Vec<_>>();
    assert!(
        manifest_names.contains(&"padded"),
        "a dynamic-shape value root must not be mistaken for an unresolved generic"
    );
    assert_eq!(root_names, manifest_names);
}

#[test]
fn eval_rejects_unknown_manifest_target_without_fallback() {
    let output = Command::new(chelis_bin())
        .args(["eval", "--target", "not-a-target", "cast(1, i32)"])
        .output()
        .expect("run eval with invalid target");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "invalid target must fail");
    assert!(stderr.contains("unknown eval target `not-a-target`"));
    assert!(!stderr.contains("using `eval`"));
}

/// For every entry in the root manifest, the C lane's combined output (build
/// stderr and artifact stdout) must contain either a rendered value or a
/// [05-UNS-1] diagnostic naming that root.
#[test]
fn manifest_completeness_c_lane() {
    let source = r#"
type Pair = | Pair { left: i32, right: i32 }
scalar_root = cast(7, i32)
tensor_root = insert(scalar_to_tensor(cast(0.25, f32)), 0, 2i64)
tuple_root = (cast(1, i32), (cast(2, i32), cast(3, i32)))
record_root = Pair { left: cast(4, i32), right: cast(5, i32) }
"#;
    let (json, stderr, eval_success) = eval_json_file(source, Some("c"));
    assert!(eval_success, "C-target manifest probe failed: {stderr}");
    let manifest_names = json["manifest"]["entries"]
        .as_array()
        .expect("manifest entries")
        .iter()
        .map(|entry| entry["name"].as_str().unwrap().to_string())
        .collect::<Vec<_>>();

    let build = build_c(source);
    assert!(build.success, "C build failed: {}", build.stderr);
    assert!(build.emitted_c.contains("int main(void)"));
    let output = build.link_and_run();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let run_stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "generated C failed: {run_stderr}");
    for name in manifest_names {
        assert!(
            stdout
                .lines()
                .any(|line| line.starts_with(&format!("{name} = "))),
            "C artifact omitted manifest root `{name}`.\nstdout: {stdout}\nstderr: {run_stderr}"
        );
    }
}
