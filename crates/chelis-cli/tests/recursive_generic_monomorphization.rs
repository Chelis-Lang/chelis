//! Acceptance oracle for `openspec/changes/add-bounded-monomorphization`
//! (chelis#1158): bounded memoized monomorphization of recursive generic
//! host calls, plus the spec/04 §3.1.1 uniform-recursive-instantiation rule.
//!
//! Run: `cargo nextest run -p chelis-cli --test
//! recursive_generic_monomorphization --no-fail-fast`
//!
//! Scenario map (delta spec -> test), per the change's task 5.4:
//!
//! `generic-monomorphization` capability:
//! - Direct recursion at one instantiation ->
//!   `direct_recursion_at_one_instantiation_compiles_links_and_runs`
//! - Distinct instantiations get distinct specializations ->
//!   `distinct_instantiations_get_distinct_specializations`
//! - Mutual recursion specializes as a group ->
//!   `mutual_recursion_specializes_as_a_group`
//! - Memoization terminates compilation ->
//!   `memoized_specialization_reuses_one_symbol_across_call_sites`
//! - Build artifacts link cleanly ->
//!   `build_artifacts_link_cleanly_and_match_eval`
//! - A surviving unsupported call fails closed ->
//!   `surviving_unsupported_call_fails_closed_without_artifacts`
//! - No shipped diagnostic cites a closed tracking issue ->
//!   `no_shipped_diagnostic_cites_closed_issue_941`
//!
//! `type-system` capability (spec/04 §3.1.1 [04-INF-2]/[04-INF-3]):
//! - Direct uniform recursion is accepted ->
//!   `direct_uniform_recursion_is_accepted_by_the_checker`
//! - Mutual uniform recursion is accepted ->
//!   `mutual_uniform_recursion_is_accepted_by_the_checker`
//! - Polymorphic recursion is a check-time type error ->
//!   `polymorphic_recursion_is_a_check_time_type_error`
//! - Rejection is lane-uniform ->
//!   `polymorphic_recursion_rejection_is_lane_uniform`
//!
//! Named regression (chelis#941 body's minimized reproducer):
//! - `issue_941_minimized_reproducer_compiles`
//!
//! Determinism golden (change task 4.2):
//! - `specialized_symbol_set_is_deterministic_across_builds`

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::process::Command as StdCommand;
use tempfile::{TempDir, tempdir};

#[path = "common/mod.rs"]
mod common;

use common::{link_generated, write_file};

/// Direct recursion at one instantiation: `depth` over `Box[a]` recursing at
/// the caller's own `Box[a]`, applied at `Box[int32]`. Prints `3`.
const DIRECT_ONE_INSTANTIATION: &str = "\
type Box[a] =
  | Empty
  | Full { value: a }
def depth[a](box: Box[a], n: int32) -> int32 =
  if n <= 0 then 0 else depth(box, n - 1) + 1
def concrete() -> int32 = depth(Full { value: cast(7, int32) }, 3)
out = print(concrete())
";

/// The same recursive generic applied at `Box[int32]` and `Box[bool]`.
/// Prints `5` (2 + 3).
const DIRECT_TWO_INSTANTIATIONS: &str = "\
type Box[a] =
  | Empty
  | Full { value: a }
def depth[a](box: Box[a], n: int32) -> int32 =
  if n <= 0 then 0 else depth(box, n - 1) + 1
def concrete() -> int32 =
  depth(Full { value: cast(7, int32) }, 2) + depth(Full { value: true }, 3)
out = print(concrete())
";

/// One instantiation reached from two distinct call sites: memoization must
/// produce exactly one specialized definition. Prints `5`.
const DIRECT_TWO_CALL_SITES: &str = "\
type Box[a] =
  | Empty
  | Full { value: a }
def depth[a](box: Box[a], n: int32) -> int32 =
  if n <= 0 then 0 else depth(box, n - 1) + 1
def concrete() -> int32 =
  depth(Full { value: cast(7, int32) }, 2) + depth(Full { value: cast(9, int32) }, 3)
out = print(concrete())
";

/// Mutual recursion over `Box[a]` at one instantiation. Prints `4`.
const MUTUAL_ONE_INSTANTIATION: &str = "\
type Box[a] =
  | Empty
  | Full { value: a }
def ping[a](box: Box[a], n: int32) -> int32 =
  if n <= 0 then 0 else pong(box, n - 1) + 1
def pong[a](box: Box[a], n: int32) -> int32 =
  if n <= 0 then 100 else ping(box, n - 1) + 1
def concrete() -> int32 = ping(Full { value: cast(1.0, f32) }, 4)
out = print(concrete())
";

/// Polymorphic recursion: `f` over `a` recursively calls `f` at `Box[a]`.
/// spec/04 §3.1.1 [04-INF-3] makes this a check-time type error.
const POLYMORPHIC_RECURSION: &str = "\
type Box[a] =
  | Empty
  | Full { value: a }
def f[a](x: a, n: int32) -> int32 =
  if n <= 0 then 0 else f(Full { value: x }, n - 1) + 1
def main() -> int32 = f(1, 3)
";

/// A recursive generic call whose instantiation never resolves (`a` is
/// unconstrained at the call site): the specializer cannot serve it, so the
/// build must stay on the fail-closed [05-UNS] boundary.
const UNRESOLVED_INSTANTIATION: &str = "\
type Box[a] =
  | Empty
  | Full { value: a }
def loop[a](box: Box[a]) -> bool =
  match box with {
    | Empty => true
    | Full { value: item } => loop(Empty)
  }
def main() -> bool = loop(Empty)
";

/// The minimized reproducer from the chelis#941 issue body, verbatim shape.
const ISSUE_941_REPRODUCER: &str = "\
type Box[a] =
  | Empty
  | Full { value: a }
def loop[a](box: Box[a]) -> bool =
  match box with {
    | Empty => true
    | Full { value: item } => loop(Empty)
  }
def main() -> bool = loop(Full { value: cast(1.0, f32) })
";

fn chelis() -> Command {
    let mut cmd = Command::cargo_bin("chelis").expect("chelis binary");
    cmd.env("CHELIS_STYLE_GATE_DISABLE", "1");
    cmd
}

fn build_ok(source: &str, stem: &str) -> (TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{stem}.ch"));
    let out_dir = dir.path().join("out");
    write_file(&path, source);
    chelis()
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    (dir, out_dir)
}

/// Build expecting failure; returns stderr and the would-be C artifact path.
fn build_err(source: &str, stem: &str) -> (TempDir, PathBuf, String) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{stem}.ch"));
    let out_dir = dir.path().join("out");
    write_file(&path, source);
    let assert = chelis()
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&assert.get_output().stderr).into_owned();
    (dir, out_dir.join(format!("{stem}.c")), stderr)
}

fn eval_first_line(source: &str, stem: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{stem}.ch"));
    write_file(&path, source);
    let output = chelis()
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    String::from_utf8_lossy(&output)
        .lines()
        .next()
        .unwrap_or_default()
        .to_string()
}

fn run_first_line(out_dir: &std::path::Path, stem: &str) -> String {
    let source_file = format!("{stem}.c");
    let status = link_generated(out_dir, &source_file, "run");
    assert!(status.success(), "generated C must link: {status}");
    let output = StdCommand::new(out_dir.join("run"))
        .output()
        .expect("run compiled binary");
    assert!(
        output.status.success(),
        "compiled binary failed: {}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .unwrap_or_default()
        .to_string()
}

fn read_generated_c(out_dir: &std::path::Path, stem: &str) -> String {
    fs::read_to_string(out_dir.join(format!("{stem}.c"))).expect("read generated C")
}

/// Collect every C identifier in `text` that begins with `prefix`.
fn identifiers_with_prefix(text: &str, prefix: &str) -> BTreeSet<String> {
    let bytes = text.as_bytes();
    let mut found = BTreeSet::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < bytes.len()
                && ((bytes[i] as char).is_ascii_alphanumeric() || bytes[i] == b'_')
            {
                i += 1;
            }
            let ident = &text[start..i];
            if ident.starts_with(prefix) && ident.len() > prefix.len() {
                found.insert(ident.to_string());
            }
        } else {
            i += 1;
        }
    }
    found
}

fn count_occurrences(text: &str, needle: &str) -> usize {
    text.match_indices(needle).count()
}

// ---------------------------------------------------------------------------
// generic-monomorphization capability
// ---------------------------------------------------------------------------

#[test]
fn direct_recursion_at_one_instantiation_compiles_links_and_runs() {
    let (_dir, out_dir) = build_ok(DIRECT_ONE_INSTANTIATION, "direct_one");
    let c_source = read_generated_c(&out_dir, "direct_one");
    let specialized = identifiers_with_prefix(&c_source, "depth");
    assert_eq!(
        specialized.len(),
        1,
        "exactly one specialized `depth` definition expected, found: {specialized:?}"
    );
    let symbol = specialized.iter().next().expect("one symbol");
    assert!(
        count_occurrences(&c_source, symbol) >= 2,
        "recursive edge must target the specialized symbol `{symbol}` \
         (expected definition plus self-call)"
    );
    assert_eq!(run_first_line(&out_dir, "direct_one"), "3");
}

#[test]
fn distinct_instantiations_get_distinct_specializations() {
    let (_dir, out_dir) = build_ok(DIRECT_TWO_INSTANTIATIONS, "direct_two");
    let c_source = read_generated_c(&out_dir, "direct_two");
    let specialized = identifiers_with_prefix(&c_source, "depth");
    assert_eq!(
        specialized.len(),
        2,
        "one specialized definition per instantiation expected, found: {specialized:?}"
    );
    for symbol in &specialized {
        assert!(
            count_occurrences(&c_source, symbol) >= 2,
            "each instantiation's recursive edge must target its own symbol `{symbol}`"
        );
    }
    assert_eq!(run_first_line(&out_dir, "direct_two"), "5");
}

#[test]
fn mutual_recursion_specializes_as_a_group() {
    let (_dir, out_dir) = build_ok(MUTUAL_ONE_INSTANTIATION, "mutual_one");
    let c_source = read_generated_c(&out_dir, "mutual_one");
    let ping = identifiers_with_prefix(&c_source, "ping");
    let pong = identifiers_with_prefix(&c_source, "pong");
    assert_eq!(
        ping.len(),
        1,
        "one specialized `ping` expected, found: {ping:?}"
    );
    assert_eq!(
        pong.len(),
        1,
        "one specialized `pong` expected, found: {pong:?}"
    );
    assert_eq!(run_first_line(&out_dir, "mutual_one"), "4");
}

#[test]
fn memoized_specialization_reuses_one_symbol_across_call_sites() {
    let (_dir, out_dir) = build_ok(DIRECT_TWO_CALL_SITES, "memoized");
    let c_source = read_generated_c(&out_dir, "memoized");
    let specialized = identifiers_with_prefix(&c_source, "depth");
    assert_eq!(
        specialized.len(),
        1,
        "two call sites at one instantiation must share one memoized \
         specialization, found: {specialized:?}"
    );
    assert_eq!(run_first_line(&out_dir, "memoized"), "5");
}

#[test]
fn build_artifacts_link_cleanly_and_match_eval() {
    let eval = eval_first_line(DIRECT_TWO_INSTANTIATIONS, "link_clean_eval");
    let (_dir, out_dir) = build_ok(DIRECT_TWO_INSTANTIATIONS, "link_clean");
    let compiled = run_first_line(&out_dir, "link_clean");
    assert_eq!(eval, "5");
    assert_eq!(
        compiled, eval,
        "compiled output must match the eval lane exactly"
    );
}

#[test]
fn surviving_unsupported_call_fails_closed_without_artifacts() {
    let (_dir, c_artifact, stderr) = build_err(UNRESOLVED_INSTANTIATION, "unresolved");
    assert!(
        stderr.contains("unsupported"),
        "rejection must be branded `unsupported:`, got:\n{stderr}"
    );
    assert!(
        !c_artifact.exists(),
        "no C artifact may be written on rejection: {}",
        c_artifact.display()
    );
}

#[test]
fn no_shipped_diagnostic_cites_closed_issue_941() {
    let (_dir, _artifact, stderr) = build_err(UNRESOLVED_INSTANTIATION, "citation");
    assert!(
        !stderr.contains("chelis#941"),
        "no shipped diagnostic may cite the closed chelis#941, got:\n{stderr}"
    );
    assert!(
        stderr.contains("chelis#1158") || stderr.contains("[04-") || stderr.contains("[05-"),
        "rejection must carry its authority (open issue or deciding atom) per \
         [05-UNS-5], got:\n{stderr}"
    );
}

// ---------------------------------------------------------------------------
// type-system capability (spec/04 §3.1.1)
// ---------------------------------------------------------------------------

#[test]
fn direct_uniform_recursion_is_accepted_by_the_checker() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("uniform_direct.ch");
    write_file(&path, DIRECT_ONE_INSTANTIATION);
    chelis()
        .args(["check", path.to_str().unwrap()])
        .assert()
        .success();
}

#[test]
fn mutual_uniform_recursion_is_accepted_by_the_checker() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("uniform_mutual.ch");
    write_file(&path, MUTUAL_ONE_INSTANTIATION);
    chelis()
        .args(["check", path.to_str().unwrap()])
        .assert()
        .success();
}

#[test]
fn polymorphic_recursion_is_a_check_time_type_error() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("poly_rec.ch");
    write_file(&path, POLYMORPHIC_RECURSION);
    // `chelis check` reports its error list in the stdout JSON report and
    // exits nonzero.
    let assert = chelis()
        .args(["check", path.to_str().unwrap()])
        .assert()
        .failure();
    let output = assert.get_output();
    let report = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        report.contains("polymorphic recursion"),
        "diagnostic must name the violation, got:\n{report}"
    );
    assert!(
        report.contains("`f`"),
        "diagnostic must name the function, got:\n{report}"
    );
    assert!(
        report.contains("Box["),
        "diagnostic must name the differing recursive instantiation, got:\n{report}"
    );
    assert!(
        report.contains("the caller's own instantiation `[a]`"),
        "diagnostic must name the caller instantiation, got:\n{report}"
    );
    assert!(
        report.contains("[04-INF-3]"),
        "diagnostic must cite the deciding spec/04 §3.1.1 atom, got:\n{report}"
    );
}

#[test]
fn polymorphic_recursion_rejection_is_lane_uniform() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("poly_rec_lanes.ch");
    write_file(&path, POLYMORPHIC_RECURSION);

    let eval_assert = chelis()
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .failure();
    let eval_stderr = String::from_utf8_lossy(&eval_assert.get_output().stderr).into_owned();

    let out_dir = dir.path().join("out");
    let build_assert = chelis()
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .failure();
    let build_stderr = String::from_utf8_lossy(&build_assert.get_output().stderr).into_owned();

    // Both lanes wrap the check-time error in their own invocation framing
    // (`error: Type errors:` vs `error: Check errors: Type errors:`); the
    // check-time error itself starts at the violation name and must be
    // byte-identical across lanes.
    let atom_error = |stderr: &str, lane: &str| -> String {
        let line = stderr
            .lines()
            .find(|line| line.contains("[04-INF-3]"))
            .unwrap_or_else(|| panic!("{lane} stderr must carry the atom-citing line:\n{stderr}"));
        let start = line
            .find("polymorphic recursion")
            .unwrap_or_else(|| panic!("{lane} atom line must name the violation:\n{line}"));
        line[start..].to_string()
    };
    assert_eq!(
        atom_error(&eval_stderr, "eval"),
        atom_error(&build_stderr, "build"),
        "eval and build must report the identical check-time error"
    );
    assert!(
        !build_stderr.contains("Lowering error"),
        "rejection must happen at check time, before any lane-specific stage:\n{build_stderr}"
    );
}

// ---------------------------------------------------------------------------
// Named regression: the chelis#941 minimized reproducer
// ---------------------------------------------------------------------------

#[test]
fn issue_941_minimized_reproducer_compiles() {
    // The verbatim reproducer has no top-level binding, so the emitted C is
    // object-mode (no host `main`); compile it as an object like the sibling
    // issue_935 suite does. Link-clean coverage for recursive generics lives
    // in `build_artifacts_link_cleanly_and_match_eval`.
    let (_dir, out_dir) = build_ok(ISSUE_941_REPRODUCER, "issue_941_repro");
    let c_source = read_generated_c(&out_dir, "issue_941_repro");
    let specialized = identifiers_with_prefix(&c_source, "loop");
    assert_eq!(
        specialized.len(),
        1,
        "one specialized `loop` definition expected, found: {specialized:?}"
    );
    let symbol = specialized.iter().next().expect("one symbol");
    assert!(
        count_occurrences(&c_source, symbol) >= 2,
        "the reproducer's recursive edge must target `{symbol}`"
    );
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: true,
            needs_blas: false,
        },
    );
    let output = StdCommand::new(&toolchain.compiler)
        .current_dir(&out_dir)
        .args(&toolchain.compile_flags)
        .args(["-I.", "-c", "issue_941_repro.c"])
        .output()
        .expect("invoke C compiler");
    assert!(
        output.status.success(),
        "the chelis#941 reproducer must compile: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// ---------------------------------------------------------------------------
// Determinism golden (change task 4.2)
// ---------------------------------------------------------------------------

#[test]
fn specialized_symbol_set_is_deterministic_across_builds() {
    let (_dir_a, out_a) = build_ok(DIRECT_TWO_INSTANTIATIONS, "golden_a");
    let (_dir_b, out_b) = build_ok(DIRECT_TWO_INSTANTIATIONS, "golden_b");
    let symbols_a = identifiers_with_prefix(&read_generated_c(&out_a, "golden_a"), "depth");
    let symbols_b = identifiers_with_prefix(&read_generated_c(&out_b, "golden_b"), "depth");
    assert_eq!(
        symbols_a, symbols_b,
        "specialized symbol set must be identical across consecutive builds"
    );
    assert_eq!(
        symbols_a.len(),
        2,
        "two instantiations must emit exactly two specialized symbols: {symbols_a:?}"
    );
}
