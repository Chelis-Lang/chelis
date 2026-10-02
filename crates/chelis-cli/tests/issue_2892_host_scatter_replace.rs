//! chelis#2892: `scatter_replace` ([05-OP-52]) runs in the host interpreter
//! behind `chelis eval --file` and `chelis test`, with the same values the
//! compiled C program prints. [05-HOST-1] requires the complete contract in
//! every execution mode, so each accepted program here is run on both lanes and
//! the printed lines are compared with each other and with the value the
//! atom's last-write-wins rule gives.
//!
//! The rejected rows are the atom's failure clause: an index outside the
//! scattered axis fails loudly on both lanes instead of being clipped or
//! skipped.
use assert_cmd::Command;
use std::path::Path;
use std::process::Command as StdCommand;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::{build_and_run, gcc_available, write_file};

/// The issue's program: distinct `i64` positions on a one-axis base.
const ISSUE_REPRO: &str = "module Demo.Main
def run(seed: i64) -> List[i64] = {
  base = to_tensor([0i64, 0i64, 0i64])
  positions = to_tensor([2i64, 0i64])
  updates = to_tensor([7i64, 9i64])
  to_list(scatter_replace(base, positions, updates, 0i32))
}
result = run(0i64)
";

/// Duplicate positions: the last update in row-major update order wins.
const DUPLICATE_POSITIONS: &str = "module Demo.Main
def run(seed: i64) -> List[i32] = {
  base = to_tensor([1i32, 2i32, 3i32, 4i32])
  positions = to_tensor([1i64, 3i64, 1i64, 1i64])
  updates = to_tensor([10i32, 20i32, 30i32, 40i32])
  to_list(scatter_replace(base, positions, updates, 0i32))
}
result = run(0i64)
";

/// A rank-2 base scattered along axis 1: each row of the updates is written
/// into the selected columns of the same row.
const AXIS_ONE: &str = "module Demo.Main
def run(seed: i64) -> List[f32] = {
  base = reshape(to_tensor([0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32]), [2i64, 3i64])
  positions = to_tensor([2i32, 0i32])
  updates = reshape(to_tensor([1.5f32, 2.5f32, 3.5f32, 4.5f32]), [2i64, 2i64])
  to_list(reshape(scatter_replace(base, positions, updates, 1i32), [6i64]))
}
result = run(0i64)
";

/// Bool data is selected and replaced without entering arithmetic.
const BOOL_DATA: &str = "module Demo.Main
def run(seed: i64) -> List[bool] = {
  base = to_tensor([false, false, false])
  positions = to_tensor([1i32])
  updates = to_tensor([true])
  to_list(scatter_replace(base, positions, updates, 0i32))
}
result = run(0i64)
";

/// [05-SPARSE-1] reads an index at its exact stored width, `i8` included.
/// Compiled C rejects this program at an internal invariant (chelis#2929),
/// so it is an eval row only until that is fixed.
const NARROW_INDEX: &str = "module Demo.Main
def run(seed: i64) -> List[i64] = {
  base = to_tensor([0i64, 0i64, 0i64])
  positions = to_tensor([2i8, 0i8, 2i8])
  updates = to_tensor([7i64, 9i64, 11i64])
  to_list(scatter_replace(base, positions, updates, 0i32))
}
result = run(0i64)
";

/// The position is computed from a runtime argument, so the checker cannot
/// reject it; it lands outside the three-element axis at run time.
const RUNTIME_OUT_OF_RANGE: &str = "module Demo.Main
def put(base: tensor[3, i64], positions: tensor[1, i64]) -> List[i64] = to_list(scatter_replace(base, positions, to_tensor([7i64]), 0i32))
def run(seed: i64) -> List[i64] = put(to_tensor([0i64, 0i64, 0i64]), add(to_tensor([3i64]), to_tensor([seed])))
result = run(0i64)
";

/// The same, below zero.
const RUNTIME_NEGATIVE: &str = "module Demo.Main
def put(base: tensor[3, i64], positions: tensor[1, i64]) -> List[i64] = to_list(scatter_replace(base, positions, to_tensor([7i64]), 0i32))
def run(seed: i64) -> List[i64] = put(to_tensor([0i64, 0i64, 0i64]), sub(to_tensor([seed]), to_tensor([1i64])))
result = run(0i64)
";

const ACCEPTED: &[(&str, &str, &str)] = &[
    ("issue_repro", ISSUE_REPRO, "result = [9, 0, 7]"),
    (
        "duplicate_positions",
        DUPLICATE_POSITIONS,
        "result = [1, 40, 3, 20]",
    ),
    (
        "axis_one",
        AXIS_ONE,
        "result = [2.5, 0.0, 1.5, 4.5, 0.0, 3.5]",
    ),
    ("bool_data", BOOL_DATA, "result = [false, true, false]"),
];

const EVAL_ONLY: &[(&str, &str, &str)] = &[("narrow_index", NARROW_INDEX, "result = [9, 0, 11]")];

fn eval_file(dir: &Path, name: &str, source: &str) -> std::process::Output {
    let path = dir.join(format!("{name}.ch"));
    write_file(&path, source);
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval runs")
}

fn result_line(stdout: &str) -> &str {
    stdout
        .lines()
        .find(|line| line.starts_with("result = "))
        .unwrap_or_else(|| panic!("no `result = ` line in:\n{stdout}"))
}

#[test]
fn eval_file_runs_scatter_replace_with_the_atoms_values() {
    let dir = tempdir().expect("tempdir");
    for (name, source, expected) in ACCEPTED.iter().chain(EVAL_ONLY) {
        let out = eval_file(dir.path(), name, source);
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success(),
            "{name}: eval rejected scatter_replace\nstdout: {stdout}\nstderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(result_line(&stdout), *expected, "{name}: eval value");
    }
}

#[test]
fn eval_file_and_compiled_c_print_the_same_scatter_replace_values() {
    if !gcc_available() {
        eprintln!("skipping: no host C compiler");
        return;
    }
    let dir = tempdir().expect("tempdir");
    for (name, source, expected) in ACCEPTED {
        let eval = eval_file(dir.path(), name, source);
        assert!(eval.status.success(), "{name}: eval failed");
        let eval_stdout = String::from_utf8_lossy(&eval.stdout).into_owned();
        let c_stdout = build_and_run(source, name);
        assert_eq!(
            result_line(&c_stdout),
            *expected,
            "{name}: compiled C value"
        );
        assert_eq!(
            result_line(&eval_stdout),
            result_line(&c_stdout),
            "{name}: eval and compiled C disagree"
        );
    }
}

#[test]
fn eval_file_rejects_a_runtime_index_outside_the_axis() {
    let dir = tempdir().expect("tempdir");
    for (name, source, index) in [
        ("runtime_out_of_range", RUNTIME_OUT_OF_RANGE, 3),
        ("runtime_negative", RUNTIME_NEGATIVE, -1),
    ] {
        let out = eval_file(dir.path(), name, source);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            !out.status.success(),
            "{name}: eval accepted an index outside the axis\nstdout: {}",
            String::from_utf8_lossy(&out.stdout)
        );
        assert!(
            stderr.contains(&format!("scatter index {index} out of bounds")),
            "{name}: eval must report scatter's own bounds failure\nstderr: {stderr}"
        );
    }
}

#[test]
fn compiled_c_rejects_a_runtime_index_outside_the_axis() {
    if !gcc_available() {
        eprintln!("skipping: no host C compiler");
        return;
    }
    for (name, source) in [
        ("runtime_out_of_range", RUNTIME_OUT_OF_RANGE),
        ("runtime_negative", RUNTIME_NEGATIVE),
    ] {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join(format!("{name}.ch"));
        let out_dir = dir.path().join(format!("{name}-out"));
        write_file(&path, source);
        Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        let run = StdCommand::new(out_dir.join(name))
            .output()
            .expect("compiled binary runs");
        let stderr = String::from_utf8_lossy(&run.stderr);
        assert!(
            !run.status.success(),
            "{name}: compiled C accepted an index outside the axis\nstdout: {}",
            String::from_utf8_lossy(&run.stdout)
        );
        assert!(
            !String::from_utf8_lossy(&run.stdout).contains("result = "),
            "{name}: compiled C printed a result for an index outside the axis"
        );
        assert!(
            stderr.contains("scatter_replace"),
            "{name}: compiled C must name the failing operation\nstderr: {stderr}"
        );
    }
}

/// `chelis test` runs a test body through the same host interpreter, so a
/// test that calls `scatter_replace` passes instead of stopping at the
/// unsupported-builtin error.
#[test]
fn chelis_test_runs_scatter_replace() {
    let dir = tempdir().expect("tempdir");
    let pkg = dir.path().join("scatter-probe");
    write_file(
        &pkg.join("reef.toml"),
        &format!(
            "[package]\nname = \"scatter-probe\"\nversion = \"0.1.0\"\ncompiler = \"={}\"\nmodule_prefix = \"Probe\"\n",
            chelis_compiler_api::COMPILER_VERSION
        ),
    );
    write_file(
        &pkg.join("src/main.ch"),
        "module Probe.Main\n\ndef noop() -> unit = test_assert(true, \"noop\")\n",
    );
    write_file(
        &pkg.join("tests/scatter.ch"),
        "module Probe.Tests.Scatter

def test_scatter_replace() -> unit = {
  base = to_tensor([0i64, 0i64, 0i64])
  got = to_list(scatter_replace(base, to_tensor([2i64, 0i64, 2i64]), to_tensor([7i64, 9i64, 11i64]), 0i32))
  _ = test_assert(eq(index(got, 0i64), 9i64), \"position 0\")
  _ = test_assert(eq(index(got, 1i64), 0i64), \"position 1\")
  test_assert(eq(index(got, 2i64), 11i64), \"position 2\")
}
",
    );
    let out = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(&pkg)
        .args(["test", "tests/"])
        .output()
        .expect("chelis test runs");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success() && stdout.contains("1 passed, 0 failed"),
        "chelis test did not pass the scatter_replace test\nstdout: {stdout}\nstderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}
