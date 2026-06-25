//! Chelis-Lang/chelis#527 — read-then-reuse of a `List[tensor]` must
//! compile and run.
//!
//! At 0.10.0 the idiomatic optimizer shape "read a list's length /
//! elements, then reuse the list" compiled; at 0.10.1 the #343
//! linearity fix (correctly typing `List[tensor]` parameters whose
//! name collides with a Deep tag, e.g. `params`) exposed that `len`
//! and `index` *consumed* their `List` argument, so the later reuse
//! became a use-after-consume with no non-consuming form to express
//! it. `len` / `index` now borrow their `List` argument by default
//! (the runtime `chelis_list_len` / `chelis_list_index` take a
//! `const chelis_list *` and never free it), restoring the pattern.
//!
//! This is the executable acceptance oracle: it builds a faithful
//! self-contained replica of School's `adamw_step_walk` /
//! `adamw_step_list` read-then-reuse shape — a recursive
//! `index(params, i)`-then-recurse walk under a `len(params)`-then-walk
//! driver, with the parameter literally named `params` (the
//! tag-colliding identifier #343 began consume-tracking) — compiles
//! the generated C, runs it, and checks the numeric output.

use assert_cmd::Command;
use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;
use tempfile::tempdir;

fn generated_source_needs_blas(out_dir: &Path, source: &str) -> bool {
    fs::read_to_string(out_dir.join(source))
        .map(|text| text.contains("cblas_sgemm(") || text.contains("\"chelis_blas.h\""))
        .unwrap_or(false)
}

/// Link the chelis-generated C against the platform host toolchain.
/// Mirrors `link_generated` from `issue_218_numerical_correctness.rs`.
fn link_generated(out_dir: &Path, source: &str, binary: &str) -> std::process::ExitStatus {
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: true,
            needs_blas: generated_source_needs_blas(out_dir, source),
        },
    );
    let mut cmd = StdCommand::new(&toolchain.compiler);
    cmd.current_dir(out_dir);
    cmd.arg("-O2");
    cmd.args(&toolchain.compile_flags);
    cmd.arg(source);
    cmd.args(["-L.", "-lchelis_runtime"]);
    cmd.args(&toolchain.link_flags);
    cmd.args(["-o", binary]);
    cmd.status().expect("host compiler should run")
}

/// Parse a printed Chelis tensor of the form
/// `name = tensor(shape=[..], data=[v0, v1, ...])` into its flat data.
fn parse_tensor_data(stdout: &str, name: &str) -> Vec<f64> {
    let prefix = format!("{name} = tensor(");
    let line = stdout
        .lines()
        .find(|l| l.starts_with(&prefix))
        .unwrap_or_else(|| panic!("output does not contain `{prefix}` line:\n{stdout}"));
    let marker = "data=[";
    let start = line.find(marker).expect("data marker") + marker.len();
    let end = line[start..].find(']').expect("closing bracket");
    line[start..start + end]
        .split(',')
        .map(|s| s.trim().parse::<f64>().expect("numeric"))
        .collect()
}

fn gcc_available() -> bool {
    StdCommand::new(chelis_backend_c::toolchain::c_compiler())
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn build_and_run(source: &str, name: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    fs::write(&path, source).expect("write source");

    Command::cargo_bin("chelis")
        .expect("binary")
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

    let source_file = format!("{name}.c");
    let status = link_generated(&out_dir, &source_file, name);
    assert!(status.success(), "link failed: {status}");

    let run = StdCommand::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    assert!(
        run.status.success(),
        "binary failed: {}\nstderr: {}",
        run.status,
        String::from_utf8_lossy(&run.stderr),
    );
    String::from_utf8(run.stdout).expect("utf-8 stdout")
}

/// Faithful replica of School's `adamw_step_list` + `adamw_step_walk`
/// read-then-reuse shape: `len(params)` then a recursive
/// `index(params, i)`-then-recurse walk that reuses `params` on every
/// frame. Reduces `[[1,2],[3,4],[5,6]]` to its elementwise sum
/// `[9, 12]`. Builds, runs, and checks the result.
#[test]
fn issue_527_len_index_then_reuse_list_builds_and_runs() {
    if !gcc_available() {
        eprintln!("skipping: c compiler not available");
        return;
    }
    let source = "\
def walk(params: List[tensor[2, f32]], i: int64, total: int64) -> tensor[2, f32] =\n\
  if gte(i, total) then to_tensor([0.0, 0.0]) else\n\
    {\n\
      p = index(params, i)\n\
      rest = walk(params, add(i, cast(1, int64)), total)\n\
      add(p, rest)\n\
    }\n\
def reduce_list(params: List[tensor[2, f32]]) -> tensor[2, f32] =\n\
  {\n\
    total = cast(len(params), int64)\n\
    walk(params, cast(0, int64), total)\n\
  }\n\
def run() -> tensor[2, f32] =\n\
  {\n\
    a = to_tensor([1.0, 2.0])\n\
    b = to_tensor([3.0, 4.0])\n\
    c = to_tensor([5.0, 6.0])\n\
    ps: List[tensor[2, f32]] = [a, b, c]\n\
    reduce_list(ps)\n\
  }\n\
out = run()\n";

    let stdout = build_and_run(source, "list_read_then_reuse");
    let actual = parse_tensor_data(&stdout, "out");
    let expected = [9.0, 12.0];
    assert_eq!(
        actual.len(),
        expected.len(),
        "unexpected output arity: {actual:?}"
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-5,
            "element {i}: actual={a} expected={e}\nfull stdout:\n{stdout}"
        );
    }
}
