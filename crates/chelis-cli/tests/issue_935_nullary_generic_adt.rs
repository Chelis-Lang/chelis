//! chelis#935: a checked return context must carry concrete applied type
//! arguments into nullary generic ADT construction before C host lowering.
//!
//! Positive coverage locks the exact four-line reproducer as a compilable C
//! object and an observable eval/C execution pair. Negative parity keeps an
//! unconstrained nullary generic constructor on the existing fail-loud
//! chelis#730 boundary; this fix must not invent an ABI type.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use std::process::Command as StdCommand;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

const EXACT_REPRO: &str = "\
type Box[a] =
  | Empty
  | Full { value: a }
def empty[a]() -> Box[a] = Empty
def main() -> Box[f32] = empty()
";

fn build(source: &str, stem: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{stem}.ch"));
    let out_dir = dir.path().join("out");
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
    (dir, out_dir)
}

#[test]
fn exact_nullary_generic_reproducer_c_compiles() {
    let (_dir, out_dir) = build(EXACT_REPRO, "nullary_generic");
    let source = "nullary_generic.c";
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: true,
            needs_blas: false,
        },
    );
    let output = StdCommand::new(&toolchain.compiler)
        .current_dir(&out_dir)
        .args(&toolchain.compile_flags)
        .args(["-I.", "-c", source])
        .output()
        .expect("invoke C compiler");
    assert!(
        output.status.success(),
        "generated C must compile: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn checked_context_reaches_nullary_constructors_nested_under_if_and_let() {
    let source = "\
type Box[a] =
  | Empty
  | Full { value: a }
def choose(flag: bool) -> Box[f32] = {
  selected = if flag then Empty else Empty
  selected
}
def main() -> Box[f32] = choose(true)
";
    let (_dir, out_dir) = build(source, "nested_nullary_generic");
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: true,
            needs_blas: false,
        },
    );
    let output = StdCommand::new(&toolchain.compiler)
        .current_dir(&out_dir)
        .args(&toolchain.compile_flags)
        .args(["-I.", "-c", "nested_nullary_generic.c"])
        .output()
        .expect("invoke C compiler");
    assert!(
        output.status.success(),
        "nested checked context must produce valid C: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn concrete_call_context_reaches_nullary_generic_match_patterns() {
    let source = "\
type Box[a] =
  | Empty
  | Full { value: a }
def empty[a]() -> Box[a] = Empty
def is_empty[a](value: Box[a]) -> bool =
  match value with {
    | Empty => true
    | Full { value: item } => false
  }
def concrete() -> Box[f32] = empty()
def main() -> bool = is_empty(concrete())
";
    let (_dir, out_dir) = build(source, "nullary_generic_match");
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: true,
            needs_blas: false,
        },
    );
    let output = StdCommand::new(&toolchain.compiler)
        .current_dir(&out_dir)
        .args(&toolchain.compile_flags)
        .args(["-I.", "-c", "nullary_generic_match.c"])
        .output()
        .expect("invoke C compiler");
    assert!(
        output.status.success(),
        "generic match specialization must produce valid C: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn fold_empty_accumulator_uses_checked_callback_type() {
    // chelis#939: the checker-owned callback parameter, not the source
    // spelling of `[]`, determines the accumulator type.
    let source = "\
def keep_unmasked(masks: List[bool], rows: List[int64]) -> List[int64] =
  fold(
    fn (acc: List[int64], row: int64) ->
      if index(masks, row) then acc else append(acc, row),
    [],
    rows
  )
def main() -> List[int64] = keep_unmasked([false], [cast(0, int64)])
";
    let (_dir, out_dir) = build(source, "fold_checked_accumulator");
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: true,
            needs_blas: false,
        },
    );
    let output = StdCommand::new(&toolchain.compiler)
        .current_dir(&out_dir)
        .args(&toolchain.compile_flags)
        .args(["-I.", "-c", "fold_checked_accumulator.c"])
        .output()
        .expect("invoke C compiler");
    assert!(
        output.status.success(),
        "fold's checked accumulator must resolve an empty init list: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn nested_adt_dimension_uses_rank_specialization_not_value_monomorphization() {
    // chelis#940: `n` is not a stored value parameter. Its only runtime role
    // is the tensor dimension reached through Frame[n] -> Column[n] ->
    // tensor[n, f32], so the concrete call must follow rank specialization.
    let source = "\
type Column[n] =
  | FloatCol(tensor[n, f32])
type Hamt[a] =
  | Leaf { value: a }
type Frame[n] =
  | Frame { cols: Hamt[Column[n]] }
def singleton[a](value: a) -> Hamt[a] = Leaf { value }
def from_column[n](column: Column[n]) -> Frame[n] =
  Frame { cols: singleton(column) }
def total[n](frame: Frame[n]) -> f32 =
  match frame with {
    | Frame { cols } =>
      match cols with {
        | Leaf { value: column } =>
          match column with {
            | FloatCol(values) =>
              fold(
                fn (acc: f32, value: f32) -> add(acc, value),
                cast(0.0, f32),
                to_list(values)
              )
          }
      }
  }
def concrete_frame() -> Frame[2] =
  from_column(FloatCol(to_tensor([cast(1.0, f32), cast(2.0, f32)])))
def concrete() -> f32 = total(concrete_frame())
out = print(concrete())
";
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("nested_adt_dimension.ch");
    write_file(&path, source);
    let eval = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let (_build_dir, out_dir) = build(source, "nested_adt_dimension");
    let linked = common::link_generated(&out_dir, "nested_adt_dimension.c", "run");
    assert!(linked.success(), "nested-ADT generated C must link");
    let compiled = StdCommand::new(out_dir.join("run"))
        .output()
        .expect("run nested-ADT generated binary");
    assert!(
        compiled.status.success(),
        "nested-ADT binary failed: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&eval)
            .lines()
            .next()
            .unwrap_or_default(),
        "3.0"
    );
    assert_eq!(
        String::from_utf8_lossy(&compiled.stdout)
            .lines()
            .next()
            .unwrap_or_default(),
        "3.0"
    );
}

#[test]
fn nested_adt_dimension_and_dtype_specialize_from_checked_metadata() {
    // chelis#948: the dimension parameter is representation-erased, but the
    // dtype is stored in the tensor ABI. Both facts come from the checker's
    // ADT registry; authored type-parameter names are not re-parsed here.
    let source = "\
type Column[rows, dtype] =
  | Column(tensor[rows, dtype])
type Hamt[payload] =
  | Leaf { value: payload }
type Frame[rows, dtype] =
  | Frame { cols: Hamt[Column[rows, dtype]] }
def singleton[payload](value: payload) -> Hamt[payload] = Leaf { value }
def from_column[rows, dtype](column: Column[rows, dtype]) -> Frame[rows, dtype] =
  Frame { cols: singleton(column) }
def first[rows, dtype](frame: Frame[rows, dtype]) -> dtype =
  match frame with {
    | Frame { cols } =>
      match cols with {
        | Leaf { value: column } =>
          match column with {
            | Column(values) => index(to_list(values), 0)
          }
      }
  }
def concrete() -> int16 =
  first(from_column(Column(to_tensor([cast(321, int16), cast(7, int16)]))))
out = print(concrete())
";
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("nested_adt_dtype.ch");
    write_file(&path, source);
    let eval = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let (_build_dir, out_dir) = build(source, "nested_adt_dtype");
    let linked = common::link_generated(&out_dir, "nested_adt_dtype.c", "run");
    assert!(
        linked.success(),
        "mixed dimension/dtype generated C must link"
    );
    let compiled = StdCommand::new(out_dir.join("run"))
        .output()
        .expect("run mixed dimension/dtype generated binary");
    assert!(
        compiled.status.success(),
        "mixed dimension/dtype binary failed: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&eval)
            .lines()
            .next()
            .unwrap_or_default(),
        "321"
    );
    assert_eq!(
        String::from_utf8_lossy(&compiled.stdout)
            .lines()
            .next()
            .unwrap_or_default(),
        "321"
    );
}

// chelis#1158 retired the two recursive-generic rejection tests that lived
// here (`invoked_recursive_generic_fails_before_emitting_an_undefined_symbol`
// and `invoked_mutual_recursive_generic_cycle_fails_loudly`). Both programs
// now build and run through bounded memoized monomorphization; their positive
// successors are `direct_recursive_generic_builds_and_runs` and
// `mutually_recursive_generics_build_and_run` in
// `issue_941_recursive_generic_mono.rs`, which also keeps the narrowed
// `[05-UNS-1]` boundary under test for polymorphic recursion.

#[test]
fn concrete_nullary_generic_constructor_eval_and_c_agree() {
    let source = "\
type Box[a] =
  | Empty
  | Full { value: a }
def empty[a]() -> Box[a] = Empty
def concrete() -> Box[f32] = empty()
out = print(concrete())
";
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("nullary_generic_run.ch");
    write_file(&path, source);
    let eval = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let (_build_dir, out_dir) = build(source, "nullary_generic_run");
    let linked = common::link_generated(&out_dir, "nullary_generic_run.c", "run");
    assert!(linked.success(), "generated C must link");
    let compiled = StdCommand::new(out_dir.join("run"))
        .output()
        .expect("run generated binary");
    assert!(
        compiled.status.success(),
        "generated binary failed: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let eval_first = String::from_utf8_lossy(&eval)
        .lines()
        .next()
        .unwrap_or_default()
        .to_string();
    let c_first = String::from_utf8_lossy(&compiled.stdout)
        .lines()
        .next()
        .unwrap_or_default()
        .to_string();
    assert_eq!(c_first, eval_first, "eval/C constructor observation parity");
    assert_eq!(c_first, "Empty");
}

#[test]
fn unconstrained_nullary_generic_constructor_remains_fail_loud() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("unconstrained.ch");
    write_file(
        &path,
        "\
type Box[a] =
  | Empty
  | Full { value: a }
out = print(Empty)
",
    );
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            dir.path().join("out").to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains(
            "unresolved host type variable `a`",
        ))
        .stderr(predicates::str::contains("[05-UNS-1]"))
        .stderr(predicates::str::contains("chelis#730"));
}
