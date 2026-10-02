//! chelis#2597: an option at a top-level exit renders in compiled C exactly as
//! `chelis eval` renders it ([05-OBS-6], [05-OP-32]).
//!
//! An option nested inside a list, tuple or data-type value already rendered
//! through the runtime. A root whose whole value is an option was rejected at
//! build time, an option component of a manifested tuple or record root built
//! and then aborted at run time, and `print` of an option was rejected. Each
//! now renders `Some(...)` or `None`.

mod common;

use assert_cmd::Command;
use common::{gcc_available, link_generated};
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn eval_stdout(dir: &Path, file: &str) -> String {
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(dir)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", file])
        .output()
        .expect("run chelis eval");
    let stdout = String::from_utf8(output.stdout).expect("stdout UTF-8");
    assert!(
        output.status.success(),
        "eval failed\nstdout: {stdout}\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    stdout
}

/// Build `source` to C, link it, run it, and return its stdout. `None` when
/// no host C compiler exists.
fn compiled_stdout(dir: &Path, stem: &str) -> Option<String> {
    let out = dir.join(format!("{stem}-out"));
    let build = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(dir)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            &format!("{stem}.ch"),
            "--target",
            "c",
            "--output",
            out.to_str().expect("UTF-8 output dir"),
        ])
        .output()
        .expect("run chelis build");
    assert!(
        build.status.success(),
        "build failed\nstderr: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    if !gcc_available() {
        eprintln!("skipped the native run of {stem}.c: no host C compiler");
        return None;
    }
    let status = link_generated(&out, &format!("{stem}.c"), stem);
    assert!(status.success(), "link failed: {status}");
    let run = std::process::Command::new(out.join(stem))
        .output()
        .expect("compiled binary runs");
    let stdout = String::from_utf8(run.stdout).expect("compiled stdout UTF-8");
    assert!(
        run.status.success(),
        "compiled {stem} failed: {}\nstdout: {stdout}\nstderr: {}",
        run.status,
        String::from_utf8_lossy(&run.stderr)
    );
    Some(stdout)
}

/// Write `source`, then require eval to print `expected` and compiled C to
/// print the same bytes.
fn assert_lanes_print(stem: &str, source: &str, expected: &str) {
    let dir = tempdir().expect("tempdir");
    fs::write(dir.path().join(format!("{stem}.ch")), source).expect("write source");
    let evaluated = eval_stdout(dir.path(), &format!("{stem}.ch"));
    assert_eq!(evaluated, expected, "eval output");
    let Some(compiled) = compiled_stdout(dir.path(), stem) else {
        return;
    };
    assert_eq!(compiled, evaluated, "compiled C must print eval's bytes");
}

#[test]
fn a_root_whose_value_is_an_option_renders_in_both_lanes() {
    assert_lanes_print(
        "whole",
        "def none_i64() -> Option[i64] = None
def latest(flag: bool) -> Option[string] = fold(fn (acc: Option[string], x: string) -> Some(x), None, [\"p\", \"q\"])
a = Some(string_concat(\"ab\", \"c\"))
d = none_i64()
g = Some(Some(2i64))
h = Some(to_tensor([1.0f32, 2.0f32]))
picked = latest(true)
",
        "none_i64 = None
a = Some(abc)
d = None
g = Some(Some(2))
h = Some(tensor(shape=[2], data=[1.0, 2.0]))
picked = Some(q)
",
    );
}

#[test]
fn an_option_component_of_a_tuple_or_record_root_renders_in_both_lanes() {
    assert_lanes_print(
        "components",
        "def none_of(seed: i64) -> Option[i64] = None
type Zone =
  | Zone { name: string, next: Option[(i64, Option[i64])] }
b = (Some(string_concat(\"ab\", \"c\")), 1i64)
c = ((Some(1.5f32), 2i64), 3i64)
t = (Some(true), Some(()), Some(2.5f64), Some(cast(3, i8)), none_of(0i64))
e = Zone { name: \"UTC\", next: Some((0i64, None)) }
",
        "b.0 = Some(abc)
b.1 = 1
c.0.0 = Some(1.5)
c.0.1 = 2
c.1 = 3
t.0 = Some(true)
t.1 = Some(())
t.2 = Some(2.5)
t.3 = Some(3)
t.4 = None
e.name = UTC
e.next = Some((0, None))
",
    );
}

#[test]
fn printing_an_option_renders_in_both_lanes() {
    assert_lanes_print(
        "printed",
        "def pair() -> (i64, Option[i64]) = (2i64, None)
def show() -> unit ! { IO } = {
  _ = print(Some(1i64))
  print(Some(pair()))
}
r = show()
",
        "Some(1)\nSome((2, None))\npair.0 = 2\npair.1 = None\nr = ()\n",
    );
}

#[test]
fn an_option_nested_in_a_list_keeps_its_existing_rendering() {
    // Control: the container path that already rendered options.
    assert_lanes_print(
        "nested",
        "f = [Some(1i64), None]\n",
        "f = [Some(1), None]\n",
    );
}
