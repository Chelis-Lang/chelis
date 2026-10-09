//! Chelis-Lang/chelis#3398: a borrow after an ordinary consume is consuming
//! fan-out that an inserted copy repairs (spec/04 section 8.3, spec/05
//! section 1.3.1), and every such copy is visible in `chelis cost`.
//!
//! `crates/chelis-types/tests/issue_3398_borrow_after_consume.rs` pins the
//! checker verdicts and the repair records. This file is the end-to-end half:
//! the repaired shapes check with an empty error list, evaluate, and compile to
//! C with the same output on both lanes; a use after `drop` stays a check
//! error; and `chelis cost --json` publishes each repair with its source span
//! and the later use that forced it, byte-identically across runs.

#[path = "common/mod.rs"]
mod common;

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use assert_cmd::Command;
use common::{build_and_run, gcc_available, write_file};
use serde_json::Value;
use tempfile::tempdir;

/// Every borrow-after-consume shape the decision names, plus the consuming
/// chain and the captures, in one canonical program.
const LANES: &str = r#"def eats(x: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(x, 0i32))
def look(x: &tensor[2, f32]) -> f32 = tensor_to_scalar(sum(x, 0i32))
def bump(x: tensor[2, f32]) -> tensor[2, f32] = add(x, to_tensor([10.0f32, 10.0f32]))
def s(t: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(mul(t, t), 0i32))
def sq(r: tensor[2, f32]) -> tensor[2, f32] = mul(r, r)
def eat_rows(m: tensor[3, 2, f32]) -> f32 = tensor_to_scalar(sum(sum(m, 0i32), 0i32))
def consume_then_borrow(x: tensor[2, f32]) -> f32 = {
  u = eats(x)
  add(u, look(x))
}
def value_then_grad(t: tensor[2, f32]) -> (f32, tensor[2, f32]) = {
  v = s(t)
  g = grad(s)(t)
  (v, g)
}
def total_then_vmap(m: tensor[3, 2, f32]) -> (f32, tensor[3, 2, f32]) = {
  t = eat_rows(m)
  (t, vmap(sq)(m))
}
def bump_then_read(x: tensor[2, f32]) -> tensor[2, f32] = {
  y = bump(x)
  z = realize(x)
  add(add(y, z), sigmoid(x))
}
def captures(x: tensor[2, f32]) -> f32 = {
  u = eats(x)
  g = fn (k: f32) -> add(k, look(x))
  h = fn (k: f32) -> add(k, eats(x))
  add(g(u), h(u))
}
def chain(x: tensor[2, f32]) -> f32 = {
  a = eats(x)
  b = eats(x)
  c = eats(x)
  add(add(a, b), add(c, look(x)))
}
borrowed = consume_then_borrow(to_tensor([1.0f32, 2.0f32]))
grad_out = value_then_grad(to_tensor([1.0f32, 2.0f32]))
vmap_out = total_then_vmap(to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32], [5.0f32, 6.0f32]]))
reuse = bump_then_read(to_tensor([1.0f32, 2.0f32]))
capture = captures(to_tensor([1.0f32, 2.0f32]))
fan_out = chain(to_tensor([1.0f32, 2.0f32]))
"#;

const LANES_OUTPUT: &str = "\
borrowed = 6.0
grad_out.0 = 5.0
grad_out.1 = tensor(shape=[2], data=[2.0, 4.0])
vmap_out.0 = 21.0
vmap_out.1 = tensor(shape=[3, 2], data=[1.0, 4.0, 9.0, 16.0, 25.0, 36.0])
reuse = tensor(shape=[2], data=[12.731058, 14.880797])
capture = 12.0
fan_out = 12.0
";

/// The tensor-carrying ADT form of the grad case. The C lane does not lower a
/// `grad` application over an ADT argument in this position whatever its
/// ownership (the grad-first order is refused the same way), so this shape is
/// pinned on the evaluator.
const ADT: &str = r#"type Lin[i] =
  | Lin { w: tensor[i, f32] }
def loss(p: Lin[2]) -> f32 = tensor_to_scalar(sum(mul(p.w, p.w), 0i32))
def value_then_grad(p: Lin[2]) -> (f32, Lin[2]) = {
  v = loss(p)
  g = grad(loss)(p)
  (v, g)
}
out = value_then_grad(Lin { w: to_tensor([1.0f32, 3.0f32]) })
"#;

fn chelis(args: &[&str]) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(args)
        .output()
        .expect("chelis should run")
}

fn write_source(dir: &Path, name: &str, source: &str) -> String {
    let path = dir.join(format!("{name}.ch"));
    write_file(&path, source);
    path.to_str().unwrap().to_string()
}

fn eval_stdout(source: &str, name: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = write_source(dir.path(), name, source);
    let out = chelis(&["eval", "--file", &path]);
    assert!(
        out.status.success(),
        "eval failed for `{name}`: {}\nstderr:\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("utf-8 stdout")
}

fn check_report(source: &str) -> (i32, Value) {
    let dir = tempdir().expect("tempdir");
    let path = write_source(dir.path(), "probe", source);
    let out = chelis(&["check", &path]);
    let report = serde_json::from_slice(&out.stdout).expect("check emits JSON");
    (out.status.code().expect("exit code"), report)
}

fn cost_json_bytes(path: &str) -> Vec<u8> {
    let out = chelis(&["cost", path, "--json"]);
    assert!(
        out.status.success(),
        "chelis cost --json failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

/// The source text a `surf:a..b` span identity names.
fn span_text<'a>(source: &'a str, id: &str) -> &'a str {
    let (start, end) = id
        .strip_prefix("surf:")
        .and_then(|range| range.split_once(".."))
        .unwrap_or_else(|| panic!("`{id}` is not a Surf span identity"));
    &source[start.parse::<usize>().unwrap()..end.parse::<usize>().unwrap()]
}

#[test]
fn repaired_shapes_agree_on_the_evaluator_and_compiled_c() {
    let eval = eval_stdout(LANES, "issue3398_lanes");
    assert_eq!(eval, LANES_OUTPUT);
    if !gcc_available() {
        eprintln!("skipping build lane: c compiler not available");
        return;
    }
    assert_eq!(build_and_run(LANES, "issue3398_lanes"), eval);
}

#[test]
fn adt_grad_after_its_value_is_repaired_on_the_evaluator() {
    assert_eq!(
        eval_stdout(ADT, "issue3398_adt"),
        "out.0 = 10.0\nout.1 = Lin(tensor(shape=[2], data=[2.0, 6.0]))\n"
    );
}

/// Inserted copies are not diagnostics: a repaired program's check report
/// has an empty error list and a zero exit.
#[test]
fn check_reports_no_errors_for_repaired_programs() {
    for source in [LANES, ADT] {
        let (code, report) = check_report(source);
        assert_eq!(report["errors"], Value::Array(Vec::new()), "{report}");
        assert_eq!(report["score"], 1.0, "{report}");
        assert_eq!(code, 0, "{report}");
    }
}

#[test]
fn a_borrow_after_drop_stays_a_check_error() {
    let source = "def look(x: &tensor[2, f32]) -> f32 = tensor_to_scalar(sum(x, 0i32))\n\
                  def f(x: tensor[2, f32]) -> f32 = {\n  u = drop(x)\n  look(x)\n}\n";
    let (code, report) = check_report(source);
    let errors = report["errors"].as_array().expect("errors array");
    assert!(
        errors.iter().any(|error| error["kind"] == "UseAfterConsume"
            && error["message"]
                .as_str()
                .unwrap()
                .contains("call to `drop`")),
        "{report}"
    );
    assert_ne!(code, 0, "{report}");
}

/// The published shape: `copy_repairs` is an array of objects with exactly
/// these members, and each later use carries exactly `at` and `kind`.
#[test]
fn cost_json_publishes_each_copy_repair() {
    let dir = tempdir().expect("tempdir");
    let path = write_source(dir.path(), "lanes", LANES);
    let json: Value = serde_json::from_slice(&cost_json_bytes(&path)).expect("cost JSON");
    let repairs = json["copy_repairs"].as_array().expect("copy_repairs array");
    let members = |value: &Value| {
        value
            .as_object()
            .expect("object")
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>()
    };
    for repair in repairs {
        assert_eq!(
            members(repair),
            [
                "binding",
                "consumed_by",
                "copy_at",
                "declaration",
                "forced_by"
            ]
            .map(String::from)
            .into(),
            "{repair}"
        );
        for later in repair["forced_by"].as_array().expect("forced_by array") {
            assert_eq!(
                members(later),
                ["at", "kind"].map(String::from).into(),
                "{later}"
            );
            assert!(
                ["consume", "borrow", "capture", "drop"].contains(&later["kind"].as_str().unwrap()),
                "{later}"
            );
        }
    }

    // Semantics: where each copy sits, and which later use forced it.
    let in_decl = |name: &str| {
        repairs
            .iter()
            .filter(|repair| repair["declaration"] == name)
            .collect::<Vec<_>>()
    };
    let summary = |repair: &Value| {
        (
            repair["binding"].as_str().unwrap().to_string(),
            span_text(LANES, repair["copy_at"].as_str().unwrap()).to_string(),
            repair["consumed_by"].as_str().unwrap().to_string(),
            repair["forced_by"]
                .as_array()
                .unwrap()
                .iter()
                .map(|later| {
                    (
                        later["kind"].as_str().unwrap().to_string(),
                        span_text(LANES, later["at"].as_str().unwrap()).to_string(),
                    )
                })
                .collect::<Vec<_>>(),
        )
    };
    let owned = |pairs: &[(&str, &str)]| {
        pairs
            .iter()
            .map(|(kind, at)| (kind.to_string(), at.to_string()))
            .collect::<Vec<_>>()
    };
    let one = |name: &str| {
        let found = in_decl(name);
        assert_eq!(found.len(), 1, "{name}: {found:?}");
        summary(found[0])
    };
    assert_eq!(
        one("consume_then_borrow"),
        (
            "x".into(),
            "eats(x)".into(),
            "call to `eats`".into(),
            owned(&[("borrow", "x")])
        )
    );
    assert_eq!(
        one("value_then_grad"),
        (
            "t".into(),
            "s(t)".into(),
            "call to `s`".into(),
            owned(&[("borrow", "t")])
        )
    );
    assert_eq!(
        one("total_then_vmap"),
        (
            "m".into(),
            "eat_rows(m)".into(),
            "call to `eat_rows`".into(),
            owned(&[("borrow", "m")])
        )
    );
    let captures = one("captures");
    assert_eq!(captures.1, "eats(x)");
    assert_eq!(
        captures
            .3
            .iter()
            .map(|(kind, _)| kind.as_str())
            .collect::<Vec<_>>(),
        ["capture", "capture"]
    );
    let chain = in_decl("chain")
        .into_iter()
        .map(summary)
        .collect::<Vec<_>>();
    assert_eq!(
        chain
            .iter()
            .map(|repair| repair.3[0].0.as_str())
            .collect::<Vec<_>>(),
        ["consume", "consume", "borrow"],
        "{chain:?}"
    );
    let reuse = in_decl("bump_then_read")
        .into_iter()
        .map(summary)
        .collect::<Vec<_>>();
    assert_eq!(
        reuse,
        [
            (
                "x".into(),
                "bump(x)".into(),
                "call to `bump`".into(),
                owned(&[("consume", "x")])
            ),
            (
                "x".into(),
                "realize(x)".into(),
                "realize".into(),
                owned(&[("borrow", "x")])
            ),
        ]
    );
}

#[test]
fn cost_json_without_fan_out_reports_no_repairs() {
    let dir = tempdir().expect("tempdir");
    let path = write_source(
        dir.path(),
        "plain",
        "def look(x: &tensor[2, f32]) -> f32 = tensor_to_scalar(sum(x, 0i32))\n\
         def eats(x: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(x, 0i32))\n\
         def f(x: tensor[2, f32]) -> f32 = add(look(x), eats(x))\n",
    );
    let json: Value = serde_json::from_slice(&cost_json_bytes(&path)).expect("cost JSON");
    assert_eq!(json["copy_repairs"], Value::Array(Vec::new()), "{json}");
}

/// Determinism: separate processes over the same file emit the same bytes.
#[test]
fn cost_json_is_byte_identical_across_runs() {
    let dir = tempdir().expect("tempdir");
    let path = write_source(dir.path(), "lanes", LANES);
    let first = cost_json_bytes(&path);
    for _ in 0..3 {
        assert_eq!(cost_json_bytes(&path), first);
    }
    let copy = dir.path().join("again.ch");
    fs::copy(&path, &copy).expect("copy fixture");
    let again: Value =
        serde_json::from_slice(&cost_json_bytes(copy.to_str().unwrap())).expect("cost JSON");
    let first: Value = serde_json::from_slice(&first).expect("cost JSON");
    assert_eq!(again["copy_repairs"], first["copy_repairs"]);
}

#[test]
fn cost_text_lists_each_copy_repair() {
    let dir = tempdir().expect("tempdir");
    let path = write_source(dir.path(), "lanes", LANES);
    let out = chelis(&["cost", &path]);
    assert!(out.status.success());
    let text = String::from_utf8(out.stdout).expect("utf-8");
    let line = text
        .lines()
        .find(|line| line.starts_with("copy_repair consume_then_borrow: "))
        .unwrap_or_else(|| panic!("no repair line for consume_then_borrow:\n{text}"));
    assert!(
        line.contains("`x` copied at surf:")
            && line.contains("(call to `eats`), for borrow at surf:"),
        "{line}"
    );
}
