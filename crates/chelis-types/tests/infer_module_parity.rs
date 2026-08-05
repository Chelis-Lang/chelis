//! Behavior baseline for the `infer` module split (openspec
//! `modularize-type-inference`).
//!
//! The split moves ~21,000 lines of inference code out of one file and into a
//! role-based module tree. Every step of that move is supposed to be
//! behavior-preserving, but the existing suite exercises inference through
//! many narrow issue-specific assertions: a reordered diagnostic or a dropped
//! metadata stamp can slip between them. These fixtures close that gap by
//! recording the *whole* observable output of a check — the annotated Deep
//! tree with its metadata, the type environment, the inference statistics,
//! and the ordered diagnostics — and comparing it byte for byte.
//!
//! The recorded baselines live in `tests/fixtures/infer_parity/`. A missing
//! baseline is written on first run and the test FAILS, so a deleted or
//! never-recorded fixture can never pass silently. A baseline is only allowed
//! to change when the change is a deliberate, reviewed behavior change; for
//! this refactor the correct number of baseline edits is zero.

use std::fs;
use std::path::PathBuf;

use chelis_deep::Expr;
use chelis_deep::printer::print_canonical;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::{CheckedProgram, InferResult, check_ir_program};

fn surf_to_deep(source: &str, label: &str) -> Vec<Expr> {
    let decls = parse_surf(source).unwrap_or_else(|e| panic!("{label}: surf parse failed: {e:?}"));
    chelis_macros::expand_program(
        &desugar_program(&decls),
        &chelis_macros::ExpansionOptions::default(),
    )
    .unwrap_or_else(|e| panic!("{label}: macro expansion failed: {e:?}"))
    .into_exprs()
}

/// The full observable output of an accepted check: the annotated tree (which
/// carries the type metadata inline), the type environment, and the honest
/// visit counters.
fn accepted_snapshot(checked: &CheckedProgram) -> String {
    let mut out = String::new();

    out.push_str("== annotated deep ==\n");
    out.push_str(&print_canonical(checked.annotated_exprs()));
    if !out.ends_with('\n') {
        out.push('\n');
    }

    out.push_str("\n== type env ==\n");
    let mut env: Vec<(&String, String)> = checked
        .type_env()
        .iter()
        .map(|(name, ty)| (name, chelis_deep::printer::print_expr_flat(ty)))
        .collect();
    env.sort();
    for (name, ty) in env {
        out.push_str(&format!("{name} : {ty}\n"));
    }

    out.push_str("\n== signature inference ==\n");
    let mut functions: Vec<_> = checked.signature_inference().functions.iter().collect();
    functions.sort_by_key(|(name, _)| *name);
    for (binding, function) in functions {
        out.push_str(&format!("binding = {binding}\n"));
        out.push_str(&format!("  name = {}\n", function.name));
        out.push_str(&format!(
            "  authored_signature = {}\n",
            function.authored_signature
        ));
        out.push_str(&format!(
            "  authored_signature_type = {:?}\n",
            function.authored_signature_type
        ));
        out.push_str(&format!(
            "  recursive_cycle = {}\n",
            function.recursive_cycle
        ));
        out.push_str(&format!(
            "  checked_signature = {:?}\n",
            function.checked_signature
        ));
        out.push_str(&format!(
            "  display_signature = {:?}\n",
            function.display_signature
        ));
        for param in &function.params {
            out.push_str(&format!(
                "  param[{}] name = {}; written = {}; inferred_read_only = {}; \
                 checked_type = {:?}; display_type = {:?}\n",
                param.index,
                param.name,
                param.written,
                param.inferred_read_only,
                param.checked_type,
                param.display_type,
            ));
        }
    }

    out.push_str("\n== infer stats ==\n");
    let stats = checked.infer_stats();
    out.push_str(&format!(
        "typed_nodes = {}\ntotal_nodes = {}\n",
        stats.typed_nodes, stats.total_nodes
    ));

    out
}

/// The full observable output of a rejected check: every diagnostic in order,
/// with the fields a consumer can actually see.
fn rejected_snapshot(report: &InferResult) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "typed_nodes = {}\ntotal_nodes = {}\ndiagnostics = {}\n\n",
        report.typed_nodes,
        report.total_nodes,
        report.errors.len()
    ));
    for (index, error) in report.errors.iter().enumerate() {
        out.push_str(&format!("[{index}] kind = {:?}\n", error.kind));
        out.push_str(&format!("    message = {}\n", error.message));
        out.push_str(&format!("    severity = {}\n", error.severity));
        out.push_str(&format!("    expected = {:?}\n", error.expected));
        out.push_str(&format!("    got = {:?}\n", error.got));
        out.push_str(&format!("    span_offset = {:?}\n", error.span_offset));
        out.push_str(&format!("    span_id = {:?}\n", error.span_id));
        for hint in &error.suggestions {
            out.push_str(&format!("    hint = {hint}\n"));
        }
    }
    out
}

fn baseline_path(name: &str) -> PathBuf {
    let (section, test_name) = name
        .split_once('_')
        .expect("the parity fixture name has a section prefix");
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/infer_parity")
        .join(format!("infer_module_parity__{section}__{test_name}.snap"))
}

/// Compare against the recorded baseline. A missing baseline is recorded and
/// then reported as a failure: the point of the fixture is the comparison, so
/// a first run must never be mistaken for a passing run.
fn assert_matches_baseline(name: &str, actual: &str) {
    let path = baseline_path(name);
    if !path.exists() {
        fs::create_dir_all(path.parent().expect("fixture dir")).expect("create fixture dir");
        fs::write(&path, actual).expect("write baseline");
        panic!(
            "{name}: no recorded baseline. Wrote {}. Review it, then run the test again.",
            path.display()
        );
    }
    let expected = fs::read_to_string(&path).expect("read baseline");
    let shown = path.display();
    assert!(
        expected == actual,
        "{name}: inference output differs from the recorded baseline at {shown}.\n\
         The infer module split must not change checked trees, metadata, \
         statistics, or diagnostics.\n\
         --- recorded ---\n{expected}\n--- actual ---\n{actual}"
    );
}

/// A program that must type-check cleanly, compared on its whole output.
fn assert_accepted_parity(name: &str, source: &str) {
    let deep = surf_to_deep(source, name);
    match check_ir_program(&deep) {
        Ok(checked) => assert_matches_baseline(name, &accepted_snapshot(&checked)),
        Err(report) => {
            let msgs: Vec<&str> = report.errors.iter().map(|e| e.message.as_str()).collect();
            panic!("{name}: fixture is supposed to type-check, but it failed: {msgs:#?}");
        }
    }
}

/// A program that must be rejected, compared on its whole ordered diagnostic
/// list.
fn assert_rejected_parity(name: &str, source: &str) {
    let deep = surf_to_deep(source, name);
    match check_ir_program(&deep) {
        Ok(_) => panic!("{name}: fixture is supposed to be rejected, but it type-checked"),
        Err(report) => {
            assert!(
                !report.errors.is_empty(),
                "{name}: a rejection must carry at least one diagnostic"
            );
            assert_matches_baseline(name, &rejected_snapshot(&report));
        }
    }
}

// ── Accepted fixtures ───────────────────────────────────────────────────────

#[test]
fn accepted_generic_calls() {
    assert_accepted_parity(
        "accepted_generic_calls",
        r#"module Parity.GenericCalls
def apply_twice(f: (int32) -> int32, x: int32) -> int32 = f(f(x))
def bump(n: int32) -> int32 = add(n, 1)
def run() -> int32 = apply_twice(bump, 3)
def identity_pair(a: int32, b: f32) -> (int32, f32) = (a, b)
def use_pair() -> int32 = identity_pair(1, 2.0).0
"#,
    );
}

#[test]
fn accepted_shape_operations() {
    assert_accepted_parity(
        "accepted_shape_operations",
        r#"module Parity.ShapeOps
def project(x: tensor[64, 32, f32], w: tensor[32, 8, f32]) -> tensor[64, 8, f32] = matmul(x, w)
def broadcast_bias(b: tensor[1, f32]) -> tensor[64, f32] = expand(b, 0, 64i64)
def swap(x: tensor[4, 6, f32]) -> tensor[6, 4, f32] = permute(x, 1, 0)
def collapse(x: tensor[4, 6, f32]) -> tensor[24, f32] = reshape(x, [cast(24, int64)])
def reduce_rows(x: tensor[4, 6, f32]) -> tensor[4, f32] = mean(x, 1)
"#,
    );
}

#[test]
fn accepted_collections() {
    assert_accepted_parity(
        "accepted_collections",
        r#"module Parity.Collections
values: List[int32] = [1, 2, 3, 4]
doubled = map(fn (v: int32) -> mul(v, 2), values)
kept = filter(fn (v: int32) -> gt(v, 2), values)
total = fold(fn (acc: int32, v: int32) -> add(acc, v), 0, values)
count = len(values)
first_two = take(values, cast(2, int64))
paired = zip(values, values)
"#,
    );
}

#[test]
fn accepted_records() {
    assert_accepted_parity(
        "accepted_records",
        r#"module Parity.Records
type Point =
  | Point { x: f32, y: f32 }
def make(a: f32, b: f32) -> Point = Point { x: a, y: b }
def read_x(p: Point) -> f32 = p.x
def shift(p: Point, dx: f32) -> Point = Point { x: add(p.x, dx), y: p.y }
"#,
    );
}

#[test]
fn accepted_patterns() {
    assert_accepted_parity(
        "accepted_patterns",
        r#"module Parity.Patterns
type Shape =
  | Circle(f32)
  | Rect(f32, f32)
def area(s: Shape) -> f32 = match s with {
  | Circle(r) => mul(3.14, mul(r, r))
  | Rect(w, h) => mul(w, h)
}
def first_or(xs: List[int32], fallback: int32) -> int32 = match xs with {
  | Cons(head, tail) => head
  | Nil => fallback
}
"#,
    );
}

#[test]
fn accepted_transforms() {
    assert_accepted_parity(
        "accepted_transforms",
        r#"module Parity.Transforms
def process(x: tensor[features, f32]) -> tensor[features, f32] = relu(x)
def batch_process(xs: tensor[batch, features, f32]) -> tensor[batch, features, f32] = xs |> vmap(process)
"#,
    );
}

// ── Rejected fixtures ───────────────────────────────────────────────────────

#[test]
fn rejected_numeric_restrictions() {
    assert_rejected_parity(
        "rejected_numeric_restrictions",
        r#"module Parity.NumericRestrictions
def mixed(n: int32) -> f32 = add(1.0, n)
def int_div(a: int32, b: int32) -> int32 = div(a, b)
"#,
    );
}

#[test]
fn rejected_invalid_shapes() {
    assert_rejected_parity(
        "rejected_invalid_shapes",
        r#"module Parity.InvalidShapes
def bad_matmul(x: tensor[64, 32, f32], w: tensor[16, 8, f32]) -> tensor[64, 8, f32] = matmul(x, w)
def bad_reshape(x: tensor[4, 6, f32]) -> tensor[25, f32] = reshape(x, [cast(25, int64)])
"#,
    );
}

#[test]
fn rejected_collection_callbacks() {
    assert_rejected_parity(
        "rejected_collection_callbacks",
        r#"module Parity.CollectionCallbacks
values: List[int32] = [1, 2, 3]
wrong_element = map(fn (v: f32) -> mul(v, 2.0), values)
wrong_predicate = filter(fn (v: int32) -> v, values)
"#,
    );
}

#[test]
fn rejected_records() {
    assert_rejected_parity(
        "rejected_records",
        r#"module Parity.RecordsRejected
type Point =
  | Point { x: f32, y: f32 }
def read_missing(p: Point) -> f32 = p.z
def build_wrong(a: int32) -> Point = Point { x: a, y: 1.0 }
"#,
    );
}

#[test]
fn rejected_patterns() {
    assert_rejected_parity(
        "rejected_patterns",
        r#"module Parity.PatternsRejected
type Shape =
  | Circle(f32)
  | Rect(f32, f32)
def unknown_ctor(s: Shape) -> f32 = match s with {
  | Circle(r) => r
  | Triangle(b, h) => mul(b, h)
  | Rect(w, h) => mul(w, h)
}
def bad_result(s: Shape) -> f32 = match s with {
  | Circle(r) => r
  | Rect(w, h) => 1
}
"#,
    );
}

#[test]
fn rejected_transforms() {
    assert_rejected_parity(
        "rejected_transforms",
        r#"module Parity.TransformsRejected
def label(x: tensor[features, f32]) -> string = "constant"
def batch_label(xs: tensor[batch, features, f32]) -> tensor[batch, features, f32] = xs |> vmap(label)
"#,
    );
}
