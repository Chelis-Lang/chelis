//! Behavioral checks across inference routes. Historical byte-for-byte
//! snapshots pinned source metadata and rejection wording, not language rules.

use chelis_deep::Expr;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_ir_program;

fn surf_to_deep(source: &str, label: &str) -> Vec<Expr> {
    let decls = parse_surf(source).unwrap_or_else(|e| panic!("{label}: surf parse failed: {e:?}"));
    chelis_macros::expand_program(
        &desugar_program(&decls).expect("Surf fixture must desugar"),
        &chelis_macros::ExpansionOptions::default(),
    )
    .unwrap_or_else(|e| panic!("{label}: macro expansion failed: {e:?}"))
    .into_exprs()
}

fn assert_accepted_parity(name: &str, source: &str, expected_types: &[(&str, &str)]) {
    let deep = surf_to_deep(source, name);
    let checked = check_ir_program(&deep)
        .unwrap_or_else(|report| panic!("{name} must check cleanly: {:#?}", report.errors));
    assert_eq!(
        checked.type_env().len(),
        expected_types.len(),
        "{name}: exported binding set changed"
    );
    for &(binding, expected_type) in expected_types {
        let ty = checked
            .type_env()
            .get(binding)
            .unwrap_or_else(|| panic!("{name}: missing exported binding {binding}"));
        assert_eq!(
            chelis_deep::printer::print_expr_flat(ty),
            expected_type,
            "{name}: wrong public type for {binding}"
        );
    }
}

fn assert_rejected_parity(name: &str, source: &str, expected: &[(&str, &[&str])]) {
    let deep = surf_to_deep(source, name);
    let report = check_ir_program(&deep)
        .err()
        .unwrap_or_else(|| panic!("{name}: invalid program was admitted"));
    assert_eq!(
        report.errors.len(),
        expected.len(),
        "{name}: missing or cascading extra rejection: {:#?}",
        report.errors
    );
    let mut unmatched = report.errors.iter().collect::<Vec<_>>();
    for &(kind, operands) in expected {
        let position = unmatched.iter().position(|error| {
            error.kind.diagnostic_name() == kind
                && if let Some(identifier) = error.kind.unresolved_identifier() {
                    operands.len() == 1 && operands[0] == identifier
                } else {
                    operands
                        .iter()
                        .all(|operand| error.message.contains(operand))
                }
        });
        let at = position.unwrap_or_else(|| {
            panic!(
                "{name}: missing {kind} on {operands:?}: {:#?}",
                report.errors
            )
        });
        unmatched.swap_remove(at);
    }
}

// ── Accepted fixtures ───────────────────────────────────────────────────────

#[test]
fn accepted_generic_calls() {
    assert_accepted_parity(
        "accepted_generic_calls",
        r#"module Parity.GenericCalls
def apply_twice(f: (i32) -> i32, x: i32) -> i32 = f(f(x))
def bump(n: i32) -> i32 = add(n, 1)
def run() -> i32 = apply_twice(bump, 3)
def identity_pair(a: i32, b: f32) -> (i32, f32) = (a, b)
def use_pair() -> i32 = identity_pair(1, 2.0).0
"#,
        &[
            (
                "apply_twice",
                "(t-fn {} (t-fn {} (t-prim {} i32) (t-prim {} i32)) (t-prim {} i32) (t-prim {} i32))",
            ),
            ("bump", "(t-fn {} (t-prim {} i32) (t-prim {} i32))"),
            (
                "identity_pair",
                "(t-fn {} (t-prim {} i32) (t-prim {} f32) (t-tuple {} (t-prim {} i32) (t-prim {} f32)))",
            ),
            ("run", "(t-fn {} (t-prim {} i32))"),
            ("use_pair", "(t-fn {} (t-prim {} i32))"),
        ],
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
def collapse(x: tensor[4, 6, f32]) -> tensor[24, f32] = reshape(x, [cast(24, i64)])
def reduce_rows(x: tensor[4, 6, f32]) -> tensor[4, f32] = mean(x, 1)
"#,
        &[
            (
                "broadcast_bias",
                "(t-fn {} (t-tensor {} (d-lit {} 1) (t-prim {} f32)) (t-tensor {} (d-lit {} 64) (t-prim {} f32)))",
            ),
            (
                "collapse",
                "(t-fn {} (t-tensor {} (d-lit {} 4) (d-lit {} 6) (t-prim {} f32)) (t-tensor {} (d-lit {} 24) (t-prim {} f32)))",
            ),
            (
                "project",
                "(t-fn {} (t-tensor {} (d-lit {} 64) (d-lit {} 32) (t-prim {} f32)) (t-tensor {} (d-lit {} 32) (d-lit {} 8) (t-prim {} f32)) (t-tensor {} (d-lit {} 64) (d-lit {} 8) (t-prim {} f32)))",
            ),
            (
                "reduce_rows",
                "(t-fn {} (t-tensor {} (d-lit {} 4) (d-lit {} 6) (t-prim {} f32)) (t-tensor {} (d-lit {} 4) (t-prim {} f32)))",
            ),
            (
                "swap",
                "(t-fn {} (t-tensor {} (d-lit {} 4) (d-lit {} 6) (t-prim {} f32)) (t-tensor {} (d-lit {} 6) (d-lit {} 4) (t-prim {} f32)))",
            ),
        ],
    );
}

#[test]
fn accepted_collections() {
    assert_accepted_parity(
        "accepted_collections",
        r#"module Parity.Collections
values: List[i32] = [1, 2, 3, 4]
doubled = map(fn (v: i32) -> mul(v, 2), values)
kept = filter(fn (v: i32) -> gt(v, 2), values)
total = fold(fn (acc: i32, v: i32) -> add(acc, v), 0, values)
item_total = len(values)
first_two = take(values, cast(2, i64))
paired = zip(values, values)
"#,
        &[
            ("doubled", "(t-adt {} List (t-prim {} i32))"),
            ("first_two", "(t-adt {} List (t-prim {} i32))"),
            ("item_total", "(t-prim {} i64)"),
            ("kept", "(t-adt {} List (t-prim {} i32))"),
            (
                "paired",
                "(t-adt {} List (t-tuple {} (t-prim {} i32) (t-prim {} i32)))",
            ),
            ("total", "(t-prim {} i32)"),
            ("values", "(t-adt {} List (t-prim {} i32))"),
        ],
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
        &[
            (
                "make",
                "(t-fn {} (t-prim {} f32) (t-prim {} f32) (t-adt {} Point))",
            ),
            ("read_x", "(t-fn {} (t-adt {} Point) (t-prim {} f32))"),
            (
                "shift",
                "(t-fn {} (t-adt {} Point) (t-prim {} f32) (t-adt {} Point))",
            ),
        ],
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
def first_or(xs: List[i32], fallback: i32) -> i32 = match xs with {
  | Cons(head, tail) => head
  | Nil => fallback
}
"#,
        &[
            ("area", "(t-fn {} (t-adt {} Shape) (t-prim {} f32))"),
            (
                "first_or",
                "(t-fn {} (t-adt {} List (t-prim {} i32)) (t-prim {} i32) (t-prim {} i32))",
            ),
        ],
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
        &[
            (
                "batch_process",
                "(t-fn {} (t-tensor {} (d-name {} batch) (d-name {} features) (t-prim {} f32)) (t-tensor {} (d-name {} batch) (d-name {} features) (t-prim {} f32)))",
            ),
            (
                "process",
                "(t-fn {} (t-tensor {} (d-name {} features) (t-prim {} f32)) (t-tensor {} (d-name {} features) (t-prim {} f32)))",
            ),
        ],
    );
}

// ── Rejected fixtures ───────────────────────────────────────────────────────

#[test]
fn rejected_numeric_restrictions() {
    assert_rejected_parity(
        "rejected_numeric_restrictions",
        r#"module Parity.NumericRestrictions
def mixed(n: i32) -> f32 = add(1.0, n)
def int_div(a: i32, b: i32) -> i32 = div(a, b)
"#,
        &[
            ("PrecisionMismatch", &["f32", "i32"]),
            ("PrecisionMismatch", &["div"]),
        ],
    );
}

#[test]
fn rejected_invalid_shapes() {
    assert_rejected_parity(
        "rejected_invalid_shapes",
        r#"module Parity.InvalidShapes
def bad_matmul(x: tensor[64, 32, f32], w: tensor[16, 8, f32]) -> tensor[64, 8, f32] = matmul(x, w)
def bad_reshape(x: tensor[4, 6, f32]) -> tensor[25, f32] = reshape(x, [cast(25, i64)])
"#,
        &[
            ("DimensionMismatch", &["matmul", "32", "16"]),
            ("DimensionMismatch", &["reshape", "25", "24"]),
        ],
    );
}

#[test]
fn rejected_collection_callbacks() {
    assert_rejected_parity(
        "rejected_collection_callbacks",
        r#"module Parity.CollectionCallbacks
values: List[i32] = [1, 2, 3]
wrong_element = map(fn (v: f32) -> mul(v, 2.0), values)
wrong_predicate = filter(fn (v: i32) -> v, values)
"#,
        &[
            ("PrecisionMismatch", &["f32", "i32"]),
            ("PrecisionMismatch", &["filter", "bool"]),
        ],
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
def build_wrong(a: i32) -> Point = Point { x: a, y: 1.0 }
"#,
        &[
            ("TypeMismatch", &["Point", "z"]),
            ("PrecisionMismatch", &["i32", "f32"]),
        ],
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
        &[
            ("UnknownConstructor", &["Triangle"]),
            ("UnboundVariable", &["b"]),
            ("UnboundVariable", &["h"]),
            ("PrecisionMismatch", &["f32", "i32"]),
        ],
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
        &[("TypeMismatch", &["batch_label", "string", "f32"])],
    );
}
