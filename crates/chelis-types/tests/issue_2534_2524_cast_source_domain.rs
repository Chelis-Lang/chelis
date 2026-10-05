//! chelis#2534 and chelis#2524: [05-OP-63] admits exactly the numeric dtypes
//! and `bool` as a `cast` source, whether the source is written directly or
//! bound later through a lambda parameter.
//!
//! Each program below checked at score 1 on `main` `32e4122e4`. A checked
//! `cast` to a concrete target decided a settled scalar source without its
//! source domain, so a `string` source passed directly and through a lambda
//! (chelis#2524). A `cast` to a declaration's dtype binder imposed nothing on a
//! source that was still an inference variable at the cast, so a `string` or
//! an unbounded binder reaching it through a lambda passed, and `eval` trapped
//! (chelis#2534). The variable source now suspends on the same decision the
//! settled source gets, and is decided when it binds or reported at the
//! declaration boundary if it never does.
//!
//! Every program is asserted on both checker ingresses (chelis#1107), and each
//! rejection has an accepted twin.

use chelis_deep::Expr;
use chelis_macros::{ExpansionOptions, expand_program};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::errors::CheckError;
use chelis_types::{check_ir_program, check_typed_program};

fn desugared(source: &str) -> Vec<Expr> {
    let decls = parse_surf(source).unwrap_or_else(|e| panic!("surf must parse: {source}\n{e:?}"));
    desugar_program(&decls).expect("Surf fixture must desugar")
}

fn expanded(source: &str) -> Vec<Expr> {
    expand_program(&desugared(source), &ExpansionOptions::default())
        .expect("macro expand")
        .into_exprs()
}

fn rendered(errors: &[CheckError]) -> Vec<String> {
    let mut out: Vec<String> = errors
        .iter()
        .map(|e| format!("[{:?}] {}", e.kind, e.message))
        .collect();
    out.sort();
    out
}

fn agreed_diagnostics(source: &str) -> Vec<String> {
    let typed = match check_typed_program(&desugared(source)) {
        Ok(_) => Vec::new(),
        Err(result) => rendered(&result.errors),
    };
    let ir = match check_ir_program(&expanded(source)) {
        Ok(_) => Vec::new(),
        Err(result) => rendered(&result.errors),
    };
    assert_eq!(
        typed, ir,
        "both checker ingresses must return the same diagnostics for:\n{source}"
    );
    typed
}

fn accepts(source: &str) {
    let diagnostics = agreed_diagnostics(source);
    assert!(
        diagnostics.is_empty(),
        "must type-check:\n{source}\ngot {diagnostics:#?}"
    );
}

/// A rejection preserving its kind and directional operand types at both ingresses.
fn rejects_with(source: &str, kind: &str, expected: &str, got: &str) {
    for checked in [
        check_typed_program(&desugared(source)),
        check_ir_program(&expanded(source)),
    ] {
        let report = checked.expect_err("an invalid cast or later call must reject");
        assert!(
            report.errors.iter().any(|error| {
                error.kind.diagnostic_name() == kind
                    && error.expected.as_deref() == Some(expected)
                    && error.got.as_deref() == Some(got)
            }),
            "{source}: expected {kind} {expected} -> {got}, got {:?}",
            report.errors
        );
    }
}

/// REGRESSION TEST (chelis#2534), both witnesses.
#[test]
fn a_variable_source_of_a_binder_target_cast_is_decided_when_it_binds() {
    rejects_with(
        "def f[q: Int](s: string) -> q = (fn (y) -> cast(y, q))(s)",
        "CastNonTensor",
        "numeric or bool scalar",
        "string",
    );
    accepts("def f[q: Int](s: i64) -> q = (fn (y) -> cast(y, q))(s)");
    accepts("def f[q: Int](s: bool) -> q = (fn (y) -> cast(y, q))(s)");

    let unbounded = "module P.Main\nexport (main)\n\
                     def f[a, q: Float](k: a) -> q = (fn (y) -> cast(y, q))(k)\n\
                     def go(s: string) -> f32 = f(s)\n\
                     def main() -> tensor[1, f32] = to_tensor([go(\"x\")])\n";
    rejects_with(unbounded, "CastNonTensor", "numeric or bool scalar", "`a`");
    accepts(
        &unbounded
            .replace("f[a, q: Float]", "f[a: Numeric, q: Float]")
            .replace("go(s: string)", "go(s: f64)")
            .replace("go(\"x\")", "go(1.5f64)"),
    );
}

/// REGRESSION TEST (chelis#2524). The concrete-target rule is the one
/// `cast_result_from_settled_source` applies, and both the direct call and the
/// discharge of a lambda parameter reach it.
#[test]
fn a_string_is_no_cast_source() {
    rejects_with(
        "def f() -> i64 = (fn (y) -> cast(y, i64))(\"s\")",
        "CastNonTensor",
        "numeric or bool scalar",
        "string",
    );
    rejects_with(
        "def f(s: string) -> i64 = cast(s, i64)",
        "CastNonTensor",
        "numeric or bool scalar",
        "string",
    );
    accepts("def f() -> i64 = (fn (y) -> cast(y, i64))(true)");
    accepts("def f(s: i32) -> i64 = cast(s, i64)");
}

/// NEGATIVE PARITY for the gate: the variable source's result is the settled
/// answer, so a tensor reaching a binder-target cast through a lambda keeps its
/// dimensions, as the direct cast does (the `cast` form of chelis#2535).
#[test]
fn a_tensor_reaching_a_binder_target_cast_through_a_lambda_keeps_its_shape() {
    accepts("def f[q: Float](b: tensor[2, f32]) -> tensor[2, q] = (fn (y) -> cast(y, q))(b)");
    accepts("def f[q: Float](b: tensor[2, f32]) -> tensor[2, q] = cast(b, q)");
}

/// REGRESSION TEST (chelis#2584 round 1, P1-1). A `let`-bound lambda whose
/// cast operand is still a variable carries a pending gate, so it stays
/// monomorphic until its first application binds the operand ([04-INF-1]).
/// It was generalized instead, every application bound a fresh copy of the
/// operand, and the gate stayed on the never-bound template: the well-typed
/// forms were rejected with `got ?N`, and the string form named no string.
#[test]
fn a_let_bound_cast_lambda_is_decided_at_its_application() {
    for program in [
        "def f[p: Float](x: f32) -> p = {\n  g = fn (y) -> cast(y, p)\n  g(x)\n}\n",
        "def f[p: Float](xs: List[f32]) -> List[p] = {\n  conv = fn (y) -> cast(y, p)\n  map(conv, xs)\n}\n",
        "def f[p: Float](x: f32) -> p = {\n  g = fn (y) -> cast(y, p)\n  a = g(x)\n  g(2.0f32)\n}\n",
    ] {
        accepts(program);
    }
    rejects_with(
        "def f[p: Float](x: f32) -> p = {\n  g = fn (y) -> cast(y, p)\n  g(\"s\")\n}\n",
        "CastNonTensor",
        "numeric or bool scalar",
        "string",
    );
    // Its first application fixes the operand type for every later use.
    rejects_with(
        "def f[p: Float](x: f32, n: i32) -> p = {\n  g = fn (y) -> cast(y, p)\n  a = g(x)\n  g(n)\n}\n",
        "PrecisionMismatch",
        "f32",
        "i32",
    );
}

/// NEGATIVE PARITY: the same holds for the concrete-target gate, which a
/// generalized `let` lambda also orphaned (chelis#1489's recorded blind spot).
#[test]
fn a_let_bound_concrete_target_cast_lambda_is_decided_at_its_application() {
    accepts("def f(x: f32) -> f64 = {\n  g = fn (y) -> cast(y, f64)\n  a = g(x)\n  g(2.0f32)\n}\n");
    rejects_with(
        "def f(x: f32) -> i64 = {\n  g = fn (y) -> cast(y, i64)\n  g(\"s\")\n}\n",
        "CastNonTensor",
        "numeric or bool scalar",
        "string",
    );
}

/// REGRESSION TEST (chelis#2584 round 2, P3). [04-INF-1] keeps the whole
/// lambda monomorphic while it carries a deferred obligation, not only the
/// checked operand: every later use has the first application's
/// instantiation. An unchecked second parameter used at `i32` and then at
/// `string` was polymorphic beside the checked one.
#[test]
fn a_lambda_carrying_a_pending_cast_is_monomorphic_as_a_whole() {
    rejects_with(
        "def f(a: f32) -> (f64, i32, string) = {\n  g = fn (y, z) -> (cast(y, f64), z)\n  \
         (u, i) = g(a, 1)\n  (v, s) = g(a, \"s\")\n  (add(u, v), i, s)\n}\n",
        "PrecisionMismatch",
        "i32",
        "string",
    );
    rejects_with(
        "def f(a: f32) -> (f64, i32, string) = {\n  mk = fn (k) -> fn (y) -> (cast(y, f64), k)\n  \
         g1 = mk(1i32)\n  g2 = mk(\"s\")\n  (g1(a).0, g1(a).1, g2(a).1)\n}\n",
        "PrecisionMismatch",
        "i32",
        "string",
    );
    // One instantiation throughout checks, and a lambda with no pending
    // obligation still generalizes.
    accepts(
        "def f(a: f32) -> (f64, i32) = {\n  g = fn (y, z) -> (cast(y, f64), z)\n  \
         (u, i) = g(a, 1)\n  (v, j) = g(a, 2)\n  (add(u, v), add(i, j))\n}\n",
    );
    accepts("def f(a: f32) -> (i32, string) = {\n  id = fn (z) -> z\n  (id(1i32), id(\"s\"))\n}\n");
}
