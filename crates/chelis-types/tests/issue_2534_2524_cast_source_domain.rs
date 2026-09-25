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

/// A rejection carrying a diagnostic that contains every fragment.
fn rejects_with(source: &str, fragments: &[&str]) {
    let diagnostics = agreed_diagnostics(source);
    assert!(
        diagnostics
            .iter()
            .any(|message| fragments.iter().all(|fragment| message.contains(fragment))),
        "expected a diagnostic containing all of {fragments:?} for:\n{source}\ngot {diagnostics:#?}"
    );
}

/// REGRESSION TEST (chelis#2534), both witnesses.
#[test]
fn a_variable_source_of_a_binder_target_cast_is_decided_when_it_binds() {
    rejects_with(
        "def f[q: Int](s: string) -> q = (fn (y) -> cast(y, q))(s)",
        &[
            "[CastNonTensor] cast to a quantified scalar dtype requires a numeric or bool scalar, got string",
        ],
    );
    accepts("def f[q: Int](s: i64) -> q = (fn (y) -> cast(y, q))(s)");
    accepts("def f[q: Int](s: bool) -> q = (fn (y) -> cast(y, q))(s)");

    let unbounded = "module P.Main\nexport (main)\n\
                     def f[a, q: Float](k: a) -> q = (fn (y) -> cast(y, q))(k)\n\
                     def go(s: string) -> f32 = f(s)\n\
                     def main() -> tensor[1, f32] = to_tensor([go(\"x\")])\n";
    rejects_with(
        unbounded,
        &["cast to a quantified scalar dtype requires a numeric or bool scalar, got `a`"],
    );
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
        &["[CastNonTensor] cast requires a numeric or bool source, got string"],
    );
    rejects_with(
        "def f(s: string) -> i64 = cast(s, i64)",
        &["[CastNonTensor] cast requires a numeric or bool source, got string"],
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
