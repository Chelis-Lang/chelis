//! chelis#3149: [04-INF-9] admits a builtin as a function value exactly when
//! its whole operation contract travels with the value. Every other builtin
//! is applicable only by name, and naming it in any value position is a
//! `TypeMismatch` that names the builtin.
//!
//! Before the fix, `op = sum` (or `max_reduce`, `cumsum`, `trace`, `lt`, ...)
//! instantiated a scheme whose result was an unconstrained variable, so a call
//! through the alias checked against any declared result: a wrong extent, a
//! wrong dtype, `f32` for a comparison. The specialized result rules run only
//! at a direct application.
//!
//! The admission predicate is read from the builtin registry
//! (`builtin_value_contract_carried`), never from a list in this file, so a
//! builtin added later with a rule its scheme does not carry is rejected in
//! value position without anyone remembering to add it here.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::errors::{CheckError, CheckErrorKind};
use chelis_types::types::Type;
use chelis_types::{
    BUILTINS, InferenceDisposition, builtin_env, builtin_value_contract_carried, check_ir_program,
};

const MARKER: &str = "is applicable only by name";

fn surf_to_deep(source: &str) -> Vec<chelis_deep::Expr> {
    let decls = parse_str(source).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls).expect("Surf fixture must desugar"),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
}

fn errors_of(exprs: &[chelis_deep::Expr]) -> Vec<CheckError> {
    match check_ir_program(exprs) {
        Ok(_) => Vec::new(),
        Err(report) => report.errors,
    }
}

fn by_name_rejections(errors: &[CheckError], builtin: &str) -> Vec<CheckError> {
    let named = format!("builtin `{builtin}` {MARKER}");
    errors
        .iter()
        .filter(|error| error.message.contains(&named))
        .cloned()
        .collect()
}

fn assert_rejected_by_name(source: &str, builtin: &str) {
    let errors = errors_of(&surf_to_deep(source));
    let rejected = by_name_rejections(&errors, builtin);
    assert!(
        !rejected.is_empty(),
        "`{builtin}` in value position must be refused as applicable only by name\n\
         {source}\n{errors:#?}"
    );
    for error in &rejected {
        assert!(
            matches!(error.kind, CheckErrorKind::TypeMismatch),
            "the by-name refusal is a TypeMismatch ([04-INF-9]): {error:#?}"
        );
        assert!(
            error.message.contains("[04-INF-9]")
                && error.suggestions.iter().any(|hint| hint.contains("lambda")),
            "the refusal cites [04-INF-9] and suggests a lambda wrapper: {error:#?}"
        );
    }
}

fn assert_checks(source: &str) {
    let errors = errors_of(&surf_to_deep(source));
    assert!(errors.is_empty(), "must check clean\n{source}\n{errors:#?}");
}

/// The registry guard. For every builtin, value-position admission by the
/// checker equals "the whole rule is carried on the value". The fixture is
/// Deep, so builtin names that are Surf keywords (`and`, `not`) are covered.
#[test]
fn value_admission_equals_carried_contract_for_every_builtin() {
    let (env, _) = builtin_env();
    let mut mismatches = Vec::new();
    for decl in BUILTINS {
        let scheme = env
            .lookup(decl.name)
            .unwrap_or_else(|| panic!("registry builtin `{}` has no scheme", decl.name));
        let carried = builtin_value_contract_carried(decl, scheme);
        let source = format!("(def {{}} op (var {{}} {}))", decl.name);
        let exprs = chelis_deep::parser::parse_str(&source).expect("Deep fixture must parse");
        let refused = !by_name_rejections(&errors_of(&exprs), decl.name).is_empty();
        if refused == carried {
            mismatches.push(format!(
                "{}: carried={carried} but refused={refused}",
                decl.name
            ));
        }
    }
    assert!(mismatches.is_empty(), "{mismatches:#?}");
}

/// The registry's `GenericAccepted` declaration claims the scheme states the
/// whole contract. A scheme whose result mentions a variable no parameter
/// fixes, with no carried relation, cannot state it, so that claim would
/// admit an unconstrained result as a value.
#[test]
fn generic_accepted_schemes_determine_their_results() {
    fn vars(ty: &Type) -> Vec<u32> {
        chelis_types::env::free_tvars(ty)
            .into_iter()
            .map(|v| v.0)
            .collect()
    }
    let (env, _) = builtin_env();
    for decl in BUILTINS {
        if !matches!(decl.inference, InferenceDisposition::GenericAccepted { .. }) {
            continue;
        }
        let scheme = env
            .lookup(decl.name)
            .expect("registry builtin has a scheme");
        let Type::Fn(params, result) = &scheme.body else {
            continue;
        };
        let fixed: Vec<u32> = params.iter().flat_map(vars).collect();
        let free_result = vars(result).into_iter().any(|v| !fixed.contains(&v));
        assert!(
            !free_result || !scheme.constraints.is_empty() || scheme.result_origin.is_some(),
            "`{}` is declared GenericAccepted but its scheme leaves the result free",
            decl.name
        );
    }
}

// The chelis#3149 witnesses (fix-trio probes Al1..Al7), plus a comparison
// aliased at a float result. Each checked clean before the fix.

#[test]
fn al1_sum_alias_at_operand_precision_is_refused() {
    assert_rejected_by_name(
        "module Al1\nop = sum\ndef g[p: Numeric](x: tensor[3, p]) -> tensor[p] = op(x, 0i32)\n",
        "sum",
    );
}

#[test]
fn al2_sum_alias_at_wrong_extent_and_dtype_is_refused() {
    assert_rejected_by_name(
        "module Al2\nop = sum\ndef g[p: Numeric](x: tensor[3, p]) -> tensor[9, f64] = op(x, 0i32)\n",
        "sum",
    );
}

#[test]
fn al3_max_reduce_alias_is_refused() {
    assert_rejected_by_name(
        "module Al3\nop = max_reduce\ndef g[p: Numeric](x: tensor[3, p]) -> tensor[9, f64] = op(x, 0i32)\n",
        "max_reduce",
    );
}

#[test]
fn al4_cumsum_alias_over_int_is_refused() {
    assert_rejected_by_name(
        "module Al4\nop = cumsum\ndef g[p: Int](x: tensor[3, p]) -> tensor[3, p] = op(x, 0i32)\n",
        "cumsum",
    );
}

#[test]
fn al5_sum_alias_declared_i8_is_refused() {
    assert_rejected_by_name(
        "module Al5\nop = sum\ndef g(x: tensor[3, i8]) -> tensor[i8] = op(x, 0i32)\n",
        "sum",
    );
}

#[test]
fn al6_cumsum_alias_declared_i8_is_refused() {
    assert_rejected_by_name(
        "module Al6\nop = cumsum\ndef g(x: tensor[3, i8]) -> tensor[3, i8] = op(x, 0i32)\n",
        "cumsum",
    );
}

#[test]
fn al7_trace_alias_declared_i8_is_refused() {
    assert_rejected_by_name(
        "module Al7\nop = trace\ndef g(x: tensor[2, 2, i8]) -> tensor[i8] = op(x, 0, 1)\n",
        "trace",
    );
}

#[test]
fn comparison_alias_declared_float_is_refused() {
    assert_rejected_by_name(
        "module Lt\nop = lt\ndef g(x: tensor[3, f32]) -> tensor[3, f32] = op(x, x)\n",
        "lt",
    );
}

// Every value position, not only a top-level alias.

#[test]
fn block_binding_is_refused() {
    assert_rejected_by_name(
        "def g(x: tensor[3, f32]) -> tensor[3, f32] = {\n  op = exp\n  op(x)\n}\n",
        "exp",
    );
}

#[test]
fn user_function_argument_is_refused() {
    assert_rejected_by_name(
        "def ap(f: tensor[3, f32] -> tensor[3, f32], x: tensor[3, f32]) -> tensor[3, f32] = f(x)\n\
         def g(x: tensor[3, f32]) -> tensor[3, f32] = ap(exp, x)\n",
        "exp",
    );
}

#[test]
fn map_callback_is_refused() {
    assert_rejected_by_name(
        "def g(xs: List[i64]) -> List[string] = map(to_string, xs)\n",
        "to_string",
    );
}

#[test]
fn fold_callback_is_refused() {
    assert_rejected_by_name("def g(xs: List[i64]) -> i64 = fold(add, 0i64, xs)\n", "add");
}

#[test]
fn vmap_callback_is_refused() {
    assert_rejected_by_name(
        "def g(x: tensor[2, 3, f32]) -> tensor[2, 3, f32] = vmap(exp)(x)\n",
        "exp",
    );
}

#[test]
fn grad_callback_is_refused() {
    assert_rejected_by_name("def g(x: f32) -> f32 = grad(exp)(x)\n", "exp");
}

#[test]
fn list_element_is_refused() {
    assert_rejected_by_name(
        "def g(x: tensor[3, f32]) -> tensor[3, f32] = {\n  ops = [exp, log]\n  x\n}\n",
        "exp",
    );
}

#[test]
fn tuple_element_is_refused() {
    assert_rejected_by_name(
        "def g(x: tensor[3, f32]) -> tensor[3, f32] = {\n  ops = (exp, x)\n  x\n}\n",
        "exp",
    );
}

#[test]
fn returned_builtin_is_refused() {
    assert_rejected_by_name(
        "def pick() -> tensor[3, f32] -> tensor[3, f32] = exp\n",
        "exp",
    );
}

// Negative parity: what stays admitted.

#[test]
fn direct_calls_still_check() {
    assert_checks(
        "def g(x: tensor[3, f32]) -> tensor[f32] = sum(x, 0i32)\n\
         def h(x: tensor[3, f32]) -> tensor[3, bool] = lt(x, x)\n\
         def k(x: tensor[3, f32]) -> tensor[3, f32] = x |> exp\n",
    );
}

#[test]
fn shape_routed_direct_calls_still_check() {
    // These routes infer their callee outside the generic application path.
    assert_checks(
        "def g(x: tensor[2, 3, f32]) -> tensor[3, 2, f32] = permute(x, 1i32, 0i32)\n\
         def h(x: tensor[2, 3, f32]) -> tensor[6, f32] = reshape(x, [6i64])\n\
         def k(x: tensor[3, f32]) -> tensor[2, 3, f32] = insert(x, 0i32, 2i64)\n",
    );
}

#[test]
fn explicitly_typed_lambda_wrapper_is_admitted() {
    assert_checks(
        "def g(x: tensor[2, 3, f32]) -> tensor[2, f32] = vmap(fn (r: tensor[3, f32]) -> sum(r, 0i32))(x)\n",
    );
}

#[test]
fn lexical_binding_with_a_builtin_name_is_an_ordinary_value() {
    assert_checks(
        "def g(sum: tensor[3, f32] -> tensor[3, f32], x: tensor[3, f32]) -> tensor[3, f32] = {\n  \
         op = sum\n  op(x)\n}\n",
    );
}

#[test]
fn carried_contract_builtins_stay_values() {
    assert_checks(
        "def main() = {\n  seed = key_from_seed\n  halves = split_key\n  folded = fold_in\n  \
         children = split_keys\n  seeds = to_tensor([1i64, 2i64])\n  keys = seed(seeds)\n  \
         (left, right) = halves(keys)\n  \
         (seed(7i64), seed(seeds), folded(left, seeds), children(right, 2i64))\n}\n",
    );
    assert_checks("def g(xs: List[List[i64]]) -> List[i64] = map(len, xs)\n");
}
