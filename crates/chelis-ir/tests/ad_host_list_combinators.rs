use chelis_deep::ast::{Atom, Expr, ExprCarrier};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_with_strict};
use chelis_ir::lower::lower_subexpr_program;
use chelis_unord::UnordMap;

fn checked_deep(src: &str) -> chelis_types::CheckedProgram {
    let exprs = chelis_deep::parser::parse_and_stamp(src).expect("Deep fixture");
    let checked = chelis_types::check_ir_program(&exprs, chelis_types::ProgramOrigin::Unlinked)
        .unwrap_or_else(|report| {
            panic!(
                "type check failed: {:?}",
                report
                    .errors
                    .iter()
                    .map(|error| error.message.clone())
                    .collect::<Vec<_>>()
            )
        });
    let checked = chelis_effects::check_program(&checked).expect("effects");
    chelis_types::check_linearity(&checked).expect("linearity")
}

fn def_name_and_body(expr: &Expr) -> Option<(String, Expr)> {
    let ExprCarrier::DecodedNode(chelis_deep::DeepTag::Def, _, kids) = expr.carrier() else {
        return None;
    };
    let name = match kids.first()? {
        Expr::Atom(Atom::Name(name), _) => name.clone(),
        _ => return None,
    };
    Some((name, kids.get(1)?.clone()))
}

fn eval_out(src: &str) -> TensorValue {
    let checked = checked_deep(src);
    let mut defs = UnordMap::new();
    let mut out_expr = None;
    for expr in checked.exprs() {
        if let Some((name, body)) = def_name_and_body(expr) {
            if name == "out" {
                out_expr = Some(body.clone());
            }
            defs.insert(name, body);
        }
    }
    let out_expr = out_expr.expect("source must define out");
    let type_env = checked
        .type_env()
        .iter()
        .map(|(name, ty)| (name.clone(), ty.clone()))
        .collect();
    let dag = lower_subexpr_program(&out_expr, UnordMap::new(), type_env, defs);
    let roots = dag.roots().to_vec();
    assert_eq!(roots.len(), 1, "out subexpression should have one root");
    let values = eval_tensor_roots_with_strict(&dag, &roots, |_| None).expect("subexpression eval");
    values[&roots[0]].clone()
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch actual={actual:?} expected={expected:?}"
    );
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tol,
            "{label}: element {index} actual={actual} expected={expected} diff={diff} tol={tol}",
        );
    }
}

fn cdf_identity_black_scholes_delta(
    spot: f64,
    strike: f64,
    rate: f64,
    sigma: f64,
    time: f64,
) -> f64 {
    let root_time = time.sqrt();
    let vol_root_time = sigma * root_time;
    let carry = (rate + 0.5 * sigma * sigma) * time;
    let discount = (-rate * time).exp();
    let d1 = ((spot / strike).ln() + carry) / vol_root_time;
    d1 + (1.0 / vol_root_time) - (strike * discount) / (spot * vol_root_time)
}

fn cdf_identity_black_scholes_gamma(
    spot: f64,
    strike: f64,
    rate: f64,
    sigma: f64,
    time: f64,
) -> f64 {
    let root_time = time.sqrt();
    let vol_root_time = sigma * root_time;
    let discount = (-rate * time).exp();
    (1.0 / (spot * vol_root_time)) + (strike * discount) / (spot * spot * vol_root_time)
}

#[test]
fn subexpr_grad_through_to_list_to_tensor_identity() {
    let out = eval_out(include_str!(
        "fixtures/core/tests/ad_host_list_combinators/subexpr_grad_through_to_list_to_tensor_identity.deep"
    ));
    assert_eq!(out.shape, vec![3]);
    assert_close(&out.to_f64_lossy_vec(), &[1.0, 1.0, 1.0], 1e-6, "boundary");
}

#[test]
fn subexpr_grad_through_let_bound_to_list_map() {
    let out = eval_out(include_str!(
        "fixtures/core/tests/ad_host_list_combinators/subexpr_grad_through_let_bound_to_list_map.deep"
    ));
    assert_eq!(out.shape, vec![3]);
    assert_close(&out.to_f64_lossy_vec(), &[2.0, 4.0, 6.0], 1e-5, "map");
}

#[test]
fn subexpr_nested_grad_through_map_is_second_derivative() {
    let out = eval_out(include_str!(
        "fixtures/core/tests/ad_host_list_combinators/subexpr_nested_grad_through_map_is_second_derivative.deep"
    ));
    assert_eq!(out.shape, vec![1]);
    assert_close(&out.to_f64_lossy_vec(), &[18.0], 1e-5, "nested grad");
}

#[test]
fn subexpr_grad_black_scholes_f64_args_differentiates_only_wrt_spots() {
    let out = eval_out(include_str!(
        "fixtures/core/tests/ad_host_list_combinators/subexpr_grad_black_scholes_f64_args_differentiates_only_wrt_spots.deep"
    ));
    assert_eq!(out.shape, vec![3]);
    let expected = [90.0, 100.0, 110.0]
        .into_iter()
        .map(|spot| cdf_identity_black_scholes_delta(spot, 100.0, 0.05, 0.30, 1.25))
        .collect::<Vec<_>>();
    assert_close(
        &out.to_f64_lossy_vec(),
        &expected,
        1e-10,
        "Black-Scholes f64 delta",
    );
}

#[test]
fn subexpr_nested_grad_named_black_scholes_f64_args_computes_gamma() {
    let out = eval_out(include_str!(
        "fixtures/core/tests/ad_host_list_combinators/subexpr_nested_grad_named_black_scholes_f64_args_computes_gamma.deep"
    ));
    assert_eq!(out.shape, vec![1]);
    let expected = [cdf_identity_black_scholes_gamma(
        100.0, 100.0, 0.05, 0.30, 1.25,
    )];
    assert_close(
        &out.to_f64_lossy_vec(),
        &expected,
        1e-10,
        "Black-Scholes f64 gamma",
    );
}

#[test]
fn subexpr_vmap_grad_black_scholes_f64_args_uses_host_list_ad_path() {
    let out = eval_out(include_str!(
        "fixtures/core/tests/ad_host_list_combinators/subexpr_vmap_grad_black_scholes_f64_args_uses_host_list_ad_path.deep"
    ));
    assert_eq!(out.shape, vec![3, 1]);
    let expected = [90.0, 100.0, 110.0]
        .into_iter()
        .map(|spot| cdf_identity_black_scholes_delta(spot, 100.0, 0.05, 0.30, 1.25))
        .collect::<Vec<_>>();
    assert_close(
        &out.to_f64_lossy_vec(),
        &expected,
        1e-10,
        "vmap Black-Scholes f64 delta",
    );
}

#[test]
fn subexpr_vmap_grad_through_map_matches_unbatched_gradients() {
    let out = eval_out(include_str!(
        "fixtures/core/tests/ad_host_list_combinators/subexpr_vmap_grad_through_map_matches_unbatched_gradients.deep"
    ));
    assert_eq!(out.shape, vec![2, 3]);
    assert_close(
        &out.to_f64_lossy_vec(),
        &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        1e-5,
        "vmap grad map",
    );
}

#[test]
fn subexpr_grad_through_filter_uses_constant_primal_mask() {
    let out = eval_out(include_str!(
        "fixtures/core/tests/ad_host_list_combinators/subexpr_grad_through_filter_uses_constant_primal_mask.deep"
    ));
    assert_eq!(out.shape, vec![4]);
    assert_close(
        &out.to_f64_lossy_vec(),
        &[0.0, 6.0, 1.0, 0.0],
        1e-5,
        "filter",
    );
}

#[test]
fn subexpr_grad_through_fold_recurrence() {
    let out = eval_out(include_str!(
        "fixtures/core/tests/ad_host_list_combinators/subexpr_grad_through_fold_recurrence.deep"
    ));
    assert_eq!(out.shape, vec![3]);
    assert_close(&out.to_f64_lossy_vec(), &[0.5, 2.0, 6.0], 1e-5, "fold");
}
