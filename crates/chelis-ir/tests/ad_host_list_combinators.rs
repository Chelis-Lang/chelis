use chelis_deep::ast::{Atom, Expr, List};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_with_strict};
use chelis_ir::lower::lower_subexpr_program;
use chelis_unord::UnordMap;

fn get_tag(list: &List) -> Option<chelis_deep::DeepTag> {
    list.tag()
}

fn children(list: &List) -> &[Expr] {
    if list.elements.len() > 2 {
        &list.elements[2..]
    } else {
        &[]
    }
}

fn checked_surf(src: &str) -> chelis_types::CheckedProgram {
    let decls = chelis_surf::parser::parse_str(src).expect("surf parse");
    let exprs = chelis_macros::expand_program(
        &chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar"),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expansion")
    .into_exprs();
    let checked = chelis_types::check_ir_program(&exprs).unwrap_or_else(|report| {
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
    let Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some(chelis_deep::DeepTag::Def) {
        return None;
    }
    let kids = children(list);
    let name = match kids.first()? {
        Expr::Atom(Atom::Name(name), _) => name.clone(),
        _ => return None,
    };
    Some((name, kids.get(1)?.clone()))
}

fn eval_out(src: &str) -> TensorValue {
    let checked = checked_surf(src);
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

const BLACK_SCHOLES_F64_ARGS_BODY: &str = "\
def bs_price(spot: f64, strike: f64, rate: f64, sigma: f64, time: f64) -> f64 = {
  root_time = sqrt(time)
  vol_root_time = mul(sigma, root_time)
  carry = mul(add(rate, mul(cast(0.5, f64), mul(sigma, sigma))), time)
  discount = exp(neg(mul(rate, time)))
  d1 = div(add(log(div(spot, strike)), carry), vol_root_time)
  d2 = sub(d1, vol_root_time)
  sub(mul(spot, d1), mul(mul(strike, discount), d2))
}

def bs_total(spots: tensor[3, f64], strike: f64, rate: f64, sigma: f64, time: f64) -> f64 = {
  xs = to_list(copy(spots))
  prices = map(fn (spot: f64) -> bs_price(spot, strike, rate, sigma, time), xs)
  tensor_to_scalar(sum(to_tensor(prices), cast(0, i32)))
}

def bs_total_one(spots: tensor[1, f64], strike: f64, rate: f64, sigma: f64, time: f64) -> f64 = {
  xs = to_list(copy(spots))
  prices = map(fn (spot: f64) -> bs_price(spot, strike, rate, sigma, time), xs)
  tensor_to_scalar(sum(to_tensor(prices), cast(0, i32)))
}

def bs_total_one_delta(spots: tensor[1, f64], strike: f64, rate: f64, sigma: f64, time: f64) -> f64 =
  tensor_to_scalar(sum(grad(bs_total_one, wrt=spots)(spots, strike, rate, sigma, time), cast(0, i32)))
";

#[test]
fn subexpr_grad_through_to_list_to_tensor_identity() {
    let out = eval_out(
        "\
def loss(x: tensor[3, f32]) -> f32 = {
  y = to_tensor(to_list(copy(x)))
  tensor_to_scalar(sum(y, cast(0, i32)))
}
out = grad(loss)(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)]))
",
    );
    assert_eq!(out.shape, vec![3]);
    assert_close(&out.to_f64_lossy_vec(), &[1.0, 1.0, 1.0], 1e-6, "boundary");
}

#[test]
fn subexpr_grad_through_let_bound_to_list_map() {
    let out = eval_out(
        "\
def loss(x: tensor[3, f32]) -> f32 = {
  xs = to_list(copy(x))
  ys = map(fn (v: f32) -> mul(v, v), xs)
  tensor_to_scalar(sum(to_tensor(ys), cast(0, i32)))
}
out = grad(loss)(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)]))
",
    );
    assert_eq!(out.shape, vec![3]);
    assert_close(&out.to_f64_lossy_vec(), &[2.0, 4.0, 6.0], 1e-5, "map");
}

#[test]
fn subexpr_nested_grad_through_map_is_second_derivative() {
    let out = eval_out(
        "\
def loss(x: tensor[1, f32]) -> f32 = {
  xs = to_list(copy(x))
  ys = map(fn (v: f32) -> mul(mul(v, v), v), xs)
  tensor_to_scalar(sum(to_tensor(ys), cast(0, i32)))
}
def first_derivative_sum(x: tensor[1, f32]) -> f32 =
  tensor_to_scalar(sum(grad(loss)(x), cast(0, i32)))
out = grad(first_derivative_sum)(to_tensor([cast(3.0, f32)]))
",
    );
    assert_eq!(out.shape, vec![1]);
    assert_close(&out.to_f64_lossy_vec(), &[18.0], 1e-5, "nested grad");
}

#[test]
fn subexpr_grad_black_scholes_f64_args_differentiates_only_wrt_spots() {
    let out = eval_out(&format!(
        "{BLACK_SCHOLES_F64_ARGS_BODY}
out = grad(bs_total, wrt=spots)(
  to_tensor([cast(90.0, f64), cast(100.0, f64), cast(110.0, f64)]),
  cast(100.0, f64),
  cast(0.05, f64),
  cast(0.3, f64),
  cast(1.25, f64)
)
"
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
    let out = eval_out(&format!(
        "{BLACK_SCHOLES_F64_ARGS_BODY}
out = grad(bs_total_one_delta, wrt=spots)(
  to_tensor([cast(100.0, f64)]),
  cast(100.0, f64),
  cast(0.05, f64),
  cast(0.3, f64),
  cast(1.25, f64)
)
"
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
    let out = eval_out(&format!(
        "{BLACK_SCHOLES_F64_ARGS_BODY}
def bs_total_one_batched(spots: tensor[1, f64], strike: f64, rate: f64, sigma: f64, time: f64) -> f64 =
  bs_total_one(spots, strike, rate, sigma, time)
out = vmap(grad(bs_total_one_batched, wrt=spots))(
  to_tensor([[cast(90.0, f64)], [cast(100.0, f64)], [cast(110.0, f64)]]),
  cast(100.0, f64),
  cast(0.05, f64),
  cast(0.3, f64),
  cast(1.25, f64)
)
"
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
    let out = eval_out(
        "\
def loss(row: tensor[3, f32]) -> f32 = {
  xs = to_list(copy(row))
  ys = map(fn (v: f32) -> mul(v, v), xs)
  tensor_to_scalar(sum(to_tensor(ys), cast(0, i32)))
}
out = vmap(grad(loss))(to_tensor([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)], [cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)]]))
",
    );
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
    let out = eval_out(
        "\
def loss(x: tensor[4, f32]) -> f32 = {
  selected = to_tensor(filter(fn (v: f32) -> gt(v, cast(0.0, f32)), to_list(copy(x))))
  squares = map(fn (v: f32) -> mul(v, v), to_list(selected))
  tensor_to_scalar(sum(to_tensor(squares), cast(0, i32)))
}
out = grad(loss)(to_tensor([cast(-2.0, f32), cast(3.0, f32), cast(0.5, f32), cast(-1.0, f32)]))
",
    );
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
    let out = eval_out(
        "\
def loss(x: tensor[3, f32]) -> f32 =
  fold(fn (acc: f32, v: f32) -> add(mul(acc, cast(0.5, f32)), mul(v, v)), cast(1.0, f32), to_list(copy(x)))
out = grad(loss)(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)]))
",
    );
    assert_eq!(out.shape, vec![3]);
    assert_close(&out.to_f64_lossy_vec(), &[0.5, 2.0, 6.0], 1e-5, "fold");
}
