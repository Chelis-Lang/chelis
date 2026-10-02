//! chelis#2993 and chelis#3017: a host-lane `grad` over a scalar def lowers
//! through the reverse-mode DAG that `chelis eval` evaluates (spec/06 section
//! 2.3), at the operand dtype ([04-NUM-8]). The lowered value is one rank-zero
//! tensor helper projected to a scalar; every helper node is at the operand
//! dtype; and the helper computes the same bits as the eval lane's DAG for the
//! same application.
use chelis_deep::ExprCarrier;
use chelis_deep::ast::{Atom, Expr};
use chelis_ir::ConcreteHostType as HostType;
use chelis_ir::eval::eval_tensor_roots_with_strict;
use chelis_ir::host::{
    ConcreteHostExpr as HostExpr, ConcreteHostExprKind as HostExprKind, HostTensorHelper,
    try_lower_compiled_program,
};
use chelis_ir::lower::lower_subexpr_program;
use chelis_types::types::Prim;
use chelis_unord::UnordMap;

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

/// The lowered `out` global: its declared host type and value, and the
/// program's global tensor helpers.
fn lowered_out(src: &str) -> (HostType, HostExpr, Vec<HostTensorHelper>) {
    let compiled = try_lower_compiled_program(&checked_surf(src)).expect("host lowering");
    let host = compiled
        .host
        .expect("a scalar grad program lowers to the host lane");
    let out = host
        .globals
        .into_iter()
        .find(|binding| binding.name.rsplit('.').next() == Some("out"))
        .expect("an `out` global");
    (out.ty, out.value, host.global_tensor_helpers)
}

/// The helper a scalar gradient lowers to: `let g = <helper>() in
/// tensor_to_scalar(g)`, with the helper's rank-zero result and the
/// projection both at `prim`.
fn gradient_helper<'a>(
    src: &str,
    value: &HostExpr,
    helpers: &'a [HostTensorHelper],
    prim: Prim,
) -> &'a HostTensorHelper {
    let HostExprKind::Let { bindings, body, ty } = &value.kind else {
        panic!("{src}: a scalar gradient is a bound reverse-DAG result: {value:?}");
    };
    assert_eq!(*ty, HostType::Scalar(prim), "{src}");
    let [binding] = bindings.as_slice() else {
        panic!("{src}: one gradient binding: {value:?}");
    };
    let HostExprKind::TensorCall { helper, args, .. } = &binding.value.kind else {
        panic!("{src}: the gradient is a tensor helper call: {value:?}");
    };
    assert!(
        args.is_empty(),
        "{src}: a top-level gradient takes no inputs"
    );
    let HostExprKind::Builtin { name, ty, .. } = &body.kind else {
        panic!("{src}: the gradient is projected to a scalar: {value:?}");
    };
    assert_eq!(
        (name.as_str(), ty),
        ("tensor_to_scalar", &HostType::Scalar(prim)),
        "{src}"
    );
    let helper = &helpers[*helper];
    assert!(helper.output.dims.is_empty(), "{src}: a rank-zero gradient");
    assert_eq!(helper.output.precision, prim, "{src}");
    helper
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

/// The eval lane's value for `out`: the subexpression lowered to a DAG and
/// evaluated, as `chelis eval` does for a gradient application.
fn eval_lane_out(src: &str) -> String {
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
    let type_env = checked
        .type_env()
        .iter()
        .map(|(name, ty)| (name.clone(), ty.clone()))
        .collect();
    let dag = lower_subexpr_program(
        &out_expr.expect("source must define out"),
        UnordMap::new(),
        type_env,
        defs,
    );
    let roots = dag.roots().to_vec();
    let values = eval_tensor_roots_with_strict(&dag, &roots, |_| None).expect("eval lane");
    format!("{:?}", values[&roots[0]].storage())
}

fn helper_value(helper: &HostTensorHelper) -> String {
    let roots = helper.dag.roots().to_vec();
    assert_eq!(roots.len(), 1, "one gradient root");
    let values = eval_tensor_roots_with_strict(&helper.dag, &roots, |_| None).expect("helper");
    format!("{:?}", values[&roots[0]].storage())
}

/// chelis#3017's witness bodies, built only from correctly rounded
/// arithmetic, then the #2993 oracle bodies. `{t}` is the dtype.
const BODIES: [&str; 10] = [
    "mul(x, x)",
    "add(mul(x, x), x)",
    "div(x, add(x, 1.0{t}))",
    "mul(x, sqrt(x))",
    "div(sqrt(x), add(x, 1.0{t}))",
    "sqrt(sqrt(x))",
    "sqrt(sqrt(sqrt(x)))",
    "mul(exp(x), x)",
    "div(1.0{t}, x)",
    "log(x)",
];

fn program(dtype: &str, body: &str, input: &str) -> String {
    let body = body.replace("{t}", dtype);
    format!("def f(x: {dtype}) -> {dtype} = {body}\nout = grad(f)({input}{dtype})\n")
}

fn assert_reverse_dag_at(dtype: &str, prim: Prim) {
    for body in BODIES {
        for input in ["0.374", "0.648", "0.1"] {
            let src = program(dtype, body, input);
            let (ty, value, helpers) = lowered_out(&src);
            assert_eq!(ty, HostType::Scalar(prim), "{src}: result dtype");
            let helper = gradient_helper(&src, &value, &helpers, prim);
            for node in helper.dag.nodes() {
                if node.output_type.precision.is_float() {
                    assert_eq!(
                        node.output_type.precision, prim,
                        "{src}: {:?} computes at another width",
                        node.op
                    );
                }
            }
            assert_eq!(
                helper_value(helper),
                eval_lane_out(&src),
                "{src}: the host helper and the eval lane differ"
            );
        }
    }
}

#[test]
fn f32_scalar_grad_is_the_eval_lanes_reverse_dag_at_f32() {
    assert_reverse_dag_at("f32", Prim::F32);
}

#[test]
fn f64_scalar_grad_is_the_eval_lanes_reverse_dag_at_f64() {
    assert_reverse_dag_at("f64", Prim::F64);
}

#[test]
fn f32_multi_parameter_gradient_tuple_is_f32() {
    let src =
        "def f(x: f32, y: f32) -> f32 = mul(x, y)\nout = grad(f, wrt=(x, y))(0.5f32, 2.0f32)\n";
    let (ty, _, helpers) = lowered_out(src);
    assert_eq!(
        ty,
        HostType::Tuple(vec![HostType::Float32, HostType::Float32])
    );
    assert!(!helpers.is_empty(), "{src}: the gradient is a reverse DAG");
    for helper in &helpers {
        assert!(
            helper.dag.nodes().iter().all(|node| {
                !node.output_type.precision.is_float() || node.output_type.precision == Prim::F32
            }),
            "{src}: every helper node is f32"
        );
    }
}

#[test]
fn host_collection_transform_body_keeps_its_rejection() {
    // A pure-scalar body that reaches a host collection transform still
    // declines to the unresolved-transform marker rather than a DAG.
    let src = "def f(x: f32) -> f32 = fold(fn (acc: f32, y: f32) -> add(acc, mul(y, x)), 0.0f32, [1.0f32, 2.0f32])\nout = grad(f)(0.5f32)\n";
    let (_, value, _) = lowered_out(src);
    let HostExprKind::Call { function, .. } = &value.kind else {
        panic!("{src}: expected the unresolved-transform marker, got {value:?}");
    };
    assert!(
        chelis_ir::host::is_host_unresolved_marker(function),
        "{src}: {function}"
    );
}
