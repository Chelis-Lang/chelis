//! chelis#2993: the host-lane forward-mode scalar AD pass builds its dual
//! trees at the operand dtype ([04-NUM-8], [05-OP-64]). A top-level f32
//! `grad` application lowers to an f32 result whose every intermediate is
//! f32 and whose every constant is finalized to f32; nothing defaults to
//! double.
use chelis_ir::ConcreteHostType as HostType;
use chelis_ir::host::{
    ConcreteHostExpr as HostExpr, ConcreteHostExprKind as HostExprKind, try_lower_compiled_program,
};
use chelis_types::types::Prim;

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

/// The lowered `out` global: its declared host type and value.
fn lowered_out(src: &str) -> (HostType, HostExpr) {
    let compiled = try_lower_compiled_program(&checked_surf(src)).expect("host lowering");
    let host = compiled.host.expect("a scalar grad program lowers to the host lane");
    let out = host
        .globals
        .into_iter()
        .find(|binding| binding.name.rsplit('.').next() == Some("out"))
        .expect("an `out` global");
    (out.ty, out.value)
}

/// Every scalar dtype the tree computes at, and every float literal that is
/// not finalized by a `cast` (an untyped `f64` constant).
#[derive(Default)]
struct Widths {
    computed: Vec<(String, HostType)>,
    bare_float_literals: Vec<f64>,
}

fn collect(expr: &HostExpr, under_cast: bool, out: &mut Widths) {
    match &expr.kind {
        HostExprKind::Float(value) => {
            if !under_cast {
                out.bare_float_literals.push(*value);
            }
        }
        HostExprKind::Int(_) | HostExprKind::Var(..) => {}
        HostExprKind::Builtin { name, args, ty } => {
            out.computed.push((name.clone(), ty.clone()));
            for arg in args {
                collect(arg, name == "cast", out);
            }
        }
        HostExprKind::Tuple(items, _) => {
            for item in items {
                collect(item, false, out);
            }
        }
        HostExprKind::FormalIngress { value, .. } => collect(value, under_cast, out),
        HostExprKind::ResultClaimScope { body, .. } => collect(body, under_cast, out),
        other => panic!("unexpected node in a scalar dual tree: {other:?}"),
    }
}

fn widths(expr: &HostExpr) -> Widths {
    let mut out = Widths::default();
    collect(expr, false, &mut out);
    out
}

/// The issue's oracle bodies, each `f: T -> T`. `{t}` is the dtype.
const BODIES: [&str; 4] = [
    "mul(exp(x), x)",
    "div(1.0{t}, x)",
    "log(x)",
    "sqrt(x)",
];

fn program(dtype: &str, body: &str) -> String {
    let body = body.replace("{t}", dtype);
    format!("def f(x: {dtype}) -> {dtype} = {body}\nout = grad(f)(0.7{dtype})\n")
}

fn assert_at_width(dtype: &str, prim: Prim) {
    for body in BODIES {
        let src = program(dtype, body);
        let (ty, value) = lowered_out(&src);
        assert_eq!(ty, HostType::Scalar(prim), "{src}: result dtype");
        let widths = widths(&value);
        assert!(
            !widths.computed.is_empty(),
            "{src}: the derivative is computed, not a folded constant"
        );
        for (name, ty) in &widths.computed {
            assert_eq!(
                *ty,
                HostType::Scalar(prim),
                "{src}: `{name}` computes at {ty:?}"
            );
        }
        if prim != Prim::F64 {
            assert!(
                widths.bare_float_literals.is_empty(),
                "{src}: constants {:?} stay at f64",
                widths.bare_float_literals
            );
        }
    }
}

#[test]
fn f32_scalar_grad_computes_every_op_and_constant_at_f32() {
    assert_at_width("f32", Prim::F32);
}

#[test]
fn f64_scalar_grad_control_stays_at_f64() {
    assert_at_width("f64", Prim::F64);
}

#[test]
fn f32_constants_inside_the_derivative_rule_are_f32() {
    // `tanh` and `sqrt` introduce rule constants (1 and 2).
    for body in ["tanh(x)", "sqrt(x)"] {
        let src = program("f32", body);
        let (ty, value) = lowered_out(&src);
        assert_eq!(ty, HostType::Float32, "{src}");
        let widths = widths(&value);
        assert!(widths.bare_float_literals.is_empty(), "{src}");
        assert!(
            widths.computed.iter().all(|(_, ty)| *ty == HostType::Float32),
            "{src}: {:?}",
            widths.computed
        );
    }
}

#[test]
fn f32_multi_parameter_gradient_tuple_is_f32() {
    let src = "def f(x: f32, y: f32) -> f32 = mul(x, y)\nout = grad(f, wrt=(x, y))(0.5f32, 2.0f32)\n";
    let (ty, value) = lowered_out(src);
    assert_eq!(ty, HostType::Tuple(vec![HostType::Float32, HostType::Float32]));
    let HostExprKind::Tuple(_, tuple_ty) = &value.kind else {
        panic!("a two-parameter gradient is a tuple: {value:?}");
    };
    assert_eq!(*tuple_ty, ty);
    let widths = widths(&value);
    assert!(widths.computed.iter().all(|(_, ty)| *ty == HostType::Float32));
    assert!(
        widths.bare_float_literals.is_empty(),
        "seeds stay at f64: {:?}",
        widths.bare_float_literals
    );
}

#[test]
fn mixed_width_body_is_not_lowered_at_an_assumed_width() {
    // A body that changes width through a checked `cast` node is outside
    // this pass: it falls through to the unresolved-transform marker (the
    // existing rejection path) instead of being computed at one width.
    let src = "def f(x: f32) -> f64 = mul(cast(mul(x, x), f64), 2.0f64)\nout = grad(f)(0.5f32)\n";
    let (_, value) = lowered_out(src);
    let HostExprKind::Call { function, .. } = &value.kind else {
        panic!("{src}: expected the unresolved-transform marker, got {value:?}");
    };
    assert!(
        chelis_ir::host::is_host_unresolved_marker(function),
        "{src}: {function}"
    );
}
