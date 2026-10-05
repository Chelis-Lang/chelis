use chelis_ir::dag::{Dag, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_with_strict};
use chelis_ir::tier2::{
    lower_gelu, lower_gelu_tanh, lower_sigmoid, lower_silu, lower_standard_normal_cdf,
};
use chelis_types::types::Prim;
use chelis_types::{FloatUnOp, ScalarValue, float_unop, scalar_from_f64, tensor_from_scalars};

type ActivationLowerer = fn(
    chelis_ir::dag::Owner,
    &mut Dag,
    chelis_ir::dag::NodeId,
    &TensorType,
    Option<&str>,
) -> chelis_ir::dag::NodeId;

// `tanh` is the [05-OP-46] Tier 1 primitive, not a Tier 2 lowering.
fn lower_tanh(
    owner: chelis_ir::dag::Owner,
    dag: &mut Dag,
    x: chelis_ir::dag::NodeId,
    ty: &TensorType,
    _parent_span: Option<&str>,
) -> chelis_ir::dag::NodeId {
    dag.add_node(owner, RiscOp::Tanh, vec![x], ty.clone(), None)
}

// `erf` and `erfc` are [05-OP-46] Tier 1 primitives too.
fn lower_erf(
    owner: chelis_ir::dag::Owner,
    dag: &mut Dag,
    x: chelis_ir::dag::NodeId,
    ty: &TensorType,
    _parent_span: Option<&str>,
) -> chelis_ir::dag::NodeId {
    dag.add_node(owner, RiscOp::Erf, vec![x], ty.clone(), None)
}

fn lower_erfc(
    owner: chelis_ir::dag::Owner,
    dag: &mut Dag,
    x: chelis_ir::dag::NodeId,
    ty: &TensorType,
    _parent_span: Option<&str>,
) -> chelis_ir::dag::NodeId {
    dag.add_node(owner, RiscOp::Erfc, vec![x], ty.clone(), None)
}

fn evaluate_tier2(prim: Prim, input: ScalarValue, lower: ActivationLowerer) -> ScalarValue {
    let ty = TensorType {
        dims: vec![],
        precision: prim,
    };
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load {
            name: "x".to_string().into(),
        },
        vec![],
        ty.clone(),
        None,
    );
    let root = lower(decl.into(), &mut dag, x, &ty, None);
    let input = TensorValue::from_storage(vec![], tensor_from_scalars(prim, &[input]));
    let values =
        eval_tensor_roots_with_strict(&dag, &[root], |name| (name == "x").then(|| input.clone()))
            .unwrap();
    values[&root].storage().scalar_at(0)
}

#[test]
fn reduced_float_scalar_activations_are_rank_zero_tier2_instances() {
    let cases = [
        (
            Prim::F16,
            FloatUnOp::Sigmoid,
            0.0007328987121582031,
            lower_sigmoid as ActivationLowerer,
        ),
        (
            Prim::F16,
            FloatUnOp::Tanh,
            5.960464477539063e-8,
            lower_tanh as ActivationLowerer,
        ),
        (
            Prim::F16,
            FloatUnOp::Silu,
            2.980_232_238_769_531e-7,
            lower_silu as ActivationLowerer,
        ),
        (
            Prim::F16,
            FloatUnOp::Gelu,
            2.980_232_238_769_531e-7,
            lower_gelu as ActivationLowerer,
        ),
        (
            Prim::Bf16,
            FloatUnOp::Sigmoid,
            0.005889892578125,
            lower_sigmoid as ActivationLowerer,
        ),
        (
            Prim::Bf16,
            FloatUnOp::Tanh,
            f64::from(f32::from_bits(0x0001_0000)),
            lower_tanh as ActivationLowerer,
        ),
        (
            Prim::Bf16,
            FloatUnOp::Silu,
            0.00555419921875,
            lower_silu as ActivationLowerer,
        ),
        (
            Prim::Bf16,
            FloatUnOp::Gelu,
            0.0030975341796875,
            lower_gelu as ActivationLowerer,
        ),
    ];

    for (prim, op, image, lower) in cases {
        let input = scalar_from_f64("activation_surface_parity", prim, image).unwrap();
        assert_eq!(
            float_unop(op, input).unwrap(),
            evaluate_tier2(prim, input, lower),
            "scalar {} must equal the rank-zero Tier-2 DAG at {}",
            op.name(),
            prim.name()
        );
    }
}

/// Every section 3.3 graph and the [05-OP-46] `erf`/`erfc` agree between the
/// evaluator's scalar lane and the IR's tensor evaluation of the `tier2`
/// lowering, at every float dtype, on the witnesses of the `Phi` graph: the
/// signed zeros, the infinities, NaN, the deep left tail, and `|x| = 64`.
#[test]
fn section_3_3_graphs_agree_between_the_scalar_lane_and_the_ir_on_the_witnesses() {
    let lowerings: [(FloatUnOp, ActivationLowerer); 8] = [
        (FloatUnOp::Sigmoid, lower_sigmoid),
        (FloatUnOp::Silu, lower_silu),
        (FloatUnOp::Gelu, lower_gelu),
        (FloatUnOp::GeluTanh, lower_gelu_tanh),
        (FloatUnOp::StandardNormalCdf, lower_standard_normal_cdf),
        (FloatUnOp::Tanh, lower_tanh),
        (FloatUnOp::Erf, lower_erf),
        (FloatUnOp::Erfc, lower_erfc),
    ];
    let witnesses = [
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        1.0,
        -0.75,
        -3.0,
        -5.0,
        -9.336,
        -12.6,
        -36.5,
        63.75,
        -63.75,
        64.0,
        -64.0,
        -65.0,
        1e-30,
    ];
    for prim in [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64] {
        for (op, lower) in lowerings {
            for image in witnesses {
                let Ok(input) = scalar_from_f64("activation_surface_parity", prim, image) else {
                    continue;
                };
                let scalar = float_unop(op, input).unwrap();
                let ir = evaluate_tier2(prim, input, lower);
                let both_nan = scalar.as_f64_lossy().is_nan() && ir.as_f64_lossy().is_nan();
                assert!(
                    both_nan || scalar == ir,
                    "{} at {}({image}): scalar {:?}, IR {:?}",
                    op.name(),
                    prim.name(),
                    scalar,
                    ir
                );
            }
        }
    }
}
