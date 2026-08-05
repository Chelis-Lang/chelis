use chelis_ir::dag::{Dag, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_with_strict};
use chelis_ir::tier2::{lower_gelu, lower_sigmoid, lower_silu, lower_tanh};
use chelis_types::types::Prim;
use chelis_types::{FloatUnOp, ScalarValue, float_unop, scalar_from_f64, tensor_from_scalars};

type ActivationLowerer =
    fn(&mut Dag, chelis_ir::dag::NodeId, &TensorType, Option<&str>) -> chelis_ir::dag::NodeId;

fn evaluate_tier2(prim: Prim, input: ScalarValue, lower: ActivationLowerer) -> ScalarValue {
    let ty = TensorType {
        dims: vec![],
        precision: prim,
    };
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load {
            name: "x".to_string().into(),
        },
        vec![],
        ty.clone(),
        None,
    );
    let root = lower(&mut dag, x, &ty, None);
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
            2.9802322387695313e-7,
            lower_silu as ActivationLowerer,
        ),
        (
            Prim::F16,
            FloatUnOp::Gelu,
            2.9802322387695313e-7,
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
