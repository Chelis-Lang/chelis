use chelis_checker_core::{
    build_compiled_library_context, check_ir_with_context, check_linearity,
    check_linearity_with_context,
};
use chelis_dag_core::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir_core::{eval, lower, lowering_trace, pipeline, specialize, vmap};
use chelis_ir_core::resources::{Operation, Resources};
use chelis_numeric_core::dtype_semantics::scalar_from_i64;
use chelis_numeric_core::prim::Prim;
use chelis_stack_services::StackServices;
use chelis_unord::UnordSet;

const STACK_BYTES: usize = 512 * 1024 * 1024;

// The caller reserves this stack before any portable operation starts.
struct ReservedThreadStack;

impl StackServices for ReservedThreadStack {
    fn grow<R>(&self, bytes: usize, run: impl FnOnce() -> R) -> R {
        assert!(bytes <= STACK_BYTES);
        run()
    }

    fn maybe_grow<R>(&self, red_zone: usize, bytes: usize, run: impl FnOnce() -> R) -> R {
        assert!(red_zone < STACK_BYTES);
        assert!(bytes <= STACK_BYTES);
        run()
    }

    fn remaining_stack(&self) -> Option<usize> {
        None
    }
}

fn run() {
    let stack = ReservedThreadStack;
    let resources = Resources::new(&stack);
    let operation = Operation::new(&resources);
    let checker = operation.checker_resources();
    let library_exprs = chelis_deep::parser::parse_str(
        "(def {} base (lit {type: (t-tensor {} (t-prim {} i32))} 3))",
        &stack,
    ).expect("the library syntax must parse");
    let (context, checked_library) = build_compiled_library_context(&library_exprs, &checker)
        .expect("the canonical checker must accept the library");
    let library_facts = check_linearity(checked_library.facts(), &checker)
        .expect("the library must satisfy linearity");
    let library = lower::try_lower_program_to_library(&library_facts, &operation)
        .expect("the library must lower");
    assert_eq!(library.dag().roots().len(), 1);
    assert_eq!(library.program_types()["base"].precision, Prim::Int32);

    let extension_exprs = chelis_deep::parser::parse_str(
        "(def {} out (app {type: (t-tensor {} (t-prim {} i32))} (var {} add) (var {type: (t-tensor {} (t-prim {} i32))} base) (lit {type: (t-tensor {} (t-prim {} i32))} 2)))",
        &stack,
    ).expect("the extension syntax must parse");
    let checked_extension = check_ir_with_context(&context, &extension_exprs, &checker)
        .expect("the canonical checker must accept the extension");
    let extension_facts = check_linearity_with_context(
        &library_facts, checked_extension.facts(), &checker,
    ).expect("the extension must satisfy contextual linearity");
    let composed = lower::try_lower_program_with_context(&library, &extension_facts, &operation)
        .expect("the extension must compose");
    assert_eq!(composed.dag.roots().len(), 2);
    let composed_values = eval::eval_tensor_roots_with_strict(
        &composed.dag, composed.dag.roots(), |_| None, &operation,
    ).expect("the composed integer graph must evaluate");
    let final_root = composed.dag.roots().last().expect("the extension has a root");
    let final_value = &composed_values[final_root];
    assert_eq!(final_value.prim(), Prim::Int32);
    assert_eq!(final_value.storage().scalar_at(0).as_i64_exact(), Some(5));

    let (traced, trace) = lowering_trace::try_lower_program_to_library_with_trace(
        &library_facts, &operation,
    ).expect("the trace must complete");
    assert_eq!(trace.contexts.len(), 1);
    assert!(trace.gradients.is_empty());
    assert!(trace.boundaries.is_empty());
    assert_eq!(
        format!("{:?}", trace.normalization.after_drops),
        format!("{:?}", traced.dag()),
    );
    assert_eq!(format!("{:?}", traced.dag()), format!("{:?}", library.dag()));

    let mut forward = Dag::new();
    let declaration = forward.declare("square");
    let x = forward.add_node(
        declaration, RiscOp::Load { name: "x".into() }, Vec::new(),
        TensorType::scalar_f32(), None,
    );
    let squared = forward.add_node(
        declaration, RiscOp::Mul, vec![x, x], TensorType::scalar_f32(), None,
    );
    forward.add_root(squared);
    for specialized in [
        specialize::specialize_for_exact_arithmetic(&forward),
        specialize::specialize_for_blas(&forward),
    ] {
        assert_eq!(specialized.nodes().len(), 2);
        let values = eval::eval_tensor_roots_with_strict(
            &specialized, specialized.roots(),
            |name| (name == "x").then(|| eval::TensorValue::scalar(3.0)), &operation,
        ).expect("the specialized square must evaluate");
        assert_eq!(
            values[&specialized.roots()[0]].storage().scalar_at(0),
            scalar_from_i64("consumer", Prim::F32, 9).unwrap(),
        );
    }

    let (batched, node_map) = vmap::vectorize_axis0_with_node_map(&forward, DimInfo::Lit(2))
        .expect("the square must support two mapped rows");
    assert_eq!(node_map.len(), forward.nodes().len());
    let batched_root = node_map[squared.0];
    assert_eq!(batched.get(batched_root).unwrap().output_type.dims, vec![DimInfo::Lit(2)]);
    let batched_values = eval::eval_tensor_roots_with_strict(
        &batched, &[batched_root],
        |name| (name == "x").then(|| eval::TensorValue::from_vec(vec![2], vec![2.0, 3.0])),
        &operation,
    ).expect("the batched square must evaluate");
    let batched_value = &batched_values[&batched_root];
    assert_eq!(batched_value.prim(), Prim::F32);
    assert_eq!(batched_value.shape, vec![2]);
    assert_eq!(batched_value.storage().scalar_at(0), scalar_from_i64("consumer", Prim::F32, 4).unwrap());
    assert_eq!(batched_value.storage().scalar_at(1), scalar_from_i64("consumer", Prim::F32, 9).unwrap());

    let differentiated = pipeline::grad_then_fuse_checked(&forward, squared, &[x])
        .expect("the square must differentiate before fusion");
    let derivative = differentiated.grad_nodes[&x];
    let differentiated_values = eval::eval_tensor_roots_with_strict(
        &differentiated.dag, &[differentiated.output_node, derivative],
        |name| (name == "x").then(|| eval::TensorValue::scalar(3.0)), &operation,
    ).expect("the fused derivative must evaluate");
    assert_eq!(
        differentiated_values[&derivative].storage().scalar_at(0),
        scalar_from_i64("consumer", Prim::F32, 6).unwrap(),
    );

    let mut keyed = Dag::new();
    let declaration = keyed.declare("captured_key");
    let key = keyed.add_node(
        declaration, RiscOp::Load { name: "key".into() }, Vec::new(),
        TensorType { dims: Vec::new(), precision: Prim::Key }, None,
    );
    keyed.add_root(key);
    let captures = [String::from("key")].into_iter().collect::<UnordSet<_>>();
    assert!(
        vmap::vectorize_axis0_with_captures(&keyed, DimInfo::Lit(2), &captures).is_err(),
        "a captured key must not broadcast across rows",
    );
    let mapped_keys = vmap::vectorize_axis0_with_captures(
        &keyed, DimInfo::Lit(2), &UnordSet::new(),
    ).expect("independent mapped keys must remain admissible");
    let mapped_key_type = &mapped_keys.get(mapped_keys.roots()[0]).unwrap().output_type;
    assert_eq!(mapped_key_type.dims, vec![DimInfo::Lit(2)]);
    assert_eq!(mapped_key_type.precision, Prim::Key);
    println!("IR_LOWER_CONSUMER {{\"lowered_roots\":1,\"contextual_roots\":2,\"contextual_i32\":5,\"batch_f32\":[4,9],\"derivative_f32\":6,\"trace_contexts\":1,\"refusals\":1}}");
}

fn main() {
    std::thread::Builder::new()
        .name("ir-lower-consumer".into())
        .stack_size(STACK_BYTES)
        .spawn(run)
        .expect("the platform must reserve the consumer stack")
        .join()
        .expect("the portable consumer must complete");
}
