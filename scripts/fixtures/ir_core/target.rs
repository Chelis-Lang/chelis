#![no_std]
extern crate alloc;

use alloc::vec;
use chelis_dag_core::dag::{Dag, RiscOp, TensorType};
use chelis_ir_core::eval::{TensorValue, eval_tensor_roots_with_strict};
use chelis_ir_core::resources::Resources;
use chelis_numeric_core::prim::Prim;
use chelis_stack_services::StackServices;

pub fn evaluate<S: StackServices>(stack: &S) {
    let mut dag = Dag::new();
    let declaration = dag.declare("target");
    let root = dag.add_node(declaration, RiscOp::Load { name: "x".into() },
        vec![], TensorType { dims: vec![], precision: Prim::Int64 }, None);
    dag.add_root(root);
    Resources::new(stack).run(|operation| {
        let values = eval_tensor_roots_with_strict(&dag, &[root], |_| Some(
            TensorValue::finalize_from_wide_int("target", Prim::Int64, vec![], vec![9]).unwrap()), operation).unwrap();
        assert_eq!(values[&root].storage().to_i64_exact_vec(), Some(vec![9]));
    });
}
