mod support;

use chelis_backend_hip::memory::MemoryPlan;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::ownership::{
    LiveByteBound, lower_dag_ownership, plan_hip_storage, verify_ownership,
};
use chelis_types::types::Prim;

#[test]
fn physical_peak_estimate_matches_retained_slot_formula() {
    for (reduce, drop_first, expected) in [(false, false, 8), (true, false, 24), (false, true, 20)]
    {
        let mut dag = Dag::new();
        let scalar = TensorType::scalar_f32();
        let first_type = if reduce || drop_first {
            TensorType {
                dims: vec![DimInfo::Lit(4)],
                precision: Prim::F32,
            }
        } else {
            scalar.clone()
        };
        let first = dag.add_node(
            RiscOp::synth_const(Prim::F32, 1.0),
            vec![],
            first_type.clone(),
            None,
        );
        let result = if drop_first {
            dag.add_node(RiscOp::Drop, vec![first], first_type, None);
            dag.add_node(RiscOp::synth_const(Prim::F32, 2.0), vec![], scalar, None)
        } else {
            let op = if reduce {
                RiscOp::Sum {
                    axis: 0,
                    accumulator: Prim::F32,
                }
            } else {
                RiscOp::Neg
            };
            let second = dag.add_node(op, vec![first], scalar.clone(), None);
            dag.add_node(RiscOp::Neg, vec![second], scalar, None)
        };
        dag.add_root(result);
        // Plan the verified graph directly: the preparation DCE can remove an
        // unused Drop chain, which would not exercise its physical allocation.
        let verified = || verify_ownership(lower_dag_ownership(dag.clone()).unwrap()).unwrap();
        let shared = plan_hip_storage(verified()).unwrap();
        let plan = MemoryPlan::from_shared(&shared);
        assert_eq!(shared.max_live_bytes(), LiveByteBound::Exact(expected));
        assert_eq!(plan.peak_device_bytes_estimate(), Some(expected as usize));
        assert_eq!(
            plan.peak_device_bytes_at(&Default::default()).unwrap(),
            expected as usize
        );
        let generated = chelis_backend_hip::codegen_hip(verified(), "physical_peak_probe").unwrap();
        assert_eq!(
            generated.peak_device_bytes_estimate,
            Some(expected as usize)
        );
        assert_eq!(
            generated.peak_device_bytes_at(&Default::default()).unwrap(),
            expected as usize
        );
        assert_eq!(
            generated.peak_device_bytes_formula,
            plan.peak_device_bytes_formula()
        );
        // Even an explicit descriptor Drop does not free the HIP slot pool.
        // In each emitted entry path, every slot allocation precedes cleanup.
        let mut allocated = 0;
        let mut freed = 0;
        for line in generated.c_source.lines() {
            if line.contains("chelis_gpu_alloc(") {
                assert_eq!(freed, 0);
                allocated += 1;
            } else if line.contains("chelis_gpu_free(") {
                assert_eq!(allocated, shared.slots().len());
                freed += 1;
                if freed == allocated {
                    allocated = 0;
                    freed = 0;
                }
            }
        }
        assert_eq!((allocated, freed), (0, 0));
    }
}
