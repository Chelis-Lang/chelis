//! spec/11-ffi §2.1: generated entry transports opaque owners and owns escapes.
//! These emission controls supplement the executed device-owner fixture and
//! generated-program lifetime mutations; they are not hardware acceptance.
mod support;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;

#[test]
fn device_entry_borrows_opaque_inputs_and_clones_escaping_results_at_dynamic_rank() {
    for rank in [0, 1, 8, 9, 33] {
        let mut dag = Dag::new();
        let input = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Lit(1); rank],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_root(input);
        let generated = support::codegen_hip(&dag, "owner_entry").unwrap();
        let signature = "const chelis_device_tensor_owner *const *inputs";
        assert!(
            generated.c_source.contains(signature),
            "rank {rank}: device entry must borrow opaque owner handles"
        );
        assert!(
            generated
                .c_source
                .contains("chelis_device_tensor_clone(inputs[0])")
        );
        for legacy in [
            "chelis_gpu_tensor **inputs",
            "chelis_gpu_clone(",
            "outputs[0] = inputs[0];",
            "CHELIS_GPU_MAX_DIM",
        ] {
            assert!(
                !generated.c_source.contains(legacy),
                "rank {rank}: retained bypass {legacy}"
            );
        }
    }
}

#[test]
fn support_root_exposes_generated_packet_and_official_sdk_without_owner_definition() {
    let header = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/runtime/chelis_hip_runtime.h"
    ))
    .unwrap();
    assert!(header.contains("#include \"chelis_device_owner.h\""));
    assert!(header.contains("#include <hipblas/hipblas.h>"));
    for legacy in [
        "__has_include",
        "CHELIS_GPU_MAX_DIM",
        "int shape[",
        "int strides[",
        "hipblasDatatype_t",
        "chelis_device_owner.cpp",
        "struct chelis_device_tensor_owner {",
    ] {
        assert!(
            !header.contains(legacy),
            "published support root retained {legacy}"
        );
    }
}
