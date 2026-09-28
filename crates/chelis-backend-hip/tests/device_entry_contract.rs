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
        let decl = dag.declare("test");
        let input = dag.add_node(
            decl,
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

#[test]
fn kernel_marshaling_uses_checked_program_rank_and_int64_geometry() {
    // [05-OP-33] admits scalar, empty and arbitrary-rank checked metadata.
    // The same rank must govern parameter declarations and launch arguments.
    for rank in [0, 1, 8, 9, 33] {
        for extent in [0, 1] {
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            let ty = TensorType {
                dims: vec![DimInfo::Lit(extent); rank],
                precision: Prim::F32,
            };
            let input = dag.add_node(
                decl,
                RiscOp::Load { name: "x".into() },
                vec![],
                ty.clone(),
                None,
            );
            let output = dag.add_node(decl, RiscOp::Neg, vec![input], ty, None);
            dag.add_root(output);
            let generated = support::codegen_hip(&dag, "rank_entry").unwrap();
            let source = &generated.c_source;
            let last_axis = rank.max(1) - 1;
            assert!(source.contains(&format!("chelis_device_metadata a_s{last_axis}")));
            assert!(source.contains(&format!("_a_s{last_axis} =")));
            assert!(!source.contains(&format!("chelis_device_metadata a_s{}", rank.max(1))));
            assert!(source.contains("chelis_device_metadata out_size"));
            assert!(source.contains("(chelis_device_metadata)blockIdx.x * blockDim.x"));
            for legacy in [
                "int indices[",
                "int out_size",
                "int t1_size",
                "int64_t a_s",
                "int64_t out_size",
                "->storage_size",
            ] {
                assert!(!source.contains(legacy), "rank {rank}: retained {legacy}");
            }
        }
    }
}

#[test]
fn movement_views_are_complete_before_publication_and_released_before_slots() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let input = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
            precision: Prim::F32,
        },
        None,
    );
    let permuted = dag.add_node(
        decl,
        RiscOp::Permute { axes: vec![1, 0] },
        vec![input],
        TensorType {
            dims: vec![DimInfo::Lit(3), DimInfo::Lit(2)],
            precision: Prim::F32,
        },
        None,
    );
    dag.add_root(permuted);
    let source = support::codegen_hip(&dag, "permuted_entry")
        .unwrap()
        .c_source;
    assert!(source.contains("chelis_metadata_plan_view("));
    assert!(source.contains("chelis_device_tensor_borrow("));
    assert!(source.contains("chelis_device_tensor_clone(o_t1)"));
    assert!(!source.contains("d_t1->strides[0] ="));
    assert!(!source.contains("chelis_gpu_alloc_view("));
    let view_release = source.find("chelis_device_tensor_release(o_t1);").unwrap();
    let slot_release = source
        .find("chelis_device_tensor_release(chelis_slot0);")
        .unwrap();
    assert!(view_release < slot_release);
}

#[test]
fn input_device_preflight_precedes_projection_and_modules_belong_to_the_invocation() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let ty = TensorType {
        dims: vec![DimInfo::Lit(1)],
        precision: Prim::F32,
    };
    let input = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        ty.clone(),
        None,
    );
    let output = dag.add_node(decl, RiscOp::Neg, vec![input], ty, None);
    dag.add_root(output);
    let source = support::codegen_hip(&dag, "context_entry")
        .unwrap()
        .c_source;
    let device = source.split_once("void context_entry_device(").unwrap().1;
    let preflight = device
        .find("chelis_device_tensor_device(inputs[slot])")
        .unwrap();
    assert!(preflight < device.find("chelis_device_tensor_view(inputs[").unwrap());
    assert!(preflight < device.find("chelis_compile_kernel(").unwrap());
    assert!(device.contains("hipModuleUnload("));
    assert!(!source.contains("static hipModule_t"));
}
