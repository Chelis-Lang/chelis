//! Structural tests for HIP code generation (S1-S13).
//!
//! These verify the generated C/HIP source is well-formed WITHOUT requiring
//! a GPU or HIP runtime. They run in default CI.

use chelis_backend_hip::codegen_hip;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::fuse::fuse;
use chelis_types::types::Prim;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn scalar_f32() -> TensorType {
    TensorType::scalar_f32()
}

fn vec_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn mat_f32(rows: usize, cols: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(rows), DimInfo::Lit(cols)],
        precision: Prim::F32,
    }
}

fn write_temp_file(dir: &Path, name: &str, contents: &str) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, contents).expect("write temp file");
    path
}

fn hipcc_available() -> bool {
    Command::new("hipcc")
        .arg("--version")
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false)
}

fn hip_runtime_src_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(chelis_backend_hip::runtime_dir())
}

fn cpu_runtime_src_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../chelis-backend-c/runtime")
}

/// Build a simple DAG: const(a) + const(b)
fn dag_add_consts() -> Dag {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
    let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
    let c = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32());
    dag.add_root(c);
    dag
}

/// Build a DAG with a Load input
fn dag_with_load() -> Dag {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4));
    let c = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(4));
    let out = dag.add_node(RiscOp::Add, vec![x, c], vec_f32(4));
    dag.add_root(out);
    dag
}

// ===========================================================================
// S1: Generated code includes runtime header
// ===========================================================================

#[test]
fn s1_includes_hip_runtime_header() {
    let dag = dag_add_consts();
    let result = codegen_hip(&dag, "test_s1");
    assert!(
        result
            .c_source
            .contains("#include \"chelis_hip_runtime.h\""),
        "Generated code must include HIP runtime header"
    );
}

#[test]
fn s1_neg_no_cpu_runtime_include() {
    let dag = dag_add_consts();
    let result = codegen_hip(&dag, "test_s1_neg");
    // Should NOT directly include the CPU-only runtime (it's included via hip runtime)
    assert!(
        !result
            .c_source
            .contains("#include \"chelis_runtime.h\"\n#include \"chelis_runtime.h\""),
        "Should not double-include CPU runtime"
    );
}

// ===========================================================================
// S2: Kernel strings are const char* literals
// ===========================================================================

#[test]
fn s2_kernel_strings_are_const_char() {
    let dag = dag_add_consts();
    let result = codegen_hip(&dag, "test_s2");
    assert!(
        result.c_source.contains("const char *"),
        "Kernel source strings should be const char* literals"
    );
}

#[test]
fn s2_kernel_strings_escape_embedded_quotes() {
    let dag = dag_add_consts();
    let result = codegen_hip(&dag, "test_quote_escape");
    assert!(
        result
            .c_source
            .contains("extern \\\"C\\\" __global__ void kernel_add"),
        "Kernel source must escape embedded quotes inside generated C string literals"
    );
}

#[test]
fn s2_neg_const_only_dag_only_fill_kernel() {
    let mut dag = Dag::new();
    let c = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], scalar_f32());
    dag.add_root(c);
    let result = codegen_hip(&dag, "test_s2_neg");
    // A const-only DAG should only have the fill kernel, no compute kernels
    assert!(
        result.c_source.contains("kernel_fill"),
        "Const-only DAG should have a fill kernel"
    );
    assert!(
        !result.c_source.contains("kernel_add"),
        "Const-only DAG should not have compute kernels"
    );
}

// ===========================================================================
// S3: Each elementwise op has a kernel
// ===========================================================================

#[test]
fn s3_all_elementwise_ops_emit_kernels() {
    let ops_and_names: Vec<(RiscOp, &str)> = vec![
        (RiscOp::Add, "add"),
        (RiscOp::Mul, "mul"),
        (RiscOp::MaxElem, "max_elem"),
        (RiscOp::CmpLt, "cmplt"),
    ];
    for (op, name) in &ops_and_names {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
        let c = dag.add_node(op.clone(), vec![a, b], scalar_f32());
        dag.add_root(c);
        let result = codegen_hip(&dag, &format!("test_{name}"));
        assert!(
            result.c_source.contains("chelis_launch_kernel"),
            "Binary op '{name}' must emit a kernel launch"
        );
    }
}

#[test]
fn s3_all_unary_ops_emit_kernels() {
    let ops_and_names: Vec<(RiscOp, &str)> = vec![
        (RiscOp::Neg, "neg"),
        (RiscOp::Exp, "exp"),
        (RiscOp::Log, "log"),
        (RiscOp::Sin, "sin"),
        (RiscOp::Sqrt, "sqrt"),
    ];
    for (op, name) in &ops_and_names {
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
        let c = dag.add_node(op.clone(), vec![a], scalar_f32());
        dag.add_root(c);
        let result = codegen_hip(&dag, &format!("test_{name}"));
        assert!(
            result.c_source.contains("chelis_launch_kernel"),
            "Unary op '{name}' must emit a kernel launch"
        );
    }
}

// ===========================================================================
// S4: Reduction kernels use naive pattern (inner loop over axis)
// ===========================================================================

#[test]
fn s4_sum_reduction_emits_kernel() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(3, 4));
    let s = dag.add_node(RiscOp::Sum { axis: 1 }, vec![x], vec_f32(3));
    dag.add_root(s);
    let result = codegen_hip(&dag, "test_sum");
    assert!(
        result.c_source.contains("chelis_launch_kernel"),
        "Sum reduction must emit a kernel launch"
    );
}

#[test]
fn s4_max_reduce_emits_kernel() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(3, 4));
    let m = dag.add_node(RiscOp::MaxReduce { axis: 0 }, vec![x], vec_f32(4));
    dag.add_root(m);
    let result = codegen_hip(&dag, "test_max_reduce");
    assert!(
        result.c_source.contains("chelis_launch_kernel"),
        "MaxReduce must emit a kernel launch"
    );
}

// ===========================================================================
// S5: Movement ops are NOT kernels (host-side metadata only)
// ===========================================================================

#[test]
fn s5_reshape_no_kernel_launch() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(2, 3));
    let r = dag.add_node(
        RiscOp::Reshape {
            new_shape: vec![DimInfo::Lit(6)],
        },
        vec![x],
        vec_f32(6),
    );
    dag.add_root(r);
    let result = codegen_hip(&dag, "test_reshape");
    // Reshape itself must not add a kernel launch
    assert!(
        result.c_source.contains("chelis_gpu_alloc_view"),
        "Reshape must use alloc_view (metadata-only), not a kernel"
    );
}

#[test]
fn s5_permute_no_kernel_launch() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(2, 3));
    let p = dag.add_node(RiscOp::Permute { axes: vec![1, 0] }, vec![x], mat_f32(3, 2));
    dag.add_root(p);
    let result = codegen_hip(&dag, "test_permute");
    assert!(
        result.c_source.contains("chelis_gpu_alloc_view"),
        "Permute must use alloc_view (metadata-only)"
    );
}

#[test]
fn s5_expand_no_kernel_launch() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(3));
    let e = dag.add_node(RiscOp::Expand { axis: 0, size: 4 }, vec![x], mat_f32(4, 3));
    dag.add_root(e);
    let result = codegen_hip(&dag, "test_expand");
    assert!(
        result.c_source.contains("chelis_gpu_alloc_view"),
        "Expand must use alloc_view (metadata-only)"
    );
}

#[test]
fn s5_realize_materializes_with_kernel_not_view() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(6));
    let s = dag.add_node(RiscOp::Stride { strides: vec![2] }, vec![x], vec_f32(3));
    let r = dag.add_node(RiscOp::Realize, vec![s], vec_f32(3));
    dag.add_root(r);

    let result = codegen_hip(&dag, "test_realize");
    assert!(
        result.c_source.contains("kernel_cast"),
        "Realize must materialize through a copy-style kernel launch"
    );
    assert!(
        !result.c_source.contains(
            "chelis_gpu_alloc_view(1, (int[]){ 3 }, CHELIS_F32, d_t1->data, d_t1->storage_size)"
        ),
        "Realize must not lower to a metadata-only view"
    );
}

// ===========================================================================
// S6: Kernel launch uses correct grid/block (ceil(size/256))
// ===========================================================================

#[test]
fn s6_grid_block_in_launch() {
    let dag = dag_add_consts();
    let result = codegen_hip(&dag, "test_s6");
    // Generated code should reference block size 256
    assert!(
        result.c_source.contains("256"),
        "Block size 256 should appear in generated code"
    );
}

// ===========================================================================
// S7: Host code follows DAG topological order
// ===========================================================================

#[test]
fn s7_topo_order_preserved() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
    let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
    let c = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32());
    let d = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
    let e = dag.add_node(RiscOp::Mul, vec![c, d], scalar_f32());
    dag.add_root(e);
    let result = codegen_hip(&dag, "test_topo");
    let src = &result.c_source;
    // t0 (const), t1 (const), t2 (add), t3 (const), t4 (mul)
    // Each must appear in order
    let pos_t2 = src.find("d_t2").expect("d_t2 should exist");
    let pos_t4 = src.find("d_t4").expect("d_t4 should exist");
    assert!(pos_t2 < pos_t4, "add (t2) must appear before mul (t4)");
}

// ===========================================================================
// S8: Input/output label contract matches C backend
// ===========================================================================

#[test]
fn s8_input_labels_match() {
    let dag = dag_with_load();
    let result = codegen_hip(&dag, "test_labels");
    assert_eq!(result.input_labels, vec!["x"]);
}

#[test]
fn s8_output_labels_root() {
    let dag = dag_add_consts();
    let result = codegen_hip(&dag, "test_labels");
    assert_eq!(result.output_labels, vec!["root0"]);
}

#[test]
fn s8_duplicate_load_single_slot() {
    let mut dag = Dag::new();
    let x1 = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4));
    let x2 = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4));
    let out = dag.add_node(RiscOp::Add, vec![x1, x2], vec_f32(4));
    dag.add_root(out);
    let result = codegen_hip(&dag, "test_dup_load");
    assert_eq!(
        result.input_labels,
        vec!["x"],
        "Repeated Load(x) shares one input slot"
    );
}

// ===========================================================================
// S9: cmplt kernel emits 1.0f/0.0f (not integer bool)
// ===========================================================================

#[test]
fn s9_cmplt_float_result() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
    let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
    let c = dag.add_node(
        RiscOp::CmpLt,
        vec![a, b],
        TensorType {
            dims: vec![],
            precision: Prim::Bool,
        },
    );
    dag.add_root(c);
    let result = codegen_hip(&dag, "test_cmplt");
    assert!(
        result.c_source.contains("1.0f") && result.c_source.contains("0.0f"),
        "cmplt kernel must produce float 1.0f/0.0f, not integer bool"
    );
}

// ===========================================================================
// S10: Device helpers in every compute kernel
// ===========================================================================

#[test]
fn s10_device_helpers_present() {
    let dag = dag_add_consts();
    let result = codegen_hip(&dag, "test_helpers");
    assert!(
        result.c_source.contains("chelis_flat_to_indices"),
        "Device helper chelis_flat_to_indices must be in kernel source"
    );
    assert!(
        result.c_source.contains("chelis_indices_to_flat"),
        "Device helper chelis_indices_to_flat must be in kernel source"
    );
    assert!(
        result.c_source.contains("chelis_gpu_failure = 0"),
        "Kernel source must declare the debug failure flag"
    );
    assert!(
        result.c_source.contains("CHELIS_GUARD_INDEX"),
        "Kernel source must include bounds-guard instrumentation"
    );
}

// ===========================================================================
// S11: Kernel JIT uses static caching
// ===========================================================================

#[test]
fn s11_static_module_caching() {
    let dag = dag_add_consts();
    let result = codegen_hip(&dag, "test_cache");
    assert!(
        result.c_source.contains("static hipModule_t"),
        "Kernel modules must be cached with 'static hipModule_t'"
    );
}

#[test]
fn s11_launches_reset_and_check_failure_flag() {
    let dag = dag_add_consts();
    let result = codegen_hip(&dag, "test_failure_checks");
    let src = &result.c_source;
    let launch_count = src.matches("chelis_launch_kernel").count();
    assert_eq!(
        src.matches("chelis_prepare_kernel_launch").count(),
        launch_count,
        "Every kernel launch must reset the debug failure flag"
    );
    assert_eq!(
        src.matches("chelis_finalize_kernel_launch").count(),
        launch_count,
        "Every kernel launch must check the debug failure flag"
    );

    let prepare_pos = src
        .find("chelis_prepare_kernel_launch(mod_kernel_fill)")
        .expect("fill reset present");
    let launch_pos = src
        .find("chelis_launch_kernel(mod_kernel_fill")
        .expect("fill launch present");
    let finalize_pos = src
        .find("chelis_finalize_kernel_launch(mod_kernel_fill")
        .expect("fill finalize present");
    assert!(
        prepare_pos < launch_pos && launch_pos < finalize_pos,
        "Failure flag reset/check must bracket the kernel launch"
    );
}

// ===========================================================================
// S12: Host↔device transfers bracket the function
// ===========================================================================

#[test]
fn s12_transfers_present() {
    let dag = dag_with_load();
    let result = codegen_hip(&dag, "test_transfer");
    assert!(
        result.c_source.contains("chelis_host_to_device"),
        "Input transfer (host→device) must be present"
    );
    assert!(
        result.c_source.contains("chelis_device_to_host"),
        "Output transfer (device→host) must be present"
    );
}

#[test]
fn s12_transfer_order() {
    let dag = dag_with_load();
    let result = codegen_hip(&dag, "test_transfer_order");
    let src = &result.c_source;
    let h2d_pos = src.find("chelis_host_to_device").expect("h2d present");
    let d2h_pos = src.find("chelis_device_to_host").expect("d2h present");
    assert!(
        h2d_pos < d2h_pos,
        "host_to_device must come before device_to_host"
    );
}

// ===========================================================================
// S13: Cleanup frees GPU tensors; views don't double-free
// ===========================================================================

#[test]
fn s13_cleanup_frees_intermediates() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
    let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
    let c = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32());
    dag.add_root(c);
    let result = codegen_hip(&dag, "test_cleanup");
    // a (t0) and b (t1) are intermediates, should be freed
    assert!(
        result.c_source.contains("chelis_gpu_free(d_t0)"),
        "Intermediate const t0 must be freed"
    );
    assert!(
        result.c_source.contains("chelis_gpu_free(d_t1)"),
        "Intermediate const t1 must be freed"
    );
}

#[test]
fn s13_outputs_not_freed() {
    let dag = dag_add_consts();
    let result = codegen_hip(&dag, "test_no_free_output");
    // t2 is the output (root), should NOT be freed
    assert!(
        !result.c_source.contains("chelis_gpu_free(d_t2)"),
        "Output tensor must not be freed"
    );
}

#[test]
fn s13_loads_not_freed() {
    let dag = dag_with_load();
    let result = codegen_hip(&dag, "test_no_free_load");
    // Load tensors are borrowed, must not be freed
    assert!(
        !result.c_source.contains("chelis_gpu_free(d_t0)"),
        "Load tensor must not be freed (borrowed from caller)"
    );
}

#[test]
fn s13_views_use_view_free() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(2, 3));
    let p = dag.add_node(RiscOp::Permute { axes: vec![1, 0] }, vec![x], mat_f32(3, 2));
    let c = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], mat_f32(3, 2));
    let out = dag.add_node(RiscOp::Add, vec![p, c], mat_f32(3, 2));
    dag.add_root(out);
    let result = codegen_hip(&dag, "test_view_free");
    // Permute (t1) is a view — must use chelis_gpu_free_view, NOT chelis_gpu_free
    assert!(
        result.c_source.contains("chelis_gpu_free_view(d_t1)"),
        "View (permute) must use chelis_gpu_free_view, not chelis_gpu_free"
    );
    assert!(
        !result.c_source.contains("chelis_gpu_free(d_t1)"),
        "View (permute) must NOT use chelis_gpu_free (would double-free device memory)"
    );
}

#[test]
fn s14_generated_hip_source_compiles_when_hipcc_available() {
    if !hipcc_available() {
        eprintln!("skipping: hipcc not available");
        return;
    }

    let dag = dag_add_consts();
    let result = codegen_hip(&dag, "test_compile");
    let tmp = tempfile::tempdir().expect("tempdir");

    let hip_rt = hip_runtime_src_dir();
    let cpu_rt = cpu_runtime_src_dir();
    write_temp_file(
        tmp.path(),
        "chelis_hip_runtime.h",
        &fs::read_to_string(hip_rt.join("chelis_hip_runtime.h")).expect("hip runtime header"),
    );
    write_temp_file(
        tmp.path(),
        "chelis_runtime.h",
        &fs::read_to_string(cpu_rt.join("chelis_runtime.h")).expect("cpu runtime header"),
    );
    write_temp_file(
        tmp.path(),
        "chelis_runtime.c",
        &fs::read_to_string(cpu_rt.join("chelis_runtime.c")).expect("cpu runtime source"),
    );
    write_temp_file(tmp.path(), "model.cpp", &result.c_source);
    let main_cpp = r#"
#include "chelis_runtime.h"
void test_compile(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {
    chelis_tensor *outputs[1] = {0};
    test_compile(NULL, 0, outputs, 1);
    chelis_free(outputs[0]);
    return 0;
}
"#;
    write_temp_file(tmp.path(), "main.cpp", main_cpp);

    let bin_path = tmp.path().join("hip_compile_smoke");
    let output = Command::new("hipcc")
        .arg("-O2")
        .arg(tmp.path().join("main.cpp"))
        .arg(tmp.path().join("model.cpp"))
        .arg(tmp.path().join("chelis_runtime.c"))
        .arg("-lhiprtc")
        .arg("-o")
        .arg(&bin_path)
        .output()
        .expect("run hipcc");
    assert!(
        output.status.success(),
        "hipcc failed:\nstderr: {}\nsource:\n{}",
        String::from_utf8_lossy(&output.stderr),
        result.c_source
    );
}

// ===========================================================================
// SF1: FusedElem emits single kernel launch
// ===========================================================================

#[test]
fn sf1_fused_elem_single_kernel_launch() {
    // Build a DAG with add→neg, fuse it, codegen_hip, count kernel launches.
    // The fused chain should produce exactly 1 kernel launch for the fused ops
    // (plus 2 fill launches for the constants).
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(4));
    let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], vec_f32(4));
    let c = dag.add_node(RiscOp::Add, vec![a, b], vec_f32(4));
    let d = dag.add_node(RiscOp::Neg, vec![c], vec_f32(4));
    dag.add_root(d);

    let fused = fuse(&dag);
    let result = codegen_hip(&fused, "test_sf1");

    // Count kernel launches: should be 2 fills + 1 fused = 3 total
    let launch_count = result.c_source.matches("chelis_launch_kernel").count();
    // Without fusion we'd have 2 fills + add + neg = 4 launches.
    // With fusion: 2 fills + 1 fused = 3.
    assert_eq!(
        launch_count, 3,
        "Fused add→neg should produce 3 kernel launches (2 fill + 1 fused), got {launch_count}"
    );
}

// ===========================================================================
// SF2: Fused kernel has chained computation in body
// ===========================================================================

#[test]
fn sf2_fused_kernel_chained_computation() {
    // The generated fused kernel source should contain chained register operations.
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(4));
    let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], vec_f32(4));
    let c = dag.add_node(RiscOp::Add, vec![a, b], vec_f32(4));
    let d = dag.add_node(RiscOp::Neg, vec![c], vec_f32(4));
    dag.add_root(d);

    let fused = fuse(&dag);
    let result = codegen_hip(&fused, "test_sf2");
    let src = &result.c_source;

    // The embedded kernel source (escaped in a C string literal) should contain
    // chained register variables: float v0 = ... and float v1 = ...
    assert!(
        src.contains("float v0 ="),
        "Fused kernel must contain 'float v0 =' for first step"
    );
    assert!(
        src.contains("float v1 ="),
        "Fused kernel must contain 'float v1 =' for second step"
    );
}

// ===========================================================================
// SF3: No intermediate GPU allocation for fused chain
// ===========================================================================

#[test]
fn sf3_no_intermediate_alloc_in_fused_chain() {
    // Between the fused kernel alloc and its launch, there should be no extra
    // chelis_gpu_alloc calls (the intermediate is computed in registers).
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(4));
    let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], vec_f32(4));
    let c = dag.add_node(RiscOp::Add, vec![a, b], vec_f32(4));
    let d = dag.add_node(RiscOp::Neg, vec![c], vec_f32(4));
    dag.add_root(d);

    let fused = fuse(&dag);
    let result = codegen_hip(&fused, "test_sf3");

    // Without fusion: 2 const allocs + add alloc + neg alloc = 4 allocs.
    // With fusion: 2 const allocs + 1 fused output alloc = 3 allocs.
    let alloc_count = result.c_source.matches("chelis_gpu_alloc(").count();
    assert_eq!(
        alloc_count, 3,
        "Fused chain should have 3 GPU allocs (2 const + 1 fused output), got {alloc_count}"
    );
}

// ===========================================================================
// SFR1: Elementwise→reduction fusion eliminates intermediate buffer
// ===========================================================================

#[test]
fn sfr1_fused_elem_into_reduction_no_intermediate_alloc() {
    // add(x, const) → neg → sum should fuse: add→neg becomes FusedElem,
    // then the FusedElem feeds sum as sole consumer → inlined into reduction.
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], mat_f32(3, 4));
    let c = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(3, 4));
    let added = dag.add_node(RiscOp::Add, vec![x, c], mat_f32(3, 4));
    let negated = dag.add_node(RiscOp::Neg, vec![added], mat_f32(3, 4));
    let summed = dag.add_node(RiscOp::Sum { axis: 1 }, vec![negated], vec_f32(3));
    dag.add_root(summed);

    let fused = fuse(&dag);
    let result = codegen_hip(&fused, "test_sfr1");
    let src = &result.c_source;

    // With elem→elem fusion: add→neg becomes FusedElem.
    // With elem→reduction fusion: the FusedElem is inlined into the sum kernel.
    // Result: 1 const fill + 1 fused reduction = 2 kernel launches.
    let launch_count = src.matches("chelis_launch_kernel").count();
    assert_eq!(
        launch_count, 2,
        "Fused add→neg→sum should produce 2 kernel launches (1 fill + 1 fused reduce), got {launch_count}"
    );

    // The fused reduction kernel name should appear in the source
    assert!(
        src.contains("kernel_fused_sum_"),
        "Should contain a fused sum kernel name"
    );

    // The fused reduction kernel should read from external inputs (ext0, ext1)
    // embedded in the kernel string
    assert!(
        src.contains("ext0") && src.contains("ext1"),
        "Fused reduction kernel should reference external inputs ext0, ext1"
    );
}

// ===========================================================================
// SFR2: Fused reduction with max_reduce
// ===========================================================================

#[test]
fn sfr2_fused_elem_into_max_reduce() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], mat_f32(3, 4));
    let c = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(3, 4));
    let added = dag.add_node(RiscOp::Add, vec![x, c], mat_f32(3, 4));
    let negated = dag.add_node(RiscOp::Neg, vec![added], mat_f32(3, 4));
    let maxed = dag.add_node(RiscOp::MaxReduce { axis: 1 }, vec![negated], vec_f32(3));
    dag.add_root(maxed);

    let fused = fuse(&dag);
    let result = codegen_hip(&fused, "test_sfr2");
    let src = &result.c_source;

    assert!(
        src.contains("kernel_fused_maxred_"),
        "Should contain a fused max-reduce kernel name"
    );

    let launch_count = src.matches("chelis_launch_kernel").count();
    assert_eq!(
        launch_count, 2,
        "Fused add→neg→max_reduce should produce 2 launches (1 fill + 1 fused reduce), got {launch_count}"
    );
}

// ===========================================================================
// SFR3: Multi-consumer FusedElem is NOT inlined into reduction
// ===========================================================================

#[test]
fn sfr3_multi_consumer_fused_elem_not_inlined() {
    // If the FusedElem has multiple consumers, it must be materialized.
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], mat_f32(3, 4));
    let c = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(3, 4));
    let added = dag.add_node(RiscOp::Add, vec![x, c], mat_f32(3, 4));
    let negated = dag.add_node(RiscOp::Neg, vec![added], mat_f32(3, 4));
    // Two consumers of the fused chain output:
    let summed = dag.add_node(RiscOp::Sum { axis: 1 }, vec![negated], vec_f32(3));
    dag.add_root(negated); // negated is also a root → 2 consumers
    dag.add_root(summed);

    let fused = fuse(&dag);
    let result = codegen_hip(&fused, "test_sfr3");
    let src = &result.c_source;

    // Since the FusedElem output is both a root and consumed by the sum,
    // it should NOT be inlined — we should see a normal sum kernel.
    assert!(
        !src.contains("kernel_fused_sum_"),
        "Multi-consumer FusedElem should NOT be inlined into reduction"
    );
    assert!(
        src.contains("kernel_sum_ax"),
        "Should use standard reduction kernel for multi-consumer case"
    );
}

#[test]
fn sfr4_realize_blocks_fused_kernel_emission() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4));
    let y = dag.add_node(RiscOp::Load { name: "y".into() }, vec![], vec_f32(4));
    let added = dag.add_node(RiscOp::Add, vec![x, y], vec_f32(4));
    let realized = dag.add_node(RiscOp::Realize, vec![added], vec_f32(4));
    let negated = dag.add_node(RiscOp::Neg, vec![realized], vec_f32(4));
    dag.add_root(negated);

    let fused = fuse(&dag);
    let result = codegen_hip(&fused, "test_realize_barrier");
    let src = &result.c_source;

    assert!(
        !src.contains("kernel_fused_"),
        "realize() must prevent fused kernel emission across the barrier"
    );
    assert!(
        src.matches("chelis_launch_kernel").count() >= 2,
        "realize() barrier should leave separate launches for add/copy/neg"
    );
}
