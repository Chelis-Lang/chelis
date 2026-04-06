//! Adversarial red-team tests for the HIP backend.
//!
//! These probe edge cases not covered by S1-S13 structural tests.

use chelis_backend_hip::codegen_hip;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;

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

// ===========================================================================
// RT1: Load -> Permute -> Add -> Sum (stride handling through chain)
// ===========================================================================

#[test]
fn rt1_load_permute_add_sum_chain() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], mat_f32(3, 4));
    let p = dag.add_node(RiscOp::Permute { axes: vec![1, 0] }, vec![x], mat_f32(4, 3));
    let c = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(4, 3));
    let a = dag.add_node(RiscOp::Add, vec![p, c], mat_f32(4, 3));
    let s = dag.add_node(RiscOp::Sum { axis: 1 }, vec![a], vec_f32(4));
    dag.add_root(s);
    let result = codegen_hip(&dag, "test_chain");
    let src = &result.c_source;

    // Verify the full chain is emitted
    assert!(src.contains("chelis_host_to_device"), "Load must transfer");
    assert!(
        src.contains("chelis_gpu_alloc_view"),
        "Permute must be a view"
    );
    assert!(src.contains("kernel_add"), "Add kernel present");
    assert!(src.contains("kernel_sum"), "Sum kernel present");
    // Verify permute stride reorder exists
    assert!(
        src.contains("d_t1->strides[0] = d_t0->strides[1]"),
        "Permute must reorder strides: dim 0 gets old dim 1"
    );
    assert!(
        src.contains("d_t1->strides[1] = d_t0->strides[0]"),
        "Permute must reorder strides: dim 1 gets old dim 0"
    );
}

// ===========================================================================
// RT2: Two Loads and a Store
// ===========================================================================

#[test]
fn rt2_two_loads_store() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4));
    let y = dag.add_node(RiscOp::Load { name: "y".into() }, vec![], vec_f32(4));
    let add = dag.add_node(RiscOp::Add, vec![x, y], vec_f32(4));
    let store = dag.add_node(
        RiscOp::Store {
            name: "result".into(),
        },
        vec![add],
        vec_f32(4),
    );
    dag.add_root(store);
    let result = codegen_hip(&dag, "test_store");
    let src = &result.c_source;

    // Store should alias its input
    assert!(
        src.contains("d_t3 = d_t2"),
        "Store must alias its input, not allocate"
    );
    // Output labels should include the store name
    assert_eq!(result.output_labels, vec!["result"]);
    // Input labels should be x, y in order
    assert_eq!(result.input_labels, vec!["x", "y"]);
    // Verify both loads transfer data
    let h2d_count = src.matches("chelis_host_to_device").count();
    assert_eq!(
        h2d_count, 2,
        "Two loads should produce two host_to_device transfers"
    );
}

// ===========================================================================
// RT3: Scalar-only DAG (all ndim=0 tensors, treated as ndim=1 size=1)
// ===========================================================================

#[test]
fn rt3_scalar_only_dag() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Const { value: 3.125 }, vec![], scalar_f32());
    let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
    let c = dag.add_node(RiscOp::Mul, vec![a, b], scalar_f32());
    dag.add_root(c);
    let result = codegen_hip(&dag, "test_scalar");
    let src = &result.c_source;

    // Scalar should be treated as ndim=1, size=1
    assert!(
        src.contains("chelis_gpu_alloc(1, (int[]){1}"),
        "Scalar must be allocated as ndim=1 size=1"
    );
    // Grid should be ceil(1/256) = 1
    assert!(
        src.contains("dim3(1)"),
        "Scalar kernel launch should have grid=1"
    );
}

// ===========================================================================
// RT4: Fan-out (same const feeds two ops)
// ===========================================================================

#[test]
fn rt4_fanout_same_input_two_ops() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], vec_f32(4));
    let a = dag.add_node(RiscOp::Add, vec![x, x], vec_f32(4));
    let b = dag.add_node(RiscOp::Mul, vec![x, a], vec_f32(4));
    dag.add_root(b);
    let result = codegen_hip(&dag, "test_fanout");
    let src = &result.c_source;

    // x (d_t0) is used by both add and mul, should not be freed until after both
    // The output is t2, so t0 and t1 should be freed
    assert!(
        src.contains("chelis_gpu_free(d_t0)"),
        "t0 should be freed (not an output)"
    );
    assert!(
        src.contains("chelis_gpu_free(d_t1)"),
        "t1 should be freed (not an output)"
    );
    // t0 should not be freed before it's used by both ops
    // Verify topological order: t0 defined before t1 (add) and t2 (mul)
    let t0_def = src.find("d_t0 = chelis_gpu_alloc").expect("t0 alloc");
    // Look for the actual kernel launch calls (not the static module cache)
    let t1_alloc = src
        .find("d_t1 = chelis_gpu_alloc")
        .expect("t1 alloc (add output)");
    let t2_alloc = src
        .find("d_t2 = chelis_gpu_alloc")
        .expect("t2 alloc (mul output)");
    assert!(
        t0_def < t1_alloc && t1_alloc < t2_alloc,
        "Fan-out: t0 must be allocated before t1 (add) and t2 (mul)"
    );
}

// ===========================================================================
// RT5: Stride op must multiply strides, not just copy
// ===========================================================================

#[test]
fn rt5_stride_op_multiplies_strides() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(6));
    let _s = dag.add_node(RiscOp::Stride { strides: vec![2] }, vec![x], vec_f32(3));
    let result = codegen_hip(&dag, "test_stride");
    let src = &result.c_source;

    // The stride op MUST multiply strides by the stride factors.
    // If it just copies strides, every-other-element access is broken.
    assert!(
        src.contains("* 2"),
        "Stride op must multiply strides by factor 2, got just a copy:\n{src}"
    );
}

// ===========================================================================
// RT6: Multiple Store nodes produce correct output slots
// ===========================================================================

#[test]
fn rt6_multiple_stores() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(4));
    let y = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], vec_f32(4));
    let add = dag.add_node(RiscOp::Add, vec![x, y], vec_f32(4));
    let store_a = dag.add_node(
        RiscOp::Store {
            name: "out_a".into(),
        },
        vec![add],
        vec_f32(4),
    );
    let mul = dag.add_node(RiscOp::Mul, vec![x, y], vec_f32(4));
    let store_b = dag.add_node(
        RiscOp::Store {
            name: "out_b".into(),
        },
        vec![mul],
        vec_f32(4),
    );
    dag.add_root(store_a);
    dag.add_root(store_b);
    let result = codegen_hip(&dag, "test_multi_store");

    assert_eq!(result.output_labels, vec!["out_a", "out_b"]);
    let src = &result.c_source;
    // Both stores should produce device_to_host transfers
    let d2h_count = src.matches("chelis_device_to_host").count();
    assert_eq!(
        d2h_count, 2,
        "Two stores should produce two device_to_host transfers"
    );
}

// ===========================================================================
// RT7: Reduction with different axis/size gets different kernel name
// ===========================================================================

#[test]
fn rt7_different_reductions_different_kernels() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(3, 4));
    let sum_ax0 = dag.add_node(RiscOp::Sum { axis: 0 }, vec![x], vec_f32(4));
    let sum_ax1 = dag.add_node(RiscOp::Sum { axis: 1 }, vec![x], vec_f32(3));
    dag.add_root(sum_ax0);
    dag.add_root(sum_ax1);
    let result = codegen_hip(&dag, "test_diff_reductions");
    let src = &result.c_source;

    // These must be DIFFERENT kernels (different axis, different axis_size)
    assert!(
        src.contains("kernel_sum_ax0_sz3"),
        "Sum on axis 0 with axis_size=3 should have unique kernel name"
    );
    assert!(
        src.contains("kernel_sum_ax1_sz4"),
        "Sum on axis 1 with axis_size=4 should have unique kernel name"
    );
}

// ===========================================================================
// RT8: Load-as-output uses wrong input slot when Store nodes exist
// ===========================================================================

#[test]
fn rt8_load_as_output_with_store() {
    // A DAG where a Load is also a root, AND there's a Store node.
    // The Store is collected as output[0], then the Load-root as output[1].
    // The output emission for the Load currently does `outputs[slot] = inputs[slot]`
    // which would be `outputs[1] = inputs[1]` — but the Load might be inputs[0]!
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4));
    let c = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(4));
    let add = dag.add_node(RiscOp::Add, vec![x, c], vec_f32(4));
    let store = dag.add_node(
        RiscOp::Store {
            name: "computed".into(),
        },
        vec![add],
        vec_f32(4),
    );
    dag.add_root(store);
    dag.add_root(x); // Load is also a root
    let result = codegen_hip(&dag, "test_load_output");
    let src = &result.c_source;

    // output_specs: [Store("computed") at slot 0, Load("x") at slot 1]
    // input_labels: ["x"] -> input slot 0
    // The Load should map to inputs[0], NOT inputs[1]
    assert_eq!(result.input_labels, vec!["x"]);
    assert_eq!(result.output_labels, vec!["computed", "root1"]);

    // The generated code for the Load-as-output should reference inputs[0], not inputs[1]
    // Current buggy code would emit: outputs[1] = inputs[1]  (wrong!)
    // Correct code should emit: outputs[1] = inputs[0]
    assert!(
        !src.contains("outputs[1] = inputs[1]"),
        "BUG: Load-as-output must use the correct input slot, not the output slot. \
         Load 'x' is inputs[0] but appears as outputs[1]. Code incorrectly maps to inputs[1]."
    );
}

// ===========================================================================
// RT9: Expand then add verifies stride-0 in generated code
// ===========================================================================

#[test]
fn rt9_expand_sets_stride_zero() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(3));
    let e = dag.add_node(RiscOp::Expand { axis: 0, size: 4 }, vec![x], mat_f32(4, 3));
    dag.add_root(e);
    let result = codegen_hip(&dag, "test_expand_stride");
    let src = &result.c_source;

    // The expand must set stride[0] = 0
    assert!(
        src.contains("strides[0] = 0"),
        "Expand must explicitly set stride to 0 on the expanded axis"
    );
}

// ===========================================================================
// RT10: Reshape view does NOT copy strides (it inherits data pointer only)
// ===========================================================================

#[test]
fn rt10_reshape_view_correct() {
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
    let src = &result.c_source;

    // Reshape creates a view with the new shape
    assert!(src.contains("chelis_gpu_alloc_view"));
    // The view shares the data pointer
    assert!(
        src.contains("d_t0->data"),
        "Reshape view must share data pointer with input"
    );
}

// ===========================================================================
// RT11: Store cleanup — Store aliased node must not be double-freed
// ===========================================================================

#[test]
fn rt11_store_no_double_free() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(4));
    let y = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], vec_f32(4));
    let add = dag.add_node(RiscOp::Add, vec![x, y], vec_f32(4));
    let store = dag.add_node(RiscOp::Store { name: "out".into() }, vec![add], vec_f32(4));
    dag.add_root(store);
    let result = codegen_hip(&dag, "test_store_free");
    let src = &result.c_source;

    // Store (t3) aliases Add (t2): d_t3 = d_t2
    // Store is an output, so t3 is skipped in cleanup. Good.
    // But t2 (the Add) is NOT an output, so it WILL be freed.
    // This means d_t2 gets freed, but d_t3 (== d_t2) was already used for device_to_host.
    // This is OK because device_to_host happens before cleanup.
    // But it would be a problem if d_t3 is used after cleanup.

    // The device_to_host must happen BEFORE chelis_gpu_free(d_t2)
    let d2h_pos = src.find("chelis_device_to_host").expect("d2h present");
    let free_t2 = src.find("chelis_gpu_free(d_t2)").expect("t2 freed");
    assert!(
        d2h_pos < free_t2,
        "device_to_host must happen before freeing the aliased node"
    );
}

// ===========================================================================
// RT12: Cast op generates kernel
// ===========================================================================

#[test]
fn rt12_cast_emits_kernel() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(4));
    let c = dag.add_node(
        RiscOp::Cast {
            new_precision: Prim::F32,
        },
        vec![x],
        vec_f32(4),
    );
    dag.add_root(c);
    let result = codegen_hip(&dag, "test_cast");
    assert!(
        result.c_source.contains("kernel_cast"),
        "Cast must emit a kernel"
    );
}

// ===========================================================================
// RT13: grid_1d(0) -- zero-element tensor
// ===========================================================================

#[test]
fn rt13_zero_size_grid() {
    // This tests if grid_1d(0) panics or produces invalid grid config
    let (grid, block) = chelis_backend_hip::launch::grid_1d(0);
    assert_eq!(grid, 0, "Zero-element grid should have 0 blocks");
    assert_eq!(block, 256, "Block size unchanged for zero elements");
}
