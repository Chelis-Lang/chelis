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

fn tensor3_f32(a: usize, b: usize, c: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(a), DimInfo::Lit(b), DimInfo::Lit(c)],
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

    // Store should alias its input through a metadata wrapper, not by allocating fresh storage.
    assert!(
        src.contains("chelis_gpu_alloc_view")
            && src.contains("d_t2->data")
            && src.contains("d_t2->storage_size"),
        "Store must alias its input rather than owning a new slot"
    );
    // Output labels should include the store name
    assert_eq!(result.output_labels, vec!["result"]);
    // Input labels should be x, y in order
    assert_eq!(result.input_labels, vec!["x", "y"]);
    // Repeated loads of different input labels still transfer once each.
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

    // x and the add intermediate should have their wrappers freed; backing slots free at end.
    assert!(
        src.contains("chelis_gpu_free_view(d_t0)"),
        "t0 wrapper should be freed"
    );
    assert!(
        src.contains("chelis_gpu_free_view(d_t1)"),
        "t1 wrapper should be freed"
    );
    // t0 should not be freed before it's used by both ops
    // Verify topological order: t0 defined before t1 (add) and t2 (mul)
    let t0_def = src
        .find("d_t0 = chelis_gpu_alloc_view")
        .expect("t0 wrapper alloc");
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

    // The store aliases the add result through a metadata wrapper, so the output copy must
    // happen before the backing slot is released.
    let d2h_pos = src.find("chelis_device_to_host").expect("d2h present");
    let free_slot = src
        .find("chelis_gpu_free(chelis_slot2)")
        .expect("aliased slot freed");
    assert!(
        d2h_pos < free_slot,
        "device_to_host must happen before freeing the aliased backing slot"
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

// ===========================================================================
// RT14: scalar staged reduction scratch stays inline, outside slot planner
// ===========================================================================

#[test]
fn rt14_staged_scalar_reduction_allocates_inline_scratch() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(1024));
    let sum = dag.add_node(RiscOp::Sum { axis: 0 }, vec![x], scalar_f32());
    dag.add_root(sum);
    let result = codegen_hip(&dag, "test_stage_scratch");
    let src = &result.c_source;

    assert!(
        src.contains("hipMalloc(&t1_partials0"),
        "staged scalar reductions should allocate scratch inline"
    );
    assert!(
        src.contains("hipFree(t1_partials0)"),
        "inline staged scratch must be freed in the same emission block"
    );
    assert!(
        !src.contains("chelis_slot2"),
        "staged scratch must not be routed through the slot planner"
    );
}

// ===========================================================================
// RT15: contiguous matmul specializes, non-contiguous matmul does not
// ===========================================================================

#[test]
fn rt15_matmul_specialization_respects_contiguity() {
    let mut contiguous = Dag::new();
    let a = contiguous.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(2, 3));
    let b = contiguous.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(3, 4));
    let ea = contiguous.add_node(
        RiscOp::Expand { axis: 2, size: 4 },
        vec![a],
        tensor3_f32(2, 3, 4),
    );
    let eb = contiguous.add_node(
        RiscOp::Expand { axis: 0, size: 2 },
        vec![b],
        tensor3_f32(2, 3, 4),
    );
    let mul = contiguous.add_node(RiscOp::Mul, vec![ea, eb], tensor3_f32(2, 3, 4));
    let sum = contiguous.add_node(RiscOp::Sum { axis: 1 }, vec![mul], mat_f32(2, 4));
    contiguous.add_root(sum);
    let contiguous_src = codegen_hip(&contiguous, "test_matmul_contig").c_source;
    assert!(
        contiguous_src.contains("chelis_hipblas_sgemm_row_major"),
        "contiguous matmul should specialize to hipBLAS"
    );

    let mut fallback = Dag::new();
    let base_a = fallback.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(3, 2));
    let a_perm = fallback.add_node(
        RiscOp::Permute { axes: vec![1, 0] },
        vec![base_a],
        mat_f32(2, 3),
    );
    let b = fallback.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(3, 4));
    let ea = fallback.add_node(
        RiscOp::Expand { axis: 2, size: 4 },
        vec![a_perm],
        tensor3_f32(2, 3, 4),
    );
    let eb = fallback.add_node(
        RiscOp::Expand { axis: 0, size: 2 },
        vec![b],
        tensor3_f32(2, 3, 4),
    );
    let mul = fallback.add_node(RiscOp::Mul, vec![ea, eb], tensor3_f32(2, 3, 4));
    let sum = fallback.add_node(RiscOp::Sum { axis: 1 }, vec![mul], mat_f32(2, 4));
    fallback.add_root(sum);
    let fallback_src = codegen_hip(&fallback, "test_matmul_fallback").c_source;
    assert!(
        !fallback_src.contains("chelis_hipblas_sgemm_row_major"),
        "non-contiguous matmul must stay on the generic reduction path"
    );
}
