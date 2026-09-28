//! Adversarial red-team tests for the HIP backend.
//!
//! These probe edge cases not covered by S1-S13 structural tests.

use chelis_backend_hip::HipCodegenResult;
mod support;
use chelis_ir::dag::{ComparisonKind, Dag, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;
use chelis_types::unsupported::{RejectionAuthorityKind, Stage, Unsupported, UnsupportedKind};
use support::codegen_hip;

fn scalar_f32() -> TensorType {
    TensorType::scalar_f32()
}

fn vec_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn vec_f64(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F64,
    }
}

fn vec_bool(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::Bool,
    }
}

fn vec_i64(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::Int64,
    }
}

fn assert_hip_dtype_rejection(error: Unsupported, dtype: &str, issue: u32) {
    assert_eq!(
        error.what.as_ref(),
        &UnsupportedKind::Dtype(dtype.to_string()),
        "the rejection must carry the exact typed dtype"
    );
    assert_eq!(
        error.stage,
        Stage::Codegen("hip"),
        "the rejection must identify the HIP codegen stage"
    );
    assert_eq!(
        error.authority.kind(),
        RejectionAuthorityKind::Unimplemented,
        "the target capability gap must remain typed as unimplemented"
    );
    assert_eq!(
        error.authority.issue().map(|issue| issue.number()),
        Some(issue),
        "the rejection must carry the exact implementation owner"
    );
}

fn expect_hip_codegen_rejection(
    result: Result<HipCodegenResult, Unsupported>,
    message: &str,
) -> Unsupported {
    match result {
        Err(error) => error,
        Ok(_) => panic!("{message}"),
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
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        mat_f32(3, 4),
        None,
    );
    let p = dag.add_node(
        decl,
        RiscOp::Permute { axes: vec![1, 0] },
        vec![x],
        mat_f32(4, 3),
        None,
    );
    let c = dag.add_node(
        decl,
        RiscOp::synth_const(mat_f32(4, 3).precision, 1.0),
        vec![],
        mat_f32(4, 3),
        None,
    );
    let a = dag.add_node(decl, RiscOp::Add, vec![p, c], mat_f32(4, 3), None);
    let s = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![a],
        vec_f32(4),
        None,
    );
    dag.add_root(s);
    let result = codegen_hip(&dag, "test_chain").unwrap();
    let src = &result.c_source;

    // Verify the full chain is emitted
    assert!(
        src.contains("chelis_device_tensor_copy_from_host(chelis_slot0, inputs[0])"),
        "Load must transfer into its owned device slot"
    );
    assert!(
        src.contains("chelis_metadata_plan *plan_t1 = chelis_metadata_plan_view")
            && src.contains(
                "chelis_device_tensor_owner *o_t1 = chelis_device_tensor_borrow(plan_t1, d_t0->data",
            ),
        "Permute must be a borrowed metadata view over the input owner"
    );
    assert!(src.contains("kernel_add"), "Add kernel present");
    assert!(src.contains("kernel_sum"), "Sum kernel present");
    // Verify permute stride reorder exists
    assert!(
        src.contains(
            "plan_t1_strides[2] = { chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)(d_t0->strides[1])),",
        ),
        "Permute must reorder strides: dim 0 gets old dim 1"
    );
    assert!(
        src.contains("chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)(d_t0->strides[0])) };",),
        "Permute must reorder strides: dim 1 gets old dim 0"
    );
    assert!(
        !src.contains("chelis_device_tensor_alloc(plan_t1)"),
        "Permute must not allocate independent backing storage"
    );
}

// ===========================================================================
// RT2: Two Loads and a Store
// ===========================================================================

#[test]
fn rt2_two_loads_store() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let y = dag.add_node(
        decl,
        RiscOp::Load { name: "y".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let add = dag.add_node(decl, RiscOp::Add, vec![x, y], vec_f32(4), None);
    let store = dag.add_node(
        decl,
        RiscOp::Store {
            name: "result".into(),
        },
        vec![add],
        vec_f32(4),
        None,
    );
    dag.add_root(store);
    let result = codegen_hip(&dag, "test_store").unwrap();
    let src = &result.c_source;

    // Store should alias its input through a metadata wrapper, not by allocating fresh storage.
    assert!(
        src.contains("chelis_metadata_plan *plan_t3 = chelis_metadata_plan_view")
            && src.contains(
                "chelis_device_tensor_owner *o_t3 = chelis_device_tensor_borrow(plan_t3, d_t2->data",
            )
            && !src.contains("slot_plan3"),
        "Store must borrow its input storage through a metadata owner, not allocate a new slot"
    );
    // Output labels should include the store name
    assert_eq!(result.output_labels, vec!["result"]);
    // Input labels should be x, y in order
    assert_eq!(result.input_labels, vec!["x", "y"]);
    // Repeated loads of different input labels still transfer once each.
    let h2d_count = src.matches("chelis_device_tensor_copy_from_host").count();
    assert_eq!(
        h2d_count, 2,
        "Two loads should produce two host-to-device transfers"
    );
}

// ===========================================================================
// RT3: Scalar-only DAG (all ndim=0 tensors, treated as ndim=1 size=1)
// ===========================================================================

#[test]
fn rt3_scalar_only_dag() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::synth_const(scalar_f32().precision, 3.125),
        vec![],
        scalar_f32(),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::synth_const(scalar_f32().precision, 2.0),
        vec![],
        scalar_f32(),
        None,
    );
    let c = dag.add_node(decl, RiscOp::Mul, vec![a, b], scalar_f32(), None);
    dag.add_root(c);
    let result = codegen_hip(&dag, "test_scalar").unwrap();
    let src = &result.c_source;

    // Scalars retain rank-0 metadata while the owner computes a logical count of one.
    assert!(
        src.contains(
            "chelis_metadata_plan *slot_plan0 = chelis_metadata_plan_new(chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)(0)), NULL,",
        ) && src.contains("chelis_device_metadata fill_size = d_t0->count;"),
        "Scalar storage must use rank-0 metadata and launch over its runtime count"
    );
    assert!(
        !src.contains("slot_plan0_shape"),
        "A rank-0 scalar must not acquire a synthetic shape array"
    );
    // Grid should be derived from the runtime size even for scalar tensors.
    assert!(
        src.contains(
            "chelis_launch_kernel(mod_kernel_fill_f32, \"kernel_fill_f32\", (fill_size / 256 + (fill_size % 256 != 0)), (256), fill_args);",
        ),
        "Scalar kernel launch should use the runtime size expression"
    );
}

// ===========================================================================
// RT4: Fan-out (same const feeds two ops)
// ===========================================================================

#[test]
fn rt4_fanout_same_input_two_ops() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::synth_const(vec_f32(4).precision, 2.0),
        vec![],
        vec_f32(4),
        None,
    );
    let a = dag.add_node(decl, RiscOp::Add, vec![x, x], vec_f32(4), None);
    let b = dag.add_node(decl, RiscOp::Mul, vec![x, a], vec_f32(4), None);
    dag.add_root(b);
    let result = codegen_hip(&dag, "test_fanout").unwrap();
    let src = &result.c_source;

    // x and the add intermediate should have their wrappers freed; backing slots free at end.
    assert!(
        src.contains("chelis_device_tensor_release(o_t0)"),
        "t0 wrapper should be freed"
    );
    assert!(
        src.contains("chelis_device_tensor_release(o_t1)"),
        "t1 wrapper should be freed"
    );
    assert!(
        !src.contains("chelis_device_tensor_release(d_t0)")
            && !src.contains("chelis_device_tensor_release(d_t1)"),
        "Observed tensor views are non-owning and must never be released as owners"
    );
    // t0 should not be freed before it's used by both ops
    // Verify topological order: t0 defined before t1 (add) and t2 (mul)
    let t0_def = src
        .find("o_t0 = chelis_device_tensor_borrow")
        .expect("t0 owner wrapper");
    // Look for the actual kernel launch calls (not the static module cache)
    let t1_alloc = src
        .find("o_t1 = chelis_device_tensor_borrow")
        .expect("t1 owner wrapper (add output)");
    let t2_alloc = src
        .find("o_t2 = chelis_device_tensor_borrow")
        .expect("t2 owner wrapper (mul output)");
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
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(6),
        None,
    );
    let _s = dag.add_node(
        decl,
        RiscOp::Stride {
            strides: vec![chelis_ir::dag::RtDim::Lit(2)],
        },
        vec![x],
        vec_f32(3),
        None,
    );
    let result = codegen_hip(&dag, "test_stride").unwrap();
    let src = &result.c_source;

    // The stride op MUST multiply strides by the stride factors.
    // If it just copies strides, every-other-element access is broken.
    assert!(
        src.contains(
            "chelis_int_checked_mul(d_t0->strides[0], INT64_C(2), 64, \"numeric trap: overflow in stride at i64\")",
        ) && src.contains("chelis_metadata_plan *plan_t1 = chelis_metadata_plan_view"),
        "Stride op must checked-multiply the borrowed view stride by factor 2:\n{src}"
    );
    assert!(
        !src.contains(
            "plan_t1_strides[1] = { chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)(d_t0->strides[0])) };",
        ),
        "Stride op must not silently copy the source stride unchanged"
    );
}

// ===========================================================================
// RT6: Multiple Store nodes produce correct output slots
// ===========================================================================

#[test]
fn rt6_multiple_stores() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::synth_const(vec_f32(4).precision, 1.0),
        vec![],
        vec_f32(4),
        None,
    );
    let y = dag.add_node(
        decl,
        RiscOp::synth_const(vec_f32(4).precision, 2.0),
        vec![],
        vec_f32(4),
        None,
    );
    let add = dag.add_node(decl, RiscOp::Add, vec![x, y], vec_f32(4), None);
    let store_a = dag.add_node(
        decl,
        RiscOp::Store {
            name: "out_a".into(),
        },
        vec![add],
        vec_f32(4),
        None,
    );
    let mul = dag.add_node(decl, RiscOp::Mul, vec![x, y], vec_f32(4), None);
    let store_b = dag.add_node(
        decl,
        RiscOp::Store {
            name: "out_b".into(),
        },
        vec![mul],
        vec_f32(4),
        None,
    );
    dag.add_root(store_a);
    dag.add_root(store_b);
    let result = codegen_hip(&dag, "test_multi_store").unwrap();

    assert_eq!(result.output_labels, vec!["out_a", "out_b"]);
    let src = &result.c_source;
    // Both stores should produce device-to-host transfers.
    let d2h_count = src.matches("chelis_device_tensor_copy_to_host").count();
    assert_eq!(
        d2h_count, 2,
        "Two stores should produce two device-to-host transfers"
    );
}

// ===========================================================================
// RT7: Reduction with different axis/size gets different kernel name
// ===========================================================================

#[test]
fn rt7_different_reductions_different_kernels() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::synth_const(mat_f32(3, 4).precision, 1.0),
        vec![],
        mat_f32(3, 4),
        None,
    );
    let sum_ax0 = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![x],
        vec_f32(4),
        None,
    );
    let sum_ax1 = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![x],
        vec_f32(3),
        None,
    );
    dag.add_root(sum_ax0);
    dag.add_root(sum_ax1);
    let result = codegen_hip(&dag, "test_diff_reductions").unwrap();
    let src = &result.c_source;

    // These must be DIFFERENT kernels (different reduction axis).
    assert!(
        src.contains("kernel_sum_ax0"),
        "Sum on axis 0 should have a unique kernel name"
    );
    assert!(
        src.contains("kernel_sum_ax1"),
        "Sum on axis 1 should have a unique kernel name"
    );
}

// ===========================================================================
// RT8: Load-as-output uses wrong input slot and ownership when Store nodes exist
// ===========================================================================

#[test]
fn rt8_load_as_output_with_store() {
    // A DAG where a Load is also a root, AND there's a Store node.
    // The Store is collected as output[0], then the Load-root as output[1].
    // The load-root output must use the correct input slot and must be an owned
    // tensor because host callers free every output slot.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let c = dag.add_node(
        decl,
        RiscOp::synth_const(vec_f32(4).precision, 1.0),
        vec![],
        vec_f32(4),
        None,
    );
    let add = dag.add_node(decl, RiscOp::Add, vec![x, c], vec_f32(4), None);
    let store = dag.add_node(
        decl,
        RiscOp::Store {
            name: "computed".into(),
        },
        vec![add],
        vec_f32(4),
        None,
    );
    dag.add_root(store);
    dag.add_root(x); // Load is also a root
    let result = codegen_hip(&dag, "test_load_output").unwrap();
    let src = &result.c_source;

    // output_specs: [Store("computed") at slot 0, Load("x") at slot 1]
    // input_labels: ["x"] -> input slot 0
    // The Load should map to inputs[0], NOT inputs[1]
    assert_eq!(result.input_labels, vec!["x"]);
    assert_eq!(result.output_labels, vec!["computed", "root1"]);

    // The generated code for the Load-as-output should reference inputs[0], not inputs[1],
    // and it should clone to an owned contiguous host tensor rather than aliasing input storage.
    assert!(
        !src.contains("outputs[1] = inputs[1]"),
        "BUG: Load-as-output must use the correct input slot, not the output slot. \
         Load 'x' is inputs[0] but appears as outputs[1]. Code incorrectly maps to inputs[1]."
    );
    assert!(
        src.contains("outputs[1] = chelis_contiguous(inputs[0]);"),
        "Load-as-output must return an owned host tensor so callers can free outputs independently"
    );
}

// ===========================================================================
// RT9: Expand then add verifies stride-0 in generated code
// ===========================================================================

#[test]
fn rt9_expand_sets_stride_zero() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::synth_const(vec_f32(3).precision, 1.0),
        vec![],
        vec_f32(3),
        None,
    );
    let e = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::Lit(4),
        },
        vec![x],
        mat_f32(4, 3),
        None,
    );
    dag.add_root(e);
    let result = codegen_hip(&dag, "test_expand_stride").unwrap();
    let src = &result.c_source;

    // The expand must encode stride[0] = 0 in a borrowed metadata view.
    assert!(
        src.contains(
            "plan_t1_strides[2] = { chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)(0)),",
        ) && src.contains("chelis_metadata_plan *plan_t1 = chelis_metadata_plan_view")
            && src.contains(
                "chelis_device_tensor_owner *o_t1 = chelis_device_tensor_borrow(plan_t1, d_t0->data",
            ),
        "Expand must encode a zero stride on the expanded axis while borrowing input storage"
    );
    assert!(
        !src.contains("slot_plan1"),
        "Expand is a view and must not allocate independent backing storage"
    );
}

// ===========================================================================
// RT10: Reshape materializes with checked source and destination metadata
// ===========================================================================

#[test]
fn rt10_reshape_view_correct() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::synth_const(mat_f32(2, 3).precision, 1.0),
        vec![],
        mat_f32(2, 3),
        None,
    );
    let r = dag.add_node(
        decl,
        RiscOp::Reshape {
            new_shape: vec![chelis_ir::dag::RtDim::Lit(6)],
        },
        vec![x],
        vec_f32(6),
        None,
    );
    dag.add_root(r);
    let result = codegen_hip(&dag, "test_reshape").unwrap();
    let src = &result.c_source;

    // Reshape validates equal element counts before allocating and launching a logical copy.
    assert!(
        src.contains(
            "if (chelis_metadata_plan_count(reshape_input1) != chelis_metadata_plan_count(reshape_output1))",
        ) && src.contains(
            "if (d_t1->count != d_t0->count) chelis_numeric_trap(\"numeric trap: domain in materialize at i64\");",
        ),
        "Reshape must validate source and destination element counts"
    );
    assert!(
        src.contains("chelis_device_tensor_alloc(slot_plan1)")
            && src.contains("kernel_reshape_CHELIS_DTYPE_F32"),
        "Reshape must materialize the new layout into an independently owned output slot"
    );
    assert!(
        !src.contains(
            "chelis_device_tensor_owner *o_t1 = chelis_device_tensor_borrow(plan_t1, d_t0->data",
        ) && !src.contains("chelis_metadata_plan *plan_t1 = chelis_metadata_plan_view"),
        "Reshape materialization must not alias the input data or preserve its old strides"
    );
}

// ===========================================================================
// RT11: Store cleanup — Store aliased node must not be double-freed
// ===========================================================================

#[test]
fn rt11_store_no_double_free() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::synth_const(vec_f32(4).precision, 1.0),
        vec![],
        vec_f32(4),
        None,
    );
    let y = dag.add_node(
        decl,
        RiscOp::synth_const(vec_f32(4).precision, 2.0),
        vec![],
        vec_f32(4),
        None,
    );
    let add = dag.add_node(decl, RiscOp::Add, vec![x, y], vec_f32(4), None);
    let store = dag.add_node(
        decl,
        RiscOp::Store { name: "out".into() },
        vec![add],
        vec_f32(4),
        None,
    );
    dag.add_root(store);
    let result = codegen_hip(&dag, "test_store_free").unwrap();
    let src = &result.c_source;

    // The store aliases the add result through a metadata wrapper, so the output copy must
    // happen before the backing slot is released.
    let d2h_pos = src
        .find("chelis_device_tensor_copy_to_host(output_guard_0, o_t3)")
        .expect("device-to-host copy present");
    let free_alias = src
        .find("chelis_device_tensor_release(o_t3)")
        .expect("store alias owner released");
    let free_slot = src
        .find("chelis_device_tensor_release(chelis_slot2)")
        .expect("aliased slot freed");
    assert!(
        d2h_pos < free_alias && free_alias < free_slot,
        "device-to-host copy must precede releasing the store alias and its backing slot"
    );
    assert_eq!(
        src.matches("chelis_device_tensor_release(o_t3)").count(),
        2,
        "the host and device entrypoints must each release their store alias exactly once"
    );
    assert!(
        !src.contains("chelis_device_tensor_release(d_t3)"),
        "the non-owning observed store view must never be released directly"
    );
}

// ===========================================================================
// RT12: Cast op generates kernel
// ===========================================================================

#[test]
fn rt12_cast_emits_kernel() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::synth_const(vec_f32(4).precision, 1.0),
        vec![],
        vec_f32(4),
        None,
    );
    let c = dag.add_node(
        decl,
        RiscOp::Cast {
            new_precision: Prim::F64,
        },
        vec![x],
        vec_f64(4),
        None,
    );
    dag.add_root(c);
    let result = codegen_hip(&dag, "test_cast").unwrap();
    assert!(
        result.c_source.contains("kernel_cast"),
        "Cast must emit a kernel"
    );
}

// ===========================================================================
// RT12b: Cast to bool rejects rather than emitting a four-byte kernel
// ===========================================================================

/// chelis#1360. This case used to be RT12 itself, asserting only that
/// `kernel_cast` appeared in the output. It did appear - as `kernel_cast_f32`,
/// writing `N * 4` bytes into the `N * 1` byte allocation that chelis#1308's
/// tagged carrier now sizes for `CHELIS_DTYPE_BOOL`. The assertion was true
/// and the emitted program overran its device heap by `3N` bytes, so the test
/// now pins the rejection instead of the kernel name.
#[test]
fn rt12b_cast_to_bool_is_rejected_not_emitted_as_f32() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::synth_const(vec_f32(4).precision, 1.0),
        vec![],
        vec_f32(4),
        None,
    );
    let c = dag.add_node(
        decl,
        RiscOp::Cast {
            new_precision: Prim::Bool,
        },
        vec![x],
        vec_bool(4),
        None,
    );
    dag.add_root(c);
    let error = match codegen_hip(&dag, "test_cast_bool") {
        Err(error) => error,
        Ok(_) => panic!("a bool result has no HIP kernel family (chelis#1364)"),
    };
    assert_hip_dtype_rejection(error, "bool", 1364);
}

/// chelis#1364 owns the reverse direction too: a real Bool8 family must read
/// one-byte inputs and produce the requested destination representation.
#[test]
fn rt12b_cast_from_bool_carries_the_bool_family_authority() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_bool(4),
        None,
    );
    let c = dag.add_node(
        decl,
        RiscOp::Cast {
            new_precision: Prim::F32,
        },
        vec![x],
        vec_f32(4),
        None,
    );
    dag.add_root(c);
    let error = expect_hip_codegen_rejection(
        codegen_hip(&dag, "test_cast_from_bool"),
        "cast from bool requires the chelis#1364 HIP Bool8 family",
    );
    assert_hip_dtype_rejection(error, "bool", 1364);
}

/// Copy materializes its input through the cast template, so Bool8 support is
/// part of the same chelis#1364 family rather than the generic chelis#689
/// fallback class.
#[test]
fn rt12b_copy_bool_carries_the_bool_family_authority() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_bool(4),
        None,
    );
    let copy = dag.add_node(decl, RiscOp::Copy, vec![x], vec_bool(4), None);
    dag.add_root(copy);
    let error = expect_hip_codegen_rejection(
        codegen_hip(&dag, "test_copy_bool"),
        "copying bool requires the chelis#1364 HIP Bool8 family",
    );
    assert_hip_dtype_rejection(error, "bool", 1364);
}

/// Realize also materializes through the cast template and therefore has the
/// same exact Bool8 capability owner as casts and copies.
#[test]
fn rt12b_realize_bool_carries_the_bool_family_authority() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_bool(4),
        None,
    );
    let owned = dag.add_node(decl, RiscOp::Copy, vec![x], vec_bool(4), None);
    let realize = dag.add_node(decl, RiscOp::Realize, vec![owned], vec_bool(4), None);
    dag.add_root(realize);
    let error = expect_hip_codegen_rejection(
        codegen_hip(&dag, "test_realize_bool"),
        "realizing bool requires the chelis#1364 HIP Bool8 family",
    );
    assert_hip_dtype_rejection(error, "bool", 1364);
}

/// The chelis#1364 authority is operation-aware, not a blanket replacement
/// for every bool rejection. A generic float-family op still belongs to the
/// chelis#689 no-typed-kernel fallback class.
#[test]
fn rt12b_unrelated_bool_numeric_op_retains_generic_authority() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_bool(4),
        None,
    );
    let neg = dag.add_node(decl, RiscOp::Neg, vec![x], vec_bool(4), None);
    dag.add_root(neg);
    let error = expect_hip_codegen_rejection(
        codegen_hip(&dag, "test_neg_bool"),
        "bool negation has no generic HIP arithmetic family",
    );
    assert_hip_dtype_rejection(error, "bool", 689);
}

/// The operation-aware bool path must not disturb the non-bool fallback that
/// chelis#689 actually owns.
#[test]
fn rt12b_non_bool_materialization_retains_generic_authority() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_i64(4),
        None,
    );
    let owned = dag.add_node(decl, RiscOp::Copy, vec![x], vec_i64(4), None);
    let realize = dag.add_node(decl, RiscOp::Realize, vec![owned], vec_i64(4), None);
    dag.add_root(realize);
    let error = expect_hip_codegen_rejection(
        codegen_hip(&dag, "test_realize_i64"),
        "i64 realize has no generic HIP arithmetic family",
    );
    assert_hip_dtype_rejection(error, "i64", 689);
}

/// chelis#1360 companion: direct `Compare(CmpLt)` uses one-byte Bool storage.
#[test]
fn rt12c_cmplt_to_bool_is_not_emitted_at_operand_width() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::synth_const(vec_f32(4).precision, 1.0),
        vec![],
        vec_f32(4),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::synth_const(vec_f32(4).precision, 2.0),
        vec![],
        vec_f32(4),
        None,
    );
    let c = dag.add_node(
        decl,
        RiscOp::Compare(ComparisonKind::CmpLt),
        vec![a, b],
        vec_bool(4),
        None,
    );
    dag.add_root(c);
    let source = codegen_hip(&dag, "test_cmplt_bool").unwrap().c_source;
    assert!(source.contains("unsigned char *out"), "{source}");
    assert!(!source.contains("float *out"), "{source}");
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
// RT14: scalar reductions use the generic runtime-sized kernel path
// ===========================================================================

#[test]
fn rt14_staged_scalar_reduction_allocates_inline_scratch() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::synth_const(vec_f32(1024).precision, 1.0),
        vec![],
        vec_f32(1024),
        None,
    );
    let sum = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![x],
        scalar_f32(),
        None,
    );
    dag.add_root(sum);
    let result = codegen_hip(&dag, "test_stage_scratch").unwrap();
    let src = &result.c_source;

    assert!(src.contains("kernel_sum_ax0"));
    assert!(!src.contains("hipMalloc(&t1_partials0"));
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
    let contiguous_decl = contiguous.declare("test");
    let a = contiguous.add_node(
        contiguous_decl,
        RiscOp::synth_const(mat_f32(2, 3).precision, 1.0),
        vec![],
        mat_f32(2, 3),
        None,
    );
    let b = contiguous.add_node(
        contiguous_decl,
        RiscOp::synth_const(mat_f32(3, 4).precision, 1.0),
        vec![],
        mat_f32(3, 4),
        None,
    );
    let ea = contiguous.add_node(
        contiguous_decl,
        RiscOp::Expand {
            axis: 2,
            size: chelis_ir::dag::RtDim::Lit(4),
        },
        vec![a],
        tensor3_f32(2, 3, 4),
        None,
    );
    let eb = contiguous.add_node(
        contiguous_decl,
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::Lit(2),
        },
        vec![b],
        tensor3_f32(2, 3, 4),
        None,
    );
    let mul = contiguous.add_node(
        contiguous_decl,
        RiscOp::Mul,
        vec![ea, eb],
        tensor3_f32(2, 3, 4),
        None,
    );
    let sum = contiguous.add_node(
        contiguous_decl,
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![mul],
        mat_f32(2, 4),
        None,
    );
    contiguous.add_root(sum);
    let contiguous_src = codegen_hip(&contiguous, "test_matmul_contig")
        .unwrap()
        .c_source;
    assert!(
        contiguous_src.contains("chelis_hipblas_sgemm_row_major"),
        "contiguous matmul should specialize to hipBLAS"
    );

    let mut fallback = Dag::new();
    let fallback_decl = fallback.declare("test");
    let base_a = fallback.add_node(
        fallback_decl,
        RiscOp::synth_const(mat_f32(3, 2).precision, 1.0),
        vec![],
        mat_f32(3, 2),
        None,
    );
    let a_perm = fallback.add_node(
        fallback_decl,
        RiscOp::Permute { axes: vec![1, 0] },
        vec![base_a],
        mat_f32(2, 3),
        None,
    );
    let b = fallback.add_node(
        fallback_decl,
        RiscOp::synth_const(mat_f32(3, 4).precision, 1.0),
        vec![],
        mat_f32(3, 4),
        None,
    );
    let ea = fallback.add_node(
        fallback_decl,
        RiscOp::Expand {
            axis: 2,
            size: chelis_ir::dag::RtDim::Lit(4),
        },
        vec![a_perm],
        tensor3_f32(2, 3, 4),
        None,
    );
    let eb = fallback.add_node(
        fallback_decl,
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::Lit(2),
        },
        vec![b],
        tensor3_f32(2, 3, 4),
        None,
    );
    let mul = fallback.add_node(
        fallback_decl,
        RiscOp::Mul,
        vec![ea, eb],
        tensor3_f32(2, 3, 4),
        None,
    );
    let sum = fallback.add_node(
        fallback_decl,
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![mul],
        mat_f32(2, 4),
        None,
    );
    fallback.add_root(sum);
    let fallback_src = codegen_hip(&fallback, "test_matmul_fallback")
        .unwrap()
        .c_source;
    assert!(
        !fallback_src.contains("chelis_hipblas_sgemm_row_major"),
        "non-contiguous matmul must stay on the generic reduction path"
    );
}
