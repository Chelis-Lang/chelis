//! Primitive matmul preserves its dtype and canonical reduction tree in C.
//! Enabling explicit BLAS-node emission does not authorize replacing a
//! primitive contraction with a library-dependent arithmetic graph, and
//! computing the product inside the sum (chelis#3370) stores no rank-3
//! intermediate.

use chelis_backend_c::CodegenOptions;
mod support;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;
use support::codegen_with_options;

fn t(prim: Prim, dims: Vec<usize>) -> TensorType {
    TensorType {
        dims: dims.into_iter().map(DimInfo::Lit).collect(),
        precision: prim,
    }
}

/// Regression lock: end-to-end F64 matmul must NOT emit cblas_sgemm.
#[test]
fn f64_matmul_subgraph_stays_off_blas_path_in_c_backend() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        t(Prim::F64, vec![8, 16]),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        t(Prim::F64, vec![16, 4]),
        None,
    );
    let ea = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 2,
            size: chelis_ir::dag::RtDim::Lit(4),
        },
        vec![a],
        t(Prim::F64, vec![8, 16, 4]),
        None,
    );
    let eb = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::Lit(8),
        },
        vec![b],
        t(Prim::F64, vec![8, 16, 4]),
        None,
    );
    let mul = dag.add_node(
        decl,
        RiscOp::Mul,
        vec![ea, eb],
        t(Prim::F64, vec![8, 16, 4]),
        None,
    );
    let sum = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 1,
            accumulator: Prim::F64,
        },
        vec![mul],
        t(Prim::F64, vec![8, 4]),
        None,
    );
    dag.add_root(sum);

    // Exercise the option used by C entry points, including its negative
    // guarantee: primitive arithmetic does not acquire a vendor summary.
    let result = codegen_with_options(
        &dag,
        "p0_f64_matmul_miscompile",
        CodegenOptions {
            use_blas: true,
            ..CodegenOptions::default()
        },
    )
    .unwrap();

    assert!(!result.c_source.contains("cblas_sgemm("));
    assert!(!result.c_source.contains("cblas_dgemm("));
    assert!(result.c_source.contains("__sum_level_"));

    // F64 slot allocation should still be present (the data is F64;
    // the generic expand+mul+sum path computes against it).
    assert!(
        result.c_source.contains("CHELIS_DTYPE_F64"),
        "F64 slot allocation must be present on the generic path; got source:\n{}",
        result.c_source
    );

    // With BLAS specialization skipped, the BLAS link requirement must
    // NOT be set — the generic path doesn't need -lopenblas.
    assert!(
        !result.requirements.needs_blas,
        "F64 matmul on the generic path must not surface needs_blas; \
         got requirements = {:?}",
        result.requirements
    );
}

/// F32 has the same exact-arithmetic obligation as F64.
#[test]
fn f32_matmul_subgraph_preserves_canonical_arithmetic_in_c_backend() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        t(Prim::F32, vec![8, 16]),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        t(Prim::F32, vec![16, 4]),
        None,
    );
    let ea = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 2,
            size: chelis_ir::dag::RtDim::Lit(4),
        },
        vec![a],
        t(Prim::F32, vec![8, 16, 4]),
        None,
    );
    let eb = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::Lit(8),
        },
        vec![b],
        t(Prim::F32, vec![8, 16, 4]),
        None,
    );
    let mul = dag.add_node(
        decl,
        RiscOp::Mul,
        vec![ea, eb],
        t(Prim::F32, vec![8, 16, 4]),
        None,
    );
    let sum = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 1,
            accumulator: Prim::F32,
        },
        vec![mul],
        t(Prim::F32, vec![8, 4]),
        None,
    );
    dag.add_root(sum);

    let result = codegen_with_options(
        &dag,
        "f32_matmul_canonical",
        CodegenOptions {
            use_blas: true,
            ..CodegenOptions::default()
        },
    )
    .unwrap();

    assert!(!result.c_source.contains("cblas_sgemm("));
    assert!(!result.requirements.needs_blas);
    assert!(result.c_source.contains("CHELIS_DTYPE_F32"));
    assert!(result.c_source.contains("__sum_level_"));
}

/// spec/05 section 4.1's `[m,k] x [k,n]` graph; `observe_product` also
/// returns the product, so nothing may skip storing it.
fn matmul_dag(prim: Prim, [m, k, n]: [usize; 3], observe_product: bool) -> Dag {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        t(prim, vec![m, k]),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        t(prim, vec![k, n]),
        None,
    );
    let expand = |axis, size| RiscOp::Expand {
        axis,
        size: chelis_ir::dag::RtDim::Lit(size),
    };
    let ea = dag.add_node(decl, expand(2, n), vec![a], t(prim, vec![m, k, n]), None);
    let eb = dag.add_node(decl, expand(0, m), vec![b], t(prim, vec![m, k, n]), None);
    let mul = dag.add_node(
        decl,
        RiscOp::Mul,
        vec![ea, eb],
        t(prim, vec![m, k, n]),
        None,
    );
    let sum = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 1,
            accumulator: prim,
        },
        vec![mul],
        t(prim, vec![m, n]),
        None,
    );
    dag.add_root(sum);
    if observe_product {
        dag.add_root(mul);
    }
    dag
}

/// chelis#3370: the issue's 256x2304 by 2304x196 matmul stores neither the
/// expanded operands nor their 256x2304x196 product. The sum reads both
/// operands through the expansions' checked plans and folds the canonical
/// tree; only the result is allocated.
#[test]
fn matmul_stores_no_rank_three_intermediate() {
    for prim in [Prim::F32, Prim::F64] {
        let dag = matmul_dag(prim, [256, 2304, 196], false);
        let c = codegen_with_options(&dag, "issue_3370", CodegenOptions::default())
            .unwrap()
            .c_source;
        assert!(!c.contains("chelis_alloc(3,"), "{prim:?}:\n{c}");
        assert_eq!(c.matches("chelis_tensor_expand_plan(").count(), 2, "{c}");
        assert_eq!(c.matches("chelis_movement_check_target(").count(), 2, "{c}");
        assert_eq!(c.matches("chelis_movement_plan_release(").count(), 2, "{c}");
        assert!(c.contains("chelis_shape_reduction_plan("), "{c}");
        assert!(c.contains("__sum_level_"), "{c}");
    }
}

/// Negative control: a product something else also reads is stored, and its
/// sum reads the stored product.
#[test]
fn an_observed_matmul_product_is_still_stored() {
    let dag = matmul_dag(Prim::F32, [4, 6, 3], true);
    let c = codegen_with_options(&dag, "issue_3370_observed", CodegenOptions::default())
        .unwrap()
        .c_source;
    assert_eq!(c.matches("chelis_alloc(3,").count(), 3, "{c}");
    assert!(!c.contains("chelis_shape_reduction_plan("), "{c}");
}
