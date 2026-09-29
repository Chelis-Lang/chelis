//! Test 2 — Brittle Pattern Matcher
//!
//! Probes whether the IR-level specialization pass survives a "useless"
//! intermediate cast. The test framing comes from a
//! third-party review: if a user inserts a no-op cast (e.g. `cast(.., f32)`)
//! between the canonical lowering steps, does the pattern-match still fire,
//! or does the program degrade silently to the scalar fallback?
//!
//! There are two relevant facts to capture in this file:
//!
//! 1. The shared IR specialization pass now has a scoped dense-gather
//!    recognizer. It matches the internal `OneHot + Expand + Mul + Sum` tree
//!    and rewrites it to first-class `RiscOp::Gather` before DCE/codegen.
//!    Arbitrary historical const/eq one-hot encodings are not covered because
//!    they no longer preserve the original index operand.
//!
//! 2. The raw backend detector still keys off `Sum -> Mul -> (Expand, Expand)`,
//!    but M1 added `chelis_ir::specialize`, which runs closed-list no-op
//!    cleanup first. The user-facing contract is that identity `Cast(f32)`
//!    no longer hides a matmul from BLAS specialization.

use chelis_backend_c::blas::detect_matmul_pattern as detect_verified_matmul_pattern;
use chelis_ir::dag::{Dag, DimExpr, DimInfo, RiscOp, TensorType};
use chelis_ir::specialize::specialize_for_blas;
use chelis_types::types::Prim;
mod support;

fn detect_matmul_pattern(
    dag: &Dag,
    sum: chelis_ir::dag::NodeId,
) -> Option<chelis_backend_c::blas::MatmulInfo> {
    let verified = support::verified_dag(dag, chelis_backend_c::CodegenOptions::default());
    detect_verified_matmul_pattern(verified.emission(), sum)
}

fn mat(r: usize, c: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(r), DimInfo::Lit(c)],
        precision: Prim::F32,
    }
}

fn t3(a: usize, b: usize, c: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(a), DimInfo::Lit(b), DimInfo::Lit(c)],
        precision: Prim::F32,
    }
}

fn vec_i64(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::Int64,
    }
}

#[test]
fn canonical_matmul_pattern_is_detected() {
    // Reproduce the canonical Tier 2 matmul lowering:
    //   A: [m=2, k=3], B: [k=3, n=4]
    //   ea = expand(A, axis=2, size=4)  -> [2, 3, 4]
    //   eb = expand(B, axis=0, size=2)  -> [2, 3, 4]
    //   p  = mul(ea, eb)                 -> [2, 3, 4]
    //   c  = sum(p, axis=1)              -> [2, 4]
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::synth_const(mat(2, 3).precision, 1.0),
        vec![],
        mat(2, 3),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::synth_const(mat(3, 4).precision, 1.0),
        vec![],
        mat(3, 4),
        None,
    );
    let ea = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 2,
            size: chelis_ir::dag::RtDim::Lit(4),
        },
        vec![a],
        t3(2, 3, 4),
        None,
    );
    let eb = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::Lit(2),
        },
        vec![b],
        t3(2, 3, 4),
        None,
    );
    let mul = dag.add_node(decl, RiscOp::Mul, vec![ea, eb], t3(2, 3, 4), None);
    let sum = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![mul],
        mat(2, 4),
        None,
    );

    let info = detect_matmul_pattern(&dag, sum)
        .expect("canonical Sum-Mul-Expand-Expand pattern must be recognized");
    assert_eq!(info.m, 2);
    assert_eq!(info.k, 3);
    assert_eq!(info.n, 4);
}

#[test]
fn cast_perturbed_matmul_specializes_after_noop_cleanup() {
    // Same DAG as above, but with a `Cast(_, f32)` inserted between each
    // Expand and the Mul. Mathematically a no-op (precision unchanged); the
    // pattern matcher should not care. In practice it does — the detector
    // keys off `mul.inputs[i].op == Expand`, so any node in between hides
    // the Expand and the BLAS specializer falls through.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::synth_const(mat(2, 3).precision, 1.0),
        vec![],
        mat(2, 3),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::synth_const(mat(3, 4).precision, 1.0),
        vec![],
        mat(3, 4),
        None,
    );
    let ea = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 2,
            size: chelis_ir::dag::RtDim::Lit(4),
        },
        vec![a],
        t3(2, 3, 4),
        None,
    );
    let eb = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::Lit(2),
        },
        vec![b],
        t3(2, 3, 4),
        None,
    );
    // Useless casts: f32 -> f32. Identity at the value level, hostile to the
    // pattern matcher.
    let ca = dag.add_node(
        decl,
        RiscOp::Cast {
            new_precision: Prim::F32,
        },
        vec![ea],
        t3(2, 3, 4),
        None,
    );
    let cb = dag.add_node(
        decl,
        RiscOp::Cast {
            new_precision: Prim::F32,
        },
        vec![eb],
        t3(2, 3, 4),
        None,
    );
    let mul = dag.add_node(decl, RiscOp::Mul, vec![ca, cb], t3(2, 3, 4), None);
    let sum = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![mul],
        mat(2, 4),
        None,
    );
    dag.add_root(sum);

    assert!(
        detect_matmul_pattern(&dag, sum).is_none(),
        "the raw backend detector remains intentionally strict; the IR \
         specialize pass is responsible for removing identity casts before \
         detection"
    );

    let specialized = specialize_for_blas(&dag);
    assert!(
        specialized.nodes().iter().any(|node| {
            matches!(
                &node.op,
                RiscOp::BlasMatmul {
                    batch_dims,
                    m,
                    n,
                    k,
                    ..
                } if batch_dims.is_empty()
                    && *m == DimExpr::Concrete(2)
                    && *n == DimExpr::Concrete(4)
                    && *k == DimExpr::Concrete(3)
            )
        }),
        "identity casts must not prevent the IR specialize pass from replacing \
         the matmul pattern with a specialized BLAS node"
    );
    assert!(
        !specialized
            .nodes()
            .iter()
            .any(|node| matches!(node.op, RiscOp::Mul | RiscOp::Expand { .. })),
        "post-specialization DCE must remove the orphan Mul and Expand nodes"
    );
}

#[test]
fn internal_one_hot_gather_tree_specializes_to_sparse_gather() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let values = dag.add_node(
        decl,
        RiscOp::synth_const(mat(3, 2).precision, 1.0),
        vec![],
        mat(3, 2),
        None,
    );
    let indices = dag.add_node(
        decl,
        RiscOp::synth_const(vec_i64(4).precision, 0.0),
        vec![],
        vec_i64(4),
        None,
    );
    let one_hot = dag.add_node(
        decl,
        RiscOp::OneHot { vocab: 3 },
        vec![indices],
        mat(4, 3),
        None,
    );
    let expanded_one_hot = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 2,
            size: chelis_ir::dag::RtDim::Lit(2),
        },
        vec![one_hot],
        t3(4, 3, 2),
        None,
    );
    let expanded_values = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::Lit(4),
        },
        vec![values],
        t3(4, 3, 2),
        None,
    );
    let product = dag.add_node(
        decl,
        RiscOp::Mul,
        vec![expanded_one_hot, expanded_values],
        t3(4, 3, 2),
        None,
    );
    let out = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![product],
        mat(4, 2),
        None,
    );
    dag.add_root(out);

    let specialized = specialize_for_blas(&dag);
    assert!(
        specialized.nodes().iter().any(|node| matches!(
            node.op,
            RiscOp::Gather {
                axis: 0,
                batch_rank: 0
            }
        )),
        "the IR specialize pass must collapse the internal dense gather tree \
         to first-class sparse Gather"
    );
    assert!(
        !specialized.nodes().iter().any(|node| {
            matches!(
                node.op,
                RiscOp::OneHot { .. } | RiscOp::Mul | RiscOp::Expand { .. }
            )
        }),
        "post-specialization DCE must remove the dense one-hot/product tree"
    );
}
