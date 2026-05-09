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
//! 1. There is *no* `detect_gather_pattern` in Chelis today. `gather` is
//!    host-only (`crates/chelis-ir/src/host.rs:4640`); the spec §3.5
//!    `one_hot+expand+mul+sum` lowering is not wired. So the original Test 2
//!    framing — "implement spec gather lowering, see whether it folds back to
//!    a hardware lookup" — has nothing to break: the lowering doesn't ship.
//!    The risk re-arms only if Phase 3h adds the lowering paired with a
//!    recognizer. Until then, gather doesn't OOM because the dangerous
//!    decomposition isn't generated.
//!
//! 2. The raw backend detector still keys off `Sum -> Mul -> (Expand, Expand)`,
//!    but M1 added `chelis_ir::specialize`, which runs closed-list no-op
//!    cleanup first. The user-facing contract is that identity `Cast(f32)`
//!    no longer hides a matmul from BLAS specialization.

use chelis_backend_c::blas::detect_matmul_pattern;
use chelis_ir::dag::{Dag, DimExpr, DimInfo, RiscOp, TensorType};
use chelis_ir::specialize::specialize_for_blas;
use chelis_types::types::Prim;

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

#[test]
fn canonical_matmul_pattern_is_detected() {
    // Reproduce the canonical Tier 2 matmul lowering:
    //   A: [m=2, k=3], B: [k=3, n=4]
    //   ea = expand(A, axis=2, size=4)  -> [2, 3, 4]
    //   eb = expand(B, axis=0, size=2)  -> [2, 3, 4]
    //   p  = mul(ea, eb)                 -> [2, 3, 4]
    //   c  = sum(p, axis=1)              -> [2, 4]
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat(2, 3), None);
    let b = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat(3, 4), None);
    let ea = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: DimExpr::Concrete(4),
        },
        vec![a],
        t3(2, 3, 4),
        None,
    );
    let eb = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(2),
        },
        vec![b],
        t3(2, 3, 4),
        None,
    );
    let mul = dag.add_node(RiscOp::Mul, vec![ea, eb], t3(2, 3, 4), None);
    let sum = dag.add_node(RiscOp::Sum { axis: 1 }, vec![mul], mat(2, 4), None);

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
    let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat(2, 3), None);
    let b = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat(3, 4), None);
    let ea = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: DimExpr::Concrete(4),
        },
        vec![a],
        t3(2, 3, 4),
        None,
    );
    let eb = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(2),
        },
        vec![b],
        t3(2, 3, 4),
        None,
    );
    // Useless casts: f32 -> f32. Identity at the value level, hostile to the
    // pattern matcher.
    let ca = dag.add_node(
        RiscOp::Cast {
            new_precision: Prim::F32,
        },
        vec![ea],
        t3(2, 3, 4),
        None,
    );
    let cb = dag.add_node(
        RiscOp::Cast {
            new_precision: Prim::F32,
        },
        vec![eb],
        t3(2, 3, 4),
        None,
    );
    let mul = dag.add_node(RiscOp::Mul, vec![ca, cb], t3(2, 3, 4), None);
    let sum = dag.add_node(RiscOp::Sum { axis: 1 }, vec![mul], mat(2, 4), None);
    dag.add_root(sum);

    assert!(
        detect_matmul_pattern(&dag, sum).is_none(),
        "the raw backend detector remains intentionally strict; the IR \
         specialize pass is responsible for removing identity casts before \
         detection"
    );

    let specialized = specialize_for_blas(&dag);
    assert!(
        specialized
            .nodes()
            .iter()
            .any(|node| matches!(node.op, RiscOp::BlasMatmul { m: 2, n: 4, k: 3 })),
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
fn structural_no_gather_recognizer_today() {
    // Sentinel: there is no gather→pointer-lookup recognizer in this snapshot.
    // If a future commit adds one (e.g., `detect_gather_pattern` in this crate
    // or in `chelis-ir/src/optimize.rs`), the gather OOM trap re-arms unless
    // the same change adds tests like `cast_perturbed_gather_pattern_misses`
    // analogous to the matmul case above. This test exists to be a noisy
    // reminder when grep'ing.
    //
    // Verify by absence: the C backend's blas module exports only
    // `MatmulInfo`/`detect_matmul_pattern`, not any gather variant.
    //
    // This test passes by construction; the comment is the durable artifact.
    let probe = std::any::type_name::<chelis_backend_c::blas::MatmulInfo>();
    assert!(probe.ends_with("::blas::MatmulInfo"));
}
