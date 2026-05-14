//! RT-2 finding: spec §5.7.1 narrowness rule is enforced for
//! BlasMatmul but NOT for Sum at the IR-verify layer.
//!
//! verify.rs lines 392-403: `Sum` is checked only for
//! `output_type.precision == accumulator` — not for the narrowness
//! rule that the constructor `sum_with_accumulator` enforces.
//!
//! verify.rs lines 404-468: `BlasMatmul` IS checked for narrowness
//! against the spec default.
//!
//! Symptom: a hand-built (or lowering-produced) Sum with int8
//! accumulator on int8 operand passes verify, then the C backend's
//! emit_reduce_sum dispatches the int8 path with int8 accumulator,
//! silently overflowing on values whose sum exceeds 127.
//!
//! Spec §5.7.1: "The accumulator parameter is permitted only when it
//! is at least as wide as the operand precision and is not narrower
//! than the documented default."

use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::verify;
use chelis_types::types::Prim;

fn vec_t(n: usize, p: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: p,
    }
}

fn scalar_t(p: Prim) -> TensorType {
    TensorType {
        dims: vec![],
        precision: p,
    }
}

#[test]
fn ir_verify_accepts_sum_int8_with_int8_accumulator_silent_overflow() {
    // Hand-build a Sum with int8 accumulator (violates §5.7.1).
    let mut dag = Dag::new();
    let inp = dag.add_node(
        RiscOp::Load {
            name: "xs".to_string().into(),
        },
        vec![],
        vec_t(3, Prim::Int8),
        None,
    );
    let bad = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::Int8,
        },
        vec![inp],
        scalar_t(Prim::Int8),
        None,
    );
    dag.add_root(bad);
    let errs = verify::verify(&dag);
    assert!(
        !errs.is_empty(),
        "IR verify ASYMMETRY BUG: spec §5.7.1 narrowness rule for Sum is \
         not enforced at the verify layer (BlasMatmul has the rule but \
         Sum does not). A hand-built Sum {{accumulator: Int8}} on an int8 \
         operand passes verify cleanly; the constructor \
         `sum_with_accumulator` would have rejected it. Symptom: the \
         lowering or any other producer that constructs RiscOp::Sum \
         directly bypasses the spec rule. Fix: add the narrowness check \
         to the Sum arm of verify.rs at line 392."
    );
    assert!(
        errs.iter().any(|e| e.contains("§5.7.1")),
        "verify error must cite §5.7.1; got: {errs:?}"
    );
}

#[test]
fn ir_verify_accepts_sum_int16_with_int16_accumulator_silent_overflow() {
    let mut dag = Dag::new();
    let inp = dag.add_node(
        RiscOp::Load {
            name: "xs".to_string().into(),
        },
        vec![],
        vec_t(3, Prim::Int16),
        None,
    );
    let bad = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::Int16,
        },
        vec![inp],
        scalar_t(Prim::Int16),
        None,
    );
    dag.add_root(bad);
    let errs = verify::verify(&dag);
    assert!(
        !errs.is_empty(),
        "IR verify ASYMMETRY BUG: int16 operand + int16 accumulator passes \
         verify, despite spec §5.7.1 requiring int32 default."
    );
}

#[test]
fn ir_verify_accepts_sum_bf16_with_bf16_accumulator_narrowness_violation() {
    let mut dag = Dag::new();
    let inp = dag.add_node(
        RiscOp::Load {
            name: "xs".to_string().into(),
        },
        vec![],
        vec_t(3, Prim::Bf16),
        None,
    );
    let bad = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::Bf16,
        },
        vec![inp],
        scalar_t(Prim::Bf16),
        None,
    );
    dag.add_root(bad);
    let errs = verify::verify(&dag);
    assert!(
        !errs.is_empty(),
        "IR verify ASYMMETRY BUG: bf16 operand + bf16 accumulator passes \
         verify; spec §5.7.1 requires f32 default for bf16."
    );
}
