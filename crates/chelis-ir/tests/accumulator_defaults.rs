//! WS-A0 acceptance tests (d)–(f): the IR `Sum` and `BlasMatmul`
//! constructors must default the `accumulator` precision per
//! spec/04-type-system.md §5.7.1, must reject integer matmul per
//! §5.7.2, and must reject explicitly-narrower-than-default `reduce_sum`
//! accumulators per §5.7.1.
//!
//! These pin the spec table directly so any drift in the default
//! lookup or the rejection path is caught at the IR level — before
//! it reaches a backend or test fixture that might tolerate the
//! wrong precision.

use chelis_ir::dag::RiscOp;
use chelis_types::types::Prim;

// ----------------------------------------------------------------
// (d) accumulator-default resolution test
// ----------------------------------------------------------------

#[test]
fn reduce_sum_default_accumulator_matches_spec_5_7_1_table() {
    // One assertion per row of the §5.7.1 table for reduce_sum.
    let cases: &[(Prim, Prim)] = &[
        (Prim::Bf16, Prim::F32),
        (Prim::F16, Prim::F32),
        (Prim::F32, Prim::F32),
        (Prim::F64, Prim::F64),
        (Prim::Int8, Prim::Int32),
        (Prim::Int16, Prim::Int32),
        (Prim::Int32, Prim::Int32),
        (Prim::Int64, Prim::Int64),
    ];
    for (operand, expected) in cases {
        let acc = RiscOp::default_reduce_sum_accumulator(*operand).unwrap_or_else(|err| {
            panic!(
                "operand `{}` must yield default accumulator `{}`, got err: {err}",
                operand.name(),
                expected.name()
            )
        });
        assert_eq!(
            acc,
            *expected,
            "spec §5.7.1: reduce_sum default accumulator for operand `{}` must be `{}`, got `{}`",
            operand.name(),
            expected.name(),
            acc.name(),
        );
    }
}

#[test]
fn matmul_default_accumulator_matches_spec_5_7_1_table() {
    // §5.7.1 matmul rows. Integer rows are rejected per §5.7.2; see the
    // dedicated test below.
    let cases: &[(Prim, Prim)] = &[
        (Prim::Bf16, Prim::F32),
        (Prim::F16, Prim::F32),
        (Prim::F32, Prim::F32),
        (Prim::F64, Prim::F64),
    ];
    for (operand, expected) in cases {
        let acc = RiscOp::default_matmul_accumulator(*operand).unwrap_or_else(|err| {
            panic!(
                "operand `{}` must yield default accumulator `{}`, got err: {err}",
                operand.name(),
                expected.name()
            )
        });
        assert_eq!(
            acc,
            *expected,
            "spec §5.7.1: matmul default accumulator for operand `{}` must be `{}`, got `{}`",
            operand.name(),
            expected.name(),
            acc.name(),
        );
    }
}

#[test]
fn sum_default_constructor_pins_accumulator_into_node() {
    // The `sum_default` constructor is the canonical "no explicit
    // accumulator parameter" lowering path. Verify the field actually
    // ends up populated on the constructed node.
    let op = RiscOp::sum_default(0, Prim::Bf16).expect("bf16 reduce_sum should construct");
    match op {
        RiscOp::Sum { axis, accumulator } => {
            assert_eq!(axis, 0);
            assert_eq!(accumulator, Prim::F32, "bf16 → f32 default per §5.7.1");
        }
        other => panic!("expected RiscOp::Sum, got {other:?}"),
    }

    let op = RiscOp::sum_default(2, Prim::Int8).expect("int8 reduce_sum should construct");
    match op {
        RiscOp::Sum { axis, accumulator } => {
            assert_eq!(axis, 2);
            assert_eq!(accumulator, Prim::Int32, "int8 → int32 default per §5.7.1");
        }
        other => panic!("expected RiscOp::Sum, got {other:?}"),
    }
}

#[test]
fn matmul_default_constructor_pins_accumulator_into_node() {
    use chelis_ir::dag::DimExpr;
    let op = RiscOp::matmul_default(
        vec![],
        DimExpr::Concrete(2),
        DimExpr::Concrete(3),
        DimExpr::Concrete(4),
        Prim::F16,
    )
    .expect("f16 matmul should construct");
    match op {
        RiscOp::BlasMatmul { accumulator, .. } => {
            assert_eq!(accumulator, Prim::F32, "f16 → f32 default per §5.7.1");
        }
        other => panic!("expected RiscOp::BlasMatmul, got {other:?}"),
    }
}

// ----------------------------------------------------------------
// (e) integer matmul rejection test
// ----------------------------------------------------------------

#[test]
fn integer_matmul_rejected_per_spec_5_7_2() {
    use chelis_ir::dag::DimExpr;
    for operand in [Prim::Int8, Prim::Int16, Prim::Int32, Prim::Int64] {
        let result = RiscOp::matmul_default(
            vec![],
            DimExpr::Concrete(2),
            DimExpr::Concrete(2),
            DimExpr::Concrete(2),
            operand,
        );
        let err = result.expect_err(&format!(
            "matmul on integer operand `{}` must be rejected per spec §5.7.2",
            operand.name()
        ));
        assert!(
            err.contains("§5.7.2"),
            "rejection diagnostic must cite spec/04-type-system.md §5.7.2 for operand `{}`; got: {err}",
            operand.name()
        );
        assert!(
            err.contains("not admitted") && err.contains("integer"),
            "rejection diagnostic must explain integer operand is not admitted for matmul; got: {err}"
        );

        // The default-accumulator helper is the same rejection path.
        let default_err = RiscOp::default_matmul_accumulator(operand)
            .expect_err("default_matmul_accumulator must reject integer operands per §5.7.2");
        assert!(default_err.contains("§5.7.2"));
    }
}

#[test]
fn matmul_with_accumulator_also_rejects_integer_operands() {
    use chelis_ir::dag::DimExpr;
    let result = RiscOp::matmul_with_accumulator(
        vec![],
        DimExpr::Concrete(2),
        DimExpr::Concrete(2),
        DimExpr::Concrete(2),
        Prim::Int32,
        Prim::Int64,
    );
    let err = result.expect_err("explicit-accumulator matmul on int32 must be rejected per §5.7.2");
    assert!(err.contains("§5.7.2"));
}

// ----------------------------------------------------------------
// (f) narrow-accumulator rejection test
// ----------------------------------------------------------------

#[test]
fn reduce_sum_int8_with_explicit_int8_accumulator_is_rejected_per_spec_5_7_1() {
    // Per §5.7.1: the explicit accumulator must be at least as wide as
    // the documented default. For int8 operands the default is int32,
    // so int8 accumulator is too narrow.
    let err = RiscOp::sum_with_accumulator(0, Prim::Int8, Prim::Int8)
        .expect_err("int8 operand + int8 accumulator must be rejected per §5.7.1");
    assert!(
        err.contains("narrower than"),
        "diagnostic must explain the accumulator is narrower than the default; got: {err}"
    );
    assert!(
        err.contains("§5.7.1"),
        "diagnostic must cite spec/04-type-system.md §5.7.1; got: {err}"
    );
    assert!(
        err.contains("int32"),
        "diagnostic must mention the spec-default accumulator (`int32`); got: {err}"
    );
}

#[test]
fn reduce_sum_int16_with_explicit_int16_accumulator_is_rejected() {
    // Same shape as int8 (default accumulator is int32 for int16).
    let err = RiscOp::sum_with_accumulator(0, Prim::Int16, Prim::Int16)
        .expect_err("int16 operand + int16 accumulator must be rejected per §5.7.1");
    assert!(err.contains("§5.7.1"));
    assert!(err.contains("int32"));
}

#[test]
fn reduce_sum_bf16_with_explicit_bf16_accumulator_is_rejected() {
    // Default accumulator for bf16 is f32; explicit bf16 is narrower.
    let err = RiscOp::sum_with_accumulator(0, Prim::Bf16, Prim::Bf16)
        .expect_err("bf16 operand + bf16 accumulator must be rejected per §5.7.1");
    assert!(err.contains("§5.7.1"));
    assert!(err.contains("f32"));
}

#[test]
fn reduce_sum_f32_with_explicit_f32_accumulator_is_accepted() {
    // Per §5.7.1 the f32 row accepts an f32 accumulator (it IS the
    // default); not a rejection case but a sanity sibling so the
    // narrowness rule cannot mistakenly reject the equal-width case.
    let op = RiscOp::sum_with_accumulator(0, Prim::F32, Prim::F32)
        .expect("f32 operand + f32 accumulator must be accepted per §5.7.1");
    match op {
        RiscOp::Sum { accumulator, .. } => assert_eq!(accumulator, Prim::F32),
        other => panic!("expected RiscOp::Sum, got {other:?}"),
    }
}

#[test]
fn reduce_sum_bf16_with_explicit_f64_accumulator_is_accepted() {
    // f64 is wider than the default f32 for bf16 operands; per §5.7.1
    // wider-than-default accumulators are permitted.
    let op = RiscOp::sum_with_accumulator(0, Prim::Bf16, Prim::F64)
        .expect("bf16 operand + f64 accumulator must be accepted per §5.7.1");
    match op {
        RiscOp::Sum { accumulator, .. } => assert_eq!(accumulator, Prim::F64),
        other => panic!("expected RiscOp::Sum, got {other:?}"),
    }
}
