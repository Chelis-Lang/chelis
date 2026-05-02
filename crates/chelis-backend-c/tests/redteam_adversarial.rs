//! Adversarial red-team validation tests for Level 3b SIMD math codegen.
//!
//! Items 4–9 from the red-team checklist.

use chelis_backend_c::{CodegenOptions, MathLib, codegen_with_options};
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::fuse::fuse;
use chelis_types::types::Prim;

fn vec_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

/// Build a DAG: single Load → chain of 5 unary math ops.
/// exp → log → sin → sqrt → exp
fn build_five_op_math_dag(n: usize) -> Dag {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(n), None);
    let e1 = dag.add_node(RiscOp::Exp, vec![a], vec_f32(n), None);
    let l1 = dag.add_node(RiscOp::Log, vec![e1], vec_f32(n), None);
    let s1 = dag.add_node(RiscOp::Sin, vec![l1], vec_f32(n), None);
    let sq = dag.add_node(RiscOp::Sqrt, vec![s1], vec_f32(n), None);
    dag.add_node(RiscOp::Exp, vec![sq], vec_f32(n), None);
    fuse(&dag)
}

// ---- Item 4: 5-op Sleef fused kernel ----

#[test]
fn redteam_five_op_sleef_contains_all_macros() {
    // Adversarial: 5 sequential math ops (exp→log→sin→sqrt→exp) with Sleef forced.
    // All five Sleef macros must appear. AVX2 load/store must appear. Scalar tail must appear.
    let dag = build_five_op_math_dag(16);
    let result = codegen_with_options(
        &dag,
        "test_five_ops",
        CodegenOptions {
            math_lib_override: Some(MathLib::Sleef),
            ..CodegenOptions::default()
        },
    );
    let src = &result.c_source;

    // All five Sleef macros (exp appears twice — first and last step).
    assert!(
        src.contains("CHELIS_EXPF8("),
        "5-op kernel missing CHELIS_EXPF8 macro:\n{src}"
    );
    assert!(
        src.contains("CHELIS_LOGF8("),
        "5-op kernel missing CHELIS_LOGF8 macro:\n{src}"
    );
    assert!(
        src.contains("CHELIS_SINF8("),
        "5-op kernel missing CHELIS_SINF8 macro:\n{src}"
    );
    assert!(
        src.contains("CHELIS_SQRTF8("),
        "5-op kernel missing CHELIS_SQRTF8 macro:\n{src}"
    );

    // Count CHELIS_EXPF8 occurrences — should be at least 2 (steps 0 and 4).
    let exp_count = src.matches("CHELIS_EXPF8(").count();
    assert!(
        exp_count >= 2,
        "expected >= 2 CHELIS_EXPF8 occurrences, got {exp_count}:\n{src}"
    );

    // AVX2 load and store intrinsics.
    assert!(
        src.contains("_mm256_loadu_ps("),
        "5-op kernel missing _mm256_loadu_ps:\n{src}"
    );
    assert!(
        src.contains("_mm256_storeu_ps("),
        "5-op kernel missing _mm256_storeu_ps:\n{src}"
    );

    // Scalar tail loop guard.
    assert!(
        src.contains("for (; __i < "),
        "5-op kernel missing scalar tail loop:\n{src}"
    );

    // Sleef guard must be present.
    assert!(
        src.contains("#ifdef CHELIS_HAS_SLEEF"),
        "5-op kernel missing #ifdef CHELIS_HAS_SLEEF:\n{src}"
    );
}

// ---- Item 5: size=0 tensor ----

#[test]
fn redteam_zero_size_tensor_loop_does_not_execute() {
    // n=0: the AVX2 loop guard `__i + 8 <= 0` is always false, and the scalar
    // tail guard `__i < 0` is also always false.  The generated C must still compile.
    // We verify the source contains the standard loop structure (correct loop guard form).
    let dag = build_five_op_math_dag(0);
    let result = codegen_with_options(
        &dag,
        "test_zero_size",
        CodegenOptions {
            math_lib_override: Some(MathLib::Sleef),
            ..CodegenOptions::default()
        },
    );
    let src = &result.c_source;

    // The AVX2 loop must use the standard guard form.
    assert!(
        src.contains("__i + 8 <= "),
        "zero-size kernel missing AVX2 loop guard form:\n{src}"
    );
    // The scalar tail loop must use the standard guard form.
    assert!(
        src.contains("for (; __i < "),
        "zero-size kernel missing scalar tail guard form:\n{src}"
    );
}

// ---- Item 6: MathLib::None — scalar fallback ----

#[test]
fn redteam_math_lib_none_uses_scalar_expf() {
    // MathLib::None: no math header, no Sleef guards, must have omp simd and expf().
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(8), None);
    dag.add_node(RiscOp::Exp, vec![a], vec_f32(8), None);
    let dag = fuse(&dag);

    let result = codegen_with_options(
        &dag,
        "test_none_path",
        CodegenOptions {
            math_lib_override: Some(MathLib::None),
            ..CodegenOptions::default()
        },
    );
    let src = &result.c_source;

    // Must NOT contain math header.
    assert!(
        !src.contains("#include \"chelis_math.h\""),
        "MathLib::None must not emit chelis_math.h:\n{src}"
    );

    // Must NOT contain Sleef guard.
    assert!(
        !src.contains("#ifdef CHELIS_HAS_SLEEF"),
        "MathLib::None must not emit Sleef guard:\n{src}"
    );

    // Must use omp parallel for simd (Level-1 path).
    assert!(
        src.contains("#pragma omp parallel for simd"),
        "MathLib::None must use Level-1 omp simd loop:\n{src}"
    );

    // Must use scalar expf().
    assert!(
        src.contains("expf("),
        "MathLib::None must emit scalar expf():\n{src}"
    );
}

// ---- Item 7: Pure arithmetic kernel with Sleef forced ----

#[test]
fn redteam_pure_arithmetic_with_sleef_forced_uses_level1_path() {
    // Add-only kernel with Sleef forced: must NOT emit Sleef guard, MUST use omp simd.
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(8), None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], vec_f32(8), None);
    dag.add_node(RiscOp::Add, vec![a, b], vec_f32(8), None);
    let dag = fuse(&dag);

    let result = codegen_with_options(
        &dag,
        "test_arith_sleef",
        CodegenOptions {
            math_lib_override: Some(MathLib::Sleef),
            ..CodegenOptions::default()
        },
    );
    let src = &result.c_source;

    // Must NOT contain Sleef guard.
    assert!(
        !src.contains("#ifdef CHELIS_HAS_SLEEF"),
        "pure-arithmetic kernel must not emit Sleef guard:\n{src}"
    );

    // Must use omp parallel for simd.
    assert!(
        src.contains("#pragma omp parallel for simd"),
        "pure-arithmetic kernel must use Level-1 omp simd loop:\n{src}"
    );
}

// ---- Item 9: VForce dispatch ----

#[test]
fn redteam_vforce_single_exp_kernel() {
    // Single-op exp kernel with VForce forced: must emit vvexpf, must NOT emit Sleef guard.
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(8), None);
    dag.add_node(RiscOp::Exp, vec![a], vec_f32(8), None);
    let dag = fuse(&dag);

    let result = codegen_with_options(
        &dag,
        "test_vforce_exp",
        CodegenOptions {
            math_lib_override: Some(MathLib::VForce),
            ..CodegenOptions::default()
        },
    );
    let src = &result.c_source;

    assert!(
        src.contains("vvexpf"),
        "VForce single-exp kernel must emit vvexpf:\n{src}"
    );
    assert!(
        !src.contains("#ifdef CHELIS_HAS_SLEEF"),
        "VForce single-exp kernel must NOT emit Sleef guard:\n{src}"
    );
}

#[test]
fn redteam_vforce_two_op_kernel_falls_through_to_level1() {
    // Two-op kernel (add + exp): NOT a simple vForce kernel (is_simple_vforce_kernel=false).
    // Must NOT emit vvexpf, MUST emit omp parallel for simd (Level-1 path).
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(8), None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], vec_f32(8), None);
    let add = dag.add_node(RiscOp::Add, vec![a, b], vec_f32(8), None);
    dag.add_node(RiscOp::Exp, vec![add], vec_f32(8), None);
    let dag = fuse(&dag);

    let result = codegen_with_options(
        &dag,
        "test_vforce_two_op",
        CodegenOptions {
            math_lib_override: Some(MathLib::VForce),
            ..CodegenOptions::default()
        },
    );
    let src = &result.c_source;

    assert!(
        !src.contains("vvexpf"),
        "two-op VForce kernel must NOT emit vvexpf (falls to Level-1):\n{src}"
    );
    assert!(
        src.contains("#pragma omp parallel for simd"),
        "two-op VForce kernel must use Level-1 omp simd loop:\n{src}"
    );
}
