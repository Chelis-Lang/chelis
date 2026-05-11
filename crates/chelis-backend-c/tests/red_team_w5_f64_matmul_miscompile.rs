//! Wave 5 red-team — **P0 SILENT MISCOMPILE** confirmation.
//!
//! Demonstrates end-to-end that an F64 matmul subgraph reaches C
//! codegen as `RiscOp::BlasMatmul` (because the IR specializer has no
//! precision gate) and the C backend then emits `cblas_sgemm` —
//! single-precision BLAS — against the F64 data.
//!
//! This is a SILENT MISCOMPILE: no diagnostic, no fail-closed panic,
//! just `cblas_sgemm` reading F64 data through `float*` strides. The
//! emitted code would (at runtime) compute mathematical nonsense.
//!
//! The fix is a precision filter in
//! `crates/chelis-ir/src/specialize.rs::detect_matmul_pattern` (or in
//! `crates/chelis-backend-c/src/emit.rs::emit_blas_matmul` to reject
//! non-F32 BlasMatmul). The latter is more conservative — it catches
//! the bug at every call site.
//!
//! This test PINS the current (buggy) behavior. When the fix lands,
//! this test must be removed or flipped to assert a panic/skip.
//!
//! Per the W5 brief's "escalate structural workarounds" rule, the
//! red-team agent does NOT fix this in-place; it surfaces the finding
//! with severity P0 to the orchestrator.

use chelis_backend_c::{CodegenOptions, codegen_with_options};
use chelis_ir::dag::{Dag, DimExpr, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;

fn t(prim: Prim, dims: Vec<usize>) -> TensorType {
    TensorType {
        dims: dims.into_iter().map(DimInfo::Lit).collect(),
        precision: prim,
    }
}

/// P0 confirmation: end-to-end F64 matmul → cblas_sgemm.
#[test]
fn p0_f64_matmul_subgraph_emits_cblas_sgemm_in_c_backend_silent_miscompile() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        t(Prim::F64, vec![8, 16]),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        t(Prim::F64, vec![16, 4]),
        None,
    );
    let ea = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: DimExpr::Concrete(4),
        },
        vec![a],
        t(Prim::F64, vec![8, 16, 4]),
        None,
    );
    let eb = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(8),
        },
        vec![b],
        t(Prim::F64, vec![8, 16, 4]),
        None,
    );
    let mul = dag.add_node(
        RiscOp::Mul,
        vec![ea, eb],
        t(Prim::F64, vec![8, 16, 4]),
        None,
    );
    let sum = dag.add_node(
        RiscOp::Sum { axis: 1 },
        vec![mul],
        t(Prim::F64, vec![8, 4]),
        None,
    );
    dag.add_root(sum);

    // The CLI's `chelis build` path uses `use_blas: true`
    // (`crates/chelis-cli/src/main.rs:4060`). Mirror that here so the
    // test reproduces the end-user-visible code path, not the
    // library-default no-BLAS path.
    let result = codegen_with_options(
        &dag,
        "p0_f64_matmul_miscompile",
        CodegenOptions {
            use_blas: true,
            ..CodegenOptions::default()
        },
    );

    // The IR specializer must have replaced the matmul with BlasMatmul
    // (no precision gate). The C backend must have emitted cblas_sgemm
    // — single-precision — against the F64 data. This is the silent
    // miscompile. Lock both legs.
    assert!(
        result.c_source.contains("cblas_sgemm("),
        "P0 BUG: F64 matmul subgraph is reaching the C backend as \
         BlasMatmul (silent specialization) and emitting cblas_sgemm \
         against double-precision data. This is a SILENT MISCOMPILE. \
         Emitted C source:\n{}",
        result.c_source
    );

    // Sanity: the data buffer the call addresses is the F64 buffer.
    // Look for `double *` typed accesses near the cblas_sgemm site —
    // the chelis runtime treats data as untyped void*, but the
    // `chelis_slotN` allocation should be sized for F64. Lock the
    // slot allocation tag.
    assert!(
        result.c_source.contains("CHELIS_F64"),
        "F64 slot allocation must be present (proves F64 ground truth); \
         got source:\n{}",
        result.c_source
    );

    // The BLAS requirement should be set on the requirements struct so
    // the user's build invokes -lopenblas — confirming the silently-wrong
    // binary would actually link.
    let requires_blas = format!("{:?}", result.requirements)
        .to_lowercase()
        .contains("blas");
    assert!(
        requires_blas,
        "F64 matmul→BLAS specialization should surface a BLAS requirement, \
         confirming the user would *successfully build* the silently \
         wrong binary; got requirements = {:?}",
        result.requirements
    );
}
