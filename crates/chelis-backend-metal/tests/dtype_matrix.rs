//! WS-M1: Per-dtype matrix coverage for the Metal backend.
//!
//! Exercises the structural emission contract for every active Metal
//! dtype (per spec/04-type-system.md §1.1.3). Runs in default CI on
//! every platform (no Metal device or `clang++` required); the
//! evaluator-vs-Metal byte-identical agreement gate that requires an
//! actual GPU lives in `gpu_correctness.rs` (`#[ignore]` by default,
//! manual gate per CLAUDE.md "One Acceptance Oracle Per Phase").
//!
//! Coverage matrix (one test per (dtype, op) pair the dtype admits):
//!
//! | dtype | add | mul | reduce_sum (acc per §5.7.1) | matmul        |
//! |-------|-----|-----|----------------------------|---------------|
//! | f32   |  X  |  X  | f32 → f32                  | MPS           |
//! | f16   |  X  |  X  | f16 → f32                  | MPS           |
//! | bf16  |  X  |  X  | bf16 → f32                 | tiled MSL     |
//! | i8  |  X  |  X  | i8 → i32               | rejected (F1) |
//! | i16 |  X  |  X  | i16 → i32              | rejected (F1) |
//! | i32 |  X  |  X  | i32 → i32              | rejected (F1) |
//! | i64 |  X  |  X  | i64 → i64              | rejected (F1) |
//! | bool  |  -  |  -  | bool → bool                | rejected      |
//!
//! Negative coverage:
//! - f64 reaches `require_metal_admissible` and returns a typed rejection
//!   without an artifact (the user-facing rejection normally happens at
//!   the CLI gate per spec/04-type-system.md §1.1.3).
//! - mixed-precision matmul fails the `precision` check in `MatmulInfo`
//!   construction; the detector returns `None` and matmul never reaches
//!   codegen.

mod support;
use chelis_backend_metal::dtype;
use chelis_ir::dag::{Dag, DagNode, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;
use support::{codegen_metal, try_codegen_metal};

fn vec_prec(n: usize, p: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: p,
    }
}

fn scalar_prec(p: Prim) -> TensorType {
    TensorType {
        dims: vec![],
        precision: p,
    }
}

fn build_add_dag(prec: Prim) -> Dag {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_prec(4, prec),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        vec_prec(4, prec),
        None,
    );
    let s = dag.add_node(RiscOp::Add, vec![a, b], vec_prec(4, prec), None);
    let stored = dag.add_node(
        RiscOp::Store { name: "out".into() },
        vec![s],
        vec_prec(4, prec),
        None,
    );
    dag.add_root(stored);
    dag
}

fn build_mul_dag(prec: Prim) -> Dag {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_prec(4, prec),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        vec_prec(4, prec),
        None,
    );
    let s = dag.add_node(RiscOp::Mul, vec![a, b], vec_prec(4, prec), None);
    let stored = dag.add_node(
        RiscOp::Store { name: "out".into() },
        vec![s],
        vec_prec(4, prec),
        None,
    );
    dag.add_root(stored);
    dag
}

fn build_relu_dag(prec: Prim) -> Dag {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_prec(6, prec),
        None,
    );
    let g = dag.add_node(
        RiscOp::Load { name: "g".into() },
        vec![],
        vec_prec(6, prec),
        None,
    );
    let relu = dag.add_node(RiscOp::Relu, vec![x], vec_prec(6, prec), None);
    let adjoint = dag.add_node(RiscOp::ReluAdjoint, vec![x, g], vec_prec(6, prec), None);
    dag.add_root(relu);
    dag.add_root(adjoint);
    dag
}

fn build_reduce_sum_dag(prec: Prim) -> Dag {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_prec(8, prec),
        None,
    );
    let acc = dtype::sum_accumulator(prec);
    let r = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: acc,
        },
        vec![a],
        scalar_prec(acc),
        None,
    );
    dag.add_root(r);
    dag
}

fn build_matmul_dag(prec: Prim) -> Dag {
    let m = 4usize;
    let k = 4usize;
    let n = 4usize;
    let mut dag = Dag::new();
    let mat = |r: usize, c: usize| TensorType {
        dims: vec![DimInfo::Lit(r), DimInfo::Lit(c)],
        precision: prec,
    };
    let cube = |a: usize, b: usize, c: usize| TensorType {
        dims: vec![DimInfo::Lit(a), DimInfo::Lit(b), DimInfo::Lit(c)],
        precision: prec,
    };
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], mat(m, k), None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], mat(k, n), None);
    let ea = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: chelis_ir::dag::RtDim::Lit(n),
        },
        vec![a],
        cube(m, k, n),
        None,
    );
    let eb = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::Lit(m),
        },
        vec![b],
        cube(m, k, n),
        None,
    );
    let mul = dag.add_node(RiscOp::Mul, vec![ea, eb], cube(m, k, n), None);
    let acc = dtype::sum_accumulator(prec);
    let sum = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: acc,
        },
        vec![mul],
        TensorType {
            dims: vec![DimInfo::Lit(m), DimInfo::Lit(n)],
            precision: acc,
        },
        None,
    );
    let root = if acc == prec {
        sum
    } else {
        dag.add_node(
            RiscOp::Cast {
                new_precision: prec,
            },
            vec![sum],
            mat(m, n),
            None,
        )
    };
    dag.add_root(root);
    dag
}

fn assert_real_kernel(src: &str, ctx: &str) {
    assert!(
        src.contains("kernel void") || src.contains("chelis_metal_mps_gemm_"),
        "{ctx}: emit must produce a real kernel or MPS dispatch. \nSource was:\n{src}"
    );
}

// ---------------------------------------------------------------------------
// Add coverage (every active dtype except bool)
// ---------------------------------------------------------------------------

#[test]
fn add_emits_typed_kernel_for_each_active_dtype() {
    let cases: &[(Prim, &str, &str)] = &[
        (Prim::F32, "float", "CHELIS_DTYPE_F32"),
        (Prim::F16, "half", "CHELIS_DTYPE_F16"),
        (Prim::Bf16, "bfloat", "CHELIS_DTYPE_BF16"),
        (Prim::Int8, "char", "CHELIS_DTYPE_I8"),
        (Prim::Int16, "short", "CHELIS_DTYPE_I16"),
        (Prim::Int32, "int", "CHELIS_DTYPE_I32"),
        (Prim::Int64, "long", "CHELIS_DTYPE_I64"),
    ];
    for (prec, msl, runtime_tag) in cases {
        let dag = build_add_dag(*prec);
        let result = codegen_metal(&dag, "addk");
        let src = &result.mm_source;
        assert_real_kernel(src, &format!("add({prec:?})"));
        let suffix = dtype::kernel_suffix(*prec);
        let kernel_name = format!("k_binary{suffix}_2");
        assert!(
            src.contains(&format!("kernel void {kernel_name}")),
            "add({prec:?}) must emit `{kernel_name}`: {src}"
        );
        assert!(
            src.contains(&format!("device const {msl}* a")),
            "add({prec:?}) operand must be `device const {msl}*`: {src}"
        );
        assert!(
            src.contains(&format!("device {msl}* out")),
            "add({prec:?}) output must be `device {msl}*`: {src}"
        );
        assert!(
            src.contains(runtime_tag),
            "add({prec:?}) Store must use runtime tag {runtime_tag}: {src}"
        );
    }
}

#[test]
fn add_bf16_wraps_kernel_in_msl_320_guard() {
    let dag = build_add_dag(Prim::Bf16);
    let result = codegen_metal(&dag, "addk_bf16");
    let src = &result.mm_source;
    assert!(
        src.contains("#if __METAL_VERSION__ >= 320"),
        "bf16 add kernel must wrap body in the MSL 3.2+ guard (Apple7+ requirement): {src}"
    );
}

#[test]
fn relu_and_adjoint_emit_exact_selection_at_every_metal_float_width() {
    for (prec, msl) in [
        (Prim::F32, "float"),
        (Prim::F16, "half"),
        (Prim::Bf16, "bfloat"),
    ] {
        let src = codegen_metal(&build_relu_dag(prec), &format!("relu_{}", prec.name())).mm_source;
        assert_real_kernel(&src, &format!("relu({prec:?})"));
        assert!(
            src.contains(&format!(
                "out[tid] = a[tid] < ({msl})0 ? ({msl})0 : a[tid];"
            )),
            "{src}"
        );
        assert!(
            src.contains(&format!(
                "out[tid] = ({msl})0 < a[tid] ? b[tid] : ({msl})0;"
            )),
            "{src}"
        );
        assert!(!src.contains("fmax"), "{src}");
        if prec == Prim::Bf16 {
            assert!(src.contains("#if __METAL_VERSION__ >= 320"), "{src}");
        }
    }
}

// ---------------------------------------------------------------------------
// Mul coverage (every active dtype except bool)
// ---------------------------------------------------------------------------

#[test]
fn mul_emits_typed_kernel_for_each_active_dtype() {
    let cases: &[Prim] = &[
        Prim::F32,
        Prim::F16,
        Prim::Bf16,
        Prim::Int8,
        Prim::Int16,
        Prim::Int32,
        Prim::Int64,
    ];
    for prec in cases {
        let dag = build_mul_dag(*prec);
        let result = codegen_metal(&dag, "mulk");
        let src = &result.mm_source;
        assert_real_kernel(src, &format!("mul({prec:?})"));
        let suffix = dtype::kernel_suffix(*prec);
        assert!(
            src.contains(&format!("kernel void k_binary{suffix}_")),
            "mul({prec:?}) must emit `k_binary{suffix}_*`: {src}"
        );
        assert!(
            src.contains("a[tid] * b[tid]"),
            "mul({prec:?}) body must call the * operator: {src}"
        );
    }
}

// ---------------------------------------------------------------------------
// reduce_sum coverage with §5.7.1 accumulator promotion
// ---------------------------------------------------------------------------

#[test]
fn reduce_sum_promotes_accumulator_per_spec() {
    // (operand, expected accumulator, expected runtime output tag).
    let cases: &[(Prim, Prim, &str)] = &[
        (Prim::F32, Prim::F32, "CHELIS_DTYPE_F32"),
        (Prim::F16, Prim::F32, "CHELIS_DTYPE_F32"),
        (Prim::Bf16, Prim::F32, "CHELIS_DTYPE_F32"),
        (Prim::Int8, Prim::Int32, "CHELIS_DTYPE_I32"),
        (Prim::Int16, Prim::Int32, "CHELIS_DTYPE_I32"),
        (Prim::Int32, Prim::Int32, "CHELIS_DTYPE_I32"),
        (Prim::Int64, Prim::Int64, "CHELIS_DTYPE_I64"),
    ];
    for (operand, expected_acc, runtime_tag) in cases {
        let dag = build_reduce_sum_dag(*operand);
        let result = codegen_metal(&dag, "redsum");
        let src = &result.mm_source;
        assert_real_kernel(src, &format!("reduce_sum({operand:?})"));
        let acc_msl = dtype::msl_type(*expected_acc);
        let in_msl = dtype::msl_type(*operand);
        assert!(
            src.contains(&format!("device const {in_msl}* input")),
            "reduce_sum({operand:?}) input ptr must be `device const {in_msl}*`: {src}"
        );
        assert!(
            src.contains(&format!("device {acc_msl}* output")),
            "reduce_sum({operand:?}) output ptr must be `device {acc_msl}*` per §5.7.1: {src}"
        );
        assert!(
            src.contains(runtime_tag),
            "reduce_sum({operand:?}) root output must declare {runtime_tag}: {src}"
        );
    }
}

#[test]
fn reduce_sum_bf16_wraps_kernel_in_msl_320_guard() {
    let dag = build_reduce_sum_dag(Prim::Bf16);
    let result = codegen_metal(&dag, "redsum_bf16");
    let src = &result.mm_source;
    assert!(
        src.contains("#if __METAL_VERSION__ >= 320"),
        "bf16 reduce_sum kernel must wrap body in the MSL 3.2+ guard: {src}"
    );
}

// ---------------------------------------------------------------------------
// Matmul coverage: f32/f16 → MPS; bf16 → tiled; integer → F1 reject
// ---------------------------------------------------------------------------

#[test]
fn matmul_f32_dispatches_to_mps() {
    let dag = build_matmul_dag(Prim::F32);
    let result = codegen_metal(&dag, "mmk");
    let src = &result.mm_source;
    assert!(
        src.contains("chelis_metal_mps_gemm_f32("),
        "f32 matmul must call MPS f32 helper: {src}"
    );
    assert!(
        !src.contains("kernel void k_matmul"),
        "f32 matmul must not emit any tiled MSL kernel: {src}"
    );
}

#[test]
fn matmul_f16_dispatches_to_mps() {
    let dag = build_matmul_dag(Prim::F16);
    let result = codegen_metal(&dag, "mmk_f16");
    let src = &result.mm_source;
    assert!(
        src.contains("chelis_metal_mps_gemm_f16("),
        "f16 matmul must call MPS f16 helper: {src}"
    );
    assert!(
        !src.contains("kernel void k_matmul"),
        "f16 matmul must not emit any tiled MSL kernel: {src}"
    );
}

#[test]
fn matmul_bf16_dispatches_to_tiled_msl() {
    let dag = build_matmul_dag(Prim::Bf16);
    let result = codegen_metal(&dag, "mmk_bf16");
    let src = &result.mm_source;
    assert!(
        src.contains("kernel void k_matmul_bf16_"),
        "bf16 matmul must emit the tiled MSL kernel with the bf16 suffix: {src}"
    );
    assert!(
        !src.contains("chelis_metal_mps_gemm_"),
        "bf16 matmul must not call MPS (no public bfloat MPS GEMM): {src}"
    );
    assert!(
        src.contains("#if __METAL_VERSION__ >= 320"),
        "bf16 matmul kernel must be guarded by MSL 3.2+: {src}"
    );
}

#[test]
fn matmul_int_rejected_at_codegen_returns_typed_unsupported() {
    // Integer matmul never reaches a healthy backend (rejected at
    // type-check per §5.7.2). The Metal blas detector also rejects it,
    // so the matmul subgraph isn't recognized; the lone Sum that
    // remains then trips the require_static_rank1 / accumulator path.
    // Codegen must fail before it can advertise an artifact.
    for prec in [Prim::Int8, Prim::Int16, Prim::Int32, Prim::Int64] {
        let dag = build_matmul_dag(prec);
        let error = try_codegen_metal(&dag, "mmk_int").unwrap_err();
        assert_eq!(
            error.stage,
            chelis_types::unsupported::Stage::Codegen("metal")
        );
        assert!(error.to_string().starts_with("unsupported:"), "{error}");
    }
}

// ---------------------------------------------------------------------------
// Negative coverage: f64 returns a typed error (CLI gate normally rejects first)
// ---------------------------------------------------------------------------

#[test]
fn f64_codegen_returns_typed_unsupported() {
    // f64 is hard-rejected at the CLI gate (test elsewhere) AND at the
    // codegen entry. Reaching codegen with f64 is a contract drift; the
    // emitter must not synthesize a kernel using a fictional MSL `double`
    // type or any other artifact.
    let dag = build_add_dag(Prim::F64);
    let error = try_codegen_metal(&dag, "addk_f64").unwrap_err();
    assert!(error.to_string().contains("[04-TGT-1]"), "{error}");
}

// ---------------------------------------------------------------------------
// Link line: MPS framework must be present after WS-M1
// ---------------------------------------------------------------------------

#[test]
fn link_flags_include_metal_performance_shaders() {
    let dag = build_matmul_dag(Prim::F32);
    let result = codegen_metal(&dag, "mmk");
    assert!(
        result
            .link_flags
            .iter()
            .any(|f| f == "MetalPerformanceShaders"),
        "WS-M1 link line must include MetalPerformanceShaders framework, got: {:?}",
        result.link_flags
    );
}

// ---------------------------------------------------------------------------
// Peak device bytes honors per-dtype element width (RT-4-Fixups F2 lesson)
// ---------------------------------------------------------------------------

#[test]
fn peak_device_bytes_honors_per_dtype_width() {
    // i8 add of two 4-element tensors: 2 inputs + 1 output = 3 buffers
    // × 4 elements × 1 byte = 12 bytes (without sizeof(float)
    // hardcoding regression).
    let dag = build_add_dag(Prim::Int8);
    let result = codegen_metal(&dag, "addk_i8");
    assert_eq!(
        result.peak_device_bytes_estimate.unwrap(),
        12,
        "i8 add (2 in + 1 out, 4 elems each) should be 12 bytes; \
         a sizeof(float) regression would report 48"
    );

    // f16 add: 4 elems × 2 bytes × 3 buffers = 24 bytes.
    let dag = build_add_dag(Prim::F16);
    let result = codegen_metal(&dag, "addk_f16");
    assert_eq!(
        result.peak_device_bytes_estimate.unwrap(),
        24,
        "f16 add should be 24 bytes; sizeof(float) regression would report 48"
    );

    // i64 add: 4 elems × 8 bytes × 3 buffers = 96 bytes.
    let dag = build_add_dag(Prim::Int64);
    let result = codegen_metal(&dag, "addk_i64");
    assert_eq!(
        result.peak_device_bytes_estimate.unwrap(),
        96,
        "i64 add should be 96 bytes; sizeof(float) regression would report 48 (under-report)"
    );
}

// Silence the unused-import warning when only some constructors are
// pulled in by individual cfg-gated tests in the future.
#[allow(dead_code)]
fn _used_imports(_n: &DagNode) {}
