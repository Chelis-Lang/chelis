//! Structural tests for HIP code generation (S1-S13).
//!
//! These verify the generated C/HIP source is well-formed WITHOUT requiring
//! a GPU or HIP runtime. They run in default CI.

use chelis_backend_hip::codegen_hip;
use chelis_ir::dag::{
    Dag, DimInfo, ExtremaKind, ExtremaOperand, RiscOp, RtAxis, RtDim, TensorType,
};
use chelis_ir::fuse::fuse;
use chelis_types::types::Prim;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn scalar_f32() -> TensorType {
    TensorType::scalar_f32()
}

fn vec_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn vec_i32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::Int32,
    }
}

fn vec_i64(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::Int64,
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

fn tensor4_f32(a: usize, b: usize, c: usize, d: usize) -> TensorType {
    TensorType {
        dims: vec![
            DimInfo::Lit(a),
            DimInfo::Lit(b),
            DimInfo::Lit(c),
            DimInfo::Lit(d),
        ],
        precision: Prim::F32,
    }
}

fn write_temp_file(dir: &Path, name: &str, contents: &str) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, contents).expect("write temp file");
    path
}

fn hipcc_available() -> bool {
    Command::new("hipcc")
        .arg("--version")
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false)
}

fn hip_runtime_src_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(chelis_backend_hip::runtime_dir())
}

fn cpu_runtime_include_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../chelis-runtime/include")
}

fn cpu_runtime_library_path() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("../../target/debug/deps"),
        manifest_dir.join("../../target/release/deps"),
    ];
    if let Ok(dir) = env::var("CHELIS_RUNTIME_DIR") {
        let candidate_dir = PathBuf::from(dir);
        if let Some(path) = fs::read_dir(&candidate_dir).ok().and_then(|entries| {
            entries.flatten().map(|entry| entry.path()).find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.starts_with("libchelis_runtime") && name.ends_with(".a"))
                    .unwrap_or(false)
            })
        }) {
            return path;
        }
    }
    for dir in candidates {
        if let Some(path) = fs::read_dir(&dir).ok().and_then(|entries| {
            entries.flatten().map(|entry| entry.path()).find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.starts_with("libchelis_runtime") && name.ends_with(".a"))
                    .unwrap_or(false)
            })
        }) {
            return path;
        }
    }
    panic!("could not locate libchelis_runtime.a for backend-hip tests");
}

fn copy_runtime_artifacts(dst: &Path) {
    let include_dir = cpu_runtime_include_dir();
    for header in &[
        "chelis_runtime.h",
        "chelis_runtime_dtype.h",
        "chelis_blas.h",
        "chelis_simd.h",
        "chelis_math.h",
    ] {
        write_temp_file(
            dst,
            header,
            &fs::read_to_string(include_dir.join(header))
                .unwrap_or_else(|_| panic!("read {header}")),
        );
    }
    fs::copy(cpu_runtime_library_path(), dst.join("libchelis_runtime.a"))
        .expect("copy rust runtime library");
}

fn host_entry_source<'a>(source: &'a str, func_name: &str) -> &'a str {
    source
        .split(&format!("void {func_name}_device("))
        .next()
        .unwrap_or(source)
}

fn hip_runtime_header() -> String {
    fs::read_to_string(hip_runtime_src_dir().join("chelis_hip_runtime.h"))
        .expect("read HIP runtime header")
}

#[test]
fn hip_runtime_header_owns_its_rank_cap_and_dtype_sizing_dependencies() {
    let header = hip_runtime_header();
    assert!(header.contains("#include \"chelis_runtime_dtype.h\""));
    assert!(header.contains("CHELIS_GPU_MAX_DIM"));
    for removed in ["CHELIS_MAX_DIM", "chelis_runtime_dtype_size_checked"] {
        assert!(
            !header.contains(removed),
            "HIP runtime header still depends on removed host symbol `{removed}`"
        );
    }

    // chelis#1360: this used to assert the header CONTAINED
    // `case CHELIS_DTYPE_F64:` / `case CHELIS_DTYPE_BOOL:`, which pinned a
    // second copy of the width table in place. The copy is how the defect
    // happened - chelis#1308 narrowed bool to one byte here while the
    // emitter went on dispatching four-byte kernels - so the assertion is
    // inverted: the header must delegate to the runtime's one authority and
    // must not restate the per-dtype widths at all.
    assert!(
        header.contains("return (size_t)chelis_dtype_size((chelis_dtype)dtype);"),
        "chelis_gpu_dtype_size must delegate to the runtime width authority"
    );
    for restated in ["case CHELIS_DTYPE_F64:", "case CHELIS_DTYPE_BOOL:"] {
        assert!(
            !header.contains(restated),
            "HIP runtime header restates the dtype width table (`{restated}`); \
             delegate to chelis_dtype_size instead (chelis#1360)"
        );
    }
}

/// Build a simple DAG: const(a) + const(b)
fn dag_add_consts() -> Dag {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 1.0),
        vec![],
        scalar_f32(),
        None,
    );
    let b = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 2.0),
        vec![],
        scalar_f32(),
        None,
    );
    let c = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32(), None);
    dag.add_root(c);
    dag
}

#[test]
fn s1_runtime_passes_wavefront_size_macro_to_hiprtc() {
    let header = hip_runtime_header();

    assert!(
        header.contains("hipGetDeviceProperties(&props, device)"),
        "runtime must discover the active device before compiling HIPRTC kernels"
    );
    assert!(
        header.contains("\"-D__AMDGCN_WAVEFRONT_SIZE=%d\""),
        "HIPRTC compile flags must define the wavefront-size macro used by HIP headers"
    );
    assert!(
        header.contains("props.warpSize"),
        "wavefront-size macro should be derived from the active device warp size"
    );
    assert!(
        header.contains("hiprtcCompileProgram(prog, 2, compile_opts)"),
        "HIPRTC invocation must pass both debug-bounds and wavefront-size options"
    );
}

/// Build a DAG with a Load input
fn dag_with_load() -> Dag {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4), None);
    let c = dag.add_node(
        RiscOp::synth_const(vec_f32(4).precision, 1.0),
        vec![],
        vec_f32(4),
        None,
    );
    let out = dag.add_node(RiscOp::Add, vec![x, c], vec_f32(4), None);
    dag.add_root(out);
    dag
}

// ===========================================================================
// S1: Generated code includes runtime header
// ===========================================================================

#[test]
fn s1_includes_hip_runtime_header() {
    let dag = dag_add_consts();
    let result = codegen_hip(&dag, "test_s1").unwrap();
    assert!(
        result
            .c_source
            .contains("#include \"chelis_hip_runtime.h\""),
        "Generated code must include HIP runtime header"
    );
}

#[test]
fn s1_neg_no_cpu_runtime_include() {
    let dag = dag_add_consts();
    let result = codegen_hip(&dag, "test_s1_neg").unwrap();
    // Should NOT directly include the CPU-only runtime (it's included via hip runtime)
    assert!(
        !result
            .c_source
            .contains("#include \"chelis_runtime.h\"\n#include \"chelis_runtime.h\""),
        "Should not double-include CPU runtime"
    );
}

// ===========================================================================
// S2: Kernel strings are const char* literals
// ===========================================================================

#[test]
fn s2_kernel_strings_are_const_char() {
    let dag = dag_add_consts();
    let result = codegen_hip(&dag, "test_s2").unwrap();
    assert!(
        result.c_source.contains("const char *"),
        "Kernel source strings should be const char* literals"
    );
}

#[test]
fn s2_kernel_strings_escape_embedded_quotes() {
    let dag = dag_add_consts();
    let result = codegen_hip(&dag, "test_quote_escape").unwrap();
    assert!(
        result
            .c_source
            .contains("extern \\\"C\\\" __global__ void kernel_add"),
        "Kernel source must escape embedded quotes inside generated C string literals"
    );
}

#[test]
fn s2_neg_const_only_dag_only_fill_kernel() {
    let mut dag = Dag::new();
    let c = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 0.0),
        vec![],
        scalar_f32(),
        None,
    );
    dag.add_root(c);
    let result = codegen_hip(&dag, "test_s2_neg").unwrap();
    // A const-only DAG should only have the fill kernel, no compute kernels
    assert!(
        result.c_source.contains("kernel_fill"),
        "Const-only DAG should have a fill kernel"
    );
    assert!(
        !result.c_source.contains("kernel_add"),
        "Const-only DAG should not have compute kernels"
    );
}

// ===========================================================================
// S3: Each elementwise op has a kernel
// ===========================================================================

#[test]
fn s3_all_elementwise_ops_emit_kernels() {
    let ops_and_names: Vec<(RiscOp, &str)> = vec![
        (RiscOp::Add, "add"),
        (RiscOp::Sub, "sub"),
        (RiscOp::Mul, "mul"),
        (RiscOp::MaxElem, "max_elem"),
        (RiscOp::MinElem, "min_elem"),
        (RiscOp::CmpLt, "cmplt"),
    ];
    for (op, name) in &ops_and_names {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        let c = dag.add_node(op.clone(), vec![a, b], scalar_f32(), None);
        dag.add_root(c);
        let result = codegen_hip(&dag, &format!("test_{name}")).unwrap();
        assert!(
            result.c_source.contains("chelis_launch_kernel"),
            "Binary op '{name}' must emit a kernel launch"
        );
    }
}

#[test]
fn direct_extrema_and_adjoint_emit_bit_preserving_kernels() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(4), None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], vec_f32(4), None);
    let g = dag.add_node(RiscOp::Load { name: "g".into() }, vec![], vec_f32(4), None);
    let sub = dag.add_node(RiscOp::Sub, vec![a, b], vec_f32(4), None);
    let max = dag.add_node(RiscOp::MaxElem, vec![a, b], vec_f32(4), None);
    let min = dag.add_node(RiscOp::MinElem, vec![a, b], vec_f32(4), None);
    let adjoint = dag.add_node(
        RiscOp::ExtremaAdjoint {
            kind: ExtremaKind::Max,
            operand: ExtremaOperand::Left,
        },
        vec![a, b, g],
        vec_f32(4),
        None,
    );
    for root in [sub, max, min, adjoint] {
        dag.add_root(root);
    }

    let source = codegen_hip(&dag, "direct_arithmetic_structure")
        .unwrap()
        .c_source;
    for kernel in [
        "kernel_sub_f32",
        "kernel_max_elem",
        "kernel_min_elem",
        "kernel_max_adjoint_left_f32",
    ] {
        assert!(source.contains(kernel), "missing {kernel}: {source}");
    }
    assert!(source.contains("bool select_left = isnan(av) || (!isnan(bv) && av >= bv);"));
    assert!(source.contains("bool select_left = isnan(av) || (!isnan(bv) && av <= bv);"));
    assert!(!source.contains("fmaxf(av, bv)"), "{source}");
    assert!(!source.contains("fminf(av, bv)"), "{source}");
}

#[test]
fn direct_signed_integer_extrema_chains_stay_on_typed_hip_kernels() {
    for (precision, suffix) in [
        (Prim::Int8, "i8"),
        (Prim::Int16, "i16"),
        (Prim::Int32, "i32"),
        (Prim::Int64, "i64"),
    ] {
        let ty = TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision,
        };
        let mut dag = Dag::new();
        let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], ty.clone(), None);
        let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], ty.clone(), None);
        let c = dag.add_node(RiscOp::Load { name: "c".into() }, vec![], ty.clone(), None);
        let maximum = dag.add_node(RiscOp::MaxElem, vec![a, b], ty.clone(), None);
        let minimum = dag.add_node(RiscOp::MinElem, vec![maximum, c], ty, None);
        dag.add_root(minimum);

        let fused = fuse(&dag);
        assert!(
            fused
                .nodes()
                .iter()
                .all(|node| !matches!(node.op, RiscOp::FusedElem { .. })),
            "{precision:?} extrema must not enter the float-only fused HIP family"
        );
        let source = codegen_hip(
            &fused,
            &format!("direct_integer_extrema_{}", precision.name()),
        )
        .unwrap_or_else(|error| panic!("{precision:?} HIP codegen failed: {error}"))
        .c_source;
        assert!(
            source.contains(&format!("kernel_max_elem_{suffix}")),
            "{precision:?}: missing typed max kernel: {source}"
        );
        assert!(
            source.contains(&format!("kernel_min_elem_{suffix}")),
            "{precision:?}: missing typed min kernel: {source}"
        );
    }
}

// chelis#178: floor_div / trunc_div emit correctly-shaped HIP kernels.
// `trunc_div` on integers is the native C `/` quotient (guarded);
// `floor_div` on integers carries the remainder-sign correction; on
// floats it is `floorf(a / b)`. This locks the kernel-string shape
// without needing a GPU.
#[test]
fn s3_floor_trunc_div_emit_typed_int_kernels() {
    // trunc_div (int): native `/`, integer-suffixed kernel name.
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_i32(4), None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], vec_i32(4), None);
    let c = dag.add_node(RiscOp::TruncDiv, vec![a, b], vec_i32(4), None);
    dag.add_root(c);
    let trunc_src = codegen_hip(&dag, "test_trunc_div").unwrap().c_source;
    assert!(
        trunc_src.contains("kernel_trunc_div_i32"),
        "trunc_div(int32) must emit a dtype-suffixed kernel; got:\n{trunc_src}"
    );

    // floor_div (int): sign-correction kernel (contains the remainder
    // modulo and the `-= 1` adjustment).
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_i32(4), None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], vec_i32(4), None);
    let c = dag.add_node(RiscOp::FloorDiv, vec![a, b], vec_i32(4), None);
    dag.add_root(c);
    let floor_src = codegen_hip(&dag, "test_floor_div").unwrap().c_source;
    assert!(
        floor_src.contains("kernel_floor_div_i32") && floor_src.contains("q -= 1"),
        "floor_div(int32) must emit a sign-corrected kernel; got:\n{floor_src}"
    );

    // floor_div (float): `floorf(a / b)`.
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(4), None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], vec_f32(4), None);
    let c = dag.add_node(RiscOp::FloorDiv, vec![a, b], vec_f32(4), None);
    dag.add_root(c);
    let floor_f_src = codegen_hip(&dag, "test_floor_div_f").unwrap().c_source;
    assert!(
        floor_f_src.contains("floorf("),
        "floor_div(f32) must emit floorf(a / b); got:\n{floor_f_src}"
    );
}

#[test]
fn s3_all_unary_ops_emit_kernels() {
    let ops_and_names: Vec<(RiscOp, &str)> = vec![
        (RiscOp::Neg, "neg"),
        (RiscOp::Exp, "exp"),
        (RiscOp::Log, "log"),
        (RiscOp::Sin, "sin"),
        (RiscOp::Sqrt, "sqrt"),
    ];
    for (op, name) in &ops_and_names {
        let mut dag = Dag::new();
        let a = dag.add_node(
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let c = dag.add_node(op.clone(), vec![a], scalar_f32(), None);
        dag.add_root(c);
        let result = codegen_hip(&dag, &format!("test_{name}")).unwrap();
        assert!(
            result.c_source.contains("chelis_launch_kernel"),
            "Unary op '{name}' must emit a kernel launch"
        );
    }
}

// ===========================================================================
// S4: Reduction kernels use naive pattern (inner loop over axis)
// ===========================================================================

#[test]
fn s4_sum_reduction_emits_kernel() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::synth_const(mat_f32(3, 4).precision, 1.0),
        vec![],
        mat_f32(3, 4),
        None,
    );
    let s = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![x],
        vec_f32(3),
        None,
    );
    dag.add_root(s);
    let result = codegen_hip(&dag, "test_sum").unwrap();
    assert!(
        result.c_source.contains("chelis_launch_kernel"),
        "Sum reduction must emit a kernel launch"
    );
}

#[test]
fn s4_max_reduce_emits_kernel() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::synth_const(mat_f32(3, 4).precision, 1.0),
        vec![],
        mat_f32(3, 4),
        None,
    );
    let m = dag.add_node(RiscOp::MaxReduce { axis: 0 }, vec![x], vec_f32(4), None);
    dag.add_root(m);
    let result = codegen_hip(&dag, "test_max_reduce").unwrap();
    assert!(
        result.c_source.contains("chelis_launch_kernel"),
        "MaxReduce must emit a kernel launch"
    );
}

// ===========================================================================
// S5: Movement ops are NOT kernels (host-side metadata only)
// ===========================================================================

#[test]
fn s5_reshape_no_kernel_launch() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::synth_const(mat_f32(2, 3).precision, 1.0),
        vec![],
        mat_f32(2, 3),
        None,
    );
    let r = dag.add_node(
        RiscOp::Reshape {
            new_shape: vec![chelis_ir::dag::RtDim::Lit(6)],
        },
        vec![x],
        vec_f32(6),
        None,
    );
    dag.add_root(r);
    let result = codegen_hip(&dag, "test_reshape").unwrap();
    // Reshape itself must not add a kernel launch
    assert!(
        result.c_source.contains("chelis_gpu_alloc_view"),
        "Reshape must use alloc_view (metadata-only), not a kernel"
    );
}

#[test]
fn s5_permute_no_kernel_launch() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::synth_const(mat_f32(2, 3).precision, 1.0),
        vec![],
        mat_f32(2, 3),
        None,
    );
    let p = dag.add_node(
        RiscOp::Permute { axes: vec![1, 0] },
        vec![x],
        mat_f32(3, 2),
        None,
    );
    dag.add_root(p);
    let result = codegen_hip(&dag, "test_permute").unwrap();
    assert!(
        result.c_source.contains("chelis_gpu_alloc_view"),
        "Permute must use alloc_view (metadata-only)"
    );
}

#[test]
fn s5_expand_no_kernel_launch() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::synth_const(vec_f32(3).precision, 1.0),
        vec![],
        vec_f32(3),
        None,
    );
    let e = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::Lit(4),
        },
        vec![x],
        mat_f32(4, 3),
        None,
    );
    dag.add_root(e);
    let result = codegen_hip(&dag, "test_expand").unwrap();
    assert!(
        result.c_source.contains("chelis_gpu_alloc_view"),
        "Expand must use alloc_view (metadata-only)"
    );
}

#[test]
fn s5_input_axis_expand_reads_witness_metadata() {
    let mut dag = Dag::new();
    let value = dag.add_node(
        RiscOp::Load {
            name: "value".into(),
        },
        vec![],
        scalar_f32(),
        None,
    );
    let witness_ty = TensorType {
        dims: vec![DimInfo::Named("n".into(), None)],
        precision: Prim::F32,
    };
    let witness = dag.add_node(
        RiscOp::Load {
            name: "witness".into(),
        },
        vec![],
        witness_ty.clone(),
        None,
    );
    let expanded = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::InputAxis {
                tensor: 1,
                axis: RtAxis::Lit(0),
            },
        },
        vec![value, witness],
        witness_ty,
        None,
    );
    dag.add_root(expanded);

    let result = codegen_hip(&dag, "test_input_axis_expand").unwrap();
    assert!(
        result.c_source.contains("inputs[1]->shape[0]"),
        "InputAxis extent must be sourced from the witness metadata: {}",
        result.c_source
    );
    assert!(
        result.c_source.contains("chelis_gpu_alloc_view"),
        "InputAxis expand remains a metadata-only view"
    );
}

#[test]
fn s5_realize_materializes_with_kernel_not_view() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(6), None);
    let s = dag.add_node(
        RiscOp::Stride {
            strides: vec![chelis_ir::dag::RtDim::Lit(2)],
        },
        vec![x],
        vec_f32(3),
        None,
    );
    let r = dag.add_node(RiscOp::Realize, vec![s], vec_f32(3), None);
    dag.add_root(r);

    let result = codegen_hip(&dag, "test_realize").unwrap();
    assert!(
        result.c_source.contains("kernel_cast"),
        "Realize must materialize through a copy-style kernel launch"
    );
    assert!(
        !result.c_source.contains(
            "chelis_gpu_alloc_view(1, (int[]){ 3 }, CHELIS_DTYPE_F32, d_t1->data, d_t1->storage_size)"
        ),
        "Realize must not lower to a metadata-only view"
    );
}

#[test]
fn s5_copy_materializes_and_drop_emits_no_kernel_or_wrapper() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(6), None);
    let copy = dag.add_node(RiscOp::Copy, vec![x], vec_f32(6), None);
    dag.add_node(RiscOp::Drop, vec![x], vec_f32(6), None);
    dag.add_root(copy);

    let result = codegen_hip(&dag, "test_copy_drop").unwrap();
    let host = host_entry_source(&result.c_source, "test_copy_drop");

    assert!(
        result.c_source.contains("kernel_cast"),
        "Copy must materialize through a copy-style kernel launch"
    );
    assert_eq!(
        host.matches("chelis_launch_kernel").count(),
        1,
        "Only Copy should launch; Drop must be non-computational:\n{host}"
    );
    assert!(
        !host.contains("d_t2"),
        "Drop should not allocate or free a device tensor wrapper:\n{host}"
    );
}

// ===========================================================================
// S6: Kernel launch uses correct grid/block (ceil(size/256))
// ===========================================================================

#[test]
fn s6_grid_block_in_launch() {
    let dag = dag_add_consts();
    let result = codegen_hip(&dag, "test_s6").unwrap();
    // Generated code should reference block size 256
    assert!(
        result.c_source.contains("256"),
        "Block size 256 should appear in generated code"
    );
}

// ===========================================================================
// S7: Host code follows DAG topological order
// ===========================================================================

#[test]
fn s7_topo_order_preserved() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 1.0),
        vec![],
        scalar_f32(),
        None,
    );
    let b = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 2.0),
        vec![],
        scalar_f32(),
        None,
    );
    let c = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32(), None);
    let d = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 3.0),
        vec![],
        scalar_f32(),
        None,
    );
    let e = dag.add_node(RiscOp::Mul, vec![c, d], scalar_f32(), None);
    dag.add_root(e);
    let result = codegen_hip(&dag, "test_topo").unwrap();
    let src = &result.c_source;
    // t0 (const), t1 (const), t2 (add), t3 (const), t4 (mul)
    // Each must appear in order
    let pos_t2 = src.find("d_t2").expect("d_t2 should exist");
    let pos_t4 = src.find("d_t4").expect("d_t4 should exist");
    assert!(pos_t2 < pos_t4, "add (t2) must appear before mul (t4)");
}

// ===========================================================================
// S8: Input/output label contract matches C backend
// ===========================================================================

#[test]
fn s8_input_labels_match() {
    let dag = dag_with_load();
    let result = codegen_hip(&dag, "test_labels").unwrap();
    assert_eq!(result.input_labels, vec!["x"]);
}

#[test]
fn s8_output_labels_root() {
    let dag = dag_add_consts();
    let result = codegen_hip(&dag, "test_labels").unwrap();
    assert_eq!(result.output_labels, vec!["root0"]);
}

#[test]
fn s8_duplicate_load_single_slot() {
    let mut dag = Dag::new();
    let x1 = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4), None);
    let x2 = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4), None);
    let out = dag.add_node(RiscOp::Add, vec![x1, x2], vec_f32(4), None);
    dag.add_root(out);
    let result = codegen_hip(&dag, "test_dup_load").unwrap();
    assert_eq!(
        result.input_labels,
        vec!["x"],
        "Repeated Load(x) shares one input slot"
    );
}

// ===========================================================================
// S9: cmplt over a bool result is rejected, not emitted at the operand width
// ===========================================================================

/// chelis#1360. This test used to assert the opposite: that the `cmplt` kernel
/// "must produce float `1.0f`/`0.0f`, not integer bool". That was correct
/// while the HIP runtime stored a bool tensor as a four-byte binary32 payload,
/// and `chelis_gpu_dtype_size(CHELIS_DTYPE_BOOL)` returned 4 to match.
///
/// chelis#1308 replaced that payload with the tagged carrier's one-byte
/// `Repr::Bool8`. The kernel side did not follow, so the assertion above went
/// on holding - `1.0f` and `0.0f` were still in the emitted source - while the
/// emitted program wrote `N * 4` bytes into an `N * 1` byte `hipMalloc` and
/// read back the low bytes of the float stream. Both halves of the assertion
/// were true and the program was corrupt, which is why the test is now the
/// rejection rather than the spelling.
#[test]
fn s9_cmplt_bool_result_is_rejected() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 1.0),
        vec![],
        scalar_f32(),
        None,
    );
    let b = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 2.0),
        vec![],
        scalar_f32(),
        None,
    );
    let c = dag.add_node(
        RiscOp::CmpLt,
        vec![a, b],
        TensorType {
            dims: vec![],
            precision: Prim::Bool,
        },
        None,
    );
    dag.add_root(c);
    let error = match codegen_hip(&dag, "test_cmplt") {
        Err(error) => error,
        Ok(_) => panic!(
            "a bool result is one byte and the cmplt template writes at the \
             operand width; emitting it overruns the allocation (chelis#1360)"
        ),
    };
    assert!(
        format!("{error:?}").contains("bool"),
        "the rejection must name the offending dtype; got: {error:?}"
    );
}

// ===========================================================================
// S10: Device helpers in every compute kernel
// ===========================================================================

#[test]
fn s10_device_helpers_present() {
    let dag = dag_add_consts();
    let result = codegen_hip(&dag, "test_helpers").unwrap();
    assert!(
        result.c_source.contains("chelis_flat_to_indices"),
        "Device helper chelis_flat_to_indices must be in kernel source"
    );
    assert!(
        result.c_source.contains("chelis_indices_to_flat"),
        "Device helper chelis_indices_to_flat must be in kernel source"
    );
    assert!(
        result.c_source.contains("chelis_gpu_failure = 0"),
        "Kernel source must declare the debug failure flag"
    );
    assert!(
        result.c_source.contains("CHELIS_GUARD_INDEX"),
        "Kernel source must include bounds-guard instrumentation"
    );
}

// ===========================================================================
// S11: Kernel JIT uses static caching
// ===========================================================================

#[test]
fn s11_static_module_caching() {
    let dag = dag_add_consts();
    let result = codegen_hip(&dag, "test_cache").unwrap();
    assert!(
        result.c_source.contains("static hipModule_t"),
        "Kernel modules must be cached with 'static hipModule_t'"
    );
}

#[test]
fn s11_launches_reset_and_check_failure_flag() {
    let dag = dag_add_consts();
    let result = codegen_hip(&dag, "test_failure_checks").unwrap();
    let src = &result.c_source;
    let launch_count = src.matches("chelis_launch_kernel").count();
    assert_eq!(
        src.matches("chelis_prepare_kernel_launch").count(),
        launch_count,
        "Every kernel launch must reset the debug failure flag"
    );
    assert_eq!(
        src.matches("chelis_finalize_kernel_launch").count(),
        launch_count,
        "Every kernel launch must check the debug failure flag"
    );

    // WS-A2: kernel names are now precision-suffixed (`mod_kernel_fill_f32` /
    // `mod_kernel_fill_f64`); the fixture is f32 so check that variant.
    let prepare_pos = src
        .find("chelis_prepare_kernel_launch(mod_kernel_fill_f32)")
        .expect("fill reset present");
    let launch_pos = src
        .find("chelis_launch_kernel(mod_kernel_fill_f32")
        .expect("fill launch present");
    let finalize_pos = src
        .find("chelis_finalize_kernel_launch(mod_kernel_fill_f32")
        .expect("fill finalize present");
    assert!(
        prepare_pos < launch_pos && launch_pos < finalize_pos,
        "Failure flag reset/check must bracket the kernel launch"
    );
}

// ===========================================================================
// S12: Host↔device transfers bracket the function
// ===========================================================================

#[test]
fn s12_transfers_present() {
    let dag = dag_with_load();
    let result = codegen_hip(&dag, "test_transfer").unwrap();
    assert!(
        result.c_source.contains("chelis_host_to_device"),
        "Input transfer (host→device) must be present"
    );
    assert!(
        result.c_source.contains("chelis_device_to_host"),
        "Output transfer (device→host) must be present"
    );
}

#[test]
fn s12_transfer_order() {
    let dag = dag_with_load();
    let result = codegen_hip(&dag, "test_transfer_order").unwrap();
    let src = &result.c_source;
    let h2d_pos = src.find("chelis_host_to_device").expect("h2d present");
    let d2h_pos = src.find("chelis_device_to_host").expect("d2h present");
    assert!(
        h2d_pos < d2h_pos,
        "host_to_device must come before device_to_host"
    );
}

#[test]
fn s12_duplicate_load_transfers_once() {
    let mut dag = Dag::new();
    let x0 = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4), None);
    let x1 = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4), None);
    let out = dag.add_node(RiscOp::Add, vec![x0, x1], vec_f32(4), None);
    dag.add_root(out);

    let result = codegen_hip(&dag, "test_dup_transfer_once").unwrap();
    assert_eq!(
        result.c_source.matches("chelis_host_to_device").count(),
        1,
        "Repeated loads of the same input should share one host→device transfer"
    );
}

#[test]
fn s12_peak_estimate_reported() {
    let dag = dag_with_load();
    let result = codegen_hip(&dag, "test_peak_estimate").unwrap();
    assert!(
        result
            .peak_device_bytes_estimate
            .expect("concrete DAG should report bytes")
            > 0,
        "HIP codegen should report a nonzero peak device-memory estimate"
    );
    assert!(!result.peak_device_bytes_formula.is_empty());
}

#[test]
fn s12_symbolic_peak_memory_reports_formula_without_fake_estimate() {
    let mut dag = Dag::new();
    let symbolic = TensorType {
        dims: vec![DimInfo::Named("batch".into(), None)],
        precision: Prim::F32,
    };
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        symbolic.clone(),
        None,
    );
    dag.add_root(x);

    let result = codegen_hip(&dag, "test_symbolic_peak").unwrap();
    assert_eq!(result.peak_device_bytes_estimate, None);
    assert_eq!(result.peak_device_bytes_formula, "(batch * 4)");
    assert_eq!(
        result
            .peak_device_bytes_at(&chelis_unord::UnordMap::from([(
                String::from("batch"),
                64usize,
            )]))
            .unwrap(),
        256
    );
}

#[test]
fn s12_symbolic_repeated_occurrences_check_every_non_canonical_input() {
    let mut dag = Dag::new();
    let symbolic = TensorType {
        dims: vec![DimInfo::Named("batch".into(), None)],
        precision: Prim::F32,
    };
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        symbolic.clone(),
        None,
    );
    let y = dag.add_node(
        RiscOp::Load { name: "y".into() },
        vec![],
        symbolic.clone(),
        None,
    );
    let z = dag.add_node(
        RiscOp::Load { name: "z".into() },
        vec![],
        symbolic.clone(),
        None,
    );
    let xy = dag.add_node(RiscOp::Add, vec![x, y], symbolic.clone(), None);
    let xyz = dag.add_node(RiscOp::Add, vec![xy, z], symbolic, None);
    dag.add_root(xyz);

    let result = codegen_hip(&dag, "test_symbolic_repeats").unwrap();
    assert!(result.c_source.contains("int batch = inputs[0]->shape[0];"));
    assert!(
        result.c_source.contains("inputs[1]->shape[0] != batch"),
        "second occurrence must be checked against the canonical binding"
    );
    assert!(
        result.c_source.contains("inputs[2]->shape[0] != batch"),
        "third occurrence must also be checked against the canonical binding"
    );
}

#[test]
fn s12_slot_backed_kernels_iterate_over_logical_size_after_dce() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::synth_const(vec_f32(8).precision, 1.0),
        vec![],
        vec_f32(8),
        None,
    );
    let b = dag.add_node(
        RiscOp::synth_const(vec_f32(8).precision, 2.0),
        vec![],
        vec_f32(8),
        None,
    );
    let _wide = dag.add_node(RiscOp::Add, vec![a, b], vec_f32(8), None);
    let small = dag.add_node(
        RiscOp::synth_const(vec_f32(4).precision, 3.0),
        vec![],
        vec_f32(4),
        None,
    );
    let out = dag.add_node(RiscOp::Neg, vec![small], vec_f32(4), None);
    dag.add_root(out);

    let result = codegen_hip(&dag, "test_logical_size").unwrap();
    let src = &result.c_source;

    assert!(
        src.contains("int fill_size = d_t"),
        "Slot-backed fill kernels must iterate over logical size"
    );
    assert!(
        !src.lines().any(|line| {
            let trimmed = line.trim_start();
            trimmed.starts_with("int fill_size =") && trimmed.contains("->storage_size;")
        }),
        "Fill kernels must not iterate over slot capacity"
    );
    assert!(
        src.contains("_size = d_t") && src.contains("->size;"),
        "Elementwise outputs must use logical size even when backed by a reused slot"
    );
}

// ===========================================================================
// S13: Cleanup frees GPU tensors; views don't double-free
// ===========================================================================

#[test]
fn s13_cleanup_frees_intermediates() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 1.0),
        vec![],
        scalar_f32(),
        None,
    );
    let b = dag.add_node(
        RiscOp::synth_const(scalar_f32().precision, 2.0),
        vec![],
        scalar_f32(),
        None,
    );
    let c = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32(), None);
    dag.add_root(c);
    let result = codegen_hip(&dag, "test_cleanup").unwrap();
    // a (t0) and b (t1) are intermediates, so both their wrappers and slot owners are freed.
    assert!(
        result.c_source.contains("chelis_gpu_free_view(d_t0)"),
        "Intermediate wrapper t0 must be freed"
    );
    assert!(
        result.c_source.contains("chelis_gpu_free_view(d_t1)"),
        "Intermediate wrapper t1 must be freed"
    );
    assert!(
        result.c_source.contains("chelis_gpu_free(chelis_slot0)"),
        "Intermediate slot 0 must be freed"
    );
    assert!(
        result.c_source.contains("chelis_gpu_free(chelis_slot1)"),
        "Intermediate slot 1 must be freed"
    );
}

#[test]
fn s13_outputs_not_freed() {
    let dag = dag_add_consts();
    let result = codegen_hip(&dag, "test_no_free_output").unwrap();
    // The output wrapper is freed after the host copy, and the backing slot is freed once at the end.
    assert!(
        result.c_source.contains("chelis_gpu_free_view(d_t2)"),
        "Output wrapper must be freed after the output transfer"
    );
    assert!(
        result.c_source.contains("chelis_gpu_free(chelis_slot0)")
            || result.c_source.contains("chelis_gpu_free(chelis_slot1)")
            || result.c_source.contains("chelis_gpu_free(chelis_slot2)"),
        "At least one backing slot must be freed during cleanup"
    );
}

#[test]
fn s13_input_copies_are_freed() {
    let dag = dag_with_load();
    let result = codegen_hip(&dag, "test_free_input_copy").unwrap();
    assert!(
        result.c_source.contains("chelis_gpu_free_view(d_t0)"),
        "The per-node load wrapper must be freed"
    );
    assert!(
        result.c_source.contains("chelis_gpu_free(chelis_slot0)"),
        "The unique input device slot must be freed"
    );
}

#[test]
fn s13_views_use_view_free() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::synth_const(mat_f32(2, 3).precision, 1.0),
        vec![],
        mat_f32(2, 3),
        None,
    );
    let p = dag.add_node(
        RiscOp::Permute { axes: vec![1, 0] },
        vec![x],
        mat_f32(3, 2),
        None,
    );
    let c = dag.add_node(
        RiscOp::synth_const(mat_f32(3, 2).precision, 2.0),
        vec![],
        mat_f32(3, 2),
        None,
    );
    let out = dag.add_node(RiscOp::Add, vec![p, c], mat_f32(3, 2), None);
    dag.add_root(out);
    let result = codegen_hip(&dag, "test_view_free").unwrap();
    // Permute (t1) is a view — must use chelis_gpu_free_view, NOT chelis_gpu_free
    assert!(
        result.c_source.contains("chelis_gpu_free_view(d_t1)"),
        "View (permute) must use chelis_gpu_free_view, not chelis_gpu_free"
    );
    assert!(
        !result.c_source.contains("chelis_gpu_free(d_t1)"),
        "View (permute) must NOT use chelis_gpu_free (would double-free device memory)"
    );
}

#[test]
fn s14_generated_hip_source_compiles_when_hipcc_available() {
    if !hipcc_available() {
        eprintln!("skipping: hipcc not available");
        return;
    }

    let dag = dag_add_consts();
    let result = codegen_hip(&dag, "test_compile").unwrap();
    let tmp = tempfile::tempdir().expect("tempdir");

    let hip_rt = hip_runtime_src_dir();
    write_temp_file(
        tmp.path(),
        "chelis_hip_runtime.h",
        &fs::read_to_string(hip_rt.join("chelis_hip_runtime.h")).expect("hip runtime header"),
    );
    copy_runtime_artifacts(tmp.path());
    write_temp_file(tmp.path(), "model.cpp", &result.c_source);
    let main_cpp = r#"
#include "chelis_runtime.h"
extern "C" void test_compile(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {
    chelis_tensor *outputs[1] = {0};
    test_compile(NULL, 0, outputs, 1);
    chelis_free(outputs[0]);
    return 0;
}
"#;
    write_temp_file(tmp.path(), "main.cpp", main_cpp);

    let bin_path = tmp.path().join("hip_compile_smoke");
    let output = Command::new("hipcc")
        .arg("-O2")
        .arg(tmp.path().join("main.cpp"))
        .arg(tmp.path().join("model.cpp"))
        .arg(format!("-L{}", tmp.path().display()))
        .arg("-lchelis_runtime")
        .arg("-lpthread")
        .arg("-ldl")
        .arg("-lhiprtc")
        .arg("-o")
        .arg(&bin_path)
        .output()
        .expect("run hipcc");
    assert!(
        output.status.success(),
        "hipcc failed:\nstderr: {}\nsource:\n{}",
        String::from_utf8_lossy(&output.stderr),
        result.c_source
    );
}

// ===========================================================================
// S15: Phase 1d optimized reductions + hipBLAS
// ===========================================================================

#[test]
fn s15_segmented_reduction_strategy_names_track_axis_size() {
    let mut tiny = Dag::new();
    let x_tiny = tiny.add_node(
        RiscOp::synth_const(mat_f32(3, 8).precision, 1.0),
        vec![],
        mat_f32(3, 8),
        None,
    );
    let sum_tiny = tiny.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![x_tiny],
        vec_f32(3),
        None,
    );
    tiny.add_root(sum_tiny);
    let tiny_result = codegen_hip(&tiny, "test_tiny_reduce").unwrap();
    assert!(
        tiny_result.c_source.contains("kernel_sum_ax1"),
        "generic symbolic-capable reductions should emit the axis-specific kernel"
    );

    let mut small = Dag::new();
    let x_small = small.add_node(
        RiscOp::synth_const(mat_f32(3, 16).precision, 1.0),
        vec![],
        mat_f32(3, 16),
        None,
    );
    let sum_small = small.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![x_small],
        vec_f32(3),
        None,
    );
    small.add_root(sum_small);
    let small_result = codegen_hip(&small, "test_small_reduce").unwrap();
    assert!(
        small_result.c_source.contains("kernel_sum_ax1"),
        "axis_size=16 should use the same runtime-sized reduction kernel"
    );

    let mut large = Dag::new();
    let x_large = large.add_node(
        RiscOp::synth_const(mat_f32(3, 128).precision, 1.0),
        vec![],
        mat_f32(3, 128),
        None,
    );
    let sum_large = large.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![x_large],
        vec_f32(3),
        None,
    );
    large.add_root(sum_large);
    let large_result = codegen_hip(&large, "test_large_reduce").unwrap();
    assert!(
        large_result.c_source.contains("kernel_sum_ax1"),
        "axis_size=128 should use the same runtime-sized reduction kernel"
    );
    assert!(
        large_result
            .c_source
            .contains("chelis_launch_kernel(mod_kernel_sum_ax1"),
        "reductions should launch the generic runtime-sized kernel"
    );
}

#[test]
fn s15_scalar_reduction_uses_staged_kernels_and_estimate() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::synth_const(vec_f32(1024).precision, 1.0),
        vec![],
        vec_f32(1024),
        None,
    );
    let sum = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![x],
        scalar_f32(),
        None,
    );
    dag.add_root(sum);
    let result = codegen_hip(&dag, "test_scalar_stage").unwrap();

    assert!(
        result.c_source.contains("kernel_sum_ax0"),
        "scalar reductions should use the generic runtime-sized reduction kernel"
    );
    assert_eq!(
        result.peak_device_bytes_estimate,
        Some(4100),
        "generic scalar reductions should report only slot-plan bytes without staged scratch"
    );
}

#[test]
fn s15_matmul_emits_hipblas_and_link_flag() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::synth_const(mat_f32(2, 3).precision, 1.0),
        vec![],
        mat_f32(2, 3),
        None,
    );
    let b = dag.add_node(
        RiscOp::synth_const(mat_f32(3, 4).precision, 1.0),
        vec![],
        mat_f32(3, 4),
        None,
    );
    let ea = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: chelis_ir::dag::RtDim::Lit(4),
        },
        vec![a],
        tensor3_f32(2, 3, 4),
        None,
    );
    let eb = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::Lit(2),
        },
        vec![b],
        tensor3_f32(2, 3, 4),
        None,
    );
    let mul = dag.add_node(RiscOp::Mul, vec![ea, eb], tensor3_f32(2, 3, 4), None);
    let sum = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![mul],
        mat_f32(2, 4),
        None,
    );
    dag.add_root(sum);
    let result = codegen_hip(&dag, "test_hipblas_matmul").unwrap();

    assert!(
        result.c_source.contains("chelis_hipblas_sgemm_row_major"),
        "contiguous rank-2 matmul should lower to hipBLAS"
    );
    assert!(
        result.link_flags.iter().any(|flag| flag == "-lhipblas"),
        "hipBLAS specialization must surface the extra link flag"
    );
}

#[test]
fn sparse_gather_i64_emits_typed_hip_kernel_and_runtime_allocation() {
    let mut dag = Dag::new();
    let table = dag.add_node(
        RiscOp::Load {
            name: "table".into(),
        },
        vec![],
        mat_f32(8, 4),
        None,
    );
    let indices = dag.add_node(
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        vec_i64(3),
        None,
    );
    let out = dag.add_node(
        RiscOp::Gather { axis: 0 },
        vec![table, indices],
        mat_f32(3, 4),
        None,
    );
    dag.add_root(out);

    let result = codegen_hip(&dag, "test_sparse_gather_i64").unwrap();
    assert!(result.c_source.contains("kernel_gather_i64"));
    assert!(result.c_source.contains("const long long *indices"));
    assert!(result.c_source.contains("CHELIS_DTYPE_I64"));
    assert!(
        result
            .c_source
            .contains("chelis_gpu_dtype_size(dst->dtype)")
            || hip_runtime_header().contains("chelis_gpu_dtype_size")
    );
    assert!(
        !result
            .c_source
            .contains("sparse gather/scatter kernels are not implemented")
    );
}

#[test]
fn sparse_scatter_add_i32_emits_atomic_add_kernel() {
    let mut dag = Dag::new();
    let target = dag.add_node(
        RiscOp::Load {
            name: "target".into(),
        },
        vec![],
        mat_f32(3, 2),
        None,
    );
    let indices = dag.add_node(
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        vec_i32(4),
        None,
    );
    let updates = dag.add_node(
        RiscOp::Load {
            name: "updates".into(),
        },
        vec![],
        mat_f32(4, 2),
        None,
    );
    let out = dag.add_node(
        RiscOp::ScatterAdd { axis: 0 },
        vec![target, indices, updates],
        mat_f32(3, 2),
        None,
    );
    dag.add_root(out);

    let result = codegen_hip(&dag, "test_sparse_scatter_i32").unwrap();
    assert!(result.c_source.contains("kernel_scatter_add_i32"));
    assert!(result.c_source.contains("const int *indices"));
    assert!(
        result
            .c_source
            .contains("atomicAdd(&out[dst], updates[i]);")
    );
    assert!(result.c_source.contains("hipMemcpyDeviceToDevice"));
}

#[test]
fn s15_batched_matmul_emits_hipblas_strided_batched_helper_and_link_flag() {
    let mut dag = Dag::new();
    let a_ty = tensor4_f32(2, 3, 4, 5);
    let b_ty = tensor4_f32(2, 3, 5, 6);
    let a = dag.add_node(
        RiscOp::synth_const(a_ty.precision, 1.0),
        vec![],
        a_ty.clone(),
        None,
    );
    let b = dag.add_node(
        RiscOp::synth_const(b_ty.precision, 1.0),
        vec![],
        b_ty.clone(),
        None,
    );
    let out = chelis_ir::tier2::lower_matmul(&mut dag, a, b, &a_ty, &b_ty, None);
    dag.add_root(out);
    let result = codegen_hip(&dag, "test_hipblas_batched_matmul").unwrap();

    assert!(
        result
            .c_source
            .contains("chelis_hipblas_sgemm_strided_batched_row_major"),
        "eligible rank-4 batched matmul should lower to the hipBLAS strided-batched helper"
    );
    assert!(
        result.c_source.contains(", 20LL, 30LL, 24LL);"),
        "strided-batched helper call should pass concrete row-major per-batch strides"
    );
    assert!(
        !result
            .c_source
            .contains("chelis_hipblas_sgemm_batched_row_major"),
        "eligible batched matmul should not use the per-batch helper loop"
    );
    assert!(
        !result.c_source.contains("kernel_sum_ax3"),
        "batched BLAS specialization should remove the generic contraction reduction"
    );
    assert!(
        result.link_flags.iter().any(|flag| flag == "-lhipblas"),
        "batched hipBLAS specialization must surface the extra link flag"
    );
}

#[test]
fn s15_batched_matmul_symbolic_batch_emits_strided_batched_helper() {
    let mut dag = Dag::new();
    let a_ty = TensorType {
        dims: vec![
            DimInfo::Named("batch".into(), None),
            DimInfo::Lit(4),
            DimInfo::Lit(5),
        ],
        precision: Prim::F32,
    };
    let b_ty = TensorType {
        dims: vec![
            DimInfo::Named("batch".into(), None),
            DimInfo::Lit(5),
            DimInfo::Lit(6),
        ],
        precision: Prim::F32,
    };
    let out_ty = TensorType {
        dims: vec![
            DimInfo::Named("batch".into(), None),
            DimInfo::Lit(4),
            DimInfo::Lit(6),
        ],
        precision: Prim::F32,
    };
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], a_ty, None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], b_ty, None);
    let out = dag.add_node(
        RiscOp::BlasMatmul {
            batch_dims: vec![chelis_ir::dag::DimExpr::Sym("batch".into())],
            m: chelis_ir::dag::DimExpr::Concrete(4),
            n: chelis_ir::dag::DimExpr::Concrete(6),
            k: chelis_ir::dag::DimExpr::Concrete(5),
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![a, b],
        out_ty,
        None,
    );
    dag.add_root(out);
    let result = codegen_hip(&dag, "test_hipblas_symbolic_batch_strided_matmul").unwrap();

    assert!(
        result
            .c_source
            .contains("chelis_hipblas_sgemm_strided_batched_row_major"),
        "symbolic batch with concrete matrix dimensions should use strided-batched hipBLAS"
    );
    assert!(
        result.c_source.contains(", batch, 20LL, 30LL, 24LL);"),
        "runtime symbolic batch count should be passed with concrete matrix strides"
    );
}

#[test]
fn s15_batched_matmul_symbolic_matrix_dim_uses_helper_loop_fallback() {
    let mut dag = Dag::new();
    let a_ty = TensorType {
        dims: vec![
            DimInfo::Named("batch".into(), None),
            DimInfo::Named("m".into(), None),
            DimInfo::Named("k".into(), None),
        ],
        precision: Prim::F32,
    };
    let b_ty = TensorType {
        dims: vec![
            DimInfo::Named("batch".into(), None),
            DimInfo::Named("k".into(), None),
            DimInfo::Named("n".into(), None),
        ],
        precision: Prim::F32,
    };
    let out_ty = TensorType {
        dims: vec![
            DimInfo::Named("batch".into(), None),
            DimInfo::Named("m".into(), None),
            DimInfo::Named("n".into(), None),
        ],
        precision: Prim::F32,
    };
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], a_ty, None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], b_ty, None);
    let out = dag.add_node(
        RiscOp::BlasMatmul {
            batch_dims: vec![chelis_ir::dag::DimExpr::Sym("batch".into())],
            m: chelis_ir::dag::DimExpr::Sym("m".into()),
            n: chelis_ir::dag::DimExpr::Sym("n".into()),
            k: chelis_ir::dag::DimExpr::Sym("k".into()),
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![a, b],
        out_ty,
        None,
    );
    dag.add_root(out);
    let result = codegen_hip(&dag, "test_hipblas_batched_loop_fallback").unwrap();

    assert!(
        result
            .c_source
            .contains("chelis_hipblas_sgemm_batched_row_major"),
        "symbolic matrix dimensions should preserve the existing helper-loop fallback"
    );
    assert!(
        !result
            .c_source
            .contains("chelis_hipblas_sgemm_strided_batched_row_major"),
        "fallback path must not call strided-batched hipBLAS when matrix strides are not concrete"
    );
    assert!(
        result.link_flags.iter().any(|flag| flag == "-lhipblas"),
        "helper-loop fallback still requires hipBLAS"
    );
}

#[test]
fn s15_batched_matmul_noncontiguous_batch_layout_uses_helper_loop_fallback() {
    let mut dag = Dag::new();
    let base_a = dag.add_node(
        RiscOp::synth_const(mat_f32(4, 5).precision, 1.0),
        vec![],
        mat_f32(4, 5),
        None,
    );
    let a = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::Lit(3),
        },
        vec![base_a],
        tensor3_f32(3, 4, 5),
        None,
    );
    let b = dag.add_node(
        RiscOp::synth_const(tensor3_f32(3, 5, 6).precision, 1.0),
        vec![],
        tensor3_f32(3, 5, 6),
        None,
    );
    let out = dag.add_node(
        RiscOp::BlasMatmul {
            batch_dims: vec![chelis_ir::dag::DimExpr::Concrete(3)],
            m: chelis_ir::dag::DimExpr::Concrete(4),
            n: chelis_ir::dag::DimExpr::Concrete(6),
            k: chelis_ir::dag::DimExpr::Concrete(5),
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![a, b],
        tensor3_f32(3, 4, 6),
        None,
    );
    dag.add_root(out);
    let result = codegen_hip(&dag, "test_hipblas_noncontiguous_batch_loop_fallback").unwrap();

    assert!(
        result
            .c_source
            .contains("chelis_hipblas_sgemm_batched_row_major"),
        "non-contiguous leading batch layout should preserve the helper-loop fallback"
    );
    assert!(
        !result
            .c_source
            .contains("chelis_hipblas_sgemm_strided_batched_row_major"),
        "non-contiguous leading batch layout must not call strided-batched hipBLAS"
    );
}

#[test]
fn s15_noncontiguous_matmul_falls_back_to_generic_reduction() {
    let mut dag = Dag::new();
    let base_a = dag.add_node(
        RiscOp::synth_const(mat_f32(3, 2).precision, 1.0),
        vec![],
        mat_f32(3, 2),
        None,
    );
    let a = dag.add_node(
        RiscOp::Permute { axes: vec![1, 0] },
        vec![base_a],
        mat_f32(2, 3),
        None,
    );
    let b = dag.add_node(
        RiscOp::synth_const(mat_f32(3, 4).precision, 1.0),
        vec![],
        mat_f32(3, 4),
        None,
    );
    let ea = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: chelis_ir::dag::RtDim::Lit(4),
        },
        vec![a],
        tensor3_f32(2, 3, 4),
        None,
    );
    let eb = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::Lit(2),
        },
        vec![b],
        tensor3_f32(2, 3, 4),
        None,
    );
    let mul = dag.add_node(RiscOp::Mul, vec![ea, eb], tensor3_f32(2, 3, 4), None);
    let sum = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![mul],
        mat_f32(2, 4),
        None,
    );
    dag.add_root(sum);
    let result = codegen_hip(&dag, "test_generic_matmul").unwrap();

    assert!(
        !result.c_source.contains("chelis_hipblas_sgemm_row_major"),
        "non-contiguous matmul operands must fall back to the generic reduction path"
    );
    assert!(
        result.c_source.contains("kernel_sum_ax1"),
        "fallback matmul should still emit the generic reduction kernel"
    );
}

#[test]
fn s15_device_entrypoint_is_emitted_for_direct_gpu_execution() {
    let dag = dag_with_load();
    let result = codegen_hip(&dag, "test_device_entry").unwrap();

    assert!(
        result
            .c_source
            .contains("extern \"C\" void test_device_entry_device(chelis_gpu_tensor **inputs, int n_in, chelis_gpu_tensor **outputs, int n_out)"),
        "HIP codegen must emit the device-native ABI entrypoint for Python direct execution"
    );
    assert!(
        result
            .c_source
            .contains("outputs[0] = chelis_gpu_clone(d_t"),
        "device-native outputs must be surfaced as owned GPU tensors"
    );
}

// ===========================================================================
// SF1: FusedElem emits single kernel launch
// ===========================================================================

#[test]
fn sf1_fused_elem_single_kernel_launch() {
    // Build a DAG with add→neg, fuse it, codegen_hip, count kernel launches.
    // The fused chain should produce exactly 1 kernel launch for the fused ops
    // (plus 2 fill launches for the constants).
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::synth_const(vec_f32(4).precision, 1.0),
        vec![],
        vec_f32(4),
        None,
    );
    let b = dag.add_node(
        RiscOp::synth_const(vec_f32(4).precision, 2.0),
        vec![],
        vec_f32(4),
        None,
    );
    let c = dag.add_node(RiscOp::Add, vec![a, b], vec_f32(4), None);
    let d = dag.add_node(RiscOp::Neg, vec![c], vec_f32(4), None);
    dag.add_root(d);

    let fused = fuse(&dag);
    let result = codegen_hip(&fused, "test_sf1").unwrap();
    let host_src = host_entry_source(&result.c_source, "test_sf1");

    // Count kernel launches: should be 2 fills + 1 fused = 3 total
    let launch_count = host_src.matches("chelis_launch_kernel").count();
    // Without fusion we'd have 2 fills + add + neg = 4 launches.
    // With fusion: 2 fills + 1 fused = 3.
    assert_eq!(
        launch_count, 3,
        "Fused add→neg should produce 3 kernel launches (2 fill + 1 fused), got {launch_count}"
    );
}

// ===========================================================================
// SF2: Fused kernel has chained computation in body
// ===========================================================================

#[test]
fn sf2_fused_kernel_chained_computation() {
    // The generated fused kernel source should contain chained register operations.
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::synth_const(vec_f32(4).precision, 1.0),
        vec![],
        vec_f32(4),
        None,
    );
    let b = dag.add_node(
        RiscOp::synth_const(vec_f32(4).precision, 2.0),
        vec![],
        vec_f32(4),
        None,
    );
    let c = dag.add_node(RiscOp::Add, vec![a, b], vec_f32(4), None);
    let d = dag.add_node(RiscOp::Neg, vec![c], vec_f32(4), None);
    dag.add_root(d);

    let fused = fuse(&dag);
    let result = codegen_hip(&fused, "test_sf2").unwrap();
    let src = &result.c_source;

    // The embedded kernel source (escaped in a C string literal) should contain
    // chained register variables: float v0 = ... and float v1 = ...
    assert!(
        src.contains("float v0 ="),
        "Fused kernel must contain 'float v0 =' for first step"
    );
    assert!(
        src.contains("float v1 ="),
        "Fused kernel must contain 'float v1 =' for second step"
    );
}

// ===========================================================================
// SF3: No intermediate GPU allocation for fused chain
// ===========================================================================

#[test]
fn sf3_no_intermediate_alloc_in_fused_chain() {
    // Between the fused kernel alloc and its launch, there should be no extra
    // chelis_gpu_alloc calls (the intermediate is computed in registers).
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::synth_const(vec_f32(4).precision, 1.0),
        vec![],
        vec_f32(4),
        None,
    );
    let b = dag.add_node(
        RiscOp::synth_const(vec_f32(4).precision, 2.0),
        vec![],
        vec_f32(4),
        None,
    );
    let c = dag.add_node(RiscOp::Add, vec![a, b], vec_f32(4), None);
    let d = dag.add_node(RiscOp::Neg, vec![c], vec_f32(4), None);
    dag.add_root(d);

    let fused = fuse(&dag);
    let result = codegen_hip(&fused, "test_sf3").unwrap();
    let host_src = host_entry_source(&result.c_source, "test_sf3");

    // Without fusion: 2 const allocs + add alloc + neg alloc = 4 allocs.
    // With fusion: 2 const allocs + 1 fused output alloc = 3 allocs.
    let alloc_count = host_src.matches("chelis_gpu_alloc(").count();
    assert_eq!(
        alloc_count, 3,
        "Fused chain should have 3 GPU allocs (2 const + 1 fused output), got {alloc_count}"
    );
}

// ===========================================================================
// SFR1: Elementwise→reduction fusion eliminates intermediate buffer
// ===========================================================================

#[test]
fn sfr1_fused_elem_into_reduction_no_intermediate_alloc() {
    // add(x, const) → neg → sum should fuse: add→neg becomes FusedElem,
    // then the FusedElem feeds sum as sole consumer → inlined into reduction.
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        mat_f32(3, 4),
        None,
    );
    let c = dag.add_node(
        RiscOp::synth_const(mat_f32(3, 4).precision, 1.0),
        vec![],
        mat_f32(3, 4),
        None,
    );
    let added = dag.add_node(RiscOp::Add, vec![x, c], mat_f32(3, 4), None);
    let negated = dag.add_node(RiscOp::Neg, vec![added], mat_f32(3, 4), None);
    let summed = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![negated],
        vec_f32(3),
        None,
    );
    dag.add_root(summed);

    let fused = fuse(&dag);
    let result = codegen_hip(&fused, "test_sfr1").unwrap();
    let src = host_entry_source(&result.c_source, "test_sfr1");

    // With elem→elem fusion: add→neg becomes FusedElem.
    // With elem→reduction fusion: the FusedElem is inlined into the sum kernel.
    // Result: 1 const fill + 1 fused reduction = 2 kernel launches.
    let launch_count = src.matches("chelis_launch_kernel").count();
    assert_eq!(
        launch_count, 2,
        "Fused add→neg→sum should produce 2 kernel launches (1 fill + 1 fused reduce), got {launch_count}"
    );

    // The fused reduction kernel name should appear in the source
    assert!(
        src.contains("kernel_fused_sum_"),
        "Should contain a fused sum kernel name"
    );

    // The fused reduction kernel should read from external inputs (ext0, ext1)
    // embedded in the kernel string
    assert!(
        src.contains("ext0") && src.contains("ext1"),
        "Fused reduction kernel should reference external inputs ext0, ext1"
    );
}

// ===========================================================================
// SFR2: Fused reduction with max_reduce
// ===========================================================================

#[test]
fn sfr2_fused_elem_into_max_reduce() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        mat_f32(3, 4),
        None,
    );
    let c = dag.add_node(
        RiscOp::synth_const(mat_f32(3, 4).precision, 1.0),
        vec![],
        mat_f32(3, 4),
        None,
    );
    let added = dag.add_node(RiscOp::Add, vec![x, c], mat_f32(3, 4), None);
    let negated = dag.add_node(RiscOp::Neg, vec![added], mat_f32(3, 4), None);
    let maxed = dag.add_node(
        RiscOp::MaxReduce { axis: 1 },
        vec![negated],
        vec_f32(3),
        None,
    );
    dag.add_root(maxed);

    let fused = fuse(&dag);
    let result = codegen_hip(&fused, "test_sfr2").unwrap();
    let src = host_entry_source(&result.c_source, "test_sfr2");

    assert!(
        src.contains("kernel_fused_maxred_"),
        "Should contain a fused max-reduce kernel name"
    );

    let launch_count = src.matches("chelis_launch_kernel").count();
    assert_eq!(
        launch_count, 2,
        "Fused add→neg→max_reduce should produce 2 launches (1 fill + 1 fused reduce), got {launch_count}"
    );
}

// ===========================================================================
// SFR3: Multi-consumer FusedElem is NOT inlined into reduction
// ===========================================================================

#[test]
fn sfr3_multi_consumer_fused_elem_not_inlined() {
    // If the FusedElem has multiple consumers, it must be materialized.
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        mat_f32(3, 4),
        None,
    );
    let c = dag.add_node(
        RiscOp::synth_const(mat_f32(3, 4).precision, 1.0),
        vec![],
        mat_f32(3, 4),
        None,
    );
    let added = dag.add_node(RiscOp::Add, vec![x, c], mat_f32(3, 4), None);
    let negated = dag.add_node(RiscOp::Neg, vec![added], mat_f32(3, 4), None);
    // Two consumers of the fused chain output:
    let summed = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![negated],
        vec_f32(3),
        None,
    );
    dag.add_root(negated); // negated is also a root → 2 consumers
    dag.add_root(summed);

    let fused = fuse(&dag);
    let result = codegen_hip(&fused, "test_sfr3").unwrap();
    let src = &result.c_source;

    // Since the FusedElem output is both a root and consumed by the sum,
    // it should NOT be inlined — we should see a normal sum kernel.
    assert!(
        !src.contains("kernel_fused_sum_"),
        "Multi-consumer FusedElem should NOT be inlined into reduction"
    );
    assert!(
        src.contains("kernel_sum_ax"),
        "Should use standard reduction kernel for multi-consumer case"
    );
}

#[test]
fn sfr4_realize_blocks_fused_kernel_emission() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4), None);
    let y = dag.add_node(RiscOp::Load { name: "y".into() }, vec![], vec_f32(4), None);
    let added = dag.add_node(RiscOp::Add, vec![x, y], vec_f32(4), None);
    let realized = dag.add_node(RiscOp::Realize, vec![added], vec_f32(4), None);
    let negated = dag.add_node(RiscOp::Neg, vec![realized], vec_f32(4), None);
    dag.add_root(negated);

    let fused = fuse(&dag);
    let result = codegen_hip(&fused, "test_realize_barrier").unwrap();
    let src = &result.c_source;

    assert!(
        !src.contains("kernel_fused_"),
        "realize() must prevent fused kernel emission across the barrier"
    );
    assert!(
        src.matches("chelis_launch_kernel").count() >= 2,
        "realize() barrier should leave separate launches for add/copy/neg"
    );
}

// ---------------------------------------------------------------------------
// WS-8A: pad / shrink HIP kernel structural coverage. The GPU==eval numeric
// proof lives in the manual `gpu_correctness` oracle; these CI-default tests
// pin the codegen shape (kernel source present + real launch, not a
// metadata-only view) and the dtype-suffix dispatch.
// ---------------------------------------------------------------------------

#[test]
fn s8a_pad_emits_kernel_and_launch_not_view() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4), None);
    let p = dag.add_node(
        RiscOp::zero_pad(
            Prim::F32,
            vec![(chelis_ir::dag::RtDim::Lit(1), chelis_ir::dag::RtDim::Lit(1))],
        ),
        vec![x],
        vec_f32(6),
        None,
    );
    dag.add_root(p);

    let result = codegen_hip(&dag, "test_pad").unwrap();
    let src = &result.c_source;
    assert!(
        src.contains("__global__ void kernel_pad"),
        "pad must emit a kernel source string"
    );
    assert!(
        src.contains("chelis_launch_kernel"),
        "pad must launch its kernel, not lower to a metadata-only view"
    );
    // The fill value is reconstructed from its exact f32 bit pattern, not a
    // lossy decimal (sibling of #189/#250).
    assert!(
        src.contains("chelis_f32_from_bits"),
        "pad fill must use exact-bit-pattern reconstruction"
    );
}

#[test]
fn s8a_shrink_emits_kernel_and_launch_not_view() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(6), None);
    let s = dag.add_node(
        RiscOp::Shrink {
            bounds: vec![(chelis_ir::dag::RtDim::Lit(1), chelis_ir::dag::RtDim::Lit(5))],
        },
        vec![x],
        vec_f32(4),
        None,
    );
    dag.add_root(s);

    let result = codegen_hip(&dag, "test_shrink").unwrap();
    let src = &result.c_source;
    assert!(
        src.contains("__global__ void kernel_shrink"),
        "shrink must emit a kernel source string"
    );
    assert!(
        src.contains("chelis_launch_kernel"),
        "shrink must launch its kernel, not lower to a metadata-only view"
    );
}

#[test]
fn s8a_pad_f64_uses_dtype_suffix() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::F64,
        },
        None,
    );
    let p = dag.add_node(
        RiscOp::zero_pad(
            Prim::F64,
            vec![(chelis_ir::dag::RtDim::Lit(1), chelis_ir::dag::RtDim::Lit(1))],
        ),
        vec![x],
        TensorType {
            dims: vec![DimInfo::Lit(6)],
            precision: Prim::F64,
        },
        None,
    );
    dag.add_root(p);

    let result = codegen_hip(&dag, "test_pad_f64").unwrap();
    let src = &result.c_source;
    assert!(
        src.contains("kernel_pad_f64"),
        "f64 pad must dispatch to the dtype-suffixed kernel name"
    );
    assert!(
        src.contains("double fill") || src.contains("double a") || src.contains("double *out"),
        "f64 pad kernel must use the double C type"
    );
}

#[test]
fn s8a_shrink_i32_uses_dtype_suffix() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_i32(6), None);
    let s = dag.add_node(
        RiscOp::Shrink {
            bounds: vec![(chelis_ir::dag::RtDim::Lit(1), chelis_ir::dag::RtDim::Lit(5))],
        },
        vec![x],
        vec_i32(4),
        None,
    );
    dag.add_root(s);

    let result = codegen_hip(&dag, "test_shrink_i32").unwrap();
    let src = &result.c_source;
    assert!(
        src.contains("kernel_shrink_i32"),
        "i32 shrink must dispatch to the dtype-suffixed kernel name"
    );
    assert!(
        src.contains("int32_t"),
        "i32 shrink kernel must use the int32_t C type"
    );
}
