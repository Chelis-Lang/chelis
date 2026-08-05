//! chelis#919: `emit_fused_elem` is parameterized on the IR-pinned
//! dtype instead of being f32-hardcoded behind a `panic!`.
//!
//! The pre-fix guard carried the comment "the fuse pass currently only
//! produces f32 fused chains in practice; this guard catches a future
//! regression". That was false. The fuse pass produces f64 chains today
//! for every ordinary f64 activation, and the guard was on the live
//! path: it only looked unreachable because `chelis-python`'s artifact
//! gate rejected f64 one layer earlier. Being a `panic!` rather than a
//! `Result`, it crossed the pyo3 FFI boundary as a `PanicException`.
//!
//! What is pinned here:
//!   - an f64 fused chain emits `double` step variables, `double`
//!     data pointers with the reinterpreting cast that
//!     `chelis_runtime.h`'s `float *data` requires, and the
//!     double-precision libm symbols (`exp`, not `expf`)
//!   - the emitted f64 C actually compiles
//!   - the f32 emission is byte-identical to the pre-fix form, casts
//!     included (the in-place aliasing tests pin that exact text)
//!   - the dtypes still unsupported here fail as a branded
//!     `unsupported:` diagnostic, never a panic
//!   - `emit_fused_reduce`, which is still f32-only, likewise rejects
//!     rather than panics

use chelis_backend_c::codegen;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::fuse::fuse;
use chelis_types::types::Prim;

fn vec_ty(n: usize, precision: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision,
    }
}

fn mat_ty(rows: usize, cols: usize, precision: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(rows), DimInfo::Lit(cols)],
        precision,
    }
}

/// `exp(x) * x` — chelis#919's minimal repro. `exp(x)` alone and
/// `x * y` alone both compiled at f64 before the fix; composing them
/// fuses the two into one chain, and that chain panicked.
fn exp_times_x_dag(precision: Prim) -> Dag {
    let mut dag = Dag::new();
    let ty = vec_ty(4, precision);
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
    let e = dag.add_node(RiscOp::Exp, vec![x], ty.clone(), None);
    let m = dag.add_node(RiscOp::Mul, vec![e, x], ty, None);
    dag.add_root(m);
    fuse(&dag)
}

#[test]
fn f64_fused_chain_emits_double_step_variables() {
    let dag = exp_times_x_dag(Prim::F64);
    let src = codegen(&dag, "f64_fused_chain")
        .expect("an f64 fused chain must emit, not panic")
        .c_source;

    assert!(
        src.contains("double v0"),
        "f64 fused steps must be declared `double`; got:\n{src}"
    );
    assert!(
        !src.contains("float v0"),
        "an f64 fused chain must not declare `float` step variables: that is \
         the silent narrowing chelis#919 is about; got:\n{src}"
    );
}

#[test]
fn f64_fused_chain_uses_double_precision_math_symbols() {
    let dag = exp_times_x_dag(Prim::F64);
    let src = codegen(&dag, "f64_fused_math")
        .expect("an f64 fused chain must emit, not panic")
        .c_source;

    // `expf(` and `exp(` are disjoint substrings, so this is an exact
    // discrimination rather than a prefix match.
    assert!(
        src.contains("exp("),
        "f64 fused `exp` must call the double-precision libm symbol; got:\n{src}"
    );
    assert!(
        !src.contains("expf("),
        "an f64 fused chain must not call `expf`: passing a double to the \
         single-precision entry point narrows the value before the call, which \
         is exactly the f32 result chelis#919 reports; got:\n{src}"
    );
}

#[test]
fn f64_fused_chain_reinterprets_the_float_data_pointer() {
    let dag = exp_times_x_dag(Prim::F64);
    let src = codegen(&dag, "f64_fused_pointers")
        .expect("an f64 fused chain must emit, not panic")
        .c_source;

    // `chelis_runtime.h` declares the payload as `float *data`, so f64
    // access needs an explicit cast. Without it the loop advances four
    // bytes per element and reads half of each double — the
    // `[-2.8569523e-32, 2.0897851, 0.0, 1.875]` signature.
    assert!(
        src.contains("double* restrict __out_"),
        "the f64 fused output pointer must be `double*`; got:\n{src}"
    );
    assert!(
        src.contains("(double*)t"),
        "the f64 fused output pointer must reinterpret `float *data`; got:\n{src}"
    );
    assert!(
        src.contains("const double* restrict __ext0_"),
        "the f64 fused input pointer must be `const double*`; got:\n{src}"
    );
    assert!(
        src.contains("(const double*)t"),
        "the f64 fused input pointer must reinterpret `float *data`; got:\n{src}"
    );
}

#[test]
fn f64_fused_chain_compiles() {
    let dag = exp_times_x_dag(Prim::F64);
    let result = codegen(&dag, "f64_fused_compile").expect("f64 fused codegen");

    let dir = std::env::temp_dir().join("chelis_issue_919_f64_fused");
    std::fs::create_dir_all(&dir).expect("create temp dir");
    std::fs::write(dir.join("f64_fused_compile.c"), &result.c_source).expect("write c");
    std::fs::write(dir.join("f64_fused_compile.h"), &result.h_header).expect("write h");

    let include_dir =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../chelis-runtime/include");
    for header in &[
        "chelis_runtime.h",
        "chelis_runtime_dtype.h",
        "chelis_blas.h",
        "chelis_simd.h",
        "chelis_math.h",
    ] {
        let src = std::fs::read_to_string(include_dir.join(header)).expect("read runtime header");
        std::fs::write(dir.join(header), &src).expect("stage runtime header");
    }

    let output = std::process::Command::new(chelis_backend_c::toolchain::c_compiler())
        .args([
            "-fsyntax-only",
            "-I",
            dir.to_str().unwrap(),
            dir.join("f64_fused_compile.c").to_str().unwrap(),
        ])
        .output();

    match output {
        Ok(o) if o.status.success() => {}
        Ok(o) => panic!(
            "f64 fused C codegen does NOT compile:\n{}",
            String::from_utf8_lossy(&o.stderr)
        ),
        Err(e) => eprintln!("C compiler not available ({e}), skipping compile check"),
    }
}

#[test]
fn f32_fused_chain_emission_is_unchanged() {
    // Negative parity for the widening: the f32 lane must keep the
    // pre-#919 text exactly, cast-free. `fused_in_place_forall_alias`
    // asserts these literal strings, so a stray unconditional cast
    // would be a real regression rather than a cosmetic one.
    let dag = exp_times_x_dag(Prim::F32);
    let src = codegen(&dag, "f32_fused_chain")
        .expect("f32 fused codegen")
        .c_source;

    assert!(
        src.contains("float v0"),
        "f32 fused steps must stay `float`; got:\n{src}"
    );
    assert!(
        src.contains("expf("),
        "f32 fused `exp` must stay on the single-precision symbol; got:\n{src}"
    );
    assert!(
        !src.contains("(double*)") && !src.contains("double v0"),
        "the f32 lane must not acquire any double-precision emission; got:\n{src}"
    );
    assert!(
        !src.contains("(float*)t"),
        "the f32 lane must keep the cast-free `t{{n}}->data` pointer form; got:\n{src}"
    );
}

#[test]
fn integer_fused_chain_is_a_diagnostic_not_a_panic() {
    // The fused-elem path admits f32 and f64 only. Integer chains need
    // integer step operators rather than libm calls (chelis#729 owns that
    // dtype capability; chelis#691's direct `emit_binary_func` / abs nodes
    // are repaired), and reduced floats (bf16/f16) need the
    // convert-then-compute routing.
    // Both must reject through the `Result` channel: a `panic!` here
    // reaches Python as a `PanicException` rather than a diagnostic,
    // which is the failure mode chelis#919 reports.
    let mut dag = Dag::new();
    let ty = vec_ty(4, Prim::Int32);
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
    let y = dag.add_node(RiscOp::Load { name: "y".into() }, vec![], ty.clone(), None);
    let a = dag.add_node(RiscOp::Add, vec![x, y], ty.clone(), None);
    let n = dag.add_node(RiscOp::Neg, vec![a], ty, None);
    dag.add_root(n);
    let fused = fuse(&dag);

    let err = codegen(&fused, "int32_fused_reject_probe")
        .map(|_| ())
        .expect_err("an int32 fused chain must be rejected, not emitted");
    let rendered = err.to_string();
    assert!(
        rendered.starts_with("unsupported:"),
        "the rejection must come through the branded diagnostic channel; got: {rendered}"
    );
    assert!(
        rendered.contains("int32"),
        "the rejection must name the offending dtype; got: {rendered}"
    );
}

#[test]
fn f64_fused_reduce_is_a_diagnostic_not_a_panic() {
    // `emit_fused_reduce` is a separate, still-f32-only path: its body
    // needs a `chelis_fill_f64` zero, a `double` accumulator cascade,
    // and `fmax`. chelis#919 does not widen it (chelis#951 owns that
    // half) — but it is reachable from ordinary Surf, because
    // `sum(exp(x), 0)` at f64 inlines the elementwise node into the
    // reduction, so it must reject rather than panic.
    let mut dag = Dag::new();
    let ty = mat_ty(3, 4, Prim::F64);
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
    let e = dag.add_node(RiscOp::Exp, vec![x], ty.clone(), None);
    let n = dag.add_node(RiscOp::Neg, vec![e], ty, None);
    let s = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: Prim::F64,
        },
        vec![n],
        vec_ty(3, Prim::F64),
        None,
    );
    dag.add_root(s);
    let fused = fuse(&dag);

    let err = codegen(&fused, "f64_fused_reduce_reject_probe")
        .map(|_| ())
        .expect_err("an f64 fused reduction must be rejected, not emitted");
    let rendered = err.to_string();
    assert!(
        rendered.starts_with("unsupported:"),
        "the rejection must come through the branded diagnostic channel; got: {rendered}"
    );
    assert!(
        rendered.contains("f64"),
        "the rejection must name the offending dtype; got: {rendered}"
    );
}
