//! Red team: verify C backend fused codegen compiles with gcc -fsyntax-only.

mod support;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::fuse::fuse;
use chelis_types::types::Prim;
use support::codegen;

mod common;

fn vec_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn mat_f32(rows: usize, cols: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(rows), DimInfo::Lit(cols)],
        precision: Prim::F32,
    }
}

#[test]
fn c_fused_codegen_compiles() {
    // Build a fusible DAG
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let y = dag.add_node(
        decl,
        RiscOp::Load { name: "y".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let a = dag.add_node(decl, RiscOp::Add, vec![x, y], vec_f32(4), None);
    let b = dag.add_node(decl, RiscOp::Neg, vec![a], vec_f32(4), None);
    let c = dag.add_node(decl, RiscOp::Exp, vec![b], vec_f32(4), None);
    dag.add_root(c);

    // Fuse and codegen
    let fused = fuse(&dag);
    let result = codegen(&fused, "test_fused").unwrap();

    // Write to temp file and syntax-check with gcc
    let probe = common::probe_dir("redteam_c_fused");
    let dir = probe.path().to_path_buf();

    std::fs::write(dir.join("test_fused.c"), &result.c_source).unwrap();
    std::fs::write(dir.join("test_fused.h"), &result.h_header).unwrap();

    let include_dir =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../chelis-runtime/include");
    for header in &[
        "chelis_runtime.h",
        "chelis_runtime_views.h",
        "chelis_runtime_dtype.h",
        "chelis_blas.h",
        "chelis_simd.h",
        "chelis_math.h",
    ] {
        let src = std::fs::read_to_string(include_dir.join(header)).unwrap();
        std::fs::write(dir.join(header), &src).unwrap();
    }

    // Verify the source contains a fused loop (float v0, float v1, etc.)
    assert!(
        result.c_source.contains("float v0"),
        "Fused C code should have float v0 step variable"
    );
    assert!(
        result.c_source.contains("float v1"),
        "Fused C code should have float v1 step variable"
    );
    assert!(
        result.c_source.contains("float v2"),
        "Fused C code should have float v2 step variable"
    );

    // Verify single loop (not multiple loops for the chain)
    let loop_count = result.c_source.matches("for (int i = 0;").count();
    // Should have at most 2 loops: one for the fused chain + possibly one for const fill
    // The key is that 3 ops (add, neg, exp) are NOT 3 separate loops
    assert!(
        loop_count <= 2,
        "Fused chain should NOT produce 3 separate loops (got {loop_count})"
    );

    // Try gcc -fsyntax-only
    let output = std::process::Command::new(chelis_backend_c::toolchain::c_compiler())
        .args([
            "-fsyntax-only",
            "-I",
            dir.to_str().unwrap(),
            dir.join("test_fused.c").to_str().unwrap(),
        ])
        .output();

    match output {
        Ok(o) if o.status.success() => {
            // Great, compiles cleanly
        }
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            panic!("Fused C codegen does NOT compile:\n{stderr}");
        }
        Err(e) => {
            panic!("C compiler is required for the fused contract: {e}");
        }
    }
}

// ===========================================================================
// Elementwise→reduction fusion: C backend structural tests
// ===========================================================================

#[test]
fn c_fused_reduce_sum_no_intermediate() {
    // add(x, const) → neg → sum(axis=1): add→neg fuses into FusedElem,
    // then the FusedElem feeds sum as sole consumer → inlined into reduction.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        mat_f32(3, 4),
        None,
    );
    let c = dag.add_node(
        decl,
        RiscOp::synth_const(mat_f32(3, 4).precision, 1.0),
        vec![],
        mat_f32(3, 4),
        None,
    );
    let added = dag.add_node(decl, RiscOp::Add, vec![x, c], mat_f32(3, 4), None);
    let negated = dag.add_node(decl, RiscOp::Neg, vec![added], mat_f32(3, 4), None);
    let summed = dag.add_node(
        decl,
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
    let result = codegen(&fused, "test_fused_reduce").unwrap();
    let src = &result.c_source;

    // The fused reduction should contain the elementwise step variables (v0 etc.)
    // inside the reduction loop, NOT in a separate elementwise loop.
    assert!(
        src.contains("float v0"),
        "Fused reduce should contain step variable 'float v0'"
    );
    // Fused leaves enter the same canonical tree as materialized Sum.
    assert!(src.contains("__sum_level_"));
    assert!(src.contains("] = v"));
    assert!(!src.contains("acc0 +="));

    // There should be NO separate allocation for the FusedElem output.
    // The tree scratch is now a checked runtime tensor. It is distinct from
    // materializing the fused elementwise result across every output group.
    let alloc_count = src.matches("chelis_alloc(").count();
    assert_eq!(
        alloc_count, 3,
        "Fused add→neg→sum requires one constant, one result, and checked tree scratch"
    );
    let fused_node = fused
        .nodes()
        .iter()
        .find(|node| matches!(node.op, RiscOp::FusedElem { .. }))
        .unwrap();
    assert!(!src.contains(&format!("chelis_tensor *t{} =", fused_node.id.0)));
    assert!(src.contains("chelis_reduction_check_scratch("));
}

#[test]
fn c_fused_reduce_max_no_intermediate() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        mat_f32(3, 4),
        None,
    );
    let c = dag.add_node(
        decl,
        RiscOp::synth_const(mat_f32(3, 4).precision, 1.0),
        vec![],
        mat_f32(3, 4),
        None,
    );
    let added = dag.add_node(decl, RiscOp::Add, vec![x, c], mat_f32(3, 4), None);
    let negated = dag.add_node(decl, RiscOp::Neg, vec![added], mat_f32(3, 4), None);
    let maxed = dag.add_node(
        decl,
        RiscOp::MaxReduce { axis: 1 },
        vec![negated],
        vec_f32(3),
        None,
    );
    dag.add_root(maxed);

    let fused = fuse(&dag);
    let result = codegen(&fused, "test_fused_maxred").unwrap();
    let src = &result.c_source;

    assert!(
        src.contains("float v0"),
        "Fused max-reduce should contain step variable"
    );
    assert!(
        src.contains("chelis_fmax_propnan_f32(acc, v"),
        "Fused max-reduce should accumulate via the NaN-propagating max \
         helper (#172 torch parity), not NaN-dropping fmaxf"
    );

    let alloc_count = src.matches("chelis_alloc(").count();
    assert_eq!(
        alloc_count, 2,
        "Fused add→neg→max_reduce should have 2 allocs, got {alloc_count}"
    );
}

#[test]
fn c_fused_reduce_compiles() {
    // Verify the fused reduction C code compiles with gcc -fsyntax-only.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        mat_f32(3, 4),
        None,
    );
    let c = dag.add_node(
        decl,
        RiscOp::synth_const(mat_f32(3, 4).precision, 1.0),
        vec![],
        mat_f32(3, 4),
        None,
    );
    let added = dag.add_node(decl, RiscOp::Add, vec![x, c], mat_f32(3, 4), None);
    let negated = dag.add_node(decl, RiscOp::Neg, vec![added], mat_f32(3, 4), None);
    let summed = dag.add_node(
        decl,
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
    let result = codegen(&fused, "test_fused_reduce_compile").unwrap();

    let probe = common::probe_dir("redteam_c_fused_reduce");
    let dir = probe.path().to_path_buf();

    std::fs::write(dir.join("test.c"), &result.c_source).unwrap();
    std::fs::write(dir.join("test.h"), &result.h_header).unwrap();

    let include_dir =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../chelis-runtime/include");
    for header in &[
        "chelis_runtime.h",
        "chelis_runtime_views.h",
        "chelis_runtime_dtype.h",
        "chelis_blas.h",
        "chelis_simd.h",
        "chelis_math.h",
    ] {
        let src = std::fs::read_to_string(include_dir.join(header)).unwrap();
        std::fs::write(dir.join(header), &src).unwrap();
    }

    let output = std::process::Command::new(chelis_backend_c::toolchain::c_compiler())
        .args([
            "-fsyntax-only",
            "-I",
            dir.to_str().unwrap(),
            dir.join("test.c").to_str().unwrap(),
        ])
        .output();

    match output {
        Ok(o) if o.status.success() => {}
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            panic!(
                "Fused reduction C codegen does NOT compile:\n{stderr}\nSource:\n{}",
                result.c_source
            );
        }
        Err(e) => {
            panic!("C compiler is required for the fused contract: {e}");
        }
    }
}

/// Write `result` and every public runtime header into a new directory. The
/// returned guard keeps the directory, and the path names the unit.
fn stage_unit(
    label: &str,
    result: &chelis_backend_c::CodegenResult,
) -> (tempfile::TempDir, std::path::PathBuf) {
    let probe = common::probe_dir(label);
    for (name, contents) in chelis_runtime_bundle::PUBLIC_HEADERS {
        std::fs::write(probe.path().join(name), contents).unwrap();
    }
    let unit = probe.path().join(format!("{label}.c"));
    std::fs::write(&unit, &result.c_source).unwrap();
    std::fs::write(unit.with_extension("h"), &result.h_header).unwrap();
    (probe, unit)
}

/// Run the test compiler on `unit` under the profile with `extra`, and return
/// its standard output. The test fails when the compiler fails.
fn run_profile_compiler(
    unit: &std::path::Path,
    result: &chelis_backend_c::CodegenResult,
    extra: &[&str],
) -> String {
    let toolchain = chelis_backend_c::toolchain::test_toolchain(result.requirements);
    let output = chelis_backend_c::toolchain::tool_command(&toolchain.compiler)
        .args(&toolchain.compile_flags)
        .arg("-I")
        .arg(unit.parent().unwrap())
        .args(extra)
        .arg(unit)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{} failed on {}:\n{}",
        toolchain.compiler,
        unit.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

/// chelis#3704: a unit that calls no `chelis_simd.h` helper does not include
/// the header. So its compile reads neither `arm_neon.h` nor `immintrin.h`.
#[test]
fn a_unit_without_simd_helper_calls_reads_no_intrinsics_header() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let y = dag.add_node(
        decl,
        RiscOp::Load { name: "y".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let sum = dag.add_node(decl, RiscOp::Add, vec![x, y], vec_f32(4), None);
    dag.add_root(sum);
    let result = codegen(&dag, "no_simd_helpers").unwrap();
    let (_probe, unit) = stage_unit("no_simd_helpers", &result);
    let dependencies = run_profile_compiler(&unit, &result, &["-M"]);
    // The dependency list names the runtime header, so it lists headers at all.
    assert!(dependencies.contains("chelis_runtime.h"), "{dependencies}");
    for header in ["chelis_simd.h", "arm_neon.h", "immintrin.h"] {
        assert!(
            !dependencies.contains(header),
            "the unit reads {header}:\n{dependencies}"
        );
    }
}

/// chelis#3704: a fused max reduction calls `chelis_fmax_propnan_f32`, so its
/// unit includes `chelis_simd.h` once. The chain also calls a correctly rounded
/// kernel, so the unit carries kernel text too, and it compiles under the
/// profile.
#[test]
fn a_fused_max_reduce_unit_includes_the_simd_header_and_compiles() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        mat_f32(3, 4),
        None,
    );
    let c = dag.add_node(
        decl,
        RiscOp::synth_const(mat_f32(3, 4).precision, 1.0),
        vec![],
        mat_f32(3, 4),
        None,
    );
    let added = dag.add_node(decl, RiscOp::Add, vec![x, c], mat_f32(3, 4), None);
    let raised = dag.add_node(decl, RiscOp::Exp, vec![added], mat_f32(3, 4), None);
    let negated = dag.add_node(decl, RiscOp::Neg, vec![raised], mat_f32(3, 4), None);
    let maxed = dag.add_node(
        decl,
        RiscOp::MaxReduce { axis: 1 },
        vec![negated],
        vec_f32(3),
        None,
    );
    dag.add_root(maxed);
    let result = codegen(&fuse(&dag), "fused_max_simd_header").unwrap();
    for call in ["chelis_fmax_propnan_f32(", "chelis_cr_expf("] {
        assert!(
            result.c_source.contains(call),
            "the unit does not call {call}"
        );
    }
    assert_eq!(
        result
            .c_source
            .matches("#include \"chelis_simd.h\"\n")
            .count(),
        1
    );
    let (probe, unit) = stage_unit("fused_max_simd_header", &result);
    let object = probe.path().join("fused_max_simd_header.o");
    run_profile_compiler(&unit, &result, &["-c", "-o", object.to_str().unwrap()]);
    assert!(object.is_file());
}
