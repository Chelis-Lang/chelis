//! Red team: verify C backend fused codegen compiles with gcc -fsyntax-only.

use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::fuse::fuse;
use chelis_types::types::Prim;

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
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4));
    let y = dag.add_node(RiscOp::Load { name: "y".into() }, vec![], vec_f32(4));
    let a = dag.add_node(RiscOp::Add, vec![x, y], vec_f32(4));
    let b = dag.add_node(RiscOp::Neg, vec![a], vec_f32(4));
    let c = dag.add_node(RiscOp::Exp, vec![b], vec_f32(4));
    dag.add_root(c);

    // Fuse and codegen
    let fused = fuse(&dag);
    let result = chelis_backend_c::codegen(&fused, "test_fused");

    // Write to temp file and syntax-check with gcc
    let dir = std::env::temp_dir().join("chelis_redteam_c_fused");
    std::fs::create_dir_all(&dir).unwrap();

    std::fs::write(dir.join("test_fused.c"), &result.c_source).unwrap();
    std::fs::write(dir.join("test_fused.h"), &result.h_header).unwrap();

    // Copy runtime files
    let rt_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("runtime");
    let h_src = std::fs::read_to_string(rt_dir.join("chelis_runtime.h")).unwrap();
    let c_src = std::fs::read_to_string(rt_dir.join("chelis_runtime.c")).unwrap();
    std::fs::write(dir.join("chelis_runtime.h"), &h_src).unwrap();
    std::fs::write(dir.join("chelis_runtime.c"), &c_src).unwrap();

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
    let output = std::process::Command::new("gcc")
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
            eprintln!("gcc not available ({e}), skipping compile check");
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
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], mat_f32(3, 4));
    let c = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(3, 4));
    let added = dag.add_node(RiscOp::Add, vec![x, c], mat_f32(3, 4));
    let negated = dag.add_node(RiscOp::Neg, vec![added], mat_f32(3, 4));
    let summed = dag.add_node(RiscOp::Sum { axis: 1 }, vec![negated], vec_f32(3));
    dag.add_root(summed);

    let fused = fuse(&dag);
    let result = chelis_backend_c::codegen(&fused, "test_fused_reduce");
    let src = &result.c_source;

    // The fused reduction should contain the elementwise step variables (v0 etc.)
    // inside the reduction loop, NOT in a separate elementwise loop.
    assert!(
        src.contains("float v0"),
        "Fused reduce should contain step variable 'float v0'"
    );
    assert!(
        src.contains("acc += v"),
        "Fused reduce should accumulate into acc from a step variable"
    );

    // There should be NO separate allocation for the FusedElem output.
    // With fusion: 1 const alloc + 1 reduction alloc = 2 chelis_alloc calls.
    // Without fusion: 1 const alloc + 1 FusedElem alloc + 1 reduction alloc = 3.
    let alloc_count = src.matches("chelis_alloc(").count();
    assert_eq!(
        alloc_count, 2,
        "Fused add→neg→sum should have 2 allocs (1 const + 1 reduction output), got {alloc_count}"
    );
}

#[test]
fn c_fused_reduce_max_no_intermediate() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], mat_f32(3, 4));
    let c = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(3, 4));
    let added = dag.add_node(RiscOp::Add, vec![x, c], mat_f32(3, 4));
    let negated = dag.add_node(RiscOp::Neg, vec![added], mat_f32(3, 4));
    let maxed = dag.add_node(RiscOp::MaxReduce { axis: 1 }, vec![negated], vec_f32(3));
    dag.add_root(maxed);

    let fused = fuse(&dag);
    let result = chelis_backend_c::codegen(&fused, "test_fused_maxred");
    let src = &result.c_source;

    assert!(
        src.contains("float v0"),
        "Fused max-reduce should contain step variable"
    );
    assert!(
        src.contains("fmaxf(acc, v"),
        "Fused max-reduce should use fmaxf to accumulate"
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
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], mat_f32(3, 4));
    let c = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], mat_f32(3, 4));
    let added = dag.add_node(RiscOp::Add, vec![x, c], mat_f32(3, 4));
    let negated = dag.add_node(RiscOp::Neg, vec![added], mat_f32(3, 4));
    let summed = dag.add_node(RiscOp::Sum { axis: 1 }, vec![negated], vec_f32(3));
    dag.add_root(summed);

    let fused = fuse(&dag);
    let result = chelis_backend_c::codegen(&fused, "test_fused_reduce_compile");

    let dir = std::env::temp_dir().join("chelis_redteam_c_fused_reduce");
    std::fs::create_dir_all(&dir).unwrap();

    std::fs::write(dir.join("test.c"), &result.c_source).unwrap();
    std::fs::write(dir.join("test.h"), &result.h_header).unwrap();

    let rt_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("runtime");
    let h_src = std::fs::read_to_string(rt_dir.join("chelis_runtime.h")).unwrap();
    let c_src = std::fs::read_to_string(rt_dir.join("chelis_runtime.c")).unwrap();
    std::fs::write(dir.join("chelis_runtime.h"), &h_src).unwrap();
    std::fs::write(dir.join("chelis_runtime.c"), &c_src).unwrap();

    let output = std::process::Command::new("gcc")
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
            eprintln!("gcc not available ({e}), skipping compile check");
        }
    }
}
