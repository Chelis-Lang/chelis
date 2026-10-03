//! Generated C dispatch for common tensor operations.
//!
//! Matmul retains its primitive product and canonical adjacent-pair Sum
//! under [05-OP-30]. Its C selection is independent of accelerator dispatch.
//! The other operations retain their existing lowering/runtime paths;
//! these source checks measure selection, not numerical conformance.

use std::fs;
use std::process::Command;

use assert_cmd::cargo::CommandCargoExt;
use tempfile::tempdir;

#[derive(Debug)]
struct Dispatch {
    name: &'static str,
    c_lines: usize,
    allocs: usize,
    sgemm_calls: usize,
    runtime_call: bool,
    sparse_gather_loop: bool,
    adjacent_pair_sum: bool,
    fused_kernels: usize,
    generic_loops: usize,
}

fn build_and_classify(name: &'static str, source: &str) -> Dispatch {
    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join(format!("op_{name}.ch"));
    let out_dir = dir.path().join(format!("{name}_out"));
    fs::write(&src_path, source).expect("write source");

    // chelis build runs fmt-check + lint as gates (post #7), so we ask
    // chelis to format the source first.
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["fmt", "--inplace", src_path.to_str().unwrap()])
        .status()
        .expect("chelis fmt should run");

    let status = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args([
            "build",
            "--emit-c",
            src_path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .status()
        .expect("chelis build should run");
    assert!(status.success(), "chelis build failed for {name}");

    let c_path = out_dir.join(format!("op_{name}.c"));
    let c = fs::read_to_string(&c_path).expect("read generated c");
    let c_lines = c.lines().count();
    let adjacent_pair_sum = c.contains("__sum_level_") && c.contains("__next_n_");
    let allocs = c.matches("chelis_alloc(").count();
    // Only count actual *call sites* (indented, with open paren).
    let sgemm_calls = c.matches("    cblas_sgemm(").count();
    // Generic gather/scatter etc. emit a single exact tagged runtime call.
    let runtime_call = c.contains("chelis_tensor_scatter_replace(")
        || c.contains("chelis_tensor_scatter_add(")
        || c.contains("chelis_tensor_gather(")
        || c.contains("chelis_tensor_where(")
        || c.contains("chelis_tensor_cumsum(")
        || c.contains("chelis_tensor_sort(");
    let sparse_gather_loop = c.contains("chelis_tensor_sparse_plan(")
        && c.contains("chelis_sparse_index_slot(")
        && c.contains("chelis_sparse_data_index(")
        && c.contains("chelis_sparse_check_target(")
        && c.contains("chelis_sparse_plan_release(")
        && c.contains("CHELIS_DTYPE_I64");
    // Each parallel loop is an `omp for` inside the region that pins the
    // floating-point environment on every worker thread.
    let fused_kernels = c.matches("#pragma omp for simd").count();
    // Generic non-SIMD parallel-for loops (used for reductions etc.).
    let generic_loops = c.matches("\n        #pragma omp for\n").count();

    Dispatch {
        name,
        c_lines,
        allocs,
        sgemm_calls,
        runtime_call,
        sparse_gather_loop,
        adjacent_pair_sum,
        fused_kernels,
        generic_loops,
    }
}

#[test]
fn specialized_kernel_dispatch_reality_for_common_ops() {
    let matmul = "def f(a: tensor[64, 128, f32], b: tensor[128, 32, f32]) -> tensor[64, 32, f32] = matmul(a, b)\n";
    let softmax = "def f(x: tensor[64, 128, f32]) -> tensor[64, 128, f32] = softmax(x, 1)\n";
    let layernorm = "def f(x: tensor[64, 128, f32], gamma: tensor[128, f32], beta: tensor[128, f32]) -> tensor[64, 128, f32] = layer_norm(x, gamma, beta, 0.00001f32)\n";
    let scatter = "def f(base: tensor[10, 4, f32], bin_ids: tensor[64, i64], updates: tensor[64, 4, f32]) -> tensor[10, 4, f32] = scatter(base, bin_ids, updates, 0, \"add\")\n";
    let gather = "def f(table: tensor[1000, 128, f32], indices: tensor[64, i64]) -> tensor[64, 128, f32] = gather(table, indices, 0)\n";

    let m = build_and_classify("matmul", matmul);
    let s = build_and_classify("softmax", softmax);
    let l = build_and_classify("layernorm", layernorm);
    let sc = build_and_classify("scatter", scatter);
    let g = build_and_classify("gather", gather);

    eprintln!("== Specialization Dispatch Reality ==");
    eprintln!("(C backend; accelerator selection is tested separately)");
    eprintln!();
    eprintln!(
        "{:<10} | {:>7} | {:>6} | {:>10} | {:>11} | {:>13} | {:>13}",
        "operation",
        "C lines",
        "allocs",
        "BLAS calls",
        "runtime fn",
        "fused kernels",
        "generic loops",
    );
    for d in [&m, &s, &l, &sc, &g] {
        eprintln!(
            "{:<10} | {:>7} | {:>6} | {:>10} | {:>11} | {:>13} | {:>13}",
            d.name,
            d.c_lines,
            d.allocs,
            d.sgemm_calls,
            if d.runtime_call { "YES" } else { "no" },
            d.fused_kernels,
            d.generic_loops,
        );
    }
    eprintln!();
    eprintln!("Verdict per row:");
    eprintln!("  matmul     -> primitive product and canonical adjacent-pair Sum");
    eprintln!("  softmax    -> generic Tier 2 lowering, multiple kernels, no fused softmax");
    eprintln!("  layer_norm -> generic Tier 2 lowering, several backing slots, no fused layernorm");
    eprintln!(
        "  scatter    -> single chelis_tensor_scatter_add() runtime call (no parallel-radix-sort)"
    );
    eprintln!(
        "  gather     -> first-class sparse C loop (no dense one-hot, no warp-aware GPU path)"
    );

    // ---- Locked assertions ----
    //
    assert_eq!(
        m.sgemm_calls, 0,
        "vendor GEMM must not replace canonical matmul arithmetic"
    );
    assert!(
        m.adjacent_pair_sum,
        "matmul must retain its adjacent-pair Sum"
    );

    assert_eq!(
        s.sgemm_calls, 0,
        "softmax must NOT spuriously match the matmul pattern. Got {} \
         BLAS calls.",
        s.sgemm_calls
    );
    assert!(
        s.allocs >= 3,
        "softmax §4.2 lowering should still produce multiple backing slots \
         under C memory planning. Got {}. If this drops to one and the loop \
         evidence changes, a softmax-fusion pass shipped. Update.",
        s.allocs
    );

    assert_eq!(
        l.sgemm_calls, 0,
        "layer_norm must NOT spuriously match the matmul pattern. Got {} \
         BLAS calls.",
        l.sgemm_calls
    );
    assert!(
        l.allocs >= 6,
        "layer_norm §4.4 lowering should still produce several backing slots \
         under C memory planning (mean/variance/normalization stages are not \
         fused into one specialized kernel). Got {}.",
        l.allocs
    );

    assert!(
        sc.runtime_call,
        "scatter must lower to a chelis_tensor_scatter_add() runtime call \
         (the only generic dispatch path today). If this changes, a \
         specialised scatter pattern matcher shipped. Update the test."
    );
    assert_eq!(
        sc.sgemm_calls, 0,
        "scatter must NOT match the matmul pattern. Got {} BLAS calls.",
        sc.sgemm_calls
    );

    assert!(
        !g.runtime_call,
        "tensor-lane gather must not fall back to chelis_tensor_gather() once sparse IR lowering is wired"
    );
    assert!(
        g.sparse_gather_loop,
        "tensor-lane gather must emit the bounded sparse C loop over indices/values/output"
    );
    assert_eq!(
        g.sgemm_calls, 0,
        "gather must NOT match the matmul pattern. Got {} BLAS calls.",
        g.sgemm_calls
    );
}
