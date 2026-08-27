//! Test 6 — Specialization Dispatch (the cuBLAS / cuDF / cuDNN analogue).
//!
//! Background framing (paraphrased from a third-party review): "Generic
//! CUDA is slow. Highly specialized CUDA is fast. And the compiler needs
//! the high-level intent to know *which* specialized CUDA to use."
//!
//! When you 'just compile to CUDA' (or, in this test, 'just compile to C
//! with cblas+OpenMP'), the GPU/CPU can technically execute any composition
//! of basic math primitives — but the *speed* depends entirely on whether
//! the compiler recognises the high-level operation and dispatches to the
//! pre-tuned hardware-aware kernel:
//!
//!   * **matmul → cuBLAS / cblas_sgemm** — uses Tensor Cores, SRAM tiling,
//!     and a hand-written cache schedule. ~100× faster than naive math.
//!   * **softmax → cuDNN softmax** — single fused kernel that reads each
//!     element exactly twice, with online-softmax for numerical stability.
//!     Streaming over the reduction axis avoids materialising the
//!     intermediate exp/max/sum tensors.
//!   * **layer_norm → cuDNN layernorm** — fused kernel computing mean,
//!     variance, normalised output in a single pass over the reduction
//!     axis.
//!   * **scatter/gather → cuDF / Thrust scatter** — parallel radix sort
//!     followed by segmented reduction, avoiding warp divergence and
//!     atomic conflicts. The basis of GPU-accelerated dataframes.
//!
//! Every one of these specialised kernels exists because the *generic*
//! decomposition into RISC primitives runs into a different performance
//! pathology: memory bandwidth wall (matmul), redundant memory traffic
//! (softmax/layer_norm), or warp divergence (scatter/gather).
//!
//! This test compiles all five operations to C and tabulates which one
//! hits a specialised library call vs which falls through to generic
//! parallel-for loops. The C-backend's BLAS path (cblas_sgemm) is the
//! exact analogue of what cuBLAS dispatch would provide on GPU. If
//! Chelis cannot dispatch a specialised kernel for an operation in the
//! C backend, it almost certainly cannot dispatch a specialised kernel
//! for the operation on GPU either — the dispatch logic lives in the
//! shared IR optimisation/codegen pipeline, not in the backend.
//!
//! ## Empirical findings (locked here)
//!
//! Running each operation through `chelis build --target c` and inspecting
//! the emitted code:
//!
//! | Operation   | Codegen path                          | Specialised? |
//! |-------------|---------------------------------------|--------------|
//! | matmul      | `cblas_sgemm()` direct call           | **YES**      |
//! | softmax     | §4.2 lowering: 4 allocs + 3+ scalar parallel-for loops | no |
//! | layer_norm  | §4.4 lowering: 10+ allocs + many loops              | no |
//! | scatter     | Exact `chelis_tensor_scatter_add()` runtime call    | no |
//! | gather      | First-class sparse C loop                           | partial |
//!
//! **Translation to the user's framing:**
//!   * matmul gets the equivalent of "Tensor Cores + cuBLAS dispatch."
//!   * softmax / layer_norm get the equivalent of "naive CUDA cores
//!     reading from main memory" — no fused reduction, no online-softmax,
//!     no SRAM tiling.
//!   * scatter still gets a generic runtime call. Tensor-lane gather now
//!     lowers to a first-class sparse C loop, which avoids dense one-hot
//!     materialization but is not yet a GPU warp-aware/cuDF-style path.
//!
//! ## Implication for the cross-library AD claim — Coral specifically
//!
//! "AD flows through Coral / Nautilus / Octant because everything compiles
//! to the same RISC primitive set" is correct **for the AD side**. But
//! "AD flows through" does not imply "those library operations get
//! specialised codegen." For the dataframe case (Coral, the pandas
//! analogue at https://github.com/Chelis-Lang/coral), the gap is even
//! sharper than this test directly probes: Coral's `agg_sum` /
//! `agg_mean` / `agg_count` etc. (`coral/src/groupby.ch`) are
//! implemented as host-lane operations over `List[List[int64]]`:
//!
//! ```chelis-surf-fragment
//! // From Coral.GroupBy.agg_sum, FloatCol branch:
//! to_tensor(map(fn (rows: List[int64]) ->
//!   sum_f32(select_float_rows(to_list(xs), rows)),
//! groups))
//! ```
//!
//! That `map` over `List[List[int64]]` is host-lane: it never enters
//! the tensor IR DAG. Per `spec/06-transformations.md` §2.10, "`grad`
//! is fully supported on the tensor lane... `grad` is **not** currently
//! supported on the host lane." So:
//!
//!   * AD does NOT flow through a Coral `groupby + agg_sum`. The
//!     official hello-chelis example
//!     (`hello-chelis/src/coral/adthroughdataframe.ch`) acknowledges
//!     this in plain text: *"The chelis 0.6.1 evaluator does not
//!     currently lower `grad` through the Frame ADT, so we keep the
//!     differentiable expression on raw tensors."* The example
//!     deliberately separates the AD-bearing `sum(mul(w, w), 0)` from
//!     the dataframe wrapper.
//!   * No specialised dispatch is even possible because the operation
//!     isn't in the IR. A `cuDF`-style parallel-radix-sort + segmented
//!     reduction would require recognising a pattern that doesn't
//!     exist in the DAG at all — the dataframe scaffolding is
//!     host-side `List[List[int64]]` plus `map`/`fold` higher-order
//!     functions.
//!
//! What this test directly probes (matmul / softmax / layer_norm /
//! scatter / gather) is the *closest tensor-lane* analogue to the Coral
//! and Nautilus operations. The "10000 basic math ops" graph the
//! third-party review describes is exactly what the softmax and
//! layer_norm tabulations below show: dozens of allocations, many small
//! kernels, no recognition of the high-level operation, no dispatch to
//! a specialised library call. For Coral specifically the situation is
//! one step further upstream — the operation isn't even in the tensor
//! IR to be dispatched.
//!
//! Closing this gap requires either (a) more pattern matchers in the
//! IR optimise pass (one per operation type that has a specialised
//! kernel), (b) keeping the high-level Tier 2 operation in the IR as a
//! first-class node and emitting backend-specific calls directly from
//! it (without round-tripping through the RISC decomposition), or (c)
//! lifting Coral's host-lane `groupby` into a tensor-DAG primitive so
//! it can be specialised at all. Today's design picks (a) only for
//! `matmul` — every other operation is generic.

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
    let sparse_gather_loop = c.contains("indices_data")
        && c.contains("values_data")
        && c.contains("_out_data")
        && c.contains("_g =")
        && c.contains("CHELIS_DTYPE_I64");
    let fused_kernels = c.matches("parallel for simd").count();
    // Generic non-SIMD parallel-for loops (used for reductions etc.).
    let generic_loops = c.matches("\n    #pragma omp parallel for\n").count();

    Dispatch {
        name,
        c_lines,
        allocs,
        sgemm_calls,
        runtime_call,
        sparse_gather_loop,
        fused_kernels,
        generic_loops,
    }
}

#[test]
fn specialized_kernel_dispatch_reality_for_common_ops() {
    let matmul = "def f(a: tensor[64, 128, f32], b: tensor[128, 32, f32]) -> tensor[64, 32, f32] = matmul(a, b)\n";
    let softmax = "def f(x: tensor[64, 128, f32]) -> tensor[64, 128, f32] = softmax(x, 1)\n";
    let layernorm = "def f(x: tensor[64, 128, f32], gamma: tensor[128, f32], beta: tensor[128, f32]) -> tensor[64, 128, f32] = layer_norm(x, gamma, beta)\n";
    let scatter = "def f(base: tensor[10, 4, f32], bin_ids: tensor[64, int64], updates: tensor[64, 4, f32]) -> tensor[10, 4, f32] = scatter(base, bin_ids, updates, 0, \"add\")\n";
    let gather = "def f(table: tensor[1000, 128, f32], indices: tensor[64, int64]) -> tensor[64, 128, f32] = gather(table, indices, 0)\n";

    let m = build_and_classify("matmul", matmul);
    let s = build_and_classify("softmax", softmax);
    let l = build_and_classify("layernorm", layernorm);
    let sc = build_and_classify("scatter", scatter);
    let g = build_and_classify("gather", gather);

    eprintln!("== Specialization Dispatch Reality ==");
    eprintln!("(C backend; same dispatch logic governs HIP/Metal targets)");
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
    eprintln!("  matmul     -> SPECIALISED via cblas_sgemm (the cuBLAS analogue)");
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
    // The dispatch reality is asymmetric: only matmul currently gets a
    // specialised library-call codegen. Every other common ML operation
    // either falls through to RISC decomposition (softmax, layer_norm) or
    // bottoms out in a generic runtime function (scatter, gather).

    assert_eq!(
        m.sgemm_calls, 1,
        "matmul must lower to exactly one cblas_sgemm call (the cuBLAS \
         analogue). Got {}.",
        m.sgemm_calls
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
