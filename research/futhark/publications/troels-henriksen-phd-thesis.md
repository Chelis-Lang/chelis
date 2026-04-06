# Design and Implementation of the Futhark Programming Language

## Metadata
- **Authors:** Troels Henriksen (Supervisors: Cosmin Eugen Oancea and Fritz Henglein)
- **Venue/Year:** PhD Thesis, University of Copenhagen, December 2017

## Summary
This thesis presents the design and implementation of Futhark, a purely functional, data-parallel array language optimized for high-performance GPU execution. Futhark bridges the gap between high-level functional programming and low-level hardware constraints by providing a machine-neutral programming model that compiles to efficient OpenCL code. The language deliberately restricts features that hinder predictable parallel execution (e.g., recursion, lazy evaluation, unrestricted side effects) while introducing targeted constructs like explicit parallel combinators, size parameters, and uniqueness types to enable safe, high-performance array programming.

The core challenge addressed is the impedance mismatch between nested, high-level functional parallelism and the flat, restricted execution model of modern GPUs. Rather than relying on full flattening (which often introduces excessive overhead and destroys locality information) or manual low-level tuning, the thesis introduces a "moderate flattening" strategy. This approach selectively exploits easily accessible parallelism, efficiently sequentializes excess parallelism, and preserves high-level structural invariants to enable downstream locality optimizations.

The work matters because it demonstrates that functional array languages can achieve performance competitive with hand-tuned GPU code while maintaining strong safety guarantees, modularity, and hardware agnosticism. By open-sourcing the compiler and providing a comprehensive benchmark suite, the thesis establishes a practical foundation for future research in parallel functional programming, compiler optimization, and automatic GPU code generation.

## Key Contributions
- **Moderate Flattening Transformation:** A novel algorithm that rewrites nested parallel constructs into flat, perfectly nested patterns suitable for GPUs, selectively exploiting parallelism while sequentializing excess work without destroying access-pattern information.
- **Size-Dependent Type Inference:** A lightweight system using existential types and function slicing to statically infer array shapes, enabling compile-time reasoning about dimensions without programmer annotations.
- **Extended Parallel Combinators & Fusion Algebra:** Introduction of `stream_red`, `stream_map`, `redomap`, and `stream_seq` combinators with aggressive vertical and horizontal fusion rules that preserve parallelism and avoid computation duplication.
- **Uniqueness Types for Safe In-Place Updates:** A type-system extension that guarantees referential transparency while allowing efficient, alias-safe in-place array modifications and parallel scatter operations.
- **Comprehensive Empirical Validation:** Evaluation across 21 non-trivial benchmarks showing performance competitive with hand-written OpenCL/CUDA code and superior to existing functional GPU frameworks like Accelerate.
- **Open-Source Compiler Release:** A fully documented, ~45k-line Haskell compiler made publicly available to serve as both a research testbed and a practical tool for parallel array programming.

## Technical Approach
The Futhark compiler is structured as a syntax-directed translation pipeline operating on a typed intermediate representation (IR). Key technical components include:
- **Size Inference Pipeline:** Transforms an untyped IR into a sized IR by injecting existential size parameters for array dimensions. Redundant size computations are eliminated via aggressive inlining, copy propagation, and dead-code elimination, ensuring negligible runtime overhead.
- **Fusion via Dataflow Reduction:** Implements a T2-reduction algorithm on a dependency graph to fuse producer-consumer and independent SOACs. Extended combinators (`redomap`, `scanomap`, `stream_seq`) enable fusion across complex patterns (e.g., map-reduce, scan-map) without duplicating work or losing parallelism.
- **Moderate Flattening & Kernel Extraction:** Applies map-loop interchange, distribution, and SOAC decomposition rules to extract flat parallel kernels. The algorithm avoids exploiting parallelism inside branches or when it would create irregular arrays, preserving analyzable structure for later passes.
- **Locality Optimizations:** Automatically repairs non-coalesced memory accesses by inserting `manifest` operations that transpose arrays in memory. Performs 1D and 2D loop tiling by recognizing `stream_seq` invariants and mapping chunks to GPU local memory (`local` construct).
- **Uniqueness & Alias Analysis:** Combines intra-procedural alias tracking with uniqueness annotations to statically verify that in-place updates and function calls do not violate functional purity or introduce data races.

## Results
- Evaluated on 21 benchmarks from Rodinia, Parboil, FinPar, and Accelerate, executed on NVIDIA K40 and AMD W8100 GPUs.
- Performance ranges from **0.21× slowdown to 13× speedup** compared to reference implementations, with the majority of benchmarks showing competitive or superior performance.
- Key optimization impacts: Fusion yields up to **10× speedup** (e.g., Crystal, LocVolCalib); in-place updates provide **8.3× speedup** for k-means; coalescing via transposition delivers up to **9.26× speedup**; loop tiling improves performance by **1.1–2.3×**.
- Demonstrates that high-level functional abstractions, when paired with targeted compiler transformations, can match or exceed hand-tuned GPU code without manual tuning or dataset-specific heuristics.

## Relevance
- **Language Design:** Illustrates how to balance expressiveness and performance by restricting problematic features (recursion, laziness, side effects) while adding targeted, semantics-preserving constructs (uniqueness types, streaming SOACs, size parameters).
- **Compiler Optimization:** Provides practical techniques for bridging high-level functional IRs and low-level GPU architectures, including moderate flattening, graph-based fusion, automatic memory layout transformation, and locality-aware tiling.
- **ML & Tensor Systems:** Highly relevant to modern ML compiler stacks (e.g., JAX, TVM, XLA, PyTorch) that rely on functional array semantics, operator fusion, and automatic parallelization. Futhark's size inference, fusion algebra, and memory coalescing strategies directly inform how tensor compilers optimize compute graphs for accelerators.
- **Research & Reproducibility:** Offers a fully open, well-documented compiler that serves as a baseline for exploring auto-tuning, irregular parallelism handling, multi-versioned code generation, and next-generation GPU programming models.
