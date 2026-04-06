# A T2 Graph-Reduction Approach To Fusion

## Metadata
- **Authors:** Troels Henriksen, Cosmin E. Oancea
- **Venue/Year:** FHPC'13 (Functional High-Performance Computing), ACM, September 2013

## Summary
Fusion is a foundational compiler optimization that eliminates intermediate data structures by merging producer-consumer operations, significantly reducing memory bandwidth overhead and sometimes improving asymptotic space complexity. While functional languages naturally expose high-level algebraic invariants (e.g., `map`, `reduce`) that make fusion tractable, traditional approaches are often overly conservative: they typically fuse only when a producer's output is consumed exactly once to avoid duplicating computation. In imperative or array-parallel contexts, safely fusing multi-consumer dependencies has historically required complex, "heroic" dependency analyses that are rarely adopted in production compilers.

This paper introduces a novel structural-analysis technique for conservative fusion in L0, a functional-core language designed for nested parallelism on regular multi-dimensional arrays. Inspired by the T1-T2 transformation used to test control-flow graph reducibility, the authors model the producer-consumer data-dependency graph and apply T2-like reductions to safely merge operations. This approach enables fusion even when a producer feeds multiple consumers, provided the dependency graph is reducible and no computation is duplicated. The technique integrates with a compositional algebra of second-order array combinators (SOACs) and includes mechanisms to bypass common fusion inhibitors like `size`, `split`, and `transpose`.

The work matters because it bridges the gap between functional language optimizations and imperative loop transformations, offering a principled, non-fixed-point analysis that scales to real-world parallel workloads. By preserving parallel semantics while aggressively eliminating intermediates, the approach provides a compiler infrastructure capable of generating highly efficient, hardware-independent code for data-intensive domains like quantitative finance and scientific computing.

## Key Contributions
- A structural, backward data-flow analysis that applies T2-like graph reduction to safely fuse SOACs across multiple consumers without duplicating computation.
- A compositional fusion algebra for `map`, `reduce`, `filter`, and an internal `redomap` construct that preserves parallel execution semantics.
- An inhibitor-propagation mechanism that rewrites or lifts calls to `size`, `split`, `transpose`, and `assertZip` so they do not block fusion.
- Proposal of two complementary transformations, ISWIM/IRWIM (interchanging scan/reduce with inner maps) and REDFLAT (reduce flattening), which both enable and benefit from fusion to increase exploitable parallelism.
- Implementation in the L0 compiler with detailed instrumentation statistics across six benchmarks, demonstrating successful fusion patterns in real-world and algorithmic kernels.

## Technical Approach
The core idea adapts the T2 reduction rule from control-flow graph theory to the data-dependency graph of array combinators. In a T2 reduction, a node with a single incoming edge is merged into its successor. The authors apply this concept to SOACs: if a producer's output feeds only one downstream kernel (or multiple kernels on disjoint control-flow paths), it can be structurally reduced into the consumer.

The compiler performs a bottom-up (backward) data-flow analysis over a normalized AST. It maintains a `FusionRes` structure tracking:
- `outArr`/`inpArr`: mappings of arrays to producing/consuming kernels.
- `unfusable`: a set of arrays that cannot be fused due to in-place updates, usage outside SOACs, multiple uses on shared execution paths, or incompatible SOAC combinations.
- `kers`: metadata for each fusion kernel, including input arrays and an `inplace` set tracking aliasing constraints.

Fusion is permitted only when four conditions hold: (1) outputs are not in the `unfusable` set, (2) at least one compatible consumer kernel exists, (3) all consumers are algebraically compatible, and (4) no in-place update aliasing violations would occur. The compositional algebra defines rewrite rules (e.g., `map ∘ map → map`, `reduce ∘ map → redomap`, `filter ∘ filter → filter` under subset conditions). `redomap` is introduced as an internal construct to seamlessly fuse reductions with maps/filters while preserving parallelism.

After identifying fusable kernels, a top-down pass substitutes fused SOACs, cleans up lambda bodies, and recursively re-applies fusion to nested structures. Inhibitor functions are handled by associating them with kernels and translating their arguments to reference original producer arrays, effectively "pushing" them out of the fusion path.

## Results
The fusion algorithm was implemented in the L0 compiler (~1000 lines of Haskell). Since L0 was interpreted at the time of publication, the authors report compiler instrumentation statistics rather than runtime speedups across six benchmarks: two real-world financial kernels (P0, P1), a stochastic volatility solver (P2), a shortest-path algorithm (P3), flat-parallel matrix multiplication (P4), and maximal segment sum (P5).

Key findings include:
- `map ∘ map` and `reduce ∘ map` fusions are the most frequent, with `reduce ∘ replicate` also common due to size-matching patterns.
- The analysis successfully transforms flat-parallel matrix multiplication into a standard three-level nested structure, eliminating all intermediate `replicate` arrays and materializations.
- A single `redomap ∘ filter` fusion in a Sobol random-number generator would theoretically increase parallelism by 32× and enable efficient GPU execution via segmented reductions.
- The `unfusable` tracking correctly prevents illegal fusions across in-place updates, shared control-flow paths, and incompatible SOAC combinations.
- The authors note that applying the proposed ISWIM transformation to benchmark P0 would unlock an additional parallelism dimension of size 365, highlighting the synergy between fusion and loop interchange.

## Relevance
- **Language Design:** Demonstrates how a carefully constrained core language with second-order combinators and uniqueness types can expose high-level invariants that make advanced, safe optimizations tractable. It shows how functional purity and imperative performance can be reconciled through compiler-enforced aliasing and consumption semantics.
- **Compilers:** Introduces a novel, non-iterative structural analysis for fusion that handles multi-consumer dependencies conservatively. The T2-reduction metaphor offers a fresh, mathematically grounded alternative to fixed-point data-flow or rewrite-rule-based fusion, with clear termination and correctness guarantees.
- **ML Systems & Tensor Compilers:** Highly applicable to modern computational graph compilers (e.g., JAX, TVM, XLA) where fusion is critical for minimizing memory traffic and kernel launch overhead. The compositional algebra, inhibitor handling, and safe multi-consumer fusion directly address challenges in optimizing nested parallelism, segmented reductions, and automatic differentiation graphs. The approach to avoiding work duplication while fusing shared intermediates is particularly relevant for memory-bound ML workloads and GPU/TPU code generation.
