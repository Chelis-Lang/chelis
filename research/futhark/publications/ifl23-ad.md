# Reverse-Mode AD of Multi-Reduce and Scan in Futhark

## Metadata
- **Authors:** Lotte Maria Bruun, Ulrik Stuhr Larsen, Nikolaj Hinnerskov, Cosmin Oancea
- **Venue/Year:** IFL 2023 (The 35th Symposium on Implementation and Application of Functional Languages)

## Summary
Automatic Differentiation (AD) is foundational to machine learning and scientific computing, yet efficiently computing gradients for high-level parallel array operations on modern hardware remains challenging. This paper presents reverse-mode AD algorithms for three core parallel combinators in the Futhark functional array language: `reduce`, `scan` (prefix sum), and `reduce-by-index` (multi-reduce/histogram). Rather than lowering these operations to scalar memory accesses before differentiation, the authors differentiate them directly at the combinator level, preserving parallel semantics and enabling compiler-level reasoning.

The general-case differentiation rules for these combinators often rewrite gradients in terms of less efficient operations (e.g., differentiating `reduce` requires `scan`, and `reduce-by-index` requires sorting and segmented scans). While asymptotically sound, these general rules incur large constant overheads. To address this, the paper introduces a suite of compiler-driven specializations for common operators, vectorized operations, invertible operators, and sparse Jacobian structures. These transformations drastically reduce memory traffic and synchronization costs, making high-level AD practical for GPU execution.

The work culminates in the first comprehensive empirical evaluation of reverse-mode AD for these combinators on an NVIDIA A100 GPU. The results demonstrate that specialized rules achieve low AD overheads (typically 1.5×–4×) and consistently outperform prior high-level AD approaches. The paper establishes that differentiating bulk-parallel operators at a high level is not only theoretically sound but also highly performant when paired with targeted compiler optimizations.

## Key Contributions
- **General-case reverse-mode AD algorithms** for `reduce-by-index` and `scan`, with formal derivations that preserve parallel work complexity.
- **A suite of compiler specializations** for common operators (+, min, max, *), vectorized operations, invertible/commutative operators, and block-diagonal sparsity patterns that dramatically reduce constant-factor overheads.
- **First GPU performance evaluation** of reverse-mode AD for `reduce`, `scan`, and `reduce-by-index`, providing a practical baseline and demonstrating significant speedups over existing high-level AD systems (e.g., PPAD).
- **Demonstration that high-level AD can be both asymptotically efficient and practically fast** on GPUs when integrated with a compiler that supports fusion, operator interchange, and sparsity-aware code generation.

## Technical Approach
The core methodology revolves around differentiating second-order array combinators directly, using program transformation rules rather than tape-based or memory-level differentiation.

- **General-Case Derivations:** 
  - `reduce` differentiation computes forward and reverse partial accumulations via exclusive scans, then applies the chain rule element-wise.
  - `reduce-by-index` generalizes this by sorting key-value pairs, computing segmented forward/reverse scans, and scattering adjoints back to original indices.
  - `scan` differentiation is derived via loop unrolling and dependence analysis, resulting in a backward linear recurrence solved by another scan over computed Jacobians.
- **Specializations:** 
  - **Common Operators:** Hand-optimized rules avoid expensive scans/sorts (e.g., `reduce(+)` becomes a simple broadcast-add; `reduce(*)` tracks zero counts to enable division-based gradients).
  - **Vectorized Interchange:** Rewrite rules (`Irwim`, `Iswim`, `Irbiwim`) push `map` outside `reduce`/`scan`/`reduce-by-index`, reducing differentiation to scalar operators and avoiding inefficient array-typed combinators.
  - **Invertible Operators:** Users can declare lifted operators with inverses, allowing gradients to be computed within the original combinator (e.g., `reduce` stays a `reduce`) by tracking auxiliary state.
  - **Sparsity Optimizations:** Compiler analysis detects Block-Diagonal (BDS) and Redundant Block-Diagonal (RBDS) Jacobian structures (common in matrix multiplication and linear recurrences), shrinking intermediate representations and improving GPU cache/scratchpad utilization.
- **Compiler Integration:** All transformations are implemented in Futhark's compiler, leveraging existing fusion, dead-code elimination, and single-pass GPU scan implementations to minimize memory traffic and kernel launches.

## Results
Evaluated on an NVIDIA A100 GPU using single-precision floats across varying dataset sizes, performance is measured via **AD overhead** (runtime of differentiated code / runtime of primal).

- **`reduce`:** Specialized rules achieve overheads of 1.5×–3.3×. The general case for matrix multiplication yields 4.8×–7.8× overhead. Outperforms the PPAD baseline significantly, especially for vectorized multiplication (PPAD: 300×–700× vs. Ours: <4×).
- **`scan`:** Addition overhead <1.8×. Matrix multiplication with RBDS sparsity peaks at 7.9× (3×3) but drops to 4.1× (5×5). Vectorized multiplication achieves <0.6× overhead. PPAD with aggressive dead-code elimination narrows the gap but remains slower on low-arity operators.
- **`reduce-by-index`:** Addition/min/multiplication overheads range from 1.5×–2.6×. The general sorting-based approach for non-invertible ops (`sumOfProd`, `satAdd`) is inefficient (up to 81×), but applying the invertible-operator specialization manually reduces overhead to ~1.1×–1.8×.
- **Key Takeaway:** General-case algorithms are theoretically sound but suffer from high constant factors due to sorting, extra scans, and large Jacobian materialization. Specializations are essential for practical GPU performance, consistently keeping overheads within a small constant factor of the primal.

## Relevance
- **Language Design:** Demonstrates how to design AD for functional array languages by differentiating higher-order combinators directly, preserving parallel semantics and avoiding the semantic gap between high-level code and low-level gradient computation.
- **Compilers:** Highlights that theoretical AD efficiency is insufficient without compiler-level optimizations. The paper showcases the critical role of operator interchange, sparsity detection, fusion, and dead-code elimination in bridging the gap between mathematical derivations and hardware performance.
- **ML Systems:** Provides a blueprint for building efficient, parallelism-preserving gradient computation in ML frameworks. By avoiding tape-based storage and differentiating bulk operations at a high level, this approach mitigates memory bottlenecks common in reverse-mode AD, offering a path toward scalable, hardware-aware autodiff for scientific computing and large-scale model training.
