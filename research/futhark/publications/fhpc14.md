# Size Slicing - A Hybrid Approach to Size Inference in Futhark

## Metadata
- **Authors:** Troels Henriksen, Martin Elsman, Cosmin E. Oancea
- **Venue/Year:** FHPC'14 (ACM, 2014)

## Summary
This paper introduces "size slicing," a hybrid shape inference technique for Futhark, a purely functional array language designed for nested parallelism on massively parallel hardware like GPGPUs. Compiling high-level array programs to accelerators typically requires static memory allocation and precise knowledge of array shapes at compile time. However, Futhark prioritizes programmer productivity by allowing dynamic shape operations (e.g., `filter`), which traditionally complicate static analysis and prevent straightforward static allocation. To bridge this gap, the authors propose a compiler transformation that infers precise shape information at runtime while preserving the original program's asymptotic computational complexity.

The core innovation is size slicing, which separates the computation of array shapes from the computation of their values. When a cost model determines that shape computation is inexpensive (e.g., contains no loops or recurrences), the compiler splits functions into a lightweight shape slice and a value slice. The shape slice is executed first, eliminating existential quantifiers and enabling static memory allocation for the subsequent value computation. For inherently shape-dynamic constructs, the system falls back to computing shapes alongside values, ensuring correctness without sacrificing asymptotic performance. This hybrid strategy allows Futhark to maintain a flexible, high-level syntax while generating efficient, statically-allocated code for GPGPUs.

The technique is integrated into Futhark's compiler pipeline and relies heavily on standard optimizations like inlining, fusion, and dead-code elimination to reduce shape-related overhead to negligible levels. By decoupling shape inference from value computation, the approach avoids the strictness and complexity of full dependent type systems while still providing the compiler with the information needed for aggressive parallelization, bounds checking, and memory management.

## Key Contributions
- Introduces **size slicing**, a hybrid analysis that separates shape computation from value computation to enable static memory allocation on GPGPUs.
- Formalizes shape inference using **existentially-quantified dependent types**, allowing the compiler to handle dynamic shapes without restricting the source language or burdening programmers.
- Provides a set of **syntax-directed transformation rules** that annotate the intermediate representation with shape information and insert runtime assertions for array regularity invariants.
- Demonstrates that **standard compiler optimizations** (inlining, constant folding, dead-code elimination, loop hoisting) can reduce shape-computation overhead to O(1) or asymptotically negligible levels.
- Validates the approach through qualitative evaluation on micro-benchmarks and real-world applications, showing minimal overhead while preserving asymptotic work complexity.

## Technical Approach
- **Shape-Dependent Typing:** Extends Futhark's IR with dependent types where array dimensions are tracked as explicit integer parameters. Result shapes are existentially quantified (`∃s`), meaning they are computed at runtime alongside values.
- **Existential Transformation:** Every function is transformed into an `f_ext` version that returns both shape integers and array values. Assumed invariants (e.g., regularity in `mapT`) are made explicit via runtime assertions.
- **Size Slicing:** A cost-model-driven pass splits `f_ext` into `f_shape` (returns only dimensions) and `f_value` (takes dimensions as inputs, returns only data). Slicing is applied only when `f_shape` contains no recurrences/loops, ensuring it remains cheap and loop-free.
- **SOAC Handling:** For combinators like `mapT`, `filterT`, and `reduceT`, the compiler uses either pre-assertion (compute shapes first, verify uniformity across elements) or intra-assertion (check shape of first element, assert for others). Later optimization passes hoist and simplify these checks.
- **Optimization Pipeline:** The transformed IR undergoes aggressive simplification. This eliminates redundant shape computations, collapses `map`-over-shapes into `replicate`, and reduces assertion checks to O(1) operations. The pipeline also separates predicate checks from computational kernels, enabling safe hoisting out of parallel loops.
- **Fallback Mechanism:** When slicing is unsafe or too costly (e.g., recursive shape dependencies), the system retains the existentially-quantified form, computing shapes and values together while guaranteeing asymptotic work preservation.

## Results
- Evaluated qualitatively on micro-benchmarks (matrix multiplication, Floyd-Warshall) and three real-world applications (100s–1000s LOC).
- Human inspection of generated code confirms that shape computation and bounds-checking overhead is asymptotically smaller than the original program's work.
- In most cases, overhead is reduced to **O(1) operations** after optimization. For complex real-world benchmarks, overhead is on the order of hundreds of operations compared to tens of millions of computational operations.
- The approach successfully eliminates existential quantifiers in common cases, enabling static memory allocation and making the generated code suitable for GPGPU execution without dynamic allocation or heavy runtime checks.

## Relevance
- **Language Design:** Demonstrates how to integrate dependent-type-like shape tracking into a practical, high-level array language without burdening programmers with complex type annotations or restricting expressiveness. It offers a "pay-as-you-go" model for shape safety.
- **Compilers:** Provides a blueprint for hybrid static/dynamic analysis in compiler pipelines. The combination of existential quantification, function slicing, and aggressive optimization bridges the gap between high-level functional abstractions and low-level hardware constraints (e.g., static allocation, kernel generation).
- **ML Systems:** Highly relevant to modern tensor/array compilers (e.g., XLA, TVM, MLIR) that must handle dynamic shapes in neural networks. The size-slicing technique offers a principled way to separate shape inference from computation, enabling efficient memory planning, operator fusion, and hardware code generation while supporting dynamic operations like masking, filtering, or variable-length sequences.
