# Compiling a functional array language with non-semantic memory information

## Metadata
- **Authors:** Philip Munksgaard, Cosmin Oancea, Troels Henriksen
- **Venue/Year:** IFL 2022 (Symposium on Implementation and Application of Functional Languages)

## Summary
Functional array languages abstract away memory management, exposing only value-based semantics to programmers. While this simplifies programming, it creates a significant challenge for optimizing compilers, which must infer allocation strategies, memory layouts, and data movement to achieve high performance. Traditional approaches lower functional code to imperative intermediate representations (IRs) early in the compilation pipeline, sacrificing high-level functional reasoning and making certain optimizations difficult or impossible to express.

This paper introduces a novel IR design that extends a purely functional core language with "non-semantic" memory information. These memory annotations do not affect the program's observable results but provide the compiler with explicit control over memory blocks and array layouts. By attaching Linear Memory Access Descriptors (LMADs) to array types, the compiler can statically reason about and transform memory access patterns (e.g., slicing, transposing, indexing) at compile time without runtime overhead. The approach preserves functional semantics while enabling imperative-style optimizations like allocation hoisting, memory reuse, and in-place updates.

The work is highly relevant for high-performance compilation of array-oriented languages, particularly for parallel hardware like GPUs. The authors formalize the approach through two toy languages (`Fun` and `FunMem`), provide a translation algorithm, and demonstrate a critical optimization called memory expansion. The design is already deployed in the Futhark compiler, where it has proven effective for generating efficient parallel code while maintaining the safety and expressiveness of functional programming.

## Key Contributions
- Formal definition of `Fun` (memory-agnostic) and `FunMem` (memory-extended) with complete static and dynamic semantics.
- A systematic translation algorithm from `Fun` to `FunMem` that inserts memory allocations and LMAD-based index functions while preserving program validity.
- Demonstration of **memory expansion**, an optimization that hoists dynamic allocations out of parallel GPU kernels by partitioning a pre-allocated shared memory block across threads.
- A deliberate design trade-off: an intentionally unsound type system for memory safety that prioritizes transformation flexibility, with correctness enforced via validity preservation and compiler pass invariants rather than strict typing.

## Technical Approach
The core technical idea is to treat memory information as an **erasable, non-semantic extension** to a functional IR. Arrays in `FunMem` are typed as `[shape]@mem→L`, where `mem` identifies a memory block and `L` is an LMAD (Linear Memory Access Descriptor) defining the layout. An LMAD consists of a base offset and per-dimension strides, enabling compile-time computation of flat memory addresses for affine transformations like slicing, transposing, and indexing.

The paper defines two evaluation semantics for `FunMem`:
1. **Value-based semantics:** Strips memory annotations to recover the original `Fun` program, ensuring functional equivalence.
2. **Heap-based semantics:** Tracks explicit memory blocks, allocations, and read/write traces. Parallel kernels are checked for data races by ensuring disjoint write sets across iterations.

A program is **valid** if both semantics yield identical results. The translation from `Fun` to `FunMem` systematically inserts `alloc` statements for fresh arrays, propagates LMADs through layout operations, and handles control flow (`if` expressions) by returning supporting metadata (sizes, memory blocks, offsets) from both branches. Finally, the paper shows a straightforward lowering from `FunMem` to a simple imperative language (`Imp`), where LMADs are symbolically evaluated to generate flat array accesses.

## Results
The paper is primarily formal and algorithmic, focusing on language semantics, translation correctness, and optimization design rather than empirical benchmarking. It establishes that the `Fun`-to-`FunMem` translation preserves validity and that memory expansion correctly partitions memory across parallel threads while maintaining GPU-friendly coalesced access patterns. The authors note that these techniques are implemented in the **Futhark compiler**, where they have enabled significant real-world performance improvements (referencing prior work [9] for empirical results). Additional optimizations built on this IR include memory block merging (reusing blocks for non-overlapping lifetimes) and memory short-circuiting (eliminating redundant copies by constructing arrays directly in their destination layout).

## Relevance
- **Language Design:** Demonstrates how to safely embed low-level memory concepts into a high-level functional IR without breaking referential transparency or complicating the type system. The "non-semantic" annotation pattern offers a blueprint for adding hardware-aware features to pure languages.
- **Compilers:** Provides a practical IR architecture for array/tensor compilers. LMADs enable compile-time layout reasoning, while the dual-semantics validity model allows aggressive, imperative-style optimizations (allocation hoisting, in-place updates, register-like array reuse) without sacrificing functional correctness guarantees.
- **ML Systems:** Directly applicable to ML compilers and tensor DSLs (e.g., JAX, TVM, PyTorch's functional APIs). The approach addresses core challenges in ML workloads: managing large tensor allocations, avoiding unnecessary memory copies, optimizing memory access patterns for GPUs/TPUs, and supporting nested parallelism—all while preserving the high-level functional abstractions that make ML code composable and debuggable.
