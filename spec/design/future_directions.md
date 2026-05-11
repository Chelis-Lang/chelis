# Future Directions

Long-horizon entries that are not active work. No design commitment, no
spec lock, no implementation scoping. Entries exist so directions are
not lost between planning cycles and so future capability discussions
can ground against them.

## Kernel authoring as a first-class Chelis capability

Chelis today is a tensor frontend that dispatches to externally-authored kernels (cblas_sgemm, hipblasSgemmStridedBatched, kernel_gather_i64, etc.) through backend-specific lowering paths. Operations without library coverage fall through to generic loops that the target compiler optimizes. The compiler never authors a GPU kernel itself.

The long-horizon direction is to extend Chelis into a full-stack tensor compiler that includes kernel authoring as a first-class capability. A Chelis function would be expressible as either tensor-level code (the current model) or kernel-level code (new), with the same type system, AD machinery, and effect system composing across both layers. The compiler would author GPU kernels itself when no library kernel covers a needed pattern — fused attention blocks, custom reductions, domain-specific fused elementwise chains, FlashAttention-style multi-op kernels.

The strategic value is structural: every "performance tail" gap today is "we need a library kernel that doesn't exist." With kernel authoring in Chelis, the gap becomes "we need a recognizer that emits the right kernel" — and the kernel doesn't have to exist beforehand. Dimension types extend into the kernel layer, foreclosing the index-race class of bugs by construction rather than via witness types. AD composes through kernels automatically, which existing GPU kernel languages (Triton in particular) cannot offer.

The audience implication is that Chelis becomes a unifying language across the researcher-engineer divide in AI/ML workloads. Researchers continue to work at the tensor level; performance engineers gain a path to drop into kernel authoring within the same language and type system, rather than dropping out to CUDA C++ or Triton. The differentiator versus Triton is dimension types from the start, AD composition through kernels, effects as first-class, and full-stack integration with the tensor layer.

Relevant prior art to revisit when this becomes active work: NVIDIA Research's cuda-oxide (Rust-to-PTX compiler via the pliron MLIR-shaped IR framework, pure Rust, no LLVM monorepo dependency), Triton (the existing high-level kernel language), the LLVM NVPTX target. Pliron specifically is a candidate substrate for the kernel-layer IR; its dialect/operation/region model accommodates the kind of multi-target lowering Chelis would need.

Implementation would proceed in phases independently of any specific timeline: kernel functions as a separate compilation target emitting PTX/MSL/HIP-IR; GPU primitives (thread indices, shared memory, atomics) lifted into the type system; tensor↔kernel integration through the existing specialization substrate; cross-cutting features (AD through kernels, specialization on dimension parameters, kernel-level in-place fusion). FlashAttention-style fused attention is the obvious customer-visible v0.1 proof target.

This is recorded as aspirational future work. No design commitment, no spec lock, no implementation scoping. The entry exists so the direction is not lost between planning cycles and so future capability discussions can ground against it.
