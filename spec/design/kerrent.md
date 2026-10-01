# Kerrent: GPU kernel authorship inside Chelis

## Goal

A Chelis language feature for authoring GPU kernels in Chelis source code. Users write kernel-level computations using Chelis syntax, with Chelis's type discipline applied; the compiler lowers these to executable GPU kernels.

The name is Kerrent. It is a core language feature, not a shell — adding kernel authorship requires new syntax, new IR layers, and new lowering paths that can't be expressed as library code.

## Why this matters

Today, Chelis's tensor operations lower through the RISC DAG to backends that either call vendor libraries (BLAS, hipBLAS) or emit generic loops the target compiler optimizes. Patterns that vendor libraries don't cover and that don't fuse cleanly under generic compilation — FlashAttention, custom domain reductions, fused operations across windows broader than the current in-place set — sit in a gap.

That gap is the dominant performance cost for transformer-shaped workloads. The attention block materializes the full [batch, heads, seq, seq] intermediate because there's no fused-attention kernel for Chelis to call. Sequence length scales the cost quadratically. Modern frameworks ship hand-authored kernels for these cases; Chelis doesn't, and falls behind on the workloads where these kernels matter.

Kerrent closes the gap by letting Chelis author its own kernels — under the same type system, same dimension types, same AD machinery, same property verification — and call them from regular tensor-level Chelis code. When Kerrent ships the attention kernel, the transformer block in Chelis no longer materializes the full attention intermediate.

The broader strategic value is bigger than performance. Kerrent unlocks several patterns that aren't expressible without it: custom kernels for domain operations vendor libraries don't serve, dimension-typed safety for GPU code that today relies on convention, AD that composes through kernels without hand-written backward passes. These are differentiating capabilities, not just performance fixes.

## Target backend technology

The first effort targets Triton as the kernel compiler.

Triton is a mature GPU kernel compiler from OpenAI with cross-vendor support (NVIDIA via PTX, AMD via AMDGCN). Its IR is an MLIR dialect, well-documented and stable. Its compiler handles tile selection, shared memory allocation, autotuning, and target-code emission. The integration scope is bounded: Chelis emits Triton IR, Triton compiles it, the resulting kernel is callable from regular Chelis code through the existing backend kernel-launch machinery.

The alternatives considered and rejected for the first effort:

- **Direct MLIR with custom dialects.** Most flexible but most work. Chelis would need to build its own GPU dialect, write lowering passes to NVPTX and ROCDL, ship without Triton's autotuning infrastructure. Right long-term direction; wrong v1.
- **CUDA-oxide (Rust to PTX).** NVIDIA-only. Would force Chelis kernels into Rust-shaped syntax or require reimplementing cuda-oxide's lowering. Doesn't compose cleanly with Chelis's dimension types.
- **TVM/TileLang.** Comparable cost to Triton with less ecosystem momentum and weaker cross-vendor story.

Triton wins on speed-to-working-capability, cross-vendor reach (NVIDIA + AMD), and ecosystem stability. The integration is bounded enough to ship as a v1 milestone rather than as an open-ended research effort.

## What Kerrent v1 delivers

The user-facing capability:

```
kernel def fused_attention[B, H, S, D](
  q: tile[B, H, S, D, f16],
  k: tile[B, H, S, D, f16],
  v: tile[B, H, S, D, f16]
) -> tile[B, H, S, D, f16] = ...
```

A new `kernel` keyword marks the function as a kernel rather than tensor-level code. The function body uses tile-level operations (load, store, dot, reduce, mask) that don't exist in regular tensor Chelis. Dimension types extend to tile-level scope. The function is callable from regular Chelis tensor code; the call lowers through the existing backend to a Triton-compiled kernel.

The capability includes:

- A `kernel` annotation on Chelis functions distinguishing them from tensor-level code
- Tile-level primitives: `tile.load`, `tile.store`, `tile.dot`, `tile.reduce`, `tile.mask`, and a small set of others matching Triton IR's operator set
- Dimension types extending to tile scope, so kernel-level code gets the same shape-checking discipline as tensor-level code
- Lowering from kernel-annotated Chelis through a kernel dialect to Triton IR
- Build integration: when a Chelis program compiles, kernel-annotated functions get Triton-compiled and the resulting binaries link into the final executable
- Runtime integration: kernel calls from tensor-level Chelis emit launches against the Triton-compiled artifacts, using the existing HIP/CUDA backend machinery for argument marshaling and grid configuration

What v1 doesn't include:

- Thread-level addressing. Triton works at block level; Kerrent v1 inherits that abstraction.
- Custom shared memory patterns beyond what Triton's primitives express.
- Warp-level primitives (shuffles, ballots, votes).
- AD composition through kernels. v1 kernels are forward-only; backward passes are hand-written or absent.
- Property verification on kernel bodies beyond standard dimension typing.

These are all addressed in later phases.

## v1 milestones

The work decomposes into bounded milestones:

**Milestone 1: Kernel syntax and parser.** The `kernel` annotation and tile-level primitives parse correctly. Kernel-annotated functions get a distinct AST representation that the rest of the pipeline can distinguish from tensor-level functions.

**Milestone 2: Kernel IR layer.** A new IR layer between Chelis's RISC DAG and Triton IR. Kernel functions lower to this intermediate representation; tile-level operations become first-class IR nodes. Type checking on kernel functions runs against this IR.

**Milestone 3: Triton IR emission.** Lowering pass from the kernel IR to Triton's MLIR dialect. Each tile-level operation has a defined Triton IR equivalent. Output validates against Triton's IR specification.

**Milestone 4: Build integration.** Triton compiler invocation from the Chelis build pipeline. Kernel artifacts (compiled PTX or AMDGCN) get produced and linked. Build system handles the Triton dependency cleanly.

**Milestone 5: Runtime integration.** Kernel calls from tensor-level Chelis lower to launches against Triton artifacts. Memory layout matching at the kernel boundary works correctly. Existing backends (HIP, CUDA) handle the launch machinery.

**Milestone 6: First production kernel.** FlashAttention-shaped fused attention kernel written in Kerrent, replacing the current attention decomposition.

The first five milestones are infrastructure. Milestone 6 is the proof point: a single hand-authored Kerrent kernel that demonstrates the capability end-to-end and produces a measurable performance improvement.

## Guarantees Kerrent v1 provides

The guarantees Chelis users get when their code includes Kerrent kernels:

**Dimension type checking at kernel boundaries.** A kernel's input and output dimension types are checked at every call site. Mismatched shapes fail at type-check, not at runtime.

**Dimension type checking inside kernel bodies.** Tile-level operations have typed inputs and outputs. A `tile.dot` between mismatched tiles fails at type-check. Subscript indexing into tiles checks bounds when the bounds are statically known.

**No undefined behavior from shape errors.** The shape errors that bite hand-written CUDA — wrong stride, transposed tensor, off-by-one in tile size — become compile-time errors in Kerrent.

**Existing Chelis property verification extends to kernel-using code.** Properties attached to functions that call kernels verify by sampling. The kernel itself is opaque to property sampling (it runs as compiled code); the function's contract is verified as usual.

**Cross-vendor portability.** A Kerrent kernel runs on NVIDIA via Triton's PTX backend and on AMD via Triton's AMDGCN backend. Same source, both targets.

What Kerrent v1 doesn't guarantee:

- Correctness of the kernel logic itself. The user is responsible for writing a correct algorithm. Kerrent verifies shape discipline, not algorithmic correctness.
- Performance bounds. Triton handles autotuning; the resulting performance is whatever Triton produces.
- AD correctness through kernels. v1 kernels are forward-only.

## What Kerrent v1 unlocks for users

Concrete capabilities that ship when v1 ships:

**Fused attention for transformer inference.** With a FlashAttention-shaped Kerrent kernel replacing the current attention decomposition, transformer inference (encoder-only, eventually decoder when KV cache support lands) no longer materializes the attention intermediate, so its quadratic memory cost in sequence length goes away.

**Custom domain kernels.** Users in domains where vendor libraries don't cover the needed operations can author kernels in Kerrent. Finance-specific reductions, sparse-pattern operations for graph workloads, custom convolutions for image processing — all become expressible in Chelis source.

**Type-safe GPU code.** The class of GPU correctness bugs that depend on convention (correct stride, correct tile alignment, correct memory layout) become impossible by construction.

**Single-source cross-platform deployment.** A Chelis program with Kerrent kernels runs on both NVIDIA and AMD without source changes. Users deploying across vendor environments stop maintaining parallel kernel implementations.

The audience for this is the users Chelis already serves: AI/ML practitioners with mainstream inference workloads, finance users with domain-specific computation needs, research users building differentiable simulators and probabilistic programs.

## Addendums: paths beyond v1

The v1 effort is bounded and addresses the dominant near-term gaps. Several extensions take the capability further.

### Addendum A: AD composition through kernels

The single most strategically interesting extension. Today, kernel authors write backward passes by hand. Triton kernels with gradient flow require parallel forward and backward kernels coded explicitly. Chelis's AD machinery doesn't extend through kernels in v1.

The work: extend the AD transform to operate on kernel-level IR. A Kerrent kernel for forward computation gets a corresponding Kerrent kernel for backward computation, generated by the AD transform. The transformation operates at the tile level, producing tile-level backward operations.

The value: research workloads — gradient analysis, model interpretability, fine-tuning kernel-shaped operations — become tractable without hand-coding backward passes. This is genuinely unique. PyTorch and JAX don't offer it; Triton itself doesn't offer it.

The complexity: real. The AD transform at tile level requires new gradient rules for tile-level operations. Some operations don't have clean tile-level gradients and need composition through multiple tiles. The infrastructure is substantial but the path is clear.

### Addendum B: Thread-level addressing

Triton operates at block level; the kernel addresses tiles rather than individual threads. Some patterns (irregular memory access, fine-grained synchronization, certain communication patterns) need thread-level control.

The work: extend Kerrent with thread-level primitives — thread index types, shared memory regions with explicit synchronization, warp-level operations (shuffles, ballots). The lowering still goes through Triton where possible; for patterns Triton can't express, route through direct MLIR dialect emission.

The value: workloads that need thread-level control become expressible. Particle methods, irregular graph algorithms, certain sparse patterns.

The complexity: moderate. The type system needs extending to thread scope. The lowering needs to handle the case where Triton can't express the pattern and fall back to lower-level emission.

### Addendum C: Custom shared memory patterns

Triton handles shared memory allocation automatically based on its tile model. Some patterns benefit from explicit control: shared memory used as a fixed cache, shared memory partitioned in non-tile ways, shared memory with custom synchronization.

The work: expose shared memory as a region type in Kerrent. The user can allocate shared memory regions, define access patterns, control synchronization explicitly.

The value: patterns where Triton's automatic shared memory allocation is suboptimal can be hand-tuned. Specific cases in dense linear algebra, certain reductions, custom communication patterns.

The complexity: moderate. The type system needs region types with explicit lifetimes. The lowering needs to route around Triton's automatic allocation when the user is managing it manually.

### Addendum D: MLIR-direct lowering

For patterns Triton's IR doesn't express well, route through direct MLIR dialect emission rather than Triton IR. This is the cuda-oxide-shaped path: Chelis defines its own GPU dialects, lowers them to NVPTX and ROCDL through standard MLIR pipelines.

The work: build a Chelis GPU dialect parallel to (or replacing) Triton emission. Lower from kernel IR through this dialect to target code. Reuse upstream MLIR infrastructure for NVPTX and ROCDL emission.

The value: capability that Triton can't express. Vendor-specific intrinsics (tensor cores beyond what Triton exposes, specific instructions only available on specific architectures). The platonic ideal of Chelis-native kernel compilation.

The complexity: substantial. This is rebuilding parts of what Triton already does. Worth doing only when Triton's abstraction has become a meaningful constraint.

### Addendum E: Verified kernel bodies

Property verification today covers function-level contracts. Extending it to kernel bodies — proving that a kernel's tile-level operations satisfy stated properties — unlocks a unique capability.

The work: extend the property verification machinery to sample tile-level inputs and verify kernel-level properties. Define what properties make sense at the tile level (memory access bounds, output value ranges, structural invariants).

The value: kernels with verified properties for trust-sensitive deployments. The combination of dimension-typed GPU code plus property-verified GPU code doesn't exist anywhere.

The complexity: substantial. Property sampling at the tile level is harder than at the tensor level because the operations are lower-level. The infrastructure needs careful design.

### Addendum F: Direct kernel platforms beyond NVIDIA and AMD

Triton covers NVIDIA and AMD. Apple Silicon, embedded GPUs, novel accelerators sit outside this. The MLIR-direct path (Addendum D) opens these as targets.

The work: lower Chelis's kernel IR through MLIR to target dialects for these platforms. Apple Silicon would go through MLIR's Metal compute support (when that stabilizes). Embedded and novel accelerators go through whatever MLIR dialect their vendor provides.

The value: cross-platform deployment that includes platforms vendor libraries don't cover. Edge deployment, novel hardware research, Apple Silicon as a first-class target.

The complexity: depends on the target. Each platform has its own MLIR maturity level and its own quirks.

## Sequencing the addendums

The addendums divide into priority tiers:

**High priority once v1 ships.** Addendum A (AD through kernels) is the highest-leverage extension. It's the capability that genuinely differentiates Chelis from other frameworks and aligns with the differentiable-language strategic direction. Should be the first major Kerrent extension.

**Medium priority.** Addendum B (thread-level) and Addendum C (shared memory) extend the capability surface to patterns Triton can't express well. Worth doing when user demand surfaces specific patterns that Triton handles poorly.

**Lower priority.** Addendum D (MLIR direct), Addendum E (verified kernels), Addendum F (additional platforms) are larger investments with longer payoff windows. Each is real strategic value but each is substantial work. Schedule when other priorities clear.

The v1 effort plus Addendum A together is the right scope to plan against. v1 produces the kernel authorship capability. Addendum A produces the AD-through-kernels capability that no other framework offers. The combination is what makes Kerrent strategically distinctive rather than just performance-competitive.

## Net

Kerrent is the path to kernel authorship in Chelis. v1 targets Triton as the kernel compiler, ships a bounded language feature with dimension-typed GPU code and cross-vendor portability, and produces the FlashAttention-shaped kernel that closes the dominant transformer performance gap. The work decomposes into six milestones, each bounded.

The strategic value of Kerrent extends beyond performance. Type-safe GPU code, custom domain kernels, and (via Addendum A) AD composition through kernels are capabilities no other framework offers in combination. The audience that values these is the same audience that values the rest of Chelis: AI/ML practitioners, finance users, research users in differentiable programming.

The first effort is real work but bounded by what Triton provides. The addendums extend the capability over time as user demand and project priority justify. The v1 plus Addendum A combination is the most-leverage scope to plan against; everything else is opportunistic extension.
