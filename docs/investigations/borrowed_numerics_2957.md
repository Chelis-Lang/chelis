# Borrowed numerics (chelis#2957): assessment and prior art

Assessment of the structural fix for the borrowed-numerics class tracked by
[chelis#2957](https://github.com/Chelis-Lang/chelis/issues/2957). Taken 2026-10-02 at
`main` `bbc40a92a`. This record is evidence and a recommendation; the decision is made in
the numbered spec, and the implementation sequence in a design document.

## 1. The problem in one paragraph

Chelis takes its transcendentals from outside the compiler: host libm, Apple Accelerate
vForce, a latent Sleef path, the C compiler's constant folder, a C compiler and flags
chosen from the environment, and the process floating-point environment. `chelis eval`
and `chelis build --target c` therefore disagree (up to 32 ULP for `tanh` at f32), one
built program can return three values for one input depending on what the optimizer can
see, and bits change across machines. That breaks 0.19 guarantee 2 (#1362) and the
determinism contract in `spec/00-context.md` ("for fixed program text, compiler build,
target, and declared inputs, every check, evaluation, and build result is a function of
those inputs").

## 2. What the active documents say

The documents disagree with one another, so the language decision precedes any code.

- `[05-OP-46]` defines `exp`, `log`, `sin`, `cos`, `tan`, `atan` as "its named
  mathematical operation". Only `sqrt` is required to be correctly rounded.
- `[05-OBS-3]` gives those six ops a 1-ULP cross-lane bound and states that the table
  covers "one lane's libm or SLEEF or vForce against another's". Every operation absent
  from the table, compound builtins included, has a zero-ULP bound.
- The compound activations (`sigmoid`, `tanh`, `silu`, `gelu`, `softmax`; spec/05 §3.3
  and `[05-OP-48]`) are pinned graphs that contain `exp`. A graph over a 1-ULP leaf
  cannot meet a zero-ULP bound unless the leaf is exact, so these rows are unachievable
  as written.
- `[05-OP-35]` (`contracts::normal_cdf`) resolves the same tension by requiring the
  embedded `exp` to be correctly rounded. No lane provides a correctly rounded `exp`.
  #1311 recorded a 2026-08-25 decision to replace that clause with a pinned `exp` graph.
- `spec/00-context.md` does not define "target". `spec/design/cross_lane_gate.md` counts
  libm identity, flags and library closure as part of the proof scope; the product
  `chelis build` profile (`-march=native`, `CHELIS_CC`) is not pinned.
- `spec/design/dtype_semantics.md` ("Not a numerics-accuracy project ... not new
  kernels") and `spec/design/faithful_observation.md` ("Not kernel accuracy") make owning
  kernels a non-goal, and the canonical reference plans more Sleef/vForce integration.
  `spec/04-type-system.md` states the opposite intent: the lanes "should agree exactly".
- `spec/08-backends.md` sets a blanket 1e-4 Metal tolerance and permits MSL fast math,
  against `spec/05`'s statement that blanket bounds are not a conforming oracle.

## 3. What the code does

About 25 production sites compute a transcendental, and eval and C are written
independently per op.

- Eval: Rust `f32::exp` and friends in `crates/chelis-types/src/dtype_semantics.rs`;
  f64-then-narrow in `crates/chelis-ir/src/optimize.rs` constant folding and in the host
  softmax (`crates/chelis-compiler-api/src/runtime/host_ops.rs`).
- C: libm names in `crates/chelis-backend-c/src/host_emit.rs` and `emit.rs`; vForce
  (`vforce_func`) and Sleef (`sleef_macro`, `simd_step_expr`) routes selected by
  `MathLib::detect()` in `crates/chelis-backend-c/src/lib.rs`.
- Prove: f64 libm in `crates/chelis-prove/src/concrete_eval.rs` and `contracts.rs`, each
  with its own copy of the Abramowitz-Stegun `erf`.
- Metal and HIP: device built-ins (MSL fast by default; ROCm ocml).

The RISC DAG expands compounds once (`crates/chelis-ir/src/tier2.rs`), so the C tensor
path, the IR evaluator and the GPU backends share one composition and differ only in the
leaves. The host-scalar path re-implements the activations by hand in
`dtype_semantics.rs` and `host_emit.rs`. The only definition consumed identically by both
lanes is the Surf stdlib route (`packages/chelis-std/src/contracts.ch`). There is no
range-reduction or minimax code in the repository, and no direct dependency on a Rust
math crate.

## 4. Prior art

Sources were fetched on 2026-10-02. "(not re-verified)" marks a claim resting on recall
or a secondary summary.

### 4.1 Languages and libraries

| system | guarantee | scope | how |
|---|---|---|---|
| Java `StrictMath` | bitwise: must equal fdlibm 5.3 results ([docs](https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/lang/StrictMath.html)) | everywhere | fdlibm ported to pure Java through JDK 21 ([JDK-8171407](https://bugs.openjdk.org/browse/JDK-8171407)); JEP 306 made all FP strict in Java 17 ([JEP 306](https://openjdk.org/jeps/306)) |
| Java `Math` | within 1 ulp, semi-monotonic; "not ... bit-for-bit" ([docs](https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/lang/Math.html)) | none | JIT intrinsics |
| Go `math` | no bitwise promise | none | pure Go plus assembly fast paths; `math.Exp(1)` differed between amd64 and arm64 ([golang/go#20319](https://github.com/golang/go/issues/20319)); the spec permits fusion across statements (not re-verified) |
| Julia | none; measured 0.49-2.4 ULP in 1.12.7 ([arXiv 2509.05666](https://arxiv.org/abs/2509.05666)) | none | pure-Julia kernels replaced openlibm ([JuliaLang/julia#26434](https://github.com/JuliaLang/julia/issues/26434)); `muladd` fuses only where hardware FMA exists (not re-verified) |
| Rust std | `f32::exp`: "Unspecified precision ... can even differ within the same execution"; `sqrt`, `mul_add` correctly rounded ([docs](https://doc.rust-lang.org/std/primitive.f32.html)) | none | platform libm; NaN bits non-deterministic by RFC 3514 ([RFC](https://rust-lang.github.io/rfcs/3514-float-semantics.html)) |
| WebAssembly | IEEE core ops; no transcendentals; deterministic profile canonicalizes NaNs ([profiles](https://webassembly.github.io/spec/core/appendix/profiles.html)) | everywhere for core ops | host or toolchain supplies transcendentals |
| C23 / IEEE 754-2019 | C23 reserves `cr_` names for correctly rounded functions ([N2715](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n2715.htm)); IEEE 754-2019 recommends correct rounding (clause not re-verified) | standard text | none mandated |
| glibc | CORE-MATH correctly rounded float functions from 2.41 (`tanhf`, `atanf`, `erff`, `cbrtf`, `expm1f`, `log1pf`, ...), first double functions in 2.43; `expf`, `logf`, `sinf`, `cosf`, `powf` are not on the list ([2.41](https://www.phoronix.com/news/GNU-C-Library-2.41-Features), [2.42](https://lists.gnu.org/archive/html/info-gnu/2025-07/msg00011.html)) | per version | varies by glibc version and CPU variant |
| LLVM libc | correctly rounded in all rounding modes by default ([docs](https://libc.llvm.org/math/index.html)); MSVC adopting it under `/Zc:cmath` for "identical results across platforms and versions" ([blog](https://devblogs.microsoft.com/cppblog/bringing-correctly-rounded-math-to-production-with-llvm-libc/)) | everywhere | fast path plus bounded wide-integer fallback |
| CORE-MATH | correctly rounded, all rounding modes, binary32 and binary64 ([project](https://core-math.gitlabpages.inria.fr/)) | everywhere | claims lower average cycles than glibc on its benchmarks |
| RLIBM | correctly rounded float, one polynomial for all rounding modes; 1.1x faster than glibc float libm ([arXiv 2104.04043](https://arxiv.org/abs/2104.04043)) | everywhere | polynomials fitted to rounding intervals |

### 4.2 Array and ML systems

- NumPy 1.22 added SVML AVX-512 paths with up to 4 ULP error on Linux only
  ([notes](https://numpy.org/doc/stable/release/1.22.0-notes.html)); results differ by
  CPU vendor ([numpy/numpy#23523](https://github.com/numpy/numpy/issues/23523)).
- JAX/XLA: `--xla_gpu_deterministic_ops` gives run-to-run determinism on one stack; CPU
  and GPU still differ ([jax-ml/jax#26795](https://github.com/jax-ml/jax/issues/26795)).
  XLA's own `tanh` approximation is backend-specific (not re-verified).
- PyTorch `use_deterministic_algorithms` and TensorFlow `enable_op_determinism` cover
  kernel choice and ordering on one hardware and software stack, not transcendentals
  (not re-verified).
- Halide ships its own `fast_exp`/`fast_log` polynomials; float64 calls the system libm
  ([docs](https://halide-lang.org/docs/namespace_halide.html)).

### 4.3 Deterministic simulation

- Box2D v3.1 is cross-platform deterministic across x64/ARM and MSVC/GCC/Clang, checked
  in CI: no fast math, `-ffp-contract=off`, its own `atan2` and `cos`/`sin`, `sqrtf`
  kept because it is correctly rounded ([post](https://box2d.org/posts/2024/08/determinism/)).
- Rapier's `enhanced-determinism` uses the Rust `libm` crate and disables SIMD and
  parallelism ([docs](https://rapier.rs/docs/user_guides/rust/determinism/)).

### 4.4 GPUs

- CUDA documents maximum errors (`expf` 2, `sinf` 2, `tanhf` 2 ULP); FMA contraction is
  on by default and results may change across versions and architectures
  ([guide](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)).
- Vulkan GLSL.std.450 `exp` is allowed (3 + 2|x|) ULP
  ([spec](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html)).
- Metal defaults to fast math; `precise::` `sin`/`cos` are within 4 ulp
  ([MSL spec](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)).
- No vendor GPU math library guarantees correct rounding. LLVM libc's GPU builds skip the
  accurate pass by default ([docs](https://libc.llvm.org/math/index.html)).

### 4.5 Patterns

1. Specified bitwise reproducibility always comes from shipping the algorithm (Java
   `StrictMath`, Box2D, Rapier, CORE-MATH or LLVM libc linked in). An N-ULP bound never
   yields identical bits across implementations, versions or CPUs.
2. Pinned source is not enough on its own: contraction and fast math must be forbidden in
   every lane, and FMA must be an explicit operation (Go's amd64/arm64 split).
3. Correct rounding gives one answer independent of the algorithm, and is in production
   (glibc, LLVM libc, MSVC). For binary32 it costs about nothing on a CPU; for binary64
   it costs about libm speed plus a rare, bounded slow path.
4. SIMD and GPU are where correctly rounded transcendentals are not yet routine.
5. NaN payloads are the residual non-determinism once arithmetic is pinned.

## 5. Assessment of the proposed fix

#2957 proposes that every zero-ULP or libm-dependent op be computed from pinned graphs
over IEEE `+ - * /`, `sqrt` and FMA, with the build profile and FP environment fixed by
the compiler.

The second half is needed under every option considered here. The first half is right
that the compiler must own the arithmetic, but it makes the implementation normative:

1. **The graph becomes the definition.** If `exp` means "what graph G returns", G's
   coefficients and operation order are language semantics, and any accuracy
   improvement after 0.19 is a breaking change (the anti-churn concern #1311 already
   records). This is Java's position with fdlibm.
2. **It complicates `chelis prove`.** A correctly rounded op is modelled as "the exact
   real function, rounded once", the same shape as IEEE `+ - * /` and `sqrt`. A pinned
   graph is either opaque to proof or needs a per-function error envelope.
3. **The named primitive set is likely too small (hypothesis).** Standard `exp` needs
   exact scaling by 2^k, `log` an exponent and mantissa split, and `sin` at large
   arguments a multi-word reduction by pi/2; implementations use exponent manipulation,
   bit casts and tables, not only `+ - * / sqrt fma`.
4. **It fixes an accuracy level by accident.** Whatever G achieves becomes the answer.
   "Correctly rounded" is the only definition of `exp` that admits no choice by a lane.

## 6. Options

| option | closes the class | cost | assessment |
|---|---|---|---|
| A. Keep host libm, widen the tolerance table, define "target" to include libm and compiler | no: compound error has no safe finite bound (32 ULP `tanh`), and cross-machine determinism is given up | low | abandons guarantee 2 |
| B. Fix instances (pin `normal_cdf`'s `exp`, drop vForce, clean flags) | no: Rust std `exp` stays unspecified and glibc varies by version | low | patches instances |
| C. Pinned graphs are the semantics (#2957 as written) | yes on CPU | medium, with churn later | workable, see section 5 |
| D. Correct rounding is the semantics; compiler-owned correctly rounded implementations in every lane | yes; algorithms can change without changing results | medium: binary32 cheap, binary64 needs a CORE-MATH-class implementation | recommended, with the approximate tier of section 7 |

## 7. Recommendation (option D with an explicit approximate tier)

**Rule, for the numbered spec.** Every float transcendental primitive returns the
correctly rounded (round-to-nearest-even) value of the exact function at `[04-NUM-8]`'s
arithmetic width, with the IEEE exceptional cases. f16 and bf16 compute at f32 and then
finalize, as `[04-NUM-8]` already requires. `[05-OBS-3]` then has no tolerance rows, and
compounds are exact because they are graphs of exact leaves and exact IEEE operations
with contraction forbidden.

**Implementation, for a design document.** One correctly rounded implementation per
function, consumed by every lane: for example vendored CORE-MATH C (MIT licence),
compiled into the `chelis` binary for eval and into the runtime or emitted header for
built programs, with emitted C calling Chelis-owned names rather than libm names. Because
a correctly rounded result does not depend on the algorithm, compiler constant folding and
the choice of C compiler stop mattering numerically. The vForce and Sleef routes and
`MathLib` detection are removed; the hand-written host-scalar activations are routed
through one definition; the eval f64 funnels compute at the declared width. A vectorized
binary32 kernel is admissible later if it passes the exhaustive oracle.

**GPU cost and the approximate tier.** Correct rounding does not exclude GPUs: they
execute IEEE `+ - * /`, `sqrt` and FMA exactly when fast math and contraction are off, so
a correctly rounded algorithm built from them gives CPU bits on a GPU. What any
CPU-equals-GPU bitwise promise excludes, under option C as much as option D, is the
hardware special-function units (MUFU-class `exp2`, `sin`, `rcp` approximations) and
vendor fast math. Correct rounding costs more than a pinned f32 graph on GPUs, because
the cheap CPU technique (evaluate in f64, round once) meets 1/32 to 1/64 f64 throughput
on consumer parts; a double-f32 formulation is the likely route, at an unmeasured
multiple of a faithful f32 polynomial. Element-wise transcendentals are usually
bandwidth-bound, and the known exception is attention, where `exp` throughput limits
kernels (FlashAttention-4 is reported to emulate `exp2` with a polynomial on the FMA
units; not re-verified).

So the language carries two tiers, chosen in the program text rather than by the lane,
following Java's `StrictMath`/`Math` split but made explicit at each call:

- `exp`, `log`, `sin`, ... are correctly rounded and identical on every target.
- A separately named approximate family (spelling to be decided in the spec, for example
  `exp_approx`) carries a per-op ULP bound stated in the spec. Its result is a function
  of program text, compiler build, declared target and inputs, so the determinism
  contract holds, but different targets may return different values within the bound.
  A CPU lane may implement it with the correctly rounded kernel, which satisfies any
  bound, so eval and `build --target c` still agree bitwise; a GPU lane may use the
  special-function units. `chelis prove` models it as the exact value rounded once,
  widened by the stated bound, and the cross-lane oracle applies the bound only to these
  ops.

The approximate tier is opt-in and visible at the call, so no reviewer has to infer
where reproducibility was traded for speed. Matmul and einsum are a separate GPU
question: `[05-OP-33]` already pins the contraction order, which vendor GEMM and
tensor-core accumulation do not follow, independent of this decision (#1290, #1315).

**Profile and environment, needed under every option.** The product build adopts the
strict profile (no fast math, no contraction, no `-march=native` unless the CPU is a
declared input); compiler subprocesses get a cleared environment with an allowlist rather
than a blocklist; library and binding entry points save, set and restore
round-to-nearest with flush-to-zero off; NaN results are canonical; and spec/00 defines
"target" as the declared target triple, so host libm and the host C compiler are not
inputs.

**Fit with the tenets.** Unambiguity: correct rounding is the one reading of `exp` with
no lane choice. Composition: compounds remain graphs and become exact. Small language:
transcendentals stay primitives with exact semantics, implemented in the runtime.
Future-proof: the rule is decided fully now even if binary64 lands after binary32.

**Open decisions.**

- `tanh`: the spec's `2*sigmoid(2x)-1` form loses 620 ULP at 1e-3 and 14,932 ULP at 1e-5
  in both lanes. Either make `tanh` a correctly rounded primitive, or add `expm1` and
  define `tanh(x) = expm1(2x) / (expm1(2x) + 2)`.
- The approximate tier's spelling, which ops it covers, and each op's bound.
- GPU lanes (experimental): the exact tier needs Chelis-owned device kernels; until they
  exist HIP and Metal are fenced on the known-issues page (#1170).
- Option D reverses the #1311 decision and the non-goals in `dtype_semantics.md` and
  `faithful_observation.md`. That decision rested on correct rounding being work no
  vendor library does; section 4 shows that premise no longer holds for CPUs.

## 8. Oracles

- Exhaustive binary32: every input of each function, eval and a built executable each
  compared bitwise against MPFR (manual or nightly gate), with the #2957 witnesses as a
  per-PR canary.
- The same input as a scalar literal, a run-time tensor (contiguous, permuted, rank-0)
  and a fused kernel gives identical bits.
- A build with `CCC_OVERRIDE_OPTIONS`, `NIX_CFLAGS_COMPILE` or a fast-math `CHELIS_CC`
  wrapper is rejected or bit-identical.
- An x86_64-linux gcc build and a macOS arm64 clang build give identical bits on one
  corpus.
- Eval and C are bit-equal on `sigmoid`, `silu`, `gelu`, `softmax` and `normal_cdf` over
  the #2952 and #2959 probe sets.
