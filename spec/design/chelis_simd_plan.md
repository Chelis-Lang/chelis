# SIMD Support Plan for Chelis

**Owner:** Core compiler (`chelis-lang/chelis`), primarily `crates/chelis-backend-c/` and `crates/chelis-runtime/`
**Status:** Level 0 (zero explicit SIMD today). Auto-vectorization only via C compiler.
**Priority:** Medium. Not blocking any shell. Becomes high priority before OOPSLA benchmarks.

---

## Current State

The C backend emits scalar loops. A fused `exp(add(mul(a, b), c))` on `tensor[1000, f32]` compiles to:

```c
for (int i = 0; i < 1000; i++) {
    out[i] = a[i] * b[i] + c[i];
}
```

The C compiler (gcc or clang) may auto-vectorize this at `-O2`. Whether it does depends on loop body complexity, pointer aliasing analysis, alignment knowledge, and compiler heuristics. Simple loops (add, mul, fma) auto-vectorize reliably. Complex fused bodies (Horner polynomials inside `erf`, convergence-checked continued fractions in `gamma_inc`) typically don't because control flow defeats the vectorizer.

Measured performance from Nautilus benchmarks (all CPU, no explicit SIMD):
- Compound expression fusion: 3.0-3.6x over numpy at n=100k
- Special functions: erfinv 6.5x, normal_inv_cdf 6.8x
- Distribution CDFs: 2.3x after convergence optimization
- These wins are from fusion (one loop instead of multiple), not from SIMD

Performance left on the table: the auto-vectorizer is conservative. It won't vectorize anything it can't prove is safe. The generated C doesn't provide the hints (alignment, restrict, pragmas) that would help.

---

## Level 1: Help the Auto-Vectorizer

**Effort:** Small (codegen changes only, no new dependencies)
**Impact:** Moderate (unlocks auto-vectorization on loops currently skipped due to aliasing)
**When:** v0.1.9 ecosystem release or next codegen improvement pass

### Changes

**`restrict` on every tensor pointer parameter.** The C backend currently emits bare `float*` parameters. The auto-vectorizer can't prove `out` and `a` don't overlap, so it either emits a runtime alias check or skips vectorization. Adding `restrict` tells the compiler these don't alias.

This is correct because Chelis's linearity system guarantees no aliasing — a tensor is consumed at most once, so two live tensor pointers never point to the same memory. The type system has already proved the precondition that `restrict` requires.

```c
// Before (current)
void fused_0(float* out, float* a, float* b, int n) { ... }

// After
void fused_0(float* restrict out, const float* restrict a, 
             const float* restrict b, int n) { ... }
```

Also add `const` on input pointers — another hint the vectorizer uses.

**Alignment attributes on runtime allocations.** Ensure `chelis_alloc` returns 32-byte-aligned memory (for AVX2) or 16-byte-aligned (for NEON). Add `__attribute__((aligned(32)))` hints in the generated C where appropriate.

```c
// In chelis_runtime:
void* chelis_alloc(size_t bytes) {
    void* ptr;
    posix_memalign(&ptr, 32, bytes);  // 32-byte aligned for AVX2
    return ptr;
}
```

**Vectorization pragmas on elementwise loops.**

```c
#pragma omp simd              // OpenMP SIMD directive (works on gcc and clang)
for (int i = 0; i < n; i++) {
    out[i] = a[i] * b[i] + c[i];
}
```

`#pragma omp simd` is more portable than gcc-specific `#pragma GCC ivdep`. It works with both gcc and clang (when OpenMP is available). On Apple clang without OpenMP, fall back to `#pragma clang loop vectorize(enable) interleave(enable)`.

The toolchain resolver (from the macOS CPU support work) already knows which compiler is active. Emit the right pragma for the compiler.

### Where in the codebase

- `crates/chelis-backend-c/src/emit.rs` — add `restrict` and `const` to emitted function signatures
- `crates/chelis-backend-c/src/host_emit.rs` — same for host-lane function signatures
- `crates/chelis-runtime/` — ensure allocation alignment
- Toolchain resolver — emit compiler-appropriate SIMD pragmas

### Testing

- Inspect generated C for `restrict` and `const` on all tensor pointer parameters
- Compile a fused elementwise kernel with `gcc -O2 -ftree-vectorize -fopt-info-vec` and verify the vectorizer reports success on loops that previously failed
- Numerical results unchanged (restrict is a correctness-neutral annotation)

---

## Level 2: Hand-Written SIMD Reductions in the Runtime

**Effort:** Medium (new C functions in the runtime, platform-specific)
**Impact:** Targeted (reductions are where auto-vectorization is weakest)
**When:** When profiling shows reductions are a bottleneck (likely during OOPSLA benchmark preparation)

### What

Auto-vectorization of reductions is notoriously poor. A `sum` over a million-element tensor uses a sequential accumulator unless the compiler is very clever about splitting it into partial sums. Hand-written SIMD reductions are well-understood and give reliable 4-8x speedups.

### New runtime functions

```c
// x86 AVX2
float chelis_sum_f32_avx2(const float* restrict data, int n);
float chelis_max_f32_avx2(const float* restrict data, int n);
float chelis_min_f32_avx2(const float* restrict data, int n);
int   chelis_argmax_f32_avx2(const float* restrict data, int n);
int   chelis_argmin_f32_avx2(const float* restrict data, int n);

// ARM NEON
float chelis_sum_f32_neon(const float* restrict data, int n);
float chelis_max_f32_neon(const float* restrict data, int n);
float chelis_min_f32_neon(const float* restrict data, int n);
int   chelis_argmax_f32_neon(const float* restrict data, int n);
int   chelis_argmin_f32_neon(const float* restrict data, int n);
```

Each function: SIMD accumulation loop (8 floats per iteration on AVX2, 4 on NEON), horizontal reduction at the end, scalar tail for remaining elements.

Dispatch: the runtime selects the right implementation based on `#ifdef __AVX2__` / `#ifdef __ARM_NEON`. The generated C calls `chelis_sum_f32(data, n)` — a single entry point that routes internally.

### Implementation sketch (sum, AVX2)

```c
float chelis_sum_f32_avx2(const float* restrict data, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(data + i));
    }
    // Horizontal sum: 8 -> 4 -> 2 -> 1
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 sum4 = _mm_add_ps(lo, hi);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    float result = _mm_cvtss_f32(sum4);
    for (; i < n; i++) result += data[i];
    return result;
}
```

### Implementation sketch (sum, NEON)

```c
float chelis_sum_f32_neon(const float* restrict data, int n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        acc = vaddq_f32(acc, vld1q_f32(data + i));
    }
    float result = vaddvq_f32(acc);
    for (; i < n; i++) result += data[i];
    return result;
}
```

### Where in the codebase

- `crates/chelis-runtime/src/simd_reductions.c` (or `.rs` with inline asm / intrinsics via `std::arch`)
- Platform-gated: AVX2 functions compile only on x86, NEON functions compile only on ARM
- Fallback: scalar loop (the current behavior) if neither is available

### Testing

- Each SIMD reduction matches the scalar loop result within f32 tolerance
- Test at various sizes: n=1 (scalar tail only), n=7 (partial SIMD + tail), n=1024 (full SIMD), n=100003 (large + odd tail)

---

## Level 3: Vectorized Math Library Integration

**Effort:** Medium (build dependency + emitter mapping)
**Impact:** Large (10-20x on math-heavy fused kernels like erf, normal_cdf)
**When:** Before OOPSLA benchmarks. This is the highest-impact SIMD investment.

### What

Scalar math functions (`expf`, `logf`, `sinf`, `sqrtf`) don't have hardware SIMD equivalents on most architectures. The auto-vectorizer can't vectorize a loop containing `expf()` because there's no `expf_x8()` instruction. Vectorized math libraries provide SIMD-width implementations of these functions using polynomial approximations.

### Library options

| Library | License | Platforms | Functions | Integration |
|---|---|---|---|---|
| **Sleef** | Boost | x86 (SSE/AVX/AVX512), ARM (NEON/SVE), RISC-V | exp, log, sin, cos, tan, atan, erf, erfc, pow, cbrt, etc. | C library, link against libsleef |
| **Accelerate vForce** | Apple | macOS/iOS (ARM only) | vexpf, vlogf, vsinf, vcosf, vtanf, etc. | Already linked via `-framework Accelerate` |
| **SVML** | Intel proprietary | x86 only (with ICC/ICX) | Full math suite | Not portable, not open source |

**Recommendation:** Use Sleef on Linux (both x86 and ARM). Use Accelerate vForce on macOS (already linked). This covers both platforms without a proprietary dependency.

### Codegen change

When the C emitter encounters a fused elementwise kernel containing a math function (exp, log, sin, sqrt, erf, etc.), and the target platform has a vectorized math library available, emit calls to the vectorized version instead of the scalar version.

```c
// Before (scalar, current)
#pragma omp simd
for (int i = 0; i < n; i++) {
    out[i] = expf(a[i] * b[i] + c[i]);
}

// After (Sleef, x86 AVX2)
int i = 0;
for (; i + 8 <= n; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 vc = _mm256_loadu_ps(c + i);
    __m256 vfma = _mm256_add_ps(_mm256_mul_ps(va, vb), vc);
    __m256 vexp = Sleef_expf8_u10(vfma);  // 8-wide exp, 1 ULP accuracy
    _mm256_storeu_ps(out + i, vexp);
}
for (; i < n; i++) {
    out[i] = expf(a[i] * b[i] + c[i]);  // scalar tail
}

// After (vForce, macOS)
int n_int = (int)n;
float* temp = alloca(n * sizeof(float));
// Compute a*b+c into temp
for (int i = 0; i < n; i++) temp[i] = a[i] * b[i] + c[i];
vvexpf(out, temp, &n_int);  // vectorized exp over entire array
```

The Sleef path emits explicit SIMD loops (the emitter controls the loop structure). The vForce path emits array-level calls (vForce operates on whole arrays, not SIMD registers). Both are valid and both eliminate the scalar `expf` bottleneck.

### Mapping table (Sleef function names)

| RISC op | Scalar C | Sleef AVX2 (f32) | Sleef NEON (f32) | Accelerate vForce |
|---|---|---|---|---|
| exp | expf | Sleef_expf8_u10 | Sleef_expf4_u10 | vvexpf |
| log | logf | Sleef_logf8_u10 | Sleef_logf4_u10 | vvlogf |
| sin | sinf | Sleef_sinf8_u10 | Sleef_sinf4_u10 | vvsinf |
| cos | cosf | Sleef_cosf8_u10 | Sleef_cosf4_u10 | vvcosf |
| sqrt | sqrtf | _mm256_sqrt_ps (hw) | vsqrtq_f32 (hw) | vvsqrtf |
| pow | powf | Sleef_powf8_u10 | Sleef_powf4_u10 | vvpowf |
| erf | — (Horner) | Sleef_erff8_u10 | Sleef_erff4_u10 | — |

Note: sqrt has hardware SIMD instructions on both x86 and ARM. The auto-vectorizer usually handles this correctly. Sleef is needed for the transcendental functions (exp, log, sin, etc.) that don't have hardware equivalents.

### Build system changes

```toml
# Linux: add sleef as a system dependency
# In crates/chelis-runtime/build.rs or the release build script:
# pkg-config --libs sleef → -lsleef
# Or vendor sleef source and build it with cmake

# macOS: Accelerate is already linked (-framework Accelerate)
# vForce functions are in Accelerate.h, already included via chelis_blas.h
```

The toolchain resolver decides: if building on Linux and libsleef is available, emit Sleef calls. If building on macOS, emit vForce calls. If neither is available, fall back to scalar (current behavior). This is a runtime detection in the toolchain resolver, not a compile-time #ifdef in the generated C — the emitter already knows the target platform.

### Where in the codebase

- `crates/chelis-backend-c/src/emit.rs` — when emitting a fused kernel body, check if the platform has vectorized math and emit the SIMD loop variant
- `crates/chelis-runtime/build.rs` — detect Sleef availability, link against it
- Toolchain resolver — propagate vectorized math library availability to the emitter
- `include/chelis_math.h` — optional shim header (like `chelis_blas.h`) that includes `<sleef.h>` on Linux or `<Accelerate/Accelerate.h>` on macOS

### Testing

- Fused kernel output with Sleef matches scalar output within 1 ULP (Sleef's `_u10` variants guarantee this)
- Benchmark: `erf` at n=100k with Sleef vs without. Expected: 3-5x additional speedup on top of existing fusion wins.
- Benchmark: `normal_cdf` (compound: erf + exp + mul + add) with Sleef. Expected: 5-10x total over numpy.
- Fallback: if Sleef is not installed, the generated C compiles and runs correctly (scalar path)

---

## Level 4: Full SIMD-Width-Aware Codegen

**Effort:** Large (parallel emit path in the C backend)
**Impact:** Diminishing if Level 3 is done
**When:** Only if benchmarks after Level 3 show significant remaining performance gaps

### What

Instead of emitting scalar loops and relying on auto-vectorization or library calls, emit explicitly vectorized C code for the entire fused kernel body. The emitter knows the SIMD width (8 for AVX2, 4 for NEON) and emits intrinsics for every operation.

This is the most complex level and gives diminishing returns if Level 3 (library integration) handles the math functions. The remaining benefit is for fused kernels that mix math and arithmetic in ways the auto-vectorizer still can't handle after Level 1's hints.

### When this matters

The auto-vectorizer + Sleef combination handles most cases. Level 4 matters for:
- Complex fused bodies where the auto-vectorizer gives up despite `restrict` and pragmas
- Fused bodies containing `where` (masked select) — `_mm256_blendv_ps` on AVX2 is faster than a scalar branch
- Gather operations — AVX2 has `_mm256_i32gather_ps` which is faster than scalar index loops for some access patterns

### Recommendation

Don't build Level 4 unless benchmarking after Levels 1-3 shows a clear, measurable gap. The engineering cost is high (a parallel SIMD emit path for every RISC op) and the benefit is incremental. Record it as a future optimization option.

---

## Execution Summary

```
Level 1 (auto-vectorizer hints):
├── restrict on all tensor pointer params
├── const on input pointers
├── Aligned allocation in runtime
├── Compiler-appropriate SIMD pragmas
└── Ship with: v0.1.9 ecosystem release or next codegen pass

Level 2 (hand-written SIMD reductions):
├── chelis_sum_f32, chelis_max_f32, chelis_min_f32
├── chelis_argmax_f32, chelis_argmin_f32
├── AVX2 + NEON implementations + scalar fallback
└── Ship with: when profiling shows reduction bottleneck

Level 3 (vectorized math library):
├── Sleef on Linux (both x86 + ARM)
├── Accelerate vForce on macOS
├── Emitter maps math ops to vectorized equivalents in fused kernels
├── chelis_math.h shim header
└── Ship with: before OOPSLA benchmarks

Level 4 (full SIMD codegen):
├── Only if Levels 1-3 leave measurable gaps
└── Record as future option, don't build speculatively
```

---

## Relationship to Other Backends

SIMD improvements apply only to the C backend's CPU path. The HIP backend uses GPU parallelism (thousands of threads, not SIMD lanes). The Metal backend (planned) uses Apple GPU cores. These are parallel approaches to performance, not alternatives:

- Small tensors (n < 10k): CPU with SIMD wins (no GPU launch overhead)
- Large tensors (n > 100k): GPU wins (massive parallelism)
- Medium tensors: depends on the operation and the hardware

The C backend with good SIMD support is the right execution path for the finance use case (Shoals/CProof) where tensor sizes are moderate (portfolios of hundreds to thousands of instruments, not ImageNet-scale batches) and latency matters more than throughput.
