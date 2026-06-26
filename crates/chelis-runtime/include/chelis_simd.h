#ifndef CHELIS_SIMD_H
#define CHELIS_SIMD_H

/* Self-contained SIMD reduction helpers for Chelis generated C code.
 * All functions are static inline — no external linkage.
 * Include order: this file is included by chelis_runtime.h unconditionally.
 *
 * `restrict` is a C99 keyword only; guard it so this header compiles in C++
 * contexts (e.g. HIP/CUDA host wrappers that include chelis_runtime.h).
 */
#ifdef __cplusplus
#define CHELIS_RESTRICT
#else
#define CHELIS_RESTRICT restrict
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* ------------------------------------------------------------------ */
/* chelis_sum_f32                                                       */
/*                                                                      */
/* Stride-4 ILP cascade matching torch's CPU `row_sum` (issue #163,     */
/* `num_levels=4 ilp_factor=4` in                                       */
/* pytorch/aten/src/ATen/native/cpu/SumKernel.cpp). Bit-exact with      */
/* torch's `.sum()` for n <= 16 on f32. The earlier AVX2 _hadd_ps       */
/* path produced a different roundoff than torch's cascade; we trade    */
/* a small constant factor of peak throughput for parity. Modern        */
/* compilers (-O3) auto-vectorize the four independent lanes well.      */
/* ------------------------------------------------------------------ */
static inline float chelis_sum_f32(const float * CHELIS_RESTRICT data, int n) {
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        acc0 += data[i];
        acc1 += data[i + 1];
        acc2 += data[i + 2];
        acc3 += data[i + 3];
    }
    for (; i < n; i++) {
        switch (i & 3) {
            case 0: acc0 += data[i]; break;
            case 1: acc1 += data[i]; break;
            case 2: acc2 += data[i]; break;
            default: acc3 += data[i]; break;
        }
    }
    return (acc0 + acc1) + (acc2 + acc3);
}

/* ------------------------------------------------------------------ */
/* chelis_fmax_propnan_f32 / chelis_fmin_propnan_f32                    */
/*                                                                      */
/* #172: scalar max/min that PROPAGATE NaN, unlike C99 `fmaxf`/`fminf`  */
/* (which return the non-NaN argument). Used by the strided / windowed  */
/* reduction loops so they match `chelis_max_f32` / `chelis_min_f32`    */
/* (and torch) on a NaN input — `max(a, NaN) == NaN` regardless of      */
/* operand order. The contiguous fast path uses the SIMD helpers above; */
/* these keep the slow path consistent.                                 */
/* ------------------------------------------------------------------ */
static inline float chelis_fmax_propnan_f32(float a, float b) {
    if (a != a || b != b) return __builtin_nanf("");
    return a > b ? a : b;
}
static inline float chelis_fmin_propnan_f32(float a, float b) {
    if (a != a || b != b) return __builtin_nanf("");
    return a < b ? a : b;
}

/* ------------------------------------------------------------------ */
/* chelis_max_f32                                                       */
/*                                                                      */
/* #172: PROPAGATES NaN, matching `torch.max` (a slice containing any   */
/* NaN returns NaN, at every position). The SIMD `_mm256_max_ps` /      */
/* `vmaxq_f32` drop NaN in a position-dependent way (the hardware max    */
/* returns the non-NaN operand), so we detect NaN separately with an    */
/* unordered-compare accumulator and force the result to NaN if any     */
/* lane was unordered. This is deterministic across positions and       */
/* across the AVX2 / NEON / scalar paths, and matches the position-     */
/* independent NaN propagation of `torch.max`. The scalar `>` reduction */
/* never sets `result` to NaN (NaN compares false), so the explicit     */
/* NaN scan is also what makes the scalar path propagate.               */
/* ------------------------------------------------------------------ */
static inline float chelis_max_f32(const float * CHELIS_RESTRICT data, int n) {
    if (n <= 0) return -__builtin_inff();
#ifdef __AVX2__
    __m256 vacc = _mm256_set1_ps(-__builtin_inff());
    __m256 vnan = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        vacc = _mm256_max_ps(vacc, v);
        /* unordered compare: lane is all-ones iff v != v (NaN) */
        vnan = _mm256_or_ps(vnan, _mm256_cmp_ps(v, v, _CMP_UNORD_Q));
    }
    /* extract 8 floats and take scalar max */
    float buf[8];
    _mm256_storeu_ps(buf, vacc);
    float result = buf[0];
    for (int k = 1; k < 8; k++) {
        if (buf[k] > result) result = buf[k];
    }
    int saw_nan = _mm256_movemask_ps(vnan) != 0;
    /* scalar tail */
    for (; i < n; i++) {
        if (data[i] != data[i]) saw_nan = 1;
        if (data[i] > result) result = data[i];
    }
    return saw_nan ? __builtin_nanf("") : result;
#elif defined(__ARM_NEON)
    float32x4_t vacc = vdupq_n_f32(-__builtin_inff());
    int saw_nan = 0;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        vacc = vmaxq_f32(vacc, v);
        /* vceqq_f32(v, v) is 0 in any NaN lane; reduce-min of the mask is 0 */
        if (vminvq_u32(vceqq_f32(v, v)) == 0) saw_nan = 1;
    }
    float result = vmaxvq_f32(vacc);
    for (; i < n; i++) {
        if (data[i] != data[i]) saw_nan = 1;
        if (data[i] > result) result = data[i];
    }
    return saw_nan ? __builtin_nanf("") : result;
#else
    float result = -__builtin_inff();
    int saw_nan = 0;
    for (int i = 0; i < n; i++) {
        if (data[i] != data[i]) saw_nan = 1;
        if (data[i] > result) result = data[i];
    }
    return saw_nan ? __builtin_nanf("") : result;
#endif
}

/* ------------------------------------------------------------------ */
/* chelis_min_f32                                                       */
/*                                                                      */
/* #172: PROPAGATES NaN, matching `torch.min` (same rationale as        */
/* chelis_max_f32 above). The SIMD min drops NaN position-dependently;  */
/* an unordered-compare accumulator detects NaN and forces the result   */
/* to NaN.                                                              */
/* ------------------------------------------------------------------ */
static inline float chelis_min_f32(const float * CHELIS_RESTRICT data, int n) {
    if (n <= 0) return __builtin_inff();
#ifdef __AVX2__
    __m256 vacc = _mm256_set1_ps(__builtin_inff());
    __m256 vnan = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        vacc = _mm256_min_ps(vacc, v);
        vnan = _mm256_or_ps(vnan, _mm256_cmp_ps(v, v, _CMP_UNORD_Q));
    }
    float buf[8];
    _mm256_storeu_ps(buf, vacc);
    float result = buf[0];
    for (int k = 1; k < 8; k++) {
        if (buf[k] < result) result = buf[k];
    }
    int saw_nan = _mm256_movemask_ps(vnan) != 0;
    for (; i < n; i++) {
        if (data[i] != data[i]) saw_nan = 1;
        if (data[i] < result) result = data[i];
    }
    return saw_nan ? __builtin_nanf("") : result;
#elif defined(__ARM_NEON)
    float32x4_t vacc = vdupq_n_f32(__builtin_inff());
    int saw_nan = 0;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        vacc = vminq_f32(vacc, v);
        if (vminvq_u32(vceqq_f32(v, v)) == 0) saw_nan = 1;
    }
    float result = vminvq_f32(vacc);
    for (; i < n; i++) {
        if (data[i] != data[i]) saw_nan = 1;
        if (data[i] < result) result = data[i];
    }
    return saw_nan ? __builtin_nanf("") : result;
#else
    float result = __builtin_inff();
    int saw_nan = 0;
    for (int i = 0; i < n; i++) {
        if (data[i] != data[i]) saw_nan = 1;
        if (data[i] < result) result = data[i];
    }
    return saw_nan ? __builtin_nanf("") : result;
#endif
}

/* ------------------------------------------------------------------ */
/* chelis_argmax_f32                                                    */
/* AVX2 index tracking is complex — scalar scan for correctness.       */
/* ------------------------------------------------------------------ */
static inline int chelis_argmax_f32(const float * CHELIS_RESTRICT data, int n) {
    if (n <= 0) return -1;
    int best = 0;
    float best_val = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] > best_val) {
            best_val = data[i];
            best = i;
        }
    }
    return best;
}

/* ------------------------------------------------------------------ */
/* chelis_argmin_f32                                                    */
/* ------------------------------------------------------------------ */
static inline int chelis_argmin_f32(const float * CHELIS_RESTRICT data, int n) {
    if (n <= 0) return -1;
    int best = 0;
    float best_val = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] < best_val) {
            best_val = data[i];
            best = i;
        }
    }
    return best;
}

#endif /* CHELIS_SIMD_H */
