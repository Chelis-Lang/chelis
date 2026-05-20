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
/* chelis_max_f32                                                       */
/* NaN in input produces implementation-defined results: AVX2           */
/* _mm256_max_ps returns the second operand when the first is NaN,      */
/* which makes NaN propagation position-dependent.                      */
/* ------------------------------------------------------------------ */
static inline float chelis_max_f32(const float * CHELIS_RESTRICT data, int n) {
    if (n <= 0) return -__builtin_inff();
#ifdef __AVX2__
    __m256 vacc = _mm256_set1_ps(-__builtin_inff());
    int i = 0;
    for (; i + 7 < n; i += 8) {
        vacc = _mm256_max_ps(vacc, _mm256_loadu_ps(data + i));
    }
    /* extract 8 floats and take scalar max */
    float buf[8];
    _mm256_storeu_ps(buf, vacc);
    float result = buf[0];
    for (int k = 1; k < 8; k++) {
        if (buf[k] > result) result = buf[k];
    }
    /* scalar tail */
    for (; i < n; i++) {
        if (data[i] > result) result = data[i];
    }
    return result;
#elif defined(__ARM_NEON)
    float32x4_t vacc = vdupq_n_f32(-__builtin_inff());
    int i = 0;
    for (; i + 3 < n; i += 4) {
        vacc = vmaxq_f32(vacc, vld1q_f32(data + i));
    }
    float result = vmaxvq_f32(vacc);
    for (; i < n; i++) {
        if (data[i] > result) result = data[i];
    }
    return result;
#else
    float result = -__builtin_inff();
    for (int i = 0; i < n; i++) {
        if (data[i] > result) result = data[i];
    }
    return result;
#endif
}

/* ------------------------------------------------------------------ */
/* chelis_min_f32                                                       */
/* NaN behavior is implementation-defined (same as chelis_max_f32).    */
/* ------------------------------------------------------------------ */
static inline float chelis_min_f32(const float * CHELIS_RESTRICT data, int n) {
    if (n <= 0) return __builtin_inff();
#ifdef __AVX2__
    __m256 vacc = _mm256_set1_ps(__builtin_inff());
    int i = 0;
    for (; i + 7 < n; i += 8) {
        vacc = _mm256_min_ps(vacc, _mm256_loadu_ps(data + i));
    }
    float buf[8];
    _mm256_storeu_ps(buf, vacc);
    float result = buf[0];
    for (int k = 1; k < 8; k++) {
        if (buf[k] < result) result = buf[k];
    }
    for (; i < n; i++) {
        if (data[i] < result) result = data[i];
    }
    return result;
#elif defined(__ARM_NEON)
    float32x4_t vacc = vdupq_n_f32(__builtin_inff());
    int i = 0;
    for (; i + 3 < n; i += 4) {
        vacc = vminq_f32(vacc, vld1q_f32(data + i));
    }
    float result = vminvq_f32(vacc);
    for (; i < n; i++) {
        if (data[i] < result) result = data[i];
    }
    return result;
#else
    float result = __builtin_inff();
    for (int i = 0; i < n; i++) {
        if (data[i] < result) result = data[i];
    }
    return result;
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
