#ifndef CHELIS_STUB_IMMINTRIN_H
#define CHELIS_STUB_IMMINTRIN_H
/* The x86 AVX2 surface chelis_simd.h consumes, declared as plain functions
 * so the AVX2 arm parses under the inventory's `avx2` configuration without
 * the compiler's intrinsics headers. A new intrinsic fails the scan until it
 * is declared here. */
typedef float __m256 __attribute__((__vector_size__(32)));
#define _CMP_EQ_OQ 0
#define _CMP_LT_OS 1
#define _CMP_LE_OS 2
#define _CMP_UNORD_Q 3
#define _CMP_NEQ_UQ 4
#define _CMP_NLT_US 5
#define _CMP_NLE_US 6
#define _CMP_ORD_Q 7
#define _CMP_GT_OS 14
#define _CMP_GE_OS 13
__m256 _mm256_loadu_ps(const float *pointer);
void _mm256_storeu_ps(float *pointer, __m256 value);
__m256 _mm256_set1_ps(float value);
__m256 _mm256_setzero_ps(void);
__m256 _mm256_max_ps(__m256 a, __m256 b);
__m256 _mm256_min_ps(__m256 a, __m256 b);
__m256 _mm256_or_ps(__m256 a, __m256 b);
__m256 _mm256_cmp_ps(__m256 a, __m256 b, int predicate);
int _mm256_movemask_ps(__m256 value);
#endif
