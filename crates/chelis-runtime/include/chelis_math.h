#pragma once

#if defined(__AVX2__) && defined(CHELIS_HAS_SLEEF)
#include <sleef.h>
/* 8-wide float functions */
#define CHELIS_EXPF8(x)   Sleef_expf8_u10(x)
#define CHELIS_LOGF8(x)   Sleef_logf8_u10(x)
#define CHELIS_SINF8(x)   Sleef_sinf8_u10(x)
#define CHELIS_COSF8(x)   Sleef_cosf8_u10(x)
#define CHELIS_SQRTF8(x)  Sleef_sqrtf8_u05(x)
#define CHELIS_ERFF8(x)   Sleef_erff8_u10(x)
#define CHELIS_MATH_WIDTH 8
#define CHELIS_MATH_VEC   __m256
#elif defined(__APPLE__)
#include <Accelerate/Accelerate.h>
/* vForce batch functions -- whole-array API, handled differently in L3b */
#define CHELIS_MATH_USE_VFORCE 1
#define CHELIS_MATH_WIDTH 1
#else
/* Scalar fallback -- use Level 1 fast path */
#define CHELIS_MATH_SCALAR_ONLY 1
#define CHELIS_MATH_WIDTH 1
#endif
