/* Bare-target support for the unchanged CORE-MATH kernels. */
#ifndef CHELIS_CRMATH_FREESTANDING_PROFILE_H
#define CHELIS_CRMATH_FREESTANDING_PROFILE_H

#if !defined(__SIZEOF_INT128__)
#if !defined(__clang__) || !defined(__BITINT_MAXWIDTH__) || __BITINT_MAXWIDTH__ < 128
#error "chelis-crmath needs Clang with 128-bit _BitInt support on this target"
#endif
#define __int128 _BitInt(128)
#endif

/* Each helper implements the corresponding IEEE operation without a C library. */
double chelis_crmath_soft_fma(double, double, double);
float chelis_crmath_soft_fmaf(float, float, float);
double chelis_crmath_soft_floor(double);
double chelis_crmath_soft_fmax(double, double);
double chelis_crmath_soft_rint(double);
double chelis_crmath_soft_round(double);
float chelis_crmath_soft_roundf(float);
double chelis_crmath_soft_ldexp(double, int);
float chelis_crmath_soft_nanf(void);
double chelis_crmath_soft_sqrt(double);
float chelis_crmath_soft_sqrtf(float);

#define __builtin_fma chelis_crmath_soft_fma
#define __builtin_fmaf chelis_crmath_soft_fmaf
#define __builtin_floor chelis_crmath_soft_floor
#define __builtin_fmax chelis_crmath_soft_fmax
#define __builtin_rint chelis_crmath_soft_rint
#define __builtin_round chelis_crmath_soft_round
#define __builtin_roundf chelis_crmath_soft_roundf
#define __builtin_ldexp chelis_crmath_soft_ldexp
#define __builtin_nanf(tag) chelis_crmath_soft_nanf()
#define __builtin_sqrt chelis_crmath_soft_sqrt
#define __builtin_sqrtf chelis_crmath_soft_sqrtf

#endif
