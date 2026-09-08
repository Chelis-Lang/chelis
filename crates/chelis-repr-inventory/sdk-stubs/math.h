#ifndef STUB_MATH_H
#define STUB_MATH_H
double fabs(double); float fabsf(float); double sqrt(double); float sqrtf(float);
double exp(double); float expf(float); double log(double); float logf(float);
double sin(double); float sinf(float); double cos(double); float cosf(float);
double tanh(double); float tanhf(float); double pow(double, double); float powf(float, float);
double floor(double); float floorf(float); double ceil(double); float ceilf(float);
double round(double); float roundf(float); double trunc(double); float truncf(float);
double fmod(double, double); float fmodf(float, float); double erf(double); float erff(float);
double fmax(double, double); float fmaxf(float, float); double fmin(double, double); float fminf(float, float);
double rint(double); float rintf(float); double nearbyint(double); float nearbyintf(float);
double ldexp(double, int); double frexp(double, int *); int isnan(double); int isinf(double);
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
#define HUGE_VAL (__builtin_huge_val())
#define isnan(x) __builtin_isnan(x)
#define isinf(x) __builtin_isinf(x)
#define isfinite(x) __builtin_isfinite(x)
#define signbit(x) __builtin_signbit(x)
#endif
