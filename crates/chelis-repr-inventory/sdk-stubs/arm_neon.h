#ifndef CHELIS_STUB_ARM_NEON_H
#define CHELIS_STUB_ARM_NEON_H
/* The NEON surface chelis_simd.h consumes, declared as plain functions so
 * the NEON arm parses under the inventory's `neon` configuration without the
 * compiler's intrinsics header. A new intrinsic fails the scan until it is
 * declared here. */
#include <stdint.h>
typedef float float32x4_t __attribute__((__vector_size__(16)));
typedef uint32_t uint32x4_t __attribute__((__vector_size__(16)));
float32x4_t vld1q_f32(const float *pointer);
float32x4_t vdupq_n_f32(float value);
float32x4_t vmaxq_f32(float32x4_t a, float32x4_t b);
float32x4_t vminq_f32(float32x4_t a, float32x4_t b);
float vmaxvq_f32(float32x4_t value);
float vminvq_f32(float32x4_t value);
uint32x4_t vceqq_f32(float32x4_t a, float32x4_t b);
uint32_t vminvq_u32(uint32x4_t value);
#endif
