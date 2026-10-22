/* Clang lowers these fixed-size copies to target memory operations. */
#ifndef CHELIS_CRMATH_FREESTANDING_STRING_H
#define CHELIS_CRMATH_FREESTANDING_STRING_H
#include <stddef.h>
void *memcpy(void *restrict, const void *restrict, size_t);
#endif
