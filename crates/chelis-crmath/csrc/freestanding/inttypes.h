/* Format macros for unused CORE-MATH debug printers. */
#ifndef CHELIS_CRMATH_FREESTANDING_INTTYPES_H
#define CHELIS_CRMATH_FREESTANDING_INTTYPES_H
#include <stdint.h>
#if __SIZEOF_LONG__ == 8
#define PRIx64 "lx"
#define PRId64 "ld"
#else
#define PRIx64 "llx"
#define PRId64 "lld"
#endif
#endif
