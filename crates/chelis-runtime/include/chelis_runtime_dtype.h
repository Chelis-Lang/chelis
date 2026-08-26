#ifndef CHELIS_RUNTIME_DTYPE_H
#define CHELIS_RUNTIME_DTYPE_H

#include <stdint.h>

typedef uint8_t chelis_dtype;
enum {
    CHELIS_DTYPE_F32 = 0,
    CHELIS_DTYPE_F64 = 1,
    CHELIS_DTYPE_I32 = 2,
    CHELIS_DTYPE_BOOL = 3,
    CHELIS_DTYPE_I64 = 4,
    CHELIS_DTYPE_BF16 = 5,
    CHELIS_DTYPE_F16 = 6,
    CHELIS_DTYPE_I8 = 7,
    CHELIS_DTYPE_I16 = 8
};

#endif
