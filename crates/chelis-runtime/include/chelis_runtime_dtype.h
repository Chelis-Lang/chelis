#ifndef CHELIS_RUNTIME_DTYPE_H
#define CHELIS_RUNTIME_DTYPE_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define CHELIS_F32 0
#define CHELIS_F64 1
#define CHELIS_I32 2
#define CHELIS_BOOL 3
#define CHELIS_I64 4
#define CHELIS_BF16 5
#define CHELIS_F16 6
#define CHELIS_I8 7
#define CHELIS_I16 8

static inline size_t chelis_runtime_dtype_size_checked(int dtype) {
    switch (dtype) {
        case CHELIS_F32: return 4;
        case CHELIS_F64: return 8;
        case CHELIS_I32: return 4;
        case CHELIS_BOOL: return 1;
        case CHELIS_I64: return 8;
        case CHELIS_BF16: return 2;
        case CHELIS_F16: return 2;
        case CHELIS_I8: return 1;
        case CHELIS_I16: return 2;
        default:
            fprintf(stderr, "invalid Chelis runtime dtype id: %d\n", dtype);
            abort();
    }
}

#endif
