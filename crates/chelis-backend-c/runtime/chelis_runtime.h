#ifndef CHELIS_RUNTIME_H
#define CHELIS_RUNTIME_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define CHELIS_F32 0
#define CHELIS_F64 1
#define CHELIS_I32 2
#define CHELIS_BOOL 3
#define CHELIS_MAX_DIM 8

typedef struct {
    float *data;
    int shape[CHELIS_MAX_DIM];
    int strides[CHELIS_MAX_DIM];
    int ndim;
    int dtype;
    int size;
} chelis_tensor;

chelis_tensor* chelis_alloc(int ndim, const int *shape, int dtype);
void chelis_free(chelis_tensor *t);
void chelis_fill_f32(chelis_tensor *t, float val);

/* Stride-aware indexing helpers */
static inline void chelis_flat_to_indices(int flat, const int *shape, int ndim, int *out) {
    for (int d = ndim - 1; d >= 0; d--) {
        out[d] = flat % shape[d];
        flat /= shape[d];
    }
}

static inline int chelis_indices_to_flat(const int *indices, const int *strides, int ndim) {
    int flat = 0;
    for (int d = 0; d < ndim; d++) {
        flat += indices[d] * strides[d];
    }
    return flat;
}

static inline int chelis_is_contiguous(const chelis_tensor *t) {
    int expected = 1;
    for (int d = t->ndim - 1; d >= 0; d--) {
        if (t->strides[d] != expected) return 0;
        expected *= t->shape[d];
    }
    return 1;
}

chelis_tensor* chelis_contiguous(const chelis_tensor *t);
void chelis_print_f32(const chelis_tensor *t);

#endif
