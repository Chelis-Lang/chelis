#include "chelis_runtime.h"

chelis_tensor* chelis_alloc(int ndim, const int *shape, int dtype) {
    chelis_tensor *t = (chelis_tensor*)calloc(1, sizeof(chelis_tensor));
    t->ndim = ndim;
    t->dtype = dtype;
    t->size = 1;
    for (int d = 0; d < ndim; d++) {
        t->shape[d] = shape[d];
        t->size *= shape[d];
    }
    /* Row-major strides */
    for (int d = ndim - 1; d >= 0; d--) {
        t->strides[d] = (d == ndim - 1) ? 1 : t->strides[d + 1] * t->shape[d + 1];
    }
    if (t->size == 0) t->size = 1; /* scalar */
    t->data = (float*)calloc(t->size, sizeof(float));
    return t;
}

void chelis_free(chelis_tensor *t) {
    if (t) { free(t->data); free(t); }
}

void chelis_fill_f32(chelis_tensor *t, float val) {
    for (int i = 0; i < t->size; i++) t->data[i] = val;
}

chelis_tensor* chelis_contiguous(const chelis_tensor *t) {
    if (chelis_is_contiguous(t)) {
        /* Already contiguous, just copy */
        chelis_tensor *out = chelis_alloc(t->ndim, t->shape, t->dtype);
        memcpy(out->data, t->data, t->size * sizeof(float));
        return out;
    }
    chelis_tensor *out = chelis_alloc(t->ndim, t->shape, t->dtype);
    int indices[CHELIS_MAX_DIM];
    for (int i = 0; i < out->size; i++) {
        chelis_flat_to_indices(i, out->shape, out->ndim, indices);
        int src = chelis_indices_to_flat(indices, t->strides, t->ndim);
        out->data[i] = t->data[src];
    }
    return out;
}

void chelis_print_f32(const chelis_tensor *t) {
    printf("tensor(shape=[");
    for (int d = 0; d < t->ndim; d++) { if (d) printf(","); printf("%d", t->shape[d]); }
    printf("], data=[");
    int n = t->size < 10 ? t->size : 10;
    for (int i = 0; i < n; i++) { if (i) printf(", "); printf("%.6f", t->data[i]); }
    if (t->size > 10) printf(", ...");
    printf("])\n");
}
