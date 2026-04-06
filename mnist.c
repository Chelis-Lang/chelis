#include "chelis_runtime.h"

void mnist(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out) {
    if (n_in != 6) {
        fprintf(stderr, "mnist: expected %d inputs, got %d\n", 6, n_in);
        abort();
    }
    if (inputs == NULL) {
        fprintf(stderr, "mnist: inputs array is NULL but %d inputs are required\n", 6);
        abort();
    }
    if (n_out != 18) {
        fprintf(stderr, "mnist: expected %d outputs, got %d\n", 18, n_out);
        abort();
    }
    if (outputs == NULL) {
        fprintf(stderr, "mnist: outputs array is NULL but %d outputs are required\n", 18);
        abort();
    }
    chelis_tensor *t0 = inputs[0];
    chelis_tensor *t1 = inputs[1];
    chelis_tensor *t2 = inputs[2];
    chelis_tensor *t3 = inputs[3];
    chelis_tensor *t4 = inputs[4];
    chelis_tensor *t5 = inputs[5];
    chelis_tensor *t6 = chelis_alloc_view(3, (int[]){ 32, 784, 128 }, CHELIS_F32, t0->data);
    if (t6->ndim == t0->ndim) {
        for (int d = 0; d < t0->ndim; d++) t6->strides[d] = t0->strides[d];
        t6->strides[2] = 0;
    } else {
        t6->strides[0] = t0->strides[0];
        t6->strides[1] = t0->strides[1];
        t6->strides[2] = 0;
        for (int d = 2; d < t0->ndim; d++) t6->strides[d+1] = t0->strides[d];
    }
    chelis_tensor *t7 = chelis_alloc_view(3, (int[]){ 32, 784, 128 }, CHELIS_F32, t2->data);
    if (t7->ndim == t2->ndim) {
        for (int d = 0; d < t2->ndim; d++) t7->strides[d] = t2->strides[d];
        t7->strides[0] = 0;
    } else {
        t7->strides[0] = 0;
        for (int d = 0; d < t2->ndim; d++) t7->strides[d+1] = t2->strides[d];
    }
    chelis_tensor *t8 = chelis_alloc(3, (int[]){ 32, 784, 128 }, CHELIS_F32);
    #pragma omp parallel for
    for (int i = 0; i < t8->size; i++) {
        int indices[CHELIS_MAX_DIM];
        chelis_flat_to_indices(i, t8->shape, t8->ndim, indices);
        int idx_a = chelis_indices_to_flat(indices, t6->strides, t6->ndim);
        int idx_b = chelis_indices_to_flat(indices, t7->strides, t7->ndim);
        t8->data[i] = t6->data[idx_a] * t7->data[idx_b];
    }
    chelis_tensor *t9 = chelis_alloc(2, (int[]){ 32, 128 }, CHELIS_F32);
    chelis_fill_f32(t9, 0.0f);
    #pragma omp parallel for
    for (int outer = 0; outer < t9->size; outer++) {
        float acc = 0.0f;
        int out_indices[CHELIS_MAX_DIM];
        chelis_flat_to_indices(outer, t9->shape, t9->ndim, out_indices);
        for (int k = 0; k < 784; k++) {
            int full_indices[CHELIS_MAX_DIM];
            int out_d = 0;
            for (int d = 0; d < t8->ndim; d++) {
                if (d == 1) {
                    full_indices[d] = k;
                } else {
                    full_indices[d] = out_indices[out_d];
                    out_d++;
                }
            }
            int src_idx = chelis_indices_to_flat(full_indices, t8->strides, t8->ndim);
            acc += t8->data[src_idx];
        }
        t9->data[outer] = acc;
    }
    chelis_tensor *t10 = chelis_alloc_view(2, (int[]){ 32, 128 }, CHELIS_F32, t3->data);
    if (t10->ndim == t3->ndim) {
        for (int d = 0; d < t3->ndim; d++) t10->strides[d] = t3->strides[d];
        t10->strides[0] = 0;
    } else {
        t10->strides[0] = 0;
        for (int d = 0; d < t3->ndim; d++) t10->strides[d+1] = t3->strides[d];
    }
    chelis_tensor *t11 = chelis_alloc(2, (int[]){ 32, 128 }, CHELIS_F32);
    #pragma omp parallel for
    for (int i = 0; i < t11->size; i++) {
        int indices[CHELIS_MAX_DIM];
        chelis_flat_to_indices(i, t11->shape, t11->ndim, indices);
        int idx_a = chelis_indices_to_flat(indices, t9->strides, t9->ndim);
        int idx_b = chelis_indices_to_flat(indices, t10->strides, t10->ndim);
        t11->data[i] = t9->data[idx_a] + t10->data[idx_b];
    }
    chelis_tensor *t12 = chelis_alloc(2, (int[]){ 32, 128 }, CHELIS_F32);
    chelis_fill_f32(t12, 0.00000000f);
    chelis_tensor *t13 = chelis_alloc(2, (int[]){ 32, 128 }, CHELIS_F32);
    #pragma omp parallel for
    for (int i = 0; i < t13->size; i++) {
        int indices[CHELIS_MAX_DIM];
        chelis_flat_to_indices(i, t13->shape, t13->ndim, indices);
        int idx_a = chelis_indices_to_flat(indices, t11->strides, t11->ndim);
        int idx_b = chelis_indices_to_flat(indices, t12->strides, t12->ndim);
        t13->data[i] = fmaxf(t11->data[idx_a], t12->data[idx_b]);
    }
    chelis_tensor *t14 = chelis_alloc_view(3, (int[]){ 32, 128, 10 }, CHELIS_F32, t13->data);
    if (t14->ndim == t13->ndim) {
        for (int d = 0; d < t13->ndim; d++) t14->strides[d] = t13->strides[d];
        t14->strides[2] = 0;
    } else {
        t14->strides[0] = t13->strides[0];
        t14->strides[1] = t13->strides[1];
        t14->strides[2] = 0;
        for (int d = 2; d < t13->ndim; d++) t14->strides[d+1] = t13->strides[d];
    }
    chelis_tensor *t15 = chelis_alloc_view(3, (int[]){ 32, 128, 10 }, CHELIS_F32, t4->data);
    if (t15->ndim == t4->ndim) {
        for (int d = 0; d < t4->ndim; d++) t15->strides[d] = t4->strides[d];
        t15->strides[0] = 0;
    } else {
        t15->strides[0] = 0;
        for (int d = 0; d < t4->ndim; d++) t15->strides[d+1] = t4->strides[d];
    }
    chelis_tensor *t16 = chelis_alloc(3, (int[]){ 32, 128, 10 }, CHELIS_F32);
    #pragma omp parallel for
    for (int i = 0; i < t16->size; i++) {
        int indices[CHELIS_MAX_DIM];
        chelis_flat_to_indices(i, t16->shape, t16->ndim, indices);
        int idx_a = chelis_indices_to_flat(indices, t14->strides, t14->ndim);
        int idx_b = chelis_indices_to_flat(indices, t15->strides, t15->ndim);
        t16->data[i] = t14->data[idx_a] * t15->data[idx_b];
    }
    chelis_tensor *t17 = chelis_alloc(2, (int[]){ 32, 10 }, CHELIS_F32);
    chelis_fill_f32(t17, 0.0f);
    #pragma omp parallel for
    for (int outer = 0; outer < t17->size; outer++) {
        float acc = 0.0f;
        int out_indices[CHELIS_MAX_DIM];
        chelis_flat_to_indices(outer, t17->shape, t17->ndim, out_indices);
        for (int k = 0; k < 128; k++) {
            int full_indices[CHELIS_MAX_DIM];
            int out_d = 0;
            for (int d = 0; d < t16->ndim; d++) {
                if (d == 1) {
                    full_indices[d] = k;
                } else {
                    full_indices[d] = out_indices[out_d];
                    out_d++;
                }
            }
            int src_idx = chelis_indices_to_flat(full_indices, t16->strides, t16->ndim);
            acc += t16->data[src_idx];
        }
        t17->data[outer] = acc;
    }
    chelis_tensor *t18 = chelis_alloc_view(2, (int[]){ 32, 10 }, CHELIS_F32, t5->data);
    if (t18->ndim == t5->ndim) {
        for (int d = 0; d < t5->ndim; d++) t18->strides[d] = t5->strides[d];
        t18->strides[0] = 0;
    } else {
        t18->strides[0] = 0;
        for (int d = 0; d < t5->ndim; d++) t18->strides[d+1] = t5->strides[d];
    }
    chelis_tensor *t19 = chelis_alloc(2, (int[]){ 32, 10 }, CHELIS_F32);
    #pragma omp parallel for
    for (int i = 0; i < t19->size; i++) {
        int indices[CHELIS_MAX_DIM];
        chelis_flat_to_indices(i, t19->shape, t19->ndim, indices);
        int idx_a = chelis_indices_to_flat(indices, t17->strides, t17->ndim);
        int idx_b = chelis_indices_to_flat(indices, t18->strides, t18->ndim);
        t19->data[i] = t17->data[idx_a] + t18->data[idx_b];
    }
    chelis_tensor *t20 = chelis_alloc(1, (int[]){ 32 }, CHELIS_F32);
    #pragma omp parallel for
    for (int outer = 0; outer < t20->size; outer++) {
        float acc = -INFINITY;
        int out_indices[CHELIS_MAX_DIM];
        chelis_flat_to_indices(outer, t20->shape, t20->ndim, out_indices);
        for (int k = 0; k < 10; k++) {
            int full_indices[CHELIS_MAX_DIM];
            int out_d = 0;
            for (int d = 0; d < t19->ndim; d++) {
                if (d == 1) {
                    full_indices[d] = k;
                } else {
                    full_indices[d] = out_indices[out_d];
                    out_d++;
                }
            }
            int src_idx = chelis_indices_to_flat(full_indices, t19->strides, t19->ndim);
            acc = fmaxf(acc, t19->data[src_idx]);
        }
        t20->data[outer] = acc;
    }
    chelis_tensor *t21 = chelis_alloc_view(2, (int[]){ 32, 10 }, CHELIS_F32, t20->data);
    if (t21->ndim == t20->ndim) {
        for (int d = 0; d < t20->ndim; d++) t21->strides[d] = t20->strides[d];
        t21->strides[1] = 0;
    } else {
        t21->strides[0] = t20->strides[0];
        t21->strides[1] = 0;
        for (int d = 1; d < t20->ndim; d++) t21->strides[d+1] = t20->strides[d];
    }
    chelis_tensor *t22 = chelis_alloc(2, (int[]){ 32, 10 }, CHELIS_F32);
    #pragma omp parallel for
    for (int i = 0; i < t22->size; i++) {
        int indices[CHELIS_MAX_DIM];
        chelis_flat_to_indices(i, t22->shape, t22->ndim, indices);
        int idx = chelis_indices_to_flat(indices, t21->strides, t21->ndim);
        t22->data[i] = -t21->data[idx];
    }
    chelis_tensor *t23 = chelis_alloc(2, (int[]){ 32, 10 }, CHELIS_F32);
    #pragma omp parallel for
    for (int i = 0; i < t23->size; i++) {
        int indices[CHELIS_MAX_DIM];
        chelis_flat_to_indices(i, t23->shape, t23->ndim, indices);
        int idx_a = chelis_indices_to_flat(indices, t19->strides, t19->ndim);
        int idx_b = chelis_indices_to_flat(indices, t22->strides, t22->ndim);
        t23->data[i] = t19->data[idx_a] + t22->data[idx_b];
    }
    chelis_tensor *t24 = chelis_alloc(2, (int[]){ 32, 10 }, CHELIS_F32);
    #pragma omp parallel for
    for (int i = 0; i < t24->size; i++) {
        int indices[CHELIS_MAX_DIM];
        chelis_flat_to_indices(i, t24->shape, t24->ndim, indices);
        int idx = chelis_indices_to_flat(indices, t23->strides, t23->ndim);
        t24->data[i] = expf(t23->data[idx]);
    }
    chelis_tensor *t25 = chelis_alloc(1, (int[]){ 32 }, CHELIS_F32);
    chelis_fill_f32(t25, 0.0f);
    #pragma omp parallel for
    for (int outer = 0; outer < t25->size; outer++) {
        float acc = 0.0f;
        int out_indices[CHELIS_MAX_DIM];
        chelis_flat_to_indices(outer, t25->shape, t25->ndim, out_indices);
        for (int k = 0; k < 10; k++) {
            int full_indices[CHELIS_MAX_DIM];
            int out_d = 0;
            for (int d = 0; d < t24->ndim; d++) {
                if (d == 1) {
                    full_indices[d] = k;
                } else {
                    full_indices[d] = out_indices[out_d];
                    out_d++;
                }
            }
            int src_idx = chelis_indices_to_flat(full_indices, t24->strides, t24->ndim);
            acc += t24->data[src_idx];
        }
        t25->data[outer] = acc;
    }
    chelis_tensor *t26 = chelis_alloc_view(2, (int[]){ 32, 10 }, CHELIS_F32, t25->data);
    if (t26->ndim == t25->ndim) {
        for (int d = 0; d < t25->ndim; d++) t26->strides[d] = t25->strides[d];
        t26->strides[1] = 0;
    } else {
        t26->strides[0] = t25->strides[0];
        t26->strides[1] = 0;
        for (int d = 1; d < t25->ndim; d++) t26->strides[d+1] = t25->strides[d];
    }
    chelis_tensor *t27 = chelis_alloc(2, (int[]){ 32, 10 }, CHELIS_F32);
    #pragma omp parallel for
    for (int i = 0; i < t27->size; i++) {
        int indices[CHELIS_MAX_DIM];
        chelis_flat_to_indices(i, t27->shape, t27->ndim, indices);
        int idx = chelis_indices_to_flat(indices, t26->strides, t26->ndim);
        t27->data[i] = logf(t26->data[idx]);
    }
    chelis_tensor *t28 = chelis_alloc(2, (int[]){ 32, 10 }, CHELIS_F32);
    #pragma omp parallel for
    for (int i = 0; i < t28->size; i++) {
        int indices[CHELIS_MAX_DIM];
        chelis_flat_to_indices(i, t28->shape, t28->ndim, indices);
        int idx = chelis_indices_to_flat(indices, t27->strides, t27->ndim);
        t28->data[i] = -t27->data[idx];
    }
    chelis_tensor *t29 = chelis_alloc(2, (int[]){ 32, 10 }, CHELIS_F32);
    #pragma omp parallel for
    for (int i = 0; i < t29->size; i++) {
        int indices[CHELIS_MAX_DIM];
        chelis_flat_to_indices(i, t29->shape, t29->ndim, indices);
        int idx = chelis_indices_to_flat(indices, t28->strides, t28->ndim);
        t29->data[i] = expf(t28->data[idx]);
    }
    chelis_tensor *t30 = chelis_alloc(2, (int[]){ 32, 10 }, CHELIS_F32);
    #pragma omp parallel for
    for (int i = 0; i < t30->size; i++) {
        int indices[CHELIS_MAX_DIM];
        chelis_flat_to_indices(i, t30->shape, t30->ndim, indices);
        int idx_a = chelis_indices_to_flat(indices, t24->strides, t24->ndim);
        int idx_b = chelis_indices_to_flat(indices, t29->strides, t29->ndim);
        t30->data[i] = t24->data[idx_a] * t29->data[idx_b];
    }
    chelis_tensor *t31 = chelis_alloc(2, (int[]){ 32, 10 }, CHELIS_F32);
    #pragma omp parallel for
    for (int i = 0; i < t31->size; i++) {
        int indices[CHELIS_MAX_DIM];
        chelis_flat_to_indices(i, t31->shape, t31->ndim, indices);
        int idx = chelis_indices_to_flat(indices, t30->strides, t30->ndim);
        t31->data[i] = logf(t30->data[idx]);
    }
    chelis_tensor *t32 = chelis_alloc(2, (int[]){ 32, 10 }, CHELIS_F32);
    #pragma omp parallel for
    for (int i = 0; i < t32->size; i++) {
        int indices[CHELIS_MAX_DIM];
        chelis_flat_to_indices(i, t32->shape, t32->ndim, indices);
        int idx_a = chelis_indices_to_flat(indices, t31->strides, t31->ndim);
        int idx_b = chelis_indices_to_flat(indices, t1->strides, t1->ndim);
        t32->data[i] = t31->data[idx_a] * t1->data[idx_b];
    }
    chelis_tensor *t33 = chelis_alloc(1, (int[]){ 32 }, CHELIS_F32);
    chelis_fill_f32(t33, 0.0f);
    #pragma omp parallel for
    for (int outer = 0; outer < t33->size; outer++) {
        float acc = 0.0f;
        int out_indices[CHELIS_MAX_DIM];
        chelis_flat_to_indices(outer, t33->shape, t33->ndim, out_indices);
        for (int k = 0; k < 10; k++) {
            int full_indices[CHELIS_MAX_DIM];
            int out_d = 0;
            for (int d = 0; d < t32->ndim; d++) {
                if (d == 1) {
                    full_indices[d] = k;
                } else {
                    full_indices[d] = out_indices[out_d];
                    out_d++;
                }
            }
            int src_idx = chelis_indices_to_flat(full_indices, t32->strides, t32->ndim);
            acc += t32->data[src_idx];
        }
        t33->data[outer] = acc;
    }
    chelis_tensor *t34 = chelis_alloc(1, (int[]){ 32 }, CHELIS_F32);
    #pragma omp parallel for
    for (int i = 0; i < t34->size; i++) {
        int indices[CHELIS_MAX_DIM];
        chelis_flat_to_indices(i, t34->shape, t34->ndim, indices);
        int idx = chelis_indices_to_flat(indices, t33->strides, t33->ndim);
        t34->data[i] = -t33->data[idx];
    }
    chelis_tensor *t35 = chelis_alloc(1, (int[]){1}, CHELIS_F32);
    chelis_fill_f32(t35, 0.0f);
    #pragma omp parallel for
    for (int outer = 0; outer < t35->size; outer++) {
        float acc = 0.0f;
        int out_indices[CHELIS_MAX_DIM];
        chelis_flat_to_indices(outer, t35->shape, t35->ndim, out_indices);
        for (int k = 0; k < 32; k++) {
            int full_indices[CHELIS_MAX_DIM];
            int out_d = 0;
            for (int d = 0; d < t34->ndim; d++) {
                if (d == 0) {
                    full_indices[d] = k;
                } else {
                    full_indices[d] = out_indices[out_d];
                    out_d++;
                }
            }
            int src_idx = chelis_indices_to_flat(full_indices, t34->strides, t34->ndim);
            acc += t34->data[src_idx];
        }
        t35->data[outer] = acc;
    }
    chelis_tensor *t36 = chelis_alloc(1, (int[]){1}, CHELIS_F32);
    chelis_fill_f32(t36, 32.00000000f);
    chelis_tensor *t37 = chelis_alloc(1, (int[]){1}, CHELIS_F32);
    #pragma omp parallel for
    for (int i = 0; i < t37->size; i++) {
        int indices[CHELIS_MAX_DIM];
        chelis_flat_to_indices(i, t37->shape, t37->ndim, indices);
        int idx = chelis_indices_to_flat(indices, t36->strides, t36->ndim);
        t37->data[i] = logf(t36->data[idx]);
    }
    chelis_tensor *t38 = chelis_alloc(1, (int[]){1}, CHELIS_F32);
    #pragma omp parallel for
    for (int i = 0; i < t38->size; i++) {
        int indices[CHELIS_MAX_DIM];
        chelis_flat_to_indices(i, t38->shape, t38->ndim, indices);
        int idx = chelis_indices_to_flat(indices, t37->strides, t37->ndim);
        t38->data[i] = -t37->data[idx];
    }
    chelis_tensor *t39 = chelis_alloc(1, (int[]){1}, CHELIS_F32);
    #pragma omp parallel for
    for (int i = 0; i < t39->size; i++) {
        int indices[CHELIS_MAX_DIM];
        chelis_flat_to_indices(i, t39->shape, t39->ndim, indices);
        int idx = chelis_indices_to_flat(indices, t38->strides, t38->ndim);
        t39->data[i] = expf(t38->data[idx]);
    }
    chelis_tensor *t40 = chelis_alloc(1, (int[]){1}, CHELIS_F32);
    #pragma omp parallel for
    for (int i = 0; i < t40->size; i++) {
        int indices[CHELIS_MAX_DIM];
        chelis_flat_to_indices(i, t40->shape, t40->ndim, indices);
        int idx_a = chelis_indices_to_flat(indices, t35->strides, t35->ndim);
        int idx_b = chelis_indices_to_flat(indices, t39->strides, t39->ndim);
        t40->data[i] = t35->data[idx_a] * t39->data[idx_b];
    }
    outputs[0] = chelis_contiguous(t0);
    outputs[1] = chelis_contiguous(t1);
    outputs[2] = chelis_contiguous(t2);
    outputs[3] = chelis_contiguous(t3);
    outputs[4] = chelis_contiguous(t4);
    outputs[5] = chelis_contiguous(t5);
    outputs[6] = chelis_contiguous(t9);
    outputs[7] = chelis_contiguous(t10);
    outputs[8] = chelis_contiguous(t11);
    outputs[9] = chelis_contiguous(t13);
    outputs[10] = chelis_contiguous(t17);
    outputs[11] = chelis_contiguous(t18);
    outputs[12] = chelis_contiguous(t19);
    outputs[13] = chelis_contiguous(t30);
    outputs[14] = chelis_contiguous(t31);
    outputs[15] = chelis_contiguous(t32);
    outputs[16] = chelis_contiguous(t34);
    outputs[17] = chelis_contiguous(t40);
    chelis_free(t6);
    chelis_free(t7);
    chelis_free(t8);
    chelis_free(t12);
    chelis_free(t14);
    chelis_free(t15);
    chelis_free(t16);
    chelis_free(t20);
    chelis_free(t21);
    chelis_free(t22);
    chelis_free(t23);
    chelis_free(t24);
    chelis_free(t25);
    chelis_free(t26);
    chelis_free(t27);
    chelis_free(t28);
    chelis_free(t29);
    chelis_free(t33);
    chelis_free(t35);
    chelis_free(t36);
    chelis_free(t37);
    chelis_free(t38);
    chelis_free(t39);
}