#ifndef CHELIS_HIP_RUNTIME_H
#define CHELIS_HIP_RUNTIME_H

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if __has_include(<hipblas/hipblas.h>)
#include <hipblas/hipblas.h>
#define CHELIS_HAS_HIPBLAS_HEADER 1
#else
#define CHELIS_HAS_HIPBLAS_HEADER 0
typedef enum {
    HIPBLAS_STATUS_SUCCESS = 0
} hipblasStatus_t;
typedef struct hipblasContext *hipblasHandle_t;
typedef enum {
    HIPBLAS_OP_N = 111,
    HIPBLAS_OP_T = 112,
    HIPBLAS_OP_C = 113
} hipblasOperation_t;
extern hipblasStatus_t hipblasCreate(hipblasHandle_t *handle);
extern hipblasStatus_t hipblasDestroy(hipblasHandle_t handle);
extern hipblasStatus_t hipblasSgemm(
    hipblasHandle_t handle,
    hipblasOperation_t transa,
    hipblasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha,
    const float *A,
    int lda,
    const float *B,
    int ldb,
    const float *beta,
    float *C,
    int ldc
);
extern hipblasStatus_t hipblasSgemmStridedBatched(
    hipblasHandle_t handle,
    hipblasOperation_t transa,
    hipblasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha,
    const float *A,
    int lda,
    long long strideA,
    const float *B,
    int ldb,
    long long strideB,
    const float *beta,
    float *C,
    int ldc,
    long long strideC,
    int batchCount
);
extern hipblasStatus_t hipblasDgemm(
    hipblasHandle_t handle,
    hipblasOperation_t transa,
    hipblasOperation_t transb,
    int m,
    int n,
    int k,
    const double *alpha,
    const double *A,
    int lda,
    const double *B,
    int ldb,
    const double *beta,
    double *C,
    int ldc
);
extern hipblasStatus_t hipblasDgemmStridedBatched(
    hipblasHandle_t handle,
    hipblasOperation_t transa,
    hipblasOperation_t transb,
    int m,
    int n,
    int k,
    const double *alpha,
    const double *A,
    int lda,
    long long strideA,
    const double *B,
    int ldb,
    long long strideB,
    const double *beta,
    double *C,
    int ldc,
    long long strideC,
    int batchCount
);
#endif

/* Re-use the CPU tensor struct for host-side data. */
#include "chelis_runtime.h"

#define CHELIS_HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", \
                __FILE__, __LINE__, hipGetErrorString(err)); \
        abort(); \
    } \
} while (0)

#define CHELIS_HIPRTC_CHECK(call) do { \
    hiprtcResult err = (call); \
    if (err != HIPRTC_SUCCESS) { \
        fprintf(stderr, "hiprtc error at %s:%d: %s\n", \
                __FILE__, __LINE__, hiprtcGetErrorString(err)); \
        abort(); \
    } \
} while (0)

#define CHELIS_HIPBLAS_CHECK(call) do { \
    hipblasStatus_t status = (call); \
    if (status != HIPBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "hipBLAS error at %s:%d: status=%d\n", __FILE__, __LINE__, (int)status); \
        abort(); \
    } \
} while (0)

/* GPU tensor: device pointer + shape metadata on host. */
typedef struct {
    float *data;                    /* device pointer (hipMalloc) */
    int shape[CHELIS_MAX_DIM];
    int strides[CHELIS_MAX_DIM];
    int ndim;
    int dtype;
    int size;                       /* total elements */
    int storage_size;               /* backing allocation elements */
} chelis_gpu_tensor;

/* ---- Allocation / deallocation ---- */

static inline size_t chelis_gpu_dtype_size(int dtype) {
    return (dtype == CHELIS_I64 || dtype == CHELIS_F64) ? sizeof(int64_t) : sizeof(float);
}

static inline chelis_gpu_tensor* chelis_gpu_alloc(int ndim, const int *shape, int dtype) {
    chelis_gpu_tensor *t = (chelis_gpu_tensor*)calloc(1, sizeof(chelis_gpu_tensor));
    t->ndim = ndim;
    t->dtype = dtype;
    t->size = 1;
    for (int d = 0; d < ndim; d++) {
        t->shape[d] = shape[d];
        t->size *= shape[d];
    }
    if (t->size == 0) t->size = 1; /* scalar */
    t->storage_size = t->size;
    /* Row-major strides */
    for (int d = ndim - 1; d >= 0; d--) {
        t->strides[d] = (d == ndim - 1) ? 1 : t->strides[d + 1] * t->shape[d + 1];
    }
    CHELIS_HIP_CHECK(hipMalloc(&t->data, t->size * chelis_gpu_dtype_size(dtype)));
    return t;
}

/* Allocate a GPU tensor that shares device memory (view, not owned). */
static inline chelis_gpu_tensor* chelis_gpu_alloc_view(
    int ndim, const int *shape, int dtype, float *device_data, int storage_size
) {
    chelis_gpu_tensor *t = (chelis_gpu_tensor*)calloc(1, sizeof(chelis_gpu_tensor));
    t->ndim = ndim;
    t->dtype = dtype;
    t->size = 1;
    for (int d = 0; d < ndim; d++) {
        t->shape[d] = shape[d];
        t->size *= shape[d];
    }
    if (t->size == 0) t->size = 1;
    t->storage_size = storage_size;
    for (int d = ndim - 1; d >= 0; d--) {
        t->strides[d] = (d == ndim - 1) ? 1 : t->strides[d + 1] * t->shape[d + 1];
    }
    t->data = device_data;
    return t;
}

/* Free a GPU tensor that OWNS its device memory (from chelis_gpu_alloc).
   Frees both the device data (hipFree) and the host-side struct (free).
   Do NOT call this on views — use chelis_gpu_free_view instead. */
static inline void chelis_gpu_free(chelis_gpu_tensor *t) {
    if (t) {
        (void)hipFree(t->data);
        free(t);
    }
}

/* Free a GPU tensor VIEW that BORROWS device memory (from chelis_gpu_alloc_view).
   Frees only the host-side metadata struct. Does NOT free device data.
   Movement ops (reshape, permute, expand, stride) create views. */
static inline void chelis_gpu_free_view(chelis_gpu_tensor *t) {
    if (t) free(t);
}

/* ---- Host ↔ Device transfer ---- */

static inline void chelis_host_to_device(chelis_gpu_tensor *dst, const chelis_tensor *src) {
    CHELIS_HIP_CHECK(hipMemcpy(dst->data, src->data,
                               dst->size * chelis_gpu_dtype_size(dst->dtype),
                               hipMemcpyHostToDevice));
}

static inline void chelis_device_to_host(chelis_tensor *dst, const chelis_gpu_tensor *src) {
    CHELIS_HIP_CHECK(hipMemcpy(dst->data, src->data,
                               dst->size * chelis_gpu_dtype_size(src->dtype),
                               hipMemcpyDeviceToHost));
}

static inline chelis_gpu_tensor* chelis_gpu_clone(const chelis_gpu_tensor *src) {
    chelis_gpu_tensor *dst = chelis_gpu_alloc(src->ndim, src->shape, src->dtype);
    for (int d = 0; d < src->ndim; d++) {
        dst->strides[d] = src->strides[d];
    }
    dst->size = src->size;
    dst->storage_size = src->size;
    CHELIS_HIP_CHECK(hipMemcpy(
        dst->data,
        src->data,
        src->size * chelis_gpu_dtype_size(src->dtype),
        hipMemcpyDeviceToDevice
    ));
    return dst;
}

/* ---- Host-side layout helpers ---- */

static inline int chelis_gpu_is_contiguous(const chelis_gpu_tensor *t) {
    int expected = 1;
    for (int d = t->ndim - 1; d >= 0; d--) {
        if (t->strides[d] != expected) return 0;
        expected *= t->shape[d];
    }
    return 1;
}

/* ---- hipBLAS helpers ---- */

static inline void chelis_hipblas_sgemm_row_major(
    const chelis_gpu_tensor *a,
    const chelis_gpu_tensor *b,
    chelis_gpu_tensor *out,
    int m,
    int n,
    int k
) {
    hipblasHandle_t handle;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHELIS_HIPBLAS_CHECK(hipblasCreate(&handle));
    /* hipBLAS is column-major by default. Swap A/B and m/n to preserve row-major semantics. */
    CHELIS_HIPBLAS_CHECK(hipblasSgemm(
        handle,
        HIPBLAS_OP_N,
        HIPBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        b->data,
        n,
        a->data,
        k,
        &beta,
        out->data,
        n
    ));
    CHELIS_HIPBLAS_CHECK(hipblasDestroy(handle));
}

static inline int chelis_gpu_matrix_slices_contiguous(
    const chelis_gpu_tensor *t,
    int trailing_cols
) {
    if (t->ndim < 2) return 0;
    return t->strides[t->ndim - 1] == 1
        && t->strides[t->ndim - 2] == trailing_cols;
}

static inline void chelis_hipblas_sgemm_batched_row_major(
    const chelis_gpu_tensor *a,
    const chelis_gpu_tensor *b,
    chelis_gpu_tensor *out,
    int m,
    int n,
    int k
) {
    if (!chelis_gpu_matrix_slices_contiguous(a, k)
        || !chelis_gpu_matrix_slices_contiguous(b, n)
        || !chelis_gpu_matrix_slices_contiguous(out, n)) {
        fprintf(stderr, "chelis_hipblas_sgemm_batched_row_major: matrix slices must be contiguous\n");
        abort();
    }

    int batch_ndim = out->ndim - 2;
    int batch_count = 1;
    for (int d = 0; d < batch_ndim; d++) {
        batch_count *= out->shape[d];
    }

    hipblasHandle_t handle;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHELIS_HIPBLAS_CHECK(hipblasCreate(&handle));
    for (int batch = 0; batch < batch_count; batch++) {
        int rem = batch;
        int a_offset = 0;
        int b_offset = 0;
        int out_offset = 0;
        for (int d = batch_ndim - 1; d >= 0; d--) {
            int coord = rem % out->shape[d];
            rem /= out->shape[d];
            a_offset += coord * a->strides[d];
            b_offset += coord * b->strides[d];
            out_offset += coord * out->strides[d];
        }
        CHELIS_HIPBLAS_CHECK(hipblasSgemm(
            handle,
            HIPBLAS_OP_N,
            HIPBLAS_OP_N,
            n,
            m,
            k,
            &alpha,
            b->data + b_offset,
            n,
            a->data + a_offset,
            k,
            &beta,
            out->data + out_offset,
            n
        ));
    }
    CHELIS_HIPBLAS_CHECK(hipblasDestroy(handle));
}

static inline void chelis_hipblas_sgemm_strided_batched_row_major(
    const chelis_gpu_tensor *a,
    const chelis_gpu_tensor *b,
    chelis_gpu_tensor *out,
    int m,
    int n,
    int k,
    int batch_count,
    long long a_batch_stride,
    long long b_batch_stride,
    long long out_batch_stride
) {
    if (!chelis_gpu_matrix_slices_contiguous(a, k)
        || !chelis_gpu_matrix_slices_contiguous(b, n)
        || !chelis_gpu_matrix_slices_contiguous(out, n)) {
        fprintf(stderr, "chelis_hipblas_sgemm_strided_batched_row_major: matrix slices must be contiguous\n");
        abort();
    }

    hipblasHandle_t handle;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHELIS_HIPBLAS_CHECK(hipblasCreate(&handle));
    /* hipBLAS is column-major by default. Swap A/B and m/n to preserve row-major semantics. */
    CHELIS_HIPBLAS_CHECK(hipblasSgemmStridedBatched(
        handle,
        HIPBLAS_OP_N,
        HIPBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        b->data,
        n,
        b_batch_stride,
        a->data,
        k,
        a_batch_stride,
        &beta,
        out->data,
        n,
        out_batch_stride,
        batch_count
    ));
    CHELIS_HIPBLAS_CHECK(hipblasDestroy(handle));
}

/* ---- f64 hipBLAS helpers (WS-A2) ---- */

/* The chelis_gpu_tensor struct's `data` field is typed as `float *`
 * historically, but the underlying device allocation is sized via
 * `chelis_gpu_dtype_size(dtype)` so the same struct holds f32 (4 B) and
 * f64 (8 B) elements. The f64 helpers below cast through `void *` to
 * the right pointer type before calling hipblasDgemm; arithmetic on the
 * cast `double *` advances by 8 bytes per element, so the existing
 * row-major-via-column-major-swap convention is preserved. */

static inline void chelis_hipblas_dgemm_row_major(
    const chelis_gpu_tensor *a,
    const chelis_gpu_tensor *b,
    chelis_gpu_tensor *out,
    int m,
    int n,
    int k
) {
    hipblasHandle_t handle;
    const double alpha = 1.0;
    const double beta = 0.0;
    CHELIS_HIPBLAS_CHECK(hipblasCreate(&handle));
    CHELIS_HIPBLAS_CHECK(hipblasDgemm(
        handle,
        HIPBLAS_OP_N,
        HIPBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        (const double *)(const void *)b->data,
        n,
        (const double *)(const void *)a->data,
        k,
        &beta,
        (double *)(void *)out->data,
        n
    ));
    CHELIS_HIPBLAS_CHECK(hipblasDestroy(handle));
}

static inline void chelis_hipblas_dgemm_batched_row_major(
    const chelis_gpu_tensor *a,
    const chelis_gpu_tensor *b,
    chelis_gpu_tensor *out,
    int m,
    int n,
    int k
) {
    if (!chelis_gpu_matrix_slices_contiguous(a, k)
        || !chelis_gpu_matrix_slices_contiguous(b, n)
        || !chelis_gpu_matrix_slices_contiguous(out, n)) {
        fprintf(stderr, "chelis_hipblas_dgemm_batched_row_major: matrix slices must be contiguous\n");
        abort();
    }

    int batch_ndim = out->ndim - 2;
    int batch_count = 1;
    for (int d = 0; d < batch_ndim; d++) {
        batch_count *= out->shape[d];
    }

    hipblasHandle_t handle;
    const double alpha = 1.0;
    const double beta = 0.0;
    const double *a_base = (const double *)(const void *)a->data;
    const double *b_base = (const double *)(const void *)b->data;
    double *out_base = (double *)(void *)out->data;
    CHELIS_HIPBLAS_CHECK(hipblasCreate(&handle));
    for (int batch = 0; batch < batch_count; batch++) {
        int rem = batch;
        long long a_offset = 0;
        long long b_offset = 0;
        long long out_offset = 0;
        for (int d = batch_ndim - 1; d >= 0; d--) {
            int coord = rem % out->shape[d];
            rem /= out->shape[d];
            a_offset += (long long)coord * a->strides[d];
            b_offset += (long long)coord * b->strides[d];
            out_offset += (long long)coord * out->strides[d];
        }
        CHELIS_HIPBLAS_CHECK(hipblasDgemm(
            handle,
            HIPBLAS_OP_N,
            HIPBLAS_OP_N,
            n,
            m,
            k,
            &alpha,
            b_base + b_offset,
            n,
            a_base + a_offset,
            k,
            &beta,
            out_base + out_offset,
            n
        ));
    }
    CHELIS_HIPBLAS_CHECK(hipblasDestroy(handle));
}

static inline void chelis_hipblas_dgemm_strided_batched_row_major(
    const chelis_gpu_tensor *a,
    const chelis_gpu_tensor *b,
    chelis_gpu_tensor *out,
    int m,
    int n,
    int k,
    int batch_count,
    long long a_batch_stride,
    long long b_batch_stride,
    long long out_batch_stride
) {
    if (!chelis_gpu_matrix_slices_contiguous(a, k)
        || !chelis_gpu_matrix_slices_contiguous(b, n)
        || !chelis_gpu_matrix_slices_contiguous(out, n)) {
        fprintf(stderr, "chelis_hipblas_dgemm_strided_batched_row_major: matrix slices must be contiguous\n");
        abort();
    }

    hipblasHandle_t handle;
    const double alpha = 1.0;
    const double beta = 0.0;
    CHELIS_HIPBLAS_CHECK(hipblasCreate(&handle));
    CHELIS_HIPBLAS_CHECK(hipblasDgemmStridedBatched(
        handle,
        HIPBLAS_OP_N,
        HIPBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        (const double *)(const void *)b->data,
        n,
        b_batch_stride,
        (const double *)(const void *)a->data,
        k,
        a_batch_stride,
        &beta,
        (double *)(void *)out->data,
        n,
        out_batch_stride,
        batch_count
    ));
    CHELIS_HIPBLAS_CHECK(hipblasDestroy(handle));
}

/* ---- JIT compilation ---- */

static inline hipModule_t chelis_compile_kernel(const char *source, const char *name) {
    hiprtcProgram prog;
    CHELIS_HIPRTC_CHECK(hiprtcCreateProgram(&prog, source, name, 0, NULL, NULL));
    int device = 0;
    hipDeviceProp_t props;
    int wavefront_size = 64;
    CHELIS_HIP_CHECK(hipGetDevice(&device));
    CHELIS_HIP_CHECK(hipGetDeviceProperties(&props, device));
    if (props.warpSize > 0) {
        wavefront_size = props.warpSize;
    }
    char wavefront_opt[64];
    snprintf(
        wavefront_opt,
        sizeof(wavefront_opt),
        "-D__AMDGCN_WAVEFRONT_SIZE=%d",
        wavefront_size
    );
#ifndef NDEBUG
    const char *compile_opts[] = { "-DCHELIS_DEBUG_BOUNDS=1", wavefront_opt };
#else
    const char *compile_opts[] = { "-DCHELIS_DEBUG_BOUNDS=0", wavefront_opt };
#endif
    hiprtcResult compile_result = hiprtcCompileProgram(prog, 2, compile_opts);
    if (compile_result != HIPRTC_SUCCESS) {
        size_t log_size;
        hiprtcGetProgramLogSize(prog, &log_size);
        char *log = (char*)malloc(log_size);
        hiprtcGetProgramLog(prog, log);
        fprintf(stderr, "hiprtc compile error for '%s':\n%s\n", name, log);
        free(log);
        hiprtcDestroyProgram(&prog);
        abort();
    }
    size_t code_size;
    CHELIS_HIPRTC_CHECK(hiprtcGetCodeSize(prog, &code_size));
    char *code = (char*)malloc(code_size);
    CHELIS_HIPRTC_CHECK(hiprtcGetCode(prog, code));
    hiprtcDestroyProgram(&prog);

    hipModule_t module;
    CHELIS_HIP_CHECK(hipModuleLoadData(&module, code));
    free(code);
    return module;
}

static inline void chelis_launch_kernel(
    hipModule_t module, const char *name,
    dim3 grid, dim3 block, void **args
) {
    hipFunction_t func;
    CHELIS_HIP_CHECK(hipModuleGetFunction(&func, module, name));
    CHELIS_HIP_CHECK(hipModuleLaunchKernel(
        func,
        grid.x, grid.y, grid.z,
        block.x, block.y, block.z,
        0, /* shared mem */
        0, /* stream (default) */
        args,
        NULL
    ));
}

#ifndef NDEBUG
static inline hipDeviceptr_t chelis_gpu_failure_symbol(hipModule_t module) {
    hipDeviceptr_t symbol;
    size_t symbol_size = 0;
    CHELIS_HIP_CHECK(hipModuleGetGlobal(&symbol, &symbol_size, module, "chelis_gpu_failure"));
    if (symbol_size < sizeof(int)) {
        fprintf(stderr, "HIP error: chelis_gpu_failure has unexpected size %zu\n", symbol_size);
        abort();
    }
    return symbol;
}

static inline void chelis_prepare_kernel_launch(hipModule_t module) {
    int zero = 0;
    hipDeviceptr_t symbol = chelis_gpu_failure_symbol(module);
    CHELIS_HIP_CHECK(hipMemcpyHtoD(symbol, &zero, sizeof(zero)));
}

static inline void chelis_finalize_kernel_launch(hipModule_t module, const char *name) {
    int failure = 0;
    hipDeviceptr_t symbol = chelis_gpu_failure_symbol(module);
    CHELIS_HIP_CHECK(hipDeviceSynchronize());
    CHELIS_HIP_CHECK(hipMemcpyDtoH(&failure, symbol, sizeof(failure)));
    if (failure != 0) {
        fprintf(stderr, "Chelis GPU kernel '%s' reported device failure code %d\n", name, failure);
        abort();
    }
}
#else
static inline void chelis_prepare_kernel_launch(hipModule_t module) {
    (void)module;
}

static inline void chelis_finalize_kernel_launch(hipModule_t module, const char *name) {
    (void)module;
    (void)name;
}
#endif

/* Device-side helpers are emitted in kernel source strings by the Rust
   code generator (kernels.rs), not compiled from this header. */

#endif
