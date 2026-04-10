#ifndef CHELIS_HIP_RUNTIME_H
#define CHELIS_HIP_RUNTIME_H

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
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
    CHELIS_HIP_CHECK(hipMalloc(&t->data, t->size * sizeof(float)));
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
                               dst->size * sizeof(float),
                               hipMemcpyHostToDevice));
}

static inline void chelis_device_to_host(chelis_tensor *dst, const chelis_gpu_tensor *src) {
    CHELIS_HIP_CHECK(hipMemcpy(dst->data, src->data,
                               dst->size * sizeof(float),
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
        src->size * sizeof(float),
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

/* ---- JIT compilation ---- */

static inline hipModule_t chelis_compile_kernel(const char *source, const char *name) {
    hiprtcProgram prog;
    CHELIS_HIPRTC_CHECK(hiprtcCreateProgram(&prog, source, name, 0, NULL, NULL));
#ifndef NDEBUG
    const char *compile_opts[] = { "-DCHELIS_DEBUG_BOUNDS=1" };
#else
    const char *compile_opts[] = { "-DCHELIS_DEBUG_BOUNDS=0" };
#endif
    hiprtcResult compile_result = hiprtcCompileProgram(prog, 1, compile_opts);
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
