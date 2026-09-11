#ifndef CHELIS_HIP_RUNTIME_H
#define CHELIS_HIP_RUNTIME_H

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "chelis_runtime_dtype.h"
#include <hipblas/hipblas.h>
#if !defined(hipblasVersionMajor) || hipblasVersionMajor < 3
#error "Chelis requires matching official hipBLAS 3+ headers and libraries"
#endif
#include "chelis_device_owner.h"

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

/* ---- hipBLAS helpers ---- */

static inline void chelis_hipblas_sgemm_row_major(
    const chelis_gpu_tensor *a,
    const chelis_gpu_tensor *b,
    const chelis_gpu_tensor *out,
    int64_t m,
    int64_t n,
    int64_t k
) {
    hipblasHandle_t handle;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHELIS_HIPBLAS_CHECK(hipblasCreate(&handle));
    /* hipBLAS is column-major by default. Swap A/B and m/n to preserve row-major semantics. */
    CHELIS_HIPBLAS_CHECK(hipblasSgemm_64(
        handle,
        HIPBLAS_OP_N,
        HIPBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        (const float *)b->data,
        n,
        (const float *)a->data,
        k,
        &beta,
        (float *)out->data,
        n
    ));
    CHELIS_HIPBLAS_CHECK(hipblasDestroy(handle));
}

static inline int64_t chelis_gpu_matrix_slices_contiguous(
    const chelis_gpu_tensor *t,
    int64_t trailing_cols
) {
    if (t->rank < 2) return 0;
    return t->strides[t->rank - 1] == 1
        && t->strides[t->rank - 2] == trailing_cols;
}

/* Reduced-precision storage with the explicitly selected f32 accumulator.
 * The official hipBLAS API distinguishes hipDataType storage from
 * hipblasComputeType_t computation; no local declaration fallback exists. */
static inline void chelis_hipblas_bf16_gemm_f32_acc_row_major(
    const chelis_gpu_tensor *a,
    const chelis_gpu_tensor *b,
    const chelis_gpu_tensor *out,
    int64_t m,
    int64_t n,
    int64_t k
) {
    hipblasHandle_t handle;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHELIS_HIPBLAS_CHECK(hipblasCreate(&handle));
    CHELIS_HIPBLAS_CHECK(hipblasGemmEx_64(
        handle,
        HIPBLAS_OP_N,
        HIPBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        b->data,
        HIP_R_16BF,
        n,
        a->data,
        HIP_R_16BF,
        k,
        &beta,
        out->data,
        HIP_R_16BF,
        n,
        HIPBLAS_COMPUTE_32F,
        HIPBLAS_GEMM_DEFAULT
    ));
    CHELIS_HIPBLAS_CHECK(hipblasDestroy(handle));
}

/* WS-A3: f16 matmul with f32 accumulator per spec §5.7.1. Same shape
 * as the bf16 wrapper above; differs only in the operand/result data
 * type (HIP_R_16F). */
static inline void chelis_hipblas_f16_gemm_f32_acc_row_major(
    const chelis_gpu_tensor *a,
    const chelis_gpu_tensor *b,
    const chelis_gpu_tensor *out,
    int64_t m,
    int64_t n,
    int64_t k
) {
    hipblasHandle_t handle;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHELIS_HIPBLAS_CHECK(hipblasCreate(&handle));
    CHELIS_HIPBLAS_CHECK(hipblasGemmEx_64(
        handle,
        HIPBLAS_OP_N,
        HIPBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        b->data,
        HIP_R_16F,
        n,
        a->data,
        HIP_R_16F,
        k,
        &beta,
        out->data,
        HIP_R_16F,
        n,
        HIPBLAS_COMPUTE_32F,
        HIPBLAS_GEMM_DEFAULT
    ));
    CHELIS_HIPBLAS_CHECK(hipblasDestroy(handle));
}

static inline void chelis_hipblas_sgemm_batched_row_major(
    const chelis_gpu_tensor *a,
    const chelis_gpu_tensor *b,
    const chelis_gpu_tensor *out,
    int64_t m,
    int64_t n,
    int64_t k
) {
    if (!chelis_gpu_matrix_slices_contiguous(a, k)
        || !chelis_gpu_matrix_slices_contiguous(b, n)
        || !chelis_gpu_matrix_slices_contiguous(out, n)) {
        fprintf(stderr, "chelis_hipblas_sgemm_batched_row_major: matrix slices must be contiguous\n");
        abort();
    }

    int64_t batch_ndim = out->rank - 2;
    if (out->count == 0) return;
    const int64_t batch_count = out->count / m / n;

    hipblasHandle_t handle;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHELIS_HIPBLAS_CHECK(hipblasCreate(&handle));
    for (int64_t batch = 0; batch < batch_count; batch++) {
        int64_t rem = batch;
        int64_t a_offset = 0;
        int64_t b_offset = 0;
        int64_t out_offset = 0;
        for (int64_t d = batch_ndim - 1; d >= 0; d--) {
            int64_t coord = rem % out->shape[d];
            rem /= out->shape[d];
            a_offset += coord * a->strides[d];
            b_offset += coord * b->strides[d];
            out_offset += coord * out->strides[d];
        }
        CHELIS_HIPBLAS_CHECK(hipblasSgemm_64(
            handle,
            HIPBLAS_OP_N,
            HIPBLAS_OP_N,
            n,
            m,
            k,
            &alpha,
            (const float *)b->data + b_offset,
            n,
            (const float *)a->data + a_offset,
            k,
            &beta,
            (float *)out->data + out_offset,
            n
        ));
    }
    CHELIS_HIPBLAS_CHECK(hipblasDestroy(handle));
}

static inline void chelis_hipblas_sgemm_strided_batched_row_major(
    const chelis_gpu_tensor *a,
    const chelis_gpu_tensor *b,
    const chelis_gpu_tensor *out,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t batch_count,
    int64_t a_batch_stride,
    int64_t b_batch_stride,
    int64_t out_batch_stride
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
    CHELIS_HIPBLAS_CHECK(hipblasSgemmStridedBatched_64(
        handle,
        HIPBLAS_OP_N,
        HIPBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        (const float *)b->data,
        n,
        b_batch_stride,
        (const float *)a->data,
        k,
        a_batch_stride,
        &beta,
        (float *)out->data,
        n,
        out_batch_stride,
        batch_count
    ));
    CHELIS_HIPBLAS_CHECK(hipblasDestroy(handle));
}

/* ---- f64 hipBLAS helpers (WS-A2) ---- */

static inline void chelis_hipblas_dgemm_row_major(
    const chelis_gpu_tensor *a,
    const chelis_gpu_tensor *b,
    const chelis_gpu_tensor *out,
    int64_t m,
    int64_t n,
    int64_t k
) {
    hipblasHandle_t handle;
    const double alpha = 1.0;
    const double beta = 0.0;
    CHELIS_HIPBLAS_CHECK(hipblasCreate(&handle));
    CHELIS_HIPBLAS_CHECK(hipblasDgemm_64(
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
    const chelis_gpu_tensor *out,
    int64_t m,
    int64_t n,
    int64_t k
) {
    if (!chelis_gpu_matrix_slices_contiguous(a, k)
        || !chelis_gpu_matrix_slices_contiguous(b, n)
        || !chelis_gpu_matrix_slices_contiguous(out, n)) {
        fprintf(stderr, "chelis_hipblas_dgemm_batched_row_major: matrix slices must be contiguous\n");
        abort();
    }

    int64_t batch_ndim = out->rank - 2;
    if (out->count == 0) return;
    const int64_t batch_count = out->count / m / n;

    hipblasHandle_t handle;
    const double alpha = 1.0;
    const double beta = 0.0;
    const double *a_base = (const double *)(const void *)a->data;
    const double *b_base = (const double *)(const void *)b->data;
    double *out_base = (double *)(void *)out->data;
    CHELIS_HIPBLAS_CHECK(hipblasCreate(&handle));
    for (int64_t batch = 0; batch < batch_count; batch++) {
        int64_t rem = batch;
        int64_t a_offset = 0;
        int64_t b_offset = 0;
        int64_t out_offset = 0;
        for (int64_t d = batch_ndim - 1; d >= 0; d--) {
            int64_t coord = rem % out->shape[d];
            rem /= out->shape[d];
            a_offset += (int64_t)coord * a->strides[d];
            b_offset += (int64_t)coord * b->strides[d];
            out_offset += (int64_t)coord * out->strides[d];
        }
        CHELIS_HIPBLAS_CHECK(hipblasDgemm_64(
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
    const chelis_gpu_tensor *out,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t batch_count,
    int64_t a_batch_stride,
    int64_t b_batch_stride,
    int64_t out_batch_stride
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
    CHELIS_HIPBLAS_CHECK(hipblasDgemmStridedBatched_64(
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
    int64_t grid_count, int64_t block_count, void **args
) {
    int device = 0;
    hipDeviceProp_t properties;
    CHELIS_HIP_CHECK(hipGetDevice(&device));
    CHELIS_HIP_CHECK(hipGetDeviceProperties(&properties, device));
    if (grid_count < 0 || block_count <= 0) {
        chelis_numeric_trap("numeric trap: domain in launch at int64");
    }
    if (grid_count > properties.maxGridSize[0]
        || block_count > properties.maxThreadsDim[0]
        || block_count > properties.maxThreadsPerBlock) {
        chelis_numeric_trap("numeric trap: overflow in launch at int64");
    }
    if (grid_count == 0) return;
    const dim3 grid((unsigned int)grid_count);
    const dim3 block((unsigned int)block_count);
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
