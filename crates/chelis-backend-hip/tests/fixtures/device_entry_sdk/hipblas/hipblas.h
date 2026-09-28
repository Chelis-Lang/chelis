/* Explicit CPU fixture declarations, never SDK compatibility evidence. */
#ifndef CHELIS_ENTRY_FIXTURE_HIPBLAS_H
#define CHELIS_ENTRY_FIXTURE_HIPBLAS_H
#include <stdint.h>
#define hipblasVersionMajor 3
typedef void *hipblasHandle_t;
typedef enum { HIPBLAS_STATUS_SUCCESS = 0 } hipblasStatus_t;
typedef enum { HIPBLAS_OP_N = 111, HIPBLAS_OP_T = 112, HIPBLAS_OP_C = 113 } hipblasOperation_t;
typedef enum { HIP_R_16F = 2, HIP_R_16BF = 14 } hipDataType;
typedef enum { HIPBLAS_COMPUTE_32F = 68 } hipblasComputeType_t;
typedef enum { HIPBLAS_GEMM_DEFAULT = 160 } hipblasGemmAlgo_t;
extern "C" {
hipblasStatus_t hipblasCreate(hipblasHandle_t *);
hipblasStatus_t hipblasDestroy(hipblasHandle_t);
hipblasStatus_t hipblasSgemm_64(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t,
    int64_t, int64_t, int64_t, const float *, const float *, int64_t,
    const float *, int64_t, const float *, float *, int64_t);
hipblasStatus_t hipblasDgemm_64(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t,
    int64_t, int64_t, int64_t, const double *, const double *, int64_t,
    const double *, int64_t, const double *, double *, int64_t);
hipblasStatus_t hipblasSgemmStridedBatched_64(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t,
    int64_t, int64_t, int64_t, const float *, const float *, int64_t, int64_t,
    const float *, int64_t, int64_t, const float *, float *, int64_t, int64_t, int64_t);
hipblasStatus_t hipblasDgemmStridedBatched_64(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t,
    int64_t, int64_t, int64_t, const double *, const double *, int64_t, int64_t,
    const double *, int64_t, int64_t, const double *, double *, int64_t, int64_t, int64_t);
hipblasStatus_t hipblasGemmEx_64(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t,
    int64_t, int64_t, int64_t, const void *, const void *, hipDataType, int64_t,
    const void *, hipDataType, int64_t, const void *, void *, hipDataType, int64_t,
    hipblasComputeType_t, hipblasGemmAlgo_t);
}
#endif
