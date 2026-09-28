#ifndef CHELIS_STUB_HIPBLAS_H
#define CHELIS_STUB_HIPBLAS_H
/* The required hipBLAS 3+ surface chelis_hip_runtime.h consumes. */
#define hipblasVersionMajor 3
typedef enum { HIPBLAS_STATUS_SUCCESS = 0 } hipblasStatus_t;
typedef struct hipblasContext *hipblasHandle_t;
typedef enum { HIPBLAS_OP_N = 111, HIPBLAS_OP_T = 112, HIPBLAS_OP_C = 113 } hipblasOperation_t;
typedef enum { HIPBLAS_R_16F = 150, HIPBLAS_R_16B = 168, HIPBLAS_R_32F = 151, HIPBLAS_R_64F = 152 } hipblasDatatype_t;
typedef enum { HIPBLAS_GEMM_DEFAULT = 160 } hipblasGemmAlgo_t;
typedef enum { HIPBLAS_COMPUTE_32F = 300 } hipblasComputeType_t;
hipblasStatus_t hipblasCreate(hipblasHandle_t *handle);
hipblasStatus_t hipblasDestroy(hipblasHandle_t handle);
hipblasStatus_t hipblasSgemm(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc);
hipblasStatus_t hipblasSgemm_64(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, long long m, long long n, long long k, const float *alpha, const float *A, long long lda, const float *B, long long ldb, const float *beta, float *C, long long ldc);
hipblasStatus_t hipblasSgemmStridedBatched_64(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, long long m, long long n, long long k, const float *alpha, const float *A, long long lda, long long strideA, const float *B, long long ldb, long long strideB, const float *beta, float *C, long long ldc, long long strideC, long long batchCount);
hipblasStatus_t hipblasSgemmStridedBatched(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, long long strideA, const float *B, int ldb, long long strideB, const float *beta, float *C, int ldc, long long strideC, int batchCount);
hipblasStatus_t hipblasDgemm(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc);
hipblasStatus_t hipblasDgemm_64(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, long long m, long long n, long long k, const double *alpha, const double *A, long long lda, const double *B, long long ldb, const double *beta, double *C, long long ldc);
hipblasStatus_t hipblasDgemmStridedBatched(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, long long strideA, const double *B, int ldb, long long strideB, const double *beta, double *C, int ldc, long long strideC, int batchCount);
hipblasStatus_t hipblasDgemmStridedBatched_64(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, long long m, long long n, long long k, const double *alpha, const double *A, long long lda, long long strideA, const double *B, long long ldb, long long strideB, const double *beta, double *C, long long ldc, long long strideC, long long batchCount);
hipblasStatus_t hipblasGemmEx(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n, int k, const void *alpha, const void *A, hipblasDatatype_t aType, int lda, const void *B, hipblasDatatype_t bType, int ldb, const void *beta, void *C, hipblasDatatype_t cType, int ldc, hipblasDatatype_t computeType, hipblasGemmAlgo_t algo);
hipblasStatus_t hipblasGemmEx_64(hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, long long m, long long n, long long k, const void *alpha, const void *A, hipDataType aType, long long lda, const void *B, hipDataType bType, long long ldb, const void *beta, void *C, hipDataType cType, long long ldc, hipblasComputeType_t computeType, hipblasGemmAlgo_t algo);
#endif
