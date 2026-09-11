#ifndef STUB_HIP_RUNTIME_H
#define STUB_HIP_RUNTIME_H
#include <stddef.h>
typedef enum { hipSuccess = 0, hipErrorUnknown = 999 } hipError_t;
typedef enum { hipMemcpyHostToHost = 0, hipMemcpyHostToDevice = 1, hipMemcpyDeviceToHost = 2, hipMemcpyDeviceToDevice = 3 } hipMemcpyKind;
typedef struct chelis_stub_hipModule *hipModule_t;
typedef struct chelis_stub_hipFunction *hipFunction_t;
typedef void *hipDeviceptr_t;
#ifdef __cplusplus
struct dim3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
    constexpr dim3(unsigned int x_value = 1, unsigned int y_value = 1,
                   unsigned int z_value = 1)
        : x(x_value), y(y_value), z(z_value) {}
};
#else
typedef struct { unsigned int x, y, z; } dim3;
#endif
typedef struct {
    int warpSize;
    int maxGridSize[3];
    int maxThreadsDim[3];
    int maxThreadsPerBlock;
} hipDeviceProp_t;
typedef enum {
    HIP_R_16F = 2,
    HIP_R_16BF = 14
} hipDataType;
const char *hipGetErrorString(hipError_t error);
hipError_t hipMalloc(void **pointer, size_t size);
hipError_t hipFree(void *pointer);
hipError_t hipMemcpy(void *dst, const void *src, size_t size, hipMemcpyKind kind);
hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void *src, size_t size);
hipError_t hipMemcpyDtoH(void *dst, hipDeviceptr_t src, size_t size);
hipError_t hipDeviceSynchronize(void);
hipError_t hipGetDevice(int *device);
hipError_t hipGetDeviceProperties(hipDeviceProp_t *props, int device);
hipError_t hipModuleLoadData(hipModule_t *module, const void *image);
hipError_t hipModuleGetFunction(hipFunction_t *function, hipModule_t module, const char *name);
hipError_t hipModuleGetGlobal(hipDeviceptr_t *dptr, size_t *bytes, hipModule_t module, const char *name);
hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned shared, void *stream, void **args, void **extra);
#endif
