#ifndef STUB_HIP_RUNTIME_H
#define STUB_HIP_RUNTIME_H
#include <stddef.h>
typedef enum { hipSuccess = 0, hipErrorUnknown = 999 } hipError_t;
typedef enum { hipMemcpyHostToHost = 0, hipMemcpyHostToDevice = 1, hipMemcpyDeviceToHost = 2, hipMemcpyDeviceToDevice = 3 } hipMemcpyKind;
typedef struct chelis_stub_hipModule *hipModule_t;
typedef struct chelis_stub_hipFunction *hipFunction_t;
typedef void *hipDeviceptr_t;
typedef struct { unsigned int x, y, z; } dim3;
typedef struct { int warpSize; } hipDeviceProp_t;
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
