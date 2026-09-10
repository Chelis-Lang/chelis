/* CPU execution fixture only. This is not an installed AMD SDK. */
#ifndef CHELIS_ENTRY_FIXTURE_HIP_API_H
#define CHELIS_ENTRY_FIXTURE_HIP_API_H
#include <stddef.h>
#include <stdint.h>
typedef enum { hipSuccess = 0, hipErrorInvalidValue = 1 } hipError_t;
typedef enum { hipMemcpyHostToHost, hipMemcpyHostToDevice, hipMemcpyDeviceToHost,
               hipMemcpyDeviceToDevice, hipMemcpyDefault } hipMemcpyKind;
typedef void *hipModule_t;
typedef void *hipFunction_t;
typedef uintptr_t hipDeviceptr_t;
typedef struct {
    int warpSize;
    int maxGridSize[3];
    int maxThreadsDim[3];
    int maxThreadsPerBlock;
} hipDeviceProp_t;
#ifdef __cplusplus
extern "C" {
#endif
hipError_t hipMalloc(void **pointer, size_t bytes);
hipError_t hipFree(void *pointer);
hipError_t hipMemset(void *pointer, int value, size_t bytes);
hipError_t hipMemcpy(void *destination, const void *source, size_t bytes, hipMemcpyKind kind);
const char *hipGetErrorString(hipError_t error);
hipError_t hipGetDevice(int *device);
hipError_t hipGetDeviceProperties(hipDeviceProp_t *properties, int device);
hipError_t hipModuleUnload(hipModule_t module);
hipError_t hipModuleLoadData(hipModule_t *module, const void *code);
hipError_t hipModuleGetFunction(hipFunction_t *function, hipModule_t module, const char *name);
hipError_t hipModuleLaunchKernel(hipFunction_t function, unsigned int gx, unsigned int gy,
    unsigned int gz, unsigned int bx, unsigned int by, unsigned int bz,
    unsigned int shared, void *stream, void **arguments, void **extra);
hipError_t hipDeviceSynchronize(void);
#ifdef __cplusplus
}
#endif
#endif
