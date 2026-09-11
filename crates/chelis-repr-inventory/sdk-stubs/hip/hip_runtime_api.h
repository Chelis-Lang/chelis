#ifndef CHELIS_STUB_HIP_RUNTIME_API_H
#define CHELIS_STUB_HIP_RUNTIME_API_H
#include "hip_runtime.h"
typedef enum {
    hipMemoryTypeHost = 0,
    hipMemoryTypeDevice = 1
} hipMemoryType;
typedef struct {
    hipMemoryType type;
    int device;
} hipPointerAttribute_t;
hipError_t hipSetDevice(int device);
hipError_t hipPointerGetAttributes(hipPointerAttribute_t *attributes, const void *pointer);
hipError_t hipMemset(void *pointer, int value, size_t size);
#endif
