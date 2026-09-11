/* Explicit CPU execution fixture, not an installed SDK or hardware receipt. */
#ifndef CHELIS_DEVICE_OWNER_FIXTURE_HIP_API_H
#define CHELIS_DEVICE_OWNER_FIXTURE_HIP_API_H
#include <stddef.h>
typedef enum { hipSuccess = 0, hipErrorInvalidValue = 1 } hipError_t;
typedef enum { hipMemcpyHostToHost, hipMemcpyHostToDevice, hipMemcpyDeviceToHost, hipMemcpyDeviceToDevice, hipMemcpyDefault } hipMemcpyKind;
#ifdef __cplusplus
extern "C" {
#endif
typedef enum { hipMemoryTypeHost = 1, hipMemoryTypeDevice = 2 } hipMemoryType;
typedef struct { hipMemoryType type; int device; void *devicePointer; void *hostPointer; int isManaged; unsigned allocationFlags; } hipPointerAttribute_t;
hipError_t hipGetDevice(int *device);
hipError_t hipSetDevice(int device);
hipError_t hipDeviceSynchronize(void);
hipError_t hipPointerGetAttributes(hipPointerAttribute_t *attributes, const void *pointer);
hipError_t hipMalloc(void **pointer, size_t bytes);
hipError_t hipFree(void *pointer);
hipError_t hipMemset(void *pointer, int value, size_t bytes);
hipError_t hipMemcpy(void *destination, const void *source, size_t bytes, hipMemcpyKind kind);
const char *hipGetErrorString(hipError_t error);
#ifdef __cplusplus
}
#endif
#endif
