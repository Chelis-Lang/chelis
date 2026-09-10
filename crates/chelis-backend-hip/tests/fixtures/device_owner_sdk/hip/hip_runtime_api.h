/* Explicit CPU execution fixture, not an installed SDK or hardware receipt. */
#ifndef CHELIS_DEVICE_OWNER_FIXTURE_HIP_API_H
#define CHELIS_DEVICE_OWNER_FIXTURE_HIP_API_H
#include <stddef.h>
typedef enum { hipSuccess = 0, hipErrorInvalidValue = 1 } hipError_t;
typedef enum { hipMemcpyHostToHost, hipMemcpyHostToDevice, hipMemcpyDeviceToHost, hipMemcpyDeviceToDevice, hipMemcpyDefault } hipMemcpyKind;
#ifdef __cplusplus
extern "C" {
#endif
hipError_t hipMalloc(void **pointer, size_t bytes);
hipError_t hipFree(void *pointer);
hipError_t hipMemset(void *pointer, int value, size_t bytes);
hipError_t hipMemcpy(void *destination, const void *source, size_t bytes, hipMemcpyKind kind);
const char *hipGetErrorString(hipError_t error);
#ifdef __cplusplus
}
#endif
#endif
