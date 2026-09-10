#ifndef CHELIS_DEVICE_OWNER_H
#define CHELIS_DEVICE_OWNER_H

#include "chelis_runtime.h"
#include "chelis_device_descriptor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct chelis_device_tensor_owner chelis_device_tensor_owner;

/* [05-OP-33]. Plans transfer to alloc/borrow; packet import borrows storage.
 * Observations remain valid only while their opaque owner lives. The caller
 * retains borrowed storage and the artifact library until its last use. */
chelis_device_tensor_owner *chelis_device_tensor_alloc(chelis_metadata_plan *plan);
chelis_device_tensor_owner *chelis_device_tensor_borrow(chelis_metadata_plan *plan, void *data, chelis_scalar byte_capacity);
chelis_device_tensor_owner *chelis_device_tensor_import(const chelis_gpu_tensor *packet);
const chelis_gpu_tensor *chelis_device_tensor_view(const chelis_device_tensor_owner *owner);
chelis_device_tensor_owner *chelis_device_tensor_clone(const chelis_device_tensor_owner *source);
void chelis_device_tensor_release(chelis_device_tensor_owner *owner);
void chelis_device_tensor_copy_from_host(chelis_device_tensor_owner *destination, const chelis_tensor *source);
void chelis_device_tensor_copy_to_host(chelis_tensor_write *destination, const chelis_device_tensor_owner *source);

#ifdef __cplusplus
}
#endif
#endif
