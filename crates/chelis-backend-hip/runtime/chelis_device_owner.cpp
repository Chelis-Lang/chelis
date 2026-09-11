// [05-OP-33]: private device ownership, linked as a separate artifact input.
// The shared Rust metadata plan is the only count/stride/byte arithmetic owner.
#include "chelis_device_owner.h"
#include <hip/hip_runtime_api.h>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace {
[[noreturn]] void domain() {
    chelis_numeric_trap("numeric trap: domain in metadata_plan at int64");
    std::abort();
}
[[noreturn]] void overflow() {
    chelis_numeric_trap("numeric trap: overflow in metadata_plan at int64");
    std::abort();
}
void hip_checked(hipError_t status) {
    if (status != hipSuccess) {
        std::fprintf(stderr, "HIP device owner failure: %s\n", hipGetErrorString(status));
        std::abort();
    }
}
int32_t current_device() {
    int device = -1;
    hip_checked(hipGetDevice(&device));
    if (device < 0) domain();
    return device;
}
class DeviceScope final {
public:
    explicit DeviceScope(int32_t device) : previous_(current_device()) {
        if (previous_ != device) hip_checked(hipSetDevice(device));
    }
    ~DeviceScope() { if (current_device() != previous_) hip_checked(hipSetDevice(previous_)); }
    DeviceScope(const DeviceScope &) = delete;
    DeviceScope &operator=(const DeviceScope &) = delete;
private:
    const int32_t previous_;
};
struct PlanRelease {
    void operator()(chelis_metadata_plan *plan) const {
        if (plan) chelis_metadata_plan_release(plan);
    }
};
using Plan = std::unique_ptr<chelis_metadata_plan, PlanRelease>;
struct DeviceRelease {
    int32_t device = 0;
    void operator()(void *data) const {
        if (data) {
            DeviceScope scope(device);
            hip_checked(hipFree(data));
        }
    }
};
using Allocation = std::unique_ptr<void, DeviceRelease>;
chelis_scalar integer(int64_t value) {
    return chelis_scalar_from_bits(CHELIS_DTYPE_I64, static_cast<uint64_t>(value));
}
chelis_scalar exemplar(chelis_dtype dtype) {
    // Untrusted tagged request, checked by the Rust plan constructor. Calling
    // a dtype-specific constructor here would select the wrong trap operation.
    return {dtype, {0, 0, 0, 0, 0, 0, 0}, 0};
}
void array_preflight(int32_t rank, const int64_t *values) {
    if (rank < 0) domain();
    if (rank == 0) return;
    if (!values || reinterpret_cast<uintptr_t>(values) % alignof(int64_t)) domain();
    // Check foreign-array and adapter-container projections before any read.
    // Descriptor products, strides, and payload bytes remain Rust-owned.
    auto length = static_cast<size_t>(rank);
    if (length > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) / sizeof(int64_t)
        || length > std::vector<chelis_scalar>().max_size()) overflow();
}
std::vector<chelis_scalar> tagged_array(int32_t rank, const int64_t *values) {
    array_preflight(rank, values);
    std::vector<chelis_scalar> result;
    result.reserve(static_cast<size_t>(rank));
    for (int32_t axis = 0; axis < rank; ++axis) result.push_back(integer(values[axis]));
    return result;
}
Plan canonical_plan(const chelis_metadata_plan *source) {
    auto rank = chelis_metadata_plan_rank(source);
    auto shape = tagged_array(rank, chelis_metadata_plan_shape(source));
    return Plan(chelis_metadata_plan_new(integer(rank), shape.data(), exemplar(chelis_metadata_plan_dtype(source))));
}
bool canonical_strides(const chelis_metadata_plan *plan) {
    auto canonical = canonical_plan(plan);
    auto rank = chelis_metadata_plan_rank(plan);
    auto left = chelis_metadata_plan_strides(plan);
    auto right = chelis_metadata_plan_strides(canonical.get());
    for (int32_t axis = 0; axis < rank; ++axis) if (left[axis] != right[axis]) return false;
    return true;
}
void validate_data(const chelis_metadata_plan *plan, const void *data) {
    auto count = chelis_metadata_plan_count(plan);
    if (!count) return;
    // Both observations come from one checked plan. No dtype width table.
    auto width = chelis_metadata_plan_byte_count(plan) / count;
    if (!data || reinterpret_cast<uintptr_t>(data) % static_cast<uintptr_t>(width)) domain();
    hipPointerAttribute_t attributes{};
    if (hipPointerGetAttributes(&attributes, data) != hipSuccess
        || attributes.type != hipMemoryTypeDevice || attributes.device != current_device()) domain();
}
template<class Operation> auto allocate_metadata(Operation operation) -> decltype(operation()) {
    // No C++ exception may unwind across the C ABI into a Rust caller.
    try { return operation(); }
    catch (const std::exception &error) {
        std::fprintf(stderr, "device metadata allocation failed: %s\n", error.what());
        std::abort();
    }
}
}

// This definition is deliberately absent from every published header. Its
// immutable packet observes the same retained plan and actual storage owner.
struct chelis_device_tensor_owner final {
    // Empty owned storage also has a null allocation; storage disposition must
    // therefore be explicit and must never be inferred from pointer nullness.
    chelis_device_tensor_owner(Plan plan, Allocation allocation, bool borrowed_storage,
                               void *data, int64_t capacity, int32_t device)
        : plan_(std::move(plan)), allocation_(std::move(allocation)),
          borrowed_storage_(borrowed_storage), device_(device),
          packet_{data, chelis_metadata_plan_shape(plan_.get()), chelis_metadata_plan_strides(plan_.get()),
                  chelis_metadata_plan_count(plan_.get()), capacity,
                  chelis_metadata_plan_rank(plan_.get()), chelis_metadata_plan_dtype(plan_.get()),
                  static_cast<uint8_t>(borrowed_storage_ ? 0 : 1), {0, 0}} {}
    chelis_device_tensor_owner(const chelis_device_tensor_owner &) = delete;
    chelis_device_tensor_owner &operator=(const chelis_device_tensor_owner &) = delete;
    const chelis_gpu_tensor *view() const { return &packet_; }
    const chelis_metadata_plan *plan() const { return plan_.get(); }
    bool owns_storage() const { return !borrowed_storage_; }
    int32_t device() const { return device_; }
    void require_current() const { if (current_device() != device_) domain(); }
private:
    Plan plan_;
    Allocation allocation_;
    const bool borrowed_storage_;
    const int32_t device_;
    const chelis_gpu_tensor packet_;
};

namespace {
const chelis_device_tensor_owner &owner(const chelis_device_tensor_owner *value) {
    if (!value || reinterpret_cast<uintptr_t>(value) % alignof(chelis_device_tensor_owner)) domain();
    return *value;
}
chelis_device_tensor_owner *borrow_plan(Plan plan, void *data, chelis_scalar capacity) {
    chelis_metadata_plan_check_capacity(plan.get(), capacity);
    validate_data(plan.get(), data);
    // check_capacity has validated the exact canonical int64 carrier.
    auto bytes = static_cast<int64_t>(capacity.bits);
    return new chelis_device_tensor_owner(std::move(plan), Allocation{}, true, data, bytes, current_device());
}
chelis_device_tensor_owner *allocate_plan(Plan plan) {
    auto bytes = chelis_metadata_plan_byte_count(plan.get());
    chelis_metadata_plan_check_capacity(plan.get(), integer(bytes));
    auto device = current_device();
    void *data = nullptr;
    if (bytes) hip_checked(hipMalloc(&data, static_cast<size_t>(bytes)));
    Allocation allocation(data, DeviceRelease{device});
    if (bytes) {
        hip_checked(hipMemset(data, 0, static_cast<size_t>(bytes)));
        hip_checked(hipDeviceSynchronize());
    }
    return new chelis_device_tensor_owner(std::move(plan), std::move(allocation), false, data, bytes, device);
}
void require_matching(const chelis_read_view &host, const chelis_gpu_tensor &device) {
    if (host.count != device.count || host.dtype != device.dtype) domain();
    for (auto byte : host.reserved) if (byte) domain();
}
void require_matching(const chelis_write_view &host, const chelis_gpu_tensor &device) {
    if (host.count != device.count || host.dtype != device.dtype) domain();
    for (auto byte : host.reserved) if (byte) domain();
}
}

extern "C" chelis_device_tensor_owner *chelis_device_tensor_alloc(chelis_metadata_plan *plan) {
    return allocate_metadata([&] { return allocate_plan(Plan(plan)); });
}
extern "C" chelis_device_tensor_owner *chelis_device_tensor_borrow(
    chelis_metadata_plan *plan, void *data, chelis_scalar byte_capacity) {
    return allocate_metadata([&] { return borrow_plan(Plan(plan), data, byte_capacity); });
}
extern "C" chelis_device_tensor_owner *chelis_device_tensor_import(const chelis_gpu_tensor *packet) {
    return allocate_metadata([&] {
        if (!packet || reinterpret_cast<uintptr_t>(packet) % alignof(chelis_gpu_tensor)) domain();
        if (packet->ownership != 0 || packet->reserved[0] || packet->reserved[1]) domain();
        // Preflight both foreign arrays before either is read or converted.
        array_preflight(packet->rank, packet->shape);
        array_preflight(packet->rank, packet->strides);
        auto shape = tagged_array(packet->rank, packet->shape);
        auto strides = tagged_array(packet->rank, packet->strides);
        Plan plan(chelis_metadata_plan_view(integer(packet->rank), shape.data(), strides.data(),
                                           exemplar(packet->dtype), integer(packet->byte_capacity)));
        if (packet->count != chelis_metadata_plan_count(plan.get())) domain();
        return borrow_plan(std::move(plan), packet->data, integer(packet->byte_capacity));
    });
}
extern "C" const chelis_gpu_tensor *chelis_device_tensor_view(const chelis_device_tensor_owner *value) {
    owner(value).require_current();
    return owner(value).view();
}
extern "C" int32_t chelis_device_tensor_device(const chelis_device_tensor_owner *value) {
    return owner(value).device();
}
extern "C" chelis_device_tensor_owner *chelis_device_tensor_clone(const chelis_device_tensor_owner *source) {
    return allocate_metadata([&] {
        auto &input = owner(source);
        input.require_current();
        // Canonical output representability is checked before device allocation,
        // even when an admitted source view is empty with huge supplied strides.
        std::unique_ptr<chelis_device_tensor_owner> result(allocate_plan(canonical_plan(input.plan())));
        auto count = input.view()->count;
        if (count) {
            auto width = chelis_metadata_plan_byte_count(input.plan()) / count;
            for (int64_t index = 0; index < count; ++index) {
                auto from = chelis_metadata_plan_byte_offset(input.plan(), integer(index));
                auto to = chelis_metadata_plan_byte_offset(result->plan(), integer(index));
                hip_checked(hipMemcpy(static_cast<unsigned char *>(result->view()->data) + to,
                                      static_cast<const unsigned char *>(input.view()->data) + from,
                                      static_cast<size_t>(width), hipMemcpyDeviceToDevice));
            }
        }
        if (count) hip_checked(hipDeviceSynchronize());
        return result.release();
    });
}
extern "C" void chelis_device_tensor_release(chelis_device_tensor_owner *value) {
    owner(value);
    delete value;
}
extern "C" void chelis_device_tensor_copy_from_host(chelis_device_tensor_owner *destination,
                                                     const chelis_tensor *source) {
    allocate_metadata([&] {
        auto &output = owner(destination);
        output.require_current();
        auto input = chelis_tensor_read_view(source);
        require_matching(input, *output.view());
        if (!output.owns_storage() || !canonical_strides(output.plan())) domain();
        auto bytes = chelis_metadata_plan_byte_count(output.plan());
        if (bytes) {
            hip_checked(hipMemcpy(output.view()->data, input.data, static_cast<size_t>(bytes), hipMemcpyHostToDevice));
            hip_checked(hipDeviceSynchronize());
        }
    });
}
extern "C" void chelis_device_tensor_copy_to_host(chelis_tensor_write *destination,
                                                   const chelis_device_tensor_owner *source) {
    auto &input = owner(source);
    input.require_current();
    auto output = chelis_tensor_write_view(destination);
    require_matching(output, *input.view());
    auto count = input.view()->count;
    if (!count) return;
    auto width = chelis_metadata_plan_byte_count(input.plan()) / count;
    auto *to = static_cast<unsigned char *>(output.data);
    for (int64_t index = 0; index < count; ++index) {
        auto from = chelis_metadata_plan_byte_offset(input.plan(), integer(index));
        // The host write guard is contiguous and its equal count/dtype proves
        // exactly logical-byte-count bytes; advancing one checked width stays
        // inside that live view, including its final one-past pointer.
        hip_checked(hipMemcpy(to, static_cast<const unsigned char *>(input.view()->data) + from,
                              static_cast<size_t>(width), hipMemcpyDeviceToHost));
        to += width;
    }
    hip_checked(hipDeviceSynchronize());
}
