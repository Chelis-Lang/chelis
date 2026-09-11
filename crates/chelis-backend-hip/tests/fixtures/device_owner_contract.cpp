// [05-OP-33], spec/11-ffi §2.1. Executes the actual separate owner implementation
// and actual Rust metadata archive; only HIP allocation/copy calls are simulated.
#include "chelis_device_owner.h"
#include <hip/hip_runtime_api.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>
#include <thread>

static std::map<uintptr_t, size_t> allocations;
static std::map<uintptr_t, int> allocation_devices;
static thread_local int current_device = 0;
struct PendingCopy { void *destination; const void *source; size_t bytes; int device; };
static std::vector<PendingCopy> pending_copies;
static size_t releases = 0;
static size_t plan_releases = 0;
// Only the companion translation unit redirects this call for observation;
// this wrapper always invokes the actual runtime plan finalizer.
extern "C" void fixture_metadata_plan_release(chelis_metadata_plan *plan) {
    ++plan_releases;
    chelis_metadata_plan_release(plan);
}
static void require_device(const void *pointer, size_t bytes) {
    if (!bytes) return;
    auto position = allocations.upper_bound(reinterpret_cast<uintptr_t>(pointer));
    assert(position != allocations.begin());
    --position;
    size_t offset = reinterpret_cast<uintptr_t>(pointer) - position->first;
    assert(offset <= position->second && bytes <= position->second - offset);
    assert(allocation_devices.at(position->first) == current_device);
}
extern "C" hipError_t hipGetDevice(int *device) { *device = current_device; return hipSuccess; }
extern "C" hipError_t hipSetDevice(int device) { assert(device >= 0 && device < 3); current_device = device; return hipSuccess; }
extern "C" hipError_t hipPointerGetAttributes(hipPointerAttribute_t *attributes, const void *pointer) {
    auto position = allocations.upper_bound(reinterpret_cast<uintptr_t>(pointer));
    if (position == allocations.begin()) return hipErrorInvalidValue;
    --position;
    auto offset = reinterpret_cast<uintptr_t>(pointer) - position->first;
    if (offset >= position->second) return hipErrorInvalidValue;
    *attributes = {hipMemoryTypeDevice, allocation_devices.at(position->first), const_cast<void *>(pointer), nullptr, 0, 0};
    return hipSuccess;
}
extern "C" hipError_t hipDeviceSynchronize() {
    for (auto copy = pending_copies.begin(); copy != pending_copies.end();) {
        if (copy->device != current_device) { ++copy; continue; }
        require_device(copy->source, copy->bytes);
        require_device(copy->destination, copy->bytes);
        std::memcpy(copy->destination, copy->source, copy->bytes);
        copy = pending_copies.erase(copy);
    }
    return hipSuccess;
}
extern "C" hipError_t hipMalloc(void **pointer, size_t bytes) {
    assert(bytes > 0); // empty owners must not allocate a synthetic element
    *pointer = std::malloc(bytes);
    assert(*pointer);
    assert(allocations.emplace(reinterpret_cast<uintptr_t>(*pointer), bytes).second);
    allocation_devices.emplace(reinterpret_cast<uintptr_t>(*pointer), current_device);
    return hipSuccess;
}
extern "C" hipError_t hipFree(void *pointer) {
    if (!pointer) return hipSuccess;
    require_device(pointer, 1);
    assert(pending_copies.empty()); // no unfinished copy may retain released storage
    allocation_devices.erase(reinterpret_cast<uintptr_t>(pointer));
    assert(allocations.erase(reinterpret_cast<uintptr_t>(pointer)) == 1);
    ++releases;
    std::free(pointer);
    return hipSuccess;
}
extern "C" hipError_t hipMemset(void *pointer, int value, size_t bytes) {
    require_device(pointer, bytes);
    if (bytes) std::memset(pointer, value, bytes);
    return hipSuccess;
}
extern "C" hipError_t hipMemcpy(void *destination, const void *source, size_t bytes, hipMemcpyKind kind) {
    if (kind == hipMemcpyHostToDevice || kind == hipMemcpyDeviceToDevice) require_device(destination, bytes);
    if (kind == hipMemcpyDeviceToHost || kind == hipMemcpyDeviceToDevice) require_device(source, bytes);
    if (bytes && kind == hipMemcpyDeviceToDevice) pending_copies.push_back({destination, source, bytes, current_device});
    else if (bytes) std::memcpy(destination, source, bytes);
    return hipSuccess;
}
extern "C" const char *hipGetErrorString(hipError_t) { return "fixture HIP error"; }
static chelis_scalar integer(int64_t value) { return chelis_scalar_from_bits(CHELIS_DTYPE_I64, static_cast<uint64_t>(value)); }
static chelis_scalar exemplar(chelis_dtype dtype) { return chelis_scalar_from_bits(dtype, 0); }
static chelis_metadata_plan *contiguous(const std::vector<int64_t> &shape, chelis_dtype dtype) {
    std::vector<chelis_scalar> tagged;
    for (auto extent : shape) tagged.push_back(integer(extent));
    return chelis_metadata_plan_new(integer(shape.size()), tagged.data(), exemplar(dtype));
}
static chelis_gpu_tensor packet(void *data, const int64_t *shape, const int64_t *strides,
                                int64_t count, int64_t capacity, int32_t rank, chelis_dtype dtype) {
    return {data, shape, strides, count, capacity, rank, dtype, 0, {0, 0}};
}
static void logical_clone_and_transfer() {
    const chelis_dtype dtypes[] = {CHELIS_DTYPE_F32, CHELIS_DTYPE_F64, CHELIS_DTYPE_I32,
        CHELIS_DTYPE_I64, CHELIS_DTYPE_BOOL, CHELIS_DTYPE_F16, CHELIS_DTYPE_BF16,
        CHELIS_DTYPE_I8, CHELIS_DTYPE_I16};
    for (auto dtype : dtypes) {
        size_t width = chelis_dtype_size(dtype);
        for (int layout = 0; layout < 3; ++layout) {
            int64_t shape[2] = {layout == 0 ? 3 : 2, layout == 1 ? 2 : 3};
            if (layout == 0) shape[1] = 2;
            int64_t strides[2] = {layout == 0 ? 1 : layout == 1 ? 4 : 0, layout == 0 ? 3 : layout == 1 ? 2 : 1};
            std::vector<int64_t> indices = layout == 0 ? std::vector<int64_t>{0,3,1,4,2,5} :
                layout == 1 ? std::vector<int64_t>{0,2,4,6} : std::vector<int64_t>{0,1,2,0,1,2};
            void *data = nullptr;
            hipMalloc(&data, 7 * width);
            // Canonical bool bytes; other dtypes are copied bit-for-bit, including NaNs.
            for (size_t byte = 0; byte < 7 * width; ++byte)
                static_cast<unsigned char *>(data)[byte] = dtype == CHELIS_DTYPE_BOOL ? byte % 2 : (byte * 31 + 7) % 256;
            std::vector<unsigned char> expected;
            for (auto index : indices) {
                auto begin = static_cast<unsigned char *>(data) + index * width;
                expected.insert(expected.end(), begin, begin + width);
            }
            auto raw = packet(data, shape, strides, indices.size(), 7 * width, 2, dtype);
            auto source = chelis_device_tensor_import(&raw);
            auto clone = chelis_device_tensor_clone(source);
            auto copied = chelis_device_tensor_view(clone);
            assert(copied->ownership == 1 && copied->data != data && copied->count == raw.count);
            assert(copied->strides[1] == 1 && copied->strides[0] == shape[1]);
            assert(copied->byte_capacity == static_cast<int64_t>(expected.size()));
            size_t before = releases;
            size_t plans_before = plan_releases;
            chelis_device_tensor_release(source);
            assert(plan_releases == plans_before + 1);
            assert(releases == before); // a borrow never frees its input allocation
            hipFree(data); // escaping clone must survive both metadata and source storage
            auto host = chelis_alloc(2, shape, dtype);
            auto guard = chelis_tensor_begin_write(host);
            chelis_device_tensor_copy_to_host(guard, clone);
            chelis_tensor_end_write(guard);
            assert(std::memcmp(chelis_tensor_read_view(host).data, expected.data(), expected.size()) == 0);
            auto fresh = chelis_device_tensor_alloc(contiguous({shape[0], shape[1]}, dtype));
            chelis_device_tensor_copy_from_host(fresh, host);
            assert(std::memcmp(chelis_device_tensor_view(fresh)->data, expected.data(), expected.size()) == 0);
            before = releases;
            plans_before = plan_releases;
            chelis_device_tensor_release(clone);
            chelis_device_tensor_release(fresh);
            assert(plan_releases == plans_before + 2);
            assert(releases == before + 2);
            chelis_tensor_release(host);
        }
    }
}
static void empty_scalar_and_dynamic_rank() {
    for (int rank : {0, 1, 8, 9, 33}) {
        auto owner = chelis_device_tensor_alloc(contiguous(std::vector<int64_t>(rank, 1), CHELIS_DTYPE_F64));
        auto view = chelis_device_tensor_view(owner);
        assert(view->rank == rank && view->count == 1 && view->byte_capacity == 8);
        assert(rank != 0 || (view->shape == nullptr && view->strides == nullptr));
        chelis_device_tensor_release(owner);
    }
    auto empty = chelis_device_tensor_alloc(contiguous({2,0,3}, CHELIS_DTYPE_F64));
    auto clone = chelis_device_tensor_clone(empty);
    assert(chelis_device_tensor_view(clone)->count == 0);
    assert(chelis_device_tensor_view(clone)->byte_capacity == 0);
    assert(allocations.empty());
    chelis_device_tensor_release(empty);
    chelis_device_tensor_release(clone);
}
static void explicit_borrow_retains_metadata_only() {
    void *data = nullptr;
    hipMalloc(&data, 48);
    chelis_scalar shape[] = {integer(2), integer(3)};
    chelis_scalar strides[] = {integer(3), integer(1)};
    auto plan = chelis_metadata_plan_view(integer(2), shape, strides, exemplar(CHELIS_DTYPE_F64), integer(48));
    auto owner = chelis_device_tensor_borrow(plan, data, integer(48));
    shape[0] = integer(99);
    strides[0] = integer(99);
    auto observed = chelis_device_tensor_view(owner);
    assert(observed->shape[0] == 2 && observed->strides[0] == 3 && observed->ownership == 0);
    assert(observed->data == data && observed->byte_capacity == 48);
    size_t before = releases;
    size_t plans_before = plan_releases;
    chelis_device_tensor_release(owner);
    assert(plan_releases == plans_before + 1);
    assert(releases == before);
    hipFree(data);
}
static void actual_device_and_foreign_thread_finalization() {
    hipSetDevice(1);
    auto value = chelis_device_tensor_alloc(contiguous({2}, CHELIS_DTYPE_F64));
    assert(chelis_device_tensor_device(value) == 1);
    hipSetDevice(2);
    assert(chelis_device_tensor_device(value) == 1); // observation is not inferred from current context
    size_t before = releases;
    std::thread finalizer([value] {
        int previous = -1;
        hipGetDevice(&previous);
        assert(previous == 0);
        chelis_device_tensor_release(value);
        int after = -1;
        hipGetDevice(&after);
        assert(after == previous);
    });
    finalizer.join();
    assert(releases == before + 1);
    int current = -1;
    hipGetDevice(&current);
    assert(current == 2);
    auto empty = chelis_device_tensor_alloc(contiguous({0}, CHELIS_DTYPE_F64));
    assert(chelis_device_tensor_device(empty) == 2);
    hipSetDevice(0);
    chelis_device_tensor_release(empty);
    hipGetDevice(&current);
    assert(current == 0);
}
static void rejected(const char *mode) {
    if (!std::strcmp(mode, "empty-clone-overflow")) {
        chelis_scalar dimensions[] = {integer(0), integer(INT64_MAX), integer(INT64_MAX)};
        chelis_scalar strides[] = {integer(INT64_MAX), integer(INT64_MAX), integer(INT64_MAX)};
        auto plan = chelis_metadata_plan_view(integer(3), dimensions, strides, exemplar(CHELIS_DTYPE_F64), integer(0));
        auto empty = chelis_device_tensor_borrow(plan, nullptr, integer(0));
        chelis_device_tensor_clone(empty);
        std::exit(0);
    }
    int64_t shape[] = {2,3}, strides[] = {3,1};
    void *data = nullptr;
    hipMalloc(&data, 48);
    auto raw = packet(data, shape, strides, 6, 48, 2, CHELIS_DTYPE_F64);
    if (!std::strcmp(mode, "reserved")) raw.reserved[1] = 1;
    if (!std::strcmp(mode, "dtype")) raw.dtype = 255;
    if (!std::strcmp(mode, "owned-import")) raw.ownership = 1;
    if (!std::strcmp(mode, "count")) raw.count = 5;
    if (!std::strcmp(mode, "capacity")) raw.byte_capacity = 47;
    if (!std::strcmp(mode, "negative-stride")) strides[0] = -1;
    if (!std::strcmp(mode, "rank")) raw.rank = -1;
    if (!std::strcmp(mode, "null-shape")) raw.shape = nullptr;
    if (!std::strcmp(mode, "null-data")) raw.data = nullptr;
    if (!std::strcmp(mode, "borrow-capacity")) {
        chelis_device_tensor_borrow(contiguous({2,3}, CHELIS_DTYPE_F64), data, integer(47));
    }
    if (!std::strcmp(mode, "gapped-allocation")) {
        chelis_scalar dimensions[] = {integer(2), integer(2)};
        chelis_scalar gaps[] = {integer(4), integer(2)};
        auto plan = chelis_metadata_plan_view(integer(2), dimensions, gaps, exemplar(CHELIS_DTYPE_F64), integer(56));
        chelis_device_tensor_alloc(plan);
    }
    if (!std::strcmp(mode, "pointer-device")) hipSetDevice(1);
    auto source = chelis_device_tensor_import(&raw);
    if (!std::strcmp(mode, "context-view")) { hipSetDevice(1); chelis_device_tensor_view(source); }
    if (!std::strcmp(mode, "context-clone")) { hipSetDevice(1); chelis_device_tensor_clone(source); }
    if (!std::strcmp(mode, "context-transfer")) {
        auto host = chelis_alloc(2, shape, CHELIS_DTYPE_F64);
        auto guard = chelis_tensor_begin_write(host);
        hipSetDevice(1);
        chelis_device_tensor_copy_to_host(guard, source);
    }
    if (!std::strcmp(mode, "borrow-write")) {
        auto host = chelis_alloc(2, shape, CHELIS_DTYPE_F64);
        chelis_device_tensor_copy_from_host(source, host);
    } else if (!std::strcmp(mode, "dtype-transfer") || !std::strcmp(mode, "count-transfer")) {
        auto host = chelis_alloc(2, shape, CHELIS_DTYPE_F64);
        auto destination = chelis_device_tensor_alloc(contiguous(
            {2, !std::strcmp(mode, "count-transfer") ? 2 : 3},
            !std::strcmp(mode, "dtype-transfer") ? CHELIS_DTYPE_I64 : CHELIS_DTYPE_F64));
        chelis_device_tensor_copy_from_host(destination, host);
    }
    std::fprintf(stderr, "malformed device request returned successfully\n");
    std::exit(0); // parent must reject success, not count a harness assertion as a trap
}
int main(int argc, char **argv) {
    assert(argc == 2);
    if (!std::strcmp(argv[1], "logical-order")) logical_clone_and_transfer();
    else if (!std::strcmp(argv[1], "dynamic-rank")) empty_scalar_and_dynamic_rank();
    else if (!std::strcmp(argv[1], "explicit-borrow")) explicit_borrow_retains_metadata_only();
    else if (!std::strcmp(argv[1], "device-context")) actual_device_and_foreign_thread_finalization();
    else rejected(argv[1]);
    assert(allocations.empty());
    assert(pending_copies.empty());
}
