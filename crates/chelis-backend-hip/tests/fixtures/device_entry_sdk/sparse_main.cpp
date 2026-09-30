// Execute exact emitted sparse kernels on supplied noncontiguous input layouts.
#include "chelis_device_owner.h"
#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define REQUIRE(condition) do { if (!(condition)) { \
    fprintf(stderr, "sparse entry fixture violation at line %d: %s\n", __LINE__, #condition); abort(); \
} } while (0)
extern "C" void entry_probe_device(const chelis_device_tensor_owner *const *, int32_t,
    chelis_device_tensor_owner **, int32_t);
extern "C" size_t fixture_live_allocations();
extern "C" size_t fixture_live_modules();

static chelis_device_tensor_owner *input(const void *host, size_t bytes, chelis_dtype dtype,
    const std::vector<int64_t> &shape, const std::vector<int64_t> &strides, int64_t count,
    std::vector<void *> &storage) {
    void *data = nullptr;
    REQUIRE(hipMalloc(&data, bytes) == hipSuccess);
    REQUIRE(hipMemcpy(data, host, bytes, hipMemcpyHostToDevice) == hipSuccess);
    chelis_gpu_tensor packet = {};
    packet.data = data; packet.shape = shape.data(); packet.strides = strides.data();
    packet.rank = (int32_t)shape.size(); packet.count = count;
    packet.byte_capacity = (int64_t)bytes; packet.dtype = dtype;
    storage.push_back(data);
    return chelis_device_tensor_import(&packet);
}

int main(int argc, char **argv) {
    const bool wrong_device = argc > 1 && !strcmp(argv[1], "wrong-second-device");
    const bool wrong_index = argc > 1 && !strcmp(argv[1], "wrong-index");
    std::vector<void *> storage;
    std::vector<chelis_device_tensor_owner *> inputs;
    float target[] = {1, 3, 5, 2, 4, 6};
    inputs.push_back(input(target, sizeof(target), CHELIS_DTYPE_F32, {3, 2}, {1, 3}, 6, storage));
    if (wrong_device) REQUIRE(hipSetDevice(1) == hipSuccess);
#if TEST_INDEX64
    using Index = int64_t;
    const auto index_dtype = CHELIS_DTYPE_I64;
#else
    using Index = int32_t;
    const auto index_dtype = CHELIS_DTYPE_I32;
#endif
#if TEST_SPARSE == 3
    Index indices[] = {2, 2, 0, 1};
    if (wrong_index) indices[0] = 3;
    inputs.push_back(input(indices, sizeof(indices), index_dtype, {2, 2}, {1, 2}, 4, storage));
#else
    Index indices[] = {2, 777, 0, 777, 2, 777, 1};
    if (wrong_index) indices[0] = 3;
    inputs.push_back(input(indices, sizeof(indices), index_dtype, {4}, {2}, 4, storage));
#endif
    if (wrong_device) REQUIRE(hipSetDevice(0) == hipSuccess);
#if TEST_SPARSE == 3
    float updates[] = {10, 30, 20, 40};
    inputs.push_back(input(updates, sizeof(updates), CHELIS_DTYPE_F32, {2, 2}, {1, 2}, 4, storage));
#elif TEST_SPARSE != 0
    float updates[] = {10, 30, 50, 70, 20, 40, 60, 80};
    inputs.push_back(input(updates, sizeof(updates), CHELIS_DTYPE_F32, {4, 2}, {1, 4}, 8, storage));
#endif
    std::vector<const chelis_device_tensor_owner *> borrowed(inputs.begin(), inputs.end());
    chelis_device_tensor_owner *outputs[2] = {};
    entry_probe_device(borrowed.data(), (int32_t)borrowed.size(), outputs, 2);
    if (wrong_device) return 0;
    float unchanged[6];
    REQUIRE(hipMemcpy(unchanged, storage[0], sizeof(target), hipMemcpyDeviceToHost) == hipSuccess);
    REQUIRE(!memcmp(unchanged, target, sizeof(target)));
    for (size_t i = 0; i < inputs.size(); ++i) {
        REQUIRE(outputs[0] != inputs[i] && outputs[1] != inputs[i]);
        chelis_device_tensor_release(inputs[i]);
        REQUIRE(hipFree(storage[i]) == hipSuccess);
    }
    const float target_expected[] = {1, 2, 3, 4, 5, 6};
#if TEST_SPARSE == 0
    const float expected[] = {5, 6, 1, 2, 5, 6, 3, 4};
#elif TEST_SPARSE == 1
    const float expected[] = {31, 42, 73, 84, 65, 86};
#elif TEST_SPARSE == 2
    const float expected[] = {30, 40, 70, 80, 50, 60};
#else
    const float expected[] = {1, 20, 3, 40, 30, 6};
#endif
    for (int i = 0; i < 2; ++i) {
        const auto *view = chelis_device_tensor_view(outputs[i]);
        REQUIRE(view->ownership == 1);
        auto *host = chelis_alloc(view->rank, view->shape, view->dtype);
        auto *write = chelis_tensor_begin_write(host);
        chelis_device_tensor_copy_to_host(write, outputs[i]);
        chelis_tensor_end_write(write);
        const size_t expected_bytes = i == 0 ? sizeof(target_expected) : sizeof(expected);
        REQUIRE(view->count * sizeof(float) == expected_bytes);
        REQUIRE(!memcmp(chelis_tensor_read_view(host).data, i == 0 ? target_expected : expected, expected_bytes));
        chelis_tensor_release(host);
        chelis_device_tensor_release(outputs[i]);
    }
    REQUIRE(fixture_live_allocations() == 0 && fixture_live_modules() == 0);
    puts("DEVICE ENTRY CPU EXECUTION: PASS");
}
