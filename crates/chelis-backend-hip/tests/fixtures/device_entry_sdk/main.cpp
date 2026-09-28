// Runs actual generated entry code against the companion and Rust metadata owner.
#include "chelis_device_owner.h"
#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifndef TEST_EMPTY_RESULT
#define TEST_EMPTY_RESULT 0
#endif

#define REQUIRE(condition) do { if (!(condition)) { \
    fprintf(stderr, "device entry fixture violation at line %d: %s\n", __LINE__, #condition); abort(); \
} } while (0)
extern "C" void entry_probe_device(const chelis_device_tensor_owner *const *, int32_t,
    chelis_device_tensor_owner **, int32_t);
extern "C" size_t fixture_live_allocations();
extern "C" size_t fixture_live_modules();

static void execute(const char *mode) {
    int rank = TEST_RANK + (!strcmp(mode, "wrong-rank") ? 1 : 0);
    const bool empty = !strcmp(mode, "empty");
    std::vector<int64_t> shape(rank, 1), strides(rank, 0);
    int64_t count = rank == 0 ? 1 : (empty ? 0 : 3);
    int64_t storage_count = rank == 0 ? 1 : (empty ? 0 : 5);
    if (rank != 0) { shape.back() = count; strides.back() = 2; }
    float host[6] = {1.0f, 99.0f, 2.0f, 99.0f, 3.0f, 6.0f};
#if TEST_MATRIX
    shape = {2, 3}; strides = {1, 2}; count = storage_count = 6;
    for (int i = 0; i < 6; ++i) host[i] = (float)(i + 1);
#endif
    void *data = nullptr;
    if (storage_count != 0) {
        REQUIRE(hipMalloc(&data, storage_count * sizeof(float)) == hipSuccess);
        REQUIRE(hipMemcpy(data, host, storage_count * sizeof(float), hipMemcpyHostToDevice) == hipSuccess);
    }
    chelis_gpu_tensor packet = {};
    packet.data = data;
    packet.shape = rank == 0 ? nullptr : shape.data();
    packet.strides = rank == 0 ? nullptr : strides.data();
    packet.count = count;
    packet.byte_capacity = storage_count * sizeof(float);
    packet.rank = rank;
    packet.dtype = !strcmp(mode, "wrong-dtype") ? CHELIS_DTYPE_I32 : CHELIS_DTYPE_F32;
    chelis_device_tensor_owner *input = chelis_device_tensor_import(&packet);
    if (!strcmp(mode, "wrong-device")) REQUIRE(hipSetDevice(1) == hipSuccess);
    const chelis_device_tensor_owner *inputs[] = { input };
    chelis_device_tensor_owner *outputs[2] = {};
    entry_probe_device(inputs, 1, outputs, 2);
    // A malformed call returning normally must make the negative test fail.
    if (!strcmp(mode, "wrong-rank") || !strcmp(mode, "wrong-dtype") || !strcmp(mode, "wrong-device")) return;
    REQUIRE(outputs[0] != input && outputs[1] != input && outputs[0] != outputs[1]);
    const int64_t output_count = TEST_EMPTY_RESULT ? 0 : count;
    for (auto output : outputs) {
        const chelis_gpu_tensor *view = chelis_device_tensor_view(output);
        REQUIRE(view->ownership == 1 && view->count == output_count);
        REQUIRE(!output_count || (view->data != data && view->byte_capacity == output_count * sizeof(float)));
    }
    float unchanged[6] = {};
    REQUIRE(hipMemcpy(unchanged, data, storage_count * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess);
    REQUIRE(!memcmp(unchanged, host, storage_count * sizeof(float)));
    chelis_device_tensor_release(input);
    REQUIRE(hipFree(data) == hipSuccess);
    // Escapes stay readable after every borrowed input/slot lifetime has ended.
    for (int output_index = 0; output_index < 2; ++output_index) {
        auto output = outputs[output_index];
        const chelis_gpu_tensor *view = chelis_device_tensor_view(output);
        chelis_tensor *host_output = chelis_alloc(view->rank, view->shape, view->dtype);
        auto guard = chelis_tensor_begin_write(host_output);
        chelis_device_tensor_copy_to_host(guard, output);
        chelis_tensor_end_write(guard);
        const float *values = (const float *)chelis_tensor_read_view(host_output).data;
        for (int64_t i = 0; i < output_count; ++i) {
#if TEST_MATRIX
            const float expected = (float)(i + 1);
#else
            const float expected = (output_index == 0 ? 1.0f : -1.0f) * (float)(i + 1);
#endif
            REQUIRE(values[i] == expected);
        }
        chelis_tensor_release(host_output);
        chelis_device_tensor_release(output);
    }
    REQUIRE(fixture_live_allocations() == 0);
    REQUIRE(fixture_live_modules() == 0);
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "positive";
    execute(mode);
    if (!strcmp(mode, "alternate-devices")) {
        REQUIRE(hipSetDevice(1) == hipSuccess);
        execute(mode);
        REQUIRE(hipSetDevice(0) == hipSuccess);
        execute(mode);
    }
    puts("DEVICE ENTRY CPU EXECUTION: PASS");
}
