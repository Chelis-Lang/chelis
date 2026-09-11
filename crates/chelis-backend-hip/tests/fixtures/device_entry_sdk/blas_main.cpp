// Prepared device input layout and owner lifetime witness; CPU BLAS transport.
#include "chelis_device_owner.h"
#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define REQUIRE(condition) do { if (!(condition)) { \
    fprintf(stderr, "BLAS entry fixture violation at line %d: %s\n", __LINE__, #condition); abort(); \
} } while (0)
extern "C" void entry_probe_device(const chelis_device_tensor_owner *const *, int32_t,
    chelis_device_tensor_owner **, int32_t);
extern "C" size_t fixture_live_allocations();
extern "C" size_t fixture_live_modules();

int main(int argc, char **argv) {
    const bool wrong_device = argc > 1 && !strcmp(argv[1], "wrong-second-device");
    float physical[2][6] = {{1, 4, 2, 5, 3, 6}, {1, 3, 5, 2, 4, 6}};
    int64_t shape[2][2] = {{2, 3}, {3, 2}}, strides[2][2] = {{1, 2}, {1, 3}};
    chelis_device_tensor_owner *inputs[2];
    void *data[2];
    for (int i = 0; i < 2; ++i) {
        if (i == 1 && wrong_device) REQUIRE(hipSetDevice(1) == hipSuccess);
        REQUIRE(hipMalloc(&data[i], sizeof(physical[i])) == hipSuccess);
        REQUIRE(hipMemcpy(data[i], physical[i], sizeof(physical[i]), hipMemcpyHostToDevice) == hipSuccess);
        chelis_gpu_tensor packet = {};
        packet.data = data[i]; packet.shape = shape[i]; packet.strides = strides[i];
        packet.count = 6; packet.byte_capacity = sizeof(physical[i]); packet.rank = 2; packet.dtype = CHELIS_DTYPE_F32;
        inputs[i] = chelis_device_tensor_import(&packet);
    }
    REQUIRE(hipSetDevice(0) == hipSuccess);
    const chelis_device_tensor_owner *borrowed[] = {inputs[0], inputs[1]};
    chelis_device_tensor_owner *outputs[2] = {};
    entry_probe_device(borrowed, 2, outputs, 2);
    if (wrong_device) return 0;
    for (int i = 0; i < 2; ++i) {
        float unchanged[6];
        REQUIRE(hipMemcpy(unchanged, data[i], sizeof(unchanged), hipMemcpyDeviceToHost) == hipSuccess);
        REQUIRE(!memcmp(unchanged, physical[i], sizeof(unchanged)));
        REQUIRE(outputs[0] != inputs[i] && outputs[1] != inputs[i]);
        chelis_device_tensor_release(inputs[i]);
        REQUIRE(hipFree(data[i]) == hipSuccess);
    }
    const float expected_input[] = {1, 2, 3, 4, 5, 6}, expected_product[] = {22, 28, 49, 64};
    for (int i = 0; i < 2; ++i) {
        const auto *view = chelis_device_tensor_view(outputs[i]);
        REQUIRE(view->ownership == 1 && view->count == (i == 0 ? 6 : 4));
        auto *host = chelis_alloc(view->rank, view->shape, view->dtype);
        auto *write = chelis_tensor_begin_write(host);
        chelis_device_tensor_copy_to_host(write, outputs[i]);
        chelis_tensor_end_write(write);
        REQUIRE(!memcmp(chelis_tensor_read_view(host).data, i == 0 ? expected_input : expected_product, (size_t)view->count * sizeof(float)));
        chelis_tensor_release(host);
        chelis_device_tensor_release(outputs[i]);
    }
    REQUIRE(fixture_live_allocations() == 0 && fixture_live_modules() == 0);
    puts("DEVICE ENTRY CPU EXECUTION: PASS");
}
