// CPU fixture only: generated kernel bodies execute as compiled C++ functions.
// The memory checks make premature free, capacity overwrite and leaks observable.
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <limits.h>

#define REQUIRE(condition) do { if (!(condition)) { \
    fprintf(stderr, "device entry fixture violation at line %d: %s\n", __LINE__, #condition); abort(); \
} } while (0)

using Launch = void (*)(unsigned int, unsigned int, void **);
extern "C" Launch fixture_kernel(const char *name);
struct Allocation { size_t bytes; int device; };
static std::map<void *, Allocation> allocations;
static int current_device = 0;
struct Module { Launch launch; int device; };
static std::map<void *, Module> modules;
static const unsigned char guard = 0xa7;

static bool contains(const void *pointer, size_t bytes) {
    const uintptr_t address = (uintptr_t)pointer;
    for (const auto &allocation : allocations) {
        const uintptr_t start = (uintptr_t)allocation.first;
        if (address >= start && address - start <= allocation.second.bytes
            && bytes <= allocation.second.bytes - (address - start)) {
            REQUIRE(allocation.second.device == current_device);
            return true;
        }
    }
    return false;
}

extern "C" size_t fixture_live_allocations() { return allocations.size(); }
extern "C" size_t fixture_live_modules() { return modules.size(); }
extern "C" hipError_t hipMalloc(void **pointer, size_t bytes) {
    REQUIRE(bytes <= 1024 * 1024);
    *pointer = malloc(bytes + 32);
    REQUIRE(*pointer != nullptr);
    memset((unsigned char *)*pointer + bytes, guard, 32);
    REQUIRE(allocations.emplace(*pointer, Allocation{bytes, current_device}).second);
    return hipSuccess;
}
extern "C" hipError_t hipFree(void *pointer) {
    if (!pointer) return hipSuccess;
    auto entry = allocations.find(pointer);
    REQUIRE(entry != allocations.end());
    REQUIRE(entry->second.device == current_device);
    for (size_t i = 0; i < 32; ++i)
        REQUIRE(((const unsigned char *)pointer)[entry->second.bytes + i] == guard);
    allocations.erase(entry);
    free(pointer);
    return hipSuccess;
}
extern "C" hipError_t hipMemset(void *pointer, int value, size_t bytes) {
    REQUIRE(bytes == 0 || contains(pointer, bytes));
    if (bytes) memset(pointer, value, bytes);
    return hipSuccess;
}
extern "C" hipError_t hipMemcpy(void *destination, const void *source, size_t bytes, hipMemcpyKind kind) {
    if (!bytes) return hipSuccess;
    if (kind == hipMemcpyHostToDevice || kind == hipMemcpyDeviceToDevice)
        REQUIRE(contains(destination, bytes));
    if (kind == hipMemcpyDeviceToHost || kind == hipMemcpyDeviceToDevice)
        REQUIRE(contains(source, bytes));
    memcpy(destination, source, bytes);
    return hipSuccess;
}
extern "C" const char *hipGetErrorString(hipError_t) { return "CPU fixture HIP error"; }
extern "C" hipError_t hipGetDevice(int *device) { *device = current_device; return hipSuccess; }
extern "C" hipError_t hipSetDevice(int device) {
    REQUIRE(device == 0 || device == 1);
    current_device = device;
    return hipSuccess;
}
extern "C" hipError_t hipPointerGetAttributes(hipPointerAttribute_t *attributes, const void *pointer) {
    const uintptr_t address = (uintptr_t)pointer;
    for (const auto &allocation : allocations) {
        const uintptr_t start = (uintptr_t)allocation.first;
        if (address >= start && address - start < allocation.second.bytes) {
            *attributes = {hipMemoryTypeDevice, allocation.second.device, allocation.first, nullptr, 0, 0};
            return hipSuccess;
        }
    }
    return hipErrorInvalidValue;
}
extern "C" hipError_t hipGetDeviceProperties(hipDeviceProp_t *properties, int) {
    *properties = {64, {INT_MAX, 65535, 65535}, {1024, 1024, 64}, 1024};
    return hipSuccess;
}
extern "C" hipError_t hipDeviceSynchronize() { return hipSuccess; }

// The actual source was compiled alongside this fixture. Module lookup only
// selects that compiled function; it does not interpret or replace its work.
extern "C" hipError_t hipModuleUnload(hipModule_t module) {
    auto found = modules.find(module);
    REQUIRE(found != modules.end() && found->second.device == current_device);
    modules.erase(found);
    free(module);
    return hipSuccess;
}
extern "C" hipError_t hipModuleLoadData(hipModule_t *module, const void *code) {
    *module = malloc(1);
    REQUIRE(*module != nullptr);
    REQUIRE(modules.emplace(*module, Module{fixture_kernel((const char *)code), current_device}).second);
    return hipSuccess;
}
extern "C" hipError_t hipModuleGetFunction(hipFunction_t *function, hipModule_t module, const char *) {
    REQUIRE(modules.at(module).device == current_device);
    *function = module;
    return hipSuccess;
}
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t function,
    unsigned int gx, unsigned int gy, unsigned int gz,
    unsigned int bx, unsigned int by, unsigned int bz,
    unsigned int, void *, void **arguments, void **) {
    REQUIRE(gy == 1 && gz == 1 && by == 1 && bz == 1);
    REQUIRE(gx <= 16 && bx <= 256);
    const auto &module = modules.at(function);
    REQUIRE(module.device == current_device);
    module.launch(gx, bx, arguments);
    return hipSuccess;
}
extern "C" hiprtcResult hiprtcCreateProgram(hiprtcProgram *program, const char *, const char *name,
    int, const char **, const char **) {
    *program = new std::string(name);
    return HIPRTC_SUCCESS;
}
extern "C" hiprtcResult hiprtcCompileProgram(hiprtcProgram, int, const char **) { return HIPRTC_SUCCESS; }
extern "C" hiprtcResult hiprtcGetProgramLogSize(hiprtcProgram, size_t *size) { *size = 1; return HIPRTC_SUCCESS; }
extern "C" hiprtcResult hiprtcGetProgramLog(hiprtcProgram, char *log) { *log = 0; return HIPRTC_SUCCESS; }
extern "C" hiprtcResult hiprtcGetCodeSize(hiprtcProgram program, size_t *size) {
    *size = ((std::string *)program)->size() + 1; return HIPRTC_SUCCESS;
}
extern "C" hiprtcResult hiprtcGetCode(hiprtcProgram program, char *code) {
    strcpy(code, ((std::string *)program)->c_str()); return HIPRTC_SUCCESS;
}
extern "C" hiprtcResult hiprtcDestroyProgram(hiprtcProgram *program) {
    delete (std::string *)*program; *program = nullptr; return HIPRTC_SUCCESS;
}
extern "C" const char *hiprtcGetErrorString(hiprtcResult) { return "CPU fixture"; }
