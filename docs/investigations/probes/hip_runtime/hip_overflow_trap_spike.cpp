// HIP GPU overflow-trap feasibility spike (feeds #729 Phase-1 freeze, condition 1b).
//
// This is the HIP (gfx1151) half of the [04-NUM-3] "traps in every lane"
// evidence; the Metal half is Robert's spike on #737. Methodology is mirrored
// exactly so the overhead percentages compare across backends:
//
//   * per-element integer-overflow DETECTION (not raise): a device buffer with
//     an atomic flag + first-failing-index, written with atomicOr / atomicMin
//     (HIP device atomics are relaxed / device-scope). The host reads it after
//     hipDeviceSynchronize and would raise the branded C2 trap end-of-dispatch.
//   * timing = hipEventRecord / hipEventElapsedTime GPU events, median of 15
//     iterations after 3 warmup, checked/unchecked interleaved per round to
//     equalize clock drift.
//   * memory-bound cells: 2^24 elements, one op per element (the shape of
//     chelis's actual elementwise HIP kernels).
//   * compute-bound cells: 2^22 elements x 64 chained in-register ops, with
//     data-dependent operands so the compiler cannot delete the checks.
//   * checks standardized to the same hand-written forms Robert validated on
//     Metal: sign-bit trick for add/sub, widening for i32 mul, high-multiply
//     (__mul64hi, the HIP analog of MSL mulhi) for i64 mul.
//
// Correctness gates per cell: an in-range run must leave the flag clear; a
// planted overflow in each polarity must set the flag with the exact index;
// the checked output must be byte-identical (memcmp) to the unchecked output;
// and the unchecked output is spot-checked against a host oracle.
//
// Build (under scripts/hip_test.py env):
//   hipcc -O2 hip_overflow_trap_spike.cpp -o hip_overflow_trap_spike
// Run:
//   ./hip_overflow_trap_spike

#include <hip/hip_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#define HIP_CHECK(expr)                                                        \
    do {                                                                       \
        hipError_t _e = (expr);                                                \
        if (_e != hipSuccess) {                                                 \
            fprintf(stderr, "HIP error %s at %s:%d: %s\n", #expr, __FILE__,    \
                    __LINE__, hipGetErrorString(_e));                          \
            std::exit(2);                                                      \
        }                                                                      \
    } while (0)

static const int BLOCK = 256;
static const int MEM_N = 1 << 24;   // 16,777,216 elements
static const int CB_N = 1 << 22;    // 4,194,304 elements
static const int CB_CHAIN = 64;     // chained ops per element

// Detection state: a flag plus the first (minimum) failing global index.
struct FlagBuf {
    unsigned int flag;
    unsigned int idx;
};

__device__ inline void mark(FlagBuf* st, unsigned int g) {
    atomicOr(&st->flag, 1u);
    atomicMin(&st->idx, g);
}

// ------------------------------------------------------------------ overflow predicates (device)
__device__ inline bool add_ovf_i32(int x, int y, int r) { return (~(x ^ y) & (x ^ r)) < 0; }
__device__ inline bool sub_ovf_i32(int x, int y, int r) { return ((x ^ y) & (x ^ r)) < 0; }
__device__ inline bool add_ovf_i64(long long x, long long y, long long r) { return (~(x ^ y) & (x ^ r)) < 0; }
__device__ inline bool sub_ovf_i64(long long x, long long y, long long r) { return ((x ^ y) & (x ^ r)) < 0; }

// ================================================================== memory-bound kernels
#define GIDX int g = blockIdx.x * blockDim.x + threadIdx.x

__global__ void add_i32_u(const int* a, const int* b, int* o, int n) { GIDX; if (g < n) o[g] = a[g] + b[g]; }
__global__ void add_i32_c(const int* a, const int* b, int* o, FlagBuf* st, int n) {
    GIDX; if (g >= n) return; int x = a[g], y = b[g], r = x + y; if (add_ovf_i32(x, y, r)) mark(st, g); o[g] = r;
}
__global__ void sub_i32_u(const int* a, const int* b, int* o, int n) { GIDX; if (g < n) o[g] = a[g] - b[g]; }
__global__ void sub_i32_c(const int* a, const int* b, int* o, FlagBuf* st, int n) {
    GIDX; if (g >= n) return; int x = a[g], y = b[g], r = x - y; if (sub_ovf_i32(x, y, r)) mark(st, g); o[g] = r;
}
__global__ void mul_i32_u(const int* a, const int* b, int* o, int n) { GIDX; if (g < n) o[g] = a[g] * b[g]; }
__global__ void mul_i32_c(const int* a, const int* b, int* o, FlagBuf* st, int n) {
    GIDX; if (g >= n) return; int x = a[g], y = b[g];
    long long w = (long long)x * (long long)y;
    if (w < (long long)INT_MIN || w > (long long)INT_MAX) mark(st, g);
    o[g] = (int)w;
}

__global__ void add_i64_u(const long long* a, const long long* b, long long* o, int n) { GIDX; if (g < n) o[g] = a[g] + b[g]; }
__global__ void add_i64_c(const long long* a, const long long* b, long long* o, FlagBuf* st, int n) {
    GIDX; if (g >= n) return; long long x = a[g], y = b[g], r = x + y; if (add_ovf_i64(x, y, r)) mark(st, g); o[g] = r;
}
__global__ void sub_i64_u(const long long* a, const long long* b, long long* o, int n) { GIDX; if (g < n) o[g] = a[g] - b[g]; }
__global__ void sub_i64_c(const long long* a, const long long* b, long long* o, FlagBuf* st, int n) {
    GIDX; if (g >= n) return; long long x = a[g], y = b[g], r = x - y; if (sub_ovf_i64(x, y, r)) mark(st, g); o[g] = r;
}
__global__ void mul_i64_u(const long long* a, const long long* b, long long* o, int n) {
    GIDX; if (g >= n) return; long long x = a[g], y = b[g];
    o[g] = (long long)((unsigned long long)x * (unsigned long long)y);
}
__global__ void mul_i64_c(const long long* a, const long long* b, long long* o, FlagBuf* st, int n) {
    GIDX; if (g >= n) return; long long x = a[g], y = b[g];
    long long lo = (long long)((unsigned long long)x * (unsigned long long)y);
    long long hi = __mul64hi(x, y);
    if (hi != (lo >> 63)) mark(st, g);   // overflow iff hi is not the sign-extension of lo
    o[g] = lo;
}

// ================================================================== compute-bound kernels
// 64 ops per element, all in-register. Operands are masked with a RUNTIME mask
// (a kernel argument, not a literal) so the compiler cannot prove they are
// bounded and therefore cannot delete the overflow check -- but at runtime the
// mask keeps every value in-range so timing measures the CHECK ALU cost, not
// the atomic mark() path. (An earlier draft masked with compile-time literals;
// the backend proved the i32 widen-mul check dead and reported a spurious +0.3%.
// The runtime mask is what forces every cell's check to survive to codegen.)
// Add cells chain the accumulator; mul cells XOR independent products (a 64-long
// mul chain would overflow); both are 64 checked ops/element either way.

__global__ void add_i32_cb_u(const int* a, int* o, int mask, int n) {
    GIDX; if (g >= n) return; int acc = a[g] & mask; int y = (a[g] >> 3) & mask;
    for (int k = 0; k < CB_CHAIN; k++) { int d = ((a[g] >> (k & 15)) & 1) ? y : -y; acc = acc + d; }
    o[g] = acc;
}
__global__ void add_i32_cb_c(const int* a, int* o, FlagBuf* st, int mask, int n) {
    GIDX; if (g >= n) return; int acc = a[g] & mask; int y = (a[g] >> 3) & mask;
    for (int k = 0; k < CB_CHAIN; k++) { int d = ((a[g] >> (k & 15)) & 1) ? y : -y; int r = acc + d; if (add_ovf_i32(acc, d, r)) mark(st, g); acc = r; }
    o[g] = acc;
}
__global__ void mul_i32_cb_u(const int* a, int* o, int mask, int n) {
    GIDX; if (g >= n) return; int x = a[g] & mask; int acc = 0;
    for (int k = 0; k < CB_CHAIN; k++) { int y = ((a[g] >> (k & 15)) ^ x) & mask; int p = x * y; acc ^= p; }
    o[g] = acc;
}
__global__ void mul_i32_cb_c(const int* a, int* o, FlagBuf* st, int mask, int n) {
    GIDX; if (g >= n) return; int x = a[g] & mask; int acc = 0;
    for (int k = 0; k < CB_CHAIN; k++) { int y = ((a[g] >> (k & 15)) ^ x) & mask; long long w = (long long)x * (long long)y; if (w < (long long)INT_MIN || w > (long long)INT_MAX) mark(st, g); acc ^= (int)w; }
    o[g] = acc;
}
__global__ void add_i64_cb_u(const long long* a, long long* o, long long mask, int n) {
    GIDX; if (g >= n) return; long long acc = a[g] & mask; long long y = (a[g] >> 3) & mask;
    for (int k = 0; k < CB_CHAIN; k++) { long long d = ((a[g] >> (k & 15)) & 1) ? y : -y; acc = acc + d; }
    o[g] = acc;
}
__global__ void add_i64_cb_c(const long long* a, long long* o, FlagBuf* st, long long mask, int n) {
    GIDX; if (g >= n) return; long long acc = a[g] & mask; long long y = (a[g] >> 3) & mask;
    for (int k = 0; k < CB_CHAIN; k++) { long long d = ((a[g] >> (k & 15)) & 1) ? y : -y; long long r = acc + d; if (add_ovf_i64(acc, d, r)) mark(st, g); acc = r; }
    o[g] = acc;
}
__global__ void mul_i64_cb_u(const long long* a, long long* o, long long mask, int n) {
    GIDX; if (g >= n) return; long long x = a[g] & mask; long long acc = 0;
    for (int k = 0; k < CB_CHAIN; k++) { long long y = ((a[g] >> (k & 15)) ^ x) & mask; long long lo = (long long)((unsigned long long)x * (unsigned long long)y); acc ^= lo; }
    o[g] = acc;
}
__global__ void mul_i64_cb_c(const long long* a, long long* o, FlagBuf* st, long long mask, int n) {
    GIDX; if (g >= n) return; long long x = a[g] & mask; long long acc = 0;
    for (int k = 0; k < CB_CHAIN; k++) { long long y = ((a[g] >> (k & 15)) ^ x) & mask; long long lo = (long long)((unsigned long long)x * (unsigned long long)y); long long hi = __mul64hi(x, y); if (hi != (lo >> 63)) mark(st, g); acc ^= lo; }
    o[g] = acc;
}

// ================================================================== host harness
static float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

template <typename LaunchFn>
static float time_launch(LaunchFn&& launch) {
    hipEvent_t s, e;
    HIP_CHECK(hipEventCreate(&s));
    HIP_CHECK(hipEventCreate(&e));
    HIP_CHECK(hipEventRecord(s));
    launch();
    HIP_CHECK(hipEventRecord(e));
    HIP_CHECK(hipEventSynchronize(e));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, s, e));
    HIP_CHECK(hipEventDestroy(s));
    HIP_CHECK(hipEventDestroy(e));
    return ms;
}

struct Row {
    const char* cell;
    float unchecked_ms;
    float checked_ms;
};

static void print_table(const char* title, const std::vector<Row>& rows) {
    printf("\n%s\n", title);
    printf("| cell | unchecked (ms) | checked (ms) | overhead |\n");
    printf("|---|---|---|---|\n");
    for (const auto& r : rows) {
        float ov = (r.unchecked_ms > 0) ? (r.checked_ms - r.unchecked_ms) / r.unchecked_ms * 100.0f : 0.0f;
        printf("| %s | %.4f | %.4f | %+.1f%% |\n", r.cell, r.unchecked_ms, r.checked_ms, ov);
    }
}

static int g_fail = 0;
static void expect(bool cond, const char* what) {
    if (!cond) { fprintf(stderr, "CORRECTNESS FAIL: %s\n", what); g_fail = 1; }
}

// reset device flag buffer to {0, UINT_MAX}
static void reset_flag(FlagBuf* d) {
    FlagBuf h{0u, std::numeric_limits<unsigned int>::max()};
    HIP_CHECK(hipMemcpy(d, &h, sizeof(FlagBuf), hipMemcpyHostToDevice));
}
static FlagBuf read_flag(FlagBuf* d) {
    FlagBuf h{};
    HIP_CHECK(hipMemcpy(&h, d, sizeof(FlagBuf), hipMemcpyDeviceToHost));
    return h;
}

int main() {
    int dev = 0;
    HIP_CHECK(hipSetDevice(dev));
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    printf("device: %s (gcnArch %s)\n", prop.name, prop.gcnArchName);
    printf("mem-bound N = %d (2^24), compute-bound N = %d (2^22) x %d chained ops\n",
           MEM_N, CB_N, CB_CHAIN);

    std::mt19937_64 rng(0xC0FFEE);

    // ---------- device buffers ----------
    FlagBuf* d_flag = nullptr;
    HIP_CHECK(hipMalloc(&d_flag, sizeof(FlagBuf)));

    // int32 memory-bound buffers
    std::vector<int> a32(MEM_N), b32(MEM_N), o32u(MEM_N), o32c(MEM_N);
    std::vector<long long> a64(MEM_N), b64(MEM_N), o64u(MEM_N), o64c(MEM_N);

    int *da32, *db32, *do32u, *do32c;
    long long *da64, *db64, *do64u, *do64c;
    HIP_CHECK(hipMalloc(&da32, MEM_N * sizeof(int)));
    HIP_CHECK(hipMalloc(&db32, MEM_N * sizeof(int)));
    HIP_CHECK(hipMalloc(&do32u, MEM_N * sizeof(int)));
    HIP_CHECK(hipMalloc(&do32c, MEM_N * sizeof(int)));
    HIP_CHECK(hipMalloc(&da64, MEM_N * sizeof(long long)));
    HIP_CHECK(hipMalloc(&db64, MEM_N * sizeof(long long)));
    HIP_CHECK(hipMalloc(&do64u, MEM_N * sizeof(long long)));
    HIP_CHECK(hipMalloc(&do64c, MEM_N * sizeof(long long)));

    int mem_grid = (MEM_N + BLOCK - 1) / BLOCK;
    int cb_grid = (CB_N + BLOCK - 1) / BLOCK;

    std::vector<Row> mem_rows;
    std::vector<Row> cb_rows;

    // ============================================================ memory-bound: int32 add
    {
        std::uniform_int_distribution<int> d(-100000000, 100000000);
        for (int i = 0; i < MEM_N; i++) { a32[i] = d(rng); b32[i] = d(rng); }
        HIP_CHECK(hipMemcpy(da32, a32.data(), MEM_N * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(db32, b32.data(), MEM_N * sizeof(int), hipMemcpyHostToDevice));

        auto lu = [&]() { add_i32_u<<<mem_grid, BLOCK>>>(da32, db32, do32u, MEM_N); };
        auto lc = [&]() { add_i32_c<<<mem_grid, BLOCK>>>(da32, db32, do32c, d_flag, MEM_N); };

        // correctness: in-range clear
        reset_flag(d_flag);
        lc(); HIP_CHECK(hipDeviceSynchronize());
        expect(read_flag(d_flag).flag == 0, "add_i32 in-range must leave flag clear");
        // memcmp
        lu(); HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(o32u.data(), do32u, MEM_N * sizeof(int), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(o32c.data(), do32c, MEM_N * sizeof(int), hipMemcpyDeviceToHost));
        expect(std::memcmp(o32u.data(), o32c.data(), MEM_N * sizeof(int)) == 0, "add_i32 checked==unchecked");
        // oracle sample
        for (int s = 0; s < 1000; s++) { int i = (int)(rng() % MEM_N); expect(o32u[i] == a32[i] + b32[i], "add_i32 oracle"); }
        // planted overflow, positive polarity at P
        int P = 12345; a32[P] = INT_MAX; b32[P] = 1;
        HIP_CHECK(hipMemcpy(da32, a32.data(), MEM_N * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(db32, b32.data(), MEM_N * sizeof(int), hipMemcpyHostToDevice));
        reset_flag(d_flag); lc(); HIP_CHECK(hipDeviceSynchronize());
        { FlagBuf f = read_flag(d_flag); expect(f.flag == 1 && f.idx == (unsigned)P, "add_i32 planted +overflow index"); }
        // planted overflow, negative polarity at Q
        int Q = 777; a32[P] = 0; b32[P] = 0; a32[Q] = INT_MIN; b32[Q] = -1;
        HIP_CHECK(hipMemcpy(da32, a32.data(), MEM_N * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(db32, b32.data(), MEM_N * sizeof(int), hipMemcpyHostToDevice));
        reset_flag(d_flag); lc(); HIP_CHECK(hipDeviceSynchronize());
        { FlagBuf f = read_flag(d_flag); expect(f.flag == 1 && f.idx == (unsigned)Q, "add_i32 planted -overflow index"); }
        // restore in-range for timing
        a32[Q] = 5; b32[Q] = 7;
        HIP_CHECK(hipMemcpy(da32, a32.data(), MEM_N * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(db32, b32.data(), MEM_N * sizeof(int), hipMemcpyHostToDevice));

        // timing: 3 warmup, 15 measured, interleaved
        for (int w = 0; w < 3; w++) { time_launch(lu); reset_flag(d_flag); time_launch(lc); }
        std::vector<float> tu, tc;
        for (int r = 0; r < 15; r++) { tu.push_back(time_launch(lu)); reset_flag(d_flag); tc.push_back(time_launch(lc)); }
        mem_rows.push_back({"add int32 (sign-bit)", median(tu), median(tc)});
    }

    // ============================================================ memory-bound: int32 sub
    {
        std::uniform_int_distribution<int> d(-100000000, 100000000);
        for (int i = 0; i < MEM_N; i++) { a32[i] = d(rng); b32[i] = d(rng); }
        HIP_CHECK(hipMemcpy(da32, a32.data(), MEM_N * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(db32, b32.data(), MEM_N * sizeof(int), hipMemcpyHostToDevice));
        auto lu = [&]() { sub_i32_u<<<mem_grid, BLOCK>>>(da32, db32, do32u, MEM_N); };
        auto lc = [&]() { sub_i32_c<<<mem_grid, BLOCK>>>(da32, db32, do32c, d_flag, MEM_N); };
        reset_flag(d_flag); lc(); HIP_CHECK(hipDeviceSynchronize());
        expect(read_flag(d_flag).flag == 0, "sub_i32 in-range clear");
        lu(); HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(o32u.data(), do32u, MEM_N * sizeof(int), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(o32c.data(), do32c, MEM_N * sizeof(int), hipMemcpyDeviceToHost));
        expect(std::memcmp(o32u.data(), o32c.data(), MEM_N * sizeof(int)) == 0, "sub_i32 checked==unchecked");
        for (int s = 0; s < 1000; s++) { int i = (int)(rng() % MEM_N); expect(o32u[i] == a32[i] - b32[i], "sub_i32 oracle"); }
        int P = 22222; a32[P] = INT_MAX; b32[P] = -1;  // MAX - (-1) overflow
        HIP_CHECK(hipMemcpy(da32, a32.data(), MEM_N * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(db32, b32.data(), MEM_N * sizeof(int), hipMemcpyHostToDevice));
        reset_flag(d_flag); lc(); HIP_CHECK(hipDeviceSynchronize());
        { FlagBuf f = read_flag(d_flag); expect(f.flag == 1 && f.idx == (unsigned)P, "sub_i32 planted +overflow"); }
        int Q = 333; a32[P] = 1; b32[P] = 1; a32[Q] = INT_MIN; b32[Q] = 1;  // MIN - 1 underflow
        HIP_CHECK(hipMemcpy(da32, a32.data(), MEM_N * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(db32, b32.data(), MEM_N * sizeof(int), hipMemcpyHostToDevice));
        reset_flag(d_flag); lc(); HIP_CHECK(hipDeviceSynchronize());
        { FlagBuf f = read_flag(d_flag); expect(f.flag == 1 && f.idx == (unsigned)Q, "sub_i32 planted -overflow"); }
        a32[Q] = 9; b32[Q] = 4;
        HIP_CHECK(hipMemcpy(da32, a32.data(), MEM_N * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(db32, b32.data(), MEM_N * sizeof(int), hipMemcpyHostToDevice));
        for (int w = 0; w < 3; w++) { time_launch(lu); reset_flag(d_flag); time_launch(lc); }
        std::vector<float> tu, tc;
        for (int r = 0; r < 15; r++) { tu.push_back(time_launch(lu)); reset_flag(d_flag); tc.push_back(time_launch(lc)); }
        mem_rows.push_back({"sub int32 (sign-bit)", median(tu), median(tc)});
    }

    // ============================================================ memory-bound: int32 mul
    {
        std::uniform_int_distribution<int> d(-30000, 30000);
        for (int i = 0; i < MEM_N; i++) { a32[i] = d(rng); b32[i] = d(rng); }
        HIP_CHECK(hipMemcpy(da32, a32.data(), MEM_N * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(db32, b32.data(), MEM_N * sizeof(int), hipMemcpyHostToDevice));
        auto lu = [&]() { mul_i32_u<<<mem_grid, BLOCK>>>(da32, db32, do32u, MEM_N); };
        auto lc = [&]() { mul_i32_c<<<mem_grid, BLOCK>>>(da32, db32, do32c, d_flag, MEM_N); };
        reset_flag(d_flag); lc(); HIP_CHECK(hipDeviceSynchronize());
        expect(read_flag(d_flag).flag == 0, "mul_i32 in-range clear");
        lu(); HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(o32u.data(), do32u, MEM_N * sizeof(int), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(o32c.data(), do32c, MEM_N * sizeof(int), hipMemcpyDeviceToHost));
        expect(std::memcmp(o32u.data(), o32c.data(), MEM_N * sizeof(int)) == 0, "mul_i32 checked==unchecked");
        for (int s = 0; s < 1000; s++) { int i = (int)(rng() % MEM_N); expect(o32u[i] == a32[i] * b32[i], "mul_i32 oracle"); }
        int P = 44444; a32[P] = 100000; b32[P] = 100000;  // 1e10 overflow
        HIP_CHECK(hipMemcpy(da32, a32.data(), MEM_N * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(db32, b32.data(), MEM_N * sizeof(int), hipMemcpyHostToDevice));
        reset_flag(d_flag); lc(); HIP_CHECK(hipDeviceSynchronize());
        { FlagBuf f = read_flag(d_flag); expect(f.flag == 1 && f.idx == (unsigned)P, "mul_i32 planted +overflow"); }
        int Q = 555; a32[P] = 3; b32[P] = 3; a32[Q] = -100000; b32[Q] = 100000;
        HIP_CHECK(hipMemcpy(da32, a32.data(), MEM_N * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(db32, b32.data(), MEM_N * sizeof(int), hipMemcpyHostToDevice));
        reset_flag(d_flag); lc(); HIP_CHECK(hipDeviceSynchronize());
        { FlagBuf f = read_flag(d_flag); expect(f.flag == 1 && f.idx == (unsigned)Q, "mul_i32 planted -overflow"); }
        a32[Q] = 6; b32[Q] = 7;
        HIP_CHECK(hipMemcpy(da32, a32.data(), MEM_N * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(db32, b32.data(), MEM_N * sizeof(int), hipMemcpyHostToDevice));
        for (int w = 0; w < 3; w++) { time_launch(lu); reset_flag(d_flag); time_launch(lc); }
        std::vector<float> tu, tc;
        for (int r = 0; r < 15; r++) { tu.push_back(time_launch(lu)); reset_flag(d_flag); tc.push_back(time_launch(lc)); }
        mem_rows.push_back({"mul int32 (widen)", median(tu), median(tc)});
    }

    // ============================================================ memory-bound: int64 add
    {
        std::uniform_int_distribution<long long> d(-400000000000000000LL, 400000000000000000LL);
        for (int i = 0; i < MEM_N; i++) { a64[i] = d(rng); b64[i] = d(rng); }
        HIP_CHECK(hipMemcpy(da64, a64.data(), MEM_N * sizeof(long long), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(db64, b64.data(), MEM_N * sizeof(long long), hipMemcpyHostToDevice));
        auto lu = [&]() { add_i64_u<<<mem_grid, BLOCK>>>(da64, db64, do64u, MEM_N); };
        auto lc = [&]() { add_i64_c<<<mem_grid, BLOCK>>>(da64, db64, do64c, d_flag, MEM_N); };
        reset_flag(d_flag); lc(); HIP_CHECK(hipDeviceSynchronize());
        expect(read_flag(d_flag).flag == 0, "add_i64 in-range clear");
        lu(); HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(o64u.data(), do64u, MEM_N * sizeof(long long), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(o64c.data(), do64c, MEM_N * sizeof(long long), hipMemcpyDeviceToHost));
        expect(std::memcmp(o64u.data(), o64c.data(), MEM_N * sizeof(long long)) == 0, "add_i64 checked==unchecked");
        for (int s = 0; s < 1000; s++) { int i = (int)(rng() % MEM_N); expect(o64u[i] == a64[i] + b64[i], "add_i64 oracle"); }
        int P = 66666; a64[P] = INT64_MAX; b64[P] = 1;
        HIP_CHECK(hipMemcpy(da64, a64.data(), MEM_N * sizeof(long long), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(db64, b64.data(), MEM_N * sizeof(long long), hipMemcpyHostToDevice));
        reset_flag(d_flag); lc(); HIP_CHECK(hipDeviceSynchronize());
        { FlagBuf f = read_flag(d_flag); expect(f.flag == 1 && f.idx == (unsigned)P, "add_i64 planted +overflow"); }
        int Q = 888; a64[P] = 2; b64[P] = 3; a64[Q] = INT64_MIN; b64[Q] = -1;
        HIP_CHECK(hipMemcpy(da64, a64.data(), MEM_N * sizeof(long long), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(db64, b64.data(), MEM_N * sizeof(long long), hipMemcpyHostToDevice));
        reset_flag(d_flag); lc(); HIP_CHECK(hipDeviceSynchronize());
        { FlagBuf f = read_flag(d_flag); expect(f.flag == 1 && f.idx == (unsigned)Q, "add_i64 planted -overflow"); }
        a64[Q] = 100; b64[Q] = 200;
        HIP_CHECK(hipMemcpy(da64, a64.data(), MEM_N * sizeof(long long), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(db64, b64.data(), MEM_N * sizeof(long long), hipMemcpyHostToDevice));
        for (int w = 0; w < 3; w++) { time_launch(lu); reset_flag(d_flag); time_launch(lc); }
        std::vector<float> tu, tc;
        for (int r = 0; r < 15; r++) { tu.push_back(time_launch(lu)); reset_flag(d_flag); tc.push_back(time_launch(lc)); }
        mem_rows.push_back({"add int64 (sign-bit)", median(tu), median(tc)});
    }

    // ============================================================ memory-bound: int64 sub
    {
        std::uniform_int_distribution<long long> d(-400000000000000000LL, 400000000000000000LL);
        for (int i = 0; i < MEM_N; i++) { a64[i] = d(rng); b64[i] = d(rng); }
        HIP_CHECK(hipMemcpy(da64, a64.data(), MEM_N * sizeof(long long), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(db64, b64.data(), MEM_N * sizeof(long long), hipMemcpyHostToDevice));
        auto lu = [&]() { sub_i64_u<<<mem_grid, BLOCK>>>(da64, db64, do64u, MEM_N); };
        auto lc = [&]() { sub_i64_c<<<mem_grid, BLOCK>>>(da64, db64, do64c, d_flag, MEM_N); };
        reset_flag(d_flag); lc(); HIP_CHECK(hipDeviceSynchronize());
        expect(read_flag(d_flag).flag == 0, "sub_i64 in-range clear");
        lu(); HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(o64u.data(), do64u, MEM_N * sizeof(long long), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(o64c.data(), do64c, MEM_N * sizeof(long long), hipMemcpyDeviceToHost));
        expect(std::memcmp(o64u.data(), o64c.data(), MEM_N * sizeof(long long)) == 0, "sub_i64 checked==unchecked");
        for (int s = 0; s < 1000; s++) { int i = (int)(rng() % MEM_N); expect(o64u[i] == a64[i] - b64[i], "sub_i64 oracle"); }
        int P = 99999; a64[P] = INT64_MAX; b64[P] = -1;
        HIP_CHECK(hipMemcpy(da64, a64.data(), MEM_N * sizeof(long long), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(db64, b64.data(), MEM_N * sizeof(long long), hipMemcpyHostToDevice));
        reset_flag(d_flag); lc(); HIP_CHECK(hipDeviceSynchronize());
        { FlagBuf f = read_flag(d_flag); expect(f.flag == 1 && f.idx == (unsigned)P, "sub_i64 planted +overflow"); }
        int Q = 1234; a64[P] = 1; b64[P] = 1; a64[Q] = INT64_MIN; b64[Q] = 1;
        HIP_CHECK(hipMemcpy(da64, a64.data(), MEM_N * sizeof(long long), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(db64, b64.data(), MEM_N * sizeof(long long), hipMemcpyHostToDevice));
        reset_flag(d_flag); lc(); HIP_CHECK(hipDeviceSynchronize());
        { FlagBuf f = read_flag(d_flag); expect(f.flag == 1 && f.idx == (unsigned)Q, "sub_i64 planted -overflow"); }
        a64[Q] = 50; b64[Q] = 20;
        HIP_CHECK(hipMemcpy(da64, a64.data(), MEM_N * sizeof(long long), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(db64, b64.data(), MEM_N * sizeof(long long), hipMemcpyHostToDevice));
        for (int w = 0; w < 3; w++) { time_launch(lu); reset_flag(d_flag); time_launch(lc); }
        std::vector<float> tu, tc;
        for (int r = 0; r < 15; r++) { tu.push_back(time_launch(lu)); reset_flag(d_flag); tc.push_back(time_launch(lc)); }
        mem_rows.push_back({"sub int64 (sign-bit)", median(tu), median(tc)});
    }

    // ============================================================ memory-bound: int64 mul
    {
        std::uniform_int_distribution<long long> d(-2000000000LL, 2000000000LL);
        for (int i = 0; i < MEM_N; i++) { a64[i] = d(rng); b64[i] = d(rng); }
        HIP_CHECK(hipMemcpy(da64, a64.data(), MEM_N * sizeof(long long), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(db64, b64.data(), MEM_N * sizeof(long long), hipMemcpyHostToDevice));
        auto lu = [&]() { mul_i64_u<<<mem_grid, BLOCK>>>(da64, db64, do64u, MEM_N); };
        auto lc = [&]() { mul_i64_c<<<mem_grid, BLOCK>>>(da64, db64, do64c, d_flag, MEM_N); };
        reset_flag(d_flag); lc(); HIP_CHECK(hipDeviceSynchronize());
        expect(read_flag(d_flag).flag == 0, "mul_i64 in-range clear");
        lu(); HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(o64u.data(), do64u, MEM_N * sizeof(long long), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(o64c.data(), do64c, MEM_N * sizeof(long long), hipMemcpyDeviceToHost));
        expect(std::memcmp(o64u.data(), o64c.data(), MEM_N * sizeof(long long)) == 0, "mul_i64 checked==unchecked");
        for (int s = 0; s < 1000; s++) { int i = (int)(rng() % MEM_N); expect(o64u[i] == a64[i] * b64[i], "mul_i64 oracle"); }
        int P = 101010; a64[P] = 4000000000LL; b64[P] = 4000000000LL;  // 1.6e19 overflow
        HIP_CHECK(hipMemcpy(da64, a64.data(), MEM_N * sizeof(long long), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(db64, b64.data(), MEM_N * sizeof(long long), hipMemcpyHostToDevice));
        reset_flag(d_flag); lc(); HIP_CHECK(hipDeviceSynchronize());
        { FlagBuf f = read_flag(d_flag); expect(f.flag == 1 && f.idx == (unsigned)P, "mul_i64 planted +overflow"); }
        int Q = 2020; a64[P] = 3; b64[P] = 3; a64[Q] = -4000000000LL; b64[Q] = 4000000000LL;
        HIP_CHECK(hipMemcpy(da64, a64.data(), MEM_N * sizeof(long long), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(db64, b64.data(), MEM_N * sizeof(long long), hipMemcpyHostToDevice));
        reset_flag(d_flag); lc(); HIP_CHECK(hipDeviceSynchronize());
        { FlagBuf f = read_flag(d_flag); expect(f.flag == 1 && f.idx == (unsigned)Q, "mul_i64 planted -overflow"); }
        a64[Q] = 12; b64[Q] = 13;
        HIP_CHECK(hipMemcpy(da64, a64.data(), MEM_N * sizeof(long long), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(db64, b64.data(), MEM_N * sizeof(long long), hipMemcpyHostToDevice));
        for (int w = 0; w < 3; w++) { time_launch(lu); reset_flag(d_flag); time_launch(lc); }
        std::vector<float> tu, tc;
        for (int r = 0; r < 15; r++) { tu.push_back(time_launch(lu)); reset_flag(d_flag); tc.push_back(time_launch(lc)); }
        mem_rows.push_back({"mul int64 (mulhi)", median(tu), median(tc)});
    }

    // ============================================================ compute-bound cells
    // Fresh data-dependent inputs; correctness = in-range clear + memcmp.
    const int MASK32 = 0x7FFF;                 // operands <= 32767; product < INT_MAX
    const long long MASK64 = 0x3FFFFFFFLL;     // operands <= ~1.07e9; product < INT64_MAX
    auto run_cb_i32 = [&](const char* label, void (*ku)(const int*, int*, int, int), void (*kc)(const int*, int*, FlagBuf*, int, int)) {
        std::uniform_int_distribution<int> d(0, 2000000000);
        for (int i = 0; i < CB_N; i++) a32[i] = d(rng);
        HIP_CHECK(hipMemcpy(da32, a32.data(), CB_N * sizeof(int), hipMemcpyHostToDevice));
        auto lu = [&]() { ku<<<cb_grid, BLOCK>>>(da32, do32u, MASK32, CB_N); };
        auto lc = [&]() { kc<<<cb_grid, BLOCK>>>(da32, do32c, d_flag, MASK32, CB_N); };
        reset_flag(d_flag); lc(); HIP_CHECK(hipDeviceSynchronize());
        expect(read_flag(d_flag).flag == 0, "cb i32 in-range clear");
        lu(); HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(o32u.data(), do32u, CB_N * sizeof(int), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(o32c.data(), do32c, CB_N * sizeof(int), hipMemcpyDeviceToHost));
        expect(std::memcmp(o32u.data(), o32c.data(), CB_N * sizeof(int)) == 0, "cb i32 checked==unchecked");
        for (int w = 0; w < 3; w++) { time_launch(lu); reset_flag(d_flag); time_launch(lc); }
        std::vector<float> tu, tc;
        for (int r = 0; r < 15; r++) { tu.push_back(time_launch(lu)); reset_flag(d_flag); tc.push_back(time_launch(lc)); }
        cb_rows.push_back({label, median(tu), median(tc)});
    };
    auto run_cb_i64 = [&](const char* label, void (*ku)(const long long*, long long*, long long, int), void (*kc)(const long long*, long long*, FlagBuf*, long long, int)) {
        std::uniform_int_distribution<long long> d(0, 2000000000000000000LL);
        for (int i = 0; i < CB_N; i++) a64[i] = d(rng);
        HIP_CHECK(hipMemcpy(da64, a64.data(), CB_N * sizeof(long long), hipMemcpyHostToDevice));
        auto lu = [&]() { ku<<<cb_grid, BLOCK>>>(da64, do64u, MASK64, CB_N); };
        auto lc = [&]() { kc<<<cb_grid, BLOCK>>>(da64, do64c, d_flag, MASK64, CB_N); };
        reset_flag(d_flag); lc(); HIP_CHECK(hipDeviceSynchronize());
        expect(read_flag(d_flag).flag == 0, "cb i64 in-range clear");
        lu(); HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(o64u.data(), do64u, CB_N * sizeof(long long), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(o64c.data(), do64c, CB_N * sizeof(long long), hipMemcpyDeviceToHost));
        expect(std::memcmp(o64u.data(), o64c.data(), CB_N * sizeof(long long)) == 0, "cb i64 checked==unchecked");
        for (int w = 0; w < 3; w++) { time_launch(lu); reset_flag(d_flag); time_launch(lc); }
        std::vector<float> tu, tc;
        for (int r = 0; r < 15; r++) { tu.push_back(time_launch(lu)); reset_flag(d_flag); tc.push_back(time_launch(lc)); }
        cb_rows.push_back({label, median(tu), median(tc)});
    };

    run_cb_i32("add int32 (sign-bit)", add_i32_cb_u, add_i32_cb_c);
    run_cb_i32("mul int32 (widen)", mul_i32_cb_u, mul_i32_cb_c);
    run_cb_i64("add int64 (sign-bit)", add_i64_cb_u, add_i64_cb_c);
    run_cb_i64("mul int64 (mulhi)", mul_i64_cb_u, mul_i64_cb_c);

    print_table("Memory-bound (2^24 elements, one op per element):", mem_rows);
    print_table("Compute-bound ceiling (2^22 elements x 64 chained ops):", cb_rows);

    printf("\ncorrectness: %s\n", g_fail ? "FAIL (see stderr)" : "all cells passed");
    return g_fail;
}
