#ifndef CHELIS_METAL_RUNTIME_H
#define CHELIS_METAL_RUNTIME_H

/* Objective-C++ runtime for the Chelis Metal backend.
 *
 * Peer of `chelis_hip_runtime.h`. Compiled into emitted .mm host files via
 * `clang++ -fobjc-arc -framework Metal -framework Foundation`. There is NO
 * Rust crate dep that wraps Metal — the backend emits pure strings; this
 * header is the only Apple-SDK touchpoint, and it is consumed by the user's
 * clang++, not by cargo.
 *
 * Architectural mapping HIP -> Metal (mirrors hiprtc):
 *   hipMalloc / hipFree                -> [device newBufferWithLength:options:]
 *   hipMemcpy{H2D,D2H}                 -> memcpy over [buf contents]
 *                                         (StorageModeShared = unified memory)
 *   hiprtcCreateProgram + Compile      -> [device newLibraryWithSource:options:error:]
 *   hipModuleLoadData                  -> the same call returns id<MTLLibrary>
 *   hipModuleGetFunction               -> [lib newFunctionWithName:]
 *   hipModuleLaunchKernel              -> [encoder dispatchThreads:threadsPerThreadgroup:]
 *
 * See `spec/design/chelis_metal_backend_plan.md` §3.3.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "chelis_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Device + queue (lazily initialized singletons) ---- */

static inline id<MTLDevice> chelis_metal_device(void) {
    static id<MTLDevice> dev = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            fprintf(stderr, "chelis Metal: MTLCreateSystemDefaultDevice() returned nil\n");
            abort();
        }
    });
    return dev;
}

static inline id<MTLCommandQueue> chelis_metal_queue(void) {
    static id<MTLCommandQueue> q = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        q = [chelis_metal_device() newCommandQueue];
    });
    return q;
}

/* ---- Pipeline cache keyed by (source-pointer-identity, function-name) ----
 *
 * The emitter stores each unique kernel as a single `static NSString *const`
 * raw-string literal, so the source pointer is a stable cache key. For
 * dynamically generated sources (none in M1-M5), pass distinct keys.
 */

static inline id<MTLComputePipelineState>
chelis_metal_get_pipeline(NSString *src, NSString *fn) {
    static NSMutableDictionary *cache = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        cache = [NSMutableDictionary dictionary];
    });

    NSString *key = [NSString stringWithFormat:@"%p::%@", (void *)src, fn];
    id<MTLComputePipelineState> pso = nil;
    @synchronized (cache) {
        pso = cache[key];
        if (!pso) {
            NSError *err = nil;
            id<MTLLibrary> lib = [chelis_metal_device()
                newLibraryWithSource:src options:nil error:&err];
            if (!lib) {
                fprintf(stderr, "chelis Metal: MSL compile failed for `%s`: %s\n",
                        [fn UTF8String],
                        [[err localizedDescription] UTF8String] ?: "(no error info)");
                abort();
            }
            id<MTLFunction> func = [lib newFunctionWithName:fn];
            if (!func) {
                fprintf(stderr, "chelis Metal: function `%s` not found in compiled library\n",
                        [fn UTF8String]);
                abort();
            }
            pso = [chelis_metal_device()
                newComputePipelineStateWithFunction:func error:&err];
            if (!pso) {
                fprintf(stderr, "chelis Metal: pipeline state for `%s` failed: %s\n",
                        [fn UTF8String],
                        [[err localizedDescription] UTF8String] ?: "(no error info)");
                abort();
            }
            cache[key] = pso;
        }
    }
    return pso;
}

/* ---- Buffer alloc over StorageModeShared (unified memory) ----
 *
 * Buffers are owned by ARC at the call site of chelis_metal_alloc. When
 * the local `id<MTLBuffer>` strong reference goes out of scope (typically
 * at the end of the emitted entrypoint function), ARC releases the
 * buffer automatically. There is intentionally NO `chelis_metal_free`
 * function: any explicit free call on an ARC-owned id is undefined
 * behavior, and an empty `(void)buf` body would mislead contributors
 * into thinking it does something. If a future phase wants explicit
 * lifetime control (e.g., for a planner-driven slot reuse pattern), the
 * right move is to switch the buffer-handle type to a non-ARC-managed
 * value (e.g. `__bridge_retained void*`) at that boundary, not to
 * resurrect a no-op free.
 */

static inline id<MTLBuffer> chelis_metal_alloc(size_t bytes) {
    if (bytes == 0) bytes = 1; /* Metal does not allow zero-length buffers */
    id<MTLBuffer> buf = [chelis_metal_device()
        newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!buf) {
        fprintf(stderr, "chelis Metal: newBufferWithLength(%zu) returned nil\n", bytes);
        abort();
    }
    return buf;
}

/* ---- Host <-> device transfer (memcpy on Apple Silicon) ---- */

static inline void chelis_metal_host_to_device(id<MTLBuffer> dst,
                                               const void *src, size_t bytes) {
    memcpy([dst contents], src, bytes);
}

static inline void chelis_metal_device_to_host(void *dst,
                                               id<MTLBuffer> src, size_t bytes) {
    memcpy(dst, [src contents], bytes);
}

/* ---- Encode and dispatch one kernel ----
 *
 * Buffers are bound at indices 0..n_buffers-1. A single contiguous uniforms
 * blob (typically packed scalar dims and counts) is bound at index n_buffers
 * via setBytes:.
 */

static inline void chelis_metal_launch(
    id<MTLComputePipelineState> pso,
    NSUInteger grid_x, NSUInteger tg_x,
    /* __unsafe_unretained so callers can pass strong-typed local arrays
     * (e.g. `id<MTLBuffer> bufs[3] = { ... }`) without ARC complaining
     * about __autoreleasing write-back. The callee never retains past
     * the dispatch, so unsafe_unretained is the right qualifier. */
    __unsafe_unretained id<MTLBuffer> *buffers, NSUInteger n_buffers,
    const void *uniforms, NSUInteger uniforms_bytes
) {
    id<MTLCommandBuffer> cb = [chelis_metal_queue() commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    for (NSUInteger i = 0; i < n_buffers; ++i) {
        [enc setBuffer:buffers[i] offset:0 atIndex:i];
    }
    if (uniforms_bytes > 0) {
        [enc setBytes:uniforms length:uniforms_bytes atIndex:n_buffers];
    }
    NSUInteger max_tg = [pso maxTotalThreadsPerThreadgroup];
    if (tg_x > max_tg) tg_x = max_tg;
    /* Zero from the emitter is a planner bug, not a runtime concern.
     * In debug builds we abort with a clear message so the regression
     * surfaces immediately; in release we conservatively clamp to 1
     * so a user-deployed binary doesn't crash on a stale build. */
    if (tg_x == 0 || grid_x == 0) {
#ifndef NDEBUG
        fprintf(stderr, "chelis Metal: chelis_metal_launch received zero "
                        "grid_x=%lu / tg_x=%lu — emitter planner bug\n",
                (unsigned long)grid_x, (unsigned long)tg_x);
        abort();
#else
        if (tg_x == 0) tg_x = 1;
        if (grid_x == 0) grid_x = 1;
#endif
    }
    MTLSize grid = MTLSizeMake(grid_x, 1, 1);
    MTLSize tg   = MTLSizeMake(tg_x, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

/* ---- 2D launch (used by the tiled matmul kernel) ----
 *
 * Dispatches a (grid_x, grid_y) compute grid with (tg_x, tg_y)
 * threadgroup size. Buffers and uniforms bind exactly as in the 1D
 * launch.
 */

static inline void chelis_metal_launch2d(
    id<MTLComputePipelineState> pso,
    NSUInteger grid_x, NSUInteger grid_y,
    NSUInteger tg_x, NSUInteger tg_y,
    __unsafe_unretained id<MTLBuffer> *buffers, NSUInteger n_buffers,
    const void *uniforms, NSUInteger uniforms_bytes
) {
    id<MTLCommandBuffer> cb = [chelis_metal_queue() commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    for (NSUInteger i = 0; i < n_buffers; ++i) {
        [enc setBuffer:buffers[i] offset:0 atIndex:i];
    }
    if (uniforms_bytes > 0) {
        [enc setBytes:uniforms length:uniforms_bytes atIndex:n_buffers];
    }
    NSUInteger max_tg = [pso maxTotalThreadsPerThreadgroup];
    NSUInteger tg_total = tg_x * tg_y;
    if (tg_total > max_tg) {
        /* For the tiled matmul we expect tg_total = 16*16 = 256, well
         * under the typical 1024 ceiling, so this is defensive. */
#ifndef NDEBUG
        fprintf(stderr, "chelis Metal: chelis_metal_launch2d threadgroup "
                        "%lu*%lu=%lu exceeds pso max %lu\n",
                (unsigned long)tg_x, (unsigned long)tg_y,
                (unsigned long)tg_total, (unsigned long)max_tg);
        abort();
#else
        tg_x = 1;
        tg_y = 1;
#endif
    }
    if (tg_x == 0 || tg_y == 0 || grid_x == 0 || grid_y == 0) {
#ifndef NDEBUG
        fprintf(stderr, "chelis Metal: chelis_metal_launch2d received "
                        "zero grid (%lu,%lu) / tg (%lu,%lu) — planner bug\n",
                (unsigned long)grid_x, (unsigned long)grid_y,
                (unsigned long)tg_x, (unsigned long)tg_y);
        abort();
#else
        if (tg_x == 0) tg_x = 1;
        if (tg_y == 0) tg_y = 1;
        if (grid_x == 0) grid_x = 1;
        if (grid_y == 0) grid_y = 1;
#endif
    }
    MTLSize grid = MTLSizeMake(grid_x, grid_y, 1);
    MTLSize tg   = MTLSizeMake(tg_x, tg_y, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

#endif /* CHELIS_METAL_RUNTIME_H */
