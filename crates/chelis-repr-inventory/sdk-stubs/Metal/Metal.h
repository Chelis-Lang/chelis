#ifndef STUB_METAL_H
#define STUB_METAL_H
#import <Foundation/Foundation.h>
typedef NSUInteger MTLResourceOptions;
enum { MTLResourceStorageModeShared = 0 };
typedef NSInteger MTLGPUFamily;
enum { MTLGPUFamilyApple1 = 1001, MTLGPUFamilyApple2, MTLGPUFamilyApple3, MTLGPUFamilyApple4, MTLGPUFamilyApple5, MTLGPUFamilyApple6, MTLGPUFamilyApple7, MTLGPUFamilyApple8, MTLGPUFamilyApple9 };
typedef struct { NSUInteger width, height, depth; } MTLSize;
static inline MTLSize MTLSizeMake(NSUInteger w, NSUInteger h, NSUInteger d) { MTLSize s = { w, h, d }; return s; }
@protocol MTLBuffer
- (void *)contents;
@end
@protocol MTLFunction
@end
@protocol MTLLibrary
- (id<MTLFunction>)newFunctionWithName:(NSString *)name;
@end
@protocol MTLComputePipelineState
- (NSUInteger)maxTotalThreadsPerThreadgroup;
@end
@protocol MTLComputeCommandEncoder
- (void)setComputePipelineState:(id<MTLComputePipelineState>)state;
- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index;
- (void)setBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index;
- (void)dispatchThreads:(MTLSize)threads threadsPerThreadgroup:(MTLSize)group;
- (void)endEncoding;
@end
@protocol MTLCommandBuffer
- (id<MTLComputeCommandEncoder>)computeCommandEncoder;
- (void)commit;
- (void)waitUntilCompleted;
@end
@protocol MTLCommandQueue
- (id<MTLCommandBuffer>)commandBuffer;
@end
@protocol MTLDevice
- (id<MTLCommandQueue>)newCommandQueue;
- (id<MTLLibrary>)newLibraryWithSource:(NSString *)source options:(id)options error:(NSError **)error;
- (id<MTLComputePipelineState>)newComputePipelineStateWithFunction:(id<MTLFunction>)function error:(NSError **)error;
- (BOOL)supportsFamily:(MTLGPUFamily)family;
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)length options:(MTLResourceOptions)options;
@end
id<MTLDevice> MTLCreateSystemDefaultDevice(void);
#endif
