#ifndef STUB_MPS_H
#define STUB_MPS_H
#import <Metal/Metal.h>
typedef NSUInteger MPSDataType;
enum { MPSDataTypeFloat32 = 0x10000020, MPSDataTypeFloat16 = 0x10000010 };
@interface MPSMatrixDescriptor : NSObject
+ (instancetype)matrixDescriptorWithRows:(NSUInteger)rows columns:(NSUInteger)columns rowBytes:(NSUInteger)rowBytes dataType:(MPSDataType)dataType;
@end
@interface MPSMatrix : NSObject
- (instancetype)initWithBuffer:(id<MTLBuffer>)buffer descriptor:(MPSMatrixDescriptor *)descriptor;
@end
@interface MPSMatrixMultiplication : NSObject
- (instancetype)initWithDevice:(id<MTLDevice>)device transposeLeft:(BOOL)tl transposeRight:(BOOL)tr resultRows:(NSUInteger)rows resultColumns:(NSUInteger)cols interiorColumns:(NSUInteger)inner alpha:(double)alpha beta:(double)beta;
- (void)encodeToCommandBuffer:(id<MTLCommandBuffer>)cb leftMatrix:(MPSMatrix *)a rightMatrix:(MPSMatrix *)b resultMatrix:(MPSMatrix *)c;
@end
#endif
