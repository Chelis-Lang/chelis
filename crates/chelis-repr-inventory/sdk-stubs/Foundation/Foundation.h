#ifndef STUB_FOUNDATION_H
#define STUB_FOUNDATION_H
#include <stddef.h>
#include <dispatch/dispatch.h>
typedef unsigned long NSUInteger;
typedef long NSInteger;
typedef signed char BOOL;
#define YES ((BOOL)1)
#define NO ((BOOL)0)
#ifndef nil
#define nil ((id)0)
#endif
@interface NSObject
+ (instancetype)alloc;
- (instancetype)init;
@end
@interface NSString : NSObject
+ (instancetype)stringWithFormat:(NSString *)format, ...;
- (const char *)UTF8String;
@end
@interface NSError : NSObject
- (NSString *)localizedDescription;
@end
@interface NSDictionary : NSObject
- (id)objectForKeyedSubscript:(id)key;
@end
@interface NSMutableDictionary : NSDictionary
+ (instancetype)dictionary;
- (void)setObject:(id)object forKeyedSubscript:(id)key;
@end
#endif
