#ifndef STUB_DISPATCH_H
#define STUB_DISPATCH_H
typedef long dispatch_once_t;
typedef void (^dispatch_block_t)(void);
void dispatch_once(dispatch_once_t *predicate, dispatch_block_t block);
#endif
