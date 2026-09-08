#ifndef STUB_STDLIB_H
#define STUB_STDLIB_H
#include <stddef.h>
void abort(void);
void exit(int status);
void *malloc(size_t size);
void *calloc(size_t count, size_t size);
void *realloc(void *pointer, size_t size);
void free(void *pointer);
int posix_memalign(void **memptr, size_t alignment, size_t size);
char *getenv(const char *name);
long strtol(const char *s, char **end, int base);
double strtod(const char *s, char **end);
#endif
