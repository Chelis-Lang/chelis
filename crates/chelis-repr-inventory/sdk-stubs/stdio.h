#ifndef STUB_STDIO_H
#define STUB_STDIO_H
#include <stddef.h>
typedef struct chelis_stub_FILE FILE;
extern FILE *stderr;
extern FILE *stdout;
int fprintf(FILE *stream, const char *format, ...);
int printf(const char *format, ...);
int snprintf(char *buffer, size_t size, const char *format, ...);
int fputs(const char *s, FILE *stream);
int fflush(FILE *stream);
#endif
