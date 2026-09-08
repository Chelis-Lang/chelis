#ifndef STUB_HIPRTC_H
#define STUB_HIPRTC_H
#include <stddef.h>
typedef enum { HIPRTC_SUCCESS = 0, HIPRTC_ERROR_UNKNOWN = 999 } hiprtcResult;
typedef struct chelis_stub_hiprtcProgram *hiprtcProgram;
const char *hiprtcGetErrorString(hiprtcResult result);
hiprtcResult hiprtcCreateProgram(hiprtcProgram *prog, const char *src, const char *name, int n, const char **headers, const char **names);
hiprtcResult hiprtcCompileProgram(hiprtcProgram prog, int n, const char **options);
hiprtcResult hiprtcGetProgramLogSize(hiprtcProgram prog, size_t *size);
hiprtcResult hiprtcGetProgramLog(hiprtcProgram prog, char *log);
hiprtcResult hiprtcDestroyProgram(hiprtcProgram *prog);
hiprtcResult hiprtcGetCodeSize(hiprtcProgram prog, size_t *size);
hiprtcResult hiprtcGetCode(hiprtcProgram prog, char *code);
#endif
