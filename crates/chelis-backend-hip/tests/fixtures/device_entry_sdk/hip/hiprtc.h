/* CPU execution fixture only; no runtime compilation is simulated. */
#ifndef CHELIS_ENTRY_FIXTURE_HIPRTC_H
#define CHELIS_ENTRY_FIXTURE_HIPRTC_H
#include <stddef.h>
typedef void *hiprtcProgram;
typedef enum { HIPRTC_SUCCESS = 0 } hiprtcResult;
extern "C" {
hiprtcResult hiprtcCreateProgram(hiprtcProgram *, const char *, const char *, int, const char **, const char **);
hiprtcResult hiprtcCompileProgram(hiprtcProgram, int, const char **);
hiprtcResult hiprtcGetProgramLogSize(hiprtcProgram, size_t *);
hiprtcResult hiprtcGetProgramLog(hiprtcProgram, char *);
hiprtcResult hiprtcGetCodeSize(hiprtcProgram, size_t *);
hiprtcResult hiprtcGetCode(hiprtcProgram, char *);
hiprtcResult hiprtcDestroyProgram(hiprtcProgram *);
const char *hiprtcGetErrorString(hiprtcResult);
}
#endif
