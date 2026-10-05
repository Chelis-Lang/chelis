/* Rust-facing shims for chelis-crmath (spec/design/correctly_rounded_math.md section 4.1).
 *
 * The amalgamation's definitions are all static; this unit includes it and exposes one
 * non-static shim per entry so the crate's Rust API can call the kernels. These symbols
 * live only inside the Rust binary that links chelis-crmath: they are not a published
 * C ABI and appear in no header. The `raw` shims bypass the NaN canonicalization and
 * exist only so the crate's tests can show that the wrapper is what canonicalizes. */

#include "crmath_amalgamation.c"

#define CHELIS_CRMATH_SHIM(name, ctype)                                             \
  ctype chelis_crmath_ffi_##name(ctype x) { return chelis_cr_##name(x); }           \
  ctype chelis_crmath_ffi_raw_##name(ctype x) { return chelis_cr_##name##__cr_##name(x); }

CHELIS_CRMATH_SHIM(expf, float)
CHELIS_CRMATH_SHIM(logf, float)
CHELIS_CRMATH_SHIM(sinf, float)
CHELIS_CRMATH_SHIM(cosf, float)
CHELIS_CRMATH_SHIM(tanf, float)
CHELIS_CRMATH_SHIM(atanf, float)
CHELIS_CRMATH_SHIM(tanhf, float)
CHELIS_CRMATH_SHIM(erff, float)
CHELIS_CRMATH_SHIM(erfcf, float)
CHELIS_CRMATH_SHIM(exp, double)
CHELIS_CRMATH_SHIM(log, double)
CHELIS_CRMATH_SHIM(sin, double)
CHELIS_CRMATH_SHIM(cos, double)
CHELIS_CRMATH_SHIM(tan, double)
CHELIS_CRMATH_SHIM(atan, double)
CHELIS_CRMATH_SHIM(tanh, double)
CHELIS_CRMATH_SHIM(erf, double)
CHELIS_CRMATH_SHIM(erfc, double)
