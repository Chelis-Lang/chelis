Generated C includes `chelis_simd.h` only in a unit that calls one of its
helpers, and `chelis_runtime.h` no longer includes it. On aarch64, that header
includes `arm_neon.h`. With GCC 16 on aarch64-linux, the C unit of a small
integer program now compiles in 39 ms instead of 136 ms. With Apple clang, it
compiles in 39 ms instead of 77 ms.

`chelis build` compiles the unit while the compiler canary runs, so its wall
time changes only where the unit takes longer than the canary. A C program that
calls these helpers must now include `chelis_simd.h` itself. See
[#3704](https://github.com/Chelis-Lang/chelis/issues/3704).
