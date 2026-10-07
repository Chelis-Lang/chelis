The correctly rounded `sin`, `cos`, and `tan` kernels at f32 and `sin`, `exp`,
and `erfc` at f64 no longer call the C library's `roundeven`. On baseline x86-64,
that call made the compiler canary, which every native build compiles first, fail
to link on musl and on glibc before 2.25. GCC 10 and 11 make the same call on
aarch64. The kernels now round their reduced argument to an integer with `rint`,
which every C99 C library has. In the rounding mode that Chelis pins, `rint`
gives the same value, so results are unchanged. See
[#3280](https://github.com/Chelis-Lang/chelis/issues/3280).
