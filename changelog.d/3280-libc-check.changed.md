`chelis build` refuses a C compiler that compiles against another C library than
the one its carried runtime archive was built for, before compiling anything, and
names both C libraries. The Linux releases carry a glibc runtime, so on a musl
system such as Alpine, `chelis build` with the system compiler reports that
mismatch instead of failing at the compiler check or at link time. See
[#3280](https://github.com/Chelis-Lang/chelis/issues/3280).
