The correctly rounded `sin`, `cos`, and `tan` kernels at f32 and `sin`, `exp`,
and `erfc` at f64 no longer call the C library's `roundeven`: they round their
reduced argument to an integer with arithmetic of their own. On baseline x86-64
that call kept the compiler canary, and with it every native build, from linking
on a C library without `roundeven`, such as musl or glibc before 2.25. Results
are unchanged. See [#3280](https://github.com/Chelis-Lang/chelis/issues/3280).
