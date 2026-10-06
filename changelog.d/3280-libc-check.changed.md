`chelis build` refuses a C compiler that compiles against another C library than
the one its carried runtime archive was built for, before compiling anything, and
names that C library. The Linux glibc builds refuse a musl compiler and the musl
build refuses a glibc one, instead of failing at the compiler check or at link
time. On Alpine `chelisup` now installs the musl build, whose `chelis build` uses
the system compiler. See
[#3280](https://github.com/Chelis-Lang/chelis/issues/3280).
