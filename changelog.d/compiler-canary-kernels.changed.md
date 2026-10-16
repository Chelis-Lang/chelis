The floating-point canary of `chelis build` now checks only the correctly rounded
kernels that the program's generated C carries, plus one sentinel kernel. It still
runs every arithmetic, comparison, conversion, and expression-shape row. With GCC 16
on aarch64-linux, a build of a program without kernels took 311 ms instead of 873 ms.
With Apple clang on macOS, it took 429 ms instead of 556 ms. See
[#3712](https://github.com/Chelis-Lang/chelis/issues/3712).
