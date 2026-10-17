`CHELIS_C_INCLUDE_PATH` and `CHELIS_C_LIBRARY_PATH` declare the directories
of external C headers and libraries, such as OpenBLAS. Each is a
colon-separated list of absolute directories. Each include directory adds
`-isystemDIR` to the C compile. Each library directory adds `-LDIR` and
`-Wl,-rpath,DIR` to the C link.

`chelis build`, the compiler check and the Python bindings use these flags. A
relative or empty entry fails the build. On Linux, the `chelis` Nix package
declares the OpenBLAS directories, so a program that calls BLAS builds and
runs. See [#3814](https://github.com/Chelis-Lang/chelis/issues/3814).
