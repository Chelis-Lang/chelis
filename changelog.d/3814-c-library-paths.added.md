`CHELIS_C_INCLUDE_PATH` and `CHELIS_C_LIBRARY_PATH` declare the directories
of external C headers and libraries, such as OpenBLAS. Each is a
colon-separated list of absolute directories. Each include directory adds
`-isystemDIR` to the C compile. Each library directory adds `-LDIR` and
`-Wl,-rpath,DIR` to the C link.

The C builds of `chelis build`, the compiler check and the C compile of the
Python bindings use these flags. So does the host program of a HIP or Metal
build or of a HIP binding. A device-only HIP or Metal build or HIP binding
compiles no C source and does not use them. A relative entry fails a build that
uses them. On Linux, the `chelis` Nix package declares the OpenBLAS directories,
so a program that calls BLAS builds and runs. See
[#3814](https://github.com/Chelis-Lang/chelis/issues/3814) and
[#3893](https://github.com/Chelis-Lang/chelis/issues/3893).
