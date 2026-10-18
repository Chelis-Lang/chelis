The flake exports a `chelis-unwrapped` package and the overlay adds it. It holds
the compiler, the runtime archive, and the six runtime headers in the layout of a
release toolchain. It has no C compiler, binutils, or OpenBLAS. Use it when the
host supplies the C toolchain. It runs the same compiler binary as the `chelis`
package. See [#3911](https://github.com/Chelis-Lang/chelis/issues/3911).
