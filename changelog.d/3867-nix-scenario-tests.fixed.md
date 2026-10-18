The install guide lists the three Nix flake systems `x86_64-linux`,
`aarch64-linux`, and `aarch64-darwin`. It also says that `x86_64-darwin` is not
supported. The flake contract suite now fails when the guide omits one of these
statements. New negative checks cover three more cases. These are a Linux BLAS
build without the OpenBLAS variables, a failed `chelisup uninstall` of a linked
toolchain, and a crate build without a declared external file. See
[#3867](https://github.com/Chelis-Lang/chelis/issues/3867).
