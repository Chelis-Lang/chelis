The `chelisup` of the Nix flake now removes the Nix garbage-collector root of a
linked store toolchain when `chelisup install` replaces that link with a release
toolchain. Previously the root stayed until `chelisup uninstall`, and it kept the
old store closure alive. A failed install keeps the root. The install guide now
says that `CHELIS_CC` and `CHELIS_AR` need a path to select a host tool with the
Nix `chelis` package. See
[#3890](https://github.com/Chelis-Lang/chelis/issues/3890) and
[#3894](https://github.com/Chelis-Lang/chelis/issues/3894).
