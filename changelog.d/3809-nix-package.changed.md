The Nix `chelis` package carries a C compiler and binutils from Nix. Its `bin/chelis` is a wrapper that puts them first on `PATH`, and the compiler is `libexec/chelis`. Thus `chelis build` runs without a host C toolchain.

The flake also has `overlays.default`, `formatter.<system>`, and package metadata. A documentation edit no longer changes a package derivation. The install guide has a Nix section with the binary cache settings. See [#3809](https://github.com/Chelis-Lang/chelis/issues/3809).
