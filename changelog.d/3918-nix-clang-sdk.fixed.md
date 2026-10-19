On macOS, `chelis build` no longer runs `/usr/bin/xcrun` or sets `SDKROOT` for
a C compiler in the Nix store. The Nix `chelis` package puts such a clang on
`PATH`, and that compiler uses the SDK of its Nix wrapper. Previously the
lines that `--emit-c` prints named the Xcode SDK, but the compile used the Nix
SDK. A host compiler such as `/usr/bin/cc` still gets the SDK of the
`xcode-select` default. See [#3918](https://github.com/Chelis-Lang/chelis/issues/3918).
