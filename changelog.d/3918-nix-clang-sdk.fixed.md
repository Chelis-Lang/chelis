On macOS, `chelis build` no longer runs `/usr/bin/xcrun` or sets `SDKROOT` for
a C compiler or `ar` of a Nix cc or binutils wrapper. The Nix `chelis` package
puts such a clang on `PATH`, and that compiler uses the SDK of its Nix wrapper.
Previously the lines that `--emit-c` prints named the Xcode SDK, but the
compile used the Nix SDK. See [#3918](https://github.com/Chelis-Lang/chelis/issues/3918).

An unwrapped compiler in the Nix store, named in `CHELIS_CC`, and a host
compiler such as `/usr/bin/cc` still get the SDK of the `xcode-select` default.
See [#3948](https://github.com/Chelis-Lang/chelis/issues/3948).
