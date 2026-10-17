`chelisup link <dir>` registers an external toolchain directory that contains
`bin/chelis`. It uses the version that `bin/chelis --version` reports, and the
`chelis` shim then runs that toolchain like an installed one. `chelisup
uninstall <version>` removes only the link, and it now rejects a version that
is not an exact `X.Y.Z`. The Nix `chelisup` package keeps a garbage-collector
root for a linked store path and removes it on uninstall. See
[#3815](https://github.com/Chelis-Lang/chelis/issues/3815).
