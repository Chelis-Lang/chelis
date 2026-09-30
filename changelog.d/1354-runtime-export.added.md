`chelis runtime export <dir>` writes the runtime archive and public runtime
headers that the `chelis` binary carries, with a receipt recording their
SHA-256 digests. Release tarballs, the ecosystem container toolchain and both
Nix runtime outputs verify the copied archive and all six public headers
against that compiler's sealed export; native Nix checks link and run a C
consumer against each output's own include directory and archive. Crossed or
missing files reject before publication and, for export-capable releases,
before installation. Releases through 0.18.11 retain their unchecked-runtime
warning. The installed-artifact canary also refuses any staged receipt that
omits, adds or misstates a public header digest. See
[#1354](https://github.com/Chelis-Lang/chelis/issues/1354).
