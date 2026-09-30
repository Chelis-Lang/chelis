`chelis runtime export <dir>` writes the runtime archive and public runtime
headers that the `chelis` binary carries, with a receipt recording their
SHA-256 digests. Release tarballs, the ecosystem container toolchain and both
Nix runtime outputs verify the copied archive and all six public headers
against that compiler's sealed export. Crossed or missing files reject before
publication or installation; the installed-artifact canary also refuses any
staged receipt that omits, adds or misstates a public header digest. See
[#1354](https://github.com/Chelis-Lang/chelis/issues/1354).
