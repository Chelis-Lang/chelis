`chelis runtime export <dir>` writes the runtime archive and public runtime
headers that the `chelis` binary carries, with a receipt recording the archive's
SHA-256. Release tarballs and the Nix `chelis` package ship its output, so their
`lib/libchelis_runtime.a` is the archive `chelis build` stages. See
[#1354](https://github.com/Chelis-Lang/chelis/issues/1354).
