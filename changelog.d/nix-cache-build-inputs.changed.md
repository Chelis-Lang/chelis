The `chelis-lang` Cachix cache also receives the cvc5 tree and the Rust crate
builds of the Nix packages. A Nix build of a later revision downloads the
unchanged ones and compiles only the crates that the change affects. A
maintainer can fill the cache between releases with a dispatch of
`nix-packages.yml` on `main`. See
[#3610](https://github.com/Chelis-Lang/chelis/issues/3610).
