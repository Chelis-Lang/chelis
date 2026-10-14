Published releases push the Nix `chelis`, `chelis-runtime`, and `chelisup`
packages to the `chelis-lang` Cachix cache, and `flake.nix` names that cache.
After you accept the flake configuration, `nix run` and `nix build` of a
released revision download these packages and do not compile cvc5 and the Rust
crates. See [#3610](https://github.com/Chelis-Lang/chelis/issues/3610).
