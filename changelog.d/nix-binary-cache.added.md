Published releases push the Nix `chelis`, `chelis-runtime`, and `chelisup`
packages to the `chelis-lang` Cachix cache, and `flake.nix` names that cache.
After `release.yml` publishes a release, it starts the native Nix checks on the
release tag, and that run pushes the packages.
When Nix trusts that cache, `nix run` and `nix build` of a released revision
download these packages and do not compile cvc5 and the Rust crates. A trusted
Nix user passes `--accept-flake-config`. On a multi-user install, an
administrator adds the cache to the system Nix configuration. See
[#3610](https://github.com/Chelis-Lang/chelis/issues/3610).
