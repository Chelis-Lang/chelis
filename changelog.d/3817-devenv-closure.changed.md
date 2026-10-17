Runs of the Nix package workflow on `main` and on release tags now push the
Devenv shell closure to the `chelis-lang` Cachix cache. That closure holds Kache,
the Rust toolchain, and the shell profile. A later run on an unchanged shell
downloads these paths instead of a new build. The base Devenv shell no longer
builds the legacy schema-27 Kache fixture. Only the new `kache-regression`
profile builds it, for the Kache executable cache manual gate. See
[#3817](https://github.com/Chelis-Lang/chelis/issues/3817) and
[#3714](https://github.com/Chelis-Lang/chelis/issues/3714).
