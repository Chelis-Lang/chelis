A daily scheduled run of the Nix package workflow builds `main` and pushes the
packages to the `chelis-lang` Cachix cache. The run skips its native builds when
the cache already holds the `chelis`, `chelis-runtime`, and `chelisup` outputs of
that revision. Each pull request now runs the `Nix flake evaluation` check. It
evaluates the flake for every supported system with no builds and no import from
derivation, and it checks the format of the Nix files. See
[#3816](https://github.com/Chelis-Lang/chelis/issues/3816) and
[#1441](https://github.com/Chelis-Lang/chelis/issues/1441).
