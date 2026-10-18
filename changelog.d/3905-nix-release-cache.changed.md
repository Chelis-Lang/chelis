The release workflow now starts the Nix package checks of a release tag right
after its release notes preflight, beside the release builds. Before, it started
them only after it published the release. So the `chelis-lang` Cachix cache can
hold the Nix packages of a release when the release becomes public. Thus `nix run`
on a new tag builds less from source. The evaluation-only Nix jobs install Nix
without Devenv, and the CaDiCaL build of the cvc5 tree no longer runs its test
suites. See [#3905](https://github.com/Chelis-Lang/chelis/issues/3905),
[#3907](https://github.com/Chelis-Lang/chelis/issues/3907), and
[#3908](https://github.com/Chelis-Lang/chelis/issues/3908).
