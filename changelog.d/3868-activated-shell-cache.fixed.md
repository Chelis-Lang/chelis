Runs of the Nix package workflow on `main` and on release tags now push the
activated Devenv shell and its shell environment to the `chelis-lang` Cachix
cache. Before this change, they pushed the shell of the `devenv test` smoke step
in place of the activated shell. Thus each run built six small shell derivations
again during activation.
`scripts/ci_devenv.py` now keeps the `.devenv/gc/activated-shell` and
`.devenv/gc/activated-shell-env` roots. See
[#3868](https://github.com/Chelis-Lang/chelis/issues/3868).
