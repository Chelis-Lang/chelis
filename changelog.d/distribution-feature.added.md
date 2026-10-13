`chelis-cli` has a `distribution` feature, the feature set of every distributed
compiler: `smt`, `sealed-runtime`, and `clarabel-provider`. The release workflow
and the Nix flake `chelis` package build it. A downstream package selects the
same set with `--features distribution`.
