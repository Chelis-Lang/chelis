# The flake package links the same cvc5 tree as a release build.
# cvc5-sys 0.3.1 runs configure.sh --static --auto-download -DBUILD_GMP=1.
# That configuration uses the Production build type, LibPoly on, and GPL off.
# nix/ci-cvc5.nix builds that configuration with Nix dependencies.
args@{
  pkgs,
  lib,
  root,
}:
import ./ci-cvc5.nix args
