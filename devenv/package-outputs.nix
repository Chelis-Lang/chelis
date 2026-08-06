{ pkgs, ... }:

let
  # The Devenv Rust importer creates a separate crate2nix graph for one Cargo root.
  # The public flake keeps the multi-artifact package contract authoritative.
  # Devenv evaluates this unlocked local Git input in impure mode.
  repoFlake = builtins.getFlake "git+file://${toString ../.}";
  system = pkgs.stdenv.hostPlatform.system;
  packageNames = (import ../nix/contracts.nix).packageNames;
  repoPackages = repoFlake.packages.${system};
  mkOutput = name: {
    inherit name;
    value = repoPackages.${name};
  };
in
{
  outputs = builtins.listToAttrs (builtins.map mkOutput packageNames);
}
