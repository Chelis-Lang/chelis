{
  inputs,
  lib,
  pkgs,
  toolchain,
  ...
}:

let
  ciPkgs = import inputs.ci-nixpkgs { system = pkgs.stdenv.system; };
  actionPackageAuthority = import ./packages.nix {
    inherit ciPkgs toolchain;
  };
in
{
  imports = [ ./profiles.nix ];

  _module.args = {
    ciPkgs = lib.mkDefault ciPkgs;
    inherit actionPackageAuthority;
    sharedContainerConstructors = import ./container.nix;
  };

  packages = actionPackageAuthority.common;
}
