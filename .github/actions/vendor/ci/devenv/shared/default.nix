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

  # The portable base carries only common tools. Devenv's defaults would add a
  # Nix compiler wrapper (and on macOS a Nix Apple SDK) to every shell, and
  # anything it links then depends on /nix/store. A profile opts back into that
  # toolchain by declaring `c-toolchain = "nix"`; see profiles.nix.
  stdenv = lib.mkDefault pkgs.stdenvNoCC;
  apple.sdk = lib.mkDefault null;
}
