{
  config,
  lib,
  pkgs,
  ...
}:
let
  supported = builtins.elem pkgs.stdenv.hostPlatform.system [
    "x86_64-linux"
    "aarch64-darwin"
  ];
in
{
  config = lib.mkIf supported {
    outputs.release-chelisup = import ../nix/release-chelisup.nix {
      inherit lib pkgs;
      root = ../.;
      rustToolchain = config.languages.rust.toolchainPackage;
    };
  };
}
