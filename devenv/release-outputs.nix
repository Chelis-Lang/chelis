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
  isLinux = pkgs.stdenv.hostPlatform.system == "x86_64-linux";
in
{
  config = lib.mkIf supported (
    lib.mkMerge [
      {
        outputs.release-chelisup = import ../nix/release-chelisup.nix {
          inherit lib pkgs;
          root = ../.;
          rustToolchain = config.languages.rust.toolchainPackage;
        };
      }
      (lib.mkIf isLinux {
        outputs.release-chelis = import ../nix/release-chelis.nix {
          inherit lib pkgs;
          root = ../.;
          rustToolchain = config.languages.rust.toolchainPackage;
        };
      })
    ]
  );
}
