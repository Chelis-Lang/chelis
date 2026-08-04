{
  config,
  pkgs,
  lib,
  ...
}:
let
  crate2nix = config.lib.getInput {
    name = "crate2nix";
    url = "github:nix-community/crate2nix";
    attribute = "chelis.rust.importWorkspace";
  };
in
{
  options.chelis.rust.importWorkspace = lib.mkOption {
    type = lib.types.functionTo (lib.types.functionTo lib.types.attrs);
    description = "Import the filtered Chelis Cargo workspace as one crate2nix graph.";
  };

  options.chelis.rust.workspaceGraph = lib.mkOption {
    type = lib.types.attrs;
    internal = true;
    description = "The lazy Chelis workspace graph for native contract checks.";
  };

  config.chelis.rust.importWorkspace =
    workspace:
    { cvc5 }:
    import ../nix/workspace.nix {
      inherit
        crate2nix
        cvc5
        lib
        pkgs
        ;
      root = workspace;
      toolchain = config.languages.rust.toolchainPackage;
    };
}
