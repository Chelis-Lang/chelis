{
  pkgs,
  lib,
  root,
  crate2nix,
  cvc5,
}:
let
  source = import ./source.nix { inherit lib root; };
  toolchain = pkgs.rust-bin.fromRustupToolchainFile (source + "/rust-toolchain.toml");
  workspace = import ./workspace.nix {
    inherit
      crate2nix
      cvc5
      lib
      pkgs
      root
      toolchain
      ;
  };
  requireWorkspaceMember =
    name:
    if builtins.hasAttr name workspace.cargoGraph.workspaceMembers then
      builtins.getAttr name workspace.cargoGraph.workspaceMembers
    else
      throw "the crate2nix graph is missing required workspace member ${name}";
  compilerCrate = (requireWorkspaceMember "chelis-cli").build.override {
    features = [ "smt" ];
  };
  runtimeCrate = (requireWorkspaceMember "chelis-runtime").build.override {
    features = [ ];
  };
  chelisupCrate = (requireWorkspaceMember "chelisup").build.override {
    features = [ ];
  };
  artifacts = import ./artifacts.nix {
    inherit
      chelisupCrate
      compilerCrate
      lib
      pkgs
      runtimeCrate
      ;
    inherit (workspace) source version;
  };
in
workspace
// artifacts
// {
  inherit
    chelisupCrate
    compilerCrate
    runtimeCrate
    ;
}
