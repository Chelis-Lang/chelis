{
  config,
  pkgs,
  lib,
  ...
}:
let
  source = import ../nix/source.nix {
    inherit lib;
    root = ../.;
  };
  cvc5 = import ../nix/cvc5.nix {
    inherit pkgs lib;
    root = source;
  };
  graph = config.chelis.rust.importWorkspace ../. { inherit cvc5; };
  requireWorkspaceMember =
    name:
    if builtins.hasAttr name graph.cargoGraph.workspaceMembers then
      builtins.getAttr name graph.cargoGraph.workspaceMembers
    else
      throw "the Devenv graph is missing required workspace member ${name}";
  compilerCrate = (requireWorkspaceMember "chelis-cli").build.override {
    features = [ "smt" ];
  };
  runtimeCrate = (requireWorkspaceMember "chelis-runtime").build.override {
    features = [ ];
  };
  chelisupCrate = (requireWorkspaceMember "chelisup").build.override {
    features = [ ];
  };
  artifacts = import ../nix/artifacts.nix {
    inherit
      chelisupCrate
      compilerCrate
      lib
      pkgs
      runtimeCrate
      ;
    inherit (graph) source version;
  };
in
{
  chelis.rust.workspaceGraph = graph;

  outputs = rec {
    chelis = artifacts.chelis;
    chelis-runtime = artifacts.runtime;
    chelisup = artifacts.chelisup;
    default = chelis;
  };
}
