{
  pkgs,
  lib,
  root,
  crate2nix,
  toolchain,
  cvc5,
}:
let
  source = import ./source.nix { inherit lib root; };
  crateSource = import ./source.nix {
    inherit lib root;
    includeRoots = [
      "crates"
      "grammars"
      "tree-sitter-chelis"
    ];
  };
  manifest = lib.importTOML (source + "/Cargo.toml");
  version = manifest.workspace.package.version;
  crate2nixManifest = lib.importTOML (crate2nix + "/crate2nix/Cargo.toml");
  crate2nixVersion =
    assert crate2nixManifest.package.version == "0.15.0";
    crate2nixManifest.package.version;
  crate2nixTools = pkgs.callPackage (crate2nix + "/tools.nix") { };
  buildRustCrateForPkgs = import ./crate-overrides.nix {
    inherit
      crateSource
      cvc5
      pkgs
      toolchain
      ;
  };
  generatedCargoNix =
    assert crate2nixVersion == "0.15.0";
    (crate2nixTools.generatedCargoNix {
      name = "chelis";
      src = source;
      cargo = toolchain;
      additionalCargoNixArgs = [
        "--no-default-features"
        "--features"
        "chelis-cli/smt"
      ];
    }).overrideAttrs
      (_: {
        CARGO_NET_OFFLINE = "true";
      });
  cargoGraph = import generatedCargoNix {
    inherit buildRustCrateForPkgs pkgs;
    rootFeatures = [ ];
  };
in
{
  inherit
    cargoGraph
    crate2nixVersion
    generatedCargoNix
    source
    toolchain
    version
    ;
}
