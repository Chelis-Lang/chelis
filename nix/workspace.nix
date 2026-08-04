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
  regenerationSource = import ./source.nix {
    inherit lib root;
    includeRoots = [
      ".cargo"
      "Cargo.lock"
      "Cargo.nix"
      "Cargo.toml"
      "crate-hashes.json"
      "crate2nix.json"
      "crates"
      "rust-toolchain.toml"
      "scripts"
      "tree-sitter-chelis"
    ];
  };
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
  cargoNix = source + "/Cargo.nix";
  buildRustCrateForPkgs = import ./crate-overrides.nix {
    inherit
      crateSource
      cvc5
      pkgs
      toolchain
      ;
  };
  cargoGraph =
    assert crate2nixVersion == "0.15.0";
    import cargoNix {
      inherit buildRustCrateForPkgs pkgs;
      rootFeatures = [ ];
    };
  regenerationCheck = import ./crate2nix-regeneration.nix {
    inherit
      crate2nix
      pkgs
      toolchain
      ;
    source = regenerationSource;
  };
in
{
  inherit
    cargoGraph
    cargoNix
    crate2nixVersion
    regenerationCheck
    source
    toolchain
    version
    ;
}
