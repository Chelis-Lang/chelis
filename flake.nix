{
  description = "Chelis semantic core and pinned Aeneas verification tools";

  inputs = {
    aeneas.url = "github:AeneasVerif/aeneas/ee9bb3935375aacfa8ea3682fc21b051b79fc931";
    nixpkgs.url = "github:NixOS/nixpkgs/f205b5574fd0cb7da5b702a2da51507b7f4fdd1b";
    rust-overlay = {
      url = "github:oxalica/rust-overlay/89abdfd661ea493cde2f73d7b1332cb34150aa43";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { aeneas, nixpkgs, rust-overlay, ... }:
    let
      forAllSystems = nixpkgs.lib.genAttrs [
        "aarch64-darwin"
        "x86_64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];
      # Each verification/toolchain/<name>/SERIES lists that series' patches in
      # application order (generated and checked by verification/toolchain/series.py).
      series =
        name:
        let
          directory = ./verification/toolchain + "/${name}";
          lines = nixpkgs.lib.splitString "\n" (builtins.readFile (directory + "/SERIES"));
        in
        map (file: directory + "/${file}") (
          builtins.filter (line: line != "" && !nixpkgs.lib.hasPrefix "#" line) lines
        );
      # Every patch applies exactly to the pinned upstream source: no fuzz.
      patchFlags = [
        "-p1"
        "-F0"
      ];
      perSystem = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ rust-overlay.overlays.default ];
          };
          upstream = aeneas.packages.${system};
          modelRust = aeneas.inputs.charon.packages.${system}.rustToolchain;
          isNameMatcherParser = p: (p.pname or "") == "name_matcher_parser";
          patchedAeneasSource = pkgs.applyPatches {
            name = "aeneas-source";
            src = aeneas;
            patches = series "aeneas";
            inherit patchFlags;
          };
          tools = rec {
            charon = upstream.charon.overrideAttrs (old: {
              patches = (old.patches or [ ]) ++ series "charon";
              inherit patchFlags;
            });
            # charon-ml's name_matcher_parser is a separate dune package (a propagated input
            # built from the same source), so the charon-ml series must patch it too.
            name-matcher-parser =
              (nixpkgs.lib.findFirst isNameMatcherParser (throw "charon-ml has no name_matcher_parser input")
                upstream.charon-ml.propagatedBuildInputs
              ).overrideAttrs
                (old: {
                  patches = (old.patches or [ ]) ++ series "charon-ml";
                  inherit patchFlags;
                });
            charon-ml = upstream.charon-ml.overrideAttrs (old: {
              patches = (old.patches or [ ]) ++ series "charon-ml";
              inherit patchFlags;
              propagatedBuildInputs = map (
                p: if isNameMatcherParser p then name-matcher-parser else p
              ) old.propagatedBuildInputs;
            });
            aeneas = (upstream.aeneas.override { inherit charon charon-ml; }).overrideAttrs (_: {
              src = "${patchedAeneasSource}/src";
            });
            rust = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
            model-rustc = pkgs.writeShellScriptBin "model-rustc" ''
              exec ${modelRust}/bin/rustc "$@"
            '';
          };
        in
        {
          packages = tools // {
            default = tools.aeneas;
            aeneas-source = patchedAeneasSource;
          };
          devShells.default = pkgs.mkShell {
            packages = [
              tools.aeneas
              tools.charon
              tools.rust
              tools.model-rustc
              pkgs.elan
              pkgs.python311
            ];
          };
        }
      );
    in
    {
      packages = nixpkgs.lib.mapAttrs (_: value: value.packages) perSystem;
      devShells = nixpkgs.lib.mapAttrs (_: value: value.devShells) perSystem;
    };
}
