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
      perSystem = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ rust-overlay.overlays.default ];
          };
          upstream = aeneas.packages.${system};
          modelRust = aeneas.inputs.charon.packages.${system}.rustToolchain;
          patchedAeneasSource = pkgs.applyPatches {
            name = "aeneas-source";
            src = aeneas;
            patches = [ ./verification/aeneas-stdlib-models.patch ];
          };
          tools = rec {
            charon = upstream.charon.overrideAttrs (old: {
              patches = (old.patches or [ ]) ++ [
                ./verification/charon-box-init.patch
                ./verification/charon-closure-clauses.patch
                ./verification/charon-nonlocal-continuations.patch
                ./verification/charon-local-trait-bounds.patch
                ./verification/charon-closure-regions.patch
                ./verification/charon-hidden-traits.patch
              ];
            });
            charon-ml = upstream.charon-ml.overrideAttrs (old: {
              patches = (old.patches or [ ]) ++ [
                ./verification/charon-ml-binders.patch
              ];
            });
            aeneas = (upstream.aeneas.override { inherit charon charon-ml; }).overrideAttrs (old: {
              src = "${patchedAeneasSource}/src";
              patches = (old.patches or [ ]) ++ [
                ./verification/aeneas-strict-warnings.patch
                ./verification/aeneas-loop-continuations.patch
                ./verification/aeneas-trait-models.patch
                ./verification/aeneas-shared-borrows.patch
                ./verification/aeneas-floating-scalars.patch
                ./verification/aeneas-collection-contract.patch
              ];
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
