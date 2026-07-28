{
  description = "Chelis compiler, runtime, and toolchain installer packages";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/f205b5574fd0cb7da5b702a2da51507b7f4fdd1b";
    rust-overlay = {
      url = "github:oxalica/rust-overlay/19a19f3921ae195f2fbd85f5dc57e6d1df63aa0b";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      rust-overlay,
    }:
    let
      lib = nixpkgs.lib;
      contracts = import ./nix/contracts.nix;
      forAllSystems = lib.genAttrs contracts.supportedSystems;
      perSystem = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ rust-overlay.overlays.default ];
          };
          cvc5 = import ./nix/cvc5.nix {
            inherit pkgs lib;
            root = self;
          };
          built = import ./nix/packages.nix {
            inherit
              cvc5
              lib
              pkgs
              ;
            root = self;
          };
          packages = rec {
            inherit (built) chelis chelisup;
            chelis-runtime = built.runtime;
            default = chelis;
          };
          apps = rec {
            chelis = {
              type = "app";
              program = "${packages.chelis}/bin/chelis";
            };
            chelisup = {
              type = "app";
              program = "${packages.chelisup}/bin/chelisup";
            };
            default = chelis;
          };
          checks = import ./nix/checks.nix {
            inherit
              apps
              built
              contracts
              cvc5
              lib
              packages
              pkgs
              ;
            root = self;
          };
        in
        {
          inherit apps checks packages;
        }
      );
    in
    {
      packages = lib.mapAttrs (_system: outputs: outputs.packages) perSystem;
      apps = lib.mapAttrs (_system: outputs: outputs.apps) perSystem;
      checks = lib.mapAttrs (_system: outputs: outputs.checks) perSystem;
    };
}
