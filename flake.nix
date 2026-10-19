{
  description = "Chelis compiler, runtime, and toolchain installer packages";

  # Runs on main and on release tags push the packages and their build inputs here.
  nixConfig.extra-substituters = [ "https://chelis-lang.cachix.org" ];
  nixConfig.extra-trusted-public-keys = [
    "chelis-lang.cachix.org-1:F4W65FQ7sunLoF2q6rzkWNTIBd3P/tJ/NFcv12OS9v8="
  ];

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/f205b5574fd0cb7da5b702a2da51507b7f4fdd1b";
    rust-overlay = {
      url = "github:oxalica/rust-overlay/89abdfd661ea493cde2f73d7b1332cb34150aa43";
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
      reef = import ./nix/reef.nix {
        inherit lib;
        chelisForSystem = system: self.packages.${system}.chelis;
      };
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
            inherit (built) chelis chelis-unwrapped chelisup;
            chelis-runtime = built.runtime;
            default = chelis;
          };
          apps = rec {
            chelis = {
              type = "app";
              program = "${packages.chelis}/bin/chelis";
              meta.description = packages.chelis.meta.description;
            };
            chelisup = {
              type = "app";
              program = "${packages.chelisup}/bin/chelisup";
              meta.description = packages.chelisup.meta.description;
            };
            default = chelis;
          };
          # `nix fmt` and the nixFormat check use this one nixfmt.
          nixfmt = pkgs.nixfmt;
          formatter = pkgs.writeShellApplication {
            name = "chelis-nix-format";
            runtimeInputs = [
              nixfmt
              pkgs.git
            ];
            # Without a file argument, the formatter formats each tracked
            # `.nix` file except the generated `Cargo.nix`. Options go to
            # nixfmt, so `nix fmt -- --check` checks the same files.
            text = ''
              options=()
              files=()
              for argument in "$@"; do
                case "$argument" in
                  -*) options+=("$argument") ;;
                  .) ;;
                  *) files+=("$argument") ;;
                esac
              done
              if [ "''${#files[@]}" -eq 0 ]; then
                top="$(git rev-parse --show-toplevel)"
                while IFS= read -r -d "" file; do
                  files+=("$top/$file")
                done < <(git -C "$top" ls-files -z -- "*.nix" ":!:Cargo.nix")
              fi
              exec nixfmt "''${options[@]}" "''${files[@]}"
            '';
            passthru = { inherit nixfmt; };
          };
          checks = import ./nix/checks.nix {
            reefBuilder = import ./nix/tests/reef-builder.nix {
              inherit pkgs lib;
              inherit (built) version;
              inherit (reef) buildChelisPackage;
              chelis = packages.chelis;
            };
            inherit
              apps
              built
              contracts
              cvc5
              lib
              nixfmt
              packages
              pkgs
              ;
            root = self;
          };
        in
        {
          inherit
            apps
            checks
            formatter
            packages
            ;
          # CI caches this pinned toolchain closure between runs; exposing it
          # here keeps the locked package contract unchanged.
          legacy = {
            cvc5-dir = cvc5.dir;
          };
        }
      );
    in
    {
      lib = reef;
      packages = lib.mapAttrs (_system: outputs: outputs.packages) perSystem;
      apps = lib.mapAttrs (_system: outputs: outputs.apps) perSystem;
      checks = lib.mapAttrs (_system: outputs: outputs.checks) perSystem;
      formatter = lib.mapAttrs (_system: outputs: outputs.formatter) perSystem;
      # The overlay adds the packages of the consumer system. On a system
      # that the flake does not support, it adds nothing.
      overlays.default =
        _final: prev:
        let
          system = prev.stdenv.hostPlatform.system;
        in
        lib.optionalAttrs (self.packages ? ${system}) {
          inherit (self.packages.${system})
            chelis
            chelis-runtime
            chelis-unwrapped
            chelisup
            ;
        };
      legacyPackages = lib.mapAttrs (_system: outputs: outputs.legacy) perSystem;
    };
}
