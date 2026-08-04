{
  pkgs,
  source,
  crate2nix,
  toolchain,
}:
let
  cargoDeps = pkgs.rustPlatform.importCargoLock {
    lockFile = source + "/Cargo.lock";
    outputHashes = {
      "carcara-1.1.0" = "sha256-MiGxAA7LcagOohofUrt0FMURqsn85lq0d4Prz4SsfI8=";
    };
  };
  crate2nixCargo = pkgs.callPackage (crate2nix + "/crate2nix/Cargo.nix") {
    strictDeprecation = true;
  };
  crate2nixCommand = crate2nixCargo.rootCrate.build;
in
pkgs.runCommand "crate2nix-regeneration"
  {
    inherit cargoDeps;
    CARGO_NET_OFFLINE = "true";
    nativeBuildInputs = [
      crate2nixCommand
      pkgs.nix
      pkgs.python311
      pkgs.rustPlatform.cargoSetupHook
      toolchain
    ];
  }
  ''
    export HOME="$TMPDIR/home"
    mkdir -p "$HOME"
    test "$(crate2nix --version)" = "crate2nix 0.15.0"

    cp -R ${source} workspace
    chmod -R u+w workspace
    cd workspace
    cargoSetupPostUnpackHook
    rm Cargo.nix
    crate2nix generate \
      "--no-default-features" \
      "--features" \
      "chelis-cli/smt" \
      "--output" \
      "Cargo.nix"
    python3 scripts/check_crate2nix_sync.py --write
    cmp Cargo.nix ${source}/Cargo.nix
    touch "$out"
  ''
