{
  pkgs,
  lib,
  root,
  cvc5,
}:
let
  source = import ./source.nix { inherit lib root; };
  manifest = lib.importTOML (source + "/Cargo.toml");
  version = manifest.workspace.package.version;
  toolchain = pkgs.rust-bin.fromRustupToolchainFile (source + "/rust-toolchain.toml");
  rustPlatform = pkgs.makeRustPlatform {
    cargo = toolchain;
    rustc = toolchain;
  };
  cargoLock = {
    lockFile = source + "/Cargo.lock";
    outputHashes = {
      "carcara-1.1.0" = "sha256-MiGxAA7LcagOohofUrt0FMURqsn85lq0d4Prz4SsfI8=";
    };
  };
  common = {
    inherit cargoLock version;
    src = source;
    doCheck = false;
    strictDeps = true;
    preBuild = ''
      export CARGO_TARGET_DIR="$out/cargo-target"
    '';
  };

  compiler = rustPlatform.buildRustPackage (
    common
    // {
      pname = "chelis-cli";
      cargoBuildFlags = [
        "-p"
        "chelis-cli"
        "--bin"
        "chelis"
        "--features"
        "smt"
      ];
      nativeBuildInputs = [
        pkgs.llvmPackages.libclang
        pkgs.pkg-config
      ];
      LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
      CVC5_DIR = "${cvc5.dir}";
      installPhase = ''
        runHook preInstall
        artifact="$(find "$CARGO_TARGET_DIR" -type f -path '*/release/chelis' -print -quit)"
        test -n "$artifact"
        install -Dm755 "$artifact" "$out/bin/chelis"
        rm -rf "$CARGO_TARGET_DIR"
        runHook postInstall
      '';
      meta.mainProgram = "chelis";
    }
  );

  runtime = rustPlatform.buildRustPackage (
    common
    // {
      pname = "chelis-runtime";
      cargoBuildFlags = [
        "-p"
        "chelis-runtime"
        "--lib"
      ];
      installPhase = ''
        runHook preInstall
        artifact="$(find "$CARGO_TARGET_DIR" -type f -path '*/release/libchelis_runtime.a' -print -quit)"
        test -n "$artifact"
        install -Dm444 "$artifact" "$out/lib/libchelis_runtime.a"
        rm -rf "$CARGO_TARGET_DIR"
        mkdir -p "$out/include"
        ${lib.concatMapStringsSep "\n" (header: ''
          install -Dm444 "crates/chelis-runtime/include/${header}" "$out/include/${header}"
        '') (import ./contracts.nix).publicRuntimeHeaders}
        runHook postInstall
      '';
    }
  );

  chelisup = rustPlatform.buildRustPackage (
    common
    // {
      pname = "chelisup";
      cargoBuildFlags = [
        "-p"
        "chelisup"
        "--bin"
        "chelisup"
      ];
      installPhase = ''
        runHook preInstall
        artifact="$(find "$CARGO_TARGET_DIR" -type f -path '*/release/chelisup' -print -quit)"
        test -n "$artifact"
        install -Dm755 "$artifact" "$out/bin/chelisup"
        rm -rf "$CARGO_TARGET_DIR"
        runHook postInstall
      '';
      meta.mainProgram = "chelisup";
    }
  );

  chelis =
    pkgs.runCommand "chelis-${version}"
      {
        inherit version;
        meta.mainProgram = "chelis";
      }
      ''
        mkdir -p "$out/bin" "$out/lib" "$out/include"
        cp ${compiler}/bin/chelis "$out/bin/chelis"
        cp ${runtime}/lib/libchelis_runtime.a "$out/lib/libchelis_runtime.a"
        cp ${runtime}/include/*.h "$out/include/"
      '';
in
{
  inherit
    chelis
    chelisup
    compiler
    runtime
    source
    toolchain
    version
    ;
}
