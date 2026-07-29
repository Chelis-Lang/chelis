{
  pkgs,
  lib,
  root,
  crate2nix,
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
  toolchain = pkgs.rust-bin.fromRustupToolchainFile (source + "/rust-toolchain.toml");
  buildRustCrateForPkgs =
    cratePkgs:
    cratePkgs.buildRustCrate.override {
      cargo = toolchain;
      rustc = toolchain;
      defaultCrateOverrides = cratePkgs.defaultCrateOverrides // {
        "chelis-cli" = attrs: {
          src = crateSource;
          sourceRoot = "chelis-source/crates/chelis-cli";
        };
        "chelis-compiler-api" = attrs: {
          src = crateSource;
          sourceRoot = "chelis-source/crates/chelis-compiler-api";
        };
        "chelis-cove" = attrs: {
          src = crateSource;
          sourceRoot = "chelis-source/crates/chelis-cove";
        };
        "cvc5-sys" = attrs: {
          nativeBuildInputs = (attrs.nativeBuildInputs or [ ]) ++ [
            pkgs.llvmPackages.libclang
            pkgs.pkg-config
          ];
          CVC5_DIR = "${cvc5.dir}";
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
        };
        "tree-sitter-chelis" = attrs: {
          src = crateSource;
          sourceRoot = "chelis-source/tree-sitter-chelis";
        };
      };
    };
  cargoGraph =
    assert crate2nixVersion == "0.15.0";
    import (root + "/Cargo.nix") {
      inherit buildRustCrateForPkgs pkgs;
      rootFeatures = [ ];
    };
  compilerCrate = cargoGraph.workspaceMembers."chelis-cli".build.override {
    features = [ "smt" ];
  };
  runtimeCrate = cargoGraph.workspaceMembers."chelis-runtime".build.override {
    features = [ ];
  };
  chelisupCrate = cargoGraph.workspaceMembers."chelisup".build.override {
    features = [ ];
  };
  compiler = pkgs.runCommand "chelis-cli-${version}" { } ''
    test -x ${compilerCrate}/bin/chelis
    install -Dm755 ${compilerCrate}/bin/chelis $out/bin/chelis
  '';
  runtime = pkgs.runCommand "chelis-runtime-${version}" { } ''
    artifact="$(find ${runtimeCrate.lib}/lib -type f -name 'libchelis_runtime-*.a' -print -quit)"
    test -n "$artifact"
    install -Dm444 "$artifact" $out/lib/libchelis_runtime.a
    mkdir -p $out/include
    ${lib.concatMapStringsSep "\n" (header: ''
      install -Dm444 ${source}/crates/chelis-runtime/include/${header} $out/include/${header}
    '') (import ./contracts.nix).publicRuntimeHeaders}
  '';
  chelisup = pkgs.runCommand "chelisup-${version}" { } ''
    test -x ${chelisupCrate}/bin/chelisup
    install -Dm755 ${chelisupCrate}/bin/chelisup $out/bin/chelisup
  '';
  chelis =
    pkgs.runCommand "chelis-${version}"
      {
        inherit version;
        meta.mainProgram = "chelis";
      }
      ''
        mkdir -p $out/bin $out/lib $out/include
        cp ${compiler}/bin/chelis $out/bin/chelis
        cp ${runtime}/lib/libchelis_runtime.a $out/lib/libchelis_runtime.a
        cp ${runtime}/include/*.h $out/include/
      '';
in
{
  inherit
    cargoGraph
    chelis
    chelisup
    chelisupCrate
    compiler
    compilerCrate
    crate2nixVersion
    runtime
    runtimeCrate
    source
    toolchain
    version
    ;
}
