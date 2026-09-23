{
  config,
  pkgs,
  lib,
  ...
}:

let
  # Darwin stdenv uses the pinned Nixpkgs clang wrapper. The shims provide
  # the command names that the existing tests require.
  gccShim = pkgs.writeShellScriptBin "gcc" ''
    exec ${pkgs.stdenv.cc}/bin/cc -Wno-unused-command-line-argument "$@"
  '';

  gxxShim = pkgs.writeShellScriptBin "g++" ''
    exec ${pkgs.stdenv.cc}/bin/c++ -Wno-unused-command-line-argument "$@"
  '';

  kacheSource = pkgs.fetchFromGitHub {
    owner = "kunobi-ninja";
    repo = "kache";
    rev = "a21d020142b1248537cd548ccde99a04c0a44820";
    hash = "sha256-mrd4hlV0UXWLuo6GQXz44w1q0rrwzqvqlgcex9BHA4Q=";
  };

  kacheRustPlatform = pkgs.makeRustPlatform {
    cargo = config.languages.rust.toolchainPackage;
    rustc = config.languages.rust.toolchainPackage;
  };

  kacheBuildSource = pkgs.runCommand "kache-0.16.0-build-source" { } ''
    mkdir -p "$out"
    cp ${kacheSource}/Cargo.toml ${kacheSource}/Cargo.lock "$out/"
    cp -R ${kacheSource}/assets ${kacheSource}/crates ${kacheSource}/src "$out/"
  '';

  # Oracle-only producer for the pre-relocation cache representation. Keeping
  # this exact schema-27 binary available lets the executable-cache regression
  # prove that the schema-28 consumer cannot reuse an ordinary upgrade entry.
  legacyKache = kacheRustPlatform.buildRustPackage {
    pname = "kache-schema-27-fixture";
    version = "0.16.0";
    src = kacheBuildSource;
    patches = [ ../nix/patches/kache-0.16.0-chelis-contract.patch ];
    cargoHash = "sha256-VQJB5kGyXUjmfcKMeD2ggllbloeKXOpwjWOCVVsb0Rk=";
    cargoBuildFlags = [
      "-p"
      "kache"
    ];
    doCheck = false;
    RUSTC_WRAPPER = "";
    SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
  };

  patchedKache = kacheRustPlatform.buildRustPackage {
    pname = "kache";
    version = "0.16.0";
    src = kacheBuildSource;
    patches = [
      ../nix/patches/kache-0.16.0-chelis-contract.patch
      ../nix/patches/kache-0.16.0-relocatable-macos-executables.patch
    ];
    cargoHash = "sha256-VQJB5kGyXUjmfcKMeD2ggllbloeKXOpwjWOCVVsb0Rk=";
    cargoBuildFlags = [
      "-p"
      "kache"
    ];
    cargoTestFlags = [
      "-p"
      "kache"
    ];
    checkFlags = lib.optionals pkgs.stdenv.hostPlatform.isDarwin [
      "--skip=store::tests::test_exclude_from_indexing_sets_tmutil_xattr"
    ];
    # Upstream tests bind loopback mock servers; keep the sandbox enabled.
    __darwinAllowLocalNetworking = true;
    preCheck = ''
      ulimit -n 4096 2>/dev/null || true
    '';
    RUSTC_WRAPPER = "";
    SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
    meta.mainProgram = "kache";
  };
in
{
  languages = {
    rust = {
      enable = true;
      toolchainFile = ../rust-toolchain.toml;
    };

    python = {
      enable = true;
      package = pkgs.python311;
      venv.enable = true;
      uv.enable = true;
    };
  };

  # Cargo uses this override instead of the manual .venv fallback from
  # .cargo/config.toml while the Devenv shell is active.
  env = {
    PYO3_PYTHON = "${config.env.DEVENV_STATE}/venv/bin/python";
    CHELIS_IDENTITY_REAL_CARGO = "${config.languages.rust.toolchainPackage}/bin/cargo";
    CHELIS_IDENTITY_PROVENANCE = "source-worktree";
    RUSTC_WRAPPER = "${patchedKache}/bin/kache";
    CARGO_BUILD_RUSTC_WRAPPER = "${patchedKache}/bin/kache";
    RUSTC_WORKSPACE_WRAPPER = "";
    CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER = "";
    KACHE_CONFIG = "${config.devenv.root}/.kache.toml";
    KACHE_DISABLED = "0";
    KACHE_SCHEMA_27_WRAPPER = "${legacyKache}/bin/kache";
  }
  // lib.optionalAttrs pkgs.stdenv.isDarwin {
    # This is a native build. Keep Nix's SDK-aware compiler wrapper, but stop
    # cc-rs from injecting the redundant arm64-apple-macosx target alias that
    # the wrapper correctly rejects as a possible cross-target invocation.
    CC_aarch64_apple_darwin = "${pkgs.stdenv.cc}/bin/cc";
    CXX_aarch64_apple_darwin = "${pkgs.stdenv.cc}/bin/c++";
    CRATE_CC_NO_DEFAULTS = "1";
  };

  packages =
    with pkgs;
    [
      cargo-llvm-cov
      cargo-nextest
      cmake
      git
      # Vendored GMP needs m4 on cold local builds, not just CI.
      m4
      mdbook
      pkg-config
      pyright
      patchedKache
      # Direct rustc-private drivers link the pinned compiler's LLVM dependency.
      zlib
    ]
    ++ lib.optionals stdenv.isLinux [
      # The chelis#893 Phase 0 inventory reads C and Objective-C headers
      # through clang's front end; on Linux `cc` is gcc, which has no AST
      # dump, so the driver is added explicitly. Darwin's cc wrapper already
      # exposes `clang`.
      clang
      gcc
      openblas
      valgrind
    ]
    ++ lib.optionals stdenv.isDarwin [
      gccShim
      gxxShim
    ];
}
