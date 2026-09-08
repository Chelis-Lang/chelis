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

  cvc5 = import ../nix/cvc5.nix {
    inherit pkgs lib;
    root = ../.;
  };

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
    preCheck = ''
      ulimit -n 4096 2>/dev/null || true
    '';
    RUSTC_WRAPPER = "";
    SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
    meta.mainProgram = "kache";
  };
in
{
  # The v2.2.2 modules accidentally retain `latest-version = 2.2.1`. Correct
  # that release metadata locally so the reviewed 2.2.0-through-2.2.2 CLI
  # range still reports the pinned module release as its update target.
  devenv.latestVersion = "2.2.2";

  languages = {
    rust = {
      enable = true;
      toolchainFile = ../rust-toolchain.toml;
      lsp.package = pkgs.rust-analyzer;
    };

    python = {
      enable = true;
      package = pkgs.python311.withPackages (ps: [ ps.numpy ]);
      venv.enable = true;
      uv.enable = true;
    };
  };

  # Cargo uses this override instead of the manual .venv fallback from
  # .cargo/config.toml while the Devenv shell is active.
  env = {
    PYO3_PYTHON = "${config.env.DEVENV_STATE}/venv/bin/python";
    RUSTC_WRAPPER = "${patchedKache}/bin/kache";
    CARGO_BUILD_RUSTC_WRAPPER = "${patchedKache}/bin/kache";
    RUSTC_WORKSPACE_WRAPPER = "";
    CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER = "";
    KACHE_CONFIG = "${config.devenv.root}/.kache.toml";
    KACHE_DISABLED = "0";
    KACHE_SCHEMA_27_WRAPPER = "${legacyKache}/bin/kache";

    # bindgen (cvc5-sys and every other -sys crate that generates bindings)
    # must load the Nix libclang: without this the clang-sys probe finds a
    # host libclang whose own dependencies are not on the shell loader path
    # (observed on ubuntu-latest: /usr/lib/llvm-18 libclang failing to load
    # libstdc++.so.6). Same path nix/packages.nix gives the cvc5-sys crate
    # override.
    LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

    # Solver stack wiring (openspec converge-full-prove-on-devenv).
    # z3-sys links the prebuilt Nix libz3 through these two variables (the
    # same override scripts/z3_test.py honors first). On Linux, that command
    # sets the loader path for each Z3 invocation. The shell does not set a
    # global loader path that can shadow system libraries.
    Z3_SYS_Z3_HEADER = "${pkgs.z3.dev}/include/z3.h";
    Z3_LIBRARY_PATH_OVERRIDE = "${pkgs.z3.lib}/lib";
    # carcara's `gmp-mpfr-sys/use-system-libs` probe compiles against system
    # GMP; these entries let it find the Nix GMP without the per-machine
    # CPATH/LIBRARY_PATH recipe docs/smt_build_setup.md carries for brew.
    CPATH = "${pkgs.gmp.dev}/include";
    LIBRARY_PATH = "${pkgs.gmp}/lib";

    # Linux wheels from downstream probes need the Nix C++ and zlib runtimes.
    # Consumers opt into this path. The project shell does not set a global
    # LD_LIBRARY_PATH that can shadow system libraries.
    CHELIS_PYTHON_WHEEL_LIBRARY_PATH = lib.optionalString pkgs.stdenv.isLinux (
      lib.makeLibraryPath [
        pkgs.stdenv.cc.cc.lib
        pkgs.zlib
      ]
    );
  }
  // lib.optionalAttrs pkgs.stdenv.isDarwin {
    # This is a native build. Keep Nix's SDK-aware compiler wrapper, but stop
    # cc-rs from injecting the redundant arm64-apple-macosx target alias that
    # the wrapper correctly rejects as a possible cross-target invocation.
    CC_aarch64_apple_darwin = "${pkgs.stdenv.cc}/bin/cc";
    CXX_aarch64_apple_darwin = "${pkgs.stdenv.cc}/bin/c++";
    CRATE_CC_NO_DEFAULTS = "1";
  };

  outputs.cvc5-dir = cvc5.dir;

  profiles = {
    ci.module.env = {
      CARGO_PROFILE_DEV_DEBUG = "0";
      CARGO_PROFILE_TEST_DEBUG = "0";
    }
    // lib.optionalAttrs pkgs.stdenv.isLinux {
      # Nixpkgs GCC does not enable GNU build IDs by default in the Devenv shell.
      # Chelis cache fingerprints require the fast linker-ID path.
      RUSTFLAGS = "-C link-arg=-Wl,--build-id=sha1";
    };

    sanitizers = {
      extends = [ "ci" ];
      module.env = {
        CHELIS_C_TEST_EXTRA_FLAGS = "-O1 -fsanitize=address,undefined -fno-omit-frame-pointer";
        ASAN_OPTIONS = "detect_leaks=1:halt_on_error=1";
        UBSAN_OPTIONS = "print_stacktrace=1:halt_on_error=1";
      };
    };

    smt = {
      extends = [ "ci" ];
      module.env.CVC5_DIR = "${config.outputs.cvc5-dir}";
    };
  };

  packages =
    with pkgs;
    [
      cargo-llvm-cov
      cargo-nextest
      cmake
      # The chelis-prove solver stack: z3 (WI-12 engine), gappa (envelope
      # re-validation), gmp (carcara's rug), m4 + make (the vendored
      # Arb/FLINT and GMP source builds under --features arb).
      gappa
      git
      gmp
      gnum4
      gnumake
      mdbook
      patchedKache
      pkg-config
      pyright
      z3
    ]
    # crate2nix and openspec come from the ci consumer module composed in
    # devenv.yaml (config.outputs.*), so their pins live once in ci.
    ++ [
      config.outputs.crate2nix
      config.outputs.openspec
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
