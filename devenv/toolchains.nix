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
in
{
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
  env.PYO3_PYTHON = "${config.env.DEVENV_STATE}/venv/bin/python";

  # Linux wheels from downstream probes need the Nix C++ and zlib runtimes.
  # Consumers opt into this path. The project shell does not set a global
  # LD_LIBRARY_PATH that can shadow system libraries.
  env.CHELIS_PYTHON_WHEEL_LIBRARY_PATH = lib.optionalString pkgs.stdenv.isLinux (
    lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
    ]
  );

  outputs.cvc5-dir = cvc5.dir;

  profiles = {
    ci.module.env = {
      CARGO_PROFILE_DEV_DEBUG = "0";
      CARGO_PROFILE_TEST_DEBUG = "0";
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

  # bindgen (cvc5-sys and every other -sys crate that generates bindings)
  # must load the Nix libclang: without this the clang-sys probe finds a
  # host libclang whose own dependencies are not on the shell loader path
  # (observed on ubuntu-latest: /usr/lib/llvm-18 libclang failing to load
  # libstdc++.so.6). Same path nix/packages.nix gives the cvc5-sys crate
  # override.
  env.LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

  # Solver stack wiring (openspec converge-full-prove-on-devenv).
  # z3-sys links the prebuilt Nix libz3 through these two variables (the
  # same override scripts/z3_test.py honors first). On Linux, that command
  # sets the loader path for each Z3 invocation. The shell does not set a
  # global loader path that can shadow system libraries.
  env.Z3_SYS_Z3_HEADER = "${pkgs.z3.dev}/include/z3.h";
  env.Z3_LIBRARY_PATH_OVERRIDE = "${pkgs.z3.lib}/lib";
  # carcara's `gmp-mpfr-sys/use-system-libs` probe compiles against system
  # GMP; these entries let it find the Nix GMP without the per-machine
  # CPATH/LIBRARY_PATH recipe docs/smt_build_setup.md carries for brew.
  env.CPATH = "${pkgs.gmp.dev}/include";
  env.LIBRARY_PATH = "${pkgs.gmp}/lib";

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
      pkg-config
      z3
    ]
    # crate2nix and openspec come from the ci consumer module composed in
    # devenv.yaml (config.outputs.*), so their pins live once in ci.
    ++ [
      config.outputs.crate2nix
      config.outputs.openspec
    ]
    ++ lib.optionals stdenv.isLinux [
      gcc
      openblas
      valgrind
    ]
    ++ lib.optionals stdenv.isDarwin [
      gccShim
      gxxShim
    ];
}
