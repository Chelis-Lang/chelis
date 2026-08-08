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

  # bindgen (cvc5-sys and every other -sys crate that generates bindings)
  # must load the Nix libclang: without this the clang-sys probe finds a
  # host libclang whose own dependencies are not on the shell loader path
  # (observed on ubuntu-latest: /usr/lib/llvm-18 libclang failing to load
  # libstdc++.so.6). Same path nix/packages.nix gives the cvc5-sys crate
  # override.
  env.LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

  # Solver stack wiring (openspec converge-full-prove-on-devenv).
  # z3-sys links the prebuilt Nix libz3 through these two variables (the
  # same override scripts/z3_test.py honors first). On Linux the runtime
  # loader also needs the directory; the CI z3 steps export
  # LD_LIBRARY_PATH="$Z3_LIBRARY_PATH_OVERRIDE" per step instead of a
  # global loader path that would shadow system libraries.
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
