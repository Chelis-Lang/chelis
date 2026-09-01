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
  env.PYO3_PYTHON = "${config.env.DEVENV_STATE}/venv/bin/python";

  packages =
    with pkgs;
    [
      # cargo-llvm-cov is deliberately absent. Nothing in this repository runs
      # it: no CI job, no script, and the coverage baseline that would use it
      # is chelis#803, still open. Re-add it there, pinned and exercised by a
      # gate, rather than carrying a shell promise nothing checks (chelis#1441).
      cargo-nextest
      cmake
      git
      pkg-config
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
