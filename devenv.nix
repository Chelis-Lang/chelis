{ pkgs, lib, ... }:

{
  languages = {
    rust = {
      enable = true;
      toolchainFile = ./rust-toolchain.toml;
    };

    python = {
      enable = true;
      package = pkgs.python311;
      uv.enable = true;
    };
  };

  packages =
    with pkgs;
    [
      cargo-llvm-cov
      cargo-nextest
      cmake
      git
      pkg-config
    ]
    ++ lib.optionals stdenv.isLinux [
      gcc
      openblas
      valgrind
    ];

  # Chelis links PyO3 against this exact project-local Python environment.
  enterShell = ''
    if [ ! -x .venv/bin/python ]; then
      uv venv --python ${pkgs.python311}/bin/python3.11 .venv
    fi
  '';

  enterTest = ''
    cargo --version
    cargo nextest --version
    cargo llvm-cov --version
    cmake --version
    .venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 11)'
  '';
}
