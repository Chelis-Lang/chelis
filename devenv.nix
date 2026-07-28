{ pkgs, lib, ... }:

let
  # About twenty C-backend and CLI tests invoke the compiler as literally
  # `gcc` (`Command::new("gcc")`). macOS has no such binary: Apple's toolchain
  # is clang, and `xcrun gcc` answers "error: tool 'gcc' not found". Without
  # this the whole compile-and-run tier fails on darwin, which is easy to
  # misread as a code baseline rather than a missing tool -- it was read that
  # way here, and it hid the fact that nothing local was checking emitted C.
  #
  # Points the name at the platform's default C compiler (clang on darwin)
  # rather than at real GCC, which on darwin fights the SDK headers. The
  # generated C is plain C99 over float/int buffers plus memcpy, so the
  # distinction does not matter to it, and BLAS-linked tests already ask for
  # `-framework Accelerate` on macOS.
  #
  # `-Wno-unused-command-line-argument` is load-bearing: the nix cc-wrapper
  # injects the shell's `-L/nix/store/...` search paths on every invocation,
  # including compile-only (`-c`) ones where a linker path is unused. Several
  # tests compile with `-Werror`, which promotes that unused-argument warning
  # to an error and fails on the wrapper's own flags rather than on the code
  # under test. It precedes "$@" so a test's own `-Werror` still applies to
  # every other diagnostic.
  gccShim = pkgs.writeShellScriptBin "gcc" ''
    exec ${pkgs.stdenv.cc}/bin/cc -Wno-unused-command-line-argument "$@"
  '';

  # `exec_simd_header_compiles_as_cxx` checks the emitted SIMD header is valid
  # C++17, so the C++ driver needs the same treatment.
  gxxShim = pkgs.writeShellScriptBin "g++" ''
    exec ${pkgs.stdenv.cc}/bin/c++ -Wno-unused-command-line-argument "$@"
  '';
in
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
    ]
    # No openblas here: `blas_link_flags` asks for `-framework Accelerate` on
    # macOS, which ships with the OS.
    ++ lib.optionals stdenv.isDarwin [
      gccShim
      gxxShim
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
    gcc --version
    .venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 11)'
  '';
}
