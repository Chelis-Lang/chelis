{ pkgs, lib, ... }:

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
    ++ lib.optionals stdenv.isDarwin [
      gccShim
      gxxShim
    ];

  enterShell = ''
    if [ ! -x .venv/bin/python ]; then
      uv venv --python ${pkgs.python311}/bin/python3.11 .venv
    fi
  '';

  tasks."chelis:toolchain-test" = {
    description = "Check the common development tools";
    before = [ "devenv:enterTest" ];
    exec = ''
      set -eu

      require_command() {
        if ! command -v "$1" >/dev/null 2>&1; then
          printf 'missing required command: %s\n' "$1" >&2
          return 1
        fi
      }

      for command_name in rustc cargo uv cmake git pkg-config; do
        require_command "$command_name"
      done

      rustc --version
      cargo --version
      cargo nextest --version
      cargo llvm-cov --version
      uv --version
      cmake --version
      git --version
      pkg-config --version
    '';
  };

  tasks."chelis:python-test" = {
    description = "Check the project Python interpreter";
    before = [ "devenv:enterTest" ];
    exec = ''
      set -eu
      if [ ! -x .venv/bin/python ]; then
        printf '%s\n' 'missing project Python interpreter: .venv/bin/python' >&2
        exit 1
      fi
      .venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 11)'
    '';
  };

  tasks."chelis:c-compiler-test" = {
    description = "Check the managed C compiler";
    before = [ "devenv:enterTest" ];
    exec = ''
      set -eu

      compiler_path="$(command -v gcc 2>/dev/null || true)"
      case "$compiler_path" in
        /nix/store/*) ;;
        *)
          printf 'unmanaged shell command: %s (%s)\n' \
            gcc "''${compiler_path:-not found}" >&2
          exit 1
          ;;
      esac
      gcc --version

      probe_dir="$(mktemp -d)"
      cleanup_probe_dir() {
        rm -rf "$probe_dir"
      }
      trap cleanup_probe_dir EXIT

      cat > "$probe_dir/valid.c" <<'EOF'
      int main(void) {
          return 0;
      }
      EOF
      gcc -std=c11 -Wall -Wextra -Werror -fsyntax-only "$probe_dir/valid.c"

      cat > "$probe_dir/warning.c" <<'EOF'
      int main(void) {
          int unused = 0;
          return 0;
      }
      EOF
      if gcc -std=c11 -Wall -Wextra -Werror -fsyntax-only \
        "$probe_dir/warning.c" >"$probe_dir/warning.log" 2>&1; then
        printf '%s\n' 'compiler warning probe unexpectedly succeeded' >&2
        exit 1
      fi
    '';
  };

  tasks."chelis:cpp-compiler-test" = {
    description = "Check the managed C++ compiler";
    before = [ "devenv:enterTest" ];
    exec = ''
      set -eu

      compiler_path="$(command -v g++ 2>/dev/null || true)"
      case "$compiler_path" in
        /nix/store/*) ;;
        *)
          printf 'unmanaged shell command: %s (%s)\n' \
            g++ "''${compiler_path:-not found}" >&2
          exit 1
          ;;
      esac
      g++ --version

      probe_dir="$(mktemp -d)"
      cleanup_probe_dir() {
        rm -rf "$probe_dir"
      }
      trap cleanup_probe_dir EXIT

      cat > "$probe_dir/valid.cpp" <<'EOF'
      int main() {
          return 0;
      }
      EOF
      g++ -std=c++17 -Wall -Wextra -Werror -c \
        "$probe_dir/valid.cpp" -o "$probe_dir/valid-cxx.o"
    '';
  };
}
