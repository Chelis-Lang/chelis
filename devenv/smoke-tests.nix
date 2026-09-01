{ ... }:

{
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

      for command_name in rustc cargo uv cmake git mdbook pkg-config pyright kache; do
        require_command "$command_name"
      done

      rustc --version
      cargo --version
      cargo nextest --version
      cargo llvm-cov --version
      uv --version
      cmake --version
      git --version
      mdbook --version
      pkg-config --version
      pyright --version
      kache --version

      if [ ! -f "$LIBCLANG_PATH/libclang.dylib" ] && \
         [ ! -f "$LIBCLANG_PATH/libclang.so" ]; then
        printf 'missing pinned libclang 18 under: %s\n' "$LIBCLANG_PATH" >&2
        exit 1
      fi
    '';
  };

  tasks."chelis:python-test" = {
    description = "Check the Devenv Python interpreter";
    after = [ "devenv:python:virtualenv" ];
    before = [ "devenv:enterTest" ];
    exec = ''
      set -eu
      if [ ! -x "$PYO3_PYTHON" ]; then
        printf '%s\n' 'missing Devenv Python virtual environment' >&2
        exit 1
      fi
      "$PYO3_PYTHON" -c \
        'import os, pathlib, sys; assert sys.version_info[:2] == (3, 11); assert pathlib.Path(os.environ["PYO3_PYTHON"]).samefile(sys.executable)'
    '';
  };

  tasks."chelis:c-compiler-test" = {
    description = "Check the managed C compiler";
    after = [ "devenv:files" ];
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

      probe_dir=".devenv/generated/compiler-probes"
      for probe in "$probe_dir/valid.c" "$probe_dir/warning.c"; do
        if [ ! -L "$probe" ]; then
          printf 'missing Devenv compiler probe: %s\n' "$probe" >&2
          exit 1
        fi
      done

      gcc -std=c11 -Wall -Wextra -Werror -fsyntax-only "$probe_dir/valid.c"
      if gcc -std=c11 -Wall -Wextra -Werror -fsyntax-only \
        "$probe_dir/warning.c" >/dev/null 2>&1; then
        printf '%s\n' 'compiler warning probe unexpectedly succeeded' >&2
        exit 1
      fi
    '';
  };

  tasks."chelis:cpp-compiler-test" = {
    description = "Check the managed C++ compiler";
    after = [ "devenv:files" ];
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

      probe=".devenv/generated/compiler-probes/valid.cpp"
      if [ ! -L "$probe" ]; then
        printf 'missing Devenv compiler probe: %s\n' "$probe" >&2
        exit 1
      fi

      object_path="$(mktemp)"
      cleanup_object() {
        rm -f "$object_path"
      }
      trap cleanup_object EXIT
      g++ -std=c++17 -Wall -Wextra -Werror -c "$probe" -o "$object_path"
    '';
  };

  tasks."chelis:kache-test" = {
    description = "Check the repository-owned Kache wrapper and no-cache control";
    after = [ "devenv:python:virtualenv" ];
    before = [ "devenv:enterTest" ];
    exec = ''
      set -eu
      "$PYO3_PYTHON" scripts/kache_toolchain_smoke.py
    '';
  };

  tasks."chelis:pyright-test" = {
    description = "Check the repository-owned Pyright analysis scope";
    after = [ "devenv:python:virtualenv" ];
    before = [ "devenv:enterTest" ];
    exec = ''
      set -eu
      pyright --version
      "$PYO3_PYTHON" scripts/check_pyright_scope.py
    '';
  };

  tasks."chelis:docs-test" = {
    description = "Build the documentation with the pinned mdBook";
    before = [ "devenv:enterTest" ];
    exec = ''
      set -eu
      mdbook --version
      mdbook build docs/book
    '';
  };

  tasks."chelis:darwin-tree-sitter-test" = {
    description = "Check the Darwin native compiler and tree-sitter parser agreement";
    after = [ "devenv:python:virtualenv" ];
    before = [ "devenv:enterTest" ];
    exec = ''
      set -eu
      "$PYO3_PYTHON" scripts/darwin_tree_sitter_smoke.py
    '';
  };
}
