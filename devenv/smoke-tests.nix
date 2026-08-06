{ config, ... }:

{
  tasks."chelis:cargo-nix-fresh" = {
    description = "Check the committed Cargo.nix graph is fresh";
    after = [ "devenv:enterShell" ];
    before = [ "devenv:enterTest" ];
    exec = ''
      set -eu
      export PATH="${config.outputs.crate2nix}/bin:$PATH"
      "$VIRTUAL_ENV/bin/python" \
        "${config.devenv.root}/scripts/regenerate_cargo_nix.py" --check
    '';
  };

  tasks."chelis:toolchain-test" = {
    description = "Check the common development tools";
    after = [ "devenv:enterShell" ];
    before = [ "devenv:enterTest" ];
    exec = ''
      set -eu

      require_command() {
        if ! command -v "$1" >/dev/null 2>&1; then
          printf 'missing required command: %s\n' "$1" >&2
          return 1
        fi
      }

      for command_name in rustc cargo rust-analyzer uv cmake git pkg-config openspec; do
        require_command "$command_name"
      done

      openspec_version="$(openspec --version)"
      if [ "$openspec_version" != "1.6.0" ]; then
        printf 'unexpected OpenSpec version: %s\n' "$openspec_version" >&2
        exit 1
      fi

      rustc --version
      cargo --version
      rust-analyzer --version
      cargo nextest --version
      cargo llvm-cov --version
      uv --version
      cmake --version
      git --version
      pkg-config --version
    '';
  };

  tasks."chelis:python-test" = {
    description = "Check the Devenv Python interpreter";
    after = [ "devenv:enterShell" ];
    before = [ "devenv:enterTest" ];
    exec = ''
      set -eu
      if [ ! -x "$VIRTUAL_ENV/bin/python" ]; then
        printf '%s\n' 'missing Devenv Python virtual environment' >&2
        exit 1
      fi
      if [ "$PYO3_PYTHON" != "$VIRTUAL_ENV/bin/python" ]; then
        printf 'PyO3 interpreter differs from the active environment: %s\n' \
          "$PYO3_PYTHON" >&2
        exit 1
      fi
      "$VIRTUAL_ENV/bin/python" -c \
        'import os, pathlib, sys; assert sys.version_info[:2] == (3, 11); assert pathlib.Path(os.environ["PYO3_PYTHON"]).samefile(sys.executable)'
    '';
  };

  tasks."chelis:c-compiler-test" = {
    description = "Check the managed C compiler";
    after = [ "devenv:enterShell" ];
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
    after = [ "devenv:enterShell" ];
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
}
