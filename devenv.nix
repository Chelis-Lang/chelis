{
  config,
  inputs,
  lib,
  pkgs,
  ...
}:

let
  # Keep generated Nix sources small and deterministic. The retained files are
  # the repository source needed by crate2nix; local build and devenv state are
  # never inputs to the package derivation.
  workspaceSrc = lib.cleanSourceWith {
    src = ./.;
    filter =
      path: type:
      let
        root = toString ./. + "/";
        relative = lib.removePrefix root (toString path);
        excluded = [
          ".devenv"
          ".direnv"
          ".git"
          ".local"
          ".venv"
          "devenv.lock"
          "devenv.nix"
          "devenv.yaml"
          "target"
        ];
        isExcluded = name: relative == name || lib.hasPrefix (name + "/") relative;
      in
      !(lib.any isExcluded excluded);
  };

  # devenv's languages.rust.import helper targets a single root crate. Chelis
  # has a virtual workspace manifest, so use the same crate2nix machinery and
  # select chelis-cli explicitly from workspaceMembers.
  crate2nixTools = pkgs.callPackage "${inputs.crate2nix}/tools.nix" { };
  generatedCargoNix = crate2nixTools.generatedCargoNix {
    name = "chelis";
    src = workspaceSrc;
  };
  workspaceCrateOverride = directory: _: {
    src = workspaceSrc;
    sourceRoot = "source/crates/${directory}";
  };
  cargoNix = pkgs.callPackage generatedCargoNix {
    release = false;
    buildRustCrateForPkgs =
      _:
      pkgs.buildRustCrate.override {
        rustc = config.languages.rust.toolchainPackage;
        cargo = config.languages.rust.toolchainPackage;
        defaultCrateOverrides = pkgs.defaultCrateOverrides // {
          # These crates embed repository-level headers, examples, docs, or
          # grammars. Preserve the Cargo workspace layout around each source.
          "chelis-cli" = workspaceCrateOverride "chelis-cli";
          "chelis-compiler-api" = workspaceCrateOverride "chelis-compiler-api";
          "chelis-cove" = workspaceCrateOverride "chelis-cove";
          "chelis-types" = workspaceCrateOverride "chelis-types";
          "tree-sitter-chelis" = _: {
            src = workspaceSrc;
            sourceRoot = "source/tree-sitter-chelis";
          };
        };
      };
  };
  chelisNix = cargoNix.workspaceMembers."chelis-cli".build;

  # uv2nix consumes py/pyproject.toml and py/uv.lock. The resulting environment
  # supplies the chelis-tools commands without pip-mutating the Nix profile.
  chelisTools = config.languages.python.import ./py {
    packageName = "chelis-tools";
  };

  python = config.languages.python.package;
  pythonLibraryPath = lib.makeLibraryPath [ python ];

  # The source and integration tests invoke gcc/g++ literally. The macOS SDK
  # exposes Clang, while /usr/bin/gcc and /usr/bin/g++ are unavailable on this
  # host, so provide compiler-name compatibility without introducing GNU GCC.
  # Nix's compiler wrapper injects linker search paths even for `-fsyntax-only`;
  # Clang reports those wrapper-owned arguments as unused, which trips tests
  # that intentionally combine syntax-only checking with `-Werror`. Suppress
  # only that driver noise for syntax-only invocations, at the environment seam
  # that introduced it rather than in the product test harness.
  darwinCompilerAlias =
    name: compiler:
    pkgs.writeTextFile {
      name = "chelis-darwin-${name}";
      destination = "/bin/${name}";
      executable = true;
      text = ''
        #!${python.interpreter}
        import os
        import sys

        args = sys.argv[1:]
        if "-fsyntax-only" in args:
            args.insert(0, "-Qunused-arguments")
        os.execv("${compiler}", ["${compiler}", *args])
      '';
    };
  darwinCompilerAliases = pkgs.symlinkJoin {
    name = "chelis-darwin-compiler-aliases";
    paths = [
      (darwinCompilerAlias "gcc" "${pkgs.stdenv.cc}/bin/clang")
      (darwinCompilerAlias "g++" "${pkgs.stdenv.cc}/bin/clang++")
    ];
  };

  tidePort = config.processes.tide.ports.http.value;
in
{
  languages.rust = {
    enable = true;
    toolchainFile = ./rust-toolchain.toml;
  };

  languages.python = {
    enable = true;
    version = "3.11";
  };

  packages = [
    chelisTools
    pkgs.cargo-nextest
    pkgs.cmake
    pkgs.crate2nix
    pkgs.git
    pkgs.github-cli
    pkgs.gmp
    pkgs.jq
    pkgs.libclang
    pkgs.m4
    pkgs.maturin
    pkgs.mdbook
    pkgs.mpfr
    pkgs.nodejs_22
    pkgs.pkg-config
    pkgs.shellcheck
    pkgs.uv
    pkgs.z3
  ]
  ++ lib.optionals pkgs.stdenv.isLinux [
    pkgs.gcc
    pkgs.openblas
  ]
  ++ lib.optionals pkgs.stdenv.isDarwin [
    darwinCompilerAliases
  ];

  env = {
    # Cargo's checked-in config also names this path. Exporting the absolute
    # form keeps direct pyo3 tooling and subprocesses on the same interpreter.
    PYO3_PYTHON = "${config.devenv.root}/.venv/bin/python";
    LIBCLANG_PATH = "${lib.getLib pkgs.libclang}/lib";
    Z3_LIBRARY_PATH_OVERRIDE = "${lib.getLib pkgs.z3}/lib";
  }
  // lib.optionalAttrs pkgs.stdenv.isLinux {
    LD_LIBRARY_PATH = lib.makeLibraryPath [
      python
      pkgs.openblas
      pkgs.z3
    ];
  }
  // lib.optionalAttrs pkgs.stdenv.isDarwin {
    DYLD_LIBRARY_PATH = pythonLibraryPath;
  };

  tasks = {
    "chelis:python-venv" = {
      description = "Create the uv-managed Python 3.11 environment required by pyo3";
      exec = ''
        python3 scripts/ci_setup_uv_python.py --python ${python.interpreter}
      '';
      status = ''
        test -x .venv/bin/python && \
          .venv/bin/python -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 11))'
      '';
    };

    "devenv:enterShell".after = [ "chelis:python-venv" ];

    "chelis:build" = {
      description = "Build every workspace target with Cargo";
      after = [ "chelis:python-venv" ];
      exec = "cargo build --workspace --all-targets";
    };

    "chelis:gate-local" = {
      description = "Run the repository's local pre-push gate";
      after = [ "chelis:python-venv" ];
      exec = ".venv/bin/python scripts/gate.py --local";
    };
  };

  scripts = {
    chelis-dev = {
      description = "Run the workspace chelis CLI (arguments are forwarded)";
      exec = ''
        cargo run -p chelis-cli --bin chelis -- "$@"
      '';
    };

    chelis-gate = {
      description = "Run scripts/gate.py with the project Python";
      exec = ''
        .venv/bin/python scripts/gate.py "$@"
      '';
    };
  };

  processes.tide = {
    exec = ''
      cargo run -p chelis-cli --bin chelis -- tide serve \
        --host 127.0.0.1 --port ${toString tidePort}
    '';
    ports.http.allocate = 8080;
    ready.exec = ''
      ${python.interpreter} -c \
        'import socket; socket.create_connection(("127.0.0.1", ${toString tidePort}), timeout=1).close()'
    '';
    after = [ "chelis:python-venv" ];
    start.enable = !config.devenv.isTesting;
    restart.on = "on_failure";
    watch = {
      paths = [
        ./Cargo.toml
        ./Cargo.lock
        ./crates
        ./tree-sitter-chelis
        ./grammars
      ];
      extensions = [
        "c"
        "h"
        "rs"
        "toml"
      ];
      ignore = [ "target" ];
    };
  };

  enterShell = ''
    source .venv/bin/activate
  '';

  # Authoritative acceptance oracle for this development environment.
  enterTest = ''
    set -euo pipefail

    rustc --version
    cargo --version
    cargo clippy --version
    rustfmt --version
    cargo nextest --version
    crate2nix --version
    .venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 11)'
    loc-report --help >/dev/null
    gcc --version >/dev/null
    g++ --version >/dev/null

    # Referencing the crate2nix output makes its build part of this oracle.
    test -x ${chelisNix}/bin/chelis
    ${chelisNix}/bin/chelis --version

    cargo build --workspace --all-targets
    cargo nextest run -p chelis-python
    cargo nextest run -p chelis-backend-c --test exec_compile \
      -E 'test(/exec_(math_none_exp_kernel_correct_output|simd_header_compiles_as_cxx)/)'
    target/debug/chelis --version

    tide_log="''${TMPDIR:-/tmp}/chelis-tide-devenv-test-$$.log"
    target/debug/chelis tide serve --host 127.0.0.1 --port ${toString tidePort} \
      >"$tide_log" 2>&1 &
    tide_pid=$!
    trap 'kill "$tide_pid" 2>/dev/null || true' EXIT
    for _ in {1..60}; do
      if ${python.interpreter} -c \
        'import socket; socket.create_connection(("127.0.0.1", ${toString tidePort}), timeout=1).close()' \
        2>/dev/null; then
        break
      fi
      sleep 0.25
    done
    ${python.interpreter} -c \
      'import socket; socket.create_connection(("127.0.0.1", ${toString tidePort}), timeout=1).close()'
    ${python.interpreter} - <<'PY'
    import json
    import urllib.request

    request = urllib.request.Request(
        "http://127.0.0.1:${toString tidePort}/parse",
        data=json.dumps({
            "source_kind": "surf",
            "source": "def identity(x: f32) -> f32 = x",
        }).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        body = json.load(response)
    assert response.status == 200
    assert body["ok"] is True, body
    assert body["result"]["surf_ast"], body
    PY
    kill "$tide_pid"
    wait "$tide_pid" || true
    trap - EXIT
    rm -f "$tide_log"
  '';

  outputs = {
    chelis = chelisNix;
    "chelis-tools" = chelisTools;
  };
}
