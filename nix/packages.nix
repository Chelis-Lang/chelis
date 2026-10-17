{
  pkgs,
  lib,
  root,
  cvc5,
}:
let
  source = import ./source.nix { inherit lib root; };
  # `packages` carries packages/chelis-std, which the chelis-std-bundle build
  # script packs into the runtime the compiler embeds.
  crateSource = import ./source.nix {
    inherit lib root;
    includeRoots = [
      "crates"
      "grammars"
      "packages"
      "tree-sitter-chelis"
    ];
  };
  manifest = lib.importTOML (root + "/Cargo.toml");
  version = manifest.workspace.package.version;
  # crate2nix 0.15.0 generates the committed root Cargo.nix with the compiler
  # features below. The Cargo.nix freshness job in ci.yml runs the same
  # command, and docs/contributor_setup.md gives it to contributors.
  crate2nixVersion = "0.15.0";
  compilerFeatures = [ "distribution" ];
  crate2nixCommand = lib.concatMapStringsSep " " builtins.toJSON (
    [
      "generate"
      "--no-default-features"
    ]
    ++ lib.concatMap (feature: [
      "--features"
      "chelis-cli/${feature}"
    ]) (lib.sort lib.lessThan compilerFeatures)
  );
  toolchain = pkgs.rust-bin.fromRustupToolchainFile (root + "/rust-toolchain.toml");
  buildRustCrateForPkgs =
    cratePkgs:
    cratePkgs.buildRustCrate.override {
      cargo = toolchain;
      rustc = toolchain;
      defaultCrateOverrides = cratePkgs.defaultCrateOverrides // {
        "chelis-cli" = attrs: {
          src = crateSource;
          sourceRoot = "chelis-source/crates/chelis-cli";
        };
        "chelis-compiler-api" = attrs: {
          src = crateSource;
          sourceRoot = "chelis-source/crates/chelis-compiler-api";
        };
        "chelis-cove" = attrs: {
          src = crateSource;
          sourceRoot = "chelis-source/crates/chelis-cove";
        };
        "chelis-std-bundle" = attrs: {
          src = crateSource;
          sourceRoot = "chelis-source/crates/chelis-std-bundle";
        };
        "cvc5-sys" = attrs: {
          nativeBuildInputs = (attrs.nativeBuildInputs or [ ]) ++ [
            pkgs.llvmPackages.libclang
            pkgs.pkg-config
          ];
          CVC5_DIR = "${cvc5.dir}";
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
        };
        "tree-sitter-chelis" = attrs: {
          src = crateSource;
          sourceRoot = "chelis-source/tree-sitter-chelis";
        };
      };
    };
  cargoGraph = import (root + "/Cargo.nix") {
    inherit buildRustCrateForPkgs pkgs;
    rootFeatures = [ ];
  };
  compilerCrate = cargoGraph.workspaceMembers."chelis-cli".build.override {
    features = compilerFeatures;
  };
  chelisupCrate = cargoGraph.workspaceMembers."chelisup".build.override {
    features = [ ];
  };
  contracts = import ./contracts.nix;
  # The scripts that the package builds run, as a minimal store path. A
  # reference to the flake source gives new package derivations for each edit
  # of the repository, also for an edit of the documentation.
  packageScripts = lib.fileset.toSource {
    root = ../scripts;
    fileset = ../scripts/verify_runtime_package.py;
  };
  packageMeta = {
    homepage = "https://chelis.ch";
    license = lib.licenses.mit;
    platforms = contracts.supportedSystems;
  };
  compiler = pkgs.runCommand "chelis-cli-${version}" { } ''
    test -x ${compilerCrate}/bin/chelis
    install -Dm755 ${compilerCrate}/bin/chelis $out/bin/chelis
  '';
  # The runtime the compiler carries (spec/08-backends.md §2.1): `chelis-runtime`
  # and `chelis` ship the compiler's export, never a separately built crate.
  runtime =
    pkgs.runCommand "chelis-runtime-${version}"
      {
        meta = packageMeta // {
          description = "C runtime library and headers of the Chelis compiler";
        };
        nativeBuildInputs = [ pkgs.python311 ];
      }
      ''
        export_dir="$TMPDIR/runtime-export"
        ${compiler}/bin/chelis runtime export "$export_dir"
        install -Dm444 "$export_dir/libchelis_runtime.a" $out/lib/libchelis_runtime.a
        mkdir -p $out/include
        ${lib.concatMapStringsSep "\n" (header: ''
          install -Dm444 "$export_dir/${header}" $out/include/${header}
        '') contracts.publicRuntimeHeaders}
        python3 ${packageScripts}/verify_runtime_package.py "$export_dir" "$out"
      '';
  chelisup = pkgs.runCommand "chelisup-${version}" { } ''
    test -x ${chelisupCrate}/bin/chelisup
    install -Dm755 ${chelisupCrate}/bin/chelisup $out/libexec/chelisup
    mkdir -p $out/bin
    cat > $out/bin/chelisup <<EOF
    #!${pkgs.runtimeShell}
    set -eu

    resolve_chelis_home() {
      if [ -n "\''${CHELIS_HOME:-}" ]; then
        chelis_home="\$CHELIS_HOME"
      elif [ -n "\''${HOME:-}" ]; then
        chelis_home="\$HOME/.chelis"
      else
        printf '%s\n' 'chelisup: neither CHELIS_HOME nor HOME is set' >&2
        return 1
      fi
    }

    set_gc_root_paths() {
      gc_root_dir="\$chelis_home/nix-gcroots"
      gc_root="\$gc_root_dir/chelisup"
      staging_root="\$gc_root_dir/chelisup.next"
      partial_root="\$gc_root_dir/chelisup.partial"
    }

    copy_matches_package() {
      candidate_root="\$1"
      for installed_copy in \
        "\$chelis_home/bin/chelis" \
        "\$chelis_home/bin/chelisup"; do
        if [ -f "\$installed_copy" ] \
          && ${pkgs.diffutils}/bin/cmp -s \
            "\$candidate_root/libexec/chelisup" "\$installed_copy"; then
          return 0
        fi
      done
      return 1
    }

    installer_matches_package() {
      [ -f "\$chelis_home/bin/chelisup" ] \
        && ${pkgs.diffutils}/bin/cmp -s \
          "$out/libexec/chelisup" "\$chelis_home/bin/chelisup"
    }

    restore_nix_wrapper() {
      wrapper_tmp="\$chelis_home/bin/.chelisup-nix-wrapper.\$\$"
      ${pkgs.coreutils}/bin/rm -f "\$wrapper_tmp"
      ${pkgs.coreutils}/bin/install -m755 "$out/bin/chelisup" "\$wrapper_tmp" \
        || { ${pkgs.coreutils}/bin/rm -f "\$wrapper_tmp"; return 1; }
      ${pkgs.coreutils}/bin/mv -f "\$wrapper_tmp" "\$chelis_home/bin/chelisup" \
        || { ${pkgs.coreutils}/bin/rm -f "\$wrapper_tmp"; return 1; }
    }

    remove_gc_roots() {
      for root in "\$gc_root" "\$staging_root" "\$partial_root"; do
        if [ -e "\$root" ] || [ -L "\$root" ]; then
          ${pkgs.coreutils}/bin/rm -f "\$root"
          printf 'removed %s\n' "\$root"
        fi
      done
      ${pkgs.coreutils}/bin/rmdir "\$gc_root_dir" 2>/dev/null || true
    }

    if [ "\''${1:-}" = "self" ] && [ "\''${2:-}" = "uninstall" ]; then
      resolve_chelis_home
      set_gc_root_paths
      uninstall_status=0
      "$out/libexec/chelisup" "\$@" || uninstall_status=\$?
      if [ "\$uninstall_status" -ne 0 ]; then
        exit "\$uninstall_status"
      fi
      remove_gc_roots
      exit 0
    fi

    if [ "\''${1:-}" = "install" ]; then
      resolve_chelis_home
      set_gc_root_paths
      ${pkgs.coreutils}/bin/mkdir -p "\$gc_root_dir"

      if [ -L "\$staging_root" ]; then
        staged_package="\$(${pkgs.coreutils}/bin/readlink "\$staging_root")"
        if copy_matches_package "\$staged_package"; then
          ${pkgs.nix}/bin/nix-store \
            --add-root "\$partial_root" \
            --realise "\$staged_package" >/dev/null
        fi
        ${pkgs.coreutils}/bin/rm -f "\$staging_root"
      elif [ -e "\$staging_root" ]; then
        printf '%s\n' 'chelisup: the Nix staging root is not a symbolic link' >&2
        exit 1
      fi

      ${pkgs.nix}/bin/nix-store --add-root "\$staging_root" --realise "$out" >/dev/null

      install_status=0
      "$out/libexec/chelisup" "\$@" || install_status=\$?
      if [ "\$install_status" -ne 0 ]; then
        if copy_matches_package "$out"; then
          ${pkgs.nix}/bin/nix-store \
            --add-root "\$partial_root" \
            --realise "$out" >/dev/null
        fi
        if installer_matches_package; then
          restore_nix_wrapper
        fi
        ${pkgs.coreutils}/bin/rm -f "\$staging_root"
        exit "\$install_status"
      fi

      restore_nix_wrapper
      if ! ${pkgs.nix}/bin/nix-store --add-root "\$gc_root" --realise "$out" >/dev/null; then
        printf '%s\n' 'chelisup: failed to promote the Nix GC root' >&2
        exit 1
      fi
      ${pkgs.coreutils}/bin/rm -f "\$partial_root" "\$staging_root"
      exit 0
    fi

    # The one operand of "link DIR" or "uninstall VERSION", also after "--".
    # Empty for any other argument shape.
    single_operand() {
      operand=""
      if [ "\$#" -eq 2 ]; then
        operand="\$2"
      elif [ "\$#" -eq 3 ] && [ "\$2" = "--" ]; then
        operand="\$3"
      fi
    }

    if [ "\''${1:-}" = "link" ]; then
      resolve_chelis_home
      set_gc_root_paths
      "$out/libexec/chelisup" "\$@" || exit \$?
      single_operand "\$@"
      if [ -z "\$operand" ] || [ ! -d "\$operand" ]; then
        exit 0
      fi
      link_target="\$(${pkgs.coreutils}/bin/realpath -e -- "\$operand")"
      for entry in "\$chelis_home/toolchains"/*; do
        if [ ! -L "\$entry" ] \
          || [ "\$(${pkgs.coreutils}/bin/readlink "\$entry")" != "\$link_target" ]; then
          continue
        fi
        toolchain_root="\$gc_root_dir/toolchain-\''${entry##*/}"
        case "\$link_target" in
          ${builtins.storeDir}/*)
            store_rest="\''${link_target#${builtins.storeDir}/}"
            store_path="${builtins.storeDir}/\''${store_rest%%/*}"
            ${pkgs.coreutils}/bin/mkdir -p "\$gc_root_dir"
            if ! ${pkgs.nix}/bin/nix-store \
              --add-root "\$toolchain_root" \
              --realise "\$store_path" >/dev/null; then
              printf 'chelisup: failed to add the Nix GC root %s\n' "\$toolchain_root" >&2
              exit 1
            fi
            ;;
          *)
            ${pkgs.coreutils}/bin/rm -f "\$toolchain_root"
            ;;
        esac
      done
      exit 0
    fi

    if [ "\''${1:-}" = "uninstall" ]; then
      resolve_chelis_home
      set_gc_root_paths
      "$out/libexec/chelisup" "\$@" || exit \$?
      single_operand "\$@"
      if [ -z "\$operand" ]; then
        exit 0
      fi
      toolchain_root="\$gc_root_dir/toolchain-\$operand"
      if [ -e "\$toolchain_root" ] || [ -L "\$toolchain_root" ]; then
        ${pkgs.coreutils}/bin/rm -f "\$toolchain_root"
        printf 'removed %s\n' "\$toolchain_root"
      fi
      ${pkgs.coreutils}/bin/rmdir "\$gc_root_dir" 2>/dev/null || true
      exit 0
    fi

    exec "$out/libexec/chelisup" "\$@"
    EOF
    chmod 0755 $out/bin/chelisup
  '';
  # The arguments of the `bin/chelis` wrapper, one makeWrapper argument for
  # each element. `chelis build` finds the C compiler, `ar`, and the linker on
  # PATH. The wrapper puts the C compiler and the binutils of stdenv first, so
  # a build needs no host toolchain. The wrapper gives the compiler its
  # libexec path as argv[0]. The compiler identifies its image without
  # argv[0] (chelis#3649), so the wrapper needs no --argv0.
  # On Linux, a program whose C includes chelis_blas.h links OpenBLAS. The
  # compiler clears the C compiler environment. Thus the wrapper declares the
  # OpenBLAS directories through the two declared variables of
  # spec/08-backends.md section 7. The OpenBLAS directories come before the
  # directories that the user declares.
  chelisWrapperArgs = [
    "--prefix"
    "PATH"
    ":"
    (lib.makeBinPath [
      pkgs.stdenv.cc
      pkgs.stdenv.cc.bintools
    ])
  ]
  ++ lib.optionals pkgs.stdenv.hostPlatform.isLinux [
    "--prefix"
    "CHELIS_C_INCLUDE_PATH"
    ":"
    "${lib.getDev pkgs.openblas}/include"
    "--prefix"
    "CHELIS_C_LIBRARY_PATH"
    ":"
    "${lib.getLib pkgs.openblas}/lib"
  ];
  chelis =
    pkgs.runCommand "chelis-${version}"
      {
        inherit version;
        meta = packageMeta // {
          description = "Compiler for Chelis, a numerical language with typed tensor shapes and checked properties";
          mainProgram = "chelis";
        };
        nativeBuildInputs = [
          pkgs.makeBinaryWrapper
          pkgs.python311
        ];
      }
      ''
        mkdir -p $out/bin $out/lib $out/include $out/libexec
        cp ${compiler}/bin/chelis $out/libexec/chelis
        makeBinaryWrapper $out/libexec/chelis $out/bin/chelis \
          ${lib.escapeShellArgs chelisWrapperArgs}
        cp ${runtime}/lib/libchelis_runtime.a $out/lib/libchelis_runtime.a
        cp ${runtime}/include/*.h $out/include/
        export_dir="$TMPDIR/combined-export"
        $out/bin/chelis runtime export "$export_dir"
        python3 ${packageScripts}/verify_runtime_package.py "$export_dir" "$out"
      '';
in
{
  # The chelisup launcher above keeps its own section, so its metadata is
  # added here.
  chelisup = lib.addMetaAttrs (
    packageMeta
    // {
      description = "Installer and version selector for Chelis toolchains";
      mainProgram = "chelisup";
    }
  ) chelisup;
  inherit
    cargoGraph
    chelis
    chelisWrapperArgs
    chelisupCrate
    compiler
    compilerCrate
    crate2nixCommand
    crate2nixVersion
    crateSource
    packageScripts
    runtime
    source
    toolchain
    version
    ;
}
