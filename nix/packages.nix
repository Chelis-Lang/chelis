{
  pkgs,
  lib,
  root,
  cvc5,
}:
let
  sources = import ./source.nix { inherit lib; };
  source = sources.package;
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
  # Crate builds need only rustc, cargo, and rust-std. So the package
  # toolchain is the minimal profile of the version that rust-toolchain.toml
  # pins. The Devenv toolchain keeps the components that the file names.
  toolchain = pkgs.rust-bin.fromRustupToolchain {
    inherit ((lib.importTOML (root + "/rust-toolchain.toml")).toolchain) channel;
    profile = "minimal";
  };
  # The chelis-std inputs that the chelis-std-bundle build script packs
  # (crates/chelis-std-bundle/build/stage.rs). These are reef.toml, each
  # declared source root that exists, and each declared metadata file.
  chelisStd = "packages/chelis-std";
  chelisStdPackage = (lib.importTOML (sources.root + "/${chelisStd}/reef.toml")).package;
  chelisStdInputs = map (path: "${chelisStd}/${path}") (
    [
      "reef.toml"
      "src"
    ]
    ++ lib.filter (path: builtins.pathExists (sources.root + "/${chelisStd}/${path}")) (
      chelisStdPackage.additional_sources or [ ]
    )
    ++ lib.optional (chelisStdPackage ? readme) chelisStdPackage.readme
    ++ lib.optional (chelisStdPackage ? license-file) chelisStdPackage.license-file
  );
  hipRuntimeSources = map (name: "crates/chelis-backend-hip/runtime/${name}") [
    "chelis_device_descriptor.h"
    "chelis_device_owner.cpp"
    "chelis_device_owner.h"
    "chelis_hip_runtime.h"
  ];
  # The paths that a workspace crate compile reads outside its own source.
  # These are include_str! and include_bytes! arguments, build script inputs,
  # and files under tests/ that non-test code embeds. A crate without an
  # entry builds from its own directory only.
  crateBuildReads = {
    "chelis-cli" = hipRuntimeSources ++ [
      "crates/chelis-backend-metal/runtime/chelis_metal_runtime.h"
    ];
    "chelis-compiler-api" =
      map (name: "crates/chelis-runtime/include/${name}") [
        "chelis_blas.h"
        "chelis_runtime.h"
        "chelis_runtime_dtype.h"
        "chelis_runtime_views.h"
      ]
      ++ hipRuntimeSources
      ++ [ "packages/chelis-clarabel/src/qp.ch" ];
    "chelis-cove" = [
      "grammars/tree-sitter-chelis-deep/queries/highlights.scm"
      "grammars/tree-sitter-chelis-surf/queries/highlights.scm"
    ];
    "chelis-crmath" = map (name: "crates/chelis-crmath/tests/fixtures/${name}") [
      "binary64_worst_cases.txt"
      "canary.txt"
      "profile_obligations.txt"
    ];
    "chelis-std-bundle" = chelisStdInputs;
    "tree-sitter-chelis" = [
      "grammars/tree-sitter-chelis-deep/src/parser.c"
      "grammars/tree-sitter-chelis-deep/src/tree_sitter"
      "grammars/tree-sitter-chelis-surf/src/parser.c"
      "grammars/tree-sitter-chelis-surf/src/scanner.cc"
      "grammars/tree-sitter-chelis-surf/src/tree_sitter"
    ];
  };
  workspaceCrates = builtins.attrNames cargoGraph.workspaceMembers;
  crateSources =
    let
      unknown = lib.subtractLists workspaceCrates (builtins.attrNames crateBuildReads);
    in
    if unknown != [ ] then
      throw "nix/packages.nix: crateBuildReads names no workspace crate ${toString unknown}"
    else
      lib.genAttrs workspaceCrates (
        name:
        sources.crate {
          inherit (cargoGraph.internal.crates.${name}) src;
          buildReads = crateBuildReads.${name} or [ ];
        }
      );
  # Every file that a workspace crate build reads. The native lane check
  # records the digest of this view.
  crateSource = sources.workspaceView (
    lib.fileset.unions (map (crate: crate.fileset) (builtins.attrValues crateSources))
  );
  # `cargo build --release` gives each crate the codegen units of the release
  # profile, 16 when Cargo.toml sets no value. buildRustCrate uses one codegen
  # unit by default, which makes each long workspace crate compile serially.
  # Only the workspace crates get this value. Thus the third-party crate
  # derivations stay the same, and their cached outputs stay valid. An edit
  # of workspace source rebuilds workspace crates only.
  releaseCodegenUnits = manifest.profile.release.codegen-units or 16;
  buildRustCrateForPkgs =
    cratePkgs:
    cratePkgs.buildRustCrate.override {
      cargo = toolchain;
      rustc = toolchain;
      defaultCrateOverrides =
        cratePkgs.defaultCrateOverrides
        // lib.mapAttrs (
          _name: crate: _attrs:
          builtins.removeAttrs crate [ "fileset" ] // { codegenUnits = releaseCodegenUnits; }
        ) crateSources
        // {
          "cvc5-sys" = attrs: {
            nativeBuildInputs = (attrs.nativeBuildInputs or [ ]) ++ [
              pkgs.llvmPackages.libclang
              pkgs.pkg-config
            ];
            CVC5_DIR = "${cvc5.dir}";
            LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
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
        ${compilerCrate}/bin/chelis runtime export "$export_dir"
        install -Dm444 "$export_dir/libchelis_runtime.a" $out/lib/libchelis_runtime.a
        mkdir -p $out/include
        ${lib.concatMapStringsSep "\n" (header: ''
          install -Dm444 "$export_dir/${header}" $out/include/${header}
        '') contracts.publicRuntimeHeaders}
        python3 ${packageScripts}/verify_runtime_package.py "$export_dir" "$out"
      '';
  # The compiler without the C toolchain, in the layout of the release
  # tarball. bin/chelis is the binary of the compiler crate, which `chelis`
  # also runs, so the store holds one copy. The user supplies the C compiler
  # and binutils on PATH. verify_runtime_package.py needs real runtime files,
  # so the archive and the headers are copies.
  chelis-unwrapped =
    pkgs.runCommand "chelis-unwrapped-${version}"
      {
        inherit version;
        meta = packageMeta // {
          description = "Chelis compiler without a C toolchain, for a host C compiler and binutils";
          mainProgram = "chelis";
        };
        nativeBuildInputs = [ pkgs.python311 ];
      }
      ''
        mkdir -p $out/bin $out/lib $out/include
        test -x ${compilerCrate}/bin/chelis
        ln -s ${compilerCrate}/bin/chelis $out/bin/chelis
        cp ${runtime}/lib/libchelis_runtime.a $out/lib/libchelis_runtime.a
        cp ${runtime}/include/*.h $out/include/
        export_dir="$TMPDIR/unwrapped-export"
        $out/bin/chelis runtime export "$export_dir"
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

    # The nix-store of the Nix installation of the user. The package brings no
    # Nix of its own (chelis#3910). The search reads PATH first and then the
    # standard profile locations. It skips each entry that is not absolute.
    find_nix_store() {
      nix_store=""
      nix_store_dirs=()
      IFS=: read -r -a nix_store_dirs <<< "\''${PATH:-}"
      nix_store_dirs+=(/nix/var/nix/profiles/default/bin /run/current-system/sw/bin)
      if [ -n "\''${HOME:-}" ]; then
        nix_store_dirs+=("\$HOME/.nix-profile/bin")
      fi
      if [ -n "\''${XDG_STATE_HOME:-}" ]; then
        nix_store_dirs+=("\$XDG_STATE_HOME/nix/profile/bin")
      elif [ -n "\''${HOME:-}" ]; then
        nix_store_dirs+=("\$HOME/.local/state/nix/profile/bin")
      fi
      for nix_store_dir in "\''${nix_store_dirs[@]}"; do
        case "\$nix_store_dir" in
          /*) ;;
          *) continue ;;
        esac
        if [ -f "\$nix_store_dir/nix-store" ] && [ -x "\$nix_store_dir/nix-store" ]; then
          nix_store="\$nix_store_dir/nix-store"
          return 0
        fi
      done
      return 1
    }

    # An operation that adds a Nix GC root calls this before it changes a root
    # or runs the installer. Without nix-store, the operation stops.
    require_nix_store() {
      if ! find_nix_store; then
        printf 'chelisup: cannot add the Nix GC root %s: nix-store is not on PATH or in a standard Nix profile\n' "\$1" >&2
        exit 1
      fi
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

    # Remove each toolchain root whose entry in toolchains/ is no longer a
    # link into the store path of that root. An install replaces the link of
    # a version with a real directory.
    remove_stale_toolchain_roots() {
      for toolchain_root in "\$gc_root_dir"/toolchain-*; do
        if [ ! -L "\$toolchain_root" ]; then
          continue
        fi
        entry="\$chelis_home/toolchains/\''${toolchain_root##*/toolchain-}"
        if [ -L "\$entry" ]; then
          root_target="\$(${pkgs.coreutils}/bin/readlink "\$toolchain_root")"
          case "\$(${pkgs.coreutils}/bin/readlink "\$entry")" in
            "\$root_target" | "\$root_target"/*)
              continue
              ;;
          esac
        fi
        ${pkgs.coreutils}/bin/rm -f "\$toolchain_root"
        printf 'removed %s\n' "\$toolchain_root"
      done
    }

    # The real "update" advises the bootstrap installer, which replaces this
    # launcher at \$CHELIS_HOME/bin/chelisup. A newer Nix chelisup comes from a
    # newer release of the flake.
    if { [ "\$#" -eq 1 ] && [ "\$1" = "update" ]; } \
      || { [ "\$#" -eq 2 ] && [ "\$1" = "self" ] && [ "\$2" = "update" ]; }; then
      printf '%s\n' \
        'chelisup: this chelisup comes from the Chelis Nix flake. To get a newer chelisup, run the chelisup of a newer release:' \
        '  nix run github:Chelis-Lang/chelis/vX.Y.Z#chelisup -- install X.Y.Z' \
        'That command installs chelis X.Y.Z and puts that chelisup in the bin directory of CHELIS_HOME.' \
        'Replace X.Y.Z with a release version. Do not run the bootstrap installer, because it replaces this chelisup.' >&2
      exit 0
    fi

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

    # The real "install" and "default" copy the installer over
    # \$CHELIS_HOME/bin/chelisup. Around both, the launcher stages and promotes
    # its GC root and puts itself back after the copy.
    if [ "\''${1:-}" = "install" ] || [ "\''${1:-}" = "default" ]; then
      resolve_chelis_home
      set_gc_root_paths
      require_nix_store "\$gc_root"
      ${pkgs.coreutils}/bin/mkdir -p "\$gc_root_dir"

      if [ -L "\$staging_root" ]; then
        staged_package="\$(${pkgs.coreutils}/bin/readlink "\$staging_root")"
        if copy_matches_package "\$staged_package"; then
          "\$nix_store" \
            --add-root "\$partial_root" \
            --realise "\$staged_package" >/dev/null
        fi
        ${pkgs.coreutils}/bin/rm -f "\$staging_root"
      elif [ -e "\$staging_root" ]; then
        printf '%s\n' 'chelisup: the Nix staging root is not a symbolic link' >&2
        exit 1
      fi

      "\$nix_store" --add-root "\$staging_root" --realise "$out" >/dev/null

      install_status=0
      "$out/libexec/chelisup" "\$@" || install_status=\$?
      if [ "\$install_status" -ne 0 ]; then
        if copy_matches_package "$out"; then
          "\$nix_store" \
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
      if ! "\$nix_store" --add-root "\$gc_root" --realise "$out" >/dev/null; then
        printf '%s\n' 'chelisup: failed to promote the Nix GC root' >&2
        exit 1
      fi
      ${pkgs.coreutils}/bin/rm -f "\$partial_root" "\$staging_root"
      remove_stale_toolchain_roots
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
      single_operand "\$@"
      if [ -n "\$operand" ] && [ -d "\$operand" ]; then
        link_target="\$(${pkgs.coreutils}/bin/realpath -e -- "\$operand")"
        case "\$link_target" in
          ${builtins.storeDir}/*)
            store_rest="\''${link_target#${builtins.storeDir}/}"
            require_nix_store "for ${builtins.storeDir}/\''${store_rest%%/*} in \$gc_root_dir"
            ;;
        esac
      fi
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
            require_nix_store "\$toolchain_root"
            ${pkgs.coreutils}/bin/mkdir -p "\$gc_root_dir"
            if ! "\$nix_store" \
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
  # `libexec/chelis` is a symlink to the binary of the compiler crate, so the
  # store holds one copy of the compiler. The compiler reads no file relative
  # to its own path. The lane check compares canonical paths, and readelf
  # follows the symlink.
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
        ln -s ${compilerCrate}/bin/chelis $out/libexec/chelis
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
    chelis-unwrapped
    chelisWrapperArgs
    chelisupCrate
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
