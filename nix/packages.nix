{
  pkgs,
  lib,
  root,
  crate2nix,
  cvc5,
}:
let
  source = import ./source.nix { inherit lib root; };
  crateSource = import ./source.nix {
    inherit lib root;
    includeRoots = [
      "crates"
      "grammars"
      "tree-sitter-chelis"
    ];
  };
  manifest = lib.importTOML (source + "/Cargo.toml");
  version = manifest.workspace.package.version;
  crate2nixManifest = lib.importTOML (crate2nix + "/crate2nix/Cargo.toml");
  crate2nixVersion =
    assert crate2nixManifest.package.version == "0.15.0";
    crate2nixManifest.package.version;
  crate2nixTools = pkgs.callPackage (crate2nix + "/tools.nix") { };
  toolchain = pkgs.rust-bin.fromRustupToolchainFile (source + "/rust-toolchain.toml");
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
  generatedCargoNix =
    assert crate2nixVersion == "0.15.0";
    (crate2nixTools.generatedCargoNix {
      name = "chelis";
      src = source;
      cargo = toolchain;
      additionalCargoNixArgs = [
        "--no-default-features"
        "--features"
        "chelis-cli/smt"
      ];
    }).overrideAttrs
      (_: {
        CARGO_NET_OFFLINE = "true";
      });
  cargoGraph = import generatedCargoNix {
    inherit buildRustCrateForPkgs pkgs;
    rootFeatures = [ ];
  };
  compilerCrate = cargoGraph.workspaceMembers."chelis-cli".build.override {
    features = [
      "smt"
      "sealed-runtime"
    ];
  };
  runtimeCrate = cargoGraph.workspaceMembers."chelis-runtime".build.override {
    features = [ ];
  };
  chelisupCrate = cargoGraph.workspaceMembers."chelisup".build.override {
    features = [ ];
  };
  compiler = pkgs.runCommand "chelis-cli-${version}" { } ''
    test -x ${compilerCrate}/bin/chelis
    install -Dm755 ${compilerCrate}/bin/chelis $out/bin/chelis
  '';
  runtime = pkgs.runCommand "chelis-runtime-${version}" { } ''
    artifact="$(find ${runtimeCrate.lib}/lib -type f -name 'libchelis_runtime-*.a' -print -quit)"
    test -n "$artifact"
    install -Dm444 "$artifact" $out/lib/libchelis_runtime.a
    mkdir -p $out/include
    ${lib.concatMapStringsSep "\n" (header: ''
      install -Dm444 ${source}/crates/chelis-runtime/include/${header} $out/include/${header}
    '') (import ./contracts.nix).publicRuntimeHeaders}
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

    exec "$out/libexec/chelisup" "\$@"
    EOF
    chmod 0755 $out/bin/chelisup
  '';
  chelis =
    pkgs.runCommand "chelis-${version}"
      {
        inherit version;
        nativeBuildInputs = [ pkgs.jq ];
        meta.mainProgram = "chelis";
      }
      ''
        mkdir -p $out/bin $out/lib $out/include
        cp ${compiler}/bin/chelis $out/bin/chelis
        # Ship the runtime the compiler carries (spec/08-backends.md §2.1).
        export_dir="$TMPDIR/runtime-export"
        ${compiler}/bin/chelis runtime export "$export_dir"
        receipt="$export_dir/chelis_runtime.receipt.json"
        test "$(jq -r .mode "$receipt")" = sealed
        cp "$export_dir/libchelis_runtime.a" $out/lib/libchelis_runtime.a
        archive_sha256="$(sha256sum $out/lib/libchelis_runtime.a | cut -d ' ' -f 1)"
        test "$archive_sha256" = "$(jq -r .archive_sha256 "$receipt")"
        ${lib.concatMapStringsSep "\n" (header: ''
          cp "$export_dir/${header}" $out/include/${header}
        '') (import ./contracts.nix).publicRuntimeHeaders}
      '';
in
{
  inherit
    cargoGraph
    chelis
    chelisup
    chelisupCrate
    compiler
    compilerCrate
    crate2nixVersion
    generatedCargoNix
    runtime
    runtimeCrate
    source
    toolchain
    version
    ;
}
