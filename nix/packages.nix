{
  pkgs,
  lib,
  root,
  crate2nix,
  cvc5,
}:
let
  source = import ./source.nix { inherit lib root; };
  workspaceMembers = builtins.listToAttrs (
    map (
      member:
      let
        package = lib.importTOML (source + "/${member}/Cargo.toml");
      in
      lib.nameValuePair package.package.name member
    ) manifest.workspace.members
  );
  manifest = lib.importTOML (source + "/Cargo.toml");
  version = manifest.workspace.package.version;
  crate2nixManifest = lib.importTOML (crate2nix + "/crate2nix/Cargo.toml");
  crate2nixVersion =
    assert crate2nixManifest.package.version == "0.15.0";
    crate2nixManifest.package.version;
  crate2nixTools = pkgs.callPackage (crate2nix + "/tools.nix") { };
  toolchain = pkgs.rust-bin.fromRustupToolchainFile (source + "/rust-toolchain.toml");
  unobservedBuildRustCrateForPkgs =
    cratePkgs:
    cratePkgs.buildRustCrate.override {
      cargo = toolchain;
      rustc = toolchain;
      # All local crates see the same complete filtered workspace, including
      # root manifests, toolchain, ABI headers and sibling build inputs.
      defaultCrateOverrides =
        cratePkgs.defaultCrateOverrides
        // lib.mapAttrs (_: member: _: {
          src = source;
          sourceRoot = "chelis-source/${member}";
          workspace_member = ".";
        }) workspaceMembers
        // {
          # Avoid buildRustCrate's Cargo-metadata autodetection for Git sources.
          "arb-sys" = _: {
            workspace_member = ".";
          };
          "carcara" = _: {
            workspace_member = "carcara";
          };
          "cvc5-sys" = attrs: {
            nativeBuildInputs = (attrs.nativeBuildInputs or [ ]) ++ [
              pkgs.llvmPackages.libclang
              pkgs.pkg-config
            ];
            CVC5_DIR = "${cvc5.dir}";
            LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          };
          "pyo3-build-config" =
            attrs:
            (cratePkgs.defaultCrateOverrides."pyo3-build-config" or (_: { })) attrs
            // {
              PYO3_PYTHON = "${cratePkgs.python311}/bin/python3";
            };
          "chelis-python" = attrs: {
            src = source;
            sourceRoot = "chelis-source/${workspaceMembers.chelis-python}";
            workspace_member = ".";
            PYO3_PYTHON = "${cratePkgs.python311}/bin/python3";
            buildInputs = (attrs.buildInputs or [ ]) ++ [ cratePkgs.python311 ];
          };
        };
    };
  generatedCargoNix =
    assert crate2nixVersion == "0.15.0";
    (crate2nixTools.generatedCargoNix {
      name = "chelis";
      src = source;
      cargo = toolchain;
      # Generate the optional dependency edges used by the private feature
      # control too. Actual producer features are selected below, not here.
      additionalCargoNixArgs = [
        "--no-default-features"
        "--features"
        "chelis-cli/smt,chelis-python/extension-module,chelis-runtime/ownership-ledger"
      ];
    }).overrideAttrs
      (_: {
        CARGO_NET_OFFLINE = "true";
      });
  # The observer has no runtime dependency. Bootstrap it through an entirely
  # unobserved graph; never recursively depend on its own compiler wrapper.
  bootstrapGraph = import generatedCargoNix {
    inherit pkgs;
    buildRustCrateForPkgs = unobservedBuildRustCrateForPkgs;
    rootFeatures = [ ];
  };
  identityObserver = bootstrapGraph.workspaceMembers."chelis-runtime-identity-build".build.override {
    features = [ ];
  };
  observeBuilder = import ./runtime-identity.nix {
    inherit
      lib
      source
      toolchain
      workspaceMembers
      ;
    observer = identityObserver;
  };
  buildRustCrateForPkgs =
    cratePkgs: observeBuilder cratePkgs (unobservedBuildRustCrateForPkgs cratePkgs);
  cargoGraph = import generatedCargoNix {
    inherit buildRustCrateForPkgs pkgs;
    rootFeatures = [ ];
  };
  # Resolve both real consumer roots together, as one Cargo invocation would.
  # The synthetic root is resolver input only: no crate or descriptor is built
  # for it, and every selected producer retains its own derivation and receipts.
  producerRoot = "chelis-runtime-identity-producers";
  producerGraphFor =
    runtimeFeatures:
    cargoGraph.internal.builtRustCratesWithFeatures {
      packageId = producerRoot;
      features = [ ];
      runTests = false;
      buildRustCrateForPkgsFunc = buildRustCrateForPkgs;
      crateConfigs = cargoGraph.internal.crates // {
        ${producerRoot} = {
          dependencies = [
            {
              name = "chelis-cli";
              packageId = cargoGraph.workspaceMembers."chelis-cli".packageId;
              features = [ "smt" ];
              usesDefaultFeatures = false;
            }
            {
              name = "chelis-python";
              packageId = cargoGraph.workspaceMembers."chelis-python".packageId;
              features = [ "extension-module" ];
              usesDefaultFeatures = false;
            }
          ]
          ++ lib.optional (runtimeFeatures != [ ]) {
            name = "chelis-runtime";
            packageId = cargoGraph.workspaceMembers."chelis-runtime".packageId;
            features = runtimeFeatures;
            usesDefaultFeatures = false;
          };
        };
      };
    };
  producerGraph = producerGraphFor [ ];
  compilerCrate = producerGraph.crates.${cargoGraph.workspaceMembers."chelis-cli".packageId};
  pythonCrate = producerGraph.crates.${cargoGraph.workspaceMembers."chelis-python".packageId};
  runtimeDependency =
    consumer:
    let
      dependencies = builtins.filter (
        dependency: dependency.crateName == "chelis-runtime"
      ) consumer.dependencies;
    in
    assert lib.assertMsg (
      builtins.length dependencies == 1
    ) "each consumer producer must declare exactly one runtime dependency";
    builtins.head dependencies;
  runtimeCrate =
    assert lib.assertMsg (
      (runtimeDependency compilerCrate).drvPath == (runtimeDependency pythonCrate).drvPath
    ) "the CLI and Python producers must consume the same resolved runtime unit";
    runtimeDependency compilerCrate;
  # These real producer variants are private acceptance inputs, not packages.
  # Resolve the changed feature through the same graph, including its optional
  # dependencies, rather than modifying a compiled archive or its receipt.
  runtimeIdentityProducerControls = {
    changedRuntimeCrate =
      let
        graph = producerGraphFor [ "ownership-ledger" ];
      in
      runtimeDependency graph.crates.${cargoGraph.workspaceMembers."chelis-cli".packageId};
    missingInputRuntimeCrate = runtimeCrate.overrideAttrs (_: {
      src = lib.cleanSourceWith {
        name = "chelis-source";
        src = "${source}";
        filter = path: _: toString path != "${source}/crates/chelis-runtime/include/chelis_runtime.h";
      };
    });
  };
  chelisupCrate = cargoGraph.workspaceMembers."chelisup".build.override {
    features = [ ];
  };
  compiler = pkgs.runCommand "chelis-cli-${version}" { } ''
    test -x ${compilerCrate}/bin/chelis
    install -Dm755 ${compilerCrate}/bin/chelis $out/bin/chelis
  '';
  runtime = pkgs.runCommand "chelis-runtime-${version}" { } ''
    export CHELIS_IDENTITY_PYTHON="${pkgs.python311}/bin/python3"
    artifact="$(${identityObserver}/bin/chelis-runtime-identity-build producer-artifact \
      --lib-dir ${runtimeCrate.lib} --kind runtime)"
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
        meta.mainProgram = "chelis";
      }
      ''
        mkdir -p $out/bin $out/lib $out/include
        cp ${compiler}/bin/chelis $out/bin/chelis
        cp ${runtime}/lib/libchelis_runtime.a $out/lib/libchelis_runtime.a
        cp ${runtime}/include/*.h $out/include/
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
    identityObserver
    pythonCrate
    runtime
    runtimeCrate
    runtimeIdentityProducerControls
    source
    toolchain
    version
    ;
}
