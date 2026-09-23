{
  lib,
  source,
  toolchain,
  observer,
  workspaceMembers,
}:
cratePkgs: baseBuilder:
let
  lock = lib.importTOML (source + "/Cargo.lock");
  executable = "${observer}/bin/chelis-runtime-identity-build";
  rustcWrapper = cratePkgs.writeShellScriptBin "rustc" ''
    exec ${executable} observe-rustc ${toolchain}/bin/rustc "$@"
  '';
  index = dependency: "${lib.getLib dependency}/chelis-runtime-identity-observations.json";
  bindings = name: dependencies: ''
    ${cratePkgs.jq}/bin/jq -s 'add // []' ${lib.escapeShellArgs (map index dependencies)} \
      ${cratePkgs.writeText "empty-identity-observations.json" "[]"} \
      > "$CHELIS_IDENTITY_STATE/${name}.json"
  '';
  observe =
    attrs:
    let
      member = workspaceMembers.${attrs.crateName} or null;
      matches = builtins.filter (
        package: package.name == attrs.crateName && package.version == attrs.version
      ) lock.package;
      package =
        assert lib.assertMsg (
          builtins.length matches == 1
        ) "runtime identity requires one locked source for ${attrs.crateName}-${attrs.version}";
        builtins.head matches;
      packageSource = if member != null then "path:${member}" else package.source;
      role =
        {
          chelis-runtime = "runtime";
          chelis-cli = "cli";
          chelis-python = "python";
        }
        .${attrs.crateName} or "";
      buildProgram = "target/build/${attrs.crateName}/build_script_build";
      scriptInvocation = "${buildProgram} | tee target/build/${attrs.crateName}.opt";
      observedInvocation = "${executable} observe-build-script ${buildProgram} | tee target/build/${attrs.crateName}.opt";
      buildOutAssignment = "export OUT_DIR=$(pwd)/target/build/${attrs.crateName}.out";
    in
    assert lib.assertMsg (
      attrs.workspace_member != null
    ) "runtime identity producers require an explicit workspace member, not Cargo autodetection";
    attrs
    // {
      nativeBuildInputs = (attrs.nativeBuildInputs or [ ]) ++ [ cratePkgs.python311 ];
      CHELIS_IDENTITY_PROTOCOL = "1";
      CHELIS_IDENTITY_BACKEND = "nix";
      CHELIS_IDENTITY_WORKSPACE = source;
      CHELIS_IDENTITY_PROVENANCE = "sealed-distribution";
      CHELIS_IDENTITY_PYTHON = "${cratePkgs.python311}/bin/python3";
      CHELIS_IDENTITY_PACKAGE_SOURCE = packageSource;
      CHELIS_IDENTITY_PACKAGE_CHECKSUM = package.checksum or "";
      CHELIS_IDENTITY_ROLE = role;
      # Instrument the pinned builder's actual compiler and script execution,
      # not its approximate DEBUG/OPT_LEVEL or transitive DEP_* exports.
      configurePhase =
        assert lib.assertMsg (
          builtins.length (lib.splitString scriptInvocation attrs.configurePhase) == 2
          && lib.hasInfix buildOutAssignment attrs.configurePhase
          && lib.hasInfix "noisily rustc --crate-name build_script_build" attrs.configurePhase
          && lib.hasInfix ''RUSTC_DRIVER="rustc"'' attrs.buildPhase
        ) "buildRustCrate execution seam changed; update the runtime identity observer";
        lib.replaceStrings [ scriptInvocation ] [ observedInvocation ] attrs.configurePhase;
      preConfigure = (attrs.preConfigure or "") + ''
        export CHELIS_IDENTITY_STATE="$NIX_BUILD_TOP/chelis-identity-state"
        mkdir -p "$CHELIS_IDENTITY_STATE"
        export CHELIS_IDENTITY_DEPENDENCIES="$CHELIS_IDENTITY_STATE/dependencies.json"
        export CHELIS_IDENTITY_DIRECT_DEPENDENCIES="$CHELIS_IDENTITY_STATE/direct-dependencies.json"
        export CHELIS_IDENTITY_BUILD_DEPENDENCIES="$CHELIS_IDENTITY_STATE/build-dependencies.json"
        export CHELIS_IDENTITY_BUILD_OUTPUT="$PWD/target/build/${attrs.crateName}.opt"
        ${bindings "dependencies" attrs.completeDeps}
        ${bindings "direct-dependencies" attrs.dependencies}
        ${bindings "build-dependencies" attrs.completeBuildDeps}
        export PATH="${rustcWrapper}/bin:$PATH"
      '';
      postConfigure = (attrs.postConfigure or "") + ''
        if test -f "$CHELIS_IDENTITY_STATE/build-script.json"; then
          export CHELIS_IDENTITY_BUILD_SCRIPT="$CHELIS_IDENTITY_STATE/build-script.json"
        fi
      '';
      preBuild = (attrs.preBuild or "") + ''
        # buildRustCrate renames rustc's underscore binary name to its declared
        # Cargo target name. Bind that exact move before installation/fixup.
        identity_build_bin_definition="$(declare -f build_bin)"
        if [[ "$identity_build_bin_definition" != *'mv "$out_dir/$crate_name_" "$out_dir/$crate_name"'* ]]; then
          echo "buildRustCrate binary relocation seam changed" >&2
          exit 1
        fi
        eval "''${identity_build_bin_definition/build_bin/chelis_identity_build_bin}"
        export -f chelis_identity_build_bin
        build_bin() {
          chelis_identity_build_bin "$@" || return
          local declared="$1" observed="''${1//-/_}" directory="''${3:-target/bin}"
          if [[ "$declared" != "$observed" ]]; then
            local suffix=""
            if [[ -f "$directory/$declared.wasm" ]]; then suffix=".wasm"; fi
            ${executable} observe-output-move \
              "$directory/$observed$suffix" "$directory/$declared$suffix"
          fi
        }
      '';
      # These are the pinned builder's actual install mappings, also used for
      # path-valued links metadata consumed by downstream build scripts.
      installPhase =
        assert lib.assertMsg (
          lib.hasInfix "cp -r target/build/* $lib/lib" attrs.installPhase
          && lib.hasInfix "cp -r target/lib/* $lib/lib" attrs.installPhase
          && lib.hasInfix "cp -rP target/bin/* $out/bin" attrs.installPhase
        ) "buildRustCrate installation seam changed; update receipt relocation";
        attrs.installPhase;
      # Installing after fixup both binds the final bytes and re-decodes producer
      # records. Cached/substituted dependencies carry these same immutable indices.
      postFixup = (attrs.postFixup or "") + ''
        ${executable} install-observations \
          --state "$CHELIS_IDENTITY_STATE" --lib-dir "$lib" --bin-dir "$out/bin" \
          --build-out-dir "$OUT_DIR" --installed-build-out-dir "$lib/lib/${attrs.crateName}.out"
      '';
    };
in
baseBuilder.override {
  stdenv = cratePkgs.stdenv // {
    mkDerivation = attrs: cratePkgs.stdenv.mkDerivation (observe attrs);
  };
}
