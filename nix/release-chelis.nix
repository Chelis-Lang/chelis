{
  lib,
  pkgs,
  root,
  rustToolchain,
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
  contracts = import ./contracts.nix;
  manifest = lib.importTOML (source + "/Cargo.toml");
  version = manifest.workspace.package.version;
  cvc5 = import ./cvc5.nix {
    inherit pkgs lib root;
  };
  system = pkgs.stdenv.hostPlatform.system;
  isLinux = system == "x86_64-linux";
  isDarwin = system == "aarch64-darwin";
  supported = isLinux || isDarwin;
  platformSlug = if isLinux then "linux-x86_64" else "darwin-arm64";
  # cvc5-sys emits an explicit C++ runtime link, and the zstd crate resolves
  # libzstd dynamically. A driver flag such as -static-libstdc++ cannot
  # override an explicit -l, but the linker searches command-line -L
  # directories before the driver defaults for every -l. A directory holding
  # ONLY the static archives therefore forces static resolution (the
  # manylinux pattern). On Linux, libgcc_s.so.1 stays dynamic: Rust std
  # references the shared unwinder explicitly, and every supported consumer
  # system carries it. On Darwin, the Nix clang already links the system
  # /usr/lib/libc++.1.dylib, so only libzstd needs the static archive.
  zstdStatic = pkgs.zstd.override { static = true; };
  staticLibDir = pkgs.runCommand "chelis-release-static-lib-dir" { } ''
    mkdir -p $out/lib
    ${lib.optionalString isLinux ''
      ln -s ${pkgs.stdenv.cc.cc}/lib/libstdc++.a $out/lib/libstdc++.a
    ''}
    ln -s ${zstdStatic.out}/lib/libzstd.a $out/lib/libzstd.a
  '';
  # Mirror of the defaultCrateOverrides set in nix/packages.nix, plus the
  # static-archive link directory on the chelis-cli crate. Keep the two sets
  # aligned; a missing override fails the crate build loudly.
  crateOverrides = pkgs.defaultCrateOverrides // {
    "chelis-cli" = attrs: {
      src = crateSource;
      sourceRoot = "chelis-source/crates/chelis-cli";
      extraRustcOpts = [ "-Lnative=${staticLibDir}/lib" ];
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
  buildRustCrateForPkgs =
    cratePkgs:
    cratePkgs.buildRustCrate.override {
      cargo = rustToolchain;
      rustc = rustToolchain;
      defaultCrateOverrides = crateOverrides;
    };
  cargoGraph = import (root + "/Cargo.nix") {
    inherit buildRustCrateForPkgs pkgs;
    rootFeatures = [ ];
  };
  compilerCrate = cargoGraph.workspaceMembers."chelis-cli".build.override {
    features = [ "smt" ];
  };
  runtimeCrate = cargoGraph.workspaceMembers."chelis-runtime".build.override {
    features = [ ];
  };
  stagingName = "chelis-v${version}-${platformSlug}";
  assetName = "${stagingName}.tar.gz";
  allowedNeeded = [
    "ld-linux-x86-64.so.2"
    "libc.so.6"
    "libdl.so.2"
    "libgcc_s.so.1"
    "libm.so.6"
    "libpthread.so.0"
    "librt.so.1"
  ];
  linuxRewrite = ''
    chmod +w "$staging/bin/chelis"
    strip "$staging/bin/chelis"
    patchelf --set-interpreter /lib64/ld-linux-x86-64.so.2 \
      --remove-rpath "$staging/bin/chelis"
    chmod 555 "$staging/bin/chelis"

    file "$staging/bin/chelis" | grep -F 'x86-64'
    interpreter="$(patchelf --print-interpreter "$staging/bin/chelis")"
    if [ "$interpreter" != /lib64/ld-linux-x86-64.so.2 ]; then
      echo "release-chelis: unexpected interpreter $interpreter" >&2
      exit 1
    fi
    rpath="$(patchelf --print-rpath "$staging/bin/chelis")"
    if [ -n "$rpath" ]; then
      echo "release-chelis: rpath survived the rewrite: $rpath" >&2
      exit 1
    fi
    if readelf -d "$staging/bin/chelis" | grep -F '/nix/store'; then
      echo "release-chelis: /nix/store reference in the dynamic section" >&2
      exit 1
    fi
    readelf -d "$staging/bin/chelis" | grep -F '(NEEDED)' \
      | sed -e 's/.*\[//' -e 's/\]$//' > "$TMPDIR/needed"
    while IFS= read -r soname; do
      case "$soname" in
        ${lib.concatStringsSep "|" allowedNeeded}) ;;
        *)
          echo "release-chelis: forbidden dynamic dependency: $soname" >&2
          exit 1
          ;;
      esac
    done < "$TMPDIR/needed"

    # The recorded glibc floor is a reviewed contract: fail on drift in
    # either direction so a toolchain bump becomes a visible diff.
    floor="$(readelf -W --version-info "$staging/bin/chelis" \
      | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)"
    expected_floor=${lib.escapeShellArg "GLIBC_${contracts.linuxReleaseGlibcFloor}"}
    if [ "$floor" != "$expected_floor" ]; then
      echo "release-chelis: glibc floor drift: recorded $expected_floor, computed $floor" >&2
      echo "release-chelis: update linuxReleaseGlibcFloor in nix/contracts.nix in a reviewed diff" >&2
      exit 1
    fi

    # Run proof against a glibc library set alone, through an explicit
    # loader because the embedded /lib64 interpreter does not exist in the
    # sandbox.
    loader_version="$(${pkgs.glibc}/lib/ld-linux-x86-64.so.2 \
      --library-path ${pkgs.glibc}/lib \
      "$staging/bin/chelis" --version)"
    if [ "$loader_version" != "$expected_version" ]; then
      echo "release-chelis: rewritten binary failed the loader run: $loader_version" >&2
      exit 1
    fi
  '';
  darwinRewrite = ''
    chmod +w "$staging/bin/chelis"
    nix_iconv="$(otool -L "$staging/bin/chelis" | awk '
      NR > 1 && $1 ~ /^\/nix\/store\/.*\/libiconv\.2\.dylib$/ { print $1; exit }
    ')"
    if [ -n "$nix_iconv" ]; then
      install_name_tool -change "$nix_iconv" \
        /usr/lib/libiconv.2.dylib "$staging/bin/chelis"
    fi
    chmod 555 "$staging/bin/chelis"

    file "$staging/bin/chelis" | grep -E 'arm64|aarch64'
    if otool -L "$staging/bin/chelis" | tail -n +2 | grep -F '/nix/store/'; then
      echo "release-chelis: a Nix store load path remains" >&2
      exit 1
    fi
    otool -L "$staging/bin/chelis" | awk '
      NR > 1 && index($1, "/usr/lib/") != 1 \
        && index($1, "/System/Library/Frameworks/") != 1 {
        print "release-chelis: non-Apple runtime dependency: " $1 > "/dev/stderr"
        bad = 1
      }
      END { exit bad }
    '

    # The Darwin sandbox can execute the rewritten binary, so the behavior
    # probes run on the exact staged artifact.
    run_probes "$staging/bin/chelis"
  '';
in
assert lib.assertMsg supported "release-chelis supports only x86_64-linux and aarch64-darwin";
pkgs.runCommand "chelis-release-${version}-${platformSlug}"
  {
    nativeBuildInputs = [
      pkgs.coreutils
      pkgs.file
      pkgs.python311
    ]
    ++ lib.optionals isLinux [
      pkgs.binutils
      pkgs.patchelf
    ]
    ++ lib.optionals isDarwin [ pkgs.cctools ];
    meta = {
      description = "Portable Chelis ${version} toolchain tarball for ${platformSlug}";
      license = lib.licenses.mit;
      platforms = [ system ];
    };
  }
  ''
    export HOME="$TMPDIR/home"
    mkdir -p "$HOME"

    prebuilt=${compilerCrate}/bin/chelis
    test -x "$prebuilt"
    expected_version=${lib.escapeShellArg "chelis ${version}"}

    cp -R ${source}/crates/chelis-cli/tests/fixtures/release_pipe_stage \
      "$TMPDIR/release-fixture"
    chmod -R u+w "$TMPDIR/release-fixture"
    artifact="$(find ${runtimeCrate.lib}/lib -type f -name 'libchelis_runtime-*.a' -print -quit)"
    test -n "$artifact"
    mkdir -p "$TMPDIR/runtime-lib"
    cp "$artifact" "$TMPDIR/runtime-lib/libchelis_runtime.a"
    export CHELIS_RUNTIME_DIR="$TMPDIR/runtime-lib"

    run_probes() {
      probe_bin="$1"
      actual_version="$("$probe_bin" --version)"
      if [ "$actual_version" != "$expected_version" ]; then
        echo "release-chelis: expected '$expected_version', got '$actual_version'" >&2
        exit 1
      fi
      "$probe_bin" --help >/dev/null
      (cd "$TMPDIR/release-fixture" && "$probe_bin" test tests)
      (cd "$TMPDIR" && python3 ${root}/.github/scripts/verify_release_smt.py "$probe_bin")
    }

    # Behavior probes on the crate output. On Linux they cannot rerun after
    # the interpreter rewrite (the sandbox has no /lib64); on Darwin the
    # rewrite block reruns them on the staged binary.
    run_probes "$prebuilt"

    # The runtime archive must stay a glibc-consumer archive: a bundled libc
    # (the musl staticlib failure mode) defines the allocator symbols.
    if nm --defined-only "$TMPDIR/runtime-lib/libchelis_runtime.a" 2>/dev/null \
      | grep -wE 'T (malloc|free|calloc|realloc)'; then
      echo "release-chelis: runtime archive defines libc allocator symbols" >&2
      exit 1
    fi

    staging="$TMPDIR/${stagingName}"
    mkdir -p "$staging/bin" "$staging/lib" "$staging/include"
    cp "$prebuilt" "$staging/bin/chelis"

    ${if isLinux then linuxRewrite else darwinRewrite}

    cp "$TMPDIR/runtime-lib/libchelis_runtime.a" "$staging/lib/libchelis_runtime.a"
    ${lib.concatMapStringsSep "\n" (header: ''
      install -Dm444 ${source}/crates/chelis-runtime/include/${header} \
        "$staging/include/${header}"
    '') contracts.publicRuntimeHeaders}
    install -Dm444 ${root + "/README.md"} "$staging/README.md"
    install -Dm444 ${root + "/LICENSE"} "$staging/LICENSE"

    expected_tree="$(printf '%s\n' \
      bin bin/chelis lib lib/libchelis_runtime.a include \
      ${lib.concatMapStringsSep " " (header: "include/${header}") contracts.publicRuntimeHeaders} \
      README.md LICENSE | sort)"
    actual_tree="$(cd "$staging" && find . -mindepth 1 | sed 's|^\./||' | sort)"
    if [ "$actual_tree" != "$expected_tree" ]; then
      echo "release-chelis: staged tree differs from its exact contract" >&2
      printf 'expected:\n%s\nactual:\n%s\n' "$expected_tree" "$actual_tree" >&2
      exit 1
    fi

    mkdir -p "$out"
    tar -C "$TMPDIR" -czf "$out/${assetName}" ${stagingName}
    (
      cd "$out"
      sha256sum ${assetName} > ${assetName}.sha256
      sha256sum -c ${assetName}.sha256
    )

    actual_inventory="$(cd "$out" && find . -mindepth 1 -maxdepth 1 -type f | sed 's|^\./||' | sort)"
    expected_inventory="$(printf '%s\n%s\n' ${assetName} ${assetName}.sha256 | sort)"
    if [ "$actual_inventory" != "$expected_inventory" ]; then
      echo "release-chelis: unexpected output inventory" >&2
      printf 'expected:\n%s\nactual:\n%s\n' "$expected_inventory" "$actual_inventory" >&2
      exit 1
    fi
    if [ -n "$(find "$out" -mindepth 1 -maxdepth 1 ! -type f)" ]; then
      echo "release-chelis: output root contains a non-file entry" >&2
      exit 1
    fi
  ''
