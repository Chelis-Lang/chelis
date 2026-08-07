{
  lib,
  pkgs,
  root,
  rustToolchain,
}:
let
  source = import ./source.nix { inherit lib root; };
  manifest = lib.importTOML (source + "/Cargo.toml");
  version = manifest.workspace.package.version;
  system = pkgs.stdenv.hostPlatform.system;
  isLinux = system == "x86_64-linux";
  isDarwin = system == "aarch64-darwin";
  supported = isLinux || isDarwin;
  platformSlug = if isLinux then "linux-x86_64" else "darwin-arm64";
  assetName = if isLinux then "chelisup-linux-x86_64" else "chelisup-darwin-arm64";
  releasePkgs = if isLinux then pkgs.pkgsStatic else pkgs;
  releaseToolchain =
    if isLinux then
      rustToolchain.override {
        targets = [ "x86_64-unknown-linux-musl" ];
      }
    else
      rustToolchain;
  buildRustCrateForPkgs =
    cratePkgs:
    cratePkgs.buildRustCrate.override {
      cargo = releaseToolchain;
      rustc = releaseToolchain;
    };
  cargoGraph = import (root + "/Cargo.nix") {
    pkgs = releasePkgs;
    inherit buildRustCrateForPkgs;
    rootFeatures = [ ];
  };
  chelisupCrate = cargoGraph.workspaceMembers."chelisup".build.override {
    features = [ ];
  };
  nativeBuildInputs = [
    pkgs.coreutils
    pkgs.file
  ]
  ++ lib.optionals isLinux [ pkgs.binutils ]
  ++ lib.optionals isDarwin [ pkgs.cctools ];
  platformChecks =
    if isLinux then
      ''
        file "$out/$asset" | grep -F 'x86-64'
        if readelf -l "$out/$asset" | grep -q 'INTERP'; then
          echo "release-chelisup: Linux executable has an interpreter segment" >&2
          exit 1
        fi
        if readelf -d "$out/$asset" | grep -q 'NEEDED'; then
          echo "release-chelisup: Linux executable has a dynamic dependency" >&2
          exit 1
        fi
      ''
    else
      ''
        nix_iconv="$(otool -L "$out/$asset" | awk '
          NR > 1 && $1 ~ /^\/nix\/store\/.*\/libiconv\.2\.dylib$/ {
            print $1
            exit
          }
        ')"
        if [ -n "$nix_iconv" ]; then
          install_name_tool \
            -change "$nix_iconv" \
            /usr/lib/libiconv.2.dylib \
            "$out/$asset"
        fi
        file "$out/$asset" | grep -F 'arm64'
        otool -L "$out/$asset" | awk '
          NR > 1 && index($1, "/usr/lib/") != 1 \
            && index($1, "/System/Library/Frameworks/") != 1 {
            print "release-chelisup: non-Apple runtime dependency: " $1 > "/dev/stderr"
            bad = 1
          }
          END { exit bad }
        '
        if otool -L "$out/$asset" | tail -n +2 | grep -F '/nix/store/'; then
          echo "release-chelisup: Darwin executable retains a Nix store load path" >&2
          exit 1
        fi
      '';
in
assert lib.assertMsg supported "release-chelisup supports only x86_64-linux and aarch64-darwin";
pkgs.runCommand "chelisup-release-${version}-${platformSlug}"
  {
    inherit nativeBuildInputs;
    meta = {
      description = "Portable chelisup ${version} executable for ${platformSlug}";
      license = lib.licenses.mit;
      platforms = [ system ];
    };
  }
  ''
    mkdir -p "$out"
    asset=${lib.escapeShellArg assetName}
    install -Dm755 ${chelisupCrate}/bin/chelisup "$out/$asset"

    ${platformChecks}

    actual_version="$("$out/$asset" --version)"
    expected_version=${lib.escapeShellArg "chelisup ${version}"}
    if [ "$actual_version" != "$expected_version" ]; then
      echo "release-chelisup: expected version '$expected_version', got '$actual_version'" >&2
      exit 1
    fi
    "$out/$asset" --help >/dev/null

    (
      cd "$out"
      sha256sum "$asset" > "$asset.sha256"
      sha256sum -c "$asset.sha256"
    )

    actual_inventory="$(find "$out" -mindepth 1 -maxdepth 1 -type f -printf '%f\n' | sort)"
    expected_inventory="$(printf '%s\n%s\n' "$asset" "$asset.sha256" | sort)"
    if [ "$actual_inventory" != "$expected_inventory" ]; then
      echo "release-chelisup: unexpected output inventory" >&2
      printf 'expected:\n%s\nactual:\n%s\n' "$expected_inventory" "$actual_inventory" >&2
      exit 1
    fi
    if [ -n "$(find "$out" -mindepth 1 -maxdepth 1 ! -type f -print)" ]; then
      echo "release-chelisup: output root contains a non-file entry" >&2
      exit 1
    fi
  ''
