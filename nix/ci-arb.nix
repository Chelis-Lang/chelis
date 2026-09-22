{
  pkgs,
  lib,
  root,
}:
let
  cargoLock = lib.importTOML (root + "/Cargo.lock");
  locked = name: lib.findFirst (package: package.name == name) null cargoLock.package;
  arb = locked "arb-sys";
  arbRevision = lib.last (lib.splitString "#" (arb.source or ""));
  flint = locked "flint-sys";
  gmp = locked "gmp-mpfr-sys";
  compiler = "${pkgs.stdenv.cc}/bin/cc";
  cflags = "-std=gnu17 -fPIC";

  # Reuse the workspace's exact dependency graph, not a second Cargo resolution.
  # This producer builds arb-sys itself, so only that root loses source identity.
  dependency =
    specification:
    let
      parts = lib.splitString " " specification;
      matches = builtins.filter (
        package:
        package.name == builtins.head parts
        && (builtins.length parts == 1 || package.version == builtins.elemAt parts 1)
      ) cargoLock.package;
    in
    assert lib.assertMsg (
      builtins.length matches == 1
    ) "The CI native numeric dependency must resolve uniquely in Cargo.lock";
    builtins.head matches;
  node = package: {
    key = "${package.name}-${package.version}";
    inherit package;
  };
  closure = builtins.genericClosure {
    startSet = [ (node arb) ];
    operator =
      item: map (specification: node (dependency specification)) (item.package.dependencies or [ ]);
  };
  packages = map (
    item:
    if item.package.name == "arb-sys" then
      builtins.removeAttrs item.package [
        "source"
        "checksum"
      ]
    else
      item.package
  ) closure;
  # Cargo lock entries contain strings and string arrays, whose JSON encoding is
  # also valid TOML. Keep the generated lock independent of unrelated packages.
  lockContents =
    "version = 4\n\n"
    + lib.concatMapStringsSep "\n\n" (
      package:
      "[[package]]\n"
      + lib.concatMapStringsSep "\n" (key: "${key} = ${builtins.toJSON package.${key}}") (
        builtins.attrNames package
      )
    ) packages
    + "\n";
  lockFile = pkgs.writeText "ci-arb-Cargo.lock" lockContents;

  cache = pkgs.rustPlatform.buildRustPackage {
    pname = "chelis-ci-native-numeric-cache";
    version = arb.version;
    src = pkgs.fetchFromGitHub {
      owner = "Chelis-Lang";
      repo = "arb-sys";
      rev = arbRevision;
      hash = "sha256-6h/SdhNEVgVSD8DOlnJcSrAlc0LK1yc6GZNzzVkVHEg=";
    };
    cargoLock.lockFileContents = lockContents;
    cargoBuildFlags = [ "--locked" ];
    nativeBuildInputs = [ pkgs.m4 ];
    CC = compiler;
    CFLAGS = cflags;
    RUSTC_WRAPPER = "";
    # The pinned upstream source already supplies the bundled MPFR prefix.
    postPatch = ''
      cp ${lockFile} Cargo.lock
      chmod u+w Cargo.lock
    '';
    preBuild = ''
      # stdenv setup hooks replace CC; the sys-cache key must match CI exactly.
      export CC=${lib.escapeShellArg compiler}
      export CFLAGS=${lib.escapeShellArg cflags}
      export GMP_MPFR_SYS_CACHE="$out/gmp"
      export FLINT_SYS_CACHE="$out/flint"
      export ARB_SYS_CACHE="$out/arb"
    '';
    # This output is a C archive cache. Check that ABI directly rather than
    # arb-sys's unpublished Rust test setup (which omits its quickcheck dependency).
    # Workspace Arb soundness tests still exercise the real Rust consumer.
    doCheck = true;
    checkPhase =
      let
        probe = pkgs.writeText "native-numeric-check.c" ''
          #include <arb.h>
          #include <stdio.h>
          int main(void) {
            arb_t value, result;
            arb_init(value);
            arb_init(result);
            arb_set_ui(value, 9);
            arb_sqrt(result, value, 128);
            int ok = arb_is_exact(result) && arb_contains_si(result, 3);
            arb_clear(result);
            arb_clear(value);
            if (!ok) return 1;
            puts("native cache sqrt(9) = 3 (exact)");
            return 0;
          }
        '';
      in
      ''
        runHook preCheck
        ${pkgs.python3}/bin/python3 - <<'PY'
        import os, shlex, subprocess
        from pathlib import Path
        root = Path(os.environ["out"])
        archives = []
        for family, name in (("arb", "arb"), ("flint", "flint"), ("gmp", "mpfr"), ("gmp", "gmp")):
            matches = list((root / family).rglob("lib" + name + ".a"))
            if len(matches) != 1:
                raise RuntimeError("ambiguous native cache archive: " + name)
            archives.append(matches[0])
        include = Path("native-check-include")
        include.mkdir()
        (include / "flint").symlink_to(archives[1].parent, target_is_directory=True)
        command = [os.environ["CC"], *shlex.split(os.environ["CFLAGS"])]
        command += ["-I" + str(include)]
        command += ["-I" + str(path.parent) for path in archives]
        command += ["${probe}", *map(str, archives), "-lm", "-lpthread", "-o", "native-numeric-check"]
        subprocess.run(command, check=True)
        subprocess.run(["./native-numeric-check"], check=True)
        PY
        runHook postCheck
      '';
    installPhase = ''
      runHook preInstall
      test -s "$out/arb/c2b7e36/CC-${compiler}/libarb.a"
      test -s "$out/arb/c2b7e36/CC-${compiler}/arb.h"
      test -d "$out/gmp"
      test -d "$out/flint"
      runHook postInstall
    '';
    # These files are the sys crates' native archive/header cache format, not
    # installable Rust libraries. Preserve their exact compiler-keyed layout.
    dontFixup = true;
  };
in
assert lib.assertMsg (
  arb != null
  && arb.version == "0.3.6"
  && builtins.match "[0-9a-f]{40}" arbRevision != null
  &&
    (arb.source or "") == "git+https://github.com/Chelis-Lang/arb-sys?rev=${arbRevision}#${arbRevision}"
  && flint != null
  && flint.version == "0.7.3"
  && gmp != null
  && gmp.version == "1.7.1"
) "nix/ci-arb.nix must be updated with the locked native numeric build contracts";
assert lib.assertMsg (
  pkgs.stdenv.buildPlatform == pkgs.stdenv.hostPlatform
  && (pkgs.stdenv.isLinux || pkgs.stdenv.isDarwin)
) "The CI native numeric cache supports native Linux and Darwin only";
{
  inherit cache compiler cflags;
}
