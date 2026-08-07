{
  pkgs,
  lib,
  root,
}:
let
  cargoLock = lib.importTOML (root + "/Cargo.lock");
  cvc5Sys = lib.findFirst (package: package.name == "cvc5-sys") null cargoLock.package;
  expectedCvc5SysVersion = "0.3.1";
  version =
    assert cvc5Sys != null && cvc5Sys.version == expectedCvc5SysVersion;
    "1.3.1";
  source = pkgs.fetchFromGitHub {
    owner = "cvc5";
    repo = "cvc5";
    tag = "cvc5-${version}";
    hash = "sha256-nxJjrpWZfYPuuKN4CWxOHEuou4r+MdK0AjdEPZHZbHI=";
  };
  cadical = pkgs.cadical.override { version = "2.1.3"; };
  gmp = pkgs.gmp.override { withStatic = true; };
  symfpu = pkgs.stdenvNoCC.mkDerivation {
    pname = "symfpu-cvc5";
    version = "e6ac3af9c2c574498ea171c957425b407625448b";
    src = pkgs.fetchurl {
      url = "https://github.com/cvc5/symfpu/archive/e6ac3af9c2c574498ea171c957425b407625448b.tar.gz";
      hash = "sha256-gjqmY/zC9oRK5enqg87aTtOTzbPa3vzps8fEHND09wI=";
    };
    installPhase = ''
      mkdir -p "$out/include/symfpu"
      cp -R core utils "$out/include/symfpu/"
    '';
  };
  # LibPoly is cvc5's CAD backend for nonlinear real arithmetic (USE_POLY).
  # cvc5 1.3.1's cmake/FindPoly.cmake pins v0.2.0 and links the
  # position-independent static archives libpicpoly.a / libpicpolyxx.a. The
  # cvc5-sys from-source build takes USE_POLY=ON by default, so match that exact
  # version and archive shape to ship the same solver capability the Cargo build
  # does. The url hash equals cvc5's own FindPoly.cmake URL_HASH for v0.2.0.
  libpoly = pkgs.stdenv.mkDerivation {
    pname = "libpoly-cvc5";
    version = "0.2.0";
    src = pkgs.fetchurl {
      url = "https://github.com/SRI-CSL/libpoly/archive/refs/tags/v0.2.0.tar.gz";
      hash = "sha256-FGrcDT9v6AOK22uLad0WEUpL4S9SDVwfszPzdG0jOr4=";
    };
    nativeBuildInputs = [ pkgs.cmake ];
    buildInputs = [ gmp ];
    cmakeFlags = [
      "-DCMAKE_BUILD_TYPE=Release"
      "-DLIBPOLY_BUILD_PYTHON_API=OFF"
      "-DLIBPOLY_BUILD_STATIC=ON"
      "-DLIBPOLY_BUILD_STATIC_PIC=ON"
      "-DBUILD_TESTING=OFF"
      "-DGMP_INCLUDE_DIR=${gmp.dev}/include"
      "-DGMP_LIBRARY=${gmp}/lib/libgmp.a"
    ];
    # Upstream installs headers plus the default libraries, but not the
    # position-independent static archives cvc5 links. Place them explicitly.
    postInstall = ''
      picpoly="$(find . -name libpicpoly.a -print -quit)"
      picpolyxx="$(find . -name libpicpolyxx.a -print -quit)"
      test -n "$picpoly"
      test -n "$picpolyxx"
      install -Dm444 "$picpoly" "$out/lib/libpicpoly.a"
      install -Dm444 "$picpolyxx" "$out/lib/libpicpolyxx.a"
    '';
  };
  package = pkgs.cvc5.overrideAttrs (old: {
    inherit version;
    src = source;
    postPatch = ''
      substituteInPlace cmake/FindGMP.cmake \
        --replace-fail \
        'LINK_LIBRARIES ''${GMP_LIBRARIES} ''${GMPXX_LIBRARIES}' \
        'LINK_LIBRARIES ''${GMPXX_LIBRARIES} ''${GMP_LIBRARIES}'
    '';
    # Replace nixpkgs' libpoly (0.2.1) with the v0.2.0 cvc5 pins, so the found
    # poly is exactly the one the Cargo build links.
    buildInputs = (builtins.filter (dep: (dep.pname or "") != "libpoly") (old.buildInputs or [ ])) ++ [
      libpoly
    ];
    cmakeFlags = [
      "-DBUILD_SHARED_LIBS=OFF"
      "-DENABLE_GPL=OFF"
      "-DGMP_INCLUDE_DIR=${gmp.dev}/include"
      "-DGMPXX_INCLUDE_DIR=${gmp.dev}/include"
      "-DGMP_LIBRARIES=${gmp}/lib/libgmp.a"
      "-DGMPXX_LIBRARIES=${gmp}/lib/libgmpxx.a"
      "-DSymFPU_INCLUDE_DIR=${symfpu}/include"
      "-DSTATIC_BINARY=OFF"
      "-DUSE_CLN=OFF"
      "-DUSE_POLY=ON"
      "-DPoly_INCLUDE_DIR=${libpoly}/include"
      "-DPoly_LIBRARIES=${libpoly}/lib/libpicpoly.a"
      "-DPolyXX_LIBRARIES=${libpoly}/lib/libpicpolyxx.a"
    ];
    doCheck = false;
  });
  dir = pkgs.runCommand "cvc5-${version}-prebuilt-tree" { } ''
    mkdir -p "$out/cmake" "$out/include" "$out/build/include"
    mkdir -p "$out/build/src" "$out/build/deps/lib"

    cp ${source}/cmake/version-base.cmake "$out/cmake/version-base.cmake"
    cp -R ${package}/include/. "$out/include/"
    cp -R ${package}/include/. "$out/build/include/"
    cp ${package}/lib/libcvc5.a "$out/build/src/libcvc5.a"
    cp ${cadical.lib}/lib/libcadical.a "$out/build/deps/lib/libcadical.a"
    cp ${gmp}/lib/libgmp.a "$out/build/deps/lib/libgmp.a"
    cp ${libpoly}/lib/libpicpoly.a "$out/build/deps/lib/libpicpoly.a"
    cp ${libpoly}/lib/libpicpolyxx.a "$out/build/deps/lib/libpicpolyxx.a"

    test -f "$out/cmake/version-base.cmake"
    grep -F 'set(CVC5_LAST_RELEASE "${version}")' "$out/cmake/version-base.cmake"
    test -f "$out/include/cvc5/c/cvc5.h"
    test -d "$out/build/include"
    test -f "$out/build/src/libcvc5.a"
    test -d "$out/build/deps/lib"
    test -f "$out/build/deps/lib/libpicpoly.a"
    test -f "$out/build/deps/lib/libpicpolyxx.a"
  '';
in
{
  inherit
    dir
    package
    source
    version
    ;
  cvc5SysVersion = expectedCvc5SysVersion;
}
