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
  package = pkgs.cvc5.overrideAttrs (_old: {
    inherit version;
    src = source;
    postPatch = ''
      substituteInPlace cmake/FindGMP.cmake \
        --replace-fail \
        'LINK_LIBRARIES ''${GMP_LIBRARIES} ''${GMPXX_LIBRARIES}' \
        'LINK_LIBRARIES ''${GMPXX_LIBRARIES} ''${GMP_LIBRARIES}'
    '';
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
      "-DUSE_POLY=OFF"
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

    test -f "$out/cmake/version-base.cmake"
    grep -F 'set(CVC5_LAST_RELEASE "${version}")' "$out/cmake/version-base.cmake"
    test -f "$out/include/cvc5/c/cvc5.h"
    test -d "$out/build/include"
    test -f "$out/build/src/libcvc5.a"
    test -d "$out/build/deps/lib"
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
