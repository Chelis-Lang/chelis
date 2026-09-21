{
  pkgs,
  lib,
  root,
}:
let
  cargoLock = lib.importTOML (root + "/Cargo.lock");
  cvc5Sys = lib.findFirst (package: package.name == "cvc5-sys") null cargoLock.package;
  cvc5SysVersion = "0.3.1";
  cvc5SysChecksum = "1d580e4bc6b26687ef4fcec1109c81e8dc265cefe9eb21575a126f34a986c100";
  version = "1.3.1";

  # cvc5-sys 0.3.1 invokes configure.sh --static --auto-download -DBUILD_GMP=1.
  # Its configure.sh defaults to Production WITH LibPoly, unlike a bare CMake
  # invocation. Resolve those native dependencies in Nix, never in Cargo.
  source = pkgs.fetchFromGitHub {
    owner = "cvc5";
    repo = "cvc5";
    rev = "ea1b484fa54bfe56c0f8b3ac90a6e3e2f46441e7"; # cvc5-1.3.1
    hash = "sha256-nxJjrpWZfYPuuKN4CWxOHEuou4r+MdK0AjdEPZHZbHI=";
  };
  gmp = pkgs.gmp.override { withStatic = true; };

  # This is the exact rel-2.1.3-elevate source selected by FindCaDiCaL.cmake,
  # built with the same stdenv as cvc5 and the Rust linker in the CI shell.
  cadical = (pkgs.cadical.override { version = "2.1.3"; }).overrideAttrs (_old: {
    src = pkgs.fetchFromGitHub {
      owner = "arminbiere";
      repo = "cadical";
      rev = "a384d221a920d473b770df6a7221f35fc5d99e90";
      hash = "sha256-2z65Fplm75p4nzTdjH9pKBRJC+sWZewIerTUT/nOdJM=";
    };
    configurePhase = ''
      runHook preConfigure
      ./configure --quiet --no-contrib -fPIC CXXFLAGS="-std=c++11"
      runHook postConfigure
    '';
  });

  poly = (pkgs.libpoly.override { inherit gmp; }).overrideAttrs (_old: {
    version = "0.2.0";
    src = pkgs.fetchFromGitHub {
      owner = "SRI-CSL";
      repo = "libpoly";
      rev = "d4c917c2566b6da5ef6300989ca5c0282b70b9b9"; # v0.2.0
      hash = "sha256-gE2O1YfiVab/aIqheoMP8GhE+N3yho7kb5EP56pzjW8=";
    };
    cmakeFlags = [
      "-DLIBPOLY_BUILD_PYTHON_API=OFF"
      "-DLIBPOLY_BUILD_STATIC=OFF"
      "-DLIBPOLY_BUILD_STATIC_PIC=ON"
      "-DBUILD_TESTING=OFF"
      "-DGMP_INCLUDE_DIR=${gmp.dev}/include"
      "-DGMP_LIBRARY=${gmp}/lib/libgmp.a"
    ];
    # Match the two PIC archives built by cvc5's FindPoly.cmake; do not build
    # unused shared libraries or a second, non-PIC copy of the same sources.
    buildFlags = [
      "static_pic_poly"
      "static_pic_polyxx"
    ];
    installPhase = ''
      runHook preInstall
      mkdir -p "$out/lib" "$out/include/poly/polyxx"
      cp src/libpicpoly.a src/libpicpolyxx.a "$out/lib/"
      cp ../include/*.h "$out/include/poly/"
      cp ../include/polyxx/*.h "$out/include/poly/polyxx/"
      runHook postInstall
    '';
  });

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

  package =
    (pkgs.cvc5.override {
      inherit gmp symfpu;
      cadical' = cadical;
      libpoly = poly;
    }).overrideAttrs
      (_old: {
        inherit version;
        src = source;
        # Bind every archive and header to the same dependency derivation. In
        # particular, do not discover a shared GMP or Nixpkgs' newer LibPoly.
        buildInputs = [
          cadical.dev
          gmp
          poly
          symfpu
        ];
        postPatch = ''
          substituteInPlace cmake/FindGMP.cmake \
            --replace-fail \
            'LINK_LIBRARIES ''${GMP_LIBRARIES} ''${GMPXX_LIBRARIES}' \
            'LINK_LIBRARIES ''${GMPXX_LIBRARIES} ''${GMP_LIBRARIES}'
        '';
        cmakeBuildType = "Production";
        cmakeFlags = [
          "-DBUILD_SHARED_LIBS=OFF"
          "-DENABLE_AUTO_DOWNLOAD=OFF"
          "-DBUILD_GMP=OFF"
          "-DENABLE_GPL=OFF"
          "-DUSE_CLN=OFF"
          "-DUSE_POLY=ON"
          "-DCaDiCaL_INCLUDE_DIR=${cadical.dev}/include"
          "-DCaDiCaL_LIBRARIES=${cadical.lib}/lib/libcadical.a"
          "-DGMP_INCLUDE_DIR=${gmp.dev}/include"
          "-DGMPXX_INCLUDE_DIR=${gmp.dev}/include"
          "-DGMP_LIBRARIES=${gmp}/lib/libgmp.a"
          "-DGMPXX_LIBRARIES=${gmp}/lib/libgmpxx.a"
          "-DPoly_INCLUDE_DIR=${poly}/include"
          "-DPoly_LIBRARIES=${poly}/lib/libpicpoly.a"
          "-DPolyXX_LIBRARIES=${poly}/lib/libpicpolyxx.a"
          "-DSymFPU_INCLUDE_DIR=${symfpu}/include"
          # This controls only cvc5's standalone executable, not its archives.
          # Cargo supplies the CI stdenv's C++ runtime and libc at final link.
          "-DSTATIC_BINARY=OFF"
        ];
        doCheck = false;
      });

  # cvc5-sys checks the release file and build/src/libcvc5.a before linking.
  # Preserve the parser feature as well as the solver and bindgen headers.
  # Symlinks retain the native closure for recursive signed Nix cache copies.
  dir = pkgs.runCommand "cvc5-${version}-ci-prebuilt-tree" { } ''
    mkdir -p "$out/cmake" "$out/build/src/parser" "$out/build/deps/lib"
    ln -s ${source}/cmake/version-base.cmake "$out/cmake/version-base.cmake"
    ln -s ${package}/include "$out/include"
    ln -s ${package}/include "$out/build/include"
    ln -s ${package}/lib/libcvc5.a "$out/build/src/libcvc5.a"
    ln -s ${package}/lib/libcvc5parser.a "$out/build/src/parser/libcvc5parser.a"
    ln -s ${cadical.lib}/lib/libcadical.a "$out/build/deps/lib/libcadical.a"
    ln -s ${poly}/lib/libpicpoly.a "$out/build/deps/lib/libpicpoly.a"
    ln -s ${poly}/lib/libpicpolyxx.a "$out/build/deps/lib/libpicpolyxx.a"
    ln -s ${gmp}/lib/libgmp.a "$out/build/deps/lib/libgmp.a"

    grep -Fx 'set(CVC5_LAST_RELEASE "${version}")' "$out/cmake/version-base.cmake"
    for header in c/cvc5.h c/cvc5_parser.h cvc5_export.h; do
      test -f "$out/include/cvc5/$header"
    done
    for archive in "$out/build/src/libcvc5.a" \
      "$out/build/src/parser/libcvc5parser.a" "$out"/build/deps/lib/*.a; do
      test -s "$archive"
    done
  '';
in
assert lib.assertMsg (
  cvc5Sys != null && cvc5Sys.version == cvc5SysVersion && cvc5Sys.checksum == cvc5SysChecksum
) "nix/ci-cvc5.nix must be updated with the cvc5-sys native build contract";
assert lib.assertMsg (
  (pkgs.stdenv.hostPlatform.isLinux || pkgs.stdenv.hostPlatform.isDarwin)
  && pkgs.stdenv.buildPlatform == pkgs.stdenv.hostPlatform
) "The CI cvc5-sys provider supports native Linux and Darwin builds only";
{
  inherit
    dir
    package
    source
    version
    cvc5SysVersion
    ;
}
