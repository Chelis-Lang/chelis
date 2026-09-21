{ lib, pkgs, ... }:

let
  # openblas-src 0.10.16's `system` + `static` features probe
  # `pkg-config --static openblas` and return without downloading/building.
  # OpenBLAS's CMake-generated .pc omits the archive's runtime dependencies,
  # so provide complete CI metadata without changing upstream source code.
  pkgConfig = pkgs.writeTextDir "lib/pkgconfig/openblas.pc" ''
    libdir=${lib.getLib pkgs.openblas}/lib
    includedir=${lib.getDev pkgs.openblas}/include

    Name: OpenBLAS
    Description: CI LP64 OpenBLAS with GNU Fortran and OpenMP runtimes
    Version: ${pkgs.openblas.version}
    Libs: -L''${libdir} -lopenblas
    Libs.private: -L${pkgs.gfortran.cc.lib}/lib -L${pkgs.stdenv.cc.cc.lib}/lib -lgfortran -lgomp -lm -lpthread
    Cflags: -I''${includedir}
  '';
in
{
  config = {
    assertions = lib.optionals pkgs.stdenv.isLinux [
      {
        assertion =
          pkgs.stdenv.cc.isGNU
          && !pkgs.stdenv.hostPlatform.isMusl
          && pkgs.stdenv.buildPlatform.config == pkgs.stdenv.hostPlatform.config;
        message = "The CI OpenBLAS static provider requires native GNU/Linux with the pinned GCC/gfortran ABI.";
      }
    ];

    overlays = [
      (
        _final: prev:
        lib.optionalAttrs prev.stdenv.isLinux {
          # Keep the existing pinned provider and all its source patches. LP64
          # means 32-bit BLAS/LAPACK integers, even on 64-bit CI runners.
          openblas = prev.openblas.override {
            blas64 = false;
            enableStatic = true;
            enableShared = true;
          };
        }
      )
    ];

    packages = lib.optionals pkgs.stdenv.isLinux [
      pkgs.openblas
      pkgs.pkg-config
      pkgConfig
    ];

    # Prepend after package setup hooks, preserving every other .pc directory.
    # The archive lives beside the shared OpenBLAS library: pkg-config's Rust
    # consumer finds libopenblas.a and emits rustc-link-lib=static=openblas.
    # GCC's lib outputs contain shared runtimes (archives remain in its out
    # output), so Fortran/OpenMP stay dynamically linked via the same Nix ABI.
    enterShell = lib.mkAfter (
      lib.optionalString pkgs.stdenv.isLinux ''
        export PKG_CONFIG_PATH="${pkgConfig}/lib/pkgconfig''${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
      ''
    );
  };
}
