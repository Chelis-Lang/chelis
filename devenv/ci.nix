{ ... }:

{
  # Select explicitly in CI; local entry retains the existing project profile.
  profiles.ci.module =
    {
      config,
      lib,
      pkgs,
      ...
    }:
    {
      # Cargo's OpenBLAS consumer uses LP64, not Nixpkgs' x86_64 ILP64 default.
      # Keep its headers and runtime library on the same integer ABI.
      overlays = [
        (_final: prev: {
          openblas = prev.openblas.override { blas64 = false; };
        })
      ];

      packages =
        with pkgs;
        [
          gh
          gnumake
          # CMake searches every PATH entry for gmake before considering make.
          (runCommand "gmake" { } ''
            mkdir -p "$out/bin"
            ln -s ${gnumake}/bin/make "$out/bin/gmake"
          '')
          m4
          gmp
          mpfr
          nodejs
        ]
        ++ lib.optionals pkgs.stdenv.isLinux [ pkgs.gfortran ];

      env = {
        # The wrapper's Bash %q response syntax cannot preserve arbitrary path bytes.
        NIX_CC_USE_RESPONSE_FILE = "0";
      }
      // lib.optionalAttrs pkgs.stdenv.isLinux {
        # Runtime image identities must remain content-derived under the Nix linker.
        NIX_SET_BUILD_ID = "1";
        NIX_BUILD_ID_STYLE = "sha1";
        # Vendored GMP/MPFR variadic formatting fails with Clang 21 here.
        # GNU C17 keeps GMP's configure probes compatible with GCC 15 and retains PIC.
        CC = "${pkgs.stdenv.cc}/bin/cc";
        CFLAGS = "-std=gnu17 -fPIC";
        FC = "${pkgs.gfortran}/bin/gfortran";
        # Native executables and Python extensions must load the same libraries
        # used by the Nix compiler/PyO3 build, not the runner distribution's ABI.
        LD_LIBRARY_PATH = lib.makeLibraryPath [
          config.languages.python.package
          pkgs.stdenv.cc.cc.lib
          pkgs.openblas
          pkgs.gmp
          pkgs.mpfr
          pkgs.zlib
        ];
        LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
      };
    };
}
