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
      imports = [ ../nix/ci-openblas.nix ];

      # Clang's setup hook overwrites CC while mkShell collects packages.
      # Reassert the CI C provider after those hooks, before commands run.
      enterShell = lib.mkAfter (
        lib.optionalString pkgs.stdenv.isLinux ''
          export CC=${lib.escapeShellArg config.env.CC}
        ''
      );

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
        # The schema-transition oracle runs in the default development profile,
        # not ordinary CI. Do not realize its second Kache build for every job.
        KACHE_SCHEMA_27_WRAPPER = lib.mkForce "";
        # Shared compiler-cache entries require non-incremental compilation.
        CARGO_INCREMENTAL = "0";
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
          pkgs.gfortran.cc.lib
          pkgs.openblas
          pkgs.gmp
          pkgs.mpfr
          pkgs.zlib
        ];
        LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
      };
    };

  # Only the SMT lane realizes the solver closure. Every other CI worker keeps
  # the smaller base profile and its own existing feature selection.
  profiles.ci-smt = {
    extends = [ "ci" ];
    module =
      { pkgs, lib, ... }:
      let
        cvc5 = import ../nix/ci-cvc5.nix {
          inherit pkgs lib;
          root = ../.;
        };
      in
      {
        env.CVC5_DIR = "${cvc5.dir}";
        outputs.cvc5 = cvc5.dir;
      };
  };
}
