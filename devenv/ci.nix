{ ... }:

{
  # Select explicitly in CI; local entry retains the existing project profile.
  #
  # `ci-hosted` realizes only packages that the public binary caches serve.
  # GitHub-hosted runners cannot reach the private Nix cache, so nothing that
  # Nix would build from source belongs here: Cargo compiles the numerics from
  # their vendored sources, as it does on main, and the SMT lane links the
  # prebuilt cvc5 asset. `ci` layers the Nix-built numerics and the static
  # OpenBLAS provider on top for the self-hosted runner, which substitutes
  # them from the private cache.
  profiles.ci-hosted.module =
    {
      config,
      lib,
      pkgs,
      ...
    }:
    let
      numerics = import ../nix/ci-arb.nix {
        inherit pkgs lib;
        root = ../.;
      };
    in
    {
      # Clang's setup hook overwrites CC while mkShell collects packages.
      # Reassert the CI C provider after those hooks, before commands run.
      enterShell = lib.mkAfter ''
        export CC=${lib.escapeShellArg config.env.CC}
      '';

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
        ++ lib.optionals pkgs.stdenv.isLinux [
          pkgs.gfortran
          # LP64 like the runner distribution's libopenblas; nixpkgs' default
          # OpenBLAS is ILP64 on x86_64. Under the `ci` overlay this resolves
          # to the same static provider derivation.
          pkgs.openblasCompat
        ];

      env = {
        # The schema-transition oracle runs in the default development profile,
        # not ordinary CI. Do not realize its second Kache build for every job.
        KACHE_SCHEMA_27_WRAPPER = lib.mkForce "";
        # Shared compiler-cache entries require non-incremental compilation.
        CARGO_INCREMENTAL = "0";
        # The wrapper's Bash %q response syntax cannot preserve arbitrary path bytes.
        NIX_CC_USE_RESPONSE_FILE = "0";
        # The vendored GMP/MPFR configure checks fail under the pinned Clang C
        # provider, whether Cargo or the native cache producer builds them.
        CC = numerics.compiler;
        CFLAGS = numerics.cflags;
      }
      // lib.optionalAttrs pkgs.stdenv.isLinux {
        # Runtime image identities must remain content-derived under the Nix linker.
        NIX_SET_BUILD_ID = "1";
        NIX_BUILD_ID_STYLE = "sha1";
        FC = "${pkgs.gfortran}/bin/gfortran";
        # Native executables and Python extensions must load the same libraries
        # used by the Nix compiler/PyO3 build, not the runner distribution's ABI.
        LD_LIBRARY_PATH = lib.makeLibraryPath [
          config.languages.python.package
          pkgs.stdenv.cc.cc.lib
          pkgs.gfortran.cc.lib
          pkgs.openblasCompat
          pkgs.gmp
          pkgs.mpfr
          pkgs.zlib
        ];
        LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
      };
    };

  profiles.ci = {
    extends = [ "ci-hosted" ];
    module =
      { lib, pkgs, ... }:
      let
        numerics = import ../nix/ci-arb.nix {
          inherit pkgs lib;
          root = ../.;
        };
      in
      {
        imports = [ ../nix/ci-openblas.nix ];

        packages = [ numerics.cache ];

        outputs.native-numerics = numerics.cache;

        env = {
          GMP_MPFR_SYS_CACHE = "${numerics.cache}/gmp";
          FLINT_SYS_CACHE = "${numerics.cache}/flint";
          ARB_SYS_CACHE = "${numerics.cache}/arb";
        };
      };
  };

  # Only the self-hosted SMT lane realizes the solver closure. Every other CI
  # worker keeps the smaller base profile and its own existing feature selection;
  # the hosted SMT lane links the prebuilt cvc5 release asset instead.
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
