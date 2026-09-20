{
  config,
  lib,
  pkgs,
  ...
}:

{
  # Select explicitly in CI; local entry retains the existing project profile.
  profiles.ci.module = {
    packages = with pkgs; [
      gh
      gnumake
      m4
      gmp
      mpfr
      nodejs
    ];

    env = lib.optionalAttrs pkgs.stdenv.isLinux {
      # Native executables and Python extensions must load the same libraries
      # used by the Nix compiler/PyO3 build, not the runner distribution's ABI.
      LD_LIBRARY_PATH = lib.makeLibraryPath [
        config.languages.python.package
        pkgs.stdenv.cc.cc.lib
        pkgs.openblas
        pkgs.gmp
        pkgs.mpfr
      ];
      LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
    };
  };
}
