{
  supportedSystems = [
    "x86_64-linux"
    "aarch64-darwin"
  ];

  packageNames = [
    "chelis"
    "chelis-runtime"
    "chelisup"
    "default"
  ];

  appNames = [
    "chelis"
    "chelisup"
    "default"
  ];

  publicRuntimeHeaders = [
    "chelis_runtime.h"
    "chelis_runtime_dtype.h"
    "chelis_blas.h"
    "chelis_simd.h"
    "chelis_math.h"
  ];

  packageShapes = {
    chelis = {
      required = [
        "bin/chelis"
        "lib/libchelis_runtime.a"
        "include/chelis_runtime.h"
        "include/chelis_runtime_dtype.h"
        "include/chelis_blas.h"
        "include/chelis_simd.h"
        "include/chelis_math.h"
      ];
      forbidden = [ "bin/chelisup" ];
      allowedProductExecutables = [ "bin/chelis" ];
      inventory = [
        "bin"
        "include"
        "lib"
        "bin/chelis"
        "lib/libchelis_runtime.a"
        "include/chelis_runtime.h"
        "include/chelis_runtime_dtype.h"
        "include/chelis_blas.h"
        "include/chelis_simd.h"
        "include/chelis_math.h"
      ];
    };

    chelis-runtime = {
      required = [
        "lib/libchelis_runtime.a"
        "include/chelis_runtime.h"
        "include/chelis_runtime_dtype.h"
        "include/chelis_blas.h"
        "include/chelis_simd.h"
        "include/chelis_math.h"
      ];
      forbidden = [
        "bin/chelis"
        "bin/chelisup"
      ];
      allowedProductExecutables = [ ];
      inventory = [
        "include"
        "lib"
        "lib/libchelis_runtime.a"
        "include/chelis_runtime.h"
        "include/chelis_runtime_dtype.h"
        "include/chelis_blas.h"
        "include/chelis_simd.h"
        "include/chelis_math.h"
      ];
    };

    chelisup = {
      required = [
        "bin/chelisup"
        "libexec/chelisup"
      ];
      forbidden = [ "bin/chelis" ];
      allowedProductExecutables = [
        "bin/chelisup"
        "libexec/chelisup"
      ];
      inventory = [
        "bin"
        "libexec"
        "bin/chelisup"
        "libexec/chelisup"
      ];
    };
  };

  # Recorded glibc floor of the portable Linux release binary: the maximum
  # GLIBC_* symbol version in its dynamic symbol table. The release
  # derivation fails on drift in either direction, so a toolchain bump that
  # moves the floor lands as a reviewed diff beside the lock change.
  linuxReleaseGlibcFloor = "2.39";

  # glibc version of the off-Nix consumption environment that runs the
  # shipped Linux tarball in the release workflow. The recorded floor must
  # not exceed this value, or our own consumption lane cannot run the
  # artifact.
  linuxReleaseConsumptionGlibc = "2.39";

  compilerBehaviorChecks = [
    "version"
    "help"
    "release-fixture"
    "smt"
  ];

  runtimeConsumers = {
    x86_64-linux = "OpenBLAS";
    aarch64-darwin = "Accelerate";
  };
}
