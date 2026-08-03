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
