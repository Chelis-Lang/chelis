{
  config,
  lib,
  pkgs,
  ...
}:
let
  # Final nixos-20.09 revision: glibc 2.31-74 and its native GCC 9 toolchain.
  # The target ABI and bindgen's in-process library come from this input.
  # Cargo, Kache, Python and CMake remain current, pinned host executables.
  legacySource = builtins.fetchTarball {
    url = "https://codeload.github.com/NixOS/nixpkgs/tar.gz/1c1f5649bb9c1b0d98637c8c365228f57126f361";
    sha256 = "sha256-tAMJnUwfaDEB2aa31jGcu7R7bzGELM9noc91L2PbVjg=";
  };
  legacy = import legacySource {
    system = "x86_64-linux";
    config = { };
    overlays = [ ];
  };
  libc = legacy.glibc;
  cc = legacy.stdenv.cc;
  cxxRuntime = cc.cc.lib;
  libclang = legacy.llvmPackages_11.libclang;
  clangResource = "${libclang.out}/lib/clang/${libclang.version}";
  loader = "${libc}/lib/ld-linux-x86-64.so.2";
  # Use today's derivation/setup-hook API with the *entire* old compiler,
  # binutils, headers, libc and C++ runtime, not merely an old final linker.
  targetStdenv = pkgs.overrideCC pkgs.stdenv cc;
  cvc5 = import ./ci-cvc5.nix {
    inherit pkgs lib;
    root = ../.;
    stdenv = targetStdenv;
  };

  # mkShell also contains modern host tools. Their setup hooks must not add
  # modern include/library paths to cc-rs, build scripts or Cargo's linker.
  # Clean only the compiler subprocess: never export the old libc globally.
  compilerScript =
    program:
    pkgs.writeShellScript "glibc231-${program}" ''
      set -eu
      for variable in "''${!NIX_@}"; do
        case "$variable" in
          NIX_CFLAGS_*|NIX_LDFLAGS*|NIX_CXXSTDLIB_*|NIX_CC_WRAPPER_FLAGS_SET*|NIX_BINTOOLS_WRAPPER_FLAGS_SET*)
            unset "$variable"
            ;;
        esac
      done
      unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH
      unset LIBRARY_PATH COMPILER_PATH GCC_EXEC_PREFIX
      unset LD_LIBRARY_PATH LD_PRELOAD LD_AUDIT
      export NIX_CC_USE_RESPONSE_FILE=0
      export NIX_LDFLAGS="-rpath ${cxxRuntime}/lib"
      exec ${cc}/bin/${program} "$@"
    '';
  compiler = pkgs.runCommand "chelis-glibc231-compiler" { } ''
    mkdir -p "$out/bin"
    ln -s ${compilerScript "cc"} "$out/bin/cc"
    ln -s ${compilerScript "c++"} "$out/bin/c++"
    ln -s cc "$out/bin/gcc"
    ln -s c++ "$out/bin/g++"
  '';
  libraryPath = lib.makeLibraryPath [
    libc
    cxxRuntime
  ];
  runtimeClosure = pkgs.closureInfo {
    rootPaths = [
      libc
      cxxRuntime
    ];
  };

  # This is a runtime proof, not just a symbol-version scan on modern glibc.
  # Accept only native dynamic ELFs linked to this exact interpreter. Resolve
  # their libraries with the old loader first, and reject every provider outside
  # its immutable runtime closure; an RPATH may not smuggle in a modern libc.
  runner = pkgs.writeScriptBin "chelis-glibc231-run" ''
    #!${pkgs.python311}/bin/python3
    import os
    import pathlib
    import re
    import stat
    import struct
    import subprocess
    import sys

    LOADER = ${builtins.toJSON loader}
    LIBRARIES = ${builtins.toJSON libraryPath}
    STORE = pathlib.Path(${builtins.toJSON builtins.storeDir})
    LIBC = pathlib.Path(${builtins.toJSON "${libc}/lib/libc.so.6"}).resolve()
    ALLOWED = frozenset(pathlib.Path(${builtins.toJSON "${runtimeClosure}/store-paths"}).read_text().splitlines())

    def reject(message):
        raise SystemExit("glibc231 runner: " + message)

    def main():
        if len(sys.argv) < 2:
            reject("usage: chelis-glibc231-run ELF [ARG ...]")
        binary = pathlib.Path(sys.argv[1]).resolve(strict=True)
        mode = binary.stat().st_mode
        if not stat.S_ISREG(mode) or not os.access(binary, os.X_OK) or mode & (stat.S_ISUID | stat.S_ISGID):
            reject("expected a non-setid executable regular file")
        with binary.open("rb") as stream:
            header = stream.read(64)
            if len(header) != 64 or header[:7] != b"\x7fELF\x02\x01\x01":
                reject("expected a little-endian ELF64 executable")
            kind, machine = struct.unpack_from("<HH", header, 16)
            if kind not in (2, 3) or machine != 62:
                reject("expected a native x86_64 executable")
            phoff = struct.unpack_from("<Q", header, 32)[0]
            phsize, phnum = struct.unpack_from("<HH", header, 54)
            if phsize != 56 or phnum == 0 or phnum == 65535:
                reject("unsupported ELF program headers")
            interpreters = []
            for index in range(phnum):
                stream.seek(phoff + index * phsize)
                ph = stream.read(phsize)
                if len(ph) != phsize:
                    reject("truncated ELF program headers")
                if struct.unpack_from("<I", ph)[0] == 3:
                    offset = struct.unpack_from("<Q", ph, 8)[0]
                    size = struct.unpack_from("<Q", ph, 32)[0]
                    if not 1 < size <= 4096:
                        reject("invalid ELF interpreter")
                    stream.seek(offset)
                    interpreters.append(stream.read(size))
            if interpreters != [os.fsencode(LOADER) + b"\0"]:
                reject("ELF interpreter is not the pinned glibc 2.31 loader")
        environment = {key: value for key, value in os.environ.items() if not key.startswith("LD_")}
        environment["LC_ALL"] = "C"
        environment["LD_LIBRARY_PATH"] = LIBRARIES
        command = [LOADER, "--inhibit-cache", "--library-path", LIBRARIES]
        resolved = subprocess.run(command + ["--list", str(binary)], env=environment, text=True, capture_output=True)
        if resolved.returncode:
            reject("pinned loader rejected ELF dependencies: " + resolved.stderr.strip())
        saw_libc = False
        for line in resolved.stdout.splitlines():
            line = line.strip()
            if line.startswith("linux-vdso.so.1 ("):
                continue
            match = re.fullmatch(r"(?:\S+ => )?(/\S+) \(0x[0-9a-fA-F]+\)", line)
            if match is None:
                reject("unrecognized loader dependency: " + line)
            dependency = pathlib.Path(match.group(1)).resolve(strict=True)
            root = dependency
            while root.parent != STORE and root != root.parent:
                root = root.parent
            if str(root) not in ALLOWED:
                reject("dependency outside the pinned runtime closure: " + str(dependency))
            saw_libc |= dependency == LIBC
        if not saw_libc:
            reject("ELF did not resolve the pinned glibc 2.31 libc")
        # Enter through the checked PT_INTERP so /proc/self/exe remains chelis.
        # Solver containment respawns current_exe; execing ld.so would break it.
        os.execve(str(binary), [str(binary), *sys.argv[2:]], environment)

    try:
        main()
    except (OSError, ValueError, struct.error) as error:
        reject(str(error))
  '';
  # Keep the loader, runtime libraries, compiler and CVC5 archives reachable
  # through packages, not only outputs: CI publishes the profile's closure.
  runtime =
    assert lib.assertMsg (
      pkgs.stdenv.hostPlatform.system == "x86_64-linux"
      && pkgs.stdenv.buildPlatform == pkgs.stdenv.hostPlatform
    ) "ci-glibc231 requires a native x86_64-linux Nix builder";
    assert lib.assertMsg (libc.version == "2.31") "ci-glibc231 requires the pinned glibc 2.31 provider";
    pkgs.runCommand "chelis-glibc231-runtime" { } ''
      mkdir -p "$out/bin" "$out/share/chelis-glibc231"
      ln -s ${runner}/bin/chelis-glibc231-run "$out/bin/chelis-glibc231-run"
      ln -s ${libc.bin}/bin/ldd "$out/bin/chelis-glibc231-ldd"
      ln -s ${libc} "$out/share/chelis-glibc231/libc"
      ln -s ${cxxRuntime} "$out/share/chelis-glibc231/cxx-runtime"
      ln -s ${compiler} "$out/share/chelis-glibc231/compiler"
      ln -s ${cvc5.dir} "$out/share/chelis-glibc231/cvc5"
      ln -s ${libclang} "$out/share/chelis-glibc231/libclang"
      ln -s ${clangResource} "$out/share/chelis-glibc231/clang-resource"
      ln -s ${runtimeClosure}/store-paths "$out/share/chelis-glibc231/runtime-store-paths"
    '';
  emptyPkgConfig = pkgs.runCommand "chelis-glibc231-pkg-config" { } ''
    mkdir -p "$out/lib/pkgconfig"
  '';
in
{
  packages = [
    runtime
    pkgs.gh
    pkgs.gnumake
    pkgs.m4
    pkgs.nodejs
    pkgs.binutils
    (pkgs.runCommand "glibc231-gmake" { } ''
      mkdir -p "$out/bin"
      ln -s ${pkgs.gnumake}/bin/make "$out/bin/gmake"
    '')
  ];
  outputs.glibc231 = runtime;
  outputs.cvc5 = cvc5.dir;

  # The CLI's default+smt graph has no PyO3, Python, Arb/FLINT or OpenBLAS
  # link dependency. Keep the repository-owned Python 3.11 venv/PYO3_PYTHON
  # from toolchains.nix for host tooling; do not claim that this CLI-only
  # profile makes modern libpython or arbitrary workspace features compatible.
  env = {
    CHELIS_GLIBC231_LDD = "${runtime}/bin/chelis-glibc231-ldd";
    CHELIS_GLIBC231_RUNNER = "${runtime}/bin/chelis-glibc231-run";
    KACHE_SCHEMA_27_WRAPPER = lib.mkForce "";
    CARGO_INCREMENTAL = "0";
    CARGO_TARGET_DIR = lib.mkForce "${config.devenv.root}/target/ci-glibc231";
    CC = lib.mkForce "${compiler}/bin/cc";
    CXX = lib.mkForce "${compiler}/bin/c++";
    CC_x86_64_unknown_linux_gnu = "${compiler}/bin/cc";
    CXX_x86_64_unknown_linux_gnu = "${compiler}/bin/c++";
    AR = "${cc.bintools.bintools}/bin/ar";
    CFLAGS = lib.mkForce "-fPIC";
    CXXFLAGS = lib.mkForce "-fPIC -Wno-deprecated-literal-operator";
    # The linker store path participates in rustc argv and therefore Kache's
    # key even with ignore_env=true. Native Cargo OUT_DIRs are isolated too.
    # Rust's default bundled LLD would bypass the pinned old binutils wrapper.
    RUSTFLAGS = lib.mkForce "-C linker=${compiler}/bin/cc -C linker-features=-lld";
    CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER = "${compiler}/bin/cc";
    CVC5_DIR = "${cvc5.dir}";
    # Native Cargo also links build scripts with this ABI. Bindgen dlopens
    # libclang inside that process, so a modern-glibc libclang cannot work.
    LIBCLANG_PATH = lib.mkForce "${libclang}/lib";
    CLANG_PATH = "${libclang.out}/bin/clang";
    BINDGEN_EXTRA_CLANG_ARGS = "-resource-dir ${clangResource} -isystem ${libc.dev}/include";
    PKG_CONFIG_PATH = lib.mkForce "";
    PKG_CONFIG_LIBDIR = "${emptyPkgConfig}/lib/pkgconfig";
    LD_LIBRARY_PATH = lib.mkForce "";
    NIX_CC_USE_RESPONSE_FILE = "0";
    NIX_SET_BUILD_ID = "1";
    NIX_BUILD_ID_STYLE = "sha1";
  };
  enterShell = lib.mkAfter ''
    export CC=${lib.escapeShellArg config.env.CC}
    export CXX=${lib.escapeShellArg config.env.CXX}
    export AR=${lib.escapeShellArg config.env.AR}
    export LIBCLANG_PATH=${lib.escapeShellArg config.env.LIBCLANG_PATH}
    export PATH=${compiler}/bin:$PATH
    unset LD_LIBRARY_PATH LD_PRELOAD LD_AUDIT
  '';
}
