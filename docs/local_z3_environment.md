# Local Z3 setup

The optional `z3` feature lets `chelis-prove` use Z3 as an SMT solver. It links
to a prebuilt Z3 library. The default build does not compile Z3 from source, so
CMake is not required for this setup.

## Run Z3 tests

Use `scripts/z3_test.py` to find the installed library and set Cargo's linker
and runtime search paths:

```sh
scripts/z3_test.py
scripts/z3_test.py -p chelis-prove --features "smt z3" --test cross_engine_oracle
```

The helper also works inside a Devenv shell as `chelis-z3-test`. To print the
environment settings without running Cargo:

```sh
scripts/z3_test.py --print-env
```

## Library names and search paths

The link library uses a different filename on each supported platform:

| Platform | Link library | Runtime search path |
|---|---|---|
| Linux | `libz3.so` | `LD_LIBRARY_PATH` |
| macOS | `libz3.dylib` | `DYLD_LIBRARY_PATH` |
| Windows | `libz3.lib` or `liblibz3.dll.a` | `PATH` |

The library directory must contain the unversioned link-library filename.
Installing only a versioned Linux file such as `libz3.so.4` does not provide
the link target needed by Cargo.

The helper searches the Python `z3-solver` package, `pkg-config`, configured
linker and runtime paths, and common platform library locations. On macOS, it
checks the Homebrew Z3 prefix on both Apple Silicon and Intel Macs. If Z3 is
installed elsewhere, set
`Z3_LIBRARY_PATH_OVERRIDE` to the directory containing the link library. The
helper adds that directory to the platform's runtime search path while
preserving existing entries visible to its Python process.

On macOS, invoke the helper through the active Python interpreter when an
existing `DYLD_LIBRARY_PATH` must be retained:

```sh
.venv/bin/python scripts/z3_test.py -p chelis-prove --features "smt z3" --test cross_engine_oracle
```

The script's `/usr/bin/env` launcher does not pass `DYLD_LIBRARY_PATH` through
to Python on protected macOS installations. Direct Python invocation makes
the existing entries available to the helper.

Linux package managers usually provide the link library through their Z3
development package. On macOS, Homebrew installs both the command-line solver
and the link library with `brew install z3`. On Windows, place the import
library and its matching DLL in the configured library directory, or make the
DLL directory available through `PATH`.

## CI and default builds

Linux CI installs the Z3 development library on the system search paths, so
the helper can use it without an override. The `z3` feature is optional:
default builds and builds using only the `smt` feature do not link to Z3.
