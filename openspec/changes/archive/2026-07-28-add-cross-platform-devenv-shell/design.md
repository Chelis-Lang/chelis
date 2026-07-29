## Context

The `main` branch has no Devenv files. PR #894 introduces the files beside unrelated runtime and ABI changes.

Many Rust tests invoke `gcc` and `g++` by exact name. The Nix Darwin shell provides the compiler as `cc` and `c++`.

Host aliases vary by macOS installation and are not Devenv inputs. Some tests skip when `gcc` is absent, while other tests fail.

The Nix C wrapper passes linker paths to `-fsyntax-only` commands. Clang reports these unused arguments, and one test uses `-Werror`.

## Goals / Non-Goals

**Goals:**

- Add one reproducible development shell for Linux and macOS.
- Make the existing compiler command names available on macOS.
- Run the C compile-and-run tests on macOS without changes to those tests.
- Preserve the project Python 3.11 environment.
- Keep the Devenv change independent from PR #894 runtime work.

**Non-Goals:**

- Do not change compiler discovery in Rust tests.
- Do not replace GCC with clang on Linux.
- Do not add OpenMP support to the Nixpkgs clang path.
- Do not change CI, the release toolchain, runtime behavior, or the C ABI.
- Do not require Devenv for contributors who use the documented manual setup.
- Do not include the `rust-toolchain.toml` or nextest-profile changes from PR #894.

## Decisions

### D1: Add a new shell instead of changing every C test

The change adds `devenv.nix`, `devenv.yaml`, and `devenv.lock`. The lock file pins the resolved Nix inputs.

The Devenv module input uses release `v2.2` at commit `ffce215a42d09c6375c3d60dd9c4110438fc4d87`. A static test rejects URL and lock drift.

The alternative changes each test to discover a compiler or read `CC` and `CXX`. That work affects a large test surface.

This proposal keeps the test contract unchanged. The development shell provides the command names that the tests already require.

### D2: Add macOS compiler shims only inside Devenv

On macOS, `pkgs.writeShellScriptBin` creates `gcc` and `g++`. Each shim invokes the corresponding compiler from `pkgs.stdenv.cc`.

The C shim invokes `cc`. The C++ shim invokes `c++`. The current lock selects Nixpkgs clang and a Nixpkgs Apple SDK.

Each shim adds `-Wno-unused-command-line-argument` before the caller arguments. This flag suppresses unused wrapper paths during `-fsyntax-only` checks.

A negative smoke check must prove that other compiler warnings still fail under `-Werror`.

A symlink cannot add the required flag. Host Apple clang is not a pinned Devenv input and bypasses the Nix compiler wrapper.

Homebrew GCC conflicts with the platform SDK and adds an unnecessary dependency.

### D3: Keep Linux on native GCC

Linux receives GCC, OpenBLAS, and Valgrind from Nixpkgs. Linux does not receive the macOS shims.

This choice preserves the compiler and BLAS combination that Linux CI uses. It also keeps a real GCC lane for generated C.

macOS uses Accelerate from the operating system, so the shell does not add OpenBLAS there.

### D4: Use the existing project pins

The Rust environment reads `rust-toolchain.toml`. The Python environment uses Python 3.11 and uv.

The common package list contains these tools:

- `cargo-llvm-cov`
- `cargo-nextest`
- `cmake`
- `git`
- `pkg-config`

The shell creates `.venv` only when `.venv/bin/python` does not exist. It does not replace an existing project environment.

### D5: Separate smoke checks from the acceptance oracle

`devenv test` checks tool availability, Python 3.11, and C and C++ compilation under `-Werror`.

The negative compiler check uses a deliberate source warning. The compile must fail, which proves that the shim does not hide code warnings.

The authoritative acceptance oracle is:

```sh
devenv shell -- cargo nextest run -p chelis-backend-c
```

This command runs the crate tests inside the shell. It includes the tests that invoke `gcc` and `g++`.

### D6: Keep generated Devenv state out of Git

The repository tracks the three input files. `.gitignore` excludes `.devenv/` and `.devenv.flake.nix`.

The repository lint policy excludes both generated paths under §12.2. The policy still checks the tracked Devenv inputs.

The generated files remain local machine state. A contributor can remove them without loss of a reproducible input.

## Risks / Trade-offs

**[The macOS command is named `gcc` but runs Nixpkgs clang]** → The documentation states this fact. Linux and CI retain real GCC coverage.

**[The warning flag hides a real diagnostic]** → The flag names one wrapper diagnostic. A negative smoke check proves that other warnings fail.

**[The rolling input changes after an update]** → The lock file pins the current revision. A separate change must update that lock.

**[The shell differs from manual contributor setups]** → Devenv remains optional. The manual setup stays documented and supported.

**[The full backend test takes longer than a smoke check]** → `devenv test` stays small. The crate suite remains the acceptance oracle.

## Migration Plan

1. Add failing shell checks before the macOS shims.
2. Copy only the Devenv inputs and ignore rules from PR #894.
3. Add the macOS shims and the platform package lists.
4. Add the contributor documentation.
5. Run the shell checks on macOS and Linux.
6. Run the authoritative acceptance oracle on macOS.

To remove this change, delete the Devenv files and their ignore rules. Then remove the Devenv documentation.

## Open Questions

None. The extraction scope and the platform behavior are explicit.
