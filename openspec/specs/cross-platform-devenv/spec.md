# Cross-Platform Devenv Specification

## Purpose

Define the reproducible contributor shell, platform compiler commands, Python environment, smoke checks, and contributor documentation for Linux and macOS.

## Requirements

### Requirement: The repository provides reproducible Devenv inputs
The repository MUST track `devenv.nix`, `devenv.yaml`, and `devenv.lock`. The lock file MUST pin all resolved input revisions.

`devenv.yaml` MUST pin the Devenv module input to release `v2.2`. `devenv.lock` MUST resolve that input to commit `ffce215a42d09c6375c3d60dd9c4110438fc4d87`.

The repository MUST ignore `.devenv/` and `.devenv.flake.nix`. These paths contain generated local state, not reproducible inputs.

The repository lint policy MUST exclude both paths from the editable lint corpus. The policy MUST keep the three tracked inputs in the corpus.

#### Scenario: The repository checks the Devenv release pin
- **WHEN** the configured URL or locked revision differs from Devenv `v2.2`
- **THEN** the static Devenv version contract test fails

#### Scenario: A clean checkout evaluates the shell
- **WHEN** a contributor runs `devenv test` from a clean checkout
- **THEN** Devenv resolves the tracked inputs without an uncommitted input file

#### Scenario: Generated state stays untracked
- **WHEN** Devenv creates `.devenv/` or `.devenv.flake.nix`
- **THEN** Git ignores each generated path

#### Scenario: Generated state stays outside repository lint
- **WHEN** a contributor runs `chelis lint --check .` after Devenv creates local state
- **THEN** the lint excludes both generated paths and still checks the tracked Devenv inputs

### Requirement: The shell provides platform-correct compiler commands
On macOS, the shell MUST provide `gcc` and `g++` commands that invoke the Nixpkgs Darwin compiler wrappers from `pkgs.stdenv.cc`.

The shell MUST NOT use host compiler aliases as the shim implementation. These aliases are not pinned by the Devenv inputs.

The macOS shims MUST suppress only the unused wrapper-argument warning. Other warnings MUST remain errors when a caller passes `-Werror`.

On Linux, the shell MUST provide GCC from Nixpkgs. Linux MUST NOT use the macOS compiler shims.

#### Scenario: macOS compiles C through the expected command
- **WHEN** a macOS contributor compiles valid C with `gcc -Werror -fsyntax-only`
- **THEN** the pinned Nixpkgs C compiler wrapper succeeds

#### Scenario: macOS compiles C++ through the expected command
- **WHEN** a macOS contributor compiles valid C++17 with `g++ -std=c++17 -Werror -c`
- **THEN** the pinned Nixpkgs C++ compiler wrapper succeeds

#### Scenario: Host aliases do not satisfy the shell contract
- **WHEN** the macOS shims are absent and host `gcc` or `g++` aliases exist
- **THEN** `devenv test` rejects the host aliases

#### Scenario: Code warnings remain errors
- **WHEN** a macOS contributor compiles source with a deliberate warning and passes `-Werror`
- **THEN** the compiler command fails despite the wrapper-warning suppression

#### Scenario: Linux uses GCC
- **WHEN** a Linux contributor runs `gcc --version` inside the shell
- **THEN** the command reports the GCC package from Nixpkgs

### Requirement: The shell provides the Chelis development tools
The common shell MUST provide Rust from `rust-toolchain.toml`, Python 3.11, uv, `cargo-nextest`, `cargo-llvm-cov`, CMake, Git, and pkg-config.

The Linux shell MUST also provide GCC, OpenBLAS, and Valgrind. The macOS shell MUST use the system Accelerate framework instead of OpenBLAS.

#### Scenario: Common tools are available
- **WHEN** a contributor enters the shell on Linux or macOS
- **THEN** each common tool responds to its version command

#### Scenario: Linux tools are available
- **WHEN** a contributor enters the shell on Linux
- **THEN** GCC, OpenBLAS, and Valgrind are available from the Nix environment

#### Scenario: macOS omits the Linux BLAS package
- **WHEN** a contributor evaluates the package list on macOS
- **THEN** the shell does not add OpenBLAS or Valgrind

### Requirement: The shell preserves the project Python environment
If `.venv/bin/python` does not exist, the shell MUST create `.venv` with the configured Python 3.11 interpreter.

If `.venv/bin/python` exists, the shell MUST preserve the existing environment. The shell MUST NOT use the system Python for the project environment.

#### Scenario: The shell creates a missing environment
- **WHEN** a contributor enters the shell without `.venv/bin/python`
- **THEN** uv creates `.venv` with Python 3.11

#### Scenario: The shell preserves an existing environment
- **WHEN** a contributor enters the shell with an executable `.venv/bin/python`
- **THEN** the shell does not recreate or replace `.venv`

### Requirement: The shell checks its compiler contract
`devenv test` MUST check the required tool versions and the Python interpreter version.

It MUST compile valid C and C++ translation units under `-Werror`. It MUST also prove that a deliberate code warning still fails.

The smoke checks MUST run as these independent Devenv tasks before `devenv:enterTest`:

- `chelis:toolchain-test`
- `chelis:python-test`
- `chelis:c-compiler-test`
- `chelis:cpp-compiler-test`

The smoke-check graph MUST NOT start a Devenv service or long-running process.

#### Scenario: The shell smoke check passes
- **WHEN** all required tools and compiler commands satisfy the contract
- **THEN** `devenv test` runs all four named tasks
- **AND** each task reports success
- **AND** `devenv test` exits with status 0

#### Scenario: A managed compiler command is missing
- **WHEN** either `gcc` or `g++` does not resolve to a Nix store package
- **THEN** `devenv test` exits with a nonzero status and names the unmanaged command

#### Scenario: The shim hides a code warning
- **WHEN** the deliberate warning compiles successfully under `-Werror`
- **THEN** `devenv test` exits with a nonzero status

### Requirement: The shell runs the C backend test tier on macOS
The macOS shell MUST run the existing `chelis-backend-c` tests that invoke `gcc` and `g++`.

The change MUST NOT disable, ignore, or weaken a test to make this tier pass.

#### Scenario: The C backend suite passes in the shell
- **WHEN** a macOS contributor runs `devenv shell -- cargo nextest run -p chelis-backend-c`
- **THEN** the suite exits with status 0 and compiler-dependent tests run with the shell commands

#### Scenario: The compiler shims are absent
- **WHEN** the same suite runs in a macOS shell without the compiler shims
- **THEN** the compiler smoke checks fail before the result can count as acceptance evidence

### Requirement: Contributor documentation describes the optional shell
The contributor documentation MUST show the Devenv activation and smoke-check commands.

It MUST state that the macOS command shims invoke the Nixpkgs clang wrapper from `pkgs.stdenv.cc`, not host Apple clang.

The documentation MUST keep the manual setup path. It MUST NOT describe Devenv as a product or CI requirement.

#### Scenario: A contributor selects Devenv
- **WHEN** a contributor reads the source-build prerequisites
- **THEN** the documentation provides the Devenv activation, smoke-check, and backend-test commands

#### Scenario: A contributor selects manual setup
- **WHEN** a contributor does not use Devenv
- **THEN** the existing rustup, uv, and platform C-toolchain instructions remain available
