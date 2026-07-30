# Cross-Platform Devenv Specification

## Purpose

Define the reproducible contributor shell, platform compiler commands, Python environment, smoke checks, and contributor documentation for Linux and macOS.

## Requirements

### Requirement: The repository provides reproducible Devenv inputs
The repository MUST track `devenv.nix`, `devenv.yaml`, `devenv.lock`, and the local modules under `devenv/`.

The lock file MUST pin all resolved input revisions.

`devenv.yaml` MUST pin the Devenv module input to release `v2.2`. `devenv.lock` MUST resolve that input to commit `ffce215a42d09c6375c3d60dd9c4110438fc4d87`.

The repository MUST ignore `.devenv/` and `.devenv.flake.nix`. These paths contain generated local state, not reproducible inputs.

The repository lint policy MUST exclude both generated paths. It MUST keep the tracked root files and local modules in the editable corpus.

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

### Requirement: The shell composes local configuration modules
`devenv.yaml` MUST import these local modules:

- `./devenv/toolchains.nix`
- `./devenv/commands.nix`
- `./devenv/generated-files.nix`
- `./devenv/git-hooks.nix`
- `./devenv/smoke-tests.nix`

The root `devenv.nix` MUST remain a minimal composition root. Each local module MUST own one configuration concern.

#### Scenario: A local module is absent
- **WHEN** a required local import is absent or duplicated
- **THEN** the static Devenv composition test fails

#### Scenario: A contributor evaluates the composed shell
- **WHEN** a contributor runs `devenv test`
- **THEN** Devenv combines all five local modules
- **AND** the composed configuration passes the shell contract

### Requirement: The repository catalogs inactive Git hooks
The repository MUST declare the `git-hooks` input in `devenv.yaml`. The input MUST follow the configured `nixpkgs` input.

The repository MUST declare this Git hook catalog in `devenv/git-hooks.nix`:

- `actionlint`
- `check-added-large-files`
- `check-case-conflicts`
- `check-executables-have-shebangs`
- `check-json`
- `check-merge-conflicts`
- `check-python`
- `check-symlinks`
- `check-toml`
- `check-yaml`
- `detect-private-keys`
- `end-of-file-fixer`
- `fix-byte-order-marker`
- `forbid-new-submodules`
- `mixed-line-endings`
- `nixfmt`
- `rustfmt`
- `shellcheck`
- `trim-trailing-whitespace`

Each catalog entry MUST set `enable = false`. Devenv MUST NOT install or run a hook from this catalog by default.

The `nixfmt` entry MUST exclude generated `Cargo.nix`. The `rustfmt` entry MUST use check mode.

The `shellcheck` entry MUST select only `crates/chelisup/bootstrap/chelisup.sh`. The whitespace entry MUST preserve Markdown line breaks.

The repository MUST ignore `.pre-commit-config.yaml`. Devenv can generate this local file after a future hook activation.

#### Scenario: A contributor enters the shell with the inactive catalog
- **WHEN** a contributor runs `devenv shell`
- **THEN** no catalog hook is active
- **AND** Devenv does not install an active pre-commit hook

#### Scenario: A catalog entry becomes active without a policy change
- **WHEN** any catalog entry sets `enable = true`
- **THEN** the static Devenv contract test fails

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

### Requirement: Devenv manages the Python environment
The shell MUST enable Python 3.11, uv, and the Devenv Python virtual environment.

Devenv MUST store the environment at `$DEVENV_STATE/venv`. It MUST activate that environment before `devenv:enterShell`.

The shell MUST set `PYO3_PYTHON` to `$DEVENV_STATE/venv/bin/python`. It MUST NOT use the system Python for PyO3.

Devenv MUST NOT create, replace, or modify the repository-root `.venv`. The manual setup path owns that environment outside Devenv.

#### Scenario: Devenv creates its Python environment
- **WHEN** a contributor enters the shell without `$DEVENV_STATE/venv/bin/python`
- **THEN** uv creates the environment with Python 3.11
- **AND** Devenv activates the environment

#### Scenario: PyO3 uses the active Devenv interpreter
- **WHEN** a contributor builds a crate that uses PyO3 inside the shell
- **THEN** `PYO3_PYTHON` names the active Devenv virtual environment interpreter

#### Scenario: A manual environment exists
- **WHEN** a repository-root `.venv` exists before shell activation
- **THEN** Devenv preserves that environment without a change

### Requirement: The shell provides developer commands
The shell MUST provide these commands through Devenv scripts:

- `chelis-gate`
- `chelis-reap-orphans`
- `chelis-exec-preflight` on macOS
- `chelis-z3-test` on Linux
- `chelis-hip-test` on Linux

Each command MUST use the configured Devenv Python package. Each command MUST forward all arguments to its existing tested Python file.

The command definitions MUST NOT duplicate the Python implementation inside Nix.

#### Scenario: A contributor runs a command
- **WHEN** a contributor runs a supported command in the shell
- **THEN** Devenv invokes its repository Python file with all supplied arguments

#### Scenario: A platform-specific command is unavailable
- **WHEN** a contributor enters the shell on an unsupported platform
- **THEN** Devenv omits that platform-specific command

### Requirement: Devenv creates compiler probe files
Devenv MUST create these read-only compiler probe files from `devenv/generated-files.nix`:

- `.devenv/generated/compiler-probes/valid.c`
- `.devenv/generated/compiler-probes/warning.c`
- `.devenv/generated/compiler-probes/valid.cpp`

The compiler smoke tasks MUST compile these files. They MUST NOT create equivalent source with inline shell heredocs.

#### Scenario: Devenv materializes compiler probes
- **WHEN** a contributor enters the shell
- **THEN** each compiler probe is a Devenv-managed symlink
- **AND** each probe contains its declared source

#### Scenario: A compiler probe is absent
- **WHEN** a required compiler probe is absent from the declarative file module
- **THEN** the static Devenv composition test fails

### Requirement: The shell checks its compiler contract
`devenv test` MUST check the required tool versions and the Python interpreter version.

It MUST compile valid C and C++ translation units under `-Werror`. It MUST also prove that a deliberate code warning still fails.

The smoke checks MUST run after `devenv:enterShell` and before `devenv:enterTest` as these independent tasks:

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

### Requirement: Native package CI uses the reviewed portable Devenv base
Each native Nix package job MUST invoke `Chelis-Lang/ci/actions/setup-devenv@73f017c4d3179dc313844e9d5f08d17a7879c824` once before its first `run` step.

Each job MUST use `devenv-ci bash --noprofile --norc -e -o pipefail {0}` as its default shell for `run` steps.

The workflow MUST NOT duplicate the direct Nix, Cachix, or Devenv bootstrap. The reviewed action supplies Nix 2.34.4 and Devenv v2.2.

Each job MUST run `devenv test --no-tui`. The Devenv cache MUST NOT replace the complete native flake check.

The public Devenv cache does not contain the custom Chelis cvc5 derivation.

#### Scenario: Native CI checks the development shell
- **WHEN** either native Nix package job runs
- **THEN** the job invokes the reviewed portable Devenv action
- **AND** each `run` step uses the portable shell
- **AND** the job runs all four named Devenv tasks
- **AND** the job runs the complete native flake check

#### Scenario: Native CI bypasses the portable base
- **WHEN** a job omits the reviewed action, omits the portable shell, or adds a direct bootstrap
- **THEN** the native workflow contract fails

#### Scenario: The custom cvc5 output is absent from public caches
- **WHEN** the native flake check requires the custom non-GPL cvc5 derivation
- **THEN** Nix builds that derivation from source

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
The contributor documentation MUST show the Devenv activation, smoke-check, and developer commands.

It MUST state that the macOS command shims invoke the Nixpkgs clang wrapper from `pkgs.stdenv.cc`, not host Apple clang.

The documentation MUST keep the manual setup path. It MUST describe Devenv as optional for local work and required by native Nix CI only.

It MUST NOT describe Devenv as a product requirement.

#### Scenario: A contributor selects Devenv
- **WHEN** a contributor reads the source-build prerequisites
- **THEN** the documentation provides the Devenv activation, smoke-check, and backend-test commands

#### Scenario: A contributor selects manual setup
- **WHEN** a contributor does not use Devenv
- **THEN** the existing rustup, uv, and platform C-toolchain instructions remain available
